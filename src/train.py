"""
Train one run: one method, one seed.

    python -m src.train --config configs/baseline.yaml --set run.seed=0
    python -m src.train --config configs/dann.yaml --set run.seed=0 --resume

Everything lands in runs/<method>/<seed>_<timestamp>/:
    config.yaml        the resolved config, so the run can be reproduced
    last.pt            written every epoch, this is what --resume reads
    best.pt            best epoch on eval.select_on (never chosen with target labels)
    history.csv        one row per epoch, including the validation loss and the epoch time
    metrics.json       final numbers for every split, plus the total wall time
    predictions_<split>.csv   one row per building, for the figures
    tb/                TensorBoard logs
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:                                    # optional: the run works without it
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .algorithms import make_algorithm
from .config import add_config_args, load_config, save_config
from .data.crops import AugConfig, class_weights, cycle, make_datasets, make_loaders
from .metrics import CLASS_NAMES, compute_metrics, confusion, format_confusion, format_metrics
from .models.siamese import build_model, count_parameters


# ----------------------------------------------------------------------------- setup
def progress_bar(iterable, enabled, **kwargs):
    """tqdm when it is installed and switched on, otherwise the plain iterable."""
    if tqdm is None or not enabled:
        return iterable
    return tqdm(iterable, **kwargs)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hms(seconds):
    """Seconds as 1h 23m 45s, for the timing lines."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return (f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s" if m else f"{s}s")


def rng_state(loaders):
    """Everything needed to carry the random stream across a resume: python, numpy, torch, and
    the DataLoader shuffle generators (without those the data order restarts)."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    for name in ("train", "target"):
        loader = loaders.get(name)
        gen = getattr(loader, "generator", None) if loader is not None else None
        state[f"loader_{name}"] = gen.get_state() if gen is not None else None
    return state


def restore_rng(state, loaders):
    if not state:
        return False
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    for name in ("train", "target"):
        loader = loaders.get(name)
        gen = getattr(loader, "generator", None) if loader is not None else None
        if gen is not None and state.get(f"loader_{name}") is not None:
            gen.set_state(state[f"loader_{name}"])
    return True


def build_optimizer(cfg, model, algo):
    groups = model.param_groups(cfg.optim.lr, cfg.optim.backbone_lr)
    extra = algo.parameters()
    if extra:
        groups.append({"params": extra, "lr": cfg.optim.lr, "name": algo.name})
    if cfg.optim.name == "adamw":
        opt = torch.optim.AdamW(groups, weight_decay=cfg.optim.weight_decay)
    else:
        opt = torch.optim.SGD(groups, momentum=0.9, weight_decay=cfg.optim.weight_decay)

    if cfg.optim.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.optim.epochs)
    elif cfg.optim.scheduler == "step":
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, cfg.optim.epochs // 4), gamma=0.5)
    else:
        sched = None
    return opt, sched


# ----------------------------------------------------------------------------- one epoch
def train_epoch(cfg, model, algo, loaders, criterion, opt, sched, scaler, device, epoch, writer):
    model.train()
    algo.train()

    target_iter = cycle(loaders["target"]) if loaders["target"] is not None else None
    amp = scaler.is_enabled()
    n_steps = len(loaders["train"])
    sums = defaultdict(float)
    correct = seen = 0
    t0 = time.time()

    bar = progress_bar(loaders["train"], cfg.log.progress, leave=False, unit="batch",
                       desc=f"epoch {epoch + 1}/{cfg.optim.epochs}")
    for step, batch in enumerate(bar):
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        tgt = next(target_iter) if target_iter is not None else None
        progress = (epoch * n_steps + step) / max(1, cfg.optim.epochs * n_steps)

        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits, z = model(pre, post)
            cls_loss = criterion(logits, y)
            extra, logs = algo.extra_loss(model, z, tgt, device, epoch=epoch, progress=progress)
            loss = cls_loss + extra

        scaler.scale(loss).backward()
        if cfg.optim.grad_clip > 0:
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
        scaler.step(opt)
        scaler.update()

        bs = y.size(0)
        sums["loss"] += loss.item() * bs
        sums["cls_loss"] += cls_loss.item() * bs
        for k, v in logs.items():                     # whatever the method chose to report
            sums[k] += v * bs
        correct += (logits.argmax(1) == y).sum().item()
        seen += bs

        if hasattr(bar, "set_postfix") and step % 10 == 0:
            post_fix = {"loss": f"{sums['loss'] / seen:.3f}", "acc": f"{correct / seen:.3f}"}
            if "domain_acc" in logs:
                post_fix["dom"] = f"{sums['domain_acc'] / seen:.3f}"
            bar.set_postfix(post_fix)

        if writer and step % cfg.log.every_n_steps == 0:
            gstep = epoch * n_steps + step
            writer.add_scalar("train/loss", loss.item(), gstep)
            writer.add_scalar("train/cls_loss", cls_loss.item(), gstep)
            for k, v in logs.items():
                writer.add_scalar(f"train/{k}", v, gstep)

    if sched is not None:
        sched.step()

    seen = max(1, seen)
    stats = {k: v / seen for k, v in sums.items()}
    stats["train_acc"] = correct / seen
    stats["seconds"] = time.time() - t0
    return stats


@torch.no_grad()
def evaluate(model, loader, device, want_predictions=False, desc=None, show_progress=True,
             criterion=None):
    """Metrics for one split. With a criterion, also the average loss on that split, so the
    training and validation loss can be compared directly (same weighted cross-entropy)."""
    model.eval()
    y_true, y_pred, probs, uids = [], [], [], []
    loss_sum, seen = 0.0, 0

    for batch in progress_bar(loader, show_progress and desc is not None,
                              leave=False, unit="batch", desc=desc):
        y = batch["label"].to(device)
        logits, _ = model(batch["pre"].to(device), batch["post"].to(device))
        if criterion is not None:
            loss_sum += criterion(logits, y).item() * y.size(0)
            seen += y.size(0)
        p = torch.softmax(logits.float(), dim=1).cpu()
        y_pred.append(p.argmax(1).numpy())
        y_true.append(batch["label"].numpy())
        if want_predictions:
            probs.append(p.numpy())
            uids.extend(batch["uid"])

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    metrics = compute_metrics(y_true, y_pred)
    if criterion is not None and seen:
        metrics["loss"] = loss_sum / seen

    if not want_predictions:
        return metrics, None

    probs = np.concatenate(probs)
    table = pd.DataFrame({"building_uid": uids, "true_label": y_true, "pred_label": y_pred,
                          "correct": y_true == y_pred})
    for c, name in enumerate(CLASS_NAMES):
        table[f"p_{name}"] = probs[:, c]
    return metrics, table


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(path, model, algo, opt, sched, scaler, epoch, best, cfg,
                    since_best=0, rng=None):
    torch.save({
        "model": model.state_dict(),
        "algorithm": algo.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best": best,
        "since_best": since_best,
        "rng": rng,
        "run_name": cfg.run.name,
    }, path)


def load_checkpoint(path, model, algo, opt, sched, scaler, device, loaders=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    algo.load_state_dict(ckpt.get("algorithm"))
    if opt is not None:
        opt.load_state_dict(ckpt["optimizer"])
        if sched is not None and ckpt.get("scheduler") is not None:
            sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        if loaders is not None and restore_rng(ckpt.get("rng"), loaders):
            print("rng      : random streams restored, data order continues")
    return ckpt["epoch"], ckpt["best"], ckpt.get("since_best", 0)


# ----------------------------------------------------------------------------- main
def main():
    p = add_config_args(argparse.ArgumentParser(description="Train one run."))
    p.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt")
    p.add_argument("--overwrite", action="store_true",
                   help="delete an existing run folder of the same name and start again")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="auto picks cuda, then mps (Apple GPU), then cpu")
    args = p.parse_args()

    wall_start = time.time()
    cfg = load_config(args.config, args.set)
    out_root = Path(cfg.run.out_dir) / cfg.run.group
    out_dir = out_root / cfg.run.name

    # With run.timestamp on, the folder name is new every time, so --resume has to go and find
    # the most recent run for this method and seed.
    if args.resume and cfg.run.auto_name and not (out_dir / "last.pt").exists():
        earlier = sorted(d for d in out_root.glob(f"seed{cfg.run.seed}_*")
                         if (d / "last.pt").exists())
        if earlier:
            out_dir = earlier[-1]
            cfg.run.name = out_dir.name
            print(f"resuming : {out_dir}")

    # protect finished runs: a mistyped name should not quietly destroy results
    already_there = (out_dir / "last.pt").exists() or (out_dir / "metrics.json").exists()
    if already_there and not (args.resume or args.overwrite):
        sys.exit(f"\n{out_dir} already holds a run.\n"
                 f"  --resume     continue it from last.pt\n"
                 f"  --overwrite  delete it and start again\n"
                 f"  or give it another name:  --set run.name=<something>\n")
    if args.overwrite and not args.resume:
        # the whole folder goes, tb/ included: two event files in one folder make
        # TensorBoard merge the old and new run into one unreadable chart
        shutil.rmtree(out_dir, ignore_errors=True)
    if args.resume and not (out_dir / "last.pt").exists():
        print("note     : --resume given but nothing found to resume, starting from scratch")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    set_seed(cfg.run.seed)
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"run      : {out_dir}   ({cfg.method.name}, seed {cfg.run.seed})")
    print(f"device   : {device}")
    print(f"started  : {started_at}")

    # ---------------- data
    aug = AugConfig(**cfg.data.aug.model_dump())
    ds = make_datasets(cfg.data.crops_dir, cfg.method.name, max_black=cfg.data.max_black, aug=aug)
    loaders = make_loaders(ds, batch_size=cfg.data.batch_size, workers=cfg.data.workers,
                           seed=cfg.run.seed)
    weights = class_weights(ds["train_labels"], cfg.loss.class_weights).to(device)
    print(f"train    : {len(ds['train']):,} buildings"
          + (f"   target (unlabelled): {len(ds['target']):,}" if ds["target"] else ""))
    print("weights  : " + "  ".join(f"{n}={w:.3f}" for n, w in zip(CLASS_NAMES, weights.tolist())))

    # ---------------- model and method
    model = build_model(cfg).to(device)
    algo = make_algorithm(cfg)
    algo.build(model)
    algo.to(device)
    extra_params = sum(p.numel() for p in algo.parameters())
    print(f"model    : {cfg.model.backbone}, {count_parameters(model):,} parameters")
    print(f"method   : {algo.name}" + (f", {extra_params:,} extra parameters" if extra_params else ""))

    criterion = nn.CrossEntropyLoss(weight=weights)
    opt, sched = build_optimizer(cfg, model, algo)
    amp = cfg.optim.amp and device.type == "cuda"      # AMP is only reliable on CUDA
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    start_epoch, best = 0, -1.0
    since_best = 0
    history = []
    if args.resume and (out_dir / "last.pt").exists():
        start_epoch, best, since_best = load_checkpoint(
            out_dir / "last.pt", model, algo, opt, sched, scaler, device, loaders)
        start_epoch += 1
        print(f"resumed  : from epoch {start_epoch}, best so far {best:.4f}, "
              f"{since_best} epoch(s) since it")
        # keep the epochs already written, or resuming would wipe them from history.csv
        prev_csv = out_dir / "history.csv"
        if prev_csv.exists():
            prev = pd.read_csv(prev_csv)
            history = prev[prev["epoch"] < start_epoch].to_dict("records")
            print(f"history  : kept {len(history)} earlier epoch(s) from history.csv")

    if cfg.log.progress and tqdm is None:
        print("note     : tqdm not installed, no progress bar (conda install -n ecse552 tqdm)")

    writer = None
    if cfg.log.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(out_dir / "tb")
        except ImportError:
            print("note     : tensorboard not installed, skipping its logs")

    # ---------------- train
    select_split = cfg.eval.select_on
    print(f"\nepochs   : {cfg.optim.epochs}, best epoch chosen on {select_split} "
          f"({cfg.eval.select_metric})\n")
    train_start = time.time()

    for epoch in range(start_epoch, cfg.optim.epochs):
        stats = train_epoch(cfg, model, algo, loaders, criterion, opt, sched, scaler,
                            device, epoch, writer)
        val, _ = evaluate(model, loaders["eval"][select_split], device,
                          desc=f"eval {select_split}", show_progress=cfg.log.progress,
                          criterion=criterion)
        score = val[cfg.eval.select_metric]

        row = {"epoch": epoch, **stats, **{f"{select_split}/{k}": v for k, v in val.items()}}

        # Extra splits evaluated every epoch for the curves ONLY. The best epoch is still chosen
        # on select_split above - looking at a target curve must not turn into selecting on it.
        tracked = {}
        for name in cfg.eval.track_splits:
            if name == select_split:
                continue
            m, _ = evaluate(model, loaders["eval"][name], device, desc=f"track {name}",
                            show_progress=cfg.log.progress, criterion=criterion)
            tracked[name] = m
            row.update({f"{name}/{k}": v for k, v in m.items()})

        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        if writer:
            for k, v in val.items():
                writer.add_scalar(f"{select_split}/{k}", v, epoch)
            # the two losses share a prefix so TensorBoard puts them side by side:
            # train falling while val turns up is overfitting
            for name, m in tracked.items():
                for k, v in m.items():
                    writer.add_scalar(f"{name}/{k}", v, epoch)
            writer.add_scalar("loss/train", stats["loss"], epoch)
            if "loss" in val:
                writer.add_scalar("loss/val", val["loss"], epoch)
            for name, m in tracked.items():
                if "loss" in m:
                    writer.add_scalar(f"loss/{name}", m["loss"], epoch)
            writer.add_scalar("time/epoch_seconds", stats["seconds"], epoch)

        marker = ""
        if score > best:
            best = score
            since_best = 0
            save_checkpoint(out_dir / "best.pt", model, algo, opt, sched, scaler, epoch, best, cfg,
                            since_best, rng_state(loaders))
            marker = "  <- best"
        else:
            since_best += 1
        if cfg.log.save_last_every_epoch:
            save_checkpoint(out_dir / "last.pt", model, algo, opt, sched, scaler, epoch, best, cfg,
                            since_best, rng_state(loaders))

        # timing: how long this epoch took, and when the run should finish
        done = epoch - start_epoch + 1
        left = cfg.optim.epochs - epoch - 1
        per_epoch = (time.time() - train_start) / done
        eta = time.strftime("%H:%M", time.localtime(time.time() + left * per_epoch))

        if "extra_loss" in stats:
            lam = f" (lam {stats['lambda']:.2f})" if "lambda" in stats else ""   # bn_only has none
            extra = f"  extra {stats['extra_loss']:.4f}{lam}"
        else:
            extra = ""
        dom = f"  dom-acc {stats['domain_acc']:.3f}" if "domain_acc" in stats else ""
        val_loss = f"  val-loss {val['loss']:.4f}" if "loss" in val else ""
        track = "".join(f"  [{n} {m[cfg.eval.select_metric]:.4f}]" for n, m in tracked.items())
        print(f"epoch {epoch + 1:3d}/{cfg.optim.epochs}  loss {stats['loss']:.4f}{val_loss}{extra}{dom}  "
              f"train-acc {stats['train_acc']:.3f}  |  {select_split} "
              f"{cfg.eval.select_metric} {score:.4f}{track}  "
              f"({hms(stats['seconds'])}, eta {eta}){marker}")

        patience = cfg.optim.early_stop
        if patience and epoch + 1 >= cfg.optim.min_epochs and since_best >= patience:
            print(f"\nearly stop: no gain on {select_split} for {since_best} epochs "
                  f"(patience {patience}). Stopped at epoch {epoch + 1} of {cfg.optim.epochs}.")
            stopped_early = epoch + 1
            break
    else:
        stopped_early = None

    train_seconds = time.time() - train_start
    epochs_run = len(history)
    print(f"\ntraining : {hms(train_seconds)} for {epochs_run} epoch(s)"
          + (f"   ({hms(train_seconds / epochs_run)} per epoch)" if epochs_run else ""))
    print(f"best {cfg.eval.select_metric} on {select_split}: {best:.4f}")

    # ---------------- final evaluation with the best checkpoint
    if (out_dir / "best.pt").exists():
        load_checkpoint(out_dir / "best.pt", model, algo, None, None, scaler, device)

    results = {"run": cfg.run.name, "group": cfg.run.group, "run_dir": str(out_dir),
               "method": cfg.method.name, "seed": cfg.run.seed,
               "best_epoch_score": best, "epochs_run": epochs_run,
               "epochs_planned": cfg.optim.epochs,
               "stopped_early": stopped_early is not None,
               "started_at": started_at,
               "train_seconds": round(train_seconds, 1),
               "seconds_per_epoch": round(train_seconds / epochs_run, 1) if epochs_run else None,
               "device": str(device),
               "splits": {}}

    for name in cfg.eval.splits:
        m, table = evaluate(model, loaders["eval"][name], device,
                            want_predictions=cfg.eval.save_predictions,
                            desc=f"eval {name}", show_progress=cfg.log.progress,
                            criterion=criterion)
        results["splits"][name] = m
        print("\n" + format_metrics(m, title=f"[{name}]"))
        if table is not None:
            table.to_csv(out_dir / f"predictions_{name}.csv", index=False)
            print(format_confusion(confusion(table["true_label"], table["pred_label"])))
        if writer:
            for k, v in m.items():
                writer.add_scalar(f"final/{name}/{k}", v, 0)

    results["total_seconds"] = round(time.time() - wall_start, 1)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    if writer:
        writer.close()

    print(f"\ntotal    : {hms(results['total_seconds'])}  "
          f"(training {hms(train_seconds)}, rest is setup and final evaluation)")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
