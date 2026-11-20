"""
Train one run: one method, one seed.

    python -m src.train --config configs/baseline.yaml --set run.seed=0
    python -m src.train --config configs/baseline.yaml --set run.seed=0 --resume

Everything lands in runs/<run.name>/:
    config.yaml        the resolved config, so the run can be reproduced
    last.pt            written every epoch, this is what --resume reads
    best.pt            best epoch on eval.select_on (never chosen with target labels)
    history.csv        one row per epoch
    metrics.json       final numbers for every split
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
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def progress_bar(iterable, enabled, **kwargs):
    """tqdm when it is installed and switched on, otherwise the plain iterable."""
    if tqdm is None or not enabled:
        return iterable
    return tqdm(iterable, **kwargs)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
            post = {"loss": f"{sums['loss'] / seen:.3f}", "acc": f"{correct / seen:.3f}"}
            if "domain_acc" in logs:
                post["dom"] = f"{sums['domain_acc'] / seen:.3f}"
            bar.set_postfix(post)

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
def evaluate(model, loader, device, want_predictions=False, desc=None, show_progress=True):
    model.eval()
    y_true, y_pred, probs, uids = [], [], [], []
    for batch in progress_bar(loader, show_progress and desc is not None,
                              leave=False, unit="batch", desc=desc):
        logits, _ = model(batch["pre"].to(device), batch["post"].to(device))
        p = torch.softmax(logits.float(), dim=1).cpu()
        y_pred.append(p.argmax(1).numpy())
        y_true.append(batch["label"].numpy())
        if want_predictions:
            probs.append(p.numpy())
            uids.extend(batch["uid"])

    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    metrics = compute_metrics(y_true, y_pred)
    if not want_predictions:
        return metrics, None

    probs = np.concatenate(probs)
    table = pd.DataFrame({"building_uid": uids, "true_label": y_true, "pred_label": y_pred,
                          "correct": y_true == y_pred})
    for c, name in enumerate(CLASS_NAMES):
        table[f"p_{name}"] = probs[:, c]
    return metrics, table


# ----------------------------------------------------------------------------- checkpoints
def save_checkpoint(path, model, algo, opt, sched, scaler, epoch, best, cfg):
    torch.save({
        "model": model.state_dict(),
        "algorithm": algo.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best": best,
        "run_name": cfg.run.name,
    }, path)


def load_checkpoint(path, model, algo, opt, sched, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    algo.load_state_dict(ckpt.get("algorithm"))
    if opt is not None:
        opt.load_state_dict(ckpt["optimizer"])
        if sched is not None and ckpt.get("scheduler") is not None:
            sched.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"], ckpt["best"]


# ----------------------------------------------------------------------------- main
def main():
    p = add_config_args(argparse.ArgumentParser(description="Train one run."))
    p.add_argument("--resume", action="store_true", help="continue from runs/<name>/last.pt")
    p.add_argument("--overwrite", action="store_true",
                   help="delete an existing run folder of the same name and start again")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"],
                   help="auto picks cuda, then mps (Apple GPU), then cpu")
    args = p.parse_args()

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
        print(f"note     : --resume given but nothing found to resume, starting from scratch")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    set_seed(cfg.run.seed)
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    print(f"run      : {out_dir}   ({cfg.method.name}, seed {cfg.run.seed})")
    print(f"device   : {device}")

    # ---------------- data
    aug = AugConfig(**cfg.data.aug.model_dump())
    ds = make_datasets(cfg.data.crops_dir, cfg.method.name, max_black=cfg.data.max_black, aug=aug)
    loaders = make_loaders(ds, batch_size=cfg.data.batch_size, workers=cfg.data.workers,
                           seed=cfg.run.seed)
    weights = class_weights(ds["train_labels"], cfg.loss.class_weights).to(device)
    print(f"train    : {len(ds['train']):,} buildings"
          + (f"   target (unlabelled): {len(ds['target']):,}" if ds["target"] else ""))
    print("weights  : " + "  ".join(f"{n}={w:.3f}" for n, w in zip(CLASS_NAMES, weights.tolist())))

    # ---------------- model
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
    if args.resume and (out_dir / "last.pt").exists():
        start_epoch, best = load_checkpoint(out_dir / "last.pt", model, algo, opt, sched, scaler, device)
        start_epoch += 1
        print(f"resumed  : from epoch {start_epoch}, best so far {best:.4f}")

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
    history = []
    print(f"\nepochs   : {cfg.optim.epochs}, best epoch chosen on {select_split} "
          f"({cfg.eval.select_metric})\n")

    for epoch in range(start_epoch, cfg.optim.epochs):
        stats = train_epoch(cfg, model, algo, loaders, criterion, opt, sched, scaler,
                            device, epoch, writer)
        val, _ = evaluate(model, loaders["eval"][select_split], device,
                          desc=f"eval {select_split}", show_progress=cfg.log.progress)
        score = val[cfg.eval.select_metric]

        row = {"epoch": epoch, **stats, **{f"{select_split}/{k}": v for k, v in val.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        if writer:
            for k, v in val.items():
                writer.add_scalar(f"{select_split}/{k}", v, epoch)
            writer.add_scalar("train/epoch_loss", stats["loss"], epoch)

        marker = ""
        if score > best:
            best = score
            since_best = 0
            save_checkpoint(out_dir / "best.pt", model, algo, opt, sched, scaler, epoch, best, cfg)
            marker = "  <- best"
        else:
            since_best += 1
        if cfg.log.save_last_every_epoch:
            save_checkpoint(out_dir / "last.pt", model, algo, opt, sched, scaler, epoch, best, cfg)

        extra = f"  extra {stats['extra_loss']:.4f} (lam {stats['lambda']:.2f})" \
            if "extra_loss" in stats else ""
        dom = f"  dom-acc {stats['domain_acc']:.3f}" if "domain_acc" in stats else ""
        print(f"epoch {epoch + 1:3d}/{cfg.optim.epochs}  loss {stats['loss']:.4f}{extra}{dom}  "
              f"train-acc {stats['train_acc']:.3f}  |  {select_split} "
              f"{cfg.eval.select_metric} {score:.4f}  ({stats['seconds']:.0f}s){marker}")

        patience = cfg.optim.early_stop
        if patience and epoch + 1 >= cfg.optim.min_epochs and since_best >= patience:
            print(f"\nearly stop: no gain on {select_split} for {since_best} epochs "
                  f"(patience {patience}). Stopped at epoch {epoch + 1} of {cfg.optim.epochs}.")
            stopped_early = epoch + 1
            break
    else:
        stopped_early = None

    # ---------------- final evaluation with the best checkpoint
    print(f"\nbest {cfg.eval.select_metric} on {select_split}: {best:.4f}")
    if (out_dir / "best.pt").exists():
        load_checkpoint(out_dir / "best.pt", model, algo, None, None, scaler, device)

    results = {"run": cfg.run.name, "group": cfg.run.group, "run_dir": str(out_dir),
               "method": cfg.method.name, "seed": cfg.run.seed,
               "best_epoch_score": best, "epochs_run": stopped_early or cfg.optim.epochs,
               "stopped_early": stopped_early is not None, "splits": {}}
    for name in cfg.eval.splits:
        m, table = evaluate(model, loaders["eval"][name], device,
                            want_predictions=cfg.eval.save_predictions,
                            desc=f"eval {name}", show_progress=cfg.log.progress)
        results["splits"][name] = m
        print("\n" + format_metrics(m, title=f"[{name}]"))
        y_true = table["true_label"] if table is not None else None
        if table is not None:
            table.to_csv(out_dir / f"predictions_{name}.csv", index=False)
            print(format_confusion(confusion(y_true, table["pred_label"])))
        if writer:
            for k, v in m.items():
                writer.add_scalar(f"final/{name}/{k}", v, 0)

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    if writer:
        writer.close()
    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
