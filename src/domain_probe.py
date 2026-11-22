"""
How separable are two domains in a trained model's feature space?

Freezes the encoder of a finished run, extracts the bottleneck feature z for samples from two
splits, and trains a small classifier to tell which split each one came from. The accuracy it
reaches is a direct measure of how different the two domains look to that model.

    0.50   the domains are indistinguishable
    0.60   barely separable
    0.90+  obviously different

This is a cleaner diagnostic than DANN's live `domain_acc`, because there the encoder is being
pushed to hide the difference at the same time as the discriminator is trying to find it. Here
the features are frozen, so the number is a property of the model, not of a tug of war.

    python -m src.domain_probe --run runs/baseline/seed3_20261121-012402
    python -m src.domain_probe --run runs/dann/seed3_034357 --pairs source_train:target_train \
                                                                     source_train:test

Reading the result:
  * A HIGH score on the baseline means the domains really are apart in feature space, so
    aligning them is a sensible thing to attempt.
  * A LOW score on the baseline means they were already together, and CORAL or DANN have
    almost nothing to remove - which would explain a flat result for both.
  * Comparing baseline against dann shows how much DANN actually moved them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .config import load_config
from .data.crops import CropDataset
from .models.siamese import build_model


@torch.no_grad()
def extract_features(model, dataset, device, n, batch_size, seed, only_class=None):
    """Bottleneck features z for n random samples of one split, model in eval mode.

    only_class restricts to buildings of one true damage class. That matters because the two
    domains have different class mixes, so a probe can separate them on class alone. Holding the
    class fixed isolates genuine appearance difference from label shift.
    """
    rng = np.random.default_rng(seed)
    pool = np.arange(len(dataset))
    if only_class is not None:
        pool = pool[dataset.labels == only_class]
        if len(pool) == 0:
            raise ValueError(f"no class-{only_class} buildings in {dataset.split}")
    n = min(n, len(pool))
    idx = rng.choice(pool, size=n, replace=False)
    subset = torch.utils.data.Subset(dataset, idx.tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()                      # BatchNorm uses its running stats, as at deployment
    out = []
    for batch in loader:
        z = model.features(batch["pre"].to(device), batch["post"].to(device))
        out.append(z.float().cpu())
    return torch.cat(out)


def feature_gaps(z_a, z_b):
    """Where the two feature clouds actually differ.

    mean_gap    || mu_a - mu_b || / sqrt(d), the distance between the centres, in units of
                feature sd. CORAL's loss contains no mean term, so a mean offset survives it.
    coral_dist  || C_a - C_b ||_F^2 / (4 d^2), the quantity CORAL minimises, so this one
                SHOULD fall in a CORAL run.

    Together they say whether an alignment method moved what it was supposed to move.
    """
    d = z_a.shape[1]
    mu_a, mu_b = z_a.mean(0), z_b.mean(0)
    scale = torch.cat([z_a, z_b]).std().clamp_min(1e-6)
    ca, cb = torch.cov(z_a.t()), torch.cov(z_b.t())
    return {
        "mean_gap": float((mu_a - mu_b).norm() / (d ** 0.5) / scale),
        "coral_dist": float(((ca - cb) ** 2).sum() / (4.0 * d * d)),
    }


def run_probe(z_a, z_b, seed=0, epochs=30, hidden=256, lr=1e-3, val_frac=0.3, device="cpu"):
    """Train a small MLP to tell z_a from z_b. Returns accuracy and AUC on held-out features.

    The two sides are balanced to the same size, so chance is exactly 0.50.
    """
    n = min(len(z_a), len(z_b))
    g = torch.Generator().manual_seed(seed)
    z_a = z_a[torch.randperm(len(z_a), generator=g)[:n]]
    z_b = z_b[torch.randperm(len(z_b), generator=g)[:n]]

    x = torch.cat([z_a, z_b])
    y = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])

    perm = torch.randperm(len(x), generator=g)
    x, y = x[perm], y[perm]
    n_val = int(len(x) * val_frac)
    x_val, y_val, x_tr, y_tr = x[:n_val], y[:n_val], x[n_val:], y[n_val:]

    # standardise with the training half only, so the probe cannot cheat through the scaling
    mean, std = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_tr, x_val = ((x_tr - mean) / std).to(device), ((x_val - mean) / std).to(device)
    y_tr, y_val = y_tr.to(device), y_val.to(device)

    torch.manual_seed(seed)
    probe = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(),
                          nn.Dropout(0.2), nn.Linear(hidden, 2)).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    best = {"acc": 0.0, "auc": 0.0, "epoch": 0}
    for epoch in range(epochs):
        probe.train()
        order = torch.randperm(len(x_tr), generator=g).to(device)
        for i in range(0, len(x_tr), 256):
            sel = order[i:i + 256]
            opt.zero_grad(set_to_none=True)
            nn.functional.cross_entropy(probe(x_tr[sel]), y_tr[sel]).backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            logits = probe(x_val)
            acc = (logits.argmax(1) == y_val).float().mean().item()
            auc = roc_auc_score(y_val.cpu().numpy(),
                                torch.softmax(logits, 1)[:, 1].cpu().numpy())
        if acc > best["acc"]:                 # best the probe manages = separability
            best = {"acc": acc, "auc": float(auc), "epoch": epoch + 1}
    return best


def main():
    p = argparse.ArgumentParser(description="Measure how separable two domains are in a run's features.")
    p.add_argument("--run", required=True, help="a run folder, e.g. runs/baseline/seed3_...")
    p.add_argument("--checkpoint", default="best.pt", choices=["best.pt", "last.pt"])
    p.add_argument("--pairs", nargs="*", default=["source_train:target_train"],
                   help="split pairs to compare, e.g. source_train:target_train source_train:test")
    p.add_argument("--n", type=int, default=6000, help="samples per split")
    p.add_argument("--epochs", type=int, default=30, help="probe training epochs")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--per-class", action="store_true",
                   help="also probe within each damage class separately, which removes the "
                        "label-shift confound (the domains differ in class mix, and a probe "
                        "can separate them on that alone)")
    args = p.parse_args()

    run_dir = Path(args.run)
    cfg = load_config(run_dir / "config.yaml")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else
         "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
         else "cpu") if args.device == "auto" else args.device)

    model = build_model(cfg).to(device)
    ckpt = torch.load(run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"run      : {run_dir}   ({cfg.method.name}, seed {cfg.run.seed}, epoch {ckpt['epoch'] + 1})")
    print(f"device   : {device}   samples per split: {args.n:,}\n")

    CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    datasets, cache, results = {}, {}, []

    def features(split, only_class=None):
        key = (split, only_class)
        if key not in cache:
            if split not in datasets:
                datasets[split] = CropDataset(cfg.data.crops_dir, split, train=False,
                                              use_labels=False, max_black=cfg.data.max_black)
            cache[key] = extract_features(model, datasets[split], device, args.n,
                                          args.batch_size, args.seed, only_class)
            label = split if only_class is None else f"{split}[{CLASSES[only_class]}]"
            print(f"  features {label:<30} {tuple(cache[key].shape)}")
        return cache[key]

    for pair in args.pairs:
        a, b = pair.split(":")
        za, zb = features(a), features(b)
        best = run_probe(za, zb, seed=args.seed, epochs=args.epochs, device=device)
        results.append({"pair": pair, "class": "all", **best, **feature_gaps(za, zb)})
        if args.per_class:
            for c, name in enumerate(CLASSES):
                try:
                    za, zb = features(a, c), features(b, c)
                except ValueError as e:
                    print(f"  skipping {name}: {e}")
                    continue
                best_c = run_probe(za, zb, seed=args.seed, epochs=args.epochs, device=device)
                best_c["n_per_side"] = min(len(za), len(zb))
                results.append({"pair": pair, "class": name, **best_c, **feature_gaps(za, zb)})

    print(f"\n{'pair':<30}{'class':<15}{'n/side':>8}{'probe acc':>11}{'AUC':>8}{'verdict':>20}")
    for r in results:
        acc = r["acc"]
        verdict = ("indistinguishable" if acc < 0.60 else
                   "barely separable" if acc < 0.70 else
                   "clearly separable" if acc < 0.85 else "very different")
        n = r.get("n_per_side", args.n)
        print(f"{r['pair']:<30}{r['class']:<15}{n:>8,}{acc:>11.3f}{r['auc']:>8.3f}{verdict:>20}")

    print(f"\nWhere the gap lives (the 'all' rows)")
    print(f"{'pair':<30}{'mean gap':>11}{'CORAL dist':>13}{'PAD':>8}")
    for r in results:
        if r["class"] != "all":
            continue
        pad = 2 * (1 - 2 * (1 - r["acc"]))
        print(f"{r['pair']:<30}{r['mean_gap']:>11.4f}{r['coral_dist']:>13.6f}{pad:>8.3f}")
    print("mean gap  : distance between the feature centres. CORAL's loss ignores this, so it")
    print("            should NOT shrink in a CORAL run.")
    print("CORAL dist: what CORAL minimises, so it SHOULD shrink in a CORAL run.")
    print("PAD       : proxy A-distance, 2(1-2*error). 0 = identical, 2 = perfectly separable.")

    print("\n0.50 = chance.")
    print("'all' mixes two effects: the domains look different, AND they have different class")
    print("mixes. The per-class rows hold the class fixed, so they show appearance difference only.")

    out = run_dir / "domain_probe.json"
    out.write_text(json.dumps({"run": str(run_dir), "method": cfg.method.name,
                               "seed": cfg.run.seed, "n_per_split": args.n,
                               "results": results}, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
