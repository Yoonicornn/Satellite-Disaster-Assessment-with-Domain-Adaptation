"""
Summarise the experiment grid: mean +/- std over seeds, and the two differences the
bn_only control exists to separate.

    python scripts/summarize.py
    python scripts/summarize.py --runs runs --split target_test --metric macro_f1
    python scripts/summarize.py --csv results.csv        # also write a table for the report

Reads runs/<method>/<run>/metrics.json, which src/train.py writes at the end of every run.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ORDER = ["baseline", "bn_only", "coral", "dann", "oracle"]


def load(runs_dir):
    rows = []
    for f in sorted(glob.glob(str(Path(runs_dir) / "*" / "*" / "metrics.json"))):
        try:
            d = json.load(open(f))
        except Exception as e:
            print(f"  skipping {f}: {e}")
            continue
        row = {"method": d["method"], "seed": d["seed"], "run_dir": d.get("run_dir", f),
               "epochs_run": d.get("epochs_run"), "best_epoch_score": d.get("best_epoch_score"),
               "minutes": (d.get("train_seconds") or 0) / 60}
        for split, m in d["splits"].items():
            for k, v in m.items():
                row[f"{split}/{k}"] = v
        rows.append(row)
    if not rows:
        raise SystemExit(f"no metrics.json found under {runs_dir}/*/*/")
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"],
                                  [m for m in ORDER if m in set(df["method"])] +
                                  sorted(set(df["method"]) - set(ORDER)), ordered=True)
    return df.sort_values(["method", "seed"]).reset_index(drop=True)


def table(df, splits, metric):
    """mean +/- std over seeds, one row per method."""
    out = []
    for method, g in df.groupby("method", observed=True):
        row = {"method": method, "seeds": len(g)}
        for s in splits:
            col = f"{s}/{metric}"
            if col in g:
                row[s] = g[col].mean()
                # ddof=1 is the sample std, the right one for 'how much does a rerun vary'
                row[f"{s}_sd"] = g[col].std(ddof=1) if len(g) > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser(description="Summarise the experiment grid.")
    p.add_argument("--runs", default="runs")
    p.add_argument("--splits", nargs="*", default=["source_val", "target_test", "test"])
    p.add_argument("--metric", default="macro_f1")
    p.add_argument("--csv", default=None, help="also write the per-run table here")
    args = p.parse_args()

    df = load(args.runs)
    metric = args.metric

    # ---------------------------------------------------------------- every run
    print(f"all runs ({len(df)})\n")
    print(f"{'method':<10}{'seed':>5}{'ep':>4}{'min':>7}" +
          "".join(f"{s:>14}" for s in args.splits))
    for _, r in df.iterrows():
        print(f"{str(r['method']):<10}{r['seed']:>5}{r['epochs_run'] or 0:>4}{r['minutes']:>7.0f}" +
              "".join(f"{r.get(f'{s}/{metric}', float('nan')):>14.4f}" for s in args.splits))

    # ---------------------------------------------------------------- mean +/- sd
    t = table(df, args.splits, metric)
    print(f"\n\nmean +/- sd over seeds  ({metric})\n")
    print(f"{'method':<10}{'n':>3}" + "".join(f"{s:>20}" for s in args.splits))
    for _, r in t.iterrows():
        cells = ""
        for s in args.splits:
            sd = r.get(f"{s}_sd", np.nan)
            cells += f"{r[s]:>13.4f} +/-{sd:>5.4f}" if pd.notna(sd) else f"{r[s]:>20.4f}"
        print(f"{str(r['method']):<10}{int(r['seeds']):>3}" + cells)

    # ---------------------------------------------------------------- the decomposition
    have = set(t["method"].astype(str))
    if {"baseline", "bn_only"} <= have:
        print(f"\n\nwhat each part contributes  ({metric}, mean over seeds)\n")
        base_sd = t.loc[t["method"].astype(str) == "baseline"]
        for s in args.splits:
            get = lambda m: float(t.loc[t["method"].astype(str) == m, s].iloc[0])
            noise = float(base_sd[f"{s}_sd"].iloc[0]) if pd.notna(base_sd[f"{s}_sd"].iloc[0]) else np.nan
            print(f"  [{s}]   baseline seed-to-seed sd = {noise:.4f}"
                  f"{'   <- a gain smaller than this is not a result' if pd.notna(noise) else ''}")
            bn = get("bn_only") - get("baseline")
            print(f"    BatchNorm adaptation   baseline -> bn_only   {bn:+.4f}  {verdict(bn, noise)}")
            for m in ("coral", "dann"):
                if m in have:
                    d = get(m) - get("bn_only")
                    print(f"    {m + ' alignment loss':<22} bn_only  -> {m:<7} {d:+.4f}  {verdict(d, noise)}")
            print()
    else:
        print("\n(need both baseline and bn_only runs for the decomposition)")

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"per-run table written to {args.csv}")


def verdict(delta, noise):
    if not np.isfinite(noise):
        return ""
    if abs(delta) < noise:
        return "within noise"
    return "above noise" if delta > 0 else "WORSE, beyond noise"


if __name__ == "__main__":
    main()
