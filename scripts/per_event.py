"""
Break a run's scores down by event, to separate geography from hazard type.

Our target domain (us-south) mixes two events:

    tuscaloosa-tornado   different place, SAME hazard as the source (tornado)
    hurricane-michael    different place, DIFFERENT hazard

So the target gap we report is geography AND hazard mixed together. Scoring each event
separately pulls them apart, using predictions that already exist - nothing is retrained.

    python scripts/per_event.py
    python scripts/per_event.py --runs runs --split target_test
    python scripts/per_event.py --groups baseline baseline_dinov2_ft2

Reads runs/<group>/<run>/predictions_<split>.csv and joins it to crops/wind_64/manifest.csv
on building_uid, which carries the event name.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


def score(d):
    return {
        "n": len(d),
        "macro_f1": f1_score(d["true_label"], d["pred_label"],
                             labels=range(4), average="macro", zero_division=0),
        "accuracy": (d["true_label"] == d["pred_label"]).mean(),
    }


def main():
    p = argparse.ArgumentParser(description="Per-event scores for finished runs.")
    p.add_argument("--runs", default="runs")
    p.add_argument("--crops", default="crops/wind_64")
    p.add_argument("--split", default="target_test")
    p.add_argument("--groups", nargs="*", default=None,
                   help="run groups to include; default is all of them")
    args = p.parse_args()

    man = pd.read_csv(Path(args.crops) / "manifest.csv",
                      usecols=["building_uid", "event", "split"])
    man["building_uid"] = man["building_uid"].astype(str)

    rows = []
    for f in sorted(glob.glob(f"{args.runs}/*/*/predictions_{args.split}.csv")):
        group = f.split("/")[1]
        if args.groups and group not in args.groups:
            continue
        seed = f.split("/")[2].split("_")[0]
        d = pd.read_csv(f)
        d["building_uid"] = d["building_uid"].astype(str)
        d = d.merge(man, on="building_uid", how="left")
        if d["event"].isna().any():
            print(f"  warning: {d['event'].isna().sum()} rows in {f} had no event match")

        rows.append({"group": group, "seed": seed, "event": "ALL", **score(d)})
        for ev, g in d.groupby("event"):
            rows.append({"group": group, "seed": seed, "event": ev, **score(g)})

    if not rows:
        raise SystemExit(f"no predictions_{args.split}.csv found under {args.runs}/*/*/")

    df = pd.DataFrame(rows)

    # mean over seeds, one row per group x event
    agg = (df.groupby(["group", "event"])
             .agg(seeds=("macro_f1", "size"), n=("n", "first"),
                  macro_f1=("macro_f1", "mean"), sd=("macro_f1", "std"))
             .reset_index())

    print(f"split: {args.split}   (mean over seeds)\n")
    width = max(len(g) for g in agg["group"]) + 2
    print(f"{'group':<{width}}{'event':<22}{'seeds':>6}{'n':>8}{'macro_f1':>11}{'sd':>9}")
    for _, r in agg.sort_values(["group", "event"]).iterrows():
        sd = f"{r['sd']:.4f}" if pd.notna(r["sd"]) else "-"
        print(f"{r['group']:<{width}}{r['event']:<22}{int(r['seeds']):>6}"
              f"{int(r['n']):>8,}{r['macro_f1']:>11.4f}{sd:>9}")

    # the comparison the split was built to answer
    events = set(agg["event"]) - {"ALL"}
    same = [e for e in events if "tornado" in e]
    diff = [e for e in events if "hurricane" in e]
    if same and diff:
        print(f"\nsame hazard as source (tornado): {', '.join(sorted(same))}")
        print(f"different hazard (hurricane)   : {', '.join(sorted(diff))}")
        print(f"\n{'group':<{width}}{'tornado':>10}{'hurricane':>11}{'difference':>12}")
        for g in sorted(set(agg["group"])):
            sub = agg[agg["group"] == g]
            a = sub[sub["event"].isin(same)]["macro_f1"].mean()
            b = sub[sub["event"].isin(diff)]["macro_f1"].mean()
            if pd.notna(a) and pd.notna(b):
                print(f"{g:<{width}}{a:>10.4f}{b:>11.4f}{a - b:>+12.4f}")
        print("\nA large positive difference means the hazard change costs more than the place")
        print("change, and the 'geographic' gap we report is partly a hazard gap.")
        print("A small difference means geography dominates and the confound is minor.")


if __name__ == "__main__":
    main()
