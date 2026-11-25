"""
Live view of every training run: which are finished, which is going, where it is.

    python scripts/watch_runs.py            # refresh every 30 s, Ctrl-C to stop
    python scripts/watch_runs.py --once     # one snapshot, no loop
    python scripts/watch_runs.py -i 10 --runs runs

Reads history.csv, which train.py writes every epoch, so it is immune to the stdout buffering
that makes a redirected log lag behind. A run counts as FINISHED when metrics.json exists,
RUNNING when its history.csv changed in the last few minutes, and STALLED otherwise.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

COLS = [("epoch", None), ("cls_loss", "train"), ("source_val/loss", "val-loss"),
        ("source_val/macro_f1", "src_val F1"), ("target_test/macro_f1", "target F1")]


def read_run(d: Path, stale_after: float):
    hist = d / "history.csv"
    if not hist.exists():
        return None
    try:
        h = pd.read_csv(hist)
    except Exception:
        return None                       # caught mid-write; it will be there next tick
    if h.empty:
        return None

    last = h.iloc[-1]
    done = (d / "metrics.json").exists()
    age = time.time() - hist.stat().st_mtime
    state = "done" if done else ("running" if age < stale_after else f"stalled {int(age/60)}m")

    row = {"run": f"{d.parent.name}/{d.name}", "state": state, "epochs": len(h)}
    for key, label in COLS:
        if label and key in h.columns:
            row[label] = float(last[key])
    row["epoch"] = int(last["epoch"]) + 1
    # seconds per epoch, for the estimate
    row["_sec"] = float(h["seconds"].tail(5).mean()) if "seconds" in h.columns else None
    return row


def snapshot(runs_dir: Path, stale_after: float, planned: int | None):
    rows = []
    for d in sorted(runs_dir.glob("*/*")):
        if d.is_dir():
            r = read_run(d, stale_after)
            if r:
                rows.append(r)
    return rows


def render(rows, planned):
    if not rows:
        return "no runs with a history.csv yet"

    width = max(len(r["run"]) for r in rows) + 2
    head = (f"{'run':<{width}}{'state':<11}{'epoch':>7}{'train':>9}{'val-loss':>10}"
            f"{'src_val F1':>12}{'target F1':>11}{'eta':>9}")
    out = [head, "-" * len(head)]
    for r in rows:
        total = planned or r["epoch"]
        left = max(0, total - r["epoch"])
        eta = (time.strftime("%H:%M", time.localtime(time.time() + left * r["_sec"]))
               if r["_sec"] and left and r["state"] == "running" else "-")
        cell = lambda k, w, f=".4f": (f"{r[k]:>{w}{f}}" if k in r else f"{'-':>{w}}")
        progress = "{}/{}".format(r["epoch"], total)
        out.append(
            f"{r['run']:<{width}}{r['state']:<11}{progress:>7}"
            + cell("train", 9) + cell("val-loss", 10)
            + cell("src_val F1", 12) + cell("target F1", 11) + f"{eta:>9}")

    done = sum(r["state"] == "done" for r in rows)
    out.append("")
    out.append(f"{done}/{len(rows)} finished   {time.strftime('%H:%M:%S')}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description="Live view of the training runs.")
    p.add_argument("--runs", default="runs")
    p.add_argument("-i", "--interval", type=float, default=30, help="seconds between refreshes")
    p.add_argument("--once", action="store_true", help="print one snapshot and exit")
    p.add_argument("--epochs", type=int, default=None,
                   help="planned epochs, for the eta column (default: read from the run)")
    p.add_argument("--stale-after", type=float, default=600,
                   help="seconds without a new epoch before a run is called stalled")
    args = p.parse_args()

    runs_dir = Path(args.runs)
    if not runs_dir.exists():
        raise SystemExit(f"{runs_dir} does not exist")

    planned = args.epochs
    if planned is None:                   # take it from any run's config
        for c in runs_dir.glob("*/*/config.yaml"):
            for line in c.read_text().splitlines():
                if line.strip().startswith("epochs:"):
                    planned = int(line.split(":")[1])
                    break
            if planned:
                break

    while True:
        text = render(snapshot(runs_dir, args.stale_after, planned), planned)
        if args.once:
            print(text)
            return
        print("\033[2J\033[H" + text, flush=True)   # clear screen, redraw
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print()
            return


if __name__ == "__main__":
    main()
