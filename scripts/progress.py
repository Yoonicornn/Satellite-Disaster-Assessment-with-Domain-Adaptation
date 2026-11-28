import glob, os, time
import pandas as pd

rows = []
for h in sorted(glob.glob("runs/*/*/history.csv")):
    g, r = h.split("/")[1], h.split("/")[2]
    try:
        d = pd.read_csv(h)
    except Exception:
        continue
    if d.empty:
        continue
    done = os.path.exists(os.path.join(os.path.dirname(h), "metrics.json"))
    age = (time.time() - os.path.getmtime(h)) / 60
    best = d["source_val/macro_f1"].max() if "source_val/macro_f1" in d else float("nan")
    at = d["source_val/macro_f1"].idxmax() if "source_val/macro_f1" in d else None
    tgt = d["target_test/macro_f1"].iloc[at] if at is not None and "target_test/macro_f1" in d else float("nan")
    rows.append(dict(group=g, run=r, ep=int(d["epoch"].max()) + 1,
                     state="done" if done else ("running" if age < 15 else f"stalled {age:.0f}m"),
                     best_src=best, tgt_at_best=tgt))

if not rows:
    raise SystemExit("no history.csv yet")
df = pd.DataFrame(rows)
w = max(len(x) for x in df["group"]) + 2
print(f"{'group':<{w}}{'ep':>4}{'state':>12}{'best source_val':>17}{'target_test':>13}")
for _, r in df.iterrows():
    print(f"{r['group']:<{w}}{r['ep']:>4}{r['state']:>12}{r['best_src']:>17.4f}{r['tgt_at_best']:>13.4f}")
