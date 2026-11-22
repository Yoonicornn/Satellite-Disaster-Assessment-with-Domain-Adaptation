#!/usr/bin/env bash
# Run the experiment grid: every method, every seed.
#
#   bash scripts/run_grid.sh                 # run it
#   PLAN_ONLY=1 bash scripts/run_grid.sh     # just print the plan and the time estimate
#
# On macOS start it with caffeinate, or the machine sleeps and the run stops:
#   nohup caffeinate -i bash scripts/run_grid.sh > logs/grid.log 2>&1 &
#
# The order is SEED-MAJOR: all methods for seed 1, then all methods for seed 3, and so on.
# That way an interrupted night still leaves a complete four-method comparison for one seed,
# instead of three baselines and nothing to compare them with.
#
# Finished runs are skipped, so re-launching the script continues where it stopped.
# Set SKIP_DONE=0 to force everything to run again.
#
#   SEEDS="3" EPOCHS=50 bash scripts/run_grid.sh
#   METHODS="coral dann" bash scripts/run_grid.sh
#   EXTRA="data.workers=2" bash scripts/run_grid.sh

set -u
export PYTHONUNBUFFERED=1     # so epoch lines reach the log file immediately

METHODS="${METHODS:-baseline coral dann oracle}"
SEEDS="${SEEDS:-1 3 14}"
EPOCHS="${EPOCHS:-50}"          # the SAME for every method, or the comparison is unfair
SKIP_DONE="${SKIP_DONE:-1}"
PLAN_ONLY="${PLAN_ONLY:-0}"
BASE_EPOCH_SEC="${BASE_EPOCH_SEC:-90}"   # fallback guess for the baseline, used until a run exists
EXTRA="${EXTRA:-}"              # more --set overrides, e.g. EXTRA="data.workers=2 optim.lr=1e-4"
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

# fail now rather than three hours in
if [ ! -f "crops/wind_64/manifest.csv" ]; then
    echo "crops/wind_64/manifest.csv not found. Run scripts/extract_crops.py first."
    exit 1
fi
python -c "import torch" 2>/dev/null || { echo "torch not importable - wrong environment?"; exit 1; }
for m in $METHODS; do
    [ -f "configs/${m}.yaml" ] || { echo "configs/${m}.yaml not found"; exit 1; }
done

# ---------------------------------------------------------------- plan and estimate
python - "$METHODS" "$SEEDS" "$EPOCHS" "$SKIP_DONE" "$BASE_EPOCH_SEC" <<'PY'
import glob, json, sys

methods, seeds, epochs, skip_done, base_sec = sys.argv[1].split(), sys.argv[2].split(), \
    int(sys.argv[3]), sys.argv[4] == "1", float(sys.argv[5])

# how long an epoch takes, measured from earlier runs when we have them
measured = {}
for f in glob.glob("runs/*/*/metrics.json"):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if d.get("seconds_per_epoch"):
        measured.setdefault(d["method"], []).append(d["seconds_per_epoch"])
measured = {m: sum(v) / len(v) for m, v in measured.items()}

# rough cost relative to the baseline: CORAL and DANN add a target pass, the oracle trains
# on source + target together
FALLBACK = {"baseline": 1.0, "coral": 2.2, "dann": 2.2, "oracle": 2.3}


def done(method, seed):
    return any(json.load(open(f)).get("epochs_run") for f in
               glob.glob(f"runs/{method}/seed{seed}_*/metrics.json"))


def per_epoch(method):
    if method in measured:
        return measured[method], "measured"
    if "baseline" in measured:
        return measured["baseline"] * FALLBACK.get(method, 2.0), "scaled"
    return base_sec * FALLBACK.get(method, 2.0), "guess"


total = 0.0
print(f"{'#':>3}  {'seed':<5}{'method':<10}{'status':<10}{'estimate':>10}  source")
i = 0
for seed in seeds:                       # seed-major: a finished seed is a usable comparison
    for m in methods:
        i += 1
        skip = skip_done and done(m, seed)
        sec, src = per_epoch(m)
        est = sec * epochs
        if not skip:
            total += est
        print(f"{i:>3}  {seed:<5}{m:<10}{'SKIP (done)' if skip else 'to run':<10}"
              f"{'-' if skip else f'{est / 3600:.1f} h':>10}  {src}")

h = int(total // 3600)
mm = int((total % 3600) // 60)
print(f"\nepochs per run : {epochs}")
print(f"estimated time : {h}h {mm:02d}m for the runs still to do")
print("note           : estimates are rough until at least one run of each method exists")
PY

if [ "$PLAN_ONLY" = "1" ]; then
    echo
    echo "PLAN_ONLY=1, nothing was run."
    exit 0
fi

# ---------------------------------------------------------------- run
echo
echo "started : $(date)"
echo

failed=""
completed=0
grid_start=$SECONDS

for seed in $SEEDS; do
    for m in $METHODS; do
        if [ "$SKIP_DONE" = "1" ] && compgen -G "runs/${m}/seed${seed}_*/metrics.json" > /dev/null; then
            echo "--- $m seed $seed   already finished, skipping"
            continue
        fi
        log="$LOG_DIR/${m}_seed${seed}.log"
        echo "--- $m seed $seed   started $(date '+%a %H:%M:%S')   -> $log"
        start=$SECONDS
        # shellcheck disable=SC2086  -- EXTRA must word-split into separate key=value args
        if python -m src.train --config "configs/${m}.yaml" \
                 --set run.seed="$seed" optim.epochs="$EPOCHS" $EXTRA > "$log" 2>&1; then
            mins=$(( (SECONDS - start) / 60 ))
            best=$(grep -m1 'best macro_f1' "$log" || true)
            echo "    done in ${mins} min   ${best}"
            completed=$((completed + 1))
        else
            echo "    FAILED - last lines of $log:"
            tail -5 "$log" | sed 's/^/      /'
            failed="$failed ${m}_seed${seed}"
        fi
    done
done

echo
echo "finished : $(date)   ($(( (SECONDS - grid_start) / 60 )) min, $completed run(s) completed)"
[ -n "$failed" ] && echo "failed   :$failed"

# ---------------------------------------------------------------- summary
echo
python - <<'PY'
import glob, json
rows = []
for f in sorted(glob.glob("runs/*/*/metrics.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    s = d["splits"]
    rows.append((d["method"], d["seed"], d.get("epochs_run"),
                 s["source_val"]["macro_f1"], s["target_test"]["macro_f1"], s["test"]["macro_f1"]))
if not rows:
    print("no finished runs yet")
else:
    rows.sort()
    print(f"{'method':10}{'seed':>5}{'ep':>5}{'source_val':>12}{'target_test':>13}{'test':>8}")
    for m, sd, ep, a, b, c in rows:
        print(f"{m:10}{sd:>5}{ep or 0:>5}{a:>12.4f}{b:>13.4f}{c:>8.4f}")
    print("\nmean over seeds (macro-F1)")
    methods = sorted({r[0] for r in rows})
    print(f"{'method':10}{'runs':>6}{'source_val':>12}{'target_test':>13}{'test':>8}")
    for m in methods:
        sel = [r for r in rows if r[0] == m]
        n = len(sel)
        print(f"{m:10}{n:>6}" + "".join(
            f"{sum(r[k] for r in sel) / n:>{w}.4f}" for k, w in ((3, 12), (4, 13), (5, 8))))
PY
