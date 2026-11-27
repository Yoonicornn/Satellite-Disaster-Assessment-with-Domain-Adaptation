#!/usr/bin/env bash
# Push CORAL, DANN and CDAN to high alignment strength, on both backbones.
#
#   nohup caffeinate -i bash scripts/run_highlambda.sh > logs/highlambda.log 2>&1 &
#   PLAN_ONLY=1 bash scripts/run_highlambda.sh      # print the plan and the estimate only
#
# Purpose: complete the alignment-strength tradeoff curve. For each method and backbone you
# already have lambda = 0 (the baseline) and the paper's default. This adds a high point, so
# the probe study can show whether domain separability actually falls, and the macro-F1 table
# can show what that costs.
#
# A run that collapses (macro-F1 near the floor, probe near 0.50) is a RESULT, not a failure:
# it is the endpoint of the curve, where the encoder has thrown away class information to
# fool the discriminator.
#
# ResNet runs go first. They are cheap, so an interrupted night still leaves a complete
# three-method comparison on one backbone.

set -u

SEED="${SEED:-3}"
EPOCHS="${EPOCHS:-25}"
CORAL_LAMBDA="${CORAL_LAMBDA:-200}"     # 5x the current 40, matching DANN/CDAN
DANN_LMAX="${DANN_LMAX:-5}"             # 5x the paper's 1.0
CDAN_LMAX="${CDAN_LMAX:-5}"
PLAN_ONLY="${PLAN_ONLY:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
LOG_DIR="logs"

DINOV2="model.backbone=dinov2_vits14 model.input_size=126 model.trainable_blocks=2"

# name | config | extra --set args | rough minutes
JOBS=(
  "coral_r34_lam${CORAL_LAMBDA}|coral|method.lambda_weight=${CORAL_LAMBDA}|48"
  "dann_r34_lmax${DANN_LMAX}|dann|method.lambda_max=${DANN_LMAX}|48"
  "cdan_r34_lmax${CDAN_LMAX}|cdan|method.lambda_max=${CDAN_LMAX}|50"
  "coral_dinov2_ft2_lam${CORAL_LAMBDA}|coral|method.lambda_weight=${CORAL_LAMBDA} ${DINOV2}|153"
  "dann_dinov2_ft2_lmax${DANN_LMAX}|dann|method.lambda_max=${DANN_LMAX} ${DINOV2}|154"
  "cdan_dinov2_ft2_lmax${CDAN_LMAX}|cdan|method.lambda_max=${CDAN_LMAX} ${DINOV2}|152"
)

mkdir -p "$LOG_DIR"
[ -f "crops/wind_64/manifest.csv" ] || { echo "crops/wind_64/manifest.csv not found"; exit 1; }
python -c "import torch" 2>/dev/null || { echo "torch not importable - wrong environment?"; exit 1; }

# ---------------------------------------------------------------- plan
total=0
echo "seed $SEED, $EPOCHS epochs"
printf "%-34s %-7s %-46s %6s\n" "run group" "config" "overrides" "est"
for j in "${JOBS[@]}"; do
    IFS='|' read -r group cfg extra mins <<< "$j"
    [ -f "configs/${cfg}.yaml" ] || { echo "configs/${cfg}.yaml missing"; exit 1; }
    if [ "$SKIP_DONE" = "1" ] && compgen -G "runs/${group}/seed${SEED}_*/metrics.json" >/dev/null; then
        printf "%-34s %-7s %-46s %6s\n" "$group" "$cfg" "$extra" "SKIP"
    else
        printf "%-34s %-7s %-46s %5sm\n" "$group" "$cfg" "$extra" "$mins"
        total=$((total + mins))
    fi
done
echo
echo "estimated total: $((total / 60))h $((total % 60))m   (finishes about $(date -v+${total}M '+%a %H:%M' 2>/dev/null || date -d "+${total} minutes" '+%a %H:%M'))"

if [ "$PLAN_ONLY" = "1" ]; then echo; echo "PLAN_ONLY=1, nothing was run."; exit 0; fi

# ---------------------------------------------------------------- run
echo
echo "started : $(date)"
failed=""
start_all=$SECONDS

for j in "${JOBS[@]}"; do
    IFS='|' read -r group cfg extra mins <<< "$j"
    if [ "$SKIP_DONE" = "1" ] && compgen -G "runs/${group}/seed${SEED}_*/metrics.json" >/dev/null; then
        echo "--- $group   already finished, skipping"; continue
    fi
    log="$LOG_DIR/${group}.log"
    echo "--- $group   started $(date '+%a %H:%M:%S')   -> $log"
    t0=$SECONDS
    # shellcheck disable=SC2086  -- $extra must word-split into separate key=value args
    if PYTHONUNBUFFERED=1 python -m src.train --config "configs/${cfg}.yaml" \
         --set run.seed="$SEED" optim.epochs="$EPOCHS" run.group="$group" $extra \
         > "$log" 2>&1; then
        echo "    done in $(( (SECONDS - t0) / 60 )) min   $(grep -m1 'best macro_f1' "$log" || true)"
    else
        echo "    FAILED - last lines of $log:"; tail -5 "$log" | sed 's/^/      /'
        failed="$failed $group"
    fi
done

echo
echo "finished: $(date)   ($(( (SECONDS - start_all) / 60 )) min)"
[ -n "$failed" ] && echo "failed  :$failed"

echo
python scripts/summarize.py 2>/dev/null || true
