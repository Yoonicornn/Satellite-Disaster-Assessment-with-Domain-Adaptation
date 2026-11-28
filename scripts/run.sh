#!/usr/bin/env bash
# One runner for every experiment. Pick an experiment by name; the job list lives in the
# case block below, so a new experiment is a few lines here, not a new script.
#
#   bash scripts/run.sh                          list the experiments
#   bash scripts/run.sh aug_all --plan           print the plan, run nothing
#   nohup caffeinate -i bash scripts/run.sh aug_all > logs/aug_all.log 2>&1 &
#
# Knobs (environment variables):
#   SEEDS="1 2 3"        seeds to run           (default: per experiment)
#   EPOCHS=25            epochs per run
#   EXTRA="k=v k=v"      extra --set args added to every run
#   SKIP_DONE=0          redo runs that already finished (default 1 = skip them)
#
# The queue is ordered cheap-backbone-first and seed-major, so stopping it part way
# still leaves a complete comparison rather than a partial one. SKIP_DONE=1 means you
# can re-launch the same command later and it picks up where it stopped.

set -u

EXP="${1:-}"
PLAN_ONLY=0
[ "${2:-}" = "--plan" ] && PLAN_ONLY=1
[ "${PLAN:-0}" = "1" ] && PLAN_ONLY=1

EPOCHS="${EPOCHS:-25}"
EXTRA="${EXTRA:-}"
RESUME="${RESUME:-0}"              # 1 = continue a part-finished run from its last.pt
SKIP_DONE="${SKIP_DONE:-1}"
LOG_DIR="logs"

# ---------------------------------------------------------------- shared fragments
DINOV2="model.backbone=dinov2_vits14 model.input_size=126 model.trainable_blocks=2"

# full albumentations set, one shared draw per pre/post pair. No spaces in any value:
# these word-split into separate --set args.
AUG="data.aug.backend=albumentations data.aug.vflip=true data.aug.brightness_contrast=0.3"
AUG="$AUG data.aug.hue_sat_val=[10,20,10] data.aug.gamma=[80,120] data.aug.blur_sigma=[0,1.0]"
AUG="$AUG data.aug.noise_std=[0,0.05] data.aug.dropout=0.2"

# A deliberately mild photometric set, for the strength comparison. Geometric settings are
# left exactly as the existing runs have them, so the ONLY difference from the no-aug
# baseline is photometric. Changing geometry at the same time would confound the two.
AUG_MILD="data.aug.backend=albumentations data.aug.vflip=true"
AUG_MILD="$AUG_MILD data.aug.brightness_contrast=0.08 data.aug.hue_sat_val=[3,8,6]"
AUG_MILD="$AUG_MILD data.aug.gamma=[95,105] data.aug.noise_std=[0,0.01] data.aug.p_photo=0.25"

# Seeds already present in runs/baseline/, so an augmented arm is compared against the
# existing no-aug arm at the SAME seeds rather than a guessed 1 2 3.
detect_seeds() {
    local s
    s=$(ls -d runs/baseline/seed*/ 2>/dev/null | sed 's|.*/seed||; s|_.*||; s|/||' \
        | sort -un | tr '\n' ' ')
    echo "${s:-1 2 3}"
}

# Alignment strength. The configs ship the paper defaults (CORAL 40, DANN/CDAN 1.0).
# The `highlambda` experiment used x5. `moderate` uses x2: on a multiplicative knob the
# useful midpoint between x1 and x5 is the geometric one (~x2.2), not x3.
CORAL_LAMBDA="${CORAL_LAMBDA:-200}"
DANN_LMAX="${DANN_LMAX:-5}"
CDAN_LMAX="${CDAN_LMAX:-5}"

CORAL_MOD="${CORAL_MOD:-80}"        # x2 of 40
DANN_MOD="${DANN_MOD:-2}"           # x2 of 1.0
CDAN_MOD="${CDAN_MOD:-2}"

# ---------------------------------------------------------------- experiments
# Each job is:  group | config | extra --set args | rough minutes
# Keep the group name unique per configuration - summarize.py groups by it and never
# averages two different configurations together.
case "$EXP" in

  # tonight: every method, with the new augmentation, one seed
  aug_all)
    DEFAULT_SEEDS="3"
    JOBS=(
      "baseline_aug|baseline|$AUG|48"
      "bn_only_aug|bn_only|$AUG|48"
      "coral_aug|coral|$AUG|48"
      "dann_aug|dann|$AUG|48"
      "cdan_aug|cdan|$AUG|50"
      "baseline_dinov2_ft2_aug|baseline|$DINOV2 $AUG|153"
      "bn_only_dinov2_ft2_aug|bn_only|$DINOV2 $AUG|153"
      "coral_dinov2_ft2_aug|coral|$DINOV2 $AUG|153"
      "dann_dinov2_ft2_aug|dann|$DINOV2 $AUG|154"
      "cdan_dinov2_ft2_aug|cdan|$DINOV2 $AUG|152"
    ) ;;

  # the resnet half on its own (4h), if you want to stop there
  aug_r34)
    DEFAULT_SEEDS="3"
    JOBS=(
      "baseline_aug|baseline|$AUG|48"
      "bn_only_aug|bn_only|$AUG|48"
      "coral_aug|coral|$AUG|48"
      "dann_aug|dann|$AUG|48"
      "cdan_aug|cdan|$AUG|50"
    ) ;;

  # the dinov2 half on its own
  aug_dinov2)
    DEFAULT_SEEDS="3"
    JOBS=(
      "baseline_dinov2_ft2_aug|baseline|$DINOV2 $AUG|153"
      "bn_only_dinov2_ft2_aug|bn_only|$DINOV2 $AUG|153"
      "coral_dinov2_ft2_aug|coral|$DINOV2 $AUG|153"
      "dann_dinov2_ft2_aug|dann|$DINOV2 $AUG|154"
      "cdan_dinov2_ft2_aug|cdan|$DINOV2 $AUG|152"
    ) ;;

  # augmented baselines at 3 seeds: the like-for-like test of whether aug helps at all
  aug_baselines)
    DEFAULT_SEEDS="$(detect_seeds)"
    JOBS=(
      "baseline_aug|baseline|$AUG|48"
      "baseline_dinov2_ft2_aug|baseline|$DINOV2 $AUG|153"
    ) ;;

  # strength sweep: mild vs strong photometric, resnet34 baseline, all seeds.
  # The third arm (no augmentation) is the `baseline` group you already have, so this
  # gives a 3-point curve for the price of 2 arms.
  aug_strength)
    DEFAULT_SEEDS="$(detect_seeds)"
    JOBS=(
      "baseline_augmild|baseline|$AUG_MILD|48"
      "baseline_aug|baseline|$AUG|48"
    ) ;;

  # CORAL / DANN / CDAN at the hyperparameters the configs ship - the paper defaults,
  # i.e. the setup from before the high-lambda night. No lambda override at all.
  # Add WITH_AUG=1 to get these same settings with the new augmentation.
  da_default)
    DEFAULT_SEEDS="$(detect_seeds)"
    JOBS=(
      "coral|coral||48"
      "dann|dann||48"
      "cdan|cdan||50"
      "coral_dinov2_ft2|coral|$DINOV2|153"
      "dann_dinov2_ft2|dann|$DINOV2|154"
      "cdan_dinov2_ft2|cdan|$DINOV2|152"
    ) ;;

  da_default_r34)
    DEFAULT_SEEDS="$(detect_seeds)"
    JOBS=(
      "coral|coral||48"
      "dann|dann||48"
      "cdan|cdan||50"
    ) ;;

  # moderate alignment strength (x2 the paper default). Together with the configs'
  # defaults and the highlambda runs this makes a 3-point strength curve per method:
  #   CORAL  40 ->  80 -> 200       DANN/CDAN  1.0 -> 2 -> 5
  lambda_x2)
    DEFAULT_SEEDS="3"
    JOBS=(
      "coral_r34_lam${CORAL_MOD}|coral|method.lambda_weight=${CORAL_MOD}|48"
      "dann_r34_lmax${DANN_MOD}|dann|method.lambda_max=${DANN_MOD}|48"
      "cdan_r34_lmax${CDAN_MOD}|cdan|method.lambda_max=${CDAN_MOD}|50"
      "coral_dinov2_ft2_lam${CORAL_MOD}|coral|method.lambda_weight=${CORAL_MOD} $DINOV2|153"
      "dann_dinov2_ft2_lmax${DANN_MOD}|dann|method.lambda_max=${DANN_MOD} $DINOV2|154"
      "cdan_dinov2_ft2_lmax${CDAN_MOD}|cdan|method.lambda_max=${CDAN_MOD} $DINOV2|152"
    ) ;;

  # the resnet half of `moderate` on its own (2.4h)
  lambda_x2_r34)
    DEFAULT_SEEDS="3"
    JOBS=(
      "coral_r34_lam${CORAL_MOD}|coral|method.lambda_weight=${CORAL_MOD}|48"
      "dann_r34_lmax${DANN_MOD}|dann|method.lambda_max=${DANN_MOD}|48"
      "cdan_r34_lmax${CDAN_MOD}|cdan|method.lambda_max=${CDAN_MOD}|50"
    ) ;;

  # high alignment strength, no augmentation (what ran earlier this week)
  highlambda)
    DEFAULT_SEEDS="3"
    JOBS=(
      "coral_r34_lam${CORAL_LAMBDA}|coral|method.lambda_weight=${CORAL_LAMBDA}|48"
      "dann_r34_lmax${DANN_LMAX}|dann|method.lambda_max=${DANN_LMAX}|48"
      "cdan_r34_lmax${CDAN_LMAX}|cdan|method.lambda_max=${CDAN_LMAX}|50"
      "coral_dinov2_ft2_lam${CORAL_LAMBDA}|coral|method.lambda_weight=${CORAL_LAMBDA} $DINOV2|153"
      "dann_dinov2_ft2_lmax${DANN_LMAX}|dann|method.lambda_max=${DANN_LMAX} $DINOV2|154"
      "cdan_dinov2_ft2_lmax${CDAN_LMAX}|cdan|method.lambda_max=${CDAN_LMAX} $DINOV2|152"
    ) ;;

  # the original grid, no augmentation, paper-default hyperparameters
  grid)
    DEFAULT_SEEDS="1 2 3"
    JOBS=(
      "baseline|baseline||48"
      "bn_only|bn_only||48"
      "coral|coral||48"
      "dann|dann||48"
      "cdan|cdan||50"
    ) ;;

  *)
    echo "usage: bash scripts/run.sh <experiment> [--plan]"
    echo
    echo "  aug_all         every method + augmentation, both backbones   (10 runs)"
    echo "  aug_r34         every method + augmentation, resnet34 only    ( 5 runs)"
    echo "  aug_dinov2      every method + augmentation, dinov2 ft2 only  ( 5 runs)"
    echo "  aug_strength    mild vs strong photometric, resnet baseline   ( 2 x seeds)"
    echo "  aug_baselines   baseline + augmentation, both backbones       ( 2 x seeds)"
    echo "  da_default      CORAL/DANN/CDAN at the configs' defaults       ( 6 runs)"
    echo "  da_default_r34  same, resnet34 only                           ( 3 runs)"
    echo "  lambda_x2       CORAL/DANN/CDAN at x2 default                 ( 6 runs)"
    echo "  lambda_x2_r34   same, resnet34 only                           ( 3 runs)"
    echo "  highlambda      CORAL/DANN/CDAN at x5 default                 ( 6 runs)"
    echo "  grid            original grid, paper defaults, 3 seeds        (15 runs)"
    echo
    echo "knobs: SEEDS=\"1 3 14\"  EPOCHS=25  EXTRA=\"k=v\"  SKIP_DONE=0"
    echo "       WITH_AUG=1|mild   add augmentation to any experiment above"
    echo "       RESUME=1          continue a part-finished run instead of restarting it"
    exit 1 ;;
esac

SEEDS="${SEEDS:-$DEFAULT_SEEDS}"

# ---------------------------------------------------------------- WITH_AUG
# Run ANY experiment above with augmentation:  WITH_AUG=1 (strong) or WITH_AUG=mild.
# The group name gets a suffix so an augmented run never lands in the same folder as
# its un-augmented twin - summarize.py groups by folder and would otherwise average them.
WITH_AUG="${WITH_AUG:-0}"
if [ "$WITH_AUG" != "0" ]; then
    case "$WITH_AUG" in
      1|strong) sfx="_aug";     add="$AUG" ;;
      mild)     sfx="_augmild"; add="$AUG_MILD" ;;
      *) echo "WITH_AUG must be 0, 1, strong or mild (got '$WITH_AUG')"; exit 1 ;;
    esac
    NEW=()
    for j in "${JOBS[@]}"; do
        IFS='|' read -r g c e m <<< "$j"
        NEW+=("${g}${sfx}|${c}|${e} ${add}|${m}")
    done
    JOBS=("${NEW[@]}")
fi

# ---------------------------------------------------------------- checks
mkdir -p "$LOG_DIR"
[ -f "crops/wind_64/manifest.csv" ] || { echo "crops/wind_64/manifest.csv not found"; exit 1; }
python -c "import torch" 2>/dev/null || { echo "torch not importable - wrong environment?"; exit 1; }
if [[ " ${JOBS[*]} " == *albumentations* ]]; then
    python -c "import albumentations" 2>/dev/null \
      || { echo "albumentations not installed:  pip install albumentations"; exit 1; }
fi

# ---------------------------------------------------------------- plan
total=0
echo "experiment: $EXP   seeds: $SEEDS   epochs: $EPOCHS"
[ -n "$EXTRA" ] && echo "extra     : $EXTRA"
echo
printf "%-32s %-9s %-5s %6s\n" "run group" "config" "seed" "est"
for s in $SEEDS; do
  for j in "${JOBS[@]}"; do
    IFS='|' read -r group cfg extra mins <<< "$j"
    [ -f "configs/${cfg}.yaml" ] || { echo "configs/${cfg}.yaml missing"; exit 1; }
    if [ "$SKIP_DONE" = "1" ] && compgen -G "runs/${group}/seed${s}_*/metrics.json" >/dev/null; then
        printf "%-32s %-9s %-5s %6s\n" "$group" "$cfg" "$s" "SKIP"
    else
        printf "%-32s %-9s %-5s %5sm\n" "$group" "$cfg" "$s" "$mins"
        total=$((total + mins))
    fi
  done
done
echo
echo "estimated total: $((total / 60))h $((total % 60))m   (finishes about $(date -v+${total}M '+%a %H:%M' 2>/dev/null || date -d "+${total} minutes" '+%a %H:%M'))"

if [ "$PLAN_ONLY" = "1" ]; then echo; echo "--plan given, nothing was run."; exit 0; fi

# ---------------------------------------------------------------- run
echo
echo "started : $(date)"
failed=""
start_all=$SECONDS

for s in $SEEDS; do
  for j in "${JOBS[@]}"; do
    IFS='|' read -r group cfg extra mins <<< "$j"
    if [ "$SKIP_DONE" = "1" ] && compgen -G "runs/${group}/seed${s}_*/metrics.json" >/dev/null; then
        echo "--- $group seed$s   already finished, skipping"; continue
    fi
    log="$LOG_DIR/${group}_seed${s}.log"
    echo "--- $group seed$s   started $(date '+%a %H:%M:%S')   -> $log"
    t0=$SECONDS
    # shellcheck disable=SC2086  -- $extra and $EXTRA must word-split into separate args
    [ "$RESUME" = "1" ] && rflag="--resume" || rflag=""
    if PYTHONUNBUFFERED=1 python -m src.train --config "configs/${cfg}.yaml" $rflag \
         --set run.seed="$s" optim.epochs="$EPOCHS" run.group="$group" $extra $EXTRA \
         > "$log" 2>&1; then
        echo "    done in $(( (SECONDS - t0) / 60 )) min   $(tr '\r' '\n' < "$log" | grep -m1 'best macro_f1' || true)"
    else
        echo "    FAILED - last lines of $log:"; tr '\r' '\n' < "$log" | tail -5 | sed 's/^/      /'
        failed="$failed ${group}/seed${s}"
    fi
  done
done

echo
echo "finished: $(date)   ($(( (SECONDS - start_all) / 60 )) min)"
[ -n "$failed" ] && echo "failed  :$failed"

echo
python scripts/summarize.py 2>/dev/null || true
