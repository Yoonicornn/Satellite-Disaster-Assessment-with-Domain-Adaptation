# Satellite-Disaster-Assessment-with-Domain-Adaptation

Final project (ECSE552). We classify the damage level of individual buildings from pre- and
post-disaster satellite images (xBD dataset), and test whether unsupervised domain adaptation
helps when the model is moved to a **new geographic region**.

- **Task**: 4-class building damage classification (no-damage, minor, major, destroyed).
- **Domains**: three US/Caribbean regions, all from wind disasters (tornadoes and hurricanes).
- **Adaptation methods**: CORAL, DANN, CDAN, plus a BatchNorm-only control.
- **Backbones**: ResNet-34 (ImageNet) and DINOv2 ViT-S/14 (self-supervised).

For the reasoning behind every decision and all measured numbers, see **[analysis.md](analysis.md)**.

## Short version of the result

**No domain-adaptation method helped, on either backbone.** CORAL, DANN and CDAN all scored at
or below the baseline. The only adaptation effect we could measure was BatchNorm statistics
adapting to the target, worth +0.019 macro-F1, and that happens for free inside CORAL and DANN
rather than coming from their loss.

**Two things did help, and neither is a DA method.**

| change | gain on caribbean (`test`) |
|---|---|
| DINOv2 ViT-S/14 instead of ResNet-34 | **+0.116** (3 seeds) |
| photometric augmentation, ResNet baseline | **+0.086** (1 seed, 19x the baseline sd) |
| best DA method | at or below baseline |

Best caribbean score so far is **0.3778** (`bn_only` + DINOv2 ft2 + augmentation), up from 0.3498.

**We now know *why* the DA methods fail, not just that they do.** The DANN/CDAN discriminator
sits at chance (domain loss 0.67-0.69 against ln 2 = 0.693) while an offline probe separates the
same features at 0.73 - so the discriminator is too weak, the features are not aligned. CORAL's
objective converges to ~0 but contributes only 3 % of the loss and leaves the features just as
separable. [Section 8.7](#87-augmentation) and analysis.md 10.13 have the numbers.

The representation and the input distribution mattered. The alignment objectives did not.

---

## 1. Pipeline

```
geotiffs/            raw xBD download (51 GB)
   |  scripts/create_subset.py          pick one disaster type, merge the xBD folders
   v
xbd_subset/wind/     5 events, full 1024x1024 tiles + 3 CSV manifests
   |  scripts/extract_crops.py          cut one 64x64 crop per building, assign the splits
   v
crops/wind_64/       5 numpy arrays + manifest.csv        <- what training reads
   |  src/train.py                      one method, one seed, one backbone
   v
runs/<group>/seed<N>_<timestamp>/       checkpoints, metrics, predictions, TensorBoard
   |  scripts/watch_runs.py             live view while training
   |  scripts/summarize.py              mean +/- std per configuration
   |  scripts/per_event.py              scores split by event
   |  src/domain_probe.py               how separable the domains are in feature space
   v
notebooks/04_experiment_analysis.ipynb  curves, tables, confusion matrices, prediction figures
```

Crops are cut once (82 seconds) instead of opening two 1024x1024 GeoTIFFs on every sample.

---

## 2. Folder structure

```
Satellite-Disaster-Assessment-with-Domain-Adaptation/
├── configs/
│   ├── base.yaml              shared settings for every run
│   ├── baseline.yaml          method configs, each overrides only what it needs
│   ├── baseline_dinov2.yaml   the DINOv2 backbone
│   ├── bn_only.yaml
│   ├── coral.yaml
│   ├── dann.yaml
│   ├── cdan.yaml
│   ├── oracle.yaml            not used in the final experiment
│   └── event_location.json    event name -> our location label
├── crops/wind_64/             output of extract_crops.py            (git-ignored)
├── geotiffs/                  raw xBD data                          (git-ignored)
├── notebooks/
│   ├── 01_eda.ipynb                    counts, class balance, metadata
│   ├── 02_event_map.ipynb              world map of the events
│   ├── 03_view_crops.ipynb             the crops, by class / domain / event
│   └── 04_experiment_analysis.ipynb    compare runs: curves, tables, errors, figures
├── reports/                   figures and tables for the report
├── runs/                      training outputs                      (git-ignored)
├── scripts/
│   ├── create_subset.py       raw xBD -> xbd_subset/<type>/
│   ├── extract_crops.py       xbd_subset -> crops/<type>_<size>/ + the splits
│   ├── run.sh                 ONE runner; experiments are named presets inside it
│   ├── summarize.py           runs/*/*/metrics.json -> mean +/- std table
│   ├── progress.py            per-run state, including runs still training
│   ├── watch_runs.py          live progress of every run
│   └── per_event.py           scores split by event (geography vs hazard)
├── src/
│   ├── config.py              YAML loading, _base_ inheritance, --set overrides
│   ├── config_schema.py       pydantic schema: the only place a config key is allowed
│   ├── metrics.py             macro-F1, per-class F1/recall, confusion matrix
│   ├── train.py               the training loop
│   ├── domain_probe.py        domain separability of a trained run's features
│   ├── algorithms/            one file per training method
│   │   ├── base.py            Algorithm interface + SourceOnly
│   │   ├── bn_only.py         the BatchNorm control
│   │   ├── coral.py
│   │   ├── dann.py
│   │   └── cdan.py
│   ├── data/crops.py          Dataset, augmentation, class weights, split wiring
│   └── models/siamese.py      the shared network, both backbones
├── xbd_subset/wind/           output of create_subset.py            (git-ignored)
├── analysis.md                decisions, measurements and findings
└── README.md
```

**Keep out of git**: `geotiffs/`, `xbd_subset/`, `crops/`, `runs/`.

All commands are run **from the project root**.

---

## 3. Environment

Conda environment `ecse552`, Python 3.10.

```bash
conda create -n ecse552 python=3.10
conda activate ecse552
conda install -c conda-forge pytorch torchvision rasterio scikit-learn pandas matplotlib \
                             pyyaml pydantic tqdm tensorboard plotly nbformat ipykernel timm
```

Install **everything from conda-forge**, including PyTorch. Mixing a pip `torch` with
conda-forge packages gives `OMP: Error #15` on macOS, because each ships its own OpenMP runtime.

`timm` is needed only for the DINOv2 backbone. It downloads the weights from HuggingFace on
first use (88 MB for ViT-S/14) and caches them.

**Augmentation needs albumentations, and there are two traps.**

```bash
conda install -c conda-forge albumentations        # gives 2.0.8
python -m pip install --no-deps simsimd stringzilla
python -c "import albumentations as A; print(A.__version__)"   # want 2.0.8
```

1. The conda-forge package does **not** pull in `simsimd` and `stringzilla`, which `albucore`
   imports. Without them `import albumentations` fails with `ModuleNotFoundError: simsimd`.
   Both are dependency-free wheels, so pip is safe here.
2. Pin **2.0.8**. The PyPI `albumentations` package stops there; the 2.2+ line ships as a
   separate package, **AlbumentationsX**, which installs under the same import name and renamed
   every `*_limit` argument to `*_range` with no deprecation. `build_albumentations()` raises a
   clear error if it sees 2.2+ rather than failing mid-epoch.

---

## 4. Getting the data

Source: <https://xview2.org/dataset> ("Full datasets with GEOTIFF metadata").

1. Download the 6 parts (~51 GB compressed) and verify each one:
   ```bash
   shasum -a 1 <file>      # compare with the SHA1 on the website
   ```
2. Merge and extract:
   ```bash
   cat xview2_geotiff.tgz.part-a? > xview2_geotiff.tgz
   tar -xvzf xview2_geotiff.tgz
   ```
3. Put the resulting `geotiffs/` in the project root. It holds `tier1`, `tier3`, `test` and
   `hold`, each with `images/` (GeoTIFF) and `labels/` (JSON).

File names are `<event>_<number>_<pre|post>_disaster.<tif|json>`. The damage label is the
`subtype` field in the **post-disaster JSON only**. Values: `no-damage`, `minor-damage`,
`major-damage`, `destroyed`, `un-classified`.

---

## 5. Running the pipeline

### Step 1 — build the subset (once)

```bash
python scripts/create_subset.py --disaster-type wind --location-map configs/event_location.json
```

### Step 2 — cut the crops and fix the splits (once)

```bash
python scripts/extract_crops.py --seed 8 --dry-run     # check the split balance
python scripts/extract_crops.py --seed 8               # the real run, ~82 s
```

Produces `crops/wind_64/` with five splits:

| Split          | Buildings | Used in training | Role                                           |
| -------------- | --------- | ---------------- | ---------------------------------------------- |
| `source_train` | 31,339    | yes, labelled    | training data                                  |
| `source_val`   | 5,699     | no               | picks the best epoch, never uses target labels |
| `target_train` | 41,272    | yes, unlabelled  | target batches for bn_only / CORAL / DANN / CDAN |
| `target_test`  | 7,920     | no               | evaluation only — the main result              |
| `test`         | 22,617    | no               | caribbean, held out, hardest case              |

Counts are after the Dataset drops 69 crops that are more than half no-data pixels.
The split is **fixed for every run**; `run.seed` changes only the model init and data order.

### Step 3 — train

`scripts/run.sh` is the only runner. Experiments are named presets inside it, so adding one is a
few lines in that file rather than a new script.

```bash
bash scripts/run.sh                       # list the experiments
bash scripts/run.sh aug_all --plan        # print the plan and the time estimate, run nothing
nohup caffeinate -i bash scripts/run.sh aug_all > logs/aug_all.log 2>&1 &
```

| preset | what |
|---|---|
| `grid` | the original grid, paper-default hyperparameters, 3 seeds |
| `aug_all` | every method + augmentation, both backbones |
| `aug_r34` / `aug_dinov2` | one backbone only |
| `aug_baselines` | baseline + augmentation, all seeds |
| `highlambda` | CORAL/DANN/CDAN at 5x the default lambda |
| `da_default` | CORAL/DANN/CDAN at the configs' own lambda |

Knobs: `SEEDS="1 3 14"`, `EPOCHS=25`, `EXTRA="key=value"`, `SKIP_DONE=0`, `RESUME=1`,
`WITH_AUG=1|mild` (adds augmentation to any preset and suffixes the run group so it never
lands in the same folder as the un-augmented twin).

The queue runs the cheap backbone first and skips any run that already has a `metrics.json`, so
an interrupted night leaves a complete comparison and re-running the same command resumes.
`RESUME=1` continues a part-finished run from its `last.pt` instead of restarting it.

A DINOv2 run (note the `run.group`, so it does not mix with the ResNet runs):

```bash
python -m src.train --config configs/baseline_dinov2.yaml \
  --set run.seed=3 model.trainable_blocks=2 run.group=baseline_dinov2_ft2
```

| Flag                      | Meaning                                                 |
| ------------------------- | ------------------------------------------------------- |
| `--set key.sub=value ...` | override config values (one `--set`, all keys after it) |
| `--resume`                | continue the latest run of this method and seed         |
| `--overwrite`             | delete a run folder of the same name and start again    |
| `--device cpu\|mps\|cuda` | default `auto`: cuda, then mps, then cpu                |

Useful model overrides:

| Key                      | Meaning                                                           |
| ------------------------ | ----------------------------------------------------------------- |
| `model.backbone`         | `resnet34` or `dinov2_vits14` / `dinov2_vitb14`                   |
| `model.input_size`       | resize before the encoder; DINOv2 needs a multiple of 14          |
| `model.trainable_blocks` | `-1` whole encoder, `0` frozen, `n` last n blocks                 |

Check a config without training:

```bash
python -m src.config --config configs/cdan.yaml --set run.seed=1
```

### Step 4 — monitor

```bash
python scripts/progress.py            # one row per run: epoch, done/running/stalled, best so far
python scripts/watch_runs.py          # live table, refreshes every 30 s
tensorboard --logdir runs --port 6007
```

Both read `history.csv`, so they are not affected by the stdout buffering that makes a redirected
log lag behind. Use `progress.py` while runs are still going — `summarize.py` reads
`metrics.json`, which is only written when a run **finishes**, so an in-flight or crashed run is
simply absent from its table.

To stop a queue, kill the script first or it will immediately start the next run:

```bash
pkill -f "scripts/run.sh" && pkill -f "src.train" && pkill -f caffeinate
pgrep -fl "src.train|scripts/run.sh|caffeinate" || echo "all stopped"
```

### Step 5 — read the results

```bash
python scripts/summarize.py --csv results.csv
```

One row per **run group**, so different configurations are never averaged together. It also
prints the `baseline -> bn_only -> coral/dann` decomposition for the ResNet grid.

```bash
python scripts/per_event.py                      # geography vs hazard type
python -m src.domain_probe --run runs/baseline/seed3_* --pairs source_train:target_train
```

Then open `notebooks/04_experiment_analysis.ipynb` for curves, confusion matrices and
prediction figures.

---

## 6. The runs

| Config             | Sees target images | Target loss        | Target labels | Role                    |
| ------------------ | ------------------ | ------------------ | ------------- | ----------------------- |
| `baseline`         | no                 | —                  | no            | the number to beat      |
| `bn_only`          | yes                | **zero**           | no            | control: BatchNorm only |
| `coral`            | yes                | covariance match   | no            | unsupervised DA         |
| `dann`             | yes                | adversarial        | no            | unsupervised DA         |
| `cdan`             | yes                | conditional adv.   | no            | unsupervised DA         |

`bn_only` exists because CORAL and DANN forward target batches in train mode, which updates
every BatchNorm layer's running statistics with target data. That is free adaptation (AdaBN)
and has nothing to do with their loss. The control forwards the same batches and adds a loss of
zero, so the two effects can be told apart:

```
baseline -> bn_only      what BatchNorm adaptation is worth on its own
bn_only  -> coral/dann   what the alignment loss actually adds
```

**A ViT needs no such control.** DINOv2 normalises with LayerNorm, which keeps no running
statistics, so nothing adapts for free there.

`oracle` (trained on labelled target data) was dropped. Its config is still in the repo.

---

## 7. Outputs of a run

| File                      | Contents                                                      |
| ------------------------- | ------------------------------------------------------------- |
| `config.yaml`             | the resolved config, for reproducing the run                  |
| `best.pt`                 | best epoch on `source_val` macro-F1                           |
| `last.pt`                 | written every epoch; what `--resume` reads                    |
| `history.csv`             | one row per epoch                                             |
| `metrics.json`            | final numbers for all three splits, backbone, timing          |
| `predictions_<split>.csv` | per-building prediction and class probabilities               |
| `tb/`                     | TensorBoard event file                                        |

Primary metric is **macro-F1**, because 74 % of the source buildings are no-damage and plain
accuracy would reward a model that ignores the damaged classes.

---

## 8. Results

**Setup.** 25 epochs, best checkpoint on `source_val` macro-F1, never on target labels.
Crops 64x64, stratified group split with seed 8, bottleneck 256, batch 128, weighted
cross-entropy, AdamW. DINOv2 runs resize crops to 126x126 inside the encoder. Sections 8.1-8.6
are **without** photometric augmentation; 8.7 adds it.

Regenerate with `python scripts/summarize.py`.

**Read 8.4 first.** The aggregate tables below average two very different kinds of shift;
8.4 pulls them apart and is the result worth quoting.

### 8.1 Backbones — macro-F1, mean ± std over 3 seeds (1, 3, 14)

| Configuration | source_val (us-central) | target_test (us-south) | test (caribbean) |
|---|---|---|---|
| ResNet-34, fine-tuned | 0.7116 ± 0.0039 | 0.5335 ± 0.0035 | 0.2168 ± 0.0045 |
| ResNet-34, frozen *(1 seed)* | 0.6087 | 0.4411 | 0.2106 |
| DINOv2 ViT-S/14, frozen | 0.7125 ± 0.0073 | 0.5133 ± 0.0027 | **0.3422** ± 0.0106 |
| **DINOv2 ViT-S/14, last 2 blocks** | **0.7553** ± 0.0046 | **0.5572** ± 0.0051 | 0.3323 ± 0.0151 |

DINOv2 with the last two blocks fine-tuned beats the ResNet baseline on **every** split:

| split | gain | in units of the ResNet baseline's sd |
|---|---|---|
| source_val | +0.0437 | 11.2× |
| target_test | +0.0237 | 6.7× |
| test | +0.1156 | 25.5× |

The frozen-ResNet row is the control that rules out "freezing is what helped" — freezing a
ResNet makes everything worse. Frozen against frozen, DINOv2 beats ResNet by +0.112, +0.075 and
+0.124.

### 8.2 Domain adaptation on ResNet-34 — mean ± std over 3 seeds

| Method | source_val | target_test | test |
|---|---|---|---|
| baseline | 0.7116 ± 0.0039 | 0.5335 ± 0.0035 | 0.2168 ± 0.0045 |
| **bn_only (control)** | **0.7207** ± 0.0028 | **0.5350** ± 0.0024 | **0.2356** ± 0.0080 |
| CORAL | 0.7166 ± 0.0030 | 0.5317 ± 0.0030 | 0.2235 ± 0.0105 |
| DANN | 0.7130 ± 0.0028 | 0.5320 ± 0.0042 | 0.2274 ± 0.0084 |

What each part contributes, against the baseline's own seed-to-seed spread:

| Split | baseline sd | BatchNorm (baseline → bn_only) | CORAL loss (bn_only → coral) | DANN loss (bn_only → dann) |
|---|---|---|---|---|
| source_val | 0.0039 | **+0.0091** above noise | −0.0041 worse | −0.0077 worse |
| target_test | 0.0035 | +0.0015 within noise | −0.0033 within noise | −0.0030 within noise |
| test | 0.0045 | **+0.0188** above noise | −0.0120 worse | −0.0082 worse |

On `source_val` and `test` the separation is complete, not just a difference of means — every
`bn_only` seed beats every `baseline` seed.

### 8.3 Domain adaptation on DINOv2 — seed 3 only

All four rows are seed 3, so this is a like-for-like comparison. σ is the DINOv2 ft2 baseline's
own 3-seed spread.

| Method | source_val | target_test | test |
|---|---|---|---|
| baseline | 0.7606 | 0.5519 | 0.3291 |
| CORAL | 0.7565 (−0.9σ) | 0.5407 (**−2.2σ**) | 0.3370 (+0.5σ) |
| DANN | 0.7484 (**−2.7σ**) | 0.5482 (−0.7σ) | 0.2873 (**−2.8σ**) |
| CDAN | 0.7460 (**−3.2σ**) | 0.5034 (**−9.5σ**) | 0.3171 (−0.8σ) |

No method beat the baseline anywhere. CDAN — the method designed for differing class
conditionals — was the worst. This replicates the ResNet result on a completely different
backbone, which makes the finding much stronger than a single-backbone result would be.

### 8.4 Domain gap — is it region or hazard?

The aggregate gap, for reference:

| Configuration | source → target | source → test |
|---|---|---|
| ResNet-34 baseline | −0.1781 (−25.0 %) | −0.4948 (−69.5 %) |
| ResNet-34 bn_only | −0.1857 (−25.8 %) | −0.4851 (−67.3 %) |
| DINOv2 frozen | −0.1992 (−28.0 %) | −0.3703 (−52.0 %) |
| DINOv2 ft2 | −0.1981 (−26.2 %) | −0.4230 (−56.0 %) |

**But these averages hide what is actually going on.** `scripts/per_event.py` scores
`target_test` separately for its two events — `tuscaloosa-tornado` (different state, same
hazard) and `hurricane-michael` (different state, different hazard). With `source_val` and the
caribbean `test` split that gives a near-factorial design. Every cell except `source_val` is
out-of-domain for the baseline, which never sees target images.

| region | hazard | ResNet-34 | DINOv2 ft2 |
|---|---|---|---|
| us-central | tornado | 0.7116 | 0.7553 |
| us-south | tornado (tuscaloosa) | 0.7128 | 0.7708 |
| us-south | hurricane (michael) | 0.4457 | 0.4462 |
| caribbean | hurricane (matthew) | 0.2168 | 0.3323 |

Each effect, with the other factor held fixed:

| effect | ResNet-34 | DINOv2 ft2 |
|---|---|---|
| geography, small — us-central → us-south, both tornado | +0.0012 | +0.0155 |
| **hazard** — tornado → hurricane, both us-south | **−0.2671** | **−0.3246** |
| **geography, large** — us-south → caribbean, both hurricane | **−0.2289** | **−0.1139** |

**1. Geographic shift between US states is nil.** Missouri/Oklahoma → Alabama costs +0.001 —
the baseline scores the same on a region it has never seen as on its own validation split. The
−0.178 aggregate gap above is entirely `hurricane-michael`.

**2. Hazard and large-scale geography both matter, by similar amounts.** −0.267 for changing
hazard inside the US, −0.229 for changing country at fixed hazard.

**3. The backbone fixed geography, not hazard.**

| effect | DINOv2 − ResNet |
|---|---|
| geography, small | +0.0143 |
| **geography, large** | **+0.1150** |
| hazard | **−0.0575** |

DINOv2's whole +0.116 on caribbean is geographic robustness; on the hazard axis it is slightly
*worse* than ResNet.

**4. Nothing improved the hurricane event.** Across every method and both backbones the best
score on `michael` is 0.4462 against the baseline's 0.4457 — a difference of +0.0005.

**5. Alignment helps where there is no shift and hurts where there is.** On DINOv2, split by
event:

| method | tuscaloosa (small shift) | michael (large shift) |
|---|---|---|
| CORAL | +0.0086 | −0.0129 |
| DANN | +0.0028 | −0.0031 |
| CDAN | +0.0089 | −0.0455 |

Two opposite effects averaging out is why the aggregate DA numbers looked like flat noise.

**Caveats.** `hurricane-matthew` is the only non-US wind event in xBD, so the −0.229 geographic
effect rests on one event. "Geography, large" bundles country, building materials, building
size (23 px against 41 px), varying GSD and a very different label distribution — realistic
regional transfer, not a pure coordinate change. Two cells of the design do not exist in xBD's
wind data: us-central hurricane and caribbean tornado.

### 8.5 The caribbean failure

Predicted class shares against the truth, on `test` (22,617 buildings), ResNet-34:

| Class | Actual | Baseline | CORAL | DANN |
|---|---|---|---|---|
| no-damage | 17.9 % | 27.1 % | 18.8 % | 18.0 % |
| minor-damage | **54.5 %** | **5.3 %** | 5.3 % | 5.6 % |
| major-damage | 12.0 % | 3.6 % | 7.6 % | 8.6 % |
| destroyed | **15.6 %** | **64.0 %** | 68.4 % | 67.8 % |

The model calls two-thirds of the island destroyed when a sixth of it is, and almost never
predicts minor-damage when it is the majority class. Of the 15,570 buildings every method gets
wrong, **10,844 are minor-damage** — 88 % of all minor-damage buildings in the split.

### 8.6 Error overlap on `test` — the gains are churn

| Method | correct | accuracy | fixed | broken | net |
|---|---|---|---|---|---|
| Baseline | 5,575 | 0.2465 | — | — | — |
| CORAL | 5,707 | 0.2523 | 1,042 | 910 | +132 |
| DANN | 5,686 | 0.2514 | 1,013 | 902 | +111 |

Every method wrong: 15,570 (68.8 %). Every method right: 4,377 (19.4 %). A net of +132 after
fixing 1,042 and breaking 910 is churn, not adaptation.

### 8.7 Augmentation

Added after the tables above. Geometric augmentation was always on; this adds **photometric**
augmentation (brightness/contrast, hue/saturation/value, gamma, blur, noise, coarse dropout) with
**one shared random draw for the pre/post pair**, so the two crops stay aligned. See analysis.md
4.3 for the settings and 10.12 for the full result.

All augmented runs are **seed 3 only**. The sd columns are the matching 3-seed baselines'.

**Effect of augmentation, seed 3 vs seed 3:**

| method | source_val | target_test | test (caribbean) |
|---|---|---|---|
| ResNet baseline | +0.0083 | +0.0076 | **+0.0862** (19 sd) |
| ResNet bn_only | −0.0111 | −0.0078 | **+0.0772** (10 sd) |
| ResNet CORAL | −0.0057 | +0.0084 | **+0.0511** (5 sd) |
| ResNet DANN | −0.0142 | +0.0015 | **+0.0701** (8 sd) |
| DINOv2 baseline | −0.0091 | −0.0077 | **+0.0430** (3 sd) |

It moves the **caribbean split only**. `source_val` and `target_test` shift by about ±0.01,
inside or barely at the noise floor. That is what photometric augmentation should do: caribbean
is the split with the largest appearance shift, and where appearance barely differs it buys
nothing.

**Ranked on caribbean, each against its own augmented baseline:**

| ResNet-34 | test | vs base | | DINOv2 ft2 | test | vs base |
|---|---|---|---|---|---|---|
| bn_only_aug | 0.3051 | +0.0046 | | bn_only_aug | **0.3778** | +0.0057 |
| baseline_aug | 0.3005 | — | | baseline_aug | 0.3721 | — |
| dann_aug | 0.2892 | −0.0113 | | coral_aug | 0.3497 | −0.0224 |
| coral_aug | 0.2694 | −0.0311 | | cdan_aug | 0.3184 | −0.0537 |
| cdan_aug | 0.2581 | −0.0424 | | dann_aug | 0.2975 | −0.0746 |

Plain baseline + augmentation tops both backbones. Every alignment method makes it worse, and
the only thing that edges past the baseline is `bn_only`, by an amount inside the noise. This is
the same ordering as the un-augmented results, so augmentation did not change the conclusion
about DA — it raised the floor underneath it.

**Caveat.** One seed. The ResNet caribbean effect is 19x the baseline's seed-to-seed spread, so
it is safe. The DINOv2 one is about 3x and carries the "new best result" claim, so seeds 1 and
14 are an open item before that goes in the report.

### 8.8 Timing

25 epochs on an Apple Silicon Mac (MPS), batch 128:

| Configuration | minutes/run |
|---|---|
| ResNet-34 baseline | 26 |
| ResNet-34 bn_only | 33 |
| ResNet-34 coral / dann | 48 |
| DINOv2 frozen @126 | 71 |
| DINOv2 ft2 @126 | 87 |
| DINOv2 ft2 + coral / dann / cdan | 152–154 |

---

## 9. Status

**Done**: subset creation, crops and splits, Dataset, five algorithms, two backbones, training
loop with checkpoints and resume, evaluation with per-building predictions, the 12-run ResNet
grid, 6 DINOv2 backbone runs, DINOv2 DA runs, CDAN, the high-lambda runs, photometric
augmentation and the 10-run augmented grid, the frozen-feature probe study, the mechanism
diagnosis and checkpoint-selection control (analysis.md 10.13-10.14), aggregation and
monitoring scripts,
analysis notebook.

**Open**, in order of value per hour:

- [ ] **Seeds 1 and 14 for the augmented runs** (~6.75 h). Everything in 8.7 is seed 3. The
      DINOv2 caribbean gain is ~3 sd and carries the headline, so it needs error bars.
      `SEEDS="1 14" bash scripts/run.sh aug_baselines`
- [ ] **Seed ensemble.** Average the softmax of the three DINOv2 ft2 seeds. No training —
      the `predictions_*.csv` are already on disk. This is what the xView2 winners did.
- [ ] **Prior correction at inference.** Add `log(pi_target) - log(pi_source)` to the logits and
      re-score. No training. Directly attacks the miscalibration in 8.5.
- [ ] **A stronger discriminator for DANN/CDAN.** Now justified by measurement, not a hunch:
      analysis.md 10.13 shows the current one sits at chance while a probe separates the same
      features at 0.73. Raising lambda is **not** the fix — the term is already 60-65 % of the loss.
- [ ] **DINOv2 at 224 on Colab.** We run at 126 (81 patch tokens); DINOv2 was trained at 224
      (256 tokens). The biggest untried lever, but it needs a GPU.
- [ ] Add `test` to `eval.track_splits` so the caribbean curve is recoverable per epoch; right
      now only `target_test` is logged every epoch.
