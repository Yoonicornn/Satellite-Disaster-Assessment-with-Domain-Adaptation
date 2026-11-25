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

**Changing the backbone was worth six times more.** Swapping ImageNet ResNet-34 for DINOv2
ViT-S/14 with the last two blocks fine-tuned gained **+0.116 macro-F1** on the unseen region.

The representation mattered. The alignment objectives did not. [Section 8](#8-results) has the
numbers.

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
│   ├── run_grid.sh            every method x every seed, with skip/resume
│   ├── summarize.py           runs/*/*/metrics.json -> mean +/- std table
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

The ResNet grid, 4 methods x 3 seeds:

```bash
METHODS="baseline bn_only coral dann" SEEDS="3 1 14" EPOCHS=25 \
  nohup caffeinate -i bash scripts/run_grid.sh > logs/grid.log 2>&1 &
```

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
python scripts/watch_runs.py          # live table of every run, refreshes every 30 s
tensorboard --logdir runs --port 6007
```

`watch_runs.py` reads `history.csv`, so it is not affected by the stdout buffering that makes a
redirected log lag behind.

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
cross-entropy, AdamW. DINOv2 runs resize crops to 126x126 inside the encoder.

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

### 8.7 Timing

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
grid, 6 DINOv2 backbone runs, 3 DINOv2 DA runs, aggregation and monitoring scripts, domain
probe, analysis notebook.

**Open**, in order of value per hour:

- [ ] **Seed ensemble.** Average the softmax of the three DINOv2 ft2 seeds. No training —
      the `predictions_*.csv` are already on disk. This is what the xView2 winners did.
- [ ] **Prior correction at inference.** Add `log(pi_target) - log(pi_source)` to the logits and
      re-score. No training. Directly attacks the miscalibration in 8.5.
- [ ] **DINOv2 at 224 on Colab.** We run at 126 (81 patch tokens); DINOv2 was trained at 224
      (256 tokens). The biggest untried lever, but it needs a GPU.
- [ ] More seeds for the DINOv2 DA runs (currently seed 3 only).
- [ ] Re-run `domain_probe.py` on the final checkpoints; the numbers in analysis.md §10.8 are
      from the archived bottleneck-512 runs.
