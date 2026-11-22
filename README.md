# Satellite-Disaster-Assessment-with-Domain-Adaptation

Final project (ECSE552). We classify the damage level of individual buildings from pre- and
post-disaster satellite images (xBD dataset), and test whether unsupervised domain adaptation
helps when the model is moved to a **new geographic region**.

- **Task**: 4-class building damage classification (no-damage, minor, major, destroyed).
- **Domains**: three US/Caribbean regions, all from wind disasters (tornadoes and hurricanes).
- **Methods compared**: source-only baseline, a BatchNorm-only control, CORAL, DANN.

For the reasoning behind every decision and all measured numbers, see **[analysis.md](analysis.md)**.

**Short version of the result**: neither CORAL nor DANN improved transfer. The only thing that
helped was BatchNorm statistics adapting to the target, which happens for free inside CORAL and
DANN and has nothing to do with their loss. See [section 8](#8-results).

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
   |  src/train.py                      one method, one seed
   v
runs/<method>/seed<N>_<timestamp>/      checkpoints, metrics, predictions, TensorBoard
   |  scripts/summarize.py              mean +/- std across seeds
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
│   ├── bn_only.yaml
│   ├── coral.yaml
│   ├── dann.yaml
│   ├── oracle.yaml            not used in the final experiment
│   └── event_location.json    event name -> our location label
├── crops/wind_64/             output of extract_crops.py            (git-ignored)
├── geotiffs/                  raw xBD data                          (git-ignored)
├── notebooks/
│   ├── view_samples.ipynb              pre/post tiles with building outlines
│   ├── 01_eda.ipynb                    counts, class balance, metadata
│   ├── 02_event_map.ipynb              world map of the events
│   ├── 03_view_crops.ipynb             the crops, by class / domain / event
│   └── 04_experiment_analysis.ipynb    compare runs: curves, tables, errors, prediction figures
├── reports/                   figures and tables for the report
├── runs/                      training outputs                      (git-ignored)
├── scripts/
│   ├── create_subset.py       raw xBD -> xbd_subset/<type>/
│   ├── extract_crops.py       xbd_subset -> crops/<type>_<size>/ + the splits
│   ├── run_grid.sh            every method x every seed, with skip/resume
│   └── summarize.py           runs/*/*/metrics.json -> mean +/- std table
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
│   │   └── dann.py
│   ├── data/crops.py          Dataset, augmentation, class weights, split wiring
│   └── models/siamese.py      the shared network
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
                             pyyaml pydantic tqdm tensorboard plotly nbformat ipykernel
```

Install **everything from conda-forge**, including PyTorch. Mixing a pip `torch` with
conda-forge packages gives `OMP: Error #15` on macOS, because each ships its own OpenMP runtime.

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

Produces `xbd_subset/wind/` with the five wind events and three manifests: `buildings.csv`
(the one that matters), `images.csv`, `events.csv`.

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
| `target_train` | 41,272    | yes, unlabelled  | target batches for bn_only / CORAL / DANN      |
| `target_test`  | 7,920     | no               | evaluation only — the main result              |
| `test`         | 22,617    | no               | caribbean, held out, hardest case              |

Counts are after the Dataset drops 69 crops that are more than half no-data pixels.
The split is **fixed for every run**; `run.seed` changes only the model init and data order.

### Step 3 — train

The full grid, 4 methods x 3 seeds:

```bash
METHODS="baseline bn_only coral dann" SEEDS="3 1 14" EPOCHS=25 \
  nohup caffeinate -i bash scripts/run_grid.sh > logs/grid.log 2>&1 &
```

`PLAN_ONLY=1` prints the plan and a time estimate without running anything. Finished runs are
skipped, so re-launching continues where it stopped. About 8 hours for all twelve on an Apple
Silicon Mac.

One run at a time:

```bash
python -m src.train --config configs/baseline.yaml --set run.seed=3
```

| Flag                      | Meaning                                                 |
| ------------------------- | ------------------------------------------------------- |
| `--set key.sub=value ...` | override config values (one `--set`, all keys after it) |
| `--resume`                | continue the latest run of this method and seed         |
| `--overwrite`             | delete a run folder of the same name and start again    |
| `--device cpu\|mps\|cuda` | default `auto`: cuda, then mps, then cpu                |

Check a config without training:

```bash
python -m src.config --config configs/dann.yaml --set run.seed=1
```

### Step 4 — monitor

```bash
grep -E "^---|done in|FAILED" logs/grid.log     # which runs are done
wc -l runs/*/*/history.csv                      # 26 lines = finished
tensorboard --logdir runs --port 6007
```

### Step 5 — read the results

```bash
python scripts/summarize.py --csv results.csv
```

Prints every run, mean +/- std per method, and the decomposition
`baseline -> bn_only -> coral/dann`. Then open
`notebooks/04_experiment_analysis.ipynb` for curves, confusion matrices and prediction figures.

Domain separability of a finished run:

```bash
python -m src.domain_probe --run runs/baseline/seed3_* --pairs source_train:target_train
```

---

## 6. The four runs

| Config     | Sees target images | Target loss      | Target labels | Role                    |
| ---------- | ------------------ | ---------------- | ------------- | ----------------------- |
| `baseline` | no                 | —                | no            | the number to beat      |
| `bn_only`  | yes                | **zero**         | no            | control: BatchNorm only |
| `coral`    | yes                | covariance match | no            | unsupervised DA         |
| `dann`     | yes                | adversarial      | no            | unsupervised DA         |

`bn_only` exists because CORAL and DANN forward target batches in train mode, which updates every
BatchNorm layer's running statistics with target data. That is free adaptation (AdaBN) and has
nothing to do with their loss. The control forwards the same batches and adds a loss of zero, so
the two effects can be told apart:

```
baseline -> bn_only      what BatchNorm adaptation is worth on its own
bn_only  -> coral/dann   what the alignment loss actually adds
```

`oracle` (trained on labelled target data) was dropped from the final experiment. Its config is
still in the repo.

---

## 7. Outputs of a run

| File                      | Contents                                        |
| ------------------------- | ----------------------------------------------- |
| `config.yaml`             | the resolved config, for reproducing the run    |
| `best.pt`                 | best epoch on `source_val` macro-F1             |
| `last.pt`                 | written every epoch; what `--resume` reads      |
| `history.csv`             | one row per epoch                               |
| `metrics.json`            | final numbers for all three splits, plus timing |
| `predictions_<split>.csv` | per-building prediction and class probabilities |
| `tb/`                     | TensorBoard event file                          |

Primary metric is **macro-F1**, because 74 % of the source buildings are no-damage and plain
accuracy would reward a model that ignores the damaged classes.

---

## 8. Results

**Setup.** 3 seeds (1, 3, 14), 25 epochs, best checkpoint on `source_val` macro-F1.
Crops 64x64, stratified group split with seed 8, ResNet-34 with ImageNet weights, bottleneck 256,
batch 128, weighted cross-entropy, AdamW. No target labels are used anywhere.

Regenerate with `python scripts/summarize.py`.

### 8.1 Main table — macro-F1, mean ± std over 3 seeds

| Method                 | source_val (us-central) | target_test (us-south) | test (caribbean)    |
| ---------------------- | ----------------------- | ---------------------- | ------------------- |
| Baseline (source only) | 0.7116 ± 0.0039         | 0.5335 ± 0.0035        | 0.2168 ± 0.0045     |
| **bn_only (control)**  | **0.7207** ± 0.0028     | **0.5350** ± 0.0024    | **0.2356** ± 0.0080 |
| CORAL                  | 0.7166 ± 0.0030         | 0.5317 ± 0.0030        | 0.2235 ± 0.0105     |
| DANN                   | 0.7130 ± 0.0028         | 0.5320 ± 0.0042        | 0.2274 ± 0.0084     |

### 8.2 What each part contributes

The baseline's own seed-to-seed spread is the bar. Anything smaller is not a result.

| Split       | baseline sd | BatchNorm (baseline → bn_only) | CORAL loss (bn_only → coral) | DANN loss (bn_only → dann) |
| ----------- | ----------- | ------------------------------ | ---------------------------- | -------------------------- |
| source_val  | 0.0039      | **+0.0091** above noise        | −0.0041 worse                | −0.0077 worse              |
| target_test | 0.0035      | +0.0015 within noise           | −0.0033 within noise         | −0.0030 within noise       |
| test        | 0.0045      | **+0.0188** above noise        | −0.0120 worse                | −0.0082 worse              |

On `source_val` and `test` the separation is complete, not just a difference of means — every
`bn_only` seed beats every `baseline` seed.

### 8.3 Domain gap — what is lost when the region changes

| Method   | source → target   | source → test     |
| -------- | ----------------- | ----------------- |
| Baseline | −0.1781 (−25.0 %) | −0.4948 (−69.5 %) |
| bn_only  | −0.1857 (−25.8 %) | −0.4851 (−67.3 %) |
| CORAL    | −0.1849 (−25.8 %) | −0.4931 (−68.8 %) |
| DANN     | −0.1810 (−25.4 %) | −0.4856 (−68.1 %) |

No method meaningfully narrowed either gap.

### 8.4 The caribbean failure

Predicted class shares against the truth, on `test` (22,617 buildings):

| Class        | Actual     | Baseline   | CORAL  | DANN   |
| ------------ | ---------- | ---------- | ------ | ------ |
| no-damage    | 17.9 %     | 27.1 %     | 18.8 % | 18.0 % |
| minor-damage | **54.5 %** | **5.3 %**  | 5.3 %  | 5.6 %  |
| major-damage | 12.0 %     | 3.6 %      | 7.6 %  | 8.6 %  |
| destroyed    | **15.6 %** | **64.0 %** | 68.4 % | 67.8 % |

The model calls two-thirds of the island destroyed when a sixth of it is. Of the 15,570
buildings every method gets wrong, **10,844 are minor-damage** — 88 % of all minor-damage
buildings in the split.

### 8.5 Error overlap on `test`

A method can raise macro-F1 while breaking nearly as many buildings as it fixes.

| Method   | correct | accuracy | fixed | broken | net  |
| -------- | ------- | -------- | ----- | ------ | ---- |
| Baseline | 5,575   | 0.2465   | —     | —      | —    |
| CORAL    | 5,707   | 0.2523   | 1,042 | 910    | +132 |
| DANN     | 5,686   | 0.2514   | 1,013 | 902    | +111 |

Every method wrong: 15,570 (68.8 %). Every method right: 4,377 (19.4 %).

### 8.6 Run log

All twelve finished, 25 epochs each.

| Method   | seed 1 | seed 3 | seed 14 | minutes/run |
| -------- | ------ | ------ | ------- | ----------- |
| baseline | 0.5376 | 0.5316 | 0.5314  | 26          |
| bn_only  | 0.5334 | 0.5339 | 0.5378  | 33          |
| coral    | 0.5350 | 0.5313 | 0.5289  | 48          |
| dann     | 0.5364 | 0.5280 | 0.5318  | 48          |

(`target_test` macro-F1.)

---

## 9. Status

**Done**: subset creation, crops and splits, Dataset, model, four algorithms, training loop with
checkpoints and resume, evaluation with per-building predictions, the full 12-run grid,
aggregation script, domain probe, analysis notebook.
