# Satellite-Disaster-Assessment-with-Domain-Adaptation

Final project (ECSE552). We classify the damage level of individual buildings from pre- and
post-disaster satellite images (xBD dataset), and test whether domain adaptation helps when the
model is moved to a **new geographic region**.

- **Task**: 4-class building damage classification (no-damage, minor, major, destroyed).
- **Domains**: three US/Caribbean regions, taken from wind disasters (tornadoes and hurricanes).
- **Methods compared**: source-only baseline, CORAL, DANN, and a target-supervised upper bound.

This README describes the project **as it is now**: the folder structure, how the data was prepared, and the commands to run. Training code is not written yet (see [Next steps](#next-steps)).

For the reasoning behind every decision (why wind, why these splits, why this architecture),
see **[analysis.md](analysis.md)**.

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
```

The crop step exists because opening two 1024x1024 GeoTIFFs for every building would dominate
training time. It is paid once (82 seconds) instead of every epoch.

---

## 2. Folder structure

```
Satellite-Disaster-Assessment-with-Domain-Adaptation/
├── configs/
│   ├── base.yaml              shared settings for every run
│   ├── baseline.yaml          method configs, each one overrides only what it needs
│   ├── coral.yaml
│   ├── dann.yaml
│   ├── oracle.yaml
│   └── event_location.json    event name -> our location label
├── crops/wind_64/             output of extract_crops.py            (git-ignored)
├── geotiffs/                  raw xBD data                          (git-ignored)
├── notebooks/
│   ├── view_samples.ipynb              pre/post tiles with building outlines
│   ├── 01_eda.ipynb                    counts, class balance, metadata, subset planner
│   ├── 02_event_map.ipynb              world map of the events
│   ├── 03_view_crops.ipynb             the crops, by class / domain / event
|   └── 04_experiment_analysis.ipynb    compare runs: curves, tables, errors, display predicted versus ground true image
├── reports/eda/               figures and tables for the report
├── runs/                      training outputs                      (git-ignored)
├── scripts/
│   ├── create_subset.py       raw xBD -> xbd_subset/<type>/
│   └── extract_crops.py       xbd_subset -> crops/<type>_<size>/ + the splits
├── src/
│   ├── config.py              YAML loading, _base_ inheritance, --set overrides
│   ├── config_schema.py       pydantic schema: the only place a config key is allowed
│   ├── metrics.py             macro-F1, per-class F1/recall, confusion matrix
│   ├── train.py               the training loop
│   ├── algorithms/            one file per training method
│   │   ├── base.py            Algorithm interface + SourceOnly
│   │   ├── coral.py
│   │   └── dann.py
│   ├── data/crops.py          Dataset, augmentation, class weights, split wiring
│   └── models/siamese.py      the shared network
├── xbd_subset/wind/           output of create_subset.py            (git-ignored)
├── ai_build.md                decisions, measurements and rationale
└── README.md
```

**Keep out of git** (too big or can be rebuilt): `geotiffs/`, `xbd_subset/`, `cache/`, `runs/`. Add them to `.gitignore`.

All commands below are run **from the project root** (the folder that contains `scripts/`, `notebooks/`, ...).

---

## 3. Environment

Check list of libraries in requirement.txt
OR
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

## 4. The data (xBD)

Source: https://xview2.org/dataset (xView2 / xBD, "Full datasets with GEOTIFF metadata").

### 4.1 Download and extract (done once)

1. Download the 6 parts (`part 1/aa` ... `part 6/af`, about 51 GB compressed in total).
2. Check each file with the SHA1 given on the website (run in the folder with the files):
   ```bash
   shasum -a 1 <file name>
   ```
   The result must be the same as the SHA1 on the page. If not, download that part again.
3. Merge the parts, then extract (the file names are the ones written on the website, section 3):
   ```bash
   cat xview2_geotiff.tgz.part-a? > xview2_geotiff.tgz
   tar -xvzf xview2_geotiff.tgz
   ```
   You need free disk space for the merged file (about 51 GB) and for the extracted data.
4. Put the extracted `geotiffs/` folder in the project root. After the extraction the parts and the merged `.tgz` can be deleted.

### 4.2 What is in `geotiffs/`

- Four folders: `tier1`, `tier3`, `test`, `hold`. Each has `images/` (GeoTIFF) and `labels/` (JSON).
- File names: `<event>_<number>_<pre|post>_disaster.<tif|json>`, for example `joplin-tornado_00000060_post_disaster.tif`. An **event** is one disaster (for example `joplin-tornado`).
- Every image has a **pre-disaster** and a **post-disaster** version, and a label file for each.
- About the folders (source: a research paper that uses xBD, not the official xView2 documentation): `tier1` + `tier3` = training set, `hold` = validation set, `test` = test set. Some events are only in `tier3`, and some are in `tier1` + `test`.

### 4.3 Label JSON files

- `features.xy`: list of buildings. Each building has a polygon (`wkt`, in **pixel** coordinates) and `properties` (`feature_type`, `uid`, `subtype`).
- `features.lng_lat`: the same polygons in longitude/latitude.
- `metadata`: sensor, capture date, off-nadir angle, sun elevation/azimuth, `disaster`, `disaster_type`, image width/height, `catalog_id`, and others.
- The **damage label is `subtype`** and it exists **only in the post-disaster file**. Values found in the data: `no-damage`, `minor-damage`, `major-damage`, `destroyed`, `un-classified`. The pre-disaster file has the building outlines but no damage label.
- Disaster types found in the data (`disaster_type`): fire, flooding, wind, volcano, tsunami, earthquake.

### 4.4 What we found in the EDA (tier1 + tier3 + test)

- 10,101 image pairs, 370,976 buildings, 19 events (`hold` was not included in the EDA).
- Buildings per damage class: no-damage 275,062 (74.1%), minor 30,522, major 25,299, destroyed 27,337, un-classified 12,756. The classes are very imbalanced, so we use per-class recall and macro-F1, not only accuracy.
- The class mix is very different from one event to another (for example `hurricane-matthew` has 52% minor-damage and 18% no-damage, `mexico-earthquake` has 99% no-damage).
- Many fire images have no building at all (48% to 74% of the images in 4 fire events).
- Some images have black areas (empty pixels, probably where the satellite scene does not cover the whole tile). We have not measured how common this is yet.

---

## 5. Getting the data

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
`subtype` field and exists **only in the post-disaster JSON**.

---

## 6. Running the pipeline

### Step 1 — build the subset (once)

```bash
python scripts/create_subset.py --disaster-type wind --location-map configs/event_location.json
```

Produces `xbd_subset/wind/` with the five wind events and three manifests: `buildings.csv`
(one row per building, the one that matters), `images.csv`, `events.csv`.

Add `--dry-run` to see what would be copied, or `--tar` to also write `xbd_subset/wind.tar.gz`
for sharing through Google Drive.

### Step 2 — cut the crops and fix the splits (once)

```bash
python scripts/extract_crops.py --seed 8 --dry-run     # check the split balance
python scripts/extract_crops.py --seed 8               # the real run, ~82 s
```

Produces `crops/wind_64/` with five splits:

| Split          | Buildings | Role                                           |
| -------------- | --------- | ---------------------------------------------- |
| `source_train` | 31,360    | labelled training data                         |
| `source_val`   | 5,705     | picks the best epoch, never uses target labels |
| `target_train` | 41,299    | **unlabelled** during training (CORAL / DANN)  |
| `target_test`  | 7,924     | evaluation only — the main result              |
| `test`         | 22,628    | caribbean, held out, evaluation only           |

`--size 128` writes a separate `crops/wind_128/` if you want to compare crop sizes.

### Step 3 — look at the crops (optional)

Open `notebooks/03_view_crops.ipynb`. Section 6 (the same damage class across the three
domains) is the figure to use in the report.

### Step 4 — train

```bash
python -m src.train --config configs/baseline.yaml --set run.seed=0
python -m src.train --config configs/coral.yaml    --set run.seed=0
python -m src.train --config configs/dann.yaml     --set run.seed=0
python -m src.train --config configs/oracle.yaml   --set run.seed=0
```

The full grid, 4 methods x 3 seeds:

```bash
for m in baseline coral dann oracle; do
  for s in 0 1 2; do
    python -m src.train --config configs/$m.yaml --set run.seed=$s
  done
done
```

Each run writes to `runs/<method>/seed<N>_<timestamp>/`, so nothing is ever overwritten.

Useful flags:

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

### Step 5 — monitor

```bash
tensorboard --logdir runs --port 6007
```

Runs appear grouped by method. Everything is also written to `history.csv` in each run folder,
so TensorBoard is optional.

---

## 7. Outputs of a run

| File                      | Contents                                        |
| ------------------------- | ----------------------------------------------- |
| `config.yaml`             | the resolved config, for reproducing the run    |
| `best.pt`                 | best epoch on `source_val` macro-F1             |
| `last.pt`                 | written every epoch; what `--resume` reads      |
| `history.csv`             | one row per epoch                               |
| `metrics.json`            | final numbers for all three splits              |
| `predictions_<split>.csv` | per-building prediction and class probabilities |
| `tb/`                     | TensorBoard event file                          |

Primary metric is **macro-F1**, because 74 % of the buildings are no-damage and plain accuracy
would reward a model that ignores the damaged classes.

---

### 8. Results

Setup. Macro-F1 from the best checkpoint on source_val. Crops 64x64, stratified group split with seed 8, ResNet-34 with ImageNet weights, weighted cross-entropy, AdamW. Target labels are used only by the oracle row.

Regenerate every number below with notebooks/04_experiment_analysis.ipynb, section 6.

7.1 Main table — macro-F1
Method source_val (us-central) target_test (us-south) test (caribbean) Runs
Baseline (source only) 0.7242 0.5458 0.2308 1 seed, 50 epochs (preliminary)
CORAL — — —
DANN — — —
Oracle (target labels) — — —

Fill in as mean ± std once three seeds are done. The baseline row is a single run at an epoch budget that is not final, so treat it as a sanity check, not a result.

7.2 Domain gap — what is lost when the region changes
Method source → target source → test
Baseline −0.178 (−25 %) −0.493 (−68 %)
CORAL — —
DANN — —
Oracle — —

Less negative is better: it means the method kept more of its source performance on the new region. This is the table the project is actually about.

7.3 Per-class F1

target_test:

Method no-damage minor major destroyed
Baseline 0.821 0.453 0.375 0.534
CORAL — — — —
DANN — — — —
Oracle — — — —

test (caribbean):

Method no-damage minor major destroyed
Baseline 0.308 0.142 0.111 0.362
CORAL — — — —
DANN — — — —
Oracle — — — —
7.4 Error overlap against the baseline

From section 11 of the analysis notebook. A method can raise macro-F1 while breaking nearly as many buildings as it fixes, so report both.

Method split fixed broken net
CORAL target_test — — —
DANN target_test — — —
CORAL test — — —
DANN test — — —

---

## 9. Status

Done: subset creation, crop extraction and splits, Dataset, model, all four algorithms,
training loop with checkpoints and resume, evaluation with per-building predictions.

Open:

- [ ] Choose the final `optim.epochs` from a long baseline run, then use it for all 12 runs.
- [ ] Run the full grid (Colab recommended; the Mac takes about 15 hours at 25 epochs).
- [ ] Prediction figures notebook (ground truth vs predicted, from `predictions_*.csv`).
- [ ] Aggregation script: mean ± std across seeds, for the results table.
