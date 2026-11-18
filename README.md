# Satellite-Disaster-Assessment-with-Domain-Adaptation

Final project (ECSE552). The goal is to classify the damage level of buildings from pre- and post-disaster satellite images (xBD dataset), and to test domain adaptation methods (transfer learning, DANN, CORAL).

This README describes the project **as it is now**: the folder structure, how the data was prepared, and the commands to run. Training code is not written yet (see [Next steps](#next-steps)).

---

## 1. Folder structure

```
Satellite-Disaster-Assessment-with-Domain-Adaptation/
├── cache/                     # big tables made by 01_eda.ipynb (can be deleted, rebuilt on the next run)
├── configs/
│   └── event_location.json    # event name -> location name (our own labels, see section 5)
├── geotiffs/                  # RAW xBD data, as extracted from the download
│   ├── hold/  {images, labels}
│   ├── test/  {images, labels}
│   ├── tier1/ {images, labels}
│   └── tier3/ {images, labels}
├── notebooks/
│   ├── view_samples.ipynb     # show pre/post image pairs and building outlines
│   ├── 01_eda.ipynb           # data exploration: counts, class balance, metadata, subset planner
│   └── 02_event_map.ipynb     # world map of the events (longitude/latitude)
├── reports/
│   └── eda/                   # tables (.csv), figures (.png) and the map (.html) made by the notebooks
├── runs/                      # training outputs: checkpoints, metrics, TensorBoard logs
├── scripts/
│   └── create_subset.py       # extracts one disaster type from geotiffs/ into xbd_subset/
├── src/                       # pipeline code (Python modules, run from the terminal). Folders are ready, the code is planned (see section 7)
│   ├── algorithms/            # training methods: transfer learning, DANN, CORAL
│   ├── data/                  # reading labels, Dataset class
│   ├── metrics/               # evaluation metrics
│   └── models/                # network architectures
├── xbd_subset/
│   └── wind/                  # OUTPUT of create_subset.py (our working subset)
│       ├── hurricane-matthew/ {images, labels}
│       ├── hurricane-michael/ {images, labels}
│       ├── joplin-tornado/    {images, labels}
│       ├── moore-tornado/     {images, labels}
│       ├── tuscaloosa-tornado/{images, labels}
│       ├── buildings.csv      # one row per building (main file for the dataloader)
│       ├── images.csv         # one row per pre/post image pair
│       ├── events.csv         # one row per event
│       └── subset_info.json   # command used, counts, label mapping
└── README.md
```

**Keep out of git** (too big or can be rebuilt): `geotiffs/`, `xbd_subset/`, `cache/`, `runs/`. Add them to `.gitignore`.

All commands below are run **from the project root** (the folder that contains `scripts/`, `notebooks/`, ...).

---

## 2. Environment

Check list of libraries in requirement.txt

---

## 3. The data (xBD)

Source: https://xview2.org/dataset (xView2 / xBD, "Full datasets with GEOTIFF metadata").

### 3.1 Download and extract (done once)

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

### 3.2 What is in `geotiffs/`

- Four folders: `tier1`, `tier3`, `test`, `hold`. Each has `images/` (GeoTIFF) and `labels/` (JSON).
- File names: `<event>_<number>_<pre|post>_disaster.<tif|json>`, for example `joplin-tornado_00000060_post_disaster.tif`. An **event** is one disaster (for example `joplin-tornado`).
- Every image has a **pre-disaster** and a **post-disaster** version, and a label file for each.
- About the folders (source: a research paper that uses xBD, not the official xView2 documentation): `tier1` + `tier3` = training set, `hold` = validation set, `test` = test set. Some events are only in `tier3`, and some are in `tier1` + `test`.

### 3.3 Label JSON files

- `features.xy`: list of buildings. Each building has a polygon (`wkt`, in **pixel** coordinates) and `properties` (`feature_type`, `uid`, `subtype`).
- `features.lng_lat`: the same polygons in longitude/latitude.
- `metadata`: sensor, capture date, off-nadir angle, sun elevation/azimuth, `disaster`, `disaster_type`, image width/height, `catalog_id`, and others.
- The **damage label is `subtype`** and it exists **only in the post-disaster file**. Values found in the data: `no-damage`, `minor-damage`, `major-damage`, `destroyed`, `un-classified`. The pre-disaster file has the building outlines but no damage label.
- Disaster types found in the data (`disaster_type`): fire, flooding, wind, volcano, tsunami, earthquake.

### 3.4 What we found in the EDA (tier1 + tier3 + test)

- 10,101 image pairs, 370,976 buildings, 19 events (`hold` was not included in the EDA).
- Buildings per damage class: no-damage 275,062 (74.1%), minor 30,522, major 25,299, destroyed 27,337, un-classified 12,756. The classes are very imbalanced, so we use per-class recall and macro-F1, not only accuracy.
- The class mix is very different from one event to another (for example `hurricane-matthew` has 52% minor-damage and 18% no-damage, `mexico-earthquake` has 99% no-damage).
- Many fire images have no building at all (48% to 74% of the images in 4 fire events).
- Some images have black areas (empty pixels, probably where the satellite scene does not cover the whole tile). We have not measured how common this is yet.

---

## 4. Step by step: from the raw data to the subset

### Step 1. Look at the data (optional but recommended)

- `notebooks/view_samples.ipynb`: shows pre/post pairs. The paths are relative to the notebook (`../geotiffs/...`).
- `notebooks/01_eda.ipynb`: counts per event, disaster type and split; class balance; building sizes; metadata per event; sample images; a subset planner at the end.
  - Run the cells in order. The first run reads every label file and can take a few minutes. The result is saved in `cache/`, so the next runs are fast. Set `REBUILD = True` if the data changed.
  - The tables and figures are saved in `reports/eda/`.

### Step 2. Choose the events

From the EDA we chose the **wind** type, because it is the only type with enough buildings in all 4 damage classes in several events (at least 449 buildings per class in every wind event). The wind events are: `joplin-tornado`, `moore-tornado`, `tuscaloosa-tornado`, `hurricane-michael`, `hurricane-matthew`.

Check the real type of every event with a dry run (nothing is copied):

```bash
python scripts/create_subset.py --disaster-type wind --dry-run
```

### Step 3. Give a location to each event

The `location` is used to split the data into source / target / test. By default the script takes it from the event name (`joplin-tornado` gives `joplin`). For hurricanes this gives the storm name (`michael`), not a place, so we set the locations ourselves in `configs/event_location.json`:

```json
{
  "joplin-tornado": "us-central",
  "moore-tornado": "us-central",
  "tuscaloosa-tornado": "us-south",
  "hurricane-michael": "us-south",
  "hurricane-matthew": "caribbean"
}
```

These names are our own labels. To see where the events are, use `notebooks/02_event_map.ipynb` (it needs `xbd_subset/wind/images.csv` from Step 4 and an internet connection for the map background). It also saves `reports/eda/map_world_wind.html` and `event_centers_wind.csv`.

### Step 4. Create the subset

```bash
python scripts/create_subset.py --disaster-type wind --location-map configs/event_location.json
```

What the script does:

1. Reads the label files of all splits (`tier1`, `tier3`, `test`, `hold`) and finds the disaster type of each event.
2. Copies the pre/post images and labels of the chosen events to `xbd_subset/wind/<event>/images` and `.../labels`. The splits are merged. The original split is kept in the column `orig_split`.
3. Writes `buildings.csv`, `images.csv`, `events.csv` and `subset_info.json`.

The script can be run again: files already copied are skipped, and the CSV files are written again. If the same image id exists in two splits, the script stops with a message (then use `--split-folders`).

Options:

| Option                      | Meaning                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------ |
| `--disaster-type wind`      | disaster type(s) to extract (default: `wind`); several are allowed                   |
| `--dry-run`                 | only list events, types and counts, copy nothing                                     |
| `--location-map file.json`  | override the location of some events                                                 |
| `--events a b c`            | keep only these events (they must also have the chosen type)                         |
| `--splits tier1 tier3 test` | read only these splits (default: all four)                                           |
| `--out folder`              | output folder (default: `xbd_subset/<disaster type>`)                                |
| `--limit-per-event N`       | copy only N image pairs per event (quick test)                                       |
| `--split-folders`           | keep one folder per split inside each event (only if image ids repeat across splits) |
| `--tar`                     | also make `<out>.tar.gz` (to upload to Google Drive)                                 |

Examples:

```bash
# only the three tornado events, in another folder
python scripts/create_subset.py --disaster-type wind \
  --events joplin-tornado moore-tornado tuscaloosa-tornado --out xbd_subset/tornado

# another type, and an archive for Drive
python scripts/create_subset.py --disaster-type fire --tar
```

### Step 5. Check the result

```python
import pandas as pd
pd.read_csv("xbd_subset/wind/events.csv")[["event", "location", "location_source", "n_images", "n_buildings"]]
```

`location_source` is `override` for the events in `event_location.json` and `parsed_from_name` for the others.

---

## 5. The CSV files (input for the dataloader)

The images stay as full tiles. The CSV files tell the code what the samples are, so the data loader can crop each building from the pre and post image.

**`buildings.csv`** (one row per building, from the post-disaster label):

| Column                               | Meaning                                                                                                               |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| `building_uid`                       | unique id of the building                                                                                             |
| `image_id`                           | image name without `_pre_disaster` / `_post_disaster`                                                                 |
| `event`, `location`, `disaster_type` | event name, our location label, disaster type                                                                         |
| `orig_split`                         | original xBD folder (`tier1`, `tier3`, `test`, `hold`)                                                                |
| `scene_id`                           | `catalog_id` of the satellite scene (use it to avoid putting tiles of the same scene in both training and validation) |
| `damage`                             | damage label (`subtype`)                                                                                              |
| `label_id`                           | 0 = no-damage, 1 = minor-damage, 2 = major-damage, 3 = destroyed, **-1 = un-classified (filter it out)**              |
| `x_min`, `y_min`, `x_max`, `y_max`   | bounding box in pixels, inside the image                                                                              |
| `area_px`                            | polygon area in pixels                                                                                                |
| `pre_path`, `post_path`              | image paths, relative to the subset folder                                                                            |

**`images.csv`** (one row per image pair): `image_id`, `event`, `location`, `disaster_type`, `orig_split`, `pre_path`, `post_path`, `pre_label_path`, `post_label_path`, `width`, `height`, `sensor`, `gsd`, capture dates, off-nadir angles, sun elevations, `catalog_id_pre/post`, `n_buildings`, `lon`, `lat` (mean position of the buildings in the image, empty if there is no building).

**`events.csv`** (one row per event): `event`, `disaster_type`, `location`, `location_source`, `splits`, `n_images`, `n_buildings`, one count per damage class, `lon_mean`, `lat_mean`.

---

## 6. Experiment plan (tentative)

Domain adaptation **across locations**, using the `location` column:

| Role                       | Location (events)                                  | Image pairs | Buildings | no-damage | minor | major | destroyed |
| -------------------------- | -------------------------------------------------- | ----------- | --------- | --------- | ----- | ----- | --------- |
| Source (labeled, training) | `us-central` (joplin-tornado, moore-tornado)       | 376         | 37,068    | 74.7%     | 8.3%  | 3.9%  | 13.1%     |
| Target (adaptation)        | `us-south` (tuscaloosa-tornado, hurricane-michael) | 784         | 42,154    | 68.2%     | 20.4% | 6.6%  | 4.8%      |
| Final test (hold out)      | `caribbean` (hurricane-matthew)                    | 311         | 17,126    | 18.3%     | 51.9% | 13.3% | 16.5%     |

(Counts from the EDA tables, post-disaster labels, `un-classified` buildings not included.)

Points to keep in mind:

- The hazard type also changes (source = tornadoes only; target = one tornado + one hurricane; test = hurricane), so the drop in performance is not only because of the location. Report the target results per event (tuscaloosa and michael) to see the difference.
- The class mix of `caribbean` is very different from the source. DANN and CORAL align image features, but they do not correct class proportions.
- Do not use `caribbean` for any tuning. Look at it only at the end.
- For the validation set inside the source, split by image or by `scene_id`, not by building.

---

## 7. Next steps

Not done yet:

- [ ] Measure how many images have black (empty) areas, and decide what to do with them.
- [ ] Decide what to do with `un-classified` buildings (current idea: exclude them).
- [ ] Write the Dataset class (read `buildings.csv`, crop pre/post, return image, `label_id`, `location`) and the source / target / test split code.
- [ ] Write the training code for the algorithms (transfer learning, DANN, CORAL) in `src/`, with a config file, checkpoints and logs in `runs/`.
- [ ] Upload the subset to Google Drive (as a `.tar.gz`, made with `--tar`) and run the training on Colab. Colab sessions can end, so save a checkpoint every epoch and support `--resume`.
