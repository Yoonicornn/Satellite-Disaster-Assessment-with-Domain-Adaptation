# Build notes: decisions, measurements and rationale

Working notes for the ECSE552 project _Satellite Disaster Assessment with Domain Adaptation_.
The README says **how** to run things. This file says **why** everything is the way it is, and
records the numbers we measured so we do not have to rediscover them.

Keep it updated as decisions change. It is also the file to hand to anyone (or any AI session)
that needs the full context without reading the whole codebase.

Last updated: 2026-11-20.

---

## 1. Project in one paragraph

Train a building-damage classifier on labelled buildings from one region, then measure how much
it loses on a different region, then test whether unsupervised domain adaptation recovers some
of that loss. Everything is building-level classification using the bounding boxes that xBD
already provides — we do **not** do building detection or segmentation.

---

## 2. Data decisions

### 2.1 Why only "wind"

The proposal asks for domain adaptation across geography. Holding the hazard type fixed removes
one confound. From the EDA over tier1 + tier3 + test (10,101 image pairs, 370,976 buildings,
19 events), **wind** is the only `disaster_type` with enough buildings in all four damage classes
across several events (at least 449 per class in every wind event).

Wind covers both tornadoes and hurricanes. Confirmed by reading `disaster_type` from the label
metadata, not inferred from the event names.

Five wind events: `joplin-tornado`, `moore-tornado`, `tuscaloosa-tornado`, `hurricane-michael`,
`hurricane-matthew`.

### 2.2 Locations are our own labels

The script derives a location from the event name, which gives the _storm_ name for hurricanes
(`michael`), not a place. We override it in `configs/event_location.json`:

| Event                                 | Location     |
| ------------------------------------- | ------------ |
| joplin-tornado, moore-tornado         | `us-central` |
| tuscaloosa-tornado, hurricane-michael | `us-south`   |
| hurricane-matthew                     | `caribbean`  |

### 2.3 Domain roles

| Role          | Location   | Buildings | Labels used in training        |
| ------------- | ---------- | --------- | ------------------------------ |
| Source        | us-central | 37,068    | yes                            |
| Target        | us-south   | 49,226    | **no** (except the oracle run) |
| Held-out test | caribbean  | 22,630    | no                             |

**Known confound, must be stated in the report:** the hazard sub-type also changes. The source
is tornado-only, the target mixes tornado and hurricane, the test is hurricane. So the
performance drop is geography _and_ hazard type, not geography alone. Reporting the target
results per event (tuscaloosa vs michael) separates them partly.

### 2.4 The xBD folders

`tier1`, `tier3`, `test`, `hold`. A research paper using xBD describes tier1 + tier3 as the
training set, `hold` as validation and `test` as test — this is **not** official xView2
documentation, so do not cite it as such. We merge all splits and re-split by location, keeping
the original folder in the `orig_split` column.

---

## 3. What we measured

All numbers below come from our own data, not from papers. They are the evidence behind the
decisions in section 4.

### 3.1 Class distribution per domain (un-classified excluded)

| Domain              | Total  | no-damage | minor  | major  | destroyed |
| ------------------- | ------ | --------- | ------ | ------ | --------- |
| us-central (source) | 37,068 | 74.7 %    | 8.3 %  | 3.9 %  | 13.1 %    |
| us-south (target)   | 49,226 | 67.4 %    | 21.0 % | 6.9 %  | 4.7 %     |
| caribbean (test)    | 22,630 | 17.9 %    | 54.5 % | 12.0 % | 15.6 %    |

Total variation distance between these distributions:

| Pair            | TVD       |
| --------------- | --------- |
| source ↔ target | 0.156     |
| source ↔ test   | **0.567** |
| target ↔ test   | 0.495     |

**This is the most important finding so far.** CORAL and DANN align the _marginal_ feature
distribution and implicitly assume similar class proportions. At TVD 0.567 that assumption is
badly violated for caribbean, so expect adaptation to help on us-south and to help little or
not at all on caribbean. That is a result to report, not a failure.

Every class has at least 1,454 buildings in every domain, so per-class F1 is measurable
everywhere.

### 3.2 Building size

Median bounding-box side, in pixels, with `gsd` from the image metadata:

| Event              | Role   | Median side | gsd  | Median ground size |
| ------------------ | ------ | ----------- | ---- | ------------------ |
| hurricane-matthew  | test   | 23.0 px     | 2.03 | 46.7               |
| joplin-tornado     | source | 35.1 px     | 2.35 | 82.6               |
| moore-tornado      | source | 39.2 px     | 1.73 | 67.8               |
| tuscaloosa-tornado | target | 37.8 px     | 1.86 | 70.2               |
| hurricane-michael  | target | 41.0 px     | 2.07 | 84.9               |

Caribbean buildings are genuinely smaller (about 47 against 68–85 in ground units), not just
imaged at coarser resolution. Source and target are well matched on this axis.

`hurricane-matthew` is also the only event with a **varying** gsd (1.59 to 2.77); the other four
have a single constant value. Its images come from several captures.

### 3.3 Black (no-data) pixels

Only **69 of 108,916 crops (0.06 %)** are more than half black. Mean black fraction per crop is
2–4 %. The black bands visible in the full tiles sit at image edges where few buildings are.
Caribbean has the _lowest_ black fraction of all splits.

The Dataset filters at `max_black=0.5`, which removes those 69 crops and changes nothing else.

### 3.4 Pre/post sharpness

Mean squared image gradient, post divided by pre:

| Event              | Domain | post / pre |
| ------------------ | ------ | ---------- |
| hurricane-matthew  | test   | 0.7        |
| joplin-tornado     | source | 0.8        |
| moore-tornado      | source | 0.8        |
| tuscaloosa-tornado | target | 1.0        |
| hurricane-michael  | target | 1.1        |

Checked because the caribbean post crops looked blurrier by eye. The source events show the same
pattern (0.8), so there is no "blurry means test domain" shortcut for the model to learn. No
action taken.

Interesting for the report: by this measure the **target** (1.0–1.1) is further from the source
(0.8) than the test is (0.7), while class distribution and building size both say caribbean is
the outlier. The shift is multi-dimensional, and a single global alignment cannot address all
axes at once.

---

## 4. Preprocessing decisions

### 4.1 Crops

```
side = max(bbox_width, bbox_height)
box  = clip(1.5 * side, 16, 256)       square, centred on the building
                                       clamped to the image, zero-padded at edges
resize to 64 x 64, same box for pre and post
```

- **1.5x margin**: context matters for damage (debris, missing roof), and it keeps the target
  building filling about two thirds of the crop so the model knows which building is meant.
- **64 x 64**: typical box is 35–62 px, so this is close to native. 128 would be 2–4x
  upsampling for 4x the compute and no extra information. The limit is the satellite resolution,
  not the output size.
- **Scale-normalised, not a fixed window**: caribbean buildings are half the linear size of the
  others. A fixed window would leave them tiny and let the model use size as a shortcut that
  breaks on the test domain.
- **Floor 16 px** so the smallest buildings are not pure interpolation; **cap 256 px** because
  the largest boxes reach 830 px, which would be most of the tile.

### 4.2 Filters

- `un-classified` dropped (3,857 buildings). Not a damage level, so there is nothing to learn.
- `side < 5` dropped (8 buildings). Degenerate boxes only.
- We first tried `side < 8`, which drops 1.02 % overall but **18.4 % of the dropped crops were
  destroyed** — a 2x over-representation, because destroyed buildings often have small or partial
  polygons. Lowering the threshold to 5 removes that bias.

After filtering: **108,916 crops** from 112,781 buildings.

### 4.3 Augmentation

Geometric only, applied with the **same random parameters** to the pre and post crop, otherwise
the pair stops being aligned:

- horizontal flip, 90° rotations, ±10° rotation, 0.9–1.1 scale

Applied in `Dataset.__getitem__`, every epoch, only on `source_train` and `target_train`. Never
on the evaluation splits.

**No photometric augmentation.** Paper 2 removed it from its final baseline, and changing colours
would interfere with exactly what CORAL and DANN are trying to align. Keep it as a separate
ablation if there is time.

Implementation detail: the random numbers come from Python's `random`, not numpy. PyTorch's
DataLoader reseeds `random` in each worker but **not** numpy's global generator, so numpy would
give every worker the same "random" transforms.

### 4.4 Why crops are pre-extracted

Opening two 1024x1024 GeoTIFFs per sample would dominate training time. Extraction takes
82 seconds once and produces 2.7 GB of uint8 arrays that are memory-mapped during training.
It also makes the dataset much smaller to share than the raw subset.

---

## 5. Splits

### 5.1 Stratified group split

Three constraints have to hold at once:

1. a whole image goes to one split (otherwise crops from the same image leak across train/val)
2. each event contributes the same fraction (event mix preserved)
3. the damage-class mix is similar on both sides

Plain random image selection satisfies 1 and 2 but not 3, because in a tornado track one image
can be almost entirely destroyed and the next untouched. Our first split (random, seed 42) gave
`source_val` 20.1 % destroyed against 11.4 % in `source_train`.

We use `sklearn.model_selection.StratifiedGroupKFold` with `groups=image_id` and `y=label_id`,
run separately inside each event. Measured on synthetic data with the same skew, across 8 seeds:

|            | mean TVD | worst | best  |
| ---------- | -------- | ----- | ----- |
| random     | 0.124    | 0.269 | 0.027 |
| stratified | 0.040    | 0.057 | 0.022 |

### 5.2 The chosen seed

On the real data, stratified seeds still vary (0.012 to 0.108), so we scanned 10 and chose
**seed 8**:

| Pair                       | TVD   |
| -------------------------- | ----- |
| source_train ↔ source_val  | 0.012 |
| target_train ↔ target_test | 0.021 |

Resulting class shares, source_train vs source_val: 74.8/8.2/3.9/13.1 against 73.6/9.0/4.1/13.2.

This is legitimate: the split was chosen for balance **before any model was trained**, using no
test information. Choosing a split after seeing test scores would not be.

`source_val` ends up at 15.4 % of the source by building count rather than 20 %, because
`StratifiedGroupKFold` balances classes across image groups of unequal size. 5,699 validation
buildings is plenty.

### 5.3 The five splits

| Split          | Buildings | Used for                                                               |
| -------------- | --------- | ---------------------------------------------------------------------- |
| `source_train` | 31,360    | training, with labels                                                  |
| `source_val`   | 5,705     | selecting the best epoch                                               |
| `target_train` | 41,299    | unlabelled target batches (CORAL / DANN); labelled only for the oracle |
| `target_test`  | 7,924     | **main result**                                                        |
| `test`         | 22,628    | unseen domain, hardest case                                            |

`target_test` and `test` measure different things: adaptation to a domain the model has seen
unlabelled, versus generalisation to one it has never seen at all.

**The split is fixed for all 12 runs.** `run.seed` changes only the model initialisation and the
data order. Mixing split variance into the ± across seeds would make the comparison meaningless.

---

## 6. Model

### 6.1 Architecture

```
pre  crop ─┐
           ├─ ResNet-34 (shared weights, ImageNet) ─> f_pre, f_post  (512 each)
post crop ─┘
f = [f_pre, f_post, f_post − f_pre]        1536
z = Dropout(ReLU(Linear(f, 512)))          the feature CORAL and DANN align
logits = Linear(z, 4)
```

22.1 M parameters. DANN adds a 131.8 k discriminator.

### 6.2 Why these choices

- **Shared weights, two branches.** The xView2 work reports that plain 6-channel stacking and
  image subtraction both performed poorly, while shared-weight branches worked. It also halves
  the parameters.
- **ResNet-34.** Paper 2 showed a single ResNet-34 U-Net reaching 0.770 against the 12-model
  competition ensemble's 0.804, so the big backbones are not where the performance is. It is
  also the cheapest of the four, which matters for 12 runs on Colab.
- **ImageNet weights, not xView2 weights.** The competition checkpoints are U-Net segmentation
  weights and cannot be loaded into a crop classifier.
- **Correction worth remembering:** the competition's best result was _not_ "ResNet34 +
  SE-ResNeXt50". It was an ensemble of 4 architectures x 3 seeds, averaged with equal weights.
- **`f_post − f_pre`** is our addition, not from the papers. Cheap, makes the change explicit,
  and ablatable with `--set model.fusion=concat`.
- Both reference papers do whole-image segmentation with U-Nets. Only the shared-weight idea
  carries over to our crop classifier; the architectures do not.

---

## 7. Algorithms

All four use the identical network. Only the loss changes, so any difference between methods
cannot come from the architecture.

| Config     | Trains on                                | Target labels | Role               |
| ---------- | ---------------------------------------- | ------------- | ------------------ |
| `baseline` | source_train                             | no            | the number to beat |
| `coral`    | source_train + unlabelled target_train   | no            | unsupervised DA    |
| `dann`     | source_train + unlabelled target_train   | no            | unsupervised DA    |
| `oracle`   | source_train + **labelled** target_train | yes           | upper bound        |

### 7.1 CORAL

From Deep CORAL (arXiv 1607.01719), equations 1 and 2:

```
L_coral = || C_s - C_t ||_F^2 / (4 d^2)
C = (X_centred^T X_centred) / (n - 1)
L = L_cls + lambda * L_coral
```

No new parameters. Covariances are computed over the batch, so `drop_last=True` matters. The
paper sets lambda so the two loss terms are roughly equal by the end of training — watch
`train/cls_loss` against `train/extra_loss` in TensorBoard and tune `method.lambda_weight`.

### 7.2 DANN

From Ganin et al. (arXiv 1505.07818):

```
L = L_cls - lambda * L_domain            (implemented by the gradient reversal layer)
lambda_p = 2 / (1 + exp(-gamma * p)) - 1,   gamma = 10, p = progress 0 -> 1
```

The GRL is identity forward and multiplies the gradient by `-lambda` backward. Starting at 0
keeps the discriminator from wrecking the features before the classifier has learned anything.

Note: with gamma = 10 the ramp is 85 % complete by p = 0.25. If DANN is unstable, lowering
`method.gamma` to 5 stretches it and is the first knob to try. Watch `train/domain_acc`: near
0.5 means the features look the same to the discriminator (good); stuck near 1.0 means
adaptation is not happening.

### 7.3 Terminology

"Transfer learning" is ambiguous here, because _every_ run uses ImageNet pretraining. In the
report, name the runs by **which labels they see**, as in the table above. If a separate
transfer-learning row is wanted, make it frozen-backbone vs full fine-tuning.

Paper 1's method is exactly our `oracle` row: it fine-tunes xBD-pretrained weights on _labelled_
target data. Its numbers are therefore not comparable to our unsupervised results.

### 7.4 Separate forward passes

Source and target go through the model in **two separate forward passes**, never concatenated.
A single pass would let BatchNorm mix the two domains' statistics, which is a hidden form of
adaptation and would contaminate the comparison.

---

## 8. Training settings

| Setting       | Value                                                 | Source                                         |
| ------------- | ----------------------------------------------------- | ---------------------------------------------- |
| loss          | weighted cross-entropy                                |                                                |
| class weights | `w_c = N / (C * n_c)`, from the source training split | paper 2 uses `1/f_c`                           |
| optimiser     | AdamW                                                 | paper 2                                        |
| lr            | 2e-4 (new layers), 2e-5 (backbone)                    | paper 2's 2e-4; the split is standard practice |
| weight decay  | 1e-6                                                  | paper 2                                        |
| scheduler     | cosine over `optim.epochs`                            |                                                |
| batch         | 64 source + 64 target                                 |                                                |
| AMP           | CUDA only                                             | not reliable on MPS                            |

Class weights on the chosen split: `[0.334, 3.056, 6.438, 1.910]`, a 19:1 ratio. A softer
`1/sqrt(f)` option exists (`loss.class_weights=sqrt`, 4.4:1) if training is unstable — loss
spikes, validation jumping, or over-prediction of major-damage. Decide on the baseline, then use
the same choice everywhere.

We chose class weights over a `WeightedRandomSampler` for one reason specific to this project: a
sampler would make the source batch class-balanced while the target batch cannot be (no labels),
so CORAL and DANN would partly be aligning away an artefact we created ourselves.

### 8.1 Model selection

Best epoch is chosen on `source_val` macro-F1. **Never on target labels** — that is the mistake
to avoid, and paper 1 is loose about it (it used the target validation split to select models and
tune thresholds).

### 8.2 Early stopping

Implemented (`optim.early_stop`) but **off by default**, and it should stay off for the final
grid. `best.pt` already protects the reported number, and a fixed epoch budget keeps every method
on equal footing. DANN would suffer most: its validation often dips while the lambda ramp is
still rising, exactly when patience would fire.

Better approach: run one long baseline, read `history.csv`, find the plateau, and set
`optim.epochs` to that for all 12 runs.

---

## 9. Metrics

- **macro-F1** is the headline. 74 % of buildings are no-damage, so accuracy would reward a model
  that ignores the damaged classes.
- Also reported: per-class F1, recall and precision, balanced accuracy, confusion matrix.
- Reported on all three splits: `source_val` (did it learn the task), `target_test` (did
  adaptation help), `test` (does it generalise to an unseen domain).
- 3 seeds per method, report **mean ± std**. Paper 2's own numbers carry ±0.02, so a 1-point
  difference means nothing.

---

## 10. Results so far

Baseline, 2 epochs only, ResNet-34 ImageNet, seed 0 (a smoke run, not a result):

| Split            | macro-F1 | vs source_val  |
| ---------------- | -------- | -------------- |
| source_val       | 0.6455   | —              |
| target_test      | 0.5105   | −0.135 (−21 %) |
| test (caribbean) | 0.2349   | −0.411 (−64 %) |

The caribbean confusion matrix shows the model predicting "destroyed" for 12,770 of 22,617
buildings (56 %) when only 15.6 % are. Minor-damage recall is 0.055. This matches the label
shift measured in 3.1: the source has almost no minor-damage examples, and the class weights push
towards the severe classes.

Weakest class on source is major-damage (F1 0.360), which is expected at 3.9 % of the data.

Timing: ~90 s per epoch on an Apple Silicon Mac (MPS). CORAL and DANN are roughly 2–2.5x slower
because of the extra target pass; the oracle trains on 72k crops instead of 31k. Full grid on the
Mac is 13–15 hours at 25 epochs, so run it on Colab.

---

## 11. Code conventions

- **Configs**: `configs/base.yaml` holds everything shared; each method file declares
  `_base_: base.yaml` and overrides only its own keys. Command line (`--set`) beats file, file
  beats base. One `--set` per command; a second one replaces the first.
- **Validation**: `src/config_schema.py` is a pydantic schema with `extra="forbid"`, so a typo
  like `bottlenck: 512` stops the run instead of being silently ignored while the default is
  used. The method section is a discriminated union, so a DANN key inside `coral.yaml` is an
  error.
- **Adding a method**: write a subclass of `Algorithm` in `src/algorithms/`, add one line to
  `REGISTRY`. `train.py` never mentions method names.
- **Run folders**: `runs/<method>/seed<N>_<timestamp>/`. Nothing is overwritten; re-running the
  same name without `--resume` or `--overwrite` is refused.
- **Reproducibility**: the resolved config is saved into every run folder, and
  `crops/*/extract_info.json` records the crop and split parameters.

---

## 12. Papers used

| Paper                                                                                                                                                           | What we took                                                                                                                       |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| [Gerard et al., _A Simple, Strong Baseline for Building Damage Detection on xBD_ (2401.17271)](https://arxiv.org/html/2401.17271v1)                             | augmentation list, AdamW 2e-4 / wd 1e-6, class weights `1/f_c`, evidence that ResNet-34 suffices, event-level generalisation drop  |
| [Mouradi & Kshirsagar, _Robust Building Damage Detection in Cross-Disaster Settings Using Domain Adaptation_ (2603.14694)](https://arxiv.org/html/2603.14694v1) | framing of with-DA vs without-DA; its DA is supervised fine-tuning, which is our oracle                                            |
| [Sun & Saenko, _Deep CORAL_ (1607.01719)](https://arxiv.org/pdf/1607.01719)                                                                                     | the CORAL loss and the batch covariance formula                                                                                    |
| [Ganin et al., _Domain-Adversarial Training of Neural Networks_ (1505.07818)](https://arxiv.org/pdf/1505.07818)                                                 | GRL, the objective, the lambda ramp                                                                                                |
| [xView2 1st place solution](https://github.com/asmaemou/xview2_1st_place_solution)                                                                              | how seeds are run (one process per seed, seed as argv) and combined (equal-weight probability averaging in `create_submission.py`) |

**On paper 1's augmentation claim.** Its abstract says unsharp-enhanced RGB with supervised DA
performs best, and that is true as a ranking — but the margin over plain RGB is **0.0036
macro-F1** (0.5552 vs 0.5516), from single runs with no seeds or standard deviations, on a test
split of 10 % of 87 image pairs. Meanwhile DA itself is worth about +0.35. Across its
augmentation configurations the destroyed-class F1 swings from 0.0000 to 0.5156, which is
instability, not a tuning curve. Conclusion: spend runs on the DA comparison, treat augmentation
choice as a secondary ablation.

Also note paper 1's "augmentations" are image _filters_ (unsharp, CLAHE, edge detection), while
paper 2's are _geometric_ transforms. Different things; we use paper 2's.

---

## 13. Environment problems hit (and fixes)

Worth recording because they cost time and will recur on a teammate's machine.

| Symptom                                                        | Cause                                                                                  | Fix                                                                                      |
| -------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: rasterio` with `(ecse552)` in the prompt | `python` resolved to a pyenv 3.8.5 shim, which is earlier in `PATH` than the conda env | reorder the conda block after pyenv in `~/.zshrc`, or call the env's python by full path |
| `OMP: Error #15: Initializing libomp.dylib`                    | pip `torch` ships its own libomp, conda-forge `llvm-openmp` provides another           | reinstall torch from conda-forge                                                         |
| `ModuleNotFoundError: pkg_resources` from tensorboard          | setuptools 81+ removed it                                                              | `conda install -n ecse552 "setuptools<81"`                                               |
| tensorboard shows "No Runs" with empty cards                   | old pyenv TensorBoard 2.14 reading new event files                                     | run it from the conda env, or `python -m tensorboard.main`                               |

---

## 14. Open items

- [ ] Pick `optim.epochs` from a long baseline run; apply it to all 12 runs.
- [ ] Run the grid on Colab.
- [ ] Aggregation script: glob `runs/*/*/metrics.json`, produce mean ± std per method and split.
- [ ] Prediction figures: whole-image ground truth vs predicted, error grids, confusion matrices,
      from `predictions_*.csv` joined to `manifest.csv` on `building_uid`.
- [ ] Optional ablations, in priority order: `fusion=concat` (drop the difference feature),
      `class_weights=sqrt`, crop size 128, colour augmentation.
- [ ] Optional: t-SNE or UMAP of `z` for source vs target, before and after adaptation
      (step 5 of the proposal).
- [ ] Decide whether to also report the transductive protocol (evaluate on all target data
      rather than the held-out 20 %), which is what many UDA papers do.

---

## 15. Things to be careful about in the report

1. The hazard sub-type changes along with the region (2.3). Do not claim a pure geographic shift.
2. Label shift is severe for caribbean (TVD 0.567). State it, and use it to explain why marginal
   alignment struggles there.
3. The split seed was chosen for class balance before any training. Say so explicitly.
4. Model selection never uses target labels. Say so explicitly; paper 1 did not.
5. The oracle uses target labels and is an upper bound, not a competing method.
6. Report mean ± std over 3 seeds. Do not compare single runs.
7. Our numbers are not comparable to either paper's: they do whole-image segmentation scored
   against the xView2 competition metric, we do building-crop classification.
