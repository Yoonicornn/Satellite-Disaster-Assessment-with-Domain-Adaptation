# Build notes: decisions, measurements and findings

Working notes for the ECSE552 project _Satellite Disaster Assessment with Domain Adaptation_.
The README says **how** to run things. This file says **why** everything is the way it is, and
records the numbers we measured so we do not have to rediscover them.

Last updated: 2026-11-22, after the full 12-run grid.

---

## 1. Project in one paragraph

Train a building-damage classifier on labelled buildings from one region, measure how much it
loses on a different region, then test whether unsupervised domain adaptation recovers some of
that loss. Everything is building-level classification using the bounding boxes xBD already
provides — we do **not** do building detection or segmentation.

**Headline result**: neither CORAL nor DANN improved transfer. The only thing that helped was
BatchNorm statistics adapting to the target, which happens for free inside both methods and has
nothing to do with their loss. Section 10 has the numbers, section 11 the explanation.

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

| Role          | Location   | Buildings | Labels used in training |
| ------------- | ---------- | --------- | ----------------------- |
| Source        | us-central | 37,068    | yes                     |
| Target        | us-south   | 49,226    | **no**                  |
| Held-out test | caribbean  | 22,630    | no                      |

**Known confound, must be stated in the report:** the hazard sub-type also changes. The source
is tornado-only, the target mixes tornado and hurricane, the test is hurricane. So the
performance drop is geography _and_ hazard type, not geography alone. Reporting the target
results per event (tuscaloosa vs michael) would separate them partly.

### 2.4 The xBD folders

`tier1`, `tier3`, `test`, `hold`. A research paper using xBD describes tier1 + tier3 as the
training set, `hold` as validation and `test` as test — this is **not** official xView2
documentation, so do not cite it as such. We merge all splits and re-split by location, keeping
the original folder in the `orig_split` column.

---

## 3. What we measured in the data

All numbers below come from our own data, not from papers.

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

**This prediction held.** We wrote before any training that CORAL and DANN align the _marginal_
feature distribution and implicitly assume similar class proportions, so at TVD 0.567 we should
expect adaptation to fail on caribbean. It did. Section 11 explains why this is a theorem, not
bad luck.

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
2–4 %. The black bands in the full tiles sit at image edges where few buildings are. Caribbean
has the _lowest_ black fraction of all splits.

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
pattern (0.8), so there is no "blurry means test domain" shortcut for the model to learn.

Worth noting in the report: by this measure the **target** (1.0–1.1) is further from the source
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
would interfere with exactly what CORAL and DANN are trying to align.

Implementation detail: the random numbers come from Python's `random`, not numpy. PyTorch's
DataLoader reseeds `random` in each worker but **not** numpy's global generator, so numpy would
give every worker the same "random" transforms.

### 4.4 Why crops are pre-extracted

Opening two 1024x1024 GeoTIFFs per sample would dominate training time. Extraction takes
82 seconds once and produces 2.7 GB of uint8 arrays that are memory-mapped during training.

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

Class shares, source_train vs source_val: 74.8/8.2/3.9/13.1 against 73.6/9.0/4.1/13.2.

This is legitimate: the split was chosen for balance **before any model was trained**, using no
test information. Choosing a split after seeing test scores would not be.

### 5.3 The five splits

Counts after the Dataset drops the 69 mostly-black crops (what training actually sees):

| Split          | Buildings | Used for                                       |
| -------------- | --------- | ---------------------------------------------- |
| `source_train` | 31,339    | training, with labels                          |
| `source_val`   | 5,699     | selecting the best epoch                       |
| `target_train` | 41,272    | unlabelled target batches (bn_only/CORAL/DANN) |
| `target_test`  | 7,920     | **main result**                                |
| `test`         | 22,617    | unseen domain, hardest case                    |

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
z = Dropout(ReLU(Linear(f, 256)))          the feature CORAL and DANN align
logits = Linear(z, 4)
```

**21,679,172 parameters.** DANN adds a 66,306-parameter discriminator (256 → 256 → 2).

### 6.2 Why these choices

- **Shared weights, two branches.** The xView2 work reports that plain 6-channel stacking and
  image subtraction both performed poorly, while shared-weight branches worked. It also halves
  the parameters.
- **ResNet-34.** Paper 2 showed a single ResNet-34 U-Net reaching 0.770 against the 12-model
  competition ensemble's 0.804, so the big backbones are not where the performance is. It is
  also the cheapest option, which matters for 12 runs.
- **ImageNet weights, not xView2 weights.** The competition checkpoints are U-Net segmentation
  weights and cannot be loaded into a crop classifier.
- **Correction worth remembering:** the competition's best result was _not_ "ResNet34 +
  SE-ResNeXt50". It was an ensemble of 4 architectures x 3 seeds, averaged with equal weights.
- **`f_post − f_pre`** is our addition, not from the papers. Cheap, makes the change explicit,
  and ablatable with `--set model.fusion=concat`.
- **Bottleneck 256, not 512.** Changed before the final grid. CORAL estimates a `d x d`
  covariance from a single batch; at `d = 512` with batch 64 that estimate has rank 63 out of
  512 and is mostly noise. Batch was raised to 128 at the same time, so the batch now exceeds
  the feature dimension. Smaller bottleneck also means fewer head parameters, which helps the
  overfitting in section 10.3. Deep CORAL itself applies the loss at a low-dimensional layer.

---

## 7. Algorithms

All four use the identical network. Only the loss changes, so any difference between methods
cannot come from the architecture.

| Config     | Sees target images | Target loss      | Target labels | Role                    |
| ---------- | ------------------ | ---------------- | ------------- | ----------------------- |
| `baseline` | no                 | —                | no            | the number to beat      |
| `bn_only`  | yes                | **zero**         | no            | control: BatchNorm only |
| `coral`    | yes                | covariance match | no            | unsupervised DA         |
| `dann`     | yes                | adversarial      | no            | unsupervised DA         |

`oracle` (source + **labelled** target) was dropped. Its seed-3 run scored 0.5085 on
`target_test`, below the baseline's 0.5324, which we never explained. It is an upper bound, not
a competing method, and the project does not need it.

### 7.1 CORAL

From Deep CORAL (arXiv 1607.01719), equations 1 and 2:

```
L_coral = || C_s - C_t ||_F^2 / (4 d^2)
C = (X_centred^T X_centred) / (n - 1)
L = L_cls + lambda * L_coral
```

No new parameters. Covariances are over the batch, so `drop_last=True` matters. `lambda_weight`
is 40 with a 5-epoch linear ramp. Note the loss scale depends on `d`, so `lambda_weight` has to
be rechecked whenever `model.bottleneck` changes.

### 7.2 DANN

From Ganin et al. (arXiv 1505.07818):

```
L = L_cls - lambda * L_domain            (implemented by the gradient reversal layer)
lambda_p = 2 / (1 + exp(-gamma * p)) - 1,   gamma = 10, p = progress 0 -> 1
```

The GRL is identity forward and multiplies the gradient by `-lambda` backward. Starting at 0
keeps the discriminator from wrecking the features before the classifier has learned anything.
With gamma = 10 the ramp is 85 % complete by p = 0.25; **lower** gamma stretches it out.

### 7.3 bn_only — the control

CORAL and DANN forward an unlabelled target batch every training step while the model is in
train mode. Every BatchNorm layer therefore updates its running mean and variance with target
statistics. The baseline never does. That is free adaptation — essentially AdaBN — and it has
nothing to do with the alignment loss.

`bn_only` forwards the same target batches and returns a loss of exactly zero, so the effect can
be separated:

```
baseline -> bn_only      what BatchNorm adaptation is worth on its own
bn_only  -> coral/dann   what the alignment loss actually adds
```

Without this run, a CORAL gain and a BatchNorm gain are indistinguishable.

### 7.4 Separate forward passes — correction

**An earlier version of this file was wrong here.** It said that using two separate forward
passes stops BatchNorm from mixing the domains. That is only half true.

Separate passes stop the two domains being mixed inside one batch statistic. They do **not** stop
BatchNorm's _running_ statistics from being updated by target data. Measured directly on a
ResNet-18 `bn1` layer:

```
train mode, no_grad : running_mean moved by 0.001108
eval  mode, no_grad : running_mean moved by 0.000000
```

So the hidden adaptation we thought we had avoided was present in every CORAL and DANN run.
`bn_only` is the fix: it measures the effect instead of pretending it is absent.

Confirmed by comparing a baseline and a bn_only checkpoint trained with the same seed:

```
BatchNorm running stats : max abs diff 0.817259  (72 tensors)
learned weights         : max abs diff 0.004512
```

### 7.5 Terminology

"Transfer learning" is ambiguous here, because _every_ run uses ImageNet pretraining. In the
report, name the runs by **which labels and which images they see**, as in the table above.

Paper 1's method is supervised fine-tuning on labelled target data, so its numbers are not
comparable to our unsupervised results.

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
| batch         | **128 source + 128 target**                           | raised from 64 for CORAL's covariance          |
| bottleneck    | **256**                                               | see 6.2                                        |
| epochs        | 25                                                    | see 10.2                                       |
| AMP           | CUDA only                                             | not reliable on MPS                            |

Class weights on the chosen split: `[0.334, 3.056, 6.438, 1.910]`, a 19:1 ratio. A softer
`1/sqrt(f)` option exists (`loss.class_weights=sqrt`, 4.4:1).

We chose class weights over a `WeightedRandomSampler` for one reason specific to this project: a
sampler would make the source batch class-balanced while the target batch cannot be (no labels),
so CORAL and DANN would partly be aligning away an artefact we created ourselves.

### 8.1 Model selection

Best epoch is chosen on `source_val` macro-F1. **Never on target labels** — that is the mistake
to avoid, and paper 1 is loose about it (it used the target validation split to select models and
tune thresholds).

### 8.2 Early stopping

Implemented (`optim.early_stop`) but **off**, and it stayed off for the final grid. `best.pt`
already protects the reported number, and a fixed epoch budget keeps every method on equal
footing. DANN would suffer most: its validation dips while the lambda ramp is still rising,
exactly when patience would fire.

---

## 9. Metrics

- **macro-F1** is the headline. 74 % of source buildings are no-damage, so accuracy would reward
  a model that ignores the damaged classes.
- Also reported: per-class F1, recall and precision, balanced accuracy, confusion matrix.
- Reported on all three splits: `source_val` (did it learn the task), `target_test` (did
  adaptation help), `test` (does it generalise to an unseen domain).
- 3 seeds per method, **mean ± std**. The baseline's own spread is the bar for what counts as a
  difference — see 10.1.

---

## 10. Results

12 runs: 4 methods x 3 seeds (1, 3, 14), 25 epochs, best checkpoint on `source_val`.
Reproduce with `python scripts/summarize.py`.

### 10.1 Main table — macro-F1, mean ± std

| Method      | source_val          | target_test         | test (caribbean)    |
| ----------- | ------------------- | ------------------- | ------------------- |
| baseline    | 0.7116 ± 0.0039     | 0.5335 ± 0.0035     | 0.2168 ± 0.0045     |
| **bn_only** | **0.7207** ± 0.0028 | **0.5350** ± 0.0024 | **0.2356** ± 0.0080 |
| coral       | 0.7166 ± 0.0030     | 0.5317 ± 0.0030     | 0.2235 ± 0.0105     |
| dann        | 0.7130 ± 0.0028     | 0.5320 ± 0.0042     | 0.2274 ± 0.0084     |

Decomposition, against the baseline's own seed-to-seed spread:

| Split       | baseline sd | BatchNorm               | CORAL loss           | DANN loss            |
| ----------- | ----------- | ----------------------- | -------------------- | -------------------- |
| source_val  | 0.0039      | **+0.0091** above noise | −0.0041 worse        | −0.0077 worse        |
| target_test | 0.0035      | +0.0015 within noise    | −0.0033 within noise | −0.0030 within noise |
| test        | 0.0045      | **+0.0188** above noise | −0.0120 worse        | −0.0082 worse        |

Three findings:

1. **On `target_test`, nothing works.** All four methods are within ±0.004 of each other and the
   baseline spread is 0.0035. CORAL and DANN did not move the number.
2. **The only gain is BatchNorm, and it is free.** On `test` the separation is complete, not just
   a difference of means: baseline seeds are 0.2140 / 0.2143 / 0.2220, bn_only seeds are
   0.2279 / 0.2350 / 0.2438. Same on `source_val`.
3. **Both alignment losses cost a little.** CORAL and DANN sit below `bn_only` on all three
   splits, beyond noise on `source_val` and `test`.

Without the `bn_only` control we would have concluded "CORAL ≈ baseline, no effect". The real
story is that the methods did transfer something, it was BatchNorm riding along for free, and
their actual objectives subtracted from it.

**Retired result.** An earlier seed-3-only run showed CORAL at +0.008 on `target_test` over the
baseline. With three seeds that is inside the noise. Single-seed differences on this dataset
mean nothing.

### 10.2 Epoch budget

Plateau (within 1 % of the final value) lands between epochs 7 and 22 across the 12 runs; the
kept epoch lands between 7 and 24. 25 epochs is enough and not wasteful. No run was still
improving at the end.

### 10.3 Overfitting

Train `cls_loss` falls steadily while `source_val/loss` turns upward early — typically by epoch
3 to 7. `target_test/loss` rises throughout.

This does not damage the reported numbers, because `best.pt` is selected on macro-F1 rather than
loss, and macro-F1 keeps improving for a long time after the loss turns. The mechanism is
over-confidence: cross-entropy punishes confident mistakes, so the loss worsens while the ranking
of classes still improves.

Note for reading the plots: compare `cls_loss` against `source_val/loss`, not `loss`.
`loss = cls_loss + adaptation term`, so `loss` is not comparable to the validation loss or across
methods.

### 10.4 Domain gap

| Method   | source → target   | source → test     |
| -------- | ----------------- | ----------------- |
| baseline | −0.1781 (−25.0 %) | −0.4948 (−69.5 %) |
| bn_only  | −0.1857 (−25.8 %) | −0.4851 (−67.3 %) |
| coral    | −0.1849 (−25.8 %) | −0.4931 (−68.8 %) |
| dann     | −0.1810 (−25.4 %) | −0.4856 (−68.1 %) |

Two very different gaps: 25 % to us-south, 69 % to caribbean. No method narrowed either.

### 10.5 The caribbean failure is a minor-damage failure

Predicted class shares against the truth, on `test` (22,617 buildings):

| Class        | Actual     | baseline   | coral  | dann   |
| ------------ | ---------- | ---------- | ------ | ------ |
| no-damage    | 17.9 %     | 27.1 %     | 18.8 % | 18.0 % |
| minor-damage | **54.5 %** | **5.3 %**  | 5.3 %  | 5.6 %  |
| major-damage | 12.0 %     | 3.6 %      | 7.6 %  | 8.6 %  |
| destroyed    | **15.6 %** | **64.0 %** | 68.4 % | 67.8 % |

The model calls two-thirds of the island destroyed when a sixth of it is, and almost never
predicts minor-damage when it is the majority class. CORAL and DANN made this _worse_: both
shifted more mass onto `destroyed`.

Of the 15,570 buildings every method gets wrong, **10,844 are minor-damage** — 88 % of all
minor-damage buildings in the split. One class accounts for 70 % of the shared errors.

The cause is in 3.1: the source domain is 8.3 % minor-damage, the test domain is 54.5 %, and the
class weights push the model further towards the severe classes.

### 10.6 Error overlap — the gains are churn

| Method   | correct | accuracy | fixed | broken | net  |
| -------- | ------- | -------- | ----- | ------ | ---- |
| baseline | 5,575   | 0.2465   | —     | —      | —    |
| coral    | 5,707   | 0.2523   | 1,042 | 910    | +132 |
| dann     | 5,686   | 0.2514   | 1,013 | 902    | +111 |

Every method wrong: 15,570 (68.8 %). Every method right: 4,377 (19.4 %). A net of +132 out of
22,617 after fixing 1,042 and breaking 910 is churn, not adaptation. Always report fixed and
broken, not only the net.

### 10.7 Accuracy moves the opposite way to macro-F1

| Method   | source_val      | target_test     | test            |
| -------- | --------------- | --------------- | --------------- |
| baseline | 0.8758 ± 0.0017 | 0.6715 ± 0.0043 | 0.2477 ± 0.0072 |
| coral    | 0.8831 ± 0.0009 | 0.6658 ± 0.0061 | 0.2439 ± 0.0077 |
| dann     | 0.8790 ± 0.0018 | 0.6644 ± 0.0042 | 0.2459 ± 0.0064 |

CORAL raises in-domain accuracy and lowers target accuracy. Another reason macro-F1 is the
headline metric.

### 10.8 Domain probe (earlier bottleneck-512 runs)

How separable the two domains are in a trained model's frozen feature space, measured by training
a fresh MLP to tell them apart. **These numbers are from the archived bottleneck-512 runs and
should be re-measured on the final checkpoints.**

| Run        | source : target   | source : test     |
| ---------- | ----------------- | ----------------- |
| baseline   | 0.888 (AUC 0.947) | 0.971 (AUC 0.992) |
| coral λ=40 | 0.851             | —                 |
| dann       | 0.869             | —                 |

Proxy A-distance: baseline 1.552, dann 1.476, coral 1.404, baseline:test 1.884.

Two things this told us:

- The domains really are far apart, so attempting alignment was reasonable.
- CORAL and DANN barely moved them (0.888 → 0.851 / 0.869). The alignment mostly did not happen.

Per-class probes gave 0.883–0.895 and 0.973–0.982, so the separability is not an artefact of the
different class mixes — the images genuinely look different.

**Correction worth recording.** We first read DANN's live `dom-acc` (around 0.54–0.59 during
training) as evidence the domains were already aligned. That was wrong. Live `dom-acc` is not a
separability measure: the encoder is actively fighting that particular discriminator, lambda is
never truly zero, and BatchNorm running statistics are being updated by the target batches. The
frozen-feature probe said 0.888 for the same model. Use the probe, not `dom-acc`.

`src/domain_probe.py` now also reports `mean_gap` (distance between feature centres) and
`coral_dist` (the quantity CORAL minimises). CORAL's loss has no mean term, so a mean offset
survives it entirely — running both numbers before and after adaptation shows whether a method
moved what it was supposed to move.

---

## 11. Why CORAL and DANN could not work here

This is the main conclusion, and it is a known theoretical result rather than something specific
to our code.

Both methods align the **marginal** feature distribution p(z) across domains. That implicitly
assumes the label distribution is similar on both sides. Ours is not: minor-damage is 8.3 % of
the source and 54.5 % of the test domain (3.1).

Zhao et al. (ICML 2019) prove an information-theoretic **lower bound** on the joint
source + target error for any method that learns domain-invariant representations when the
marginal label distributions differ. Their abstract describes "a fundamental tradeoff between
learning invariant representations and achieving small joint error". Forcing p(z_source) and
p(z_target) together while keeping source accuracy high raises the floor on target error.

That is exactly our table: alignment sits consistently below the BatchNorm control, and the only
thing that transferred was a statistic that does not try to make features invariant.

The TVD numbers in 3.1 predicted this before any model was trained. Worth stating that way in
the report: it was a prediction, not a post-hoc excuse.

**What would address it**, in order of cost:

1. **Prior correction at inference.** Add `log(pi_target) - log(pi_source)` to the logits. Under
   label shift this is the Bayes-optimal correction, needs no retraining, and runs on the
   existing `predictions_*.csv`. Report it twice: with the true target prior (an upper bound on
   what prior correction can buy) and with an estimated one.
2. **CDAN** instead of DANN — conditions the discriminator on the classifier's prediction, so it
   aligns p(z|y) rather than p(z). One new file in `src/algorithms/`.
3. **Importance-weighted DANN** — reweight the domain loss by estimated target class
   proportions. The direct patch for the Zhao result.
4. **Pseudo-label self-training** — likely to fail here: the pseudo-labels would mostly say
   "destroyed" and reinforce the existing bias. Worth mentioning as considered and rejected.

---

## 12. Code conventions

- **Configs**: `configs/base.yaml` holds everything shared; each method file declares
  `_base_: base.yaml` and overrides only its own keys. Command line (`--set`) beats file, file
  beats base. One `--set` per command; a second one replaces the first.
- **Validation**: `src/config_schema.py` is a pydantic schema with `extra="forbid"`, so a typo
  like `bottlenck: 512` stops the run instead of being silently ignored. The method section is a
  discriminated union, so a DANN key inside `coral.yaml` is an error.
- **Adding a method**: write a subclass of `Algorithm` in `src/algorithms/`, add one line to
  `REGISTRY`, one class to the schema union, and the name to `METHODS` / `TARGET_METHODS` in
  `src/data/crops.py`. `train.py` never mentions method names.
- **Run folders**: `runs/<method>/seed<N>_<timestamp>/`. Nothing is overwritten; re-running the
  same name without `--resume` or `--overwrite` is refused.
- **Reproducibility**: the resolved config is saved into every run folder, and
  `crops/*/extract_info.json` records the crop and split parameters.

---

## 13. Papers used

| Paper                                                                                                                                                           | What we took                                                                                     |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [Gerard et al., _A Simple, Strong Baseline for Building Damage Detection on xBD_ (2401.17271)](https://arxiv.org/html/2401.17271v1)                             | augmentation list, AdamW 2e-4 / wd 1e-6, class weights `1/f_c`, evidence that ResNet-34 suffices |
| [Mouradi & Kshirsagar, _Robust Building Damage Detection in Cross-Disaster Settings Using Domain Adaptation_ (2603.14694)](https://arxiv.org/html/2603.14694v1) | framing of with-DA vs without-DA; its DA is supervised fine-tuning                               |
| [Sun & Saenko, _Deep CORAL_ (1607.01719)](https://arxiv.org/pdf/1607.01719)                                                                                     | the CORAL loss and the batch covariance formula                                                  |
| [Ganin et al., _Domain-Adversarial Training of Neural Networks_ (1505.07818)](https://arxiv.org/pdf/1505.07818)                                                 | GRL, the objective, the lambda ramp                                                              |
| [Zhao et al., _On Learning Invariant Representations for Domain Adaptation_ (ICML 2019)](https://arxiv.org/abs/1901.09453)                                      | the lower bound that explains our negative result (section 11)                                   |
| [Li et al., _Revisiting Batch Normalization For Practical Domain Adaptation_ / AdaBN (1603.04779)](https://arxiv.org/abs/1603.04779)                            | why BatchNorm statistics alone are a domain-adaptation method                                    |
| [xView2 1st place solution](https://github.com/asmaemou/xview2_1st_place_solution)                                                                              | how seeds are run and combined (equal-weight probability averaging)                              |

**On paper 1's augmentation claim.** Its abstract says unsharp-enhanced RGB with supervised DA
performs best, and that is true as a ranking — but the margin over plain RGB is **0.0036
macro-F1** (0.5552 vs 0.5516), from single runs with no seeds, on a test split of 10 % of 87
image pairs. Meanwhile DA itself is worth about +0.35. Across its augmentation configurations the
destroyed-class F1 swings from 0.0000 to 0.5156, which is instability, not a tuning curve.
Conclusion: spend runs on the DA comparison, treat augmentation as a secondary ablation.

Also note paper 1's "augmentations" are image _filters_ (unsharp, CLAHE, edge detection), while
paper 2's are _geometric_ transforms. Different things; we use paper 2's.

---

## 14. Problems hit (and fixes)

| Symptom                                                        | Cause                                                                                     | Fix                                                                                                                    |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: rasterio` with `(ecse552)` in the prompt | `python` resolved to a pyenv 3.8.5 shim, earlier in `PATH` than the conda env             | reorder the conda block after pyenv in `~/.zshrc`, or call the env's python by full path                               |
| `OMP: Error #15: Initializing libomp.dylib`                    | pip `torch` ships its own libomp, conda-forge `llvm-openmp` provides another              | reinstall torch from conda-forge                                                                                       |
| `ModuleNotFoundError: pkg_resources` from tensorboard          | setuptools 81+ removed it                                                                 | `conda install -n ecse552 "setuptools<81"`                                                                             |
| tensorboard shows "No Runs" with empty cards                   | old pyenv TensorBoard 2.14 reading new event files                                        | run it from the conda env                                                                                              |
| epoch lines missing from a redirected log                      | Python block-buffers stdout when redirected                                               | `PYTHONUNBUFFERED=1`, or read `history.csv`, which is written every epoch                                              |
| `--resume` wiped the earlier epochs from `history.csv`         | the loop started from an empty history and rewrote the file                               | load the existing CSV on resume and keep rows with `epoch < start_epoch`; also restore `since_best` and the RNG states |
| `bn_only` crashed the epoch print with `KeyError: 'lambda'`    | the print assumed any method logging `extra_loss` also logs `lambda`                      | make the lambda part conditional                                                                                       |
| results table silently showed only 3 of 4 methods              | `METHOD_ORDER` in notebook cell 2 did not list `bn_only`, and every table reindexes on it | add `bn_only` to `METHOD_ORDER` and `COLOR`                                                                            |

### 14.1 Timing

25 epochs, Apple Silicon (MPS), batch 128:

| Method   | minutes per run |
| -------- | --------------- |
| baseline | 26              |
| bn_only  | 33              |
| coral    | 48              |
| dann     | 48              |

Full 12-run grid: about 8 hours.

---

## 15. Things to be careful about in the report

1. The hazard sub-type changes along with the region (2.3). Do not claim a pure geographic shift.
2. Label shift is severe for caribbean (TVD 0.567). State it, and use it to explain why marginal
   alignment struggles there — with the Zhao et al. bound (section 11), not as hand-waving.
3. The split seed was chosen for class balance before any training. Say so explicitly.
4. Model selection never uses target labels. Say so explicitly; paper 1 did not.
5. Report mean ± std over 3 seeds against the baseline's own spread. Do not compare single runs —
   our own seed-3-only CORAL result was inside the noise (10.1).
6. Report the `bn_only` control. Without it, "CORAL helped" and "BatchNorm helped" cannot be told
   apart, and the honest answer here is the second one.
7. Report fixed and broken counts, not only net (10.6).
8. Our numbers are not comparable to either paper's: they do whole-image segmentation scored
   against the xView2 competition metric, we do building-crop classification.

---
