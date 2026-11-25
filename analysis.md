# Build notes: decisions, measurements and findings

Working notes for the ECSE552 project _Satellite Disaster Assessment with Domain Adaptation_.
The README says **how** to run things. This file says **why** everything is the way it is, and
records the numbers we measured so we do not have to rediscover them.

Last updated: 2026-11-24, after the DINOv2 backbone experiments and the CDAN runs.

---

## 1. Project in one paragraph

Train a building-damage classifier on labelled buildings from one region, measure how much it
loses on a different region, then test whether unsupervised domain adaptation recovers some of
that loss. Everything is building-level classification using the bounding boxes xBD already
provides — we do **not** do building detection or segmentation.

**Headline result, in two parts:**

1. **No adaptation method helped.** CORAL, DANN and CDAN all scored at or below the baseline, on
   two different backbones. The only measurable adaptation effect was BatchNorm statistics
   adapting to the target (+0.019 macro-F1), which happens for free inside CORAL and DANN and
   has nothing to do with their loss.
2. **The backbone was worth six times more.** Replacing ImageNet ResNet-34 with DINOv2 ViT-S/14
   (last two blocks fine-tuned) gained **+0.116 macro-F1** on the unseen region.

Section 10 has the numbers, section 11 the explanation.

---

## 2. Data decisions

### 2.1 Why only "wind"

The proposal asks for domain adaptation across regions. Holding the hazard type fixed removes
one confound. From the EDA over tier1 + tier3 + test (10,101 image pairs, 370,976 buildings,
19 events), **wind** is the only `disaster_type` with enough buildings in all four damage classes
across several events (at least 449 per class in every wind event).

`disaster_type` comes from the label metadata, not from the event name, and it describes **what
caused the damage**. Read from every event's post-disaster JSON:

| disaster_type | events |
| --- | --- |
| wind | hurricane-matthew, hurricane-michael, joplin-tornado, moore-tornado, tuscaloosa-tornado |
| flooding | hurricane-florence, hurricane-harvey, midwest-flooding, nepal-flooding |
| fire | pinery-bushfire, portugal-wildfire, santa-rosa-wildfire, socal-fire, woolsey-fire |
| tsunami | palu-tsunami, sunda-tsunami |
| volcano | guatemala-volcano, lower-puna-volcano |
| earthquake | mexico-earthquake |

**xBD has four hurricane-named events but only two are wind events.** Harvey (Houston 2017) and
Florence (Carolinas 2018) did their damage by flooding, so xBD files them under `flooding` and
our subset excludes them. Their class profile agrees: both are dominated by major-damage, the
signature of flood damage, while Michael and Matthew carry far more minor-damage.

So in the report write **"wind-damage hurricanes"**, not "hurricanes". An unqualified "we used
hurricane events" misdescribes what the model saw.

Five wind events: `joplin-tornado`, `moore-tornado`, `tuscaloosa-tornado`, `hurricane-michael`,
`hurricane-matthew`.

### 2.2 Locations are our own labels

The script derives a location from the event name, which gives the _storm_ name for hurricanes
(`michael`), not a place. We override it in `configs/event_location.json`:

| Event                                 | Location     | Hazard    |
| ------------------------------------- | ------------ | --------- |
| joplin-tornado, moore-tornado         | `us-central` | tornado   |
| tuscaloosa-tornado, hurricane-michael | `us-south`   | mixed     |
| hurricane-matthew                     | `caribbean`  | hurricane |

### 2.3 Domain roles

| Role          | Location   | Buildings | Labels used in training |
| ------------- | ---------- | --------- | ----------------------- |
| Source        | us-central | 37,068    | yes                     |
| Target        | us-south   | 49,226    | **no**                  |
| Held-out test | caribbean  | 22,630    | no                      |

### 2.4 Should we use tornado only? — answered with data

The hazard sub-type changes along with the region, so this was a fair question. Section 10.2
answers it: **a tornado-only project would have had no domain shift to study.**

Holding the hazard fixed, moving from us-central (Joplin, Moore) to us-south (Tuscaloosa) costs
**+0.0012 macro-F1**. The baseline scores the same on a region it has never seen as on its own
validation split. With no gap, there is nothing for domain adaptation to recover and the
experiment would have had no content.

Switching to tornado-only would also have dropped caribbean entirely, since `hurricane-matthew`
is the only non-US wind event. We would have lost the severe label shift (TVD 0.567) that
explains why alignment fails, and one of three domains.

Mixing hazards is also what deployment looks like: a model trained on past events is applied to
a new event that differs in place, hazard, season and sensor at once. A "pure geographic shift"
is an artificial construct.

The right move was to **measure** the confound rather than remove it.
`scripts/per_event.py` does that using predictions that already exist. See 10.2.

### 2.5 The xBD folders

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
expect adaptation to fail on caribbean. It did, on two backbones. Section 11 explains why this
is a theorem, not bad luck.

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

`hurricane-matthew` is also the only event with a **varying** gsd (1.59 to 2.77).

### 3.3 Black (no-data) pixels

Only **69 of 108,916 crops (0.06 %)** are more than half black. The Dataset filters at
`max_black=0.5`, which removes those 69 and changes nothing else.

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

Worth noting: by this measure the **target** (1.0–1.1) is further from the source (0.8) than the
test is (0.7), while class distribution and building size both say caribbean is the outlier. The
shift is multi-dimensional, and a single global alignment cannot address all axes at once.

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
  building filling about two thirds of the crop.
- **64 x 64**: typical box is 35–62 px, so this is close to native. The limit is the satellite
  resolution, not the output size.
- **Scale-normalised, not a fixed window**: caribbean buildings are half the linear size of the
  others. A fixed window would let the model use size as a shortcut that breaks on the test
  domain.
- **Floor 16 px, cap 256 px.**

DINOv2 needs an input that is a multiple of 14, so the **encoder** resizes 64 → 126 on the fly
(`model.input_size`). The crops folder is unchanged, so both backbones read identical data.

### 4.2 Filters

- `un-classified` dropped (3,857 buildings).
- `side < 5` dropped (8 buildings).
- We first tried `side < 8`, which drops 1.02 % overall but **18.4 % of the dropped crops were
  destroyed** — a 2x over-representation. Lowering the threshold to 5 removes that bias.

After filtering: **108,916 crops** from 112,781 buildings.

### 4.3 Augmentation

Geometric only, applied with the **same random parameters** to the pre and post crop: horizontal
flip, 90° rotations, ±10° rotation, 0.9–1.1 scale. Only on `source_train` and `target_train`,
never on the evaluation splits.

**No photometric augmentation**, because changing colours would interfere with exactly what
CORAL and DANN are trying to align.

Implementation detail: the random numbers come from Python's `random`, not numpy. PyTorch's
DataLoader reseeds `random` in each worker but **not** numpy's global generator.

---

## 5. Splits

### 5.1 Stratified group split

Three constraints at once: a whole image goes to one split; each event contributes the same
fraction; the damage-class mix is similar on both sides. Plain random image selection fails the
third, because in a tornado track one image can be almost entirely destroyed and the next
untouched. Our first split (random, seed 42) gave `source_val` 20.1 % destroyed against 11.4 %
in `source_train`.

We use `StratifiedGroupKFold` with `groups=image_id` and `y=label_id`, run inside each event.
On synthetic data with the same skew, across 8 seeds:

|            | mean TVD | worst | best  |
| ---------- | -------- | ----- | ----- |
| random     | 0.124    | 0.269 | 0.027 |
| stratified | 0.040    | 0.057 | 0.022 |

### 5.2 The chosen seed

Stratified seeds still vary (0.012 to 0.108), so we scanned 10 and chose **seed 8**:
source TVD 0.012, target TVD 0.021. The split was chosen for balance **before any model was
trained**, using no test information. Choosing a split after seeing test scores would not be
legitimate.

### 5.3 The five splits

Counts after the Dataset drops the 69 mostly-black crops:

| Split          | Buildings | Used for                                       |
| -------------- | --------- | ---------------------------------------------- |
| `source_train` | 31,339    | training, with labels                          |
| `source_val`   | 5,699     | selecting the best epoch                       |
| `target_train` | 41,272    | unlabelled target batches                      |
| `target_test`  | 7,920     | **main result**                                |
| `test`         | 22,617    | unseen domain, hardest case                    |

**The split is fixed for every run.** `run.seed` changes only the model initialisation and the
data order.

---

## 6. Model

### 6.1 Architecture

```
pre  crop ─┐
           ├─ encoder (shared weights) ─> f_pre, f_post
post crop ─┘
f = [f_pre, f_post, f_post − f_pre]
z = Dropout(ReLU(Linear(f, 256)))          the feature CORAL / DANN / CDAN align
logits = Linear(z, 4)
```

Two encoder families, chosen by `model.backbone`:

| Backbone | Pretraining | Params | Input | Feature |
| --- | --- | --- | --- | --- |
| `resnet34` | ImageNet, supervised | 21.7 M total | 64×64 native | 512 |
| `dinov2_vits14` | LVD-142M, self-supervised | 22.1 M | 126×126 (multiple of 14) | 384, or 768 with `cls+avg` |

For DINOv2 we use `vit_pool: cls+avg` — the class token concatenated with the mean patch token,
the usual linear-probe read-out. Taking the class token alone throws away the patch-level detail
that is a ViT's main advantage over a ResNet's 2×2 final feature map.

`model.trainable_blocks` controls freezing: `-1` whole encoder, `0` frozen, `n` last n blocks.
Note it means different things per family — DINOv2 has 12 transformer blocks, so `n=2` trains
16 % of the encoder; ResNet-34 has 5 groups and most of its parameters are late, so `n=2` trains
94 %.

### 6.2 Why these choices

- **Shared weights, two branches.** The xView2 work reports that plain 6-channel stacking and
  image subtraction both performed poorly, while shared-weight branches worked.
- **`f_post − f_pre`** is our addition, not from the papers. Ablatable with
  `--set model.fusion=concat`.
- **Correction worth remembering:** the xView2 competition's best result was _not_ "ResNet34 +
  SE-ResNeXt50". It was an ensemble of 4 architectures x 3 seeds, averaged with equal weights.
- **Bottleneck 256, not 512.** CORAL estimates a `d x d` covariance from a single batch; at
  `d = 512` with batch 64 that estimate has rank 63 out of 512 and is mostly noise. Batch was
  raised to 128 at the same time, so the batch now exceeds the feature dimension.
- **DINOv2 at 126, not 224.** 126 gives 81 patch tokens at about 6× the ResNet cost; 224 gives
  256 tokens at 21×. 126 is what fits on a laptop. **Our DINOv2 numbers are therefore a lower
  bound** — the model is being used outside its native resolution.

### 6.3 What we checked before trusting the DINOv2 claim

The suggestion to use DINOv2 came with several arguments. We checked each:

| Claim | Verdict |
| --- | --- |
| ImageNet CNNs are texture-biased | Holds (Geirhos et al., ICLR 2019) and fits our failure mode |
| ViT-S/14 "matches ResNet34's speed/memory" | **True on parameters, false on compute.** 21× at 224, 6× at 126 |
| DINOv2 patch tokens preserve spatial detail | Holds; the paper shows frozen features beating baselines on dense tasks |
| LVD-142M "includes aerial/overhead views" | **Not supported.** Named sources are ImageNet-22k/1k, Google Landmarks and fine-grained natural-image sets |
| Difference of DINOv2 embeddings is "cleaner" | Untested speculation |

A 2025 systematic study of pretraining for satellite tasks found none of the satellite-specific
pretrained models consistently beat general-purpose baselines at ViT-B scale, so "DINOv2 will
obviously be much better on overhead imagery" was not safe to assume. We measured it instead.

---

## 7. Algorithms

All methods use the identical network. Only the loss changes, so a difference between methods
cannot come from the architecture.

| Config     | Sees target images | Target loss      | Role                    |
| ---------- | ------------------ | ---------------- | ----------------------- |
| `baseline` | no                 | —                | the number to beat      |
| `bn_only`  | yes                | **zero**         | control: BatchNorm only |
| `coral`    | yes                | covariance match | unsupervised DA         |
| `dann`     | yes                | adversarial      | unsupervised DA         |
| `cdan`     | yes                | conditional adv. | unsupervised DA         |

`oracle` (source + **labelled** target) was dropped. Its seed-3 run scored 0.5085 on
`target_test`, below the baseline's 0.5324, which we never explained.

### 7.1 CORAL

Deep CORAL (arXiv 1607.01719):

```
L_coral = || C_s - C_t ||_F^2 / (4 d^2)
L = L_cls + lambda * L_coral
```

No new parameters. Covariances are over the batch, so `drop_last=True` matters. `lambda_weight`
is 40 with a 5-epoch ramp. **The loss scale depends on `d`, so `lambda_weight` has to be
rechecked whenever `model.bottleneck` changes** — this is an open item for the DINOv2 CORAL run.

### 7.2 DANN

Ganin et al. (arXiv 1505.07818):

```
L = L_cls - lambda * L_domain            (the gradient reversal layer implements the minus)
lambda_p = 2 / (1 + exp(-gamma * p)) - 1,   gamma = 10, p = progress 0 -> 1
```

With gamma = 10 the ramp is 85 % complete by p = 0.25; **lower** gamma stretches it out (we got
this backwards once).

### 7.3 CDAN

Long et al. (arXiv 1705.10667). The discriminator sees the outer product of the feature and the
predicted class distribution:

```
T(z, g) = g (x) z        flattened to (C * d) = 4 * 256 = 1024
```

so the encoder is pushed to align p(z|y) rather than p(z). The softmax is **detached** before
the outer product — the conditioning tells the discriminator which class it is looking at, and
must not train the classifier.

`entropy: true` gives CDAN+E, which weights examples by prediction confidence. We left it off:
our model is confidently *wrong* on caribbean, so +E would upweight exactly those examples.

Verified against hand calculations: the multilinear map matches an explicit outer product
exactly; entropy weight is 1.25 for a uniform prediction and 2.00 for a one-hot one; the
gradient reversal flips the encoder gradient exactly (cosine −1.0000) and leaves the
discriminator gradient untouched; the detach produces no classifier gradient.

### 7.4 bn_only — the control

CORAL and DANN forward an unlabelled target batch every training step while the model is in
train mode. Every BatchNorm layer therefore updates its running mean and variance with target
statistics. The baseline never does. That is free adaptation — essentially AdaBN — and it has
nothing to do with the alignment loss.

`bn_only` forwards the same target batches and returns a loss of exactly zero:

```
baseline -> bn_only      what BatchNorm adaptation is worth on its own
bn_only  -> coral/dann   what the alignment loss actually adds
```

**No equivalent control is needed on DINOv2.** A ViT uses LayerNorm, which keeps no running
statistics, so nothing adapts for free there. That makes the DINOv2 DA comparison cleaner than
the ResNet one.

### 7.5 Separate forward passes — correction

**An earlier version of this file was wrong here.** It said two separate forward passes stop
BatchNorm from mixing the domains. That is only half true. Separate passes stop the two domains
being mixed inside one batch statistic. They do **not** stop BatchNorm's _running_ statistics
from being updated by target data:

```
train mode, no_grad : running_mean moved by 0.001108
eval  mode, no_grad : running_mean moved by 0.000000
```

Confirmed by comparing a baseline and a bn_only checkpoint trained with the same seed:

```
BatchNorm running stats : max abs diff 0.817259  (72 tensors)
learned weights         : max abs diff 0.004512
```

### 7.6 Terminology

"Transfer learning" is ambiguous here, because _every_ run uses pretrained weights. Name the
runs by **which labels and which images they see**. Paper 1's method is supervised fine-tuning
on labelled target data, so its numbers are not comparable to our unsupervised results.

---

## 8. Training settings

| Setting       | Value                                                 | Source                                 |
| ------------- | ----------------------------------------------------- | -------------------------------------- |
| loss          | weighted cross-entropy                                |                                        |
| class weights | `w_c = N / (C * n_c)`, from the source training split | paper 2 uses `1/f_c`                   |
| optimiser     | AdamW                                                 | paper 2                                |
| lr            | 2e-4 (new layers), 2e-5 (encoder)                     | paper 2's 2e-4                         |
| weight decay  | 1e-6                                                  | paper 2                                |
| scheduler     | cosine over `optim.epochs`                            |                                        |
| batch         | 128 source + 128 target                               | raised from 64 for CORAL's covariance  |
| bottleneck    | 256                                                   | see 6.2                                |
| epochs        | 25                                                    | see 10.6                               |
| AMP           | CUDA only                                             | not reliable on MPS                    |

Class weights on the chosen split: `[0.334, 3.056, 6.438, 1.910]`, a 19:1 ratio.

We chose class weights over a `WeightedRandomSampler` for one reason specific to this project: a
sampler would make the source batch class-balanced while the target batch cannot be (no labels),
so CORAL and DANN would partly be aligning away an artefact we created ourselves.

### 8.1 Model selection

Best epoch is chosen on `source_val` macro-F1. **Never on target labels** — that is the mistake
to avoid, and paper 1 is loose about it.

### 8.2 Early stopping

Implemented but **off**, and it stayed off. Section 10.7 shows why: validation loss turns upward
10–18 epochs before macro-F1 peaks, so patience on the loss would stop every run far too early.

---

## 9. Metrics

- **macro-F1** is the headline. 74 % of source buildings are no-damage, so accuracy would reward
  a model that ignores the damaged classes.
- Also reported: per-class F1, recall, precision, balanced accuracy, confusion matrix.
- 3 seeds per configuration, **mean ± std**. The baseline's own spread is the bar for what
  counts as a difference.

---

## 10. Results

Reproduce with `python scripts/summarize.py`.

### 10.1 Backbones — macro-F1, mean ± std over 3 seeds

| Configuration | source_val | target_test | test (caribbean) |
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

**Why the frozen-ResNet row matters.** The first DINOv2 run was frozen, so it changed two things
at once — backbone and freezing. Freezing a ResNet makes everything worse (−0.103, −0.092,
−0.006), which rules out "freezing is what helped". Frozen against frozen, DINOv2 beats ResNet
by +0.112, +0.075 and +0.124 on the three splits. That is the controlled comparison.

**The frozen-vs-ft2 trade.** Unfreezing two blocks buys +0.043 on source_val and +0.044 on
target_test, and costs −0.010 on caribbean. Adaptation to the source helps nearby domains and
slightly hurts the far one — the same pattern the frozen-ResNet control showed, just milder.

### 10.2 Where the shift actually comes from

`scripts/per_event.py` scores `target_test` separately for its two events. Together with
`source_val` and the caribbean `test` split that gives a near-factorial design. Every cell
except `source_val` is out-of-domain for the baseline, which never sees target images, so the
baseline column is the clean one.

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

Three things follow.

**1. Geographic shift between US states is nil.** Missouri/Oklahoma → Alabama costs +0.001. The
−0.178 aggregate gap we report on `target_test` is entirely `hurricane-michael`.

**2. Both hazard and large-scale geography matter, by similar amounts.** Changing hazard inside
the US costs −0.267. Changing country at fixed hazard costs −0.229. An earlier draft of this
file said "geography costs nothing" — that was wrong, and came from testing only a small
geographic change.

**3. The backbone fixed geography, not hazard.**

| effect | DINOv2 − ResNet |
|---|---|
| geography, small | +0.0143 |
| **geography, large** | **+0.1150** |
| hazard | **−0.0575** |

DINOv2's entire +0.116 on caribbean is geographic robustness. On the hazard axis it is slightly
*worse* than ResNet. This also resolves what looked like an anomaly in the aggregate numbers:
DINOv2 gained +0.116 on matthew but +0.0005 on michael, both hurricanes. They differ on the
geographic axis, which is the one DINOv2 helps with.

**Nothing improved the hurricane event.** Across every method and both backbones, the best score
on `michael` is 0.4462 against the baseline's 0.4457:

| config | michael | vs its own baseline |
|---|---|---|
| baseline (ResNet) | 0.4457 | — |
| bn_only | 0.4435 | −0.0022 |
| coral | 0.4436 | −0.0021 |
| dann | 0.4435 | −0.0022 |
| DINOv2 ft2 | 0.4462 | +0.0005 |
| coral on DINOv2 | 0.4333 | −0.0129 |
| dann on DINOv2 | 0.4431 | −0.0031 |
| cdan on DINOv2 | 0.4007 | −0.0455 |

**Limits of this design.** `hurricane-matthew` is the only non-US wind event in xBD, so the
−0.229 geographic effect rests on a single event and cannot be replicated inside this dataset.
"Geography, large" also bundles country, building materials, building size (23 px against
41 px), varying GSD and a very different label distribution — realistic regional transfer, not a
pure coordinate change. Two cells of the design do not exist in xBD's wind data: us-central
hurricane and caribbean tornado.

### 10.3 Domain adaptation on ResNet-34 — mean ± std over 3 seeds

| Method | source_val | target_test | test |
|---|---|---|---|
| baseline | 0.7116 ± 0.0039 | 0.5335 ± 0.0035 | 0.2168 ± 0.0045 |
| **bn_only** | **0.7207** ± 0.0028 | **0.5350** ± 0.0024 | **0.2356** ± 0.0080 |
| coral | 0.7166 ± 0.0030 | 0.5317 ± 0.0030 | 0.2235 ± 0.0105 |
| dann | 0.7130 ± 0.0028 | 0.5320 ± 0.0042 | 0.2274 ± 0.0084 |

Decomposition, against the baseline's own seed-to-seed spread:

| Split | baseline sd | BatchNorm | CORAL loss | DANN loss |
|---|---|---|---|---|
| source_val | 0.0039 | **+0.0091** above noise | −0.0041 worse | −0.0077 worse |
| target_test | 0.0035 | +0.0015 within noise | −0.0033 within noise | −0.0030 within noise |
| test | 0.0045 | **+0.0188** above noise | −0.0120 worse | −0.0082 worse |

On `source_val` and `test` the separation is complete, not just a difference of means — every
`bn_only` seed beats every `baseline` seed.

Without the `bn_only` control we would have concluded "CORAL ≈ baseline, no effect". The real
story is that the methods did transfer something, it was BatchNorm riding along for free, and
their actual objectives subtracted from it.

**Retired result.** An earlier seed-3-only run showed CORAL at +0.008 on `target_test`. With
three seeds that is inside the noise. Single-seed differences on this dataset mean nothing.

### 10.4 Domain adaptation on DINOv2 — seed 3 only

All rows are seed 3, so this is like-for-like. σ is the DINOv2 ft2 baseline's 3-seed spread.

| Method | source_val | target_test | test |
|---|---|---|---|
| baseline | 0.7606 | 0.5519 | 0.3291 |
| CORAL | 0.7565 (−0.9σ) | 0.5407 (**−2.2σ**) | 0.3370 (+0.5σ) |
| DANN | 0.7484 (**−2.7σ**) | 0.5482 (−0.7σ) | 0.2873 (**−2.8σ**) |
| CDAN | 0.7460 (**−3.2σ**) | 0.5034 (**−9.5σ**) | 0.3171 (−0.8σ) |

No method beat the baseline anywhere. CORAL's +0.008 on caribbean is half a standard deviation —
noise. **CDAN, the method specifically designed for differing class conditionals, was the
worst.**

This replicates the ResNet result on a completely different backbone with no BatchNorm
confound, which makes the finding far stronger than a single-backbone result.

**The flat aggregate hides two opposite effects.** Split by event (10.2), the DA methods *help*
on the event that barely differs from the source and *hurt* on the one that does:

| method | tuscaloosa (small shift) | michael (large shift) |
|---|---|---|
| CORAL | +0.0086 | −0.0129 |
| DANN | +0.0028 | −0.0031 |
| CDAN | +0.0089 | −0.0455 |

That is the Zhao tradeoff visible inside a single split: alignment pays when there is little to
align and costs when there is a lot. It is also why the aggregate `target_test` numbers looked
like flat noise — two opposite effects averaging out.

**Caveat:** single seed. The deltas exceed the baseline's seed spread, but three seeds each
would be better if there is time.

### 10.5 Why CDAN failed — a partial explanation

Our first hypothesis was that CDAN forces the target's predicted label distribution toward the
source's, since aligning the joint p(g, z) implies aligning the marginal p(g). We checked it on
`target_test`:

| | no-dmg | minor | major | destr |
|---|---|---|---|---|
| source prior | 74.7 % | 8.3 % | 3.9 % | 13.1 % |
| **actual target** | 68.8 % | 19.3 % | 6.9 % | 5.1 % |
| baseline predicts | 56.7 % | 31.4 % | 6.3 % | 5.6 % |
| CDAN predicts | 58.2 % | 19.3 % | 8.3 % | 14.2 % |

**Supported:** CDAN did pull the predicted prior toward the source. Distance to the source prior
fell from 25.5 % to 16.5 %. The clearest case is `destroyed`, which went 5.6 % → 14.2 % when the
truth is 5.1 % and the source is 13.1 % — it nearly tripled its destroyed predictions and landed
almost exactly on the source prior.

**Not supported:** CDAN also got *closer to the truth overall* — distance to the actual target
distribution improved from 12.6 % to 10.5 %, and `minor-damage` went from a badly over-predicted
31.4 % to exactly the correct 19.3 %.

So the hypothesis holds for one class and fails as a global story. CDAN has a better predicted
marginal and a macro-F1 that is 0.049 lower, which means its per-example decisions got worse
even as its overall class balance improved. **Open:** compare per-class F1 from `metrics.json`
to find where the 0.049 went. The prediction is that `destroyed` F1 collapses — at 14.2 %
predicted against 5.1 % actual, precision cannot exceed about 0.36.

### 10.6 Domain gap

| Configuration | source → target | source → test |
|---|---|---|
| ResNet-34 baseline | −0.1781 (−25.0 %) | −0.4948 (−69.5 %) |
| ResNet-34 bn_only | −0.1857 (−25.8 %) | −0.4851 (−67.3 %) |
| DINOv2 frozen | −0.1992 (−28.0 %) | −0.3703 (−52.0 %) |
| DINOv2 ft2 | −0.1981 (−26.2 %) | −0.4230 (−56.0 %) |

DINOv2 narrowed the caribbean gap from 69 % to 52–56 %. No *method* narrowed anything. Note the
relative gap to us-south did not improve — DINOv2 raised both ends roughly equally there.

**These aggregates hide the event split.** Section 10.2 decomposes the same gap into a zero
geographic effect within the US and a large hazard effect. Quote 10.2 in the report, not this
table.

### 10.7 Epoch budget, and why validation loss is a bad stopping signal

Across every fine-tuned run, validation loss bottoms out 10–18 epochs **before** macro-F1 peaks:

| group | valloss_min_ep | best_f1_ep |
|---|---|---|
| baseline (ResNet) | 7, 7, 8 | 7, 20, 22 |
| bn_only | 5, 7, 7 | 16, 17, 24 |
| coral | 7, 7, 8 | 16, 20, 22 |
| dann | 1, 2, 7 | 16, 18, 18 |
| dinov2 ft2 | 4, 6, 8 | 15, 24, 25 |
| dinov2 frozen | 15, 18, 19 | 13, 20, 20 |

The mechanism is over-confidence: cross-entropy punishes confident mistakes, so the loss worsens
while the ranking of classes still improves. This justifies two choices directly — selecting on
macro-F1 rather than loss, and keeping early stopping off.

DANN has the worst loss divergence (val_rise 0.33–0.39) and was also the weakest method. Frozen
DINOv2 barely diverges at all (0.001–0.009).

**Caveat on ft2:** two of its three seeds peaked at epoch 24 and 25, the budget edge. Its
numbers are a **lower bound**. We did not extend the budget because all 25 runs share it and
that comparability is worth more than a few thousandths.

Reading the plots: compare `cls_loss` against `source_val/loss`, not `loss`.
`loss = cls_loss + adaptation term`, so `loss` is not comparable across methods.

### 10.8 The caribbean failure is a minor-damage failure

Predicted class shares against the truth, on `test` (22,617 buildings), ResNet-34:

| Class | Actual | baseline | coral | dann |
|---|---|---|---|---|
| no-damage | 17.9 % | 27.1 % | 18.8 % | 18.0 % |
| minor-damage | **54.5 %** | **5.3 %** | 5.3 % | 5.6 % |
| major-damage | 12.0 % | 3.6 % | 7.6 % | 8.6 % |
| destroyed | **15.6 %** | **64.0 %** | 68.4 % | 67.8 % |

Of the 15,570 buildings every method gets wrong, **10,844 are minor-damage** — 88 % of all
minor-damage buildings in the split. One class accounts for 70 % of the shared errors.

The cause is in 3.1: the source is 8.3 % minor-damage, the test domain is 54.5 %, and the class
weights push the model further towards the severe classes.

### 10.9 Error overlap — the gains are churn

| Method | correct | accuracy | fixed | broken | net |
|---|---|---|---|---|---|
| baseline | 5,575 | 0.2465 | — | — | — |
| coral | 5,707 | 0.2523 | 1,042 | 910 | +132 |
| dann | 5,686 | 0.2514 | 1,013 | 902 | +111 |

Every method wrong: 15,570 (68.8 %). Every method right: 4,377 (19.4 %). A net of +132 after
fixing 1,042 and breaking 910 is churn, not adaptation. Always report fixed and broken, not only
the net.

### 10.10 Why macro-F1 is low in absolute terms

| split | macro-F1 (best config) | % of in-domain |
|---|---|---|
| source_val | 0.755 | — |
| target_test | 0.557 | 74 % |
| test | 0.332 | 44 % |

The in-domain number is the ceiling, and it is only 0.755. Two reasons:

1. **macro-F1 weights all four classes equally**, so `major-damage` (F1 ~0.31 on target) costs as
   much as `no-damage` (~0.79). Those weak classes are 3.9 % and 8.3 % of the training data.
2. **The task is hard at this resolution.** A building is 23–41 pixels at ~2 m GSD, and
   "minor damage" means a few missing shingles. The xBD minor-vs-major boundary is also
   subjective for human annotators.

So the low cross-domain numbers are partly the domain gap and partly the task itself.

### 10.11 Domain probe (earlier bottleneck-512 runs)

How separable two domains are in a run's frozen feature space, measured by training a fresh MLP
to tell them apart. **These are from the archived bottleneck-512 ResNet runs and should be
re-measured on the final checkpoints.**

| Run        | source : target   | source : test     |
| ---------- | ----------------- | ----------------- |
| baseline   | 0.888 (AUC 0.947) | 0.971 (AUC 0.992) |
| coral λ=40 | 0.851             | —                 |
| dann       | 0.869             | —                 |

Proxy A-distance: baseline 1.552, dann 1.476, coral 1.404, baseline:test 1.884.

- The domains really are far apart, so attempting alignment was reasonable.
- CORAL and DANN barely moved them (0.888 → 0.851 / 0.869). The alignment mostly did not happen.
- Per-class probes gave 0.883–0.895 and 0.973–0.982, so the separability is **not** an artefact
  of the different class mixes — same-class buildings from the two regions genuinely look
  different. This was the strongest evidence in the project for trying a better backbone.

**Correction worth recording.** We first read DANN's live `dom-acc` (0.54–0.59 during training)
as evidence the domains were already aligned. That was wrong. Live `dom-acc` is not a
separability measure: the encoder is actively fighting that particular discriminator, lambda is
never truly zero, and BatchNorm running statistics are being updated by the target batches. The
frozen-feature probe said 0.888 for the same model. **Use the probe, not `dom-acc`.**

---

## 11. Why none of the methods worked

Both CORAL and DANN align the **marginal** feature distribution p(z). That implicitly assumes a
similar label distribution on both sides. Ours is not: minor-damage is 8.3 % of the source and
54.5 % of the caribbean test split.

Zhao et al. (ICML 2019) prove an information-theoretic **lower bound** on the joint
source + target error for any method that learns domain-invariant representations when the
marginal label distributions differ — "a fundamental tradeoff between learning invariant
representations and achieving small joint error". Forcing p(z_source) and p(z_target) together
while keeping source accuracy high raises the floor on target error.

CDAN was supposed to be the answer, because it conditions on the predicted class. It was the
worst performer. Conditioning on a *predicted* label does not estimate the target prior, and our
predictions on the target are themselves corrupted by the label shift, so the conditioning
signal is unreliable where it matters most.

The TVD numbers in 3.1 predicted all of this before any model was trained. State it that way in
the report: it was a prediction, not a post-hoc excuse.

**What the backbone result adds.** The same three methods failed on two very different
representations, while simply changing the representation gained +0.116 on caribbean. For this
problem, where the features come from matters far more than any alignment objective applied on
top of them.

**What the per-event split adds (10.2).** The shift has two axes of roughly equal size: hazard
type (−0.267) and large-scale geography (−0.229). Geographic shift between US states is nil.
No method improved either axis, and the one thing that did help — the backbone — helped only on
the geographic axis and was slightly worse on the hazard axis.

The split also shows the Zhao tradeoff directly rather than by inference. Within `target_test`,
CORAL and CDAN *gain* about +0.009 on `tuscaloosa`, where the shift is negligible, and *lose*
0.013 to 0.046 on `michael`, where it is large. Alignment pays when there is little to align and
costs when there is a lot. That is the predicted behaviour, measured inside one split.

**What would actually address the label shift**, in order of cost:

1. **Prior correction at inference.** Add `log(pi_target) - log(pi_source)` to the logits. Under
   label shift this is the Bayes-optimal correction, needs no retraining, and runs on the
   existing `predictions_*.csv`. Report it twice: with the true target prior (an upper bound on
   what prior correction can buy) and with an estimated one. **Still not done — cheapest
   remaining experiment.**
2. **Seed ensemble.** Average the softmax of the three ft2 seeds. Free, and it is what the
   xView2 winners did.
3. **Importance-weighted DANN** — reweight the domain loss by estimated target class
   proportions. The direct patch for the Zhao result.
4. **Pseudo-label self-training** — likely to fail here: the pseudo-labels would mostly say
   "destroyed" and reinforce the existing bias. Worth mentioning as considered and rejected.

---

## 12. Code conventions

- **Configs**: `configs/base.yaml` holds everything shared; each method file declares
  `_base_: base.yaml`. Command line (`--set`) beats file, file beats base.
- **Validation**: `src/config_schema.py` is a pydantic schema with `extra="forbid"`, so a typo
  stops the run instead of being silently ignored. The method section is a discriminated union,
  so a DANN key inside `coral.yaml` is an error.
- **Adding a method**: write a subclass of `Algorithm` in `src/algorithms/`, add one line to
  `REGISTRY`, one class to the schema union, and the name to `METHODS` / `TARGET_METHODS` in
  `src/data/crops.py`. `train.py` never mentions method names.
- **Run folders**: `runs/<group>/seed<N>_<timestamp>/`. Set `run.group` explicitly whenever a
  configuration differs from the plain method — otherwise a DINOv2 baseline lands in
  `runs/baseline/` next to the ResNet one and the summary averages them together.
- **Reproducibility**: the resolved config is saved into every run folder, `metrics.json`
  records the backbone and input size, and `crops/*/extract_info.json` records the crop and
  split parameters.

---

## 13. Papers used

| Paper | What we took |
| --- | --- |
| [Gerard et al., _A Simple, Strong Baseline for Building Damage Detection on xBD_ (2401.17271)](https://arxiv.org/html/2401.17271v1) | augmentation list, AdamW 2e-4 / wd 1e-6, class weights `1/f_c`, evidence that ResNet-34 suffices |
| [Mouradi & Kshirsagar, _Robust Building Damage Detection in Cross-Disaster Settings Using Domain Adaptation_ (2603.14694)](https://arxiv.org/html/2603.14694v1) | framing of with-DA vs without-DA; its DA is supervised fine-tuning |
| [Sun & Saenko, _Deep CORAL_ (1607.01719)](https://arxiv.org/pdf/1607.01719) | the CORAL loss and the batch covariance formula |
| [Ganin et al., _Domain-Adversarial Training of Neural Networks_ (1505.07818)](https://arxiv.org/pdf/1505.07818) | GRL, the objective, the lambda ramp |
| [Long et al., _Conditional Adversarial Domain Adaptation_ / CDAN (1705.10667)](https://arxiv.org/abs/1705.10667) | the multilinear map, entropy conditioning |
| [Zhao et al., _On Learning Invariant Representations for Domain Adaptation_ (ICML 2019)](https://arxiv.org/abs/1901.09453) | the lower bound that explains our negative result (section 11) |
| [Li et al., _Revisiting Batch Normalization For Practical Domain Adaptation_ / AdaBN (1603.04779)](https://arxiv.org/abs/1603.04779) | why BatchNorm statistics alone are a domain-adaptation method |
| [Oquab et al., _DINOv2_ (2304.07193)](https://arxiv.org/pdf/2304.07193) | the backbone; LVD-142M composition; patch-level feature quality |
| [Geirhos et al., _ImageNet-trained CNNs are biased towards texture_ (ICLR 2019)](https://arxiv.org/abs/1811.12231) | why ImageNet features may transfer badly across regions |
| [xView2 1st place solution](https://github.com/asmaemou/xview2_1st_place_solution) | how seeds are run and combined (equal-weight probability averaging) |

**On paper 1's augmentation claim.** Its abstract says unsharp-enhanced RGB with supervised DA
performs best, and that is true as a ranking — but the margin over plain RGB is **0.0036
macro-F1** (0.5552 vs 0.5516), from single runs with no seeds, on a test split of 10 % of 87
image pairs. Meanwhile DA itself is worth about +0.35. Across its augmentation configurations the
destroyed-class F1 swings from 0.0000 to 0.5156, which is instability, not a tuning curve.

Also note paper 1's "augmentations" are image _filters_ (unsharp, CLAHE, edge detection), while
paper 2's are _geometric_ transforms. We use paper 2's.

---

## 14. Problems hit (and fixes)

| Symptom | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: rasterio` with `(ecse552)` in the prompt | `python` resolved to a pyenv 3.8.5 shim, earlier in `PATH` | reorder the conda block after pyenv in `~/.zshrc` |
| `OMP: Error #15: Initializing libomp.dylib` | pip `torch` and conda-forge `llvm-openmp` each ship an OpenMP | reinstall torch from conda-forge |
| `ModuleNotFoundError: pkg_resources` from tensorboard | setuptools 81+ removed it | `conda install -n ecse552 "setuptools<81"` |
| tensorboard shows "No Runs" with empty cards | old pyenv TensorBoard 2.14 reading new event files | run it from the conda env |
| epoch lines missing from a redirected log | Python block-buffers stdout when redirected | `PYTHONUNBUFFERED=1`, or read `history.csv` |
| `--resume` wiped earlier epochs from `history.csv` | the loop started from an empty history and rewrote the file | load the existing CSV on resume; also restore `since_best` and the RNG states |
| `bn_only` crashed the epoch print with `KeyError: 'lambda'` | the print assumed any method logging `extra_loss` also logs `lambda` | make the lambda part conditional |
| results table silently showed only 3 of 4 methods | `METHOD_ORDER` in notebook cell 2 did not list `bn_only` | add it to `METHOD_ORDER` and `COLOR` |
| summary would have averaged frozen and ft2 DINOv2 together | grouping was by `(backbone, method)`, which they share | group by run folder instead |
| `SyntaxError: f-string: unmatched '['` in a helper script | nested same-type quotes in an f-string, legal only on Python 3.12+ | check scripts with `ast.parse(src, feature_version=(3,10))` |

### 14.1 Timing

25 epochs, Apple Silicon (MPS), batch 128:

| Configuration | minutes/run |
| --- | --- |
| ResNet-34 baseline | 26 |
| ResNet-34 bn_only | 33 |
| ResNet-34 coral / dann | 48 |
| DINOv2 frozen @126 | 71 |
| DINOv2 ft2 @126 | 87 |
| DINOv2 ft2 + coral / dann / cdan | 152–154 |

---

## 15. Things to be careful about in the report

1. **The hazard sub-type changes along with the region** (2.4). Do not claim a pure geographic
   shift. Run `scripts/per_event.py` and report the tornado-vs-hurricane split instead.
2. **Label shift is severe for caribbean** (TVD 0.567). Use the Zhao et al. bound to explain why
   marginal alignment struggles there, not hand-waving.
3. **The split seed was chosen for class balance before any training.** Say so explicitly.
4. **Model selection never uses target labels.** Say so explicitly; paper 1 did not.
5. **Report mean ± std against the baseline's own spread.** Do not compare single runs — our own
   seed-3-only CORAL result was inside the noise.
6. **Report the `bn_only` control** for the ResNet results. Without it, "CORAL helped" and
   "BatchNorm helped" cannot be told apart, and the honest answer is the second one.
7. **Say that no such control is needed on DINOv2** (LayerNorm has no running statistics). That
   is why the DINOv2 DA comparison is the cleaner one.
8. **Report fixed and broken counts**, not only net (10.9).
9. **State the DINOv2 caveats**: run at 126 rather than its native 224, two of three ft2 seeds
   peaked at the budget edge, and the DINOv2 DA runs are single-seed. All three make our DINOv2
   numbers a lower bound.
10. **Describe the shift by what we measured, not by the project title.** Within the wind
    category, two axes matter and they are similar in size: hazard type (−0.267) and
    large-scale geography (−0.229). Geographic shift between US states is nil (+0.001). Prefer
    "cross-event shift" to "cross-geographic shift" in the text.
11. **`hurricane-matthew` is the only non-US wind event in xBD.** The −0.229 geographic effect
    therefore rests on a single event and cannot be replicated inside this dataset. State it as
    a caveat, not a headline.
12. **"Geography, large" is not a pure geographic change.** It bundles country, building
    materials, building size (23 px against 41 px), varying GSD and a very different label
    distribution. Call it regional transfer.
13. **Say "wind-damage hurricanes", not "hurricanes"** (2.1). Two of xBD's four hurricane-named
    events are categorised as flooding and are not in our subset.
14. **Report the per-event split, not only the aggregate** (10.2). The aggregate `target_test`
    numbers average two opposite effects and look like flat noise as a result.
15. **Our numbers are not comparable to either paper's**: they do whole-image segmentation
    scored against the xView2 competition metric, we do building-crop classification.

---

## 16. Open items

In order of value per hour:

- [ ] **Run `scripts/per_event.py`** — separates geography from hazard type using predictions
      that already exist. Needed for report point 15.1.
- [ ] **Seed ensemble** — average the softmax of the three DINOv2 ft2 seeds. No training.
- [ ] **Prior correction at inference** (11, item 1). No training. The cheapest real improvement.
- [ ] **Per-class F1 for the DINOv2 DA runs** — settles the CDAN explanation in 10.4.
- [ ] **CORAL lambda check on DINOv2** — `lambda_weight: 40` was tuned on different features.
- [ ] Seeds 1 and 14 for the DINOv2 DA runs.
- [ ] DINOv2 at 224 on Colab — the biggest untried lever, needs a GPU.
- [ ] Re-run `domain_probe.py` on the final checkpoints; 10.10 is from the archived 512 runs.
- [ ] Optional ablations: `fusion=concat`, `class_weights=sqrt`, `vit_pool=cls`.
