"""
Dataset over the crops made by scripts/extract_crops.py.

One sample is one building: the pre crop, the post crop, and the damage label.
The two crops always get the SAME random augmentation, geometric and photometric, otherwise
they stop being aligned and `f_post - f_pre` stops meaning "what changed".

Two augmentation backends, chosen by data.aug.backend:
    torchvision     geometric only, the original. Keep it to reproduce the earlier runs.
    albumentations  geometric + photometric, via the `images` target, which gives every
                    image in the stack identical parameters.

Quick check (from the project root):
    python -m src.data.crops --crops crops/wind_64
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import ConcatDataset, DataLoader, Dataset

CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
N_CLASSES = len(CLASS_NAMES)
IGNORE_LABEL = -1

# ImageNet statistics, because the backbone is pretrained on ImageNet
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

METHODS = ("baseline", "bn_only", "coral", "dann", "cdan", "oracle")

# methods whose training step needs an unlabelled target batch. bn_only is in here
# because it must forward target images for BatchNorm, even though it adds no loss.
TARGET_METHODS = ("bn_only", "coral", "dann", "cdan")


# ----------------------------------------------------------------------------- augmentation
@dataclass
class AugConfig:
    """Augmentation, applied with the SAME parameters to the pre and post crop.

    Geometric keys work with either backend. The photometric keys need
    backend="albumentations" and are ignored otherwise; each is off at its default, so the
    torchvision defaults reproduce the original runs exactly.
    """
    backend: str = "torchvision"                    # torchvision | albumentations

    # --- geometric
    hflip: bool = True                              # horizontal flip
    vflip: bool = False                             # overhead imagery has no canonical "up"
    rot90: bool = True                              # 0, 90, 180 or 270 degrees
    rotate_deg: float = 10.0                        # extra small rotation, in [-x, +x]
    scale: tuple[float, float] = (0.9, 1.1)         # random zoom

    # --- photometric (albumentations only). 0 / no-op defaults = off.
    brightness_contrast: float = 0.0                # limit for both, e.g. 0.3
    hue_sat_val: tuple[float, float, float] = (0.0, 0.0, 0.0)   # shift limits
    gamma: tuple[float, float] = (100.0, 100.0)     # percent; (100,100) is a no-op
    blur_sigma: tuple[float, float] = (0.0, 0.0)    # GaussianBlur sigma range
    noise_std: tuple[float, float] = (0.0, 0.0)     # GaussNoise, fraction of the 0-1 range
    dropout: float = 0.0                            # p of CoarseDropout
    p_photo: float = 0.5                            # p for each photometric transform

    def __post_init__(self):
        # YAML gives lists; tuples make the no-op checks below behave
        self.scale = tuple(self.scale)
        self.hue_sat_val = tuple(self.hue_sat_val)
        self.gamma = tuple(self.gamma)
        self.blur_sigma = tuple(self.blur_sigma)
        self.noise_std = tuple(self.noise_std)

    @property
    def geometric_identity(self) -> bool:
        return (not self.hflip and not self.vflip and not self.rot90
                and self.rotate_deg == 0 and self.scale == (1.0, 1.0))

    @property
    def photometric_identity(self) -> bool:
        return (self.brightness_contrast == 0 and self.hue_sat_val == (0.0, 0.0, 0.0)
                and self.gamma == (100.0, 100.0) and self.blur_sigma == (0.0, 0.0)
                and self.noise_std == (0.0, 0.0) and self.dropout == 0)

    @property
    def is_identity(self) -> bool:
        return self.geometric_identity and (self.backend != "albumentations"
                                            or self.photometric_identity)


def build_albumentations(aug: AugConfig):
    """One A.Compose. The `images` target gives every image in the stack identical
    parameters, which is what keeps the pre/post pair aligned."""
    import albumentations as A

    # The `*_limit` names below were REMOVED in AlbumentationsX 2.2.0 (renamed to `*_range`,
    # with no deprecation period). AlbumentationsX installs under the same `albumentations`
    # import name, so the only way to tell is the version. pip install albumentations gives
    # 2.0.8, which is what this is written for. Fail here with something readable rather
    # than part way through the first epoch of an overnight run.
    if tuple(int(x) for x in A.__version__.split(".")[:2]) >= (2, 2):
        raise RuntimeError(
            f"albumentations {A.__version__} renamed the *_limit arguments this code uses "
            f"(2.2.0 removed them outright). Either:\n"
            f"    pip install 'albumentations==2.0.8'\n"
            f"or update build_albumentations() to the *_range names."
        )

    t = []
    if aug.hflip:
        t.append(A.HorizontalFlip(p=0.5))
    if aug.vflip:
        t.append(A.VerticalFlip(p=0.5))
    if aug.rot90:
        t.append(A.RandomRotate90(p=1.0))           # k is drawn inside, 0 included
    if aug.rotate_deg > 0 or aug.scale != (1.0, 1.0):
        t.append(A.Affine(rotate=(-aug.rotate_deg, aug.rotate_deg),
                          scale=aug.scale, p=1.0))

    bc, (h, sa, v) = aug.brightness_contrast, aug.hue_sat_val
    if bc > 0:
        t.append(A.RandomBrightnessContrast(brightness_limit=bc, contrast_limit=bc,
                                            p=aug.p_photo))
    if (h, sa, v) != (0.0, 0.0, 0.0):
        t.append(A.HueSaturationValue(hue_shift_limit=h, sat_shift_limit=sa,
                                      val_shift_limit=v, p=aug.p_photo))
    if aug.gamma != (100.0, 100.0):
        t.append(A.RandomGamma(gamma_limit=aug.gamma, p=aug.p_photo))
    if aug.blur_sigma[1] > 0:
        t.append(A.GaussianBlur(blur_limit=0, sigma_limit=aug.blur_sigma, p=aug.p_photo))
    if aug.noise_std[1] > 0:
        t.append(A.GaussNoise(std_range=aug.noise_std, p=aug.p_photo))
    if aug.dropout > 0:
        t.append(A.CoarseDropout(num_holes_range=(1, 3),
                                 hole_height_range=(0.05, 0.2),
                                 hole_width_range=(0.05, 0.2), p=aug.dropout))
    return A.Compose(t)


def augment_pair(pre: torch.Tensor, post: torch.Tensor, aug: AugConfig):
    """Same random transform on both crops. Tensors are (3, H, W) floats in [0, 1].

    Uses python's `random`, which the DataLoader reseeds in every worker. numpy's global
    generator is NOT reseeded per worker, which is a classic source of repeated augmentations.
    """
    if aug.hflip and random.random() < 0.5:
        pre, post = torch.flip(pre, [2]), torch.flip(post, [2])

    if aug.vflip and random.random() < 0.5:
        pre, post = torch.flip(pre, [1]), torch.flip(post, [1])

    if aug.rot90:
        k = random.randint(0, 3)
        if k:
            pre = torch.rot90(pre, k, [1, 2]).contiguous()
            post = torch.rot90(post, k, [1, 2]).contiguous()

    if aug.rotate_deg > 0 or aug.scale != (1.0, 1.0):
        angle = random.uniform(-aug.rotate_deg, aug.rotate_deg)
        scale = random.uniform(*aug.scale)
        kw = dict(angle=angle, translate=[0, 0], scale=scale, shear=[0.0],
                  interpolation=TF.InterpolationMode.BILINEAR)
        pre, post = TF.affine(pre, **kw), TF.affine(post, **kw)

    return pre, post


# ----------------------------------------------------------------------------- dataset
class CropDataset(Dataset):
    """Buildings of one split.

    crops_dir   folder written by scripts/extract_crops.py
    split       source_train | source_val | target_train | target_test | test
    train       True turns augmentation on. Use it only for the training splits.
    use_labels  False makes every label IGNORE_LABEL. This is how the target domain stays
                unlabeled for CORAL and DANN, so the labels cannot leak in by mistake.
    max_black   drop crops whose black (no-data) share is above this, in pre or post.
    """

    def __init__(self, crops_dir, split, train=False, use_labels=True,
                 max_black=0.5, aug: AugConfig | None = None, manifest: pd.DataFrame | None = None):
        self.dir = Path(crops_dir)
        self.split = split
        self.train = train
        self.use_labels = use_labels
        self.aug = aug if aug is not None else AugConfig()

        self.info = json.loads((self.dir / "extract_info.json").read_text())
        self.size = int(self.info["size"])

        man = manifest if manifest is not None else pd.read_csv(self.dir / "manifest.csv")
        rows = man[(man["split"] == split) & man["ok"]]
        if max_black is not None:
            black = rows[["black_pre", "black_post"]].max(axis=1)
            rows = rows[black <= max_black]
        if rows.empty:
            raise ValueError(f"no usable rows for split '{split}' in {self.dir}")

        self.meta = rows.reset_index(drop=True)
        self.rows = self.meta["row"].to_numpy(np.int64)          # index inside the .npy files
        self.labels = self.meta["label_id"].to_numpy(np.int64)
        # building_uid is the xBD polygon uid, a UUID string such as
        # "0007e545-dec1-4999-8d38-068d2ff6888c", so keep it as text
        self.uids = self.meta["building_uid"].astype(str).to_numpy()
        self._pre = self._post = None                            # memmaps, opened per worker
        self._tf = None                                          # albumentations, built per worker

    # the memmaps are opened on first use, not in __init__, so that DataLoader workers
    # (which may be started with "spawn" on macOS) each open their own handle
    def _arrays(self):
        if self._pre is None:
            self._pre = np.load(self.dir / f"{self.split}_pre.npy", mmap_mode="r")
            self._post = np.load(self.dir / f"{self.split}_post.npy", mmap_mode="r")
        return self._pre, self._post

    def __len__(self):
        return len(self.rows)

    def _albu(self):
        """Built lazily, like the memmaps, so each DataLoader worker makes its own."""
        if self._tf is None:
            self._tf = build_albumentations(self.aug)
        return self._tf

    def __getitem__(self, i):
        pre_arr, post_arr = self._arrays()
        r = int(self.rows[i])

        # np.array copies out of the read-only memmap, which also keeps torch from warning
        pre_u8, post_u8 = np.array(pre_arr[r]), np.array(post_arr[r])

        albu = self.train and self.aug.backend == "albumentations" and not self.aug.is_identity
        if albu:
            # the `images` target applies ONE draw of parameters to both crops
            out = self._albu()(images=np.stack([pre_u8, post_u8]))["images"]
            pre_u8, post_u8 = np.ascontiguousarray(out[0]), np.ascontiguousarray(out[1])

        pre = torch.from_numpy(pre_u8).permute(2, 0, 1).float() / 255.0
        post = torch.from_numpy(post_u8).permute(2, 0, 1).float() / 255.0

        if self.train and not albu and not self.aug.is_identity:
            pre, post = augment_pair(pre, post, self.aug)

        pre = TF.normalize(pre, IMAGENET_MEAN, IMAGENET_STD)
        post = TF.normalize(post, IMAGENET_MEAN, IMAGENET_STD)

        label = int(self.labels[i]) if self.use_labels else IGNORE_LABEL
        # uid stays a string; the default collate turns a batch of them into a list of strings
        return {"pre": pre, "post": post, "label": label,
                "uid": str(self.uids[i]), "index": i}

    def class_counts(self):
        return np.bincount(self.labels, minlength=N_CLASSES)

    def __repr__(self):
        counts = ", ".join(f"{n}={c:,}" for n, c in zip(CLASS_NAMES, self.class_counts()))
        return (f"CropDataset({self.split}, n={len(self):,}, train={self.train}, "
                f"labels={self.use_labels}, {counts})")


# ----------------------------------------------------------------------------- helpers
def class_weights(labels, scheme="balanced"):
    """Weights for the cross-entropy loss, from the labels of the TRAINING split.

    balanced : w_c = N / (C * n_c)            ratio about 19:1 on the wind source domain
    sqrt     : w_c proportional to 1/sqrt(f_c)  ratio about 4:1, softer
    none     : all ones
    Both weighted schemes satisfy sum_c n_c * w_c = N, so the loss keeps its usual scale.
    """
    counts = np.bincount(np.asarray(labels), minlength=N_CLASSES).astype(np.float64)
    if (counts == 0).any():
        print(f"WARNING: class(es) {np.where(counts == 0)[0].tolist()} are missing from the training split")
    n = counts.sum()
    safe = np.where(counts > 0, counts, 1.0)

    if scheme == "none":
        w = np.ones(N_CLASSES)
    elif scheme == "balanced":
        w = n / (N_CLASSES * safe)
    elif scheme == "sqrt":
        f = safe / n
        w = 1.0 / (np.sqrt(f) * np.sqrt(f).sum())
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    w = np.where(counts > 0, w, 0.0)
    return torch.tensor(w, dtype=torch.float32)


def make_datasets(crops_dir, method, max_black=0.5, aug: AugConfig | None = None):
    """Datasets for one run. See METHODS.

    Returns a dict:
        train         labelled training set
        train_labels  its labels, for class_weights()
        target        unlabelled target set for CORAL / DANN, else None
        eval          {split name: dataset}, never augmented, always labelled
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")

    crops_dir = Path(crops_dir)
    man = pd.read_csv(crops_dir / "manifest.csv")          # read once, share it
    common = dict(crops_dir=crops_dir, max_black=max_black, manifest=man)

    source_train = CropDataset(split="source_train", train=True, use_labels=True, aug=aug, **common)

    if method == "oracle":
        # the only run allowed to read the target labels
        target_labelled = CropDataset(split="target_train", train=True, use_labels=True, aug=aug, **common)
        train = ConcatDataset([source_train, target_labelled])
        train_labels = np.concatenate([source_train.labels, target_labelled.labels])
        target = None
    else:
        train = source_train
        train_labels = source_train.labels
        target = (CropDataset(split="target_train", train=True, use_labels=False, aug=aug, **common)
                  if method in TARGET_METHODS else None)

    evaluation = {s: CropDataset(split=s, train=False, use_labels=True, **common)
                  for s in ("source_val", "target_test", "test")}

    return {"train": train, "train_labels": train_labels, "target": target, "eval": evaluation}


def make_loaders(datasets, batch_size=64, workers=2, seed=0):
    """DataLoaders for the datasets returned by make_datasets.

    The target loader has the same batch size as the training loader, so every step sees the
    same number of source and target samples. drop_last keeps the batch covariance that CORAL
    needs well defined.
    """
    g = torch.Generator().manual_seed(seed)
    common = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            drop_last=True, generator=g, **common),
        "target": None,
        "eval": {name: DataLoader(ds, batch_size=batch_size * 2, shuffle=False, **common)
                 for name, ds in datasets["eval"].items()},
    }
    if datasets["target"] is not None:
        g2 = torch.Generator().manual_seed(seed + 1)
        loaders["target"] = DataLoader(datasets["target"], batch_size=batch_size, shuffle=True,
                                       drop_last=True, generator=g2, **common)
    return loaders


def cycle(loader):
    """Never-ending iterator. The target set is a different size from the source set, so the
    training loop counts source batches and pulls target batches from here."""
    while True:
        for batch in loader:
            yield batch


# ----------------------------------------------------------------------------- self-test
def _main():
    import argparse

    p = argparse.ArgumentParser(description="Sanity check of the crop datasets.")
    p.add_argument("--crops", default="crops/wind_64")
    p.add_argument("--method", default="dann", choices=METHODS)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    ds = make_datasets(args.crops, args.method)
    print(f"method: {args.method}")
    print("  train :", ds["train"] if not isinstance(ds["train"], ConcatDataset)
          else f"ConcatDataset(n={len(ds['train']):,})")
    print("  target:", ds["target"])
    for name, d in ds["eval"].items():
        print(f"  eval  : {d}")

    print("\nclass weights from the training split")
    for scheme in ("balanced", "sqrt"):
        w = class_weights(ds["train_labels"], scheme)
        print(f"  {scheme:9} " + "  ".join(f"{n}={v:.3f}" for n, v in zip(CLASS_NAMES, w.tolist())))

    loaders = make_loaders(ds, batch_size=args.batch_size, workers=args.workers)
    batch = next(iter(loaders["train"]))
    print(f"\none training batch")
    print(f"  pre   {tuple(batch['pre'].shape)}  {batch['pre'].dtype}  "
          f"range [{batch['pre'].min():.2f}, {batch['pre'].max():.2f}]")
    print(f"  post  {tuple(batch['post'].shape)}")
    print(f"  label {tuple(batch['label'].shape)}  values {sorted(set(batch['label'].tolist()))}")

    if loaders["target"] is not None:
        tb = next(cycle(loaders["target"]))
        print(f"  target batch labels: {sorted(set(tb['label'].tolist()))}  (should be [-1])")

    print("\nOK")


if __name__ == "__main__":
    _main()
