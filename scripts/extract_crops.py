#!/usr/bin/env python3
"""
Cut one square crop per building out of the pre and post images, and save everything as a few
numpy files that the training code can memory-map.

Why this step exists: during training we would otherwise open two 1024x1024 GeoTIFFs for every
single building. That is far slower than the GPU and it would dominate the Colab time. Here we
pay that cost once.

What it writes into --out:
    <split>_pre.npy     uint8, shape (n, size, size, 3)
    <split>_post.npy    uint8, same shape, same row order
    manifest.csv        one row per building: label, split, row index, and the usual metadata
    extract_info.json   every parameter used, so a run can be reproduced

Splits are made by location:
    source  -> source_train / source_val   (labels are used in training)
    target  -> target_train / target_test  (target_train is the UNLABELED set for CORAL/DANN)
    test    -> test                        (held out, never seen in training)
Images are never shared between two splits: the split is made on whole images, inside each event.

Examples (run from the project root)
    python scripts/extract_crops.py
    python scripts/extract_crops.py --size 128 --out crops/wind_128
    python scripts/extract_crops.py --limit-per-event 50 --out crops/_smoke   # quick test

Needs: numpy, pandas, rasterio, pillow.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image

SPLITS = ["source_train", "source_val", "target_train", "target_test", "test"]
UNCLASSIFIED_LABEL = -1


# ----------------------------------------------------------------------------- arguments
def parse_args():
    p = argparse.ArgumentParser(description="Cut pre/post crops for every building.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--subset-dir", default="xbd_subset/wind", help="folder made by create_subset.py")
    p.add_argument("--out", default=None, help="output folder (default: crops/<subset name>_<size>)")

    p.add_argument("--size", type=int, default=64, help="output crop size in pixels (square)")
    p.add_argument("--box-scale", type=float, default=1.5, help="crop box = scale x longest side of the bbox")
    p.add_argument("--min-box", type=float, default=16, help="smallest crop box in pixels, before resizing")
    p.add_argument("--max-box", type=float, default=256, help="largest crop box in pixels, before resizing")
    p.add_argument("--min-side", type=float, default=5, help="skip buildings whose longest bbox side is below this")
    p.add_argument("--keep-unclassified", action="store_true", help="keep un-classified buildings (label_id = -1)")

    p.add_argument("--source", default="us-central", help="location used as the source domain")
    p.add_argument("--target", default="us-south", help="location used as the target domain")
    p.add_argument("--test", default="caribbean", help="location held out for the final test")
    p.add_argument("--val-frac", type=float, default=0.2, help="share of source images kept for validation")
    p.add_argument("--target-test-frac", type=float, default=0.2, help="share of target images kept for evaluation")
    p.add_argument("--group-by", default="image_id", choices=["image_id", "scene_id"],
                   help="unit that is split: whole images, or whole satellite scenes (stricter)")
    p.add_argument("--split-method", default="stratified", choices=["stratified", "random"],
                   help="stratified keeps whole images together AND balances the damage classes "
                        "between the two sides (needs scikit-learn); random just picks images")
    p.add_argument("--seed", type=int, default=42, help="seed of the split")

    p.add_argument("--limit-per-event", type=int, default=0, help="use only the first N images per event (for a quick test)")
    p.add_argument("--overwrite", action="store_true", help="overwrite the output folder if it already exists")
    p.add_argument("--dry-run", action="store_true",
                   help="work out the split and print how balanced it is, then stop. Cuts no crops, "
                        "writes nothing. Use it to compare --seed values before the real run.")
    return p.parse_args()


# ----------------------------------------------------------------------------- helpers
def read_rgb(path):
    """Read a GeoTIFF as an (H, W, 3) uint8 array."""
    with rasterio.open(path) as src:
        bands = list(range(1, min(src.count, 3) + 1))
        arr = src.read(bands)
    if arr.dtype != np.uint8:                       # not expected for xBD, but be safe
        top = float(arr.max()) or 1.0
        arr = (arr.astype(np.float32) / top * 255.0).astype(np.uint8)
    arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr


def cut(img, cx, cy, box, size):
    """Square crop of side `box` centred on (cx, cy), zero-padded at the image edge, resized to `size`.

    Returns (crop, black_fraction). The black fraction is measured BEFORE resizing, because
    resizing blurs the exact zeros of the no-data borders.
    """
    box = int(round(box))
    x0 = int(round(cx - box / 2.0))
    y0 = int(round(cy - box / 2.0))
    x1, y1 = x0 + box, y0 + box

    h, w = img.shape[:2]
    out = np.zeros((box, box, 3), np.uint8)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]

    black = float((out.sum(axis=2) == 0).mean())
    if box != size:
        out = np.asarray(Image.fromarray(out).resize((size, size), Image.BILINEAR))
    return out, black


def _hold_out_mask(g, groups, frac, seed, method):
    """Which rows of one event go to the held-out side. Whole groups move together.

    stratified: sklearn's StratifiedGroupKFold keeps images intact while choosing WHICH images
                to hold out so that the class mix of both sides stays close. This matters because
                one image can be almost entirely destroyed (a tornado track) while the next is
                untouched, so a random choice of images easily skews the rare classes.
    random:     plain random choice of images. Kept for comparison.
    """
    n_groups = groups.nunique()
    if method == "stratified":
        n_splits = max(2, min(n_groups, int(round(1 / frac))))
        try:
            from sklearn.model_selection import StratifiedGroupKFold
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            _, hold_idx = next(splitter.split(np.zeros(len(g)), g["label_id"], groups))
            mask = np.zeros(len(g), bool)
            mask[hold_idx] = True
            return mask
        except ImportError:
            print("  WARNING: scikit-learn not installed, falling back to a random split")
        except ValueError as e:                       # too few images or a class too rare
            print(f"  WARNING: stratified split not possible here ({e}), using a random split")

    rng = np.random.default_rng(seed)
    names = np.sort(groups.unique())
    rng.shuffle(names)
    n_hold = min(len(names) - 1, max(1, int(round(len(names) * frac))))
    return groups.isin(set(names[:n_hold])).to_numpy()


def split_groups(df, group_col, frac, seed, name_a, name_b, method="stratified"):
    """Give every row name_a or name_b. Whole groups move together, and the split is done
    inside each event, so both sides keep the same event mix."""
    out = pd.Series(name_a, index=df.index, dtype=object)
    for i, (event, g) in enumerate(df.groupby("event", sort=True)):
        groups = g[group_col].astype(str)
        if groups.nunique() < 2:
            print(f"  WARNING: {event} has only {groups.nunique()} {group_col}, nothing held out for {name_b}")
            continue
        mask = _hold_out_mask(g, groups, frac, seed + i, method)
        out.loc[g.index[mask]] = name_b
    return out


def tvd(a, b):
    """Total variation distance between two class distributions. 0 = identical, 1 = no overlap."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


def balance_report(df):
    """How close each training split is to the split it is judged on.

    source_val is where every run picks its best epoch, so it should look like source_train.
    Above about 0.05 the validation set is measuring a different class mix from the one the
    model trains on; try another --seed.
    """
    counts = {s: np.bincount(g["label_id"], minlength=4)
              for s, g in df.groupby("split") if s in SPLITS}
    pairs = [("source_train", "source_val"), ("target_train", "target_test")]
    print("\nHow balanced the split is (total variation distance, lower is better)")
    worst = 0.0
    for a, b in pairs:
        if a in counts and b in counts:
            d = tvd(counts[a], counts[b])
            flag = "" if d <= 0.05 else ("  <- high, try another --seed" if d > 0.08 else "  <- a bit high")
            print(f"  {a:14} vs {b:14} {d:.3f}{flag}")
            worst = max(worst, d)
    return worst


def class_table(df, label_names):
    """Count per split and class, plus the share of each class in percent."""
    t = df.pivot_table(index="split", columns="label_id", values="building_uid",
                       aggfunc="count", fill_value=0)
    t = t.reindex(index=[s for s in SPLITS if s in t.index])
    t.columns = [label_names.get(c, str(c)) for c in t.columns]
    t["total"] = t.sum(axis=1)
    pct = (100 * t[t.columns[:-1]].div(t["total"], axis=0)).round(1)
    pct.columns = [f"{c} %" for c in pct.columns]
    return pd.concat([t, pct], axis=1)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    subset = Path(args.subset_dir)
    out_dir = Path(args.out) if args.out else Path("crops") / f"{subset.name}_{args.size}"

    if not (subset / "buildings.csv").is_file():
        sys.exit(f"Not found: {subset / 'buildings.csv'}. Run scripts/create_subset.py first.")
    if out_dir.exists() and any(out_dir.iterdir()) and not (args.overwrite or args.dry_run):
        sys.exit(f"{out_dir} already exists and is not empty. Use --overwrite, or pick another --out.")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    b = pd.read_csv(subset / "buildings.csv")
    print(f"buildings.csv : {len(b):,} rows, {b['image_id'].nunique():,} images, {b['event'].nunique()} events")

    # ---------------- filter
    n0 = len(b)
    if not args.keep_unclassified:
        b = b[b["label_id"] != UNCLASSIFIED_LABEL]
        print(f"  dropped un-classified      : {n0 - len(b):,}")

    n1 = len(b)
    b = b.copy()
    b["side"] = np.maximum(b["x_max"] - b["x_min"], b["y_max"] - b["y_min"])
    b = b[b["side"] >= args.min_side]
    print(f"  dropped side < {args.min_side:g} px        : {n1 - len(b):,}")
    print(f"  kept                       : {len(b):,}")
    if b.empty:
        sys.exit("Nothing left after filtering.")

    if args.limit_per_event:
        keep = (b.drop_duplicates("image_id").groupby("event").head(args.limit_per_event)["image_id"])
        b = b[b["image_id"].isin(set(keep))]
        print(f"  --limit-per-event {args.limit_per_event}: {len(b):,} buildings, {b['image_id'].nunique()} images")

    # ---------------- split
    roles = {args.source: "source", args.target: "target", args.test: "test"}
    unknown = sorted(set(b["location"]) - set(roles))
    if unknown:
        print(f"  WARNING: these locations have no role and are skipped: {unknown}")
    b = b[b["location"].isin(roles)].copy()
    b["role"] = b["location"].map(roles)

    b["split"] = None
    src = b[b["role"] == "source"]
    tgt = b[b["role"] == "target"]
    print(f"\n{args.split_method.capitalize()} split by whole {args.group_by}, "
          f"inside each event (seed {args.seed})")
    if len(src):
        b.loc[src.index, "split"] = split_groups(src, args.group_by, args.val_frac, args.seed,
                                                 "source_train", "source_val", args.split_method)
    if len(tgt):
        b.loc[tgt.index, "split"] = split_groups(tgt, args.group_by, args.target_test_frac, args.seed + 100,
                                                 "target_train", "target_test", args.split_method)
    b.loc[b["role"] == "test", "split"] = "test"
    b = b[b["split"].notna()].copy()

    # row index of each building inside its own split file
    b = b.sort_values(["split", "image_id", "building_uid"]).reset_index(drop=True)
    b["row"] = b.groupby("split").cumcount()

    label_names = {0: "no-damage", 1: "minor-damage", 2: "major-damage", 3: "destroyed",
                   UNCLASSIFIED_LABEL: "un-classified"}
    print("\nBuildings per split")
    print(class_table(b, label_names).to_string())
    print("\nImages per split")
    print(b.groupby("split")["image_id"].nunique().reindex([s for s in SPLITS]).dropna().astype(int).to_string())
    balance_report(b)

    if args.dry_run:
        print(f"\n--dry-run: nothing written. Re-run with other --seed values to compare, "
              f"then drop --dry-run for the real extraction.")
        return

    # ---------------- allocate the output files
    sizes = b.groupby("split").size().to_dict()
    arrays = {}
    for split, n in sizes.items():
        arrays[split] = {
            w: np.lib.format.open_memmap(out_dir / f"{split}_{w}.npy", mode="w+",
                                         dtype=np.uint8, shape=(n, args.size, args.size, 3))
            for w in ("pre", "post")
        }
    total_mb = sum(n for n in sizes.values()) * 2 * args.size * args.size * 3 / 1e6
    print(f"\nWriting {sum(sizes.values()):,} crops x 2 images = {total_mb:,.0f} MB into {out_dir}")
    if total_mb > 5000:
        print("  NOTE: that is large. On Colab, copy it to the local disk before training,"
              " and load it with np.load(..., mmap_mode='r').")

    # ---------------- cut the crops, one image at a time
    b["box"] = np.clip(b["side"] * args.box_scale, args.min_box, args.max_box)
    # filled by position; much faster than writing into the DataFrame row by row
    black_pre = np.full(len(b), np.nan)
    black_post = np.full(len(b), np.nan)
    ok_flag = np.zeros(len(b), dtype=bool)

    failures, done, n_images = [], 0, b["image_id"].nunique()
    for i, (image_id, g) in enumerate(b.groupby("image_id", sort=False), start=1):
        first = g.iloc[0]
        try:
            pre = read_rgb(subset / first["pre_path"])
            post = read_rgb(subset / first["post_path"])
        except Exception as e:                                   # keep going, report at the end
            failures.append(f"{image_id}: {e}")
            continue

        split = first["split"]
        for r in g.itertuples():
            cx = (r.x_min + r.x_max) / 2.0
            cy = (r.y_min + r.y_max) / 2.0
            crop_pre, bp = cut(pre, cx, cy, r.box, args.size)
            crop_post, bq = cut(post, cx, cy, r.box, args.size)
            arrays[split]["pre"][r.row] = crop_pre
            arrays[split]["post"][r.row] = crop_post
            black_pre[r.Index], black_post[r.Index], ok_flag[r.Index] = bp, bq, True
            done += 1

        if i % 50 == 0 or i == n_images:
            rate = done / max(time.time() - t_start, 1e-9)
            print(f"  {i:>5}/{n_images} images   {done:>7,} crops   {rate:,.0f} crops/s", flush=True)

    for split in arrays:
        for w in arrays[split]:
            arrays[split][w].flush()

    b["black_pre"], b["black_post"], b["ok"] = black_pre, black_post, ok_flag

    # ---------------- manifest and info
    cols = ["building_uid", "image_id", "event", "location", "role", "split", "row", "ok",
            "damage", "label_id", "scene_id", "orig_split",
            "x_min", "y_min", "x_max", "y_max", "area_px", "side", "box",
            "black_pre", "black_post", "pre_path", "post_path"]
    manifest = b[[c for c in cols if c in b.columns]]
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    info = {
        "subset_dir": str(subset), "out_dir": str(out_dir),
        "size": args.size, "box_scale": args.box_scale, "min_box": args.min_box, "max_box": args.max_box,
        "min_side": args.min_side, "keep_unclassified": args.keep_unclassified,
        "source": args.source, "target": args.target, "test": args.test,
        "val_frac": args.val_frac, "target_test_frac": args.target_test_frac,
        "group_by": args.group_by, "split_method": args.split_method, "seed": args.seed,
        "label_map": {v: k for k, v in label_names.items() if v != "un-classified"},
        "counts": {s: int(n) for s, n in sizes.items()},
        "n_ok": int(manifest["ok"].sum()), "n_failed_images": len(failures),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "extract_info.json").write_text(json.dumps(info, indent=2))

    # ---------------- summary
    ok = manifest[manifest["ok"]]
    print(f"\nDone in {time.time() - t_start:.0f} s")
    print(f"  crops written : {len(ok):,} of {len(manifest):,}")
    if failures:
        print(f"  images that could not be read: {len(failures)}")
        for f in failures[:10]:
            print("   ", f)
        print("  (rows of those images have ok = False in manifest.csv)")
    if len(ok):
        print("\nBlack (no-data) pixels per crop, measured before resizing")
        print(ok.groupby("split")[["black_pre", "black_post"]]
                .describe(percentiles=[.5, .9, .99]).round(3).to_string())
        heavy = ok[(ok[["black_pre", "black_post"]].max(axis=1) > 0.5)]
        print(f"\n  crops more than 50 % black: {len(heavy):,} ({100 * len(heavy) / len(ok):.2f} %)"
              "   -> filter them in the Dataset, no need to extract again")
    print(f"\nFiles in {out_dir}: " + ", ".join(sorted(p.name for p in out_dir.iterdir())))


if __name__ == "__main__":
    main()
