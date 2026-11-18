#!/usr/bin/env python3
"""
Make a subset of the xBD dataset for one or more disaster types.

What it does
1. Reads the label JSON files in all splits (tier1, tier3, test, hold) and finds the disaster type of each event.
2. Copies the pre/post images and labels of the chosen events to one folder, merging the splits:
       <out>/<event>/images/*.tif
       <out>/<event>/labels/*.json
   (the original split is kept in the CSV column "orig_split")
3. Writes three CSV files (manifests) that the ML code can read:
       images.csv     one row per pre/post image pair
       buildings.csv  one row per building (label, bounding box, location, paths)  <- main file for the dataloader
       events.csv     one row per event (type, location, counts, mean longitude/latitude)

Examples (run from the project root)
    python scripts/create_subset.py --disaster-type wind --dry-run     # only show what would be selected
    python scripts/create_subset.py --disaster-type wind
    python scripts/create_subset.py --disaster-type wind --events joplin-tornado moore-tornado tuscaloosa-tornado --out xbd_subset/tornado
    python scripts/create_subset.py --disaster-type fire --tar         # also make a .tar.gz to upload to Drive

Only the Python standard library is used.
"""
import argparse
import csv
import json
import re
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ALL_SPLITS = ["tier1", "tier3", "test", "hold"]

# Integer label for each damage class. "un-classified" (and any other name) gets -1: filter it out in the dataloader.
LABEL_MAP = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
IGNORE_LABEL = -1

# Words removed from the event name to get the location: "joplin-tornado" -> "joplin"
HAZARD_WORDS = {"tornado", "wildfire", "bushfire", "fire", "flooding", "flood", "volcano",
                "earthquake", "tsunami", "hurricane"}

NAME_RE = re.compile(r"^(?P<id>.+)_(?P<when>pre|post)_disaster$")
NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
PAIR_RE = re.compile(rf"({NUM})\s+({NUM})")
FIRST_PAIR_RE = re.compile(rf"\(+\s*({NUM})\s+({NUM})")

IMAGE_FIELDS = ["image_id", "event", "location", "disaster_type", "orig_split",
                "pre_path", "post_path", "pre_label_path", "post_label_path",
                "width", "height", "sensor", "gsd",
                "capture_date_pre", "capture_date_post", "off_nadir_pre", "off_nadir_post",
                "sun_elevation_pre", "sun_elevation_post", "catalog_id_pre", "catalog_id_post",
                "n_buildings", "lon", "lat"]
BUILDING_FIELDS = ["building_uid", "image_id", "event", "location", "disaster_type", "orig_split",
                   "scene_id", "damage", "label_id", "x_min", "y_min", "x_max", "y_max", "area_px",
                   "pre_path", "post_path"]


# ----------------------------------------------------------------------------- helpers
def read_json(path):
    with open(path) as f:
        return json.load(f)


def event_of(image_id):
    """'joplin-tornado_00000060' -> 'joplin-tornado'"""
    return image_id.rsplit("_", 1)[0]


def parse_location(event, overrides):
    """Location from the event name, or from the override file. Returns (location, source)."""
    if event in overrides:
        return overrides[event], "override"
    tokens = [t for t in event.split("-") if t not in HAZARD_WORDS]
    return ("-".join(tokens) if tokens else event), "parsed_from_name"


def wkt_points(wkt):
    return [(float(a), float(b)) for a, b in PAIR_RE.findall(wkt)]


def outer_ring_area(wkt):
    """Area of the outer ring of a POLYGON (shoelace formula). Empty string for other shapes."""
    if not wkt.startswith("POLYGON"):
        return ""
    inner = wkt[wkt.index("((") + 2: wkt.rindex("))")]
    pts = wkt_points(inner.split("),")[0])
    if len(pts) < 3:
        return ""
    s = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        s += x1 * y2 - x2 * y1
    return round(abs(s) / 2, 2)


def mean_lonlat(lab):
    """Mean longitude/latitude of the buildings of one image (from the first vertex of each polygon)."""
    lons, lats = [], []
    for b in lab.get("features", {}).get("lng_lat", []):
        m = FIRST_PAIR_RE.search(b.get("wkt", ""))
        if m:
            lons.append(float(m.group(1)))
            lats.append(float(m.group(2)))
    if not lons:
        return "", ""
    return round(sum(lons) / len(lons), 6), round(sum(lats) / len(lats), 6)


def get_buildings(lab):
    return [b for b in lab.get("features", {}).get("xy", [])
            if b.get("properties", {}).get("feature_type", "building") == "building"]


def copy_file(src, dst):
    """Copy a file. Returns False if the same file (same size) is already there, so the script can be run again."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ----------------------------------------------------------------------------- scanning
def scan(data_root, splits):
    """event -> split -> [image_id], using the post-disaster label files."""
    found = defaultdict(lambda: defaultdict(list))
    for split in splits:
        lab_dir = data_root / split / "labels"
        if not lab_dir.exists():
            print(f"WARNING: labels folder not found, split skipped: {lab_dir}")
            continue
        for p in sorted(lab_dir.glob("*_post_disaster.json")):
            m = NAME_RE.match(p.stem)
            if m:
                found[event_of(m["id"])][split].append(m["id"])
    return found


def event_type(data_root, event, by_split):
    """Disaster type of an event, from the metadata of its first label file.
    Assumption: all images of one event have the same type (it is a field of the event)."""
    for split, ids in by_split.items():
        meta = read_json(data_root / split / "labels" / f"{ids[0]}_post_disaster.json").get("metadata", {})
        if meta.get("disaster_type"):
            return meta["disaster_type"]
    return "unknown"


# ----------------------------------------------------------------------------- main
def parse_args():
    p = argparse.ArgumentParser(description="Create an xBD subset for one or more disaster types.")
    p.add_argument("--data-root", default="geotiffs", help="folder that contains tier1, tier3, test, hold (default: geotiffs)")
    p.add_argument("--splits", nargs="+", default=ALL_SPLITS, help=f"splits to read (default: {' '.join(ALL_SPLITS)})")
    p.add_argument("--disaster-type", nargs="+", default=["wind"],
                   help="disaster type(s) to extract, as in the metadata (default: wind)")
    p.add_argument("--events", nargs="+", default=None,
                   help="optional: keep only these events (they must also have the chosen disaster type)")
    p.add_argument("--out", default=None, help="output folder (default: xbd_subset/<disaster types>)")
    p.add_argument("--location-map", default=None,
                   help='optional JSON file {"event-name": "location-name"} to override the location parsed from the name')
    p.add_argument("--split-folders", action="store_true",
                   help="keep the original split in the folder layout: <out>/<event>/<split>/images. "
                        "Use it only if the same image id exists in two splits (the script stops and tells you).")
    p.add_argument("--limit-per-event", type=int, default=None, help="optional: copy only N image pairs per event (for quick tests)")
    p.add_argument("--dry-run", action="store_true", help="only list the events and counts, do not copy anything")
    p.add_argument("--tar", action="store_true", help="also make <out>.tar.gz (to upload to Drive)")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        sys.exit(f"Data folder not found: {data_root.resolve()}  (run from the project root or use --data-root)")

    overrides = json.load(open(args.location_map)) if args.location_map else {}
    wanted = {t.lower() for t in args.disaster_type}
    out_dir = Path(args.out) if args.out else Path("xbd_subset") / "_".join(sorted(wanted))

    found = scan(data_root, args.splits)
    if not found:
        sys.exit("No post-disaster label files found. Check --data-root and --splits.")
    types = {ev: event_type(data_root, ev, by_split) for ev, by_split in found.items()}

    # ---- choose events
    selected = [ev for ev in sorted(found) if types[ev].lower() in wanted]
    if args.events:
        unknown = [e for e in args.events if e not in found]
        if unknown:
            sys.exit(f"Unknown event names: {unknown}\nAvailable: {sorted(found)}")
        wrong_type = [e for e in args.events if e not in selected]
        if wrong_type:
            print(f"WARNING: not selected because their type is not {sorted(wanted)}: "
                  f"{[(e, types[e]) for e in wrong_type]}")
        selected = [e for e in selected if e in args.events]
    if not selected:
        sys.exit(f"No event selected for type {sorted(wanted)}. Disaster types found: {sorted(set(types.values()))}")

    # ---- table of all events
    print(f"\n{'event':24} {'type':12} {'location':16} {'sel':3}  image pairs per split")
    for ev in sorted(found):
        loc, _ = parse_location(ev, overrides)
        per_split = ", ".join(f"{s}: {len(ids)}" for s, ids in found[ev].items())
        print(f"{ev:24} {types[ev]:12} {loc:16} {'yes' if ev in selected else '':3}  {per_split}")
    storms = [ev for ev in selected if ev.startswith("hurricane-") and ev not in overrides]
    if storms:
        print(f"\nNOTE: for {storms} the 'location' is the storm name, not a place. "
              f"Use --location-map to give a real location, or use lon/lat in events.csv.")
    if args.dry_run:
        print("\nDry run: nothing was copied.")
        return

    # ---- the splits are merged in one folder per event, so an image id must not exist in two splits
    if not args.split_folders:
        dups = [i for ev in selected
                for i, c in Counter(i for ids in found[ev].values() for i in ids).items() if c > 1]
        if dups:
            sys.exit(f"STOP: {len(dups)} image ids exist in more than one split (example: {dups[0]}). "
                     f"Merging them would overwrite files. Run again with --split-folders.")

    # ---- copy files and build the manifests
    image_rows, building_rows, event_rows = [], [], []
    skipped = Counter()
    n_total = sum(len(ids) for ev in selected for ids in found[ev].values())
    done = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying {len(selected)} events ({n_total} image pairs) to {out_dir} ...")

    for ev in selected:
        location, loc_source = parse_location(ev, overrides)
        ev_images, ev_classes, ev_lons, ev_lats = 0, Counter(), [], []
        count_event = 0
        for split, ids in found[ev].items():
            for image_id in ids:
                done += 1
                if done % 250 == 0:
                    print(f"  {done}/{n_total}")
                if args.limit_per_event and count_event >= args.limit_per_event:
                    break
                src_img = data_root / split / "images"
                src_lab = data_root / split / "labels"
                pre_tif = src_img / f"{image_id}_pre_disaster.tif"
                post_tif = src_img / f"{image_id}_post_disaster.tif"
                pre_json = src_lab / f"{image_id}_pre_disaster.json"
                post_json = src_lab / f"{image_id}_post_disaster.json"
                if not (pre_tif.exists() and post_tif.exists() and post_json.exists()):
                    skipped["missing pre tif, post tif or post label"] += 1
                    continue
                count_event += 1

                post = read_json(post_json)
                pre = read_json(pre_json) if pre_json.exists() else {}
                meta_post, meta_pre = post.get("metadata", {}), pre.get("metadata", {})
                if meta_post.get("disaster_type") and meta_post["disaster_type"].lower() != types[ev].lower():
                    print(f"WARNING: {image_id} has type {meta_post['disaster_type']}, event type is {types[ev]}")

                # copy (one folder per event, splits merged unless --split-folders)
                rel = Path(ev) / split if args.split_folders else Path(ev)
                rel_pre_tif = rel / "images" / pre_tif.name
                rel_post_tif = rel / "images" / post_tif.name
                rel_pre_json = rel / "labels" / pre_json.name
                rel_post_json = rel / "labels" / post_json.name
                copy_file(pre_tif, out_dir / rel_pre_tif)
                copy_file(post_tif, out_dir / rel_post_tif)
                copy_file(post_json, out_dir / rel_post_json)
                has_pre_json = pre_json.exists()
                if has_pre_json:
                    copy_file(pre_json, out_dir / rel_pre_json)

                width, height = meta_post.get("width"), meta_post.get("height")
                buildings = get_buildings(post)
                lon, lat = mean_lonlat(post)
                if lon != "":
                    ev_lons.append(lon)
                    ev_lats.append(lat)

                image_rows.append({
                    "image_id": image_id, "event": ev, "location": location, "disaster_type": types[ev],
                    "orig_split": split,
                    "pre_path": rel_pre_tif.as_posix(), "post_path": rel_post_tif.as_posix(),
                    "pre_label_path": rel_pre_json.as_posix() if has_pre_json else "",
                    "post_label_path": rel_post_json.as_posix(),
                    "width": width, "height": height,
                    "sensor": meta_post.get("sensor"), "gsd": meta_post.get("gsd"),
                    "capture_date_pre": meta_pre.get("capture_date", ""), "capture_date_post": meta_post.get("capture_date", ""),
                    "off_nadir_pre": meta_pre.get("off_nadir_angle", ""), "off_nadir_post": meta_post.get("off_nadir_angle", ""),
                    "sun_elevation_pre": meta_pre.get("sun_elevation", ""), "sun_elevation_post": meta_post.get("sun_elevation", ""),
                    "catalog_id_pre": meta_pre.get("catalog_id", ""), "catalog_id_post": meta_post.get("catalog_id", ""),
                    "n_buildings": len(buildings), "lon": lon, "lat": lat,
                })

                for b in buildings:
                    props = b.get("properties", {})
                    wkt = b.get("wkt", "")
                    pts = wkt_points(wkt)
                    if not pts:
                        skipped["building without coordinates"] += 1
                        continue
                    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
                    x_min, x_max, y_min, y_max = min(xs), max(xs), min(ys), max(ys)
                    if width and height:                    # keep the box inside the image
                        x_min, x_max = max(0.0, x_min), min(float(width), x_max)
                        y_min, y_max = max(0.0, y_min), min(float(height), y_max)
                    damage = props.get("subtype", "no-subtype")
                    ev_classes[damage] += 1
                    building_rows.append({
                        "building_uid": props.get("uid"), "image_id": image_id, "event": ev, "location": location,
                        "disaster_type": types[ev], "orig_split": split,
                        "scene_id": meta_post.get("catalog_id", ""),
                        "damage": damage, "label_id": LABEL_MAP.get(damage, IGNORE_LABEL),
                        "x_min": round(x_min, 2), "y_min": round(y_min, 2),
                        "x_max": round(x_max, 2), "y_max": round(y_max, 2),
                        "area_px": outer_ring_area(wkt),
                        "pre_path": rel_pre_tif.as_posix(), "post_path": rel_post_tif.as_posix(),
                    })
                ev_images += 1

        event_rows.append({
            "event": ev, "disaster_type": types[ev], "location": location, "location_source": loc_source,
            "splits": "+".join(found[ev].keys()), "n_images": ev_images, "n_buildings": sum(ev_classes.values()),
            **{f"n_{c}": ev_classes.get(c, 0) for c in LABEL_MAP}, "n_other": sum(v for k, v in ev_classes.items() if k not in LABEL_MAP),
            "lon_mean": round(sum(ev_lons) / len(ev_lons), 5) if ev_lons else "",
            "lat_mean": round(sum(ev_lats) / len(ev_lats), 5) if ev_lats else "",
        })

    # ---- write manifests
    write_csv(out_dir / "images.csv", image_rows, IMAGE_FIELDS)
    write_csv(out_dir / "buildings.csv", building_rows, BUILDING_FIELDS)
    event_fields = list(event_rows[0].keys())
    write_csv(out_dir / "events.csv", event_rows, event_fields)
    info = {
        "made_on": datetime.now().isoformat(timespec="seconds"),
        "command_line": " ".join(sys.argv),
        "disaster_types": sorted(wanted), "events": selected, "splits_read": args.splits,
        "n_image_pairs": len(image_rows), "n_buildings": len(building_rows),
        "label_map": LABEL_MAP, "ignored_label": IGNORE_LABEL, "skipped": dict(skipped),
    }
    (out_dir / "subset_info.json").write_text(json.dumps(info, indent=2))

    # ---- summary
    print(f"\nDone. {len(image_rows)} image pairs, {len(building_rows)} buildings -> {out_dir}")
    print(f"\n{'event':24} {'location':14} {'images':>7} {'buildings':>9} " + " ".join(f"{c[:8]:>9}" for c in LABEL_MAP) + f" {'other':>7}")
    for r in event_rows:
        print(f"{r['event']:24} {r['location']:14} {r['n_images']:>7} {r['n_buildings']:>9} "
              + " ".join(f"{r['n_' + c]:>9}" for c in LABEL_MAP) + f" {r['n_other']:>7}")
    if skipped:
        print("\nSkipped:", dict(skipped))

    if args.tar:
        tar_path = out_dir.parent / f"{out_dir.name}.tar.gz"
        print(f"\nMaking {tar_path} (this can take a while) ...")
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(out_dir, arcname=out_dir.name)
        print("Archive size: %.1f MB" % (tar_path.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
