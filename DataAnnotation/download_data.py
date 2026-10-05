"""Download three small datasets to practice annotation in Label Studio.

Usage, from the repository root::

    python DataAnnotation/download_data.py

It uses only the Python standard library, so any Python 3.10+ works (no
virtual environment needed). Archives are downloaded once into
``data/annotation/downloads/`` and a fixed random sample of images (same
``--seed``, same images for everyone) is copied to::

    data/annotation/
    ├── classification/   30 Imagenette images: dog, church, or parachute
    │   ├── images/          img_001.jpg ...  (neutral names: the class is not in the name)
    │   └── ground_truth/    labels.csv       (image,label), to check your labels afterwards
    ├── segmentation/     20 photos with balloons, to outline every balloon
    │   ├── images/
    │   └── ground_truth/    via_region_data.json (the original polygons, VGG Image Annotator format)
    └── detection/        30 photos of buffalo, elephants, rhinos, and zebras, to box every animal
        ├── images/
        └── ground_truth/    labels/*.txt (YOLO format) and classes.txt

Label the files in ``images/``; look at ``ground_truth/`` only after you
finish, to measure how close your labels are.
"""

import argparse
import csv
import json
import random
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_CLASSES = {
    "n02102040": "dog",
    "n03028079": "church",
    "n03888257": "parachute",
}
BALLOON_URL = (
    "https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip"
)
WILDLIFE_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip"
WILDLIFE_CLASSES = ("buffalo", "elephant", "rhino", "zebra")


def download(url: str, target: Path) -> Path:
    """Download ``url`` to ``target`` once; later calls reuse the file."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    partial = target.with_suffix(target.suffix + ".part")
    with urlopen(url, timeout=60) as response, partial.open("wb") as f:
        shutil.copyfileobj(response, f, length=1 << 20)
    partial.rename(target)
    return target


def fresh_folder(path: Path) -> Path:
    """Create ``path`` (and its parents), emptying it if it already has files."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def classification(
    out: Path, downloads: Path, per_class: int, rng: random.Random
) -> None:
    """Copy ``per_class`` Imagenette validation images of each class, shuffled."""
    archive = download(IMAGENETTE_URL, downloads / "imagenette2-160.tgz")
    images = fresh_folder(out / "classification" / "images")
    truth = fresh_folder(out / "classification" / "ground_truth")

    with tarfile.open(archive) as tar:
        members = [
            m
            for m in tar.getmembers()
            if m.isfile()
            and m.name.endswith(".JPEG")
            and m.name.split("/")[1] == "val"
            and m.name.split("/")[2] in IMAGENETTE_CLASSES
        ]
        picked = []
        for wnid, label in IMAGENETTE_CLASSES.items():
            of_class = sorted(
                (m for m in members if m.name.split("/")[2] == wnid),
                key=lambda m: m.name,
            )
            picked += [(m, label) for m in rng.sample(of_class, per_class)]
        rng.shuffle(picked)

        rows = []
        for k, (member, label) in enumerate(picked, start=1):
            name = f"img_{k:03d}.jpg"
            (images / name).write_bytes(tar.extractfile(member).read())
            rows.append((name, label))

    with (truth / "labels.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("image", "label"), *rows])
    print(
        f"classification: {len(rows)} images ({', '.join(IMAGENETTE_CLASSES.values())})"
    )


def segmentation(out: Path, downloads: Path, count: int, rng: random.Random) -> None:
    """Copy ``count`` balloon photos (all 13 validation images first) and their polygons."""
    archive = download(BALLOON_URL, downloads / "balloon_dataset.zip")
    images = fresh_folder(out / "segmentation" / "images")
    truth = fresh_folder(out / "segmentation" / "ground_truth")

    with zipfile.ZipFile(archive) as zf:
        names = [
            n for n in zf.namelist() if n.startswith("balloon/") and n.endswith(".jpg")
        ]
        val = sorted(n for n in names if n.startswith("balloon/val/"))
        train = sorted(n for n in names if n.startswith("balloon/train/"))
        picked = val[:count] + rng.sample(train, max(0, count - len(val)))

        regions = {}
        for split in ("train", "val"):
            regions.update(json.loads(zf.read(f"balloon/{split}/via_region_data.json")))
        kept = {}
        for name in picked:
            filename = Path(name).name
            (images / filename).write_bytes(zf.read(name))
            kept.update({k: v for k, v in regions.items() if v["filename"] == filename})

    (truth / "via_region_data.json").write_text(json.dumps(kept, indent=1))
    print(f"segmentation: {len(picked)} images (balloon)")


def detection(out: Path, downloads: Path, count: int, rng: random.Random) -> None:
    """Copy ``count`` African Wildlife test images and their YOLO labels."""
    archive = download(WILDLIFE_URL, downloads / "african-wildlife.zip")
    images = fresh_folder(out / "detection" / "images")
    truth = fresh_folder(out / "detection" / "ground_truth")
    labels = truth / "labels"
    labels.mkdir()

    with zipfile.ZipFile(archive) as zf:
        test = sorted(
            n
            for n in zf.namelist()
            if n.startswith("images/test/") and n.endswith(".jpg")
        )
        # The leading number of a file name ("3 (12).jpg") is its main animal:
        # take the same number of images of each, so no class is left out.
        groups = {}
        for name in test:
            groups.setdefault(Path(name).stem.split()[0], []).append(name)
        picked = []
        for k, group in enumerate(groups.values()):
            share = count // len(groups) + (k < count % len(groups))
            picked += rng.sample(group, share)
        for name in picked:
            stem = Path(name).stem
            # "1 (103)" -> "wildlife_1_103": no spaces or brackets in file names
            clean = "wildlife_" + "_".join(re.findall(r"\d+", stem))
            (images / f"{clean}.jpg").write_bytes(zf.read(name))
            (labels / f"{clean}.txt").write_bytes(zf.read(f"labels/test/{stem}.txt"))

    (truth / "classes.txt").write_text("\n".join(WILDLIFE_CLASSES) + "\n")
    print(f"detection: {count} images ({', '.join(WILDLIFE_CLASSES)})")


def main() -> None:
    """Parse the arguments and prepare the three folders."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "data" / "annotation",
        help="output folder (default: data/annotation in the repository)",
    )
    parser.add_argument(
        "--per-class", type=int, default=10, help="classification images per class"
    )
    parser.add_argument(
        "--segmentation", type=int, default=20, help="segmentation images"
    )
    parser.add_argument("--detection", type=int, default=30, help="detection images")
    parser.add_argument(
        "--seed", type=int, default=42, help="picks the same images for everyone"
    )
    args = parser.parse_args()

    downloads = args.out / "downloads"
    classification(args.out, downloads, args.per_class, random.Random(args.seed))
    segmentation(args.out, downloads, args.segmentation, random.Random(args.seed))
    detection(args.out, downloads, args.detection, random.Random(args.seed))
    print(f"\nDone. Label the files in {args.out}/<task>/images/.")
    print("Look at ground_truth/ only after you finish, to check your labels.")


if __name__ == "__main__":
    main()
