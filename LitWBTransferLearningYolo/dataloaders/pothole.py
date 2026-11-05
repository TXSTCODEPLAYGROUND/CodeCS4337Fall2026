"""Download the IVCNZ pothole dataset and lay it out for Ultralytics.

The dataset (Chitale et al., IVCNZ 2020, MIT license) has 1,243 road photos
and 4,076 pothole boxes, already in the YOLO label format. The files ship in
one folder (``img-1.jpg`` next to ``img-1.txt``). :func:`prepare_pothole`
downloads the zip, shuffles the images with the config's seed, and writes the
train, val, and test folders Ultralytics expects, plus ``pothole.yaml``.

YOLO label format, one line per pothole, coordinates divided by the image
width or height (so they lie between 0 and 1):

.. code-block:: text

    <class> <x_center> <y_center> <width> <height>
    0 0.502404 0.638889 0.557692 0.380342

The prepared dataset looks like this:

.. code-block:: text

    pothole_yolo/
    ├── pothole.yaml
    ├── split.json                 # which image went to which split, and the seed
    ├── images/{train,val,test}/img-1.jpg
    └── labels/{train,val,test}/img-1.txt
"""

import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

POTHOLE_URL = (
    "https://github.com/jaygala24/pothole-detection/releases/download/"
    "v1.0.0/Pothole.Dataset.IVCNZ.zip"
)
"""Download link of the dataset zip (about 166 MB)."""
CLASS_NAMES = ["pothole"]
"""The dataset's only class; its YOLO class index is 0."""
SPLITS = ("train", "val", "test")
"""The three splits, in the order images are assigned after the shuffle."""
VAL_FRACTION = 0.1
"""Share of images held out for validation (choosing the best epoch)."""
TEST_FRACTION = 0.1
"""Share of images held out for the final test report."""
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
"""Image file types copied into the YOLO dataset."""


def download_pothole(data_dir: str | Path) -> Path:
    """Download and unzip the pothole dataset into ``data_dir``, unless it is already there.

    Args:
        data_dir: Directory for datasets, e.g. the repo's ``data/``.

    Returns:
        The unzipped folder, ``<data_dir>/pothole_ivcnz``.
    """
    data_dir = Path(data_dir)
    raw = data_dir / "pothole_ivcnz"
    if any(raw.rglob("*.jpg")):
        return raw
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "Pothole.Dataset.IVCNZ.zip"
    print(f"Downloading {POTHOLE_URL}")
    urllib.request.urlretrieve(POTHOLE_URL, archive)
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        # A zip entry with ".." would write outside raw.
        for name in zf.namelist():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Unsafe path in the dataset zip: {name}")
        zf.extractall(raw)
    archive.unlink()
    return raw


def yolo_to_box(
    yolo_box: tuple[float, ...], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert a normalized YOLO box to pixel corners.

    Args:
        yolo_box: ``(x_center, y_center, box_width, box_height)``, each divided
            by the image width or height.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` in pixels.
    """
    x_center, y_center, box_width, box_height = yolo_box
    return (
        (x_center - box_width / 2) * width,
        (y_center - box_height / 2) * height,
        (x_center + box_width / 2) * width,
        (y_center + box_height / 2) * height,
    )


def read_yolo_labels(label_file: str | Path) -> list[tuple[int, tuple[float, ...]]]:
    """Read a YOLO label file.

    Args:
        label_file: A ``.txt`` file with one ``class x_center y_center width height``
            line per pothole. A missing file is an image with no potholes.

    Returns:
        ``(class_index, (x_center, y_center, width, height))`` per pothole.
    """
    path = Path(label_file)
    if not path.is_file():
        return []
    labels = []
    for line in path.read_text().splitlines():
        if line.strip():
            cls, *coords = line.split()
            labels.append((int(cls), tuple(float(value) for value in coords)))
    return labels


def image_paths(dataset_dir: str | Path, split: str) -> list[Path]:
    """List the images of one split of a dataset written by :func:`prepare_pothole`.

    Args:
        dataset_dir: The ``pothole_yolo`` folder.
        split: ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        Image paths, sorted by file name.
    """
    folder = Path(dataset_dir) / "images" / split
    return sorted(
        path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )


def _image_label_pairs(source: Path) -> list[tuple[str, Path, Path]]:
    """Pair every image under ``source`` with its ``.txt`` label."""
    images = [
        path
        for path in source.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and not path.name.startswith(".")
    ]
    stems = [path.stem for path in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Two images share a file name; cannot build the YOLO split")
    pairs = []
    for image in images:
        label = image.with_suffix(".txt")
        if not label.is_file():
            raise FileNotFoundError(f"No YOLO label next to {image.name}")
        pairs.append((image.stem, image, label))
    return pairs


def _assign_splits(names: list[str], seed: int) -> dict[str, list[str]]:
    """Shuffle ``names`` with ``seed`` and cut them into train, val, and test."""
    names = list(names)
    random.Random(seed).shuffle(names)
    n_val = round(len(names) * VAL_FRACTION)
    n_test = round(len(names) * TEST_FRACTION)
    n_train = len(names) - n_val - n_test
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"{len(names)} images is too small for a train/val/test split")
    cuts = [0, n_train, n_train + n_val, len(names)]
    return {
        split: sorted(names[start:end])
        for split, start, end in zip(SPLITS, cuts, cuts[1:])
    }


def convert_to_yolo(source: str | Path, target: str | Path, seed: int = 42) -> Path:
    """Copy the flat image/label folder into Ultralytics' train, val, and test folders.

    The images are shuffled with ``seed``. About 80% go to train and 10% each
    to val and test (:data:`VAL_FRACTION`, :data:`TEST_FRACTION`). The copy is
    skipped when ``target`` already holds the same split; it is redone when
    the seed changes.

    Args:
        source: The unzipped dataset, images and ``.txt`` labels side by side.
        target: Folder for the YOLO layout, ``<data_dir>/pothole_yolo``.
        seed: Seed of the shuffle that assigns images to splits.

    Returns:
        The dataset YAML file for Ultralytics, ``<target>/pothole.yaml``.
    """
    source, target = Path(source), Path(target)
    pairs = _image_label_pairs(source)
    by_stem = {stem: (image, label) for stem, image, label in pairs}
    split = _assign_splits(sorted(by_stem), seed)
    split_file = target / "split.json"
    wanted = {"seed": seed, "split": split}

    if not (split_file.is_file() and json.loads(split_file.read_text()) == wanted):
        print(f"Preparing the pothole dataset in {target}")
        for sub in ("images", "labels"):
            shutil.rmtree(target / sub, ignore_errors=True)
        for split_name, items in split.items():
            (target / "images" / split_name).mkdir(parents=True)
            (target / "labels" / split_name).mkdir(parents=True)
            for name in items:
                image, label = by_stem[name]
                shutil.copy(image, target / "images" / split_name / image.name)
                shutil.copy(label, target / "labels" / split_name / f"{name}.txt")
        split_file.write_text(json.dumps(wanted, indent=2))

    counts = ", ".join(
        f"{split_name} {len(image_paths(target, split_name))} images"
        for split_name in SPLITS
    )
    print(f"Pothole split (seed {seed}): {counts}")

    # Rewritten every time: the absolute path differs between machines.
    yaml_file = target / "pothole.yaml"
    yaml_file.write_text(
        f"path: {target.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES))
    )
    return yaml_file


def prepare_pothole(data_dir: str | Path, seed: int = 42) -> Path:
    """Download the pothole dataset if needed and lay it out for Ultralytics.

    Args:
        data_dir: Directory for datasets. The zip is extracted to
            ``<data_dir>/pothole_ivcnz`` and the YOLO layout is written to
            ``<data_dir>/pothole_yolo``.
        seed: Seed of the shuffle that assigns images to splits.

    Returns:
        The dataset YAML file to pass to Ultralytics as ``data=``.
    """
    source = download_pothole(data_dir)
    return convert_to_yolo(source, Path(data_dir) / "pothole_yolo", seed)
