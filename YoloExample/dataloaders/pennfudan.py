"""Download the Penn-Fudan pedestrian dataset and convert it to the YOLO format.

The dataset has 170 street photos with 423 pedestrians. Each image comes with
an instance mask: a PNG of the same size where pixel value 0 is background and
values 1, 2, ... mark the pixels of person 1, person 2, and so on. YOLO needs
boxes instead, one text file per image, so :func:`convert_to_yolo` turns every
person's mask into its bounding box.

YOLO label format, one line per object, all coordinates divided by the image
width or height (so they lie between 0 and 1):

.. code-block:: text

    <class> <x_center> <y_center> <width> <height>
    0 0.411449 0.571828 0.255814 0.465672

The converted dataset looks like this, plus the ``pennfudan.yaml`` file that
tells Ultralytics where everything is:

.. code-block:: text

    pennfudan_yolo/
    ├── pennfudan.yaml
    ├── split.json                 # which image went to which split, and the seed
    ├── images/{train,val,test}/FudanPed00001.png
    └── labels/{train,val,test}/FudanPed00001.txt
"""

import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

PENNFUDAN_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
"""Download link of the dataset (about 51 MB)."""
CLASS_NAMES = ["person"]
"""The dataset's only class; its YOLO class index is 0."""
SPLITS = ("train", "val", "test")
"""The three splits, in this order in the config's ``"split_sizes"``."""


def download_pennfudan(data_dir: str | Path) -> Path:
    """Download and unzip Penn-Fudan into ``data_dir``, unless it is already there.

    Args:
        data_dir: Directory for datasets, e.g. the repo's ``data/``.

    Returns:
        The unzipped folder, ``<data_dir>/PennFudanPed``.
    """
    data_dir = Path(data_dir)
    source = data_dir / "PennFudanPed"
    if (source / "PNGImages").is_dir():
        return source
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "PennFudanPed.zip"
    print(f"Downloading {PENNFUDAN_URL} to {archive}")
    urllib.request.urlretrieve(PENNFUDAN_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    archive.unlink()
    return source


def mask_to_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Turn an instance mask into one pixel box per instance.

    Args:
        mask: ``(H, W)`` array; 0 is background, ``k > 0`` marks instance ``k``.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` per instance, in pixels, with the max
        edges exclusive (``x_max - x_min`` is the box width).
    """
    boxes = []
    for instance in np.unique(mask):
        if instance == 0:
            continue
        ys, xs = np.nonzero(mask == instance)
        boxes.append(
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        )
    return boxes


def box_to_yolo(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert a pixel box ``(x_min, y_min, x_max, y_max)`` to YOLO's normalized format.

    Args:
        box: Corners in pixels.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        ``(x_center, y_center, box_width, box_height)``, each divided by the
        image width or height.
    """
    x_min, y_min, x_max, y_max = box
    return (
        (x_min + x_max) / 2 / width,
        (y_min + y_max) / 2 / height,
        (x_max - x_min) / width,
        (y_max - y_min) / height,
    )


def yolo_to_box(
    yolo_box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    """Convert a normalized YOLO box back to pixel corners; the inverse of :func:`box_to_yolo`.

    Args:
        yolo_box: ``(x_center, y_center, box_width, box_height)``, normalized.
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
        label_file: A ``.txt`` file with one ``class x_center y_center width height`` line per object.

    Returns:
        ``(class_index, (x_center, y_center, width, height))`` per object.
    """
    labels = []
    for line in Path(label_file).read_text().splitlines():
        if line.strip():
            cls, *coords = line.split()
            labels.append((int(cls), tuple(float(c) for c in coords)))
    return labels


def convert_to_yolo(
    source: str | Path,
    target: str | Path,
    split_sizes: tuple[int, int, int] = (120, 25, 25),
    seed: int = 42,
) -> Path:
    """Convert Penn-Fudan's masks to YOLO labels and split the images.

    The images are shuffled with ``seed`` and cut into train, val, and test
    splits of ``split_sizes``. The conversion is skipped when ``target``
    already holds the same split; it is redone when the sizes or the seed change.

    Args:
        source: The unzipped dataset, with ``PNGImages/`` and ``PedMasks/``.
        target: Folder for the YOLO version of the dataset.
        split_sizes: Number of images in train, val, and test.
        seed: Seed of the shuffle that assigns images to splits.

    Returns:
        The dataset YAML file for Ultralytics, ``<target>/pennfudan.yaml``.
    """
    source, target = Path(source), Path(target)
    names = sorted(p.stem for p in (source / "PNGImages").glob("*.png"))
    if sum(split_sizes) > len(names):
        raise ValueError(
            f"split_sizes {split_sizes} need more than {len(names)} images"
        )
    random.Random(seed).shuffle(names)
    bounds = np.cumsum([0, *split_sizes])
    split = {s: sorted(names[a:b]) for s, a, b in zip(SPLITS, bounds, bounds[1:])}
    split_file = target / "split.json"
    wanted = {"seed": seed, "split": split}

    if not (split_file.is_file() and json.loads(split_file.read_text()) == wanted):
        print(f"Converting Penn-Fudan to YOLO format in {target}")
        for sub in ("images", "labels"):
            shutil.rmtree(target / sub, ignore_errors=True)
        for split_name, items in split.items():
            (target / "images" / split_name).mkdir(parents=True)
            (target / "labels" / split_name).mkdir(parents=True)
            for name in items:
                image = source / "PNGImages" / f"{name}.png"
                mask = np.array(Image.open(source / "PedMasks" / f"{name}_mask.png"))
                height, width = mask.shape
                lines = [
                    "0 " + " ".join(f"{v:.6f}" for v in box_to_yolo(b, width, height))
                    for b in mask_to_boxes(mask)
                ]
                shutil.copy(image, target / "images" / split_name / image.name)
                (target / "labels" / split_name / f"{name}.txt").write_text(
                    "\n".join(lines) + "\n"
                )
        split_file.write_text(json.dumps(wanted, indent=2))

    # Rewritten every time: the absolute path differs between machines.
    yaml_file = target / "pennfudan.yaml"
    yaml_file.write_text(
        f"path: {target.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    return yaml_file


def prepare_pennfudan(
    data_dir: str | Path,
    split_sizes: tuple[int, int, int] = (120, 25, 25),
    seed: int = 42,
) -> Path:
    """Download Penn-Fudan if needed and convert it to YOLO format.

    Args:
        data_dir: Directory for datasets; the original goes to
            ``<data_dir>/PennFudanPed`` and the YOLO version to
            ``<data_dir>/pennfudan_yolo``.
        split_sizes: Number of images in train, val, and test.
        seed: Seed of the shuffle that assigns images to splits.

    Returns:
        The dataset YAML file to pass to Ultralytics as ``data=``.
    """
    source = download_pennfudan(data_dir)
    return convert_to_yolo(source, Path(data_dir) / "pennfudan_yolo", split_sizes, seed)
