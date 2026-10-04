"""Find runs and load trained YOLO models back from them."""

import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from ultralytics import YOLO

from ..utils.paths import PROJECT_NAME, REPO_DIR, resolve_repo_path


def weights_path(weights: str) -> str:
    """Where the starting weights of a config are, or are downloaded to.

    A pretrained Ultralytics name such as ``"yolov8n.pt"`` becomes
    ``<DATA_DIR>/weights/yolov8n.pt``; Ultralytics downloads the file there the
    first time. An architecture file such as ``"yolov8n.yaml"`` (training from
    scratch) and an existing path are returned unchanged.

    Args:
        weights: The config's ``"model": {"weights": ...}``.

    Returns:
        The path or name to pass to ``YOLO(...)``.
    """
    if not weights.endswith(".pt") or Path(weights).expanduser().is_file():
        return weights
    load_dotenv(REPO_DIR / ".env")
    folder = resolve_repo_path(os.getenv("DATA_DIR", "data")) / "weights"
    folder.mkdir(parents=True, exist_ok=True)
    return str(folder / weights)


def _resolve(run: str | Path) -> Path:
    """Find ``run`` as given, in the repo root, or in ``<OUTPUT_DIR>/YoloExample``."""
    load_dotenv(REPO_DIR / ".env")
    path = Path(run).expanduser()
    outputs = resolve_repo_path(os.getenv("OUTPUT_DIR", "runs")) / PROJECT_NAME
    for candidate in (path, REPO_DIR / path, outputs / path):
        if candidate.exists():
            return candidate.resolve()
    found = (
        sorted(d.name for d in outputs.iterdir() if d.is_dir())
        if outputs.is_dir()
        else []
    )
    raise FileNotFoundError(
        f"{str(run)!r} not found here, in {REPO_DIR}, or in {outputs}. "
        f"Runs saved for this project: {found or 'none yet'}. In Colab, mount "
        "Google Drive before training and before loading."
    )


def find_run(run: str | Path) -> Path:
    """Locate a run folder saved by :func:`~YoloExample.main.main`.

    Args:
        run: A run folder (``config02/2026-10-04_17-30-00``), or a config
            folder (``config02``), which picks its newest run. Relative paths
            are tried from the current folder, the repo root, and
            ``<OUTPUT_DIR>/YoloExample``.

    Returns:
        The run folder.
    """
    path = _resolve(run)
    if path.is_file():
        return path.parent.parent if path.parent.name == "weights" else path.parent
    if (path / "config.json").is_file() or (path / "weights").is_dir():
        return path
    runs = sorted(d for d in path.iterdir() if d.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run folder in {path}")
    return runs[-1]


def find_weights(run: str | Path, which: str = "best") -> Path | None:
    """Locate the trained weights of a run.

    Args:
        run: A weights file, a run folder, or a config folder (its newest run
            with weights); see :func:`find_run`.
        which: ``"best"`` (highest validation mAP50-95) or ``"last"`` (last epoch).

    Returns:
        ``<run>/weights/<which>.pt``, or ``None`` for an evaluation-only run
        (zero-shot), which has no weights of its own.
    """
    if which not in ("best", "last"):
        raise ValueError(f"which must be 'best' or 'last', got {which!r}")
    path = _resolve(run)
    if path.is_file():
        return path
    if not ((path / "weights").is_dir() or (path / "config.json").is_file()):
        with_weights = sorted(path.glob(f"*/weights/{which}.pt"))
        if with_weights:
            return with_weights[-1]
    weights = find_run(path) / "weights" / f"{which}.pt"
    return weights if weights.is_file() else None


def load_model(run: str | Path, which: str = "best") -> YOLO:
    """Load the YOLO model of a run, ready for ``model.predict(...)`` or ``model.val(...)``.

    Example:

    .. code-block:: python

        from YoloExample import load_model

        model = load_model("config02")          # newest fine-tuned run, best weights
        results = model.predict("street.jpg", conf=0.25)
        results[0].boxes.xyxy                   # (num_boxes, 4) in pixels

    For a zero-shot run (``config01``), which trains nothing, the pretrained
    COCO model of its config is returned.

    Args:
        run: A weights file, a run folder, or a config folder; see :func:`find_weights`.
        which: ``"best"`` or ``"last"`` weights.

    Returns:
        The Ultralytics ``YOLO`` model.
    """
    weights = find_weights(run, which)
    if weights is None:
        config = json.loads((find_run(run) / "config.json").read_text())
        weights = Path(weights_path(config["model"]["weights"]))
        print(f"{run} trained nothing (zero-shot); loading its pretrained model")
    print(f"Loaded {weights}")
    return YOLO(str(weights))


def find_resume_weights(run: str | Path) -> Path:
    """Locate the ``last.pt`` of an interrupted training run, to resume it.

    Args:
        run: A ``last.pt`` file, a run folder, or a config folder (its newest
            run with a ``last.pt``).

    Returns:
        The ``last.pt`` path.

    Raises:
        FileNotFoundError: If the run has no ``last.pt`` (e.g. zero-shot).
        ValueError: If the run already finished, so there is nothing to resume.
    """
    weights = find_weights(run, "last")
    if weights is None:
        raise FileNotFoundError(
            f"No weights/last.pt in {find_run(run)}: nothing to resume"
        )
    # Ultralytics sets "epoch" to -1 when it finalizes the weights of a finished run.
    if torch.load(weights, map_location="cpu", weights_only=False).get("epoch") == -1:
        raise ValueError(
            f"{weights.parent.parent} already finished training: nothing to resume. "
            "To train longer, start a new run (e.g. with more epochs in a copy of the config)."
        )
    return weights
