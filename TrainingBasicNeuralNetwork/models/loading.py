"""Load a trained model back from a run folder."""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from ..utils.paths import PROJECT_NAME, REPO_DIR, resolve_repo_path
from .mlp import MLP

CHECKPOINT_SUFFIX = ".pt"


def find_checkpoint(run: str | Path, which: str = "best") -> Path:
    """Locate a checkpoint saved by :func:`~TrainingBasicNeuralNetwork.main.main`.

    ``run`` can be a checkpoint file, a run folder
    (``<OUTPUT_DIR>/TrainingBasicNeuralNetwork/config01/<timestamp>``), or a
    config's folder (``.../config01``), which picks its newest run that has a
    checkpoint. Relative paths are tried from the current folder, the repo
    root, and ``<OUTPUT_DIR>/TrainingBasicNeuralNetwork``, so ``"config01"``
    and ``"config01/2026-10-02_11-20-01"`` both work.

    Args:
        run: Checkpoint file, run folder, or config folder.
        which: ``"best"`` (best validation accuracy) or ``"last"`` (last epoch).

    Returns:
        The checkpoint path.

    Raises:
        ValueError: If ``which`` is not ``"best"`` or ``"last"``.
        FileNotFoundError: If no matching checkpoint exists.
    """
    if which not in ("best", "last"):
        raise ValueError(f"which must be 'best' or 'last', got {which!r}")
    load_dotenv(REPO_DIR / ".env")
    path = Path(run).expanduser()
    outputs = resolve_repo_path(os.getenv("OUTPUT_DIR", "runs")) / PROJECT_NAME
    for candidate in (path, REPO_DIR / path, outputs / path):
        if candidate.exists():
            path = candidate.resolve()
            break
    else:
        raise FileNotFoundError(
            f"{str(run)!r} not found here, in {REPO_DIR}, or in {outputs}"
        )
    if path.is_file():
        return path

    pattern = f"{which}*{CHECKPOINT_SUFFIX}"
    if not any(path.glob(pattern)):
        runs = sorted(d for d in path.iterdir() if d.is_dir() and any(d.glob(pattern)))
        if runs:
            path = runs[-1]
    checkpoints = list(path.glob(pattern))
    if not checkpoints:
        raise FileNotFoundError(f"No {pattern} checkpoint in {path} or its run folders")
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def load_model(run: str | Path, which: str = "best", device: str = "cpu") -> MLP:
    """Rebuild a trained model from its run, ready for predictions.

    Every checkpoint stores the run's config, so the network is rebuilt with
    the right ``hidden_sizes`` and ``dropout`` before its weights are loaded.
    The model is returned in eval mode.

    Example:

    .. code-block:: python

        from TrainingBasicNeuralNetwork import load_model

        model = load_model("config01")   # newest run of config01, best checkpoint
        logits = model(images)           # images: (N, 1, 28, 28), normalized

    Args:
        run: Checkpoint file, run folder, or config folder; see :func:`find_checkpoint`.
        which: ``"best"`` or ``"last"`` checkpoint.
        device: Where to put the weights, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        The trained ``MLP``.
    """
    path = find_checkpoint(run, which)
    checkpoint = torch.load(path, map_location=device)
    model_cfg = checkpoint["config"]["model"]
    model = MLP(
        hidden_sizes=model_cfg["hidden_sizes"],
        num_classes=10,
        dropout=model_cfg["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded {path}")
    return model
