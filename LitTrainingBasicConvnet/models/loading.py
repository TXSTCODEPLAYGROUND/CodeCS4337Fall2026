"""Load a trained model back from a run folder."""

import json
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

from ..utils.paths import PROJECT_NAME, REPO_DIR, resolve_config_path, resolve_repo_path
from .components import ConvNet
from .lit_convnet import LitConvNet

CHECKPOINT_SUFFIX = ".ckpt"


def find_checkpoint(run: str | Path, which: str = "best") -> Path:
    """Locate a checkpoint saved by :func:`~LitTrainingBasicConvnet.main.main`.

    ``run`` can be a checkpoint file, a run folder
    (``<OUTPUT_DIR>/LitTrainingBasicConvnet/config01/<timestamp>``), or a
    config's folder (``.../config01``), which picks its newest run that has a
    checkpoint. Relative paths are tried from the current folder, the repo
    root, and ``<OUTPUT_DIR>/LitTrainingBasicConvnet``, so ``"config01"``
    and ``"config01/2026-10-02_12-29-09"`` both work.

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


def _run_config(checkpoint: Path) -> dict:
    """Read the config of the run that saved ``checkpoint``.

    ``config.json`` is copied into the run folder when the run finishes. For a
    run that was interrupted, the project's config of the same name is used.
    """
    saved = checkpoint.parent / "config.json"
    if saved.is_file():
        return json.loads(saved.read_text())
    config_path = resolve_config_path(checkpoint.parent.parent.name)
    warnings.warn(
        f"{saved} not found (the run did not finish); using {config_path}", stacklevel=3
    )
    return json.loads(config_path.read_text())


def load_model(run: str | Path, which: str = "best", device: str = "cpu") -> LitConvNet:
    """Rebuild a trained model from its run, ready for predictions.

    The network is rebuilt from the run's config, so you don't need to remember
    its ``dropout``. The model is returned in eval mode.

    Example:

    .. code-block:: python

        from LitTrainingBasicConvnet import load_model

        model = load_model("config01")   # newest run of config01, best checkpoint
        logits = model(images)           # images: (N, 1, 28, 28), normalized

    Args:
        run: Checkpoint file, run folder, or config folder; see :func:`find_checkpoint`.
        which: ``"best"`` or ``"last"`` checkpoint.
        device: Where to put the weights, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        The trained ``LitConvNet``. It also works with ``trainer.test(model, datamodule=data)``.
    """
    checkpoint = find_checkpoint(run, which)
    model_cfg = _run_config(checkpoint)["model"]
    net = ConvNet(dropout=model_cfg["dropout"])
    model = LitConvNet.load_from_checkpoint(checkpoint, net=net, map_location=device)
    model.eval()
    print(f"Loaded {checkpoint}")
    return model
