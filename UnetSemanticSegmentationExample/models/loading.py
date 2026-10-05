"""Load a trained model back from a run folder."""

import json
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv

from ..utils.paths import PROJECT_NAME, REPO_DIR, resolve_config_path, resolve_repo_path
from .build import build_net
from .lit_unet import LitUNet

CHECKPOINT_SUFFIX = ".ckpt"


def _resolve_run(run: str | Path) -> Path:
    """Find ``run`` as given, in the repo root, or in ``<OUTPUT_DIR>/UnetSemanticSegmentationExample``."""
    load_dotenv(REPO_DIR / ".env")
    path = Path(run).expanduser()
    outputs = resolve_repo_path(os.getenv("OUTPUT_DIR", "runs")) / PROJECT_NAME
    for candidate in (path, REPO_DIR / path, outputs / path):
        if candidate.exists():
            path = candidate.resolve()
            break
    else:
        if outputs.is_dir():
            found = sorted(d.name for d in outputs.iterdir() if d.is_dir())
            hint = f"Runs saved for this project: {found or 'none yet'}"
        else:
            hint = (
                f"{outputs} does not exist: no run of {PROJECT_NAME} was saved there "
                "yet. Train first; in Colab, mount Google Drive before training and "
                "before loading"
            )
        raise FileNotFoundError(
            f"{str(run)!r} not found here, in {REPO_DIR}, or in {outputs}. {hint}"
        )
    return path


def find_checkpoint(run: str | Path, which: str = "best") -> Path:
    """Locate a checkpoint saved by :func:`~UnetSemanticSegmentationExample.main.main`.

    ``run`` can be a checkpoint file, a run folder
    (``<OUTPUT_DIR>/UnetSemanticSegmentationExample/config01/<timestamp>``), or
    a config's folder (``.../config01``), which picks its newest run that has a
    checkpoint. Relative paths are tried from the current folder, the repo
    root, and ``<OUTPUT_DIR>/UnetSemanticSegmentationExample``, so
    ``"config01"`` and ``"config01/2026-10-04_12-29-09"`` both work.

    Args:
        run: Checkpoint file, run folder, or config folder.
        which: ``"best"`` (best validation Dice) or ``"last"`` (last epoch).

    Returns:
        The checkpoint path.

    Raises:
        ValueError: If ``which`` is not ``"best"`` or ``"last"``.
        FileNotFoundError: If no matching checkpoint exists.
    """
    if which not in ("best", "last"):
        raise ValueError(f"which must be 'best' or 'last', got {which!r}")
    path = _resolve_run(run)
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


def find_resume_checkpoint(run: str | Path) -> Path:
    """Locate the checkpoint to continue training from: the run's last epoch.

    ``run`` is looked up as in :func:`find_checkpoint`; for a config folder, its
    newest run is used. A run stopped before it saved a ``last`` checkpoint
    continues from its best one.

    Args:
        run: Checkpoint file, run folder, or config folder.

    Returns:
        The checkpoint path.
    """
    path = _resolve_run(run)
    if path.is_file():
        return path
    run_dir = find_checkpoint(path, "best").parent
    last = list(run_dir.glob(f"last*{CHECKPOINT_SUFFIX}"))
    if last:
        return max(last, key=lambda p: p.stat().st_mtime)
    return find_checkpoint(run_dir, "best")


def run_config(checkpoint: Path) -> dict:
    """Read the config of the run that saved ``checkpoint``.

    ``config.json`` is copied into the run folder when the run finishes. For a
    run that was interrupted, the project's config of the same name is used.

    Args:
        checkpoint: A checkpoint inside a run folder.

    Returns:
        The config as a dict.
    """
    saved = checkpoint.parent / "config.json"
    if saved.is_file():
        return json.loads(saved.read_text())
    config_path = resolve_config_path(checkpoint.parent.parent.name)
    warnings.warn(
        f"{saved} not found (the run did not finish); using {config_path}", stacklevel=3
    )
    return json.loads(config_path.read_text())


def load_model(run: str | Path, which: str = "best", device: str = "cpu") -> LitUNet:
    """Rebuild a trained model from its run, ready for predictions.

    The network is rebuilt from the run's config (its ``"model"`` section),
    and all its weights come from the checkpoint: the ImageNet weights of a
    pretrained encoder are not downloaded again. The model is returned in
    eval mode.

    Example:

    .. code-block:: python

        from UnetSemanticSegmentationExample import load_model

        model = load_model("config02")      # newest run of config02, best checkpoint
        logits = model(images)              # images: (N, 3, 256, 256), ImageNet-normalized
        masks = torch.sigmoid(logits) >= 0.5

    Args:
        run: Checkpoint file, run folder, or config folder; see :func:`find_checkpoint`.
        which: ``"best"`` or ``"last"`` checkpoint.
        device: Where to put the weights, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        The trained ``LitUNet``. It also works with ``trainer.test(model, datamodule=data)``.
    """
    checkpoint = find_checkpoint(run, which)
    net = build_net(run_config(checkpoint)["model"], pretrained=False)
    model = LitUNet.load_from_checkpoint(checkpoint, net=net, map_location=device)
    model.eval()
    print(f"Loaded {checkpoint}")
    return model
