"""Load a trained model back from a run folder."""

from pathlib import Path

from ..utils import PROJECT
from .build import build_net
from .lit_unet import LitUNet


def find_checkpoint(run: str | Path, which: str = "best") -> Path:
    """Locate a checkpoint saved by :func:`~UnetSemanticSegmentationExample.main.main`.

    ``run`` can be a checkpoint file, a run folder
    (``<OUTPUT_DIR>/UnetSemanticSegmentationExample/config01/<timestamp>``), or
    a config's folder (``.../config01``), which picks its newest run that has a
    checkpoint. Relative paths are tried from the current folder, the repo
    root, and ``<OUTPUT_DIR>/UnetSemanticSegmentationExample``, so
    ``"config01"`` and ``"config01/2026-10-04_12-29-09"`` both work. This is
    :meth:`colabutilscs4337.project.Project.find_checkpoint` for this project.

    Args:
        run: Checkpoint file, run folder, or config folder.
        which: ``"best"`` (best validation Dice) or ``"last"`` (last epoch).

    Returns:
        The checkpoint path.
    """
    return PROJECT.find_checkpoint(run, which)


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
    net = build_net(PROJECT.run_config(checkpoint)["model"], pretrained=False)
    model = LitUNet.load_from_checkpoint(checkpoint, net=net, map_location=device)
    model.eval()
    print(f"Loaded {checkpoint}")
    return model
