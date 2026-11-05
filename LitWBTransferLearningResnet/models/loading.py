"""Load a trained model back from a run folder."""

from pathlib import Path

import torch

from ..utils import PROJECT
from .components import ResNetClassifier
from .lit_resnet import LitResNet


def find_checkpoint(run: str | Path, which: str = "best") -> Path:
    """Locate a checkpoint saved by :func:`~LitWBTransferLearningResnet.main.main`.

    ``run`` can be a checkpoint file, a run folder
    (``<OUTPUT_DIR>/LitWBTransferLearningResnet/config02/<timestamp>``), or a config's
    folder (``.../config02``), which picks its newest run. ``"config02"`` and
    ``"config02/2026-10-02_12-29-09"`` both work. This is
    :meth:`colabutilscs4337.project.Project.find_checkpoint` for this project.

    Args:
        run: Checkpoint file, run folder, or config folder.
        which: ``"best"`` (best validation accuracy) or ``"last"`` (last epoch).

    Returns:
        The checkpoint path.
    """
    return PROJECT.find_checkpoint(run, which)


def load_model(run: str | Path, which: str = "best", device: str = "cpu") -> LitResNet:
    """Rebuild a trained model from its run, ready for predictions.

    The network is rebuilt from the run's config (its ``freeze_backbone`` and
    ``dropout``), and all its weights, backbone included, come from the
    checkpoint: the ImageNet weights are not downloaded again. The model is
    returned in eval mode.

    Example:

    .. code-block:: python

        from LitWBTransferLearningResnet import load_model

        model = load_model("config02")   # newest run of config02, best checkpoint
        logits = model(images)           # images: (N, 3, 224, 224), ImageNet-normalized

    Args:
        run: Checkpoint file, run folder, or config folder; see :func:`find_checkpoint`.
        which: ``"best"`` or ``"last"`` checkpoint.
        device: Where to put the weights, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        The trained ``LitResNet``. It also works with ``trainer.test(model, datamodule=data)``.
    """
    checkpoint = find_checkpoint(run, which)
    model_cfg = PROJECT.run_config(checkpoint)["model"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net = ResNetClassifier(
        num_classes=len(state["hyper_parameters"]["class_names"]),
        pretrained=False,
        freeze_backbone=model_cfg["freeze_backbone"],
        dropout=model_cfg["dropout"],
    )
    model = LitResNet.load_from_checkpoint(checkpoint, net=net, map_location=device)
    model.eval()
    print(f"Loaded {checkpoint}")
    return model
