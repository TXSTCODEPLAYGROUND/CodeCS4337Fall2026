"""Pixel-wise losses for binary segmentation."""

from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss

LOSSES = ("bce", "dice", "bce_dice", "focal")
"""Loss names accepted by :func:`segmentation_loss`."""


def dice_loss(logits: Tensor, targets: Tensor, smooth: float = 1.0) -> Tensor:
    """Soft Dice loss: 1 minus the Dice overlap of the probabilities and the mask.

    For each image, with probabilities :math:`p_i = \\sigma(z_i)` and labels
    :math:`y_i` over its pixels :math:`i`:

    .. math::

        \\text{Dice} = \\frac{2 \\sum_i p_i y_i + s}{\\sum_i p_i + \\sum_i y_i + s},
        \\qquad L = 1 - \\text{Dice}

    It measures overlap, so it does not care how many background pixels there
    are: a small polyp counts as much as a large one. ``smooth`` (:math:`s`)
    keeps it defined for an image with no polyp.

    Args:
        logits: ``(B, 1, H, W)`` raw network outputs.
        targets: ``(B, 1, H, W)`` masks of 0.0 and 1.0.
        smooth: Added to the numerator and denominator.

    Returns:
        The mean loss over the batch, a scalar.
    """
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    dice = (2 * intersection + smooth) / (
        probs.sum(dim=1) + targets.sum(dim=1) + smooth
    )
    return (1 - dice).mean()


def segmentation_loss(name: str) -> Callable[[Tensor, Tensor], Tensor]:
    """Return the loss function named in a config.

    Args:
        name: ``"bce"`` (binary cross-entropy on the logits, averaged over
            every pixel), ``"dice"`` (:func:`dice_loss`), ``"bce_dice"`` (the
            sum of both), or ``"focal"`` (BCE that down-weights easy pixels,
            torchvision's ``sigmoid_focal_loss`` with ``alpha=0.25, gamma=2``).

    Returns:
        A function ``loss(logits, targets)`` returning a scalar.
    """
    if name == "bce":
        return F.binary_cross_entropy_with_logits
    if name == "dice":
        return dice_loss
    if name == "bce_dice":
        return lambda logits, targets: (
            F.binary_cross_entropy_with_logits(logits, targets)
            + dice_loss(logits, targets)
        )
    if name == "focal":
        return lambda logits, targets: sigmoid_focal_loss(
            logits, targets, reduction="mean"
        )
    raise ValueError(f"loss must be one of {LOSSES}, got {name!r}")
