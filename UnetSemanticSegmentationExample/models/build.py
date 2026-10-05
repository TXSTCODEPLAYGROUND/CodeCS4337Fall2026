"""Build the network named in a config's ``"model"`` section."""

from torch import nn

from .components import ResNetUNet, UNet

ARCHITECTURES = ("unet", "resnet34_unet")
"""Values of ``"architecture"`` in a config's ``"model"`` section."""


def build_net(model_cfg: dict, pretrained: bool = True) -> nn.Module:
    """Create the network of a config.

    Args:
        model_cfg: The config's ``"model"`` section. ``"architecture": "unet"``
            builds a :class:`~UnetSemanticSegmentationExample.models.components.UNet`
            with its ``"base_channels"`` and ``"depth"``;
            ``"architecture": "resnet34_unet"`` builds a
            :class:`~UnetSemanticSegmentationExample.models.components.ResNetUNet`,
            pretrained if ``"pretrained"`` is true.
        pretrained: ``False`` skips downloading the ImageNet weights, e.g.
            when all weights come from a checkpoint anyway.

    Returns:
        The network, with one output channel (binary segmentation).
    """
    architecture = model_cfg["architecture"]
    if architecture == "unet":
        return UNet(base_channels=model_cfg["base_channels"], depth=model_cfg["depth"])
    if architecture == "resnet34_unet":
        return ResNetUNet(pretrained=model_cfg["pretrained"] and pretrained)
    raise ValueError(
        f"architecture must be one of {ARCHITECTURES}, got {architecture!r}"
    )
