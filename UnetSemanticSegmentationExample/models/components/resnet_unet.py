"""A U-Net whose encoder is torchvision's ImageNet-pretrained ResNet-34."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision import models

from .unet import DoubleConv


class DecoderBlock(nn.Module):
    """Upsample by 2, concatenate the encoder's skip feature map, and convolve.

    Unlike :class:`~UnetSemanticSegmentationExample.models.components.unet.UpBlock`,
    the upsampling is bilinear interpolation (no weights), as in most U-Nets
    built on pretrained encoders, and the skip can have any number of
    channels, or be absent.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """Build the block.

        Args:
            in_channels: Channels of the deeper feature map.
            skip_channels: Channels of the encoder feature map concatenated
                after upsampling (0: no skip connection).
            out_channels: Channels of the output.
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor | None = None) -> Tensor:
        """Upsample ``x``, concatenate ``skip`` if given, and convolve."""
        x = self.up(x)
        if skip is not None:
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResNetUNet(nn.Module):
    """U-Net with a ResNet-34 encoder (pretrained on ImageNet) and a new decoder.

    The **encoder** is torchvision's ResNet-34 without its classifier
    (21.3 million parameters), cut into the five stages whose outputs become
    the skip connections. For a ``(B, 3, 256, 256)`` image:

    .. code-block:: text

        encoder.stem    conv 7x7 stride 2 + BN + ReLU  (B,  64, 128, 128) ── skip ┐
        encoder.layer1  max-pool + 3 residual blocks   (B,  64,  64,  64) ─ skip ┐ │
        encoder.layer2  4 residual blocks              (B, 128,  32,  32) skip ┐ │ │
        encoder.layer3  6 residual blocks              (B, 256,  16,  16) ─┐   │ │ │
        encoder.layer4  3 residual blocks (bottleneck) (B, 512,   8,   8)  │   │ │ │
        decoder[0]  up + concat layer3 + conv          (B, 256,  16,  16) ◄┘   │ │ │
        decoder[1]  up + concat layer2 + conv          (B, 128,  32,  32) ◄────┘ │ │
        decoder[2]  up + concat layer1 + conv          (B,  64,  64,  64) ◄──────┘ │
        decoder[3]  up + concat stem + conv            (B,  32, 128, 128) ◄────────┘
        decoder[4]  up + conv (no skip)                (B,  16, 256, 256)
        head  1 x 1 convolution                        (B,   1, 256, 256)  logits

    The **decoder** and **head** are new, randomly initialized layers
    (3.2 million parameters). With ``freeze_encoder``, the encoder gets no
    gradients and its BatchNorm layers keep their ImageNet statistics, so only
    the decoder learns; :meth:`set_encoder_frozen` unfreezes it later.
    """

    def __init__(
        self,
        out_channels: int = 1,
        pretrained: bool = True,
        freeze_encoder: bool = False,
    ) -> None:
        """Build the encoder from ResNet-34 and the new decoder.

        Args:
            out_channels: Logits per pixel (1 for binary segmentation).
            pretrained: Start the encoder from torchvision's ImageNet weights
                (``ResNet34_Weights.DEFAULT``, downloaded once, 83 MB, to
                ``~/.cache/torch``) instead of random weights.
            freeze_encoder: Start with the encoder frozen.
        """
        super().__init__()
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )
        self.encoder = nn.ModuleDict(
            {
                "stem": nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu),
                "layer1": nn.Sequential(resnet.maxpool, resnet.layer1),
                "layer2": resnet.layer2,
                "layer3": resnet.layer3,
                "layer4": resnet.layer4,
            }
        )
        self.decoder = nn.ModuleList(
            [
                DecoderBlock(512, 256, 256),
                DecoderBlock(256, 128, 128),
                DecoderBlock(128, 64, 64),
                DecoderBlock(64, 64, 32),
                DecoderBlock(32, 0, 16),
            ]
        )
        self.head = nn.Conv2d(16, out_channels, 1)
        self.encoder_frozen = False
        self.set_encoder_frozen(freeze_encoder)

    def set_encoder_frozen(self, frozen: bool) -> None:
        """Freeze or unfreeze the encoder.

        Args:
            frozen: ``True``: no gradients for the encoder, and its BatchNorm
                layers in eval mode. ``False``: the encoder trains like the rest.
        """
        self.encoder_frozen = frozen
        for p in self.encoder.parameters():
            p.requires_grad = not frozen
        self.train(self.training)

    def train(self, mode: bool = True) -> ResNetUNet:
        """Switch to training or eval mode, keeping a frozen encoder in eval mode.

        ``requires_grad = False`` stops the weights from changing, but in
        training mode BatchNorm would still update its running statistics with
        every batch.

        Args:
            mode: ``True`` for training mode, ``False`` for eval mode.

        Returns:
            The network itself.
        """
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Return logits ``(B, out_channels, H, W)`` for images ``(B, 3, H, W)``.

        ``H`` and ``W`` must be divisible by 32.
        """
        skips = []
        for name in ("stem", "layer1", "layer2", "layer3"):
            x = self.encoder[name](x)
            skips.append(x)
        x = self.encoder["layer4"](x)
        for block in self.decoder:
            x = block(x, skips.pop() if skips else None)
        return self.head(x)
