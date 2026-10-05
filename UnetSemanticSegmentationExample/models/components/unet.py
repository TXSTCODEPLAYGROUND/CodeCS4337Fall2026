"""A U-Net written out block by block, for training from scratch."""

import torch
from torch import Tensor, nn


class DoubleConv(nn.Module):
    """Two ``3 x 3`` convolutions, each followed by batch normalization and ReLU.

    ``padding=1`` keeps the height and width, so only the pooling and the
    upsampling change the size of the feature maps. (The original U-Net used
    no padding, and its output was smaller than its input.)
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Build the two convolutions.

        Args:
            in_channels: Channels of the input feature map.
            out_channels: Channels of both convolutions' outputs.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """``(B, in, H, W)`` to ``(B, out, H, W)``."""
        return self.block(x)


class UpBlock(nn.Module):
    """One decoder step: upsample, concatenate the skip connection, convolve.

    ``(B, C, H, W)`` is upsampled by a transposed convolution to
    ``(B, C/2, 2H, 2W)``, concatenated along the channels with the encoder's
    feature map of the same size ``(B, C/2, 2H, 2W)`` (the **skip
    connection**), giving ``(B, C, 2H, 2W)``, and a :class:`DoubleConv`
    reduces it to ``(B, C/2, 2H, 2W)``.
    """

    def __init__(self, in_channels: int) -> None:
        """Build the upsampling and the convolutions.

        Args:
            in_channels: Channels ``C`` of the deeper feature map; the output has ``C/2``.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = DoubleConv(in_channels, in_channels // 2)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Upsample ``x``, concatenate ``skip``, and convolve."""
        x = self.up(x)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) for binary segmentation.

    With ``base_channels=32`` and ``depth=4``, a ``(B, 3, 256, 256)`` image goes through:

    .. code-block:: text

        encoders[0]  DoubleConv           (B,  32, 256, 256) ──────────────── skip ┐
        pool + encoders[1]                (B,  64, 128, 128) ─────────── skip ┐    │
        pool + encoders[2]                (B, 128,  64,  64) ────── skip ┐    │    │
        pool + encoders[3]                (B, 256,  32,  32) ─ skip ┐    │    │    │
        pool + bottleneck                 (B, 512,  16,  16)        │    │    │    │
        decoders[0]  up + concat + conv   (B, 256,  32,  32) ◄──────┘    │    │    │
        decoders[1]                       (B, 128,  64,  64) ◄───────────┘    │    │
        decoders[2]                       (B,  64, 128, 128) ◄────────────────┘    │
        decoders[3]                       (B,  32, 256, 256) ◄─────────────────────┘
        head  1 x 1 convolution           (B,   1, 256, 256)   logits

    Each pooling halves the height and width and each encoder doubles the
    channels; the decoder undoes both. The output is one **logit** per pixel:
    ``sigmoid(logit)`` is the probability that the pixel is polyp.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
    ) -> None:
        """Build the encoder, bottleneck, decoder, and head.

        Args:
            in_channels: Channels of the input image (3 for RGB).
            out_channels: Logits per pixel: 1 for binary segmentation, the
                number of classes for multiclass segmentation.
            base_channels: Channels of the first encoder block, doubled at
                each level (the original U-Net used 64).
            depth: Number of poolings; the input side must be divisible by ``2**depth``.
        """
        super().__init__()
        self.depth = depth
        channels = [base_channels * 2**level for level in range(depth + 1)]
        self.encoders = nn.ModuleList(
            [DoubleConv(in_channels, channels[0])]
            + [DoubleConv(channels[k], channels[k + 1]) for k in range(depth - 1)]
        )
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(channels[depth - 1], channels[depth])
        self.decoders = nn.ModuleList(
            [UpBlock(channels[k]) for k in range(depth, 0, -1)]
        )
        self.head = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Return logits ``(B, out_channels, H, W)`` for images ``(B, in_channels, H, W)``."""
        skips = []
        for k, encoder in enumerate(self.encoders):
            x = encoder(x if k == 0 else self.pool(x))
            skips.append(x)
        x = self.bottleneck(self.pool(x))
        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            x = decoder(x, skip)
        return self.head(x)
