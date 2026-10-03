"""Plain ``nn.Module`` networks, with no training code."""

from .convnet import ConvBlock, ConvNet, block_channels

__all__ = ["ConvBlock", "ConvNet", "block_channels"]
