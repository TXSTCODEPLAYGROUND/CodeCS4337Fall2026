"""The ConvNet builder (Lightning's "model"): plain PyTorch, no training code.

A network is a stack of blocks, built from the config's ``"model"`` section.
Each block can have one or more convolutions, BatchNorm, a skip connection
(as in ResNet), and dropout, so the same code builds anything from a small
two-block ConvNet to a deep residual network.
"""

import torch
from torch import nn

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}
"""Activation functions that can be chosen by name in the config."""

NUM_POOLED_BLOCKS = 2
"""Blocks that end with 2x2 max pooling: 28x28 images become 14x14, then 7x7."""


class ConvBlock(nn.Module):
    """One block: convolutions, then an optional skip connection, pooling, and dropout.

    Without BatchNorm and skip connection, a block with one convolution is::

        x -> Conv 3x3 -> activation -> [MaxPool 2x2] -> Dropout

    With ``convs=2``, BatchNorm, and a skip connection (a ResNet "basic block")::

        x -> Conv 3x3 -> BatchNorm -> activation -> Conv 3x3 -> BatchNorm -> (+) -> activation -> ...
        |                                                                    |
        +---------------------- shortcut (identity or 1x1 conv) -------------+

    The shortcut adds the block's input to its output, so the convolutions only
    learn a correction to the input. Gradients flow back through the addition
    unchanged, which is what lets deep networks train.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        convs: int = 1,
        batch_norm: bool = False,
        skip_connection: bool = False,
        activation: str = "relu",
        pool: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """Build the block's layers.

        Args:
            in_channels: Channels of the block's input.
            out_channels: Channels of every convolution in the block.
            convs: Number of 3x3 convolutions (padding 1, so the image size is kept).
            batch_norm: Add ``BatchNorm2d`` after every convolution (and on
                the shortcut). The convolutions then have no bias, which the
                BatchNorm would cancel anyway.
            skip_connection: Add the block's input to the output of its last
                convolution. When the channels change, a 1x1 convolution
                adapts the input.
            activation: A key of :data:`ACTIVATIONS`.
            pool: End the block with 2x2 max pooling, halving the image size.
            dropout: Dropout probability at the end of the block.
        """
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(convs):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=not batch_norm,
                )
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            # The last activation comes after the skip connection's addition.
            if i < convs - 1:
                layers.append(ACTIVATIONS[activation]())
        self.conv_layers = nn.Sequential(*layers)

        self.shortcut: nn.Module | None = None
        if skip_connection:
            if in_channels == out_channels:
                self.shortcut = nn.Identity()
            else:
                shortcut: list[nn.Module] = [
                    nn.Conv2d(in_channels, out_channels, 1, bias=not batch_norm)
                ]
                if batch_norm:
                    shortcut.append(nn.BatchNorm2d(out_channels))
                self.shortcut = nn.Sequential(*shortcut)
        self.activation = ACTIVATIONS[activation]()
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to feature maps ``(N, in_channels, H, W)``."""
        out = self.conv_layers(x)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return self.dropout(self.pool(self.activation(out)))


class ConvNet(nn.Module):
    """A stack of :class:`ConvBlock` blocks, then a linear layer that predicts the class.

    With the defaults (the base config ``config01.json``) the network is::

        image 1x28x28 -> Conv 3x3 -> 32 channels  -> ReLU -> MaxPool -> 32x14x14
                      -> Conv 3x3 -> 64 channels  -> ReLU -> MaxPool -> 64x7x7
                      -> Flatten -> Linear(64*7*7, 10) -> 10 logits

    The first :data:`NUM_POOLED_BLOCKS` blocks end with max pooling; later
    blocks keep the 7x7 size.
    """

    def __init__(
        self,
        channels: tuple[int, ...] | list[int] = (32, 64),
        convs_per_block: int = 1,
        batch_norm: bool = False,
        skip_connections: bool = False,
        activation: str = "relu",
        dropout: float = 0.0,
        num_classes: int = 10,
        in_channels: int = 1,
        image_size: int = 28,
    ) -> None:
        """Build the blocks and the prediction layer.

        Args:
            channels: Output channels of each block, from first to last; its
                length is the number of blocks.
            convs_per_block: Number of 3x3 convolutions in every block.
            batch_norm: Use BatchNorm in every block.
            skip_connections: Give every block a skip connection.
            activation: A key of :data:`ACTIVATIONS`, used in every block.
            dropout: Dropout probability at the end of every block.
            num_classes: Number of output classes.
            in_channels: Channels of the input images (1 for grayscale).
            image_size: Height and width of the input images.

        Raises:
            ValueError: If ``activation`` is unknown, or ``channels`` or
                ``convs_per_block`` is empty or zero.
        """
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}, choose from {list(ACTIVATIONS)}"
            )
        if not channels or convs_per_block < 1:
            raise ValueError("Need at least one block with at least one convolution")
        blocks = []
        size = image_size
        for i, out_channels in enumerate(channels):
            pool = i < NUM_POOLED_BLOCKS
            blocks.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    convs=convs_per_block,
                    batch_norm=batch_norm,
                    skip_connection=skip_connections,
                    activation=activation,
                    pool=pool,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
            size = size // 2 if pool else size
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(in_channels * size * size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits of shape ``(N, num_classes)`` for images ``(N, 1, 28, 28)``."""
        return self.classifier(self.blocks(x))


def block_channels(first: int, num_blocks: int, max_channels: int = 256) -> list[int]:
    """Channels of each block, as in ResNet: doubled every time the image is halved.

    The first block has ``first`` channels; the blocks after each pooled block
    have twice as many, up to ``max_channels``. For example
    ``block_channels(32, 4)`` is ``[32, 64, 128, 128]``: the first two blocks
    halve the image, so the second and third double the channels.

    Args:
        first: Channels of the first block.
        num_blocks: Number of blocks.
        max_channels: Upper limit, which keeps deep, wide networks trainable.

    Returns:
        One channel count per block.
    """
    return [
        min(first * 2 ** min(i, NUM_POOLED_BLOCKS), max_channels)
        for i in range(num_blocks)
    ]
