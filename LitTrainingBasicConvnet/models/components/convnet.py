"""The ConvNet network (Lightning's "model"): plain PyTorch, no training code."""

import torch
from torch import nn


class ConvNet(nn.Module):
    """Small two-block convolutional network for 28x28 grayscale images.

    A plain ``nn.Module``: layers and ``forward`` only, no training code, so it
    can be used with or without Lightning.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        """Build the layers.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout probability before the final linear layer.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits of shape ``(N, num_classes)`` for images ``(N, 1, 28, 28)``."""
        return self.classifier(self.features(x))
