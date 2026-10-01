import torch
from torch import nn


class ConvNet(nn.Module):
    """Small two-block convolutional network for 28x28 grayscale images."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.25) -> None:
        """Initialize the network layers.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout probability applied before the final linear layer.
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images.

        Args:
            x: Input tensor of shape ``(N, 1, 28, 28)``.

        Returns:
            Unnormalized logits of shape ``(N, num_classes)``.
        """
        return self.classifier(self.features(x))
