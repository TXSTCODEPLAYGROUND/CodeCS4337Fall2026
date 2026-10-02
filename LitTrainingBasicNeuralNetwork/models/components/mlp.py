"""The MLP network (Lightning's "model"): plain PyTorch, no training code."""

import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected network: flatten the image, then a stack of linear layers.

    With the default ``hidden_sizes=(256, 128)`` the layers are::

        784 pixels -> Linear -> 256 -> ReLU -> Dropout
                   -> Linear -> 128 -> ReLU -> Dropout
                   -> Linear -> 10 logits
    """

    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_sizes: tuple[int, ...] | list[int] = (256, 128),
        num_classes: int = 10,
        dropout: float = 0.2,
    ) -> None:
        """Initialize the network layers.

        Args:
            input_size: Number of input values per image (pixels after flattening).
            hidden_sizes: Number of units in each hidden layer, from first to last.
            num_classes: Number of output classes.
            dropout: Dropout probability applied after every hidden layer.
        """
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = input_size
        for hidden_size in hidden_sizes:
            layers += [
                nn.Linear(in_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_features = hidden_size
        layers.append(nn.Linear(in_features, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images.

        Args:
            x: Input tensor of shape ``(N, 1, 28, 28)``.

        Returns:
            Unnormalized logits of shape ``(N, num_classes)``.
        """
        return self.layers(x)
