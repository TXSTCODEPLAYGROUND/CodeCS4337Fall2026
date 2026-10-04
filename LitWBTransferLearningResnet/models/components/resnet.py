"""ResNet-18 with a new classification head, for transfer learning."""

from torch import Tensor, nn
from torchvision import models


class ResNetClassifier(nn.Module):
    """A ResNet-18 backbone followed by a new head for ``num_classes`` classes.

    The backbone is torchvision's ResNet-18 without its last layer: images
    ``(N, 3, 224, 224)`` go in, the 512 numbers after global average pooling
    come out. The head is dropout followed by ``Linear(512, num_classes)``.

    The three transfer-learning setups of the configs:

    - **Feature extraction** (``pretrained=True, freeze_backbone=True``): the
      ImageNet weights are kept as they are and only the head is trained.
    - **Fine-tuning** (``pretrained=True, freeze_backbone=False``): everything
      is trained, starting from the ImageNet weights.
    - **From scratch** (``pretrained=False``): random weights, a baseline that
      shows what the ImageNet weights are worth.
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """Build the backbone and the head.

        Args:
            num_classes: Number of classes of the new task, e.g. 102.
            pretrained: Start the backbone from torchvision's ImageNet weights
                (``ResNet18_Weights.DEFAULT``, downloaded once to
                ``~/.cache/torch``) instead of random weights.
            freeze_backbone: Train only the head: the backbone gets no
                gradients, and its BatchNorm layers keep their ImageNet
                statistics.
            dropout: Dropout probability before the head's linear layer.
        """
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(in_features, num_classes)
        )
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        # Lightning does not call train() when fitting starts: it keeps the
        # mode each module has, so the frozen backbone must start in eval mode.
        self.train()

    def forward(self, x: Tensor) -> Tensor:
        """Return class logits ``(N, num_classes)`` for images ``(N, 3, H, W)``."""
        return self.head(self.backbone(x))

    def train(self, mode: bool = True) -> "ResNetClassifier":
        """Switch to training or eval mode, keeping a frozen backbone in eval mode.

        ``requires_grad = False`` stops the weights from changing, but in
        training mode BatchNorm would still update its running mean and
        variance with every batch, so the frozen features would drift.

        Args:
            mode: ``True`` for training mode, ``False`` for eval mode.

        Returns:
            The network itself.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
