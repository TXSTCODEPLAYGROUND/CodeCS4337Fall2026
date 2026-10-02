import lightning as L
import torch
from torch import nn
from torch.nn import functional as F


class LitConvNet(L.LightningModule):
    """Small two-block ConvNet for 28x28 grayscale images, with its training logic.

    Lightning runs the loops, moves data to the device, and calls the methods
    below: ``training_step``, ``validation_step``, and ``test_step`` for each
    batch, and ``configure_optimizers`` once at the start.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.25,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        """Build the network.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout probability before the final linear layer.
            lr: Adam learning rate.
            weight_decay: Adam weight decay (L2 penalty).
        """
        super().__init__()
        # Stores the arguments in self.hparams and in every checkpoint.
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits of shape ``(N, num_classes)`` for images ``(N, 1, 28, 28)``."""
        return self.net(x)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        """Compute and log loss and accuracy as ``<stage>_loss`` and ``<stage>_acc``."""
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        self.log_dict(
            {f"{stage}_loss": loss, f"{stage}_acc": acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(labels),
        )
        return loss

    def training_step(self, batch, batch_idx):
        """Return the training loss; Lightning runs backward and the optimizer step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Log validation loss and accuracy (no gradients, model in eval mode)."""
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Log test loss and accuracy."""
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use Adam with the learning rate and weight decay from the config."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
