"""The LightningModule (Lightning's "system"): how the network is trained and evaluated."""

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)

STAGES = ("train", "val", "test")

Batch = tuple[torch.Tensor, torch.Tensor]
"""A batch from the dataloaders: images ``(N, 1, 28, 28)`` and labels ``(N,)``."""


class LitConvNet(L.LightningModule):
    """Trains and evaluates an image classifier.

    The network is passed in, so any ``nn.Module`` that maps images to class
    logits works. Lightning runs the loops, moves data to the device, and calls
    ``training_step``, ``validation_step``, and ``test_step`` for each batch and
    ``configure_optimizers`` once at the start.

    For each stage (``train``, ``val``, ``test``) it logs, once per epoch:

    - ``<stage>_loss``: cross-entropy loss.
    - ``<stage>_acc``: accuracy over all samples.
    - ``<stage>_precision`` and ``<stage>_recall``: macro averages (the mean
      over classes, so every class counts the same).
    - ``<stage>_acc_<class>``: accuracy on the samples of one class, e.g.
      ``val_acc_Sneaker``.
    """

    def __init__(
        self,
        net: nn.Module,
        class_names: list[str],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        """Store the network, the class names, and the optimizer settings.

        Args:
            net: Network that maps images to class logits, e.g.
                :class:`~LitWBTrainingBasicConvnet.models.components.ConvNet`.
            class_names: One name per class, in label order. Used to name the
                per-class metrics.
            lr: Adam learning rate.
            weight_decay: Adam weight decay (L2 penalty).
        """
        super().__init__()
        # The network's weights are already in every checkpoint; only the plain
        # settings go to self.hparams. logger=False stops the logger from writing
        # them to hparams.yaml; main.py writes hparams.json instead.
        self.save_hyperparameters(ignore=["net"], logger=False)
        self.net = net

        num_classes = len(class_names)
        # Metric names can't contain "/" or spaces, e.g. "T-shirt/top" -> "T-shirt_top".
        self.metric_class_names = [
            name.replace("/", "_").replace(" ", "_") for name in class_names
        ]
        # torchmetrics objects keep running totals across batches, so each stage
        # needs its own copy. Lightning computes and resets them every epoch.
        overall = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes, average="micro"),
                "precision": MulticlassPrecision(num_classes, average="macro"),
                "recall": MulticlassRecall(num_classes, average="macro"),
            }
        )
        for stage in STAGES:
            setattr(self, f"{stage}_metrics", overall.clone(prefix=f"{stage}_"))
            setattr(
                self,
                f"{stage}_class_acc",
                MulticlassAccuracy(num_classes, average=None),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the network's class logits."""
        return self.net(x)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Compute and log the loss and metrics of one training batch.

        Lightning then runs ``backward`` and the optimizer step on the returned
        loss.

        Args:
            batch: Images and labels.
            batch_idx: Index of the batch in the epoch (unused).

        Returns:
            The cross-entropy loss of the batch.
        """
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Log the loss and metrics of one validation batch.

        Lightning has already switched to eval mode and turned gradients off.

        Args:
            batch: Images and labels.
            batch_idx: Index of the batch in the epoch (unused).
        """
        self._shared_step(batch, "val")

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        """Log the loss and metrics of one test batch.

        Args:
            batch: Images and labels.
            batch_idx: Index of the batch (unused).
        """
        self._shared_step(batch, "test")

    def on_train_epoch_end(self) -> None:
        """Log the per-class training accuracy of the finished epoch."""
        self._log_class_acc("train")

    def on_validation_epoch_end(self) -> None:
        """Log the per-class validation accuracy of the finished epoch."""
        self._log_class_acc("val")

    def on_test_epoch_end(self) -> None:
        """Log the per-class test accuracy."""
        self._log_class_acc("test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use Adam with the learning rate and weight decay from the config."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

    def _shared_step(self, batch: Batch, stage: str) -> torch.Tensor:
        """Compute the loss, update the stage's metrics, and log them.

        Args:
            batch: Images and labels.
            stage: ``"train"``, ``"val"``, or ``"test"``; used as the prefix of
                the logged names.

        Returns:
            The cross-entropy loss of the batch.
        """
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(dim=1)

        metrics = getattr(self, f"{stage}_metrics")
        metrics.update(preds, labels)
        getattr(self, f"{stage}_class_acc").update(preds, labels)

        # Passing the metric object (not a number) lets Lightning compute it over
        # the whole epoch. Only loss and accuracy go to the progress bar.
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(labels),
        )
        for name, metric in metrics.items():
            self.log(
                name,
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=name.endswith("_acc"),
            )
        return loss

    def _log_class_acc(self, stage: str) -> None:
        """Log ``<stage>_acc_<class>`` for every class, then reset the totals.

        Args:
            stage: ``"train"``, ``"val"``, or ``"test"``.
        """
        metric = getattr(self, f"{stage}_class_acc")
        per_class = metric.compute()
        self.log_dict(
            {
                f"{stage}_acc_{name}": value
                for name, value in zip(self.metric_class_names, per_class)
            }
        )
        metric.reset()
