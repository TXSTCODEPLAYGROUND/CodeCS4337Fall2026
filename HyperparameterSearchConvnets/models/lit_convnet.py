"""The LightningModule (Lightning's "system"): how the network is trained and evaluated."""

import math

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
)

STAGES = ("train", "val", "test")

OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}
"""Optimizers that can be chosen by name in the config."""
SCHEDULERS = ("none", "step", "cosine", "plateau")
"""Learning-rate schedulers that can be chosen by name in the config."""

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

    ``train_loss`` is the cross-entropy only, so it can be compared with
    ``val_loss``; the L1 penalty is added to the loss that is optimized.
    """

    def __init__(
        self,
        net: nn.Module,
        class_names: list[str],
        optimizer: str = "adam",
        lr: float = 1e-3,
        momentum: float = 0.9,
        l1: float = 0.0,
        l2: float = 0.0,
        scheduler: str = "none",
        step_size: int = 3,
        gamma: float = 0.5,
    ) -> None:
        """Store the network, the class names, and the optimizer settings.

        Args:
            net: Network that maps images to class logits, e.g.
                :class:`~HyperparameterSearchConvnets.models.components.ConvNet`.
            class_names: One name per class, in label order. Used to name the
                per-class metrics.
            optimizer: A key of :data:`OPTIMIZERS`: ``"adam"``, ``"adamw"``,
                ``"sgd"``, or ``"rmsprop"``.
            lr: Learning rate (the starting one, if a scheduler changes it).
            momentum: Momentum, used only by ``"sgd"`` and ``"rmsprop"``.
            l1: Weight of the L1 penalty, ``l1 * sum(|w|)`` over the
                convolution and linear weights (not the biases or BatchNorm),
                added to the training loss. It pushes weights to exactly zero.
            l2: Weight of the L2 penalty, passed to the optimizer as
                ``weight_decay``. It shrinks all weights a little every step.
                For ``"adamw"`` it is decoupled from the gradient, the form
                recommended for Adam.
            scheduler: How the learning rate changes over the epochs, one of
                :data:`SCHEDULERS`: ``"none"`` (constant), ``"step"``
                (multiplied by ``gamma`` every ``step_size`` epochs),
                ``"cosine"`` (decays to 0 over the run along a cosine), or
                ``"plateau"`` (multiplied by ``gamma`` when ``val_loss`` has
                not improved for 1 epoch).
            step_size: Epochs between learning-rate drops for ``"step"``.
            gamma: Factor applied to the learning rate by ``"step"`` and
                ``"plateau"``.

        Raises:
            ValueError: If ``optimizer`` or ``scheduler`` is unknown.
        """
        super().__init__()
        if optimizer not in OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}, choose from {list(OPTIMIZERS)}"
            )
        if scheduler not in SCHEDULERS:
            raise ValueError(
                f"Unknown scheduler {scheduler!r}, choose from {list(SCHEDULERS)}"
            )
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
            The cross-entropy loss of the batch, plus the L1 penalty.
        """
        loss = self._shared_step(batch, "train")
        if self.hparams.l1 > 0:
            # Convolution and linear weights have 2 or more dimensions; biases
            # and BatchNorm's scale and shift have one.
            weights = [p for p in self.net.parameters() if p.dim() > 1]
            loss = loss + self.hparams.l1 * sum(w.abs().sum() for w in weights)
        return loss

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

    def on_train_start(self) -> None:
        """Stretch a cosine schedule over all epochs when a run is resumed.

        Resuming restores the earlier run's scheduler, whose cosine already ends
        at a learning rate of 0 after that run's epochs. The curve is stretched
        to the new number of epochs, and the learning rate set to its value at
        the current epoch, so the extra epochs still learn.
        """
        if self.hparams.scheduler != "cosine":
            return
        scheduler = self.lr_schedulers()
        if scheduler.T_max == self.trainer.max_epochs:
            return
        scheduler.T_max = self.trainer.max_epochs
        # The scheduler updates the learning rate from its previous value, which
        # is 0 here, so set the value of the stretched curve directly.
        factor = (1 + math.cos(math.pi * scheduler.last_epoch / scheduler.T_max)) / 2
        for group, base_lr in zip(scheduler.optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = scheduler.eta_min + (base_lr - scheduler.eta_min) * factor

    def on_train_epoch_end(self) -> None:
        """Log the per-class training accuracy of the finished epoch."""
        self._log_class_acc("train")

    def on_validation_epoch_end(self) -> None:
        """Log the per-class validation accuracy of the finished epoch."""
        self._log_class_acc("val")

    def on_test_epoch_end(self) -> None:
        """Log the per-class test accuracy."""
        self._log_class_acc("test")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build the optimizer and, if one is chosen, the learning-rate scheduler.

        Returns:
            The optimizer alone for ``scheduler="none"``; otherwise a dict that
            also holds the scheduler, which Lightning steps once per epoch.
        """
        hp = self.hparams
        kwargs = {"lr": hp.lr, "weight_decay": hp.l2}
        if hp.optimizer in ("sgd", "rmsprop"):
            kwargs["momentum"] = hp.momentum
        optimizer = OPTIMIZERS[hp.optimizer](self.parameters(), **kwargs)

        if hp.scheduler == "none":
            return optimizer
        if hp.scheduler == "step":
            scheduler = lr_scheduler.StepLR(optimizer, hp.step_size, gamma=hp.gamma)
        elif hp.scheduler == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, self.trainer.max_epochs
            )
        else:
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=hp.gamma, patience=1
            )
        # "monitor" is only read by ReduceLROnPlateau.
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

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
