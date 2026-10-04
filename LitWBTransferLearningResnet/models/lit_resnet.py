"""The LightningModule (Lightning's "system"): how the network is trained and evaluated."""

import lightning as L
import torch
from torch.nn import functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from .components import ResNetClassifier

STAGES = ("train", "val", "test")

Batch = tuple[torch.Tensor, torch.Tensor]
"""A batch from the dataloaders: images ``(N, 3, 224, 224)`` and labels ``(N,)``."""


class LitResNet(L.LightningModule):
    """Trains and evaluates a :class:`~LitWBTransferLearningResnet.models.components.ResNetClassifier`.

    For each stage (``train``, ``val``, ``test``) it logs, once per epoch:

    - ``<stage>_loss``: cross-entropy loss.
    - ``<stage>_acc``: accuracy over all images.
    - ``<stage>_acc_top5``: how often the true class is among the 5 highest
      scores. With 102 similar flowers, it shows whether a wrong answer was
      still a near miss.
    - ``<stage>_precision``, ``<stage>_recall``, ``<stage>_f1``: macro
      averages (the mean over classes, so every class counts the same).

    The per-class metrics are logged as a table at the end, by
    :class:`~LitWBTransferLearningResnet.callbacks.wandb_evaluation.LogEvaluation`,
    instead of 102 charts per stage.
    """

    def __init__(
        self,
        net: ResNetClassifier,
        class_names: list[str],
        lr: float = 1e-3,
        backbone_lr: float = 1e-4,
        weight_decay: float = 0.0,
    ) -> None:
        """Store the network, the class names, and the optimizer settings.

        Args:
            net: The network; its ``head`` and ``backbone`` get their own
                learning rates.
            class_names: One name per class, in label order.
            lr: Adam learning rate of the new head.
            backbone_lr: Adam learning rate of the backbone, usually smaller
                than ``lr`` when fine-tuning, so the pretrained features are
                adjusted, not overwritten. Unused when the backbone is frozen.
            weight_decay: Adam weight decay (L2 penalty).
        """
        super().__init__()
        # The network's weights are already in every checkpoint; only the plain
        # settings go to self.hparams. logger=False stops the logger from writing
        # them to hparams.yaml; main.py writes hparams.json instead.
        self.save_hyperparameters(ignore=["net"], logger=False)
        self.net = net

        num_classes = len(class_names)
        # torchmetrics objects keep running totals across batches, so each stage
        # needs its own copy. Lightning computes and resets them every epoch.
        overall = MetricCollection(
            {
                "acc": MulticlassAccuracy(num_classes, average="micro"),
                "acc_top5": MulticlassAccuracy(num_classes, top_k=5, average="micro"),
                "precision": MulticlassPrecision(num_classes, average="macro"),
                "recall": MulticlassRecall(num_classes, average="macro"),
                "f1": MulticlassF1Score(num_classes, average="macro"),
            },
            # Automatic compute groups merge metrics whose states are equal after
            # the first batch. With a barely trained model, top-5 accuracy can
            # match the others on that batch, and recall would then be computed
            # from the top-5 counts.
            compute_groups=False,
        )
        for stage in STAGES:
            setattr(self, f"{stage}_metrics", overall.clone(prefix=f"{stage}_"))

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
        return self._shared_step(batch, "train")[0]

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Log the loss and metrics of one validation batch.

        Lightning has already switched to eval mode and turned gradients off.

        Args:
            batch: Images and labels.
            batch_idx: Index of the batch in the epoch (unused).

        Returns:
            The class logits, passed to the callbacks' ``on_validation_batch_end``.
        """
        return self._shared_step(batch, "val")[1]

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Log the loss and metrics of one test batch.

        Args:
            batch: Images and labels.
            batch_idx: Index of the batch (unused).

        Returns:
            The class logits, passed to the callbacks' ``on_test_batch_end``.
        """
        return self._shared_step(batch, "test")[1]

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use Adam, with ``lr`` for the head and ``backbone_lr`` for the backbone.

        Parameters of a frozen backbone have ``requires_grad = False`` and are
        left out of the optimizer.
        """
        groups = [{"params": list(self.net.head.parameters()), "lr": self.hparams.lr}]
        backbone = [p for p in self.net.backbone.parameters() if p.requires_grad]
        if backbone:
            groups.append({"params": backbone, "lr": self.hparams.backbone_lr})
        return torch.optim.Adam(groups, weight_decay=self.hparams.weight_decay)

    def _shared_step(
        self, batch: Batch, stage: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the loss, update the stage's metrics, and log them.

        Args:
            batch: Images and labels.
            stage: ``"train"``, ``"val"``, or ``"test"``; used as the prefix of
                the logged names.

        Returns:
            The cross-entropy loss of the batch and the class logits.
        """
        images, labels = batch
        logits = self(images)
        loss = F.cross_entropy(logits, labels)

        metrics = getattr(self, f"{stage}_metrics")
        # The logits, not the predicted classes: top-5 accuracy needs the scores.
        metrics.update(logits, labels)

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
        return loss, logits
