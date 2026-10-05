"""The LightningModule (Lightning's "system"): how a segmentation network is trained and evaluated."""

from __future__ import annotations

import lightning as L
import torch
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

from .losses import segmentation_loss

STAGES = ("train", "val", "test")

Batch = tuple[torch.Tensor, torch.Tensor]
"""A batch from the dataloaders: images ``(B, 3, H, W)`` and masks ``(B, 1, H, W)``."""


class LitUNet(L.LightningModule):
    """Trains and evaluates a binary segmentation network (a U-Net).

    For each stage (``train``, ``val``, ``test``) it logs, once per epoch,
    with every pixel of every image counted together:

    - ``<stage>_loss``: the config's loss (BCE, Dice, BCE + Dice, or focal).
    - ``<stage>_dice``: Dice coefficient, ``2 TP / (2 TP + FP + FN)``, the F1
      score of the "polyp" pixels. 1 is a perfect overlap.
    - ``<stage>_iou``: intersection over union (Jaccard index),
      ``TP / (TP + FP + FN)``; always lower than Dice.
    - ``<stage>_precision``: share of the pixels predicted as polyp that are polyp.
    - ``<stage>_recall``: share of the polyp pixels that are found.
    - ``<stage>_pixel_acc``: share of all pixels classified correctly. With
      85 % background, predicting "background" everywhere already scores 0.85,
      which is why Dice and IoU are the main metrics.

    A pixel is predicted as polyp when ``sigmoid(logit) >= threshold``.
    """

    def __init__(
        self,
        net: nn.Module,
        loss: str = "bce",
        optimizer: str = "adam",
        lr: float = 1e-3,
        encoder_lr: float | None = None,
        weight_decay: float = 0.0,
        scheduler: str = "none",
        freeze_encoder_epochs: int = 0,
        threshold: float = 0.5,
    ) -> None:
        """Store the network and the training settings.

        Args:
            net: The network; it returns one logit per pixel.
            loss: ``"bce"``, ``"dice"``, ``"bce_dice"``, or ``"focal"``
                (:func:`~UnetSemanticSegmentationExample.models.losses.segmentation_loss`).
            optimizer: ``"adam"`` or ``"adamw"`` (Adam with decoupled weight decay).
            lr: Learning rate (of the decoder, for a network with a pretrained encoder).
            encoder_lr: Learning rate of ``net.encoder``, usually smaller so
                the pretrained features are adjusted, not overwritten.
                ``None``: the whole network uses ``lr``.
            weight_decay: Weight decay of the optimizer.
            scheduler: ``"none"`` or ``"cosine"`` (the learning rates decrease
                along a half cosine, from their start value to 0 at the last epoch).
            freeze_encoder_epochs: Keep ``net.encoder`` frozen during the first
                epochs (the new decoder first learns to use the pretrained
                features), then train everything.
            threshold: Probability above which a pixel counts as polyp in the metrics.
        """
        super().__init__()
        # The network's weights are already in every checkpoint; only the plain
        # settings go to self.hparams. logger=False stops the logger from writing
        # them to hparams.yaml; main.py writes hparams.json instead.
        self.save_hyperparameters(ignore=["net"], logger=False)
        self.net = net
        self.loss_fn = segmentation_loss(loss)
        metrics = MetricCollection(
            {
                "dice": BinaryF1Score(threshold),
                "iou": BinaryJaccardIndex(threshold),
                "precision": BinaryPrecision(threshold),
                "recall": BinaryRecall(threshold),
                "pixel_acc": BinaryAccuracy(threshold),
            }
        )
        # torchmetrics objects keep running totals across batches, so each stage
        # needs its own copy. Lightning computes and resets them every epoch.
        for stage in STAGES:
            setattr(self, f"{stage}_metrics", metrics.clone(prefix=f"{stage}_"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the logits ``(B, 1, H, W)`` for images ``(B, 3, H, W)``."""
        return self.net(x)

    def on_train_epoch_start(self) -> None:
        """Freeze the encoder during the first ``freeze_encoder_epochs`` epochs."""
        if not hasattr(self.net, "set_encoder_frozen"):
            return
        frozen = self.current_epoch < self.hparams.freeze_encoder_epochs
        if frozen != self.net.encoder_frozen:
            self.net.set_encoder_frozen(frozen)
            print(
                f"Epoch {self.current_epoch}: encoder {'frozen' if frozen else 'unfrozen'}"
            )
        self.log("encoder_frozen", float(frozen), on_step=False, on_epoch=True)

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Compute and log the loss and metrics of one training batch.

        Lightning then runs ``backward`` and the optimizer step on the returned loss.

        Args:
            batch: Images and masks.
            batch_idx: Index of the batch in the epoch (unused).

        Returns:
            The loss of the batch.
        """
        return self._shared_step(batch, "train")[0]

    def validation_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Log the loss and metrics of one validation batch.

        Returns:
            The logits, passed to the callbacks' ``on_validation_batch_end``.
        """
        return self._shared_step(batch, "val")[1]

    def test_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Log the loss and metrics of one test batch.

        Returns:
            The logits, passed to the callbacks' ``on_test_batch_end``.
        """
        return self._shared_step(batch, "test")[1]

    def configure_optimizers(
        self,
    ) -> OptimizerLRSchedulerConfig | torch.optim.Optimizer:
        """Adam or AdamW, with ``encoder_lr`` for ``net.encoder`` if set, and the scheduler.

        Frozen encoder weights are in the optimizer too: while they have no
        gradient, the optimizer skips them, and they start training as soon
        as they are unfrozen.
        """
        encoder = getattr(self.net, "encoder", None)
        if encoder is not None and self.hparams.encoder_lr is not None:
            encoder_ids = {id(p) for p in encoder.parameters()}
            groups = [
                {"params": list(encoder.parameters()), "lr": self.hparams.encoder_lr},
                {
                    "params": [
                        p for p in self.net.parameters() if id(p) not in encoder_ids
                    ],
                    "lr": self.hparams.lr,
                },
            ]
        else:
            groups = [{"params": list(self.net.parameters()), "lr": self.hparams.lr}]
        optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        optimizer = optimizers[self.hparams.optimizer](
            groups, weight_decay=self.hparams.weight_decay
        )
        if self.hparams.scheduler == "none":
            return optimizer
        if self.hparams.scheduler != "cosine":
            raise ValueError(
                f"scheduler must be none or cosine: {self.hparams.scheduler!r}"
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _shared_step(
        self, batch: Batch, stage: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the loss, update the stage's metrics, and log them.

        Args:
            batch: Images ``(B, 3, H, W)`` and masks ``(B, 1, H, W)``.
            stage: ``"train"``, ``"val"``, or ``"test"``; the prefix of the logged names.

        Returns:
            The loss of the batch and the logits ``(B, 1, H, W)``.
        """
        images, masks = batch
        logits = self(images)
        loss = self.loss_fn(logits, masks)

        metrics = getattr(self, f"{stage}_metrics")
        metrics.update(torch.sigmoid(logits), masks.int())
        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(images),
        )
        for name, metric in metrics.items():
            self.log(
                name,
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=name.endswith("dice"),
            )
        return loss, logits
