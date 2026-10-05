"""A callback that logs segmentation masks and per-image scores to W&B."""

import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import Dataset


class LogSegmentation(L.Callback):
    """Log predicted masks over the images, and per-image Dice and IoU, to W&B.

    In W&B, each image shows the ground-truth mask and the predicted mask as
    layers that can be switched on and off (``wandb.Image(..., masks=...)``).

    During training, after every validation epoch:

    - ``val_progress``: the same ``num_progress`` validation images every
      epoch, so the W&B slider shows the predictions improving.

    After training, for the best checkpoint (``trainer.validate`` and
    ``trainer.test`` in ``main``), for a split ``<split>`` (``val`` or ``test``):

    - ``<split>_per_image``: one row per image with its Dice, IoU, polyp area
      (share of the image), and predicted area. Sort by ``dice`` to find the
      hardest images.
    - ``<split>_worst_examples`` and ``<split>_best_examples``: the
      ``num_examples`` images with the lowest and highest Dice.

    A callback adds behavior to training without changing the LightningModule:
    Lightning calls its ``on_...`` hooks at the matching moments.
    """

    def __init__(
        self,
        class_names: tuple[str, ...],
        mean: tuple[float, ...],
        std: tuple[float, ...],
        threshold: float = 0.5,
        num_progress: int = 8,
        num_examples: int = 8,
    ) -> None:
        """Store the settings.

        Args:
            class_names: Name of mask value 0 and of mask value 1.
            mean: Per-channel mean used to normalize the images (undone for display).
            std: Per-channel standard deviation used to normalize the images.
            threshold: Probability above which a pixel is predicted as foreground.
            num_progress: Validation images shown after every epoch.
            num_examples: Images in each of the worst and best galleries.
        """
        self.class_labels = dict(enumerate(class_names))
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.threshold = threshold
        self.num_progress = num_progress
        self.num_examples = num_examples

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Start collecting the per-image scores."""
        self._start()

    def on_test_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Start collecting the per-image scores."""
        self._start()

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Keep the batch's per-image scores, outside training.

        ``outputs`` is what ``validation_step`` returned: the logits.
        """
        if trainer.state.fn == TrainerFn.VALIDATING:
            self._collect(outputs, batch[1])

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Keep the batch's per-image scores (from ``test_step``'s logits)."""
        self._collect(outputs, batch[1])

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Log the progress gallery during training, or the full report after it."""
        if trainer.sanity_checking:
            return
        if trainer.state.fn == TrainerFn.VALIDATING:
            self._log_report(trainer, "val", trainer.datamodule.val_set)
        else:
            self._log_progress(trainer, pl_module, trainer.datamodule.val_set)

    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Log the test report."""
        self._log_report(trainer, "test", trainer.datamodule.test_set)

    def _start(self) -> None:
        """Empty the per-image score lists."""
        self.scores: list[torch.Tensor] = []

    def _collect(self, logits: torch.Tensor, masks: torch.Tensor) -> None:
        """Compute and keep each image's Dice, IoU, true and predicted area.

        An image with no polyp and no predicted polyp counts as a perfect 1.0.

        Args:
            logits: ``(B, 1, H, W)`` logits of the batch.
            masks: ``(B, 1, H, W)`` ground-truth masks of 0.0 and 1.0.
        """
        preds = (torch.sigmoid(logits.float()) >= self.threshold).flatten(1).float()
        truth = masks.flatten(1)
        intersection = (preds * truth).sum(dim=1)
        total = preds.sum(dim=1) + truth.sum(dim=1)
        union = total - intersection
        dice = torch.where(total > 0, 2 * intersection / total.clamp(min=1), 1.0)
        iou = torch.where(union > 0, intersection / union.clamp(min=1), 1.0)
        pixels = truth.shape[1]
        self.scores.append(
            torch.stack(
                [dice, iou, truth.sum(dim=1) / pixels, preds.sum(dim=1) / pixels], 1
            ).cpu()
        )

    def _picture(self, image: torch.Tensor) -> np.ndarray:
        """Undo the normalization of a ``(3, H, W)`` image: ``(H, W, 3)`` uint8."""
        image = image.cpu() * self.std + self.mean
        return (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

    def _overlay(
        self, image: torch.Tensor, mask: torch.Tensor, pred: torch.Tensor, caption: str
    ) -> wandb.Image:
        """An image with its ground-truth and predicted masks as W&B layers."""
        return wandb.Image(
            self._picture(image),
            masks={
                "ground_truth": {
                    "mask_data": mask[0].cpu().numpy().astype(np.uint8),
                    "class_labels": self.class_labels,
                },
                "prediction": {
                    "mask_data": pred[0].cpu().numpy().astype(np.uint8),
                    "class_labels": self.class_labels,
                },
            },
            caption=caption,
        )

    def _predict(
        self, pl_module: L.LightningModule, dataset: Dataset, indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the model on some images of ``dataset``.

        Returns:
            The images ``(N, 3, H, W)``, true masks and predicted masks ``(N, 1, H, W)``.
        """
        images = torch.stack([dataset[i][0] for i in indices])
        masks = torch.stack([dataset[i][1] for i in indices])
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            logits = pl_module(images.to(pl_module.device))
        pl_module.train(was_training)
        preds = (torch.sigmoid(logits.float()) >= self.threshold).cpu()
        return images, masks, preds

    def _log_progress(
        self, trainer: L.Trainer, pl_module: L.LightningModule, dataset: Dataset
    ) -> None:
        """Log the first ``num_progress`` validation images with their predicted masks."""
        logger = next(lg for lg in trainer.loggers if isinstance(lg, WandbLogger))
        indices = list(range(min(self.num_progress, len(dataset))))
        images, masks, preds = self._predict(pl_module, dataset, indices)
        logger.experiment.log(
            {
                "val_progress": [
                    self._overlay(
                        images[k], masks[k], preds[k], f"epoch {trainer.current_epoch}"
                    )
                    for k in range(len(indices))
                ],
                "trainer/global_step": trainer.global_step,
            }
        )

    def _log_report(self, trainer: L.Trainer, split: str, dataset: Dataset) -> None:
        """Log the per-image table and the worst and best galleries of a split.

        The batches come in the dataset's order (no shuffling), so score ``i``
        belongs to image ``i`` of ``dataset``.

        Args:
            trainer: The trainer, to find the W&B logger and the model.
            split: ``"val"`` or ``"test"``, the prefix of the logged names.
            dataset: The split's dataset, with the evaluation transform.
        """
        logger = next(lg for lg in trainer.loggers if isinstance(lg, WandbLogger))
        scores = torch.cat(self.scores)
        dice, iou, area, pred_area = scores.unbind(dim=1)
        names = getattr(dataset, "names", [str(i) for i in range(len(dataset))])
        order = dice.argsort()
        worst = order[: self.num_examples].tolist()
        best = order.flip(0)[: self.num_examples].tolist()
        shown = {}
        for indices in (worst, best):
            images, masks, preds = self._predict(
                trainer.lightning_module, dataset, indices
            )
            for k, i in enumerate(indices):
                shown[i] = self._overlay(
                    images[k], masks[k], preds[k], f"{names[i]}: Dice {dice[i]:.3f}"
                )

        rows = [
            [
                names[i],
                round(dice[i].item(), 4),
                round(iou[i].item(), 4),
                round(area[i].item(), 4),
                round(pred_area[i].item(), 4),
            ]
            for i in range(len(dice))
        ]
        logger.log_table(
            f"{split}_per_image",
            columns=["image", "dice", "iou", "polyp_area", "predicted_area"],
            data=rows,
        )
        logger.experiment.log(
            {
                f"{split}_worst_examples": [shown[i] for i in worst],
                f"{split}_best_examples": [shown[i] for i in best],
            }
        )
        print(
            f"{split}: mean Dice per image {dice.mean():.4f} (median {dice.median():.4f}), "
            f"{int((dice < 0.5).sum())} of {len(dice)} images below 0.5. "
            f"Worst: {', '.join(f'{names[i]} ({dice[i]:.2f})' for i in worst[:3])}"
        )
