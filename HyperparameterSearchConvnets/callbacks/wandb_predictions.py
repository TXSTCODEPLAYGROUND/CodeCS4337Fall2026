"""A callback that logs test predictions to W&B: an image table and a confusion matrix."""

import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger


class LogTestPredictions(L.Callback):
    """After testing, log the model's test predictions to W&B.

    Adds two panels to the W&B run:

    - ``test_predictions``: a table with one row per image (the image, the
      predicted class, the true class, and whether it is correct) for the first
      ``num_images`` test images. In W&B, filter it on ``correct`` to see only
      the mistakes, or group it by class.
    - ``confusion_matrix``: for every true class, how often each class was
      predicted, over the whole test set.

    A callback adds behavior to training without changing the LightningModule:
    Lightning calls its ``on_...`` hooks at the matching moments.
    """

    def __init__(
        self, class_names: list[str], mean: float, std: float, num_images: int = 1000
    ) -> None:
        """Store the settings.

        Args:
            class_names: One name per class, in label order.
            mean: Mean used to normalize the images (undone for display).
            std: Standard deviation used to normalize the images.
            num_images: Number of test images shown in the table.
        """
        self.class_names = class_names
        self.mean = mean
        self.std = std
        self.num_images = num_images

    def on_test_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Start with empty lists of labels, predictions, and table rows."""
        self.labels: list[int] = []
        self.preds: list[int] = []
        self.rows: list[list] = []

    def on_test_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: object,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Predict the batch's classes and keep them, plus table rows for the first images."""
        images, labels = batch
        preds = pl_module(images).argmax(dim=1)
        self.labels += labels.tolist()
        self.preds += preds.tolist()
        for image, label, pred in zip(images, labels, preds):
            if len(self.rows) == self.num_images:
                break
            pixels = (image.squeeze() * self.std + self.mean).clamp(0, 1)
            self.rows.append(
                [
                    wandb.Image((pixels * 255).byte().cpu().numpy()),
                    self.class_names[pred],
                    self.class_names[label],
                    bool(pred == label),
                ]
            )

    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Send the table and the confusion matrix to the W&B logger."""
        logger = next(lg for lg in trainer.loggers if isinstance(lg, WandbLogger))
        logger.log_table(
            "test_predictions",
            columns=["image", "predicted", "true", "correct"],
            data=self.rows,
        )
        confusion = wandb.plot.confusion_matrix(
            y_true=self.labels, preds=self.preds, class_names=self.class_names
        )
        logger.experiment.log({"confusion_matrix": confusion})
