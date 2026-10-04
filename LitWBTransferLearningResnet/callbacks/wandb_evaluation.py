"""A callback that logs a full evaluation of the model to W&B, for validation and test."""

import lightning as L
import torch
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.trainer.states import TrainerFn
from torch.utils.data import Dataset
from torchmetrics.functional.classification import (
    binary_average_precision,
    binary_precision_recall_curve,
)


class LogEvaluation(L.Callback):
    """Log predictions, examples, per-class metrics, and curves to W&B.

    Runs on the **test** set during ``trainer.test`` and on the **validation**
    set during ``trainer.validate``, which ``main`` calls once after training
    with the best checkpoint. The validation runs at the end of every training
    epoch are skipped, so W&B gets one evaluation per split, not one per epoch.

    For a split ``<split>`` (``val`` or ``test``), it logs:

    - ``<split>_predictions``: a table of ``num_table_images`` random images
      with the true and predicted class, the confidence, and whether the
      prediction is right and whether the true class is in the top 5. Filter on
      ``correct`` in W&B to see only the mistakes.
    - ``<split>_correct_examples``: ``num_examples`` random right predictions.
    - ``<split>_wrong_examples``: the ``num_examples`` most confident mistakes,
      captioned with the true class, the predicted class, and the confidence.
    - ``<split>_per_class``: one row per class with its number of images,
      accuracy, precision, recall, F1, and the class it is most often confused
      with. Sort it by ``f1`` to find the hardest classes. A class's accuracy
      is its recall: the fraction of its images predicted as that class.
    - ``<split>_pr_curve``: precision-recall curves of the ``num_pr_classes``
      worst and best classes by F1, each class against all others, with its F1
      and average precision (AP, the area under its curve) in the legend.
    - ``<split>_confusion_matrix``: true vs. predicted class over the split.
      Only the non-empty cells are sent; the others show as blank.
    - ``<split>_most_confused``: the ``num_confused_pairs`` most frequent
      mistakes, e.g. "true: pink primrose, predicted: tree mallow, 7 times".

    A callback adds behavior to training without changing the LightningModule:
    Lightning calls its ``on_...`` hooks at the matching moments.
    """

    def __init__(
        self,
        class_names: list[str],
        mean: tuple[float, ...],
        std: tuple[float, ...],
        num_table_images: int = 200,
        num_examples: int = 16,
        num_pr_classes: int = 5,
        num_confused_pairs: int = 10,
        seed: int = 0,
    ) -> None:
        """Store the settings.

        Args:
            class_names: One name per class, in label order.
            mean: Per-channel mean used to normalize the images (undone for display).
            std: Per-channel standard deviation used to normalize the images.
            num_table_images: Number of random images in the predictions table.
            num_examples: Number of images in each of the two example galleries.
            num_pr_classes: Number of worst and of best classes in the
                precision-recall plot.
            num_confused_pairs: Number of rows in the most-confused table.
            seed: Seed for picking the random images, so reruns show the same ones.
        """
        self.class_names = class_names
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        self.num_table_images = num_table_images
        self.num_examples = num_examples
        self.num_pr_classes = num_pr_classes
        self.num_confused_pairs = num_confused_pairs
        self.seed = seed

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Start collecting, if this is ``trainer.validate`` (not training)."""
        self._start()

    def on_test_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Start collecting the test predictions."""
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
        """Keep the batch's labels and class probabilities, outside training.

        ``outputs`` is what ``validation_step`` returned: the class logits.
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
        """Keep the batch's labels and class probabilities (from ``test_step``'s logits)."""
        self._collect(outputs, batch[1])

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Log the validation evaluation, outside training."""
        if trainer.state.fn == TrainerFn.VALIDATING:
            self._log(trainer, "val", trainer.datamodule.val_set)

    def on_test_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Log the test evaluation."""
        self._log(trainer, "test", trainer.datamodule.test_set)

    def _start(self) -> None:
        """Empty the lists of labels and probabilities."""
        self.labels: list[torch.Tensor] = []
        self.probs: list[torch.Tensor] = []

    def _collect(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Keep a batch's labels and softmax probabilities (not its images).

        Args:
            logits: The model's class logits for the batch.
            labels: The true classes.
        """
        self.labels.append(labels.cpu())
        self.probs.append(logits.float().softmax(dim=1).cpu())

    def _log(self, trainer: L.Trainer, split: str, dataset: Dataset) -> None:
        """Compute everything from the collected predictions and send it to W&B.

        The batches come in the dataset's order (no shuffling), so position
        ``i`` of the collected predictions is image ``i`` of ``dataset``; the
        images shown are loaded from it again.

        Args:
            trainer: The trainer, to find the W&B logger.
            split: ``"val"`` or ``"test"``, the prefix of the logged names.
            dataset: The split's dataset, with the evaluation transform.
        """
        logger = next(lg for lg in trainer.loggers if isinstance(lg, WandbLogger))
        labels = torch.cat(self.labels)
        probs = torch.cat(self.probs)
        confidence, preds = probs.max(dim=1)
        correct = preds == labels
        top5 = (probs.topk(5, dim=1).indices == labels[:, None]).any(dim=1)
        names = self.class_names
        num_classes = len(names)
        generator = torch.Generator().manual_seed(self.seed)

        confusion = torch.bincount(
            labels * num_classes + preds, minlength=num_classes**2
        ).view(num_classes, num_classes)
        images_per_class = confusion.sum(dim=1)
        hits = confusion.diag()
        precision = hits / confusion.sum(dim=0).clamp(min=1)
        recall = hits / images_per_class.clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
        mistakes = confusion.clone().fill_diagonal_(0)

        def picture(i: int) -> torch.Tensor:
            image = dataset[i][0] * self.std + self.mean
            return (image.clamp(0, 1) * 255).byte().permute(1, 2, 0).numpy()

        def caption(i: int) -> str:
            return (
                f"true: {names[labels[i]]}\n"
                f"predicted: {names[preds[i]]} ({confidence[i]:.0%})"
            )

        sample = torch.randperm(len(labels), generator=generator)[
            : self.num_table_images
        ]
        rows = [
            [
                wandb.Image(picture(i)),
                names[labels[i]],
                names[preds[i]],
                round(confidence[i].item(), 4),
                bool(correct[i]),
                bool(top5[i]),
            ]
            for i in sample.tolist()
        ]
        logger.log_table(
            f"{split}_predictions",
            columns=["image", "true", "predicted", "confidence", "correct", "in_top5"],
            data=rows,
        )

        right = correct.nonzero().flatten()
        right = right[torch.randperm(len(right), generator=generator)]
        wrong = (~correct).nonzero().flatten()
        wrong = wrong[confidence[wrong].argsort(descending=True)]
        logger.experiment.log(
            {
                f"{split}_correct_examples": [
                    wandb.Image(picture(i), caption=caption(i))
                    for i in right[: self.num_examples].tolist()
                ],
                f"{split}_wrong_examples": [
                    wandb.Image(picture(i), caption=caption(i))
                    for i in wrong[: self.num_examples].tolist()
                ],
            }
        )

        per_class = [
            [
                names[c],
                int(images_per_class[c]),
                round(recall[c].item(), 4),
                round(precision[c].item(), 4),
                round(recall[c].item(), 4),
                round(f1[c].item(), 4),
                names[mistakes[c].argmax()] if mistakes[c].any() else "",
            ]
            for c in range(num_classes)
        ]
        logger.log_table(
            f"{split}_per_class",
            columns=[
                "class",
                "images",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "most_confused_with",
            ],
            data=per_class,
        )

        order = f1.argsort()
        plotted = order[: self.num_pr_classes].tolist()
        plotted += order[-self.num_pr_classes :].flip(0).tolist()
        recalls, precisions, keys = [], [], []
        for c in plotted:
            # One class against all others: its probability is the score.
            scores, is_class = probs[:, c], labels == c
            precision_c, recall_c, _ = binary_precision_recall_curve(scores, is_class)
            ap = binary_average_precision(scores, is_class)
            keep = torch.linspace(0, len(recall_c) - 1, min(len(recall_c), 200)).long()
            recalls.append(recall_c[keep].tolist())
            precisions.append(precision_c[keep].tolist())
            keys.append(f"{names[c]} (F1 {f1[c]:.2f}, AP {ap:.2f})")
        counts, flat = mistakes.flatten().topk(self.num_confused_pairs)
        pairs = [
            [
                names[i // num_classes],
                names[i % num_classes],
                int(n),
                round(n.item() / images_per_class[i // num_classes].item(), 4),
            ]
            for n, i in zip(counts, flat.tolist())
            if n > 0
        ]
        logger.experiment.log(
            {
                f"{split}_pr_curve": wandb.plot.line_series(
                    xs=recalls,
                    ys=precisions,
                    keys=keys,
                    title=f"{split}: precision (y) vs. recall, "
                    f"{self.num_pr_classes} worst and best classes by F1",
                    xname="recall",
                ),
                # wandb.plot.confusion_matrix sends all 102 x 102 cells, more
                # than the 10,000 rows a W&B table holds; the empty cells are
                # left out here (they show as blank).
                f"{split}_confusion_matrix": wandb.plot_table(
                    "wandb/confusion_matrix/v1",
                    wandb.Table(
                        columns=["Actual", "Predicted", "nPredictions"],
                        data=[
                            [names[i], names[j], int(confusion[i, j])]
                            for i, j in confusion.nonzero().tolist()
                        ],
                    ),
                    {
                        "Actual": "Actual",
                        "Predicted": "Predicted",
                        "nPredictions": "nPredictions",
                    },
                    {"title": f"{split}: confusion matrix"},
                ),
                f"{split}_most_confused": wandb.Table(
                    columns=["true", "predicted", "count", "share_of_true_class"],
                    data=pairs,
                ),
            }
        )

        worst = ", ".join(f"{names[c]} ({f1[c]:.2f})" for c in order[:5].tolist())
        print(
            f"{split}: accuracy {correct.float().mean():.4f}, "
            f"top-5 {top5.float().mean():.4f}, {int((~correct).sum())} mistakes. "
            f"Lowest F1: {worst}"
        )
