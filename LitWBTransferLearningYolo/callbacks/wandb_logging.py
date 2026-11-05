"""W&B logging for Ultralytics YOLO: training curves and the final evaluation report.

Ultralytics has its own trainer, so there are no Lightning callbacks here.
Instead, :func:`epoch_logger` is registered with Ultralytics'
``model.add_callback("on_fit_epoch_end", ...)``, and the ``log_...``
functions are called by :func:`~LitWBTransferLearningYolo.main.main` after training.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import wandb
from PIL import Image
from torchvision.ops import box_iou
from ultralytics.utils.torch_utils import get_num_gradients

from ..dataloaders.pothole import image_paths, read_yolo_labels, yolo_to_box

METRIC_NAMES = {
    "metrics/precision(B)": "precision",
    "metrics/recall(B)": "recall",
    "metrics/mAP50(B)": "mAP50",
    "metrics/mAP50-95(B)": "mAP50-95",
}
"""Ultralytics' metric keys (``(B)`` = boxes) and the shorter names used in W&B."""
PLOTS = {
    "BoxPR_curve.png": "pr_curve",
    "BoxF1_curve.png": "f1_curve",
    "BoxP_curve.png": "precision_curve",
    "BoxR_curve.png": "recall_curve",
    "confusion_matrix.png": "confusion_matrix",
}
"""Plots saved by Ultralytics' validation, and their names in W&B."""


def epoch_logger(run: Any) -> Callable[[Any], None]:
    """Build the callback that logs every training epoch to W&B.

    Logged per epoch: ``train/box_loss``, ``train/cls_loss``,
    ``train/dfl_loss``, the same three losses on the validation set
    (``val/...``), ``val/precision``, ``val/recall``, ``val/mAP50``,
    ``val/mAP50-95``, and the learning rate of each parameter group
    (``lr/pg0`` ...). The first epoch also stores ``trainable_params``,
    the weights that are not frozen.

    Args:
        run: The W&B run.

    Returns:
        A function to register with ``model.add_callback("on_fit_epoch_end", ...)``.
    """
    logged: set[int] = set()

    def on_fit_epoch_end(trainer: Any) -> None:
        metrics = dict(trainer.metrics)
        # The hook runs once more after training, for the final validation of
        # best.pt; that call has no validation losses and is skipped.
        if "val/box_loss" not in metrics or trainer.epoch in logged:
            return
        logged.add(trainer.epoch)
        if trainer.epoch == 0:
            run.summary["trainable_params"] = get_num_gradients(trainer.model)
        row: dict[str, float] = {"epoch": trainer.epoch + 1}
        row.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
        for key, value in metrics.items():
            row[f"val/{METRIC_NAMES[key]}" if key in METRIC_NAMES else key] = value
        row.update(trainer.lr)
        run.log(row)

    return on_fit_epoch_end


def evaluation_metrics(metrics: Any, num_images: int) -> dict[str, float]:
    """Pick the key numbers from an Ultralytics validation result.

    Args:
        metrics: What ``model.val(...)`` returned (``DetMetrics``).
        num_images: Number of images in the evaluated split.

    Returns:
        ``images``, ``instances``, ``precision``, ``recall``, ``f1``,
        ``mAP50``, ``mAP75``, ``mAP50-95``, and ``inference_ms`` (the model's
        time per image, without pre- and post-processing).
    """
    box = metrics.box
    precision, recall = float(box.mp), float(box.mr)
    return {
        "images": num_images,
        "instances": int(sum(metrics.nt_per_class)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "mAP50": float(box.map50),
        "mAP75": float(box.map75),
        "mAP50-95": float(box.map),
        "inference_ms": float(metrics.speed["inference"]),
    }


def log_evaluation(
    run: Any, split: str, metrics: Any, num_images: int
) -> dict[str, float]:
    """Log one split's metrics and Ultralytics' validation plots to W&B.

    The numbers go to the run summary as ``<split>/mAP50-95`` etc.; the PR,
    F1, precision, and recall curves and the confusion matrix as images
    ``<split>/pr_curve`` etc.

    Args:
        run: The W&B run.
        split: ``"val"`` or ``"test"``.
        metrics: What ``model.val(...)`` returned; its ``save_dir`` holds the plots.
        num_images: Number of images in the split.

    Returns:
        The split's metrics, see :func:`evaluation_metrics`.
    """
    values = evaluation_metrics(metrics, num_images)
    for name, value in values.items():
        run.summary[f"{split}/{name}"] = value
    plots = {
        f"{split}/{name}": wandb.Image(str(Path(metrics.save_dir) / file))
        for file, name in PLOTS.items()
        if (Path(metrics.save_dir) / file).is_file()
    }
    if plots:
        run.log(plots)
    return values


def match_detections(
    pred_boxes: torch.Tensor,
    pred_conf: torch.Tensor,
    true_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """Mark each prediction as a true positive or a false positive.

    Predictions are taken from the most to the least confident; each one is
    matched to the unmatched true box it overlaps most, if their IoU reaches
    ``iou_threshold``. This is the matching behind precision, recall, and AP.

    Args:
        pred_boxes: ``(N, 4)`` predicted boxes, ``(x_min, y_min, x_max, y_max)``.
        pred_conf: ``(N,)`` confidences.
        true_boxes: ``(M, 4)`` true boxes.
        iou_threshold: Minimum IoU for a match.

    Returns:
        ``(N,)`` booleans, ``True`` for the predictions that found a pothole.
    """
    hits = torch.zeros(len(pred_boxes), dtype=torch.bool)
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return hits
    iou = box_iou(pred_boxes, true_boxes)
    taken = torch.zeros(len(true_boxes), dtype=torch.bool)
    for index in pred_conf.argsort(descending=True).tolist():
        overlaps = iou[index].masked_fill(taken, -1)
        best = int(overlaps.argmax())
        if overlaps[best] >= iou_threshold:
            hits[index] = taken[best] = True
    return hits


def log_predictions(
    run: Any,
    model: Any,
    dataset_dir: Path,
    split: str,
    *,
    imgsz: int,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
) -> None:
    """Log every image of a split with its predicted and true boxes.

    ``<split>_predictions`` is a table with one row per image: the image with
    both sets of boxes (toggle them in W&B), the number of true potholes, of
    predictions, of potholes found, of false alarms, and of missed potholes.

    Args:
        run: The W&B run.
        model: The Ultralytics ``YOLO`` model.
        dataset_dir: The prepared dataset, with ``images/<split>`` and ``labels/<split>``.
        split: ``"val"`` or ``"test"``.
        imgsz: Image size for the model.
        conf: Minimum confidence of the predictions shown and counted.
        iou_threshold: Minimum IoU for a prediction to count as finding a pothole.
    """
    images = image_paths(dataset_dir, split)
    results = model.predict(
        [str(path) for path in images], conf=conf, imgsz=imgsz, verbose=False
    )
    rows = []
    for path, result in zip(images, results, strict=True):
        image = Image.open(path).convert("RGB")
        width, height = image.size
        labels = read_yolo_labels(dataset_dir / "labels" / split / f"{path.stem}.txt")
        if labels:
            true_boxes = torch.tensor(
                [yolo_to_box(box, width, height) for _, box in labels],
                dtype=torch.float32,
            )
        else:
            true_boxes = torch.zeros((0, 4))
        pred_boxes = result.boxes.xyxy.cpu()
        pred_conf = result.boxes.conf.cpu()
        hits = match_detections(pred_boxes, pred_conf, true_boxes, iou_threshold)
        found = int(hits.sum())
        false_alarms, missed = len(hits) - found, len(true_boxes) - found

        def box_data(boxes: torch.Tensor, scores: torch.Tensor | None) -> list[dict]:
            return [
                {
                    "position": dict(
                        zip(
                            ("minX", "minY", "maxX", "maxY"),
                            corners.tolist(),
                            strict=True,
                        )
                    ),
                    "domain": "pixel",
                    "class_id": 0,
                    "box_caption": "pothole" if scores is None else f"{scores[i]:.2f}",
                    **(
                        {}
                        if scores is None
                        else {"scores": {"confidence": float(scores[i])}}
                    ),
                }
                for i, corners in enumerate(boxes)
            ]

        rows.append(
            [
                wandb.Image(
                    image,
                    boxes={
                        "predictions": {
                            "box_data": box_data(pred_boxes, pred_conf),
                            "class_labels": {0: "pothole"},
                        },
                        "ground_truth": {
                            "box_data": box_data(true_boxes, None),
                            "class_labels": {0: "pothole"},
                        },
                    },
                    caption=(
                        f"{path.stem}: {found} found, {missed} missed, "
                        f"{false_alarms} false alarms"
                    ),
                ),
                path.stem,
                len(true_boxes),
                len(hits),
                found,
                false_alarms,
                missed,
            ]
        )
    run.log(
        {
            f"{split}_predictions": wandb.Table(
                columns=[
                    "image",
                    "file",
                    "potholes",
                    "predicted",
                    "found",
                    "false_alarms",
                    "missed",
                ],
                data=rows,
            )
        }
    )
