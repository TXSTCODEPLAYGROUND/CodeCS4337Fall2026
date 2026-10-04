"""W&B logging for Ultralytics YOLO: training curves and the final evaluation report.

Ultralytics has its own trainer, so there are no Lightning callbacks here.
Instead, :func:`epoch_logger` is registered with Ultralytics'
``model.add_callback("on_fit_epoch_end", ...)``, and the ``log_...``
functions are called by :func:`~YoloExample.main.main` after training.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import wandb
from PIL import Image
from torchvision.ops import box_iou

from ..dataloaders.pennfudan import read_yolo_labels, yolo_to_box

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
"""The 10 IoU thresholds of mAP50-95: 0.50, 0.55, ..., 0.95."""
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
    (``lr/pg0`` ...).

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
        "instances": int(np.sum(metrics.nt_per_class)),
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


def log_ap_per_iou(run: Any, ap_per_iou: dict[str, np.ndarray]) -> None:
    """Plot AP at each IoU threshold, from 0.50 to 0.95, one line per split.

    mAP50-95 is the mean of this curve: it shows how fast the score drops when
    the boxes must fit more tightly.

    Args:
        run: The W&B run.
        ap_per_iou: Split name mapped to its 10 AP values (mean over classes).
    """
    run.log(
        {
            "ap_per_iou_threshold": wandb.plot.line_series(
                xs=IOU_THRESHOLDS,
                ys=[list(map(float, ap)) for ap in ap_per_iou.values()],
                keys=list(ap_per_iou),
                title="AP at each IoU threshold (mAP50-95 is the mean)",
                xname="IoU threshold",
            ),
            "ap_per_iou_table": wandb.Table(
                columns=["iou_threshold", *ap_per_iou],
                data=[
                    [t, *(round(float(ap[i]), 4) for ap in ap_per_iou.values())]
                    for i, t in enumerate(IOU_THRESHOLDS)
                ],
            ),
        }
    )


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
        ``(N,)`` booleans, ``True`` for the predictions that found a person.
    """
    hits = torch.zeros(len(pred_boxes), dtype=torch.bool)
    if len(pred_boxes) == 0 or len(true_boxes) == 0:
        return hits
    iou = box_iou(pred_boxes, true_boxes)
    taken = torch.zeros(len(true_boxes), dtype=torch.bool)
    for i in pred_conf.argsort(descending=True).tolist():
        overlaps = iou[i].masked_fill(taken, -1)
        best = int(overlaps.argmax())
        if overlaps[best] >= iou_threshold:
            hits[i] = taken[best] = True
    return hits


def log_predictions(
    run: Any,
    model: Any,
    dataset_dir: Path,
    split: str,
    *,
    classes: list[int] | None,
    imgsz: int,
    conf: float = 0.25,
    iou_threshold: float = 0.5,
    num_hardest: int = 8,
) -> None:
    """Log every image of a split with its predicted and true boxes.

    Logged:

    - ``<split>_predictions``: a table with one row per image: the image with
      both sets of boxes (toggle them in W&B, and filter the predictions by
      confidence), the number of true persons, of predictions, of persons
      found, of false alarms, and of missed persons.
    - ``<split>_hardest_examples``: the ``num_hardest`` images with the most
      false alarms plus missed persons.

    Args:
        run: The W&B run.
        model: The Ultralytics ``YOLO`` model.
        dataset_dir: The converted dataset, with ``images/<split>`` and ``labels/<split>``.
        split: ``"val"`` or ``"test"``.
        classes: Class indices to keep (``[0]``, person, for the COCO model), or ``None``.
        imgsz: Image size for the model.
        conf: Minimum confidence of the predictions shown and counted.
        iou_threshold: Minimum IoU for a prediction to count as finding a person.
        num_hardest: Number of images in the hardest-examples gallery.
    """
    images = sorted((dataset_dir / "images" / split).glob("*.png"))
    results = model.predict(
        [str(p) for p in images], conf=conf, classes=classes, imgsz=imgsz, verbose=False
    )
    rows, scored = [], []
    for path, result in zip(images, results):
        image = np.array(Image.open(path).convert("RGB"))
        height, width = image.shape[:2]
        labels = read_yolo_labels(dataset_dir / "labels" / split / f"{path.stem}.txt")
        true_boxes = torch.tensor(
            [yolo_to_box(box, width, height) for _, box in labels]
        ).reshape(-1, 4)
        pred_boxes = result.boxes.xyxy.cpu()
        pred_conf = result.boxes.conf.cpu()
        hits = match_detections(pred_boxes, pred_conf, true_boxes, iou_threshold)
        found = int(hits.sum())
        false_alarms, missed = len(hits) - found, len(true_boxes) - found

        def box_data(boxes: torch.Tensor, scores: torch.Tensor | None) -> list[dict]:
            return [
                {
                    "position": dict(zip(("minX", "minY", "maxX", "maxY"), b.tolist())),
                    "domain": "pixel",
                    "class_id": 0,
                    "box_caption": "person" if scores is None else f"{scores[i]:.2f}",
                    **(
                        {}
                        if scores is None
                        else {"scores": {"confidence": float(scores[i])}}
                    ),
                }
                for i, b in enumerate(boxes)
            ]

        picture = wandb.Image(
            image,
            boxes={
                "predictions": {
                    "box_data": box_data(pred_boxes, pred_conf),
                    "class_labels": {0: "person"},
                },
                "ground_truth": {
                    "box_data": box_data(true_boxes, None),
                    "class_labels": {0: "person"},
                },
            },
            caption=f"{path.stem}: {found} found, {missed} missed, {false_alarms} false alarms",
        )
        rows.append(
            [
                picture,
                path.stem,
                len(true_boxes),
                len(hits),
                found,
                false_alarms,
                missed,
            ]
        )
        scored.append((false_alarms + missed, picture))

    run.log(
        {
            f"{split}_predictions": wandb.Table(
                columns=[
                    "image",
                    "file",
                    "persons",
                    "predicted",
                    "found",
                    "false_alarms",
                    "missed",
                ],
                data=rows,
            ),
            f"{split}_hardest_examples": [
                picture
                for _, picture in sorted(scored, key=lambda s: -s[0])[:num_hardest]
            ],
        }
    )
