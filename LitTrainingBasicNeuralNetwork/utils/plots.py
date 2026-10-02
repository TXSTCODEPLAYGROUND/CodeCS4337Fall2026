"""Plots saved in every run folder: learning curves and example predictions."""

import csv
from pathlib import Path

import torch
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from torchvision.datasets import FashionMNIST


def read_epoch_metrics(metrics_csv: Path) -> dict[int, dict[str, float]]:
    """Read ``metrics.csv`` (written by ``CSVLogger``) into ``{epoch: {name: value}}``.

    The logger writes the training, validation, and test values of an epoch on
    separate rows; they are merged here.

    Args:
        metrics_csv: Path to the run's ``metrics.csv``.

    Returns:
        For each epoch (counted from 0), the metric names mapped to their values,
        e.g. ``{0: {"train_loss": 0.47, "val_acc": 0.89, ...}, ...}``.
    """
    by_epoch: dict[int, dict[str, float]] = {}
    with open(metrics_csv, newline="") as f:
        for row in csv.DictReader(f):
            values = by_epoch.setdefault(int(row["epoch"]), {})
            values.update(
                {
                    k: float(v)
                    for k, v in row.items()
                    if v and k not in ("epoch", "step")
                }
            )
    return by_epoch


def plot_train_vs_val(
    by_epoch: dict[int, dict[str, float]], metric: str, label: str, path: Path
) -> None:
    """Plot ``train_<metric>`` and ``val_<metric>`` against the epoch on one graph.

    Args:
        by_epoch: Metrics per epoch, from :func:`read_epoch_metrics`.
        metric: Metric name without the stage prefix, e.g. ``"loss"`` or ``"acc"``.
        label: Name shown in the title and on the y-axis, e.g. ``"accuracy"``.
        path: Where to save the PNG.
    """
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    for stage, style in (("train", "o-"), ("val", "s--")):
        ax.plot(*_series(by_epoch, f"{stage}_{metric}"), style, label=stage)
    _finish(ax, f"Training vs. validation {label}", label)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")


def plot_class_accuracy(
    by_epoch: dict[int, dict[str, float]],
    class_names: list[str],
    metric_class_names: list[str],
    path: Path,
) -> None:
    """Plot the accuracy of every class against the epoch on one graph.

    Each class has its own color; training is a solid line and validation a
    dashed line.

    Args:
        by_epoch: Metrics per epoch, from :func:`read_epoch_metrics`.
        class_names: One name per class, in label order, shown in the legend.
        metric_class_names: The same names as used in the metric keys
            (``train_acc_<name>``, ``val_acc_<name>``).
        path: Where to save the PNG.
    """
    fig = Figure(figsize=(9, 5))
    ax = fig.subplots()
    colors = colormaps["tab10"]
    for i, (name, key) in enumerate(zip(class_names, metric_class_names)):
        color = colors(i % 10)
        ax.plot(*_series(by_epoch, f"train_acc_{key}"), "-", color=color, label=name)
        ax.plot(*_series(by_epoch, f"val_acc_{key}"), "--", color=color)
    ax.plot([], [], "k-", label="train (solid)")
    ax.plot([], [], "k--", label="val (dashed)")
    _finish(ax, "Accuracy per class", "accuracy")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")


def plot_predictions(
    images: torch.Tensor,
    labels: torch.Tensor,
    preds: torch.Tensor,
    class_names: list[str],
    mean: float,
    std: float,
    title: str,
    path: Path,
) -> None:
    """Show images in a grid, each titled with its predicted and true class.

    Correct predictions are titled in green, wrong ones in red.

    Args:
        images: Normalized images of shape ``(N, 1, H, W)``.
        labels: True classes, shape ``(N,)``.
        preds: Predicted classes, shape ``(N,)``.
        class_names: One name per class, in label order.
        mean: Mean used to normalize the images (undone for display).
        std: Standard deviation used to normalize the images.
        title: Title of the figure.
        path: Where to save the PNG.
    """
    cols = 8
    rows = max(1, -(-len(images) // cols))
    fig = Figure(figsize=(cols * 1.6, rows * 1.9))
    axes = fig.subplots(rows, cols, squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, label, pred in zip(axes.flat, images, labels, preds):
        ax.imshow(image.squeeze() * std + mean, cmap="gray", vmin=0, vmax=1)
        ax.set_title(
            f"pred: {class_names[pred]}\ntrue: {class_names[label]}",
            fontsize=7,
            color="green" if pred == label else "red",
        )
    fig.suptitle(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")


def save_run_plots(
    run_dir: Path,
    class_names: list[str],
    metric_class_names: list[str],
    test_set: FashionMNIST,
    preds: torch.Tensor,
    mean: float,
    std: float,
    num_images: int = 16,
) -> list[Path]:
    """Save all plots of a finished run in ``run_dir``.

    Writes ``loss.png``, ``accuracy.png``, ``accuracy_per_class.png``,
    ``predictions.png``, and ``wrong_predictions.png``.

    Args:
        run_dir: Run folder containing ``metrics.csv``.
        class_names: One name per class, in label order.
        metric_class_names: The same names as used in the metric keys
            (see :class:`~LitTrainingBasicNeuralNetwork.models.LitMLP`).
        test_set: Test dataset, in the same order as ``preds``.
        preds: Predicted class of every test image.
        mean: Mean used to normalize the images.
        std: Standard deviation used to normalize the images.
        num_images: Number of images in each prediction grid.

    Returns:
        Paths of the PNG files written. ``wrong_predictions.png`` is skipped
        if every test image was classified correctly.
    """
    by_epoch = read_epoch_metrics(run_dir / "metrics.csv")
    paths = {
        name: run_dir / f"{name}.png"
        for name in (
            "loss",
            "accuracy",
            "accuracy_per_class",
            "predictions",
            "wrong_predictions",
        )
    }
    plot_train_vs_val(by_epoch, "loss", "loss", paths["loss"])
    plot_train_vs_val(by_epoch, "acc", "accuracy", paths["accuracy"])
    plot_class_accuracy(
        by_epoch, class_names, metric_class_names, paths["accuracy_per_class"]
    )

    labels = torch.as_tensor(test_set.targets)
    wrong = (preds != labels).nonzero().flatten()
    grids = (
        ("predictions", "Example test predictions", torch.arange(num_images)),
        ("wrong_predictions", "Wrong test predictions", wrong[:num_images]),
    )
    for name, title, indices in grids:
        if len(indices) == 0:
            continue
        images = torch.stack([test_set[int(i)][0] for i in indices])
        plot_predictions(
            images,
            labels[indices],
            preds[indices],
            class_names,
            mean,
            std,
            title,
            paths[name],
        )
    return [path for path in paths.values() if path.exists()]


def _series(
    by_epoch: dict[int, dict[str, float]], key: str
) -> tuple[list[int], list[float]]:
    """Return the epochs that have ``key`` and its value at each of them."""
    epochs = [e for e in sorted(by_epoch) if key in by_epoch[e]]
    return epochs, [by_epoch[e][key] for e in epochs]


def _finish(ax: Axes, title: str, ylabel: str) -> None:
    """Add the title, axis labels, whole-number epoch ticks, and a light grid."""
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
