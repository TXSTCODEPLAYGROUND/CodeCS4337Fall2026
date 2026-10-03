"""TensorBoard logging for one run, besides the per-epoch curves logged by the trainer.

Every run writes its TensorBoard event files into its own run folder, so
``tensorboard --logdir <OUTPUT_DIR>/<project_name>`` shows all runs of the
project, named ``<config>/<timestamp>``, side by side.
"""

import json
from pathlib import Path
from typing import Any

import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from ..dataloaders.fashion_mnist import FASHION_MNIST_MEAN, FASHION_MNIST_STD


def create_writer(run_dir: Path) -> SummaryWriter:
    """Create the TensorBoard writer of a run.

    Args:
        run_dir: The run folder; the event files are written into it.

    Returns:
        A ``SummaryWriter``. Close it at the end of the run.
    """
    return SummaryWriter(log_dir=str(run_dir))


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo the dataset normalization so images display with their real brightness."""
    return (images * FASHION_MNIST_STD[0] + FASHION_MNIST_MEAN[0]).clamp(0, 1)


def log_config(writer: SummaryWriter, config: dict[str, Any]) -> None:
    """Show the run's config in the *Text* tab.

    Args:
        writer: The run's writer.
        config: The parsed config.
    """
    # Indented lines render as a code block in TensorBoard's Markdown.
    text = "\n".join(
        "    " + line for line in json.dumps(config, indent=2).splitlines()
    )
    writer.add_text("config", text)


def log_sample_images(writer: SummaryWriter, loader: DataLoader, n: int = 32) -> None:
    """Show a grid of input images in the *Images* tab: what the network sees.

    Args:
        writer: The run's writer.
        loader: A dataloader that does not shuffle (drawing from a shuffling
            one would change the training order).
        n: Number of images in the grid.
    """
    images, _ = next(iter(loader))
    writer.add_image("data/samples", make_grid(_denormalize(images[:n]), nrow=8))


def log_model_graph(
    writer: SummaryWriter, model: nn.Module, loader: DataLoader, device: torch.device
) -> None:
    """Show the network's layers and tensor shapes in the *Graphs* tab.

    Args:
        writer: The run's writer.
        model: The network, already on ``device``.
        loader: Any dataloader; one image is used to trace the network.
        device: Device of the model.
    """
    images, _ = next(iter(loader))
    writer.add_graph(model, images[:1].to(device))


@torch.no_grad()
def log_test_predictions(
    writer: SummaryWriter,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    step: int,
    n: int = 16,
) -> None:
    """Show test predictions and a confusion matrix in the *Images* tab.

    Args:
        writer: The run's writer.
        model: The trained network, on ``device``.
        loader: The test dataloader. Its dataset must have a ``classes`` list.
        device: Device of the model.
        step: Step (epoch) the figures are logged at.
        n: Number of example images.
    """
    model.eval()
    classes = loader.dataset.classes
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(dim=1).cpu())
        labels.append(y)
    preds, labels = torch.cat(preds), torch.cat(labels)

    images = _denormalize(torch.stack([loader.dataset[i][0] for i in range(n)]))
    fig = Figure(figsize=(8, 8.5))
    for i, ax in enumerate(fig.subplots(4, n // 4).flat):
        ax.imshow(images[i, 0], cmap="gray")
        ax.set_title(
            f"{classes[preds[i]]}\n(true: {classes[labels[i]]})",
            fontsize=8,
            color="green" if preds[i] == labels[i] else "red",
        )
        ax.axis("off")
    fig.tight_layout()
    writer.add_figure("test/predictions", fig, step)

    matrix = torch.zeros(len(classes), len(classes), dtype=torch.long)
    for t, p in zip(labels, preds):
        matrix[t, p] += 1
    fig = Figure(figsize=(7, 6))
    ax = fig.subplots()
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for t in range(len(classes)):
        for p in range(len(classes)):
            color = "white" if matrix[t, p] > matrix.max() / 2 else "black"
            ax.text(
                p,
                t,
                int(matrix[t, p]),
                ha="center",
                va="center",
                fontsize=7,
                color=color,
            )
    fig.tight_layout()
    writer.add_figure("test/confusion_matrix", fig, step)


def log_hparams(
    writer: SummaryWriter, hparams: dict[str, Any], metrics: dict[str, float]
) -> None:
    """Add the run to the *HParams* tab: its settings next to its final scores.

    Args:
        writer: The run's writer.
        hparams: Settings of the run, e.g. ``{"lr": 0.001, ...}``. Values must
            be numbers, strings, or booleans.
        metrics: Final scores, e.g. ``{"test_acc": 0.89}``. They are logged as
            ``hparam/<name>``.
    """
    # run_name="." keeps the HParams entry in the run's own folder, so it is
    # the same run as the curves instead of a separate timestamped one.
    writer.add_hparams(
        hparams, {f"hparam/{k}": v for k, v in metrics.items()}, run_name="."
    )
