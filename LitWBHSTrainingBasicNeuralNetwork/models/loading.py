"""Load a trained model back from a run folder."""

from pathlib import Path

from ..utils import PROJECT
from .components import MLP
from .lit_mlp import LitMLP


def find_checkpoint(run: str | Path, which: str = "best") -> Path:
    """Locate a checkpoint saved by :func:`~LitWBHSTrainingBasicNeuralNetwork.main.main`.

    ``run`` can be a checkpoint file, a run folder
    (``<OUTPUT_DIR>/LitWBHSTrainingBasicNeuralNetwork/search01_best/<timestamp>``), or a
    config's folder (``.../search01_best``), which picks its newest run. ``"search01_best"``
    and ``"search01_best/2026-10-02_20-23-38"`` both work. This is
    :meth:`colabutilscs4337.project.Project.find_checkpoint` for this project.

    Args:
        run: Checkpoint file, run folder, or config folder.
        which: ``"best"`` (best validation accuracy) or ``"last"`` (last epoch).

    Returns:
        The checkpoint path.
    """
    return PROJECT.find_checkpoint(run, which)


def load_model(run: str | Path, which: str = "best", device: str = "cpu") -> LitMLP:
    """Rebuild a trained model from its run, ready for predictions.

    The network is rebuilt from the run's config, so you don't need to remember
    its ``hidden_sizes``, ``activation``, or ``dropout``. The model is returned in eval mode.

    Example:

    .. code-block:: python

        from LitWBHSTrainingBasicNeuralNetwork import load_model

        model = load_model("search01_best")   # newest run of search01_best, best checkpoint
        logits = model(images)                # images: (N, 1, 28, 28), normalized

    Args:
        run: Checkpoint file, run folder, or config folder; see :func:`find_checkpoint`.
        which: ``"best"`` or ``"last"`` checkpoint.
        device: Where to put the weights, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        The trained ``LitMLP``. It also works with ``trainer.test(model, datamodule=data)``.
    """
    checkpoint = find_checkpoint(run, which)
    model_cfg = PROJECT.run_config(checkpoint)["model"]
    net = MLP(
        hidden_sizes=model_cfg["hidden_sizes"],
        activation=model_cfg["activation"],
        dropout=model_cfg["dropout"],
    )
    model = LitMLP.load_from_checkpoint(checkpoint, net=net, map_location=device)
    model.eval()
    print(f"Loaded {checkpoint}")
    return model
