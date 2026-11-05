"""Train a fully connected network on Fashion-MNIST with PyTorch Lightning, tracked in W&B.

Usage:

.. code-block:: text

    From the repo root:        python -m LitWBTrainingBasicNeuralNetwork --config config01.json
    From this folder:          python main.py --config config01.json
    Python or a notebook:      from LitWBTrainingBasicNeuralNetwork import main
                               main("config01.json")
"""

if __name__ == "__main__" and not __package__:
    # Run as `python main.py`: relative imports need the package, so rerun as `python -m`.
    import runpy
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module(
        Path(__file__).resolve().parent.name, run_name="__main__", alter_sys=True
    )
    sys.exit()

from pathlib import Path

import colabutilscs4337 as cu
import lightning as L
import wandb

from .callbacks import LogTestPredictions
from .dataloaders import (
    FASHION_MNIST_CLASSES,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    FashionMNISTDataModule,
)
from .models import MLP, LitMLP
from .utils import PROJECT


def main(
    config_path: str | Path | None = None, resume_from: str | Path | None = None
) -> dict[str, float]:
    """Train, then test the best checkpoint, using a JSON config.

    Every metric is sent to `Weights & Biases <https://wandb.ai>`__ (W&B), in
    the project named in the config, together with the config itself, a table
    of test predictions, and a confusion matrix. The W&B API key is read from
    ``WANDB_API_KEY`` in the ``.env`` file at the repo root. Without it, or if
    W&B cannot log in, the run is logged offline in the run folder and can be
    uploaded later with ``wandb sync``.

    Local results go to ``<OUTPUT_DIR>/LitWBTrainingBasicNeuralNetwork/<config_name>/<timestamp>/``:
    a copy of the config, ``metrics.csv`` (a local backup of the metrics),
    ``hparams.json``, the best and last checkpoints, and W&B's ``wandb/``
    folder. A row is appended to ``runs_summary.csv`` next to the run folders.

    Args:
        config_path: Config name or path, e.g. ``"config01.json"``. When
            omitted, it is read from the required ``--config`` command-line flag.
        resume_from: Continue an earlier run instead of starting from
            scratch: ``"config01"`` (the newest run of config01), a run folder,
            or a checkpoint, found by
            :meth:`~colabutilscs4337.project.Project.find_resume_checkpoint`. The
            weights, the optimizer (including its learning rate), and the epoch
            count come from the run's last checkpoint; the config's ``"epochs"``
            more epochs are then trained into a new run folder. The model
            settings in the config must match the run's. On the command line:
            ``--resume-from config01``.

    Returns:
        Test metrics of the best checkpoint, e.g. ``test_loss``, ``test_acc``,
        ``test_precision``, ``test_recall``, and the per-class accuracies.
    """
    run = cu.lit.start_run(PROJECT, config_path, resume_from, description=__doc__)
    config = run.config
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["training"]

    L.seed_everything(config["seed"], workers=True)

    data = FashionMNISTDataModule(
        data_dir=str(PROJECT.data_dir()), seed=config["seed"], **data_cfg
    )
    net = MLP(hidden_sizes=model_cfg["hidden_sizes"], dropout=model_cfg["dropout"])
    num_params = sum(p.numel() for p in net.parameters())
    print(f"Model: {model_cfg['hidden_sizes']} hidden units, {num_params:,} parameters")
    model = LitMLP(
        net=net,
        class_names=list(FASHION_MNIST_CLASSES),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    wandb_logger = cu.lit.connect_wandb(
        project=config["wandb"]["project"],
        name=f"{run.config_name}-{run.run_id}",
        group=run.config_name,
        save_dir=run.run_dir,
        config=config,
    )
    wandb_logger.experiment.config.update({"num_params": num_params})

    trainer = L.Trainer(
        max_epochs=run.epochs,
        accelerator=config["accelerator"],
        logger=[run.csv_logger, wandb_logger],
        callbacks=[
            run.checkpoint,
            run.last_checkpoint,
            LogTestPredictions(
                model.hparams.class_names, FASHION_MNIST_MEAN[0], FASHION_MNIST_STD[0]
            ),
        ],
    )
    trainer.fit(model, datamodule=data, ckpt_path=run.resume_checkpoint)
    (test_metrics,) = trainer.test(datamodule=data, ckpt_path="best")
    if not wandb_logger.experiment.offline:
        print(f"W&B run: {wandb_logger.experiment.url}")
    # Close the run, so calling main() again in a notebook starts a new one.
    wandb.finish()

    run.save(
        {"net": model_cfg, "model": dict(model.hparams), "data": dict(data.hparams)}
    )
    run.append_summary(
        {
            "run_id": run.run_id,
            "epochs": train_cfg["epochs"],
            "best_val_acc": f"{run.checkpoint.best_model_score.item():.4f}",
            "test_acc": f"{test_metrics['test_acc']:.4f}",
            "test_precision": f"{test_metrics['test_precision']:.4f}",
            "test_recall": f"{test_metrics['test_recall']:.4f}",
            "test_loss": f"{test_metrics['test_loss']:.4f}",
            "lr": train_cfg["lr"],
            "weight_decay": train_cfg["weight_decay"],
            "batch_size": data_cfg["batch_size"],
            "hidden_sizes": "-".join(map(str, model_cfg["hidden_sizes"])),
            "dropout": model_cfg["dropout"],
            "num_params": num_params,
            "seed": config["seed"],
            "best_checkpoint": Path(run.checkpoint.best_model_path).name,
        }
    )
    return test_metrics
