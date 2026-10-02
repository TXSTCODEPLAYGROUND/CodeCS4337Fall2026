"""Train a ConvNet on Fashion-MNIST with PyTorch Lightning, tracked in W&B.

Usage:

.. code-block:: text

    From the repo root:        python -m LitWBTrainingBasicConvnet --config config01.json
    From this folder:          python main.py --config config01.json
    Python or a notebook:      from LitWBTrainingBasicConvnet import main
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

import argparse
import json
import os
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import lightning as L
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from .callbacks import LogTestPredictions
from .dataloaders import (
    FASHION_MNIST_CLASSES,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    FashionMNISTDataModule,
)
from .models import ConvNet, LitConvNet
from .utils import (
    PROJECT_NAME,
    REPO_DIR,
    append_run_summary,
    resolve_config_path,
    resolve_repo_path,
)


def main(config_path: str | Path | None = None) -> dict[str, float]:
    """Train, then test the best checkpoint, using a JSON config.

    Every metric is sent to `Weights & Biases <https://wandb.ai>`__ (W&B), in
    the project named in the config, together with the config itself, a table
    of test predictions, and a confusion matrix. The W&B API key is read from
    ``WANDB_API_KEY`` in the ``.env`` file at the repo root. Without it, or if
    W&B cannot log in, the run is logged offline in the run folder and can be
    uploaded later with ``wandb sync``.

    Local results go to ``<OUTPUT_DIR>/LitWBTrainingBasicConvnet/<config_name>/<timestamp>/``:
    a copy of the config, ``metrics.csv`` (a local backup of the metrics),
    ``hparams.json``, the best and last checkpoints, and W&B's ``wandb/``
    folder. A row is appended to ``runs_summary.csv`` next to the run folders.

    Args:
        config_path: Config name or path, e.g. ``"config01.json"``. When
            omitted, it is read from the required ``--config`` command-line flag.

    Returns:
        Test metrics of the best checkpoint, e.g. ``test_loss``, ``test_acc``,
        ``test_precision``, ``test_recall``, and the per-class accuracies.
    """
    load_dotenv(REPO_DIR / ".env")
    if config_path is None:
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--config", required=True, help="e.g. config01.json")
        config_path = parser.parse_args().config
    config_path = resolve_config_path(config_path)
    config = json.loads(config_path.read_text())
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["training"]

    L.seed_everything(config["seed"], workers=True)

    data = FashionMNISTDataModule(
        data_dir=str(resolve_repo_path(os.getenv("DATA_DIR", "data"))),
        seed=config["seed"],
        **data_cfg,
    )
    model = LitConvNet(
        net=ConvNet(dropout=model_cfg["dropout"]),
        class_names=list(FASHION_MNIST_CLASSES),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    csv_logger = CSVLogger(
        save_dir=resolve_repo_path(os.getenv("OUTPUT_DIR", "runs")) / PROJECT_NAME,
        name=config_path.stem,
        version=run_id,
    )
    run_dir = Path(csv_logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = _connect_wandb(config, config_path.stem, run_id, run_dir)
    checkpoint = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_epoch{epoch:02d}_valacc{val_acc:.4f}",
        auto_insert_metric_name=False,
        monitor="val_acc",
        mode="max",
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=train_cfg["epochs"],
        accelerator=config["accelerator"],
        logger=[csv_logger, wandb_logger],
        callbacks=[
            checkpoint,
            LogTestPredictions(
                model.hparams.class_names, FASHION_MNIST_MEAN[0], FASHION_MNIST_STD[0]
            ),
        ],
    )
    trainer.fit(model, datamodule=data)
    (test_metrics,) = trainer.test(datamodule=data, ckpt_path="best")
    if not wandb_logger.experiment.offline:
        print(f"W&B run: {wandb_logger.experiment.url}")
    # Close the run, so calling main() again in a notebook starts a new one.
    wandb.finish()

    shutil.copy(config_path, run_dir / "config.json")
    hparams = {"model": dict(model.hparams), "data": dict(data.hparams)}
    (run_dir / "hparams.json").write_text(json.dumps(hparams, indent=2))
    append_run_summary(
        run_dir.parent / "runs_summary.csv",
        {
            "run_id": run_id,
            "epochs": train_cfg["epochs"],
            "best_val_acc": f"{checkpoint.best_model_score.item():.4f}",
            "test_acc": f"{test_metrics['test_acc']:.4f}",
            "test_precision": f"{test_metrics['test_precision']:.4f}",
            "test_recall": f"{test_metrics['test_recall']:.4f}",
            "test_loss": f"{test_metrics['test_loss']:.4f}",
            "lr": train_cfg["lr"],
            "weight_decay": train_cfg["weight_decay"],
            "batch_size": data_cfg["batch_size"],
            "dropout": model_cfg["dropout"],
            "seed": config["seed"],
            "best_checkpoint": Path(checkpoint.best_model_path).name,
        },
    )
    return test_metrics


def _connect_wandb(
    config: dict, config_name: str, run_id: str, run_dir: Path
) -> WandbLogger:
    """Start the W&B run online, or offline when there is no key or no login.

    Args:
        config: The experiment config; ``config["wandb"]["project"]`` names the
            W&B project, and the whole config is saved with the run.
        config_name: Config file name without ``.json``, used as the run group.
        run_id: Timestamp of the run, added to the run name.
        run_dir: Run folder; W&B writes its local files to ``run_dir/wandb``.

    Returns:
        A ``WandbLogger`` whose run has already started, online or offline.
    """

    def start(offline: bool) -> WandbLogger:
        logger = WandbLogger(
            project=config["wandb"]["project"],
            name=f"{config_name}-{run_id}",
            group=config_name,
            save_dir=run_dir,
            config=config,
            # Passed to wandb.init: unlike offline=True, which only sets WANDB_MODE,
            # this still works after a failed online attempt.
            mode="offline" if offline else "online",
        )
        _ = logger.experiment
        return logger

    sync_hint = (
        "logging to W&B offline. Upload the run later with: "
        f"wandb sync {run_dir / 'wandb'}/offline-run-*"
    )
    if not os.getenv("WANDB_API_KEY"):
        warnings.warn(f"WANDB_API_KEY is not in .env: {sync_hint}", stacklevel=3)
        return start(offline=True)
    try:
        return start(offline=False)
    except (wandb.errors.Error, OSError) as error:
        wandb.finish()
        warnings.warn(f"Could not log in to W&B ({error}): {sync_hint}", stacklevel=3)
        return start(offline=True)
