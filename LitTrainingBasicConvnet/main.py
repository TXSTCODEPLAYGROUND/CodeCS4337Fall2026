"""Train a ConvNet on Fashion-MNIST with PyTorch Lightning.

Run from the repo root:

.. code-block:: text

    Terminal:                  python -m LitTrainingBasicConvnet --config config01.json
    Python or a notebook:      from LitTrainingBasicConvnet import main
                               main("config01.json")
"""

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import lightning as L
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .data import FashionMNISTDataModule
from .model import LitConvNet

PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent


def resolve_repo_path(path: str | Path) -> Path:
    """Return ``path`` unchanged if absolute, otherwise relative to the repo root."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_DIR / path


def resolve_config_path(config: str | Path) -> Path:
    """Find a config given as ``config01``, ``config01.json``, or ``configs/config01.json``.

    Raises:
        FileNotFoundError: If no matching file exists.
    """
    config = Path(config).expanduser()
    for base in (config, PROJECT_DIR / config, PROJECT_DIR / "configs" / config):
        for candidate in (base, base.with_suffix(".json")):
            if candidate.is_file():
                return candidate.resolve()
    available = sorted(p.name for p in (PROJECT_DIR / "configs").glob("*.json"))
    raise FileNotFoundError(f"Config {str(config)!r} not found. Available: {available}")


def main(config_path: str | Path | None = None) -> dict[str, float]:
    """Train, then test the best checkpoint, using a JSON config.

    Results go to ``<OUTPUT_DIR>/LitTrainingBasicConvnet/<config_name>/<timestamp>/``:
    a copy of the config, ``metrics.csv`` (per-epoch metrics), ``hparams.yaml``,
    and the best and last checkpoints. A row is appended to ``runs_summary.csv``
    next to the run folders. ``DATA_DIR`` and ``OUTPUT_DIR`` come from the
    ``.env`` file at the repo root.

    Args:
        config_path: Config name or path, e.g. ``"config01.json"``. When
            omitted, it is read from the required ``--config`` command-line flag.

    Returns:
        Test metrics of the best checkpoint: ``test_loss`` and ``test_acc``.
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
        dropout=model_cfg["dropout"],
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Logs land in <OUTPUT_DIR>/<project>/<config_name>/<timestamp>/.
    experiment_dir = (
        resolve_repo_path(os.getenv("OUTPUT_DIR", "runs")) / PROJECT_DIR.name
    )
    run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    logger = CSVLogger(save_dir=experiment_dir, name=config_path.stem, version=run_id)
    run_dir = Path(logger.log_dir)
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
        logger=logger,
        callbacks=[checkpoint],
    )
    trainer.fit(model, datamodule=data)
    (test_metrics,) = trainer.test(datamodule=data, ckpt_path="best")

    shutil.copy(config_path, run_dir / "config.json")
    summary_path = run_dir.parent / "runs_summary.csv"
    row = {
        "run_id": run_id,
        "epochs": train_cfg["epochs"],
        "best_val_acc": f"{checkpoint.best_model_score.item():.4f}",
        "test_acc": f"{test_metrics['test_acc']:.4f}",
        "test_loss": f"{test_metrics['test_loss']:.4f}",
        "lr": train_cfg["lr"],
        "weight_decay": train_cfg["weight_decay"],
        "batch_size": data_cfg["batch_size"],
        "dropout": model_cfg["dropout"],
        "seed": config["seed"],
        "best_checkpoint": Path(checkpoint.best_model_path).name,
    }
    is_new = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return test_metrics


if __name__ == "__main__":
    main()
