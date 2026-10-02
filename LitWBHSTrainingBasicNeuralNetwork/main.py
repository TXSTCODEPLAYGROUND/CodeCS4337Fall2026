"""Train a fully connected network on Fashion-MNIST with PyTorch Lightning, tracked in W&B.

Usage:

.. code-block:: text

    From the repo root:        python -m LitWBHSTrainingBasicNeuralNetwork --config config01.json
    From this folder:          python main.py --config config01.json
    Python or a notebook:      from LitWBHSTrainingBasicNeuralNetwork import main
                               main("config01.json")

After a hyperparameter search (see :mod:`~LitWBHSTrainingBasicNeuralNetwork.search`),
train the best settings with ``--config search01_best.json``.
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
from datetime import datetime
from pathlib import Path

import lightning as L
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from .callbacks import LogTestPredictions
from .dataloaders import (
    FASHION_MNIST_CLASSES,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    FashionMNISTDataModule,
)
from .models import MLP, LitMLP
from .utils import (
    PROJECT_NAME,
    REPO_DIR,
    append_run_summary,
    connect_wandb,
    resolve_config_path,
    resolve_repo_path,
)


def main(config_path: str | Path | None = None) -> dict[str, float]:
    """Train, then test the best checkpoint, using a JSON config.

    Every metric is sent to `Weights & Biases <https://wandb.ai>`__ (W&B), in
    the project named in the config, together with the config itself, the
    learning rate of every epoch, a table of test predictions, and a confusion
    matrix. The W&B API key is read from ``WANDB_API_KEY`` in the ``.env`` file
    at the repo root. Without it, or if W&B cannot log in, the run is logged
    offline in the run folder and can be uploaded later with ``wandb sync``.

    Local results go to ``<OUTPUT_DIR>/LitWBHSTrainingBasicNeuralNetwork/<config_name>/<timestamp>/``:
    a copy of the config, ``metrics.csv`` (a local backup of the metrics),
    ``hparams.json``, the best and last checkpoints, and W&B's ``wandb/``
    folder. A row is appended to ``runs_summary.csv`` next to the run folders.

    Args:
        config_path: Config name or path, e.g. ``"config01.json"`` or
            ``"search01_best.json"``. When omitted, it is read from the
            required ``--config`` command-line flag.

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
    data, model, num_params = build(config)
    print(f"Model: {model_cfg['hidden_sizes']} hidden units, {num_params:,} parameters")

    run_id = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    csv_logger = CSVLogger(
        save_dir=resolve_repo_path(os.getenv("OUTPUT_DIR", "runs")) / PROJECT_NAME,
        name=config_path.stem,
        version=run_id,
    )
    run_dir = Path(csv_logger.log_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = connect_wandb(
        project=config["wandb"]["project"],
        name=f"{config_path.stem}-{run_id}",
        group=config_path.stem,
        save_dir=run_dir,
        config=config,
    )
    wandb_logger.experiment.config.update({"num_params": num_params})
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
            *training_callbacks(config),
        ],
    )
    trainer.fit(model, datamodule=data)
    (test_metrics,) = trainer.test(datamodule=data, ckpt_path="best")
    if not wandb_logger.experiment.offline:
        print(f"W&B run: {wandb_logger.experiment.url}")
    # Close the run, so calling main() again in a notebook starts a new one.
    wandb.finish()

    shutil.copy(config_path, run_dir / "config.json")
    hparams = {
        "net": model_cfg,
        "model": dict(model.hparams),
        "data": dict(data.hparams),
    }
    (run_dir / "hparams.json").write_text(json.dumps(hparams, indent=2))
    append_run_summary(
        run_dir.parent / "runs_summary.csv",
        {
            "run_id": run_id,
            "epochs": train_cfg["epochs"],
            "epochs_run": trainer.current_epoch,
            "best_val_acc": f"{checkpoint.best_model_score.item():.4f}",
            "test_acc": f"{test_metrics['test_acc']:.4f}",
            "test_precision": f"{test_metrics['test_precision']:.4f}",
            "test_recall": f"{test_metrics['test_recall']:.4f}",
            "test_loss": f"{test_metrics['test_loss']:.4f}",
            "optimizer": train_cfg["optimizer"],
            "lr": train_cfg["lr"],
            "l1": train_cfg["l1"],
            "l2": train_cfg["l2"],
            "scheduler": train_cfg["scheduler"],
            "early_stopping": train_cfg["early_stopping"],
            "batch_size": data_cfg["batch_size"],
            "hidden_sizes": "-".join(map(str, model_cfg["hidden_sizes"])),
            "activation": model_cfg["activation"],
            "dropout": model_cfg["dropout"],
            "num_params": num_params,
            "seed": config["seed"],
            "best_checkpoint": Path(checkpoint.best_model_path).name,
        },
    )
    return test_metrics


def build(config: dict) -> tuple[FashionMNISTDataModule, LitMLP, int]:
    """Create the DataModule and the LightningModule described by a config.

    Used by :func:`main` and by every trial of the hyperparameter search, so
    both train exactly the same way.

    Args:
        config: A training config, with ``"seed"``, ``"data"``, ``"model"``,
            and ``"training"`` sections.

    Returns:
        The DataModule, the LightningModule, and the network's number of
        parameters.
    """
    model_cfg, train_cfg = config["model"], config["training"]
    data = FashionMNISTDataModule(
        data_dir=str(resolve_repo_path(os.getenv("DATA_DIR", "data"))),
        seed=config["seed"],
        **config["data"],
    )
    net = MLP(
        hidden_sizes=model_cfg["hidden_sizes"],
        activation=model_cfg["activation"],
        dropout=model_cfg["dropout"],
    )
    num_params = sum(p.numel() for p in net.parameters())
    model = LitMLP(
        net=net,
        class_names=list(FASHION_MNIST_CLASSES),
        optimizer=train_cfg["optimizer"],
        lr=train_cfg["lr"],
        momentum=train_cfg["momentum"],
        l1=train_cfg["l1"],
        l2=train_cfg["l2"],
        scheduler=train_cfg["scheduler"],
        step_size=train_cfg["step_size"],
        gamma=train_cfg["gamma"],
    )
    return data, model, num_params


def training_callbacks(config: dict, log_lr: bool = True) -> list[L.Callback]:
    """Callbacks set by the config's ``"training"`` section.

    Args:
        config: A training config.
        log_lr: Add a ``LearningRateMonitor``, which logs the learning rate
            every epoch (as ``lr-<Optimizer>``) to see what the scheduler does.
            It needs a logger.

    Returns:
        The learning-rate monitor, plus ``EarlyStopping`` when
        ``"early_stopping"`` is true: training stops when ``val_acc`` has not
        improved for ``"patience"`` epochs.
    """
    train_cfg = config["training"]
    callbacks: list[L.Callback] = []
    if log_lr:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    if train_cfg["early_stopping"]:
        callbacks.append(
            EarlyStopping(monitor="val_acc", mode="max", patience=train_cfg["patience"])
        )
    return callbacks
