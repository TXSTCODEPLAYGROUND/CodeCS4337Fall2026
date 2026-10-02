"""Train a fully connected network on Fashion-MNIST with PyTorch Lightning.

Usage:

.. code-block:: text

    From the repo root:        python -m LitTrainingBasicNeuralNetwork --config config01.json
    From this folder:          python main.py --config config01.json
    Python or a notebook:      from LitTrainingBasicNeuralNetwork import main
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
import torch
from dotenv import load_dotenv
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, LitLogger

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
    resolve_config_path,
    resolve_repo_path,
    save_run_plots,
)


def main(config_path: str | Path | None = None) -> dict[str, float]:
    """Train, then test the best checkpoint, using a JSON config.

    Results go to ``<OUTPUT_DIR>/LitTrainingBasicNeuralNetwork/<config_name>/<timestamp>/``:
    a copy of the config, ``metrics.csv`` (per-epoch metrics), ``hparams.json``,
    the best and last checkpoints, and PNG plots (learning curves and example
    predictions). A row is appended to ``runs_summary.csv`` next to the run
    folders. ``DATA_DIR`` and ``OUTPUT_DIR`` come from the ``.env`` file at the
    repo root.

    When the config sets ``"litlogger": true``, metrics and plots are also sent
    to `LitLogger <https://lightning.ai/docs/pytorch/stable/visualize/experiment_managers.html>`__
    on lightning.ai. This needs ``LIGHTNING_USER_ID`` and ``LIGHTNING_API_KEY``
    in ``.env``; if they are missing or it can't connect, the run continues with
    local logs only.

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
    net = MLP(hidden_sizes=model_cfg["hidden_sizes"], dropout=model_cfg["dropout"])
    num_params = sum(p.numel() for p in net.parameters())
    print(f"Model: {model_cfg['hidden_sizes']} hidden units, {num_params:,} parameters")
    model = LitMLP(
        net=net,
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
    lit_logger = None
    if config["litlogger"]:
        lit_logger = _connect_litlogger(run_dir, f"{PROJECT_NAME}-{config_path.stem}")
    loggers = [csv_logger] + ([lit_logger] if lit_logger else [])
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
        logger=loggers,
        callbacks=[checkpoint],
    )
    trainer.fit(model, datamodule=data)
    (test_metrics,) = trainer.test(datamodule=data, ckpt_path="best")
    preds = torch.cat(trainer.predict(datamodule=data, ckpt_path="best"))

    shutil.copy(config_path, run_dir / "config.json")
    hparams = {
        "net": model_cfg,
        "model": dict(model.hparams),
        "data": dict(data.hparams),
    }
    (run_dir / "hparams.json").write_text(json.dumps(hparams, indent=2))
    plots = save_run_plots(
        run_dir,
        class_names=model.hparams.class_names,
        metric_class_names=model.metric_class_names,
        test_set=data.test_set,
        preds=preds,
        mean=FASHION_MNIST_MEAN[0],
        std=FASHION_MNIST_STD[0],
    )
    if lit_logger:
        for path in [run_dir / "config.json", run_dir / "hparams.json", *plots]:
            lit_logger.log_file(str(path))
        print(f"LitLogger run: {lit_logger.url}")
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
            "hidden_sizes": "-".join(map(str, model_cfg["hidden_sizes"])),
            "dropout": model_cfg["dropout"],
            "num_params": num_params,
            "seed": config["seed"],
            "best_checkpoint": Path(checkpoint.best_model_path).name,
        },
    )
    return test_metrics


def _connect_litlogger(run_dir: Path, name: str) -> "LitLogger | None":
    """Create a LitLogger and connect now, so a failure doesn't stop training later.

    Credentials come only from ``LIGHTNING_USER_ID`` and ``LIGHTNING_API_KEY``
    (loaded from ``.env``). Without them the Lightning SDK would fall back to a
    browser login, which hangs in Colab and over SSH, so LitLogger is skipped.

    Args:
        run_dir: Run folder; LitLogger keeps its local files in ``run_dir/litlogger``.
        name: Experiment name on lightning.ai (a timestamp is added to it).

    Returns:
        The connected logger, or ``None`` if it is not configured or can't connect.
    """
    keys = ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY")
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        warnings.warn(
            f"LitLogger disabled: add {' and '.join(missing)} to .env", stacklevel=2
        )
        return None
    try:
        # save_logs=True would record the terminal by relaunching the script from
        # sys.argv, which breaks `python -m` and notebooks.
        logger = LitLogger(root_dir=run_dir / "litlogger", name=name, save_logs=False)
        _ = logger.experiment
        return logger
    except (RuntimeError, OSError) as err:
        warnings.warn(f"LitLogger disabled, logging locally only: {err}", stacklevel=2)
        return None
