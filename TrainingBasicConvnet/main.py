"""Train a ConvNet on Fashion-MNIST.

Usage:

.. code-block:: text

    From the repo root:        python -m TrainingBasicConvnet --config config01.json
    From this folder:          python main.py --config config01.json
    Python or a notebook:      from TrainingBasicConvnet import main
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
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

from .dataloaders import get_dataloaders
from .models import ConvNet
from .trainers import Trainer
from .utils import (
    PROJECT_NAME,
    REPO_DIR,
    append_run_summary,
    create_run_dir,
    get_device,
    resolve_config_path,
    resolve_repo_path,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments with a ``config`` attribute holding the config name or path.
    """
    parser = argparse.ArgumentParser(description="Train a ConvNet on Fashion-MNIST")
    parser.add_argument(
        "--config",
        required=True,
        help="Config name or path, e.g. config01.json",
    )
    return parser.parse_args()


def main(config_path: str | Path | None = None) -> None:
    """Run a full training experiment from a JSON config.

    Each invocation creates ``<OUTPUT_DIR>/<project_name>/<config_name>/<timestamp>/``
    holding the config snapshot, checkpoints, per-epoch history, and a results
    file whose name includes the test accuracy. A row is also appended to
    ``<OUTPUT_DIR>/<project_name>/<config_name>/runs_summary.csv`` for
    comparing runs.

    Environment variables are read from the ``.env`` file at the repo root,
    shared by all projects. Relative ``DATA_DIR`` and ``OUTPUT_DIR`` values are
    resolved against the repo root, so data and results land in the same place
    regardless of the current working directory.

    Args:
        config_path: Config name or path, e.g. ``"config01.json"``. When given (as
            when calling from a notebook), command-line arguments are ignored.
            When omitted, it is read from the required ``--config`` CLI flag.
            See ``utils.resolve_config_path`` for the lookup rules.
    """
    load_dotenv(REPO_DIR / ".env")
    if config_path is None:
        config_path = parse_args().config

    config_path = resolve_config_path(config_path)
    with open(config_path) as f:
        config = json.load(f)

    experiment_name = config_path.stem
    experiment_dir = resolve_repo_path(os.getenv("OUTPUT_DIR", "runs"))
    experiment_dir = experiment_dir / PROJECT_NAME / experiment_name
    run_id, run_dir = create_run_dir(experiment_dir)
    shutil.copy(config_path, run_dir / "config.json")

    seed = config.get("seed", 42)
    set_seed(seed)
    device = get_device(config.get("device", "auto"))
    print(f"Experiment: {experiment_name} | run: {run_id} | device: {device}")
    print(f"Outputs: {run_dir}")

    data_cfg = config["data"]
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=str(resolve_repo_path(os.getenv("DATA_DIR", "data"))),
        batch_size=data_cfg["batch_size"],
        val_split=data_cfg["val_split"],
        num_workers=data_cfg["num_workers"],
        seed=seed,
    )

    model = ConvNet(num_classes=10, dropout=config["model"]["dropout"])
    trainer = Trainer(model, train_loader, val_loader, config, run_dir, device)

    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()
    trainer.fit()

    trainer.load_checkpoint(trainer.best_checkpoint)
    test_loss, test_acc = trainer.evaluate(test_loader, desc="test")
    duration_s = round(time.perf_counter() - start_time, 1)
    print(f"Test loss {test_loss:.4f} | test acc {test_acc:.4f}")

    train_cfg = config["training"]
    results = {
        "experiment": experiment_name,
        "run_id": run_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_s": duration_s,
        "metrics": {
            "best_epoch": trainer.best_epoch,
            "best_val_acc": trainer.best_val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
        "checkpoints": {
            "best": trainer.best_checkpoint.name,
            "last": trainer.last_checkpoint.name,
        },
        "model": {
            "name": type(model).__name__,
            "num_params": sum(p.numel() for p in model.parameters()),
        },
        "environment": {
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "config": config,
    }
    results_name = (
        f"results_epoch{trainer.best_epoch:02d}"
        f"_valacc{trainer.best_val_acc:.4f}_testacc{test_acc:.4f}.json"
    )
    with open(run_dir / results_name, "w") as f:
        json.dump(results, f, indent=2)

    append_run_summary(
        experiment_dir / "runs_summary.csv",
        {
            "run_id": run_id,
            "epochs": train_cfg["epochs"],
            "best_epoch": trainer.best_epoch,
            "best_val_acc": f"{trainer.best_val_acc:.4f}",
            "test_acc": f"{test_acc:.4f}",
            "test_loss": f"{test_loss:.4f}",
            "lr": train_cfg["lr"],
            "weight_decay": train_cfg.get("weight_decay", 0.0),
            "batch_size": data_cfg["batch_size"],
            "dropout": config["model"]["dropout"],
            "seed": seed,
            "device": str(device),
            "duration_s": duration_s,
            "best_checkpoint": trainer.best_checkpoint.name,
        },
    )
