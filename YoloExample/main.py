"""Pedestrian detection with YOLOv8 on Penn-Fudan, tracked in W&B.

Usage:

.. code-block:: text

    From the repo root:        python -m YoloExample --config config01.json
    From this folder:          python main.py --config config01.json
    Python or a notebook:      from YoloExample import main
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
import csv
import json
import os
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import wandb
from dotenv import load_dotenv
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

from .callbacks import epoch_logger, log_ap_per_iou, log_evaluation, log_predictions
from .dataloaders import CLASS_NAMES, prepare_pennfudan
from .models.loading import find_resume_weights, weights_path
from .utils import (
    PROJECT_NAME,
    REPO_DIR,
    append_run_summary,
    compare_runs,
    resolve_config_path,
    resolve_repo_path,
)

SPLITS = ("val", "test")
"""The splits evaluated after training."""
COMPARED = [
    "val_mAP50-95",
    "test_precision",
    "test_recall",
    "test_mAP50",
    "test_mAP75",
    "test_mAP50-95",
]
"""Columns of the comparison table, besides config, strategy, and epochs."""


def main(
    config_path: str | Path | None = None, resume_from: str | Path | None = None
) -> dict[str, float]:
    """Evaluate or train YOLOv8 on Penn-Fudan with a JSON config, then evaluate it.

    The config's ``"training"`` section picks one of three strategies:

    - ``"train": false``: **zero-shot**, the baseline. The COCO-pretrained
      model (``"weights": "yolov8n.pt"``) is only evaluated, with its
      predictions restricted to COCO's ``person`` class (``config01``).
    - ``"train": true`` with ``"weights": "yolov8n.pt"``: **fine-tuned** from
      the COCO weights (``config02``).
    - ``"train": true`` with ``"weights": "yolov8n.yaml"``: **from scratch**,
      the same architecture with random weights (``config03``).

    Penn-Fudan is downloaded and converted to YOLO format on the first run
    (:func:`~YoloExample.dataloaders.pennfudan.prepare_pennfudan`). Training
    is done by Ultralytics; its best epoch (highest validation mAP50-95) is
    kept in ``weights/best.pt``. Then the model is evaluated on the
    validation and test splits.

    Everything goes to `Weights & Biases <https://wandb.ai>`__, in the project
    named in the config: the per-epoch losses and metrics, the final
    precision, recall, F1, mAP50, mAP75, and mAP50-95 of both splits, AP at
    each IoU threshold, Ultralytics' PR curves and confusion matrices, a
    table of every val and test image with its predicted and true boxes, a
    ``results`` table of this run, and a ``comparison`` table of the newest
    run of every config with its gain over the zero-shot baseline. Without
    ``WANDB_API_KEY`` in ``.env``, or if W&B cannot log in, the run is logged
    offline in the run folder.

    Local results go to ``<OUTPUT_DIR>/YoloExample/<config_name>/<timestamp>/``:
    a copy of the config, Ultralytics' files (``weights/best.pt`` and
    ``last.pt``, ``results.csv``, ``args.yaml``, plots), one ``eval_<split>/``
    folder of plots per split, and ``wandb/``. A row is appended to
    ``runs_summary.csv`` next to the run folders.

    Args:
        config_path: Config name or path, e.g. ``"config01.json"``. When
            omitted, it is read from the required ``--config`` command-line flag.
        resume_from: Finish an interrupted training run instead of starting a
            new one: ``"config02"`` (its newest run), a run folder, or a
            ``last.pt`` file. Ultralytics continues from ``weights/last.pt``
            up to the run's original number of epochs, in the same folder; a
            run that already finished cannot be resumed. On the command line:
            ``--resume-from config02``.

    Returns:
        The test metrics: ``test_precision``, ``test_recall``, ``test_f1``,
        ``test_mAP50``, ``test_mAP75``, ``test_mAP50-95``, and more.
    """
    load_dotenv(REPO_DIR / ".env")
    if config_path is None:
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--config", required=True, help="e.g. config01.json")
        parser.add_argument(
            "--resume-from",
            help="finish an interrupted training run, e.g. config02 (its newest run)",
        )
        args = parser.parse_args()
        config_path, resume_from = args.config, args.resume_from
    config_path = resolve_config_path(config_path)
    config = json.loads(config_path.read_text())
    data_cfg, model_cfg = config["data"], config["model"]
    train_cfg, eval_cfg = config["training"], config.get("evaluation", {})

    data_dir = resolve_repo_path(os.getenv("DATA_DIR", "data"))
    data_yaml = prepare_pennfudan(
        data_dir, tuple(data_cfg["split_sizes"]), config["seed"]
    )
    project_runs = resolve_repo_path(os.getenv("OUTPUT_DIR", "runs")) / PROJECT_NAME
    if resume_from is not None:
        last = find_resume_weights(resume_from)
        run_dir = last.parent.parent
        run_name = f"{run_dir.name}-resumed"
        print(f"Resuming {last}")
    else:
        run_dir = (
            project_runs
            / config_path.stem
            / datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        )
        run_name = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_name = run_dir.parent.name
    strategy = _strategy(
        model_cfg["weights"], train_cfg["train"] or resume_from is not None
    )

    run = _connect_wandb(config, config_name, run_name, run_dir)
    for prefix in ("train/*", "val/*", "lr/*"):
        run.define_metric(prefix, step_metric="epoch")
    run.summary["strategy"] = strategy
    common = {
        "imgsz": data_cfg["imgsz"],
        "batch": data_cfg["batch"],
        "workers": data_cfg["workers"],
        "device": config.get("device"),
    }

    epochs_run = best_epoch = 0
    train_minutes = 0.0
    if strategy == "zero-shot":
        model = YOLO(weights_path(model_cfg["weights"]))
    else:
        model = YOLO(
            str(last) if resume_from is not None else weights_path(model_cfg["weights"])
        )
        model.add_callback("on_fit_epoch_end", epoch_logger(run))
        start = time.perf_counter()
        if resume_from is not None:
            model.train(resume=True)
        else:
            model.train(
                data=str(data_yaml),
                epochs=train_cfg["epochs"],
                patience=train_cfg["patience"],
                optimizer=train_cfg["optimizer"],
                lr0=train_cfg["lr0"],
                seed=config["seed"],
                project=str(run_dir.parent),
                name=run_dir.name,
                exist_ok=True,
                plots=True,
                **common,
                **train_cfg.get("ultralytics_args", {}),
            )
        train_minutes = (time.perf_counter() - start) / 60
        epochs_run, best_epoch = _training_progress(run_dir / "results.csv")
        if epochs_run < train_cfg.get("epochs", epochs_run):
            print(
                f"Early stopping: mAP50-95 stopped improving, stopped after epoch {epochs_run}"
            )
        run.log(
            {
                f"training/{name}": wandb.Image(str(run_dir / file))
                for file, name in (("results.png", "curves"), ("labels.jpg", "labels"))
                if (run_dir / file).is_file()
            }
        )
        model = YOLO(str(run_dir / "weights" / "best.pt"))

    # The COCO model predicts 80 classes; only its "person" boxes are scored.
    classes = [i for i, n in model.names.items() if n in CLASS_NAMES]
    classes = classes if len(model.names) > len(CLASS_NAMES) else None
    params = get_num_params(model.model)
    gflops = get_flops(model.model, data_cfg["imgsz"])
    run.summary.update(
        {
            "params": params,
            "GFLOPs": gflops,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
        }
    )

    results: dict[str, dict[str, float]] = {}
    ap_per_iou = {}
    for split in SPLITS:
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            conf=eval_cfg.get("conf", 0.001),
            iou=eval_cfg.get("iou", 0.7),
            classes=classes,
            project=str(run_dir),
            name=f"eval_{split}",
            exist_ok=True,
            plots=True,
            verbose=False,
            **common,
        )
        num_images = len(list((data_yaml.parent / "images" / split).glob("*.png")))
        results[split] = log_evaluation(run, split, metrics, num_images)
        ap_per_iou[split] = metrics.box.all_ap.mean(axis=0)
        log_predictions(
            run,
            model,
            data_yaml.parent,
            split,
            classes=classes,
            imgsz=data_cfg["imgsz"],
            conf=eval_cfg.get("gallery_conf", 0.25),
        )
    log_ap_per_iou(run, ap_per_iou)
    columns = ["split", *results["test"]]
    run.log(
        {
            "results": wandb.Table(
                columns=columns,
                data=[
                    [s, *(round(v, 4) for v in r.values())] for s, r in results.items()
                ],
            )
        }
    )
    for split, values in results.items():
        print(
            f"{split}: precision {values['precision']:.4f}, recall {values['recall']:.4f}, "
            f"mAP50 {values['mAP50']:.4f}, mAP50-95 {values['mAP50-95']:.4f}"
        )

    summary_row = {
        "run_id": run_dir.name,
        "strategy": strategy,
        "weights": model_cfg["weights"],
        "epochs": train_cfg.get("epochs", 0),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "train_minutes": f"{train_minutes:.2f}",
        "imgsz": data_cfg["imgsz"],
        "params": params,
        "GFLOPs": f"{gflops:.1f}",
        **{f"val_{k}": f"{results['val'][k]:.4f}" for k in ("mAP50", "mAP50-95")},
        **{
            f"test_{k}": f"{results['test'][k]:.4f}"
            for k in (
                "precision",
                "recall",
                "f1",
                "mAP50",
                "mAP75",
                "mAP50-95",
                "inference_ms",
            )
        },
        "seed": config["seed"],
        "resumed": resume_from is not None,
    }
    append_run_summary(run_dir.parent / "runs_summary.csv", summary_row)
    _log_comparison(run, project_runs)
    if not run.offline:
        print(f"W&B run: {run.url}")
    # Close the run, so calling main() again in a notebook starts a new one.
    wandb.finish()

    shutil.copy(config_path, run_dir / "config.json")
    return {f"test_{k}": v for k, v in results["test"].items()}


def _strategy(weights: str, train: bool) -> str:
    """Name the run's strategy: ``"zero-shot"``, ``"fine-tuned"``, or ``"from scratch"``."""
    if not train:
        return "zero-shot"
    return "from scratch" if weights.endswith(".yaml") else "fine-tuned"


def _training_progress(results_csv: Path) -> tuple[int, int]:
    """Read the number of epochs run and the best epoch (highest val mAP50-95)."""
    with results_csv.open(newline="") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    return len(rows), int(float(best["epoch"]))


def _log_comparison(run: Any, project_runs: Path) -> None:
    """Log and print the newest run of every config, with the gain over zero-shot."""
    rows = compare_runs(project_runs)
    columns = ["config", "strategy", "epochs_run", *COMPARED, "gain_vs_zero_shot"]
    table = [
        [
            r.get(c)
            if c in ("config", "strategy")
            else int(r[c])
            if c == "epochs_run"
            else _number(r.get(c))
            for c in columns
        ]
        for r in rows
    ]
    run.log({"comparison": wandb.Table(columns=columns, data=table)})
    print(
        "Newest run of each config (gain_vs_zero_shot: test mAP50-95 minus zero-shot's):"
    )
    widths = [
        max(len(str(v)) for v in (c, *(row[i] for row in table))) + 2
        for i, c in enumerate(columns)
    ]
    print("".join(f"{c:>{w}}" for c, w in zip(columns, widths, strict=True)))
    for row in table:
        print(
            "".join(
                f"{'' if v is None else v:>{w}}"
                for v, w in zip(row, widths, strict=True)
            )
        )


def _number(value: Any) -> float | None:
    """Turn a CSV string into a rounded float, or ``None`` when empty."""
    return None if value in (None, "") else round(float(value), 4)


def _connect_wandb(config: dict, config_name: str, run_name: str, run_dir: Path) -> Any:
    """Start the W&B run online, or offline when there is no key or no login.

    Args:
        config: The experiment config; ``config["wandb"]["project"]`` names the
            W&B project, and the whole config is saved with the run.
        config_name: Config file name without ``.json``, used as the run group.
        run_name: Timestamp of the run, added to the run name.
        run_dir: Run folder; W&B writes its local files to ``run_dir/wandb``.

    Returns:
        The started W&B run, online or offline.
    """

    def start(offline: bool) -> Any:
        return wandb.init(
            project=config["wandb"]["project"],
            name=f"{config_name}-{run_name}",
            group=config_name,
            dir=run_dir,
            config=config,
            mode="offline" if offline else "online",
        )

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
