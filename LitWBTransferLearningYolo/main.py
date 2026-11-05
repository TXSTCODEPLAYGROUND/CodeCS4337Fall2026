"""Pothole detection with YOLOv8 transfer learning, tracked in W&B.

Usage:

.. code-block:: text

    From the repo root:        python -m LitWBTransferLearningYolo --config config01.json
    From this folder:          python main.py --config config01.json
    Python or a notebook:      from LitWBTransferLearningYolo import main
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

import csv
import time
from pathlib import Path
from typing import Any

import colabutilscs4337 as cu
import wandb
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

from .callbacks import epoch_logger, log_evaluation, log_predictions
from .dataloaders import image_paths, prepare_pothole
from .models.loading import find_resume_weights, weights_path, yolo_weights_dir
from .utils import PROJECT

BASELINE = "from scratch"
"""The ``strategy`` of the run the others are compared with."""
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
    """Train YOLOv8 on the pothole dataset with a JSON config, then evaluate it.

    The config's ``"model"`` and ``"training"`` sections pick one of three
    strategies. COCO (Common Objects in Context), the dataset the pretrained
    weights come from, has no pothole class, so every config trains a new head:

    - ``"weights": "yolov8n.pt"`` and ``"freeze": 10``: **frozen backbone**.
      The COCO backbone stays fixed and the neck and new head learn potholes
      (``config01``).
    - ``"weights": "yolov8n.pt"`` with nothing frozen: **fine-tuned**. Every
      layer is trained, with a smaller learning rate (``config02``).
    - ``"weights": "yolov8n.yaml"``: **from scratch**, the same architecture
      with random weights (``config03``). This is the baseline.

    The dataset is downloaded and split on the first run
    (:func:`~LitWBTransferLearningYolo.dataloaders.pothole.prepare_pothole`).
    Training is done by Ultralytics; its best epoch (highest validation
    mAP50-95) is kept in ``weights/best.pt``. Then the model is evaluated on
    the validation and test splits.

    Everything goes to `Weights & Biases <https://wandb.ai>`__, in the project
    named in the config: the per-epoch losses and metrics, the final
    precision, recall, F1, mAP50, mAP75, and mAP50-95 of both splits,
    Ultralytics' PR curves and confusion matrices, a table of every val and
    test image with its predicted and true boxes, a ``results`` table of this
    run, and a ``comparison`` table of the newest run of every config with its
    gain over the from-scratch baseline. Without ``WANDB_API_KEY`` in
    ``.env``, or if W&B cannot log in, the run is logged offline in the run
    folder.

    Local results go to
    ``<OUTPUT_DIR>/LitWBTransferLearningYolo/<config_name>/<timestamp>/``:
    a copy of the config, ``hparams.json``, Ultralytics' files
    (``weights/best.pt`` and ``last.pt``, ``results.csv``, ``args.yaml``,
    plots), one ``eval_<split>/`` folder of plots per split, and ``wandb/``.
    A row is appended to ``runs_summary.csv`` next to the run folders. In
    Colab, ``OUTPUT_DIR`` points at Google Drive (``.envcolab``), so those
    folders survive a runtime reset.

    Args:
        config_path: Config name or path, e.g. ``"config01.json"``. When
            omitted, it is read from the required ``--config`` command-line flag.
        resume_from: Continue a training run instead of starting from the
            config's weights: ``"config02"`` (its newest run), a run folder,
            or a ``last.pt`` file. An **interrupted** run is finished in its
            own folder: Ultralytics continues from ``weights/last.pt``
            (weights, optimizer, epoch) up to the run's original number of
            epochs. A **finished** run (all epochs done, or early stopping)
            is trained the config's ``"epochs"`` **more** epochs, starting
            from its ``weights/best.pt``, in a **new** run folder, with the
            config's settings and no warm-up. On the command line:
            ``--resume-from config02``.

    Returns:
        The test metrics: ``test_precision``, ``test_recall``, ``test_f1``,
        ``test_mAP50``, ``test_mAP75``, ``test_mAP50-95``, and more.
    """
    PROJECT.load_env()
    if config_path is None:
        config_path, resume_from = cu.lit.parse_cli(__doc__.splitlines()[0])
    config_path, config = PROJECT.read_config(config_path)
    data_cfg, model_cfg = config["data"], config["model"]
    train_cfg, eval_cfg = config["training"], config.get("evaluation", {})
    freeze = train_cfg.get("ultralytics_args", {}).get("freeze", 0)

    data_dir = PROJECT.data_dir()
    data_yaml = prepare_pothole(data_dir, config["seed"])
    yolo_weights_dir()
    project_runs = PROJECT.runs_dir()
    interrupted_last = continue_from = source_run = None
    if resume_from is not None:
        resume_weights, interrupted = find_resume_weights(resume_from)
        source_run = resume_weights.parent.parent
        if interrupted:
            interrupted_last = resume_weights
            print(f"Finishing the interrupted run {source_run}")
        else:
            continue_from = resume_weights
            print(
                f"{source_run.name} already finished: training {train_cfg['epochs']} "
                f"more epochs from {resume_weights}, in a new run folder"
            )
    if interrupted_last is not None:
        run_dir = source_run
        run_name = f"{run_dir.name}-resumed"
    else:
        run_dir = PROJECT.new_run_dir(config_path.stem).path
        run_name = run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_name = run_dir.parent.name
    strategy = _strategy(model_cfg["weights"], freeze)
    print(f"{strategy}: {run_dir}")

    start_weights = interrupted_last or continue_from
    hparams = {
        "seed": config["seed"],
        "strategy": strategy,
        "data": data_cfg,
        "model": model_cfg,
        "training": train_cfg,
        "resumed_from": str(start_weights) if start_weights else "",
    }
    PROJECT.save_run_files(run_dir, config_path, hparams)

    run = cu.lit.connect_wandb_run(
        project=config["wandb"]["project"],
        name=f"{config_name}-{run_name}",
        group=config_name,
        save_dir=run_dir,
        config=config,
    )
    for prefix in ("train/*", "val/*", "lr/*"):
        run.define_metric(prefix, step_metric="epoch")
    run.summary["strategy"] = strategy
    if source_run is not None:
        run.summary["resumed_from"] = source_run.name
    common = {
        "imgsz": data_cfg["imgsz"],
        "batch": data_cfg["batch"],
        "workers": data_cfg["workers"],
    }
    if config.get("device"):
        common["device"] = config["device"]

    model = YOLO(
        str(start_weights) if start_weights else weights_path(model_cfg["weights"])
    )
    model.add_callback("on_fit_epoch_end", epoch_logger(run))
    # Warm-up (3 epochs from a low learning rate) is for fresh weights; a
    # trained model would be shaken up by its high bias learning rate.
    extra_args = {"warmup_epochs": 0} if continue_from is not None else {}
    extra_args |= train_cfg.get("ultralytics_args", {})
    start = time.perf_counter()
    if interrupted_last is not None:
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
            **extra_args,
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

    params = get_num_params(model.model)
    gflops = float(get_flops(model.model, data_cfg["imgsz"]))
    run.summary.update(
        {
            "params": params,
            "GFLOPs": gflops,
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
        }
    )

    results: dict[str, dict[str, float]] = {}
    for split in SPLITS:
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            conf=eval_cfg.get("conf", 0.001),
            iou=eval_cfg.get("iou", 0.7),
            project=str(run_dir),
            name=f"eval_{split}",
            exist_ok=True,
            plots=True,
            verbose=False,
            **common,
        )
        results[split] = log_evaluation(
            run, split, metrics, len(image_paths(data_yaml.parent, split))
        )
        log_predictions(
            run,
            model,
            data_yaml.parent,
            split,
            imgsz=data_cfg["imgsz"],
            conf=eval_cfg.get("gallery_conf", 0.25),
        )
    columns = ["split", *results["test"]]
    run.log(
        {
            "results": wandb.Table(
                columns=columns,
                data=[
                    [split, *(round(value, 4) for value in row.values())]
                    for split, row in results.items()
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
        "freeze": freeze or 0,
        "lr0": train_cfg["lr0"],
        "optimizer": train_cfg["optimizer"],
        "epochs": train_cfg["epochs"],
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "train_minutes": f"{train_minutes:.2f}",
        "imgsz": data_cfg["imgsz"],
        "params": params,
        "trainable_params": run.summary.get("trainable_params", ""),
        "GFLOPs": f"{gflops:.1f}",
        **{f"val_{key}": f"{results['val'][key]:.4f}" for key in ("mAP50", "mAP50-95")},
        **{
            f"test_{key}": f"{results['test'][key]:.4f}"
            for key in (
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
        "resumed_from": source_run.name if source_run is not None else "",
    }
    cu.project.append_run_summary(run_dir.parent / "runs_summary.csv", summary_row)
    _log_comparison(run, project_runs)
    if not run.offline:
        print(f"W&B run: {run.url}")
    # Close the run, so calling main() again in a notebook starts a new one.
    wandb.finish()
    return {f"test_{key}": value for key, value in results["test"].items()}


def _strategy(weights: str, freeze: int) -> str:
    """Name the run: ``"frozen backbone"``, ``"fine-tuned"``, or ``"from scratch"``."""
    if str(weights).endswith((".yaml", ".yml")):
        return "from scratch"
    return "frozen backbone" if freeze else "fine-tuned"


def _training_progress(results_csv: Path) -> tuple[int, int]:
    """Read the number of epochs run and the best epoch (highest val mAP50-95)."""
    with results_csv.open(newline="") as f:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    best = max(rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    return len(rows), int(float(best["epoch"]))


def _log_comparison(run: Any, project_runs: Path) -> None:
    """Log and print the newest run of every config, with the gain over from scratch."""
    rows = cu.project.compare_runs(
        project_runs, "test_mAP50-95", baseline_value=BASELINE
    )
    for row in rows:
        row["gain_vs_from_scratch"] = row.pop("gain_vs_baseline")
    if not rows:
        return
    columns = ["config", "strategy", "epochs_run", *COMPARED, "gain_vs_from_scratch"]
    table = [
        [
            row.get(column)
            if column in ("config", "strategy")
            else int(row[column])
            if column == "epochs_run"
            else _number(row.get(column))
            for column in columns
        ]
        for row in rows
    ]
    logged = [["" if value is None else value for value in row] for row in table]
    run.log({"comparison": wandb.Table(columns=columns, data=logged)})
    print(
        "Newest run of each config (gain_vs_from_scratch: test mAP50-95 minus from scratch):"
    )
    widths = [
        max(len(str(value)) for value in (column, *(row[i] for row in table))) + 2
        for i, column in enumerate(columns)
    ]
    print(
        "".join(
            f"{column:>{width}}" for column, width in zip(columns, widths, strict=True)
        )
    )
    for row in table:
        print(
            "".join(
                f"{'' if value is None else value:>{width}}"
                for value, width in zip(row, widths, strict=True)
            )
        )


def _number(value: Any) -> float | None:
    """Turn a CSV string into a rounded float, or ``None`` when empty."""
    return None if value in (None, "") else round(float(value), 4)
