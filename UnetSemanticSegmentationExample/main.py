"""Binary semantic segmentation of polyps (Kvasir-SEG) with a U-Net, PyTorch Lightning, and W&B.

Usage:

.. code-block:: text

    From the repo root:        python -m UnetSemanticSegmentationExample --config config01.json
    From this folder:          python main.py --config config01.json
    Python or a notebook:      from UnetSemanticSegmentationExample import main
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

import time
from pathlib import Path

import colabutilscs4337 as cu
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor

from .callbacks import LogSegmentation, ResumableEarlyStopping
from .dataloaders import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, KvasirSegDataModule
from .models import LitUNet, build_net
from .utils import PROJECT


def main(
    config_path: str | Path | None = None, resume_from: str | Path | None = None
) -> dict[str, float]:
    """Train, then evaluate the best checkpoint on validation and test, using a JSON config.

    The three configs are three experiments; only values differ:

    - ``config01`` (baseline): U-Net from scratch, BCE loss, Adam, resize and
      horizontal flips only, constant learning rate.
    - ``config02`` (improved training): the same U-Net with strong
      augmentation, BCE + Dice loss, AdamW, and a cosine learning-rate schedule.
    - ``config03`` (fine-tuning): a U-Net whose encoder is an ImageNet-pretrained
      ResNet-34, frozen for the first ``"freeze_encoder_epochs"`` epochs, then
      trained with the smaller ``"encoder_lr"``.

    Every metric is sent to `Weights & Biases <https://wandb.ai>`__ (W&B), in
    the project named in the config, together with the config itself, the
    learning rates, and
    :class:`~UnetSemanticSegmentationExample.callbacks.wandb_segmentation.LogSegmentation`'s
    predicted masks. The W&B API key is read from ``WANDB_API_KEY`` in the
    ``.env`` file at the repo root. Without it, or if W&B cannot log in, the
    run is logged offline in the run folder and can be uploaded later with
    ``wandb sync``.

    Local results go to ``<OUTPUT_DIR>/UnetSemanticSegmentationExample/<config_name>/<timestamp>/``:
    a copy of the config, ``metrics.csv`` (a local backup of the metrics),
    ``hparams.json``, the best (highest ``val_dice``) and last checkpoints, and
    W&B's ``wandb/`` folder. A row is appended to ``runs_summary.csv`` next to
    the run folders, with the test metrics, the training time, the number of
    parameters, and the inference time per image.

    With ``"early_stopping": true`` in the config's ``"training"`` section,
    training stops once ``val_dice`` has not improved for ``"patience"`` epochs
    (:class:`~colabutilscs4337.lit.ResumableEarlyStopping`).
    ``"precision": "16-mixed"`` trains with mixed precision (faster, less GPU
    memory) when a GPU is available; on a CPU, 32-bit is used.

    Args:
        config_path: Config name or path, e.g. ``"config01.json"``. When
            omitted, it is read from the required ``--config`` command-line flag.
        resume_from: Continue an earlier run instead of starting from
            scratch: ``"config01"`` (the newest run of config01), a run folder,
            or a checkpoint, found by
            :meth:`~colabutilscs4337.project.Project.find_resume_checkpoint`.
            The weights, the optimizer, the scheduler, and the epoch count come
            from the run's last checkpoint; the config's ``"epochs"`` more
            epochs are then trained into a new run folder. The model settings
            in the config must match the run's. On the command line:
            ``--resume-from config01``.

    Returns:
        Test metrics of the best checkpoint: ``test_loss``, ``test_dice``,
        ``test_iou``, ``test_precision``, ``test_recall``, and ``test_pixel_acc``.
    """
    run = cu.lit.start_run(
        PROJECT,
        config_path,
        resume_from,
        description=__doc__.splitlines()[0],
        monitor="val_dice",
        metric_label="valdice",
    )
    config = run.config
    data_cfg, model_cfg, train_cfg = config["data"], config["model"], config["training"]

    L.seed_everything(config["seed"], workers=True)

    data = KvasirSegDataModule(
        data_dir=str(PROJECT.data_dir()), seed=config["seed"], **data_cfg
    )
    # When resuming, all weights come from the checkpoint.
    net = build_net(model_cfg, pretrained=run.resume_checkpoint is None)
    total = sum(p.numel() for p in net.parameters())
    print(f"{model_cfg['architecture']}: {total:,} parameters")
    model = LitUNet(
        net=net,
        loss=train_cfg["loss"],
        optimizer=train_cfg["optimizer"],
        lr=train_cfg["lr"],
        encoder_lr=train_cfg["encoder_lr"],
        weight_decay=train_cfg["weight_decay"],
        scheduler=train_cfg["scheduler"],
        freeze_encoder_epochs=train_cfg["freeze_encoder_epochs"],
        threshold=train_cfg["threshold"],
    )

    wandb_logger = cu.lit.connect_wandb(
        project=config["wandb"]["project"],
        name=f"{run.config_name}-{run.run_id}",
        group=run.config_name,
        save_dir=run.run_dir,
        config=config,
    )
    wandb_logger.experiment.summary["total_params"] = total

    callbacks: list[L.Callback] = [
        run.checkpoint,
        run.last_checkpoint,
        LearningRateMonitor(logging_interval="epoch"),
        LogSegmentation(
            CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, train_cfg["threshold"]
        ),
    ]
    early_stopping = train_cfg.get("early_stopping", False)
    if early_stopping:
        callbacks.append(
            ResumableEarlyStopping(
                monitor="val_dice", mode="max", patience=train_cfg.get("patience", 15)
            )
        )

    gpu = config["accelerator"] in ("auto", "gpu", "cuda") and torch.cuda.is_available()
    precision = train_cfg.get("precision", "32-true") if gpu else "32-true"
    trainer = L.Trainer(
        max_epochs=run.epochs,
        accelerator=config["accelerator"],
        precision=precision,
        logger=[run.csv_logger, wandb_logger],
        callbacks=callbacks,
    )
    start = time.perf_counter()
    trainer.fit(model, datamodule=data, ckpt_path=run.resume_checkpoint)
    train_minutes = (time.perf_counter() - start) / 60
    epochs_run = trainer.current_epoch
    if epochs_run < run.epochs:
        print(
            f"Early stopping: val_dice stopped improving, stopped after epoch {epochs_run}"
        )
    best_epoch = torch.load(
        run.checkpoint.best_model_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )["epoch"]
    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    # The best checkpoint again on the validation set, now with the per-image
    # report (the validation runs during training only log the per-epoch metrics).
    (val_metrics,) = trainer.validate(datamodule=data, ckpt_path="best")
    (test_metrics,) = trainer.test(datamodule=data, ckpt_path="best")
    device = trainer.strategy.root_device
    inference_ms = _inference_ms_per_image(model, data, device)
    summary = {
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "train_minutes": round(train_minutes, 2),
        "trainable_params": trainable,
        "model_size_mb": round(total * 4 / 1e6, 1),
        "inference_ms_per_image": round(inference_ms, 2),
        "inference_device": device.type,
    }
    wandb_logger.experiment.summary.update(summary)
    if not wandb_logger.experiment.offline:
        print(f"W&B run: {wandb_logger.experiment.url}")
    # Close the run, so calling main() again in a notebook starts a new one.
    wandb.finish()

    run.save({"model": dict(model.hparams), "data": dict(data.hparams)})
    run.append_summary(
        {
            "run_id": run.run_id,
            "architecture": model_cfg["architecture"],
            "pretrained": model_cfg["pretrained"],
            "loss": train_cfg["loss"],
            "optimizer": train_cfg["optimizer"],
            "augmentation": data_cfg["augmentation"],
            "scheduler": train_cfg["scheduler"],
            "total_params": total,
            **summary,
            "epochs": train_cfg["epochs"],
            "early_stopping": early_stopping,
            "best_val_dice": f"{val_metrics['val_dice']:.4f}",
            "best_val_iou": f"{val_metrics['val_iou']:.4f}",
            "best_val_loss": f"{val_metrics['val_loss']:.4f}",
            "test_dice": f"{test_metrics['test_dice']:.4f}",
            "test_iou": f"{test_metrics['test_iou']:.4f}",
            "test_precision": f"{test_metrics['test_precision']:.4f}",
            "test_recall": f"{test_metrics['test_recall']:.4f}",
            "test_pixel_acc": f"{test_metrics['test_pixel_acc']:.4f}",
            "test_loss": f"{test_metrics['test_loss']:.4f}",
            "lr": train_cfg["lr"],
            "encoder_lr": train_cfg["encoder_lr"],
            "weight_decay": train_cfg["weight_decay"],
            "freeze_encoder_epochs": train_cfg["freeze_encoder_epochs"],
            "image_size": data_cfg["image_size"],
            "batch_size": data_cfg["batch_size"],
            "precision": precision,
            "seed": config["seed"],
            "best_checkpoint": Path(run.checkpoint.best_model_path).name,
        }
    )
    return test_metrics


def _inference_ms_per_image(
    model: L.LightningModule, data: KvasirSegDataModule, device: torch.device
) -> float:
    """Time the model's forward pass on one validation batch, in milliseconds per image.

    The batch runs a few times first (warm-up: CUDA kernels and memory are set
    up on the first calls), then the mean of 10 timed runs is returned. The
    model is moved to ``device`` (the training device: Lightning moves it back
    to the CPU after testing).
    """
    images = next(iter(data.val_dataloader()))[0].to(device)
    model.to(device).eval()
    with torch.no_grad():
        for repeat in range(13):
            if repeat == 3:
                if images.is_cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
            model(images)
        if images.is_cuda:
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / 10 / len(images) * 1000
