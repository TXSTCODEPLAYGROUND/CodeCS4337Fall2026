"""Starting a W&B run, online when possible and offline otherwise."""

import contextlib
import os
import warnings
from pathlib import Path
from typing import Any

import wandb
from lightning.pytorch.loggers import WandbLogger
from wandb.sdk.mailbox.mailbox import MailboxClosedError


def connect_wandb(
    project: str,
    name: str,
    group: str,
    save_dir: Path,
    config: dict,
    offline: bool = False,
    **init_kwargs: Any,
) -> WandbLogger:
    """Start the W&B run online, or offline when there is no key or no login.

    Args:
        project: W&B project name.
        name: Run name shown in W&B.
        group: Run group; W&B can show or average the runs of a group together.
        save_dir: Folder where W&B writes its local files (``save_dir/wandb``).
        config: Settings saved with the run, shown as columns in W&B.
        offline: Skip the online attempt, e.g. after an earlier run of the
            same search already failed to log in.
        **init_kwargs: More arguments for ``wandb.init``, e.g. ``job_type``.

    Returns:
        A ``WandbLogger`` whose run has already started, online or offline.
        ``logger.experiment.offline`` tells which.
    """

    def init(offline: bool) -> WandbLogger:
        logger = WandbLogger(
            project=project,
            name=name,
            group=group,
            save_dir=save_dir,
            config=config,
            # Passed to wandb.init: unlike offline=True, which only sets WANDB_MODE,
            # this still works after a failed online attempt.
            mode="offline" if offline else "online",
            **init_kwargs,
        )
        _ = logger.experiment
        return logger

    def start(offline: bool) -> WandbLogger:
        try:
            return init(offline)
        except MailboxClosedError:
            # W&B's background process died (e.g. after interrupting a notebook
            # cell) and every later run in this Python session would fail; drop the
            # dead connection so the next run starts a new process.
            with contextlib.suppress(Exception):
                wandb.teardown()
            return init(offline)

    sync_hint = (
        "logging to W&B offline. Upload the run later with: "
        f"wandb sync {save_dir / 'wandb'}/offline-run-*"
    )
    if offline:
        return start(offline=True)
    if not os.getenv("WANDB_API_KEY"):
        warnings.warn(f"WANDB_API_KEY is not in .env: {sync_hint}", stacklevel=2)
        return start(offline=True)
    try:
        return start(offline=False)
    except (wandb.errors.Error, OSError) as error:
        wandb.finish()
        warnings.warn(f"Could not log in to W&B ({error}): {sync_hint}", stacklevel=2)
        return start(offline=True)
