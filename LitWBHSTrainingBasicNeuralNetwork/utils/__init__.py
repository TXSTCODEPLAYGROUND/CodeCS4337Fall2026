"""Shared helpers: paths, run summaries, and W&B runs."""

from .experiment import append_run_summary
from .paths import (
    CONFIGS_DIR,
    PROJECT_DIR,
    PROJECT_NAME,
    REPO_DIR,
    resolve_config_path,
    resolve_repo_path,
)
from .tracking import connect_wandb

__all__ = [
    "CONFIGS_DIR",
    "PROJECT_DIR",
    "PROJECT_NAME",
    "REPO_DIR",
    "append_run_summary",
    "connect_wandb",
    "resolve_config_path",
    "resolve_repo_path",
]
