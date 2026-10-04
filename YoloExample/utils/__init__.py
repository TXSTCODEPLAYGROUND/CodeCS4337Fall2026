"""Shared helpers: paths, run summaries, and the comparison of configs."""

from .experiment import BASELINE, append_run_summary, compare_runs
from .paths import (
    CONFIGS_DIR,
    PROJECT_DIR,
    PROJECT_NAME,
    REPO_DIR,
    resolve_config_path,
    resolve_repo_path,
)

__all__ = [
    "BASELINE",
    "CONFIGS_DIR",
    "PROJECT_DIR",
    "PROJECT_NAME",
    "REPO_DIR",
    "append_run_summary",
    "compare_runs",
    "resolve_config_path",
    "resolve_repo_path",
]
