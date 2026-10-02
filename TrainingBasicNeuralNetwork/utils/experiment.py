"""Run folders and the ``runs_summary.csv`` file that compares runs."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_ID_FORMAT = "%Y-%m-%d_%H-%M-%S"


def create_run_dir(experiment_dir: Path) -> tuple[str, Path]:
    """Create a new, uniquely named run directory inside ``experiment_dir``.

    The directory is named after the current local time, e.g.
    ``2026-10-01_12-40-05``, so runs sort chronologically. A numeric suffix is
    added if two runs start within the same second.

    Args:
        experiment_dir: Parent directory grouping all runs of one config.

    Returns:
        A tuple ``(run_id, run_dir)``.
    """
    base_id = datetime.now().astimezone().strftime(RUN_ID_FORMAT)
    run_id, suffix = base_id, 1
    while (experiment_dir / run_id).exists():
        run_id = f"{base_id}_{suffix}"
        suffix += 1
    run_dir = experiment_dir / run_id
    run_dir.mkdir(parents=True)
    return run_id, run_dir


def append_run_summary(summary_path: Path, row: dict[str, Any]) -> None:
    """Append one run's key metrics to a CSV shared by all runs of a config.

    The header, taken from the keys of ``row``, is written only when the file
    is first created.

    Args:
        summary_path: Path to the ``runs_summary.csv`` file.
        row: Column names mapped to this run's values.
    """
    is_new = not summary_path.exists()
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
