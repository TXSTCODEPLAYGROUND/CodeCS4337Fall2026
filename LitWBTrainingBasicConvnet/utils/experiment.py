"""The ``runs_summary.csv`` file that compares runs of a config."""

import csv
from pathlib import Path
from typing import Any


def append_run_summary(summary_path: Path, row: dict[str, Any]) -> None:
    """Append one run's key metrics to a CSV shared by all runs of a config.

    The header, taken from the keys of ``row``, is written only when the file
    is first created.

    Args:
        summary_path: Path to the ``runs_summary.csv`` file.
        row: Column names mapped to this run's values.
    """
    is_new = not summary_path.exists()
    with summary_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
