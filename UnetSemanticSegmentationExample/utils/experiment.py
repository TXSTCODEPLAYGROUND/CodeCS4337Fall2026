"""The ``runs_summary.csv`` file that compares runs of a config."""

import csv
from pathlib import Path
from typing import Any


def append_run_summary(summary_path: Path, row: dict[str, Any]) -> None:
    """Append one run's key metrics to a CSV shared by all runs of a config.

    The header is taken from the keys of ``row``. When ``row`` has columns
    the file does not have yet (e.g. after a new setting was added to the
    config), the file is rewritten with the new columns added, left empty for
    the earlier runs.

    Args:
        summary_path: Path to the ``runs_summary.csv`` file.
        row: Column names mapped to this run's values.
    """
    old_fieldnames: list[str] = []
    old_rows: list[dict[str, Any]] = []
    if summary_path.exists():
        with summary_path.open(newline="") as f:
            reader = csv.DictReader(f)
            old_rows = list(reader)
            old_fieldnames = list(reader.fieldnames or [])

    if old_fieldnames and set(row) <= set(old_fieldnames):
        with summary_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=old_fieldnames).writerow(row)
        return

    fieldnames = old_fieldnames + [key for key in row if key not in old_fieldnames]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([*old_rows, row])
