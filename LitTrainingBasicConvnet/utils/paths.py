"""Locations of the project, the repo root, and the configs."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
"""The project folder, e.g. ``.../CodeCS4337Fall2026/LitTrainingBasicConvnet``."""
PROJECT_NAME = PROJECT_DIR.name
"""The project folder's name, used as the subfolder of ``OUTPUT_DIR``."""
REPO_DIR = PROJECT_DIR.parent
"""The repo root, where ``.env``, ``data/``, and ``runs/`` live."""
CONFIGS_DIR = PROJECT_DIR / "configs"
"""The folder holding this project's JSON configs."""


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve ``path`` against the repo root unless it is absolute.

    Data and results then land in the same place whatever the current working
    directory is.

    Args:
        path: An absolute path, or a path relative to the repo root.

    Returns:
        The absolute path.
    """
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_DIR / path


def resolve_config_path(config: str | Path) -> Path:
    """Locate a JSON config file.

    ``config`` is tried as given, then relative to the project folder, then
    inside ``configs/``, each with and without ``.json``. So ``"config01"``,
    ``"config01.json"``, and ``"configs/config01.json"`` all work.

    Args:
        config: Config name or path.

    Returns:
        The absolute path to the config file.

    Raises:
        FileNotFoundError: If no matching file exists. The message lists the
            available configs.
    """
    config = Path(config).expanduser()
    for base in (config, PROJECT_DIR / config, CONFIGS_DIR / config):
        for candidate in (base, base.with_suffix(".json")):
            if candidate.is_file():
                return candidate.resolve()
    available = sorted(p.name for p in CONFIGS_DIR.glob("*.json"))
    raise FileNotFoundError(f"Config {str(config)!r} not found. Available: {available}")
