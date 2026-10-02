"""Locations of the project, the repo root, and the configs."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
"""The project folder, e.g. ``.../CodeCS4337Fall2026/TrainingBasicNeuralNetwork``."""
PROJECT_NAME = PROJECT_DIR.name
"""The project folder's name, used as the subfolder of ``OUTPUT_DIR``."""
REPO_DIR = PROJECT_DIR.parent
"""The repo root, where ``.env``, ``data/``, and ``runs/`` live."""
CONFIGS_DIR = PROJECT_DIR / "configs"
"""The folder holding this project's JSON configs."""


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve ``path`` against the repo root unless it is absolute.

    This keeps data and outputs in the same place whatever the current working
    directory is, e.g. for a notebook started outside the repo.

    Args:
        path: An absolute path, or a path relative to the repo root.

    Returns:
        The absolute path.
    """
    path = Path(path).expanduser()
    return path if path.is_absolute() else REPO_DIR / path


def resolve_config_path(config: str | Path) -> Path:
    """Locate a JSON config file.

    Lookup order: ``config`` as given (absolute, or relative to the current
    working directory), then relative to the project directory, then inside
    ``configs/``. The ``.json`` extension may be omitted, so ``"config01"``,
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
    names = [config] if config.suffix else [config.with_suffix(".json"), config]
    for name in names:
        for candidate in (name, PROJECT_DIR / name, CONFIGS_DIR / name):
            if candidate.is_file():
                return candidate.resolve()

    available = sorted(p.name for p in CONFIGS_DIR.glob("*.json"))
    raise FileNotFoundError(
        f"Config {str(config)!r} not found. Available in {CONFIGS_DIR}: {available}"
    )
