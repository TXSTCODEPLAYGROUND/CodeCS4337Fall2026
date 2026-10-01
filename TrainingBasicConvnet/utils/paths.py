from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_DIR / "configs"


def resolve_project_path(path: str | Path) -> Path:
    """Resolve ``path`` against the project directory unless it is absolute.

    This keeps outputs in the same place whether the code is run from inside
    the project folder, from the repo root, or imported from a notebook.

    Args:
        path: An absolute path, or a path relative to the project directory.

    Returns:
        The absolute path.
    """
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def resolve_config_path(config: str | Path) -> Path:
    """Locate a YAML config file.

    Lookup order: ``config`` as given (absolute, or relative to the current
    working directory), then relative to the project directory, then inside
    ``configs/``. The ``.yaml`` extension may be omitted, so ``"config01"``,
    ``"config01.yaml"``, and ``"configs/config01.yaml"`` all work.

    Args:
        config: Config name or path.

    Returns:
        The absolute path to the config file.

    Raises:
        FileNotFoundError: If no matching file exists. The message lists the
            available configs.
    """
    config = Path(config).expanduser()
    names = [config] if config.suffix else [config.with_suffix(".yaml"), config]
    for name in names:
        for candidate in (name, PROJECT_DIR / name, CONFIGS_DIR / name):
            if candidate.is_file():
                return candidate.resolve()

    available = sorted(p.name for p in CONFIGS_DIR.glob("*.yaml"))
    raise FileNotFoundError(
        f"Config {str(config)!r} not found. Available in {CONFIGS_DIR}: {available}"
    )
