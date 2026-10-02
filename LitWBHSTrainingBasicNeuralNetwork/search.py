"""Search hyperparameters with Optuna, then save the best ones as a training config.

Usage:

.. code-block:: text

    From the repo root:        python -m LitWBHSTrainingBasicNeuralNetwork.search --config search01.json
    From this folder:          python search.py --config search01.json
    Python or a notebook:      from LitWBHSTrainingBasicNeuralNetwork.search import search
                               study = search("search01.json")

Then train the best settings found, and test them, with:

.. code-block:: text

    python -m LitWBHSTrainingBasicNeuralNetwork --config search01_best.json
"""

if __name__ == "__main__" and not __package__:
    # Run as `python search.py`: relative imports need the package, so rerun as `python -m`.
    import runpy
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    package = Path(__file__).resolve().parent.name
    runpy.run_module(f"{package}.search", run_name="__main__", alter_sys=True)
    sys.exit()

import argparse
import copy
import csv
import json
import logging
import os
from pathlib import Path

import lightning as L
import optuna
import wandb
from dotenv import load_dotenv

from .callbacks import OptunaPruning
from .main import build, training_callbacks
from .utils import (
    CONFIGS_DIR,
    PROJECT_NAME,
    REPO_DIR,
    connect_wandb,
    resolve_config_path,
    resolve_repo_path,
)


def search(
    search_config_path: str | Path | None = None, n_trials: int | None = None
) -> optuna.Study:
    """Run an Optuna study: train many settings and keep the best one.

    The search config names a training config to start from
    (``"base_config"``), the study settings (``"study"``), and the ranges to
    search (``"search_space"``, see :func:`suggest_config`). Each trial trains
    on the training split and is scored by its best validation accuracy; the
    test set is never used, so the final test score stays unbiased. Optuna's
    TPE sampler picks each trial's settings from how earlier trials did, and
    the median pruner stops trials that fall behind.

    Each trial is a W&B run in the group ``<search_name>``, so the trials can
    be compared on wandb.ai (e.g. with a parallel-coordinates panel). Without
    a W&B key, they are logged offline, as in :func:`~LitWBHSTrainingBasicNeuralNetwork.main.main`.

    Results go to ``<OUTPUT_DIR>/LitWBHSTrainingBasicNeuralNetwork/<search_name>/``:

    - ``study.db``: the Optuna study (SQLite). Running the same search again
      adds trials to it, so an interrupted search can be continued.
    - ``trials.csv``: one row per trial, with its settings, score, and state.
    - ``best_config.json``: the training config of the best trial, also saved
      as ``configs/<search_name>_best.json`` so it can be trained with
      ``--config <search_name>_best.json``.

    Args:
        search_config_path: Search config name or path, e.g.
            ``"search01.json"``. When omitted, it is read from the required
            ``--config`` command-line flag.
        n_trials: Number of trials to run, overriding ``"n_trials"`` in the
            config, e.g. a small number for a quick test. ``0`` runs no new
            trial and only rewrites the results of the saved study, e.g. to
            get ``configs/<search_name>_best.json`` back after the repo folder
            was deleted. Also settable with ``--n-trials``.

    Returns:
        The Optuna study, with every trial so far (``study.trials``) and the
        best one (``study.best_trial``).
    """
    load_dotenv(REPO_DIR / ".env")
    if search_config_path is None:
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--config", required=True, help="e.g. search01.json")
        parser.add_argument(
            "--n-trials", type=int, help="override the config's n_trials"
        )
        args = parser.parse_args()
        search_config_path, n_trials = args.config, args.n_trials
    search_path = resolve_config_path(search_config_path)
    search_cfg = json.loads(search_path.read_text())
    base_config = json.loads(resolve_config_path(search_cfg["base_config"]).read_text())
    study_cfg, space = search_cfg["study"], search_cfg["search_space"]
    search_name = search_path.stem

    search_dir = (
        resolve_repo_path(os.getenv("OUTPUT_DIR", "runs")) / PROJECT_NAME / search_name
    )
    search_dir.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=search_name,
        storage=f"sqlite:///{search_dir / 'study.db'}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=study_cfg["seed"]),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=study_cfg["pruner_startup_trials"],
            n_warmup_steps=study_cfg["pruner_warmup_epochs"],
        ),
    )
    objective = _Objective(
        base_config, space, search_name, search_dir, study_cfg["log_trials_to_wandb"]
    )
    timeout = study_cfg["timeout_minutes"]

    # Lightning prints its setup (GPU, model summary) for every trial; keep only warnings.
    lightning_log = logging.getLogger("lightning.pytorch")
    level = lightning_log.level
    lightning_log.setLevel(logging.WARNING)
    try:
        study.optimize(
            objective,
            n_trials=study_cfg["n_trials"] if n_trials is None else n_trials,
            timeout=timeout * 60 if timeout else None,
            gc_after_trial=True,
        )
    finally:
        lightning_log.setLevel(level)

    _save_results(study, base_config, space, search_path, search_dir)
    return study


def suggest_config(trial: optuna.Trial, base_config: dict, space: dict) -> dict:
    """Let the trial pick a value for every hyperparameter; return the training config.

    Ranges ``[low, high]`` are searched continuously (on a log scale for
    ``lr``, ``l1``, and ``l2``, which span several powers of ten); lists of
    choices are searched as categories. To fix a hyperparameter, give a range
    with ``low == high`` or a single choice.

    Some hyperparameters only exist for some choices (Optuna's
    "define-by-run" search spaces): ``momentum`` only for ``"sgd"`` and
    ``"rmsprop"``, ``l1`` and ``l2`` only when ``regularization`` uses them,
    ``step_size`` and ``gamma`` only for the schedulers that use them, and
    ``patience`` only with early stopping.

    Args:
        trial: The Optuna trial. ``optuna.trial.FixedTrial(params)`` rebuilds
            the config of a finished trial from its parameters.
        base_config: Training config providing everything not searched
            (seed, data split, W&B project, ...).
        space: The ``"search_space"`` section of the search config.

    Returns:
        A full training config, as read by :func:`~LitWBHSTrainingBasicNeuralNetwork.main.main`.
    """
    config = copy.deepcopy(base_config)
    data, model, training = config["data"], config["model"], config["training"]

    n_layers = trial.suggest_int("n_layers", *space["n_layers"])
    model["hidden_sizes"] = [
        trial.suggest_categorical(f"units_layer{i + 1}", space["units"])
        for i in range(n_layers)
    ]
    model["activation"] = trial.suggest_categorical("activation", space["activation"])
    model["dropout"] = trial.suggest_float("dropout", *space["dropout"])

    data["batch_size"] = trial.suggest_categorical("batch_size", space["batch_size"])
    training["epochs"] = trial.suggest_int("epochs", *space["epochs"])

    training["optimizer"] = trial.suggest_categorical("optimizer", space["optimizer"])
    training["lr"] = trial.suggest_float("lr", *space["lr"], log=True)
    if training["optimizer"] in ("sgd", "rmsprop"):
        training["momentum"] = trial.suggest_float("momentum", *space["momentum"])

    regularization = trial.suggest_categorical(
        "regularization", space["regularization"]
    )
    training["l1"], training["l2"] = 0.0, 0.0
    if regularization in ("l1", "l1_l2"):
        training["l1"] = trial.suggest_float("l1", *space["l1"], log=True)
    if regularization in ("l2", "l1_l2"):
        training["l2"] = trial.suggest_float("l2", *space["l2"], log=True)

    training["scheduler"] = trial.suggest_categorical("scheduler", space["scheduler"])
    if training["scheduler"] == "step":
        training["step_size"] = trial.suggest_int("step_size", *space["step_size"])
    if training["scheduler"] in ("step", "plateau"):
        training["gamma"] = trial.suggest_float("gamma", *space["gamma"])

    training["early_stopping"] = trial.suggest_categorical(
        "early_stopping", space["early_stopping"]
    )
    if training["early_stopping"]:
        training["patience"] = trial.suggest_int("patience", *space["patience"])
    return config


class _Objective:
    """Train one trial and return its score; Optuna calls it once per trial."""

    def __init__(
        self,
        base_config: dict,
        space: dict,
        search_name: str,
        search_dir: Path,
        use_wandb: bool,
    ) -> None:
        self.base_config = base_config
        self.space = space
        self.search_name = search_name
        self.search_dir = search_dir
        self.use_wandb = use_wandb
        # Set after the first trial if W&B could not log in, so later trials
        # go straight to offline mode instead of failing again.
        self.offline = False

    def __call__(self, trial: optuna.Trial) -> float:
        config = suggest_config(trial, self.base_config, self.space)
        L.seed_everything(config["seed"], workers=True, verbose=False)
        data, model, num_params = build(config)
        trial.set_user_attr("num_params", num_params)
        pruning = OptunaPruning(trial, monitor="val_acc")

        logger = False
        if self.use_wandb:
            logger = connect_wandb(
                project=config["wandb"]["project"],
                name=f"{self.search_name}-trial{trial.number:03d}",
                group=self.search_name,
                save_dir=self.search_dir,
                config={**config, "trial": trial.number, "num_params": num_params},
                offline=self.offline,
                job_type="search",
                settings=wandb.Settings(quiet=True),
            )
            self.offline = logger.experiment.offline

        trainer = L.Trainer(
            max_epochs=config["training"]["epochs"],
            accelerator=config["accelerator"],
            logger=logger,
            callbacks=[pruning, *training_callbacks(config, log_lr=bool(logger))],
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        state = "complete"
        try:
            trainer.fit(model, datamodule=data)
        except optuna.TrialPruned:
            state = "pruned"
            raise
        finally:
            if logger:
                logger.experiment.summary.update(
                    {"best_val_acc": pruning.best, "state": state}
                )
                wandb.finish()
        return pruning.best


def _save_results(
    study: optuna.Study,
    base_config: dict,
    space: dict,
    search_path: Path,
    search_dir: Path,
) -> None:
    """Write ``trials.csv`` and the best trial's training config, and print a summary."""
    trials = study.trials
    param_names = sorted({name for t in trials for name in t.params})
    with (search_dir / "trials.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "trial",
                "state",
                "val_acc",
                "epochs_run",
                "num_params",
                "seconds",
                *param_names,
            ]
        )
        for t in trials:
            seconds = t.duration.total_seconds() if t.duration else ""
            writer.writerow(
                [
                    t.number,
                    t.state.name,
                    "" if t.value is None else f"{t.value:.4f}",
                    len(t.intermediate_values),
                    t.user_attrs.get("num_params", ""),
                    f"{seconds:.0f}" if seconds != "" else "",
                    *(t.params.get(name, "") for name in param_names),
                ]
            )

    states = [t.state for t in trials]
    print(
        f"\nStudy {study.study_name!r}: {len(trials)} trials, "
        f"{states.count(optuna.trial.TrialState.COMPLETE)} complete, "
        f"{states.count(optuna.trial.TrialState.PRUNED)} pruned. "
        f"Results in {search_dir}"
    )
    if optuna.trial.TrialState.COMPLETE not in states:
        print("No trial completed, so there is no best config yet.")
        return

    best = study.best_trial
    best_config = suggest_config(
        optuna.trial.FixedTrial(best.params), base_config, space
    )
    best_config["found_by"] = {
        "search_config": search_path.name,
        "trial": best.number,
        "val_acc": round(best.value, 4),
    }
    text = json.dumps(best_config, indent=2) + "\n"
    (search_dir / "best_config.json").write_text(text)
    config_name = f"{search_path.stem}_best.json"
    (CONFIGS_DIR / config_name).write_text(text)
    print(f"Best trial: {best.number}, val_acc {best.value:.4f}")
    for name, value in best.params.items():
        print(f"  {name}: {value}")
    print(
        f"Saved its config as configs/{config_name}. Train and test it with:\n"
        f"  python -m {PROJECT_NAME} --config {config_name}"
    )


if __name__ == "__main__":
    search()
