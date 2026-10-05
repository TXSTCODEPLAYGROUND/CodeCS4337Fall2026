---
name: cs4337-nn-training-skills
description: >-
  CS4337NNTrainingSkills. Builds and trains neural-network projects in the
  CodeCS4337Fall2026 repo following its LitWB format (PyTorch Lightning + W&B):
  self-contained package, JSON configs, LightningDataModule and LightningModule,
  checkpoints, early stopping, resume, load_model, runs_summary.csv, offline W&B
  fallback, Colab notebook, README, Sphinx docs, and requirements.txt. Use when
  creating a new project, model, or training experiment in this repo, changing
  an existing project, adding a config, or when the user mentions LitWB,
  training a network, W&B logging, or a new course project.
---

# CS4337NNTrainingSkills

Every training project in this repo is a self-contained Python package that
follows the **LitWB format**. Copy the structure of a reference project rather
than inventing a new one:

| Reference | Use it for |
| --- | --- |
| `LitWBTrainingBasicConvnet/` | **The canonical template.** Minimal LitWB project; start here. |
| `LitWBTransferLearningResnet/` | Pretrained backbones, several strategies as configs, a full val/test W&B report (`callbacks/wandb_evaluation.py`). |
| `LitWBHSTrainingBasicNeuralNetwork/` | Adding an Optuna hyperparameter search. |
| `YoloExample/` | Libraries with their own trainer (Ultralytics): same layout, configs, W&B, summaries, and docs, without Lightning. |

Read the reference project's `main.py`, `models/`, `dataloaders/`, and
`README.md` before writing code; match their naming, docstrings, and comments.

## Non-negotiable rules

- **Credentials:** API keys go only in the repo-root `.env` (git-ignored).
  Never write keys into `.envcolab` (tracked by git), configs, notebooks, or
  code. Never read or print secret values.
- **Git:** the user commits and pushes. Do not commit; give the git commands at
  the end.
- **Testing never touches the user's accounts or global settings:** run with
  `WANDB_API_KEY=` (empty) and `WANDB_MODE=offline`, `DATA_DIR` and
  `OUTPUT_DIR` under `/tmp`, and, for libraries with global settings (e.g.
  Ultralytics), a temporary config dir (`YOLO_CONFIG_DIR=/tmp/...`). Delete
  test outputs afterwards.
- Libraries must not create folders at the repo root: downloads go under
  `DATA_DIR` (e.g. `data/<dataset>/`, `data/yolo_weights/`), results under
  `OUTPUT_DIR`.

## Always keep these in sync

These apply to every task, new project or change to an existing one, unless
the user explicitly says otherwise:

1. **`requirements.txt`:** every new third-party import (in the package, the
   notebook, or the docs) is added, pinned to the version installed in `.venv`
   (`.venv/bin/pip show <pkg>`). Never remove or re-pin existing lines unless
   asked. List the imports (code and notebook) after writing code and compare
   with `requirements.txt`, ignoring the standard library, the project itself,
   `google` (Colab), and words caught from markdown prose:
   `rg -o '(?:^|")(?:import|from) (\w+)' -r '$1' <ProjectName> --no-filename -g '*.py' -g '*.ipynb' | sort -u`.
   Pip names can differ from import names (`PIL` is `pillow`, `sklearn` is
   `scikit-learn`, `dotenv` is `python-dotenv`).
2. **Notebook:** every new project gets a Colab notebook
   (`<snake_case_name>_notebook.ipynb`, see [reference.md](reference.md)).
   Skip it only if the user says so. When configs, outputs, or APIs change,
   update the existing notebook's cells to match.
3. **Docs and READMEs:** any new or changed config, module, function, metric,
   output file, or command is reflected in the project `README.md`,
   `docs/<ProjectName>.rst` (plus `docs/index.rst` for new pages and
   `docs/conf.py` mocks for new imports), and the root `README.md` (projects
   table, folder tree, Colab table). The strict docs build must pass.

## Package layout

```
<ProjectName>/                    # CamelCase, importable: from <ProjectName> import main
├── __init__.py                   # docstring example; exports main, load_model
├── __main__.py                   # from .main import main; main()
├── main.py                       # entry point (contract below)
├── README.md
├── <snake_case_name>_notebook.ipynb
├── configs/config01.json, config02.json, ...   # one config per experiment/strategy
├── dataloaders/<dataset>.py      # LightningDataModule + dataset constants
├── models/
│   ├── components/<net>.py       # plain nn.Module, no training code
│   ├── lit_<net>.py              # LightningModule
│   └── loading.py                # find_checkpoint, find_resume_checkpoint, load_model
├── callbacks/                    # ResumableEarlyStopping, W&B prediction/evaluation logging
└── utils/paths.py, experiment.py # copied from the reference project unchanged
```

Every subpackage `__init__.py` has a one-line docstring and an explicit
`__all__`. Shared at the repo root: `.venv`, `requirements.txt`, `.env`,
`.envcolab`, `data/`, `runs/`.

## Config

```json
{
  "seed": 42,
  "accelerator": "auto",
  "wandb": {"project": "<ProjectName>"},
  "data": {"batch_size": 256, "val_split": 0.1, "num_workers": 2},
  "model": {"dropout": 0.25},
  "training": {"epochs": 10, "lr": 0.001, "weight_decay": 0.0,
               "early_stopping": true, "patience": 3}
}
```

`data` is passed as `**data_cfg` to the DataModule. Different strategies or
baselines are different configs (`config01`, `config02`, ...), never flags.

## `main.py` contract

Copy `LitWBTrainingBasicConvnet/main.py` and adapt it. It must keep:

1. The header that reruns `python main.py` as `python -m <ProjectName>`, and a
   module docstring listing the three ways to run it.
2. `main(config_path=None, resume_from=None) -> dict[str, float]`: loads
   `REPO_DIR / ".env"`, parses `--config` (required) and `--resume-from` only
   when `config_path` is `None`, resolves it with `resolve_config_path`.
3. `L.seed_everything(config["seed"], workers=True)`; `DATA_DIR` and
   `OUTPUT_DIR` from the environment through `resolve_repo_path`.
4. Run folder `<OUTPUT_DIR>/<ProjectName>/<config_stem>/<YYYY-mm-dd_HH-MM-SS>/`
   via `CSVLogger(save_dir=..., name=config_path.stem, version=run_id)`.
5. `_connect_wandb` unchanged: W&B run `name=f"{config_stem}-{run_id}"`,
   `group=config_stem`, `save_dir=run_dir`, `config=config`; offline when
   `WANDB_API_KEY` is missing or login fails, with a `wandb sync` hint.
6. `ModelCheckpoint` for the best epoch on the monitored metric
   (`best_epoch{epoch:02d}_<metric>{...:.4f}`) plus `last.ckpt` rewritten every
   epoch; `ResumableEarlyStopping` when `"early_stopping": true`.
7. Resume: weights, optimizer, and epoch from the run's `last.ckpt`; the config's
   `"epochs"` **more** epochs into a **new** run folder.
8. After `fit`: `trainer.test(ckpt_path="best")`, store `epochs_run` in the W&B
   summary, print the run URL when online, then **`wandb.finish()`**.
9. Copy the config to `run_dir/config.json`, write `hparams.json` (with
   `resumed_from`), and `append_run_summary(run_dir.parent / "runs_summary.csv", {...})`
   with run id, epochs, `epochs_run`, best val metric, test metrics, key
   hyperparameters, seed, and best checkpoint name. Return the test metrics.

## LightningModule and DataModule

- The network is passed in (`net: nn.Module`);
  `self.save_hyperparameters(ignore=["net"], logger=False)`.
- One torchmetrics `MetricCollection` per stage (`overall.clone(prefix=f"{stage}_")`),
  logged with `on_step=False, on_epoch=True`. Metric names are
  `<stage>_<metric>`: `train_loss`, `val_acc`, `test_f1`, `val_acc_<class>`.
- DataModule: `save_hyperparameters(logger=False)`, download in
  `prepare_data`, seeded splits in `setup`, `persistent_workers=num_workers > 0`,
  dataset constants (`<DATASET>_MEAN`, `_STD`, `_CLASSES`) with docstrings.
- `load_model("config01")` finds the newest run's best checkpoint, rebuilds the
  network from the run's `config.json`, returns it in eval mode.

## Code style

- Google-style docstrings with Sphinx cross-references
  (`:class:`~<ProjectName>.models.lit_x.LitX``); every public function, class,
  and module constant is documented (docs build with `-W`).
- Comments only for constraints the code cannot show; no narration.
- `ruff check` and `ruff format` clean (`ruff.toml` at the repo root).

## Deliverables checklist

```
- [ ] Package with configs, main.py, models, dataloaders, callbacks, utils
- [ ] README.md (sections in reference.md)
- [ ] Colab notebook, unless the user said not to (cells in reference.md; reuse the reference notebook's setup and API-key cells)
- [ ] docs/<ProjectName>.rst + docs/index.rst toctree entry + new imports in docs/conf.py autodoc_mock_imports
- [ ] Root README.md: projects table, folder tree, Colab table
- [ ] New packages pinned in requirements.txt (version installed in .venv), existing lines untouched
- [ ] For changes to an existing project: its README, docs page, notebook, and root README updated to match
- [ ] Verification (below) passes; test outputs and __pycache__ removed
- [ ] Git commands given to the user
```

## Verification

```bash
.venv/bin/ruff check <ProjectName> && .venv/bin/ruff format --check <ProjectName>
.venv-docs/bin/sphinx-build -W --keep-going -q -b html docs /tmp/cs4337-docs
WANDB_API_KEY= WANDB_MODE=offline DATA_DIR=/tmp/data OUTPUT_DIR=/tmp/runs \
  .venv/bin/python -m <ProjectName> --config <short test config in /tmp>.json
```

Smoke-test every config with 1 to 2 epochs (a copy in `/tmp`), then
`--resume-from`, then `load_model`, and run the notebook's code cells. The GPU
needs the unsandboxed shell; in the sandbox use `num_workers: 0`. Report real
results honestly, including when a baseline wins.

## Additional resources

- README, notebook, docs, and root-README templates: [reference.md](reference.md)
