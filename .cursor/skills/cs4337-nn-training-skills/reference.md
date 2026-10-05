# CS4337NNTrainingSkills: templates

## Project README.md

Model it on `LitWBTransferLearningResnet/README.md`. Sections, in order:

1. `# <ProjectName>`, then the Colab badge
   (`https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/<ProjectName>/<notebook>.ipynb`)
   and a line linking the notebook file.
2. One paragraph: task, dataset (linked), what the configs compare, what W&B gets.
3. Which project it builds on; link
   `../LitWBTrainingBasicConvnet/README.md#set-up-your-wb-key` and
   `#no-key-or-failed-login-offline-mode` instead of repeating the W&B setup.
   Link the main README and the API reference
   (`https://txstcodeplayground.github.io/CodeCS4337Fall2026/<ProjectName>.html`).
4. `## Running`: one `python -m <ProjectName> --config configNN.json` line per
   config with a comment; download size and location; GPU advice.
5. `## The dataset`: split table (split, images, used for), preprocessing and
   augmentation.
6. `## Configs` (or the strategies): table of what each config changes and why.
7. Concept sections the project teaches (model internals, metrics with
   definitions and how to read them).
8. `## Results`: a table from real runs (seed, hardware), with caveats.
9. `## What is logged`: per-epoch metrics table, then the after-training W&B
   report table; `### Reading the report`.
10. `## Run folder`: tree of files with comments; `runs_summary.csv` columns;
    `load_model` example; resume command.
11. `## Things to try`: 4 to 6 concrete experiments.

## Notebook (`<snake_case_name>_notebook.ipynb`)

Build it with an `nbformat` script (in `/tmp`, not committed). Copy the setup
cells verbatim from the reference notebook
(`LitWBTransferLearningResnet/lit_wb_transfer_learning_resnet_notebook.ipynb`):
the Drive mount, the clone/update + `cp .envcolab .env` + install-missing-
requirements bash cell, `%cd /content/CodeCS4337Fall2026`, and the "Load API
keys" cell that reads Colab Secrets into `.env` without printing values.
Keep the reference notebook's metadata (`accelerator: GPU`).

Cells, in order:

1. Title, Colab badge, configs table, "Choose your route" table (Colab cells,
   Colab terminal, local notebook, local terminal), numbered steps, link to pro tips.
2. `## 1. Setup` (Colab only), with the copied cells.
3. `## 2. Run the project`: the repo-path cell (`REPO` = `/content/...` in
   Colab, `Path.cwd().parent` locally, added to `sys.path`), the `OUTPUT_DIR`
   cell, Option A1 (`!cd "{REPO}" && for c in ...; do python -m ...; done`),
   A2 (Colab terminal commands), B (`from <ProjectName> import main`).
4. Teaching sections specific to the project (data, model internals, metrics),
   each with runnable cells and figures.
5. `## Results`: read every config's `runs_summary.csv` (newest row) into a
   table or chart; where to look in W&B.
6. `## Load a trained model`: `load_model` call table, then predictions on
   test samples.
7. `## Keep training a run` / finish an interrupted run: `resume_from` table.
8. `## Pro tips: keep training running`: adapt the reference cell (Drive,
   idle disconnects, plan runs, free the GPU).

Run ruff on the notebook (`ruff check --fix` and `ruff format` work on
`.ipynb`), and execute its non-Colab code cells (e.g. with `nbclient`, data
and outputs in `/tmp`) before finishing. Do not commit outputs.

## Docs page (`docs/<ProjectName>.rst`)

Model it on `docs/LitWBTransferLearningResnet.rst`:

```rst
<ProjectName>
=============

One paragraph: task, dataset, which project's setup it uses (:doc:`LitWBTrainingBasicConvnet`).

Running
-------
.. code-block:: bash  (one line per config)
.. code-block:: python  (from <ProjectName> import main)

The configs
-----------
Bullets, one per config.

What is logged
--------------
Bullets with :class:/:func: references to the logging code.

Outputs
-------
Run folder, runs_summary.csv, load_model and resume_from examples.

API
---
Subsections (Entry point, Callbacks, Models, Dataloaders, Utilities), each
with ``.. automodule:: <ProjectName>.<module>`` and ``:members:``.
```

Then add the page to a toctree in `docs/index.rst`, and every new third-party
import to `autodoc_mock_imports` in `docs/conf.py`. Build with
`.venv-docs/bin/sphinx-build -W --keep-going -q -b html docs /tmp/cs4337-docs`
and check that the API entries rendered (grep the HTML for function ids).

## Root README.md

Add the project in three places, next to related projects:

1. The projects table: `| [<ProjectName>](<ProjectName>/) | description ... [API reference](...) · [Colab notebook](...) |`.
2. The folder tree: the package with its `configs/` and key files, each with a comment.
3. The Colab table: `| <ProjectName> | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](...) (what the notebook does) |`.

## Final message to the user

Lead with the outcome and the real results table; mention any decision the
user should make (e.g. a baseline that beat training); list what was added;
end with the git commands:

```bash
cd ~/Desktop/projects/CodeCS4337Fall2026
git add <ProjectName> docs/<ProjectName>.rst docs/index.rst docs/conf.py README.md requirements.txt
git commit -m "Add <ProjectName>: <one-line summary>"
git push
```
