# CS4337 Fall 2026 — Course Code

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/starternotebook.ipynb)

This repository holds the code for CS4337, Fall 2026. Each top-level folder
(for example `TrainingBasicConvnet/`) is a **self-contained project**: a small
Python package that you can either

- import and run from a notebook (e.g. Google Colab), or
- run from the terminal on your own machine.

New projects will be added over the semester. To get them, run `git pull`
inside your copy of the repository.

## Projects

| Project | Description |
| --- | --- |
| [TrainingBasicConvnet](TrainingBasicConvnet/) | Train a basic ConvNet on Fashion-MNIST |

## Repository structure

```
CodeCS4337Fall2026/
├── README.md              # This file
├── starternotebook.ipynb  # Ready-to-run Google Colab notebook (start here in Colab)
├── ruff.toml              # Code style settings (you can ignore this for now, see below)
├── .gitignore             # Files git should not track (data, runs, .env, virtualenvs, ...)
├── .env                   # Local settings for all projects: where data and results go (not in git)
├── .envcolab              # Google Colab settings for all projects (copied to .env in Colab)
├── .venv/                 # Your local Python virtual environment (you create it, not in git)
├── data/                  # Datasets shared by all projects (created on first run, not in git)
├── runs/                  # Experiment results, one subfolder per project (not in git)
└── TrainingBasicConvnet/  # One project = one Python package
    ├── __init__.py        # Makes the folder importable: from TrainingBasicConvnet import main
    ├── __main__.py        # Makes `python -m TrainingBasicConvnet` work
    ├── main.py            # Entry point: main(config) runs a full experiment
    ├── configs/           # Experiment settings (config01.json, ...)
    ├── dataloaders/       # Dataset loading
    ├── models/            # Network definitions
    ├── trainers/          # Training and evaluation loop
    ├── utils/             # Helpers (paths, seeding, run folders)
    └── requirements.txt   # Python packages this project needs
```

Every project follows the same layout, so once you know one you know them all.

### The `.env` and `.envcolab` files

One settings file at the repo root is shared by every project:

- `DATA_DIR`: where datasets are stored.
- `OUTPUT_DIR`: where experiment results are written.

Relative paths (like `data` or `runs`) are resolved against the repo root, so
it does not matter which folder you run from. Absolute paths are used as-is.

- **`.env`** holds the local settings (`DATA_DIR=data`, `OUTPUT_DIR=runs`). It
  is not tracked by git, so you can change it freely. If it is missing (for
  example after a fresh clone), the same defaults are used.
- **`.envcolab`** holds the Colab settings. In Colab it is copied to `.env`
  (the starter notebook does this for you, see
  [Running in Google Colab](#running-in-google-colab)).

### Configs

An experiment is described by a JSON file in the project's `configs/` folder
(learning rate, batch size, number of epochs, ...). To try your own settings,
copy a config, give it a new name (e.g. `config02.json`), edit it, and pass
that name when you run the project. You never need to edit the Python code to
change hyperparameters.

```json
{
  "seed": 42,
  "device": "auto",
  "data": {"batch_size": 256, "val_split": 0.1, "num_workers": 2},
  "model": {"dropout": 0.25},
  "training": {"epochs": 10, "lr": 0.001, "weight_decay": 0.0}
}
```

- `device` is `"auto"` (use a GPU if available), `"cuda"`, or `"cpu"`.
- JSON is strict: keys and text values need double quotes, there is no comma
  after the last item in a block, and comments are not allowed.
- In Colab, double-click a config in the file browser (left sidebar) to edit it,
  then save with Ctrl+S. Edits under `/content/` are lost when the runtime
  resets, so keep a copy of configs you care about in your Drive.

### The `.venv/` folder (local only)

`.venv/` is a Python *virtual environment*: a private folder holding its own
Python interpreter and the packages you install with `pip`. It keeps this
course's packages (and their exact versions) separate from the rest of your
system, so installing them cannot break other Python projects on your machine,
and vice versa.

- It does **not** come with the repository. You create it yourself once, at the
  repo root (see [Running locally](#running-locally)), and all projects share it.
- It is listed in `.gitignore`: it is large (several GB with PyTorch) and
  specific to your machine, so it is never committed.
- Activate it in every new terminal before running a project
  (`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows). Your
  prompt then starts with `(.venv)`. Run `deactivate` to leave it.
- Don't move or rename it: virtual environments store absolute paths and stop
  working when moved. If it breaks, delete the `.venv/` folder and create it
  again.
- In editors like VS Code or Cursor, select `.venv` as the Python interpreter
  so the editor uses the same packages.
- **Not needed in Colab**: Colab already provides its own Python environment.

### The `data/` folder

- Datasets are downloaded automatically the first time you run a project, then
  reused on later runs.
- **Locally**, all projects share one `data/` folder at the repo root, so a
  dataset used by several projects is downloaded only once (`DATA_DIR=data`
  in `.env`).
- **In Colab**, `.envcolab` uses `DATA_DIR=/content/data`, the fast local disk
  of the Colab machine. This disk is wiped when the runtime resets, so the
  dataset is re-downloaded each session (this only takes a few seconds for the
  datasets used here). See [Keeping your work in Google Drive](#keeping-your-work-in-google-drive)
  if you prefer to keep the dataset in Drive.
- `data/` is listed in `.gitignore`: datasets are never committed.

### The `runs/` folder

Results of all projects go into one `runs/` folder, separated by project name
and then by config name. Each time you run an experiment, a new folder named
after the current date and time is created, so no run ever overwrites another:

```
runs/
└── TrainingBasicConvnet/                  # One folder per project
    └── config01/                          # One folder per config name
        ├── runs_summary.csv               # One row per run: accuracy, lr, epochs, ...
        └── 2026-10-01_12-46-45/           # One folder per run (timestamp)
            ├── config.json                # Exact copy of the config used
            ├── history.json               # Loss and accuracy for every epoch
            ├── best_epoch09_valacc0.9243.pt   # Model state with the best validation accuracy
            ├── last_epoch10_valacc0.9193.pt   # Model state after the last epoch
            └── results_epoch09_valacc0.9243_testacc0.9186.json  # Final metrics
```

- **Locally**, results go to `runs/` at the repo root (`OUTPUT_DIR=runs` in
  `.env`).
- **In Colab**, results go to your Google Drive (see below).
- `runs/` is listed in `.gitignore`: your results are never committed.

Open `runs_summary.csv` to compare runs of the same config at a glance.

## Running in Google Colab

The easiest way is the ready-made notebook
[`starternotebook.ipynb`](starternotebook.ipynb). It sets everything up for you;
you only need to open it in Colab and run the cells from top to bottom.

### 1. Open the starter notebook in Colab

Either:

- **Open it directly from GitHub:** click
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/starternotebook.ipynb)
  (also at the top of this page), then **File → Save a copy in Drive** so your
  changes to the notebook are kept. This always opens the latest version of the
  notebook, or
- **Upload it:** download
  [`starternotebook.ipynb`](starternotebook.ipynb) from this repository, then in
  [Colab](https://colab.research.google.com) choose **File → Upload notebook**.
  Uploaded notebooks are saved automatically in the `Colab Notebooks` folder of
  your Google Drive.

The notebook asks Colab for a GPU. If it does not get one, open
**Runtime → Change runtime type** and select a GPU. Training works on CPU too,
just much slower.

### 2. Run the cells from top to bottom

| Cell | What it does |
| --- | --- |
| Mount Google Drive | Connects your Drive at `/content/drive` so results survive runtime resets. Colab asks you to allow access. |
| Setup (`%%bash`) | Clones the repo into `/content/CodeCS4337Fall2026` (or runs `git pull` if it is already there), copies `.envcolab` to `.env`, and installs only the missing requirements. |
| `%cd` | Moves the notebook into the repo folder so the projects can be imported. |
| Run | `from TrainingBasicConvnet import main` and `main("config01")` |
| Or run from the terminal | The same run as a shell command (optional, see below). |
| Results | Shows `runs_summary.csv` from your Drive. |

The setup cell is safe to run again at any time, for example to get new
projects after they are added. To run a different project, change
`PROJECT=TrainingBasicConvnet` in the setup cell and the import in the run cell.

Results are saved to
`/content/drive/MyDrive/CodeCS4337Fall2026/runs/<ProjectName>/<config>/<timestamp>/`.

### Running from the Colab terminal

You can also run a project as a terminal command in Colab, just like
[running locally](#3-run-the-experiment). The terminal runs on the same Colab
machine as the notebook, so it sees the same files, installed packages, and
Drive mount.

**First run the notebook's setup cells** (mount Drive, then the `%%bash` setup
cell). These steps are still needed, and Drive can only be mounted from a
notebook cell, not from the terminal.

Then open the terminal with the **Terminal** button at the bottom left of the
Colab window and run:

```bash
cd /content/CodeCS4337Fall2026
python -m TrainingBasicConvnet --config config01
```

or, from inside the project folder:

```bash
cd /content/CodeCS4337Fall2026/TrainingBasicConvnet
python main.py --config config01
```

You can also run the same command from a notebook cell by prefixing it with
`!` (the starter notebook has a cell for this):

```python
!cd /content/CodeCS4337Fall2026 && python -m TrainingBasicConvnet --config config01
```

The terminal is handy for long runs: you can keep using the notebook while
training runs. Results go to the same Drive folder either way. If the runtime
disconnects or resets, both the notebook and the terminal stop, and you need to
run the setup cells again.

### Why only the *missing* requirements are installed

Colab already comes with most packages (PyTorch, torchvision, NumPy, ...)
preinstalled. **Do not** run `pip install -r requirements.txt` in Colab: the
versions pinned there would replace Colab's own versions, which is slow and can
break other preinstalled packages.

The setup cell reads the project's `requirements.txt`, strips the version
numbers, and runs `pip install` only for packages that are not installed yet.
A package installed this way gets the latest version rather than the pinned
one, and a preinstalled package keeps Colab's version. These version
differences are expected and fine for this course.

### Without the notebook

If you prefer to set things up yourself in any Colab notebook, these cells do
the same as the starter notebook:

```python
from google.colab import drive

drive.mount("/content/drive")

!git clone https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026.git /content/CodeCS4337Fall2026
%cd /content/CodeCS4337Fall2026
!cp .envcolab .env
```

Then run the requirements loop from the setup cell of
[`starternotebook.ipynb`](starternotebook.ipynb), and finally:

```python
from TrainingBasicConvnet import main

main("config01")
```

### Keeping your work in Google Drive

Anything stored under `/content/` (outside `/content/drive/`) is **deleted**
when the Colab runtime disconnects or resets. That includes the cloned repo,
the dataset in `/content/data`, and any edits you made to configs.

- **Model checkpoints and results** are already saved to Drive by `.envcolab`
  (`OUTPUT_DIR=/content/drive/MyDrive/...`). Always mount Drive *before* calling
  `main()`. Otherwise the results are silently written to a local folder named
  `/content/drive/...` that is deleted with the runtime (and that folder then
  also prevents Drive from mounting until you remove it).
- **Datasets** are kept on the local disk by default because it is faster and
  they re-download quickly. To keep a dataset in Drive instead (useful for large
  datasets later in the course), edit `.env` at the repo root in Colab and
  set:

  ```
  DATA_DIR=/content/drive/MyDrive/CodeCS4337Fall2026/data
  ```

- **Your own configs or code changes**: save copies to Drive (or push them to
  your own GitHub fork), since the cloned repo disappears with the runtime.

## Running locally

### 1. Clone and create a virtual environment

You need Python 3.13. Create **one** virtual environment (`.venv/`, see
[The `.venv/` folder](#the-venv-folder-local-only)) at the repo root and reuse
it for every project:

```bash
git clone https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026.git
cd CodeCS4337Fall2026
python3.13 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install the project's requirements

Each project has its own `requirements.txt`, and **each project may need
different packages**. Install the requirements of every project you want to
run (locally, the pinned versions are the ones the code was tested with):

```bash
pip install -r TrainingBasicConvnet/requirements.txt
```

When a new project is added, install its requirements the same way.

### 3. Run the experiment

Either from inside the project folder:

```bash
cd TrainingBasicConvnet
python main.py --config config01
```

or from the repo root:

```bash
python -m TrainingBasicConvnet --config config01
```

`--config` accepts a name (`config01`), a file name (`config01.json`), or a
path (`configs/config01.json`). Results are written to
`runs/TrainingBasicConvnet/config01/<timestamp>/`, and the dataset is stored in
the shared `data/` folder, both at the repo root.

You can also call it from Python or a local Jupyter notebook started at the
repo root:

```python
from TrainingBasicConvnet import main

main("config01")
```

## About `ruff.toml` (you can ignore it for now)

[Ruff](https://docs.astral.sh/ruff/) is a tool that checks Python code for
common mistakes (unused imports, undefined names, ...) and formats it in a
consistent style. `ruff.toml` holds its settings for the whole repository.
You do **not** need Ruff to run any project.

Why it matters: consistent, lint-free code is easier to read, review, and
debug, and many bugs (like a typo in a variable name) are caught before you
even run the code. If you want to try it:

```bash
pip install ruff
ruff check .          # Report problems
ruff check . --fix    # Fix the ones that can be fixed automatically
ruff format .         # Reformat code in a consistent style
```

The only setting in `ruff.toml` allows CamelCase folder names such as
`TrainingBasicConvnet`, which Ruff would otherwise flag. We use CamelCase
names so each project can be imported as `from TrainingBasicConvnet import main`.
