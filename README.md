# CS4337 Fall 2026 — Course Code

This repository holds the code for CS4337, Fall 2026. Each top-level folder
(for example `TrainingBasicNeuralNetwork/`) is a **self-contained project**: a small
Python package that you can either

- import and run from a notebook (e.g. Google Colab), or
- run from the terminal on your own machine.

New projects will be added over the semester. To get them, see
[Getting updates](#getting-updates).

Function and class details are in the
[documentation](https://txstcodeplayground.github.io/CodeCS4337Fall2026/).

## Projects

The projects come in two series, one per network. Each series goes through
the same three steps: plain PyTorch, then PyTorch Lightning, then Lightning
with Weights & Biases tracking. A third section then searches for good
hyperparameters automatically.

| Project | Description |
| --- | --- |
| **Fully connected network (MLP)** | |
| [TrainingBasicNeuralNetwork](TrainingBasicNeuralNetwork/) | Start here: train a fully connected neural network (only `nn.Linear` layers, no convolutions) on Fashion-MNIST in plain PyTorch, and see what layers, activations, dropout, and parameters are. [API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/TrainingBasicNeuralNetwork.html) · [Colab notebook](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/TrainingBasicNeuralNetwork/training_basic_neural_network_notebook.ipynb) |
| [LitTrainingBasicNeuralNetwork](LitTrainingBasicNeuralNetwork/) | The same network trained with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/): the Lightning-Hydra-Template project layout, torchmetrics (precision, recall, per-class accuracy), learning-curve and prediction plots, and optional LitLogger tracking. [API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitTrainingBasicNeuralNetwork.html) · [Colab notebook](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitTrainingBasicNeuralNetwork/lit_training_basic_neural_network_notebook.ipynb) |
| [LitWBTrainingBasicNeuralNetwork](LitWBTrainingBasicNeuralNetwork/) | **Recommended pattern.** The Lightning MLP tracked with [Weights & Biases](https://wandb.ai): live charts, a test-prediction table, and a confusion matrix on wandb.ai, plus the network size (`num_params`) for comparing architectures. [API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitWBTrainingBasicNeuralNetwork.html) · [Colab notebook](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBTrainingBasicNeuralNetwork/lit_wb_training_basic_neural_network_notebook.ipynb) |
| **Convolutional network (ConvNet)** | |
| [TrainingBasicConvnet](TrainingBasicConvnet/) | Train a basic ConvNet on Fashion-MNIST with a training loop written in plain PyTorch, and organize the code into modules and packages. [API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/TrainingBasicConvnet.html) · [Colab notebook](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/TrainingBasicConvnet/training_basic_convnet_notebook.ipynb) |
| [LitTrainingBasicConvnet](LitTrainingBasicConvnet/) | The same ConvNet trained with PyTorch Lightning; its README explains the Lightning project layout used by all `Lit...` projects. [API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitTrainingBasicConvnet.html) · [Colab notebook](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitTrainingBasicConvnet/lit_training_basic_convnet_notebook.ipynb) |
| [LitWBTrainingBasicConvnet](LitWBTrainingBasicConvnet/) | **Recommended pattern.** The Lightning ConvNet tracked with W&B, with an offline mode when there is no key or the login fails; its README explains the W&B setup and dashboard used by all `LitWB...` projects. [API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitWBTrainingBasicConvnet.html) · [Colab notebook](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBTrainingBasicConvnet/lit_wb_training_basic_convnet_notebook.ipynb) |
| **Hyperparameter search (Optuna)** | |
| [LitWBHSTrainingBasicNeuralNetwork](LitWBHSTrainingBasicNeuralNetwork/) | LitWBTrainingBasicNeuralNetwork plus a hyperparameter search with [Optuna](https://optuna.org): network size, activation, dropout, batch size, epochs, optimizer, learning rate, L1/L2 regularization, learning-rate scheduler, and early stopping. The best trial is saved as a config to train and log to W&B. [API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitWBHSTrainingBasicNeuralNetwork.html) · [Colab notebook](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBHSTrainingBasicNeuralNetwork/lit_wbhs_training_basic_neural_network_notebook.ipynb) |

The `LitWB...` projects are the **recommended pattern** for your own
experiments: Lightning removes the training-loop boilerplate, and W&B records
every run (metrics, config, predictions) so runs can be compared without
writing plotting code. The other projects build up to it step by step.

Each project folder has its own `README.md` explaining what the project is
about and what it teaches. This README covers what all projects share: setup,
running, configs, and where data and results go.

## Repository structure

```
CodeCS4337Fall2026/
├── README.md              # This file
├── requirements.txt       # Python packages for every project (one shared .venv)
├── starternotebook.ipynb  # General Colab notebook for any project (each project also has its own)
├── ruff.toml              # Code style settings (you can ignore this for now, see below)
├── .gitignore             # Files git should not track (data, runs, .env, virtualenvs, ...)
├── .env                   # Local settings for all projects: where data and results go (not in git)
├── .envcolab              # Google Colab settings for all projects (copied to .env in Colab)
├── .venv/                 # Your local Python virtual environment (you create it, not in git)
├── data/                  # Datasets shared by all projects (created on first run, not in git)
├── runs/                  # Experiment results, one subfolder per project (not in git)
├── docs/                  # Source of the online documentation (built automatically, see below)
├── TrainingBasicNeuralNetwork/  # One project = one Python package
│   ├── README.md          # What this project is about
│   ├── training_basic_neural_network_notebook.ipynb  # This project's Colab notebook
│   ├── __init__.py        # Makes the folder importable: from TrainingBasicNeuralNetwork import main
│   ├── __main__.py        # Makes `python -m TrainingBasicNeuralNetwork` work
│   ├── main.py            # Entry point: main(config) runs a full experiment
│   ├── configs/           # Experiment settings (config01.json, ...)
│   ├── dataloaders/       # Dataset loading
│   ├── models/            # Network definitions (models/mlp.py)
│   ├── trainers/          # Training and evaluation loop
│   └── utils/             # Helpers (paths, seeding, run folders)
├── LitTrainingBasicNeuralNetwork/  # Same experiment with PyTorch Lightning
│   ├── README.md, *_notebook.ipynb, __init__.py, __main__.py, main.py, configs/  # Same roles as above
│   ├── models/            # LightningModule, and plain networks in models/components/
│   ├── dataloaders/       # LightningDataModule: download, split, batch
│   └── utils/             # Helpers (paths, run summaries, plots)
├── LitWBTrainingBasicNeuralNetwork/  # Same Lightning experiment, tracked with W&B (recommended)
│   ├── ...                # Same as LitTrainingBasicNeuralNetwork, without utils/plots.py
│   └── callbacks/         # Logs test predictions and a confusion matrix to W&B
├── TrainingBasicConvnet/      # Same three steps with a ConvNet (models/convnet.py)
├── LitTrainingBasicConvnet/   # Same layout as LitTrainingBasicNeuralNetwork
├── LitWBTrainingBasicConvnet/ # Same layout as LitWBTrainingBasicNeuralNetwork (recommended)
└── LitWBHSTrainingBasicNeuralNetwork/  # LitWBTrainingBasicNeuralNetwork plus an Optuna search
    ├── ...                    # Same layout as LitWBTrainingBasicNeuralNetwork
    ├── search.py              # Second entry point: search(config) runs the Optuna study
    ├── configs/search01.json  # Search space and study settings
    └── lit_wbhs_training_basic_neural_network_notebook.ipynb  # Colab notebook: search, plots, train the best config
```

Every project has the same entry points (`main.py`, `configs/`) and is run the
same way. Each project folder also has its own Colab notebook, named after the
project (`lit_wb_training_basic_convnet_notebook.ipynb` for
`LitWBTrainingBasicConvnet`, ...). Lightning projects have no `trainers/` folder, because Lightning's
`Trainer` replaces the hand-written training loop; their layout is explained
in the [LitTrainingBasicConvnet README](LitTrainingBasicConvnet/README.md).

### The `.env` and `.envcolab` files

One settings file at the repo root is shared by every project:

- `DATA_DIR`: where datasets are stored.
- `OUTPUT_DIR`: where experiment results are written.
- `LIGHTNING_USER_ID` and `LIGHTNING_API_KEY` (optional): your Lightning AI
  keys, only needed for LitLogger in the `LitTraining...` projects (see the
  [LitTrainingBasicConvnet README](LitTrainingBasicConvnet/README.md#litlogger-optional)).
- `WANDB_API_KEY` (optional): your Weights & Biases key, used by
  the `LitWB...` projects (see the
  [LitWBTrainingBasicConvnet README](LitWBTrainingBasicConvnet/README.md#set-up-your-wb-key)). Without
  it, or if the login fails, W&B logs locally and the run can be uploaded later.

Add keys to `.env` only, never to `.envcolab`, which is tracked by git. In
Colab, store them as Colab Secrets instead and run the project notebook's
**Load API keys** cell (see [Running in Google Colab](#running-in-google-colab)).

Relative paths (like `data` or `runs`) are resolved against the repo root, so
it does not matter which folder you run from. Absolute paths are used as-is.

- **`.env`** holds the local settings (`DATA_DIR=data`, `OUTPUT_DIR=runs`). It
  is not tracked by git, so you can change it freely. If it is missing (for
  example after a fresh clone), the same defaults are used.
- **`.envcolab`** holds the Colab settings. In Colab it is copied to `.env`
  (the project notebooks do this for you, see
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

- `device` is `"auto"` (use the best available: an NVIDIA GPU, then an Apple
  Silicon GPU, then the CPU), `"cuda"`, `"mps"` (Apple Silicon GPU), or `"cpu"`.
  Lightning projects use Lightning's name for this field, `"accelerator"`, with
  the values `"auto"`, `"gpu"`, `"mps"`, or `"cpu"`.
- `model` holds the settings of the project's network, so it differs between
  projects. TrainingBasicNeuralNetwork, for example, also sets the hidden
  layers: `"model": {"hidden_sizes": [256, 128], "dropout": 0.2}`.
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

Lightning projects use the same folders but write their own files into each
run (metrics, checkpoints, and plots); see the
[LitTrainingBasicConvnet README](LitTrainingBasicConvnet/README.md#run-folder).

- **Locally**, results go to `runs/` at the repo root (`OUTPUT_DIR=runs` in
  `.env`).
- **In Colab**, results go to your Google Drive (see below).
- `runs/` is listed in `.gitignore`: your results are never committed.

Open `runs_summary.csv` to compare runs of the same config at a glance.

## Running in Google Colab

The easiest way is the project's own notebook: every project folder has one,
already set up for that project, so there is nothing to change or copy. It
sets everything up for you; you only need to open it in Colab and run the
cells from top to bottom.

### 1. Open the project's notebook in Colab

| Project | Notebook |
| --- | --- |
| TrainingBasicNeuralNetwork | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/TrainingBasicNeuralNetwork/training_basic_neural_network_notebook.ipynb) |
| LitTrainingBasicNeuralNetwork | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitTrainingBasicNeuralNetwork/lit_training_basic_neural_network_notebook.ipynb) |
| LitWBTrainingBasicNeuralNetwork | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBTrainingBasicNeuralNetwork/lit_wb_training_basic_neural_network_notebook.ipynb) |
| TrainingBasicConvnet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/TrainingBasicConvnet/training_basic_convnet_notebook.ipynb) |
| LitTrainingBasicConvnet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitTrainingBasicConvnet/lit_training_basic_convnet_notebook.ipynb) |
| LitWBTrainingBasicConvnet | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBTrainingBasicConvnet/lit_wb_training_basic_convnet_notebook.ipynb) |
| LitWBHSTrainingBasicNeuralNetwork | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBHSTrainingBasicNeuralNetwork/lit_wbhs_training_basic_neural_network_notebook.ipynb) (search, plots, and training the best config) |

Click the badge, then **File → Save a copy in Drive** so your changes to the
notebook are kept. The badge always opens the latest version of the notebook.
You can also download a notebook from the project folder and, in
[Colab](https://colab.research.google.com), choose **File → Upload notebook**;
uploaded notebooks are saved in the `Colab Notebooks` folder of your Google
Drive.

[`starternotebook.ipynb`](starternotebook.ipynb) at the repo root
([open in Colab](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/starternotebook.ipynb))
is a general version of the same notebook: it works for any project, but you
change the project name in its run cells yourself.

The notebook asks Colab for a GPU. If it does not get one, open
**Runtime → Change runtime type** and select a GPU. Training works on CPU too,
just much slower.

### 2. Run the cells from top to bottom

The notebook has three parts:

| Part | What it does |
| --- | --- |
| **1. Setup** | Mounts Google Drive (Colab asks you to allow access), clones the repo into `/content/CodeCS4337Fall2026` (or runs `git pull` if it is already there), copies `.envcolab` to `.env`, installs only the missing requirements, and moves the notebook into the repo folder. In the projects that use a tracker, the optional **Load API keys** cell then loads `.env` and copies the project's missing keys (`WANDB_API_KEY` for `LitWB...`, `LIGHTNING_USER_ID` and `LIGHTNING_API_KEY` for `LitTraining...`) from your Colab Secrets into the environment and `.env`, printing which keys were found. Run it before the project, from the notebook or from Colab's terminal. |
| **2. Run the project** | Choose **Option A** (terminal command) or **Option B** (from Python), see below. |
| **3. Results** | Shows `runs_summary.csv` from your Drive. |

The notebooks also run on your own machine: open the notebook from its
project folder with the `.venv` as its kernel, skip the Colab-only part 1, and
start at part 2, whose first cell finds the repo folder in Colab and locally.

It ends with [pro tips](#pro-tips-keep-training-running) for keeping long
training runs alive.

The setup cells are safe to run again at any time, for example to get new
projects after they are added (the setup cell rewrites `.env`, so run the
API-key cell again after it). The setup cell installs the packages for every
project. To run a different project, open that project's notebook.

Results are saved to
`/content/drive/MyDrive/CodeCS4337Fall2026/runs/<ProjectName>/<config>/<timestamp>/`.

### Option A: run as a terminal command

This is the same command you would use on your own machine (see
[Running locally](#3-run-the-experiment)). There are two ways to run it in
Colab:

**A1. From a notebook cell (everyone).** A line starting with `!` is sent to
the shell:

```python
!cd /content/CodeCS4337Fall2026 && python -m TrainingBasicConvnet --config config01.json
```

**A2. From Colab's terminal (Colab Pro).** Open the terminal with the
**Terminal** button at the bottom left of the Colab window and run:

```bash
cd /content/CodeCS4337Fall2026
python -m TrainingBasicConvnet --config config01.json
```

The terminal runs on the same Colab machine as the notebook, so it sees the
same files, installed packages, and Drive mount. **Run the notebook's setup
cells first**: Drive can only be mounted from a notebook cell, not from the
terminal. The same goes for Colab Secrets: if a project needs an API key, run
the **Load API keys** cell too, which saves the keys to `.env` where the
terminal finds them. While training runs in the terminal, the notebook stays
free.

### Option B: run from Python

```python
from TrainingBasicConvnet import main

main("config01.json")
```

Both options run the same training and save results to the same Drive folder.

### Pro tips: keep training running

Colab disconnects runtimes that look idle (no interaction for a while, often
around 90 minutes) and caps the total runtime (around 12 hours on the free
tier, longer on paid plans). The exact limits vary and are not guaranteed. A
disconnect stops training and deletes everything under `/content/` outside
your Drive.

**Save your state in Google Drive**

- Always mount Drive *before* training, so results go straight to Drive.
- The best model so far (`best_epochXX_valaccY.pt`) is saved to Drive
  *during* training, every time validation accuracy improves. If the runtime
  disconnects mid-run, that checkpoint is kept.
- The last checkpoint, `history.json`, the results file, and the row in
  `runs_summary.csv` are written only when the run finishes, so an interrupted
  run has only its best checkpoint.
- Keep your notebook and any configs you edited in Drive too.

**Avoid idle disconnects**

- Keep the Colab tab open and don't let your computer sleep (plug in your
  laptop and turn off sleep while training).
- Check on the notebook now and then. Running in the terminal does *not*
  prevent disconnects: it shares the same runtime.
- Don't use scripts or extensions that fake activity (auto-clickers); they can
  get your Colab usage restricted.
- With Colab Pro+, *background execution* keeps the runtime alive after you
  close the browser.

**Plan your runs**

- Test a new config with few epochs first (e.g. `"epochs": 1`), then start the
  full run.
- Prefer several shorter runs over one very long run.
- After a disconnect, re-run the setup cells (and the **Load API keys** cell,
  if you use W&B or LitLogger), then start the run again.
- When done, use *Runtime → Disconnect and delete runtime* to save your GPU
  quota (or compute units on paid plans).

### Why only the *missing* requirements are installed

Colab already comes with most packages (PyTorch, torchvision, NumPy, ...)
preinstalled. **Do not** run `pip install -r requirements.txt` in Colab: the
versions pinned there would replace Colab's own versions, which is slow and can
break other preinstalled packages.

The setup cell reads `requirements.txt` at the repo root, strips the version
numbers, and runs `pip install` only for packages that are not installed yet.
A package installed this way gets the latest version rather than the pinned
one, and a preinstalled package keeps Colab's version. These version
differences are expected and fine for this course.

### Without the notebook

If you prefer to set things up yourself in any Colab notebook, these cells do
the same as the project notebooks:

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

main("config01.json")
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

### 2. Install the requirements

One `requirements.txt` at the repo root lists the packages for **every**
project, at the versions the code was tested with. All projects share the same
`.venv`, so one install covers them all:

```bash
pip install -r requirements.txt
```

When projects are added later, run this again after `git pull` (see
[Getting updates](#getting-updates)).

### 3. Run the experiment

From the repo root, use `python -m` and the project name:

```bash
python -m TrainingBasicConvnet --config config01.json
python -m LitTrainingBasicConvnet --config config01.json
```

or, from inside a project folder:

```bash
cd TrainingBasicConvnet
python main.py --config config01.json
```

Both do the same thing. The projects are packages, so when `main.py` is run
directly it puts the repo root on the import path and reruns itself as
`python -m <Project>`. Outputs and data still go to the repo root.

`--config` accepts a name (`config01`), a file name (`config01.json`), or a
path (`configs/config01.json`). Results are written to
`runs/TrainingBasicConvnet/config01/<timestamp>/`, and the dataset is stored in
the shared `data/` folder, both at the repo root.

The first line of output shows which device is used (e.g. `device: cuda`;
Lightning projects print `GPU available: True (cuda), used: True`). With the
default `"device": "auto"` (`"accelerator": "auto"` in Lightning projects):

- **NVIDIA GPU (Linux/Windows):** uses CUDA if your PyTorch install supports
  it. On Windows, the default `pip` install of PyTorch is CPU-only; for GPU
  support, use the install command from
  [pytorch.org](https://pytorch.org/get-started/locally/).
- **Mac with Apple Silicon (M1/M2/M3/M4):** uses the Apple GPU (`mps`) with the
  regular `pip` install. This needs macOS 12.3 or newer and a native (arm64)
  Python. If you hit an error saying an operation is not implemented for MPS,
  run with `PYTORCH_ENABLE_MPS_FALLBACK=1` set, or set `"device": "cpu"`.
- **Otherwise:** trains on the CPU. That is fine for this project, just slower.

You can also call a project from Python or a local Jupyter notebook started at
the repo root:

```python
from TrainingBasicConvnet import main          # or: from LitTrainingBasicConvnet import main

main("config01.json")
```

## Getting updates

New projects and fixes are added during the semester. To update your local
copy, run from the repo root:

```bash
git pull
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run `pip install -r requirements.txt` after **every** `git pull`. It installs
packages that new projects need and switches any changed versions to the pinned
ones. Packages that are already correct are left alone, so it finishes quickly
when nothing changed.

- **If `git pull` refuses to update** because you edited a file that is in the
  repo (for example `configs/config01.json`): keep your own settings in a new
  file such as `config02.json` instead, since `git pull` never touches files you
  created. To keep an edit you already made, run `git stash`, then `git pull`,
  then `git stash pop`.
- `data/`, `runs/`, and `.env` are not in git, so pulling never changes your
  datasets, results, or local settings.
- **In Colab**, just re-run the setup cells of your project's notebook. They run
  `git pull` and install any missing packages. The setup cell rewrites `.env`,
  so run the **Load API keys** cell again afterwards if you use API keys.

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

`ruff.toml` allows CamelCase folder names such as `TrainingBasicConvnet`,
which Ruff would otherwise flag. We use CamelCase names so each project can be
imported as `from TrainingBasicConvnet import main`. It also allows imports
below the short block at the top of each `main.py` that makes
`python main.py` work from inside a project folder.
