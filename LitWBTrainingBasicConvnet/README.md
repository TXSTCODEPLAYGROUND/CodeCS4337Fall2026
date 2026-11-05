# LitWBTrainingBasicConvnet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBTrainingBasicConvnet/lit_wb_training_basic_convnet_notebook.ipynb)
This project's notebook: [`lit_wb_training_basic_convnet_notebook.ipynb`](lit_wb_training_basic_convnet_notebook.ipynb).

The same experiment as [LitTrainingBasicConvnet](../LitTrainingBasicConvnet/)
(the same ConvNet, LightningModule, DataModule, metrics, and settings), but
tracked with [Weights & Biases](https://wandb.ai) (W&B) instead of LitLogger
and hand-made plots. Read that project first: this README only explains what
is different.

What changes:

1. **Every metric goes to W&B** through Lightning's
   [`WandbLogger`](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html):
   loss, accuracy, precision, recall, and per-class accuracy, for training,
   validation, and test, plus the config of the run.
2. **No plotting code.** `utils/plots.py` is gone. The learning curves are
   W&B charts, and the example predictions are an interactive table.
3. **A small callback**, [`callbacks/wandb_predictions.py`](callbacks/wandb_predictions.py),
   logs a table of test images with their predicted and true class, and a
   confusion matrix.
4. **`metrics.csv` is still saved locally** (Lightning's `CSVLogger`), as a
   backup that works without an internet connection or an account.

This Lightning + W&B setup is the **approach you must follow** when training
your own models: start new projects from this one or from
[LitWBTrainingBasicNeuralNetwork](../LitWBTrainingBasicNeuralNetwork/).

## Boilerplate lives in `colabutilscs4337`

Setting up Colab, finding paths and configs, creating run folders, resuming, and
connecting to W&B were copied into every project. They now come from the
course library [`colabutilscs4337`](https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html), which is
in [`requirements.txt`](../requirements.txt), so this project keeps only what is
specific to the experiment: the data, the model, and the metrics.

| Was in this project | Now comes from `colabutilscs4337` |
| --- | --- |
| `utils/paths.py`: repo, `configs/`, `.env`, `DATA_DIR`, `OUTPUT_DIR` paths | `cu.project.Project` (`PROJECT` in [`utils/paths.py`](utils/paths.py)) |
| `utils/experiment.py`: `runs_summary.csv` | `cu.project.append_run_summary` |
| `models/loading.py`: finding a run, its checkpoints, and its config | `Project.find_checkpoint`, `find_resume_checkpoint`, `run_config` |
| `callbacks/early_stopping.py` | `cu.lit.ResumableEarlyStopping` |
| `main.py`: command line, resume, run folder, `CSVLogger`, checkpoints, `config.json` and `hparams.json` | `cu.lit.start_run` |
| `main.py`: `_connect_wandb` | `cu.lit.connect_wandb` |
| Notebook: mount Drive, clone, `.env`, requirements, API keys | `cu.project.setup_colab`, `cu.env.load_secrets`, `cu.env.sync_env_file` |

Read the library's source when you want to see how, for example, resuming works.

How to set up and run the project and where results go is explained in the
[main README](../README.md). Function and class details are in the
[API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitWBTrainingBasicConvnet.html).

## What is W&B?

[Weights & Biases](https://docs.wandb.ai) is a widely used experiment
tracker. Each training run sends its metrics to wandb.ai while it runs, so you
can watch the curves live, compare many runs in one chart, sort runs by any
metric or config value, and share a link to your results. It is free for
students and personal projects.

With Lightning, using W&B is one extra logger: every `self.log(...)` in
[`models/lit_convnet.py`](models/lit_convnet.py) is sent to W&B without any
change to the model.

```python
# cu.lit.connect_wandb returns Lightning's WandbLogger: online, or offline without a key.
wandb_logger = cu.lit.connect_wandb(
    project="LitWBTrainingBasicConvnet", name=..., group=..., save_dir=run.run_dir, config=config
)
trainer = L.Trainer(logger=[run.csv_logger, wandb_logger], ...)
```

## Set up your W&B key

1. Create a free account at [wandb.ai](https://wandb.ai/site).
2. Copy your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
   and add it to `.env` at the repo root:

   ```
   WANDB_API_KEY=your-api-key
   ```

   `.env` is not in git, so your key stays private. Never put it in
   `.envcolab` (which *is* in git), a config, or a notebook you share. If a key
   is ever pushed by mistake, delete it and create a new one on the same page.

   **In a Colab notebook**, the setup cell overwrites `.env`, so store the key as a
   Colab *Secret* named `WANDB_API_KEY` instead (key icon in the left sidebar).
   Then run the **Load API keys** cell of
   [this project's notebook](lit_wb_training_basic_convnet_notebook.ipynb), after the setup cells
   and before running the project. It calls
   `cu.env.load_secrets("WANDB_API_KEY", persist_to=REPO / ".env")`, which copies the
   key from your Secrets to the notebook's environment and to `.env`, so every way
   of running works: from the notebook (Option A1 or B) and from Colab's terminal (A2).

   **In a Colab CLI (SSH) session**, Colab Secrets are not available. Put the key in
   a text file on your Google Drive and give its path to `setup_colab`, as
   described in [Training with the Colab CLI](#training-with-the-colab-cli).

The run prints a link to its page on wandb.ai. The W&B project is set by
`"wandb": {"project": ...}` in the config; it is created in your account the
first time you run.

### No key or failed login: offline mode

A run never stops because of W&B. If `WANDB_API_KEY` is missing (or commented
out), or W&B cannot log in (a wrong key, no internet), the run prints a warning
and W&B logs **locally** instead: every metric, the config, the prediction
table, and the confusion matrix are saved in the run folder under
`wandb/offline-run-*`. `metrics.csv` and the checkpoints are written as usual.

To upload such a run to W&B later:

1. Put a valid key in `.env` (see [Set up your W&B key](#set-up-your-wb-key)).
2. From the repo root, with `.venv` activated, load `.env` into the terminal
   (`wandb sync` does not read `.env` by itself), then run the `wandb sync`
   command printed in the warning:

   ```bash
   set -a; source .env; set +a
   wandb sync runs/LitWBTrainingBasicConvnet/config01/<timestamp>/wandb/offline-run-*
   ```

   To upload every offline run of a config at once, use `*` for the timestamp:
   `runs/LitWBTrainingBasicConvnet/config01/*/wandb/offline-run-*`.

   In Colab, run the project notebook's **Load API keys** cell first, then the
   same two commands in Colab's terminal, or with `!` in a cell
   (`!wandb sync ...`; the key cell already set the key for `!` commands).
3. Open your project on wandb.ai: the run appears with all its charts, as if
   it had been online.

Run `wandb sync` on the same computer (or the same Colab session) as the
training: W&B keeps the images of the prediction table in its own cache there
until they are uploaded. After a failed login, the run's `wandb/` folder also
holds an empty `run-*` folder from the attempt; only `offline-run-*` needs
syncing.

## What is logged

| In W&B | Comes from | Replaces in LitTrainingBasicConvnet |
| --- | --- | --- |
| `train_loss`, `val_loss`, `test_loss` | `self.log` in `LitConvNet` | `loss.png` |
| `train_acc`, `val_acc`, `test_acc` | `self.log` (torchmetrics) | `accuracy.png` |
| `*_precision`, `*_recall` | `self.log` (torchmetrics) | (`metrics.csv` only) |
| `train_acc_<class>`, `val_acc_<class>` | `self.log` (torchmetrics) | `accuracy_per_class.png` |
| `test_predictions` table | `LogTestPredictions` callback | `predictions.png`, `wrong_predictions.png` |
| `confusion_matrix` | `LogTestPredictions` callback | (new) |
| Config (seed, lr, batch size, ...) | `WandbLogger(config=config)` | `config.json` |

The `test_predictions` table holds the first 1000 test images with columns
`image`, `predicted`, `true`, and `correct`. The confusion matrix covers the
whole test set: row = true class, column = predicted class, so everything off
the diagonal is a mistake, and you can see which classes get mixed up.

## Build the dashboard

W&B makes one chart per metric automatically. To compare training and
validation on the same graph, like the plots of LitTrainingBasicConvnet, add
a few panels once; the workspace keeps them for every later run of the
project.

1. Open your project on wandb.ai and go to **Workspace**.
2. **Loss**: click **Add panels** → **Line plot**. Set **X** to `epoch` and
   **Y** to `train_loss` and `val_loss`. Click **Apply**.
3. **Accuracy**: the same, with **Y** = `train_acc` and `val_acc`.
4. **Accuracy per class**: the same, but in **Y** switch on regex
   (the `.*` button) and enter `train_acc_.*|val_acc_.*`. This draws all 20
   per-class curves on one graph. Use `val_acc_.*` alone if it is too crowded.
5. **Wrong predictions**: open the `test_predictions` table panel, click
   **Filter**, and enter `row["correct"] = false`. Sort or group by `true` to
   see which classes the model gets wrong.
6. The `confusion_matrix` panel needs no setup.

Tip: Lightning logs `epoch` and `trainer/global_step` with every value; use
`epoch` as the X axis so one point is one epoch, as in `metrics.csv`.

To compare runs, train with a second config (e.g. `config02.json` with a
different `lr`): every chart then shows one line per run, and the **Runs**
table can sort runs by `val_acc` or any config value. Runs from the same
config file are grouped by name (`config01`, ...).

## Run folder

Each run writes these files to
`runs/LitWBTrainingBasicConvnet/<config>/<timestamp>/`:

```
├── config.json                     # Exact copy of the config used
├── hparams.json                    # Optimizer and data settings
├── metrics.csv                     # Every metric, every epoch (local backup)
├── best_epoch08_valacc0.9240.ckpt  # Checkpoint with the best validation accuracy
├── last.ckpt                       # Checkpoint of the last epoch, rewritten every epoch
└── wandb/                          # W&B's local copy of the run (offline runs are synced from here)
```

To load a trained model back, e.g. the best checkpoint of the newest
`config01` run, see
[Loading a trained model](../LitTrainingBasicConvnet/README.md#loading-a-trained-model):

```python
from LitWBTrainingBasicConvnet import load_model

model = load_model("config01")
```

To train a run further instead, resume it, as described in
[Continuing training](../LitTrainingBasicConvnet/README.md#continuing-training):
`main("config01.json", resume_from="config01")`, or
`--resume-from config01` on the command line. The continued run is a new W&B
run.

## Early stopping

`config01.json` turns on early stopping in its `"training"` section:

```json
"training": {
  "epochs": 10,
  "lr": 0.001,
  "weight_decay": 0.0,
  "early_stopping": true,
  "patience": 3
}
```

Training stops when `val_acc` has not improved for `patience` epochs in a
row, so `"epochs"` becomes the **most** epochs a run can take. Set
`"early_stopping": false` to always train all of them.

- The model that is tested is still the **best checkpoint** (the epoch with
  the highest `val_acc`), not the last epoch before stopping.
- The run prints after which epoch it stopped. The epochs it actually ran are
  in `runs_summary.csv` (`epochs_run`, next to the planned `epochs`) and in
  the W&B run's summary (`epochs_run`).
- **Resuming** a run that stopped early works: the epochs without improvement
  are counted from zero again, with the config's `patience`, and the
  continued run has to beat the earlier run's best `val_acc`. Lightning's own
  `EarlyStopping` would restore its counter from the checkpoint and stop again
  after one epoch; `ResumableEarlyStopping` from `colabutilscs4337.lit`
  changes only that.

## Training with the Colab CLI

Another way to train this project, once everything is final. Its only purpose
is the training run itself:

1. **Work locally** on the project: change the code or a config, run short
   tests, fix bugs.
2. **Use Google Colab** (this project's notebook) for debugging and interactive work.
3. **Use the Colab CLI** when everything is final and the only thing left is
   training. The CLI creates a Colab runtime from your local terminal and opens
   a shell on it, so there is no notebook or browser tab to keep open.

The Colab CLI works on Linux and macOS (not Windows). For more detail, see
[google-colab-cli.md](../google-colab-cli.md) at the repo root or the
[official Colab CLI documentation](https://github.com/googlecolab/google-colab-cli).

### Step 1: Install the CLI and sign in

On your own machine:

```bash
pip install google-colab-cli     # or: uv tool install google-colab-cli
colab version                    # check that it works
```

The first command that talks to Colab (`colab version`, `colab new`, …) prints a
sign-in link: open it, sign in with your Google account, and paste the code it
shows back into the terminal. You only need to do this once per machine.

### Step 2: Create `cs4337.env` on Google Drive

Colab Secrets can only be read from a notebook, not from an SSH session, so the
keys for this project live in a plain text file on your Google Drive. Create a
file named `cs4337.env` in the **root of My Drive** (so its path in the session
will be `/content/drive/MyDrive/cs4337.env`), with one `NAME=value` per line:

```text
# Settings shared by every project in this repo.
# Relative paths are resolved against the repo root, not the current directory.

# Required
# Each project writes to <OUTPUT_DIR>/<ProjectName>/<config>/<timestamp>/.
DATA_DIR=/content/data
OUTPUT_DIR=/content/drive/MyDrive/CodeCS4337Fall2026/runs

# Required
# W&B key for LitWBTrainingBasicConvnet, from wandb.ai/authorize (uncomment and paste it):
# WANDB_API_KEY=

# Optional
# LitLogger key for LitTrainingBasicConvnet, from lightning.ai/profile (uncomment and paste it):
LIGHTNING_USER_ID=
LIGHTNING_API_KEY=

# optional
# LABEL_STUDIO_USERNAME=email@email.com   # must look like an email address
# LABEL_STUDIO_PASSWORD=password
```

`setup_colab` writes `.env` from the repo's `.envcolab` and then appends this
file, so later lines win when a name appears twice. Uncomment and set
`WANDB_API_KEY` for this project. Without it the run still trains, but W&B
logs offline into the run folder; upload it later with the `wandb sync` command
the run prints. Never put keys in `.envcolab`, which is in git.

You can also create the file from inside the session after Drive is mounted
(step 4): `read -s` hides the key while you paste it:

```bash
read -rsp "W&B API key: " key && echo "WANDB_API_KEY=$key" >> /content/drive/MyDrive/cs4337.env && unset key && echo
```

### Step 3: Start a GPU session

These commands run in your **local** terminal. A **session** is one Colab
runtime (a virtual machine) with a name you choose, here `cs4337`:

```bash
colab new -s cs4337 --gpu T4
```

`--gpu T4` is enough for this project. Other GPUs (`L4`, `A100`, `H100`), CPU
sessions, high-memory sessions, listing and stopping sessions, and connecting
from VS Code are covered in [google-colab-cli.md](../google-colab-cli.md) and the
[official Colab CLI documentation](https://github.com/googlecolab/google-colab-cli):
read those when you want more than a plain GPU session.

> **Once a session is stopped, nothing on it stays.** The cloned repo, the
> downloaded data, and every file that is not on Google Drive are deleted.
> This project writes its runs to Google Drive (step 5), so mount Drive before
> training.

### Step 4: Mount Google Drive and attach

Drive is mounted from your local terminal, not from inside the session:

```bash
colab drivemount -s cs4337          # open the link it prints and allow access
colab ssh -s cs4337                 # attach to the session
```

Inside the session, check that the GPU is there and that Drive is mounted:

```bash
nvidia-smi                          # GPU name, memory, utilization
ls /content/drive/MyDrive           # your Drive files, including cs4337.env
```

Other monitoring tools (`btop`, `gpustat`, `tmux` shortcuts) are listed in the
[main README](../README.md#3-useful-monitoring-tools) under *Useful Monitoring
Tools*. Nothing in the project mounts Drive again: the setup in step 5 uses this
mount.

### Step 5: Set up the repo and train

One call of the course library
[`colabutilscs4337`](https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html) sets up everything the
runtime needs: it clones the repo (or pulls it, if it is already there), writes
`.env` (`.envcolab` with your env file added below it), and installs only the
packages Colab doesn't already have (see [Why only the missing requirements are installed](../README.md#why-only-the-missing-requirements-are-installed)).
Drive is already mounted from your local terminal, so nothing is mounted again.
Inside the session:

```bash
pip install -q colabutilscs4337
python - <<'EOF'
import colabutilscs4337 as cu

cu.project.setup_colab(
    "https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026.git",
    env_source="/content/drive/MyDrive/cs4337.env",  # your env file on Google Drive
    mount=False,  # `colab drivemount` already mounted Drive
)
EOF
cd /content/CodeCS4337Fall2026
```

The runtime only sees what is on GitHub: if you changed code or a config
locally, push it to your own fork first and use that URL instead. If
`cs4337.env` is not on Drive, `setup_colab` stops with an error that names the
path it looked for.

Start a `tmux` session so training keeps going if the SSH connection drops or
you close your laptop, then run from the repo root:

```bash
tmux new -s train                  # if tmux is missing: apt-get install -y tmux
python -m LitWBTrainingBasicConvnet --config config01.json
```

To leave training running, press `Ctrl+B`, then `D` (detach). The process keeps
going on the runtime. To come back later: `colab ssh -s cs4337`, then
`tmux attach -t train`.

### Step 6: Monitor the run

While training (or after you detach from `tmux`):

- **W&B:** open the run link that the process prints, or go to your W&B project
  in the browser. Metrics, the config, and the system charts update there live
  when the key was present. If the run printed a `wandb sync …` command instead,
  it logged offline; run that command later from a machine with your key.
- **Run folder on Google Drive:** every file of the run (config, metrics,
  checkpoints, W&B's local `wandb/` folder) is under

  ```text
  /content/drive/MyDrive/CodeCS4337Fall2026/runs/LitWBTrainingBasicConvnet/config01/2026-11-05_14-30-00/
  ```

  The timestamp is printed when the run starts. From the session:

  ```bash
  ls /content/drive/MyDrive/CodeCS4337Fall2026/runs/LitWBTrainingBasicConvnet/config01
  ```

  or on [drive.google.com](https://drive.google.com), under **My Drive >
  CodeCS4337Fall2026 > runs > LitWBTrainingBasicConvnet**. Check that `runs_summary.csv` next to
  the run folders has one new row per run.

- **Resume a cut-off run:** start a new session, repeat the steps above, and
  continue from the last checkpoint on Drive:

  ```bash
  python -m LitWBTrainingBasicConvnet --config config01.json --resume-from config01
  ```

When everything is checked, stop the session so it no longer uses your Colab
quota:

```bash
colab stop -s cs4337
```

## Things to try

- Compare [`main.py`](main.py) with
  [`LitTrainingBasicConvnet/main.py`](../LitTrainingBasicConvnet/main.py) and
  `utils/plots.py` there: count how much code W&B replaces.
- Add a `config02.json` with more dropout or weight decay, run both, and
  compare the `val_loss` curves in one chart.
- Raise `"epochs"` to 30 and try `"patience"` 1, 3, and 5: how many epochs
  does each run take, and does a longer patience reach a better `test_acc`?
- In the confusion matrix, find the class most often confused with *Shirt*.
- Log something new: `self.log("lr", ...)` in `LitConvNet` appears in W&B
  without any other change.
