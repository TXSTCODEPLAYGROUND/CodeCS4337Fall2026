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
wandb_logger = WandbLogger(project="LitWBTrainingBasicConvnet", config=config)
trainer = L.Trainer(logger=[csv_logger, wandb_logger], ...)
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

   **In Colab**, the setup cell overwrites `.env`, so store the key as a Colab
   *Secret* named `WANDB_API_KEY` instead (key icon in the left sidebar). Then
   run the **Load API keys** cell of
   [this project's notebook](lit_wb_training_basic_convnet_notebook.ipynb), after the setup cells
   and before running the project. It copies the key from your Secrets to the
   notebook's environment and to `.env`, so every way of running works: from
   the notebook (Option A1 or B) and from Colab's terminal (A2).

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
  after one epoch; [`callbacks/early_stopping.py`](callbacks/early_stopping.py)
  changes only that.

## Training with the Colab CLI

Another way to train this project, once everything is final. Its only purpose
is the training run itself:

1. **Work locally** on the project: change the code or a config, run short
   tests, fix bugs.
2. **Use Google Colab** (this project's notebook) for debugging and interactive work.
3. **Use the Colab CLI** when everything is final and the only thing left is
   training. The CLI creates a Colab runtime from your local terminal and opens
   a shell on it, so there is no notebook or browser tab to keep open. On the
   runtime you clone the repo and run `python -m LitWBTrainingBasicConvnet --config config01.json`.

The Colab CLI works on Linux and macOS (not Windows). For more detail, see
[google-colab-cli.md](../google-colab-cli.md) at the repo root or the
[official Colab CLI documentation](https://github.com/googlecolab/google-colab-cli).

### Step 1: Install the CLI

On your own machine:

```bash
uv tool install google-colab-cli     # or: pip install google-colab-cli
colab version                         # check that it works
```

The first command that talks to Colab prints a sign-in link: open it, sign in
with your Google account, and paste the code it shows back into the terminal.

### Step 2: The basic workflow (sessions)

A **session** is one Colab runtime (a virtual machine) with a name you choose,
here `cs4337`. These commands run in your **local** terminal:

| What | Command |
| --- | --- |
| Create a session with a GPU | `colab new -s cs4337 --gpu T4` (or `L4`, `A100`, `H100`, depending on your plan) |
| Create a session without a GPU | `colab new -s cs4337` (CPU only) |
| List your sessions / check one | `colab sessions` / `colab status -s cs4337` |
| Attach to a session (open a shell on it) | `colab ssh -s cs4337` |
| Leave the shell; the session keeps running | `exit` |
| Stop the session (delete the runtime) | `colab stop -s cs4337` |

Use a GPU runtime: `--gpu T4` is enough for this project. A CPU session can't be switched to a GPU: stop it and create a new one.

> **Once a session is stopped, nothing on it stays.** The cloned repo, the
> downloaded data, and every file that is not on Google Drive are deleted.
> This project writes its runs to Google Drive (step 4), so mount Drive before
> training.

### Step 3: Mount Google Drive

Drive is mounted from your local terminal, not from inside the session. If you
are attached, `exit` first:

```bash
exit                                # only if you are inside the session
colab drivemount -s cs4337          # open the link it prints and allow access
colab ssh -s cs4337                 # attach again
ls /content/drive/MyDrive           # check: your Drive files are listed
```

### Step 4: Clone the repo and train

Inside the session, clone the repo and go to its root folder:

```bash
cd /content
git clone https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026.git
cd CodeCS4337Fall2026
cp .envcolab .env
```

The runtime only sees what is on GitHub: if you changed code or a config
locally, push it to your own fork first and clone that instead. `.envcolab`
sets `DATA_DIR=/content/data` (data on the runtime's fast local disk) and
`OUTPUT_DIR=/content/drive/MyDrive/CodeCS4337Fall2026/runs` (runs on Drive, so they are kept after a stop).

This project logs to Weights & Biases and reads the key only from `.env`.
Add your W&B API key (from [wandb.ai/authorize](https://wandb.ai/authorize))
to the runtime's `.env`. `read -s` hides the key while you paste it, so it is
not shown on screen or saved in the shell history:

```bash
read -rsp "W&B API key: " key && echo "WANDB_API_KEY=$key" >> .env && unset key && echo
```

Without a key the run still trains, but W&B logs offline into the run folder;
upload it later with the `wandb sync` command that the run prints. Never put
the key in `.envcolab`, which is in git.

Install only the packages Colab doesn't already have, as the setup of this project's notebook
does. A plain `pip install -r requirements.txt` would replace Colab's own
versions, which is slow and can break other preinstalled packages:

```bash
grep -vE '^\s*(#|$)' requirements.txt \
  | sed -E 's/[<>=!~;[ ].*//' \
  | while read -r pkg; do
      if pip show "$pkg" > /dev/null 2>&1; then echo "skip $pkg (already installed)"
      else pip install -q "$pkg" && echo "installed $pkg"; fi
    done
```

Start a `tmux` session first, so training keeps going if the SSH connection
drops or you close your laptop, then run the training from the repo root:

```bash
tmux new -s train                  # if tmux is missing: apt-get install -y tmux
python -m LitWBTrainingBasicConvnet --config config01.json
```

To leave training running, press `Ctrl+B`, then `D`. To come back later:
`colab ssh -s cs4337`, then `tmux attach -t train`.

### Step 5: Check that everything was logged

The run folders are on Google Drive, so they are kept after the session
stops. From the session:

```bash
ls /content/drive/MyDrive/CodeCS4337Fall2026/runs/LitWBTrainingBasicConvnet/config01
```

or on [drive.google.com](https://drive.google.com), under **My Drive >
CodeCS4337Fall2026 > runs > LitWBTrainingBasicConvnet**. Check that:

- each run has its own `<timestamp>` folder (the path is printed when the run starts), with its config, results, plots, and checkpoints;
- `runs_summary.csv`, next to the run folders, has one new row per run;
- the run shows up in your W&B project: the run prints its W&B link. If it printed a `wandb sync` command instead, the run was logged offline (no key or no login); run that command later, from a machine with your key, to upload it;
- if the session stopped or the run was cut off, start a new session, repeat the steps above, and continue from the last checkpoint on Drive with `--resume-from config01`, e.g. `python -m LitWBTrainingBasicConvnet --config config01.json --resume-from config01`.

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
