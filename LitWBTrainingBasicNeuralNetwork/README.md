# LitWBTrainingBasicNeuralNetwork

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBTrainingBasicNeuralNetwork/lit_wb_training_basic_neural_network_notebook.ipynb)
This project's notebook: [`lit_wb_training_basic_neural_network_notebook.ipynb`](lit_wb_training_basic_neural_network_notebook.ipynb).

The fully connected network (MLP) of
[LitTrainingBasicNeuralNetwork](../LitTrainingBasicNeuralNetwork/), tracked
with [Weights & Biases](https://wandb.ai) (W&B) exactly like
[LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/): live charts, a
table of test predictions, and a confusion matrix on wandb.ai instead of
plotting code, with `metrics.csv` kept locally as a backup. After 10 epochs
the test accuracy is about 88.6%.

This Lightning + W&B setup is the **approach you must follow** when training
your own models: start new projects from this one or from
[LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/).

Read the projects it combines first:

- [TrainingBasicNeuralNetwork](../TrainingBasicNeuralNetwork/README.md)
  explains the network: fully connected layers, ReLU, dropout, and how to
  count parameters.
- [LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/README.md)
  explains W&B: setting up your key (in `.env`, or Colab Secrets with the
  project notebook's **Load API keys** cell), offline mode and `wandb sync`,
  what is logged, and how to build the dashboard. All of it applies here
  unchanged.

How to set up and run the project and where results go is explained in the
[main README](../README.md). Function and class details are in the
[API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitWBTrainingBasicNeuralNetwork.html).

## What is different from LitWBTrainingBasicConvnet

Only the network, as in the Lightning projects without W&B:

| | LitWBTrainingBasicConvnet | This project |
| --- | --- | --- |
| Network | `models/components/convnet.py`: `ConvNet` | [`models/components/mlp.py`](models/components/mlp.py): `MLP` |
| LightningModule | `models/lit_convnet.py`: `LitConvNet` | [`models/lit_mlp.py`](models/lit_mlp.py): `LitMLP` (same code, renamed) |
| `"model"` in the config | `{"dropout": 0.25}` | `{"hidden_sizes": [256, 128], "dropout": 0.2}` |
| W&B project | `LitWBTrainingBasicConvnet` | `LitWBTrainingBasicNeuralNetwork` |
| Parameters | about 422,000 | 235,146 |

The W&B logger, the offline fallback, the prediction-table callback, and the
local backup are the same code. The run prints the number of parameters at
the start and also sends it to W&B as `num_params` in the run's config, next
to `model.hidden_sizes` and `model.dropout`. `runs_summary.csv` gets the
columns `hidden_sizes` and `num_params`, and `hparams.json` a `"net"` section.

## Comparing network sizes in W&B

Because the network size is in each run's config, W&B can compare
architectures without any extra code:

1. Make a few copies of the config with different `"hidden_sizes"`, e.g.
   `[64]`, `[256, 128]`, and `[512, 256, 128]`, and run each one.
2. In the project's **Runs** table, show the `model.hidden_sizes` and
   `num_params` columns and sort by `test_acc`: does a bigger network always
   win?
3. Add a **Scatter plot** panel with X = `num_params` and Y = `test_acc` to
   see accuracy against network size.
4. Overlay `train_acc` and `val_acc` (see the dashboard steps in the
   [LitWBTrainingBasicConvnet README](../LitWBTrainingBasicConvnet/README.md#build-the-dashboard))
   to see which sizes overfit: the training curve keeps rising while
   validation flattens.

To compare against the ConvNet, run LitWBTrainingBasicConvnet too: the two
projects log the same metric names, so their runs can be compared side by
side in a W&B report.

## Run folder

The same files as in
[LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/README.md#run-folder),
in `runs/LitWBTrainingBasicNeuralNetwork/<config>/<timestamp>/`. To load a
trained model back, e.g. the best checkpoint of the newest `config01` run, see
[Loading a trained model](../LitTrainingBasicConvnet/README.md#loading-a-trained-model):

```python
from LitWBTrainingBasicNeuralNetwork import load_model

model = load_model("config01")
```

To train a run further instead, resume it, as described in
[Continuing training](../LitTrainingBasicConvnet/README.md#continuing-training):
`main("config01.json", resume_from="config01")`, or
`--resume-from config01` on the command line. The continued run is a new W&B
run.

## Training with the Colab CLI

Another way to train this project, once everything is final. Its only purpose
is the training run itself:

1. **Work locally** on the project: change the code or a config, run short
   tests, fix bugs.
2. **Use Google Colab** (this project's notebook) for debugging and interactive work.
3. **Use the Colab CLI** when everything is final and the only thing left is
   training. The CLI creates a Colab runtime from your local terminal and opens
   a shell on it, so there is no notebook or browser tab to keep open. On the
   runtime you clone the repo and run `python -m LitWBTrainingBasicNeuralNetwork --config config01.json`.

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

This network is small: a CPU runtime trains it fine, and a T4 GPU makes it faster. A CPU session can't be switched to a GPU: stop it and create a new one.

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
python -m LitWBTrainingBasicNeuralNetwork --config config01.json
```

To leave training running, press `Ctrl+B`, then `D`. To come back later:
`colab ssh -s cs4337`, then `tmux attach -t train`.

### Step 5: Check that everything was logged

The run folders are on Google Drive, so they are kept after the session
stops. From the session:

```bash
ls /content/drive/MyDrive/CodeCS4337Fall2026/runs/LitWBTrainingBasicNeuralNetwork/config01
```

or on [drive.google.com](https://drive.google.com), under **My Drive >
CodeCS4337Fall2026 > runs > LitWBTrainingBasicNeuralNetwork**. Check that:

- each run has its own `<timestamp>` folder (the path is printed when the run starts), with its config, results, plots, and checkpoints;
- `runs_summary.csv`, next to the run folders, has one new row per run;
- the run shows up in your W&B project: the run prints its W&B link. If it printed a `wandb sync` command instead, the run was logged offline (no key or no login); run that command later, from a machine with your key, to upload it;
- if the session stopped or the run was cut off, start a new session, repeat the steps above, and continue from the last checkpoint on Drive with `--resume-from config01`, e.g. `python -m LitWBTrainingBasicNeuralNetwork --config config01.json --resume-from config01`.

When everything is checked, stop the session so it no longer uses your Colab
quota:

```bash
colab stop -s cs4337
```
