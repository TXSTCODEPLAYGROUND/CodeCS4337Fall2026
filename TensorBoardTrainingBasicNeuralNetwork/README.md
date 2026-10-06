# TensorBoardTrainingBasicNeuralNetwork

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/TensorBoardTrainingBasicNeuralNetwork/tensorboard_training_basic_neural_network_notebook.ipynb)
This project's notebook: [`tensorboard_training_basic_neural_network_notebook.ipynb`](tensorboard_training_basic_neural_network_notebook.ipynb).

[TrainingBasicNeuralNetwork](../TrainingBasicNeuralNetwork/) with
[TensorBoard](https://www.tensorflow.org/tensorboard) logging: the same fully
connected network (MLP), trained on
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) with the
same plain-PyTorch training loop, but every run also writes its training
curves, sample images, network graph, and settings for TensorBoard, a
dashboard that runs locally or inside a notebook. No account is needed, and
nothing leaves your machine (or your Google Drive, in Colab).

> **For learning only.** This project shows TensorBoard, which you will meet
> in other code. To train your own models, follow the Lightning + W&B
> approach of [LitWBTrainingBasicNeuralNetwork](../LitWBTrainingBasicNeuralNetwork/)
> and [LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/).

The network, its layers, and its parameter count are explained in the
[TrainingBasicNeuralNetwork README](../TrainingBasicNeuralNetwork/README.md).
How to set up, run, and configure the project, and where results go, is
explained in the [main README](../README.md). Function and class details are
in the [API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/TensorBoardTrainingBasicNeuralNetwork.html).

```bash
python -m TensorBoardTrainingBasicNeuralNetwork --config config01.json
tensorboard --logdir runs/TensorBoardTrainingBasicNeuralNetwork   # then open http://localhost:6006
```

## What is logged

Each run writes TensorBoard **event files** (`events.out.tfevents.*`) into
its own run folder, `<OUTPUT_DIR>/TensorBoardTrainingBasicNeuralNetwork/<config>/<timestamp>/`,
next to its checkpoints and results. Point TensorBoard at the project folder
and it lists every run as `<config>/<timestamp>`; tick runs in its sidebar to
compare them.

| TensorBoard tab | What this project logs |
| --- | --- |
| **Scalars** | Per epoch: `loss/train`, `loss/val`, `accuracy/train`, `accuracy/val`, `learning_rate`. Every 50 batches: `batch/train_loss`. At the end: `test/loss`, `test/accuracy`, and the `hparam/...` scores. |
| **Custom Scalars** | Train and val of the loss and the accuracy in one chart each, to spot overfitting. |
| **Images** | `data/samples` (input images), `test/predictions` (test images with predicted and true class), `test/confusion_matrix`. |
| **Graphs** | The network's layers and the tensor shapes between them. |
| **Histograms**, **Distributions** | Weights and gradients of every layer, per epoch. |
| **Text** | The run's config. |
| **HParams** | One row per run: `lr`, `weight_decay`, `epochs`, `batch_size`, `hidden_sizes`, `dropout`, `num_params`, next to `best_val_acc`, `test_acc`, and `test_loss`. |

The curves are written after every epoch, so a TensorBoard opened before or
during training shows the run as it trains. The test results, figures, and
HParams row appear when the run finishes.

The config's optional `tensorboard` section sets how much is logged:

```json
"tensorboard": {
  "log_every_n_steps": 50,
  "histograms": true
}
```

`log_every_n_steps` is how often (in batches) the training loss is logged;
`"histograms": false` skips the weight and gradient histograms, which are the
largest part of the event files.

## Opening TensorBoard

**In Colab or Jupyter** (section 4 of the notebook):

```python
%load_ext tensorboard
%tensorboard --logdir "{logdir}"
```

`{logdir}` is a Python variable holding the project folder under
`OUTPUT_DIR`, which the notebook cell before it sets. TensorBoard opens in the
cell's output and keeps updating in the background, so start it **before**
training to watch the run live. In Colab the runs are read from Google Drive,
so runs from earlier sessions show up too. In VS Code or Cursor the output can
stay blank: open [http://localhost:6006](http://localhost:6006) in a browser.

**From a terminal**, at the repo root with the `.venv` activated:

```bash
tensorboard --logdir runs/TensorBoardTrainingBasicNeuralNetwork
```

then open [http://localhost:6006](http://localhost:6006). Each run prints this
command, with its own folder, when it finishes.

## How the logging is added

Compared with TrainingBasicNeuralNetwork, three files change:

- [`main.py`](main.py) creates a
  [`SummaryWriter`](https://pytorch.org/docs/stable/tensorboard.html) for the
  run folder, logs the config, sample images, and the network graph before
  training, and the test results and HParams row after it, then closes the
  writer.
- [`trainers/trainer.py`](trainers/trainer.py) takes the writer and logs the
  curves (and histograms) after every epoch, and the batch loss every
  `log_every_n_steps` batches. It calls `writer.flush()` after each epoch so
  an open TensorBoard sees every epoch at once. Without a writer it trains
  exactly as before.
- [`utils/tensorboard_logging.py`](utils/tensorboard_logging.py) holds the
  logging helpers: one function per kind of record, e.g.
  `log_test_predictions` draws the prediction grid and the confusion matrix
  with Matplotlib and logs them with `writer.add_figure`.

The `SummaryWriter` comes with PyTorch (`torch.utils.tensorboard`); the
`tensorboard` package in `requirements.txt` is only the dashboard that reads
the event files. Colab has it preinstalled.

## Loading a trained model

`load_model` rebuilds the network from the config stored in the checkpoint
and loads its weights, as in
[TrainingBasicConvnet](../TrainingBasicConvnet/README.md#loading-a-trained-model):

```python
from TensorBoardTrainingBasicNeuralNetwork import load_model

model = load_model("config01")   # newest run of config01, best checkpoint
```

To train a run further instead, resume it, as described in
[Continuing training](../TrainingBasicConvnet/README.md#continuing-training):
`main("config01.json", resume_from="config01")`, or
`--resume-from config01` on the command line. The continued run is a new run
in TensorBoard; its epochs are numbered after the earlier run's, so tick both
runs to see one continuous curve.

## Training with the Colab CLI

Another way to train this project, once everything is final. Its only purpose
is the training run itself:

1. **Work locally** on the project: change the code or a config, run short
   tests, fix bugs.
2. **Use Google Colab** (this project's notebook) for debugging and interactive work.
3. **Use the Colab CLI** when everything is final and the only thing left is
   training. The CLI creates a Colab runtime from your local terminal and opens
   a shell on it, so there is no notebook or browser tab to keep open. On the
   runtime you clone the repo and run `python -m TensorBoardTrainingBasicNeuralNetwork --config config01.json`.

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

This project needs no API keys: everything is logged to the run folder.

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
python -m TensorBoardTrainingBasicNeuralNetwork --config config01.json
```

To leave training running, press `Ctrl+B`, then `D`. To come back later:
`colab ssh -s cs4337`, then `tmux attach -t train`.

### Step 5: Check that everything was logged

The run folders are on Google Drive, so they are kept after the session
stops. From the session:

```bash
ls /content/drive/MyDrive/CodeCS4337Fall2026/runs/TensorBoardTrainingBasicNeuralNetwork/config01
```

or on [drive.google.com](https://drive.google.com), under **My Drive >
CodeCS4337Fall2026 > runs > TensorBoardTrainingBasicNeuralNetwork**. Check that:

- each run has its own `<timestamp>` folder (the path is printed when the run starts), with its config, results, plots, and checkpoints;
- `runs_summary.csv`, next to the run folders, has one new row per run;
- the TensorBoard logs are in the run folder. To view them, open this project's notebook in Colab and run its TensorBoard section (it reads the runs from Drive), or download the project folder from Drive and run `tensorboard --logdir runs/TensorBoardTrainingBasicNeuralNetwork` locally;
- if the session stopped or the run was cut off, start a new session, repeat the steps above, and continue from the last checkpoint on Drive with `--resume-from config01`, e.g. `python -m TensorBoardTrainingBasicNeuralNetwork --config config01.json --resume-from config01`.

When everything is checked, stop the session so it no longer uses your Colab
quota:

```bash
colab stop -s cs4337
```

## Things to try

Copy [`configs/config01.json`](configs/config01.json) (e.g. to
`config02.json`), change one setting, run, and compare the runs in
TensorBoard:

- **Overfitting**: `"dropout": 0.0` and `"epochs": 30`. In *Custom Scalars*,
  where does `loss/val` stop falling while `loss/train` keeps going down?
  Repeat with `"dropout": 0.5`.
- **Learning rate**: `"lr": 0.01` vs. `0.0001`. Compare `batch/train_loss` in
  *Scalars*: too high is noisy, too low is slow.
- **Width and depth**: `"hidden_sizes": [512, 256]`, `[64, 32]`, or `[]`.
  Sort the *HParams* table by `test_acc`, and look at `num_params`.
- **Gradients**: in *Histograms*, compare the gradients of the first and the
  last layer, early and late in training.
- **Mistakes**: in *Images*, which classes does `test/confusion_matrix`
  confuse most, and do they look alike in `test/predictions`?
