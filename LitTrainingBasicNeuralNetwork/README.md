# LitTrainingBasicNeuralNetwork

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitTrainingBasicNeuralNetwork/lit_training_basic_neural_network_notebook.ipynb)
This project's notebook: [`lit_training_basic_neural_network_notebook.ipynb`](lit_training_basic_neural_network_notebook.ipynb).

The fully connected network of
[TrainingBasicNeuralNetwork](../TrainingBasicNeuralNetwork/) (an MLP with only
`nn.Linear` layers), trained with
[PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). Everything
else is the same as in [LitTrainingBasicConvnet](../LitTrainingBasicConvnet/):
the project layout, torchmetrics (accuracy, precision, recall, per-class
accuracy), the plots, and optional LitLogger tracking. After 10 epochs the
test accuracy is about 88.5%.

Read the two projects it combines first:

- [TrainingBasicNeuralNetwork](../TrainingBasicNeuralNetwork/README.md)
  explains the network: fully connected layers, ReLU, dropout, and how to
  count parameters.
- [LitTrainingBasicConvnet](../LitTrainingBasicConvnet/README.md) introduces
  Lightning, the project layout, the metrics, the plots, and how to turn on
  LitLogger. All of it applies here unchanged.

How to set up and run the project and where results go is explained in the
[main README](../README.md). Function and class details are in the
[API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitTrainingBasicNeuralNetwork.html).

## What is different from LitTrainingBasicConvnet

Only the network. This is the point of keeping the network
(`models/components/`, Lightning's *model*) apart from the `LightningModule`
that trains it (`models/`, the *system*):

| | LitTrainingBasicConvnet | This project |
| --- | --- | --- |
| Network | `models/components/convnet.py`: `ConvNet` | [`models/components/mlp.py`](models/components/mlp.py): `MLP` |
| LightningModule | `models/lit_convnet.py`: `LitConvNet` | [`models/lit_mlp.py`](models/lit_mlp.py): `LitMLP` (same code, renamed) |
| `"model"` in the config | `{"dropout": 0.25}` | `{"hidden_sizes": [256, 128], "dropout": 0.2}` |
| Parameters | about 422,000 | 235,146 |
| Test accuracy after 10 epochs | about 92% | about 88.5% |

The loss, metrics, logging, optimizer, plots, checkpoints, and data code did
not change at all. In [`main.py`](main.py) the network is built from the
config and passed in:

```python
net = MLP(hidden_sizes=model_cfg["hidden_sizes"], dropout=model_cfg["dropout"])
model = LitMLP(net=net, class_names=..., lr=..., weight_decay=...)
```

The run also prints the number of parameters at the start, and
`runs_summary.csv` gets two extra columns, `hidden_sizes` and `num_params`,
so runs with different network sizes can be compared.

## Reading the plots of an MLP

In `accuracy.png` and `loss.png`, the training curve keeps improving while
the validation curve flattens after about 5 epochs: the network starts to
memorize the training images (overfitting). An MLP overfits sooner than the
ConvNet, because it has no built-in knowledge of images and its first layer
alone has 200,960 weights. `accuracy_per_class.png` shows which classes
suffer most: with `config01.json`, *Shirt* reaches only about 68% test
accuracy, and *Coat* and *Pullover* about 80%, while *Trouser*, *Bag*, and
*Sandal* are above 95%.

## Run folder

The same files as in
[LitTrainingBasicConvnet](../LitTrainingBasicConvnet/README.md#run-folder),
in `runs/LitTrainingBasicNeuralNetwork/<config>/<timestamp>/`. `hparams.json`
also has a `"net"` section with the network settings (`hidden_sizes`,
`dropout`). `load_model` reads them from the run, rebuilds the network, and
loads its weights (see
[Loading a trained model](../LitTrainingBasicConvnet/README.md#loading-a-trained-model)):

```python
from LitTrainingBasicNeuralNetwork import load_model

model = load_model("config01")   # newest run of config01, best checkpoint
```

To train a run further instead, resume it, as described in
[Continuing training](../LitTrainingBasicConvnet/README.md#continuing-training):
`main("config01.json", resume_from="config01")`, or
`--resume-from config01` on the command line.

## Training with the Colab CLI

Another way to train this project, once everything is final. Its only purpose
is the training run itself:

1. **Work locally** on the project: change the code or a config, run short
   tests, fix bugs.
2. **Use Google Colab** (this project's notebook) for debugging and interactive work.
3. **Use the Colab CLI** when everything is final and the only thing left is
   training. The CLI creates a Colab runtime from your local terminal and opens
   a shell on it, so there is no notebook or browser tab to keep open. On the
   runtime you clone the repo and run `python -m LitTrainingBasicNeuralNetwork --config config01.json`.

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

LitLogger (lightning.ai) is used only when the config has
`"litlogger": true`. In that case, add your two Lightning keys (the same ones
as in your local `.env`) to the runtime's `.env`. `read -s` hides the key while
you paste it, so it is not shown on screen or saved in the shell history:

```bash
read -rp "Lightning user ID: " uid && read -rsp "Lightning API key: " key \
  && printf 'LIGHTNING_USER_ID=%s\nLIGHTNING_API_KEY=%s\n' "$uid" "$key" >> .env \
  && unset uid key && echo
```

Without them, LitLogger is skipped with a warning and the run is logged in its
run folder only. Never put the keys in `.envcolab`, which is in git.

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
python -m LitTrainingBasicNeuralNetwork --config config01.json
```

To leave training running, press `Ctrl+B`, then `D`. To come back later:
`colab ssh -s cs4337`, then `tmux attach -t train`.

### Step 5: Check that everything was logged

The run folders are on Google Drive, so they are kept after the session
stops. From the session:

```bash
ls /content/drive/MyDrive/CodeCS4337Fall2026/runs/LitTrainingBasicNeuralNetwork/config01
```

or on [drive.google.com](https://drive.google.com), under **My Drive >
CodeCS4337Fall2026 > runs > LitTrainingBasicNeuralNetwork**. Check that:

- each run has its own `<timestamp>` folder (the path is printed when the run starts), with its config, results, plots, and checkpoints;
- `runs_summary.csv`, next to the run folders, has one new row per run;
- with `"litlogger": true`, the run shows up on lightning.ai. A `LitLogger disabled` warning means the keys are missing from `.env`;
- if the session stopped or the run was cut off, start a new session, repeat the steps above, and continue from the last checkpoint on Drive with `--resume-from config01`, e.g. `python -m LitTrainingBasicNeuralNetwork --config config01.json --resume-from config01`.

When everything is checked, stop the session so it no longer uses your Colab
quota:

```bash
colab stop -s cs4337
```

## Things to try

- Change `"hidden_sizes"` (wider, deeper, or `[]` for a linear model) and
  `"dropout"` in a copy of the config, and compare the runs in
  `runs_summary.csv` and their `accuracy.png`.
- Run [LitTrainingBasicConvnet](../LitTrainingBasicConvnet/) with the same
  settings and compare the two `accuracy_per_class.png` plots.
- Compare [`main.py`](main.py) with
  [`../TrainingBasicNeuralNetwork/main.py`](../TrainingBasicNeuralNetwork/main.py):
  the same network, trained by a hand-written loop vs. by Lightning.
