# LitTrainingBasicConvnet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitTrainingBasicConvnet/lit_training_basic_convnet_notebook.ipynb)
This project's notebook: [`lit_training_basic_convnet_notebook.ipynb`](lit_training_basic_convnet_notebook.ipynb).

The same experiment as [TrainingBasicConvnet](../TrainingBasicConvnet/) (a
small ConvNet on Fashion-MNIST, with the same network and settings), rewritten
with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/). Read
that project first: this one is easiest to follow once you know what a plain
PyTorch training loop does.

What this project adds:

1. **PyTorch Lightning** runs the training loop for you.
2. **A standard project layout**, the one recommended by Lightning and used by
   the Lightning-Hydra-Template.
3. **[torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)** for
   accuracy, precision, recall, and per-class accuracy.
4. **Plots** saved with every run: learning curves and example predictions.
5. **[LitLogger](https://lightning.ai/docs/pytorch/stable/visualize/experiment_managers.html)**
   (optional) to follow runs online.

How to set up and run the project and where results go is explained in the
[main README](../README.md). Function and class details are in the
[API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitTrainingBasicConvnet.html).

## PyTorch Lightning in a nutshell

In TrainingBasicConvnet you wrote everything yourself: moving tensors to the
GPU, `zero_grad`/`backward`/`step`, switching between `model.train()` and
`model.eval()`, turning gradients off for validation, saving checkpoints, and
recording the metric history. That code is almost the same in every project,
and it is easy to get subtly wrong.

Lightning splits a project into **what to compute**, which you write, and
**how to run it**, which Lightning's `Trainer` does:

| You write | In | Lightning's `Trainer` does |
| --- | --- | --- |
| `training_step`: loss for one batch | `LightningModule` | the loop over epochs and batches, `zero_grad`, `backward`, `optimizer.step()` |
| `validation_step`, `test_step` | `LightningModule` | `model.eval()`, turning gradients off, running validation after every epoch |
| `configure_optimizers` | `LightningModule` | moving the model and every batch to the GPU (or CPU, or Apple GPU) |
| `self.log("val_acc", ...)` | `LightningModule` | averaging over the epoch and sending values to the loggers |
| `prepare_data`, `setup`, `*_dataloader` | `LightningDataModule` | calling them at the right time |
| `ModelCheckpoint(monitor="val_acc")` | `main.py` | saving the best and last checkpoints |

[`main.py`](main.py) then needs only:

```python
trainer = L.Trainer(max_epochs=10, logger=loggers, callbacks=[checkpoint])
trainer.fit(model, datamodule=data)                  # train + validate
trainer.test(datamodule=data, ckpt_path="best")      # test the best checkpoint
```

The same code runs on a CPU, one GPU, or several GPUs, by changing the
`Trainer`'s arguments, not the model.

## Project layout

The layout follows the
[Lightning style guide](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html)
and the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template),
a widely used starting point for Lightning research projects:

```
LitTrainingBasicConvnet/
├── main.py                    # Thin entry point: config -> DataModule + LightningModule + Trainer
├── configs/config01.json      # Experiment settings
├── models/
│   ├── lit_convnet.py         # LitConvNet (LightningModule): how to train, the "system"
│   └── components/
│       └── convnet.py         # ConvNet (nn.Module): the network itself, the "model"
├── dataloaders/
│   └── fashion_mnist.py       # FashionMNISTDataModule: download, split, batch
└── utils/
    ├── paths.py               # Repo and config paths
    ├── experiment.py          # runs_summary.csv
    └── plots.py               # Learning curves and prediction grids
```

- **`models/components/`** holds plain PyTorch networks (`nn.Module`): layers
  and `forward` only, no training code. They work with or without Lightning.
- **`models/`** holds the `LightningModule`s. `LitConvNet` *receives* a
  network and adds the loss, metrics, logging, and optimizer. To try another
  architecture, add it to `components/` and pass it in; the training code does
  not change.
- **`dataloaders/`** holds the `LightningDataModule`s. The dataset files stay
  in the shared `data/` folder at the repo root, outside the project. (The
  template names this code folder `data/`; it is called `dataloaders/` here so
  code is not confused with the dataset folder.)
- **`main.py`** only reads the config, builds those objects, and starts the
  `Trainer`.

The template also uses [Hydra](https://hydra.cc/) to compose YAML configs from
the command line, plus tests and many optional loggers. This project keeps one
JSON config per experiment, like TrainingBasicConvnet, so there is less to
learn at once.

## Metrics with torchmetrics

Accuracy averaged batch by batch is only correct if every batch has the same
size, and it gets harder with several GPUs. A `torchmetrics` object keeps
running totals instead: call it on every batch, and Lightning computes the
exact value over the whole epoch (combining all GPUs) and resets it.

[`LitConvNet`](models/lit_convnet.py) logs, for `train`, `val`, and `test`:

| Metric | Meaning |
| --- | --- |
| `<stage>_loss` | Cross-entropy loss |
| `<stage>_acc` | Fraction of all images classified correctly |
| `<stage>_precision` | For each class: of the images *predicted* as that class, the fraction that really are. Averaged over classes. |
| `<stage>_recall` | For each class: of the images that really *are* that class, the fraction found. Averaged over classes. |
| `<stage>_acc_<class>` | Accuracy on the images of one class, e.g. `val_acc_Shirt` |

Precision and recall are *macro* averages: every class counts the same, so a
class the model is bad at pulls them down even when overall accuracy looks
good. All values are written to `metrics.csv` and printed in a table after
testing. The per-class accuracies show which classes are hard: on
Fashion-MNIST, *Shirt* is often confused with *T-shirt/top*, *Pullover*, and
*Coat*.

## Plots

After testing, [`utils/plots.py`](utils/plots.py) saves these PNG files in the
run folder:

| File | Shows |
| --- | --- |
| `loss.png` | Training and validation loss per epoch, on one graph |
| `accuracy.png` | Training and validation accuracy per epoch, on one graph |
| `accuracy_per_class.png` | Accuracy of every class per epoch: one color per class, training solid, validation dashed |
| `predictions.png` | The first 16 test images with their predicted and true class (green if correct, red if wrong) |
| `wrong_predictions.png` | 16 test images the model got wrong |

How to read the curves: when training keeps improving but validation stops
improving or gets worse, the model is **overfitting** (memorizing the
training images). More dropout, weight decay, or fewer epochs can help.
Epochs are counted from 0, as in the checkpoint names.

## LitLogger (optional)

[LitLogger](https://pypi.org/project/litlogger/) is Lightning AI's
experiment tracker. With it, every `self.log(...)` value is also sent to
[lightning.ai](https://lightning.ai), where you can watch the curves live
while training runs (also from your phone), compare runs side by side, and
share them. The plots and config files of each run are uploaded too. It is
free for individual use, and `metrics.csv` and the plots are still saved
locally either way.

It is off by default, because it needs a Lightning AI account. To turn it on:

1. Create a free account at [lightning.ai](https://lightning.ai).
2. Get your keys: on lightning.ai, click your profile picture (top right),
   open **Global Settings**, choose **Keys**, and find the
   **Login via CLI** box under *Programmatic access*. Add both values to
   `.env` at the repo root:

   ```
   LIGHTNING_USER_ID=your-user-id
   LIGHTNING_API_KEY=your-api-key
   ```

   `.env` is not in git, so your key stays private. Never put it in
   `.envcolab` (which *is* in git), a config, or a notebook you share. If a key
   is ever pushed by mistake, create a new one on the same page.

   **In Colab**, the setup cell overwrites `.env`, so store the two values as
   Colab *Secrets* named `LIGHTNING_USER_ID` and `LIGHTNING_API_KEY` instead
   (key icon in the left sidebar). Then run the **Load API keys** cell of
   [this project's notebook](lit_training_basic_convnet_notebook.ipynb), after the setup cells
   and before running the project. It copies the keys from your Secrets to the
   notebook's environment and to `.env`, so every way of running works: from
   the notebook (Option A1 or B) and from Colab's terminal (A2).
3. Set `"litlogger": true` in your config.

The run prints a link to its page on lightning.ai. If the keys are missing
or LitLogger cannot connect, it prints a warning and the run continues with
local logs only.

## Run folder

Each run writes these files to
`runs/LitTrainingBasicConvnet/<config>/<timestamp>/`:

```
├── config.json                     # Exact copy of the config used
├── hparams.json                    # Optimizer and data settings
├── metrics.csv                     # Every metric, every epoch (Lightning's CSVLogger)
├── best_epoch08_valacc0.9240.ckpt  # Checkpoint with the best validation accuracy
├── last.ckpt                       # Checkpoint of the last epoch, rewritten every epoch
├── loss.png, accuracy.png, accuracy_per_class.png
└── predictions.png, wrong_predictions.png
```

`best_epoch08` is the ninth epoch. `ModelCheckpoint(save_last=True)` would only
copy the best checkpoint to `last.ckpt`, so `main.py` uses a second
`ModelCheckpoint` that writes `last.ckpt` after every epoch; that is the
checkpoint training is resumed from.

## Loading a trained model

`load_model` finds a run's checkpoint, rebuilds the network from the run's
`config.json`, and loads the weights, so you don't have to remember the
network's settings:

```python
from LitTrainingBasicConvnet import load_model

model = load_model("config01")                       # newest run of config01, best checkpoint
model = load_model("config01/2026-10-02_12-29-09")   # one specific run
model = load_model("config01", which="last")         # the last epoch instead of the best
logits = model(images)                               # images: (N, 1, 28, 28), normalized
```

Run names are looked up in `<OUTPUT_DIR>/LitTrainingBasicConvnet/`; a full
path to a run folder or a `.ckpt` file works too, e.g.
`load_model("config01/2026-10-02_12-29-09/best_epoch07_valacc0.9260.ckpt")`.
Each run folder keeps two checkpoints, the best and the last epoch; older ones
are replaced during training, so other epochs can't be loaded. The model comes back in eval
mode on the CPU (pass `device="cuda"` for a GPU), and works with
`trainer.test(model, datamodule=data)`.

## Continuing training

Running a config trains from scratch. To train an earlier run further, for
example because it needed more epochs or a Colab disconnect stopped it, pass
`resume_from`:

```python
from LitTrainingBasicConvnet import main

main("config01.json", resume_from="config01")   # newest run of config01
```

```bash
python -m LitTrainingBasicConvnet --config config01.json --resume-from config01
```

Lightning restores everything from the run's `last.ckpt` (`trainer.fit(...,
ckpt_path=...)`): the weights, the optimizer state including its learning
rate, and the epoch count. The config's `"epochs"` **more** epochs are then
trained, so a 10-epoch run resumed with `config01.json` ends after epoch 20.
The continued run gets its own run folder, plots, and `runs_summary.csv` row,
and `hparams.json` records `resumed_from`; the earlier run is not changed.

- `resume_from` accepts the same names as `load_model`: a config name (its
  newest run), a run folder such as `"config01/2026-10-02_12-29-09"`, or a
  checkpoint file. A run stopped before its first `last.ckpt` continues from
  its best checkpoint.
- The config's model settings must match the run's (the weights only fit the
  same network). To train a different number of extra epochs, copy the config,
  change `"epochs"`, and pass the copy with `resume_from="config01"`.
- The learning rate and weight decay come from the saved optimizer, so changing
  `lr` or `weight_decay` in the config has no effect when resuming. The data
  settings, seed, and logging come from the config.
- Resuming again continues further: `"config01"` then means the continued run.
- The best checkpoint and the test score are those of the continued run.

## Training with the Colab CLI

Another way to train this project, once everything is final. Its only purpose
is the training run itself:

1. **Work locally** on the project: change the code or a config, run short
   tests, fix bugs.
2. **Use Google Colab** (this project's notebook) for debugging and interactive work.
3. **Use the Colab CLI** when everything is final and the only thing left is
   training. The CLI creates a Colab runtime from your local terminal and opens
   a shell on it, so there is no notebook or browser tab to keep open. On the
   runtime you clone the repo and run `python -m LitTrainingBasicConvnet --config config01.json`.

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
python -m LitTrainingBasicConvnet --config config01.json
```

To leave training running, press `Ctrl+B`, then `D`. To come back later:
`colab ssh -s cs4337`, then `tmux attach -t train`.

### Step 5: Check that everything was logged

The run folders are on Google Drive, so they are kept after the session
stops. From the session:

```bash
ls /content/drive/MyDrive/CodeCS4337Fall2026/runs/LitTrainingBasicConvnet/config01
```

or on [drive.google.com](https://drive.google.com), under **My Drive >
CodeCS4337Fall2026 > runs > LitTrainingBasicConvnet**. Check that:

- each run has its own `<timestamp>` folder (the path is printed when the run starts), with its config, results, plots, and checkpoints;
- `runs_summary.csv`, next to the run folders, has one new row per run;
- with `"litlogger": true`, the run shows up on lightning.ai. A `LitLogger disabled` warning means the keys are missing from `.env`;
- if the session stopped or the run was cut off, start a new session, repeat the steps above, and continue from the last checkpoint on Drive with `--resume-from config01`, e.g. `python -m LitTrainingBasicConvnet --config config01.json --resume-from config01`.

When everything is checked, stop the session so it no longer uses your Colab
quota:

```bash
colab stop -s cs4337
```

## Things to try

- Compare [`models/lit_convnet.py`](models/lit_convnet.py) with
  [`TrainingBasicConvnet/trainers/trainer.py`](../TrainingBasicConvnet/trainers/trainer.py):
  find where each line of the hand-written loop went.
- Train for 30 epochs and look for overfitting in `loss.png`.
- Add a network to `models/components/` and pass it to `LitConvNet` in
  `main.py`.
- Turn on LitLogger and compare two configs on lightning.ai.
