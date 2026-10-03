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
├── last.ckpt                       # Checkpoint after the last epoch
├── loss.png, accuracy.png, accuracy_per_class.png
└── predictions.png, wrong_predictions.png
```

`best_epoch08` is the ninth epoch. Because the network is passed into
`LitConvNet`, give it again when reloading a checkpoint:

```python
from LitTrainingBasicConvnet.models import ConvNet, LitConvNet

model = LitConvNet.load_from_checkpoint("path/to/file.ckpt", net=ConvNet(dropout=0.25))
```

## Things to try

- Compare [`models/lit_convnet.py`](models/lit_convnet.py) with
  [`TrainingBasicConvnet/trainers/trainer.py`](../TrainingBasicConvnet/trainers/trainer.py):
  find where each line of the hand-written loop went.
- Train for 30 epochs and look for overfitting in `loss.png`.
- Add a network to `models/components/` and pass it to `LitConvNet` in
  `main.py`.
- Turn on LitLogger and compare two configs on lightning.ai.
