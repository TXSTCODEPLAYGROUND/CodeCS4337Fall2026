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
