# LitWBTrainingBasicNeuralNetwork

The fully connected network (MLP) of
[LitTrainingBasicNeuralNetwork](../LitTrainingBasicNeuralNetwork/), tracked
with [Weights & Biases](https://wandb.ai) (W&B) exactly like
[LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/): live charts, a
table of test predictions, and a confusion matrix on wandb.ai instead of
plotting code, with `metrics.csv` kept locally as a backup. After 10 epochs
the test accuracy is about 88.6%.

Read the projects it combines first:

- [TrainingBasicNeuralNetwork](../TrainingBasicNeuralNetwork/README.md)
  explains the network: fully connected layers, ReLU, dropout, and how to
  count parameters.
- [LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/README.md)
  explains W&B: setting up your key (in `.env`, or Colab Secrets with the
  starter notebook's **Load API keys** cell), offline mode and `wandb sync`,
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
in `runs/LitWBTrainingBasicNeuralNetwork/<config>/<timestamp>/`. To reload a
checkpoint, pass the network again:

```python
from LitWBTrainingBasicNeuralNetwork.models import MLP, LitMLP

net = MLP(hidden_sizes=[256, 128], dropout=0.2)
model = LitMLP.load_from_checkpoint("path/to/file.ckpt", net=net)
```
