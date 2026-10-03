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

## Things to try

- Change `"hidden_sizes"` (wider, deeper, or `[]` for a linear model) and
  `"dropout"` in a copy of the config, and compare the runs in
  `runs_summary.csv` and their `accuracy.png`.
- Run [LitTrainingBasicConvnet](../LitTrainingBasicConvnet/) with the same
  settings and compare the two `accuracy_per_class.png` plots.
- Compare [`main.py`](main.py) with
  [`../TrainingBasicNeuralNetwork/main.py`](../TrainingBasicNeuralNetwork/main.py):
  the same network, trained by a hand-written loop vs. by Lightning.
