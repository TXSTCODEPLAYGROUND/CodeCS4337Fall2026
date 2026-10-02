# TrainingBasicNeuralNetwork

Train the simplest kind of neural network, a **fully connected network**
(also called a multilayer perceptron, MLP), to recognize clothing in
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): 70,000
grayscale images of 28x28 pixels in 10 classes. After 10 epochs the test
accuracy is about 88%.

This is the starting point of the course projects. The network uses only
fully connected (`nn.Linear`) layers, no convolutions, so you can focus on
what a neural network is and how it learns. The data, training loop,
checkpoints, and project layout are the same as in
[TrainingBasicConvnet](../TrainingBasicConvnet/), which replaces this network
with a convolutional one; read its README for the training steps and how the
code is organized into modules and packages.

How to set up, run, and configure the project, and where results go, is
explained in the [main README](../README.md). Function and class details are
in the [API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/TrainingBasicNeuralNetwork.html).

## The network

[`models/mlp.py`](models/mlp.py) builds this network from the config
(`"hidden_sizes": [256, 128]`):

```
image 1x28x28 -> Flatten -> 784 values
              -> Linear(784, 256) -> ReLU -> Dropout
              -> Linear(256, 128) -> ReLU -> Dropout
              -> Linear(128, 10)  -> 10 logits, one score per class
```

- **Flatten**: a fully connected layer takes a vector, not an image, so the
  28x28 pixels are laid out in one row of 784 numbers.
- **Linear (fully connected) layer**: every output is a weighted sum of
  *every* input plus a bias, `y = W x + b`. `Linear(784, 256)` has a weight
  matrix of 784 x 256 values and 256 biases, all learned during training.
- **ReLU**: `max(0, x)`, applied to every value. Without a nonlinearity
  between them, any stack of linear layers would collapse into a single
  linear layer, and adding layers would gain nothing.
- **Dropout**: during training, randomly sets a fraction of the values to 0
  (20% here), so the network cannot rely on a few specific units. This
  reduces overfitting. It is turned off by `model.eval()` for validation and
  testing.
- **Logits**: the last layer has no ReLU. Its 10 outputs are raw scores; the
  cross-entropy loss turns them into probabilities internally, and the
  predicted class is the one with the highest score.

### Counting parameters

Each `Linear(n_in, n_out)` layer has `n_in * n_out + n_out` parameters:

| Layer | Parameters |
| --- | --- |
| `Linear(784, 256)` | 784 x 256 + 256 = 200,960 |
| `Linear(256, 128)` | 256 x 128 + 128 = 32,896 |
| `Linear(128, 10)` | 128 x 10 + 10 = 1,290 |
| **Total** | **235,146** |

The run prints this number at the start and saves it in the results file and
in `runs_summary.csv`. Most parameters are in the first layer, because it
connects every pixel to every hidden unit.

## Fully connected vs. convolutional

| | This project (MLP) | [TrainingBasicConvnet](../TrainingBasicConvnet/) |
| --- | --- | --- |
| Input | A vector of 784 pixels | The 28x28 image |
| Layers | `Linear` only | `Conv2d` + pooling, then `Linear` |
| Parameters | 235,146 | about 422,000 |
| Test accuracy after 10 epochs | about 88% | about 92% |

A fully connected layer treats each pixel as an unrelated input: it does not
know which pixels are neighbors. Shift a shirt two pixels to the right and,
for the network, it is a completely different input. A convolution instead
slides the same small filter over the whole image, so it learns local
patterns (edges, textures) that work anywhere in the image. That built-in
knowledge about images is why ConvNets do better on images. MLPs are still
the basic building block: the last layers of a ConvNet, and parts of
transformers, are fully connected layers.

## Things to try

Change only [`configs/config01.json`](configs/config01.json) (or a copy of
it, e.g. `config02.json`), run, and compare the runs in `runs_summary.csv`:

- **Width**: `"hidden_sizes": [512, 256]` or `[64, 32]`. How do the number of
  parameters and the accuracy change?
- **Depth**: `[256]` (one hidden layer) vs. `[256, 256, 128]` (three).
- **No hidden layer**: `[]` gives a linear model (`Linear(784, 10)`, also
  called multinomial logistic regression). How much do hidden layers help?
- **Dropout**: `0.0` vs. `0.5`. Train for 30 epochs and compare the training
  and validation accuracy printed after each epoch: the gap between them is
  overfitting.
- Then train [TrainingBasicConvnet](../TrainingBasicConvnet/) and compare the
  two `runs_summary.csv` files.
