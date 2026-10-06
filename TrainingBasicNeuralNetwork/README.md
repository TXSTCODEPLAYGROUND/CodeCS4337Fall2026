# TrainingBasicNeuralNetwork

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/TrainingBasicNeuralNetwork/training_basic_neural_network_notebook.ipynb)
This project's notebook: [`training_basic_neural_network_notebook.ipynb`](training_basic_neural_network_notebook.ipynb).

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

## Loading a trained model

`load_model` rebuilds the network from the config stored in the checkpoint
and loads its weights, as in
[TrainingBasicConvnet](../TrainingBasicConvnet/README.md#loading-a-trained-model):

```python
from TrainingBasicNeuralNetwork import load_model

model = load_model("config01")   # newest run of config01, best checkpoint
```

To train a run further instead, resume it, as described in
[Continuing training](../TrainingBasicConvnet/README.md#continuing-training):
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
   runtime you clone the repo and run `python -m TrainingBasicNeuralNetwork --config config01.json`.

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
python -m TrainingBasicNeuralNetwork --config config01.json
```

To leave training running, press `Ctrl+B`, then `D`. To come back later:
`colab ssh -s cs4337`, then `tmux attach -t train`.

### Step 5: Check that everything was logged

The run folders are on Google Drive, so they are kept after the session
stops. From the session:

```bash
ls /content/drive/MyDrive/CodeCS4337Fall2026/runs/TrainingBasicNeuralNetwork/config01
```

or on [drive.google.com](https://drive.google.com), under **My Drive >
CodeCS4337Fall2026 > runs > TrainingBasicNeuralNetwork**. Check that:

- each run has its own `<timestamp>` folder (the path is printed when the run starts), with its config, results, plots, and checkpoints;
- `runs_summary.csv`, next to the run folders, has one new row per run;
- if the session stopped or the run was cut off, start a new session, repeat the steps above, and continue from the last checkpoint on Drive with `--resume-from config01`, e.g. `python -m TrainingBasicNeuralNetwork --config config01.json --resume-from config01`.

When everything is checked, stop the session so it no longer uses your Colab
quota:

```bash
colab stop -s cs4337
```

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
