# TrainingBasicConvnet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/TrainingBasicConvnet/training_basic_convnet_notebook.ipynb)
This project's notebook: [`training_basic_convnet_notebook.ipynb`](training_basic_convnet_notebook.ipynb).

Train a small convolutional neural network (ConvNet) to recognize clothing in
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist): 70,000
grayscale images of 28x28 pixels in 10 classes (T-shirt/top, trouser,
pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot). After 10
epochs the test accuracy is about 92%.

Everything is written by hand in plain PyTorch, with no training framework.
The project has two goals:

1. **See every step of training a neural network**: data, model, loss,
   optimizer, the training loop, validation, checkpoints, and testing.
2. **Organize code as a Python package**: split it into modules and
   subpackages, each with one job, instead of one long script or notebook.

How to set up, run, and configure the project, and where results go, is
explained in the [main README](../README.md). Function and class details are
in the [API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/TrainingBasicConvnet.html).

## Training a neural network, step by step

Reading [`main.py`](main.py) from top to bottom follows one experiment:

1. **Seed and device**: [`utils/reproducibility.py`](utils/reproducibility.py)
   seeds Python, NumPy, and PyTorch, so a run with the same config gives
   (nearly) the same result, and picks the GPU if there is one.
2. **Data**: [`dataloaders/fashion_mnist.py`](dataloaders/fashion_mnist.py)
   downloads the dataset and normalizes the pixels. The 60,000 official training
   images are split into a **training set** (to learn from) and a
   **validation set** (to watch for overfitting and pick the best epoch). The
   10,000 **test** images are used only once, at the very end, to estimate how
   well the model does on images it has never seen. `DataLoader`s serve the
   images in shuffled batches.
3. **Model**: [`models/convnet.py`](models/convnet.py) defines the network.
   Two blocks of convolution, batch normalization, ReLU, and max pooling turn
   the 28x28 image into 64 feature maps of 7x7. A classifier flattens them
   and maps them through a hidden layer of 128 units (with dropout) to 10
   scores, one per class (the *logits*).
4. **Training loop**: [`trainers/trainer.py`](trainers/trainer.py) does the
   actual learning. For every batch:

   ```python
   outputs = model(images)            # forward pass: logits
   loss = criterion(outputs, labels)  # cross-entropy loss
   optimizer.zero_grad()              # clear the old gradients
   loss.backward()                    # backpropagation: compute new gradients
   optimizer.step()                   # update the weights (Adam)
   ```

   After each epoch the model is evaluated on the validation set with
   `model.eval()` and gradients turned off: dropout and batch normalization
   switch to inference behavior, and nothing is learned. Whenever the
   validation accuracy improves, the weights are saved as the best checkpoint.
5. **Testing**: the best checkpoint is loaded and evaluated once on the test
   set. The results and a row in `runs_summary.csv` are written.

## Organizing code into modules and packages

A **module** is one `.py` file. A **package** is a folder of modules with an
`__init__.py` file. This project is a package made of smaller packages:

| Folder | Its one job |
| --- | --- |
| [`configs/`](configs/) | Experiment settings (JSON), so changing a hyperparameter never means editing code |
| [`dataloaders/`](dataloaders/) | Load and split the data |
| [`models/`](models/) | Define networks, and load trained ones back |
| [`trainers/`](trainers/) | Train, evaluate, and save checkpoints |
| [`utils/`](utils/) | Small helpers: paths, seeding, run folders, run summaries |
| [`main.py`](main.py) | Connect the pieces: read the config, build each part, run the experiment |

A few Python features make this work:

- **`__init__.py`** marks a folder as a package and chooses what it exposes.
  For example, [`models/__init__.py`](models/__init__.py) contains
  `from .convnet import ConvNet`, so other code writes
  `from .models import ConvNet` without knowing which file defines it.
- **`__all__`** in `__init__.py` lists the public names of the package (what
  `from package import *` imports, and what tools treat as its API).
- **Relative imports** such as `from .models import ConvNet` (the dot means
  "this package") find modules inside the project wherever it is, so the
  project folder can be renamed or copied without changing imports.
- **`__main__.py`** is what Python runs for `python -m TrainingBasicConvnet`.
  It only calls `main()`.

Why bother? Each part can be read, tested, and replaced on its own. To try a
new architecture, add a file to `models/` and change one line in `main.py`;
the data and training code stay untouched. When something breaks, the folder
names tell you where to look.

## Loading a trained model

Every checkpoint stores the run's config, so `load_model` can rebuild the
network with the right settings and load its weights:

```python
from TrainingBasicConvnet import load_model

model = load_model("config01")                       # newest run of config01, best checkpoint
model = load_model("config01/2026-10-02_11-20-01")   # one specific run
model = load_model("config01", which="last")         # the last epoch instead of the best
logits = model(images)                               # images: (N, 1, 28, 28), normalized
```

Run names are looked up in `<OUTPUT_DIR>/TrainingBasicConvnet/`; a full path
to a run folder or a `.pt` file works too. The model comes back in eval mode on
the CPU (pass `device="cuda"` for a GPU).

## Things to try

- Copy `configs/config01.json` to `config02.json`, change the learning rate,
  dropout, or number of epochs, run it, and compare the runs in
  `runs_summary.csv`.
- Add a second network to `models/` (for example, a third convolution block),
  export it in `models/__init__.py`, and use it in `main.py`.
- Then look at [LitTrainingBasicConvnet](../LitTrainingBasicConvnet/), which
  runs the same experiment with PyTorch Lightning, and compare how much code
  each needs.
