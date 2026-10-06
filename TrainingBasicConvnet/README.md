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
to a run folder or a `.pt` file works too, e.g.
`load_model("config01/2026-10-02_11-20-01/best_epoch10_valacc0.9248.pt")`.
Each run folder keeps two checkpoints, the best and the last epoch; older ones
are replaced during training, so other epochs can't be loaded. The model comes back in eval mode on
the CPU (pass `device="cuda"` for a GPU).

## Continuing training

Running a config trains from scratch. To train an earlier run further, for
example because it needed more epochs or a Colab disconnect stopped it, pass
`resume_from`:

```python
from TrainingBasicConvnet import main

main("config01.json", resume_from="config01")   # newest run of config01
```

```bash
python -m TrainingBasicConvnet --config config01.json --resume-from config01
```

The trainer saves `last_epochXX_valaccY.pt` after **every** epoch (replacing
the previous one), with the model and the optimizer state. `Trainer.resume`
loads both and the epoch number, and `fit` then trains the config's `"epochs"`
**more** epochs, numbered after the checkpoint's: a 10-epoch run resumed with
`config01.json` trains epochs 11 to 20. The continued run gets its own run
folder and `runs_summary.csv` row, and its results file records
`resumed_from`; the earlier run is not changed.

- `resume_from` accepts the same names as `load_model`: a config name (its
  newest run), a run folder, or a checkpoint file. A run stopped before its
  first `last_*.pt` continues from its best checkpoint.
- The config's model settings must match the run's. To train a different
  number of extra epochs, copy the config, change `"epochs"`, and pass the copy
  with `resume_from="config01"`.
- The learning rate and weight decay come from the saved optimizer, so changing
  `lr` or `weight_decay` in the config has no effect when resuming. The data
  settings and seed come from the config.
- Resuming again continues further: `"config01"` then means the continued run.

## Training with the Colab CLI

Another way to train this project, once everything is final. Its only purpose
is the training run itself:

1. **Work locally** on the project: change the code or a config, run short
   tests, fix bugs.
2. **Use Google Colab** (this project's notebook) for debugging and interactive work.
3. **Use the Colab CLI** when everything is final and the only thing left is
   training. The CLI creates a Colab runtime from your local terminal and opens
   a shell on it, so there is no notebook or browser tab to keep open. On the
   runtime you clone the repo and run `python -m TrainingBasicConvnet --config config01.json`.

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
python -m TrainingBasicConvnet --config config01.json
```

To leave training running, press `Ctrl+B`, then `D`. To come back later:
`colab ssh -s cs4337`, then `tmux attach -t train`.

### Step 5: Check that everything was logged

The run folders are on Google Drive, so they are kept after the session
stops. From the session:

```bash
ls /content/drive/MyDrive/CodeCS4337Fall2026/runs/TrainingBasicConvnet/config01
```

or on [drive.google.com](https://drive.google.com), under **My Drive >
CodeCS4337Fall2026 > runs > TrainingBasicConvnet**. Check that:

- each run has its own `<timestamp>` folder (the path is printed when the run starts), with its config, results, plots, and checkpoints;
- `runs_summary.csv`, next to the run folders, has one new row per run;
- if the session stopped or the run was cut off, start a new session, repeat the steps above, and continue from the last checkpoint on Drive with `--resume-from config01`, e.g. `python -m TrainingBasicConvnet --config config01.json --resume-from config01`.

When everything is checked, stop the session so it no longer uses your Colab
quota:

```bash
colab stop -s cs4337
```

## Things to try

- Copy `configs/config01.json` to `config02.json`, change the learning rate,
  dropout, or number of epochs, run it, and compare the runs in
  `runs_summary.csv`.
- Add a second network to `models/` (for example, a third convolution block),
  export it in `models/__init__.py`, and use it in `main.py`.
- Then look at [LitTrainingBasicConvnet](../LitTrainingBasicConvnet/), which
  runs the same experiment with PyTorch Lightning, and compare how much code
  each needs.
