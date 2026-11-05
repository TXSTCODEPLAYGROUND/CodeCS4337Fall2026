# LitWBTransferLearningResnet

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBTransferLearningResnet/lit_wb_transfer_learning_resnet_notebook.ipynb)
This project's notebook: [`lit_wb_transfer_learning_resnet_notebook.ipynb`](lit_wb_transfer_learning_resnet_notebook.ipynb).

**Transfer learning** with a ResNet-18 pretrained on ImageNet, applied to
[Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/):
102 flower species with only **10 training images per class**. Three configs
compare the three ways of using a pretrained network, and W&B gets a full
evaluation of each model on the validation and test sets: prediction tables,
galleries of right and wrong predictions, per-class metrics,
precision-recall curves, and a confusion matrix.

The project follows the Lightning + W&B setup of
[LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/) (same `main`,
loggers, checkpoints, early stopping, resuming, and `load_model`); read that
project first, in particular how to
[set up your W&B key](../LitWBTrainingBasicConvnet/README.md#set-up-your-wb-key)
and [offline mode](../LitWBTrainingBasicConvnet/README.md#no-key-or-failed-login-offline-mode).
The model is the one built by hand in section 8 of
[ResNetWalkThrough](../ResNetWalkThrough/): a ResNet-18 backbone with a new
classification head.

How to set up and run the project and where results go is explained in the
[main README](../README.md). Function and class details are in the
[API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitWBTransferLearningResnet.html).

## Boilerplate lives in `colabutilscs4337`

Setting up Colab, finding paths and configs, creating run folders, resuming, and
connecting to W&B were copied into every project. They now come from the
course library [`colabutilscs4337`](https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html), which is
in [`requirements.txt`](../requirements.txt), so this project keeps only what is
specific to the experiment: the data, the model, the transfer-learning
strategies, and the evaluation report.

| Was in this project | Now comes from `colabutilscs4337` |
| --- | --- |
| `utils/paths.py`: repo, `configs/`, `.env`, `DATA_DIR`, `OUTPUT_DIR` paths | `cu.project.Project` (`PROJECT` in [`utils/paths.py`](utils/paths.py)) |
| `utils/experiment.py`: `runs_summary.csv` | `cu.project.append_run_summary` |
| `models/loading.py`: finding a run, its checkpoints, and its config | `Project.find_checkpoint`, `find_resume_checkpoint`, `run_config` |
| `callbacks/early_stopping.py` | `cu.lit.ResumableEarlyStopping` |
| `main.py`: command line, resume, run folder, `CSVLogger`, checkpoints, `config.json` and `hparams.json` | `cu.lit.start_run` |
| `main.py`: `_connect_wandb` | `cu.lit.connect_wandb` |
| Notebook: mount Drive, clone, `.env`, requirements, API keys | `cu.project.setup_colab`, `cu.env.load_secrets`, `cu.env.sync_env_file` |

## Running

From the repo root, one config at a time:

```bash
python -m LitWBTransferLearningResnet --config config01.json   # frozen backbone
python -m LitWBTransferLearningResnet --config config02.json   # fine-tuning
python -m LitWBTransferLearningResnet --config config03.json   # from scratch
```

The dataset (about 345 MB) is downloaded to `data/flowers-102/` on the first
run. Use a GPU: on Colab's T4, an epoch takes seconds, while on a CPU every
evaluation of the 6,149 test images takes minutes.

## The dataset

| Split | Images | Per class | Used for |
| --- | --- | --- | --- |
| train | 1,020 | 10 | training (augmented) |
| val | 1,020 | 10 | early stopping, best checkpoint, validation report |
| test | 6,149 | 20 to 238 | final test report |

This is the dataset's official split. With 10 images per class, a network
trained from scratch has very little to learn from: exactly the situation
where transfer learning pays off.

**Augmentation** ([`dataloaders/flowers102.py`](dataloaders/flowers102.py)):
every training image is a random crop (8 % to 100 % of the image, resized to
224 x 224) with a random horizontal flip, so each epoch sees new versions of the
1,020 images. Validation and test images get ImageNet's evaluation
preprocessing: resize the short side to 256, take the center 224 x 224 crop.
All images are normalized with the **ImageNet** mean and standard deviation,
the statistics the pretrained weights expect.

## Three strategies, three configs

| Config | `model` settings | Trainable parameters | Learning rates |
| --- | --- | --- | --- |
| `config01.json` **feature extraction** | `"pretrained": true, "freeze_backbone": true` | 52,326 (the head only) | head `1e-3` |
| `config02.json` **fine-tuning** | `"pretrained": true, "freeze_backbone": false` | 11,228,838 (everything) | head `1e-3`, backbone `1e-4` |
| `config03.json` **from scratch** | `"pretrained": false` | 11,228,838 (everything) | `1e-3` for both |

- **Feature extraction** keeps the ImageNet features as they are and trains
  only a new linear layer on top: fast, hard to overfit, and often already
  good, because edges, textures, and shapes learned on ImageNet transfer well.
- **Fine-tuning** also adjusts the backbone, with a 10 times smaller learning
  rate, so the pretrained features are nudged towards flowers instead of
  being overwritten by the large gradients of the still random head.
- **From scratch** starts from random weights: the baseline that shows what
  the ImageNet weights are worth. It gets more epochs and more patience
  (60 and 10), and still has only 1,020 images to learn from.

Run all three, then compare `val_acc`, `test_acc`, and `test_f1` in W&B (the
runs are grouped by config) or in each config's `runs_summary.csv`.

### What is in the model

[`models/components/resnet.py`](models/components/resnet.py):

```python
backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)   # or weights=None
backbone.fc = nn.Identity()                                    # now outputs 512 features
head = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, 102))     # the new classifier
```

Two details matter for transfer learning:

1. **Freezing** sets `requires_grad = False` on every backbone parameter, and
   the optimizer only gets the parameters that still need gradients.
2. **BatchNorm stays in eval mode when frozen.** Freezing the weights does
   not freeze BatchNorm's running mean and variance: in training mode they keep
   updating from the flower batches, so the "frozen" backbone would still
   change. `ResNetClassifier.train()` puts a frozen backbone back in eval mode
   every time Lightning switches to training.

[`models/lit_resnet.py`](models/lit_resnet.py) gives Adam two parameter
groups, the head at `lr` and the backbone at `backbone_lr`:

```python
groups = [{"params": net.head.parameters(), "lr": lr}]
groups.append({"params": trainable_backbone_params, "lr": backbone_lr})   # skipped when frozen
torch.optim.Adam(groups, weight_decay=weight_decay)
```

## What is logged

Per epoch, for `train`, `val`, and `test` (`self.log` in `LitResNet`):

| Metric | Meaning |
| --- | --- |
| `<stage>_loss` | cross-entropy loss |
| `<stage>_acc` | accuracy over all images |
| `<stage>_acc_top5` | the true class is among the 5 highest scores: with 102 similar species, it shows whether a mistake was a near miss |
| `<stage>_precision`, `<stage>_recall`, `<stage>_f1` | **macro** averages: computed per class, then averaged, so every class counts the same |

After training, `main` evaluates the **best checkpoint** on the validation set
(`trainer.validate`) and then on the test set (`trainer.test`), and
[`callbacks/wandb_evaluation.py`](callbacks/wandb_evaluation.py) logs a full
report of each. The validation runs during training only log the metrics
above, so the reports are logged once per split, not every epoch.

| In W&B (`val_...` and `test_...`) | What it shows |
| --- | --- |
| `<split>_predictions` | table of 200 random images: `image`, `true`, `predicted`, `confidence`, `correct`, `in_top5` |
| `<split>_correct_examples` | 16 random right predictions, captioned *true / predicted (confidence)* |
| `<split>_wrong_examples` | the 16 **most confident mistakes**, with the same captions |
| `<split>_per_class` | one row per class: `images`, `accuracy`, `precision`, `recall`, `f1`, `most_confused_with` |
| `<split>_pr_curve` | precision-recall curves of the 5 worst and 5 best classes by F1, with each class's F1 and average precision (AP) in the legend |
| `<split>_confusion_matrix` | true vs. predicted class (only non-empty cells are sent, the others show as blank) |
| `<split>_most_confused` | the 10 most frequent mistakes: `true`, `predicted`, `count`, `share_of_true_class` |

The run also prints a one-line summary per split: accuracy, top-5 accuracy,
number of mistakes, and the 5 classes with the lowest F1.

### Reading the report

- **Per-class table:** sort by `f1` to find the hardest species. A class's
  `accuracy` is its `recall`, the share of its images predicted correctly;
  `precision` is the share of the predictions *of* that class that are
  correct. Low precision means other species are wrongly sent to it.
- **Wrong examples:** confident mistakes are the most informative. Often the
  two species really look alike, or the photo is unusual (a bud, a close-up of
  leaves). Compare with `most_confused_with` and `<split>_most_confused`.
- **Predictions table:** click **Filter** and enter `row["correct"] = false`
  to see only the mistakes, or `row["in_top5"] = true and row["correct"] = false`
  for the near misses. Group by `true` to count mistakes per class.
- **PR curves:** for one class against all others, as the confidence
  threshold goes down, recall goes up and precision usually goes down. A curve
  that stays near the top right (AP close to 1) separates the class well at any
  threshold.
- **Validation vs. test:** the test set is 6 times larger, so its numbers are
  more reliable; the validation set was also used to choose the best
  checkpoint, which makes its numbers slightly optimistic.

### Build the dashboard

As in [LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/README.md#build-the-dashboard),
add line plots with `epoch` as X: `train_loss` and `val_loss`; `train_acc`
and `val_acc`; `val_acc` and `val_acc_top5`. With all three configs run,
every chart shows one line per run: the frozen and fine-tuned runs start far
above the from-scratch run and stay there.

## Mixed precision

This project trains in 32-bit floats (`float32`), the default. Mixed
precision runs most of the network in 16-bit floats instead, and Lightning
turns it on with one argument:

```python
trainer = L.Trainer(
    max_epochs=epochs,
    accelerator=config["accelerator"],
    precision="16-mixed",        # float16 where safe, float32 where needed
    logger=[csv_logger, wandb_logger],
    callbacks=callbacks,
)
```

To make it a config setting, add `"precision": "16-mixed"` to the config and
pass `precision=config.get("precision", "32-true")`.

**Why someone would need it:**

- **Speed.** GPUs have Tensor Cores that multiply 16-bit matrices several times
  faster than 32-bit ones. On Colab's T4, convolutions and linear layers often
  run 1.5 to 3 times faster.
- **Memory.** Activations take half the memory, so a larger batch, larger
  images, or a larger model (ResNet-50, ResNet-101) fits on the same GPU.
- **Little or no loss in accuracy.** The weights are kept in `float32`, and
  only the forward and backward computations use 16 bits.

**What Lightning does for you with `"16-mixed"`:**

- **Autocast** picks the precision per operation: matrix multiplications and
  convolutions in `float16`; numerically sensitive ones (softmax, losses, sums,
  BatchNorm statistics) stay in `float32`.
- **Gradient scaling** (`GradScaler`): `float16` cannot represent very small
  numbers, so tiny gradients would become zero. The loss is multiplied by a
  large factor before `backward`, and the gradients are divided by it again
  before the optimizer step; steps whose gradients overflow are skipped and
  the factor is lowered.

**`bf16-mixed`:** `bfloat16` has the same range as `float32` (fewer digits,
but no underflow), so no gradient scaling is needed. Use
`precision="bf16-mixed"` on GPUs that support it well (A100, L4, RTX 30xx and
newer); the T4 does not, so use `"16-mixed"` there. On a CPU, mixed precision
rarely makes training faster.

The rest of the project works unchanged: torchmetrics and the evaluation
callback convert the logits to `float32` before computing probabilities and
metrics, and checkpoints still hold `float32` weights.

## Run folder

Each run writes these files to
`runs/LitWBTransferLearningResnet/<config>/<timestamp>/`:

```
├── config.json                     # Exact copy of the config used
├── hparams.json                    # Class names, learning rates, and data settings
├── metrics.csv                     # Every metric, every epoch (local backup)
├── best_epochNN_valaccX.XXXX.ckpt  # Checkpoint with the best validation accuracy
├── last.ckpt                       # Checkpoint of the last epoch, rewritten every epoch
└── wandb/                          # W&B's local copy of the run (offline runs are synced from here)
```

`runs_summary.csv` (one row per run, next to the run folders) also records
`pretrained`, `freeze_backbone`, `trainable_params`, `test_acc_top5`, and
`test_f1`, so the three strategies can be compared without W&B.

To load a trained model back (all weights come from the checkpoint; the
ImageNet weights are not downloaded again):

```python
from LitWBTransferLearningResnet import load_model

model = load_model("config02")    # newest run of config02, best checkpoint
logits = model(images)            # images: (N, 3, 224, 224), ImageNet-normalized
```

To train a run further, resume it: `main("config02.json",
resume_from="config02")`, or `--resume-from config02` on the command line. The
model settings in the config must match the run's, in particular
`freeze_backbone`, which decides what the optimizer trains.

## Training with the Colab CLI

Another way to train this project, once everything is final. Its only purpose
is the training run itself:

1. **Work locally** on the project: change the code or a config, run short
   tests, fix bugs.
2. **Use Google Colab** (this project's notebook) for debugging and interactive work.
3. **Use the Colab CLI** when everything is final and the only thing left is
   training. The CLI creates a Colab runtime from your local terminal and opens
   a shell on it, so there is no notebook or browser tab to keep open.

The Colab CLI works on Linux and macOS (not Windows). For more detail, see
[google-colab-cli.md](../google-colab-cli.md) at the repo root or the
[official Colab CLI documentation](https://github.com/googlecolab/google-colab-cli).

### Step 1: Install the CLI and sign in

On your own machine:

```bash
pip install google-colab-cli     # or: uv tool install google-colab-cli
colab version                    # check that it works
```

The first command that talks to Colab (`colab version`, `colab new`, …) prints a
sign-in link: open it, sign in with your Google account, and paste the code it
shows back into the terminal. You only need to do this once per machine.

### Step 2: Create `cs4337.env` on Google Drive

Colab Secrets can only be read from a notebook, not from an SSH session, so the
keys for this project live in a plain text file on your Google Drive. Create a
file named `cs4337.env` in the **root of My Drive** (so its path in the session
will be `/content/drive/MyDrive/cs4337.env`), with one `NAME=value` per line:

```text
# Settings shared by every project in this repo.
# Relative paths are resolved against the repo root, not the current directory.

# Required
# Each project writes to <OUTPUT_DIR>/<ProjectName>/<config>/<timestamp>/.
DATA_DIR=/content/data
OUTPUT_DIR=/content/drive/MyDrive/CodeCS4337Fall2026/runs

# Required
# W&B key for LitWBTrainingBasicConvnet, from wandb.ai/authorize (uncomment and paste it):
# WANDB_API_KEY=

# Optional
# LitLogger key for LitTrainingBasicConvnet, from lightning.ai/profile (uncomment and paste it):
LIGHTNING_USER_ID=
LIGHTNING_API_KEY=

# optional
# LABEL_STUDIO_USERNAME=email@email.com   # must look like an email address
# LABEL_STUDIO_PASSWORD=password
```

`setup_colab` writes `.env` from the repo's `.envcolab` and then appends this
file, so later lines win when a name appears twice. Uncomment and set
`WANDB_API_KEY` for this project. Without it the run still trains, but W&B
logs offline into the run folder; upload it later with the `wandb sync` command
the run prints. Never put keys in `.envcolab`, which is in git.

You can also create the file from inside the session after Drive is mounted
(step 4): `read -s` hides the key while you paste it:

```bash
read -rsp "W&B API key: " key && echo "WANDB_API_KEY=$key" >> /content/drive/MyDrive/cs4337.env && unset key && echo
```

### Step 3: Start a GPU session

These commands run in your **local** terminal. A **session** is one Colab
runtime (a virtual machine) with a name you choose, here `cs4337`:

```bash
colab new -s cs4337 --gpu T4
```

`--gpu T4` is enough for this project. Other GPUs (`L4`, `A100`, `H100`), CPU
sessions, high-memory sessions, listing and stopping sessions, and connecting
from VS Code are covered in [google-colab-cli.md](../google-colab-cli.md) and the
[official Colab CLI documentation](https://github.com/googlecolab/google-colab-cli):
read those when you want more than a plain GPU session.

> **Once a session is stopped, nothing on it stays.** The cloned repo, the
> downloaded data, and every file that is not on Google Drive are deleted.
> This project writes its runs to Google Drive (step 5), so mount Drive before
> training.

### Step 4: Mount Google Drive and attach

Drive is mounted from your local terminal, not from inside the session:

```bash
colab drivemount -s cs4337          # open the link it prints and allow access
colab ssh -s cs4337                 # attach to the session
```

Inside the session, check that the GPU is there and that Drive is mounted:

```bash
nvidia-smi                          # GPU name, memory, utilization
ls /content/drive/MyDrive           # your Drive files, including cs4337.env
```

Other monitoring tools (`btop`, `gpustat`, `tmux` shortcuts) are listed in the
[main README](../README.md#3-useful-monitoring-tools) under *Useful Monitoring
Tools*. Nothing in the project mounts Drive again: the setup in step 5 uses this
mount.

### Step 5: Set up the repo and train

One call of the course library
[`colabutilscs4337`](https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html) sets up everything the
runtime needs: it clones the repo (or pulls it, if it is already there), writes
`.env` (`.envcolab` with your env file added below it), and installs only the
packages Colab doesn't already have (see [Why only the missing requirements are installed](../README.md#why-only-the-missing-requirements-are-installed)).
Drive is already mounted from your local terminal, so nothing is mounted again.
Inside the session:

```bash
pip install -q colabutilscs4337
python - <<'EOF'
import colabutilscs4337 as cu

cu.project.setup_colab(
    "https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026.git",
    env_source="/content/drive/MyDrive/cs4337.env",  # your env file on Google Drive
    mount=False,  # `colab drivemount` already mounted Drive
)
EOF
cd /content/CodeCS4337Fall2026
```

The runtime only sees what is on GitHub: if you changed code or a config
locally, push it to your own fork first and use that URL instead. If
`cs4337.env` is not on Drive, `setup_colab` stops with an error that names the
path it looked for.

Start a `tmux` session so training keeps going if the SSH connection drops or
you close your laptop, then run from the repo root:

```bash
tmux new -s train                  # if tmux is missing: apt-get install -y tmux
python -m LitWBTransferLearningResnet --config config01.json
```

To leave training running, press `Ctrl+B`, then `D` (detach). The process keeps
going on the runtime. To come back later: `colab ssh -s cs4337`, then
`tmux attach -t train`.

### Step 6: Monitor the run

While training (or after you detach from `tmux`):

- **W&B:** open the run link that the process prints, or go to your W&B project
  in the browser. Metrics, the config, and the system charts update there live
  when the key was present. If the run printed a `wandb sync …` command instead,
  it logged offline; run that command later from a machine with your key.
- **Run folder on Google Drive:** every file of the run (config, metrics,
  checkpoints, W&B's local `wandb/` folder) is under

  ```text
  /content/drive/MyDrive/CodeCS4337Fall2026/runs/LitWBTransferLearningResnet/config01/2026-11-05_14-30-00/
  ```

  The timestamp is printed when the run starts. From the session:

  ```bash
  ls /content/drive/MyDrive/CodeCS4337Fall2026/runs/LitWBTransferLearningResnet/config01
  ```

  or on [drive.google.com](https://drive.google.com), under **My Drive >
  CodeCS4337Fall2026 > runs > LitWBTransferLearningResnet**. Check that `runs_summary.csv` next to
  the run folders has one new row per run.

- **Resume a cut-off run:** start a new session, repeat the steps above, and
  continue from the last checkpoint on Drive:

  ```bash
  python -m LitWBTransferLearningResnet --config config01.json --resume-from config01
  ```

When everything is checked, stop the session so it no longer uses your Colab
quota:

```bash
colab stop -s cs4337
```

## Things to try

- Compare the three configs: how much better than from scratch are the
  pretrained runs, and how many epochs does each need?
- Look at `test_wrong_examples` of the fine-tuned model: are the mistakes
  species you could tell apart yourself?
- In `config02.json`, set `"backbone_lr"` equal to `"lr"` (`0.001`): does
  fine-tuning still beat feature extraction, or do the large updates damage
  the pretrained features?
- Two-stage training (feature extraction first, then fine-tuning) is common.
  Resuming a `config01` run with `config02`'s model settings fails: the
  optimizer saved in the checkpoint has one parameter group, the new one has
  two. How would you add an `"unfreeze_after"` setting instead? (Hint: a
  callback that sets `requires_grad = True` on the backbone and calls
  `optimizer.add_param_group`.)
- Turn off augmentation (use `eval_transform` for training too) and compare
  the gap between `train_acc` and `val_acc`.
- Try `models.resnet50` (2,048 features instead of 512) with mixed precision on
  a Colab GPU.
