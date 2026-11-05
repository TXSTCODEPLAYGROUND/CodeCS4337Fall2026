# UnetSemanticSegmentationExample

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/UnetSemanticSegmentationExample/unet_semantic_segmentation_example_notebook.ipynb)
This project's notebook: [`unet_semantic_segmentation_example_notebook.ipynb`](unet_semantic_segmentation_example_notebook.ipynb).

**Semantic segmentation** with a **U-Net**: for every pixel of a colonoscopy
image, decide whether it belongs to a **polyp** (a growth on the wall of the
colon that doctors look for, because some turn into cancer) or to the
background. The data is [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/):
1,000 images, each with a mask drawn by a doctor. Three configs compare a
baseline U-Net, better training of the same U-Net, and a U-Net whose encoder
is a ResNet-34 pretrained on ImageNet.

The project follows the Lightning + W&B setup of
[LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/) (same `main`,
loggers, checkpoints, early stopping, resuming, and `load_model`); read that
project first, in particular how to
[set up your W&B key](../LitWBTrainingBasicConvnet/README.md#set-up-your-wb-key)
and [offline mode](../LitWBTrainingBasicConvnet/README.md#no-key-or-failed-login-offline-mode).
Transfer learning (frozen encoder, two learning rates, BatchNorm in eval mode)
is introduced in [LitWBTransferLearningResnet](../LitWBTransferLearningResnet/).

How to set up and run the project and where results go is explained in the
[main README](../README.md). Function and class details are in the
[API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/UnetSemanticSegmentationExample.html).

## The notebook

The notebook is the main teaching material. It goes through a segmentation
project in order: **Dataset → Exploration → Tensors → DataLoader → U-Net →
Inspection → Loss → Training → Evaluation → Predictions → Comparison →
Fine-tuning → Extensions**.

| Section | What you do |
| --- | --- |
| 3. Segmentation vs. classification | The pipeline image → U-Net → logits → sigmoid → probability map → threshold → mask, and the shapes of `X` and `y` in both tasks |
| 4. Explore Kvasir-SEG | Image sizes, raw JPEG mask values and binarization, foreground share per image (class imbalance), overlays |
| 5. Files to tensors | Every preprocessing step with its shape, dtype, and value range; paired augmentation of image and mask; one DataLoader batch |
| 6. Binary cross-entropy | One pixel by hand (`y = 1`, `z = 1.5`), a 2 x 2 example against `F.binary_cross_entropy_with_logits`, reductions over `[B, 1, H, W]`, why `BCEWithLogitsLoss` is stable where sigmoid + `BCELoss` is not, and the Dice loss |
| 7. The U-Net | The architecture as a diagram, the code, concatenation, and the shape after every block |
| 8. torchinfo | How to read a summary: parameters checked by hand, multiply-adds, memory |
| 9. Netron and torchview | ONNX export shown inside the notebook (Colab and local), where the skip connections are in the graph |
| 10. A plain PyTorch loop | The training loop written out, then how it maps to Lightning |
| 11. Three experiments | The configs side by side, every setting explained, running the project |
| 12. The fine-tuned model | Pretrained vs. new layers, frozen vs. trainable parameter counts, BatchNorm in a frozen encoder, ImageNet filters |
| 13-15. Evaluation | Learning curves, predictions colored by true/false positives and negatives, probability maps, a results table with costs, and discussion questions |
| 16. Reuse | `load_model` and resuming a run |
| 17. Other datasets | 16 segmentation datasets (medical, driving, aerial, pets, people, industrial), binary and multiclass, with what to change for each |
| 18. Experiment ideas | 14 experiments, each with a hypothesis |

## Boilerplate lives in `colabutilscs4337`

Setting up Colab, finding paths and configs, creating run folders, resuming, and
connecting to W&B were copied into every project. They now come from the
course library [`colabutilscs4337`](https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html), which is
in [`requirements.txt`](../requirements.txt), so this project keeps only what is
specific to the experiment: the dataset, the U-Net models, and the
segmentation report.

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
python -m UnetSemanticSegmentationExample --config config01.json   # baseline
python -m UnetSemanticSegmentationExample --config config02.json   # improved training
python -m UnetSemanticSegmentationExample --config config03.json   # pretrained encoder
```

The dataset (44 MB) is downloaded to `data/kvasir-seg/` on the first run.
Use a GPU: each config trains up to 100 epochs, about 10 to 20 minutes on
Colab's T4 (config03 usually stops earlier).

**Download note:** the dataset's server (`datasets.simula.no`) sends an
incomplete certificate chain, which makes Python reject the connection
(`CERTIFICATE_VERIFY_FAILED`). `prepare_kvasir` then retries once with the
missing intermediate certificate, downloaded from the address given in the
server's own certificate, added to the trusted ones; certificate checking
stays on.

## The dataset

| Split | Images | Used for |
| --- | --- | --- |
| train | 800 | training (augmented) |
| val | 100 | early stopping, best checkpoint, validation report |
| test | 100 | final test report |

The split is random with the config's seed, so it is the same in every run
and in the notebook. The images come in 333 different sizes (from 332 x 487 to
1920 x 1072) and are resized to 256 x 256. The masks are JPEG files whose
values are only close to 0 and 255, so they are thresholded at 128. On
average 15 % of an image is polyp (from 0.5 % to 81 %): most pixels are
background.

[`dataloaders/kvasir_seg.py`](dataloaders/kvasir_seg.py) wraps each image and
mask as `tv_tensors.Image` and `tv_tensors.Mask`, so torchvision's
`transforms.v2` applies the same random flip, crop, or rotation to both, and
resizes masks with nearest-neighbor interpolation (they stay 0 and 1).
Images are normalized with the ImageNet mean and standard deviation.

| `"augmentation"` | Training transform |
| --- | --- |
| `"none"` | resize only |
| `"minimal"` | resize, horizontal flip |
| `"strong"` | random crop of 50 to 100 % of the image, horizontal and vertical flips, rotation up to 15°, brightness, contrast, saturation, and hue changes (image only) |

## Three experiments, three configs

The configs have the same keys; only values differ.

| Config | Model | Loss | Optimizer, schedule | Augmentation |
| --- | --- | --- | --- | --- |
| `config01.json` **baseline** | `UNet`, 7.8 million parameters, from scratch | BCE | Adam, `lr` 1e-3, constant | minimal |
| `config02.json` **improved training** | the same `UNet` | BCE + Dice | AdamW, weight decay 1e-4, cosine schedule | strong |
| `config03.json` **fine-tuning** | `ResNetUNet`, 24.4 million parameters: ResNet-34 encoder pretrained on ImageNet + new decoder | BCE + Dice | AdamW, decoder `lr` 1e-3, encoder `encoder_lr` 1e-4, encoder frozen for 3 epochs, cosine schedule | strong |

All three train up to 100 epochs with early stopping on `val_dice` (patience
15), and with mixed precision (`"precision": "16-mixed"`) when a GPU is
available. Every setting is explained in section 11 of the notebook.

### The models

[`models/components/unet.py`](models/components/unet.py) is a U-Net written
out in three small classes: `DoubleConv` (two 3 x 3 convolutions with
BatchNorm and ReLU), `UpBlock` (transposed convolution, concatenation with the
skip connection, `DoubleConv`), and `UNet` (4 encoder levels of 32 to 256
channels, a 512-channel bottleneck at 16 x 16, 4 decoder levels, a 1 x 1
convolution head that outputs one logit per pixel).

[`models/components/resnet_unet.py`](models/components/resnet_unet.py) keeps
torchvision's ResNet-34 without its classifier as the encoder (its stem and
four stages give the skip connections at 1/2, 1/4, 1/8, and 1/16 of the image
size; the last stage, at 1/32, is the bottleneck) and adds a new decoder of
bilinear upsampling, concatenation, and `DoubleConv` blocks.
`set_encoder_frozen` turns gradients off for the encoder, and `train()` keeps
a frozen encoder's BatchNorm layers in eval mode, so their ImageNet statistics
are not overwritten.

[`models/lit_unet.py`](models/lit_unet.py) (`LitUNet`) picks the loss
([`models/losses.py`](models/losses.py): `bce`, `dice`, `bce_dice`, `focal`),
unfreezes the encoder after `freeze_encoder_epochs`, and gives the optimizer
two parameter groups when `encoder_lr` is set.

## What is logged

Per epoch, for `train`, `val`, and `test`, with every pixel of every image
counted together:

| Metric | Meaning |
| --- | --- |
| `<stage>_loss` | the config's loss (not comparable between configs with different losses) |
| `<stage>_dice` | Dice coefficient, `2 TP / (2 TP + FP + FN)`: the overlap of the predicted and true polyp; the main metric |
| `<stage>_iou` | intersection over union, `TP / (TP + FP + FN)`; always lower than Dice |
| `<stage>_precision` | share of the pixels predicted as polyp that are polyp |
| `<stage>_recall` | share of the polyp pixels that are found |
| `<stage>_pixel_acc` | share of all pixels classified correctly; high even for a useless model, because 85 % of the pixels are background |

`TP`, `FP`, and `FN` are the true positive, false positive, and false negative
pixels, with a pixel predicted as polyp when its probability is at least 0.5.
The learning rate of each parameter group (`lr-AdamW/pg1` is config03's
encoder, `pg2` its decoder) and `encoder_frozen` are logged too.

[`callbacks/wandb_segmentation.py`](callbacks/wandb_segmentation.py) adds images.
Each image is one strip of four panels side by side, so the masks can be
compared at a glance:

| image | ground truth | prediction | errors |
| --- | --- | --- | --- |
| the endoscopy image | true polyp in green | predicted polyp in blue | green: true positive (polyp found), red: false positive (background predicted as polyp), yellow: false negative (polyp missed) |

The panel titles give the polyp's share of the image, and the caption gives
the image's name and Dice.

| In W&B | What it shows |
| --- | --- |
| `val_progress` | the same 8 validation images after every epoch; drag the step slider to watch the predictions improve |
| `val_per_image`, `test_per_image` | for the best checkpoint, one row per image: `dice`, `iou`, `polyp_area`, `predicted_area`; sort by `dice` to find the hardest images |
| `val_worst_examples`, `test_worst_examples` | the 8 images with the lowest Dice |
| `val_best_examples`, `test_best_examples` | the 8 images with the highest Dice |

The run summary also has `total_params`, `trainable_params`, `best_epoch`,
`train_minutes`, `model_size_mb`, and `inference_ms_per_image`.

## Results

One run of each config (seed 42, an RTX 5090 GPU; on Colab's T4, expect
training to take about 3 to 4 times longer). Scores are those of the best
checkpoint; your numbers will differ slightly.

| Config | Best val Dice | Test Dice | Test IoU | Test precision | Test recall | Epochs run (best) | Trainable parameters | Size | Inference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `config01` baseline | 0.819 | 0.810 | 0.681 | 0.868 | 0.759 | 100 (88) | 7.8 M | 31 MB | 0.6 ms/image |
| `config02` improved training | 0.864 | 0.868 | 0.766 | 0.883 | 0.853 | 100 (99) | 7.8 M | 31 MB | 0.6 ms/image |
| `config03` pretrained encoder | 0.912 | 0.883 | 0.790 | 0.856 | 0.911 | 37 (21) | 24.4 M | 98 MB | 0.5 ms/image |

- **Better training of the same network** (config02) gains about 6 points
  of Dice, mostly through **recall**: the Dice term in the loss stops the
  network from leaving polyp edges and small polyps as background, which
  BCE alone tolerates. Strong augmentation keeps its training and validation
  Dice close (0.88 and 0.86 in the last epoch): it barely overfits, and it
  was still improving at epoch 100.
- **The pretrained encoder** (config03) starts at a validation Dice of about
  0.69 after one epoch with a frozen encoder, and reaches its best score in
  21 epochs. It trades a little precision for a much higher recall.
- **Bigger is not slower here:** the ResNet-34 encoder shrinks the image 4
  times in its first two layers, while the U-Net runs 32- and 64-channel
  convolutions at full resolution, so config03 predicts slightly faster
  despite 3 times more parameters.
- With only 100 validation and 100 test images, differences of 1 to 2 points
  are within the noise of the split.

## Run folder

Each run writes these files to
`runs/UnetSemanticSegmentationExample/<config>/<timestamp>/`:

```
├── config.json                       # Exact copy of the config used
├── hparams.json                      # Model and data settings
├── metrics.csv                       # Every metric, every epoch (local backup)
├── best_epochNN_valdiceX.XXXX.ckpt   # Checkpoint with the best validation Dice
├── last.ckpt                         # Checkpoint of the last epoch, rewritten every epoch
└── wandb/                            # W&B's local copy of the run (offline runs are synced from here)
```

`runs_summary.csv` (one row per run, next to the run folders) has the test
metrics, `best_val_dice`, `best_epoch`, `epochs_run`, `train_minutes`,
`trainable_params`, `model_size_mb`, `inference_ms_per_image`, and the main
settings, so runs can be compared without W&B.

To load a trained model back (the network is rebuilt from the run's config;
all weights come from the checkpoint):

```python
from UnetSemanticSegmentationExample import load_model

model = load_model("config03")              # newest run of config03, best checkpoint
masks = torch.sigmoid(model(images)) >= 0.5   # images: (N, 3, 256, 256), ImageNet-normalized
```

To train a run further, resume it: `main("config02.json",
resume_from="config02")`, or `--resume-from config02` on the command line.
The model settings in the config must match the run's.

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
# Keys for CodeCS4337Fall2026. Never commit this file to git.
# Get the W&B key from https://wandb.ai/authorize

WANDB_API_KEY=your-api-key

# Optional: only for LitTraining* projects that use LitLogger (not this one)
# LIGHTNING_USER_ID=your-user-id
# LIGHTNING_API_KEY=your-api-key
```

This project needs `WANDB_API_KEY`. Without it the run still trains, but W&B
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
python -m UnetSemanticSegmentationExample --config config01.json
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
  /content/drive/MyDrive/CodeCS4337Fall2026/runs/UnetSemanticSegmentationExample/config01/2026-11-05_14-30-00/
  ```

  The timestamp is printed when the run starts. From the session:

  ```bash
  ls /content/drive/MyDrive/CodeCS4337Fall2026/runs/UnetSemanticSegmentationExample/config01
  ```

  or on [drive.google.com](https://drive.google.com), under **My Drive >
  CodeCS4337Fall2026 > runs > UnetSemanticSegmentationExample**. Check that `runs_summary.csv` next to
  the run folders has one new row per run.

- **Resume a cut-off run:** start a new session, repeat the steps above, and
  continue from the last checkpoint on Drive:

  ```bash
  python -m UnetSemanticSegmentationExample --config config01.json --resume-from config01
  ```

When everything is checked, stop the session so it no longer uses your Colab
quota:

```bash
colab stop -s cs4337
```

## Things to try

Section 17 of the notebook lists other segmentation datasets with what to
change for each (output channels, loss, mask encoding, image size, metrics),
and section 18 lists 14 experiments, each with a hypothesis. Three to start
with:

- config03 with `"pretrained": false`: how much of its lead comes from the
  ImageNet weights, and how much from the larger encoder?
- `"split_sizes": [100, 100, 100]` in config01 and config03: does
  pretraining matter more with fewer training images?
- `"image_size": 384`: are small polyps segmented better (sort
  `test_per_image` by `polyp_area`)?

## Dataset license

Kvasir-SEG may be used for research and education. Cite: D. Jha, P. H.
Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, and H. D.
Johansen, *Kvasir-SEG: A Segmented Polyp Dataset*, MultiMedia Modeling (MMM),
2020.
