# LitWBTransferLearningYolo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBTransferLearningYolo/lit_wb_transfer_learning_yolo_notebook.ipynb)
This project's notebook: [`lit_wb_transfer_learning_yolo_notebook.ipynb`](lit_wb_transfer_learning_yolo_notebook.ipynb).

**Transfer learning for object detection:** take YOLOv8n (YOLO, "You Only Look Once", version 8, size nano) pretrained on COCO (Common Objects in Context) and train it to find potholes. The dataset is the [IVCNZ pothole set](https://github.com/jaygala24/pothole-detection) (Chitale et al., 2020): 1,243 road photos, 4,076 boxes, one class. COCO has no pothole class, so the pretrained model cannot be scored as-is; every config trains a new 1-class head. The three configs compare a **frozen backbone**, **fine-tuning** every layer, and the same architecture **from scratch**. Weights & Biases (W&B) gets the training curves, precision, recall, F1, mAP50, mAP75, and mAP50-95 on the validation and test sets, the precision-recall curves and confusion matrices, a table of every validation and test image with its true and predicted boxes, and a comparison against the from-scratch run.

This project uses the same Ultralytics trainer, run folders, resume, and W&B wiring as [YoloExample](../YoloExample/README.md), and the same three transfer-learning strategies as [LitWBTransferLearningResnet](../LitWBTransferLearningResnet/README.md). YoloExample is the place to learn the YOLO label format, the network's output vector, and how a box is decoded; this project starts from a dataset that is already in that format. The W&B setup is the one in [LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/): [set up your W&B key](../LitWBTrainingBasicConvnet/README.md#set-up-your-wb-key) and [offline mode](../LitWBTrainingBasicConvnet/README.md#no-key-or-failed-login-offline-mode).

How to set up the repo, where `DATA_DIR` and `OUTPUT_DIR` point, and how Google Drive is mounted in Colab is in the [main README](../README.md). Function details are in the [API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitWBTransferLearningYolo.html).

## Running

From the repo root, one config at a time. Run config03 as well if you want the comparison table's gain over training from scratch:

```bash
python -m LitWBTransferLearningYolo --config config01.json   # frozen COCO backbone
python -m LitWBTransferLearningYolo --config config02.json   # fine-tune every layer
python -m LitWBTransferLearningYolo --config config03.json   # from scratch
```

The dataset zip is about 166 MB. The first run downloads it to `data/pothole_ivcnz/` and writes the YOLO layout to `data/pothole_yolo/`. The COCO weights (about 6 MB) go to `data/yolo_weights/`. Later runs read both from there.

In Colab, the notebook mounts Google Drive and copies `.envcolab` to `.env`. Datasets stay on the runtime's disk (`DATA_DIR=/content/data`). Every run folder is written under `OUTPUT_DIR=/content/drive/MyDrive/CodeCS4337Fall2026/runs`, so `weights/best.pt` and `weights/last.pt` survive a disconnect. Resume reads those folders.

A GPU is worth using. On a Colab T4, config01 and config02 are on the order of 10 to 20 minutes and config03 longer; a CPU run of all three is an overnight job. Ultralytics augments the training images itself (mosaic, flips, scale, color). The validation and test images are not augmented.

## The dataset

The images were collected from the web and labeled for the paper [Pothole Detection and Dimension Estimation System using Deep Learning (YOLO) and Image Processing](https://doi.org/10.1109/IVCNZ51579.2020.9290547) (IVCNZ 2020). The release is MIT licensed. One class, `pothole`, class index 0. With seed 42 the split is:

| Split | Images | Potholes | Used for |
| --- | --- | --- | --- |
| train | 995 | 3,232 | training, with Ultralytics' default augmentation |
| val | 124 | 504 | choosing the best epoch and early stopping |
| test | 124 | 340 | the final report, touched once |

About 80% of the images are train and 10% each are val and test (`VAL_FRACTION` and `TEST_FRACTION` in [`dataloaders/pothole.py`](dataloaders/pothole.py)). The shuffle is saved in `data/pothole_yolo/split.json`, so a later run with the same seed keeps the same images in each split.

Each label file is already YOLO format, one line per pothole:

```text
class x_center y_center width height
0 0.502404 0.638889 0.557692 0.380342
```

The four coordinates are the box center and size divided by the image width and height. [`prepare_pothole`](dataloaders/pothole.py) copies images and labels into `images/<split>/` and `labels/<split>/` and writes `pothole.yaml`, which is the file Ultralytics trains on.

## Configs

| Config | Strategy | Starting weights | What is trained |
| --- | --- | --- | --- |
| `config01.json` | **frozen backbone** | `yolov8n.pt` (COCO) | neck and the new 1-class head. `"freeze": 10` keeps layers 0 to 9 (the backbone) fixed. SGD, `lr0` 0.01, 30 epochs, patience 10 |
| `config02.json` | **fine-tuned** | `yolov8n.pt` (COCO) | every layer, so the backbone can change. SGD, `lr0` 0.001 (smaller, so 995 images do not wipe out the COCO features), 30 epochs, patience 10 |
| `config03.json` | **from scratch** | `yolov8n.yaml` (random weights) | every layer, from random weights. `optimizer` auto, `lr0` 0.01, 50 epochs, patience 15. This is the baseline |

All three use image size 640, batch 16, and the same seed. Ultralytics replaces the 80-class score layer with a 1-class layer when the weights are `yolov8n.pt` and the dataset has one class. Extra Ultralytics arguments go under `"training": {"ultralytics_args": {...}}` (see the [training settings](https://docs.ultralytics.com/modes/train/#train-settings)).

Freezing the backbone is the right first try when the new task has little data and the old features (edges, texture, road-like surfaces) are still useful. Fine-tuning lets those features move, which helps when potholes do not look like anything in COCO, and hurts when the small learning rate is still large enough to forget them. From scratch shows what the COCO weights were worth.

## Metrics

Detection scores are built in the same steps for every config. [YoloExample](../YoloExample/README.md#metrics) walks through them with the pedestrian model; the definitions used here are:

1. **IoU (intersection over union)** of a predicted box and a true box: overlap area divided by the area covered by both. 1 is a perfect fit, 0 is no overlap.
2. **Matching.** Predictions are taken from most to least confident. A prediction is a true positive when its IoU with a not-yet-matched true box is at least the threshold. Otherwise it is a false positive. Unmatched true boxes are false negatives.
3. **Precision** = true positives / (true positives + false positives). **Recall** = true positives / (true positives + false negatives). **F1** = 2 x precision x recall / (precision + recall).
4. **AP (average precision)** is the area under the precision-recall curve, over every confidence threshold. With one class, **mAP** (mean average precision) is that AP.

| Metric | Definition | What a high score means |
| --- | --- | --- |
| **mAP50** | AP at IoU ≥ 0.50 | the pothole was found; a loose box still counts |
| **mAP75** | AP at IoU ≥ 0.75 | the box also fits tightly |
| **mAP50-95** | the mean of AP at IoU 0.50, 0.55, ..., 0.95 | both. Ultralytics picks `weights/best.pt` with this number, and the comparison table uses it |

The reported precision, recall, and F1 are at the confidence that maximizes F1. mAP itself is computed with `"conf": 0.001` so the curve covers the low-confidence predictions too. `inference_ms` is the network time per image on the machine that evaluated the run, without pre- and post-processing. Compare it only between runs on the same GPU.

The three training losses, logged every epoch under `train/` and `val/`:

| Loss | What it measures |
| --- | --- |
| `box_loss` | how far the predicted box is from the true box |
| `cls_loss` | binary cross-entropy of the class score (here, pothole vs. background) |
| `dfl_loss` | Distribution Focal Loss: how far the predicted box edges are from the true edges |

With 124 test images, a difference of 0.01 to 0.02 between runs is small. Read `test_mAP50-95` together with `epochs_run`: a run that early-stops has already flattened.

## Results

Seed 42. The table below is a **1-epoch check** on an Apple M1 Pro (CPU), image size 320, batch 8, and 5% of the training images (50 of 995). It shows that training, evaluation, the comparison table, resume, and `load_model` run. It is not the comparison the configs are built for: those train 30, 30, and 50 epochs at image size 640 on all 995 training images. One epoch is not enough for the frozen run and the fine-tuned run to separate.

| Config | Strategy | Epochs run | Test precision | Test recall | Test mAP50 | Test mAP75 | Test mAP50-95 | Gain vs from scratch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| config01 | frozen backbone | 1 | 0.0068 | 0.5294 | 0.0239 | 0.0057 | 0.0084 | +0.0084 |
| config02 | fine-tuned | 1 | 0.0068 | 0.5294 | 0.0239 | 0.0057 | 0.0084 | +0.0084 |
| config03 | from scratch | 1 | 0.0006 | 0.0206 | 0.0000 | 0.0000 | 0.0000 | 0 |

At one epoch the two pretrained runs match, and both sit above training from scratch. The 1-class YOLOv8n has 3,011,043 parameters and about 8.2 GFLOPs at image size 640. With the backbone frozen, 1,738,371 parameters train; with nothing frozen, 3,011,027 train (one small fixed layer in the head never trains). Read your own `runs_summary.csv` after the full configs: that is where frozen versus fine-tuning shows up.

## What is logged

Everything goes to the W&B project `LitWBTransferLearningYolo`, one run per config run, grouped by config.

**Per epoch** ([`callbacks/wandb_logging.py`](callbacks/wandb_logging.py), registered as Ultralytics' `on_fit_epoch_end` callback), against `epoch`:

| Key | Meaning |
| --- | --- |
| `train/box_loss`, `train/cls_loss`, `train/dfl_loss` | training losses, averaged over the epoch |
| `val/box_loss`, `val/cls_loss`, `val/dfl_loss` | the same losses on the validation set |
| `val/precision`, `val/recall`, `val/mAP50`, `val/mAP50-95` | validation metrics after the epoch |
| `lr/pg0`, `lr/pg1`, `lr/pg2` | learning rate of each parameter group |

The first epoch also stores `trainable_params` in the run summary: weights with `requires_grad`, so the frozen run is smaller than the other two while `params` (every weight) stays the size of the 1-class YOLOv8n.

**After training**, the best weights are evaluated on val and test:

| In W&B | What it shows |
| --- | --- |
| run summary `val/...` and `test/...` | `images`, `instances`, `precision`, `recall`, `f1`, `mAP50`, `mAP75`, `mAP50-95`, `inference_ms`; also `strategy`, `params`, `GFLOPs`, `epochs_run`, `best_epoch` |
| `results` | one row per split with those numbers |
| `comparison` | the newest run of every config, with `gain_vs_from_scratch` |
| `<split>/pr_curve`, `f1_curve`, `precision_curve`, `recall_curve`, `confusion_matrix` | Ultralytics' plots for that split |
| `<split>_predictions` | every image, with `ground_truth` and `predictions` boxes, and the counts `potholes`, `predicted`, `found`, `false_alarms`, `missed` (confidence ≥ 0.25, IoU ≥ 0.5) |
| `training/curves`, `training/labels` | Ultralytics' training-curve figure, and where the training boxes sit |

### Reading the report

- **Comparison table.** A positive `gain_vs_from_scratch` means that starting from COCO beat random weights on test mAP50-95. Then compare config01 and config02 with each other: that is the frozen-backbone versus fine-tuning question.
- **Learning curves.** `val/mAP50-95` should rise and flatten. If `val/box_loss` rises while `train/box_loss` keeps falling, the model is memorizing the 995 training images. Patience stops a run that has not improved for that many epochs.
- **Predictions table.** Sort by `missed` or `false_alarms`. Small potholes, shadows, and patched asphalt are the usual hard cases.

## Run folder

Each run writes `<OUTPUT_DIR>/LitWBTransferLearningYolo/<config>/<timestamp>/`. Locally that is `runs/LitWBTransferLearningYolo/...`. In Colab it is `MyDrive/CodeCS4337Fall2026/runs/LitWBTransferLearningYolo/...`.

```
├── config.json        # Copy of the config, written when the run starts
├── hparams.json       # Strategy, data, model, training, and resumed_from
├── args.yaml          # Ultralytics' full training arguments
├── results.csv        # Losses and validation metrics, one row per epoch
├── results.png, labels.jpg, train_batch*.jpg, ...   # Ultralytics' plots
├── weights/
│   ├── best.pt        # Epoch with the best validation mAP50-95
│   └── last.pt        # Rewritten every epoch; resume reads this file
├── eval_val/          # Validation curves and confusion matrix
├── eval_test/         # The same plots for the test set
└── wandb/             # Local W&B files. Offline runs are uploaded from here
```

`runs_summary.csv`, next to the run folders, has one row per finished run: strategy, weights, freeze, learning rate, optimizer, epochs, epochs actually run, best epoch, training minutes, image size, parameter counts, GFLOPs, validation mAP50 and mAP50-95, the test metrics, seed, and `resumed_from`.

To load a trained model:

```python
from LitWBTransferLearningYolo import load_model

model = load_model("config02")          # newest run of config02, weights/best.pt
results = model.predict("road.jpg", conf=0.25)
print(results[0].boxes.xyxy, results[0].boxes.conf)
```

To keep training: `main("config02.json", resume_from="config02")`, or `--resume-from config02`.

| The run | What `resume_from` does |
| --- | --- |
| **interrupted** (Colab disconnect, crash) | Ultralytics finishes it from `weights/last.pt` (weights, optimizer, epoch) in the **same** folder, up to that run's original number of epochs |
| **finished** (all epochs done, or early stopping) | trains the config's `"epochs"` **more** epochs from `weights/best.pt`, in a **new** run folder, with the config's settings |

A finished run has no optimizer state left in the weights file, so continuing it starts a new stage. The learning-rate schedule starts again from `lr0`, and the 3 warm-up epochs are turned off so they do not disturb trained weights. The new row's `resumed_from` is the earlier run's folder name.

## Things to try

- Copy `config02.json` to `config04.json` and set `"weights": "yolov8s.pt"`. The small model has about 11 million parameters. Does the extra capacity beat YOLOv8n on test mAP50-95 by enough to be worth the slower inference?
- In a copy of config01, set `lr0` to `0.001`. The frozen run's head is new and random; a smaller rate learns it more slowly. Which one reaches a higher validation mAP50-95 in 30 epochs?
- Train config03 for 100 epochs (`"epochs": 100` in a copy). Does from-scratch catch the pretrained runs, or does it flatten below them?
- Add `"mosaic": 0.0` under `ultralytics_args`. Mosaic pastes four images into one. Potholes are small; turning mosaic off changes how often a pothole fills the frame.
- Set `"imgsz": 320`. Training is faster. Check test mAP75, not only mAP50: a coarser image makes tight boxes harder.
- Swap in another YOLO pothole dataset by pointing `prepare_pothole` at a folder of images with sibling `.txt` labels, and rerun the three configs on that split.
