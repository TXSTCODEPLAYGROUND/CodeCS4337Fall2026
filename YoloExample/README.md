# YoloExample

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/YoloExample/yolo_example_notebook.ipynb)
This project's notebook: [`yolo_example_notebook.ipynb`](yolo_example_notebook.ipynb).

**Object detection** with [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
on the [Penn-Fudan pedestrian dataset](https://www.cis.upenn.edu/~jshi/ped_html/):
find every person in a street photo and draw a box around them. Three configs
compare a COCO-pretrained YOLOv8n used as it is (the **zero-shot baseline**),
the same model fine-tuned on Penn-Fudan, and the same architecture trained
from scratch. Every run is tracked in W&B: training curves, precision, recall,
F1, mAP50, mAP75, and **mAP50-95** (mAP@0.5:0.05:0.95) on the validation and
test sets, AP at each IoU threshold, PR curves, a table of every image with its
true and predicted boxes, a results table, and a comparison of all configs with
the zero-shot baseline.

The [notebook](yolo_example_notebook.ipynb) opens up YOLO step by step: the
YOLO label format and the conversion of Penn-Fudan's masks to it, the YOLOv8
structure layer by layer, what goes into the network, what each of the 84
numbers it predicts per location means, how a box is decoded from them by
hand, and how non-maximum suppression produces the final boxes.

Unlike the Lightning projects, training is done by the
[Ultralytics](https://docs.ultralytics.com/) library, which has its own
trainer; this project adds the W&B report around it. The W&B setup is the same
as in [LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/): read how to
[set up your W&B key](../LitWBTrainingBasicConvnet/README.md#set-up-your-wb-key)
and about [offline mode](../LitWBTrainingBasicConvnet/README.md#no-key-or-failed-login-offline-mode).

How to set up and run the project and where results go is explained in the
[main README](../README.md). Function details are in the
[API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/YoloExample.html).

## Running

From the repo root, one config at a time, **config01 first** (the others are
compared with it):

```bash
python -m YoloExample --config config01.json   # zero-shot baseline, no training
python -m YoloExample --config config02.json   # fine-tuned from COCO
python -m YoloExample --config config03.json   # from scratch
```

The dataset (51 MB) is downloaded to `data/PennFudanPed/` and converted to
YOLO format in `data/pennfudan_yolo/` on the first run; the COCO weights
(6 MB) go to `data/yolo_weights/`, with the small `yolo26n.pt` Ultralytics
uses to check mixed precision before training; later runs read them from
there. On Colab's T4, config02 takes a few minutes and
config03 around 15; on a recent desktop GPU, under a minute and about two.

## The dataset and the YOLO label format

Penn-Fudan has 170 photos of streets around two universities, with 423 labeled
pedestrians. The images are split once, with a fixed seed, into:

| Split | Images | People | Used for |
| --- | --- | --- | --- |
| train | 120 | 291 | training (augmented by Ultralytics: mosaic, flips, scaling, color) |
| val | 25 | 67 | choosing the best epoch, early stopping, validation report |
| test | 25 | 65 | final test report |

Ultralytics reads **one text file per image**, `labels/<split>/<name>.txt`
next to `images/<split>/<name>.png`, with one line per object:

```text
class x_center y_center width height
0 0.412343 0.570896 0.255814 0.466418
```

`class` is the index in the dataset's `names` (here `0` = `person`); the box
is its **center and size, divided by the image width and height**, so every
number is between 0 and 1 and stays valid when the image is resized. Pascal
VOC stores pixel corners (`x_min, y_min, x_max, y_max`) and COCO the top-left
corner and size in pixels (`x_min, y_min, width, height`); converting means
getting this right:

```text
x_center = (x_min + x_max) / 2 / W        width  = (x_max - x_min) / W
y_center = (y_min + y_max) / 2 / H        height = (y_max - y_min) / H
```

Penn-Fudan comes with **instance masks** (`PedMasks/<name>_mask.png`: 0 for
background, k for the pixels of the k-th person), not with YOLO labels.
[`dataloaders/pennfudan.py`](dataloaders/pennfudan.py) converts it: for each
person, the smallest box around its mask pixels (`mask_to_boxes`), converted
to YOLO numbers (`box_to_yolo`), one line per person. It also writes
`pennfudan.yaml`, which tells Ultralytics where the splits are and the class
names. The notebook draws converted labels back on their images: always check
a conversion this way, since wrong labels still train, just badly.

## YOLOv8 in short

YOLOv8n (3.2 million parameters, about 9 GFLOPs per 640 x 640 image) is a
one-stage detector: one pass through one network predicts all boxes.

- **Backbone** (layers 0 to 9: Conv, C2f, SPPF blocks) turns the
  `3 x 640 x 640` image into feature maps at three scales: 80 x 80 (stride 8),
  40 x 40 (stride 16), 20 x 20 (stride 32).
- **Neck** (layers 10 to 21, PAN-FPN) mixes the three scales with upsampling,
  concatenation, and strided convolutions.
- **Head** (layer 22, Detect) predicts, for every cell of the three maps, a box
  and the class scores, in separate branches.

The output is a tensor of shape `(batch, 84, 8400)`: **8,400 candidate boxes**
(80 x 80 + 40 x 40 + 20 x 20 cells), each described by **84 numbers**:

| Index | Meaning |
| --- | --- |
| 0, 1 | `cx`, `cy`: box center, in pixels of the 640 x 640 input |
| 2, 3 | `w`, `h`: box width and height, in pixels |
| 4 to 83 | the 80 COCO class scores, each between 0 and 1 (independent sigmoids); index 4 is `person` |

There is no objectness score (YOLOv5 had 85 numbers): a box's confidence is
its highest class score. A model trained on Penn-Fudan has one class, so 5
numbers per box. Internally, the head predicts each box as 4 distributions
over 16 distances, from the cell center to the box's left, top, right, and
bottom sides (Distribution Focal Loss); `cx, cy, w, h` are computed from their
expected values. Post-processing drops candidates under a confidence threshold
and removes overlapping duplicates with **non-maximum suppression** (NMS). The
notebook shows each step on a real image.

## Three configs and the zero-shot baseline

| Config | Strategy | `model.weights` | Training |
| --- | --- | --- | --- |
| `config01.json` | **zero-shot** (baseline) | `yolov8n.pt` (COCO) | `"train": false`: only evaluated |
| `config02.json` | **fine-tuned** | `yolov8n.pt` (COCO) | 50 epochs, SGD, `lr0` 0.001, first 10 layers (the backbone) frozen |
| `config03.json` | **from scratch** | `yolov8n.yaml` (random weights) | 150 epochs, `optimizer` auto, patience 30 |

**Zero-shot** means evaluating a model on a task without training it on that
task's data. COCO, the dataset YOLOv8n was trained on, has 80 classes, one of
which is `person`, so the COCO model can already find pedestrians.
Penn-Fudan's labels only contain people, so the evaluation passes
`classes=[0]`: only the model's `person` boxes are compared with the labels,
and its cars or handbags are ignored. The zero-shot score is the bar the
trained models must clear: training is only worth it if it beats the
baseline.

**Fine-tuning** starts from the same COCO weights; Ultralytics replaces the
80-class score layer with a new 1-class layer and keeps everything else. With
only 120 training images, how gently it trains matters a lot (test mAP50-95,
50 epochs each):

| Fine-tuning settings | Test mAP50-95 |
| --- | --- |
| `optimizer` auto (AdamW, lr 0.002), nothing frozen | 0.766 |
| `"freeze": 10`, `optimizer` auto | 0.775 |
| SGD, `lr0` 0.001, nothing frozen | 0.832 |
| **SGD, `lr0` 0.001, `"freeze": 10`** (config02) | **0.854** |
| *zero-shot baseline* | *0.826* |

Aggressive fine-tuning ends **below the baseline**: it overwrites what the
model learned from COCO's 118,000 images with what 120 images can teach.
Freezing the backbone and using a small learning rate keeps the general
"what a person looks like" features and adapts the neck and head to
Penn-Fudan's labeling style.

**From scratch** uses the same architecture with random weights: it shows what
the COCO weights are worth.

Extra Ultralytics training arguments can be added to a config's
`"training"` section under `"ultralytics_args"`, e.g. `{"freeze": 10}` or
`{"mosaic": 0.0}` (see the
[list of training arguments](https://docs.ultralytics.com/modes/train/#train-settings)).

## Metrics

All detection metrics are built from the same steps:

1. **IoU (intersection over union)** of a predicted box and a true box: the
   area of their overlap divided by the area they cover together. 1 is a
   perfect fit, 0 no overlap at all.
2. **Matching at an IoU threshold.** Predictions are taken from the most to the
   least confident. A prediction is a **true positive (TP)** if its IoU with a
   true box not yet matched is at least the threshold, otherwise a **false
   positive (FP)**: a false alarm, or a box too loose to count. True boxes left
   unmatched are **false negatives (FN)**: missed people.
3. **Precision** = TP / (TP + FP): the share of detections that are right.
   **Recall** = TP / (TP + FN): the share of people that are found.
   **F1** = 2 x precision x recall / (precision + recall): one number that is
   high only when both are.
4. **Precision-recall (PR) curve.** Lowering the confidence threshold accepts
   more boxes: recall goes up, precision usually goes down. Each threshold is
   one point of the curve.
5. **AP (average precision)**: the area under the PR curve, between 0 and 1
   (Ultralytics, like COCO, makes precision non-increasing and samples it at
   101 recall values). It summarizes the detector over **all** confidence
   thresholds, so it does not depend on picking one.
6. **mAP**: the mean of AP over the classes. With one class, mAP is the AP of
   `person`.

The IoU threshold of step 2 gives the different mAPs:

| Metric | Definition | What it rewards |
| --- | --- | --- |
| **mAP50** | AP at IoU ≥ 0.50 | finding people; a box half off still counts (the Pascal VOC metric) |
| **mAP75** | AP at IoU ≥ 0.75 | finding people with tight boxes |
| **mAP50-95**, or **mAP@0.5:0.05:0.95** | the mean of AP at IoU 0.50, 0.55, 0.60, ..., 0.95 (10 thresholds) | both, with partial credit for every threshold a box passes; the main COCO metric |

A box with IoU 0.80 counts as correct at 7 of the 10 thresholds: full credit
for mAP50, 70 % for its share of mAP50-95. mAP50-95 is the metric Ultralytics
uses to pick the best epoch (`weights/best.pt`) and the one used to compare
configs here. The W&B plot `ap_per_iou_threshold` shows AP at each of the 10
thresholds: mAP50-95 is its average, and how fast it drops shows how precise
the boxes are. In the results, the model trained from scratch has a decent
mAP50 (0.87) but a poor mAP75 (0.60): it finds people but does not fit them
tightly.

Details to keep in mind when reading the numbers:

- mAP is computed with a very low confidence threshold (`"conf": 0.001` in the
  config's `"evaluation"` section) so the PR curve covers all recall levels.
  The **precision, recall, and F1** reported are those at the single
  confidence threshold that maximizes F1 (Ultralytics' convention), read from
  the curves.
- `inference_ms` is the network's time per image on the machine that ran the
  evaluation, without pre- and post-processing. Compare it only between runs
  on the same GPU; the first evaluation of a run also includes warm-up.
- With 25 test images (65 people), differences of 0.01 to 0.02 are within
  run-to-run noise. Penn-Fudan also leaves some people unlabeled (mostly
  hidden or far in the background): finding them counts as a false positive.

The three **training losses**, logged every epoch for `train/` and `val/`:

| Loss | What it measures |
| --- | --- |
| `box_loss` | CIoU loss: 1 − IoU between predicted and true boxes, plus penalties for center distance and aspect ratio |
| `cls_loss` | binary cross-entropy of the class scores |
| `dfl_loss` | Distribution Focal Loss: how far each side's 16-bin distance distribution is from the true distance |

## Results

One run of each config (seed 42); yours will differ a little:

| Config | Epochs run | Test precision | Test recall | Test mAP50 | Test mAP75 | Test mAP50-95 | Gain vs zero-shot |
| --- | --- | --- | --- | --- | --- | --- | --- |
| config01, zero-shot | 0 | 0.872 | 0.908 | 0.950 | 0.924 | 0.826 | 0 |
| config02, fine-tuned | 50 | 0.920 | 0.885 | 0.957 | 0.926 | **0.854** | +0.027 |
| config03, from scratch | 142 (early stopping) | 0.882 | 0.815 | 0.866 | 0.601 | 0.521 | −0.306 |

## What is logged

Everything goes to the W&B project `YoloExample`, one run per config run,
grouped by config.

**Per epoch** (config02 and config03), against `epoch`
([`callbacks/wandb_logging.py`](callbacks/wandb_logging.py), registered as
Ultralytics' `on_fit_epoch_end` callback):

| Key | Meaning |
| --- | --- |
| `train/box_loss`, `train/cls_loss`, `train/dfl_loss` | training losses (average over the epoch) |
| `val/box_loss`, `val/cls_loss`, `val/dfl_loss` | the same on the validation set |
| `val/precision`, `val/recall`, `val/mAP50`, `val/mAP50-95` | validation metrics after the epoch |
| `lr/pg0`, `lr/pg1`, `lr/pg2` | learning rate of each parameter group (with warm-up over the first 3 epochs) |

**After training**, the best weights (highest validation mAP50-95) are
evaluated on the validation and test sets:

| In W&B | What it shows |
| --- | --- |
| run summary `val/...` and `test/...` | `images`, `instances`, `precision`, `recall`, `f1`, `mAP50`, `mAP75`, `mAP50-95`, `inference_ms`; also `strategy`, `params`, `GFLOPs`, `epochs_run`, `best_epoch` |
| `results` | table: one row per split with all the numbers above |
| `comparison` | table: the newest run of every config, with `gain_vs_zero_shot` (test mAP50-95 minus the zero-shot run's) |
| `ap_per_iou_threshold`, `ap_per_iou_table` | AP at each IoU threshold, 0.50 to 0.95, for both splits |
| `<split>/pr_curve`, `f1_curve`, `precision_curve`, `recall_curve` | Ultralytics' curves against the confidence threshold |
| `<split>/confusion_matrix` | person vs. background: missed people and false alarms |
| `<split>_predictions` | table of every image, with `ground_truth` and `predictions` boxes (toggle them in the image panel) and the counts `persons`, `predicted`, `found`, `false_alarms`, `missed` (confidence ≥ 0.25, IoU ≥ 0.5) |
| `<split>_hardest_examples` | the 8 images with the most false alarms plus missed people |
| `training/curves`, `training/labels` | Ultralytics' summary of the training curves, and the distribution of box positions and sizes in the training labels |

### Reading the report

- **Comparison table:** the first thing to look at. Is `gain_vs_zero_shot`
  positive? Then training was worth it.
- **Learning curves:** `val/mAP50-95` should rise and flatten; if
  `val/box_loss` rises while `train/box_loss` keeps falling, the model is
  overfitting the 120 training images. Early stopping (`patience`) stops a run
  whose mAP50-95 has not improved for that many epochs.
- **Predictions table:** sort by `missed` or `false_alarms`, then open the
  images: small, hidden, or cut-off people are the usual misses; unlabeled
  people in the background the usual "false alarms".
- **AP per IoU threshold:** a curve that stays high up to 0.8 or 0.85 means
  tight boxes; one that falls after 0.6 means loose ones.

## Run folder

Each run writes `runs/YoloExample/<config>/<timestamp>/`:

```
├── config.json        # Exact copy of the config used
├── args.yaml          # All Ultralytics training arguments (trained configs)
├── results.csv        # Losses and validation metrics, every epoch
├── results.png, labels.jpg, train_batch*.jpg, ...   # Ultralytics' training plots
├── weights/
│   ├── best.pt        # Epoch with the best validation mAP50-95
│   └── last.pt        # Last epoch, rewritten every epoch (used to resume)
├── eval_val/          # Validation plots: PR, F1, P, R curves, confusion matrices, sample predictions
├── eval_test/         # The same for the test set
└── wandb/             # W&B's local copy of the run (offline runs are synced from here)
```

`runs_summary.csv` (one row per run, next to the run folders) records the
strategy, epochs, best epoch, training time, parameters, GFLOPs, validation
mAP50 and mAP50-95, and all the test metrics.

To load a trained model back:

```python
from YoloExample import load_model

model = load_model("config02")             # newest run of config02, weights/best.pt
results = model.predict("street.jpg", conf=0.25)
print(results[0].boxes.xyxy, results[0].boxes.conf)
```

`load_model("config01")` returns the COCO model of the zero-shot config; pass
`classes=[0]` to `predict` or `val` to keep only its people.

To finish an interrupted training run (a Colab disconnect, a crash):
`main("config02.json", resume_from="config02")`, or `--resume-from config02`
on the command line. Ultralytics continues from `weights/last.pt`, in the same
folder, up to the run's original number of epochs. A run that already finished
cannot be resumed; start a new run instead.

## Ultralytics settings and license

Ultralytics keeps its own settings in `settings.json` (`yolo settings` prints
it and its location). Three of them matter here:

- `wandb` (off by default) turns on Ultralytics' **own** W&B logging. This
  project logs to W&B itself, so leave it off; turned on, every training would
  also create a second, differently organized W&B run.
- `sync` (on by default) sends anonymous usage analytics and crash reports to
  Ultralytics. Turn it off with `yolo settings sync=False`.
- `weights_dir` is where Ultralytics downloads weights, including
  `yolo26n.pt` for its one-time mixed-precision check before training. When
  the settings are first created inside this repo, it is `<repo>/weights/`.
  This project replaces it, for its own runs only, with `data/yolo_weights/`
  (`yolo_weights_dir` in [`models/loading.py`](models/loading.py)), so all
  YOLO weights live with the data; `settings.json` is not changed.

Ultralytics is licensed under [AGPL-3.0](https://www.ultralytics.com/license):
fine for coursework and open-source projects; a closed-source product that
uses it needs an Ultralytics enterprise license.

## Things to try

- Run config02 with `"optimizer": "auto"` and `"ultralytics_args": {}` (in a
  copy, e.g. `config04.json`): does it beat the zero-shot baseline?
- Fine-tune `yolov8s.pt` (11 million parameters) instead of `yolov8n.pt`. Is
  the gain worth 3 times the inference time?
- Compare the zero-shot model with and without `classes=[0]` (remove it in
  `main.py`): how much do its other classes cost?
- Fine-tune with fewer training images, e.g. `"ultralytics_args": {"freeze":
  10, "fraction": 0.25}` (30 of the 120): at what point does fine-tuning stop
  beating the baseline? `fraction` keeps the validation and test images the
  same; changing `"split_sizes"` would not.
- Look at `test_hardest_examples` of the zero-shot and fine-tuned models:
  which mistakes did fine-tuning fix?
