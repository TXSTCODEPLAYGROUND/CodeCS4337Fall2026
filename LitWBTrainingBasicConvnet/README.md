# LitWBTrainingBasicConvnet

The same experiment as [LitTrainingBasicConvnet](../LitTrainingBasicConvnet/)
(the same ConvNet, LightningModule, DataModule, metrics, and settings), but
tracked with [Weights & Biases](https://wandb.ai) (W&B) instead of LitLogger
and hand-made plots. Read that project first: this README only explains what
is different.

What changes:

1. **Every metric goes to W&B** through Lightning's
   [`WandbLogger`](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html):
   loss, accuracy, precision, recall, and per-class accuracy, for training,
   validation, and test, plus the config of the run.
2. **No plotting code.** `utils/plots.py` is gone. The learning curves are
   W&B charts, and the example predictions are an interactive table.
3. **A small callback**, [`callbacks/wandb_predictions.py`](callbacks/wandb_predictions.py),
   logs a table of test images with their predicted and true class, and a
   confusion matrix.
4. **`metrics.csv` is still saved locally** (Lightning's `CSVLogger`), as a
   backup that works without an internet connection or an account.

How to set up and run the project and where results go is explained in the
[main README](../README.md). Function and class details are in the
[API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitWBTrainingBasicConvnet.html).

## What is W&B?

[Weights & Biases](https://docs.wandb.ai) is a widely used experiment
tracker. Each training run sends its metrics to wandb.ai while it runs, so you
can watch the curves live, compare many runs in one chart, sort runs by any
metric or config value, and share a link to your results. It is free for
students and personal projects.

With Lightning, using W&B is one extra logger: every `self.log(...)` in
[`models/lit_convnet.py`](models/lit_convnet.py) is sent to W&B without any
change to the model.

```python
wandb_logger = WandbLogger(project="LitWBTrainingBasicConvnet", config=config)
trainer = L.Trainer(logger=[csv_logger, wandb_logger], ...)
```

## Set up your W&B key

1. Create a free account at [wandb.ai](https://wandb.ai/site).
2. Copy your API key from [wandb.ai/authorize](https://wandb.ai/authorize)
   and add it to `.env` at the repo root:

   ```
   WANDB_API_KEY=your-api-key
   ```

   `.env` is not in git, so your key stays private. Never put it in
   `.envcolab` (which *is* in git), a config, or a notebook you share. If a key
   is ever pushed by mistake, delete it and create a new one on the same page.

   **In Colab**, the setup cell overwrites `.env`, so store the key as a Colab
   *Secret* named `WANDB_API_KEY` instead (key icon in the left sidebar). Then
   run the **Load API keys** cell of
   [`starternotebook.ipynb`](../starternotebook.ipynb), after the setup cells
   and before running the project. It copies the key from your Secrets to the
   notebook's environment and to `.env`, so every way of running works: from
   the notebook (Option A1 or B) and from Colab's terminal (A2).

The run prints a link to its page on wandb.ai. The W&B project is set by
`"wandb": {"project": ...}` in the config; it is created in your account the
first time you run.

### No key or failed login: offline mode

A run never stops because of W&B. If `WANDB_API_KEY` is missing (or commented
out), or W&B cannot log in (a wrong key, no internet), the run prints a warning
and W&B logs **locally** instead: every metric, the config, the prediction
table, and the confusion matrix are saved in the run folder under
`wandb/offline-run-*`. `metrics.csv` and the checkpoints are written as usual.

To upload such a run to W&B later:

1. Put a valid key in `.env` (see [Set up your W&B key](#set-up-your-wb-key)).
2. From the repo root, with `.venv` activated, load `.env` into the terminal
   (`wandb sync` does not read `.env` by itself), then run the `wandb sync`
   command printed in the warning:

   ```bash
   set -a; source .env; set +a
   wandb sync runs/LitWBTrainingBasicConvnet/config01/<timestamp>/wandb/offline-run-*
   ```

   To upload every offline run of a config at once, use `*` for the timestamp:
   `runs/LitWBTrainingBasicConvnet/config01/*/wandb/offline-run-*`.

   In Colab, run the starter notebook's **Load API keys** cell first, then the
   same two commands in Colab's terminal, or with `!` in a cell
   (`!wandb sync ...`; the key cell already set the key for `!` commands).
3. Open your project on wandb.ai: the run appears with all its charts, as if
   it had been online.

Run `wandb sync` on the same computer (or the same Colab session) as the
training: W&B keeps the images of the prediction table in its own cache there
until they are uploaded. After a failed login, the run's `wandb/` folder also
holds an empty `run-*` folder from the attempt; only `offline-run-*` needs
syncing.

## What is logged

| In W&B | Comes from | Replaces in LitTrainingBasicConvnet |
| --- | --- | --- |
| `train_loss`, `val_loss`, `test_loss` | `self.log` in `LitConvNet` | `loss.png` |
| `train_acc`, `val_acc`, `test_acc` | `self.log` (torchmetrics) | `accuracy.png` |
| `*_precision`, `*_recall` | `self.log` (torchmetrics) | (`metrics.csv` only) |
| `train_acc_<class>`, `val_acc_<class>` | `self.log` (torchmetrics) | `accuracy_per_class.png` |
| `test_predictions` table | `LogTestPredictions` callback | `predictions.png`, `wrong_predictions.png` |
| `confusion_matrix` | `LogTestPredictions` callback | (new) |
| Config (seed, lr, batch size, ...) | `WandbLogger(config=config)` | `config.json` |

The `test_predictions` table holds the first 1000 test images with columns
`image`, `predicted`, `true`, and `correct`. The confusion matrix covers the
whole test set: row = true class, column = predicted class, so everything off
the diagonal is a mistake, and you can see which classes get mixed up.

## Build the dashboard

W&B makes one chart per metric automatically. To compare training and
validation on the same graph, like the plots of LitTrainingBasicConvnet, add
a few panels once; the workspace keeps them for every later run of the
project.

1. Open your project on wandb.ai and go to **Workspace**.
2. **Loss**: click **Add panels** → **Line plot**. Set **X** to `epoch` and
   **Y** to `train_loss` and `val_loss`. Click **Apply**.
3. **Accuracy**: the same, with **Y** = `train_acc` and `val_acc`.
4. **Accuracy per class**: the same, but in **Y** switch on regex
   (the `.*` button) and enter `train_acc_.*|val_acc_.*`. This draws all 20
   per-class curves on one graph. Use `val_acc_.*` alone if it is too crowded.
5. **Wrong predictions**: open the `test_predictions` table panel, click
   **Filter**, and enter `row["correct"] = false`. Sort or group by `true` to
   see which classes the model gets wrong.
6. The `confusion_matrix` panel needs no setup.

Tip: Lightning logs `epoch` and `trainer/global_step` with every value; use
`epoch` as the X axis so one point is one epoch, as in `metrics.csv`.

To compare runs, train with a second config (e.g. `config02.json` with a
different `lr`): every chart then shows one line per run, and the **Runs**
table can sort runs by `val_acc` or any config value. Runs from the same
config file are grouped by name (`config01`, ...).

## Run folder

Each run writes these files to
`runs/LitWBTrainingBasicConvnet/<config>/<timestamp>/`:

```
├── config.json                     # Exact copy of the config used
├── hparams.json                    # Optimizer and data settings
├── metrics.csv                     # Every metric, every epoch (local backup)
├── best_epoch08_valacc0.9240.ckpt  # Checkpoint with the best validation accuracy
├── last.ckpt                       # Checkpoint after the last epoch
└── wandb/                          # W&B's local copy of the run (offline runs are synced from here)
```

Reloading a checkpoint works as in LitTrainingBasicConvnet (pass `net=` again).

## Things to try

- Compare [`main.py`](main.py) with
  [`LitTrainingBasicConvnet/main.py`](../LitTrainingBasicConvnet/main.py) and
  `utils/plots.py` there: count how much code W&B replaces.
- Add a `config02.json` with more dropout or weight decay, run both, and
  compare the `val_loss` curves in one chart.
- In the confusion matrix, find the class most often confused with *Shirt*.
- Log something new: `self.log("lr", ...)` in `LitConvNet` appears in W&B
  without any other change.
