# LitWBHSTrainingBasicNeuralNetwork

[LitWBTrainingBasicNeuralNetwork](../LitWBTrainingBasicNeuralNetwork/) plus
a **hyperparameter search** with [Optuna](https://optuna.org). Instead of
guessing the learning rate, the network size, or the optimizer, Optuna trains
many settings, learns from each result which settings look promising, and
stops bad trials early. The best settings are saved as a normal config, which
is then trained and tested like any other.

Read [LitWBTrainingBasicNeuralNetwork](../LitWBTrainingBasicNeuralNetwork/README.md)
first: the network, the Lightning code, and the W&B logging are the same.
How to set up and run a project is in the [main README](../README.md).
Function and class details are in the
[API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitWBHSTrainingBasicNeuralNetwork.html).

The easiest way to run everything is the project's own notebook,
[`hyperparameter_search.ipynb`](hyperparameter_search.ipynb)
([open in Colab](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBHSTrainingBasicNeuralNetwork/hyperparameter_search.ipynb)).
It has the same setup cells as the starter notebook, then runs the search,
plots the results, and trains the best config.

## The workflow

1. **Search** with a search config. Each trial trains one setting and is
   scored by its best validation accuracy:

   ```bash
   python -m LitWBHSTrainingBasicNeuralNetwork.search --config search01.json
   ```

2. **Read the results**: the printed best trial, `trials.csv`, and Optuna's
   plots (in the notebook).
3. **Train the best config**, which the search saved as
   `configs/search01_best.json`. This run tests the model once on the test
   set, the score to report:

   ```bash
   python -m LitWBHSTrainingBasicNeuralNetwork --config search01_best.json
   ```

The test set is never used during the search. Picking settings by their test
score would tune them to the test set, and the test score would no longer
show how well the model does on new data.

From Python or a notebook:

```python
from LitWBHSTrainingBasicNeuralNetwork import main
from LitWBHSTrainingBasicNeuralNetwork.search import search

study = search("search01.json", n_trials=30)
main("search01_best.json")
```

## What changes from LitWBTrainingBasicNeuralNetwork

| | LitWBTrainingBasicNeuralNetwork | This project |
| --- | --- | --- |
| Activation | ReLU | `"activation"`: ReLU, Leaky ReLU, GELU, or tanh ([`models/components/mlp.py`](models/components/mlp.py)) |
| Optimizer | Adam | `"optimizer"`: Adam, AdamW, SGD, or RMSprop, with `"momentum"` for SGD and RMSprop ([`models/lit_mlp.py`](models/lit_mlp.py)) |
| Regularization | `"weight_decay"` (L2) | `"l1"` (penalty added to the loss) and `"l2"` (the optimizer's weight decay) |
| Learning rate | Constant | `"scheduler"`: none, step, cosine, or plateau, with `"step_size"` and `"gamma"`; the learning rate is logged every epoch |
| Early stopping | No | `"early_stopping"` and `"patience"` |
| Data-loading workers | `"num_workers": 2` | `"num_workers": "auto"`: the CPU cores available, at most 8 (8 on a big machine, 2 in Colab), so the same config is fast everywhere |
| Training data | All 54,000 training images | `"train_fraction"`: train on part of them, e.g. for faster search trials |
| Setup code | In `main()` | In `build()` in [`main.py`](main.py), shared by `main()` and every search trial |
| W&B connection | In `main.py` | [`utils/tracking.py`](utils/tracking.py), shared too |
| New files | | [`search.py`](search.py), [`callbacks/optuna_pruning.py`](callbacks/optuna_pruning.py), [`configs/search01.json`](configs/search01.json), the notebook |

[`configs/config01.json`](configs/config01.json) has the same settings as
LitWBTrainingBasicNeuralNetwork's config (Adam, learning rate 0.001,
`[256, 128]`, ReLU, no regularization, no scheduler), now written out in the
new `"training"` keys. It is the starting point of the search and the
baseline to beat.

## The search config

[`configs/search01.json`](configs/search01.json) has three parts:

- `"base_config"`: the training config every trial starts from. Everything not
  searched (seed, validation split, W&B project) comes from it.
- `"study"`: `n_trials`; `timeout_minutes` (stop starting trials after this
  time, `null` for no limit); the sampler's `seed`; `train_fraction` (the
  fraction of the training data each trial uses, see
  [below](#how-long-a-search-takes)); the pruner settings (below); and
  `log_trials_to_wandb` (`false`, see
  [below](#comparing-the-trials-in-wb)).
- `"search_space"`: what each trial may pick.

| Key | Searched as | Notes |
| --- | --- | --- |
| `n_layers` | Integer range | Number of hidden layers |
| `units` | Choices | Units of each hidden layer, picked per layer (`units_layer1`, ...) |
| `activation` | Choices | |
| `dropout` | Range | |
| `batch_size` | Choices | |
| `epochs` | Integer range | Maximum epochs; early stopping may end sooner |
| `optimizer` | Choices | |
| `lr` | Range, log scale | Each power of ten is equally likely |
| `momentum` | Range | Only for SGD and RMSprop |
| `regularization` | Choices | `"none"`, `"l1"`, `"l2"`, or `"l1_l2"` |
| `l1`, `l2` | Range, log scale | Only when `regularization` uses them |
| `scheduler` | Choices | `"none"`, `"step"`, `"cosine"`, or `"plateau"` |
| `step_size` | Integer range | Only for `"step"` |
| `gamma` | Range | Only for `"step"` and `"plateau"` |
| `early_stopping` | Choices | `[true, false]` |
| `patience` | Integer range | Only with early stopping |

The "only for" rows are a key Optuna feature: the search space is defined by
running Python code, `suggest_config()` in [`search.py`](search.py), so an
`if` decides what each trial samples. A trial using Adam has no momentum at
all, rather than a momentum value that is silently ignored.

To fix a hyperparameter instead of searching it, give one choice
(`"optimizer": ["adam"]`) or equal ends (`"epochs": [10, 10]`). To try a
different search, copy `search01.json` to `search02.json`, edit it, and pass
`--config search02.json`; each search config gets its own study and results.

## How Optuna searches

- **Trials and the objective.** A trial is one training run with one setting.
  Optuna calls the objective (`_Objective` in `search.py`) once per trial; it
  builds the config with `suggest_config()`, trains with the same `build()`
  as `main()`, and returns the best validation accuracy.
- **The TPE sampler** (Tree-structured Parzen Estimator) picks the settings.
  The first trials are random; after that it compares the settings of the
  good trials with those of the bad ones and samples more often where the
  good ones are. It usually finds good settings in far fewer trials than a
  grid or random search.
- **The median pruner** stops a trial when its validation accuracy after an
  epoch is below the median of the earlier trials at the same epoch. The
  callback [`OptunaPruning`](callbacks/optuna_pruning.py) reports the
  accuracy after every epoch and stops training when the pruner says so; the
  search then marks the trial as pruned. `pruner_startup_trials` trials always run to the end first, so
  there is something to compare with, and no trial is pruned during its first
  `pruner_warmup_epochs` epochs.
- **Seeds.** Every trial uses the base config's seed, so a trial and its
  saved config train identically: with `train_fraction` 1.0, training
  `search01_best.json` reproduces the best trial's validation accuracy.

## Results

Search results go to
`<OUTPUT_DIR>/LitWBHSTrainingBasicNeuralNetwork/search01/`:

| File | What it is |
| --- | --- |
| `study.db` | The Optuna study (an SQLite database), saved after every trial. Running the same search again **adds** trials to it, so an interrupted search continues where it stopped. To start over, delete the folder or use a new search config name. |
| `trials.csv` | One row per trial: state (`COMPLETE` or `PRUNED`), `val_acc`, epochs run, number of parameters, duration, and every sampled setting. |
| `best_config.json` | The training config of the best trial, with a `found_by` entry naming the trial. Also saved as `configs/search01_best.json`. |
| `wandb/` | Only with `"log_trials_to_wandb": true`: the trials' local W&B files (offline runs to `wandb sync` if there was no key). |

`search("search01.json", n_trials=0)` (or `--n-trials 0`) runs no new trial
and rewrites these files from the saved study. In Colab this brings back
`configs/search01_best.json` after a runtime reset deleted the repo folder.

Training the best config gives a normal run folder,
`<OUTPUT_DIR>/LitWBHSTrainingBasicNeuralNetwork/search01_best/<timestamp>/`,
as in [LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/README.md#run-folder).
`runs_summary.csv` now also records the optimizer, regularization,
scheduler, early stopping, activation, and the number of epochs actually run.

## Comparing the trials in W&B

By default the trials are **not** logged to W&B: starting and uploading one
W&B run per trial adds a few seconds to every trial and fills the project
with dozens of runs, while Optuna's plots and `trials.csv` already compare
them. Only training the best config (step 3) is logged, like any other run.

To log the trials too, set `"log_trials_to_wandb": true` in the search
config's `"study"`. Every trial is then a W&B run in the project
`LitWBHSTrainingBasicNeuralNetwork`, in the group `search01`, with job type
`search`. Its config holds the sampled settings, and its summary
`best_val_acc` and `state` (`complete` or `pruned`).

1. In the **Runs** table, filter or group by `group`, show the settings you
   care about, and sort by `best_val_acc`.
2. Add a **Parallel coordinates** panel with the searched settings (e.g.
   `training.optimizer`, `training.lr`, `model.dropout`, `num_params`) and
   `best_val_acc` as the last axis. Each trial is one line; highlighting the
   high end of `best_val_acc` shows which settings the best trials share.
3. Add a **Parameter importance** panel for `best_val_acc`.
4. The `val_acc` chart overlays all trials (pruned ones stop early), and the
   `lr-...` chart shows what each scheduler did to the learning rate.

The general W&B setup (keys, offline mode, `wandb sync`) is explained in the
[LitWBTrainingBasicConvnet README](../LitWBTrainingBasicConvnet/README.md).

## How long a search takes

On a GPU a trial takes from a few seconds (pruned after one epoch) to about a
minute or more (15 epochs with a small batch size), so the 30 trials of
`search01.json` take roughly 10 to 30 minutes. To check that everything works
first, run a few trials: `--n-trials 3`. Fewer epochs per trial make the
search faster, but favor settings that learn quickly over settings that end
best.

**Training on part of the data.** Loading images is often the bottleneck,
especially in Colab, which has only 2 CPU cores for it (`"num_workers":
"auto"` uses up to 8 cores elsewhere). To make every trial faster, train each
trial on a fraction of the 54,000 training images:

```bash
python -m LitWBHSTrainingBasicNeuralNetwork.search --config search01.json --train-fraction 0.25
```

or `search("search01.json", train_fraction=0.25)`, or `"train_fraction"` in
the search config's `"study"`. The subset is stratified: every class keeps
the same fraction of its images (e.g. about 1,350 of each class at 0.25), so
no class is over- or under-represented by chance. It is drawn with the seed,
so every trial trains on the same images.

- The validation set stays complete (6,000 images), so trials are still
  ranked reliably. Shrinking it too would make the scores noisy: at about 88%
  accuracy, 6,000 images give a score accurate to about ±0.4%, 1,200 only to
  about ±0.9%, as large as the gaps between good trials.
- The saved best config keeps the base config's `"train_fraction"` (1.0), so
  section 3 of the workflow still trains the final model on all the data.
- Settings that depend on the amount of data may not transfer exactly: with
  less data, a model overfits sooner, so the search may favor more dropout or
  L1/L2, and an epoch has fewer steps. Use the fraction to explore quickly,
  and `1.0` for a final, narrower search when you have the time.
- Keep the same fraction for all trials of a study; to change it, use a new
  search config name (e.g. `search02.json`), since scores from different
  fractions are not comparable.

`"train_fraction"` also works in a training config's `"data"` section, e.g.
for a quick test of `main()`.

## Things to try

- After a first search, read the importances: fix the unimportant settings
  and search the important ones in a narrower range (`search02.json`).
- Compare the best config's test accuracy with `config01.json`'s. How much
  did the search gain?
- Search only the learning rate and the optimizer, with everything else
  fixed: how much of the gain comes from those two?
- Add a hyperparameter: e.g. a `"label_smoothing"` value for the
  cross-entropy loss. Add it to the config, use it in `LitMLP`, and sample it
  in `suggest_config()`.
- Replace `TPESampler` with `optuna.samplers.RandomSampler` in `search.py`
  and compare how fast the best score rises.
