# LitWBTrainingBasicNeuralNetwork

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/blob/main/LitWBTrainingBasicNeuralNetwork/lit_wb_training_basic_neural_network_notebook.ipynb)
This project's notebook: [`lit_wb_training_basic_neural_network_notebook.ipynb`](lit_wb_training_basic_neural_network_notebook.ipynb).

The fully connected network (MLP) of
[LitTrainingBasicNeuralNetwork](../LitTrainingBasicNeuralNetwork/), tracked
with [Weights & Biases](https://wandb.ai) (W&B) exactly like
[LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/): live charts, a
table of test predictions, and a confusion matrix on wandb.ai instead of
plotting code, with `metrics.csv` kept locally as a backup. After 10 epochs
the test accuracy is about 88.6%.

This Lightning + W&B setup is the **approach you must follow** when training
your own models: start new projects from this one or from
[LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/).

Read the projects it combines first:

- [TrainingBasicNeuralNetwork](../TrainingBasicNeuralNetwork/README.md)
  explains the network: fully connected layers, ReLU, dropout, and how to
  count parameters.
- [LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/README.md)
  explains W&B: setting up your key (in `.env`, or Colab Secrets with the
  project notebook's **Load API keys** cell), offline mode and `wandb sync`,
  what is logged, and how to build the dashboard. All of it applies here
  unchanged.

How to set up and run the project and where results go is explained in the
[main README](../README.md). Function and class details are in the
[API reference](https://txstcodeplayground.github.io/CodeCS4337Fall2026/LitWBTrainingBasicNeuralNetwork.html).

## Boilerplate lives in `colabutilscs4337`

Setting up Colab, finding paths and configs, creating run folders, resuming, and
connecting to W&B were copied into every project. They now come from the
course library [`colabutilscs4337`](https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html), which is
in [`requirements.txt`](../requirements.txt), so this project keeps only what is
specific to the experiment: the data, the network, and the metrics.

| Was in this project | Now comes from `colabutilscs4337` |
| --- | --- |
| `utils/paths.py`: repo, `configs/`, `.env`, `DATA_DIR`, `OUTPUT_DIR` paths | `cu.project.Project` (`PROJECT` in [`utils/paths.py`](utils/paths.py)) |
| `utils/experiment.py`: `runs_summary.csv` | `cu.project.append_run_summary` |
| `models/loading.py`: finding a run, its checkpoints, and its config | `Project.find_checkpoint`, `find_resume_checkpoint`, `run_config` |
| `main.py`: command line, resume, run folder, `CSVLogger`, checkpoints, `config.json` and `hparams.json` | `cu.lit.start_run` |
| `main.py`: `_connect_wandb` | `cu.lit.connect_wandb` |
| Notebook: mount Drive, clone, `.env`, requirements, API keys | `cu.project.setup_colab`, `cu.env.load_secrets`, `cu.env.sync_env_file` |

See [LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/README.md#boilerplate-lives-in-colabutilscs4337) has the same table.

## What is different from LitWBTrainingBasicConvnet

Only the network, as in the Lightning projects without W&B:

| | LitWBTrainingBasicConvnet | This project |
| --- | --- | --- |
| Network | `models/components/convnet.py`: `ConvNet` | [`models/components/mlp.py`](models/components/mlp.py): `MLP` |
| LightningModule | `models/lit_convnet.py`: `LitConvNet` | [`models/lit_mlp.py`](models/lit_mlp.py): `LitMLP` (same code, renamed) |
| `"model"` in the config | `{"dropout": 0.25}` | `{"hidden_sizes": [256, 128], "dropout": 0.2}` |
| W&B project | `LitWBTrainingBasicConvnet` | `LitWBTrainingBasicNeuralNetwork` |
| Parameters | about 422,000 | 235,146 |

The W&B logger, the offline fallback, the prediction-table callback, and the
local backup are the same code. The run prints the number of parameters at
the start and also sends it to W&B as `num_params` in the run's config, next
to `model.hidden_sizes` and `model.dropout`. `runs_summary.csv` gets the
columns `hidden_sizes` and `num_params`, and `hparams.json` a `"net"` section.

## Comparing network sizes in W&B

Because the network size is in each run's config, W&B can compare
architectures without any extra code:

1. Make a few copies of the config with different `"hidden_sizes"`, e.g.
   `[64]`, `[256, 128]`, and `[512, 256, 128]`, and run each one.
2. In the project's **Runs** table, show the `model.hidden_sizes` and
   `num_params` columns and sort by `test_acc`: does a bigger network always
   win?
3. Add a **Scatter plot** panel with X = `num_params` and Y = `test_acc` to
   see accuracy against network size.
4. Overlay `train_acc` and `val_acc` (see the dashboard steps in the
   [LitWBTrainingBasicConvnet README](../LitWBTrainingBasicConvnet/README.md#build-the-dashboard))
   to see which sizes overfit: the training curve keeps rising while
   validation flattens.

To compare against the ConvNet, run LitWBTrainingBasicConvnet too: the two
projects log the same metric names, so their runs can be compared side by
side in a W&B report.

## Run folder

The same files as in
[LitWBTrainingBasicConvnet](../LitWBTrainingBasicConvnet/README.md#run-folder),
in `runs/LitWBTrainingBasicNeuralNetwork/<config>/<timestamp>/`. To load a
trained model back, e.g. the best checkpoint of the newest `config01` run, see
[Loading a trained model](../LitTrainingBasicConvnet/README.md#loading-a-trained-model):

```python
from LitWBTrainingBasicNeuralNetwork import load_model

model = load_model("config01")
```

To train a run further instead, resume it, as described in
[Continuing training](../LitTrainingBasicConvnet/README.md#continuing-training):
`main("config01.json", resume_from="config01")`, or
`--resume-from config01` on the command line. The continued run is a new W&B
run.

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
python -m LitWBTrainingBasicNeuralNetwork --config config01.json
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
  /content/drive/MyDrive/CodeCS4337Fall2026/runs/LitWBTrainingBasicNeuralNetwork/config01/2026-11-05_14-30-00/
  ```

  The timestamp is printed when the run starts. From the session:

  ```bash
  ls /content/drive/MyDrive/CodeCS4337Fall2026/runs/LitWBTrainingBasicNeuralNetwork/config01
  ```

  or on [drive.google.com](https://drive.google.com), under **My Drive >
  CodeCS4337Fall2026 > runs > LitWBTrainingBasicNeuralNetwork**. Check that `runs_summary.csv` next to
  the run folders has one new row per run.

- **Resume a cut-off run:** start a new session, repeat the steps above, and
  continue from the last checkpoint on Drive:

  ```bash
  python -m LitWBTrainingBasicNeuralNetwork --config config01.json --resume-from config01
  ```

When everything is checked, stop the session so it no longer uses your Colab
quota:

```bash
colab stop -s cs4337
```

