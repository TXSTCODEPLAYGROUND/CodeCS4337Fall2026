LitWBHSTrainingBasicNeuralNetwork
=================================

:doc:`LitWBTrainingBasicNeuralNetwork` plus a hyperparameter search with
`Optuna <https://optuna.org>`__: network size, activation, dropout, batch
size, epochs, optimizer, learning rate, L1/L2 regularization, learning-rate
scheduler, and early stopping. The best trial is saved as a training config,
whose training run is logged to W&B (the trials themselves only with
``"log_trials_to_wandb": true``).

1. Search, with a search config from ``LitWBHSTrainingBasicNeuralNetwork/configs/``:

   .. code-block:: bash

      python -m LitWBHSTrainingBasicNeuralNetwork.search --config search01.json

2. Train and test the best settings, saved as ``configs/search01_best.json``:

   .. code-block:: bash

      python -m LitWBHSTrainingBasicNeuralNetwork --config search01_best.json

From inside the project folder, use ``python search.py`` and ``python main.py``
with the same flags. From Python or a notebook started at the repository root:

.. code-block:: python

   from LitWBHSTrainingBasicNeuralNetwork import main
   from LitWBHSTrainingBasicNeuralNetwork.search import search

   study = search("search01.json", n_trials=30)
   main("search01_best.json")

The search config names the training config to start from
(``"base_config"``), the study settings (``"study"``), and a range or a list
of choices for every hyperparameter (``"search_space"``). Trials are scored
by their best validation accuracy; the test set is only used when the best
config is trained. The study is saved in
``<OUTPUT_DIR>/LitWBHSTrainingBasicNeuralNetwork/<search_name>/study.db``, so
running the same search again adds trials to it.

The trials save no checkpoints. After training the best config,
:func:`~LitWBHSTrainingBasicNeuralNetwork.models.loading.load_model` loads
that model back, with the network the search picked:

.. code-block:: python

   from LitWBHSTrainingBasicNeuralNetwork import load_model

   model = load_model("search01_best")   # newest run, best checkpoint

To train a run further instead, pass ``resume_from`` to :func:`~LitWBHSTrainingBasicNeuralNetwork.main.main`:
it continues from the run's last checkpoint (weights, optimizer, and epoch
count) for the config's ``"epochs"`` more epochs, into a new run folder.

.. code-block:: python

   main("search01_best.json", resume_from="search01_best")

On the command line: ``--resume-from search01_best``.

In a Colab notebook, add the key as a Colab Secret and run the project notebook's
*Load API keys* cell (``cu.env.load_secrets("WANDB_API_KEY", persist_to=...)``). In a
Colab CLI (SSH) session, Colab Secrets are not available: put the key in a text file on
your Google Drive and pass its path as ``env_source`` to
:func:`colabutilscs4337.project.setup_colab` (the project README shows the commands). The
run stops with an error if that file is missing.

The project folder has a Colab notebook, ``lit_wbhs_training_basic_neural_network_notebook.ipynb``,
that runs the search, plots the results, and trains the best config. The
`project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/LitWBHSTrainingBasicNeuralNetwork>`_
explains the search space, Optuna's sampler and pruner, and the W&B dashboard.

Shared code: colabutilscs4337
-----------------------------

Setup, paths, configs, run folders, resuming, and the W&B connection used to be copied
into every project. They now come from the course library
`colabutilscs4337 <https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html>`__ (listed in
``requirements.txt``), so this project keeps only what is specific to the experiment:

* ``utils/paths.py`` is one line, ``PROJECT = cu.project.Project(...)``: repo, config,
  data, and run folders. The search uses ``PROJECT.read_config``, ``runs_dir`` and
  ``configs_dir``.
* ``main()`` starts with ``cu.lit.start_run`` (command line, config, resume, run folder,
  ``CSVLogger``, checkpoint callbacks) and ends with ``run.save`` and
  ``run.append_summary`` (``config.json``, ``hparams.json``, ``runs_summary.csv``).
* ``cu.lit.connect_wandb`` starts the W&B run, online or offline, for ``main()`` and for
  every search trial.

API
---

Search
~~~~~~

.. automodule:: LitWBHSTrainingBasicNeuralNetwork.search
   :members:

Training
~~~~~~~~

.. automodule:: LitWBHSTrainingBasicNeuralNetwork.main
   :members:

Callbacks
~~~~~~~~~

.. automodule:: LitWBHSTrainingBasicNeuralNetwork.callbacks.optuna_pruning
   :members:

.. automodule:: LitWBHSTrainingBasicNeuralNetwork.callbacks.wandb_predictions
   :members:

Models
~~~~~~

.. automodule:: LitWBHSTrainingBasicNeuralNetwork.models.lit_mlp
   :members:

.. automodule:: LitWBHSTrainingBasicNeuralNetwork.models.components.mlp
   :members:

.. automodule:: LitWBHSTrainingBasicNeuralNetwork.models.loading
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: LitWBHSTrainingBasicNeuralNetwork.dataloaders.fashion_mnist
   :members:

Utilities
~~~~~~~~~

.. automodule:: LitWBHSTrainingBasicNeuralNetwork.utils.paths
   :members:

The rest (``runs_summary.csv``, run folders, checkpoint lookup, the W&B connection) is in
`colabutilscs4337 <https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html>`__.
