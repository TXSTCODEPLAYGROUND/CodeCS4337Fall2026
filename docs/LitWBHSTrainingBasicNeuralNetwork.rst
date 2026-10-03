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

The project folder has a Colab notebook, ``lit_wbhs_training_basic_neural_network_notebook.ipynb``,
that runs the search, plots the results, and trains the best config. The
`project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/LitWBHSTrainingBasicNeuralNetwork>`_
explains the search space, Optuna's sampler and pruner, and the W&B dashboard.

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

.. automodule:: LitWBHSTrainingBasicNeuralNetwork.utils.experiment
   :members:

.. automodule:: LitWBHSTrainingBasicNeuralNetwork.utils.tracking
   :members:
