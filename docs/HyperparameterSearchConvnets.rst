HyperparameterSearchConvnets
============================

:doc:`LitWBHSTrainingBasicNeuralNetwork` for convolutional networks: the
Optuna search also picks the **architecture** of a ConvNet built from
ResNet-like blocks (number of blocks, channels of the first block,
convolutions per block, BatchNorm, skip connections, activation, dropout),
together with the training settings. The base config, ``config01.json``, is a
plain two-block ConvNet (32 then 64 channels, one convolution each), the
baseline the search tries to beat. Like every ``LitWB...`` project, it trains
with Lightning and logs to W&B.

The network is :class:`~HyperparameterSearchConvnets.models.components.convnet.ConvNet`,
a stack of :class:`~HyperparameterSearchConvnets.models.components.convnet.ConvBlock`
blocks set by the config's ``"model"`` section:

.. code-block:: json

   "model": {
     "channels": [32, 64],
     "convs_per_block": 1,
     "batch_norm": false,
     "skip_connections": false,
     "activation": "relu",
     "dropout": 0.0
   }

The first two blocks end with max pooling (28x28 to 14x14 to 7x7). The search
picks ``num_blocks`` and ``first_channels``; the channels then double each
time the image is halved, up to ``max_channels``
(:func:`~HyperparameterSearchConvnets.models.components.convnet.block_channels`).

1. Search, with a search config from ``HyperparameterSearchConvnets/configs/``:

   .. code-block:: bash

      python -m HyperparameterSearchConvnets.search --config search01.json

2. Train and test the best settings, saved as ``configs/search01_best.json``:

   .. code-block:: bash

      python -m HyperparameterSearchConvnets --config search01_best.json

From inside the project folder, use ``python search.py`` and ``python main.py``
with the same flags. From Python or a notebook started at the repository root:

.. code-block:: python

   from HyperparameterSearchConvnets import main
   from HyperparameterSearchConvnets.search import search

   study = search("search01.json", n_trials=30)
   main("search01_best.json")

The search config names the training config to start from
(``"base_config"``), the study settings (``"study"``), and a range or a list
of choices for every hyperparameter (``"search_space"``). Trials are scored
by their best validation accuracy; the test set is only used when the best
config is trained. The study is saved in
``<OUTPUT_DIR>/HyperparameterSearchConvnets/<search_name>/study.db``, so
running the same search again adds trials to it.

The trials save no checkpoints. After training the best config,
:func:`~HyperparameterSearchConvnets.models.loading.load_model` loads
that model back, with the network the search picked:

.. code-block:: python

   from HyperparameterSearchConvnets import load_model

   model = load_model("search01_best")   # newest run, best checkpoint

To train a run further instead, pass ``resume_from`` to :func:`~HyperparameterSearchConvnets.main.main`:
it continues from the run's last checkpoint (weights, optimizer, and epoch
count) for the config's ``"epochs"`` more epochs, into a new run folder.

.. code-block:: python

   main("search01_best.json", resume_from="search01_best")

On the command line: ``--resume-from search01_best``.

The project folder has a Colab notebook, ``hyperparameter_search_convnets_notebook.ipynb``,
that runs the search, plots the results, shows the best network in Netron
(:func:`~HyperparameterSearchConvnets.utils.onnx_export.export_onnx`), and
trains the best config. The
`project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/HyperparameterSearchConvnets>`_
explains the network builder, the search space, Optuna's sampler and pruner, and the W&B dashboard.

API
---

Search
~~~~~~

.. automodule:: HyperparameterSearchConvnets.search
   :members:

Training
~~~~~~~~

.. automodule:: HyperparameterSearchConvnets.main
   :members:

Callbacks
~~~~~~~~~

.. automodule:: HyperparameterSearchConvnets.callbacks.optuna_pruning
   :members:

.. automodule:: HyperparameterSearchConvnets.callbacks.wandb_predictions
   :members:

Models
~~~~~~

.. automodule:: HyperparameterSearchConvnets.models.lit_convnet
   :members:

.. automodule:: HyperparameterSearchConvnets.models.components.convnet
   :members:

.. automodule:: HyperparameterSearchConvnets.models.loading
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: HyperparameterSearchConvnets.dataloaders.fashion_mnist
   :members:

Utilities
~~~~~~~~~

.. automodule:: HyperparameterSearchConvnets.utils.paths
   :members:

.. automodule:: HyperparameterSearchConvnets.utils.experiment
   :members:

.. automodule:: HyperparameterSearchConvnets.utils.tracking
   :members:

.. automodule:: HyperparameterSearchConvnets.utils.onnx_export
   :members:
