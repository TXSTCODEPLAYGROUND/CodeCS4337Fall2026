LitTrainingBasicNeuralNetwork
=============================

The fully connected network of :doc:`TrainingBasicNeuralNetwork`, trained with
`PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_. The project
layout, metrics, plots, and optional LitLogger tracking are the same as in
:doc:`LitTrainingBasicConvnet`; only the network differs.

Run from the repository root:

.. code-block:: bash

   python -m LitTrainingBasicNeuralNetwork --config config01.json

or, from inside the project folder:

.. code-block:: bash

   cd LitTrainingBasicNeuralNetwork
   python main.py --config config01.json

or, from Python or a notebook started at the repository root:

.. code-block:: python

   from LitTrainingBasicNeuralNetwork import main

   main("config01.json")

The config in ``LitTrainingBasicNeuralNetwork/configs/`` sets the network with
``"model": {"hidden_sizes": [256, 128], "dropout": 0.2}``, and ``"litlogger"``
turns LitLogger on or off (off by default). ``runs_summary.csv`` also records
``hidden_sizes`` and ``num_params``, and ``hparams.json`` has a ``"net"``
section with the network settings.

Because the network is passed in, give it again when loading a checkpoint:

.. code-block:: python

   from LitTrainingBasicNeuralNetwork.models import MLP, LitMLP

   model = LitMLP.load_from_checkpoint("path/to/best.ckpt", net=MLP(hidden_sizes=[256, 128]))

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/LitTrainingBasicNeuralNetwork>`_
compares the project with LitTrainingBasicConvnet.

API
---

Entry point
~~~~~~~~~~~

.. automodule:: LitTrainingBasicNeuralNetwork.main
   :members:

Models
~~~~~~

.. automodule:: LitTrainingBasicNeuralNetwork.models.lit_mlp
   :members:

.. automodule:: LitTrainingBasicNeuralNetwork.models.components.mlp
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: LitTrainingBasicNeuralNetwork.dataloaders.fashion_mnist
   :members:

Utilities
~~~~~~~~~

.. automodule:: LitTrainingBasicNeuralNetwork.utils.paths
   :members:

.. automodule:: LitTrainingBasicNeuralNetwork.utils.experiment
   :members:

.. automodule:: LitTrainingBasicNeuralNetwork.utils.plots
   :members:
