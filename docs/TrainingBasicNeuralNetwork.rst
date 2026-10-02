TrainingBasicNeuralNetwork
==========================

Train a fully connected neural network (multilayer perceptron, MLP) on
Fashion-MNIST, using only ``nn.Linear`` layers. The data, training loop, and
project layout are the same as in :doc:`TrainingBasicConvnet`; only the
network differs.

Run from the repository root:

.. code-block:: bash

   python -m TrainingBasicNeuralNetwork --config config01.json

or, from inside the project folder:

.. code-block:: bash

   cd TrainingBasicNeuralNetwork
   python main.py --config config01.json

or, from Python or a notebook started at the repository root:

.. code-block:: python

   from TrainingBasicNeuralNetwork import main

   main("config01.json")

The config in ``TrainingBasicNeuralNetwork/configs/`` sets the network with
``"model": {"hidden_sizes": [256, 128], "dropout": 0.2}``: one entry per
hidden layer, giving its number of units. ``[]`` gives a linear model with no
hidden layer.

Each run writes a timestamped folder under
``runs/TrainingBasicNeuralNetwork/<config>/`` with the config snapshot,
checkpoints, ``history.json``, and a results file, and appends a row
(including ``hidden_sizes`` and ``num_params``) to ``runs_summary.csv``.

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/TrainingBasicNeuralNetwork>`_
explains the layers, counts the parameters, and compares the network with a
ConvNet.

API
---

Entry point
~~~~~~~~~~~

.. automodule:: TrainingBasicNeuralNetwork.main
   :members:

Models
~~~~~~

.. automodule:: TrainingBasicNeuralNetwork.models.mlp
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: TrainingBasicNeuralNetwork.dataloaders.fashion_mnist
   :members:

Trainers
~~~~~~~~

.. automodule:: TrainingBasicNeuralNetwork.trainers.trainer
   :members:

Utilities
~~~~~~~~~

.. automodule:: TrainingBasicNeuralNetwork.utils.paths
   :members:

.. automodule:: TrainingBasicNeuralNetwork.utils.reproducibility
   :members:

.. automodule:: TrainingBasicNeuralNetwork.utils.experiment
   :members:
