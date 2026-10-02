LitWBTrainingBasicNeuralNetwork
===============================

The fully connected network of :doc:`LitTrainingBasicNeuralNetwork`, tracked
with `Weights & Biases <https://wandb.ai>`__ (W&B) exactly like
:doc:`LitWBTrainingBasicConvnet`; only the network differs.

Run from the repository root:

.. code-block:: bash

   python -m LitWBTrainingBasicNeuralNetwork --config config01.json

or, from inside the project folder:

.. code-block:: bash

   cd LitWBTrainingBasicNeuralNetwork
   python main.py --config config01.json

or, from Python or a notebook started at the repository root:

.. code-block:: python

   from LitWBTrainingBasicNeuralNetwork import main

   main("config01.json")

The config in ``LitWBTrainingBasicNeuralNetwork/configs/`` sets the network
with ``"model": {"hidden_sizes": [256, 128], "dropout": 0.2}`` and the W&B
project with ``"wandb": {"project": "LitWBTrainingBasicNeuralNetwork"}``. The
W&B key comes from ``WANDB_API_KEY`` in ``.env`` (in Colab: a Colab Secret and
the starter notebook's *Load API keys* cell). Without it, or if W&B cannot log
in, the run is logged offline and can be uploaded later with ``wandb sync``.

The number of parameters is sent to W&B as ``num_params`` in the run's config,
so runs with different ``hidden_sizes`` can be compared on wandb.ai. Locally,
``runs_summary.csv`` records ``hidden_sizes`` and ``num_params``.

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/LitWBTrainingBasicNeuralNetwork>`_
shows how to compare network sizes in W&B.

API
---

Entry point
~~~~~~~~~~~

.. automodule:: LitWBTrainingBasicNeuralNetwork.main
   :members:

Callbacks
~~~~~~~~~

.. automodule:: LitWBTrainingBasicNeuralNetwork.callbacks.wandb_predictions
   :members:

Models
~~~~~~

.. automodule:: LitWBTrainingBasicNeuralNetwork.models.lit_mlp
   :members:

.. automodule:: LitWBTrainingBasicNeuralNetwork.models.components.mlp
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: LitWBTrainingBasicNeuralNetwork.dataloaders.fashion_mnist
   :members:

Utilities
~~~~~~~~~

.. automodule:: LitWBTrainingBasicNeuralNetwork.utils.paths
   :members:

.. automodule:: LitWBTrainingBasicNeuralNetwork.utils.experiment
   :members:
