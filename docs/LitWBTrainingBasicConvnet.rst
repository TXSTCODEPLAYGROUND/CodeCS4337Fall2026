LitWBTrainingBasicConvnet
=========================

The same experiment as :doc:`LitTrainingBasicConvnet` (the same network,
LightningModule, DataModule, and metrics), tracked with
`Weights & Biases <https://wandb.ai>`__ (W&B) through Lightning's
``WandbLogger``. W&B charts and tables replace the plotting code, so there is
no ``utils/plots.py``. Lightning + W&B is the recommended pattern for your own
experiments; :doc:`LitWBTrainingBasicNeuralNetwork` is the same setup with a
fully connected network.

Running
-------

Run from the repository root:

.. code-block:: bash

   python -m LitWBTrainingBasicConvnet --config config01.json

or, from inside the project folder:

.. code-block:: bash

   cd LitWBTrainingBasicConvnet
   python main.py --config config01.json

or, from Python or a notebook started at the repository root:

.. code-block:: python

   from LitWBTrainingBasicConvnet import main

   main("config01.json")

The config lives in ``LitWBTrainingBasicConvnet/configs/``. Instead of
``"litlogger"``, it has ``"wandb": {"project": "LitWBTrainingBasicConvnet"}``,
the W&B project the runs are sent to.

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/LitWBTrainingBasicConvnet>`_
explains how to get a W&B key and how to build the dashboard.

W&B key and offline mode
------------------------

Copy your API key from `wandb.ai/authorize <https://wandb.ai/authorize>`_
into ``.env`` as ``WANDB_API_KEY`` (in Colab: add it as a Colab Secret and run
the project notebook's *Load API keys* cell). Without it, or if W&B cannot log in (a
wrong key, no internet), the run is logged offline in the run folder and
prints the ``wandb sync`` command that uploads it later.

What is logged
--------------

* Every ``self.log`` value of ``LitConvNet``: ``<stage>_loss``,
  ``<stage>_acc``, ``<stage>_precision``, ``<stage>_recall``, and
  ``<stage>_acc_<class>`` for ``train``, ``val``, and ``test``.
* The config of the run, so runs can be compared by any setting.
* ``test_predictions``: a table of test images with predicted and true class,
  logged by :class:`~LitWBTrainingBasicConvnet.callbacks.wandb_predictions.LogTestPredictions`.
* ``confusion_matrix``: true vs. predicted class over the whole test set.

Outputs
-------

Each run writes ``runs/LitWBTrainingBasicConvnet/<config>/<timestamp>/`` with
``config.json``, ``hparams.json``, ``metrics.csv`` (a local backup written by
``CSVLogger``), the best and last checkpoints, and ``wandb/`` (W&B's local
copy of the run). A row per run is appended to ``runs_summary.csv`` in the
config folder.

API
---

Entry point
~~~~~~~~~~~

.. automodule:: LitWBTrainingBasicConvnet.main
   :members:

Callbacks
~~~~~~~~~

.. automodule:: LitWBTrainingBasicConvnet.callbacks.wandb_predictions
   :members:

Models
~~~~~~

.. automodule:: LitWBTrainingBasicConvnet.models.lit_convnet
   :members:

.. automodule:: LitWBTrainingBasicConvnet.models.components.convnet
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: LitWBTrainingBasicConvnet.dataloaders.fashion_mnist
   :members:

Utilities
~~~~~~~~~

.. automodule:: LitWBTrainingBasicConvnet.utils.paths
   :members:

.. automodule:: LitWBTrainingBasicConvnet.utils.experiment
   :members:
