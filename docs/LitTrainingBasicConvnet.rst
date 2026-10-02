LitTrainingBasicConvnet
=======================

The same experiment as :doc:`TrainingBasicConvnet` (a small ConvNet on
Fashion-MNIST), written with `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_.
Lightning provides the training loop, device placement, logging, and
checkpointing, so the project is three short files:

* ``model.py``: :class:`~LitTrainingBasicConvnet.model.LitConvNet`, the network
  plus what to do with one batch (``training_step``, ``validation_step``,
  ``test_step``) and which optimizer to use.
* ``data.py``: :class:`~LitTrainingBasicConvnet.data.FashionMNISTDataModule`,
  which downloads the data, splits it, and builds the dataloaders.
* ``main.py``: :func:`~LitTrainingBasicConvnet.main.main`, which reads a
  config and hands the model and data to ``lightning.Trainer``.

Running
-------

Run from the repository root:

.. code-block:: bash

   python -m LitTrainingBasicConvnet --config config01.json

or, from Python or a notebook started at the repository root:

.. code-block:: python

   from LitTrainingBasicConvnet import main

   main("config01.json")

The config lives in ``LitTrainingBasicConvnet/configs/`` and has the same
fields as the other project, except that the device is set with
``"accelerator"`` (``"auto"``, ``"gpu"``, ``"mps"``, or ``"cpu"``).

Outputs
-------

Each run writes ``runs/LitTrainingBasicConvnet/<config>/<timestamp>/``:

* ``config.json``: copy of the config used
* ``metrics.csv``: loss and accuracy per epoch, written by Lightning's ``CSVLogger``
* ``hparams.yaml``: model and data settings
* ``best_epochXX_valaccY.ckpt``: checkpoint with the best validation accuracy
* ``last.ckpt``: checkpoint after the last epoch

Lightning numbers epochs from 0, so ``best_epoch09`` is the tenth epoch. A row
per run is appended to ``runs_summary.csv`` in the config folder.

API
---

Entry point
~~~~~~~~~~~

.. automodule:: LitTrainingBasicConvnet.main
   :members: main, resolve_config_path, resolve_repo_path

Model
~~~~~

.. automodule:: LitTrainingBasicConvnet.model
   :members:

Data
~~~~

.. automodule:: LitTrainingBasicConvnet.data
   :members:
