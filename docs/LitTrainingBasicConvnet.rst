LitTrainingBasicConvnet
=======================

The same experiment as :doc:`TrainingBasicConvnet` (a small ConvNet on
Fashion-MNIST), written with `PyTorch Lightning <https://lightning.ai/docs/pytorch/stable/>`_.
Lightning provides the training loop, device placement, logging, and
checkpointing, so there is no ``trainers/`` package.

Running
-------

Run from the repository root:

.. code-block:: bash

   python -m LitTrainingBasicConvnet --config config01.json

or, from inside the project folder:

.. code-block:: bash

   cd LitTrainingBasicConvnet
   python main.py --config config01.json

or, from Python or a notebook started at the repository root:

.. code-block:: python

   from LitTrainingBasicConvnet import main

   main("config01.json")

The config lives in ``LitTrainingBasicConvnet/configs/`` and has the same
fields as the other project, except that the device is set with
``"accelerator"`` (``"auto"``, ``"gpu"``, ``"mps"``, or ``"cpu"``), and
``"litlogger"`` turns online tracking on or off (see `LitLogger`_).

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/LitTrainingBasicConvnet>`_
introduces Lightning and explains the metrics and plots in more detail.

How the project is organized
----------------------------

The layout follows the structure Lightning recommends in its
`style guide <https://lightning.ai/docs/pytorch/stable/starter/style_guide.html>`_,
the same one used by the widely used
`Lightning-Hydra-Template <https://github.com/ashleve/lightning-hydra-template>`_:

.. code-block:: text

   LitTrainingBasicConvnet/
   ├── main.py                    # thin entry point: config -> DataModule + LightningModule + Trainer
   ├── configs/                   # experiment settings (JSON)
   ├── models/
   │   ├── lit_convnet.py         # LitConvNet (LightningModule): how a network is trained
   │   └── components/
   │       └── convnet.py         # ConvNet (nn.Module): what the network computes
   ├── dataloaders/
   │   └── fashion_mnist.py       # FashionMNISTDataModule: download, split, batch
   └── utils/                     # paths, runs_summary.csv, and plots

* **Model vs. system.** Lightning's style guide calls a plain network such as
  a ResNet a *model*, and the ``LightningModule`` that trains it a *system*. It
  recommends keeping them separate and passing the network in.
  :class:`~LitTrainingBasicConvnet.models.components.convnet.ConvNet` is a plain
  ``nn.Module``, identical to the network in :doc:`TrainingBasicConvnet`, with
  no Lightning code.
  :class:`~LitTrainingBasicConvnet.models.lit_convnet.LitConvNet` receives it
  as ``net`` and adds the loss, metrics, logging, and optimizer. To try
  another architecture, add a file to ``models/components/`` and pass that
  network instead; ``LitConvNet`` does not change.
* **Data.** :class:`~LitTrainingBasicConvnet.dataloaders.fashion_mnist.FashionMNISTDataModule`
  is a ``LightningDataModule``: it downloads the dataset, makes the seeded
  train/validation split, and builds the dataloaders. The dataset files stay
  in the shared ``data/`` folder at the repository root (``DATA_DIR`` in
  ``.env``), not inside the package, so every project reuses one download.
  The template calls this code folder ``data/``; here it is ``dataloaders/``,
  matching :doc:`TrainingBasicConvnet` and keeping code apart from the dataset
  folder.
* **Entry point.** :func:`~LitTrainingBasicConvnet.main.main` only reads the
  config, builds the three objects, and calls ``trainer.fit`` and
  ``trainer.test``.

The template adds things this course project leaves out: Hydra YAML configs
instead of JSON, tests, and packaging files.

Metrics
-------

``LitConvNet`` uses `torchmetrics <https://lightning.ai/docs/torchmetrics/stable/>`_,
which accumulate over all batches and are computed once per epoch. For each
stage (``train``, ``val``, ``test``) it logs ``<stage>_loss``,
``<stage>_acc``, ``<stage>_precision`` and ``<stage>_recall`` (macro averages
over the classes), and ``<stage>_acc_<class>`` for every class, e.g.
``val_acc_Shirt``. The checkpoint callback keeps the epoch with the best
``val_acc``.

Outputs
-------

Each run writes ``runs/LitTrainingBasicConvnet/<config>/<timestamp>/``:

* ``config.json``: copy of the config used
* ``metrics.csv``: every metric per epoch, written by Lightning's ``CSVLogger``
* ``hparams.json``: optimizer and data settings
* ``best_epochXX_valaccY.ckpt``: checkpoint with the best validation accuracy
* ``last.ckpt``: checkpoint after the last epoch
* ``loss.png`` and ``accuracy.png``: training and validation curves on one graph
* ``accuracy_per_class.png``: every class per epoch, training solid and validation dashed
* ``predictions.png`` and ``wrong_predictions.png``: test images with their
  predicted and true class

Lightning numbers epochs from 0, so ``best_epoch09`` is the tenth epoch. A row
per run is appended to ``runs_summary.csv`` in the config folder.

:func:`~LitTrainingBasicConvnet.models.loading.load_model` loads a trained
model back: it finds the checkpoint, rebuilds the network from the run's
``config.json``, and returns the model in eval mode:

.. code-block:: python

   from LitTrainingBasicConvnet import load_model

   model = load_model("config01")                       # newest run, best checkpoint
   model = load_model("config01/2026-10-02_12-29-09")   # one specific run
   model = load_model("config01", which="last")         # the last epoch

LitLogger
---------

`LitLogger <https://lightning.ai/docs/pytorch/stable/visualize/experiment_managers.html>`__
is Lightning AI's experiment tracker: logged metrics, the plots, and the config
of each run also appear on lightning.ai, where runs can be watched live and
compared. It needs a free Lightning AI account, so it is off by default. To
turn it on, copy ``LIGHTNING_USER_ID`` and ``LIGHTNING_API_KEY`` from
lightning.ai (profile picture, *Global Settings*, *Keys*, *Login via CLI*)
into ``.env`` (in Colab: add them as Colab Secrets and run the project
notebook's *Load API keys* cell) and set ``"litlogger": true`` in the config.
Without the keys it
is skipped with a warning. The local ``metrics.csv`` and plots are written
either way.

API
---

Entry point
~~~~~~~~~~~

.. automodule:: LitTrainingBasicConvnet.main
   :members:

Models
~~~~~~

.. automodule:: LitTrainingBasicConvnet.models.lit_convnet
   :members:

.. automodule:: LitTrainingBasicConvnet.models.components.convnet
   :members:

.. automodule:: LitTrainingBasicConvnet.models.loading
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: LitTrainingBasicConvnet.dataloaders.fashion_mnist
   :members:

Utilities
~~~~~~~~~

.. automodule:: LitTrainingBasicConvnet.utils.paths
   :members:

.. automodule:: LitTrainingBasicConvnet.utils.experiment
   :members:

.. automodule:: LitTrainingBasicConvnet.utils.plots
   :members:
