LitWBTransferLearningResnet
===========================

Transfer learning with a ResNet-18 pretrained on ImageNet, applied to
`Oxford Flowers-102 <https://www.robots.ox.ac.uk/~vgg/data/flowers/102/>`_
(102 flower species, 10 training images per class), with the Lightning + W&B
setup of :doc:`LitWBTrainingBasicConvnet`. The network is the one built in
section 8 of the
`ResNetWalkThrough notebook <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/ResNetWalkThrough>`_:
the ResNet-18 backbone with a new classification head.

Running
-------

Run from the repository root, one config per strategy:

.. code-block:: bash

   python -m LitWBTransferLearningResnet --config config01.json   # frozen backbone
   python -m LitWBTransferLearningResnet --config config02.json   # fine-tuning
   python -m LitWBTransferLearningResnet --config config03.json   # from scratch

or, from Python or a notebook started at the repository root:

.. code-block:: python

   from LitWBTransferLearningResnet import main

   main("config01.json")

The dataset (about 345 MB) is downloaded to ``data/flowers-102/`` on the
first run. A GPU is strongly recommended.

The three configs
-----------------

The configs live in ``LitWBTransferLearningResnet/configs/``; their
``"model"`` section picks the strategy:

* ``config01.json``, **feature extraction**: ``"pretrained": true,
  "freeze_backbone": true``. Only the new head is trained (52,326 parameters).
* ``config02.json``, **fine-tuning**: ``"pretrained": true,
  "freeze_backbone": false``. The whole network is trained, the head with
  ``"lr"`` and the backbone with the 10 times smaller ``"backbone_lr"``.
* ``config03.json``, **from scratch**: ``"pretrained": false``. The baseline
  that shows what the ImageNet weights are worth.

All three use early stopping on ``val_acc``
(:class:`~LitWBTransferLearningResnet.callbacks.early_stopping.ResumableEarlyStopping`).
A frozen backbone also keeps its BatchNorm layers in eval mode, so their
running statistics do not change either
(:meth:`~LitWBTransferLearningResnet.models.components.resnet.ResNetClassifier.train`).

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/LitWBTransferLearningResnet>`_
explains the strategies, how to read the W&B report, and how to turn on mixed
precision.

What is logged
--------------

* Per epoch, for ``train``, ``val``, and ``test``: ``<stage>_loss``,
  ``<stage>_acc``, ``<stage>_acc_top5``, and the macro averages
  ``<stage>_precision``, ``<stage>_recall``, and ``<stage>_f1``
  (:class:`~LitWBTransferLearningResnet.models.lit_resnet.LitResNet`).
* The config of the run and its number of trainable parameters.
* After training, for the best checkpoint on the validation set and on the
  test set, logged by
  :class:`~LitWBTransferLearningResnet.callbacks.wandb_evaluation.LogEvaluation`:
  a predictions table, galleries of right predictions and of the most
  confident mistakes, a per-class table (accuracy, precision, recall, F1, most
  confused class), precision-recall curves of the 5 worst and 5 best classes,
  a confusion matrix, and the 10 most confused class pairs.

Without ``WANDB_API_KEY`` in ``.env``, or if W&B cannot log in, the run is
logged offline, as in :doc:`LitWBTrainingBasicConvnet`.

Outputs
-------

Each run writes ``runs/LitWBTransferLearningResnet/<config>/<timestamp>/``
with ``config.json``, ``hparams.json``, ``metrics.csv``, the best and last
checkpoints, and ``wandb/``. A row per run is appended to
``runs_summary.csv`` in the config folder, including ``pretrained``,
``freeze_backbone``, ``trainable_params``, ``test_acc_top5``, and ``test_f1``.

:func:`~LitWBTransferLearningResnet.models.loading.load_model` loads a trained
model back from its run; all weights come from the checkpoint:

.. code-block:: python

   from LitWBTransferLearningResnet import load_model

   model = load_model("config02")   # newest run of config02, best checkpoint

To train a run further, pass ``resume_from`` to
:func:`~LitWBTransferLearningResnet.main.main`:

.. code-block:: python

   main("config02.json", resume_from="config02")

On the command line: ``--resume-from config02``.

API
---

Entry point
~~~~~~~~~~~

.. automodule:: LitWBTransferLearningResnet.main
   :members:

Callbacks
~~~~~~~~~

.. automodule:: LitWBTransferLearningResnet.callbacks.wandb_evaluation
   :members:

.. automodule:: LitWBTransferLearningResnet.callbacks.early_stopping
   :members:

Models
~~~~~~

.. automodule:: LitWBTransferLearningResnet.models.lit_resnet
   :members:

.. automodule:: LitWBTransferLearningResnet.models.components.resnet
   :members:

.. automodule:: LitWBTransferLearningResnet.models.loading
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: LitWBTransferLearningResnet.dataloaders.flowers102
   :members:

Utilities
~~~~~~~~~

.. automodule:: LitWBTransferLearningResnet.utils.paths
   :members:

.. automodule:: LitWBTransferLearningResnet.utils.experiment
   :members:
