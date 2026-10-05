UnetSemanticSegmentationExample
===============================

Binary **semantic segmentation** with a U-Net: for every pixel of a
colonoscopy image, decide whether it belongs to a polyp. The data is
`Kvasir-SEG <https://datasets.simula.no/kvasir-seg/>`_ (1,000 images with
hand-drawn masks), and the project uses the Lightning + W&B setup of
:doc:`LitWBTrainingBasicConvnet` (same ``main``, loggers, checkpoints, early
stopping, resuming, and ``load_model``).

Its notebook teaches the whole pipeline step by step: data exploration,
the Dataset and DataLoader with shapes printed at every step, binary
cross-entropy computed by hand, the U-Net block by block, torchinfo, Netron,
and torchview, a training loop in plain PyTorch, then the comparison of the
three experiments below, and a table of other segmentation datasets to try.

Running
-------

Run from the repository root, one config per experiment:

.. code-block:: bash

   python -m UnetSemanticSegmentationExample --config config01.json   # baseline
   python -m UnetSemanticSegmentationExample --config config02.json   # improved training
   python -m UnetSemanticSegmentationExample --config config03.json   # pretrained encoder

or, from Python or a notebook started at the repository root:

.. code-block:: python

   from UnetSemanticSegmentationExample import main

   main("config01.json")

The dataset (44 MB) is downloaded to ``data/kvasir-seg/`` on the first run
(:func:`~UnetSemanticSegmentationExample.dataloaders.kvasir_seg.prepare_kvasir`)
and split, with the seed, into 800 training, 100 validation, and 100 test
images. Every image is resized to 256 x 256. A GPU is strongly recommended.

The three configs
-----------------

The configs live in ``UnetSemanticSegmentationExample/configs/`` and have the
same keys, so every experiment is a change of values:

* ``config01.json``, **baseline**: a
  :class:`~UnetSemanticSegmentationExample.models.components.unet.UNet` from
  scratch (7.8 million parameters), binary cross-entropy (BCE), Adam with a
  constant learning rate, resize and horizontal flips only.
* ``config02.json``, **improved training**: the same U-Net with strong
  augmentation (random crops, flips, rotations, color changes), BCE + Dice
  loss, AdamW with weight decay, and a cosine learning-rate schedule.
* ``config03.json``, **fine-tuning**: a
  :class:`~UnetSemanticSegmentationExample.models.components.resnet_unet.ResNetUNet`,
  whose encoder is torchvision's ImageNet-pretrained ResNet-34 and whose
  decoder is new. The encoder is frozen for the first 3 epochs (its
  BatchNorm layers stay in eval mode), then trained with a 10 times smaller
  learning rate.

All three use early stopping on ``val_dice``
(:class:`~UnetSemanticSegmentationExample.callbacks.early_stopping.ResumableEarlyStopping`)
and mixed precision when a GPU is available.

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/UnetSemanticSegmentationExample>`_
explains every config setting, the results, and what is logged.

What is logged
--------------

* Per epoch, for ``train``, ``val``, and ``test``: ``<stage>_loss``,
  ``<stage>_dice`` (the F1 score of the polyp pixels), ``<stage>_iou``,
  ``<stage>_precision``, ``<stage>_recall``, and ``<stage>_pixel_acc``
  (:class:`~UnetSemanticSegmentationExample.models.lit_unet.LitUNet`), plus the
  learning rates and, for config03, ``encoder_frozen``.
* After every validation epoch, the same 8 validation images, each shown as
  four panels side by side: the image, the ground-truth mask, the predicted
  mask, and an error map (true positives green, false positives red, false
  negatives yellow). After training,
  for the best checkpoint, a per-image table (Dice, IoU, polyp area) and
  galleries of the worst and best images of the validation and test sets
  (:class:`~UnetSemanticSegmentationExample.callbacks.wandb_segmentation.LogSegmentation`).
* The config, the number of parameters, the training time, the best epoch,
  the model size, and the inference time per image.

Without ``WANDB_API_KEY`` in ``.env``, or if W&B cannot log in, the run is
logged offline, as in :doc:`LitWBTrainingBasicConvnet`.

Outputs
-------

Each run writes ``runs/UnetSemanticSegmentationExample/<config>/<timestamp>/``
with ``config.json``, ``hparams.json``, ``metrics.csv``, the best and last
checkpoints, and ``wandb/``. A row per run is appended to
``runs_summary.csv`` in the config folder, with the test metrics,
``train_minutes``, ``trainable_params``, ``best_epoch``, ``model_size_mb``, and
``inference_ms_per_image``.

:func:`~UnetSemanticSegmentationExample.models.loading.load_model` loads a
trained model back from its run; the network is rebuilt from the run's
config and all weights come from the checkpoint:

.. code-block:: python

   from UnetSemanticSegmentationExample import load_model

   model = load_model("config03")   # newest run of config03, best checkpoint
   masks = torch.sigmoid(model(images)) >= 0.5

To train a run further, pass ``resume_from`` to
:func:`~UnetSemanticSegmentationExample.main.main`:

.. code-block:: python

   main("config02.json", resume_from="config02")

On the command line: ``--resume-from config02``.

API
---

Entry point
~~~~~~~~~~~

.. automodule:: UnetSemanticSegmentationExample.main
   :members:

Callbacks
~~~~~~~~~

.. automodule:: UnetSemanticSegmentationExample.callbacks.wandb_segmentation
   :members:

.. automodule:: UnetSemanticSegmentationExample.callbacks.early_stopping
   :members:

Models
~~~~~~

.. automodule:: UnetSemanticSegmentationExample.models.lit_unet
   :members:

.. automodule:: UnetSemanticSegmentationExample.models.losses
   :members:

.. automodule:: UnetSemanticSegmentationExample.models.components.unet
   :members:

.. automodule:: UnetSemanticSegmentationExample.models.components.resnet_unet
   :members:

.. automodule:: UnetSemanticSegmentationExample.models.build
   :members:

.. automodule:: UnetSemanticSegmentationExample.models.loading
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: UnetSemanticSegmentationExample.dataloaders.kvasir_seg
   :members:

Utilities
~~~~~~~~~

.. automodule:: UnetSemanticSegmentationExample.utils.paths
   :members:

.. automodule:: UnetSemanticSegmentationExample.utils.experiment
   :members:
