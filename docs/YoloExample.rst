YoloExample
===========

Pedestrian detection with `YOLOv8 <https://docs.ultralytics.com/models/yolov8/>`_
on the `Penn-Fudan dataset <https://www.cis.upenn.edu/~jshi/ped_html/>`_
(170 street photos, 423 people). Training is done by the
`Ultralytics <https://docs.ultralytics.com/>`_ library; the project converts
the dataset to the YOLO label format, compares three strategies with a
zero-shot baseline, and logs a full evaluation report to W&B.

Running
-------

Run from the repository root, ``config01`` first (the others are compared
with it):

.. code-block:: bash

   python -m YoloExample --config config01.json   # zero-shot baseline, no training
   python -m YoloExample --config config02.json   # fine-tuned from COCO
   python -m YoloExample --config config03.json   # from scratch

or, from Python or a notebook started at the repository root:

.. code-block:: python

   from YoloExample import main

   main("config01.json")

The dataset (51 MB) is downloaded to ``data/PennFudanPed/`` and converted to
YOLO format in ``data/pennfudan_yolo/`` on the first run
(:func:`~YoloExample.dataloaders.pennfudan.prepare_pennfudan`). A GPU is
recommended.

The three configs
-----------------

The configs live in ``YoloExample/configs/``:

* ``config01.json``, **zero-shot**: ``"weights": "yolov8n.pt"``,
  ``"train": false``. The COCO-pretrained YOLOv8n is only evaluated, keeping
  only its ``person`` boxes (``classes=[0]``). This is the baseline.
* ``config02.json``, **fine-tuned**: the same COCO weights, trained 50 epochs
  on Penn-Fudan with SGD, a small learning rate, and the backbone frozen
  (``"ultralytics_args": {"freeze": 10}``).
* ``config03.json``, **from scratch**: ``"weights": "yolov8n.yaml"``, the same
  architecture with random weights, 150 epochs.

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/YoloExample>`_
explains the YOLO label format, the YOLOv8 structure and its output vector,
every metric (IoU, precision, recall, F1, AP, mAP50, mAP75, and mAP50-95), and
how to read the W&B report. The project's notebook walks through the network
layer by layer and decodes a prediction by hand.

What is logged
--------------

* Per epoch, against ``epoch``: the ``train/`` and ``val/`` box, class, and
  DFL losses, ``val/precision``, ``val/recall``, ``val/mAP50``,
  ``val/mAP50-95``, and the learning rates
  (:func:`~YoloExample.callbacks.wandb_logging.epoch_logger`).
* After training, for the best weights on the validation and test sets:
  precision, recall, F1, mAP50, mAP75, mAP50-95, and inference time in the run
  summary (:func:`~YoloExample.callbacks.wandb_logging.log_evaluation`),
  Ultralytics' PR, F1, precision, and recall curves and confusion matrix, AP at
  each IoU threshold (:func:`~YoloExample.callbacks.wandb_logging.log_ap_per_iou`),
  and a table of every image with its true and predicted boxes plus the
  hardest examples (:func:`~YoloExample.callbacks.wandb_logging.log_predictions`).
* A ``results`` table of the run, and a ``comparison`` table of the newest run
  of every config with its gain over the zero-shot baseline
  (:func:`~YoloExample.utils.experiment.compare_runs`).

Without ``WANDB_API_KEY`` in ``.env``, or if W&B cannot log in, the run is
logged offline, as in :doc:`LitWBTrainingBasicConvnet`.

Outputs
-------

Each run writes ``runs/YoloExample/<config>/<timestamp>/`` with
``config.json``, Ultralytics' files (``weights/best.pt`` and ``last.pt``,
``results.csv``, ``args.yaml``, training plots), ``eval_val/`` and
``eval_test/`` plots, and ``wandb/``. A row per run is appended to
``runs_summary.csv`` in the config folder.

:func:`~YoloExample.models.loading.load_model` loads a model back from its
run, as an Ultralytics ``YOLO`` model:

.. code-block:: python

   from YoloExample import load_model

   model = load_model("config02")   # newest run of config02, weights/best.pt
   results = model.predict("street.jpg", conf=0.25)

To finish an interrupted training run, pass ``resume_from`` to
:func:`~YoloExample.main.main`:

.. code-block:: python

   main("config02.json", resume_from="config02")

On the command line: ``--resume-from config02``.

API
---

Entry point
~~~~~~~~~~~

.. automodule:: YoloExample.main
   :members:

Callbacks
~~~~~~~~~

.. automodule:: YoloExample.callbacks.wandb_logging
   :members:

Models
~~~~~~

.. automodule:: YoloExample.models.loading
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: YoloExample.dataloaders.pennfudan
   :members:

Utilities
~~~~~~~~~

.. automodule:: YoloExample.utils.paths
   :members:

.. automodule:: YoloExample.utils.experiment
   :members:
