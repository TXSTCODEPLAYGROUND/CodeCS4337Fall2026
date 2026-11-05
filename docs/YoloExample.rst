YoloExample
===========

Pedestrian detection with `YOLOv8 <https://docs.ultralytics.com/models/yolov8/>`_
(YOLO, "You Only Look Once", version 8) on the `Penn-Fudan dataset <https://www.cis.upenn.edu/~jshi/ped_html/>`_
(170 street photos, 423 people). Training is done by the
`Ultralytics <https://docs.ultralytics.com/>`_ library; the project converts
the dataset to the YOLO label format, compares three strategies with a
zero-shot baseline, and logs a full evaluation report to W&B (Weights &
Biases).

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
  ``"train": false``. YOLOv8n pretrained on COCO (Common Objects in Context,
  80 classes, one of which is ``person``) is only evaluated, keeping
  only its ``person`` boxes (``classes=[0]``). This is the baseline.
* ``config02.json``, **fine-tuned**: the same COCO weights, trained 50 epochs
  on Penn-Fudan with SGD (stochastic gradient descent), a small learning rate,
  and the backbone frozen
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
  (:func:`colabutilscs4337.project.compare_runs`).

Without ``WANDB_API_KEY`` in ``.env``, or if W&B cannot log in, the run is
logged offline, as in :doc:`LitWBTrainingBasicConvnet`.

In a Colab notebook, add the key as a Colab Secret and run the project notebook's
*Load API keys* cell (``cu.env.load_secrets("WANDB_API_KEY", persist_to=...)``). In a
Colab CLI (SSH) session, Colab Secrets are not available: put the key in a text file on
your Google Drive and pass its path as ``env_source`` to
:func:`colabutilscs4337.project.setup_colab` (the project README shows the commands). The
run stops with an error if that file is missing.

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

To continue a training run, pass ``resume_from`` to
:func:`~YoloExample.main.main`:

.. code-block:: python

   main("config02.json", resume_from="config02")

On the command line: ``--resume-from config02``. An interrupted run is
finished in its own folder from ``weights/last.pt``; a finished run is trained
the config's ``"epochs"`` more epochs from its ``weights/best.pt``, in a new
run folder (:func:`~YoloExample.models.loading.find_resume_weights`).

Shared code: colabutilscs4337
-----------------------------

Setup, paths, configs, run folders, resuming, and the W&B connection used to be copied
into every project. They now come from the course library
`colabutilscs4337 <https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html>`__ (listed in
``requirements.txt``), so this project keeps only what is specific to the experiment:

* ``utils/paths.py`` is one line, ``PROJECT = cu.project.Project(...)``: repo, config,
  data, and run folders.
* ``main()`` uses ``cu.lit.parse_cli`` (command line), ``PROJECT.read_config`` and
  ``new_run_dir``, ``cu.lit.connect_wandb_run`` (the W&B run, online or offline), and
  ``cu.project.append_run_summary`` and ``compare_runs`` (``runs_summary.csv`` and the
  comparison table).
* Finding ``weights/best.pt`` and ``last.pt`` stays in ``models/loading.py``, because
  Ultralytics keeps them in a ``weights/`` folder; it uses ``PROJECT.resolve_run``.

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

The rest (``runs_summary.csv``, run folders, the comparison of configs) is in
`colabutilscs4337 <https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html>`__.
