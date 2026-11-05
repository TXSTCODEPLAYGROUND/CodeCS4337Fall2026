LitWBTransferLearningYolo
==========================

Transfer learning for pothole detection with `YOLOv8 <https://docs.ultralytics.com/models/yolov8/>`_
on the `IVCNZ pothole dataset <https://github.com/jaygala24/pothole-detection>`_
(1,243 road photos, 4,076 boxes, one class). Training is done by
`Ultralytics <https://docs.ultralytics.com/>`_. The project uses the run
folders, W&B (Weights & Biases) logging, and resume behavior of
:doc:`YoloExample`, and the same three strategies as
:doc:`LitWBTransferLearningResnet`: a frozen pretrained backbone, fine-tuning
every layer, and training from scratch. COCO (Common Objects in Context) has
no pothole class, so there is no zero-shot baseline; the from-scratch run is
the baseline.

Running
-------

Run from the repository root:

.. code-block:: bash

   python -m LitWBTransferLearningYolo --config config01.json   # frozen backbone
   python -m LitWBTransferLearningYolo --config config02.json   # fine-tune every layer
   python -m LitWBTransferLearningYolo --config config03.json   # from scratch

or, from Python or a notebook started at the repository root:

.. code-block:: python

   from LitWBTransferLearningYolo import main

   main("config01.json")

The dataset (about 166 MB) is downloaded to ``data/pothole_ivcnz/`` and laid
out for Ultralytics in ``data/pothole_yolo/`` on the first run
(:func:`~LitWBTransferLearningYolo.dataloaders.pothole.prepare_pothole`).
In Colab, ``OUTPUT_DIR`` is on Google Drive (``.envcolab``), so each run
folder survives a runtime reset. A GPU is recommended.

The three configs
-----------------

The configs live in ``LitWBTransferLearningYolo/configs/``:

* ``config01.json``, **frozen backbone**: ``"weights": "yolov8n.pt"``,
  ``"freeze": 10`` (layers 0 to 9 stay at the COCO weights), SGD, ``lr0``
  0.01, 30 epochs.
* ``config02.json``, **fine-tuned**: the same COCO weights, nothing frozen,
  SGD, ``lr0`` 0.001, 30 epochs.
* ``config03.json``, **from scratch**: ``"weights": "yolov8n.yaml"``, random
  weights, 50 epochs. The comparison table reports
  ``gain_vs_from_scratch`` against this run.

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/LitWBTransferLearningYolo>`_
has the split counts, the metric definitions, and how to read the W&B report.
:doc:`YoloExample` explains the YOLO label format and the YOLOv8 output vector.

What is logged
--------------

* Per epoch, against ``epoch``: the ``train/`` and ``val/`` box, class, and
  DFL losses, ``val/precision``, ``val/recall``, ``val/mAP50``,
  ``val/mAP50-95``, and the learning rates
  (:func:`~LitWBTransferLearningYolo.callbacks.wandb_logging.epoch_logger`).
  The first epoch stores ``trainable_params``.
* After training, for the best weights on the validation and test sets:
  precision, recall, F1, mAP50, mAP75, mAP50-95, and inference time
  (:func:`~LitWBTransferLearningYolo.callbacks.wandb_logging.log_evaluation`),
  Ultralytics' curves and confusion matrix, and a table of every image with
  its true and predicted boxes
  (:func:`~LitWBTransferLearningYolo.callbacks.wandb_logging.log_predictions`).
* A ``results`` table of the run, and a ``comparison`` table of the newest run
  of every config
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

Each run writes ``runs/LitWBTransferLearningYolo/<config>/<timestamp>/`` with
``config.json``, ``hparams.json`` (including ``resumed_from``), Ultralytics'
files (``weights/best.pt`` and ``last.pt``, ``results.csv``, ``args.yaml``,
training plots), ``eval_val/`` and ``eval_test/``, and ``wandb/``. A row per
finished run is appended to ``runs_summary.csv`` in the config folder.

:func:`~LitWBTransferLearningYolo.models.loading.load_model` loads a model
back from its run, as an Ultralytics ``YOLO`` model:

.. code-block:: python

   from LitWBTransferLearningYolo import load_model

   model = load_model("config02")   # newest run of config02, weights/best.pt
   results = model.predict("road.jpg", conf=0.25)

To continue a training run, pass ``resume_from`` to
:func:`~LitWBTransferLearningYolo.main.main`:

.. code-block:: python

   main("config02.json", resume_from="config02")

On the command line: ``--resume-from config02``. An interrupted run is
finished in its own folder from ``weights/last.pt``; a finished run is trained
the config's ``"epochs"`` more epochs from its ``weights/best.pt``, in a new
run folder (:func:`~LitWBTransferLearningYolo.models.loading.find_resume_weights`).

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

.. automodule:: LitWBTransferLearningYolo.main
   :members:

Callbacks
~~~~~~~~~

.. automodule:: LitWBTransferLearningYolo.callbacks.wandb_logging
   :members:

Models
~~~~~~

.. automodule:: LitWBTransferLearningYolo.models.loading
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: LitWBTransferLearningYolo.dataloaders.pothole
   :members:

Utilities
~~~~~~~~~

.. automodule:: LitWBTransferLearningYolo.utils.paths
   :members:

The rest (``runs_summary.csv``, run folders, the comparison of configs) is in
`colabutilscs4337 <https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html>`__.
