LitWBTrainingBasicConvnet
=========================

The same experiment as :doc:`LitTrainingBasicConvnet` (the same network,
LightningModule, DataModule, and metrics), tracked with
`Weights & Biases <https://wandb.ai>`__ (W&B) through Lightning's
``WandbLogger``. W&B charts and tables replace the plotting code, so there is
no ``utils/plots.py``. Lightning + W&B is the approach you must follow when training your own
models; :doc:`LitWBTrainingBasicNeuralNetwork` is the same setup with a
fully connected network.

Shared code: colabutilscs4337
-----------------------------

Setup, paths, configs, run folders, resuming, and the W&B connection used to be copied
into every project. They now come from the course library
`colabutilscs4337 <https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html>`__ (listed in
``requirements.txt``), so this project keeps only the data, the model, the callback
that logs predictions, and ``main()``:

* ``utils/paths.py`` is one line, ``PROJECT = cu.project.Project(...)``: repo, config,
  data, and run folders, plus finding checkpoints (``find_checkpoint``,
  ``find_resume_checkpoint``, ``run_config``).
* ``main()`` starts with ``cu.lit.start_run`` (command line, config, resume, run folder,
  ``CSVLogger``, checkpoint callbacks) and ends with ``run.save`` and
  ``run.append_summary`` (``config.json``, ``hparams.json``, ``runs_summary.csv``).
* ``cu.lit.connect_wandb`` starts the W&B run, online or offline.
* ``ResumableEarlyStopping`` is ``cu.lit.ResumableEarlyStopping``.

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
the W&B project the runs are sent to. Its ``"training"`` section also turns on
early stopping (``"early_stopping": true``, ``"patience": 3``): training stops
when ``val_acc`` has not improved for ``patience`` epochs, so ``"epochs"`` is
the most a run can take
(:class:`~colabutilscs4337.lit.ResumableEarlyStopping`).

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/LitWBTrainingBasicConvnet>`_
explains how to get a W&B key and how to build the dashboard.

W&B key and offline mode
------------------------

Copy your API key from `wandb.ai/authorize <https://wandb.ai/authorize>`_
into ``.env`` as ``WANDB_API_KEY``. Without it, or if W&B cannot log in (a wrong key, no
internet), the run is logged offline in the run folder and prints the ``wandb sync``
command that uploads it later.

In a Colab notebook, add the key as a Colab Secret and run the project notebook's
*Load API keys* cell (``cu.env.load_secrets("WANDB_API_KEY", persist_to=...)``). In a
Colab CLI (SSH) session, Colab Secrets are not available: put the key in a text file on
your Google Drive and pass its path as ``env_source`` to
:func:`colabutilscs4337.project.setup_colab` (the project README shows the commands). The
run stops with an error if that file is missing.

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

:func:`~LitWBTrainingBasicConvnet.models.loading.load_model` loads a trained
model back from its run:

.. code-block:: python

   from LitWBTrainingBasicConvnet import load_model

   model = load_model("config01")   # newest run of config01, best checkpoint

To train a run further instead, pass ``resume_from`` to :func:`~LitWBTrainingBasicConvnet.main.main`:
it continues from the run's last checkpoint (weights, optimizer, and epoch
count) for the config's ``"epochs"`` more epochs, into a new run folder.

.. code-block:: python

   main("config01.json", resume_from="config01")

On the command line: ``--resume-from config01``.

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

``ResumableEarlyStopping`` is in the library: see :class:`colabutilscs4337.lit.ResumableEarlyStopping`.

Models
~~~~~~

.. automodule:: LitWBTrainingBasicConvnet.models.lit_convnet
   :members:

.. automodule:: LitWBTrainingBasicConvnet.models.components.convnet
   :members:

.. automodule:: LitWBTrainingBasicConvnet.models.loading
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: LitWBTrainingBasicConvnet.dataloaders.fashion_mnist
   :members:

Utilities
~~~~~~~~~

.. automodule:: LitWBTrainingBasicConvnet.utils.paths
   :members:

The rest (``runs_summary.csv``, run folders, checkpoint lookup) is in
`colabutilscs4337 <https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html>`__.
