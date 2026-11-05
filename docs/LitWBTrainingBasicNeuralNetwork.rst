LitWBTrainingBasicNeuralNetwork
===============================

The fully connected network of :doc:`LitTrainingBasicNeuralNetwork`, tracked
with `Weights & Biases <https://wandb.ai>`__ (W&B) exactly like
:doc:`LitWBTrainingBasicConvnet`; only the network differs. Lightning + W&B is
the approach you must follow when training your own models.

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
W&B key comes from ``WANDB_API_KEY`` in ``.env``. Without it, or if W&B cannot log
in, the run is logged offline and can be uploaded later with ``wandb sync``.

In a Colab notebook, add the key as a Colab Secret and run the project notebook's
*Load API keys* cell (``cu.env.load_secrets("WANDB_API_KEY", persist_to=...)``). In a
Colab CLI (SSH) session, Colab Secrets are not available: put the key in a text file on
your Google Drive and pass its path as ``env_source`` to
:func:`colabutilscs4337.project.setup_colab` (the project README shows the commands). The
run stops with an error if that file is missing.

The number of parameters is sent to W&B as ``num_params`` in the run's config,
so runs with different ``hidden_sizes`` can be compared on wandb.ai. Locally,
``runs_summary.csv`` records ``hidden_sizes`` and ``num_params``.

:func:`~LitWBTrainingBasicNeuralNetwork.models.loading.load_model` loads a
trained model back, rebuilt with the run's ``hidden_sizes`` and ``dropout``:

.. code-block:: python

   from LitWBTrainingBasicNeuralNetwork import load_model

   model = load_model("config01")   # newest run of config01, best checkpoint

To train a run further instead, pass ``resume_from`` to :func:`~LitWBTrainingBasicNeuralNetwork.main.main`:
it continues from the run's last checkpoint (weights, optimizer, and epoch
count) for the config's ``"epochs"`` more epochs, into a new run folder.

.. code-block:: python

   main("config01.json", resume_from="config01")

On the command line: ``--resume-from config01``.

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/LitWBTrainingBasicNeuralNetwork>`_
shows how to compare network sizes in W&B.

Shared code: colabutilscs4337
-----------------------------

Setup, paths, configs, run folders, resuming, and the W&B connection used to be copied
into every project. They now come from the course library
`colabutilscs4337 <https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html>`__ (listed in
``requirements.txt``), so this project keeps only what is specific to the experiment:

* ``utils/paths.py`` is one line, ``PROJECT = cu.project.Project(...)``: repo, config,
  data, and run folders, plus finding checkpoints.
* ``main()`` starts with ``cu.lit.start_run`` (command line, config, resume, run folder,
  ``CSVLogger``, checkpoint callbacks) and ends with ``run.save`` and
  ``run.append_summary`` (``config.json``, ``hparams.json``, ``runs_summary.csv``).
* ``cu.lit.connect_wandb`` starts the W&B run, online or offline.

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

.. automodule:: LitWBTrainingBasicNeuralNetwork.models.loading
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: LitWBTrainingBasicNeuralNetwork.dataloaders.fashion_mnist
   :members:

Utilities
~~~~~~~~~

.. automodule:: LitWBTrainingBasicNeuralNetwork.utils.paths
   :members:

The rest (``runs_summary.csv``, run folders, checkpoint lookup) is in
`colabutilscs4337 <https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html>`__.
