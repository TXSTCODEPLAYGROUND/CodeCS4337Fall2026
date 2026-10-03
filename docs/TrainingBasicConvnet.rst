TrainingBasicConvnet
====================

Train a small convolutional network on Fashion-MNIST.

Run from the repository root:

.. code-block:: bash

   python -m TrainingBasicConvnet --config config01.json

or, from inside the project folder:

.. code-block:: bash

   cd TrainingBasicConvnet
   python main.py --config config01.json

or, from Python or a notebook started at the repository root:

.. code-block:: python

   from TrainingBasicConvnet import main

   main("config01.json")

``--config`` accepts a name (``config01``), a file name (``config01.json``),
or a path (``configs/config01.json``). Hyperparameters live in
``TrainingBasicConvnet/configs/``; the Python code does not need to change to
try a new setting.

``DATA_DIR`` and ``OUTPUT_DIR`` are read from the ``.env`` file at the
repository root. Relative paths are resolved against that root, so datasets
and results land in the same place no matter which directory you run from.
Each run writes a timestamped folder under
``runs/TrainingBasicConvnet/<config>/`` containing the config snapshot,
checkpoints, ``history.json``, and a results file. A row is also appended to
``runs_summary.csv`` for that config.

Every run trains from scratch. To load a trained model back,
:func:`~TrainingBasicConvnet.models.loading.load_model` rebuilds the network
from the config stored in the checkpoint and loads its weights:

.. code-block:: python

   from TrainingBasicConvnet import load_model

   model = load_model("config01")                 # newest run, best checkpoint
   model = load_model("config01", which="last")   # its last epoch

To train a run further instead, pass ``resume_from`` to :func:`~TrainingBasicConvnet.main.main`:
it continues from the run's last checkpoint (weights, optimizer, and epoch
count) for the config's ``"epochs"`` more epochs, into a new run folder.

.. code-block:: python

   main("config01.json", resume_from="config01")

On the command line: ``--resume-from config01``.

.. toctree::
   :maxdepth: 2
   :caption: API

   api/main
   api/models
   api/dataloaders
   api/trainers
   api/utils
