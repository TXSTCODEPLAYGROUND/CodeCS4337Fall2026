TrainingBasicConvnet
====================

Train a small convolutional network on Fashion-MNIST.

.. code-block:: python

   from TrainingBasicConvnet import main

   main("config01")

The same experiment can be started from the terminal, from the repository
root:

.. code-block:: bash

   python -m TrainingBasicConvnet --config config01

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

.. toctree::
   :maxdepth: 2
   :caption: API

   api/main
   api/models
   api/dataloaders
   api/trainers
   api/utils
