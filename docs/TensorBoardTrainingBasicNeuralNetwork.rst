TensorBoardTrainingBasicNeuralNetwork
=====================================

:doc:`TrainingBasicNeuralNetwork` with `TensorBoard
<https://www.tensorflow.org/tensorboard>`_ logging: the same fully connected
network and plain-PyTorch training loop, plus a
``torch.utils.tensorboard.SummaryWriter`` that records every run for the
TensorBoard dashboard.

.. note::

   This project is for learning TensorBoard. To train your own models, follow
   the Lightning + W&B approach of :doc:`LitWBTrainingBasicNeuralNetwork` and
   :doc:`LitWBTrainingBasicConvnet`.

Run from the repository root, then open TensorBoard on the project's runs:

.. code-block:: bash

   python -m TensorBoardTrainingBasicNeuralNetwork --config config01.json
   tensorboard --logdir runs/TensorBoardTrainingBasicNeuralNetwork

and open http://localhost:6006. In a notebook (Colab or Jupyter), the
project notebook shows TensorBoard in a cell's output:

.. code-block:: python

   %load_ext tensorboard
   %tensorboard --logdir "{logdir}"

Each run writes its event files into its own run folder,
``runs/TensorBoardTrainingBasicNeuralNetwork/<config>/<timestamp>/``, so
TensorBoard lists every run as ``<config>/<timestamp>``. Logged per run:

- **Scalars**: ``loss/train``, ``loss/val``, ``accuracy/train``,
  ``accuracy/val``, and ``learning_rate`` per epoch; ``batch/train_loss``
  every ``log_every_n_steps`` batches; ``test/loss`` and ``test/accuracy``.
- **Custom Scalars**: train and val in one chart per metric.
- **Images**: input samples, test predictions, and a confusion matrix.
- **Graphs**: the network.
- **Histograms**: weights and gradients per epoch (``"histograms": false``
  turns them off).
- **Text**: the config. **HParams**: the run's settings next to its scores.

The config's optional ``"tensorboard"`` section sets ``log_every_n_steps``
(default 50) and ``histograms`` (default ``true``).

Loading and resuming work as in :doc:`TrainingBasicNeuralNetwork`:

.. code-block:: python

   from TensorBoardTrainingBasicNeuralNetwork import load_model, main

   model = load_model("config01")                   # newest run, best checkpoint
   main("config01.json", resume_from="config01")    # train it further

The `project README <https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026/tree/main/TensorBoardTrainingBasicNeuralNetwork>`_
describes each TensorBoard tab and what to try.

API
---

Entry point
~~~~~~~~~~~

.. automodule:: TensorBoardTrainingBasicNeuralNetwork.main
   :members:

Models
~~~~~~

.. automodule:: TensorBoardTrainingBasicNeuralNetwork.models.mlp
   :members:

.. automodule:: TensorBoardTrainingBasicNeuralNetwork.models.loading
   :members:

Dataloaders
~~~~~~~~~~~

.. automodule:: TensorBoardTrainingBasicNeuralNetwork.dataloaders.fashion_mnist
   :members:

Trainers
~~~~~~~~

.. automodule:: TensorBoardTrainingBasicNeuralNetwork.trainers.trainer
   :members:

Utilities
~~~~~~~~~

.. automodule:: TensorBoardTrainingBasicNeuralNetwork.utils.tensorboard_logging
   :members:

.. automodule:: TensorBoardTrainingBasicNeuralNetwork.utils.paths
   :members:

.. automodule:: TensorBoardTrainingBasicNeuralNetwork.utils.reproducibility
   :members:

.. automodule:: TensorBoardTrainingBasicNeuralNetwork.utils.experiment
   :members:
