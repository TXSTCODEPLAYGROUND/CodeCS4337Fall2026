"""Train a basic fully connected neural network on Fashion-MNIST.

Example:

.. code-block:: python

    from TrainingBasicNeuralNetwork import main
    main("config01.json")
"""

from .main import main
from .models import load_model

__all__ = ["load_model", "main"]
