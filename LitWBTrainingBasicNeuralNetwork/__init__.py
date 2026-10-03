"""Train a basic fully connected network on Fashion-MNIST with Lightning, tracked in W&B.

Example:

.. code-block:: python

    from LitWBTrainingBasicNeuralNetwork import main
    main("config01.json")
"""

from .main import main
from .models import load_model

__all__ = ["load_model", "main"]
