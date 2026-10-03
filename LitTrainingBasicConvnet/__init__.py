"""Train a basic ConvNet on Fashion-MNIST with PyTorch Lightning.

Example:

.. code-block:: python

    from LitTrainingBasicConvnet import main
    main("config01.json")
"""

from .main import main
from .models import load_model

__all__ = ["load_model", "main"]
