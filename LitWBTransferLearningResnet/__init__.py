"""Transfer learning with ResNet-18 on Oxford Flowers-102, with PyTorch Lightning and W&B.

Example:

.. code-block:: python

    from LitWBTransferLearningResnet import main
    main("config01.json")
"""

from .main import main
from .models import load_model

__all__ = ["load_model", "main"]
