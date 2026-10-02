"""Train a basic ConvNet on Fashion-MNIST with PyTorch Lightning, tracked in W&B.

Example:

.. code-block:: python

    from LitWBTrainingBasicConvnet import main
    main("config01.json")
"""

from .main import main

__all__ = ["main"]
