"""Binary semantic segmentation of polyps (Kvasir-SEG) with a U-Net, PyTorch Lightning, and W&B.

Example:

.. code-block:: python

    from UnetSemanticSegmentationExample import main
    main("config01.json")
"""

from .main import main
from .models import load_model

__all__ = ["load_model", "main"]
