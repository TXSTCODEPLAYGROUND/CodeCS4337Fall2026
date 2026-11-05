"""Pothole detection with YOLOv8 transfer learning, tracked in W&B.

Example:

.. code-block:: python

    from LitWBTransferLearningYolo import main
    main("config01.json")
"""

from .main import main
from .models import load_model

__all__ = ["load_model", "main"]
