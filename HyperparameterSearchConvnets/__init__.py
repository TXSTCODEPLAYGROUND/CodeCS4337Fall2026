"""Search the architecture and training settings of a ConvNet with Optuna, then train the best.

Example:

.. code-block:: python

    from HyperparameterSearchConvnets import main
    from HyperparameterSearchConvnets.search import search

    search("search01.json")      # writes configs/search01_best.json
    main("search01_best.json")   # trains and tests the best settings
"""

from .main import main
from .models import load_model

__all__ = ["load_model", "main"]
