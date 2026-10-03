"""Search the hyperparameters of a fully connected network with Optuna, then train the best.

Example:

.. code-block:: python

    from LitWBHSTrainingBasicNeuralNetwork import main
    from LitWBHSTrainingBasicNeuralNetwork.search import search

    search("search01.json")      # writes configs/search01_best.json
    main("search01_best.json")   # trains and tests the best settings
"""

from .main import main
from .models import load_model

__all__ = ["load_model", "main"]
