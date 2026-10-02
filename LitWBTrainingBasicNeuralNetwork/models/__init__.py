"""LightningModules: how a network is trained and evaluated.

The networks themselves are in :mod:`~LitWBTrainingBasicNeuralNetwork.models.components`.
"""

from .components import MLP
from .lit_mlp import LitMLP

__all__ = ["MLP", "LitMLP"]
