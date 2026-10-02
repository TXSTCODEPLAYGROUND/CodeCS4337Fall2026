"""LightningModules: how a network is trained and evaluated.

The networks themselves are in :mod:`~LitTrainingBasicConvnet.models.components`.
"""

from .components import ConvNet
from .lit_convnet import LitConvNet

__all__ = ["ConvNet", "LitConvNet"]
