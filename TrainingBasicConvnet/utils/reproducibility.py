"""Seeding for reproducible runs, and choosing the device to train on."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU and CUDA) RNGs for reproducibility.

    Args:
        seed: The seed value to apply to every random number generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    """Resolve the device to train on.

    Args:
        requested: ``"auto"`` to pick the best available device (CUDA, then
            Apple Silicon GPU via MPS, then CPU), or an explicit device string
            such as ``"cpu"``, ``"cuda:0"``, or ``"mps"``.

    Returns:
        The resolved ``torch.device``.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)
