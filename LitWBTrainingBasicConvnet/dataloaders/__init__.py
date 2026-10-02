"""LightningDataModules: download, split, and batch the datasets."""

from .fashion_mnist import (
    FASHION_MNIST_CLASSES,
    FASHION_MNIST_MEAN,
    FASHION_MNIST_STD,
    FashionMNISTDataModule,
)

__all__ = [
    "FASHION_MNIST_CLASSES",
    "FASHION_MNIST_MEAN",
    "FASHION_MNIST_STD",
    "FashionMNISTDataModule",
]
