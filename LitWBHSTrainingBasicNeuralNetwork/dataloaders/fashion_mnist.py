"""The Fashion-MNIST LightningDataModule and the dataset's constants."""

import os

import lightning as L
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

FASHION_MNIST_MEAN = (0.2860,)
"""Mean pixel value of the training images, used to normalize them."""
FASHION_MNIST_STD = (0.3530,)
"""Standard deviation of the training pixels, used to normalize them."""
FASHION_MNIST_CLASSES = datasets.FashionMNIST.classes
"""The 10 class names, in label order: ``"T-shirt/top"``, ``"Trouser"``, ..."""
MAX_AUTO_WORKERS = 8
"""Upper limit for ``num_workers="auto"``: more workers rarely speed up this
small dataset, and every trial of a search starts its workers again."""


def resolve_num_workers(num_workers: int | str) -> int:
    """Turn ``"auto"`` into a number of workers; return numbers unchanged.

    Args:
        num_workers: A number of workers, or ``"auto"`` for the CPU cores this
            process may use, at most :data:`MAX_AUTO_WORKERS` (e.g. 8 on a
            24-core machine, 2 in Colab).

    Returns:
        The number of data-loading worker processes.
    """
    if num_workers != "auto":
        return int(num_workers)
    if hasattr(os, "sched_getaffinity"):
        cores = len(os.sched_getaffinity(0))
    else:
        cores = os.cpu_count() or 1
    return min(cores, MAX_AUTO_WORKERS)


class FashionMNISTDataModule(L.LightningDataModule):
    """Fashion-MNIST train, validation, and test dataloaders.

    The dataset files live in ``data_dir`` (the shared ``data/`` folder at the
    repo root by default), not in this package. The official training set is
    split into train and validation subsets with a seeded generator, so the
    split is the same in every run. Training can use only a fraction of the
    training split (``train_fraction``); the validation and test sets are
    always complete, so scores stay comparable.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        val_split: float = 0.1,
        num_workers: int | str = "auto",
        train_fraction: float = 1.0,
        seed: int = 42,
    ) -> None:
        """Store the settings; nothing is downloaded or loaded yet.

        Args:
            data_dir: Directory where the dataset is stored or will be downloaded.
            batch_size: Number of samples per batch.
            val_split: Fraction of the training set used for validation.
            num_workers: Number of subprocesses used for data loading, or
                ``"auto"`` (see :func:`resolve_num_workers`).
            train_fraction: Fraction of the training split to train on, e.g.
                ``0.25`` for a 4 times faster epoch. The subset is stratified
                (the same fraction of every class, with scikit-learn's
                ``train_test_split``) and drawn with the seed, so it is the
                same in every run.
            seed: Random seed for the train/validation split and the subset.

        Raises:
            ValueError: If ``train_fraction`` is not in ``(0, 1]``.
        """
        super().__init__()
        if not 0 < train_fraction <= 1:
            raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")
        self.save_hyperparameters(logger=False)
        # Store the actual number, so hparams.json records what was used.
        self.hparams.num_workers = resolve_num_workers(num_workers)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
            ]
        )

    def prepare_data(self) -> None:
        """Download the dataset (Lightning calls this once, before ``setup``)."""
        datasets.FashionMNIST(self.hparams.data_dir, train=True, download=True)
        datasets.FashionMNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: str) -> None:
        """Create the train/validation split and the test set."""
        full_train = datasets.FashionMNIST(
            self.hparams.data_dir, train=True, transform=self.transform
        )
        val_size = int(len(full_train) * self.hparams.val_split)
        self.train_set, self.val_set = random_split(
            full_train,
            [len(full_train) - val_size, val_size],
            generator=torch.Generator().manual_seed(self.hparams.seed),
        )
        if self.hparams.train_fraction < 1:
            # stratify: every class keeps the same fraction of its images, where a
            # plain random sample could over- or under-represent a class by chance.
            keep, _ = train_test_split(
                range(len(self.train_set)),
                train_size=self.hparams.train_fraction,
                stratify=full_train.targets[self.train_set.indices],
                random_state=self.hparams.seed,
            )
            self.train_set = Subset(self.train_set, keep)
        self.test_set = datasets.FashionMNIST(
            self.hparams.data_dir, train=False, transform=self.transform
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled training batches."""
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Validation batches, in a fixed order."""
        return self._loader(self.val_set)

    def test_dataloader(self) -> DataLoader:
        """Test batches, in a fixed order."""
        return self._loader(self.test_set)

    def _loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Build a DataLoader with the batch size and workers from the config.

        Args:
            dataset: The dataset to batch.
            shuffle: Whether to reshuffle the samples every epoch.

        Returns:
            The DataLoader.
        """
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
        )
