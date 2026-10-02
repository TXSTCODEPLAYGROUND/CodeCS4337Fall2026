"""The Fashion-MNIST LightningDataModule and the dataset's constants."""

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

FASHION_MNIST_MEAN = (0.2860,)
"""Mean pixel value of the training images, used to normalize them."""
FASHION_MNIST_STD = (0.3530,)
"""Standard deviation of the training pixels, used to normalize them."""
FASHION_MNIST_CLASSES = datasets.FashionMNIST.classes
"""The 10 class names, in label order: ``"T-shirt/top"``, ``"Trouser"``, ..."""


class FashionMNISTDataModule(L.LightningDataModule):
    """Fashion-MNIST train, validation, and test dataloaders.

    The dataset files live in ``data_dir`` (the shared ``data/`` folder at the
    repo root by default), not in this package. The official training set is
    split into train and validation subsets with a seeded generator, so the
    split is the same in every run.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        val_split: float = 0.1,
        num_workers: int = 2,
        seed: int = 42,
    ) -> None:
        """Store the settings; nothing is downloaded or loaded yet.

        Args:
            data_dir: Directory where the dataset is stored or will be downloaded.
            batch_size: Number of samples per batch.
            val_split: Fraction of the training set used for validation.
            num_workers: Number of subprocesses used for data loading.
            seed: Random seed for the train/validation split.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
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

    def predict_dataloader(self) -> DataLoader:
        """The test set again, used by ``trainer.predict`` for example predictions."""
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
