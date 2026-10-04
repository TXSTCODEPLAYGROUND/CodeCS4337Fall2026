"""The Oxford Flowers-102 LightningDataModule and the dataset's constants."""

import lightning as L
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
"""Mean of each color channel over ImageNet, used to normalize the images.

The pretrained ResNet was trained on images normalized this way, so new
images must be normalized the same way, not with Flowers-102's own statistics.
"""
IMAGENET_STD = (0.229, 0.224, 0.225)
"""Standard deviation of each color channel over ImageNet."""
FLOWERS102_CLASSES = datasets.Flowers102.classes
"""The 102 flower species, in label order: ``"pink primrose"``, ..."""
IMAGE_SIZE = 224
"""Side of the square images fed to the network, as in ImageNet training."""


class Flowers102DataModule(L.LightningDataModule):
    """Oxford Flowers-102 train, validation, and test dataloaders.

    Uses the dataset's official split: 1,020 training images (10 per class),
    1,020 validation images (10 per class), and 6,149 test images. The images
    are downloaded (about 330 MB) to ``data_dir``, the shared ``data/`` folder
    at the repo root by default.

    Training images are augmented: a random crop of 8 % to 100 % of the image,
    resized to 224 x 224, and a random horizontal flip, so every epoch sees
    slightly different versions of the 1,020 images. Validation and test images
    get ImageNet's evaluation preprocessing: resized to 256 on the short side,
    then the center 224 x 224 crop. All images are normalized with the ImageNet
    mean and standard deviation.
    """

    def __init__(
        self, data_dir: str, batch_size: int = 64, num_workers: int = 2
    ) -> None:
        """Store the settings; nothing is downloaded or loaded yet.

        Args:
            data_dir: Directory where the dataset is stored or will be downloaded.
            batch_size: Number of images per batch.
            num_workers: Number of subprocesses used for loading and augmenting
                the images.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)
        normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        self.train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def prepare_data(self) -> None:
        """Download the dataset (Lightning calls this once, before ``setup``)."""
        datasets.Flowers102(self.hparams.data_dir, split="train", download=True)

    def setup(self, stage: str) -> None:
        """Create the three datasets of the official split."""
        root = self.hparams.data_dir
        self.train_set = datasets.Flowers102(
            root, split="train", transform=self.train_transform
        )
        self.val_set = datasets.Flowers102(
            root, split="val", transform=self.eval_transform
        )
        self.test_set = datasets.Flowers102(
            root, split="test", transform=self.eval_transform
        )

    def train_dataloader(self) -> DataLoader:
        """Shuffled, augmented training batches."""
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
