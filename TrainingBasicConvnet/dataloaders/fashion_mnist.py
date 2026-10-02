"""Fashion-MNIST: download, normalize, split, and batch the images."""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test dataloaders for Fashion-MNIST.

    The official training set is split into train and validation subsets using
    a seeded generator so the split is reproducible across runs. Images are
    converted to tensors and normalized with the Fashion-MNIST mean and std.

    Args:
        data_dir: Directory where the dataset is stored or will be downloaded.
        batch_size: Number of samples per batch.
        val_split: Fraction of the training set reserved for validation.
        num_workers: Number of subprocesses used for data loading.
        seed: Random seed for the train/validation split.

    Returns:
        A tuple ``(train_loader, val_loader, test_loader)``. Only the train
        loader shuffles its data.
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(FASHION_MNIST_MEAN, FASHION_MNIST_STD),
        ]
    )

    full_train = datasets.FashionMNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_set = datasets.FashionMNIST(
        data_dir, train=False, download=True, transform=transform
    )

    val_size = int(len(full_train) * val_split)
    train_size = len(full_train) - val_size
    train_set, val_set = random_split(
        full_train,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader
