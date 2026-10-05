"""Kvasir-SEG: download, binary masks, paired augmentation, and the LightningDataModule."""

import shutil
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2

KVASIR_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
"""The official Kvasir-SEG archive (44 MB): 1,000 polyp images and their masks."""
KVASIR_INTERMEDIATE_CERT_URL = (
    "http://crt.sectigo.com/SectigoPublicServerAuthenticationCADVR36.crt"
)
"""The certificate that ``datasets.simula.no`` forgets to send.

Browsers fetch a missing intermediate certificate on their own; Python does
not, so the download fails with ``CERTIFICATE_VERIFY_FAILED``.
:func:`prepare_kvasir` then adds this one (the address is written in the
server's own certificate) and retries, still verifying the connection.
"""
KVASIR_FOLDER = "kvasir-seg"
"""Subfolder of ``DATA_DIR`` holding the extracted dataset."""
CLASS_NAMES = ("background", "polyp")
"""Mask values: 0 is background, 1 is polyp."""
MASK_THRESHOLD = 128
"""Mask pixels above this gray level are polyp.

The masks are stored as JPEG, so compression leaves values near 0 and near
255 (e.g. 3 or 251) instead of exactly 0 and 255.
"""
IMAGENET_MEAN = (0.485, 0.456, 0.406)
"""Mean of each color channel over ImageNet, used to normalize every image.

The pretrained encoder of ``config03`` expects images normalized this way;
the U-Nets trained from scratch use the same normalization so that only the
model differs between configs.
"""
IMAGENET_STD = (0.229, 0.224, 0.225)
"""Standard deviation of each color channel over ImageNet."""
SPLITS = ("train", "val", "test")
"""Names of the three splits, in the order of ``split_sizes``."""


def prepare_kvasir(data_dir: str | Path) -> Path:
    """Download and extract Kvasir-SEG once, and return its folder.

    Args:
        data_dir: The shared data folder, e.g. ``<repo>/data``.

    Returns:
        ``<data_dir>/kvasir-seg/Kvasir-SEG``, with ``images/`` and ``masks/``
        (same file names) and ``kavsir_bboxes.json``.
    """
    root = Path(data_dir) / KVASIR_FOLDER
    dataset_dir = root / "Kvasir-SEG"
    if (dataset_dir / "masks").is_dir():
        return dataset_dir
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "kvasir-seg.zip"
    print(f"Downloading Kvasir-SEG (44 MB) to {root}")
    try:
        _download(KVASIR_URL, archive)
    except urllib.error.URLError as error:
        if not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
        context = ssl.create_default_context()
        with urllib.request.urlopen(KVASIR_INTERMEDIATE_CERT_URL, timeout=60) as r:
            context.load_verify_locations(cadata=ssl.DER_cert_to_PEM_cert(r.read()))
        _download(KVASIR_URL, archive, context)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    archive.unlink()
    return dataset_dir


def _download(url: str, target: Path, context: ssl.SSLContext | None = None) -> None:
    """Save ``url`` to ``target``, through a temporary file."""
    partial = target.with_suffix(".part")
    with (
        urllib.request.urlopen(url, context=context, timeout=60) as r,
        partial.open("wb") as f,
    ):
        shutil.copyfileobj(r, f)
    partial.rename(target)


def split_samples(
    dataset_dir: Path, split_sizes: tuple[int, int, int], seed: int
) -> dict[str, list[str]]:
    """Split the image names into train, validation, and test, reproducibly.

    Args:
        dataset_dir: The folder returned by :func:`prepare_kvasir`.
        split_sizes: Number of images in train, validation, and test; they
            must add up to at most 1,000.
        seed: Seed of the random permutation; the same seed gives the same split.

    Returns:
        ``{"train": [...], "val": [...], "test": [...]}``: file names such as
        ``"cju0qkwl35piu0993l0dewei2.jpg"``.
    """
    names = sorted(p.name for p in (dataset_dir / "images").glob("*.jpg"))
    if sum(split_sizes) > len(names):
        raise ValueError(
            f"split_sizes {split_sizes} need more than {len(names)} images"
        )
    order = torch.randperm(len(names), generator=torch.Generator().manual_seed(seed))
    shuffled = [names[i] for i in order.tolist()]
    bounds = np.cumsum([0, *split_sizes])
    return {
        split: shuffled[bounds[k] : bounds[k + 1]] for k, split in enumerate(SPLITS)
    }


def load_binary_mask(path: Path) -> np.ndarray:
    """Read a Kvasir-SEG mask as 0 (background) and 1 (polyp).

    Args:
        path: A file of ``masks/``: an RGB JPEG, white polyp on black.

    Returns:
        A ``(H, W)`` ``uint8`` array of 0 and 1.
    """
    gray = np.array(Image.open(path).convert("L"))
    return (gray >= MASK_THRESHOLD).astype(np.uint8)


def build_transform(image_size: int, augmentation: str, train: bool) -> v2.Compose:
    """The transform applied to an (image, mask) pair.

    The image and its mask go through the transform **together**, as
    ``tv_tensors.Image`` and ``tv_tensors.Mask``: a flip or crop moves both the
    same way, the mask is resized with nearest-neighbor interpolation (so it
    stays 0 or 1), and color changes touch only the image.

    Args:
        image_size: Side of the square output, e.g. 256.
        augmentation: ``"none"``, ``"minimal"`` (random horizontal flip), or
            ``"strong"`` (random crop of 50-100 % of the image, horizontal and
            vertical flips, rotation up to 15 degrees, color jitter).
        train: Augment (training set); otherwise only resize (validation, test).

    Returns:
        A transform taking and returning ``(image, mask)``: the image becomes
        ``float32`` ``(3, S, S)`` normalized with the ImageNet statistics, the
        mask stays ``uint8`` ``(1, S, S)``.
    """
    size = (image_size, image_size)
    if not train or augmentation == "none":
        geometry = [v2.Resize(size)]
    elif augmentation == "minimal":
        geometry = [v2.Resize(size), v2.RandomHorizontalFlip()]
    elif augmentation == "strong":
        geometry = [
            v2.RandomResizedCrop(size, scale=(0.5, 1.0)),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(15),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ]
    else:
        raise ValueError(
            f"augmentation must be none, minimal, or strong: {augmentation!r}"
        )
    return v2.Compose(
        [
            *geometry,
            v2.ToDtype({tv_tensors.Image: torch.float32, "others": None}, scale=True),
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class KvasirSegDataset(Dataset):
    """Kvasir-SEG images and binary masks, as tensors ready for the network.

    One sample is ``(image, mask)``:

    - ``image``: ``float32`` ``(3, S, S)``, normalized (roughly between -2 and 2.6).
    - ``mask``: ``float32`` ``(1, S, S)``, 1.0 for polyp pixels, 0.0 elsewhere,
      the target of the binary cross-entropy loss.
    """

    def __init__(
        self, dataset_dir: Path, names: list[str], transform: v2.Compose
    ) -> None:
        """Store where the files are; nothing is loaded yet.

        Args:
            dataset_dir: The folder returned by :func:`prepare_kvasir`.
            names: File names of this split's images (and masks).
            transform: Applied to each ``(image, mask)`` pair; see :func:`build_transform`.
        """
        self.dataset_dir = Path(dataset_dir)
        self.names = names
        self.transform = transform

    def __len__(self) -> int:
        """Number of images in the split."""
        return len(self.names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load, binarize, transform, and return one image and its mask."""
        name = self.names[index]
        image = Image.open(self.dataset_dir / "images" / name).convert("RGB")
        image = tv_tensors.Image(torch.from_numpy(np.array(image)).permute(2, 0, 1))
        mask = load_binary_mask(self.dataset_dir / "masks" / name)
        mask = tv_tensors.Mask(torch.from_numpy(mask)[None])
        image, mask = self.transform(image, mask)
        return image.as_subclass(torch.Tensor), mask.as_subclass(torch.Tensor).float()


class KvasirSegDataModule(L.LightningDataModule):
    """Kvasir-SEG train, validation, and test dataloaders.

    The 1,000 images are split once, with ``seed``, into ``split_sizes``
    images (800 / 100 / 100 by default). Every image is resized to
    ``image_size`` x ``image_size``; training images are also augmented as
    set by ``augmentation`` (:func:`build_transform`).
    """

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        batch_size: int = 16,
        num_workers: int = 2,
        augmentation: str = "minimal",
        split_sizes: tuple[int, int, int] = (800, 100, 100),
        seed: int = 42,
    ) -> None:
        """Store the settings; nothing is downloaded or loaded yet.

        Args:
            data_dir: The shared data folder; the dataset goes to ``<data_dir>/kvasir-seg``.
            image_size: Side of the square images fed to the network.
            batch_size: Number of images per batch.
            num_workers: Subprocesses loading and augmenting images.
            augmentation: ``"none"``, ``"minimal"``, or ``"strong"``.
            split_sizes: Number of train, validation, and test images.
            seed: Seed of the split.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

    def prepare_data(self) -> None:
        """Download the dataset (Lightning calls this once, before ``setup``)."""
        prepare_kvasir(self.hparams.data_dir)

    def setup(self, stage: str) -> None:
        """Split the images and create the three datasets."""
        self.dataset_dir = prepare_kvasir(self.hparams.data_dir)
        self.splits = split_samples(
            self.dataset_dir, tuple(self.hparams.split_sizes), self.hparams.seed
        )
        size, augmentation = self.hparams.image_size, self.hparams.augmentation
        for split in SPLITS:
            transform = build_transform(size, augmentation, train=split == "train")
            dataset = KvasirSegDataset(self.dataset_dir, self.splits[split], transform)
            setattr(self, f"{split}_set", dataset)

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
        """Build a DataLoader with the batch size and workers from the config."""
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )
