"""Food-101 datasets and loaders shared by both benchmark profiles."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import ExperimentConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _transforms(image_size: int) -> tuple[Any, Any]:
    """Return the augmentation family used by the reference experiment."""

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.TrivialAugmentWide(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_datasets(config: ExperimentConfig) -> tuple[Any, Any, str]:
    """Build from an ImageFolder layout or torchvision Food101.

    ImageFolder is selected when ``DATA_ROOT/train`` and ``DATA_ROOT/test``
    exist. Otherwise torchvision's official Food101 adapter is used, which can
    download and validate the public archive.
    """

    train_transform, eval_transform = _transforms(config.image_size)
    data_root = Path(config.data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    if train_dir.is_dir() and test_dir.is_dir():
        train_data = datasets.ImageFolder(train_dir, transform=train_transform)
        eval_data = datasets.ImageFolder(test_dir, transform=eval_transform)
        source = "imagefolder"
    else:
        train_data = datasets.Food101(
            data_root, split="train", transform=train_transform, download=config.download
        )
        eval_data = datasets.Food101(
            data_root, split="test", transform=eval_transform, download=config.download
        )
        source = "torchvision.datasets.Food101"

    train_classes = list(train_data.classes)
    eval_classes = list(eval_data.classes)
    if train_classes != eval_classes:
        raise RuntimeError("training and evaluation class mappings differ")
    if len(train_classes) != config.num_classes:
        raise RuntimeError(
            f"expected {config.num_classes} classes, found {len(train_classes)}"
        )
    return train_data, eval_data, source


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def build_loaders(
    config: ExperimentConfig, *, use_cuda: bool
) -> tuple[DataLoader[Any], DataLoader[Any], dict[str, Any]]:
    train_data, eval_data, source = build_datasets(config)
    generator = torch.Generator().manual_seed(config.seed)
    shared: dict[str, Any] = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": use_cuda,
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": _seed_worker,
        "generator": generator,
    }
    if config.num_workers > 0:
        shared["prefetch_factor"] = 2

    train_loader = DataLoader(train_data, shuffle=True, **shared)
    eval_loader = DataLoader(eval_data, shuffle=False, **shared)
    metadata = {
        "source": source,
        "root": str(config.data_root),
        "train_examples": len(train_data),
        "eval_examples": len(eval_data),
        "classes": len(train_data.classes),
    }
    return train_loader, eval_loader, metadata
