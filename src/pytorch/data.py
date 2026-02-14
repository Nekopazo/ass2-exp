"""Data loading pipeline for PyTorch experiments (Phase 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import yaml

Split = Literal["train", "val", "test"]
DatasetName = Literal["cifar10", "cifar100", "tiny_imagenet"]


class NumpyImageDataset(Dataset):
    """Simple numpy-backed dataset with optional transform."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.images = images
        self.labels = labels.astype(np.int64, copy=False)
        self.transform = transform

    @staticmethod
    def _ensure_rgb(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        return image.astype(np.uint8, copy=False)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self._ensure_rgb(self.images[index])
        label = int(self.labels[index])
        pil_image = Image.fromarray(image, mode="RGB")
        if self.transform is None:
            return transforms.ToTensor()(pil_image), label
        return self.transform(pil_image), label


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_stats(stats_path: Path, dataset: DatasetName) -> tuple[list[float], list[float]]:
    with stats_path.open("r", encoding="utf-8") as f:
        stats = yaml.safe_load(f)
    if dataset not in stats:
        raise KeyError(f"Dataset {dataset} not found in {stats_path}")
    mean = [float(v) for v in stats[dataset]["mean"]]
    std = [float(v) for v in stats[dataset]["std"]]
    return mean, std


def _load_cifar(data_dir: Path, dataset: DatasetName) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_cls = datasets.CIFAR10 if dataset == "cifar10" else datasets.CIFAR100
    train_set = dataset_cls(root=str(data_dir), train=True, download=False)
    test_set = dataset_cls(root=str(data_dir), train=False, download=False)
    train_images = np.asarray(train_set.data, dtype=np.uint8)
    train_labels = np.asarray(train_set.targets, dtype=np.int64)
    test_images = np.asarray(test_set.data, dtype=np.uint8)
    test_labels = np.asarray(test_set.targets, dtype=np.int64)
    return train_images, train_labels, test_images, test_labels


def _load_tiny_imagenet(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images_path = data_dir / "tiny_imagenet_train_images.npy"
    train_labels_path = data_dir / "tiny_imagenet_train_labels.npy"
    test_images_path = data_dir / "tiny_imagenet_test_images.npy"
    test_labels_path = data_dir / "tiny_imagenet_test_labels.npy"
    required = [train_images_path, train_labels_path, test_images_path, test_labels_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Tiny-ImageNet numpy files are missing. Expected: " + ", ".join(missing)
        )

    train_images = np.load(train_images_path, mmap_mode="r")
    train_labels = np.load(train_labels_path)
    test_images = np.load(test_images_path, mmap_mode="r")
    test_labels = np.load(test_labels_path)
    return train_images, train_labels.astype(np.int64), test_images, test_labels.astype(np.int64)


def _load_full_arrays(data_dir: Path, dataset: DatasetName) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset in {"cifar10", "cifar100"}:
        return _load_cifar(data_dir=data_dir, dataset=dataset)
    if dataset == "tiny_imagenet":
        return _load_tiny_imagenet(data_dir=data_dir)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _index_file(splits_dir: Path, dataset: DatasetName, split: Split) -> Path:
    if dataset == "tiny_imagenet":
        if split == "train":
            return splits_dir / "tiny_imagenet_train_indices.npy"
        if split == "val":
            return splits_dir / "tiny_imagenet_val_indices.npy"
        raise ValueError("Tiny-ImageNet test split uses official test array and has no index file.")

    prefix = "cifar10" if dataset == "cifar10" else "cifar100"
    if split == "train":
        return splits_dir / f"{prefix}_train_indices.npy"
    if split == "val":
        return splits_dir / f"{prefix}_val_indices.npy"
    raise ValueError(f"{dataset} test split uses official test array and has no index file.")


def _build_transform(
    dataset: DatasetName,
    split: Split,
    config: Mapping[str, Any],
    mean: list[float],
    std: list[float],
) -> transforms.Compose:
    ops: list[Any] = []
    if split == "train":
        if dataset in {"cifar10", "cifar100"}:
            aug_cfg = config["augmentation"]["cifar"]
        else:
            aug_cfg = config["augmentation"]["tiny_imagenet"]
        crop_cfg = aug_cfg["random_crop"]
        ops.append(
            transforms.RandomCrop(
                size=int(crop_cfg["size"]),
                padding=int(crop_cfg["padding"]),
            )
        )
        if bool(aug_cfg.get("random_horizontal_flip", True)):
            ops.append(transforms.RandomHorizontalFlip())

    ops.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(ops)


def load_split_arrays(
    dataset: DatasetName,
    split: Split,
    project_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    root = Path(project_root) if project_root is not None else _default_project_root()
    data_dir = root / "data"
    splits_dir = data_dir / "splits"
    train_images, train_labels, test_images, test_labels = _load_full_arrays(data_dir, dataset)

    if split == "test":
        return test_images, test_labels

    idx_path = _index_file(splits_dir=splits_dir, dataset=dataset, split=split)
    indices = np.load(idx_path)
    return train_images[indices], train_labels[indices]


def get_dataloader(
    dataset: DatasetName,
    split: Split,
    config: Mapping[str, Any],
    project_root: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int = 8,
) -> DataLoader:
    """Build a split-aware DataLoader with plan-required settings."""
    root = Path(project_root) if project_root is not None else _default_project_root()
    stats_path = root / "configs" / "dataset_stats.yaml"
    mean, std = _load_stats(stats_path, dataset)
    images, labels = load_split_arrays(dataset=dataset, split=split, project_root=root)

    transform = _build_transform(dataset=dataset, split=split, config=config, mean=mean, std=std)
    ds = NumpyImageDataset(images=images, labels=labels, transform=transform)

    is_train = split == "train"
    effective_batch_size = int(batch_size or config["training"]["batch_size"])
    loader_kwargs: dict[str, Any] = {
        "dataset": ds,
        "batch_size": effective_batch_size,
        "shuffle": is_train,
        "drop_last": is_train,
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)
