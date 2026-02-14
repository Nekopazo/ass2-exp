"""Data loading pipeline for Keras experiments (Phase 4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import tensorflow as tf
from torchvision import datasets
import yaml

Split = Literal["train", "val", "test"]
DatasetName = Literal["cifar10", "cifar100", "tiny_imagenet"]


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_stats(stats_path: Path, dataset: DatasetName) -> tuple[np.ndarray, np.ndarray]:
    with stats_path.open("r", encoding="utf-8") as f:
        stats = yaml.safe_load(f)
    if dataset not in stats:
        raise KeyError(f"Dataset {dataset} not found in {stats_path}")
    mean = np.asarray(stats[dataset]["mean"], dtype=np.float32)
    std = np.asarray(stats[dataset]["std"], dtype=np.float32)
    return mean, std


def _ensure_rgb_batch(images: np.ndarray) -> np.ndarray:
    if images.ndim == 3:
        images = np.repeat(images[..., None], 3, axis=-1)
    elif images.ndim == 4 and images.shape[-1] == 1:
        images = np.repeat(images, 3, axis=-1)
    elif images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Unexpected image tensor shape: {images.shape}")
    return images.astype(np.uint8, copy=False)


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
    train_images = _ensure_rgb_batch(np.load(train_images_path, mmap_mode="r"))
    train_labels = np.load(train_labels_path).astype(np.int64)
    test_images = _ensure_rgb_batch(np.load(test_images_path, mmap_mode="r"))
    test_labels = np.load(test_labels_path).astype(np.int64)
    return train_images, train_labels, test_images, test_labels


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


def _build_preprocess_fn(
    dataset: DatasetName,
    split: Split,
    config: Mapping[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
):
    is_train = split == "train"
    if dataset in {"cifar10", "cifar100"}:
        aug_cfg = config["augmentation"]["cifar"]
    else:
        aug_cfg = config["augmentation"]["tiny_imagenet"]

    crop_size = int(aug_cfg["random_crop"]["size"])
    padding = int(aug_cfg["random_crop"]["padding"])
    do_flip = bool(aug_cfg.get("random_horizontal_flip", True))

    mean_t = tf.constant(mean, dtype=tf.float32)
    std_t = tf.constant(std, dtype=tf.float32)

    def preprocess(image: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.cast(image, tf.float32)
        if image.shape.rank == 2:
            image = tf.repeat(image[:, :, tf.newaxis], repeats=3, axis=2)
        elif image.shape.rank == 3 and image.shape[-1] == 1:
            image = tf.repeat(image, repeats=3, axis=2)
        if is_train:
            image = tf.image.resize_with_crop_or_pad(
                image,
                target_height=crop_size + 2 * padding,
                target_width=crop_size + 2 * padding,
            )
            image = tf.image.random_crop(image, size=[crop_size, crop_size, 3])
            if do_flip:
                image = tf.image.random_flip_left_right(image)
        image = image / 255.0
        image = (image - mean_t) / std_t
        label = tf.cast(label, tf.int64)
        return image, label

    return preprocess


def get_tf_dataset(
    dataset: DatasetName,
    split: Split,
    config: Mapping[str, Any],
    project_root: str | Path | None = None,
    batch_size: int | None = None,
    shuffle_seed: int = 42,
) -> tf.data.Dataset:
    """Build split-aware tf.data pipeline matching the PyTorch data protocol."""
    root = Path(project_root) if project_root is not None else _default_project_root()
    stats_path = root / "configs" / "dataset_stats.yaml"
    mean, std = _load_stats(stats_path, dataset)
    images, labels = load_split_arrays(dataset=dataset, split=split, project_root=root)

    ds = tf.data.Dataset.from_tensor_slices((images, labels))
    if split == "train":
        ds = ds.shuffle(
            buffer_size=int(labels.shape[0]),
            seed=int(shuffle_seed),
            reshuffle_each_iteration=True,
        )

    preprocess = _build_preprocess_fn(
        dataset=dataset,
        split=split,
        config=config,
        mean=mean,
        std=std,
    )
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    effective_batch_size = int(batch_size or config["training"]["batch_size"])
    ds = ds.batch(effective_batch_size, drop_remainder=(split == "train"))
    return ds.prefetch(tf.data.AUTOTUNE)
