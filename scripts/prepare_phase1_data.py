#!/usr/bin/env python3
"""Prepare datasets, fixed splits, and dataset statistics for Phase 1."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from torchvision import datasets


SEED = 42


def ensure_rgb_uint8(image: np.ndarray) -> np.ndarray:
    """Convert image array to uint8 RGB with shape (H, W, 3)."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Unexpected image shape: {image.shape}")
    return image.astype(np.uint8, copy=False)


def save_split_indices(
    labels: np.ndarray, train_size: int, val_size: int, train_path: Path, val_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(labels.shape[0])
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, random_state=SEED, stratify=labels
    )
    if train_idx.size != train_size or val_idx.size != val_size:
        raise RuntimeError("Split size mismatch")
    np.save(train_path, train_idx)
    np.save(val_path, val_idx)
    return train_idx, val_idx


def validate_split(indices_train: np.ndarray, indices_val: np.ndarray, total_size: int) -> None:
    train_set = set(indices_train.tolist())
    val_set = set(indices_val.tolist())
    if train_set & val_set:
        raise RuntimeError("Train/val index intersection is non-empty")
    full = train_set | val_set
    expected = set(range(total_size))
    if full != expected:
        raise RuntimeError("Train/val index union does not cover full index space")


def compute_mean_std(images_uint8: np.ndarray) -> tuple[list[float], list[float]]:
    x = images_uint8.astype(np.float64) / 255.0  # float64 !
    mean = x.mean(axis=(0, 1, 2), dtype=np.float64)
    std = x.std(axis=(0, 1, 2), dtype=np.float64)
    return mean.tolist(), std.tolist()


def check_normalization(images_uint8: np.ndarray, mean: Iterable[float], std: Iterable[float]) -> None:
    x = images_uint8.astype(np.float64) / 255.0  # float64 !
    mean_arr = np.asarray(mean, dtype=np.float64).reshape(1, 1, 1, 3)
    std_arr = np.asarray(std, dtype=np.float64).reshape(1, 1, 1, 3)
    z = (x - mean_arr) / std_arr
    z_mean = z.mean(axis=(0, 1, 2), dtype=np.float64)
    z_std = z.std(axis=(0, 1, 2), dtype=np.float64)
    if not np.all(np.abs(z_mean) < 5e-3):
        raise RuntimeError(f"Standardized mean too far from 0: {z_mean}")
    if not np.all(np.abs(z_std - 1.0) < 5e-3):
        raise RuntimeError(f"Standardized std too far from 1: {z_std}")


def load_tiny_imagenet(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_train_images = data_dir / "tiny_imagenet_train_images.npy"
    local_train_labels = data_dir / "tiny_imagenet_train_labels.npy"
    local_test_images = data_dir / "tiny_imagenet_test_images.npy"
    local_test_labels = data_dir / "tiny_imagenet_test_labels.npy"
    if all(
        p.exists()
        for p in [local_train_images, local_train_labels, local_test_images, local_test_labels]
    ):
        print("[Tiny-ImageNet] Loading existing local npy files.")
        return (
            np.load(local_train_images),
            np.load(local_train_labels),
            np.load(local_test_images),
            np.load(local_test_labels),
        )

    tiny_root = data_dir / "tiny-imagenet-200"
    if tiny_root.exists():
        return load_tiny_from_folder(tiny_root)

    return load_tiny_from_hf(data_dir)


def load_tiny_from_folder(tiny_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from PIL import Image

    print(f"[Tiny-ImageNet] Loading from local folder: {tiny_root}")
    wnids = (tiny_root / "wnids.txt").read_text(encoding="utf-8").strip().splitlines()
    cls_to_id = {wnid.strip(): i for i, wnid in enumerate(wnids)}

    train_images: list[np.ndarray] = []
    train_labels: list[int] = []
    train_root = tiny_root / "train"
    for wnid in wnids:
        img_dir = train_root / wnid / "images"
        for img_path in sorted(img_dir.glob("*.JPEG")):
            with Image.open(img_path) as img:
                train_images.append(ensure_rgb_uint8(np.array(img)))
            train_labels.append(cls_to_id[wnid])

    val_ann = tiny_root / "val" / "val_annotations.txt"
    val_map: dict[str, int] = {}
    for line in val_ann.read_text(encoding="utf-8").strip().splitlines():
        fields = line.split("\t")
        if len(fields) >= 2:
            val_map[fields[0]] = cls_to_id[fields[1]]

    val_images: list[np.ndarray] = []
    val_labels: list[int] = []
    val_img_dir = tiny_root / "val" / "images"
    for img_path in sorted(val_img_dir.glob("*.JPEG")):
        with Image.open(img_path) as img:
            val_images.append(ensure_rgb_uint8(np.array(img)))
        val_labels.append(val_map[img_path.name])

    x_train = np.stack(train_images, axis=0)
    y_train = np.asarray(train_labels, dtype=np.int64)
    x_test = np.stack(val_images, axis=0)
    y_test = np.asarray(val_labels, dtype=np.int64)
    return x_train, y_train, x_test, y_test


def load_tiny_from_hf(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as err:
        raise RuntimeError(
            "Tiny-ImageNet not found locally and `datasets` is not installed. "
            "Please provide local `data/tiny-imagenet-200` folder or install `datasets`."
        ) from err

    candidates = ["zh-plus/tiny-imagenet", "Maysee/tiny-imagenet"]
    last_err: Exception | None = None
    ds = None
    for name in candidates:
        try:
            ds = load_dataset(name, cache_dir=str(data_dir / "hf_cache"))
            print(f"[Tiny-ImageNet] Loaded dataset: {name}")
            break
        except Exception as err:  # noqa: BLE001
            last_err = err
            print(f"[Tiny-ImageNet] Failed to load {name}: {err}")
    if ds is None:
        raise RuntimeError(f"Unable to load Tiny-ImageNet from candidates: {candidates}") from last_err

    if "train" not in ds or "valid" not in ds:
        raise RuntimeError(f"Unexpected splits in Tiny-ImageNet dataset: {list(ds.keys())}")

    train_split = ds["train"]
    valid_split = ds["valid"]
    train_images = [ensure_rgb_uint8(np.array(item["image"])) for item in train_split]
    valid_images = [ensure_rgb_uint8(np.array(item["image"])) for item in valid_split]
    x_train = np.stack(train_images, axis=0)
    y_train = np.asarray(train_split["label"], dtype=np.int64)
    x_test = np.stack(valid_images, axis=0)
    y_test = np.asarray(valid_split["label"], dtype=np.int64)
    return x_train, y_train, x_test, y_test


def load_cifar(
    data_dir: Path, dataset_cls: type[datasets.CIFAR10], name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        train_set = dataset_cls(root=str(data_dir), train=True, download=False)
        test_set = dataset_cls(root=str(data_dir), train=False, download=False)
        print(f"[{name}] Loaded from local cache.")
    except Exception:  # noqa: BLE001
        train_set = dataset_cls(root=str(data_dir), train=True, download=True)
        test_set = dataset_cls(root=str(data_dir), train=False, download=True)
        print(f"[{name}] Downloaded from remote source.")

    x_train = np.asarray(train_set.data, dtype=np.uint8)
    y_train = np.asarray(train_set.targets, dtype=np.int64)
    x_test = np.asarray(test_set.data, dtype=np.uint8)
    y_test = np.asarray(test_set.targets, dtype=np.int64)
    return x_train, y_train, x_test, y_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    data_dir = project_root / "data"
    splits_dir = data_dir / "splits"
    cfg_dir = project_root / "configs"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    # Step 1.1 CIFAR-10
    x10_train, y10_train, x10_test, y10_test = load_cifar(data_dir, datasets.CIFAR10, "CIFAR-10")
    print(f"[CIFAR-10] train: {x10_train.shape}, test: {x10_test.shape}, classes: {len(np.unique(y10_train))}")

    # Step 1.2 CIFAR-100
    x100_train, y100_train, x100_test, y100_test = load_cifar(data_dir, datasets.CIFAR100, "CIFAR-100")
    print(
        f"[CIFAR-100] train: {x100_train.shape}, test: {x100_test.shape}, classes: {len(np.unique(y100_train))}"
    )

    # Step 1.3 Tiny-ImageNet
    tiny_train_x, tiny_train_y, tiny_test_x, tiny_test_y = load_tiny_imagenet(data_dir)
    np.save(data_dir / "tiny_imagenet_train_images.npy", tiny_train_x)
    np.save(data_dir / "tiny_imagenet_train_labels.npy", tiny_train_y)
    np.save(data_dir / "tiny_imagenet_test_images.npy", tiny_test_x)
    np.save(data_dir / "tiny_imagenet_test_labels.npy", tiny_test_y)
    print(
        "[Tiny-ImageNet] train/test shapes:",
        tiny_train_x.shape,
        tiny_test_x.shape,
        "classes:",
        len(np.unique(tiny_train_y)),
    )
    sample_shapes = [tiny_train_x[i].shape for i in [0, 1, 2, 3, 4]]
    print(f"[Tiny-ImageNet] sample image shapes (5): {sample_shapes}")

    # Step 1.4 CIFAR-10 splits
    c10_train_idx, c10_val_idx = save_split_indices(
        y10_train,
        train_size=45_000,
        val_size=5_000,
        train_path=splits_dir / "cifar10_train_indices.npy",
        val_path=splits_dir / "cifar10_val_indices.npy",
    )
    validate_split(c10_train_idx, c10_val_idx, total_size=50_000)
    c10_val_counts = np.bincount(y10_train[c10_val_idx], minlength=10)
    if not np.all(c10_val_counts == 500):
        raise RuntimeError(f"CIFAR-10 val class counts mismatch: {c10_val_counts}")
    print(f"[CIFAR-10 split] train={c10_train_idx.size}, val={c10_val_idx.size}, val_counts={c10_val_counts.tolist()}")

    # Step 1.5 CIFAR-100 splits
    c100_train_idx, c100_val_idx = save_split_indices(
        y100_train,
        train_size=45_000,
        val_size=5_000,
        train_path=splits_dir / "cifar100_train_indices.npy",
        val_path=splits_dir / "cifar100_val_indices.npy",
    )
    validate_split(c100_train_idx, c100_val_idx, total_size=50_000)
    c100_val_counts = np.bincount(y100_train[c100_val_idx], minlength=100)
    if not np.all(c100_val_counts == 50):
        raise RuntimeError(f"CIFAR-100 val class counts mismatch")
    print(
        f"[CIFAR-100 split] train={c100_train_idx.size}, val={c100_val_idx.size}, "
        f"val_count_unique={np.unique(c100_val_counts).tolist()}"
    )

    # Step 1.6 Tiny-ImageNet splits
    tiny_train_idx, tiny_val_idx = save_split_indices(
        tiny_train_y,
        train_size=90_000,
        val_size=10_000,
        train_path=splits_dir / "tiny_imagenet_train_indices.npy",
        val_path=splits_dir / "tiny_imagenet_val_indices.npy",
    )
    validate_split(tiny_train_idx, tiny_val_idx, total_size=100_000)
    tiny_val_counts = np.bincount(tiny_train_y[tiny_val_idx], minlength=200)
    if not np.all(tiny_val_counts == 50):
        raise RuntimeError("Tiny-ImageNet val class counts mismatch")
    if tiny_test_x.shape[0] != 10_000 or tiny_test_y.min() < 0 or tiny_test_y.max() > 199:
        raise RuntimeError("Tiny-ImageNet test split check failed")
    print(
        f"[Tiny-ImageNet split] train={tiny_train_idx.size}, val={tiny_val_idx.size}, "
        f"val_count_unique={np.unique(tiny_val_counts).tolist()}, test={tiny_test_x.shape[0]}"
    )

    # Step 1.7 mean/std
    c10_mean, c10_std = compute_mean_std(x10_train[c10_train_idx])
    c100_mean, c100_std = compute_mean_std(x100_train[c100_train_idx])
    tiny_mean, tiny_std = compute_mean_std(tiny_train_x[tiny_train_idx])

    check_normalization(x10_train[c10_train_idx], c10_mean, c10_std)
    check_normalization(x100_train[c100_train_idx], c100_mean, c100_std)
    check_normalization(tiny_train_x[tiny_train_idx], tiny_mean, tiny_std)

    stats = {
        "cifar10": {"mean": [round(v, 6) for v in c10_mean], "std": [round(v, 6) for v in c10_std]},
        "cifar100": {"mean": [round(v, 6) for v in c100_mean], "std": [round(v, 6) for v in c100_std]},
        "tiny_imagenet": {"mean": [round(v, 6) for v in tiny_mean], "std": [round(v, 6) for v in tiny_std]},
    }
    with (cfg_dir / "dataset_stats.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(stats, f, sort_keys=False)

    print("[dataset_stats.yaml]")
    print(yaml.safe_dump(stats, sort_keys=False))
    print("Phase 1 complete.")


if __name__ == "__main__":
    main()
