#!/usr/bin/env python3
"""
Keras vs PyTorch 图像分类 Benchmark — 完整实现 (Stage B)
==============================================================
阶段 B：PlantVillage + CIFAR-10 迁移学习实验。
对比 Keras 与 PyTorch 在 ResNet50、VGG16、MobileNetV2 上的表现。

用法示例:
  # Phase 1: 数据准备
  python ass2_code.py prepare_data                       # 准备全部数据集
  python ass2_code.py prepare_data --dataset cifar10     # 只准备 CIFAR-10

  # Phase 2: 生成配置文件
  python ass2_code.py generate_configs

  # Phase 3: 参数对齐表
  python ass2_code.py param_comparison

  # Phase 5: 训练
  python ass2_code.py train --framework pytorch --dataset plantvillage --model resnet50 --fold 0
  python ass2_code.py train --framework keras --dataset cifar10 --model mobilenetv2 --fold 2

  # Phase 6: 结果汇总
  python ass2_code.py aggregate

  # Phase 7: 运行全部实验
  python ass2_code.py run_all

  # Phase 8: 统计分析
  python ass2_code.py statistical_tests

  # Phase 9: 可视化
  python ass2_code.py plot_curves
  python ass2_code.py plot_bars
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# PyTorch NVML workaround: 部分 HPC/容器环境 NVML 初始化失败，
# 禁用 NVML 相关检查以避免 CUDACachingAllocator 崩溃。
# 必须在 import torch 之前设置。
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# TF GPU 配置（在任何 TF 操作之前调用）
# ---------------------------------------------------------------------------

_tf_gpu_configured = False

def _select_best_gpu() -> int:
    """选择空闲显存最多的 GPU，返回其索引。"""
    import torch
    n = torch.cuda.device_count()
    if n <= 1:
        return 0
    best_idx, best_free = 0, 0
    for i in range(n):
        free, _ = torch.cuda.mem_get_info(i)
        print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): free={free / 1024**3:.1f} GB")
        if free > best_free:
            best_free = free
            best_idx = i
    return best_idx


def setup_pytorch_gpu() -> "torch.device":
    """选择空闲显存最多的 GPU 并返回 torch.device。"""
    import torch
    if not torch.cuda.is_available():
        print("[PyTorch GPU] No CUDA device, using CPU")
        return torch.device("cpu")
    idx = _select_best_gpu()
    torch.cuda.set_device(idx)
    print(f"[PyTorch GPU] Using GPU {idx} ({torch.cuda.get_device_name(idx)})")
    return torch.device(f"cuda:{idx}")


def setup_tf_gpu():
    """配置 TensorFlow GPU：允许显存按需增长，多卡时选择空闲 GPU。"""
    global _tf_gpu_configured
    if _tf_gpu_configured:
        return
    _tf_gpu_configured = True

    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # 选择空闲显存最多的 GPU
    if len(gpus) > 1:
        try:
            import torch
            best = _select_best_gpu()
        except Exception:
            best = 1
        tf.config.set_visible_devices(gpus[best], 'GPU')
        print(f"[TF GPU] Using {gpus[best].name} (index {best}, most free memory)")
    else:
        print(f"[TF GPU] Using {gpus[0].name}")


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DATASETS_INFO = {
    "plantvillage": {"num_classes": 38},
    "cifar10": {"num_classes": 10},
}
DATASETS = list(DATASETS_INFO.keys())
INPUT_SIZE = 224
MODELS = ["resnet50", "vgg16", "mobilenetv2"]
FOLDS = [0, 1, 2]
FRAMEWORKS = ["keras", "pytorch"]


def get_num_classes(dataset: str) -> int:
    return DATASETS_INFO[dataset]["num_classes"]


# ===================================================================
#  Phase 2: 配置文件生成
# ===================================================================

def generate_train_config() -> dict:
    """返回 train_config.yaml 的内容字典。"""
    return {
        "optimizer": {
            "type": "SGD",
            "momentum": 0.9,
            "nesterov": True,
            "weight_decay": 1e-4,
        },
        "lr": {
            "base_lr": 0.04,
            "scheduler": "cosine",
            "eta_min": 0.0,
            "warmup_epochs": 5,
            "warmup_start_lr": 0.0,
            "warmup_mode": "per_step",
        },
        "training": {
            "batch_size": 128,
            "loss": "cross_entropy",
        },
        "early_stopping": {
            "monitor": "val_loss",
            "patience": 7,
            "min_delta": 0.0,
        },
        "epochs": 30,
        "cross_validation": {
            "n_splits": 3,
            "seed": 42,
        },
        "augmentation": {
            "random_resized_crop": {
                "size": 224,
                "scale": [0.8, 1.0],
            },
            "random_horizontal_flip": True,
        },
        "input_size": 224,
    }


def generate_experiment_matrix() -> dict:
    return {
        "name": "迁移学习 — Keras vs PyTorch Benchmark",
        "datasets": ["plantvillage", "cifar10"],
        "models": ["resnet50", "vgg16", "mobilenetv2"],
        "folds": [0, 1, 2],
        "frameworks": ["keras", "pytorch"],
        "transfer_learning": True,
        "total_runs": 36,
    }


def cmd_generate_configs(args: argparse.Namespace) -> None:
    """Phase 2: 生成配置文件。"""
    configs_dir = PROJECT_ROOT / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    train_cfg_path = configs_dir / "train_config.yaml"
    with train_cfg_path.open("w", encoding="utf-8") as f:
        yaml.dump(generate_train_config(), f, default_flow_style=False, allow_unicode=True)
    print(f"[OK] Generated {train_cfg_path}")

    exp_matrix_path = configs_dir / "experiment_matrix.yaml"
    with exp_matrix_path.open("w", encoding="utf-8") as f:
        yaml.dump(generate_experiment_matrix(), f, default_flow_style=False, allow_unicode=True)
    print(f"[OK] Generated {exp_matrix_path}")


# ===================================================================
#  Phase 1: 数据准备
# ===================================================================

def _prepare_plantvillage(data_dir: Path, splits_dir: Path) -> Tuple[np.ndarray, np.ndarray, Path]:
    """加载/准备 PlantVillage 数据，返回 (images, labels, images_path)。"""
    from sklearn.model_selection import StratifiedKFold, train_test_split

    pv_images_path = data_dir / "plantvillage_images.npy"
    pv_labels_path = data_dir / "plantvillage_labels.npy"
    pv_class_names_path = data_dir / "plantvillage_class_names.json"

    print("=" * 60)
    print("Loading PlantVillage...")

    if pv_images_path.exists() and pv_labels_path.exists():
        pv_images = np.load(str(pv_images_path), mmap_mode="r")
        pv_labels = np.load(str(pv_labels_path))
        print(f"  Loaded from npy: images={pv_images.shape}, labels={pv_labels.shape}")
    else:
        pv_dir = data_dir / "plantvillage"
        if not pv_dir.exists():
            print("  [ERROR] PlantVillage data not found.")
            print(f"  Expected: {pv_images_path} + {pv_labels_path}")
            print(f"  Or directory: {pv_dir}/")
            print("  Please download PlantVillage from Kaggle and place it under data/plantvillage/")
            return None, None, pv_images_path

        from PIL import Image as PILImage

        def find_class_root(root: Path, max_depth: int = 3) -> Path:
            subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
            if not subdirs:
                return root
            for sd in subdirs:
                has_images = any(
                    f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                    for f in sd.iterdir() if f.is_file()
                )
                if has_images:
                    return root
            if max_depth > 0 and len(subdirs) <= 2:
                return find_class_root(subdirs[0], max_depth - 1)
            return root

        pv_dir = find_class_root(pv_dir)
        print(f"  Loading from directory: {pv_dir}")
        images_list = []
        labels_list = []
        class_names = sorted([d.name for d in pv_dir.iterdir() if d.is_dir()])
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        for class_name, idx in class_to_idx.items():
            class_dir = pv_dir / class_name
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    img = PILImage.open(str(img_file)).convert("RGB").resize((224, 224))
                    images_list.append(np.asarray(img, dtype=np.uint8))
                    labels_list.append(idx)

        pv_images = np.stack(images_list, axis=0)
        pv_labels = np.asarray(labels_list, dtype=np.int64)
        np.save(str(pv_images_path), pv_images)
        np.save(str(pv_labels_path), pv_labels)
        with pv_class_names_path.open("w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)
        print(f"  Saved: images={pv_images.shape}, labels={pv_labels.shape}")

    print(f"  Total images: {pv_images.shape[0]}, Classes: {len(set(pv_labels.tolist()))}")

    # 测试集划分 + 三折 CV
    _generate_splits("plantvillage", pv_labels, splits_dir)

    return pv_images, pv_labels, pv_images_path


def _prepare_cifar10(data_dir: Path, splits_dir: Path) -> Tuple[np.ndarray, np.ndarray, Path]:
    """加载/准备 CIFAR-10 数据，返回 (images, labels, images_path)。"""

    c10_images_path = data_dir / "cifar10_images.npy"
    c10_labels_path = data_dir / "cifar10_labels.npy"
    c10_class_names_path = data_dir / "cifar10_class_names.json"

    print("=" * 60)
    print("Loading CIFAR-10...")

    if c10_images_path.exists() and c10_labels_path.exists():
        c10_images = np.load(str(c10_images_path))
        c10_labels = np.load(str(c10_labels_path))
        print(f"  Loaded from npy: images={c10_images.shape}, labels={c10_labels.shape}")
    else:
        print("  Downloading CIFAR-10 via keras...")
        import tensorflow as tf
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        c10_images = np.concatenate([x_train, x_test], axis=0)           # (60000, 32, 32, 3)
        c10_labels = np.concatenate([y_train.squeeze(), y_test.squeeze()], axis=0).astype(np.int64)
        np.save(str(c10_images_path), c10_images)
        np.save(str(c10_labels_path), c10_labels)
        class_names = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]
        with c10_class_names_path.open("w", encoding="utf-8") as f:
            json.dump(class_names, f, ensure_ascii=False, indent=2)
        print(f"  Saved: images={c10_images.shape}, labels={c10_labels.shape}")

    print(f"  Total images: {c10_images.shape[0]}, Classes: {len(set(c10_labels.tolist()))}")

    # 测试集划分 + 三折 CV
    _generate_splits("cifar10", c10_labels, splits_dir)

    return c10_images, c10_labels, c10_images_path


def _generate_splits(dataset: str, labels: np.ndarray, splits_dir: Path):
    """为指定数据集生成 test split + 3-fold CV 索引。"""
    from sklearn.model_selection import StratifiedKFold, train_test_split

    print(f"  Generating test split + 3-fold CV for {dataset}...")
    all_indices = np.arange(len(labels))
    trainval_indices, test_indices = train_test_split(
        all_indices, test_size=0.2, stratify=labels, random_state=42
    )
    np.save(str(splits_dir / f"{dataset}_test_indices.npy"), test_indices)
    np.save(str(splits_dir / f"{dataset}_trainval_indices.npy"), trainval_indices)
    print(f"  Test: {len(test_indices)}, TrainVal: {len(trainval_indices)}")

    trainval_labels = labels[trainval_indices]
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for fold_idx, (fold_train_local, fold_val_local) in enumerate(skf.split(trainval_indices, trainval_labels)):
        np.save(str(splits_dir / f"{dataset}_fold{fold_idx}_train_indices.npy"), fold_train_local)
        np.save(str(splits_dir / f"{dataset}_fold{fold_idx}_val_indices.npy"), fold_val_local)
        print(f"  Fold {fold_idx}: train={len(fold_train_local)}, val={len(fold_val_local)}")

    all_val = np.concatenate([
        np.load(str(splits_dir / f"{dataset}_fold{i}_val_indices.npy")) for i in range(3)
    ])
    assert len(set(all_val.tolist())) == len(trainval_indices), f"{dataset} val indices overlap"
    print(f"  [OK] {dataset} 3-fold CV indices verified.")


def _compute_dataset_stats(dataset: str, images_path: Path, splits_dir: Path) -> dict:
    """为指定数据集计算 per-fold channel mean/std。"""
    print(f"  Computing per-fold channel mean/std for {dataset}...")
    images_full = np.load(str(images_path))
    ds_stats = {}
    for fold_idx in range(3):
        trainval_idx = np.load(str(splits_dir / f"{dataset}_trainval_indices.npy"))
        fold_train_local = np.load(str(splits_dir / f"{dataset}_fold{fold_idx}_train_indices.npy"))
        global_train_idx = trainval_idx[fold_train_local]
        fold_images = images_full[global_train_idx].astype(np.float64) / 255.0
        mean = fold_images.mean(axis=(0, 1, 2)).tolist()
        std = fold_images.std(axis=(0, 1, 2)).tolist()
        ds_stats[f"fold{fold_idx}"] = {
            "mean": [round(v, 6) for v in mean],
            "std": [round(v, 6) for v in std],
        }
        print(f"  fold{fold_idx}: mean={[round(v,4) for v in mean]}, std={[round(v,4) for v in std]}")
    return ds_stats


def cmd_prepare_data(args: argparse.Namespace) -> None:
    """Phase 1: 准备数据集 + 生成折索引 + 计算 mean/std。"""
    data_dir = PROJECT_ROOT / "data"
    splits_dir = data_dir / "splits"
    configs_dir = PROJECT_ROOT / "configs"
    data_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    target = getattr(args, "dataset", "all")

    stats = {}

    # --- PlantVillage ---
    if target in ("all", "plantvillage"):
        pv_images, pv_labels, pv_images_path = _prepare_plantvillage(data_dir, splits_dir)
        if pv_images is not None:
            stats["plantvillage"] = _compute_dataset_stats("plantvillage", pv_images_path, splits_dir)

    # --- CIFAR-10 ---
    if target in ("all", "cifar10"):
        c10_images, c10_labels, c10_images_path = _prepare_cifar10(data_dir, splits_dir)
        if c10_images is not None:
            stats["cifar10"] = _compute_dataset_stats("cifar10", c10_images_path, splits_dir)

    # 合并已有 stats（保留其他数据集的统计）
    stats_path = configs_dir / "dataset_stats.yaml"
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as f:
            existing_stats = yaml.safe_load(f) or {}
        existing_stats.update(stats)
        stats = existing_stats

    with stats_path.open("w", encoding="utf-8") as f:
        yaml.dump(stats, f, default_flow_style=False)
    print(f"[OK] Saved {stats_path}")


# ===================================================================
#  共享工具: LR Schedule + Early Stopping
# ===================================================================

@dataclass(frozen=True)
class WarmupCosineLRSchedule:
    """Per-step LR: linear warmup then cosine decay."""
    base_lr: float
    warmup_steps: int
    total_steps: int
    eta_min: float = 0.0
    warmup_start_lr: float = 0.0

    @property
    def cosine_steps(self) -> int:
        return max(1, self.total_steps - self.warmup_steps)

    def lr_at(self, step: int) -> float:
        step = max(0, min(int(step), self.total_steps - 1))
        if self.warmup_steps > 0 and step < self.warmup_steps:
            ratio = float(step) / float(self.warmup_steps)
            return float(self.warmup_start_lr + (self.base_lr - self.warmup_start_lr) * ratio)
        progress = float(step - self.warmup_steps) / float(self.cosine_steps)
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(self.eta_min + (self.base_lr - self.eta_min) * cosine)


class EarlyStopping:
    """Monitor val_loss with patience."""
    def __init__(self, patience: int = 15, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def load_config(path: Optional[str] = None) -> dict:
    if path is None:
        path = str(PROJECT_ROOT / "configs" / "train_config.yaml")
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset_stats(dataset: str, fold: int) -> Tuple[List[float], List[float]]:
    stats_path = PROJECT_ROOT / "configs" / "dataset_stats.yaml"
    with stats_path.open("r", encoding="utf-8") as f:
        stats = yaml.safe_load(f)
    ds_stats = stats[dataset]
    fold_key = f"fold{fold}"
    if fold_key in ds_stats:
        mean = [float(v) for v in ds_stats[fold_key]["mean"]]
        std = [float(v) for v in ds_stats[fold_key]["std"]]
    elif "mean" in ds_stats:
        mean = [float(v) for v in ds_stats["mean"]]
        std = [float(v) for v in ds_stats["std"]]
    else:
        raise KeyError(f"No stats found for {dataset} fold{fold} in {stats_path}")
    return mean, std


# ===================================================================
#  Phase 3: PyTorch 模型
# ===================================================================

def build_pytorch_model(model_name: str, num_classes: int = 38):
    """构建 PyTorch 模型 (迁移学习)。"""
    import torch.nn as nn

    if model_name == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        nn.init.kaiming_normal_(model.fc.weight)
        nn.init.zeros_(model.fc.bias)
        return model

    elif model_name == "vgg16":
        from torchvision.models import vgg16_bn
        model = vgg16_bn(weights="IMAGENET1K_V1")
        model.classifier[6] = nn.Linear(4096, num_classes)
        nn.init.normal_(model.classifier[6].weight, 0, 0.01)
        nn.init.zeros_(model.classifier[6].bias)
        return model

    elif model_name == "mobilenetv2":
        from torchvision.models import mobilenet_v2
        model = mobilenet_v2(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    else:
        raise ValueError(f"Unsupported model: {model_name}")


def count_pytorch_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ===================================================================
#  Phase 3: Keras 模型
# ===================================================================

def build_keras_model(model_name: str, num_classes: int = 38, weight_decay: float = 1e-4):
    """构建 Keras 模型 (迁移学习)。"""
    setup_tf_gpu()
    import tensorflow as tf

    if model_name == "resnet50":
        return _build_keras_resnet50(num_classes, weight_decay)
    elif model_name == "vgg16":
        return _build_keras_vgg16(num_classes, weight_decay)
    elif model_name == "mobilenetv2":
        return _build_keras_mobilenetv2(num_classes, weight_decay)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def _build_keras_resnet50(num_classes: int, weight_decay: float):
    import tensorflow as tf

    base = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet",
        input_shape=(224, 224, 3), pooling="avg",
    )
    x = base.output
    outputs = tf.keras.layers.Dense(
        num_classes,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay / 2),
        name="predictions",
    )(x)
    return tf.keras.Model(inputs=base.input, outputs=outputs, name="resnet50")


def _build_keras_vgg16(num_classes: int, weight_decay: float):
    import tensorflow as tf
    from tensorflow.keras import layers, Model, regularizers

    reg = regularizers.l2(weight_decay / 2)

    vgg_cfg = [
        (64, 2), (128, 2), (256, 3), (512, 3), (512, 3),
    ]

    inputs = layers.Input(shape=(224, 224, 3), name="input")
    x = inputs
    for block_idx, (filters, num_convs) in enumerate(vgg_cfg):
        for c in range(num_convs):
            x = layers.Conv2D(
                filters, 3, padding="same", use_bias=True,
                kernel_regularizer=reg,
                name=f"block{block_idx+1}_conv{c+1}",
            )(x)
            x = layers.BatchNormalization(name=f"block{block_idx+1}_bn{c+1}")(x)
            x = layers.ReLU(name=f"block{block_idx+1}_relu{c+1}")(x)
        x = layers.MaxPooling2D(2, strides=2, name=f"block{block_idx+1}_pool")(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(4096, activation="relu", kernel_regularizer=reg, name="fc1")(x)
    x = layers.Dense(4096, activation="relu", kernel_regularizer=reg, name="fc2")(x)
    outputs = layers.Dense(num_classes, kernel_regularizer=reg, name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="vgg16_bn")

    # 迁移学习: 从官方 VGG16 复制 Conv + Dense 权重
    print("  Loading ImageNet weights for Keras VGG16-BN (Conv+Dense weight copy)...")
    src_model = tf.keras.applications.VGG16(weights="imagenet", include_top=True)
    src_convs = [l for l in src_model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    dst_convs = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    for s, d in zip(src_convs, dst_convs):
        d.set_weights(s.get_weights())
    src_denses = [l for l in src_model.layers if isinstance(l, tf.keras.layers.Dense)]
    dst_denses = [l for l in model.layers if isinstance(l, tf.keras.layers.Dense)]
    for s, d in zip(src_denses[:-1], dst_denses[:-1]):
        if s.get_weights()[0].shape == d.get_weights()[0].shape:
            d.set_weights(s.get_weights())
    del src_model
    print("  [OK] Weight copy done. BN layers use default init.")

    return model


def _build_keras_mobilenetv2(num_classes: int, weight_decay: float):
    import tensorflow as tf

    base = tf.keras.applications.MobileNetV2(
        include_top=False, weights="imagenet",
        input_shape=(224, 224, 3), pooling="avg",
    )
    x = base.output
    outputs = tf.keras.layers.Dense(
        num_classes,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay / 2),
        name="predictions",
    )(x)
    return tf.keras.Model(inputs=base.input, outputs=outputs, name="mobilenetv2")


def count_keras_params(model) -> int:
    return int(sum(
        np.prod(w.shape) for w in model.trainable_weights
    ))


# ===================================================================
#  Phase 3: 参数对齐表
# ===================================================================

def cmd_param_comparison(args: argparse.Namespace) -> None:
    """Phase 3: 生成参数对齐表。"""
    setup_tf_gpu()
    import torch
    import tensorflow as tf

    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "param_comparison.csv"

    rows = []
    for dataset in DATASETS:
        nc = get_num_classes(dataset)
        for model_name in MODELS:
            print(f"Building {model_name} (224x224, {nc} classes, {dataset})...")

            # PyTorch
            pt_model = build_pytorch_model(model_name, num_classes=nc)
            pt_params = count_pytorch_params(pt_model)

            dummy = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                out = pt_model(dummy)
            assert out.shape == (1, nc), f"PyTorch output shape mismatch: {out.shape}"

            # Keras
            tf.keras.backend.clear_session()
            k_model = build_keras_model(model_name, num_classes=nc)
            k_params = count_keras_params(k_model)

            dummy_k = tf.random.normal((1, 224, 224, 3))
            out_k = k_model(dummy_k, training=False)
            assert out_k.shape == (1, nc), f"Keras output shape mismatch: {out_k.shape}"

            diff_pct = abs(pt_params - k_params) / max(pt_params, 1) * 100
            rows.append({
                "dataset": dataset,
                "model": model_name,
                "input_size": 224,
                "num_classes": nc,
                "pytorch_params": pt_params,
                "keras_params": k_params,
                "diff_pct": round(diff_pct, 4),
            })
            print(f"  PT={pt_params:,}  Keras={k_params:,}  diff={diff_pct:.4f}%")

            del pt_model, k_model

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] Saved {csv_path}")


# ===================================================================
#  Phase 4: PyTorch 数据加载器
# ===================================================================

_data_cache: Dict[str, np.ndarray] = {}


def _load_dataset_split(dataset: str, fold: int, split: str):
    """加载指定数据集的 fold/split 数据，返回 (images_np, labels_np)。"""
    data_dir = PROJECT_ROOT / "data"
    splits_dir = data_dir / "splits"

    img_key = f'{dataset}_images'
    lbl_key = f'{dataset}_labels'
    if img_key not in _data_cache:
        _data_cache[img_key] = np.load(str(data_dir / f"{dataset}_images.npy"))
        _data_cache[lbl_key] = np.load(str(data_dir / f"{dataset}_labels.npy"))
    images = _data_cache[img_key]
    labels = _data_cache[lbl_key]

    if split == "test":
        test_idx = np.load(str(splits_dir / f"{dataset}_test_indices.npy"))
        return images[test_idx], labels[test_idx]
    else:
        trainval_idx = np.load(str(splits_dir / f"{dataset}_trainval_indices.npy"))
        fold_local_idx = np.load(str(splits_dir / f"{dataset}_fold{fold}_{split}_indices.npy"))
        global_idx = trainval_idx[fold_local_idx]
        return images[global_idx], labels[global_idx]


def get_pytorch_dataloader(
    dataset: str,
    fold: int,
    split: str,
    config: dict,
    num_workers: int = 8,
):
    """构建 PyTorch DataLoader (Phase 4)。"""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image as PILImage

    batch_size = int(config["training"]["batch_size"])
    mean, std = load_dataset_stats(dataset, fold)

    images, labels = _load_dataset_split(dataset, fold, split)

    if split == "train":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    class NumpyImageDataset(Dataset):
        def __init__(self, images, labels, transform):
            self.images = images
            self.labels = labels.astype(np.int64, copy=False)
            self.transform = transform

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            image = self.images[index].astype(np.uint8)
            pil_image = PILImage.fromarray(image, mode="RGB")
            tensor = self.transform(pil_image)
            return tensor, int(self.labels[index])

    ds = NumpyImageDataset(images, labels, transform)
    is_train = (split == "train")
    loader_kwargs = {
        "dataset": ds,
        "batch_size": batch_size,
        "shuffle": is_train,
        "drop_last": is_train,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(**loader_kwargs)


# ===================================================================
#  Phase 4: Keras 数据加载器
# ===================================================================

def get_keras_dataset(
    dataset: str,
    fold: int,
    split: str,
    config: dict,
    shuffle_seed: int = 42,
):
    """构建 Keras tf.data.Dataset (Phase 4)。"""
    setup_tf_gpu()
    import tensorflow as tf

    batch_size = int(config["training"]["batch_size"])
    mean, std = load_dataset_stats(dataset, fold)

    images, labels = _load_dataset_split(dataset, fold, split)
    images = np.ascontiguousarray(images)
    labels = np.ascontiguousarray(labels).astype(np.int64)

    is_train = (split == "train")
    n_samples = len(labels)

    # 保留 uint8 格式送入 tf.data，节省 4x 内存（对比预转 float32）
    mean_tf = tf.constant(mean, dtype=tf.float32)
    std_tf = tf.constant(std, dtype=tf.float32)

    with tf.device('/cpu:0'):
        ds = tf.data.Dataset.from_tensor_slices((images, labels))

    del images, labels  # 释放 numpy 副本

    ds = ds.cache()

    if is_train:
        ds = ds.shuffle(buffer_size=min(n_samples, 10000), seed=shuffle_seed, reshuffle_each_iteration=True)

    if is_train:
        def augment_and_normalize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - mean_tf) / std_tf
            image = tf.image.resize(image, (256, 256))
            image = tf.image.random_crop(image, size=[224, 224, 3])
            image = tf.image.random_flip_left_right(image)
            return image, label
        ds = ds.map(augment_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        def normalize_and_resize(image, label):
            image = tf.cast(image, tf.float32) / 255.0
            image = (image - mean_tf) / std_tf
            image = tf.image.resize(image, (224, 224))
            return image, label
        ds = ds.map(normalize_and_resize, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=is_train)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ===================================================================
#  Phase 5: PyTorch 训练循环
# ===================================================================

def train_pytorch(
    dataset: str,
    model_name: str,
    fold: int,
    config: dict,
    epochs_override: Optional[int] = None,
    num_workers: int = 8,
) -> None:
    """Phase 5: PyTorch 完整训练循环。"""
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score, confusion_matrix

    num_classes = get_num_classes(dataset)

    seed = 42 + fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

    batch_size = int(config["training"]["batch_size"])
    base_lr = float(config["lr"]["base_lr"])
    max_epochs = epochs_override if epochs_override else int(config["epochs"])
    patience = int(config["early_stopping"]["patience"])
    min_delta = float(config["early_stopping"]["min_delta"])

    device = setup_pytorch_gpu()
    print(f"[PyTorch] device={device}, dataset={dataset}, model={model_name}, "
          f"fold={fold}, lr={base_lr}, epochs={max_epochs}")

    train_loader = get_pytorch_dataloader(dataset, fold, "train", config, num_workers)
    val_loader = get_pytorch_dataloader(dataset, fold, "val", config, num_workers)
    test_loader = get_pytorch_dataloader(dataset, fold, "test", config, num_workers)

    model = build_pytorch_model(model_name, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=base_lr,
        momentum=float(config["optimizer"]["momentum"]),
        nesterov=bool(config["optimizer"]["nesterov"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    steps_per_epoch = len(train_loader)
    total_steps = max_epochs * steps_per_epoch
    warmup_epochs = int(config["lr"]["warmup_epochs"])
    warmup_steps = warmup_epochs * steps_per_epoch
    lr_schedule = WarmupCosineLRSchedule(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=float(config["lr"]["eta_min"]),
        warmup_start_lr=float(config["lr"]["warmup_start_lr"]),
    )

    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    logs_dir = PROJECT_ROOT / "logs"
    ckpt_dir = logs_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / f"pytorch_{dataset}_{model_name}_fold{fold}.csv"
    ckpt_path = ckpt_dir / f"pytorch_{dataset}_{model_name}_fold{fold}.pt"
    test_json_path = logs_dir / f"pytorch_{dataset}_{model_name}_fold{fold}_test.json"
    confusion_path = logs_dir / f"pytorch_{dataset}_{model_name}_fold{fold}_confusion.npy"

    csv_columns = [
        "epoch", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "val_macro_f1",
        "epoch_time_seconds", "learning_rate",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    best_val_acc = -1.0
    best_epoch = -1
    global_step = 0

    def evaluate_pt(loader):
        model.eval()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, targets)
                preds = torch.argmax(logits, dim=1)
                bs = int(targets.size(0))
                total_loss += float(loss.item()) * bs
                total_correct += int((preds == targets).sum().item())
                total_seen += bs
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(targets.cpu().numpy().tolist())
        avg_loss = total_loss / max(1, total_seen)
        acc = total_correct / max(1, total_seen)
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
        return avg_loss, acc, macro_f1, all_preds, all_targets

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        last_lr = lr_schedule.lr_at(global_step)

        train_start = time.perf_counter()

        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            current_lr = lr_schedule.lr_at(global_step)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            last_lr = current_lr

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(logits.detach(), dim=1)
            bs = int(targets.size(0))
            running_loss += float(loss.item()) * bs
            running_correct += int((preds == targets).sum().item())
            running_seen += bs
            global_step += 1

        epoch_train_time = time.perf_counter() - train_start
        train_loss = running_loss / max(1, running_seen)
        train_acc = running_correct / max(1, running_seen)

        val_loss, val_acc, val_macro_f1, _, _ = evaluate_pt(val_loader)

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
            "epoch_time_seconds": float(epoch_train_time),
            "learning_rate": float(last_lr),
        }

        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow(row)

        print(f"  [Epoch {epoch}/{max_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_f1={val_macro_f1:.4f} time={epoch_train_time:.1f}s lr={last_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
            }, ckpt_path)

        if early_stopper.step(val_loss):
            print(f"  [Early Stop] at epoch {epoch} (patience={patience})")
            break

    # --- Test evaluation ---
    print("  Loading best checkpoint for test evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_macro_f1, test_preds, test_targets = evaluate_pt(test_loader)

    cm = confusion_matrix(test_targets, test_preds)
    np.save(str(confusion_path), cm)

    with test_json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "framework": "pytorch",
            "dataset": dataset,
            "model": model_name,
            "fold": fold,
            "transfer_learning": True,
            "best_epoch_by_val_accuracy": best_epoch,
            "val_accuracy_best": best_val_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "epochs_trained": epoch,
        }, f, indent=2)

    print(f"  [Done] test_acc={test_acc:.4f}, test_f1={test_macro_f1:.4f}")
    print(f"  CSV: {csv_path}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Test JSON: {test_json_path}")
    print(f"  Confusion: {confusion_path}")


# ===================================================================
#  Phase 5: Keras 训练循环
# ===================================================================

def train_keras(
    dataset: str,
    model_name: str,
    fold: int,
    config: dict,
    epochs_override: Optional[int] = None,
) -> None:
    """Phase 5: Keras 自定义训练循环 (tf.GradientTape)。"""
    import tensorflow as tf
    from sklearn.metrics import f1_score, confusion_matrix

    setup_tf_gpu()
    num_classes = get_num_classes(dataset)

    seed = 42 + fold
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)

    batch_size = int(config["training"]["batch_size"])
    base_lr = float(config["lr"]["base_lr"])
    max_epochs = epochs_override if epochs_override else int(config["epochs"])
    patience = int(config["early_stopping"]["patience"])
    min_delta = float(config["early_stopping"]["min_delta"])
    weight_decay = float(config["optimizer"]["weight_decay"])

    print(f"[Keras] dataset={dataset}, model={model_name}, "
          f"fold={fold}, lr={base_lr}, epochs={max_epochs}")

    train_ds = get_keras_dataset(dataset, fold, "train", config, shuffle_seed=seed)
    val_ds = get_keras_dataset(dataset, fold, "val", config, shuffle_seed=seed)
    test_ds = get_keras_dataset(dataset, fold, "test", config, shuffle_seed=seed)

    # 计算 steps
    splits_dir = PROJECT_ROOT / "data" / "splits"
    _fold_local = np.load(str(splits_dir / f"{dataset}_fold{fold}_train_indices.npy"))
    n_train_samples = len(_fold_local)
    train_steps = n_train_samples // batch_size  # drop_remainder=True

    total_steps = max_epochs * train_steps
    warmup_epochs = int(config["lr"]["warmup_epochs"])
    warmup_steps = warmup_epochs * train_steps
    lr_schedule = WarmupCosineLRSchedule(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=float(config["lr"]["eta_min"]),
        warmup_start_lr=float(config["lr"]["warmup_start_lr"]),
    )

    model = build_keras_model(model_name, num_classes=num_classes, weight_decay=weight_decay)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=base_lr,
        momentum=float(config["optimizer"]["momentum"]),
        nesterov=bool(config["optimizer"]["nesterov"]),
    )

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    logs_dir = PROJECT_ROOT / "logs"
    ckpt_dir = logs_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / f"keras_{dataset}_{model_name}_fold{fold}.csv"
    ckpt_path = ckpt_dir / f"keras_{dataset}_{model_name}_fold{fold}.keras"
    test_json_path = logs_dir / f"keras_{dataset}_{model_name}_fold{fold}_test.json"
    confusion_path = logs_dir / f"keras_{dataset}_{model_name}_fold{fold}_confusion.npy"

    csv_columns = [
        "epoch", "train_loss", "train_accuracy",
        "val_loss", "val_accuracy", "val_macro_f1",
        "epoch_time_seconds", "learning_rate",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()

    best_val_acc = -1.0
    best_epoch = -1
    global_step = 0
    train_vars = model.trainable_variables

    @tf.function
    def train_step(images, labels, current_lr):
        optimizer.learning_rate.assign(current_lr)
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            ce_loss = loss_fn(labels, logits)
            reg_loss = tf.add_n(model.losses) if model.losses else 0.0
            total_loss = ce_loss + reg_loss
        grads = tape.gradient(total_loss, train_vars)
        optimizer.apply_gradients(zip(grads, train_vars))
        preds = tf.argmax(logits, axis=1, output_type=tf.int64)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32))
        return ce_loss, correct

    @tf.function
    def eval_step(images, labels):
        logits = model(images, training=False)
        loss = loss_fn(labels, logits)
        preds = tf.argmax(logits, axis=1, output_type=tf.int64)
        correct = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32))
        return loss, preds, correct

    def evaluate_keras(ds):
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        all_preds = []
        all_targets = []
        for images, labels in ds:
            loss, preds, correct = eval_step(images, labels)
            bs = int(labels.shape[0])
            total_loss += float(loss.numpy()) * bs
            total_correct += int(correct.numpy())
            total_seen += bs
            all_preds.extend(preds.numpy().tolist())
            all_targets.extend(labels.numpy().tolist())
        avg_loss = total_loss / max(1, total_seen)
        acc = total_correct / max(1, total_seen)
        macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
        return avg_loss, acc, macro_f1, all_preds, all_targets

    actual_epoch = 0
    for epoch in range(1, max_epochs + 1):
        actual_epoch = epoch
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        last_lr = lr_schedule.lr_at(global_step)

        train_start = time.perf_counter()

        for images, labels in train_ds:
            current_lr = lr_schedule.lr_at(global_step)
            last_lr = current_lr

            ce_loss, correct = train_step(images, labels, tf.constant(current_lr, dtype=tf.float32))

            bs = int(labels.shape[0])
            running_loss += float(ce_loss.numpy()) * bs
            running_correct += int(correct.numpy())
            running_seen += bs
            global_step += 1

        epoch_train_time = time.perf_counter() - train_start
        train_loss = running_loss / max(1, running_seen)
        train_acc = running_correct / max(1, running_seen)

        val_loss, val_acc, val_macro_f1, _, _ = evaluate_keras(val_ds)

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
            "epoch_time_seconds": float(epoch_train_time),
            "learning_rate": float(last_lr),
        }

        with csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns)
            writer.writerow(row)

        print(f"  [Epoch {epoch}/{max_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
              f"val_f1={val_macro_f1:.4f} time={epoch_train_time:.1f}s lr={last_lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model.save(ckpt_path, include_optimizer=False)

        if early_stopper.step(val_loss):
            print(f"  [Early Stop] at epoch {epoch} (patience={patience})")
            break

    # --- Test evaluation ---
    print("  Loading best checkpoint for test evaluation...")
    best_model = tf.keras.models.load_model(ckpt_path, compile=False, safe_mode=False)
    total_loss_t = 0.0
    total_correct_t = 0
    total_seen_t = 0
    all_preds_t = []
    all_targets_t = []
    for images, labels in test_ds:
        logits = best_model(images, training=False)
        loss = loss_fn(labels, logits)
        preds = tf.argmax(logits, axis=1, output_type=tf.int64)
        bs = int(labels.shape[0])
        total_loss_t += float(loss.numpy()) * bs
        total_correct_t += int(tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32)).numpy())
        total_seen_t += bs
        all_preds_t.extend(preds.numpy().tolist())
        all_targets_t.extend(labels.numpy().tolist())

    test_acc = total_correct_t / max(1, total_seen_t)
    test_macro_f1 = float(f1_score(all_targets_t, all_preds_t, average="macro"))

    cm = confusion_matrix(all_targets_t, all_preds_t)
    np.save(str(confusion_path), cm)

    with test_json_path.open("w", encoding="utf-8") as f:
        json.dump({
            "framework": "keras",
            "dataset": dataset,
            "model": model_name,
            "fold": fold,
            "transfer_learning": True,
            "best_epoch_by_val_accuracy": best_epoch,
            "val_accuracy_best": best_val_acc,
            "test_loss": float(total_loss_t / max(1, total_seen_t)),
            "test_accuracy": test_acc,
            "test_macro_f1": test_macro_f1,
            "epochs_trained": actual_epoch,
        }, f, indent=2)

    print(f"  [Done] test_acc={test_acc:.4f}, test_f1={test_macro_f1:.4f}")
    print(f"  CSV: {csv_path}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Test JSON: {test_json_path}")
    print(f"  Confusion: {confusion_path}")


# ===================================================================
#  Phase 5: 统一训练入口
# ===================================================================

def cmd_train(args: argparse.Namespace) -> None:
    """统一训练命令入口。"""
    config = load_config(args.config if hasattr(args, "config") and args.config else None)
    dataset = args.dataset

    if args.framework == "pytorch":
        train_pytorch(
            dataset=dataset,
            model_name=args.model,
            fold=args.fold,
            config=config,
            epochs_override=args.epochs_override,
            num_workers=args.num_workers,
        )
    elif args.framework == "keras":
        train_keras(
            dataset=dataset,
            model_name=args.model,
            fold=args.fold,
            config=config,
            epochs_override=args.epochs_override,
        )
    else:
        raise ValueError(f"Unknown framework: {args.framework}")


# ===================================================================
#  Phase 6: 结果汇总
# ===================================================================

def extract_run_summary(csv_path: Path, test_json_path: Path, config: dict) -> dict:
    """Phase 6: 从单次运行的 CSV 和 JSON 提取汇总数据。"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    with test_json_path.open("r", encoding="utf-8") as f:
        test_data = json.load(f)

    framework = test_data["framework"]
    model_name = test_data["model"]
    fold = test_data["fold"]
    batch_size = int(config["training"]["batch_size"])

    best_idx = df["val_accuracy"].idxmax()

    dataset_name = test_data.get("dataset", "plantvillage")

    splits_dir = PROJECT_ROOT / "data" / "splits"
    fold_train_path = splits_dir / f"{dataset_name}_fold{fold}_train_indices.npy"
    if fold_train_path.exists():
        total_train = len(np.load(str(fold_train_path)))
    else:
        total_train = 30000

    num_batches = total_train // batch_size
    samples_per_epoch = num_batches * batch_size

    time_per_epoch_avg = df["epoch_time_seconds"].mean()
    total_training_time = df["epoch_time_seconds"].sum()
    images_per_sec_avg = samples_per_epoch / time_per_epoch_avg if time_per_epoch_avg > 0 else 0

    return {
        "framework": framework,
        "dataset": dataset_name,
        "model": model_name,
        "fold": fold,
        "val_acc_best": float(df["val_accuracy"].max()),
        "val_f1_best": float(df["val_macro_f1"].max()),
        "test_accuracy": test_data["test_accuracy"],
        "test_macro_f1": test_data["test_macro_f1"],
        "epoch_best": int(df.loc[best_idx, "epoch"]),
        "epochs_trained": int(df["epoch"].max()),
        "time_per_epoch_avg": round(time_per_epoch_avg, 2),
        "total_training_time": round(total_training_time, 2),
        "images_per_sec_avg": round(images_per_sec_avg, 1),
    }


def cmd_aggregate(args: argparse.Namespace) -> None:
    """Phase 6: 全量结果汇总。"""
    import pandas as pd

    config = load_config()
    logs_dir = Path(args.log_dir) if args.log_dir else PROJECT_ROOT / "logs"
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "results" / "results.csv"
    summary_path = Path(args.summary_output) if args.summary_output else PROJECT_ROOT / "results" / "results_summary.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for csv_file in sorted(logs_dir.glob("*_fold*.csv")):
        if "_test" in csv_file.name:
            continue
        stem = csv_file.stem
        test_json = logs_dir / f"{stem}_test.json"
        if not test_json.exists():
            print(f"  [SKIP] No test JSON for {csv_file.name}")
            continue

        try:
            summary = extract_run_summary(csv_file, test_json, config)
            rows.append(summary)
        except Exception as e:
            print(f"  [ERROR] {csv_file.name}: {e}")

    if not rows:
        print("[WARN] No results found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"[OK] Saved {output_path} ({len(df)} rows)")

    group_cols = ["framework", "dataset", "model"]
    agg_df = df.groupby(group_cols).agg(
        mean_test_accuracy=("test_accuracy", "mean"),
        std_test_accuracy=("test_accuracy", "std"),
        mean_test_macro_f1=("test_macro_f1", "mean"),
        std_test_macro_f1=("test_macro_f1", "std"),
        mean_time_per_epoch=("time_per_epoch_avg", "mean"),
        std_time_per_epoch=("time_per_epoch_avg", "std"),
        mean_images_per_sec=("images_per_sec_avg", "mean"),
        std_images_per_sec=("images_per_sec_avg", "std"),
    ).reset_index()

    agg_df.to_csv(summary_path, index=False)
    print(f"[OK] Saved {summary_path} ({len(agg_df)} rows)")


# ===================================================================
#  Phase 7: 批量运行
# ===================================================================

def cmd_run_all(args: argparse.Namespace) -> None:
    """Phase 7: 运行全部训练。"""
    config = load_config()
    target_datasets = getattr(args, "datasets", None) or DATASETS
    combos = []
    for ds in target_datasets:
        for fw in FRAMEWORKS:
            for model_name in MODELS:
                for fold in FOLDS:
                    combos.append((ds, fw, model_name, fold))

    print(f"Total: {len(combos)} runs")
    for i, (ds, fw, mn, fold) in enumerate(combos, 1):
        print(f"\n{'='*60}")
        print(f"Run {i}/{len(combos)}: {fw} {ds} {mn} fold{fold}")
        print(f"{'='*60}")

        test_json = PROJECT_ROOT / "logs" / f"{fw}_{ds}_{mn}_fold{fold}_test.json"
        if test_json.exists():
            print(f"  [SKIP] Already completed: {test_json}")
            continue

        if fw == "pytorch":
            train_pytorch(ds, mn, fold, config)
        else:
            train_keras(ds, mn, fold, config)


# ===================================================================
#  Phase 8: 统计分析
# ===================================================================

def cmd_statistical_tests(args: argparse.Namespace) -> None:
    """Phase 8: 统计检验。"""
    import pandas as pd
    from scipy import stats as sp_stats

    analysis_dir = PROJECT_ROOT / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    results_path = PROJECT_ROOT / "results" / "results.csv"
    if not results_path.exists():
        print("[ERROR] results.csv not found. Run 'aggregate' first.")
        return

    df = pd.read_csv(results_path)

    combos = [(ds, mn) for ds in DATASETS for mn in MODELS]

    metrics = ["test_accuracy", "test_macro_f1", "images_per_sec_avg"]
    stat_rows = []

    for ds, mn in combos:
        for metric in metrics:
            keras_vals = df[(df["dataset"] == ds) & (df["model"] == mn) & (df["framework"] == "keras")]
            pytorch_vals = df[(df["dataset"] == ds) & (df["model"] == mn) & (df["framework"] == "pytorch")]

            if len(keras_vals) < 3 or len(pytorch_vals) < 3:
                print(f"  [SKIP] Insufficient data for ({ds}, {mn}, {metric})")
                continue

            k_sorted = keras_vals.sort_values("fold")[metric].values
            p_sorted = pytorch_vals.sort_values("fold")[metric].values

            t_stat, p_value = sp_stats.ttest_rel(p_sorted, k_sorted)

            diff = p_sorted - k_sorted
            d_mean = diff.mean()
            d_std = diff.std(ddof=1)
            cohens_d = d_mean / d_std if d_std > 0 else 0.0

            n1, n2 = len(p_sorted), len(k_sorted)
            more = sum(1 for a in p_sorted for b in k_sorted if a > b)
            less = sum(1 for a in p_sorted for b in k_sorted if a < b)
            cliffs_delta = (more - less) / (n1 * n2) if n1 * n2 > 0 else 0.0

            delta = float(p_sorted.mean() - k_sorted.mean())

            stat_rows.append({
                "dataset": ds,
                "model": mn,
                "metric": metric,
                "pytorch_mean": round(float(p_sorted.mean()), 6),
                "keras_mean": round(float(k_sorted.mean()), 6),
                "delta": round(delta, 6),
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(p_value), 6),
                "cohens_d": round(float(cohens_d), 4),
                "cliffs_delta": round(float(cliffs_delta), 4),
            })

    stat_df = pd.DataFrame(stat_rows)
    stat_csv = analysis_dir / "statistical_results.csv"
    stat_df.to_csv(stat_csv, index=False)
    print(f"[OK] Saved {stat_csv} ({len(stat_df)} rows)")

    # --- 显著性汇总 ---
    summary_rows = []
    for _, row in stat_df.iterrows():
        sig = "Yes" if row["p_value"] < 0.05 else "No"
        d_abs = abs(row["cohens_d"])
        if d_abs < 0.2:
            effect = "Negligible"
        elif d_abs < 0.5:
            effect = "Small"
        elif d_abs < 0.8:
            effect = "Medium"
        else:
            effect = "Large"

        summary_rows.append({
            "dataset": row["dataset"],
            "model": row["model"],
            "metric": row["metric"],
            "p_value": row["p_value"],
            "significant_alpha_0.05": sig,
            "cohens_d": row["cohens_d"],
            "effect_size": effect,
            "cliffs_delta": row["cliffs_delta"],
            "delta_pytorch_minus_keras": row["delta"],
        })

    sig_df = pd.DataFrame(summary_rows)
    sig_csv = analysis_dir / "significance_summary.csv"
    sig_df.to_csv(sig_csv, index=False)

    sig_md = analysis_dir / "significance_summary.md"
    with sig_md.open("w", encoding="utf-8") as f:
        f.write("# Significance Summary\n\n")
        f.write("| Dataset | Model | Metric | p-value | Significant (a=0.05) | Cohen's d | Effect Size | Cliff's d | D (PT-Keras) |\n")
        f.write("|---------|-------|--------|---------|---------------------|-----------|-------------|-----------|-------------|\n")
        for _, r in sig_df.iterrows():
            f.write(f"| {r['dataset']} | {r['model']} | {r['metric']} | "
                    f"{r['p_value']:.6f} | {r['significant_alpha_0.05']} | "
                    f"{r['cohens_d']:.4f} | {r['effect_size']} | "
                    f"{r['cliffs_delta']:.4f} | {r['delta_pytorch_minus_keras']:.6f} |\n")

    print(f"[OK] Saved {sig_csv}")
    print(f"[OK] Saved {sig_md}")


# ===================================================================
#  Phase 9: 可视化
# ===================================================================

def cmd_plot_curves(args: argparse.Namespace) -> None:
    """Phase 9: 训练曲线图。"""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = PROJECT_ROOT / "logs"

    for ds in DATASETS:
        for mn in MODELS:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.suptitle(f"{ds} — {mn}", fontsize=14)

            for fw in FRAMEWORKS:
                color = "tab:blue" if fw == "pytorch" else "tab:orange"
                for fold in FOLDS:
                    csv_path = logs_dir / f"{fw}_{ds}_{mn}_fold{fold}.csv"
                    if not csv_path.exists():
                        continue
                    df = pd.read_csv(csv_path)
                    alpha = 0.3
                    axes[0, 0].plot(df["epoch"], df["train_loss"], color=color, alpha=alpha, linewidth=0.8)
                    axes[0, 1].plot(df["epoch"], df["val_accuracy"], color=color, alpha=alpha, linewidth=0.8)
                    axes[1, 0].plot(df["epoch"], df["val_macro_f1"], color=color, alpha=alpha, linewidth=0.8)
                    axes[1, 1].plot(df["epoch"], df["learning_rate"], color=color, alpha=alpha, linewidth=0.8)

                all_dfs = []
                for fold in FOLDS:
                    csv_path = logs_dir / f"{fw}_{ds}_{mn}_fold{fold}.csv"
                    if csv_path.exists():
                        all_dfs.append(pd.read_csv(csv_path))
                if all_dfs:
                    min_len = min(len(d) for d in all_dfs)
                    trimmed = [d.iloc[:min_len] for d in all_dfs]
                    mean_df = pd.concat(trimmed).groupby("epoch").mean().reset_index()
                    axes[0, 0].plot(mean_df["epoch"], mean_df["train_loss"], color=color, linewidth=2, label=fw)
                    axes[0, 1].plot(mean_df["epoch"], mean_df["val_accuracy"], color=color, linewidth=2, label=fw)
                    axes[1, 0].plot(mean_df["epoch"], mean_df["val_macro_f1"], color=color, linewidth=2, label=fw)
                    axes[1, 1].plot(mean_df["epoch"], mean_df["learning_rate"], color=color, linewidth=2, label=fw)

            axes[0, 0].set_title("Train Loss"); axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss"); axes[0, 0].legend()
            axes[0, 1].set_title("Val Accuracy"); axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Accuracy"); axes[0, 1].legend()
            axes[1, 0].set_title("Val Macro-F1"); axes[1, 0].set_xlabel("Epoch"); axes[1, 0].set_ylabel("Macro-F1"); axes[1, 0].legend()
            axes[1, 1].set_title("Learning Rate"); axes[1, 1].set_xlabel("Epoch"); axes[1, 1].set_ylabel("LR"); axes[1, 1].legend()

            plt.tight_layout()
            out_path = plots_dir / f"curves_{ds}_{mn}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"[OK] Saved {out_path}")


def cmd_plot_bars(args: argparse.Namespace) -> None:
    """Phase 9: 框架对比柱状图。"""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = PROJECT_ROOT / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_path = PROJECT_ROOT / "results" / "results_summary.csv"
    if not summary_path.exists():
        print("[ERROR] results_summary.csv not found. Run 'aggregate' first.")
        return

    df = pd.read_csv(summary_path)

    for metric_col, std_col, ylabel, title_suffix, filename_suffix in [
        ("mean_test_accuracy", "std_test_accuracy", "Test Accuracy", "Accuracy Comparison", "comparison_accuracy.png"),
        ("mean_images_per_sec", "std_images_per_sec", "Images/sec", "Training Speed Comparison", "comparison_speed.png"),
    ]:
        # 所有数据集的模型组合
        combo_labels = []
        for ds in DATASETS:
            for mn in MODELS:
                combo_labels.append(f"{ds}\n{mn}")

        fig, ax = plt.subplots(figsize=(max(10, len(combo_labels) * 2), 6))
        x = np.arange(len(combo_labels))
        width = 0.35

        keras_vals, keras_errs = [], []
        pytorch_vals, pytorch_errs = [], []

        for ds in DATASETS:
            for mn in MODELS:
                for fw, vals, errs in [("keras", keras_vals, keras_errs), ("pytorch", pytorch_vals, pytorch_errs)]:
                    row = df[(df["dataset"] == ds) & (df["model"] == mn) & (df["framework"] == fw)]
                    if len(row) > 0:
                        vals.append(float(row[metric_col].values[0]))
                        errs.append(float(row[std_col].values[0]))
                    else:
                        vals.append(0)
                        errs.append(0)

        ax.bar(x - width/2, keras_vals, width, yerr=keras_errs, label="Keras", color="tab:orange", capsize=3)
        ax.bar(x + width/2, pytorch_vals, width, yerr=pytorch_errs, label="PyTorch", color="tab:blue", capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels(combo_labels, fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title_suffix)
        ax.legend()
        plt.tight_layout()

        out_path = plots_dir / filename_suffix
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved {out_path}")


# ===================================================================
#  Shell 脚本生成
# ===================================================================

def cmd_generate_scripts(args: argparse.Namespace) -> None:
    """生成 run_all.sh 和分框架并行脚本。"""
    scripts_dir = PROJECT_ROOT / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # 分框架脚本（可用双 GPU 并行执行）
    for fw in FRAMEWORKS:
        lines = ["#!/bin/bash", "set -e", "",
                 f"# Transfer Learning Benchmark — {fw}", ""]
        for ds in DATASETS:
            for mn in MODELS:
                for fold in FOLDS:
                    cmd = (f"python ass2_code.py train --framework {fw} "
                           f"--dataset {ds} --model {mn} --fold {fold}")
                    lines.append(f'echo "=== {fw} {ds} {mn} fold{fold} ==="')
                    lines.append(cmd)
                    lines.append("")
        lines.append(f'echo "{fw} runs complete."')
        fw_script = scripts_dir / f"run_{fw}.sh"
        with fw_script.open("w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        fw_script.chmod(0o755)
        print(f"[OK] Generated {fw_script}")

    # 合并脚本
    total_runs = len(DATASETS) * len(FRAMEWORKS) * len(MODELS) * len(FOLDS)
    lines = ["#!/bin/bash", "set -e", "",
             f"# Transfer Learning Benchmark ({total_runs} runs)",
             "# 双 GPU 并行: bash scripts/run_pytorch.sh & bash scripts/run_keras.sh & wait", ""]
    for ds in DATASETS:
        for fw in FRAMEWORKS:
            for mn in MODELS:
                for fold in FOLDS:
                    cmd = (f"python ass2_code.py train --framework {fw} "
                           f"--dataset {ds} --model {mn} --fold {fold}")
                    lines.append(f'echo "=== {fw} {ds} {mn} fold{fold} ==="')
                    lines.append(cmd)
                    lines.append("")

    lines.append('echo "All runs complete."')
    lines.append("")
    lines.append("# Auto-aggregate")
    lines.append("python ass2_code.py aggregate")

    script_path = scripts_dir / "run_all.sh"
    with script_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    print(f"[OK] Generated {script_path}")


# ===================================================================
#  主入口: CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Keras vs PyTorch Benchmark — Transfer Learning (PlantVillage + CIFAR-10)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    prep_parser = subparsers.add_parser("prepare_data", help="Phase 1: Download data + generate folds + compute stats")
    prep_parser.add_argument("--dataset", default="all", choices=["all", "plantvillage", "cifar10"])

    subparsers.add_parser("generate_configs", help="Phase 2: Generate YAML config files")
    subparsers.add_parser("param_comparison", help="Phase 3: Parameter alignment table")

    train_parser = subparsers.add_parser("train", help="Phase 5: Train a single run")
    train_parser.add_argument("--framework", required=True, choices=["keras", "pytorch"])
    train_parser.add_argument("--dataset", required=True, choices=["plantvillage", "cifar10"])
    train_parser.add_argument("--model", required=True, choices=["resnet50", "vgg16", "mobilenetv2"])
    train_parser.add_argument("--fold", required=True, type=int, choices=[0, 1, 2])
    train_parser.add_argument("--config", default=None)
    train_parser.add_argument("--epochs_override", type=int, default=None)
    train_parser.add_argument("--num_workers", type=int, default=8)

    agg_parser = subparsers.add_parser("aggregate", help="Phase 6: Aggregate results")
    agg_parser.add_argument("--log_dir", default=None)
    agg_parser.add_argument("--output", default=None)
    agg_parser.add_argument("--summary_output", default=None)

    subparsers.add_parser("run_all", help="Phase 7: Run all experiments")

    subparsers.add_parser("statistical_tests", help="Phase 8: Statistical analysis")

    subparsers.add_parser("plot_curves", help="Phase 9: Training curves")
    subparsers.add_parser("plot_bars", help="Phase 9: Comparison bar charts")

    subparsers.add_parser("generate_scripts", help="Generate shell script for all runs")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    cmd_map = {
        "prepare_data": cmd_prepare_data,
        "generate_configs": cmd_generate_configs,
        "param_comparison": cmd_param_comparison,
        "train": cmd_train,
        "aggregate": cmd_aggregate,
        "run_all": cmd_run_all,
        "statistical_tests": cmd_statistical_tests,
        "plot_curves": cmd_plot_curves,
        "plot_bars": cmd_plot_bars,
        "generate_scripts": cmd_generate_scripts,
    }

    cmd_map[args.command](args)


if __name__ == "__main__":
    main()
