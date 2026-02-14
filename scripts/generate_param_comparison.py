#!/usr/bin/env python3
"""Generate Phase 3 parameter comparison table and run basic model checks."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import tensorflow as tf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.keras.models import build_model as build_keras_model
from src.keras.models import count_parameters as count_keras_parameters
from src.pytorch.models import build_model as build_pytorch_model
from src.pytorch.models import count_parameters as count_pytorch_parameters

OUTPUT_CSV = PROJECT_ROOT / "results" / "param_comparison.csv"

MODELS = ("mobilenetv2", "resnet50", "convnext_tiny")
INPUT_SIZES = (32, 64)
NUM_CLASSES_LIST = (10, 100, 200)


def _validate_pytorch_forward(model_name: str, num_classes: int, input_size: int) -> int:
    model = build_pytorch_model(model_name=model_name, num_classes=num_classes, input_size=input_size)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, input_size, input_size)
        y = model(x)
    if tuple(y.shape) != (1, num_classes):
        raise AssertionError(
            f"PyTorch output shape mismatch: {model_name}, input={input_size}, "
            f"classes={num_classes}, got={tuple(y.shape)}"
        )

    if model_name == "resnet50" and input_size == 64:
        with torch.no_grad():
            feat = model.relu(model.bn1(model.conv1(x)))
            feat = model.maxpool(feat)
            feat = model.layer1(feat)
            feat = model.layer2(feat)
            feat = model.layer3(feat)
            feat = model.layer4(feat)
        if tuple(feat.shape[-2:]) != (4, 4):
            raise AssertionError(f"PyTorch ResNet50 tiny feature map is not 4x4: {tuple(feat.shape)}")

    return count_pytorch_parameters(model)


def _validate_keras_forward(model_name: str, num_classes: int, input_size: int) -> int:
    model = build_keras_model(model_name=model_name, num_classes=num_classes, input_size=input_size)
    x = tf.random.normal((1, input_size, input_size, 3))
    y = model(x, training=False)
    if tuple(y.shape) != (1, num_classes):
        raise AssertionError(
            f"Keras output shape mismatch: {model_name}, input={input_size}, "
            f"classes={num_classes}, got={tuple(y.shape)}"
        )

    if model_name == "resnet50" and input_size == 64:
        layer_name = "conv5_block3_out"
        feat_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        feat = feat_model(x, training=False)
        if tuple(feat.shape[1:3]) != (4, 4):
            raise AssertionError(f"Keras ResNet50 tiny feature map is not 4x4: {tuple(feat.shape)}")

    params = count_keras_parameters(model)
    tf.keras.backend.clear_session()
    return params


def main() -> None:
    rows: list[dict[str, float | int | str]] = []

    for model_name in MODELS:
        for input_size in INPUT_SIZES:
            for num_classes in NUM_CLASSES_LIST:
                pt_params = _validate_pytorch_forward(model_name, num_classes, input_size)
                k_params = _validate_keras_forward(model_name, num_classes, input_size)
                diff_pct = abs(pt_params - k_params) / pt_params * 100.0
                rows.append(
                    {
                        "model": model_name,
                        "input_size": input_size,
                        "num_classes": num_classes,
                        "pytorch_params": pt_params,
                        "keras_params": k_params,
                        "diff_pct": round(diff_pct, 6),
                    }
                )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "input_size",
                "num_classes",
                "pytorch_params",
                "keras_params",
                "diff_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Constraint checks from Phase 3.7
    for row in rows:
        if row["model"] in {"mobilenetv2", "resnet50"} and row["diff_pct"] >= 1.0:
            raise AssertionError(f"diff_pct >= 1% for {row}")
        if row["model"] == "convnext_tiny" and row["diff_pct"] >= 5.0:
            raise AssertionError(f"diff_pct >= 5% for {row}")

    print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
