"""Model builders for PyTorch experiments (Phase 3)."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
from torchvision.models import (
    convnext_tiny,
    mobilenet_v2,
    resnet50,
)

ModelName = Literal["mobilenetv2", "resnet50", "convnext_tiny"]


def build_model(model_name: ModelName, num_classes: int, input_size: int) -> nn.Module:
    """Build a model for the given dataset setup.

    Args:
        model_name: One of mobilenetv2/resnet50/convnext_tiny.
        num_classes: Number of classes in target dataset.
        input_size: Input image size (expected 32 for CIFAR or 64 for Tiny-ImageNet).
    """
    if model_name == "mobilenetv2":
        return build_mobilenetv2(num_classes=num_classes, input_size=input_size)
    if model_name == "resnet50":
        return build_resnet50(num_classes=num_classes, input_size=input_size)
    if model_name == "convnext_tiny":
        return build_convnext_tiny(num_classes=num_classes, input_size=input_size)
    raise ValueError(f"Unsupported model_name: {model_name}")


def build_mobilenetv2(num_classes: int, input_size: int) -> nn.Module:
    model = mobilenet_v2(weights=None)

    if input_size == 32:
        first_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=first_conv.in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def build_resnet50(num_classes: int, input_size: int) -> nn.Module:
    model = resnet50(weights=None)

    if input_size == 32:
        model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    # Tiny-ImageNet (64x64) keeps default conv1 but still removes maxpool.
    # CIFAR also removes maxpool to avoid aggressive early downsampling.
    model.maxpool = nn.Identity()

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_convnext_tiny(num_classes: int, input_size: int) -> nn.Module:
    model = convnext_tiny(weights=None)

    stem_conv = model.features[0][0]
    if input_size == 32:
        model.features[0][0] = nn.Conv2d(
            in_channels=stem_conv.in_channels,
            out_channels=stem_conv.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )
    elif input_size == 64:
        model.features[0][0] = nn.Conv2d(
            in_channels=stem_conv.in_channels,
            out_channels=stem_conv.out_channels,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )

    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
