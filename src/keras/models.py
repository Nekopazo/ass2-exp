"""Model builders for Keras experiments (Phase 3)."""

from __future__ import annotations

from typing import Literal

import tensorflow as tf

ModelName = Literal["mobilenetv2", "resnet50", "convnext_tiny"]


def _clone_with_layer_rewrites(
    base_model: tf.keras.Model,
    rewrite_fn,
) -> tf.keras.Model:
    def clone_function(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        rewritten = rewrite_fn(layer)
        if rewritten is not None:
            return rewritten
        return layer.__class__.from_config(layer.get_config())

    return tf.keras.models.clone_model(base_model, clone_function=clone_function)


def _identity_from(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    return tf.keras.layers.Lambda(lambda x: x, name=layer.name)


def build_model(model_name: ModelName, num_classes: int, input_size: int) -> tf.keras.Model:
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


def build_mobilenetv2(num_classes: int, input_size: int) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=None,
        input_shape=(input_size, input_size, 3),
        pooling="avg",
    )

    if input_size == 32:

        def rewrite(layer: tf.keras.layers.Layer):
            if layer.name == "Conv1" and isinstance(layer, tf.keras.layers.Conv2D):
                cfg = layer.get_config()
                cfg["strides"] = (1, 1)
                cfg["padding"] = "same"
                return tf.keras.layers.Conv2D.from_config(cfg)
            return None

        base = _clone_with_layer_rewrites(base, rewrite)

    outputs = tf.keras.layers.Dense(num_classes, name="predictions")(base.output)
    return tf.keras.Model(inputs=base.input, outputs=outputs, name=f"mobilenetv2_{input_size}")


def build_resnet50(num_classes: int, input_size: int) -> tf.keras.Model:
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(input_size, input_size, 3),
        pooling="avg",
    )

    def rewrite(layer: tf.keras.layers.Layer):
        if input_size == 32 and layer.name == "conv1_pad":
            return _identity_from(layer)
        if input_size == 32 and layer.name == "conv1_conv" and isinstance(layer, tf.keras.layers.Conv2D):
            cfg = layer.get_config()
            cfg["kernel_size"] = (3, 3)
            cfg["strides"] = (1, 1)
            cfg["padding"] = "same"
            return tf.keras.layers.Conv2D.from_config(cfg)
        if layer.name in {"pool1_pad", "pool1_pool"}:
            return _identity_from(layer)
        return None

    base = _clone_with_layer_rewrites(base, rewrite)
    outputs = tf.keras.layers.Dense(num_classes, name="predictions")(base.output)
    return tf.keras.Model(inputs=base.input, outputs=outputs, name=f"resnet50_{input_size}")


def build_convnext_tiny(num_classes: int, input_size: int) -> tf.keras.Model:
    base = tf.keras.applications.ConvNeXtTiny(
        include_top=False,
        weights=None,
        input_shape=(input_size, input_size, 3),
        pooling="avg",
    )

    def rewrite(layer: tf.keras.layers.Layer):
        if layer.name == "convnext_tiny_stem_conv" and isinstance(layer, tf.keras.layers.Conv2D):
            cfg = layer.get_config()
            if input_size == 32:
                cfg["kernel_size"] = (3, 3)
                cfg["strides"] = (1, 1)
                cfg["padding"] = "same"
            elif input_size == 64:
                cfg["kernel_size"] = (2, 2)
                cfg["strides"] = (2, 2)
                cfg["padding"] = "valid"
            return tf.keras.layers.Conv2D.from_config(cfg)
        return None

    base = _clone_with_layer_rewrites(base, rewrite)
    outputs = tf.keras.layers.Dense(num_classes, name="predictions")(base.output)
    return tf.keras.Model(inputs=base.input, outputs=outputs, name=f"convnext_tiny_{input_size}")


def count_parameters(model: tf.keras.Model) -> int:
    return int(
        sum(
            tf.keras.backend.count_params(weight)
            for weight in model.trainable_weights
        )
    )
