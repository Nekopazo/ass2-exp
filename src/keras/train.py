"""Keras training loop for Phase 5."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common import (
    DATASET_NUM_CLASSES,
    WarmupCosineLRSchedule,
    get_dataset_input_size,
    get_total_epochs,
    optimizer_steps_per_epoch,
)
from src.keras.data import get_tf_dataset, load_split_arrays
from src.keras.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keras training script")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "tiny_imagenet"])
    parser.add_argument("--model", required=True, choices=["mobilenetv2", "resnet50", "convnext_tiny"])
    parser.add_argument("--precision", required=True, choices=["fp32", "amp"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "train_config.yaml"))
    parser.add_argument("--epochs_override", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[float, float, float]:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    for images, labels in dataset:
        logits = model(images, training=False)
        loss = loss_fn(labels, logits)

        preds = tf.argmax(logits, axis=1, output_type=tf.int64)
        batch_size_actual = int(labels.shape[0])

        total_loss += float(loss.numpy()) * batch_size_actual
        total_correct += int(tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32)).numpy())
        total_seen += batch_size_actual

        all_preds.extend(preds.numpy().tolist())
        all_targets.extend(labels.numpy().tolist())

    avg_loss = total_loss / max(1, total_seen)
    acc = total_correct / max(1, total_seen)
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    return avg_loss, acc, macro_f1


def zero_like_vars(variables: list[tf.Variable]) -> list[tf.Tensor]:
    return [tf.zeros_like(v, dtype=tf.float32) for v in variables]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    precision = args.precision
    if precision == "amp":
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")

    set_seed(args.seed)

    dataset = args.dataset
    model_name = args.model

    batch_size = int(config["training"]["batch_size"])
    grad_accum_steps = int(config["training"]["gradient_accumulation_steps"])
    if grad_accum_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")

    num_classes = DATASET_NUM_CLASSES[dataset]
    input_size = get_dataset_input_size(dataset, config)
    total_epochs = get_total_epochs(dataset, config, args.epochs_override)

    train_images, _ = load_split_arrays(dataset=dataset, split="train", project_root=PROJECT_ROOT)
    micro_batches_per_epoch = int(train_images.shape[0]) // batch_size
    update_steps_per_epoch = optimizer_steps_per_epoch(micro_batches_per_epoch, grad_accum_steps)
    total_steps = total_epochs * update_steps_per_epoch
    warmup_steps = int(config["lr"]["warmup_epochs"]) * update_steps_per_epoch
    lr_schedule = WarmupCosineLRSchedule(
        base_lr=float(config["lr"]["base_lr"]),
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=float(config["lr"].get("eta_min", 0.0)),
        warmup_start_lr=float(config["lr"].get("warmup_start_lr", 0.0)),
    )

    train_ds = get_tf_dataset(
        dataset=dataset,
        split="train",
        config=config,
        project_root=PROJECT_ROOT,
        batch_size=batch_size,
        shuffle_seed=args.seed,
    )
    val_ds = get_tf_dataset(
        dataset=dataset,
        split="val",
        config=config,
        project_root=PROJECT_ROOT,
        batch_size=batch_size,
        shuffle_seed=args.seed,
    )
    test_ds = get_tf_dataset(
        dataset=dataset,
        split="test",
        config=config,
        project_root=PROJECT_ROOT,
        batch_size=batch_size,
        shuffle_seed=args.seed,
    )

    model = build_model(model_name=model_name, num_classes=num_classes, input_size=input_size)

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=float(config["lr"]["base_lr"]),
        momentum=float(config["optimizer"]["momentum"]),
        nesterov=bool(config["optimizer"]["nesterov"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    if precision == "amp":
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    logs_dir = PROJECT_ROOT / "logs"
    ckpt_dir = logs_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / f"keras_{dataset}_{model_name}_{precision}_seed{args.seed}.csv"
    ckpt_path = ckpt_dir / f"keras_{dataset}_{model_name}_{precision}_seed{args.seed}.h5"
    test_json_path = logs_dir / f"keras_{dataset}_{model_name}_{precision}_seed{args.seed}_test.json"

    csv_columns = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "epoch_time_seconds",
        "learning_rate",
    ]

    history: list[dict[str, float]] = []
    best_val_acc = -1.0
    best_epoch = -1
    global_step = 0

    train_vars = model.trainable_variables

    for epoch in range(1, total_epochs + 1):
        epoch_start_lr = lr_schedule.lr_at(global_step)
        print(f"epoch={epoch} start_lr={epoch_start_lr:.10f}")

        grad_sums = zero_like_vars(train_vars)
        accum_counter = 0
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        last_lr = epoch_start_lr

        train_start = time.perf_counter()

        for batch_idx, (images, labels) in enumerate(train_ds, start=1):
            current_lr = lr_schedule.lr_at(global_step)
            inner_optimizer = optimizer.inner_optimizer if hasattr(optimizer, "inner_optimizer") else optimizer
            inner_optimizer.learning_rate.assign(current_lr)
            last_lr = current_lr

            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = loss_fn(labels, logits)
                scaled_loss = loss / float(grad_accum_steps)
                if hasattr(optimizer, "scale_loss"):
                    scaled_loss = optimizer.scale_loss(scaled_loss)

            grads = tape.gradient(scaled_loss, train_vars)

            grads = [
                tf.zeros_like(v, dtype=tf.float32) if g is None else tf.cast(g, tf.float32)
                for g, v in zip(grads, train_vars)
            ]
            grad_sums = [acc + g for acc, g in zip(grad_sums, grads)]

            preds = tf.argmax(logits, axis=1, output_type=tf.int64)
            batch_size_actual = int(labels.shape[0])
            running_loss += float(loss.numpy()) * batch_size_actual
            running_correct += int(tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.int32)).numpy())
            running_seen += batch_size_actual

            accum_counter += 1
            is_last_batch = batch_idx == micro_batches_per_epoch
            if accum_counter == grad_accum_steps or is_last_batch:
                optimizer.apply_gradients(zip(grad_sums, train_vars))
                grad_sums = zero_like_vars(train_vars)
                accum_counter = 0
                global_step += 1

        epoch_train_time = time.perf_counter() - train_start

        train_loss = running_loss / max(1, running_seen)
        train_acc = running_correct / max(1, running_seen)
        val_loss, val_acc, val_macro_f1 = evaluate(model, val_ds)

        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_macro_f1),
            "epoch_time_seconds": float(epoch_train_time),
            "learning_rate": float(last_lr),
        }
        history.append(row)

        print(
            " ".join(
                [
                    f"train_loss={train_loss:.6f}",
                    f"train_acc={train_acc:.6f}",
                    f"val_loss={val_loss:.6f}",
                    f"val_acc={val_acc:.6f}",
                    f"val_f1={val_macro_f1:.6f}",
                    f"time={epoch_train_time:.2f}s",
                    f"lr={last_lr:.8f}",
                ]
            )
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model.save(ckpt_path, include_optimizer=False)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in history:
            output_row = row.copy()
            output_row["epoch"] = int(output_row["epoch"])
            writer.writerow(output_row)

    best_model = tf.keras.models.load_model(ckpt_path, compile=False, safe_mode=False)
    test_loss, test_acc, test_macro_f1 = evaluate(best_model, test_ds)

    with test_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "framework": "keras",
                "dataset": dataset,
                "model": model_name,
                "precision": precision,
                "seed": args.seed,
                "best_epoch_by_val_accuracy": best_epoch,
                "val_accuracy_best": best_val_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_macro_f1": test_macro_f1,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    print(f"saved_csv={csv_path}")
    print(f"saved_ckpt={ckpt_path}")
    print(f"saved_test={test_json_path}")


if __name__ == "__main__":
    main()
