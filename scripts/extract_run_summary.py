#!/usr/bin/env python3
"""Extract one-run summary metrics from an epoch-level training log CSV."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_NAME_RE = re.compile(
    r"^(?P<framework>keras|pytorch)_(?P<dataset>cifar10|cifar100|tiny_imagenet)_"
    r"(?P<model>mobilenetv2|resnet50|convnext_tiny)_(?P<precision>fp32|amp)_seed(?P<seed>\d+)\.csv$"
)

TRAIN_SPLIT_SIZES = {
    "cifar10": 45000,
    "cifar100": 45000,
    "tiny_imagenet": 90000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a one-line run summary from a training CSV log.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to one training CSV log file.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "train_config.yaml",
        help="Training config path (used for batch_size).",
    )
    parser.add_argument("--as-json", action="store_true", help="Print JSON instead of CSV header+row.")
    return parser.parse_args()


def parse_run_name(csv_path: Path) -> dict[str, Any]:
    match = RUN_NAME_RE.match(csv_path.name)
    if not match:
        raise ValueError(f"Unexpected log filename format: {csv_path.name}")
    info = match.groupdict()
    info["seed"] = int(info["seed"])
    return info


def _read_batch_size(config_path: Path) -> int:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to read batch_size from config.") from exc

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    batch_size = int(config["training"]["batch_size"])
    if batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")
    return batch_size


def _late_half(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    start_idx = n // 2
    return df.iloc[start_idx:]


def _read_test_metrics(csv_path: Path) -> tuple[float, float]:
    stem = csv_path.stem
    test_path = csv_path.with_name(f"{stem}_test.json")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test metrics file: {test_path}")
    with test_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return float(payload["test_accuracy"]), float(payload["test_macro_f1"])


def summarize_run(csv_path: Path, config_path: Path) -> dict[str, Any]:
    run = parse_run_name(csv_path)
    df = pd.read_csv(csv_path)
    required_cols = {
        "epoch",
        "val_accuracy",
        "val_macro_f1",
        "epoch_time_seconds",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"Empty CSV log: {csv_path}")

    batch_size = _read_batch_size(config_path)
    train_size = TRAIN_SPLIT_SIZES[run["dataset"]]
    num_batches = train_size // batch_size
    images_per_epoch = num_batches * batch_size

    best_acc_idx = int(df["val_accuracy"].idxmax())
    acc_best = float(df.loc[best_acc_idx, "val_accuracy"])
    epoch_best = int(df.loc[best_acc_idx, "epoch"])
    f1_best = float(df["val_macro_f1"].max())

    test_accuracy, test_macro_f1 = _read_test_metrics(csv_path)

    avg_time = float(df["epoch_time_seconds"].mean())
    late_df = _late_half(df)
    avg_time_late = float(late_df["epoch_time_seconds"].mean())

    if avg_time <= 0.0 or avg_time_late <= 0.0:
        raise ValueError("epoch_time_seconds contains non-positive average; cannot compute throughput.")

    images_per_sec_avg = float(images_per_epoch / avg_time)
    images_per_sec_avg_late = float(images_per_epoch / avg_time_late)

    if not (0.0 <= acc_best <= 1.0 and 0.0 <= f1_best <= 1.0 and 0.0 <= test_accuracy <= 1.0 and 0.0 <= test_macro_f1 <= 1.0):
        raise ValueError("One or more metric values are outside [0, 1].")
    if not (math.isfinite(images_per_sec_avg) and math.isfinite(images_per_sec_avg_late)):
        raise ValueError("Throughput is non-finite.")

    return {
        "framework": run["framework"],
        "dataset": run["dataset"],
        "model": run["model"],
        "precision": run["precision"],
        "seed": run["seed"],
        "acc_best": acc_best,
        "f1_best": f1_best,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
        "epoch_best": epoch_best,
        "time_per_epoch_avg": avg_time,
        "time_per_epoch_avg_late": avg_time_late,
        "images_per_sec_avg": images_per_sec_avg,
        "images_per_sec_avg_late": images_per_sec_avg_late,
    }


def main() -> None:
    args = parse_args()
    summary = summarize_run(csv_path=args.csv, config_path=args.config)

    if args.as_json:
        print(json.dumps(summary, ensure_ascii=True))
        return

    columns = list(summary.keys())
    print(",".join(columns))
    print(",".join(str(summary[col]) for col in columns))


if __name__ == "__main__":
    main()
