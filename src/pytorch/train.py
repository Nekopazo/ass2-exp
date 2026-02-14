"""PyTorch training loop for Phase 5."""

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
import torch
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
from src.pytorch.data import get_dataloader
from src.pytorch.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch training script")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "tiny_imagenet"])
    parser.add_argument("--model", required=True, choices=["mobilenetv2", "resnet50", "convnext_tiny"])
    parser.add_argument("--precision", required=True, choices=["fp32", "amp"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "train_config.yaml"))
    parser.add_argument("--epochs_override", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    all_preds: list[int] = []
    all_targets: list[int] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)

            preds = torch.argmax(logits, dim=1)
            total_loss += float(loss.item()) * int(targets.size(0))
            total_correct += int((preds == targets).sum().item())
            total_seen += int(targets.size(0))

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(targets.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, total_seen)
    acc = total_correct / max(1, total_seen)
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro"))
    return avg_loss, acc, macro_f1


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(args.seed)

    dataset = args.dataset
    model_name = args.model
    precision = args.precision

    batch_size = int(config["training"]["batch_size"])
    grad_accum_steps = int(config["training"]["gradient_accumulation_steps"])
    if grad_accum_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be > 0")

    num_classes = DATASET_NUM_CLASSES[dataset]
    input_size = get_dataset_input_size(dataset, config)
    total_epochs = get_total_epochs(dataset, config, args.epochs_override)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if precision == "amp" and device.type != "cuda":
        print("[warn] AMP requested but CUDA is unavailable, falling back to fp32.")
        precision = "fp32"

    train_loader = get_dataloader(
        dataset=dataset,
        split="train",
        config=config,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    val_loader = get_dataloader(
        dataset=dataset,
        split="val",
        config=config,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    test_loader = get_dataloader(
        dataset=dataset,
        split="test",
        config=config,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )

    model = build_model(model_name=model_name, num_classes=num_classes, input_size=input_size).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["lr"]["base_lr"]),
        momentum=float(config["optimizer"]["momentum"]),
        nesterov=bool(config["optimizer"]["nesterov"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )

    micro_batches_per_epoch = len(train_loader)
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

    use_amp = precision == "amp" and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logs_dir = PROJECT_ROOT / "logs"
    ckpt_dir = logs_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_path = logs_dir / f"pytorch_{dataset}_{model_name}_{precision}_seed{args.seed}.csv"
    ckpt_path = ckpt_dir / f"pytorch_{dataset}_{model_name}_{precision}_seed{args.seed}.pt"
    test_json_path = logs_dir / f"pytorch_{dataset}_{model_name}_{precision}_seed{args.seed}_test.json"

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

    for epoch in range(1, total_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_start_lr = lr_schedule.lr_at(global_step)
        print(f"epoch={epoch} start_lr={epoch_start_lr:.10f}")

        train_start = time.perf_counter()
        running_loss = 0.0
        running_correct = 0
        running_seen = 0
        accum_counter = 0
        last_lr = epoch_start_lr

        for batch_idx, (images, targets) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            current_lr = lr_schedule.lr_at(global_step)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
            last_lr = current_lr

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()

            preds = torch.argmax(logits.detach(), dim=1)
            batch_size_actual = int(targets.size(0))
            running_loss += float(loss.item()) * batch_size_actual
            running_correct += int((preds == targets).sum().item())
            running_seen += batch_size_actual

            accum_counter += 1
            is_last_batch = batch_idx == micro_batches_per_epoch
            if accum_counter == grad_accum_steps or is_last_batch:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                accum_counter = 0
                global_step += 1

        epoch_train_time = time.perf_counter() - train_start

        train_loss = running_loss / max(1, running_seen)
        train_acc = running_correct / max(1, running_seen)
        val_loss, val_acc, val_macro_f1 = evaluate(model, val_loader, criterion, device)

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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_macro_f1,
                    "seed": args.seed,
                },
                ckpt_path,
            )

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in history:
            output_row = row.copy()
            output_row["epoch"] = int(output_row["epoch"])
            writer.writerow(output_row)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc, test_macro_f1 = evaluate(model, test_loader, criterion, device)

    with test_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "framework": "pytorch",
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
