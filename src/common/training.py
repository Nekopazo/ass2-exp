"""Shared training helpers for Phase 5 scripts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

DATASET_NUM_CLASSES: dict[str, int] = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
}


def get_dataset_input_size(dataset: str, config: Mapping[str, object]) -> int:
    input_cfg = config["input_size"]
    if dataset in {"cifar10", "cifar100"}:
        return int(input_cfg["cifar"])
    if dataset == "tiny_imagenet":
        return int(input_cfg["tiny_imagenet"])
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_total_epochs(dataset: str, config: Mapping[str, object], epochs_override: int | None) -> int:
    if epochs_override is not None:
        if epochs_override <= 0:
            raise ValueError("epochs_override must be > 0")
        return int(epochs_override)
    return int(config["epochs"][dataset])


def optimizer_steps_per_epoch(num_micro_batches: int, grad_accum_steps: int) -> int:
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be > 0")
    if num_micro_batches <= 0:
        raise ValueError("num_micro_batches must be > 0")
    return int(math.ceil(num_micro_batches / grad_accum_steps))


@dataclass(frozen=True)
class WarmupCosineLRSchedule:
    """Per-step LR: linear warmup then cosine decay."""

    base_lr: float
    warmup_steps: int
    total_steps: int
    eta_min: float = 0.0
    warmup_start_lr: float = 0.0

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.base_lr < 0 or self.eta_min < 0 or self.warmup_start_lr < 0:
            raise ValueError("Learning rates must be non-negative")

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
