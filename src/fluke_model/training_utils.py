"""Shared training utilities: device selection and LR scheduling.

Both `scripts/train_embedder.py` and `scripts/finetune_miewid_head.py` need the
same warmup-then-cosine schedule and the same `auto` device resolver. Keeping
one canonical implementation here avoids the previous copy/paste drift.
"""

from __future__ import annotations

import math

import torch


def select_device(name: str) -> torch.device:
    """Resolve a device name; `auto` picks the best available accelerator."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    kind: str,
    epochs: int,
    warmup_epochs: int,
    base_lr: float,
    min_lr: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Linear-warmup + cosine-decay LR schedule, applied per-epoch.

    Returns None when `kind == "none"`. The cosine schedule decays from
    `base_lr` to `min_lr` over the post-warmup epochs.
    """
    if kind == "none":
        return None

    warmup = max(0, min(warmup_epochs, epochs - 1))
    decay_span = max(1, epochs - warmup)
    floor_factor = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(epoch_idx: int) -> float:
        # `epoch_idx` is 0-based and increments after each scheduler.step().
        if warmup > 0 and epoch_idx < warmup:
            return float(epoch_idx + 1) / float(warmup)
        progress = (epoch_idx - warmup) / decay_span
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor_factor + (1.0 - floor_factor) * cos

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
