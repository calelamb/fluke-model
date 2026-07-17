"""Tests for the shared LR schedule helper used by training scripts."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.training_utils import build_scheduler, select_device  # noqa: E402


def _make_optimizer(lr: float = 3e-4) -> torch.optim.Optimizer:
    model = torch.nn.Linear(2, 2)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def _step(
    optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler
) -> None:
    optimizer.step()
    scheduler.step()


def test_scheduler_none_returns_none():
    sch = build_scheduler(
        _make_optimizer(),
        kind="none",
        epochs=10,
        warmup_epochs=2,
        base_lr=3e-4,
        min_lr=1e-6,
    )
    assert sch is None


def test_warmup_then_cosine_decay():
    base_lr = 3e-4
    min_lr = 1e-6
    optimizer = _make_optimizer(base_lr)
    sch = build_scheduler(
        optimizer,
        kind="cosine",
        epochs=20,
        warmup_epochs=2,
        base_lr=base_lr,
        min_lr=min_lr,
    )
    assert sch is not None

    # Epoch 1 (before any step): half-warmup
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr / 2, rel=1e-6)

    # After 1 step: full base lr
    _step(optimizer, sch)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-6)

    # After 2 steps: still at peak (cosine is at progress=0)
    _step(optimizer, sch)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-6)

    # Walk through remaining epochs; LR should monotonically decrease and
    # land near (but no lower than) the floor.
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(18):
        _step(optimizer, sch)
        lrs.append(optimizer.param_groups[0]["lr"])

    descending = all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1))
    assert descending, f"LR not monotonically decreasing: {lrs}"
    assert lrs[-1] >= min_lr - 1e-12
    assert lrs[-1] <= base_lr / 10


def test_zero_warmup_starts_at_peak():
    base_lr = 3e-4
    optimizer = _make_optimizer(base_lr)
    sch = build_scheduler(
        optimizer,
        kind="cosine",
        epochs=10,
        warmup_epochs=0,
        base_lr=base_lr,
        min_lr=1e-6,
    )
    assert sch is not None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-6)


def test_warmup_clamped_to_epochs():
    """If warmup_epochs >= epochs, the scheduler should not blow up."""
    base_lr = 3e-4
    optimizer = _make_optimizer(base_lr)
    sch = build_scheduler(
        optimizer,
        kind="cosine",
        epochs=2,
        warmup_epochs=10,
        base_lr=base_lr,
        min_lr=1e-6,
    )
    assert sch is not None
    for _ in range(2):
        _step(optimizer, sch)
    final_lr = optimizer.param_groups[0]["lr"]
    assert math.isfinite(final_lr)
    assert final_lr > 0


def test_select_device_explicit_cpu():
    """Explicit device names bypass the auto resolver."""
    assert select_device("cpu") == torch.device("cpu")


def test_select_device_auto_returns_real_device():
    """`auto` must resolve to one of cuda/mps/cpu without error."""
    device = select_device("auto")
    assert device.type in {"cuda", "mps", "cpu"}
