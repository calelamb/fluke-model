"""Tests for the LR schedule helper used by train_embedder.py."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "train_embedder.py"
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_embedder_mod", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def train_module():
    return _load_train_module()


def _make_optimizer(lr: float = 3e-4) -> torch.optim.Optimizer:
    model = torch.nn.Linear(2, 2)
    return torch.optim.AdamW(model.parameters(), lr=lr)


def test_scheduler_none_returns_none(train_module):
    sch = train_module._build_scheduler(
        _make_optimizer(),
        kind="none",
        epochs=10,
        warmup_epochs=2,
        base_lr=3e-4,
        min_lr=1e-6,
    )
    assert sch is None


def test_warmup_then_cosine_decay(train_module):
    base_lr = 3e-4
    min_lr = 1e-6
    optimizer = _make_optimizer(base_lr)
    sch = train_module._build_scheduler(
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
    sch.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-6)

    # After 2 steps: still at peak (cosine is at progress=0)
    sch.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-6)

    # Walk through remaining epochs; LR should monotonically decrease and
    # land near (but no lower than) the floor.
    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(18):
        sch.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    descending = all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1))
    assert descending, f"LR not monotonically decreasing: {lrs}"
    assert lrs[-1] >= min_lr - 1e-12
    assert lrs[-1] <= base_lr / 10


def test_zero_warmup_starts_at_peak(train_module):
    base_lr = 3e-4
    optimizer = _make_optimizer(base_lr)
    sch = train_module._build_scheduler(
        optimizer,
        kind="cosine",
        epochs=10,
        warmup_epochs=0,
        base_lr=base_lr,
        min_lr=1e-6,
    )
    assert sch is not None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr, rel=1e-6)


def test_warmup_clamped_to_epochs(train_module):
    """If warmup_epochs >= epochs, the scheduler should not blow up."""
    base_lr = 3e-4
    optimizer = _make_optimizer(base_lr)
    sch = train_module._build_scheduler(
        optimizer,
        kind="cosine",
        epochs=2,
        warmup_epochs=10,
        base_lr=base_lr,
        min_lr=1e-6,
    )
    assert sch is not None
    # Should not raise across the configured epoch count.
    for _ in range(2):
        sch.step()
    final_lr = optimizer.param_groups[0]["lr"]
    assert math.isfinite(final_lr)
    assert final_lr > 0
