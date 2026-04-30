"""Tests for lightweight trainable model utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.trainable import BalancedBatchSampler, batch_hard_triplet_loss  # noqa: E402


def test_balanced_batch_sampler_repeats_each_identity():
    labels = [0, 0, 1, 1, 2, 2, 3, 3]
    sampler = BalancedBatchSampler(labels, identities_per_batch=2, images_per_identity=2, seed=1)
    batch = next(iter(sampler))
    batch_labels = [labels[i] for i in batch]
    assert len(batch) == 4
    assert sorted(batch_labels).count(batch_labels[0]) >= 1
    assert len(set(batch_labels)) == 2
    for label in set(batch_labels):
        assert batch_labels.count(label) == 2


def test_balanced_batch_sampler_requires_repeated_ids():
    with pytest.raises(ValueError):
        BalancedBatchSampler([0, 1, 2], identities_per_batch=2, images_per_identity=2)


def test_triplet_loss_accepts_l2_normalized_embeddings():
    raw = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ]
    )
    embeddings = torch.nn.functional.normalize(raw, dim=-1)
    labels = torch.tensor([0, 0, 1, 1])
    loss = batch_hard_triplet_loss(embeddings, labels, margin=0.2)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_triplet_loss_validates_shapes():
    with pytest.raises(ValueError):
        batch_hard_triplet_loss(torch.zeros(2, 3), torch.zeros(3, dtype=torch.long))
