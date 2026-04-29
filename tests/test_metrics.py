"""Unit tests for fluke_model.metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.metrics import top_k_accuracy, mean_reciprocal_rank


def test_top_k_accuracy_perfect():
    preds = [["a", "b", "c"], ["x", "y", "z"]]
    truths = ["a", "x"]
    assert top_k_accuracy(preds, truths, k=1) == 1.0
    assert top_k_accuracy(preds, truths, k=3) == 1.0


def test_top_k_accuracy_partial():
    preds = [["a", "b"], ["c", "d"], ["e", "f"]]
    truths = ["b", "c", "x"]
    # query 1: b at rank 2 (in top-3, not top-1)
    # query 2: c at rank 1
    # query 3: x missing
    assert top_k_accuracy(preds, truths, k=1) == pytest.approx(1 / 3)
    assert top_k_accuracy(preds, truths, k=3) == pytest.approx(2 / 3)


def test_top_k_accuracy_invalid_k():
    with pytest.raises(ValueError):
        top_k_accuracy([["a"]], ["a"], k=0)


def test_top_k_accuracy_length_mismatch():
    with pytest.raises(ValueError):
        top_k_accuracy([["a"]], ["a", "b"], k=1)


def test_top_k_accuracy_empty():
    assert top_k_accuracy([], [], k=1) == 0.0


def test_mrr_perfect():
    preds = [["a", "b"], ["c", "d"]]
    truths = ["a", "c"]
    assert mean_reciprocal_rank(preds, truths) == 1.0


def test_mrr_mixed():
    preds = [["a", "b", "c"], ["c", "d"], ["e", "f"]]
    truths = ["b", "c", "x"]
    # 1/2, 1/1, 0  -> 1.5 / 3
    expected = (0.5 + 1.0 + 0.0) / 3
    assert mean_reciprocal_rank(preds, truths) == pytest.approx(expected)


def test_mrr_all_missing():
    preds = [["a"], ["b"]]
    truths = ["x", "y"]
    assert mean_reciprocal_rank(preds, truths) == 0.0
