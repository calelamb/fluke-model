"""Re-identification evaluation metrics.

These are the only numbers the M-Model-0 prototype reports:
- top_k_accuracy(k): fraction of queries whose true individual is in the top-k retrieved list.
- mean_reciprocal_rank: 1/rank averaged over queries; rank=None contributes 0.
"""

from __future__ import annotations

from typing import Sequence


def top_k_accuracy(
    predictions: Sequence[Sequence[str]],
    truths: Sequence[str],
    k: int,
) -> float:
    """Compute top-k accuracy.

    Args:
        predictions: For each query, an ordered list of predicted individual ids
            (highest similarity first). Lists shorter than k are allowed; missing
            entries simply don't count as a hit.
        truths: Ground-truth individual id per query.
        k: Cutoff. Must be >= 1.

    Returns:
        Fraction of queries where truths[i] appears in predictions[i][:k].
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(predictions) != len(truths):
        raise ValueError(
            f"predictions and truths must have the same length; got {len(predictions)} vs {len(truths)}"
        )
    if not predictions:
        return 0.0

    hits = 0
    for preds, truth in zip(predictions, truths):
        if truth in list(preds)[:k]:
            hits += 1
    return hits / len(predictions)


def mean_reciprocal_rank(
    predictions: Sequence[Sequence[str]],
    truths: Sequence[str],
) -> float:
    """Compute mean reciprocal rank (MRR).

    For each query: if the true id appears at rank r (1-indexed) in predictions,
    contribute 1/r. If it doesn't appear at all, contribute 0.
    """
    if len(predictions) != len(truths):
        raise ValueError(
            f"predictions and truths must have the same length; got {len(predictions)} vs {len(truths)}"
        )
    if not predictions:
        return 0.0

    total = 0.0
    for preds, truth in zip(predictions, truths):
        rank = _rank_of(list(preds), truth)
        if rank is not None:
            total += 1.0 / rank
    return total / len(predictions)


def _rank_of(predictions: list[str], truth: str) -> int | None:
    """Return the 1-indexed rank of `truth` in `predictions`, or None if absent."""
    for i, p in enumerate(predictions):
        if p == truth:
            return i + 1
    return None
