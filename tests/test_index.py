"""Unit tests for fluke_model.index — round-trip and aggregation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.index import (  # noqa: E402
    aggregate_per_individual,
    build_index,
    load_index,
    save_index,
    search,
)


def _l2(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return (vecs / norms).astype(np.float32)


def test_build_search_round_trip(tmp_path: Path):
    rng = np.random.default_rng(7)
    raw = rng.standard_normal((6, 32)).astype(np.float32)
    embeddings = _l2(raw)
    metadata = [{"path": f"p{i}", "individual_id": f"ind_{i % 3}"} for i in range(6)]
    bundle = build_index(embeddings, metadata, embedder_name="test")

    save_index(bundle, tmp_path)
    reloaded = load_index(tmp_path)
    assert reloaded.embedder_name == "test"
    assert reloaded.embed_dim == 32
    assert len(reloaded.metadata) == 6

    # Querying with the first row should retrieve itself first.
    hits = search(reloaded, embeddings[0], k=3)
    assert len(hits) == 3
    assert hits[0][1]["path"] == "p0"
    assert hits[0][0] == pytest.approx(1.0, abs=1e-5)


def test_build_validation_errors():
    with pytest.raises(ValueError):
        build_index(np.zeros((3,), dtype=np.float32), [{}], embedder_name="x")
    with pytest.raises(ValueError):
        build_index(np.zeros((2, 4), dtype=np.float32), [{}], embedder_name="x")


def test_aggregate_per_individual_mean_top3():
    hits = [
        (0.9, {"individual_id": "a", "path": "p1"}),
        (0.8, {"individual_id": "a", "path": "p2"}),
        (0.7, {"individual_id": "a", "path": "p3"}),
        (0.6, {"individual_id": "a", "path": "p4"}),
        (0.5, {"individual_id": "b", "path": "p5"}),
    ]
    out = aggregate_per_individual(hits, top_n=3)
    assert out[0][0] == "a"
    assert out[0][1] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert out[1][0] == "b"
    assert out[1][1] == pytest.approx(0.5)


def test_search_drops_invalid_indices(tmp_path):
    rng = np.random.default_rng(0)
    embeddings = _l2(rng.standard_normal((2, 8)).astype(np.float32))
    metadata = [{"path": "a", "individual_id": "i1"}, {"path": "b", "individual_id": "i2"}]
    bundle = build_index(embeddings, metadata, embedder_name="t")
    # Asking for more neighbors than exist must not error.
    hits = search(bundle, embeddings[0], k=10)
    assert len(hits) == 2
