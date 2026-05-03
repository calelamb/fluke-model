"""Tests for the MiewID fine-tune utilities (cache round-trip + heads)."""

from __future__ import annotations

import os

# FAISS (loaded by sibling tests) and PyTorch both link an OpenMP runtime.
# On macOS that combination occasionally segfaults inside batch_norm. This
# guard is the standard escape hatch and only affects the test process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.miewid_finetune import (  # noqa: E402
    MIEWID_FEATURE_DIM,
    CachedFeatureDataset,
    CachedFeatures,
    LinearHead,
    MLPHead,
    build_head,
    load_cached_features,
    save_cached_features,
)
from fluke_model.orca_data import OrcaManifestRow  # noqa: E402


def _make_cache(n: int = 6, dim: int = 2152) -> CachedFeatures:
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    return CachedFeatures(
        paths=[f"/tmp/img_{i:03d}.jpg" for i in range(n)],
        features=raw,
        embedder_name="miewid-msv3",
        image_size=440,
    )


def test_cache_round_trip(tmp_path: Path):
    cache = _make_cache()
    out = tmp_path / "cache.npz"
    save_cached_features(out, cache)
    loaded = load_cached_features(out)
    assert loaded.paths == cache.paths
    assert loaded.embedder_name == cache.embedder_name
    assert loaded.image_size == cache.image_size
    np.testing.assert_array_equal(loaded.features, cache.features)


def test_cache_select_preserves_order():
    cache = _make_cache(n=5)
    selected = cache.select([cache.paths[2], cache.paths[0], cache.paths[4]])
    np.testing.assert_array_equal(selected[0], cache.features[2])
    np.testing.assert_array_equal(selected[1], cache.features[0])
    np.testing.assert_array_equal(selected[2], cache.features[4])


def test_cache_select_unknown_path_raises():
    cache = _make_cache(n=3)
    with pytest.raises(KeyError, match="not in feature cache"):
        cache.select([cache.paths[0], "/tmp/missing.jpg"])


def test_cached_feature_dataset_alignment():
    cache = _make_cache(n=4)
    rows = [
        OrcaManifestRow(path=cache.paths[2], image="a.jpg", individual_id="id-A", species="killer_whale"),
        OrcaManifestRow(path=cache.paths[0], image="b.jpg", individual_id="id-B", species="killer_whale"),
        OrcaManifestRow(path=cache.paths[3], image="c.jpg", individual_id="id-A", species="killer_whale"),
    ]
    ds = CachedFeatureDataset(cache, rows)
    assert len(ds) == 3
    feat0, label0, path0 = ds[0]
    np.testing.assert_array_equal(feat0.numpy(), cache.features[2])
    assert path0 == cache.paths[2]
    assert label0 == ds.label_to_idx["id-A"]

    feat1, label1, _path1 = ds[1]
    np.testing.assert_array_equal(feat1.numpy(), cache.features[0])
    assert label1 == ds.label_to_idx["id-B"]


def test_linear_head_shape_and_normalization():
    head = LinearHead(in_features=MIEWID_FEATURE_DIM, embed_dim=128)
    x = torch.randn(4, MIEWID_FEATURE_DIM)
    y = head(x)
    assert y.shape == (4, 128)
    norms = y.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_mlp_head_shape_and_normalization():
    # use_bn=False sidesteps a known macOS OpenMP conflict between FAISS
    # (imported by sibling tests) and PyTorch's batch_norm kernel that can
    # segfault inside the same pytest process. The BatchNorm-on path is
    # exercised end-to-end by the miewid-mlp-512-256-001 training run.
    head = MLPHead(
        in_features=MIEWID_FEATURE_DIM, hidden_dim=256, embed_dim=64, use_bn=False
    )
    head.eval()
    x = torch.randn(8, MIEWID_FEATURE_DIM)
    with torch.no_grad():
        y = head(x)
    assert y.shape == (8, 64)
    norms = y.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-4)


def test_build_head_dispatch():
    linear = build_head("linear", in_features=2152, embed_dim=128)
    assert isinstance(linear, LinearHead)
    mlp = build_head("mlp", in_features=2152, hidden_dim=256, embed_dim=64)
    assert isinstance(mlp, MLPHead)


def test_build_head_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unknown head kind"):
        build_head("transformer")
