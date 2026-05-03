"""MiewID fine-tuning utilities — cached features and trainable heads.

Full backbone fine-tuning lives at BYU GPU lab (see docs/byu-gpu-finetuning-plan.md).
On the M3 Pro we run a cheaper variant: cache MiewID's frozen 2152-dim embeddings
once, then train a small learnable head on top. The head is what specializes the
embedding space for orcas without re-running the heavy backbone every epoch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from fluke_model.orca_data import OrcaManifestRow

MIEWID_FEATURE_DIM = 2152


@dataclass(frozen=True)
class CachedFeatures:
    """In-memory cache of MiewID features for a list of manifest rows."""

    paths: list[str]
    features: np.ndarray  # (N, 2152) float32, L2-normalized
    embedder_name: str
    image_size: int

    def index_for(self, paths: list[str]) -> np.ndarray:
        """Return integer indices for the given paths in cache order."""
        lookup = {p: i for i, p in enumerate(self.paths)}
        try:
            return np.array([lookup[p] for p in paths], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"Path not in feature cache: {exc.args[0]}") from exc

    def select(self, paths: list[str]) -> np.ndarray:
        idx = self.index_for(paths)
        return self.features[idx]


def save_cached_features(path: str | Path, cache: CachedFeatures) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        paths=np.array(cache.paths, dtype=object),
        features=cache.features.astype(np.float32),
        embedder_name=np.array(cache.embedder_name),
        image_size=np.array(cache.image_size, dtype=np.int32),
    )


def load_cached_features(path: str | Path) -> CachedFeatures:
    # allow_pickle=True is required because `paths` is saved as a dtype=object
    # NumPy array of Python strings. The cache file is locally produced by
    # cache_miewid_features.py; do not load caches from untrusted sources.
    raw = np.load(path, allow_pickle=True)
    return CachedFeatures(
        paths=list(raw["paths"]),
        features=raw["features"].astype(np.float32),
        embedder_name=str(raw["embedder_name"]),
        image_size=int(raw["image_size"]),
    )


class CachedFeatureDataset(Dataset):
    """Dataset that returns (feature, label_idx, path) tuples for cached vectors."""

    def __init__(self, cache: CachedFeatures, rows: list[OrcaManifestRow]):
        self.rows = rows
        self.label_to_idx = {label: i for i, label in enumerate(sorted({r.individual_id for r in rows}))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self._features = cache.select([row.path for row in rows])

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        row = self.rows[idx]
        feature = torch.from_numpy(self._features[idx]).clone()
        label = self.label_to_idx[row.individual_id]
        return feature, label, row.path


class LinearHead(nn.Module):
    """Single Linear projection followed by L2 normalization."""

    def __init__(self, in_features: int = MIEWID_FEATURE_DIM, embed_dim: int = 256):
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_features, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class MLPHead(nn.Module):
    """Two-layer MLP with optional bottleneck and L2-normalized output."""

    def __init__(
        self,
        in_features: int = MIEWID_FEATURE_DIM,
        hidden_dim: int = 512,
        embed_dim: int = 256,
        dropout: float = 0.1,
        use_bn: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        layers: list[nn.Module] = [nn.Linear(in_features, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, embed_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def build_head(kind: str, **kwargs: int | float | bool) -> nn.Module:
    """Factory for head architectures.

    Accepted kwargs match the constructors of `LinearHead` and `MLPHead`:
    `in_features`, `embed_dim`, plus `hidden_dim`, `dropout`, `use_bn` for MLP.
    """
    if kind == "linear":
        return LinearHead(**kwargs)
    if kind == "mlp":
        return MLPHead(**kwargs)
    raise ValueError(f"Unknown head kind: {kind}. Choices: linear, mlp")


@dataclass(frozen=True)
class HeadCheckpointMetadata:
    """Persisted alongside the head weights for reload."""

    embedder_name: str
    embedder_dim: int
    head_kind: str
    embed_dim: int
    hidden_dim: int | None
    dropout: float | None
    use_bn: bool | None
    image_size: int


def save_head_checkpoint(
    path: str | Path,
    head: nn.Module,
    metadata: HeadCheckpointMetadata,
    *,
    epoch: int,
    metrics: dict,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state": head.state_dict(),
            "metadata": metadata.__dict__,
            "epoch": epoch,
            "metrics": metrics,
        },
        out,
    )


@torch.no_grad()
def project_cached_features(
    head: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Apply a trained head to cached MiewID vectors and return projected embeddings."""
    head.eval()
    out_chunks: list[np.ndarray] = []
    for start in range(0, len(features), batch_size):
        chunk = features[start : start + batch_size]
        tensor = torch.from_numpy(chunk).to(device)
        out_chunks.append(head(tensor).cpu().numpy().astype(np.float32))
    if not out_chunks:
        return np.zeros((0, head.embed_dim), dtype=np.float32)
    return np.concatenate(out_chunks, axis=0)
