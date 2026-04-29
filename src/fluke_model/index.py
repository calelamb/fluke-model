"""FAISS index helpers for the M-Model-0 prototype.

Inner-product index over L2-normalized vectors == cosine similarity. We use
`IndexFlatIP` because the catalog scale (a few thousand vectors) makes exact
search trivially fast on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass(frozen=True)
class IndexBundle:
    """A FAISS index plus its row metadata.

    Attributes:
        index: The faiss index. Vectors are L2-normalized and stored at
            row positions matching `metadata`.
        metadata: One dict per row: { "path": str, "individual_id": str }.
        embedder_name: The embedder used to produce the vectors.
        embed_dim: Embedding dimensionality.
    """

    index: faiss.Index
    metadata: list[dict]
    embedder_name: str
    embed_dim: int


def build_index(
    embeddings: np.ndarray,
    metadata: list[dict],
    embedder_name: str,
) -> IndexBundle:
    """Build an `IndexFlatIP` from already-L2-normalized embeddings.

    Args:
        embeddings: Shape (N, D), float32, rows L2-normalized.
        metadata: Length N list of metadata dicts.
        embedder_name: For traceability when the bundle is reloaded.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D (N, D); got shape {embeddings.shape}")
    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"embeddings rows ({embeddings.shape[0]}) != len(metadata) ({len(metadata)})"
        )
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    dim = int(embeddings.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return IndexBundle(index=index, metadata=list(metadata), embedder_name=embedder_name, embed_dim=dim)


def save_index(bundle: IndexBundle, out_dir: str | Path) -> None:
    """Persist a bundle to <out_dir>/index.faiss + <out_dir>/metadata.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(bundle.index, str(out_dir / "index.faiss"))
    payload = {
        "embedder_name": bundle.embedder_name,
        "embed_dim": bundle.embed_dim,
        "metadata": bundle.metadata,
    }
    (out_dir / "metadata.json").write_text(json.dumps(payload, indent=2))


def load_index(in_dir: str | Path) -> IndexBundle:
    """Load a bundle previously persisted by `save_index`."""
    in_dir = Path(in_dir)
    index = faiss.read_index(str(in_dir / "index.faiss"))
    payload = json.loads((in_dir / "metadata.json").read_text())
    return IndexBundle(
        index=index,
        metadata=payload["metadata"],
        embedder_name=payload["embedder_name"],
        embed_dim=payload["embed_dim"],
    )


def search(bundle: IndexBundle, query: np.ndarray, k: int) -> list[tuple[float, dict]]:
    """Search the index. Returns up to k (score, metadata) pairs sorted by descending score.

    Args:
        bundle: The IndexBundle to query.
        query: Shape (D,) or (1, D) float32, L2-normalized.
        k: Number of neighbors to return.
    """
    if query.ndim == 1:
        query = query[None, :]
    if query.dtype != np.float32:
        query = query.astype(np.float32)
    scores, idxs = bundle.index.search(query, k)
    out: list[tuple[float, dict]] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        out.append((float(score), bundle.metadata[idx]))
    return out


def aggregate_per_individual(
    hits: list[tuple[float, dict]],
    top_n: int = 3,
) -> list[tuple[str, float]]:
    """Aggregate raw FAISS hits into per-individual scores using mean-of-top-N.

    Returns a list of (individual_id, score) sorted by descending score.
    """
    by_id: dict[str, list[float]] = {}
    for score, meta in hits:
        ind = meta["individual_id"]
        by_id.setdefault(ind, []).append(score)
    aggregated = [
        (ind, float(sum(sorted(scores, reverse=True)[:top_n]) / min(top_n, len(scores))))
        for ind, scores in by_id.items()
    ]
    aggregated.sort(key=lambda x: -x[1])
    return aggregated
