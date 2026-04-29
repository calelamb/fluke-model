#!/usr/bin/env python3
"""Embed a manifest of catalog photos and persist a FAISS index.

Usage:
    python scripts/embed_catalog.py \
        --embedder dinov2-small \
        --manifest data/beluga-id-2022/manifest.csv \
        --out data/index/dinov2-small
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Allow running without `uv run python -m`; add src to sys.path.
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.embedders import load_embedder, EmbedderUnavailable  # noqa: E402
from fluke_model.index import build_index, save_index  # noqa: E402
from fluke_model.io import load_image, read_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed a manifest and write a FAISS index.")
    parser.add_argument("--embedder", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, help="Output directory for the index bundle.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="If > 0, only embed the first N rows.")
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        print(f"Manifest is empty: {args.manifest}", file=sys.stderr)
        return 2

    try:
        embedder = load_embedder(args.embedder)
    except EmbedderUnavailable as e:
        print(f"Embedder '{args.embedder}' unavailable: {e}", file=sys.stderr)
        return 3

    print(f"Embedding {len(rows)} rows with {args.embedder} (dim={embedder.embed_dim})...")
    all_vecs: list[np.ndarray] = []
    metadata: list[dict] = []

    for start in tqdm(range(0, len(rows), args.batch_size)):
        chunk = rows[start : start + args.batch_size]
        images = [load_image(r.path) for r in chunk]
        try:
            vecs = embedder.embed_fn(images)
        except Exception as e:
            print(f"\nbatch failed at row {start}: {e}", file=sys.stderr)
            continue
        all_vecs.append(vecs)
        for r in chunk:
            metadata.append({"path": r.path, "individual_id": r.individual_id})

    if not all_vecs:
        print("No vectors produced.", file=sys.stderr)
        return 4

    embeddings = np.concatenate(all_vecs, axis=0).astype(np.float32)
    bundle = build_index(embeddings, metadata, embedder_name=args.embedder)
    save_index(bundle, args.out)
    print(f"Saved index ({embeddings.shape[0]} vectors, dim={embeddings.shape[1]}) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
