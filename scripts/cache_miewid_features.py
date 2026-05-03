#!/usr/bin/env python3
"""Pre-compute MiewID features for every image in an orca manifest.

Caching the 2152-dim MiewID embeddings once turns the M3 Pro head-training loop
from "MiewID forward + head + backward" into "head + backward". MiewID is the
expensive component; without caching, every epoch re-runs the heavy backbone.

Usage:

    python scripts/cache_miewid_features.py \\
        --manifest data/manifests/orca_all.jsonl \\
        --out artifacts/miewid_features/orca_all.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.embedders import load_embedder  # noqa: E402
from fluke_model.miewid_finetune import (  # noqa: E402
    MIEWID_FEATURE_DIM,
    CachedFeatures,
    save_cached_features,
)
from fluke_model.orca_data import read_jsonl_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache MiewID features for an orca manifest.")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data/manifests/orca_all.jsonl"))
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "artifacts/miewid_features/orca_all.npz")
    )
    parser.add_argument("--embedder", default="miewid-msv3")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=440, help="Reported in cache metadata.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    rows = read_jsonl_manifest(manifest_path)
    paths = [row.path for row in rows]
    print(f"Caching {args.embedder} features for {len(paths)} images")

    embedder = load_embedder(args.embedder)
    if embedder.embed_dim != MIEWID_FEATURE_DIM:
        print(
            f"  warning: expected dim {MIEWID_FEATURE_DIM}, embedder reports {embedder.embed_dim}",
            file=sys.stderr,
        )

    started = time.time()
    chunks: list[np.ndarray] = []
    for start in tqdm(range(0, len(paths), args.batch_size), desc=args.embedder):
        batch_paths = paths[start : start + args.batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        chunks.append(embedder.embed_fn(images))
    features = np.concatenate(chunks, axis=0).astype(np.float32)
    elapsed = time.time() - started

    cache = CachedFeatures(
        paths=paths,
        features=features,
        embedder_name=embedder.name,
        image_size=args.image_size,
    )
    save_cached_features(args.out, cache)
    print(f"Wrote {args.out}")
    print(f"  shape: {features.shape}")
    print(f"  elapsed: {elapsed:.1f}s ({elapsed / max(len(paths), 1):.3f}s/image)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
