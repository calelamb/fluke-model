#!/usr/bin/env python3
"""Run a single-image identification against a saved FAISS index.

Usage:
    python scripts/identify.py --index data/index/dinov2-small --image path/to/photo.jpg --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.embedders import load_embedder, EmbedderUnavailable  # noqa: E402
from fluke_model.index import aggregate_per_individual, load_index, search  # noqa: E402
from fluke_model.io import load_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Identify a single image against a saved index.")
    parser.add_argument("--index", required=True, help="Index bundle directory (from embed_catalog.py).")
    parser.add_argument("--image", required=True, help="Query image path.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--neighbors", type=int, default=20, help="Raw FAISS neighbors to retrieve before aggregation.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plaintext.")
    args = parser.parse_args()

    bundle = load_index(args.index)
    try:
        embedder = load_embedder(bundle.embedder_name)
    except EmbedderUnavailable as e:
        print(f"Cannot reload embedder '{bundle.embedder_name}': {e}", file=sys.stderr)
        return 2

    image = load_image(args.image)
    query_vec = embedder.embed_fn([image])[0]
    raw_hits = search(bundle, query_vec, k=min(args.neighbors, bundle.index.ntotal))
    aggregated = aggregate_per_individual(raw_hits, top_n=3)
    top = aggregated[: args.top_k]

    if args.json:
        print(json.dumps({"query": args.image, "matches": [{"individual_id": i, "score": s} for i, s in top]}, indent=2))
    else:
        print(f"Query: {args.image}")
        print(f"Top-{args.top_k} matches (embedder: {bundle.embedder_name}):")
        for rank, (ind, score) in enumerate(top, start=1):
            print(f"  {rank}. {ind}  score={score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
