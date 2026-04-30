#!/usr/bin/env python3
"""Create closed-set train/val/test splits for an orca manifest."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.orca_data import (  # noqa: E402
    read_jsonl_manifest,
    split_by_individual_images,
    verify_manifest_images,
    write_jsonl_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Split an orca JSONL manifest.")
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data/manifests/happywhale_orca.jsonl"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "data/manifests/happywhale_orca_splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--min-images-per-individual", type=int, default=2)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl_manifest(args.manifest)
    missing = verify_manifest_images(rows)
    if missing and not args.allow_missing:
        print(f"{len(missing)} images are missing. First missing: {missing[0]}", file=sys.stderr)
        return 2

    train, val, test, stats = split_by_individual_images(
        rows,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        min_images_per_individual=args.min_images_per_individual,
    )

    out_dir = Path(args.out_dir)
    write_jsonl_manifest(out_dir / "train.jsonl", train)
    write_jsonl_manifest(out_dir / "val.jsonl", val)
    write_jsonl_manifest(out_dir / "test.jsonl", test)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest),
        **stats,
    }
    (out_dir / "split_summary.json").write_text(json.dumps(payload, indent=2))

    print(f"Wrote splits to {out_dir}")
    print(f"Train: {len(train)} | val: {len(val)} | test: {len(test)}")
    print(f"Dropped individuals: {stats['dropped_individuals']} ({stats['dropped_images']} images)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
