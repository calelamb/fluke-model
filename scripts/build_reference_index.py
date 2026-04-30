#!/usr/bin/env python3
"""Build a MiewID FAISS reference index from a JSON/JSONL reference manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.identify_runtime import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    build_reference_index,
    reference_from_payload,
)


def read_manifest(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return list(payload["references"])
    return list(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Fluke reference-photo FAISS index.")
    parser.add_argument("--manifest", required=True, help="JSON/JSONL reference manifest.")
    parser.add_argument("--out-dir", default=str(DEFAULT_INDEX_DIR))
    args = parser.parse_args()

    references = [reference_from_payload(row) for row in read_manifest(Path(args.manifest))]
    result = build_reference_index(references, out_dir=Path(args.out_dir))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
