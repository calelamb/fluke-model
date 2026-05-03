#!/usr/bin/env python3
"""Concatenate per-source orca manifests into a combined training manifest.

Reads per-source JSONL manifests (HappyWhale, FinID-20, etc.), concatenates
the rows, and writes a unified manifest that downstream scripts treat as a
single dataset. No deduplication is attempted because individual_ids should
already be namespaced per source (HappyWhale ids are 12-char hex; FinID-20
ids are prefixed with `finid_`).
"""

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
    OrcaManifestRow,
    manifest_stats,
    read_jsonl_manifest,
    write_jsonl_manifest,
)

DEFAULT_SOURCES = [
    str(REPO_ROOT / "data/manifests/happywhale_orca.jsonl"),
    str(REPO_ROOT / "data/manifests/finid20_orca.jsonl"),
]
DEFAULT_OUT = REPO_ROOT / "data/manifests/orca_all.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine orca manifests across sources.")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="Per-source JSONL manifest (repeatable). Defaults to HappyWhale + FinID-20.",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    sources = args.source if args.source else DEFAULT_SOURCES

    all_rows: list[OrcaManifestRow] = []
    per_source_stats: dict[str, dict] = {}
    skipped: list[str] = []

    for src in sources:
        path = Path(src)
        if not path.exists():
            print(f"  SKIP missing source: {path}", file=sys.stderr)
            skipped.append(str(path))
            continue
        rows = read_jsonl_manifest(path)
        per_source_stats[str(path)] = manifest_stats(rows)
        all_rows.extend(rows)
        print(f"  + {path}: {len(rows)} rows, {per_source_stats[str(path)]['individuals']} individuals")

    if not all_rows:
        print("No rows from any source; nothing to combine.", file=sys.stderr)
        return 2

    seen_paths: set[str] = set()
    deduped: list[OrcaManifestRow] = []
    for row in all_rows:
        if row.path in seen_paths:
            continue
        seen_paths.add(row.path)
        deduped.append(row)
    if len(deduped) != len(all_rows):
        print(f"  warning: {len(all_rows) - len(deduped)} duplicate paths removed")

    out_path = Path(args.out)
    write_jsonl_manifest(out_path, deduped)
    stats = manifest_stats(deduped)
    summary = {
        "manifest": str(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "skipped_sources": skipped,
        "per_source_stats": per_source_stats,
        "combined_stats": stats,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {out_path}")
    print(f"Combined rows: {stats['rows']} | individuals: {stats['individuals']}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
