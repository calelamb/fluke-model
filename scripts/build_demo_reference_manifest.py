#!/usr/bin/env python3
"""Build a demo-mode reference manifest from licensed orca data.

Demo mode is the V0 of the identifier feature: match user uploads against the
combined HappyWhale + FinID-20 catalog (45 individuals) so we can ship and
test the whole pipeline before Salish Sea catalog licensing closes.

This script reads `data/manifests/orca_all_splits/train.jsonl` (the train
portion is treated as the reference catalog) and produces a JSONL manifest in
the schema that `IdentifierRuntime` / `build_reference_index.py` consume:

    {"referencePhotoId", "catalogId", "name", "url", "side", "quality", "crop"}

URLs are emitted as `file://` so the runtime can load images straight from
disk in dev. Production references live behind authenticated object-storage
URLs and would skip this script.

The val + test splits are intentionally NOT included in references so they
can be used as held-out queries when smoke-testing the service.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.orca_data import OrcaManifestRow, read_jsonl_manifest  # noqa: E402

DEFAULT_TRAIN = REPO_ROOT / "data/manifests/orca_all_splits/train.jsonl"
DEFAULT_OUT = REPO_ROOT / "data/manifests/demo_reference.jsonl"


def display_name_for(individual_id: str) -> str:
    """Render a short display name for an anonymized individual id.

    HappyWhale ids are 12-char hex; FinID-20 ids are prefixed `finid_`. We
    abbreviate to the first 8 characters and prefix with the source so the
    UI has something readable to show until real names are licensed.
    """
    if individual_id.startswith("finid_"):
        return f"FinID {individual_id[len('finid_') :][:8].upper()}"
    return f"HW {individual_id[:8].upper()}"


def build_references(
    rows: list[OrcaManifestRow],
    *,
    max_per_individual: int,
) -> list[dict]:
    """Group rows by individual_id, take up to `max_per_individual` per id."""
    grouped: dict[str, list[OrcaManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[row.individual_id].append(row)

    references: list[dict] = []
    for individual_id in sorted(grouped):
        items = grouped[individual_id][:max_per_individual]
        for idx, row in enumerate(items):
            references.append(
                {
                    "referencePhotoId": f"{individual_id}::{idx:02d}",
                    "catalogId": individual_id,
                    "name": display_name_for(individual_id),
                    "url": f"file://{row.path}",
                    "side": "UNKNOWN",
                    "quality": "USABLE",
                    "source_dataset": row.source_dataset,
                }
            )
    return references


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the demo-mode reference manifest.")
    parser.add_argument("--train-manifest", default=str(DEFAULT_TRAIN))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--max-per-individual",
        type=int,
        default=5,
        help="Cap reference photos per individual; trades index size for coverage.",
    )
    args = parser.parse_args()

    train_path = Path(args.train_manifest)
    if not train_path.exists():
        print(f"Train manifest not found: {train_path}", file=sys.stderr)
        print("Run scripts/split_manifest.py first.", file=sys.stderr)
        return 2

    rows = read_jsonl_manifest(train_path)
    if not rows:
        print(f"No rows in {train_path}", file=sys.stderr)
        return 3

    references = build_references(rows, max_per_individual=args.max_per_individual)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for ref in references:
            f.write(json.dumps(ref) + "\n")

    summary = {
        "manifest": str(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_train_manifest": str(train_path),
        "max_per_individual": args.max_per_individual,
        "reference_count": len(references),
        "individual_count": len({ref["catalogId"] for ref in references}),
        "by_source": dict.fromkeys({ref["source_dataset"] for ref in references}, 0),
    }
    for ref in references:
        summary["by_source"][ref["source_dataset"]] += 1
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {out_path}")
    print(f"  references: {summary['reference_count']}")
    print(f"  individuals: {summary['individual_count']}")
    print(f"  by source: {summary['by_source']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
