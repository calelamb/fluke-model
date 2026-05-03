#!/usr/bin/env python3
"""Build an OrcaManifestRow JSONL from the FinID-20 (Zenodo 16786268) dataset.

FinID-20 ships pre-cropped fluke images grouped by anonymized individual id:

    cropped_images/
      05a9a14eed/
        05a9a14eed_fin0_IMG_0000.jpg
        ...
      ...
      train.csv  val.csv  test.csv  (built-in 350/75/75 split)

We reuse the OrcaManifestRow schema with `source_dataset="finid20"` and
prefix the individual_id with `finid_` to avoid any collision with
HappyWhale's 12-char hex ids.

This script does NOT use the built-in split CSVs. The combined orca pipeline
re-splits everything together via scripts/split_manifest.py, so each dataset's
manifest is just a flat list of (path, individual_id) rows.
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
    verify_manifest_images,
    write_jsonl_manifest,
)

DEFAULT_RAW = REPO_ROOT / "data/finid-20/raw"
DEFAULT_OUT = REPO_ROOT / "data/manifests/finid20_orca.jsonl"


def collect_rows(cropped_dir: Path) -> list[OrcaManifestRow]:
    """Walk cropped_images/<individual>/*.jpg and produce manifest rows."""
    rows: list[OrcaManifestRow] = []
    if not cropped_dir.exists():
        return rows
    for individual_dir in sorted(p for p in cropped_dir.iterdir() if p.is_dir()):
        individual_id = f"finid_{individual_dir.name.lower()}"
        for image_path in sorted(individual_dir.glob("*.jpg")):
            rows.append(
                OrcaManifestRow(
                    path=str(image_path.resolve()),
                    image=image_path.name,
                    individual_id=individual_id,
                    species="killer_whale",
                    source_dataset="finid20",
                )
            )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FinID-20 orca manifest.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    cropped_dir = raw_dir / "cropped_images"
    if not cropped_dir.exists():
        print(f"FinID-20 cropped_images not found: {cropped_dir}", file=sys.stderr)
        print("Run scripts/download_finid20.py first.", file=sys.stderr)
        return 2

    rows = collect_rows(cropped_dir)
    if not rows:
        print(f"No images found under {cropped_dir}", file=sys.stderr)
        return 3

    missing = verify_manifest_images(rows)
    if missing:
        print(f"{len(missing)} manifest images are missing. First missing: {missing[0]}", file=sys.stderr)
        return 4

    out_path = Path(args.out)
    write_jsonl_manifest(out_path, rows)
    stats = manifest_stats(rows)
    summary = {
        "source_dataset": "finid20",
        "doi": "10.5281/zenodo.16786268",
        "license": "CC-BY-4.0",
        "raw_dir": str(raw_dir),
        "manifest": str(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"Wrote {out_path}")
    print(f"Rows: {stats['rows']} | individuals: {stats['individuals']}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
