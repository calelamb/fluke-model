#!/usr/bin/env python3
"""Build a killer-whale/orca JSONL manifest from Happywhale metadata."""

from __future__ import annotations

import argparse
import csv
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
    is_orca_species,
    manifest_stats,
    normalize_species,
    verify_manifest_images,
    write_jsonl_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter Happywhale train.csv to killer-whale rows.")
    parser.add_argument("--csv", default=str(REPO_ROOT / "data/happywhale/train.csv"))
    parser.add_argument("--images-dir", default=str(REPO_ROOT / "data/happywhale/train_images"))
    parser.add_argument("--out", default=str(REPO_ROOT / "data/manifests/happywhale_orca.jsonl"))
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    images_dir = Path(args.images_dir)
    if not csv_path.exists():
        print(f"Happywhale CSV not found: {csv_path}", file=sys.stderr)
        print("Run scripts/download_happywhale.py first.", file=sys.stderr)
        return 2
    if not images_dir.exists():
        print(f"Happywhale image directory not found: {images_dir}", file=sys.stderr)
        return 2

    rows: list[OrcaManifestRow] = []
    species_counts: dict[str, int] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "species", "individual_id"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            print(f"{csv_path} must include columns {sorted(required)}; got {reader.fieldnames}", file=sys.stderr)
            return 3
        for record in reader:
            species = record["species"]
            normalized = normalize_species(species)
            species_counts[normalized] = species_counts.get(normalized, 0) + 1
            if not is_orca_species(species):
                continue
            image = record["image"]
            rows.append(
                OrcaManifestRow(
                    path=str((images_dir / image).resolve()),
                    image=image,
                    individual_id=record["individual_id"],
                    species=species,
                    source_dataset="happywhale",
                )
            )

    if not rows:
        print("No killer-whale/orca rows found in Happywhale metadata.", file=sys.stderr)
        print(f"Observed species labels: {sorted(species_counts)}", file=sys.stderr)
        return 4

    missing = verify_manifest_images(rows)
    if missing and not args.allow_missing:
        print(f"{len(missing)} manifest images are missing. First missing: {missing[0]}", file=sys.stderr)
        return 5
    if missing:
        missing_set = set(missing)
        rows = [row for row in rows if row.path not in missing_set]

    out_path = Path(args.out)
    write_jsonl_manifest(out_path, rows)
    stats = manifest_stats(rows)
    summary = {
        "source_csv": str(csv_path),
        "images_dir": str(images_dir),
        "manifest": str(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "filter": "species in killer whale/orca aliases",
        "species_counts": dict(sorted(species_counts.items())),
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
