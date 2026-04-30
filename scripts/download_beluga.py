#!/usr/bin/env python3
"""Idempotent download of the Beluga ID 2022 dataset.

Source: https://lila.science/datasets/beluga-id-2022/
License: CDLA-Permissive-2.0 (https://cdla.io/permissive-2-0/)

The dataset is hosted as a single ZIP on lila.science. We:
  1. Download to data/beluga-id-2022/beluga-id-2022.zip if not present.
  2. Extract to data/beluga-id-2022/raw/ if not present.
  3. Walk the extracted tree to build data/beluga-id-2022/manifest.csv with
     (path, individual_id) rows.

If the network call fails or the dataset is unreachable, the script writes a
SHORTFALL note and exits non-zero so the eval pipeline knows to use the
synthetic fallback (see scripts/evaluate.py --synthetic).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

# Per https://lila.science/datasets/beluga-id-2022/ the official tarball is hosted
# on Microsoft's lila storage. We try the canonical URL first; if it 404s we surface
# a clear message rather than silently failing.
# GCP mirror is usually fastest for the lila-hosted Wild-Me datasets. AWS and Azure
# mirrors are listed at https://lila.science/datasets/beluga-id-2022/ as fallbacks.
DATASET_URL = "https://storage.googleapis.com/public-datasets-lila/wild-me/beluga.coco.tar.gz"
DATASET_MIRRORS = [
    "https://storage.googleapis.com/public-datasets-lila/wild-me/beluga.coco.tar.gz",
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/wild-me/beluga.coco.tar.gz",
    "http://us-west-2.opendata.source.coop.s3.amazonaws.com/agentmorris/lila-wildlife/wild-me/beluga.coco.tar.gz",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "beluga-id-2022"


def download_with_progress(url: str, dest: Path) -> None:
    """Download `url` to `dest`, printing simple progress."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    print(f"  -> {dest}")

    def _hook(blocks_done: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100, (blocks_done * block_size * 100) // total_size)
            mb = (blocks_done * block_size) // (1024 * 1024)
            print(f"\r  {pct}% ({mb} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _hook)
    print()


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive_path.name} -> {dest_dir}")
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_dir)
    elif archive_path.name.endswith(".tar.gz") or archive_path.suffix in (".tgz", ".gz"):
        import tarfile

        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(dest_dir)
    else:
        raise ValueError(f"Unknown archive type: {archive_path}")


def find_coco_annotations(raw_dir: Path) -> Path | None:
    """Locate the COCO annotations JSON within the extracted tree."""
    for p in raw_dir.rglob("*.json"):
        # Beluga ID 2022 ships annotations as `beluga.coco.json` or similar.
        if "coco" in p.name.lower() or p.name.lower() == "annotations.json":
            return p
    return None


def build_manifest_from_coco(coco_path: Path, raw_dir: Path, out_csv: Path) -> int:
    """Walk a COCO JSON to produce (path, individual_id) rows.

    Beluga ID 2022 stores per-image annotations with the individual id as a
    `name` or `individual` attribute. We probe a few field names since lila
    datasets vary. Returns the number of rows written.
    """
    with coco_path.open() as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    rows: list[tuple[str, str]] = []

    for ann in coco.get("annotations", []):
        img = images_by_id.get(ann.get("image_id"))
        if img is None:
            continue
        # Try common field names for the individual id
        ind_id = (
            ann.get("name")
            or ann.get("individual")
            or ann.get("individual_id")
            or ann.get("identity")
            or img.get("name")
            or img.get("individual")
        )
        if not ind_id:
            continue
        # File path is relative to `raw_dir` per COCO convention
        file_name = img.get("file_name") or img.get("filename")
        if not file_name:
            continue
        candidate = raw_dir / file_name
        if not candidate.exists():
            # Try locating it under raw_dir/images or any subdir
            matches = list(raw_dir.rglob(Path(file_name).name))
            if not matches:
                continue
            candidate = matches[0]
        rows.append((str(candidate.resolve()), str(ind_id)))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "individual_id"])
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Beluga ID 2022.")
    parser.add_argument("--url", default=DATASET_URL, help="Override the source URL.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Destination directory.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    archive_name = Path(args.url).name
    archive_path = data_dir / archive_name
    manifest_csv = data_dir / "manifest.csv"

    print(f"Beluga ID 2022 download to {data_dir}")
    print("  license: CDLA-Permissive-2.0")
    print()

    # Step 1: download
    if not archive_path.exists():
        try:
            download_with_progress(args.url, archive_path)
        except Exception as e:
            print(f"DOWNLOAD FAILED: {e}", file=sys.stderr)
            print("Recommend running scripts/evaluate.py --synthetic for stub numbers.", file=sys.stderr)
            return 2
    else:
        print(f"  archive already present: {archive_path}")

    # Step 2: extract
    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        try:
            extract_archive(archive_path, raw_dir)
        except Exception as e:
            print(f"EXTRACT FAILED: {e}", file=sys.stderr)
            return 3
    else:
        print(f"  already extracted: {raw_dir}")

    # Step 3: build manifest
    coco_path = find_coco_annotations(raw_dir)
    if coco_path is None:
        print("MANIFEST FAILED: no COCO annotations JSON found under raw/", file=sys.stderr)
        return 4

    n = build_manifest_from_coco(coco_path, raw_dir, manifest_csv)
    print(f"  manifest written: {manifest_csv} ({n} rows)")
    if n < 50:
        print("  WARNING: < 50 rows — annotations parsing may have missed fields.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
