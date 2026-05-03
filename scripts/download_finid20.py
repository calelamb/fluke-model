#!/usr/bin/env python3
"""Idempotent download of the FinID-20 killer whale photo-id dataset.

Source: https://zenodo.org/records/16786268
DOI: 10.5281/zenodo.16786268
License: CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)

Citation: Bergler et al., "Advances in Deep Learning-Driven Photo Identification
and Meta Analysis of Cetaceans in Large Data Repositories - Supplementary Data."

The dataset is 500 cropped images of 20 Bigg's killer whale individuals
(25 photos per individual) with YOLO bounding-box annotations and a published
train/val/test split. This is the public sample of the larger FIN-PRINT dataset.

We:
  1. Download the ZIP to data/finid-20/finid-20.zip if not present.
  2. Extract to data/finid-20/raw/ if not present.
  3. Print a summary of the extracted structure.

The manifest builder lives in scripts/build_finid20_manifest.py.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://zenodo.org/api/records/16786268/files/finId-20.zip/content"
DATASET_LICENSE = "CC-BY-4.0"
DATASET_DOI = "10.5281/zenodo.16786268"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "finid-20"


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


def extract_zip(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive_path.name} -> {dest_dir}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(dest_dir)


def summarize_extracted(raw_dir: Path) -> None:
    """Print top-level structure and a count of images per common extension."""
    if not raw_dir.exists():
        return
    top_entries = sorted(p.name for p in raw_dir.iterdir())
    print(f"  raw entries: {top_entries[:10]}{' ...' if len(top_entries) > 10 else ''}")
    image_exts = {".jpg", ".jpeg", ".png"}
    image_count = sum(1 for p in raw_dir.rglob("*") if p.suffix.lower() in image_exts)
    txt_count = sum(1 for p in raw_dir.rglob("*.txt"))
    json_count = sum(1 for p in raw_dir.rglob("*.json"))
    print(f"  images: {image_count} | .txt files: {txt_count} | .json files: {json_count}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download FinID-20.")
    parser.add_argument("--url", default=DATASET_URL, help="Override the source URL.")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Destination directory.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    raw_dir = data_dir / "raw"
    archive_path = data_dir / "finid-20.zip"

    print(f"FinID-20 download to {data_dir}")
    print(f"  license: {DATASET_LICENSE}")
    print(f"  doi:     {DATASET_DOI}")
    print()

    if not archive_path.exists():
        try:
            download_with_progress(args.url, archive_path)
        except Exception as e:
            print(f"DOWNLOAD FAILED: {e}", file=sys.stderr)
            return 2
    else:
        size_mb = archive_path.stat().st_size // (1024 * 1024)
        print(f"  archive already present: {archive_path} ({size_mb} MB)")

    if not raw_dir.exists() or not any(raw_dir.iterdir()):
        try:
            extract_zip(archive_path, raw_dir)
        except Exception as e:
            print(f"EXTRACT FAILED: {e}", file=sys.stderr)
            return 3
    else:
        print(f"  already extracted: {raw_dir}")

    summarize_extracted(raw_dir)
    print()
    print("Next: scripts/build_finid20_manifest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
