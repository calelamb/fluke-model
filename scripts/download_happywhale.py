#!/usr/bin/env python3
"""Download the Happywhale public dataset through the official Kaggle API.

This script intentionally does not scrape anything. It requires the user to
authenticate with Kaggle and accept the Happywhale competition/dataset terms
through Kaggle before download.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data/happywhale"


def has_kaggle_credentials() -> bool:
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")) or (
        Path.home() / ".kaggle/kaggle.json"
    ).exists()


def unzip_archives(out_dir: Path) -> list[str]:
    extracted: list[str] = []
    for archive in sorted(out_dir.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        extracted.append(archive.name)
    return extracted


def verify_happywhale_files(out_dir: Path) -> dict:
    train_csv = out_dir / "train.csv"
    train_images = out_dir / "train_images"
    if not train_csv.exists():
        raise FileNotFoundError(f"Expected metadata file not found: {train_csv}")
    if not train_images.exists():
        raise FileNotFoundError(f"Expected image directory not found: {train_images}")
    image_count = sum(1 for p in train_images.iterdir() if p.is_file())
    return {
        "train_csv": str(train_csv),
        "train_images": str(train_images),
        "train_image_count": image_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Happywhale via Kaggle.")
    parser.add_argument("--competition", default="happy-whale-and-dolphin")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true", help="Re-download even if train.csv already exists.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not has_kaggle_credentials():
        print(
            "Kaggle credentials are missing.\n\n"
            "1. Create a Kaggle API token from https://www.kaggle.com/settings\n"
            "2. Save it as ~/.kaggle/kaggle.json, or set KAGGLE_USERNAME and KAGGLE_KEY.\n"
            "3. Open the Happywhale dataset/competition page in Kaggle and accept the terms.\n"
            "4. Re-run this script.",
            file=sys.stderr,
        )
        return 2

    if (out_dir / "train.csv").exists() and not args.force:
        print(f"Happywhale metadata already exists at {out_dir}. Use --force to re-download.")
    else:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
        except Exception as e:
            print(f"kaggle package is unavailable: {e}", file=sys.stderr)
            print("Run `uv sync` first.", file=sys.stderr)
            return 2

        api = KaggleApi()
        try:
            api.authenticate()
            api.competition_download_files(args.competition, path=str(out_dir), quiet=False)
        except Exception as e:
            print(f"Kaggle download failed: {e}", file=sys.stderr)
            print(
                "Check credentials and make sure you accepted the Happywhale terms in Kaggle.",
                file=sys.stderr,
            )
            return 3

    extracted = unzip_archives(out_dir)
    try:
        summary = verify_happywhale_files(out_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 4

    summary.update(
        {
            "dataset": args.competition,
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "extracted_archives": extracted,
            "terms_note": (
                "Downloaded through Kaggle official access path. Review Kaggle/Happywhale "
                "terms before publishing derived artifacts."
            ),
        }
    )
    summary_path = out_dir / "metadata_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Happywhale files verified. Wrote {summary_path}")
    print(f"Train images: {summary['train_image_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
