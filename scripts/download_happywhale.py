#!/usr/bin/env python3
"""Download the Happywhale public dataset through the official Kaggle API.

This script intentionally does not scrape anything. It requires the user to
authenticate with Kaggle and accept the Happywhale competition/dataset terms
through Kaggle before download.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.orca_data import is_orca_species, manifest_stats, OrcaManifestRow  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data/happywhale"


def has_kaggle_credentials() -> bool:
    kaggle_dir = Path.home() / ".kaggle"
    return (
        bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        or bool(os.environ.get("KAGGLE_API_TOKEN"))
        or (kaggle_dir / "kaggle.json").exists()
        or (kaggle_dir / "access_token").exists()
    )


def unzip_archives(out_dir: Path) -> list[str]:
    extracted: list[str] = []
    for archive in sorted(out_dir.glob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        extracted.append(archive.name)
    return extracted


def unzip_file_archive(archive: Path, dest_dir: Path, *, remove_archive: bool = True) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
        extracted = zf.namelist()
    if remove_archive:
        archive.unlink(missing_ok=True)
    return extracted


def ensure_train_csv(api, competition: str, out_dir: Path) -> Path:
    train_csv = out_dir / "train.csv"
    if train_csv.exists():
        return train_csv
    print("Downloading train.csv metadata...")
    api.competition_download_file(competition, "train.csv", path=str(out_dir), quiet=False)
    archive = out_dir / "train.csv.zip"
    if archive.exists():
        unzip_file_archive(archive, out_dir)
    if not train_csv.exists():
        raise FileNotFoundError(f"Expected train.csv after download: {train_csv}")
    return train_csv


def read_orca_rows(train_csv: Path) -> list[dict]:
    with train_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"image", "species", "individual_id"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"{train_csv} must include {sorted(required)}; got {reader.fieldnames}")
        return [row for row in reader if is_orca_species(row["species"])]


def choose_download_rows(rows: list[dict], max_images: int) -> list[dict]:
    """Choose a balanced subset when max_images is set.

    We prefer repeated identities and keep each selected identity's image group
    together. Metric learning needs positive pairs; a broad one-photo-per-ID
    sample is mostly useless for the first training pass.
    """
    if max_images <= 0 or len(rows) <= max_images:
        return rows
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["individual_id"]].append(row)
    groups = [items for _id, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))]
    selected: list[dict] = []
    for group in groups:
        if len(group) < 2:
            continue
        remaining = max_images - len(selected)
        if remaining <= 0:
            break
        if len(group) <= remaining:
            selected.extend(group)
        elif remaining >= 2:
            selected.extend(group[:remaining])
            break
    return selected


def download_orca_images(api, competition: str, rows: list[dict], images_dir: Path) -> dict:
    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed: list[str] = []
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        image = row["image"]
        target = images_dir / image
        if target.exists():
            skipped += 1
            continue
        archive = images_dir / f"{image}.zip"
        try:
            api.competition_download_file(
                competition,
                f"train_images/{image}",
                path=str(images_dir),
                quiet=True,
            )
            if archive.exists():
                unzip_file_archive(archive, images_dir)
            if not target.exists():
                raise FileNotFoundError(f"download did not produce {target}")
            downloaded += 1
        except Exception as e:
            failed.append(f"{image}: {e}")
        if idx % 25 == 0 or idx == total:
            print(f"  images {idx}/{total} | downloaded {downloaded} | skipped {skipped} | failed {len(failed)}")
    return {"downloaded": downloaded, "skipped": skipped, "failed": failed}


def verify_happywhale_files(out_dir: Path, *, require_images: bool = True) -> dict:
    train_csv = out_dir / "train.csv"
    train_images = out_dir / "train_images"
    if not train_csv.exists():
        raise FileNotFoundError(f"Expected metadata file not found: {train_csv}")
    if require_images and not train_images.exists():
        raise FileNotFoundError(f"Expected image directory not found: {train_images}")
    image_count = sum(1 for p in train_images.glob("*.jpg")) if train_images.exists() else 0
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
    parser.add_argument(
        "--full-archive",
        action="store_true",
        help="Download the full 57GB competition archive. Default downloads only orca rows.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Limit selective orca-image downloads for a first local run (0 = all orca images).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not has_kaggle_credentials():
        print(
            "Kaggle credentials are missing.\n\n"
            "1. Create a Kaggle API token from https://www.kaggle.com/settings\n"
            "2. Save it as ~/.kaggle/access_token, ~/.kaggle/kaggle.json, or set KAGGLE_API_TOKEN.\n"
            "3. Open the Happywhale dataset/competition page in Kaggle and accept the terms.\n"
            "4. Re-run this script.",
            file=sys.stderr,
        )
        return 2

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        print(f"kaggle package is unavailable: {e}", file=sys.stderr)
        print("Run `uv sync` first.", file=sys.stderr)
        return 2

    api = KaggleApi()
    try:
        api.authenticate()
        if args.full_archive:
            if (out_dir / "train.csv").exists() and not args.force:
                print(f"Happywhale metadata already exists at {out_dir}. Use --force to re-download.")
            else:
                print("Downloading the full Happywhale archive (~57GB).")
                api.competition_download_files(args.competition, path=str(out_dir), quiet=False)
            extracted = unzip_archives(out_dir)
            summary = verify_happywhale_files(out_dir)
        else:
            train_csv = ensure_train_csv(api, args.competition, out_dir)
            orca_rows = read_orca_rows(train_csv)
            selected_rows = choose_download_rows(orca_rows, args.max_images)
            print(
                f"Happywhale metadata has {len(orca_rows)} killer-whale/orca rows; "
                f"downloading {len(selected_rows)} image files."
            )
            download_summary = download_orca_images(
                api,
                args.competition,
                selected_rows,
                out_dir / "train_images",
            )
            if download_summary["failed"]:
                print("Some image downloads failed:", file=sys.stderr)
                for item in download_summary["failed"][:10]:
                    print(f"  {item}", file=sys.stderr)
                if len(download_summary["failed"]) > 10:
                    print(f"  ... {len(download_summary['failed']) - 10} more", file=sys.stderr)
                return 5
            extracted = []
            summary = verify_happywhale_files(out_dir)
            selected_manifest_rows = [
                OrcaManifestRow(
                    path=str((out_dir / "train_images" / row["image"]).resolve()),
                    image=row["image"],
                    individual_id=row["individual_id"],
                    species=row["species"],
                    source_dataset="happywhale",
                )
                for row in selected_rows
            ]
            summary["orca_rows_in_metadata"] = len(orca_rows)
            summary["selected_orca_rows"] = len(selected_rows)
            summary["selected_orca_stats"] = manifest_stats(selected_manifest_rows)
            summary["download_summary"] = {
                "downloaded": download_summary["downloaded"],
                "skipped": download_summary["skipped"],
                "failed": len(download_summary["failed"]),
            }
    except Exception as e:
        print(f"Kaggle download failed: {e}", file=sys.stderr)
        print(
            "Check credentials and make sure you accepted the Happywhale terms in Kaggle.",
            file=sys.stderr,
        )
        return 3

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
