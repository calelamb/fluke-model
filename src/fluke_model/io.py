"""Filesystem and manifest helpers for the M-Model-0 prototype."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from PIL import Image


@dataclass(frozen=True)
class ManifestRow:
    """One photo in a re-identification manifest.

    Attributes:
        path: Absolute path to the image file.
        individual_id: The catalog identity this photo belongs to.
    """

    path: str
    individual_id: str


def read_manifest(manifest_path: str | Path) -> list[ManifestRow]:
    """Read a CSV manifest of (path, individual_id) rows.

    The manifest must have a header with at least `path` and `individual_id` columns.
    Returns an immutable list of ManifestRow objects.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows: list[ManifestRow] = []
    with manifest_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "path" not in reader.fieldnames or "individual_id" not in reader.fieldnames:
            raise ValueError(
                f"Manifest {manifest_path} must have columns 'path' and 'individual_id'; "
                f"got {reader.fieldnames}"
            )
        for r in reader:
            rows.append(ManifestRow(path=r["path"], individual_id=r["individual_id"]))
    return rows


def write_manifest(manifest_path: str | Path, rows: Iterable[ManifestRow]) -> None:
    """Write a manifest CSV (path, individual_id) at the given location."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "individual_id"])
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def load_image(path: str | Path) -> Image.Image:
    """Load an image as RGB. Caller is responsible for closing or letting the GC do it."""
    return Image.open(path).convert("RGB")


def write_json(path: str | Path, payload: dict) -> None:
    """Write JSON with two-space indent, ensuring parent dir exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)


def read_json(path: str | Path) -> dict:
    """Read JSON; raises if the file is missing."""
    with open(path) as f:
        return json.load(f)
