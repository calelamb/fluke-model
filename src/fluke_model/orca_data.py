"""Public orca dataset manifest helpers.

The first Fluke training path uses official public dataset releases, not
scraped images. This module keeps the data rules small and testable:

- normalize/filter Happywhale species labels to killer whale/orca rows
- write JSONL manifests that reference local image paths
- split images per individual without leaking image paths across splits
- verify every manifest row still points at an image on disk
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ORCA_SPECIES_ALIASES = {
    "killer_whale",
    "killerwhale",
    "killer whale",
    "orca",
    "orcinus_orca",
    "orcinus orca",
}


@dataclass(frozen=True)
class OrcaManifestRow:
    """One image row in the public-orca training manifest."""

    path: str
    individual_id: str
    species: str
    source_dataset: str = "happywhale"
    image: str | None = None


def normalize_species(value: str) -> str:
    """Normalize noisy dataset species labels for comparison."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def is_orca_species(value: str) -> bool:
    """Return True when a dataset species label means killer whale/orca."""
    normalized = normalize_species(value)
    return normalized in {normalize_species(v) for v in ORCA_SPECIES_ALIASES}


def write_jsonl_manifest(path: str | Path, rows: Iterable[OrcaManifestRow]) -> None:
    """Write an OrcaManifestRow JSONL manifest."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def read_jsonl_manifest(path: str | Path) -> list[OrcaManifestRow]:
    """Read an OrcaManifestRow JSONL manifest."""
    manifest = Path(path)
    rows: list[OrcaManifestRow] = []
    with manifest.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                rows.append(
                    OrcaManifestRow(
                        path=payload["path"],
                        individual_id=payload["individual_id"],
                        species=payload.get("species", "killer_whale"),
                        source_dataset=payload.get("source_dataset", "happywhale"),
                        image=payload.get("image"),
                    )
                )
            except KeyError as e:
                raise ValueError(f"{manifest}:{line_no} missing required key {e}") from e
    return rows


def verify_manifest_images(rows: Iterable[OrcaManifestRow]) -> list[str]:
    """Return missing image paths from a manifest."""
    missing: list[str] = []
    for row in rows:
        if not Path(row.path).exists():
            missing.append(row.path)
    return missing


def manifest_stats(rows: Iterable[OrcaManifestRow]) -> dict:
    """Return compact traceability stats for a manifest."""
    rows = list(rows)
    by_id = Counter(row.individual_id for row in rows)
    by_species = Counter(normalize_species(row.species) for row in rows)
    return {
        "rows": len(rows),
        "individuals": len(by_id),
        "species": dict(sorted(by_species.items())),
        "images_per_individual": {
            "min": min(by_id.values()) if by_id else 0,
            "max": max(by_id.values()) if by_id else 0,
            "median": _median_int(list(by_id.values())),
        },
        "individuals_with_1_image": sum(1 for n in by_id.values() if n == 1),
        "individuals_with_2_plus_images": sum(1 for n in by_id.values() if n >= 2),
    }


def split_by_individual_images(
    rows: list[OrcaManifestRow],
    *,
    seed: int = 42,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    min_images_per_individual: int = 2,
) -> tuple[list[OrcaManifestRow], list[OrcaManifestRow], list[OrcaManifestRow], dict]:
    """Split images per individual for closed-set metric-learning evaluation.

    Individuals with fewer than `min_images_per_individual` images are excluded.
    Remaining individuals can appear in train/val/test, but each image path is in
    exactly one split. This is the usual closed-set retrieval setup: the model
    learns identities with some reference photos, then ranks held-out photos of
    those same identities.
    """
    if min_images_per_individual < 2:
        raise ValueError("min_images_per_individual must be >= 2 for closed-set eval")
    if not (0 <= val_fraction < 1 and 0 < test_fraction < 1):
        raise ValueError("val_fraction must be [0,1); test_fraction must be (0,1)")
    if val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction + test_fraction must be < 1")

    grouped: dict[str, list[OrcaManifestRow]] = defaultdict(list)
    for row in rows:
        grouped[row.individual_id].append(row)

    rng = random.Random(seed)
    train: list[OrcaManifestRow] = []
    val: list[OrcaManifestRow] = []
    test: list[OrcaManifestRow] = []
    dropped: dict[str, int] = {}

    for individual_id in sorted(grouped):
        items = list(grouped[individual_id])
        if len(items) < min_images_per_individual:
            dropped[individual_id] = len(items)
            continue

        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_fraction))
        n_val = round(n * val_fraction) if n >= 3 else 0
        if n - n_test - n_val < 1:
            n_val = max(0, n - n_test - 1)

        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])

    _assert_no_path_overlap(train, val, test)
    stats = {
        "seed": seed,
        "val_fraction": val_fraction,
        "test_fraction": test_fraction,
        "min_images_per_individual": min_images_per_individual,
        "dropped_individuals": len(dropped),
        "dropped_images": sum(dropped.values()),
        "train": manifest_stats(train),
        "val": manifest_stats(val),
        "test": manifest_stats(test),
    }
    return train, val, test, stats


def _assert_no_path_overlap(*splits: list[OrcaManifestRow]) -> None:
    seen: set[str] = set()
    for split in splits:
        for row in split:
            if row.path in seen:
                raise ValueError(f"image path leaked across splits: {row.path}")
            seen.add(row.path)


def _median_int(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return round((ordered[mid - 1] + ordered[mid]) / 2)
