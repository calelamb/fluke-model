"""Tests for public-orca dataset manifest and split rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.orca_data import (  # noqa: E402
    OrcaManifestRow,
    is_orca_species,
    read_jsonl_manifest,
    split_by_individual_images,
    verify_manifest_images,
    write_jsonl_manifest,
)


def test_orca_species_aliases():
    assert is_orca_species("killer_whale")
    assert is_orca_species("Killer Whale")
    assert is_orca_species("orca")
    assert is_orca_species("orcinus-orca")
    assert not is_orca_species("humpback_whale")


def test_jsonl_manifest_round_trip(tmp_path: Path):
    rows = [
        OrcaManifestRow(path="/tmp/a.jpg", image="a.jpg", individual_id="id1", species="killer_whale"),
        OrcaManifestRow(path="/tmp/b.jpg", image="b.jpg", individual_id="id2", species="killer_whale"),
    ]
    path = tmp_path / "manifest.jsonl"
    write_jsonl_manifest(path, rows)
    loaded = read_jsonl_manifest(path)
    assert loaded == rows


def test_verify_manifest_images(tmp_path: Path):
    image = tmp_path / "exists.jpg"
    image.write_bytes(b"not-a-real-image-but-path-exists")
    rows = [
        OrcaManifestRow(path=str(image), individual_id="id1", species="killer_whale"),
        OrcaManifestRow(path=str(tmp_path / "missing.jpg"), individual_id="id1", species="killer_whale"),
    ]
    assert verify_manifest_images(rows) == [str(tmp_path / "missing.jpg")]


def test_split_drops_singletons_and_has_no_image_leakage():
    rows: list[OrcaManifestRow] = []
    for individual in ["a", "b", "c"]:
        for i in range(4):
            rows.append(
                OrcaManifestRow(
                    path=f"/tmp/{individual}-{i}.jpg",
                    individual_id=individual,
                    species="killer_whale",
                )
            )
    rows.append(OrcaManifestRow(path="/tmp/single.jpg", individual_id="singleton", species="killer_whale"))

    train, val, test, stats = split_by_individual_images(
        rows,
        seed=7,
        val_fraction=0.25,
        test_fraction=0.25,
        min_images_per_individual=2,
    )

    assert stats["dropped_individuals"] == 1
    assert stats["dropped_images"] == 1
    assert {r.individual_id for r in test} == {"a", "b", "c"}
    all_paths = [r.path for r in train + val + test]
    assert len(all_paths) == len(set(all_paths))
    assert "/tmp/single.jpg" not in all_paths


def test_split_validates_eval_minimum():
    rows = [OrcaManifestRow(path="/tmp/a.jpg", individual_id="a", species="killer_whale")]
    with pytest.raises(ValueError):
        split_by_individual_images(rows, min_images_per_individual=1)
