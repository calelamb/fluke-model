"""Tests for Happywhale selective download planning."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "download_happywhale.py"
SPEC = importlib.util.spec_from_file_location("download_happywhale", SCRIPT_PATH)
assert SPEC and SPEC.loader
download_happywhale = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_happywhale)


def make_row(individual_id: str, index: int, species: str = "killer_whale") -> dict[str, str]:
    return {
        "image": f"{individual_id}-{index}.jpg",
        "individual_id": individual_id,
        "species": species,
    }


def test_choose_repeated_identity_rows_prefers_repeated_identities():
    rows = [
        *[make_row("a", i, "kiler_whale") for i in range(10)],
        *[make_row("b", i) for i in range(8)],
        *[make_row("c", i) for i in range(2)],
        make_row("singleton", 0),
    ]

    selected = download_happywhale.choose_repeated_identity_rows(
        rows,
        target_identities=2,
        min_images_per_identity=8,
        max_images_per_identity=9,
    )

    assert len(selected) == 17
    assert {row["individual_id"] for row in selected} == {"a", "b"}
    assert sum(row["individual_id"] == "a" for row in selected) == 9
    assert sum(row["individual_id"] == "b" for row in selected) == 8


def test_write_download_plan_counts_existing_and_missing_images(tmp_path: Path):
    rows = [
        *[make_row("a", i, "kiler_whale") for i in range(2)],
        *[make_row("b", i) for i in range(2)],
    ]
    images_dir = tmp_path / "train_images"
    images_dir.mkdir()
    (images_dir / "a-0.jpg").write_bytes(b"present")
    args = argparse.Namespace(
        target_identities=2,
        min_images_per_identity=2,
        max_images_per_identity=2,
        max_images=0,
    )

    plan = download_happywhale.write_download_plan(
        tmp_path / "orca_download_plan.json",
        rows=rows,
        images_dir=images_dir,
        selection_mode="repeated_identity",
        args=args,
    )

    assert plan["planned_rows"] == 4
    assert plan["planned_identities"] == 2
    assert plan["already_present"] == 1
    assert plan["missing"] == 3
    assert plan["identities"][0]["species"] == ["kiler_whale"]
