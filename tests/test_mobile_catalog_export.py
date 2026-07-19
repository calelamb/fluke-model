"""Verified mobile catalog handoff contract for a future iOS bundle install."""

from __future__ import annotations

from pathlib import Path

import pytest

from fluke_model.mobile_catalog_export import export_verified_mobile_catalog
from test_mobile_release import _valid_coreml_spec, build_release_fixture


def test_export_copies_only_exact_verified_catalog_files(tmp_path: Path) -> None:
    release = build_release_fixture(tmp_path / "source")
    output = tmp_path / "IdentifierCatalog"

    manifest = export_verified_mobile_catalog(
        release,
        output,
        app_build=2,
        package_loader=lambda _path: _valid_coreml_spec(),
    )

    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        "metadata.json",
        "references.f16",
    }
    assert manifest.reference_count == 1
    assert (output / "manifest.json").read_bytes() == (
        release / "catalog" / "manifest.json"
    ).read_bytes()


def test_export_rejects_incompatible_app_build(tmp_path: Path) -> None:
    release = build_release_fixture(tmp_path / "source")

    with pytest.raises(ValueError, match="app build"):
        export_verified_mobile_catalog(
            release,
            tmp_path / "output",
            app_build=101,
            package_loader=lambda _path: _valid_coreml_spec(),
        )


def test_export_rejects_existing_destination(tmp_path: Path) -> None:
    release = build_release_fixture(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError, match="must not already exist"):
        export_verified_mobile_catalog(
            release,
            output,
            app_build=2,
            package_loader=lambda _path: _valid_coreml_spec(),
        )
