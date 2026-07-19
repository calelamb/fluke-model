"""Atomic export of an independently verified catalog for an iOS bundle boundary."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from fluke_model.mobile_catalog import MobileCatalogManifest, validate_published_mobile_catalog
from fluke_model.mobile_release import MobileReleaseReport, verify_mobile_release_directory
from fluke_model.mobile_release_builder import (
    INDEX_VERSION,
    MAXIMUM_REFERENCE_COUNT,
    MODEL_VERSION,
)
from fluke_model.secure_snapshot import publish_directory_no_replace, snapshot_regular_file

_CATALOG_FILES = ("manifest.json", "metadata.json", "references.f16")


def export_verified_mobile_catalog(
    release_dir: Path,
    output_dir: Path,
    *,
    app_build: int,
) -> MobileCatalogManifest:
    """Reverify a release, enforce iOS compatibility, and copy exactly three files."""
    return _export_verified_mobile_catalog(
        release_dir,
        output_dir,
        app_build=app_build,
        verifier=verify_mobile_release_directory,
    )


def _export_verified_mobile_catalog_for_testing(
    release_dir: Path,
    output_dir: Path,
    *,
    app_build: int,
    verifier: Callable[[Path], MobileReleaseReport],
) -> MobileCatalogManifest:
    """Exercise catalog export with an explicit test-only release verifier."""
    return _export_verified_mobile_catalog(
        release_dir,
        output_dir,
        app_build=app_build,
        verifier=verifier,
    )


def _export_verified_mobile_catalog(
    release_dir: Path,
    output_dir: Path,
    *,
    app_build: int,
    verifier: Callable[[Path], MobileReleaseReport],
) -> MobileCatalogManifest:
    release = Path(release_dir)
    output = Path(output_dir)
    _reject_symlinks(release, "mobile release")
    _reject_symlinks(output, "catalog output")
    resolved_release = release.resolve(strict=False)
    resolved_output = output.resolve(strict=False)
    if (
        resolved_release == resolved_output
        or resolved_release.is_relative_to(resolved_output)
        or resolved_output.is_relative_to(resolved_release)
    ):
        raise ValueError("catalog output overlaps the verified release")
    if output.exists():
        raise FileExistsError("catalog output must not already exist")
    if isinstance(app_build, bool) or not isinstance(app_build, int) or app_build <= 0:
        raise ValueError("app build must be a positive integer")
    report = verifier(release)
    if not report.ready:
        failed = tuple(gate.name for gate in report.gates if not gate.passed)
        raise ValueError(f"mobile release is not verified: {', '.join(failed)}")
    validated = validate_published_mobile_catalog(release / "catalog")
    manifest = validated.manifest
    if manifest.reference_count > MAXIMUM_REFERENCE_COUNT:
        raise ValueError("catalog exceeds the 50000-reference iOS loader limit")
    if manifest.model_version != MODEL_VERSION or manifest.index_version != INDEX_VERSION:
        raise ValueError("catalog model/index version is incompatible with the iOS loader")
    if not manifest.minimum_app_build <= app_build <= manifest.maximum_app_build:
        raise ValueError("app build is outside the catalog compatibility range")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        for name in _CATALOG_FILES:
            snapshot_regular_file(release / "catalog" / name, staging / name)
        copied = validate_published_mobile_catalog(staging)
        if copied.manifest_sha256 != validated.manifest_sha256:
            raise ValueError("exported catalog digest changed during publication")
        publish_directory_no_replace(staging, output)
        return copied.manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _reject_symlinks(path: Path, name: str) -> None:
    absolute = Path(path).absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise ValueError(f"{name} path contains a symbolic link component")
