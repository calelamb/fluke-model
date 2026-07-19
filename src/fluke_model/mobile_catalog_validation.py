"""Strict reread validation for published mobile catalogs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from fluke_model.mobile_catalog import (
    SCORE_SEMANTICS,
    MobileCatalogManifest,
    ReferenceRow,
    ValidatedMobileCatalog,
    _reject_symlink_components,
    _validate_rows,
    sha256_file,
)

_SCHEMA_VERSION = 1
_VECTOR_DTYPE = "float16"
_CATALOG_FILES = frozenset({"manifest.json", "metadata.json", "references.f16"})
_MANIFEST_KEYS = {
    "schemaVersion",
    "manifestVersion",
    "modelId",
    "modelRevision",
    "modelVersion",
    "modelSha256",
    "preprocessingVersion",
    "embeddingDimension",
    "dtype",
    "indexVersion",
    "referenceCount",
    "catalogCount",
    "vectorsSha256",
    "metadataSha256",
    "rightsAttestationSha256",
    "scoreSemantics",
    "scoreThreshold",
    "marginThreshold",
}
_METADATA_KEYS = {"referencePhotoId", "whaleId", "catalogId", "sourceId"}
_SHA256_LENGTH = 64
_FLOAT16_BYTES = 2
_NORM_ATOL = 2e-3


def validate_published_mobile_catalog(catalog_dir: Path) -> ValidatedMobileCatalog:
    """Validate exact files, schemas, identities, digests, and Float16 vectors."""
    root = Path(catalog_dir)
    _validate_catalog_directory(root)
    manifest_path = root / "manifest.json"
    metadata_path = root / "metadata.json"
    vectors_path = root / "references.f16"
    manifest = _parse_manifest(_load_json_mapping(manifest_path, "catalog manifest"))
    rows = _parse_metadata(_load_json(metadata_path, "catalog metadata"))
    _validate_metadata_contract(rows, manifest)
    _validate_catalog_hashes(metadata_path, vectors_path, manifest)
    _validate_vectors(vectors_path, manifest)
    return ValidatedMobileCatalog(
        manifest=manifest,
        rows=rows,
        manifest_sha256=sha256_file(manifest_path),
    )


def _validate_catalog_directory(root: Path) -> None:
    _reject_symlink_components(root, name="published mobile catalog")
    if root.is_symlink() or not root.is_dir():
        raise ValueError("published mobile catalog must be a regular directory")
    entries = tuple(root.iterdir())
    if frozenset(path.name for path in entries) != _CATALOG_FILES:
        raise ValueError("published mobile catalog must contain exactly the required files")
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"published mobile catalog entry must be a regular file: {path.name}")


def _parse_manifest(payload: Mapping[str, Any]) -> MobileCatalogManifest:
    if set(payload) != _MANIFEST_KEYS:
        raise ValueError("catalog manifest fields do not match the exact schema")
    schema_version = _exact_schema_version(payload["schemaVersion"])
    embedding_dimension = _positive_integer(payload["embeddingDimension"], "embeddingDimension")
    reference_count = _positive_integer(payload["referenceCount"], "referenceCount")
    catalog_count = _positive_integer(payload["catalogCount"], "catalogCount")
    return MobileCatalogManifest(
        schema_version=schema_version,
        manifest_version=_nonempty_text(payload["manifestVersion"], "manifestVersion"),
        model_id=_nonempty_text(payload["modelId"], "modelId"),
        model_revision=_nonempty_text(payload["modelRevision"], "modelRevision"),
        model_version=_nonempty_text(payload["modelVersion"], "modelVersion"),
        model_sha256=_sha256(payload["modelSha256"], "modelSha256"),
        preprocessing_version=_nonempty_text(
            payload["preprocessingVersion"], "preprocessingVersion"
        ),
        embedding_dimension=embedding_dimension,
        dtype=_exact_text(payload["dtype"], _VECTOR_DTYPE, "dtype"),
        index_version=_nonempty_text(payload["indexVersion"], "indexVersion"),
        reference_count=reference_count,
        catalog_count=catalog_count,
        vectors_sha256=_sha256(payload["vectorsSha256"], "vectorsSha256"),
        metadata_sha256=_sha256(payload["metadataSha256"], "metadataSha256"),
        rights_attestation_sha256=_sha256(
            payload["rightsAttestationSha256"], "rightsAttestationSha256"
        ),
        score_semantics=_exact_text(
            payload["scoreSemantics"], SCORE_SEMANTICS, "scoreSemantics"
        ),
        score_threshold=_finite_threshold(payload["scoreThreshold"], "scoreThreshold"),
        margin_threshold=_finite_threshold(payload["marginThreshold"], "marginThreshold"),
    )


def _parse_metadata(payload: object) -> tuple[ReferenceRow, ...]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("catalog metadata must be a non-empty JSON array")
    return _validate_rows(tuple(_parse_metadata_row(value) for value in payload))


def _parse_metadata_row(value: object) -> ReferenceRow:
    if not isinstance(value, dict) or set(value) != _METADATA_KEYS:
        raise ValueError("catalog metadata row fields do not match the exact schema")
    return ReferenceRow(
        reference_photo_id=_nonempty_text(value["referencePhotoId"], "referencePhotoId"),
        whale_id=_nonempty_text(value["whaleId"], "whaleId"),
        catalog_id=_nonempty_text(value["catalogId"], "catalogId"),
        source_id=_nonempty_text(value["sourceId"], "sourceId"),
    )


def _validate_metadata_contract(
    rows: tuple[ReferenceRow, ...], manifest: MobileCatalogManifest
) -> None:
    if len(rows) != manifest.reference_count:
        raise ValueError("referenceCount does not match catalog metadata")
    distinct_catalogs = len(frozenset(row.catalog_id for row in rows))
    if distinct_catalogs != manifest.catalog_count:
        raise ValueError("catalogCount does not match distinct catalog IDs")
    reference_ids = tuple(row.reference_photo_id for row in rows)
    if reference_ids != tuple(sorted(reference_ids)):
        raise ValueError("catalog metadata must be sorted by referencePhotoId")


def _validate_catalog_hashes(
    metadata_path: Path,
    vectors_path: Path,
    manifest: MobileCatalogManifest,
) -> None:
    if sha256_file(metadata_path) != manifest.metadata_sha256:
        raise ValueError("catalog metadata digest does not match manifest")
    if sha256_file(vectors_path) != manifest.vectors_sha256:
        raise ValueError("catalog vector digest does not match manifest")


def _validate_vectors(vectors_path: Path, manifest: MobileCatalogManifest) -> None:
    raw = vectors_path.read_bytes()
    expected_bytes = (
        manifest.reference_count * manifest.embedding_dimension * _FLOAT16_BYTES
    )
    if len(raw) != expected_bytes:
        raise ValueError("catalog vector length does not match manifest dimensions")
    vectors = np.frombuffer(raw, dtype="<f2").reshape(
        manifest.reference_count, manifest.embedding_dimension
    )
    values = vectors.astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("catalog vectors must be finite")
    norms = np.linalg.vector_norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=_NORM_ATOL, rtol=0.0):
        raise ValueError("catalog vectors must be L2 normalized")


def _load_json_mapping(path: Path, name: str) -> Mapping[str, Any]:
    value = _load_json(path, name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _load_json(path: Path, name: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error


def _exact_schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != _SCHEMA_VERSION:
        raise ValueError("schemaVersion must be the integer 1")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_threshold(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and within [-1, 1]")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and within [-1, 1]") from error
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and within [-1, 1]")
    return result


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _exact_text(value: object, expected: str, name: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{name} must be {expected}")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA256")
    valid = len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )
    if not valid:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value
