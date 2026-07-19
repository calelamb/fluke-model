"""Deterministic, fail-closed mobile reference catalog publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from fluke_model.rights import EXPECTED_MODEL_LICENSE_SPDX, ModelRights, RightsError

_SCHEMA_VERSION = 1
_VECTOR_DTYPE = "float16"
SCORE_SEMANTICS = "cosineSimilarity"
_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_RIGHTS_BYTES = 1024 * 1024
_SHA256_LENGTH = 64
_RIGHTS_KEYS = {
    "schema_version",
    "approved_by",
    "approved_at",
    "commercial_use_allowed",
    "model",
    "data_sources",
}
_MODEL_RIGHTS_KEYS = {
    "model_id",
    "revision",
    "license_spdx",
    "evidence_url",
    "commercial_use_allowed",
}
_DATA_RIGHTS_KEYS = {
    "source_id",
    "license_or_permission",
    "evidence_url",
    "commercial_use_allowed",
    "redistribution_allowed",
    "mobile_ml_use_allowed",
}
_METADATA_KEYS = {"referencePhotoId", "whaleId", "catalogId", "sourceId"}
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


@dataclass(frozen=True)
class ReferenceRow:
    """Stable mobile identity metadata paired with one embedding row."""

    reference_photo_id: str
    whale_id: str
    catalog_id: str
    source_id: str


@dataclass(frozen=True)
class MobileCatalogRelease:
    """Immutable release inputs not derived from the reference vectors."""

    manifest_version: str
    model_id: str
    model_revision: str
    model_version: str
    model_sha256: str
    preprocessing_version: str
    embedding_dimension: int
    index_version: str
    score_semantics: str
    score_threshold: float
    margin_threshold: float
    rights_attestation_path: Path


@dataclass(frozen=True)
class MobileCatalogManifest:
    """Exact manifest contract consumed by the iOS bundle loader."""

    schema_version: int
    manifest_version: str
    model_id: str
    model_revision: str
    model_version: str
    model_sha256: str
    preprocessing_version: str
    embedding_dimension: int
    dtype: str
    index_version: str
    reference_count: int
    catalog_count: int
    vectors_sha256: str
    metadata_sha256: str
    rights_attestation_sha256: str
    score_semantics: str
    score_threshold: float
    margin_threshold: float


@dataclass(frozen=True)
class ValidatedMobileCatalog:
    """Immutable evidence reread from a fully validated published catalog."""

    manifest: MobileCatalogManifest
    rows: tuple[ReferenceRow, ...]
    manifest_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))


@dataclass(frozen=True)
class MobileDataRights:
    """Per-source permissions required for an offline redistributed ML bundle."""

    source_id: str
    license_or_permission: str
    evidence_url: str
    commercial_use_allowed: bool
    redistribution_allowed: bool
    mobile_ml_use_allowed: bool


@dataclass(frozen=True)
class MobileRightsAttestation:
    """Rights evidence for the exact model and every bundled reference source."""

    schema_version: int
    approved_by: str
    approved_at: datetime
    commercial_use_allowed: bool
    model: ModelRights
    data_sources: tuple[MobileDataRights, ...]

    def validate_for(
        self,
        *,
        model_id: str,
        model_revision: str,
        reference_source_ids: tuple[str, ...],
    ) -> None:
        """Require exact model/source coverage and all production bundle permissions."""
        _validate_attestation_header(self, model_id=model_id, model_revision=model_revision)
        sources = _unique_rights_sources(self.data_sources)
        expected = frozenset(reference_source_ids)
        actual = frozenset(source.source_id for source in sources)
        missing = tuple(sorted(expected - actual))
        if missing:
            raise RightsError(f"reference source is not covered by the attestation: {missing[0]}")
        extra = tuple(sorted(actual - expected))
        if extra:
            raise RightsError(f"rights attestation contains unused source: {extra[0]}")
        for source in sources:
            _validate_source_rights(source)


def manifest_payload(value: MobileCatalogManifest) -> dict[str, object]:
    """Serialize with the literal client-facing keys; never expose Python names."""
    return {
        "schemaVersion": value.schema_version,
        "manifestVersion": value.manifest_version,
        "modelId": value.model_id,
        "modelRevision": value.model_revision,
        "modelVersion": value.model_version,
        "modelSha256": value.model_sha256,
        "preprocessingVersion": value.preprocessing_version,
        "embeddingDimension": value.embedding_dimension,
        "dtype": value.dtype,
        "indexVersion": value.index_version,
        "referenceCount": value.reference_count,
        "catalogCount": value.catalog_count,
        "vectorsSha256": value.vectors_sha256,
        "metadataSha256": value.metadata_sha256,
        "rightsAttestationSha256": value.rights_attestation_sha256,
        "scoreSemantics": value.score_semantics,
        "scoreThreshold": value.score_threshold,
        "marginThreshold": value.margin_threshold,
    }


def validate_embeddings(values: np.ndarray, expected_dimension: int) -> np.ndarray:
    """Return an owned float32 row-major copy after validating the model contract."""
    if not isinstance(expected_dimension, int) or isinstance(expected_dimension, bool):
        raise ValueError("embedding dimension must be a positive integer")
    if expected_dimension <= 0:
        raise ValueError("embedding dimension must be a positive integer")
    array = np.array(values, dtype=np.float32, order="C", copy=True)
    if array.ndim != 2 or array.shape[1] != expected_dimension:
        raise ValueError("reference embedding shape does not match the model contract")
    if not np.isfinite(array).all():
        raise ValueError("reference embeddings must be finite")
    norms = np.linalg.vector_norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("reference embeddings must be L2 normalized")
    return array


def sha256_file(path: Path) -> str:
    """Hash one regular, non-symlink file without unbounded memory use."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"SHA256 source must be a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_published_mobile_catalog(catalog_dir: Path) -> ValidatedMobileCatalog:
    """Reread and fully validate an exact Task 3 catalog without trusting publication state."""
    from fluke_model.mobile_catalog_validation import validate_published_mobile_catalog as validate

    return validate(Path(catalog_dir))


def write_mobile_catalog(
    output_dir: Path,
    embeddings: np.ndarray,
    rows: Sequence[ReferenceRow],
    release: MobileCatalogRelease,
) -> MobileCatalogManifest:
    """Validate, stage, reread, and atomically publish one mobile catalog."""
    destination = Path(output_dir)
    _validate_release(release)
    rights_path = Path(release.rights_attestation_path)
    _validate_catalog_paths(destination, rights_path)
    _validate_output_destination(destination)
    normalized_rows = _validate_rows(rows)
    normalized_embeddings = validate_embeddings(embeddings, release.embedding_dimension)
    if normalized_embeddings.shape[0] != len(normalized_rows):
        raise ValueError("reference embedding row count does not match metadata row count")
    rights, rights_sha256 = _load_mobile_rights(rights_path)
    source_ids = tuple(sorted({row.source_id for row in normalized_rows}))
    rights.validate_for(
        model_id=release.model_id,
        model_revision=release.model_revision,
        reference_source_ids=source_ids,
    )
    sorted_rows, sorted_embeddings = _sort_catalog(normalized_rows, normalized_embeddings)
    return _stage_and_publish(
        destination,
        sorted_embeddings,
        sorted_rows,
        release,
        rights_sha256=rights_sha256,
    )


def _validate_release(release: MobileCatalogRelease) -> None:
    if not isinstance(release, MobileCatalogRelease):
        raise TypeError("release must be a MobileCatalogRelease")
    text_fields = (
        release.manifest_version,
        release.model_id,
        release.model_revision,
        release.model_version,
        release.preprocessing_version,
        release.index_version,
        release.score_semantics,
    )
    if any(not isinstance(value, str) or not value.strip() for value in text_fields):
        raise ValueError("release text fields must be non-empty strings")
    if release.score_semantics != SCORE_SEMANTICS:
        raise ValueError(f"score semantics must be {SCORE_SEMANTICS}")
    if not isinstance(release.embedding_dimension, int) or isinstance(
        release.embedding_dimension, bool
    ):
        raise ValueError("embedding dimension must be a positive integer")
    if release.embedding_dimension <= 0:
        raise ValueError("embedding dimension must be a positive integer")
    _validate_sha256("model SHA256", release.model_sha256)
    _validate_threshold("score threshold", release.score_threshold)
    _validate_threshold("margin threshold", release.margin_threshold)


def _validate_catalog_paths(destination: Path, rights_path: Path) -> None:
    _reject_symlink_components(destination, name="mobile catalog output")
    _reject_symlink_components(rights_path, name="rights attestation")
    resolved_destination = destination.resolve(strict=False)
    resolved_rights = rights_path.resolve(strict=False)
    if _paths_overlap(resolved_destination, resolved_rights):
        raise ValueError("mobile catalog output and rights attestation paths overlap")


def _reject_symlink_components(path: Path, *, name: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise ValueError(f"{name} path contains a symbolic link component")


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _validate_rows(rows: Sequence[ReferenceRow]) -> tuple[ReferenceRow, ...]:
    normalized = tuple(rows)
    if not normalized:
        raise ValueError("mobile catalog must contain at least one reference")
    if any(not isinstance(row, ReferenceRow) for row in normalized):
        raise TypeError("reference rows must be ReferenceRow records")
    for row in normalized:
        fields = (row.reference_photo_id, row.whale_id, row.catalog_id, row.source_id)
        if any(not isinstance(value, str) or not value.strip() for value in fields):
            raise ValueError("reference identity fields must be non-empty strings")
    reference_ids = tuple(row.reference_photo_id for row in normalized)
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("referencePhotoId values must be unique")
    _validate_stable_identity_mapping(normalized)
    return normalized


def _validate_stable_identity_mapping(rows: tuple[ReferenceRow, ...]) -> None:
    whale_ids = frozenset(row.whale_id for row in rows)
    for whale_id in whale_ids:
        catalogs = frozenset(row.catalog_id for row in rows if row.whale_id == whale_id)
        if len(catalogs) != 1:
            raise ValueError(f"whaleId must have one stable catalogId: {whale_id}")
    catalog_ids = frozenset(row.catalog_id for row in rows)
    for catalog_id in catalog_ids:
        whales = frozenset(row.whale_id for row in rows if row.catalog_id == catalog_id)
        if len(whales) != 1:
            raise ValueError(f"catalogId must have one stable whaleId: {catalog_id}")


def _load_mobile_rights(path: Path) -> tuple[MobileRightsAttestation, str]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise RightsError("rights attestation must be a regular file")
    if source.stat().st_size > _MAX_RIGHTS_BYTES:
        raise RightsError("rights attestation exceeds the size limit")
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RightsError(f"rights attestation is invalid JSON: {error}") from error
    attestation = _rights_from_payload(_require_mapping(payload, "rights attestation"))
    return attestation, hashlib.sha256(raw).hexdigest()


def _rights_from_payload(payload: Mapping[str, Any]) -> MobileRightsAttestation:
    _require_exact_keys(payload, _RIGHTS_KEYS, "rights attestation")
    model_payload = _require_mapping(payload["model"], "model rights")
    source_payloads = payload["data_sources"]
    if not isinstance(source_payloads, list) or not source_payloads:
        raise RightsError("rights attestation data_sources must be a non-empty array")
    try:
        approved_at = datetime.fromisoformat(_require_string(payload, "approved_at"))
    except ValueError as error:
        raise RightsError("rights approval date must be valid ISO-8601") from error
    return MobileRightsAttestation(
        schema_version=_require_integer(payload, "schema_version"),
        approved_by=_require_string(payload, "approved_by"),
        approved_at=approved_at,
        commercial_use_allowed=_require_boolean(payload, "commercial_use_allowed"),
        model=_model_rights_from_payload(model_payload),
        data_sources=tuple(
            _data_rights_from_payload(_require_mapping(value, "data source rights"))
            for value in source_payloads
        ),
    )


def _model_rights_from_payload(payload: Mapping[str, Any]) -> ModelRights:
    _require_exact_keys(payload, _MODEL_RIGHTS_KEYS, "model rights")
    return ModelRights(
        model_id=_require_string(payload, "model_id"),
        revision=_require_string(payload, "revision"),
        license_spdx=_require_string(payload, "license_spdx"),
        evidence_url=_require_string(payload, "evidence_url"),
        commercial_use_allowed=_require_boolean(payload, "commercial_use_allowed"),
    )


def _data_rights_from_payload(payload: Mapping[str, Any]) -> MobileDataRights:
    _require_exact_keys(payload, _DATA_RIGHTS_KEYS, "data source rights")
    return MobileDataRights(
        source_id=_require_string(payload, "source_id"),
        license_or_permission=_require_string(payload, "license_or_permission"),
        evidence_url=_require_string(payload, "evidence_url"),
        commercial_use_allowed=_require_boolean(payload, "commercial_use_allowed"),
        redistribution_allowed=_require_boolean(payload, "redistribution_allowed"),
        mobile_ml_use_allowed=_require_boolean(payload, "mobile_ml_use_allowed"),
    )


def _validate_attestation_header(
    value: MobileRightsAttestation, *, model_id: str, model_revision: str
) -> None:
    if value.schema_version != _SCHEMA_VERSION:
        raise RightsError("unsupported rights attestation schema")
    if not value.commercial_use_allowed or not value.model.commercial_use_allowed:
        raise RightsError("rights attestation does not permit commercial production use")
    if not value.approved_by.strip() or value.approved_at.tzinfo is None:
        raise RightsError("rights attestation requires an approver and timezone-aware approval date")
    if (value.model.model_id, value.model.revision) != (model_id, model_revision):
        raise RightsError("rights attestation does not match the configured model revision")
    if value.model.license_spdx != EXPECTED_MODEL_LICENSE_SPDX:
        raise RightsError("rights attestation does not match the pinned model license")
    if not _is_https_url(value.model.evidence_url):
        raise RightsError("model rights evidence must be an absolute HTTPS URL")


def _unique_rights_sources(
    sources: tuple[MobileDataRights, ...],
) -> tuple[MobileDataRights, ...]:
    source_ids = tuple(source.source_id for source in sources)
    if any(not source_id.strip() for source_id in source_ids):
        raise RightsError("rights source IDs must be non-empty")
    if len(source_ids) != len(set(source_ids)):
        raise RightsError("rights source IDs must be unique")
    return sources


def _validate_source_rights(source: MobileDataRights) -> None:
    if not source.commercial_use_allowed or not source.license_or_permission.strip():
        raise RightsError(
            f"reference source lacks commercial production rights: {source.source_id}"
        )
    if not source.redistribution_allowed:
        raise RightsError(f"reference source lacks redistribution permission: {source.source_id}")
    if not source.mobile_ml_use_allowed:
        raise RightsError(f"reference source lacks mobile ML permission: {source.source_id}")
    if not _is_https_url(source.evidence_url):
        raise RightsError(f"reference source rights evidence must use HTTPS: {source.source_id}")


def _sort_catalog(
    rows: tuple[ReferenceRow, ...], embeddings: np.ndarray
) -> tuple[tuple[ReferenceRow, ...], np.ndarray]:
    ordered_pairs = tuple(sorted(enumerate(rows), key=lambda value: value[1].reference_photo_id))
    sorted_rows = tuple(row for _, row in ordered_pairs)
    indices = np.array(tuple(index for index, _ in ordered_pairs), dtype=np.intp)
    return sorted_rows, np.array(embeddings[indices], dtype=np.float32, order="C", copy=True)


def _stage_and_publish(
    destination: Path,
    embeddings: np.ndarray,
    rows: tuple[ReferenceRow, ...],
    release: MobileCatalogRelease,
    *,
    rights_sha256: str,
) -> MobileCatalogManifest:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        vectors_bytes = embeddings.astype("<f2").tobytes(order="C")
        metadata = _metadata_payload(rows)
        (staging / "references.f16").write_bytes(vectors_bytes)
        _write_json(staging / "metadata.json", metadata)
        manifest = _build_manifest(staging, rows, release, rights_sha256=rights_sha256)
        _write_json(staging / "manifest.json", manifest_payload(manifest))
        _validate_staged_catalog(staging, manifest, metadata, vectors_bytes)
        _publish_staging(staging, destination)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _metadata_payload(rows: tuple[ReferenceRow, ...]) -> list[dict[str, str]]:
    return [
        {
            "referencePhotoId": row.reference_photo_id,
            "whaleId": row.whale_id,
            "catalogId": row.catalog_id,
            "sourceId": row.source_id,
        }
        for row in rows
    ]


def _build_manifest(
    staging: Path,
    rows: tuple[ReferenceRow, ...],
    release: MobileCatalogRelease,
    *,
    rights_sha256: str,
) -> MobileCatalogManifest:
    return MobileCatalogManifest(
        schema_version=_SCHEMA_VERSION,
        manifest_version=release.manifest_version,
        model_id=release.model_id,
        model_revision=release.model_revision,
        model_version=release.model_version,
        model_sha256=release.model_sha256,
        preprocessing_version=release.preprocessing_version,
        embedding_dimension=release.embedding_dimension,
        dtype=_VECTOR_DTYPE,
        index_version=release.index_version,
        reference_count=len(rows),
        catalog_count=len({row.catalog_id for row in rows}),
        vectors_sha256=sha256_file(staging / "references.f16"),
        metadata_sha256=sha256_file(staging / "metadata.json"),
        rights_attestation_sha256=rights_sha256,
        score_semantics=release.score_semantics,
        score_threshold=release.score_threshold,
        margin_threshold=release.margin_threshold,
    )


def _validate_staged_catalog(
    staging: Path,
    manifest: MobileCatalogManifest,
    expected_metadata: list[dict[str, str]],
    expected_vectors: bytes,
) -> None:
    try:
        raw_manifest = json.loads((staging / "manifest.json").read_text(encoding="utf-8"))
        raw_metadata = json.loads((staging / "metadata.json").read_text(encoding="utf-8"))
        raw_vectors = (staging / "references.f16").read_bytes()
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"staged mobile catalog cannot be reread: {error}") from error
    if not isinstance(raw_manifest, dict) or set(raw_manifest) != _MANIFEST_KEYS:
        raise ValueError("staged manifest schema does not match the mobile contract")
    if raw_manifest != manifest_payload(manifest):
        raise ValueError("staged manifest values changed during publication")
    if raw_metadata != expected_metadata or not _valid_metadata_schema(raw_metadata):
        raise ValueError("staged metadata does not match the mobile contract")
    if raw_vectors != expected_vectors:
        raise ValueError("staged reference vectors changed during publication")
    _validate_staged_hashes(staging, manifest, raw_vectors)


def _validate_staged_hashes(
    staging: Path, manifest: MobileCatalogManifest, raw_vectors: bytes
) -> None:
    expected_bytes = manifest.reference_count * manifest.embedding_dimension * 2
    if len(raw_vectors) != expected_bytes:
        raise ValueError("staged reference vector length does not match the manifest")
    if sha256_file(staging / "references.f16") != manifest.vectors_sha256:
        raise ValueError("staged reference vector digest does not match the manifest")
    if sha256_file(staging / "metadata.json") != manifest.metadata_sha256:
        raise ValueError("staged metadata digest does not match the manifest")


def _valid_metadata_schema(payload: object) -> bool:
    if not isinstance(payload, list):
        return False
    return all(
        isinstance(row, dict)
        and set(row) == _METADATA_KEYS
        and all(isinstance(value, str) and value for value in row.values())
        for row in payload
    )


def _validate_output_destination(destination: Path) -> None:
    if destination.is_symlink():
        raise ValueError("mobile catalog output must not be a symbolic link")
    if not destination.exists():
        return
    if not destination.is_dir():
        raise FileExistsError("mobile catalog output must be an empty directory")
    try:
        next(destination.iterdir())
    except StopIteration:
        return
    raise FileExistsError("mobile catalog output must be an empty directory")


def _publish_staging(staging: Path, destination: Path) -> None:
    _validate_output_destination(destination)
    if destination.exists():
        destination.rmdir()
    os.replace(staging, destination)


def _write_json(path: Path, payload: object) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n")
    path.write_text(encoded, encoding="utf-8")


def _validate_sha256(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    is_hex = all(character in "0123456789abcdef" for character in value)
    if len(value) != _SHA256_LENGTH or not is_hex:
        raise ValueError(f"{name} must be a lowercase SHA256 digest")


def _validate_threshold(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and within [-1, 1]")
    if not math.isfinite(float(value)) or not -1.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and within [-1, 1]")


def _is_https_url(value: str) -> bool:
    parsed = urlsplit(value)
    return parsed.scheme == "https" and bool(parsed.hostname) and parsed.username is None


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RightsError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise RightsError(f"{name} fields do not match the required schema")


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise RightsError(f"rights field must be a non-empty string: {key}")
    return value


def _require_boolean(payload: Mapping[str, Any], key: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise RightsError(f"rights field must be a boolean: {key}")
    return value


def _require_integer(payload: Mapping[str, Any], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RightsError(f"rights field must be an integer: {key}")
    return value
