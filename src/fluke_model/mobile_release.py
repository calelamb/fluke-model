"""Deterministic, fail-closed verification for an on-device model release."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from fluke_model.coreml_artifact import (
    COMPUTE_PRECISION,
    MINIMUM_DEPLOYMENT_TARGET,
    CoreMLExportError,
    package_tree_sha256,
)
from fluke_model.embedders import DINO_V2_MODEL_ID, DINO_V2_REVISION
from fluke_model.mobile_catalog import (
    SCORE_SEMANTICS,
    RightsError,
    _load_mobile_rights,
    sha256_file,
)
from fluke_model.mobile_export import mobile_model_contract
from fluke_model.model_artifact import DINOV2_ARTIFACT_SHA256

REPORT_SCHEMA_VERSION = 1
REPORT_FILENAME = "mobile-release-report.json"
RELEASE_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        "parity_cosine": 0.999,
        "top_1": 0.65,
        "top_3": 0.80,
        "false_accept": 0.05,
    }
)

_BOUNDARY_GATE_NAMES = (
    "input_paths",
    "package",
    "catalog",
    "digests",
    "rights",
    "embedding_shape",
    "embedding_norm",
    "required_reports",
)
_EXPORT_METADATA_KEYS = {
    "compute_precision",
    "input_shape",
    "minimum_deployment_target",
    "model_id",
    "model_revision",
    "model_sha256",
    "output_shape",
    "package_sha256",
    "preprocessing_version",
    "tool_versions",
}
_TOOL_VERSIONS = MappingProxyType(
    {
        "coremltools": "9.0",
        "numpy": "2.2.6",
        "python": "3.11.15",
        "torch": "2.13.0",
        "transformers": "5.14.0",
    }
)
_CATALOG_MANIFEST_KEYS = {
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
_CATALOG_METADATA_KEYS = {"referencePhotoId", "whaleId", "catalogId", "sourceId"}
_CLOSED_REPORT_KEYS = {
    "schemaVersion",
    "evaluationType",
    "modelPackageSha256",
    "catalogManifestSha256",
    "sampleCount",
    "top1",
    "top3",
}
_OPEN_REPORT_KEYS = {
    "schemaVersion",
    "evaluationType",
    "modelPackageSha256",
    "catalogManifestSha256",
    "sampleCount",
    "falseAcceptRate",
}
_OPEN_REPORTS = (
    ("openSet", "evaluation/open-set.json"),
    ("nonOrca", "evaluation/non-orca.json"),
    ("poorQuality", "evaluation/poor-quality.json"),
    ("occlusion", "evaluation/occlusion.json"),
    ("distributionShift", "evaluation/distribution-shift.json"),
)
_SHA256_LENGTH = 64
_EMBEDDING_DIMENSION = 384


@dataclass(frozen=True)
class ValidationEvidence:
    """One immutable non-metric validation result."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class MobileReleaseEvidence:
    """Validated release facts consumed by the threshold gate."""

    parity_cosine: float | None
    parity_sample_count: int
    top_1: float | None
    top_3: float | None
    closed_set_sample_count: int
    false_accept: float | None
    open_set_sample_count: int
    validations: tuple[ValidationEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "validations", tuple(self.validations))


@dataclass(frozen=True)
class GateResult:
    """Serializable evidence for one non-negotiable release gate."""

    name: str
    passed: bool
    observed: bool | float | int | str | None
    requirement: str
    detail: str


@dataclass(frozen=True)
class MobileReleaseReport:
    """Deterministic release decision with no timestamps or host-specific paths."""

    ready: bool
    gates: tuple[GateResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "gates", tuple(self.gates))


@dataclass(frozen=True)
class _ReleasePaths:
    root: Path
    package: Path
    export_metadata: Path
    catalog_manifest: Path
    catalog_vectors: Path
    catalog_metadata: Path
    rights: Path
    pytorch_embeddings: Path
    coreml_embeddings: Path
    closed_report: Path
    open_reports: tuple[tuple[str, Path], ...]

    def required_files(self) -> tuple[tuple[str, Path], ...]:
        return (
            ("export-metadata.json", self.export_metadata),
            ("catalog/manifest.json", self.catalog_manifest),
            ("catalog/references.f16", self.catalog_vectors),
            ("catalog/metadata.json", self.catalog_metadata),
            ("rights-attestation.json", self.rights),
            ("evaluation/parity-pytorch.npy", self.pytorch_embeddings),
            ("evaluation/parity-coreml.npy", self.coreml_embeddings),
            ("evaluation/closed-set.json", self.closed_report),
            *((relative, path) for _, relative, path in self._open_path_records()),
        )

    def _open_path_records(self) -> tuple[tuple[str, str, Path], ...]:
        return tuple(
            (kind, relative, path)
            for (kind, relative), (_, path) in zip(_OPEN_REPORTS, self.open_reports, strict=True)
        )


@dataclass(frozen=True)
class _PackageEvidence:
    digest: str | None
    metadata: Mapping[str, Any] | None
    validation: ValidationEvidence


@dataclass(frozen=True)
class _CatalogEvidence:
    manifest: Mapping[str, Any] | None
    metadata: tuple[Mapping[str, str], ...]
    manifest_digest: str | None
    validation: ValidationEvidence
    digest_validation: ValidationEvidence


@dataclass(frozen=True)
class _EmbeddingEvidence:
    parity_cosine: float | None
    sample_count: int
    shape_validation: ValidationEvidence
    norm_validation: ValidationEvidence


@dataclass(frozen=True)
class _ReportEvidence:
    top_1: float | None
    top_3: float | None
    closed_count: int
    false_accept: float | None
    open_count: int
    validation: ValidationEvidence


def verify_mobile_release(evidence: MobileReleaseEvidence) -> MobileReleaseReport:
    """Apply every approved boundary, count, and numeric release gate."""
    if not isinstance(evidence, MobileReleaseEvidence):
        raise TypeError("evidence must be MobileReleaseEvidence")
    boundary_gates = _boundary_gates(evidence.validations)
    metric_gates = (
        _count_gate("parity_samples", evidence.parity_sample_count),
        _metric_gate("parity_cosine", evidence.parity_cosine),
        _count_gate("closed_set_samples", evidence.closed_set_sample_count),
        _metric_gate("top_1", evidence.top_1),
        _metric_gate("top_3", evidence.top_3),
        _count_gate("open_set_samples", evidence.open_set_sample_count),
        _metric_gate("false_accept", evidence.false_accept),
    )
    gates = (*boundary_gates, *metric_gates)
    return MobileReleaseReport(ready=all(gate.passed for gate in gates), gates=gates)


def verify_mobile_release_directory(release_dir: Path) -> MobileReleaseReport:
    """Inspect the fixed release layout and return a complete fail-closed report."""
    paths = _release_paths(Path(release_dir))
    path_validation = _validate_input_paths(paths)
    package = _inspect_package(paths)
    catalog = _inspect_catalog(paths, package)
    rights = _inspect_rights(paths, catalog)
    embeddings = _inspect_embeddings(paths)
    reports = _inspect_reports(paths, package, catalog)
    evidence = MobileReleaseEvidence(
        parity_cosine=embeddings.parity_cosine,
        parity_sample_count=embeddings.sample_count,
        top_1=reports.top_1,
        top_3=reports.top_3,
        closed_set_sample_count=reports.closed_count,
        false_accept=reports.false_accept,
        open_set_sample_count=reports.open_count,
        validations=(
            path_validation,
            package.validation,
            catalog.validation,
            catalog.digest_validation,
            rights,
            embeddings.shape_validation,
            embeddings.norm_validation,
            reports.validation,
        ),
    )
    return verify_mobile_release(evidence)


def report_payload(report: MobileReleaseReport) -> dict[str, object]:
    """Return the exact deterministic mobile release report schema."""
    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "ready": report.ready,
        "thresholds": dict(RELEASE_THRESHOLDS),
        "gates": [
            {
                "name": gate.name,
                "passed": gate.passed,
                "observed": gate.observed,
                "requirement": gate.requirement,
                "detail": gate.detail,
            }
            for gate in report.gates
        ],
    }


def write_mobile_release_report(path: Path, report: MobileReleaseReport) -> None:
    """Atomically write canonical finite JSON without following symbolic links."""
    destination = Path(path)
    _reject_symlink_components(destination, "release report")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            report_payload(report),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_report_destination(release_dir: Path, report_path: Path) -> None:
    """Reject a report destination that could replace any release input."""
    paths = _release_paths(Path(release_dir))
    destination = Path(report_path)
    _reject_symlink_components(destination, "release report")
    resolved = destination.resolve(strict=False)
    protected = (
        paths.package,
        paths.catalog_manifest.parent,
        paths.closed_report.parent,
        *(path for _, path in paths.required_files()),
    )
    for source in protected:
        source_resolved = source.resolve(strict=False)
        if (
            resolved == source_resolved
            or resolved.is_relative_to(source_resolved)
            or source_resolved.is_relative_to(resolved)
        ):
            raise ValueError("release report path overlaps a release input")


def _release_paths(root: Path) -> _ReleasePaths:
    evaluation = root / "evaluation"
    return _ReleasePaths(
        root=root,
        package=root / "FlukeEmbedder.mlpackage",
        export_metadata=root / "export-metadata.json",
        catalog_manifest=root / "catalog" / "manifest.json",
        catalog_vectors=root / "catalog" / "references.f16",
        catalog_metadata=root / "catalog" / "metadata.json",
        rights=root / "rights-attestation.json",
        pytorch_embeddings=evaluation / "parity-pytorch.npy",
        coreml_embeddings=evaluation / "parity-coreml.npy",
        closed_report=evaluation / "closed-set.json",
        open_reports=tuple((kind, root / relative) for kind, relative in _OPEN_REPORTS),
    )


def _validate_input_paths(paths: _ReleasePaths) -> ValidationEvidence:
    problems: tuple[str, ...] = ()
    try:
        _reject_symlink_components(paths.root, "release directory")
    except ValueError as error:
        problems = (*problems, str(error))
    if not paths.root.is_dir():
        problems = (*problems, "release directory is missing or not a regular directory")
    if paths.package.is_symlink() or not paths.package.is_dir():
        problems = (*problems, "FlukeEmbedder.mlpackage is missing or not a regular directory")
    for relative, path in paths.required_files():
        try:
            _reject_symlink_components(path, relative)
        except ValueError as error:
            problems = (*problems, str(error))
        if path.is_symlink() or not path.is_file():
            problems = (*problems, f"{relative} is missing or not a regular file")
    return _validation("input_paths", not problems, problems, "all fixed release inputs are safe")


def _inspect_package(paths: _ReleasePaths) -> _PackageEvidence:
    try:
        metadata = _load_json_mapping(paths.export_metadata, "export metadata")
        _validate_export_metadata(metadata)
        digest = package_tree_sha256(paths.package)
        if digest != metadata["package_sha256"]:
            raise ValueError("Core ML package digest does not match export metadata")
    except (CoreMLExportError, OSError, UnicodeError, ValueError, TypeError) as error:
        return _PackageEvidence(None, None, _failed("package", str(error)))
    return _PackageEvidence(
        digest,
        MappingProxyType(dict(metadata)),
        _passed("package", "exact export schema, identity, package tree, and audited tools verified"),
    )


def _validate_export_metadata(payload: Mapping[str, Any]) -> None:
    if set(payload) != _EXPORT_METADATA_KEYS:
        raise ValueError("export metadata fields do not match the exact schema")
    contract = mobile_model_contract()
    exact_values = (
        ("model_id", DINO_V2_MODEL_ID),
        ("model_revision", DINO_V2_REVISION),
        ("preprocessing_version", contract.preprocessing_version),
        ("minimum_deployment_target", MINIMUM_DEPLOYMENT_TARGET),
        ("compute_precision", COMPUTE_PRECISION),
        ("model_sha256", DINOV2_ARTIFACT_SHA256["model.safetensors"]),
        ("input_shape", list(contract.input_shape)),
        ("output_shape", list(contract.output_shape)),
    )
    for name, expected in exact_values:
        if payload[name] != expected:
            raise ValueError(f"export metadata field does not match the release contract: {name}")
    _require_sha256(payload["package_sha256"], "export package digest")
    tools = payload["tool_versions"]
    if not isinstance(tools, dict) or tools != dict(_TOOL_VERSIONS):
        raise ValueError("export tool versions do not match the audited release contract")


def _inspect_catalog(paths: _ReleasePaths, package: _PackageEvidence) -> _CatalogEvidence:
    try:
        manifest = _load_json_mapping(paths.catalog_manifest, "catalog manifest")
        metadata = _load_catalog_metadata(paths.catalog_metadata)
        _validate_catalog_manifest(manifest, metadata)
        manifest_digest = sha256_file(paths.catalog_manifest)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        failed = _failed("catalog", str(error))
        return _CatalogEvidence(None, (), None, failed, _failed("digests", "catalog unavailable"))
    validation = _passed("catalog", "exact catalog schema, counts, identities, and vector length verified")
    digest_validation = _validate_catalog_digests(paths, package, manifest)
    return _CatalogEvidence(
        MappingProxyType(dict(manifest)),
        tuple(MappingProxyType(dict(row)) for row in metadata),
        manifest_digest,
        validation,
        digest_validation,
    )


def _validate_catalog_manifest(
    manifest: Mapping[str, Any], metadata: tuple[dict[str, str], ...]
) -> None:
    if set(manifest) != _CATALOG_MANIFEST_KEYS:
        raise ValueError("catalog manifest fields do not match the exact schema")
    exact_values = (
        ("schemaVersion", 1),
        ("modelId", DINO_V2_MODEL_ID),
        ("modelRevision", DINO_V2_REVISION),
        ("preprocessingVersion", mobile_model_contract().preprocessing_version),
        ("embeddingDimension", _EMBEDDING_DIMENSION),
        ("dtype", "float16"),
        ("scoreSemantics", SCORE_SEMANTICS),
    )
    if any(manifest[name] != expected for name, expected in exact_values):
        raise ValueError("catalog manifest values do not match the mobile release contract")
    reference_count = _positive_integer(manifest["referenceCount"], "catalog referenceCount")
    catalog_count = _positive_integer(manifest["catalogCount"], "catalog catalogCount")
    if reference_count != len(metadata) or catalog_count > reference_count:
        raise ValueError("catalog counts do not match metadata")
    for name in ("modelSha256", "vectorsSha256", "metadataSha256", "rightsAttestationSha256"):
        _require_sha256(manifest[name], f"catalog {name}")
    for name in ("scoreThreshold", "marginThreshold"):
        _finite_rate(manifest[name], f"catalog {name}", lower=-1.0)


def _load_catalog_metadata(path: Path) -> tuple[dict[str, str], ...]:
    payload = _load_json(path, "catalog metadata")
    if not isinstance(payload, list) or not payload:
        raise ValueError("catalog metadata must be a non-empty JSON array")
    rows = tuple(_catalog_row(row) for row in payload)
    reference_ids = tuple(row["referencePhotoId"] for row in rows)
    if len(reference_ids) != len(set(reference_ids)):
        raise ValueError("catalog referencePhotoId values must be unique")
    return rows


def _catalog_row(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _CATALOG_METADATA_KEYS:
        raise ValueError("catalog metadata row fields do not match the exact schema")
    if any(not isinstance(value[name], str) or not value[name].strip() for name in value):
        raise ValueError("catalog metadata identity fields must be non-empty strings")
    return dict(value)


def _validate_catalog_digests(
    paths: _ReleasePaths,
    package: _PackageEvidence,
    manifest: Mapping[str, Any],
) -> ValidationEvidence:
    try:
        if package.digest is None or manifest["modelSha256"] != package.digest:
            raise ValueError("catalog model digest does not match the Core ML package")
        expected_length = manifest["referenceCount"] * manifest["embeddingDimension"] * 2
        if paths.catalog_vectors.stat().st_size != expected_length:
            raise ValueError("catalog vector length does not match the manifest")
        checks = (
            (paths.catalog_vectors, manifest["vectorsSha256"], "vector"),
            (paths.catalog_metadata, manifest["metadataSha256"], "metadata"),
            (paths.rights, manifest["rightsAttestationSha256"], "rights"),
        )
        for path, expected, name in checks:
            if sha256_file(path) != expected:
                raise ValueError(f"catalog {name} digest does not match the manifest")
    except (OSError, ValueError, TypeError) as error:
        return _failed("digests", str(error))
    return _passed("digests", "package, vector, metadata, and rights digests match exactly")


def _inspect_rights(paths: _ReleasePaths, catalog: _CatalogEvidence) -> ValidationEvidence:
    try:
        if catalog.manifest is None or not catalog.metadata:
            raise RightsError("catalog is unavailable for exact rights coverage")
        attestation, digest = _load_mobile_rights(paths.rights)
        if digest != catalog.manifest["rightsAttestationSha256"]:
            raise RightsError("rights attestation digest does not match the catalog")
        source_ids = tuple(sorted({row["sourceId"] for row in catalog.metadata}))
        attestation.validate_for(
            model_id=str(catalog.manifest["modelId"]),
            model_revision=str(catalog.manifest["modelRevision"]),
            reference_source_ids=source_ids,
        )
    except (OSError, UnicodeError, RightsError, ValueError, TypeError) as error:
        return _failed("rights", str(error))
    return _passed("rights", "written model and exact-source mobile redistribution rights verified")


def _inspect_embeddings(paths: _ReleasePaths) -> _EmbeddingEvidence:
    try:
        pytorch = _load_embedding_array(paths.pytorch_embeddings, "PyTorch parity embeddings")
        coreml = _load_embedding_array(paths.coreml_embeddings, "Core ML parity embeddings")
        if pytorch.shape != coreml.shape:
            raise ValueError("PyTorch and Core ML parity embedding shapes differ")
        if pytorch.shape[0] <= 0 or pytorch.shape[1] != _EMBEDDING_DIMENSION:
            raise ValueError("parity embeddings must have positive shape (N, 384)")
    except (OSError, ValueError, TypeError) as error:
        detail = str(error)
        return _EmbeddingEvidence(None, 0, _failed("embedding_shape", detail), _failed("embedding_norm", "embedding shape unavailable"))
    shape = _passed("embedding_shape", f"{pytorch.shape[0]} paired float32 embeddings have shape (N, 384)")
    try:
        pytorch_norms = _unit_norms(pytorch, "PyTorch parity embeddings")
        coreml_norms = _unit_norms(coreml, "Core ML parity embeddings")
        cosines = np.sum(pytorch * coreml, axis=1) / (pytorch_norms * coreml_norms)
        parity = float(np.min(cosines))
        if not math.isfinite(parity):
            raise ValueError("parity cosine must be finite")
    except ValueError as error:
        return _EmbeddingEvidence(None, int(pytorch.shape[0]), shape, _failed("embedding_norm", str(error)))
    norm = _passed("embedding_norm", "all PyTorch and Core ML parity embeddings are finite and L2 normalized")
    return _EmbeddingEvidence(parity, int(pytorch.shape[0]), shape, norm)


def _load_embedding_array(path: Path, name: str) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} file is missing")
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32) or value.ndim != 2:
        raise ValueError(f"{name} must be an exact two-dimensional float32 NumPy array")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _unit_norms(values: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.vector_norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=0.0):
        raise ValueError(f"{name} must be L2 normalized")
    return norms


def _inspect_reports(
    paths: _ReleasePaths,
    package: _PackageEvidence,
    catalog: _CatalogEvidence,
) -> _ReportEvidence:
    problems: tuple[str, ...] = ()
    if package.digest is None or catalog.manifest_digest is None:
        problems = (*problems, "package or catalog digest unavailable for report provenance")
    try:
        closed = _load_evaluation_report(
            paths.closed_report,
            "closedSetRetrieval",
            _CLOSED_REPORT_KEYS,
            package.digest,
            catalog.manifest_digest,
        )
        top_1, top_3 = _closed_metrics(closed)
        closed_count = _positive_integer(closed["sampleCount"], "closed-set sampleCount")
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        problems = (*problems, f"evaluation/closed-set.json: {error}")
        top_1, top_3, closed_count = None, None, 0
    open_results, open_problems = _load_open_reports(paths, package.digest, catalog.manifest_digest)
    problems = (*problems, *open_problems)
    false_accept = max((rate for _, rate in open_results), default=None)
    open_count = sum(count for count, _ in open_results)
    validation = _validation(
        "required_reports",
        not problems,
        problems,
        "all six exact-schema, digest-bound evaluation reports are present",
    )
    return _ReportEvidence(top_1, top_3, closed_count, false_accept, open_count, validation)


def _load_open_reports(
    paths: _ReleasePaths,
    package_digest: str | None,
    catalog_digest: str | None,
) -> tuple[tuple[tuple[int, float], ...], tuple[str, ...]]:
    results: tuple[tuple[int, float], ...] = ()
    problems: tuple[str, ...] = ()
    for (kind, relative), (_, path) in zip(_OPEN_REPORTS, paths.open_reports, strict=True):
        try:
            payload = _load_evaluation_report(
                path,
                kind,
                _OPEN_REPORT_KEYS,
                package_digest,
                catalog_digest,
            )
            count = _positive_integer(payload["sampleCount"], f"{kind} sampleCount")
            rate = _finite_rate(payload["falseAcceptRate"], f"{kind} falseAcceptRate")
            results = (*results, (count, rate))
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            problems = (*problems, f"{relative}: {error}")
    return results, problems


def _load_evaluation_report(
    path: Path,
    evaluation_type: str,
    keys: set[str],
    package_digest: str | None,
    catalog_digest: str | None,
) -> Mapping[str, Any]:
    payload = _load_json_mapping(path, f"{evaluation_type} report")
    if set(payload) != keys:
        raise ValueError("report fields do not match the exact schema")
    if payload["schemaVersion"] != 1 or payload["evaluationType"] != evaluation_type:
        raise ValueError("report identity does not match its fixed release path")
    if package_digest is None or payload["modelPackageSha256"] != package_digest:
        raise ValueError("report package digest does not match the release")
    if catalog_digest is None or payload["catalogManifestSha256"] != catalog_digest:
        raise ValueError("report catalog digest does not match the release")
    return payload


def _closed_metrics(payload: Mapping[str, Any]) -> tuple[float, float]:
    top_1 = _finite_rate(payload["top1"], "closed-set top1")
    top_3 = _finite_rate(payload["top3"], "closed-set top3")
    if top_3 < top_1:
        raise ValueError("closed-set top3 must be greater than or equal to top1")
    return top_1, top_3


def _boundary_gates(validations: tuple[ValidationEvidence, ...]) -> tuple[GateResult, ...]:
    unknown = tuple(value.name for value in validations if value.name not in _BOUNDARY_GATE_NAMES)
    gates = tuple(_boundary_gate(name, validations) for name in _BOUNDARY_GATE_NAMES)
    if not unknown:
        return gates
    return (
        *gates,
        GateResult(
            name="evidence_schema",
            passed=False,
            observed=", ".join(sorted(unknown)),
            requirement="only approved boundary gate names are accepted",
            detail="release evidence contains unknown validation gates",
        ),
    )


def _boundary_gate(name: str, validations: tuple[ValidationEvidence, ...]) -> GateResult:
    matches = tuple(value for value in validations if value.name == name)
    if len(matches) != 1:
        return GateResult(
            name=name,
            passed=False,
            observed=len(matches),
            requirement="exactly one validation result is required",
            detail="validation result is missing or duplicated",
        )
    evidence = matches[0]
    return GateResult(name, evidence.passed, evidence.passed, "validation must pass", evidence.detail)


def _count_gate(name: str, value: object) -> GateResult:
    passed = isinstance(value, int) and not isinstance(value, bool) and value > 0
    observed = value if isinstance(value, (bool, int, float, str)) else None
    return GateResult(
        name=name,
        passed=passed,
        observed=observed,
        requirement="positive integer sample count",
        detail="sample count is meaningful" if passed else "sample count is missing or invalid",
    )


def _metric_gate(name: str, value: object) -> GateResult:
    threshold = RELEASE_THRESHOLDS[name]
    finite = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    if name == "false_accept":
        passed = finite and float(value) <= threshold
        comparison = "<="
    else:
        passed = finite and float(value) >= threshold
        comparison = ">="
    observed: float | str | None = float(value) if finite else ("non-finite" if isinstance(value, (int, float)) else None)
    return GateResult(
        name=name,
        passed=passed,
        observed=observed,
        requirement=f"finite value {comparison} {threshold}",
        detail="threshold met" if passed else "metric missing, non-finite, or outside threshold",
    )


def _load_json_mapping(path: Path, name: str) -> Mapping[str, Any]:
    payload = _load_json(path, name)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _load_json(path: Path, name: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} file is missing or not a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_rate(value: object, name: str, *, lower: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number within [{lower}, 1]")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number within [{lower}, 1]")
    return result


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA256")
    if len(value) != _SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def _reject_symlink_components(path: Path, name: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise ValueError(f"{name} path contains a symbolic link component")


def _validation(
    name: str,
    passed: bool,
    problems: tuple[str, ...],
    success: str,
) -> ValidationEvidence:
    return ValidationEvidence(name, passed, success if passed else "; ".join(problems))


def _passed(name: str, detail: str) -> ValidationEvidence:
    return ValidationEvidence(name, True, detail)


def _failed(name: str, detail: str) -> ValidationEvidence:
    return ValidationEvidence(name, False, detail)
