"""Deterministic, fail-closed verification for an on-device model release."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from fluke_model.embedders import DINO_V2_MODEL_ID, DINO_V2_REVISION
from fluke_model.mobile_catalog import (
    RightsError,
    ValidatedMobileCatalog,
    _load_mobile_rights,
    sha256_file,
    validate_published_mobile_catalog,
)
from fluke_model.mobile_export import mobile_model_contract
from fluke_model.mobile_release_contracts import (
    EMBEDDING_DIMENSION,
    REPORT_FILENAME,
    EmbeddingEvidence,
    EvaluationEvidence,
    PackageEvidence,
    ReleasePaths,
    ValidationEvidence,
    failed,
    inspect_embeddings,
    inspect_evaluations,
    inspect_package,
    normalize_external_detail,
    passed,
    reject_symlink_components,
    release_paths,
    require_sha256,
    safe_file_digest,
    validate_input_layout,
    validate_report_destination,
)

REPORT_SCHEMA_VERSION = 1
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
    "runtime_reexecution",
)
_EXPECTED_INPUT_ERRORS = (
    EOFError,
    OSError,
    OverflowError,
    TypeError,
    UnicodeError,
    ValueError,
)
_MAX_OBSERVED_INTEGER_BITS = 4096


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
    model_package_sha256: str | None = None
    catalog_manifest_sha256: str | None = None

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
    """Deterministic decision bound to the exact package and catalog when available."""

    ready: bool
    gates: tuple[GateResult, ...]
    model_package_sha256: str | None = None
    catalog_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "gates", tuple(self.gates))


@dataclass(frozen=True)
class _CatalogEvidence:
    validated: ValidatedMobileCatalog | None
    digest: str | None
    validation: ValidationEvidence
    digest_validation: ValidationEvidence


def verify_mobile_release(evidence: MobileReleaseEvidence) -> MobileReleaseReport:
    """Apply every approved boundary, count, and numeric release gate."""
    if not isinstance(evidence, MobileReleaseEvidence):
        raise TypeError("evidence must be MobileReleaseEvidence")
    gates = (
        _digest_gate("model_package_digest", evidence.model_package_sha256),
        _digest_gate("catalog_manifest_digest", evidence.catalog_manifest_sha256),
        *_boundary_gates(evidence.validations),
        _count_gate("parity_samples", evidence.parity_sample_count),
        _metric_gate("parity_cosine", evidence.parity_cosine),
        _count_gate("closed_set_samples", evidence.closed_set_sample_count),
        _metric_gate("top_1", evidence.top_1),
        _metric_gate("top_3", evidence.top_3),
        _count_gate("open_set_samples", evidence.open_set_sample_count),
        _metric_gate("false_accept", evidence.false_accept),
    )
    return MobileReleaseReport(
        ready=all(gate.passed for gate in gates),
        gates=gates,
        model_package_sha256=evidence.model_package_sha256,
        catalog_manifest_sha256=evidence.catalog_manifest_sha256,
    )


def verify_mobile_release_directory(
    release_dir: Path,
) -> MobileReleaseReport:
    """Re-execute the fixed production release and return a fail-closed report."""
    from fluke_model.mobile_release_builder import reexecute_mobile_release

    return _verify_mobile_release_directory_entry(
        release_dir,
        package_loader=None,
        runtime_validator=reexecute_mobile_release,
    )


def _verify_mobile_release_directory_for_testing(
    release_dir: Path,
    *,
    package_loader: Callable[[Path], Any],
    runtime_validator: Callable[[ReleasePaths, ValidatedMobileCatalog], tuple[bool, str]],
) -> MobileReleaseReport:
    """Exercise verifier contracts with explicit test-only execution adapters."""
    return _verify_mobile_release_directory_entry(
        release_dir,
        package_loader=package_loader,
        runtime_validator=runtime_validator,
    )


def _verify_mobile_release_directory_entry(
    release_dir: Path,
    *,
    package_loader: Callable[[Path], Any] | None,
    runtime_validator: Callable[[ReleasePaths, ValidatedMobileCatalog], tuple[bool, str]],
) -> MobileReleaseReport:
    raw_root = Path(release_dir)
    try:
        reject_symlink_components(raw_root, "release directory")
    except ValueError as error:
        return failed_mobile_release_report(str(error))
    root = raw_root.resolve(strict=False)
    try:
        report = _verify_mobile_release_directory(
            root,
            package_loader=package_loader,
            runtime_validator=runtime_validator,
        )
    except _EXPECTED_INPUT_ERRORS as error:
        detail = normalize_external_detail(f"release input validation failed: {error}", root)
        report = failed_mobile_release_report(detail)
    return _normalize_report_details(report, root)


def failed_mobile_release_report(
    detail: str, release_dir: Path | None = None
) -> MobileReleaseReport:
    """Build a deterministic all-failed report for a bounded external-input error."""
    stable_detail = (
        normalize_external_detail(detail, Path(release_dir)) if release_dir is not None else detail
    )
    validations = tuple(failed(name, stable_detail) for name in _BOUNDARY_GATE_NAMES)
    return verify_mobile_release(
        MobileReleaseEvidence(
            parity_cosine=None,
            parity_sample_count=0,
            top_1=None,
            top_3=None,
            closed_set_sample_count=0,
            false_accept=None,
            open_set_sample_count=0,
            validations=validations,
        )
    )


def report_payload(report: MobileReleaseReport) -> dict[str, object]:
    """Return the exact deterministic mobile release report schema."""
    return {
        "schemaVersion": REPORT_SCHEMA_VERSION,
        "modelPackageSha256": report.model_package_sha256,
        "catalogManifestSha256": report.catalog_manifest_sha256,
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
    from fluke_model.mobile_release_contracts import reject_symlink_components

    reject_symlink_components(destination, "release report")
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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_mobile_release_directory(
    release_dir: Path,
    *,
    package_loader: Callable[[Path], Any] | None,
    runtime_validator: Callable[[ReleasePaths, ValidatedMobileCatalog], tuple[bool, str]],
) -> MobileReleaseReport:
    paths = release_paths(release_dir)
    path_validation = validate_input_layout(paths)
    package = inspect_package(paths, package_loader=package_loader)
    catalog = _inspect_catalog(paths, package)
    rights = _inspect_rights(paths, catalog)
    embeddings = inspect_embeddings(paths, package.digest, catalog.digest)
    manifest = catalog.validated.manifest if catalog.validated is not None else None
    evaluations = inspect_evaluations(
        paths,
        package.digest,
        catalog.digest,
        score_threshold=manifest.score_threshold if manifest is not None else None,
        margin_threshold=manifest.margin_threshold if manifest is not None else None,
        catalog_rows=catalog.validated.rows if catalog.validated is not None else None,
    )
    runtime = _inspect_runtime_reexecution(paths, catalog, runtime_validator)
    evidence = _directory_evidence(
        path_validation,
        package,
        catalog,
        rights,
        embeddings,
        evaluations,
        runtime,
    )
    return verify_mobile_release(evidence)


def _directory_evidence(
    path_validation: ValidationEvidence,
    package: PackageEvidence,
    catalog: _CatalogEvidence,
    rights: ValidationEvidence,
    embeddings: EmbeddingEvidence,
    evaluations: EvaluationEvidence,
    runtime: ValidationEvidence,
) -> MobileReleaseEvidence:
    return MobileReleaseEvidence(
        parity_cosine=embeddings.parity_cosine,
        parity_sample_count=embeddings.sample_count,
        top_1=evaluations.top_1,
        top_3=evaluations.top_3,
        closed_set_sample_count=evaluations.closed_count,
        false_accept=evaluations.false_accept,
        open_set_sample_count=evaluations.open_count,
        validations=(
            path_validation,
            package.validation,
            catalog.validation,
            catalog.digest_validation,
            rights,
            embeddings.shape_validation,
            embeddings.norm_validation,
            evaluations.validation,
            runtime,
        ),
        model_package_sha256=package.digest,
        catalog_manifest_sha256=catalog.digest,
    )


def _inspect_runtime_reexecution(
    paths: ReleasePaths,
    catalog: _CatalogEvidence,
    validator: Callable[[ReleasePaths, ValidatedMobileCatalog], tuple[bool, str]],
) -> ValidationEvidence:
    if catalog.validated is None:
        return failed("runtime_reexecution", "catalog unavailable for runtime re-execution")
    try:
        valid, detail = validator(paths, catalog.validated)
    except _EXPECTED_INPUT_ERRORS as error:
        return failed("runtime_reexecution", str(error))
    return passed("runtime_reexecution", detail) if valid else failed(
        "runtime_reexecution", detail
    )


def _inspect_catalog(paths: ReleasePaths, package: PackageEvidence) -> _CatalogEvidence:
    manifest_digest = safe_file_digest(paths.catalog_manifest)
    try:
        validated = validate_published_mobile_catalog(paths.catalog_dir)
        _validate_mobile_catalog_contract(validated, package.digest)
    except _EXPECTED_INPUT_ERRORS as error:
        return _CatalogEvidence(
            None,
            manifest_digest,
            failed("catalog", str(error)),
            failed("digests", "catalog unavailable for complete digest validation"),
        )
    digest_validation = _validate_release_digests(paths, validated, package.digest)
    return _CatalogEvidence(
        validated,
        validated.manifest_sha256,
        passed("catalog", "complete Task 3 published catalog contract verified"),
        digest_validation,
    )


def _validate_mobile_catalog_contract(
    catalog: ValidatedMobileCatalog, package_digest: str | None
) -> None:
    manifest = catalog.manifest
    contract = mobile_model_contract()
    exact_values = (
        (manifest.model_id, DINO_V2_MODEL_ID, "modelId"),
        (manifest.model_revision, DINO_V2_REVISION, "modelRevision"),
        (manifest.preprocessing_version, contract.preprocessing_version, "preprocessingVersion"),
        (manifest.embedding_dimension, EMBEDDING_DIMENSION, "embeddingDimension"),
    )
    if any(actual != expected for actual, expected, _ in exact_values):
        mismatch = next(name for actual, expected, name in exact_values if actual != expected)
        raise ValueError(f"catalog {mismatch} does not match the release contract")
    if package_digest is None or manifest.model_sha256 != package_digest:
        raise ValueError("catalog modelSha256 does not match the Core ML package")


def _validate_release_digests(
    paths: ReleasePaths,
    catalog: ValidatedMobileCatalog,
    package_digest: str | None,
) -> ValidationEvidence:
    try:
        if package_digest is None or catalog.manifest.model_sha256 != package_digest:
            raise ValueError("catalog model digest does not match the Core ML package")
        if sha256_file(paths.rights) != catalog.manifest.rights_attestation_sha256:
            raise ValueError("catalog rights digest does not match the manifest")
    except _EXPECTED_INPUT_ERRORS as error:
        return failed("digests", str(error))
    return passed("digests", "package, vectors, metadata, and rights digests match exactly")


def _inspect_rights(paths: ReleasePaths, catalog: _CatalogEvidence) -> ValidationEvidence:
    try:
        if catalog.validated is None:
            raise RightsError("catalog is unavailable for exact rights coverage")
        attestation, digest = _load_mobile_rights(paths.rights)
        manifest = catalog.validated.manifest
        if digest != manifest.rights_attestation_sha256:
            raise RightsError("rights attestation digest does not match the catalog")
        source_ids = tuple(sorted({row.source_id for row in catalog.validated.rows}))
        attestation.validate_for(
            model_id=manifest.model_id,
            model_revision=manifest.model_revision,
            reference_source_ids=source_ids,
            required_purpose="production",
        )
    except (RightsError, *_EXPECTED_INPUT_ERRORS) as error:
        return failed("rights", str(error))
    return passed("rights", "written model and exact-source mobile redistribution rights verified")


def _boundary_gates(validations: tuple[ValidationEvidence, ...]) -> tuple[GateResult, ...]:
    unknown = tuple(value.name for value in validations if value.name not in _BOUNDARY_GATE_NAMES)
    gates = tuple(_boundary_gate(name, validations) for name in _BOUNDARY_GATE_NAMES)
    if not unknown:
        return gates
    return (
        *gates,
        GateResult(
            "evidence_schema",
            False,
            ", ".join(sorted(unknown)),
            "only approved boundary gate names are accepted",
            "release evidence contains unknown validation gates",
        ),
    )


def _boundary_gate(name: str, validations: tuple[ValidationEvidence, ...]) -> GateResult:
    matches = tuple(value for value in validations if value.name == name)
    if len(matches) != 1:
        return GateResult(
            name,
            False,
            len(matches),
            "exactly one validation result is required",
            "validation result is missing or duplicated",
        )
    evidence = matches[0]
    return GateResult(
        name, evidence.passed, evidence.passed, "validation must pass", evidence.detail
    )


def _count_gate(name: str, value: object) -> GateResult:
    is_integer = isinstance(value, int) and not isinstance(value, bool)
    passed_gate = is_integer and value > 0
    observed: bool | int | str | None
    if is_integer and value.bit_length() > _MAX_OBSERVED_INTEGER_BITS:
        observed = "out-of-range integer"
    elif isinstance(value, (bool, int, str)):
        observed = value
    else:
        observed = None
    return GateResult(
        name,
        passed_gate,
        observed,
        "positive integer sample count",
        "sample count is meaningful" if passed_gate else "sample count is missing or invalid",
    )


def _digest_gate(name: str, value: object) -> GateResult:
    valid = _is_lowercase_sha256(value)
    observed = value if isinstance(value, str) else None
    return GateResult(
        name,
        valid,
        observed,
        "valid lowercase SHA256 release identity",
        "digest is bound" if valid else "digest is missing or invalid",
    )


def _metric_gate(name: str, value: object) -> GateResult:
    threshold = RELEASE_THRESHOLDS[name]
    numeric = _finite_float(value)
    if name == "false_accept":
        passed_gate = numeric is not None and numeric <= threshold
        comparison = "<="
    else:
        passed_gate = numeric is not None and numeric >= threshold
        comparison = ">="
    observed: float | str | None = numeric
    if numeric is None and isinstance(value, (int, float)) and not isinstance(value, bool):
        observed = "non-finite or out-of-range"
    return GateResult(
        name,
        passed_gate,
        observed,
        f"finite value {comparison} {threshold}",
        "threshold met" if passed_gate else "metric missing, non-finite, or outside threshold",
    )


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _is_lowercase_sha256(value: object) -> bool:
    try:
        require_sha256(value, "release digest")
    except ValueError:
        return False
    return True


def _normalize_report_details(
    report: MobileReleaseReport, release_dir: Path
) -> MobileReleaseReport:
    gates = tuple(
        replace(gate, detail=normalize_external_detail(gate.detail, release_dir))
        for gate in report.gates
    )
    return replace(report, gates=gates)


__all__ = [
    "GateResult",
    "MobileReleaseEvidence",
    "MobileReleaseReport",
    "RELEASE_THRESHOLDS",
    "REPORT_FILENAME",
    "ValidationEvidence",
    "failed_mobile_release_report",
    "report_payload",
    "validate_report_destination",
    "verify_mobile_release",
    "verify_mobile_release_directory",
    "write_mobile_release_report",
]
