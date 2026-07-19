"""Exact external-input contracts for mobile release verification."""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from fluke_model.coreml_artifact import (
    COMPUTE_PRECISION,
    MINIMUM_DEPLOYMENT_TARGET,
    CoreMLExportError,
    package_tree_sha256,
    validate_coreml_package_interface,
)
from fluke_model.embedders import DINO_V2_MODEL_ID, DINO_V2_REVISION
from fluke_model.mobile_catalog import sha256_file
from fluke_model.mobile_export import mobile_model_contract
from fluke_model.model_artifact import DINOV2_ARTIFACT_SHA256
from fluke_model.numpy_artifact import load_bounded_parity_array

REPORT_FILENAME = "mobile-release-report.json"
EMBEDDING_DIMENSION = 384
OPEN_REPORTS = (
    ("openSet", "open-set.json"),
    ("nonOrca", "non-orca.json"),
    ("poorQuality", "poor-quality.json"),
    ("occlusion", "occlusion.json"),
    ("distributionShift", "distribution-shift.json"),
)
_ROOT_ENTRIES = frozenset(
    {
        "FlukeEmbedder.mlpackage",
        "export-metadata.json",
        "rights-attestation.json",
        "catalog",
        "evaluation",
    }
)
_CATALOG_ENTRIES = frozenset({"manifest.json", "metadata.json", "references.f16"})
_EVALUATION_ENTRIES = frozenset(
    {
        "parity-pytorch.npy",
        "parity-coreml.npy",
        "parity.json",
        "closed-set.json",
        *(filename for _, filename in OPEN_REPORTS),
    }
)
_EXPORT_METADATA_KEYS = {
    "compute_precision",
    "input_shape",
    "minimum_deployment_target",
    "model_id",
    "model_revision",
    "model_sha256",
    "source_artifact_sha256",
    "output_shape",
    "package_sha256",
    "preprocessing_version",
    "tool_versions",
}
_TOOL_VERSIONS = MappingProxyType(
    {
        "coremltools": "9.0",
        "macos": "26.5.1",
        "numpy": "2.2.6",
        "pillow": "12.3.0",
        "python": "3.11.15",
        "torch": "2.13.0",
        "transformers": "5.14.0",
        "xcode": "26.0.1 (17A400)",
    }
)
_CLOSED_REPORT_KEYS = {
    "schemaVersion",
    "evaluationType",
    "evidencePurpose",
    "provenanceUrl",
    "fixtureSetSha256",
    "modelPackageSha256",
    "catalogManifestSha256",
    "sampleCount",
    "top1",
    "top3",
}
_OPEN_REPORT_KEYS = {
    "schemaVersion",
    "evaluationType",
    "evidencePurpose",
    "provenanceUrl",
    "fixtureSetSha256",
    "modelPackageSha256",
    "catalogManifestSha256",
    "sampleCount",
    "falseAcceptRate",
}
_PARITY_REPORT_KEYS = {
    "schemaVersion",
    "evaluationType",
    "evidencePurpose",
    "provenanceUrl",
    "modelPackageSha256",
    "catalogManifestSha256",
    "sourceModelSha256",
    "preprocessingVersion",
    "fixtureSetSha256",
    "sampleCount",
    "pytorchEmbeddingsSha256",
    "coremlEmbeddingsSha256",
}
_SHA256_LENGTH = 64


@dataclass(frozen=True)
class ValidationEvidence:
    """One immutable non-metric validation result."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ReleasePaths:
    """Every fixed path in one mobile release directory."""

    root: Path
    package: Path
    export_metadata: Path
    catalog_dir: Path
    catalog_manifest: Path
    catalog_vectors: Path
    catalog_metadata: Path
    rights: Path
    evaluation_dir: Path
    pytorch_embeddings: Path
    coreml_embeddings: Path
    parity_report: Path
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
            ("evaluation/parity.json", self.parity_report),
            ("evaluation/closed-set.json", self.closed_report),
            *(
                (f"evaluation/{path.name}", path)
                for _, path in self.open_reports
            ),
        )


@dataclass(frozen=True)
class PackageEvidence:
    digest: str | None
    validation: ValidationEvidence


@dataclass(frozen=True)
class EmbeddingEvidence:
    parity_cosine: float | None
    sample_count: int
    shape_validation: ValidationEvidence
    norm_validation: ValidationEvidence


@dataclass(frozen=True)
class EvaluationEvidence:
    top_1: float | None
    top_3: float | None
    closed_count: int
    false_accept: float | None
    open_count: int
    validation: ValidationEvidence


def release_paths(root: Path) -> ReleasePaths:
    """Construct fixed paths without resolving or reading caller input."""
    release_root = Path(root)
    catalog = release_root / "catalog"
    evaluation = release_root / "evaluation"
    return ReleasePaths(
        root=release_root,
        package=release_root / "FlukeEmbedder.mlpackage",
        export_metadata=release_root / "export-metadata.json",
        catalog_dir=catalog,
        catalog_manifest=catalog / "manifest.json",
        catalog_vectors=catalog / "references.f16",
        catalog_metadata=catalog / "metadata.json",
        rights=release_root / "rights-attestation.json",
        evaluation_dir=evaluation,
        pytorch_embeddings=evaluation / "parity-pytorch.npy",
        coreml_embeddings=evaluation / "parity-coreml.npy",
        parity_report=evaluation / "parity.json",
        closed_report=evaluation / "closed-set.json",
        open_reports=tuple((kind, evaluation / filename) for kind, filename in OPEN_REPORTS),
    )


def validate_input_layout(paths: ReleasePaths) -> ValidationEvidence:
    """Require exact root/catalog/evaluation entries and safe regular input paths."""
    problems: tuple[str, ...] = ()
    try:
        reject_symlink_components(paths.root, "release directory")
    except ValueError as error:
        problems = (*problems, str(error))
    if not paths.root.is_dir():
        problems = (*problems, "release directory is missing or not a regular directory")
    else:
        problems = (*problems, *_layout_problems(paths))
    if paths.package.is_symlink() or not paths.package.is_dir():
        problems = (*problems, "FlukeEmbedder.mlpackage is missing or not a regular directory")
    for relative, path in paths.required_files():
        problems = (*problems, *_file_problems(relative, path))
    return validation(
        "input_paths",
        not problems,
        problems,
        "all fixed release inputs and exact directory layouts are safe",
    )


def inspect_package(
    paths: ReleasePaths,
    *,
    package_loader: Callable[[Path], Any] | None = None,
) -> PackageEvidence:
    """Hash the actual package and independently validate exact export metadata."""
    digest = safe_package_digest(paths.package)
    try:
        metadata = load_json_mapping(paths.export_metadata, "export metadata")
        _validate_export_metadata(metadata)
        if digest is None or digest != metadata["package_sha256"]:
            raise ValueError("Core ML package digest does not match export metadata")
        validate_coreml_package_interface(paths.package, package_loader=package_loader)
    except (CoreMLExportError, OSError, OverflowError, UnicodeError, ValueError, TypeError) as error:
        return PackageEvidence(digest, failed("package", str(error)))
    return PackageEvidence(
        digest,
        passed(
            "package",
            "exact export schema, identity, package tree, interface, and audited tools verified",
        ),
    )


def inspect_embeddings(
    paths: ReleasePaths,
    package_digest: str | None = None,
    catalog_digest: str | None = None,
) -> EmbeddingEvidence:
    """Load bounded NumPy inputs and calculate minimum paired cosine parity."""
    try:
        pytorch = _load_embedding_array(paths.pytorch_embeddings, "PyTorch parity embeddings")
        coreml = _load_embedding_array(paths.coreml_embeddings, "Core ML parity embeddings")
        if pytorch.shape != coreml.shape:
            raise ValueError("PyTorch and Core ML parity embedding shapes differ")
        if pytorch.shape[0] <= 0 or pytorch.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError("parity embeddings must have positive shape (N, 384)")
        _validate_parity_report(
            paths,
            package_digest=package_digest,
            catalog_digest=catalog_digest,
            sample_count=int(pytorch.shape[0]),
        )
    except (EOFError, OSError, OverflowError, ValueError, TypeError) as error:
        detail = str(error)
        return EmbeddingEvidence(
            None,
            0,
            failed("embedding_shape", detail),
            failed("embedding_norm", "embedding shape unavailable"),
        )
    shape = passed(
        "embedding_shape",
        f"{pytorch.shape[0]} paired float32 embeddings have shape (N, 384)",
    )
    try:
        pytorch_norms = _unit_norms(pytorch, "PyTorch parity embeddings")
        coreml_norms = _unit_norms(coreml, "Core ML parity embeddings")
        cosines = np.sum(pytorch * coreml, axis=1) / (pytorch_norms * coreml_norms)
        parity = float(np.min(cosines))
        if not math.isfinite(parity):
            raise ValueError("parity cosine must be finite")
    except (OverflowError, ValueError, TypeError) as error:
        return EmbeddingEvidence(
            None,
            int(pytorch.shape[0]),
            shape,
            failed("embedding_norm", str(error)),
        )
    return EmbeddingEvidence(
        parity,
        int(pytorch.shape[0]),
        shape,
        passed("embedding_norm", "all parity embeddings are finite and L2 normalized"),
    )


def inspect_evaluations(
    paths: ReleasePaths,
    package_digest: str | None,
    catalog_digest: str | None,
) -> EvaluationEvidence:
    """Load all digest-bound closed/open-set reports without allowing partial success."""
    problems: tuple[str, ...] = ()
    if package_digest is None or catalog_digest is None:
        problems = (*problems, "package or catalog digest unavailable for report provenance")
    try:
        closed = _load_evaluation_report(
            paths.closed_report,
            "closedSetRetrieval",
            _CLOSED_REPORT_KEYS,
            package_digest,
            catalog_digest,
        )
        top_1, top_3 = _closed_metrics(closed)
        closed_count = positive_integer(closed["sampleCount"], "closed-set sampleCount")
    except (OSError, OverflowError, UnicodeError, ValueError, TypeError) as error:
        problems = (*problems, f"evaluation/closed-set.json: {error}")
        top_1, top_3, closed_count = None, None, 0
    open_results, open_problems = _load_open_reports(
        paths, package_digest, catalog_digest
    )
    problems = (*problems, *open_problems)
    return EvaluationEvidence(
        top_1=top_1,
        top_3=top_3,
        closed_count=closed_count,
        false_accept=max((rate for _, rate in open_results), default=None),
        open_count=sum(count for count, _ in open_results),
        validation=validation(
            "required_reports",
            not problems,
            problems,
            "all six exact-schema, digest-bound evaluation reports are present",
        ),
    )


def validate_report_destination(release_dir: Path, report_path: Path) -> None:
    """Reject a report destination that could replace any release input."""
    paths = release_paths(Path(release_dir))
    destination = Path(report_path)
    reject_symlink_components(destination, "release report")
    resolved = destination.resolve(strict=False)
    release_root = paths.root.resolve(strict=False)
    canonical = (release_root / REPORT_FILENAME).resolve(strict=False)
    if resolved.is_relative_to(release_root) and resolved != canonical:
        raise ValueError(
            "report path overlaps the release input contract; path inside release directory "
            f"must be exactly {REPORT_FILENAME}"
        )
    protected = (
        paths.package,
        paths.catalog_dir,
        paths.evaluation_dir,
        *(path for _, path in paths.required_files()),
    )
    for source in protected:
        source_resolved = source.resolve(strict=False)
        overlaps = (
            resolved == source_resolved
            or resolved.is_relative_to(source_resolved)
            or source_resolved.is_relative_to(resolved)
        )
        if overlaps:
            raise ValueError("release report path overlaps a release input")


def normalize_external_detail(detail: object, release_dir: Path) -> str:
    """Replace host-specific release/temp roots before serializing gate evidence."""
    text = str(detail)
    root = Path(release_dir).resolve(strict=False)
    temporary = Path(tempfile.gettempdir()).resolve(strict=False)
    normalized = _replace_path_tokens(text, {str(root)}, "<release-dir>")
    return _replace_path_tokens(normalized, {str(temporary)}, "<temp-dir>")


def _replace_path_tokens(text: str, paths: set[str], token: str) -> str:
    normalized = text
    for value in sorted((path for path in paths if path), key=len, reverse=True):
        normalized = normalized.replace(value, token)
    return normalized


def safe_package_digest(path: Path) -> str | None:
    try:
        return package_tree_sha256(path)
    except (CoreMLExportError, OSError, OverflowError, ValueError, TypeError):
        return None


def safe_file_digest(path: Path) -> str | None:
    try:
        return sha256_file(path)
    except (OSError, OverflowError, ValueError, TypeError):
        return None


def reject_symlink_components(path: Path, name: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise ValueError(f"{name} path contains a symbolic link component")


def passed(name: str, detail: str) -> ValidationEvidence:
    return ValidationEvidence(name, True, detail)


def failed(name: str, detail: str) -> ValidationEvidence:
    return ValidationEvidence(name, False, detail)


def validation(
    name: str,
    is_valid: bool,
    problems: tuple[str, ...],
    success: str,
) -> ValidationEvidence:
    return ValidationEvidence(name, is_valid, success if is_valid else "; ".join(problems))


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite_rate(value: object, name: str, *, lower: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number within [{lower}, 1]")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number within [{lower}, 1]") from error
    if not math.isfinite(result) or not lower <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number within [{lower}, 1]")
    return result


def require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA256")
    valid = len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )
    if not valid:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def load_json_mapping(path: Path, name: str) -> Mapping[str, Any]:
    payload = _load_json(path, name)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _layout_problems(paths: ReleasePaths) -> tuple[str, ...]:
    problems: tuple[str, ...] = ()
    root_entries = frozenset(path.name for path in paths.root.iterdir())
    allowed_root = _ROOT_ENTRIES | {REPORT_FILENAME}
    if not _ROOT_ENTRIES.issubset(root_entries) or not root_entries.issubset(allowed_root):
        problems = (*problems, "release root must contain exactly required entries and optional report")
    report_path = paths.root / REPORT_FILENAME
    if (report_path.exists() or report_path.is_symlink()) and (
        report_path.is_symlink() or not report_path.is_file()
    ):
        problems = (*problems, "optional report must be a regular non-symlink file")
    problems = (*problems, *_exact_directory_problems(paths.catalog_dir, _CATALOG_ENTRIES, "catalog"))
    problems = (
        *problems,
        *_exact_directory_problems(paths.evaluation_dir, _EVALUATION_ENTRIES, "evaluation"),
    )
    return problems


def _exact_directory_problems(
    directory: Path, expected: frozenset[str], name: str
) -> tuple[str, ...]:
    if directory.is_symlink() or not directory.is_dir():
        return (f"{name} directory is missing or not a regular directory",)
    actual = frozenset(path.name for path in directory.iterdir())
    if actual != expected:
        return (f"{name} directory must contain exactly the documented files",)
    return ()


def _file_problems(relative: str, path: Path) -> tuple[str, ...]:
    problems: tuple[str, ...] = ()
    try:
        reject_symlink_components(path, relative)
    except ValueError as error:
        problems = (*problems, str(error))
    if path.is_symlink() or not path.is_file():
        problems = (*problems, f"{relative} is missing or not a regular file")
    return problems


def _validate_export_metadata(payload: Mapping[str, Any]) -> None:
    if set(payload) != _EXPORT_METADATA_KEYS:
        raise ValueError("export metadata fields do not match the exact schema")
    contract = mobile_model_contract()
    _exact_shape(payload["input_shape"], contract.input_shape, "input_shape")
    _exact_shape(payload["output_shape"], contract.output_shape, "output_shape")
    exact_values = (
        ("model_id", DINO_V2_MODEL_ID),
        ("model_revision", DINO_V2_REVISION),
        ("preprocessing_version", contract.preprocessing_version),
        ("minimum_deployment_target", MINIMUM_DEPLOYMENT_TARGET),
        ("compute_precision", COMPUTE_PRECISION),
        ("model_sha256", DINOV2_ARTIFACT_SHA256["model.safetensors"]),
    )
    for name, expected in exact_values:
        if not isinstance(payload[name], str) or payload[name] != expected:
            raise ValueError(f"export metadata field does not match release contract: {name}")
    require_sha256(payload["package_sha256"], "export package digest")
    if payload["source_artifact_sha256"] != DINOV2_ARTIFACT_SHA256:
        raise ValueError("export source artifact digests do not match the release contract")
    tools = payload["tool_versions"]
    if not isinstance(tools, dict) or tools != dict(_TOOL_VERSIONS):
        raise ValueError("export tool versions do not match the audited release contract")


def _exact_shape(value: object, expected: tuple[int, ...], name: str) -> None:
    valid = (
        isinstance(value, list)
        and len(value) == len(expected)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and tuple(value) == expected
    )
    if not valid:
        raise ValueError(f"export metadata {name} does not match the exact integer shape")


def _load_embedding_array(path: Path, name: str) -> np.ndarray:
    return load_bounded_parity_array(
        path,
        name,
        expected_columns=EMBEDDING_DIMENSION,
    )


def _unit_norms(values: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.vector_norm(values, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3, rtol=0.0):
        raise ValueError(f"{name} must be L2 normalized")
    return norms


def _load_open_reports(
    paths: ReleasePaths,
    package_digest: str | None,
    catalog_digest: str | None,
) -> tuple[tuple[tuple[int, float], ...], tuple[str, ...]]:
    results: tuple[tuple[int, float], ...] = ()
    problems: tuple[str, ...] = ()
    for kind, path in paths.open_reports:
        try:
            payload = _load_evaluation_report(
                path, kind, _OPEN_REPORT_KEYS, package_digest, catalog_digest
            )
            count = positive_integer(payload["sampleCount"], f"{kind} sampleCount")
            rate = finite_rate(payload["falseAcceptRate"], f"{kind} falseAcceptRate")
            results = (*results, (count, rate))
        except (OSError, OverflowError, UnicodeError, ValueError, TypeError) as error:
            problems = (*problems, f"evaluation/{path.name}: {error}")
    return results, problems


def _load_evaluation_report(
    path: Path,
    evaluation_type: str,
    keys: set[str],
    package_digest: str | None,
    catalog_digest: str | None,
) -> Mapping[str, Any]:
    payload = load_json_mapping(path, f"{evaluation_type} report")
    if set(payload) != keys:
        raise ValueError("report fields do not match the exact schema")
    schema = payload["schemaVersion"]
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        raise ValueError("report schemaVersion must be the integer 1")
    if not isinstance(payload["evaluationType"], str) or payload["evaluationType"] != evaluation_type:
        raise ValueError("report identity does not match its fixed release path")
    if package_digest is None or payload["modelPackageSha256"] != package_digest:
        raise ValueError("report package digest does not match the release")
    if catalog_digest is None or payload["catalogManifestSha256"] != catalog_digest:
        raise ValueError("report catalog digest does not match the release")
    _validate_production_evidence(payload)
    return payload


def _validate_parity_report(
    paths: ReleasePaths,
    *,
    package_digest: str | None,
    catalog_digest: str | None,
    sample_count: int,
) -> None:
    payload = load_json_mapping(paths.parity_report, "parity report")
    if set(payload) != _PARITY_REPORT_KEYS:
        raise ValueError("parity report fields do not match the exact schema")
    schema = payload["schemaVersion"]
    if isinstance(schema, bool) or not isinstance(schema, int) or schema != 1:
        raise ValueError("parity report schemaVersion must be the integer 1")
    if payload["evaluationType"] != "pytorchCoreMLParity":
        raise ValueError("parity report identity does not match its fixed release path")
    if package_digest is None or payload["modelPackageSha256"] != package_digest:
        raise ValueError("parity report package digest does not match the release")
    if catalog_digest is None or payload["catalogManifestSha256"] != catalog_digest:
        raise ValueError("parity report catalog digest does not match the release")
    if payload["sourceModelSha256"] != DINOV2_ARTIFACT_SHA256["model.safetensors"]:
        raise ValueError("parity report source model digest does not match the release")
    if payload["preprocessingVersion"] != mobile_model_contract().preprocessing_version:
        raise ValueError("parity report preprocessing version does not match the release")
    if positive_integer(payload["sampleCount"], "parity sampleCount") != sample_count:
        raise ValueError("parity report sampleCount does not match the arrays")
    if payload["pytorchEmbeddingsSha256"] != sha256_file(paths.pytorch_embeddings):
        raise ValueError("parity report PyTorch array digest does not match")
    if payload["coremlEmbeddingsSha256"] != sha256_file(paths.coreml_embeddings):
        raise ValueError("parity report Core ML array digest does not match")
    _validate_production_evidence(payload)


def _validate_production_evidence(payload: Mapping[str, Any]) -> None:
    if payload["evidencePurpose"] != "production":
        raise ValueError("report evidencePurpose must be production")
    provenance = payload["provenanceUrl"]
    if not isinstance(provenance, str):
        raise ValueError("report provenanceUrl must be an absolute HTTPS URL")
    parsed = urlsplit(provenance)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("report provenanceUrl must be an absolute HTTPS URL")
    require_sha256(payload["fixtureSetSha256"], "report fixtureSetSha256")


def _closed_metrics(payload: Mapping[str, Any]) -> tuple[float, float]:
    top_1 = finite_rate(payload["top1"], "closed-set top1")
    top_3 = finite_rate(payload["top3"], "closed-set top3")
    if top_3 < top_1:
        raise ValueError("closed-set top3 must be greater than or equal to top1")
    return top_1, top_3


def _load_json(path: Path, name: str) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} file is missing or not a regular file")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error
