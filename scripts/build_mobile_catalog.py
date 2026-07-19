#!/usr/bin/env python3
"""Build a deterministic rights-gated catalog for the iOS app bundle."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.coreml_artifact import (  # noqa: E402
    COMPUTE_PRECISION,
    MINIMUM_DEPLOYMENT_TARGET,
    CoreMLExportError,
    package_tree_sha256,
)
from fluke_model.embedders import DINO_V2_MODEL_ID, DINO_V2_REVISION  # noqa: E402
from fluke_model.mobile_catalog import (  # noqa: E402
    MobileCatalogRelease,
    ReferenceRow,
    SCORE_SEMANTICS,
    manifest_payload,
    write_mobile_catalog,
)
from fluke_model.mobile_export import mobile_model_contract  # noqa: E402
from fluke_model.model_artifact import DINOV2_ARTIFACT_SHA256  # noqa: E402

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
_REFERENCE_KEYS = {"referencePhotoId", "whaleId", "catalogId", "sourceId"}
_TOOL_VERSION_KEYS = {"coremltools", "numpy", "python", "torch", "transformers"}
_AUDITED_TOOL_VERSIONS = {
    "coremltools": "9.0",
    "numpy": "2.2.6",
    "torch": "2.13.0",
    "transformers": "5.14.0",
}
_PYTHON_VERSION_PATTERN = re.compile(r"3\.11\.\d+")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, rights-gated mobile reference catalog"
    )
    parser.add_argument("--model-package", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True, help="Core ML embeddings .npy")
    parser.add_argument("--references", type=Path, required=True, help="Reference rows JSON")
    parser.add_argument("--rights", type=Path, required=True, help="Rights attestation JSON")
    parser.add_argument("--manifest-version", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--index-version", required=True)
    parser.add_argument(
        "--score-semantics", choices=(SCORE_SEMANTICS,), default=SCORE_SEMANTICS
    )
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--margin-threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        _validate_path_boundaries(args.output_dir, _source_paths_from_args(args))
        export = _load_export_metadata(args.model_metadata)
        package_sha256 = package_tree_sha256(args.model_package)
        if package_sha256 != export["package_sha256"]:
            raise ValueError("Core ML package digest does not match export metadata")
        embeddings = _load_embeddings(args.embeddings)
        rows = _load_reference_rows(args.references)
        release = _release_from_args(args, export, package_sha256, embeddings)
        manifest = write_mobile_catalog(args.output_dir, embeddings, rows, release)
    except (
        CoreMLExportError,
        OSError,
        UnicodeError,
        ValueError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(manifest_payload(manifest), indent=2, sort_keys=True))
    return 0


def _load_export_metadata(path: Path) -> dict[str, Any]:
    payload = _load_json_file(path, "Core ML export metadata")
    if not isinstance(payload, dict) or set(payload) != _EXPORT_METADATA_KEYS:
        raise ValueError("Core ML export metadata fields do not match the required schema")
    _validate_export_identity(payload)
    _validate_export_shapes(payload)
    _validate_export_target(payload)
    _validate_export_tool_versions(payload["tool_versions"])
    return payload


def _source_paths_from_args(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "model package": args.model_package,
        "export metadata": args.model_metadata,
        "embeddings": args.embeddings,
        "references": args.references,
        "rights": args.rights,
    }


def _validate_path_boundaries(output_dir: Path, sources: dict[str, Path]) -> None:
    paths = {"output": Path(output_dir), **{name: Path(path) for name, path in sources.items()}}
    for name, path in paths.items():
        _reject_symlink_components(path, name=name)
    resolved_output = paths["output"].resolve(strict=False)
    for name, source in sources.items():
        resolved_source = Path(source).resolve(strict=False)
        if _paths_overlap(resolved_output, resolved_source):
            raise ValueError(f"mobile catalog output and {name} paths overlap")


def _reject_symlink_components(path: Path, *, name: str) -> None:
    absolute = path.absolute()
    components = (*reversed(absolute.parents), absolute)
    for component in components:
        if component.is_symlink():
            raise ValueError(f"{name} path contains a symbolic link component")


def _paths_overlap(first: Path, second: Path) -> bool:
    return (
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _validate_export_identity(payload: dict[str, Any]) -> None:
    if (payload["model_id"], payload["model_revision"]) != (
        DINO_V2_MODEL_ID,
        DINO_V2_REVISION,
    ):
        raise ValueError("Core ML export model identity does not match the pinned contract")
    contract = mobile_model_contract()
    if payload["preprocessing_version"] != contract.preprocessing_version:
        raise ValueError("Core ML export preprocessing version does not match the pinned contract")
    if payload["model_sha256"] != DINOV2_ARTIFACT_SHA256["model.safetensors"]:
        raise ValueError("Core ML export source model digest does not match the pinned artifact")
    package_sha256 = payload["package_sha256"]
    if not isinstance(package_sha256, str) or _SHA256_PATTERN.fullmatch(package_sha256) is None:
        raise ValueError("Core ML export package digest must be a lowercase SHA256")


def _validate_export_shapes(payload: dict[str, Any]) -> None:
    contract = mobile_model_contract()
    if not _is_exact_integer_shape(payload["input_shape"], contract.input_shape):
        raise ValueError("Core ML export input shape does not match the pinned contract")
    if not _is_exact_integer_shape(payload["output_shape"], contract.output_shape):
        raise ValueError("Core ML export output shape does not match the pinned contract")


def _validate_export_target(payload: dict[str, Any]) -> None:
    if payload["minimum_deployment_target"] != MINIMUM_DEPLOYMENT_TARGET:
        raise ValueError("Core ML export deployment target does not match the pinned contract")
    if payload["compute_precision"] != COMPUTE_PRECISION:
        raise ValueError("Core ML export precision does not match the pinned contract")


def _validate_export_tool_versions(value: object) -> None:
    if not isinstance(value, dict) or set(value) != _TOOL_VERSION_KEYS:
        raise ValueError("Core ML export tool version fields do not match the audited contract")
    for name, expected in _AUDITED_TOOL_VERSIONS.items():
        if value[name] != expected:
            raise ValueError(f"Core ML export {name} version does not match the audited contract")
    python_version = value["python"]
    if not isinstance(python_version, str) or _PYTHON_VERSION_PATTERN.fullmatch(python_version) is None:
        raise ValueError("Core ML export python version must be an audited Python 3.11.x release")


def _is_exact_integer_shape(value: object, expected: tuple[int, ...]) -> bool:
    return (
        isinstance(value, list)
        and len(value) == len(expected)
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and tuple(value) == expected
    )


def _load_embeddings(path: Path) -> np.ndarray:
    _require_regular_file(path, "Core ML embeddings")
    values = np.load(path, allow_pickle=False)
    if not isinstance(values, np.ndarray):
        raise ValueError("Core ML embeddings must contain one NumPy array")
    return values


def _load_reference_rows(path: Path) -> tuple[ReferenceRow, ...]:
    payload = _load_json_file(path, "reference manifest")
    if not isinstance(payload, list) or not payload:
        raise ValueError("reference manifest must be a non-empty JSON array")
    return tuple(_reference_row(value) for value in payload)


def _reference_row(value: object) -> ReferenceRow:
    if not isinstance(value, dict) or set(value) != _REFERENCE_KEYS:
        raise ValueError("reference manifest row fields do not match the required schema")
    if any(not isinstance(value[key], str) for key in _REFERENCE_KEYS):
        raise ValueError("reference manifest identity fields must be strings")
    return ReferenceRow(
        reference_photo_id=value["referencePhotoId"],
        whale_id=value["whaleId"],
        catalog_id=value["catalogId"],
        source_id=value["sourceId"],
    )


def _release_from_args(
    args: argparse.Namespace,
    export: dict[str, Any],
    package_sha256: str,
    embeddings: np.ndarray,
) -> MobileCatalogRelease:
    output_shape = export["output_shape"]
    dimension = output_shape[1]
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise ValueError("Core ML export embedding dimension must be an integer")
    if embeddings.ndim != 2 or embeddings.shape[1] != dimension:
        raise ValueError("Core ML embeddings do not match the export output dimension")
    return MobileCatalogRelease(
        manifest_version=args.manifest_version,
        model_id=export["model_id"],
        model_revision=export["model_revision"],
        model_version=args.model_version,
        model_sha256=package_sha256,
        preprocessing_version=export["preprocessing_version"],
        embedding_dimension=dimension,
        index_version=args.index_version,
        score_semantics=args.score_semantics,
        score_threshold=args.score_threshold,
        margin_threshold=args.margin_threshold,
        rights_attestation_path=args.rights,
    )


def _load_json_file(path: Path, name: str) -> object:
    _require_regular_file(path, name)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_regular_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular file")


if __name__ == "__main__":
    raise SystemExit(main())
