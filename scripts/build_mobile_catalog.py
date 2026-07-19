#!/usr/bin/env python3
"""Build a deterministic rights-gated catalog for the iOS app bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.coreml_artifact import package_tree_sha256  # noqa: E402
from fluke_model.mobile_catalog import (  # noqa: E402
    MobileCatalogRelease,
    ReferenceRow,
    manifest_payload,
    write_mobile_catalog,
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
_REFERENCE_KEYS = {"referencePhotoId", "whaleId", "catalogId", "sourceId"}


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
    parser.add_argument("--score-semantics", default="cosine_similarity_not_probability")
    parser.add_argument("--score-threshold", type=float, required=True)
    parser.add_argument("--margin-threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        export = _load_export_metadata(args.model_metadata)
        package_sha256 = package_tree_sha256(args.model_package)
        if package_sha256 != export["package_sha256"]:
            raise ValueError("Core ML package digest does not match export metadata")
        embeddings = _load_embeddings(args.embeddings)
        rows = _load_reference_rows(args.references)
        release = _release_from_args(args, export, package_sha256, embeddings)
        manifest = write_mobile_catalog(args.output_dir, embeddings, rows, release)
    except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(manifest_payload(manifest), indent=2, sort_keys=True))
    return 0


def _load_export_metadata(path: Path) -> dict[str, Any]:
    payload = _load_json_file(path, "Core ML export metadata")
    if not isinstance(payload, dict) or set(payload) != _EXPORT_METADATA_KEYS:
        raise ValueError("Core ML export metadata fields do not match the required schema")
    required_text = ("model_id", "model_revision", "preprocessing_version", "package_sha256")
    if any(not isinstance(payload[key], str) or not payload[key] for key in required_text):
        raise ValueError("Core ML export metadata identity fields must be non-empty strings")
    output_shape = payload["output_shape"]
    if not isinstance(output_shape, list) or len(output_shape) != 2:
        raise ValueError("Core ML export output shape is invalid")
    return payload


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
