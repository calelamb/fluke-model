"""Fail-closed construction of a reproducible, independently verified mobile release."""

from __future__ import annotations

import json
import math
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import torch
from PIL import Image, ImageOps

from fluke_model.coreml_artifact import (
    COREML_PACKAGE_NAME,
    EXPORT_METADATA_NAME,
    FixedShapeDINOv2Embedder,
    package_tree_sha256,
)
from fluke_model.embedders import DINO_V2_MODEL_ID, DINO_V2_REVISION
from fluke_model.mobile_catalog import (
    MobileCatalogRelease,
    ReferenceRow,
    SCORE_SEMANTICS,
    _load_mobile_rights,
    sha256_file,
    write_mobile_catalog,
    validate_published_mobile_catalog,
)
from fluke_model.mobile_export import mobile_model_contract
from fluke_model.mobile_release_contracts import REPORT_FILENAME
from fluke_model.mobile_release_evidence import (
    CorpusManifest,
    DecisionRecord,
    OPEN_EVALUATION_TYPES,
    canonical_decisions_payload,
    canonical_fixture_payload,
    fixture_set_sha256,
    load_corpus_manifest,
    load_published_fixture_rows,
    recompute_metrics,
)
from fluke_model.model_artifact import DINOV2_ARTIFACT_SHA256, verify_dinov2_artifact
from fluke_model.secure_snapshot import (
    publish_directory_no_replace,
    snapshot_relative_file,
    snapshot_regular_file,
    snapshot_tree,
)

MAXIMUM_REFERENCE_COUNT = 50_000
MODEL_VERSION = "dinov2-small-coreml-v1"
INDEX_VERSION = "mobile-reference-v1"
_PLAN_KEYS = {
    "schemaVersion",
    "evidencePurpose",
    "approvedBy",
    "approvedAt",
    "provenanceUrl",
    "cohortDefinitions",
    "scoreThreshold",
    "marginThreshold",
    "runtimeVersions",
}
_COHORTS = ("parity", "closedSetRetrieval", *OPEN_EVALUATION_TYPES)
_RUNTIME_VERSION_KEYS = {
    "coremltools",
    "macos",
    "numpy",
    "pillow",
    "python",
    "torch",
    "transformers",
    "xcode",
}
_IMAGE_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGE_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
_MAX_IMAGE_PIXELS = 40_000_000


@dataclass(frozen=True)
class EvaluationPlan:
    """Human-approved production evaluation intent, separate from measured output."""

    purpose: str
    approved_by: str
    approved_at: str
    provenance_url: str
    score_threshold: float
    margin_threshold: float
    runtime_versions: tuple[tuple[str, str], ...]
    cohort_definitions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohort_definitions", tuple(self.cohort_definitions))
        object.__setattr__(self, "runtime_versions", tuple(self.runtime_versions))


@dataclass(frozen=True)
class BuildOptions:
    """Explicit non-tunable release identity and compatibility inputs."""

    manifest_version: str
    minimum_app_build: int
    maximum_app_build: int


@dataclass(frozen=True)
class _EmbeddingRuntimes:
    """Real model execution boundaries; injectable only below the production CLI."""

    pytorch: Callable[[np.ndarray], np.ndarray]
    coreml: Callable[[np.ndarray], np.ndarray]


def load_evaluation_plan(path: Path) -> EvaluationPlan:
    """Load the exact human-approval schema and validate its timestamp and URL."""
    source = _regular_file(path, "evaluation plan")
    if source.stat().st_size > 1024 * 1024:
        raise ValueError("evaluation plan exceeds the maximum size")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != _PLAN_KEYS:
        raise ValueError("evaluation plan fields do not match the exact schema")
    if payload["schemaVersion"] != 1 or isinstance(payload["schemaVersion"], bool):
        raise ValueError("evaluation plan schemaVersion must be the integer 1")
    definitions = payload["cohortDefinitions"]
    if not isinstance(definitions, dict) or set(definitions) != set(_COHORTS):
        raise ValueError("evaluation plan must define every fixed evaluation cohort")
    approved_at = _timestamp(payload["approvedAt"])
    return EvaluationPlan(
        purpose=_text(payload["evidencePurpose"], "evidencePurpose"),
        approved_by=_text(payload["approvedBy"], "approvedBy"),
        approved_at=approved_at,
        provenance_url=_https(payload["provenanceUrl"], "provenanceUrl"),
        score_threshold=_threshold(payload["scoreThreshold"], "scoreThreshold"),
        margin_threshold=_threshold(payload["marginThreshold"], "marginThreshold"),
        runtime_versions=_runtime_versions(payload["runtimeVersions"]),
        cohort_definitions=tuple(
            (name, _text(definitions[name], f"cohortDefinitions.{name}"))
            for name in sorted(definitions)
        ),
    )


def require_production_approval(
    plan: EvaluationPlan, *, corpus_purpose: str, rights_purpose: str
) -> None:
    """Prohibit test/research inputs at the production builder boundary."""
    purposes = {
        "evaluation plan": plan.purpose,
        "corpus manifest": corpus_purpose,
        "rights attestation": rights_purpose,
    }
    for name, purpose in purposes.items():
        if purpose != "production":
            raise ValueError(f"{name} purpose must be production")


def rank_catalog(
    query: np.ndarray,
    references: np.ndarray,
    rows: tuple[ReferenceRow, ...],
    *,
    limit: int = 3,
) -> tuple[tuple[str, str, float], ...]:
    """Match iOS ExactCosineSearcher: top 25 refs, mean best 3 per catalog."""
    values = np.asarray(query, dtype=np.float32)
    matrix = np.asarray(references, dtype=np.float32)
    if values.ndim != 1 or matrix.ndim != 2 or matrix.shape[1] != values.shape[0]:
        raise ValueError("query and reference embedding shapes do not match")
    if matrix.shape[0] != len(rows) or not 1 <= limit <= 100:
        raise ValueError("catalog rows or result limit do not match the retrieval contract")
    scores = matrix @ values
    top_references = sorted(
        zip(rows, scores, strict=True),
        key=lambda item: (-float(item[1]), item[0].reference_photo_id),
    )[:25]
    grouped: dict[str, list[tuple[ReferenceRow, np.float32]]] = {}
    for row, score in top_references:
        grouped.setdefault(row.catalog_id, []).append((row, np.float32(score)))
    identities: list[tuple[str, str, float]] = []
    for catalog_id, matches in grouped.items():
        best = sorted(matches, key=lambda item: (-float(item[1]), item[0].reference_photo_id))[:3]
        total = np.float32(0.0)
        for _, score in best:
            total = np.float32(total + score)
        score = np.float32(total / np.float32(len(best)))
        identities.append((best[0][0].whale_id, catalog_id, float(score)))
    return tuple(sorted(identities, key=lambda item: (-item[2], item[1], item[0]))[:limit])


def build_mobile_release(
    *,
    corpus_manifest_path: Path,
    corpus_root: Path,
    evaluation_plan_path: Path,
    rights_path: Path,
    model_artifact_dir: Path,
    model_package_path: Path,
    export_metadata_path: Path,
    output_dir: Path,
    options: BuildOptions,
) -> Mapping[str, object]:
    """Build into a fresh sibling staging directory, verify, then atomically publish."""
    sources = (
        corpus_manifest_path,
        corpus_root,
        evaluation_plan_path,
        rights_path,
        model_artifact_dir,
        model_package_path,
        export_metadata_path,
    )
    _validate_boundaries(output_dir, sources)
    package_digest = package_tree_sha256(model_package_path)
    _require_package_metadata_digest(export_metadata_path, package_digest)
    manifest = load_corpus_manifest(corpus_manifest_path, corpus_root)
    plan = load_evaluation_plan(evaluation_plan_path)
    rights, _ = _load_mobile_rights(rights_path)
    require_production_approval(
        plan, corpus_purpose=manifest.purpose, rights_purpose=rights.purpose
    )
    references = tuple(row for row in manifest.rows if "reference" in row.roles)
    if not references or len(references) > MAXIMUM_REFERENCE_COUNT:
        raise ValueError("reference count must be within [1, 50000]")
    source_ids = tuple(sorted({row.source_id for row in references if row.source_id is not None}))
    rights.validate_for(
        model_id=DINO_V2_MODEL_ID,
        model_revision=DINO_V2_REVISION,
        reference_source_ids=source_ids,
        required_purpose="production",
    )
    _validate_options(options)
    return _stage_release(
        manifest=manifest,
        plan=plan,
        corpus_root=corpus_root,
        rights_path=rights_path,
        model_artifact_dir=model_artifact_dir,
        model_package_path=model_package_path,
        export_metadata_path=export_metadata_path,
        output_dir=output_dir,
        options=options,
    )


def _load_production_runtimes(
    *,
    model_artifact_dir: Path,
    model_package_path: Path,
    expected_versions: Mapping[str, str] | None = None,
) -> _EmbeddingRuntimes:
    """Load the pinned PyTorch artifact and executable Core ML package or fail."""
    if platform.system() != "Darwin":
        raise RuntimeError("production mobile release requires executable Core ML on macOS")
    verify_dinov2_artifact(model_artifact_dir)
    try:
        import coremltools
        from transformers import AutoModel

        source_model = AutoModel.from_pretrained(
            str(model_artifact_dir), local_files_only=True, use_safetensors=True
        )
        torch_model = FixedShapeDINOv2Embedder(source_model).eval()
        coreml_model = coremltools.models.MLModel(
            str(model_package_path), compute_units=coremltools.ComputeUnit.CPU_ONLY
        )
        if expected_versions is not None:
            _validate_runtime_versions(expected_versions, coremltools)
    except Exception as error:
        raise RuntimeError(f"production model runtime could not be loaded: {error}") from error

    @torch.no_grad()
    def pytorch_embed(pixels: np.ndarray) -> np.ndarray:
        output = torch_model(torch.from_numpy(np.array(pixels, dtype=np.float32, copy=True)))
        return output.detach().cpu().numpy()[0]

    def coreml_embed(pixels: np.ndarray) -> np.ndarray:
        try:
            result = coreml_model.predict({"pixels": pixels})
            return np.asarray(result["embedding"], dtype=np.float32).reshape(-1)
        except Exception as error:
            raise RuntimeError(f"Core ML prediction failed: {error}") from error

    return _EmbeddingRuntimes(pytorch=pytorch_embed, coreml=coreml_embed)


def _stage_release(
    *,
    manifest: CorpusManifest,
    plan: EvaluationPlan,
    corpus_root: Path,
    rights_path: Path,
    model_artifact_dir: Path,
    model_package_path: Path,
    export_metadata_path: Path,
    output_dir: Path,
    options: BuildOptions,
) -> Mapping[str, object]:
    from fluke_model.mobile_release import (
        report_payload,
        verify_mobile_release_directory,
        write_mobile_release_report,
    )
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError("mobile release output must not already exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        snapshot_tree(model_package_path, staging / COREML_PACKAGE_NAME)
        snapshot_tree(model_artifact_dir, staging / "source-model")
        snapshot_regular_file(export_metadata_path, staging / EXPORT_METADATA_NAME)
        snapshot_regular_file(rights_path, staging / "rights-attestation.json")
        evaluation = staging / "evaluation"
        fixtures = evaluation / "fixtures"
        fixtures.mkdir(parents=True)
        snapshot_manifest = _snapshot_corpus(manifest, corpus_root, fixtures)
        _require_package_metadata_digest(
            staging / EXPORT_METADATA_NAME, package_tree_sha256(staging / COREML_PACKAGE_NAME)
        )
        runtimes = _load_production_runtimes(
            model_artifact_dir=staging / "source-model",
            model_package_path=staging / COREML_PACKAGE_NAME,
            expected_versions=dict(plan.runtime_versions),
        )
        embeddings = _embed_corpus(snapshot_manifest, runtimes)
        catalog_rows, catalog_vectors = _catalog_inputs(snapshot_manifest, embeddings)
        package_digest = package_tree_sha256(staging / COREML_PACKAGE_NAME)
        catalog_release = MobileCatalogRelease(
            manifest_version=options.manifest_version,
            model_id=DINO_V2_MODEL_ID,
            model_revision=DINO_V2_REVISION,
            model_version=MODEL_VERSION,
            model_sha256=package_digest,
            preprocessing_version=mobile_model_contract().preprocessing_version,
            embedding_dimension=384,
            index_version=INDEX_VERSION,
            minimum_app_build=options.minimum_app_build,
            maximum_app_build=options.maximum_app_build,
            score_semantics=SCORE_SEMANTICS,
            score_threshold=plan.score_threshold,
            margin_threshold=plan.margin_threshold,
            rights_attestation_path=staging / "rights-attestation.json",
        )
        write_mobile_catalog(staging / "catalog", catalog_vectors, catalog_rows, catalog_release)
        _write_evaluation(
            staging,
            snapshot_manifest,
            plan,
            embeddings,
            catalog_rows,
            catalog_vectors,
        )
        report = verify_mobile_release_directory(staging)
        write_mobile_release_report(staging / REPORT_FILENAME, report)
        if not report.ready:
            failed = tuple(gate.name for gate in report.gates if not gate.passed)
            raise ValueError(
                f"staged mobile release failed verification gates: {', '.join(failed)}"
            )
        publish_directory_no_replace(staging, destination)
        return report_payload(report)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def reexecute_mobile_release(paths: object, catalog: object) -> tuple[bool, str]:
    """Independently execute staged artifacts and compare all caller-supplied evidence."""
    snapshot_parent = Path(tempfile.mkdtemp(prefix="orcawatch-release-verification-"))
    try:
        from fluke_model.mobile_release_contracts import release_paths

        snapshot_root = snapshot_parent / "release"
        snapshot_tree(getattr(paths, "root"), snapshot_root)
        verified_paths = release_paths(snapshot_root)
        verified_catalog = validate_published_mobile_catalog(verified_paths.catalog_dir)
        plan = load_evaluation_plan(verified_paths.evaluation_plan)
        runtimes = _load_production_runtimes(
            model_artifact_dir=verified_paths.source_model,
            model_package_path=verified_paths.package,
            expected_versions=dict(plan.runtime_versions),
        )
        rows = _bind_release_fixture_paths(
            load_published_fixture_rows(verified_paths.fixture_manifest),
            verified_paths.fixture_images,
        )
        manifest = CorpusManifest("production", plan.provenance_url, rows)
        embeddings = _embed_corpus(manifest, runtimes)
        parity = tuple(row for row in rows if "parity" in row.roles)
        expected_pytorch = np.stack([embeddings[row.fixture_id][0] for row in parity])
        expected_coreml = np.stack([embeddings[row.fixture_id][1] for row in parity])
        actual_pytorch = np.load(verified_paths.pytorch_embeddings, allow_pickle=False)
        actual_coreml = np.load(verified_paths.coreml_embeddings, allow_pickle=False)
        if not np.array_equal(actual_pytorch, expected_pytorch) or not np.array_equal(
            actual_coreml, expected_coreml
        ):
            raise ValueError("parity arrays do not match independent runtime execution")
        catalog_rows = tuple(verified_catalog.rows)
        vectors = np.fromfile(verified_paths.catalog_vectors, dtype="<f2").astype(np.float32)
        vectors = vectors.reshape(len(catalog_rows), 384)
        by_reference = {row.reference_photo_id: row for row in rows if "reference" in row.roles}
        expected_vectors = np.stack(
            [embeddings[by_reference[row.reference_photo_id].fixture_id][1] for row in catalog_rows]
        )
        if not np.array_equal(vectors, expected_vectors.astype("<f2").astype(np.float32)):
            raise ValueError("catalog vectors do not match independent Core ML execution")
        decisions = _build_decisions(manifest, embeddings, vectors, catalog_rows, plan)
        expected = canonical_decisions_payload(
            decisions,
            score_threshold=plan.score_threshold,
            margin_threshold=plan.margin_threshold,
        )
        if verified_paths.raw_decisions.read_bytes() != expected:
            raise ValueError("raw decisions do not match independent runtime execution")
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        return False, str(error)
    finally:
        shutil.rmtree(snapshot_parent, ignore_errors=True)
    return True, "pinned PyTorch and shipped Core ML evidence independently re-executed"


def _bind_release_fixture_paths(rows: tuple[object, ...], root: Path) -> tuple[object, ...]:
    bound = []
    for row in rows:
        path = root / getattr(row, "relative_path")
        digest = sha256_file(path)
        if digest != getattr(row, "image_sha256"):
            raise ValueError("release fixture image digest mismatch")
        bound.append(replace(row, path=path))
    return tuple(bound)


def _embed_corpus(
    manifest: CorpusManifest, runtimes: _EmbeddingRuntimes
) -> dict[str, tuple[np.ndarray | None, np.ndarray]]:
    results: dict[str, tuple[np.ndarray | None, np.ndarray]] = {}
    for row in manifest.rows:
        pixels = preprocess_image(row.path)
        coreml = _normalized(runtimes.coreml(pixels), f"Core ML embedding {row.fixture_id}")
        pytorch = None
        if "parity" in row.roles:
            pytorch = _normalized(runtimes.pytorch(pixels), f"PyTorch embedding {row.fixture_id}")
        results[row.fixture_id] = (pytorch, coreml)
    return results


def preprocess_image(path: Path) -> np.ndarray:
    """Apply the pinned resize/center-crop/ImageNet preprocessing contract."""
    try:
        with Image.open(path) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > _MAX_IMAGE_PIXELS:
                raise ValueError("image dimensions are outside the supported bounds")
            image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            target = (
                (256, 256 * height // width) if width <= height else (256 * width // height, 256)
            )
            resized = image.resize(target, Image.Resampling.BICUBIC)
            left = (resized.width - 224) // 2
            top = (resized.height - 224) // 2
            cropped = resized.crop((left, top, left + 224, top + 224))
            values = np.asarray(cropped, dtype=np.float32) / np.float32(255.0)
    except (OSError, ValueError) as error:
        raise ValueError(f"fixture image cannot be preprocessed: {path.name}: {error}") from error
    normalized = (values - _IMAGE_MEAN) / _IMAGE_STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def _catalog_inputs(
    manifest: CorpusManifest,
    embeddings: Mapping[str, tuple[np.ndarray | None, np.ndarray]],
) -> tuple[tuple[ReferenceRow, ...], np.ndarray]:
    fixtures = tuple(row for row in manifest.rows if "reference" in row.roles)
    rows = tuple(
        ReferenceRow(
            row.reference_photo_id or "",
            row.whale_id or "",
            row.catalog_id or "",
            row.source_id or "",
        )
        for row in fixtures
    )
    vectors = np.stack([embeddings[row.fixture_id][1] for row in fixtures]).astype(np.float32)
    return rows, vectors


def _write_evaluation(
    staging: Path,
    manifest: CorpusManifest,
    plan: EvaluationPlan,
    embeddings: Mapping[str, tuple[np.ndarray | None, np.ndarray]],
    catalog_rows: tuple[ReferenceRow, ...],
    catalog_vectors: np.ndarray,
) -> None:
    evaluation = staging / "evaluation"
    _write_json(
        evaluation / "evaluation-plan.json",
        {
            "schemaVersion": 1,
            "evidencePurpose": plan.purpose,
            "approvedBy": plan.approved_by,
            "approvedAt": plan.approved_at,
            "provenanceUrl": plan.provenance_url,
            "scoreThreshold": plan.score_threshold,
            "marginThreshold": plan.margin_threshold,
            "runtimeVersions": dict(plan.runtime_versions),
            "cohortDefinitions": dict(plan.cohort_definitions),
        },
    )
    parity_rows = tuple(row for row in manifest.rows if "parity" in row.roles)
    if not parity_rows:
        raise ValueError("corpus manifest must contain parity fixtures")
    pytorch = np.stack([embeddings[row.fixture_id][0] for row in parity_rows]).astype(np.float32)
    coreml = np.stack([embeddings[row.fixture_id][1] for row in parity_rows]).astype(np.float32)
    np.save(evaluation / "parity-pytorch.npy", pytorch, allow_pickle=False)
    np.save(evaluation / "parity-coreml.npy", coreml, allow_pickle=False)
    (evaluation / "fixture-manifest.json").write_bytes(canonical_fixture_payload(manifest.rows))
    ios_vectors = catalog_vectors.astype("<f2").astype(np.float32)
    decisions = _build_decisions(manifest, embeddings, ios_vectors, catalog_rows, plan)
    (evaluation / "decisions.json").write_bytes(
        canonical_decisions_payload(
            decisions,
            score_threshold=plan.score_threshold,
            margin_threshold=plan.margin_threshold,
        )
    )
    _write_reports(staging, plan, manifest, decisions)


def _snapshot_corpus(
    manifest: CorpusManifest, corpus_root: Path, destination: Path
) -> CorpusManifest:
    rows = []
    for row in manifest.rows:
        target = destination / row.relative_path
        digest = snapshot_relative_file(corpus_root, row.relative_path, target)
        if digest != row.image_sha256:
            raise ValueError(f"fixture image changed before authenticated snapshot: {row.fixture_id}")
        rows.append(replace(row, path=target))
    return CorpusManifest(manifest.purpose, manifest.provenance_url, tuple(rows))


def _build_decisions(
    manifest: CorpusManifest,
    embeddings: Mapping[str, tuple[np.ndarray | None, np.ndarray]],
    catalog_vectors: np.ndarray,
    catalog_rows: tuple[ReferenceRow, ...],
    plan: EvaluationPlan,
) -> tuple[DecisionRecord, ...]:
    decisions: list[DecisionRecord] = []
    evaluation_types = {"closedSetRetrieval", *OPEN_EVALUATION_TYPES}
    for row in manifest.rows:
        for evaluation_type in sorted(set(row.roles) & evaluation_types):
            ranked = rank_catalog(
                embeddings[row.fixture_id][1], catalog_vectors, catalog_rows, limit=3
            )
            top_score = ranked[0][2]
            second_score = ranked[1][2] if len(ranked) > 1 else -1.0
            accepted = _eligible(ranked, plan)
            decisions.append(
                DecisionRecord(
                    fixture_id=row.fixture_id,
                    evaluation_type=evaluation_type,
                    truth_whale_id=row.whale_id
                    if evaluation_type == "closedSetRetrieval"
                    else None,
                    ranked_whale_ids=tuple(item[0] for item in ranked),
                    top_score=top_score,
                    second_score=second_score,
                    accepted=accepted,
                )
            )
    return tuple(decisions)


def _eligible(ranked: tuple[tuple[str, str, float], ...], plan: EvaluationPlan) -> bool:
    first = np.float32(ranked[0][2])
    if first < np.float32(plan.score_threshold):
        return False
    if len(ranked) == 1:
        return True
    margin = first - np.float32(ranked[1][2]) + np.finfo(np.float32).eps
    return bool(margin >= np.float32(plan.margin_threshold))


def _write_reports(
    staging: Path,
    plan: EvaluationPlan,
    manifest: CorpusManifest,
    decisions: tuple[DecisionRecord, ...],
) -> None:
    evaluation = staging / "evaluation"
    package_digest = package_tree_sha256(staging / COREML_PACKAGE_NAME)
    catalog_digest = sha256_file(staging / "catalog" / "manifest.json")
    fixture_digest = fixture_set_sha256(manifest.rows)
    metrics = recompute_metrics(
        decisions,
        score_threshold=plan.score_threshold,
        margin_threshold=plan.margin_threshold,
    )
    common = {
        "schemaVersion": 1,
        "evidencePurpose": "production",
        "provenanceUrl": plan.provenance_url,
        "fixtureSetSha256": fixture_digest,
        "modelPackageSha256": package_digest,
        "catalogManifestSha256": catalog_digest,
    }
    pytorch_path = evaluation / "parity-pytorch.npy"
    coreml_path = evaluation / "parity-coreml.npy"
    parity_count = sum("parity" in row.roles for row in manifest.rows)
    _write_json(
        evaluation / "parity.json",
        {
            **common,
            "evaluationType": "pytorchCoreMLParity",
            "sourceModelSha256": DINOV2_ARTIFACT_SHA256["model.safetensors"],
            "preprocessingVersion": mobile_model_contract().preprocessing_version,
            "sampleCount": parity_count,
            "pytorchEmbeddingsSha256": sha256_file(pytorch_path),
            "coremlEmbeddingsSha256": sha256_file(coreml_path),
        },
    )
    filenames = {
        "closedSetRetrieval": "closed-set.json",
        **dict(
            zip(
                OPEN_EVALUATION_TYPES,
                (
                    "open-set.json",
                    "non-orca.json",
                    "poor-quality.json",
                    "occlusion.json",
                    "distribution-shift.json",
                ),
                strict=True,
            )
        ),
    }
    for evaluation_type, filename in filenames.items():
        if evaluation_type not in metrics:
            raise ValueError(f"corpus manifest lacks required cohort: {evaluation_type}")
        _write_json(
            evaluation / filename,
            {**common, "evaluationType": evaluation_type, **metrics[evaluation_type]},
        )


def _normalized(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (384,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 384-vector")
    norm = float(np.linalg.vector_norm(array))
    if not math.isfinite(norm) or abs(norm - 1.0) > 0.001:
        raise ValueError(f"{name} must be L2 normalized within the iOS tolerance")
    return np.ascontiguousarray(array, dtype=np.float32)


def _validate_options(options: BuildOptions) -> None:
    _text(options.manifest_version, "manifest version")
    if (
        isinstance(options.minimum_app_build, bool)
        or not isinstance(options.minimum_app_build, int)
        or isinstance(options.maximum_app_build, bool)
        or not isinstance(options.maximum_app_build, int)
        or options.minimum_app_build <= 0
        or options.maximum_app_build < options.minimum_app_build
    ):
        raise ValueError("app build range must contain ordered positive integers")


def _require_package_metadata_digest(path: Path, package_digest: str) -> None:
    source = _regular_file(path, "export metadata")
    if source.stat().st_size > 1024 * 1024:
        raise ValueError("export metadata exceeds the maximum size")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as error:
        raise ValueError(f"export metadata is invalid JSON: {error}") from error
    if not isinstance(payload, dict) or payload.get("package_sha256") != package_digest:
        raise ValueError("Core ML package digest does not match export metadata")


def _validate_boundaries(output: Path, sources: tuple[Path, ...]) -> None:
    destination = Path(output)
    _reject_symlinks(destination, "mobile release output")
    resolved = destination.resolve(strict=False)
    for source in sources:
        _reject_symlinks(source, "mobile release input")
        other = Path(source).resolve(strict=False)
        if resolved == other or resolved.is_relative_to(other) or other.is_relative_to(resolved):
            raise ValueError("mobile release output overlaps an input path")


def _regular_file(path: Path, name: str) -> Path:
    candidate = Path(path)
    _reject_symlinks(candidate, name)
    if not candidate.is_file():
        raise ValueError(f"{name} must be a regular file")
    return candidate


def _reject_symlinks(path: Path, name: str) -> None:
    absolute = Path(path).absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise ValueError(f"{name} path contains a symbolic link component")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _https(value: object, name: str) -> str:
    text = _text(value, name)
    parsed = urlsplit(text)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None:
        raise ValueError(f"{name} must be an absolute HTTPS URL")
    return text


def _timestamp(value: object) -> str:
    text = _text(value, "approvedAt")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("approvedAt must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("approvedAt must include a timezone")
    return text


def _threshold(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and within [-1, 1]")
    result = float(value)
    if not math.isfinite(result) or not -1 <= result <= 1:
        raise ValueError(f"{name} must be finite and within [-1, 1]")
    return result


def _runtime_versions(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, dict) or set(value) != _RUNTIME_VERSION_KEYS:
        raise ValueError("runtimeVersions fields do not match the exact schema")
    return tuple((name, _text(value[name], f"runtimeVersions.{name}")) for name in sorted(value))


def _validate_runtime_versions(expected: Mapping[str, str], coremltools: object) -> None:
    import PIL
    import transformers

    xcode_lines = (
        subprocess.run(
            ["/usr/bin/xcodebuild", "-version"], capture_output=True, check=True, text=True
        )
        .stdout.strip()
        .splitlines()
    )
    if len(xcode_lines) != 2:
        raise RuntimeError("xcode version output does not match the audited format")
    xcode = (
        f"{xcode_lines[0].removeprefix('Xcode ')} ({xcode_lines[1].removeprefix('Build version ')})"
    )
    observed = {
        "coremltools": str(getattr(coremltools, "__version__")),
        "macos": platform.mac_ver()[0],
        "numpy": np.__version__,
        "pillow": PIL.__version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "xcode": xcode,
    }
    if dict(expected) != observed:
        mismatch = next(name for name in sorted(observed) if expected.get(name) != observed[name])
        raise RuntimeError(f"production runtime version does not match approved plan: {mismatch}")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(
            payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
