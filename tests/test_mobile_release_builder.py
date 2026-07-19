"""Fail-closed orchestration and iOS-equivalent retrieval tests."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from fluke_model.mobile_catalog import ReferenceRow
from fluke_model.coreml_artifact import package_tree_sha256
from fluke_model.mobile_release_builder import (
    BuildOptions,
    MAXIMUM_REFERENCE_COUNT,
    EvaluationPlan,
    build_mobile_release,
    load_evaluation_plan,
    load_production_runtimes,
    rank_catalog,
    require_production_approval,
)

RIGHTS_FIXTURE = Path(__file__).parent / "fixtures" / "mobile-catalog" / "rights-attestation.json"


def test_rank_catalog_matches_ios_top25_mean_top3_and_stable_ties() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    references = np.array([[0.9, 0.0], [0.6, 0.0], [0.3, 0.0], [0.8, 0.0]], dtype=np.float32)
    rows = (
        ReferenceRow("a", "whale-a", "catalog-a", "source"),
        ReferenceRow("b", "whale-a", "catalog-a", "source"),
        ReferenceRow("c", "whale-a", "catalog-a", "source"),
        ReferenceRow("d", "whale-b", "catalog-b", "source"),
    )

    ranked = rank_catalog(query, references, rows, limit=3)

    assert ranked == (
        ("whale-b", "catalog-b", pytest.approx(0.8)),
        ("whale-a", "catalog-a", pytest.approx(0.6)),
    )


def test_evaluation_plan_rejects_test_purpose_for_production(tmp_path: Path) -> None:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "evidencePurpose": "test",
                "approvedBy": "Synthetic fixture generator",
                "approvedAt": "2026-07-19T00:00:00+00:00",
                "provenanceUrl": "https://example.invalid/test-plan",
                "cohortDefinitions": {
                    name: "Synthetic contract fixture only"
                    for name in (
                        "closedSetRetrieval",
                        "openSet",
                        "nonOrca",
                        "poorQuality",
                        "occlusion",
                        "distributionShift",
                        "parity",
                    )
                },
            }
        ),
        encoding="utf-8",
    )

    plan = load_evaluation_plan(path)

    with pytest.raises(ValueError, match="production"):
        require_production_approval(plan, corpus_purpose="test", rights_purpose="test")


def test_reference_limit_matches_ios_loader_contract() -> None:
    assert MAXIMUM_REFERENCE_COUNT == 50_000


def test_production_runtime_fails_instead_of_faking_coreml_off_macos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("fluke_model.mobile_release_builder.platform.system", lambda: "Linux")

    with pytest.raises(RuntimeError, match="requires executable Core ML on macOS"):
        load_production_runtimes(
            model_artifact_dir=tmp_path / "artifact",
            model_package_path=tmp_path / "model.mlpackage",
        )


def test_evaluation_plan_is_immutable() -> None:
    plan = EvaluationPlan(
        purpose="test",
        approved_by="fixture",
        approved_at="2026-07-19T00:00:00+00:00",
        provenance_url="https://example.invalid/test",
        cohort_definitions=(("parity", "test"),),
    )

    with pytest.raises(AttributeError):
        plan.purpose = "production"


def test_builder_rejects_test_inputs_without_running_or_publishing(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    image = corpus / "reference.jpg"
    image.write_bytes(b"synthetic-image")
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "evidencePurpose": "test",
                "provenanceUrl": "https://example.invalid/test-corpus",
                "rows": [
                    {
                        "fixtureId": "reference-1",
                        "relativePath": "reference.jpg",
                        "imageSha256": hashlib.sha256(b"synthetic-image").hexdigest(),
                        "roles": ["reference"],
                        "referencePhotoId": "reference-1",
                        "whaleId": "whale-1",
                        "catalogId": "catalog-1",
                        "sourceId": "synthetic-owned-fixture",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "evidencePurpose": "test",
                "approvedBy": "Synthetic fixture generator",
                "approvedAt": "2026-07-19T00:00:00+00:00",
                "provenanceUrl": "https://example.invalid/test-plan",
                "cohortDefinitions": {
                    name: "Synthetic test cohort"
                    for name in (
                        "parity",
                        "closedSetRetrieval",
                        "openSet",
                        "nonOrca",
                        "poorQuality",
                        "occlusion",
                        "distributionShift",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    package = tmp_path / "FlukeEmbedder.mlpackage"
    package.mkdir()
    (package / "Manifest.json").write_text("synthetic", encoding="utf-8")
    metadata = tmp_path / "export-metadata.json"
    metadata.write_text(
        json.dumps({"package_sha256": package_tree_sha256(package)}), encoding="utf-8"
    )
    output = tmp_path / "release"

    with pytest.raises(ValueError, match="purpose must be production"):
        build_mobile_release(
            corpus_manifest_path=manifest,
            corpus_root=corpus,
            evaluation_plan_path=plan,
            rights_path=RIGHTS_FIXTURE,
            model_artifact_dir=tmp_path / "model-artifact",
            model_package_path=package,
            export_metadata_path=metadata,
            output_dir=output,
            options=BuildOptions("test", 1, 1, 0.7, 0.1),
        )
    assert not output.exists()
