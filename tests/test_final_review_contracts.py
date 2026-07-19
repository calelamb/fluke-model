"""Whole-branch release review regressions, grouped by numbered finding."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fluke_model.coreml_artifact as coreml_artifact
import fluke_model.mobile_catalog as mobile_catalog
import fluke_model.mobile_release_contracts as release_contracts
from fluke_model.model_artifact import DINOV2_ARTIFACT_SHA256


RIGHTS_FIXTURE = Path(__file__).parent / "fixtures" / "mobile-catalog" / "rights-attestation.json"


def _valid_coreml_spec() -> object:
    input_type = SimpleNamespace(shape=[1, 3, 224, 224], dataType=65568)
    output_type = SimpleNamespace(shape=[1, 384], dataType=65568)
    return SimpleNamespace(
        description=SimpleNamespace(
            input=[
                SimpleNamespace(
                    name="pixels",
                    type=SimpleNamespace(multiArrayType=input_type),
                )
            ],
            output=[
                SimpleNamespace(
                    name="embedding",
                    type=SimpleNamespace(multiArrayType=output_type),
                )
            ],
        )
    )


def test_finding_1_public_validator_fails_closed_on_literal_synthetic_package(
    tmp_path: Path,
) -> None:
    package = tmp_path / "Synthetic.mlpackage"
    package.mkdir()
    (package / "literal.bin").write_bytes(b"not a Core ML package")

    validator = getattr(coreml_artifact, "validate_coreml_package_interface", None)

    assert callable(validator)
    with pytest.raises(coreml_artifact.CoreMLExportError, match="reload"):
        validator(package)


def test_finding_1_validator_injects_only_isolated_package_loader(tmp_path: Path) -> None:
    package = tmp_path / "Synthetic.mlpackage"
    package.mkdir()
    (package / "literal.bin").write_bytes(b"test fixture")
    observed: list[Path] = []

    def package_loader(isolated_package: Path) -> object:
        observed.append(isolated_package)
        return _valid_coreml_spec()

    validator = getattr(coreml_artifact, "validate_coreml_package_interface", None)

    assert callable(validator)
    validator(package, package_loader=package_loader)
    assert len(observed) == 1
    assert observed[0] != package
    assert observed[0].name == package.name


def test_finding_2_exact_layout_requires_parity_report() -> None:
    assert "parity.json" in release_contracts._EVALUATION_ENTRIES


def test_finding_3_committed_rights_fixture_is_test_only() -> None:
    payload = json.loads(RIGHTS_FIXTURE.read_text(encoding="utf-8"))

    assert payload["purpose"] == "test"


def test_finding_3_production_verifier_rejects_test_purpose_rights(tmp_path: Path) -> None:
    from test_mobile_release import (
        _update_json,
        build_release_fixture,
        verify_mobile_release_directory,
    )

    release_dir = build_release_fixture(tmp_path)
    rights_path = release_dir / "rights-attestation.json"
    _update_json(rights_path, purpose="test")
    manifest_path = release_dir / "catalog" / "manifest.json"
    _update_json(
        manifest_path,
        rightsAttestationSha256=mobile_catalog.sha256_file(rights_path),
    )
    catalog_digest = mobile_catalog.sha256_file(manifest_path)
    for report_path in (release_dir / "evaluation").glob("*.json"):
        _update_json(report_path, catalogManifestSha256=catalog_digest)

    report = verify_mobile_release_directory(release_dir)

    gate = next(value for value in report.gates if value.name == "rights")
    assert gate.passed is False
    assert "purpose must be production" in gate.detail


def test_finding_4_catalog_contract_has_positive_ordered_app_build_bounds() -> None:
    release_fields = mobile_catalog.MobileCatalogRelease.__dataclass_fields__
    manifest_fields = mobile_catalog.MobileCatalogManifest.__dataclass_fields__

    assert "minimum_app_build" in release_fields
    assert "maximum_app_build" in release_fields
    assert "minimum_app_build" in manifest_fields
    assert "maximum_app_build" in manifest_fields


def test_finding_5_score_semantics_is_not_a_probability() -> None:
    assert mobile_catalog.SCORE_SEMANTICS == "uncalibrated_similarity_not_probability"


def test_finding_6_export_metadata_binds_all_source_artifacts_and_tools() -> None:
    metadata = coreml_artifact.build_export_metadata(
        model_sha256=DINOV2_ARTIFACT_SHA256["model.safetensors"],
        package_sha256="a" * 64,
        source_artifact_sha256=DINOV2_ARTIFACT_SHA256,
        tool_versions={
            "coremltools": "9.0",
            "macos": "26.5.1",
            "numpy": "2.2.6",
            "pillow": "12.3.0",
            "python": "3.11.15",
            "torch": "2.13.0",
            "transformers": "5.14.0",
            "xcode": "26.0.1 (17A400)",
        },
    )

    assert metadata.as_json_dict()["source_artifact_sha256"] == DINOV2_ARTIFACT_SHA256
    assert set(metadata.tool_versions) == {
        "coremltools",
        "macos",
        "numpy",
        "pillow",
        "python",
        "torch",
        "transformers",
        "xcode",
    }


def test_finding_7_numpy_header_is_bounded_before_array_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    oversized = evaluation / "parity-pytorch.npy"
    with oversized.open("wb") as stream:
        np.lib.format.write_array_header_1_0(
            stream,
            {
                "descr": np.dtype(np.float32).str,
                "fortran_order": False,
                "shape": (1_000_001, 384),
            },
        )
    (evaluation / "parity-coreml.npy").write_bytes(oversized.read_bytes())
    monkeypatch.setattr(
        release_contracts.np,
        "load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("np.load reached before header bounds")
        ),
    )

    evidence = release_contracts.inspect_embeddings(release_contracts.release_paths(tmp_path))

    assert evidence.shape_validation.passed is False
    assert "maximum sample rows" in evidence.shape_validation.detail


def test_finding_8_release_docs_define_production_evidence_and_handoff_semantics() -> None:
    root = Path(__file__).parent.parent
    documentation = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in ("README.md", "docs/mobile-model-card.md")
    )

    assert "evaluation/parity.json" in documentation
    assert "uncalibrated_similarity_not_probability" in documentation
    assert "minimumAppBuild" in documentation
    assert "maximumAppBuild" in documentation
    assert "purpose: production" in documentation


@pytest.mark.parametrize(
    ("updates", "detail"),
    (
        ({"unexpected": True}, "schema"),
        ({"evidencePurpose": "test"}, "evidencePurpose"),
        ({"provenanceUrl": "http://example.invalid/parity"}, "HTTPS"),
        ({"fixtureSetSha256": "invalid"}, "SHA256"),
        ({"sampleCount": 3}, "sampleCount"),
        ({"pytorchEmbeddingsSha256": "0" * 64}, "PyTorch array digest"),
    ),
)
def test_finding_2_and_3_parity_report_rejects_tampered_evidence(
    tmp_path: Path,
    updates: dict[str, object],
    detail: str,
) -> None:
    from test_mobile_release import (
        _update_json,
        build_release_fixture,
        verify_mobile_release_directory,
    )

    release_dir = build_release_fixture(tmp_path)
    _update_json(release_dir / "evaluation" / "parity.json", **updates)

    report = verify_mobile_release_directory(release_dir)

    gate = next(value for value in report.gates if value.name == "embedding_shape")
    assert gate.passed is False
    assert detail in gate.detail


@pytest.mark.parametrize(
    ("updates", "detail"),
    (
        ({"evidencePurpose": "test"}, "evidencePurpose"),
        ({"provenanceUrl": "http://example.invalid/evaluation"}, "HTTPS"),
        ({"fixtureSetSha256": "invalid"}, "SHA256"),
    ),
)
def test_finding_3_evaluation_reports_require_production_provenance(
    tmp_path: Path,
    updates: dict[str, object],
    detail: str,
) -> None:
    from test_mobile_release import (
        _update_json,
        build_release_fixture,
        verify_mobile_release_directory,
    )

    release_dir = build_release_fixture(tmp_path)
    _update_json(release_dir / "evaluation" / "closed-set.json", **updates)

    report = verify_mobile_release_directory(release_dir)

    gate = next(value for value in report.gates if value.name == "required_reports")
    assert gate.passed is False
    assert detail in gate.detail


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("minimum_app_build", True),
        ("minimum_app_build", 0),
        ("maximum_app_build", 0),
        ("minimum_app_build", 101),
    ),
)
def test_finding_4_catalog_rejects_invalid_app_build_ranges(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    from dataclasses import replace

    from test_mobile_catalog import release_fixture, rows_fixture

    release = replace(release_fixture(), **{field: value})
    with pytest.raises(ValueError, match="AppBuild"):
        mobile_catalog.write_mobile_catalog(
            tmp_path / "catalog",
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            rows_fixture(),
            release,
        )
