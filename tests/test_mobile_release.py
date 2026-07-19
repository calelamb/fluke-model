"""Release parity, open-set, provenance, and CLI gate contracts."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest

from fluke_model.coreml_artifact import package_tree_sha256
from fluke_model.mobile_catalog import (
    MobileCatalogRelease,
    ReferenceRow,
    sha256_file,
    write_mobile_catalog,
)
from fluke_model.mobile_release import (
    MobileReleaseEvidence,
    RELEASE_THRESHOLDS,
    ValidationEvidence,
    report_payload,
    validate_report_destination,
    verify_mobile_release,
    verify_mobile_release_directory,
    write_mobile_release_report,
)
from fluke_model.model_artifact import DINOV2_ARTIFACT_SHA256


RIGHTS_FIXTURE = Path(__file__).parent / "fixtures" / "mobile-catalog" / "rights-attestation.json"
MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
BOUNDARY_NAMES = (
    "input_paths",
    "package",
    "catalog",
    "digests",
    "rights",
    "embedding_shape",
    "embedding_norm",
    "required_reports",
)
OPEN_COHORTS = ("openSet", "nonOrca", "poorQuality", "occlusion", "distributionShift")


def _passed_boundaries() -> tuple[ValidationEvidence, ...]:
    return tuple(
        ValidationEvidence(name=name, passed=True, detail="synthetic test evidence")
        for name in BOUNDARY_NAMES
    )


def release_evidence_fixture(
    *,
    parity_cosine: float = 0.9994,
    top_1: float = 0.70,
    top_3: float = 0.84,
    false_accept: float = 0.04,
) -> MobileReleaseEvidence:
    return MobileReleaseEvidence(
        parity_cosine=parity_cosine,
        parity_sample_count=2,
        top_1=top_1,
        top_3=top_3,
        closed_set_sample_count=20,
        false_accept=false_accept,
        open_set_sample_count=100,
        validations=_passed_boundaries(),
    )


def _export_metadata(package_digest: str) -> dict[str, object]:
    return {
        "compute_precision": "FLOAT16",
        "input_shape": [1, 3, 224, 224],
        "minimum_deployment_target": "iOS17",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_sha256": DINOV2_ARTIFACT_SHA256["model.safetensors"],
        "output_shape": [1, 384],
        "package_sha256": package_digest,
        "preprocessing_version": "dinov2-imagenet-v1",
        "tool_versions": {
            "coremltools": "9.0",
            "numpy": "2.2.6",
            "python": "3.11.15",
            "torch": "2.13.0",
            "transformers": "5.14.0",
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _evaluation_report(
    *,
    evaluation_type: str,
    package_digest: str,
    catalog_digest: str,
    sample_count: int = 20,
    false_accept_rate: float = 0.04,
) -> dict[str, object]:
    common = {
        "schemaVersion": 1,
        "evaluationType": evaluation_type,
        "modelPackageSha256": package_digest,
        "catalogManifestSha256": catalog_digest,
        "sampleCount": sample_count,
    }
    if evaluation_type == "closedSetRetrieval":
        return {**common, "top1": 0.70, "top3": 0.84}
    return {**common, "falseAcceptRate": false_accept_rate}


def build_release_fixture(tmp_path: Path) -> Path:
    release_dir = tmp_path / "release"
    package = release_dir / "FlukeEmbedder.mlpackage"
    (package / "Data").mkdir(parents=True)
    (package / "Manifest.json").write_text("synthetic-package", encoding="utf-8")
    package_digest = package_tree_sha256(package)
    _write_json(release_dir / "export-metadata.json", _export_metadata(package_digest))
    rights = release_dir / "rights-attestation.json"
    rights.write_bytes(RIGHTS_FIXTURE.read_bytes())
    catalog_digest = _write_catalog_fixture(release_dir, rights, package_digest)
    _write_evaluation_fixture(release_dir, package_digest, catalog_digest)
    return release_dir


def _write_catalog_fixture(release_dir: Path, rights: Path, package_digest: str) -> str:
    catalog = release_dir / "catalog"
    reference = np.zeros((1, 384), dtype=np.float32)
    reference[0, 0] = 1.0
    write_mobile_catalog(
        catalog,
        reference,
        (
            ReferenceRow(
                "synthetic-ref-1",
                "synthetic-whale-1",
                "SYNTHETIC-1",
                "synthetic-owned-fixture",
            ),
        ),
        MobileCatalogRelease(
            manifest_version="synthetic-test",
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            model_version="synthetic-test",
            model_sha256=package_digest,
            preprocessing_version="dinov2-imagenet-v1",
            embedding_dimension=384,
            index_version="synthetic-test",
            score_semantics="cosineSimilarity",
            score_threshold=0.7,
            margin_threshold=0.1,
            rights_attestation_path=rights,
        ),
    )
    return sha256_file(catalog / "manifest.json")


def _write_evaluation_fixture(
    release_dir: Path, package_digest: str, catalog_digest: str
) -> None:
    pytorch = np.zeros((2, 384), dtype=np.float32)
    pytorch[0, 0] = 1.0
    pytorch[1, 1] = 1.0
    coreml = pytorch.copy()
    evaluation = release_dir / "evaluation"
    evaluation.mkdir()
    np.save(evaluation / "parity-pytorch.npy", pytorch, allow_pickle=False)
    np.save(evaluation / "parity-coreml.npy", coreml, allow_pickle=False)
    _write_json(
        evaluation / "closed-set.json",
        _evaluation_report(
            evaluation_type="closedSetRetrieval",
            package_digest=package_digest,
            catalog_digest=catalog_digest,
        ),
    )
    for cohort in OPEN_COHORTS:
        filename = {
            "openSet": "open-set.json",
            "nonOrca": "non-orca.json",
            "poorQuality": "poor-quality.json",
            "occlusion": "occlusion.json",
            "distributionShift": "distribution-shift.json",
        }[cohort]
        _write_json(
            evaluation / filename,
            _evaluation_report(
                evaluation_type=cohort,
                package_digest=package_digest,
                catalog_digest=catalog_digest,
            ),
        )


def test_release_requires_every_approved_gate() -> None:
    evidence = release_evidence_fixture(
        parity_cosine=0.9994,
        top_1=0.70,
        top_3=0.84,
        false_accept=0.04,
    )

    report = verify_mobile_release(evidence)

    assert report.ready is True
    assert all(gate.passed for gate in report.gates)


@pytest.mark.parametrize(
    ("field", "value"),
    (("parity_cosine", 0.9989), ("top_1", 0.649), ("top_3", 0.799), ("false_accept", 0.051)),
)
def test_release_fails_when_any_metric_misses(field: str, value: float) -> None:
    evidence = replace(release_evidence_fixture(), **{field: value})

    assert verify_mobile_release(evidence).ready is False


def test_exact_metric_thresholds_pass() -> None:
    evidence = release_evidence_fixture(
        parity_cosine=RELEASE_THRESHOLDS["parity_cosine"],
        top_1=RELEASE_THRESHOLDS["top_1"],
        top_3=RELEASE_THRESHOLDS["top_3"],
        false_accept=RELEASE_THRESHOLDS["false_accept"],
    )

    assert verify_mobile_release(evidence).ready is True


@pytest.mark.parametrize("field", ("parity_cosine", "top_1", "top_3", "false_accept"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_release_rejects_nonfinite_metrics(field: str, value: float) -> None:
    report = verify_mobile_release(replace(release_evidence_fixture(), **{field: value}))

    assert report.ready is False
    json.dumps(report_payload(report), allow_nan=False)


@pytest.mark.parametrize(
    "field",
    ("parity_sample_count", "closed_set_sample_count", "open_set_sample_count"),
)
@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_release_requires_positive_integer_sample_counts(field: str, value: object) -> None:
    report = verify_mobile_release(replace(release_evidence_fixture(), **{field: value}))

    assert report.ready is False


def test_release_records_are_immutable() -> None:
    evidence = release_evidence_fixture()
    report = verify_mobile_release(evidence)

    with pytest.raises(FrozenInstanceError):
        evidence.top_1 = 1.0
    with pytest.raises(FrozenInstanceError):
        report.ready = False
    with pytest.raises(FrozenInstanceError):
        report.gates[0].passed = False


def test_directory_verifier_accepts_complete_digest_bound_fixture(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is True
    assert all(gate.passed for gate in report.gates)
    assert {gate.name for gate in report.gates}.issuperset(BOUNDARY_NAMES)


def test_directory_report_is_byte_deterministic(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    report = verify_mobile_release_directory(release_dir)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_mobile_release_report(first, report)
    write_mobile_release_report(second, report)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8"))["ready"] is True


@pytest.mark.parametrize(
    "relative_path",
    (
        "FlukeEmbedder.mlpackage/report.json",
        "catalog/report.json",
        "evaluation/report.json",
    ),
)
def test_report_destination_cannot_mutate_verified_release_tree(
    tmp_path: Path, relative_path: str
) -> None:
    release_dir = build_release_fixture(tmp_path)

    with pytest.raises(ValueError, match="overlap"):
        validate_report_destination(release_dir, release_dir / relative_path)


@pytest.mark.parametrize(
    "relative_path",
    (
        "rights-attestation.json",
        "catalog/manifest.json",
        "evaluation/parity-pytorch.npy",
        "evaluation/closed-set.json",
        "evaluation/open-set.json",
        "evaluation/non-orca.json",
        "evaluation/poor-quality.json",
        "evaluation/occlusion.json",
        "evaluation/distribution-shift.json",
    ),
)
def test_directory_verifier_fails_closed_for_missing_inputs(
    tmp_path: Path, relative_path: str
) -> None:
    release_dir = build_release_fixture(tmp_path)
    (release_dir / relative_path).unlink()

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert any(relative_path in gate.detail for gate in report.gates if not gate.passed)


def test_directory_verifier_rejects_package_digest_mismatch(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    (release_dir / "FlukeEmbedder.mlpackage" / "Manifest.json").write_text(
        "tampered", encoding="utf-8"
    )

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert "digest" in next(g.detail for g in report.gates if g.name == "package")


def test_directory_verifier_rejects_catalog_digest_mismatch(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    (release_dir / "catalog" / "metadata.json").write_text("[]", encoding="utf-8")

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert next(g for g in report.gates if g.name == "digests").passed is False


def test_directory_verifier_rejects_rights_digest_or_permission_change(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    rights_path = release_dir / "rights-attestation.json"
    rights = json.loads(rights_path.read_text(encoding="utf-8"))
    rights["data_sources"][0]["redistribution_allowed"] = False
    _write_json(rights_path, rights)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert next(g for g in report.gates if g.name == "rights").passed is False


@pytest.mark.parametrize("kind", ("shape", "norm", "finite"))
def test_directory_verifier_rejects_invalid_parity_embeddings(tmp_path: Path, kind: str) -> None:
    release_dir = build_release_fixture(tmp_path)
    path = release_dir / "evaluation" / "parity-coreml.npy"
    values = np.load(path, allow_pickle=False)
    if kind == "shape":
        values = values[:, :-1]
    elif kind == "norm":
        values = values * 0.5
    else:
        values[0, 0] = np.nan
    np.save(path, values, allow_pickle=False)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert any(
        not next(g for g in report.gates if g.name == name).passed
        for name in ("embedding_shape", "embedding_norm", "parity_cosine")
    )


def test_directory_verifier_uses_worst_open_set_cohort(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    path = release_dir / "evaluation" / "poor-quality.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["falseAcceptRate"] = 0.051
    _write_json(path, payload)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    gate = next(g for g in report.gates if g.name == "false_accept")
    assert gate.observed == pytest.approx(0.051)


def test_directory_verifier_requires_exact_report_schema_and_provenance(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    path = release_dir / "evaluation" / "closed-set.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    payload["catalogManifestSha256"] = "0" * 64
    _write_json(path, payload)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert "schema" in next(g.detail for g in report.gates if g.name == "required_reports")


def test_directory_verifier_rejects_symlinked_input(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    report_path = release_dir / "evaluation" / "open-set.json"
    external = tmp_path / "external.json"
    report_path.replace(external)
    report_path.symlink_to(external)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert "symbolic link" in next(g.detail for g in report.gates if g.name == "input_paths")


def test_cli_fails_nonzero_and_writes_explicit_missing_gate_report(tmp_path: Path) -> None:
    release_dir = tmp_path / "incomplete"
    release_dir.mkdir()
    script = Path(__file__).parent.parent / "scripts" / "verify_mobile_release.py"

    result = subprocess.run(
        [sys.executable, str(script), "--release-dir", str(release_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    report_path = release_dir / "mobile-release-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result.returncode == 1
    assert report["ready"] is False
    assert "rights-attestation.json" in report_path.read_text(encoding="utf-8")
    assert "evaluation/closed-set.json" in report_path.read_text(encoding="utf-8")
    assert "evaluation/open-set.json" in report_path.read_text(encoding="utf-8")


def test_cli_returns_zero_only_for_complete_release(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    script = Path(__file__).parent.parent / "scripts" / "verify_mobile_release.py"

    result = subprocess.run(
        [sys.executable, str(script), "--release-dir", str(release_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout)["ready"] is True
