"""Release parity, open-set, provenance, and CLI gate contracts."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fluke_model.mobile_release as mobile_release_module

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
    _verify_mobile_release_directory_for_testing,
    write_mobile_release_report,
)
from fluke_model.mobile_release_evidence import (
    DecisionRecord,
    FixtureRow,
    canonical_decisions_payload,
    canonical_fixture_payload,
    fixture_set_sha256,
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
    "runtime_reexecution",
)
OPEN_COHORTS = ("openSet", "nonOrca", "poorQuality", "occlusion", "distributionShift")
SYNTHETIC_PACKAGE_SHA256 = "a" * 64
SYNTHETIC_CATALOG_SHA256 = "b" * 64
FIXTURE_SET_SHA256 = "c" * 64


def _valid_coreml_spec() -> object:
    input_type = SimpleNamespace(shape=[1, 3, 224, 224], dataType=65568)
    output_type = SimpleNamespace(shape=[1, 384], dataType=65568)
    return SimpleNamespace(
        description=SimpleNamespace(
            input=[SimpleNamespace(name="pixels", type=SimpleNamespace(multiArrayType=input_type))],
            output=[
                SimpleNamespace(name="embedding", type=SimpleNamespace(multiArrayType=output_type))
            ],
        )
    )


def verify_mobile_release_directory(release_dir: Path):
    """Validate synthetic test packages only at the package-loader boundary."""
    return _verify_mobile_release_directory_for_testing(
        release_dir,
        package_loader=lambda _isolated_package: _valid_coreml_spec(),
        runtime_validator=lambda _paths, _catalog: (True, "synthetic test runtime evidence"),
    )


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
        model_package_sha256=SYNTHETIC_PACKAGE_SHA256,
        catalog_manifest_sha256=SYNTHETIC_CATALOG_SHA256,
    )


def _export_metadata(package_digest: str) -> dict[str, object]:
    return {
        "compute_precision": "FLOAT16",
        "input_shape": [1, 3, 224, 224],
        "minimum_deployment_target": "iOS17",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_sha256": DINOV2_ARTIFACT_SHA256["model.safetensors"],
        "source_artifact_sha256": dict(DINOV2_ARTIFACT_SHA256),
        "output_shape": [1, 384],
        "package_sha256": package_digest,
        "preprocessing_version": "dinov2-imagenet-v1",
        "tool_versions": {
            "coremltools": "9.0",
            "macos": "26.5.1",
            "numpy": "2.2.6",
            "pillow": "12.3.0",
            "python": "3.11.15",
            "torch": "2.13.0",
            "transformers": "5.14.0",
            "xcode": "26.0.1 (17A400)",
        },
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _update_json(path: Path, **updates: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _write_json(path, {**payload, **updates})


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
        "evidencePurpose": "production",
        "provenanceUrl": "https://example.invalid/orcawatch/production-evaluation",
        "fixtureSetSha256": FIXTURE_SET_SHA256,
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
    source_model = release_dir / "source-model"
    source_model.mkdir()
    for filename in DINOV2_ARTIFACT_SHA256:
        (source_model / filename).write_text("synthetic test artifact", encoding="utf-8")
    package_digest = package_tree_sha256(package)
    _write_json(release_dir / "export-metadata.json", _export_metadata(package_digest))
    rights = release_dir / "rights-attestation.json"
    rights.write_bytes(RIGHTS_FIXTURE.read_bytes())
    _update_json(rights, purpose="production")
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
            model_version="dinov2-small-coreml-v1",
            model_sha256=package_digest,
            preprocessing_version="dinov2-imagenet-v1",
            embedding_dimension=384,
            index_version="mobile-reference-v1",
            minimum_app_build=1,
            maximum_app_build=100,
            score_semantics="uncalibrated_similarity_not_probability",
            score_threshold=0.7,
            margin_threshold=0.1,
            rights_attestation_path=rights,
        ),
    )
    return sha256_file(catalog / "manifest.json")


def _write_evaluation_fixture(release_dir: Path, package_digest: str, catalog_digest: str) -> None:
    pytorch = np.zeros((2, 384), dtype=np.float32)
    pytorch[0, 0] = 1.0
    pytorch[1, 1] = 1.0
    coreml = pytorch.copy()
    evaluation = release_dir / "evaluation"
    evaluation.mkdir()
    _write_json(
        evaluation / "evaluation-plan.json",
        {
            "schemaVersion": 1,
            "evidencePurpose": "production",
            "approvedBy": "Synthetic verifier contract fixture",
            "approvedAt": "2026-07-19T00:00:00+00:00",
            "provenanceUrl": "https://example.invalid/orcawatch/production-evaluation",
            "scoreThreshold": 0.7,
            "marginThreshold": 0.1,
            "runtimeVersions": {
                "coremltools": "9.0",
                "macos": "26.5.1",
                "numpy": "2.2.6",
                "pillow": "12.3.0",
                "python": "3.11.15",
                "torch": "2.13.0",
                "transformers": "5.14.0",
                "xcode": "26.0.1 (17A400)",
            },
            "cohortDefinitions": {
                name: "Synthetic verifier contract only"
                for name in ("parity", "closedSetRetrieval", *OPEN_COHORTS)
            },
        },
    )
    fixture_rows, decisions = _raw_evaluation_fixture()
    fixtures = evaluation / "fixtures"
    for row in fixture_rows:
        fixture = fixtures / row.relative_path
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_bytes(row.fixture_id.encode())
    fixture_digest = fixture_set_sha256(fixture_rows)
    (evaluation / "fixture-manifest.json").write_bytes(canonical_fixture_payload(fixture_rows))
    (evaluation / "decisions.json").write_bytes(
        canonical_decisions_payload(decisions, score_threshold=0.7, margin_threshold=0.1)
    )
    np.save(evaluation / "parity-pytorch.npy", pytorch, allow_pickle=False)
    np.save(evaluation / "parity-coreml.npy", coreml, allow_pickle=False)
    _write_json(
        evaluation / "parity.json",
        {
            "schemaVersion": 1,
            "evaluationType": "pytorchCoreMLParity",
            "evidencePurpose": "production",
            "provenanceUrl": "https://example.invalid/orcawatch/production-evaluation",
            "modelPackageSha256": package_digest,
            "catalogManifestSha256": catalog_digest,
            "sourceModelSha256": DINOV2_ARTIFACT_SHA256["model.safetensors"],
            "preprocessingVersion": "dinov2-imagenet-v1",
            "fixtureSetSha256": fixture_digest,
            "sampleCount": 2,
            "pytorchEmbeddingsSha256": sha256_file(evaluation / "parity-pytorch.npy"),
            "coremlEmbeddingsSha256": sha256_file(evaluation / "parity-coreml.npy"),
        },
    )
    _write_json(
        evaluation / "closed-set.json",
        _evaluation_report(
            evaluation_type="closedSetRetrieval",
            package_digest=package_digest,
            catalog_digest=catalog_digest,
            sample_count=100,
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
                sample_count=100,
            ),
        )

    for report_path in (
        evaluation / "closed-set.json",
        *(
            evaluation / name
            for name in (
                "open-set.json",
                "non-orca.json",
                "poor-quality.json",
                "occlusion.json",
                "distribution-shift.json",
            )
        ),
    ):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        _write_json(report_path, {**payload, "fixtureSetSha256": fixture_digest})


def _raw_evaluation_fixture() -> tuple[tuple[FixtureRow, ...], tuple[DecisionRecord, ...]]:
    rows: list[FixtureRow] = [
        FixtureRow(
            fixture_id="synthetic-reference-1",
            relative_path="images/synthetic-reference-1.jpg",
            image_sha256=hashlib.sha256(b"synthetic-reference-1").hexdigest(),
            roles=("reference",),
            reference_photo_id="synthetic-ref-1",
            whale_id="synthetic-whale-1",
            catalog_id="SYNTHETIC-1",
            source_id="synthetic-owned-fixture",
            path=Path("images/synthetic-reference-1.jpg"),
        )
    ]
    decisions: list[DecisionRecord] = []
    for index in range(2):
        rows.append(_fixture_row(f"parity-{index}", "parity", whale_id=None))
    for index in range(100):
        fixture_id = f"closed-{index:03d}"
        truth = f"truth-{index:03d}"
        if index < 70:
            ranking = (truth, "other-1", "other-2")
        elif index < 84:
            ranking = ("other-1", truth, "other-2")
        else:
            ranking = ("other-1", "other-2", "other-3")
        rows.append(_fixture_row(fixture_id, "closedSetRetrieval", whale_id=truth))
        decisions.append(
            DecisionRecord(fixture_id, "closedSetRetrieval", truth, ranking, 0.8, 0.6, True)
        )
    filenames = {
        "openSet": "open",
        "nonOrca": "non-orca",
        "poorQuality": "poor-quality",
        "occlusion": "occlusion",
        "distributionShift": "distribution-shift",
    }
    for cohort, prefix in filenames.items():
        for index in range(100):
            fixture_id = f"{prefix}-{index:03d}"
            accepted = index < 4
            top_score, second_score = (0.8, 0.6) if accepted else (0.55, 0.5)
            rows.append(_fixture_row(fixture_id, cohort, whale_id=None))
            decisions.append(
                DecisionRecord(
                    fixture_id,
                    cohort,
                    None,
                    ("whale-1", "whale-2"),
                    top_score,
                    second_score,
                    accepted,
                )
            )
    return tuple(rows), tuple(decisions)


def _fixture_row(fixture_id: str, role: str, *, whale_id: str | None) -> FixtureRow:
    return FixtureRow(
        fixture_id=fixture_id,
        relative_path=f"images/{fixture_id}.jpg",
        image_sha256=hashlib.sha256(fixture_id.encode()).hexdigest(),
        roles=(role,),
        reference_photo_id=None,
        whale_id=whale_id,
        catalog_id=None,
        source_id=None,
        path=Path(f"images/{fixture_id}.jpg"),
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


def test_pure_evidence_without_release_digests_cannot_be_ready() -> None:
    evidence = replace(
        release_evidence_fixture(),
        model_package_sha256=None,
        catalog_manifest_sha256=None,
    )

    report = verify_mobile_release(evidence)
    payload = report_payload(report)

    assert set(payload) == {
        "schemaVersion",
        "modelPackageSha256",
        "catalogManifestSha256",
        "ready",
        "thresholds",
        "gates",
    }
    assert payload["modelPackageSha256"] is None
    assert payload["catalogManifestSha256"] is None
    assert report.ready is False
    assert next(g for g in report.gates if g.name == "model_package_digest").passed is False
    assert next(g for g in report.gates if g.name == "catalog_manifest_digest").passed is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_package_sha256", "invalid"),
        ("model_package_sha256", "A" * 64),
        ("catalog_manifest_sha256", "invalid"),
        ("catalog_manifest_sha256", "B" * 64),
    ),
)
def test_pure_evidence_invalid_release_digest_fails_without_escaping(
    field: str, value: str
) -> None:
    evidence = replace(release_evidence_fixture(), **{field: value})

    report = verify_mobile_release(evidence)

    assert report.ready is False
    assert getattr(report, field) == value


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
    assert report.model_package_sha256 == package_tree_sha256(
        release_dir / "FlukeEmbedder.mlpackage"
    )
    assert report.catalog_manifest_sha256 == sha256_file(release_dir / "catalog" / "manifest.json")


def test_directory_verifier_rejects_symlinked_release_root(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path / "actual")
    alias = tmp_path / "release-alias"
    alias.symlink_to(release_dir, target_is_directory=True)

    report = verify_mobile_release_directory(alias)
    details = {gate.detail for gate in report.gates if gate.name in BOUNDARY_NAMES}

    assert report.ready is False
    assert details == {"release directory path contains a symbolic link component"}
    assert str(alias) not in json.dumps(report_payload(report), sort_keys=True)


def test_directory_verifier_rejects_symlinked_release_ancestor(tmp_path: Path) -> None:
    actual_parent = tmp_path / "actual-parent"
    release_dir = build_release_fixture(actual_parent)
    alias_parent = tmp_path / "parent-alias"
    alias_parent.symlink_to(actual_parent, target_is_directory=True)

    report = verify_mobile_release_directory(alias_parent / release_dir.name)
    details = {gate.detail for gate in report.gates if gate.name in BOUNDARY_NAMES}

    assert report.ready is False
    assert details == {"release directory path contains a symbolic link component"}
    assert str(alias_parent) not in json.dumps(report_payload(report), sort_keys=True)


def test_passing_report_is_identical_for_absolute_dot_and_relative_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release_dir = build_release_fixture(tmp_path)

    absolute = report_payload(verify_mobile_release_directory(release_dir))
    monkeypatch.chdir(release_dir)
    dot = report_payload(verify_mobile_release_directory(Path(".")))
    monkeypatch.chdir(tmp_path)
    relative = report_payload(verify_mobile_release_directory(Path(release_dir.name)))
    payloads = tuple(
        json.dumps(payload, sort_keys=True).encode() for payload in (absolute, dot, relative)
    )

    assert payloads[0] == payloads[1] == payloads[2]
    assert absolute["ready"] is True


def test_stale_copied_report_is_ignored_and_rebound_to_current_release(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    _write_json(
        release_dir / "mobile-release-report.json",
        {
            "schemaVersion": 1,
            "modelPackageSha256": "0" * 64,
            "catalogManifestSha256": "1" * 64,
            "ready": True,
            "thresholds": {},
            "gates": [],
        },
    )

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is True
    assert report.model_package_sha256 != "0" * 64
    assert report.catalog_manifest_sha256 != "1" * 64


@pytest.mark.parametrize("kind", ("directory", "symlink"))
def test_optional_prior_report_must_be_a_regular_file(tmp_path: Path, kind: str) -> None:
    release_dir = build_release_fixture(tmp_path)
    report_path = release_dir / "mobile-release-report.json"
    if kind == "directory":
        report_path.mkdir()
    else:
        external = tmp_path / "external-report.json"
        external.write_text("{}", encoding="utf-8")
        report_path.symlink_to(external)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert "optional report" in next(g.detail for g in report.gates if g.name == "input_paths")


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


@pytest.mark.parametrize("relative_path", ("custom.json", "nested/report.json"))
def test_in_release_report_destination_must_use_canonical_filename(
    tmp_path: Path, relative_path: str
) -> None:
    release_dir = build_release_fixture(tmp_path)

    with pytest.raises(ValueError, match="mobile-release-report.json"):
        validate_report_destination(release_dir, release_dir / relative_path)


def test_cli_rejects_in_release_custom_report_without_poisoning_next_run(
    tmp_path: Path,
) -> None:
    release_dir = build_release_fixture(tmp_path)
    script = Path(__file__).parent.parent / "scripts" / "verify_mobile_release.py"
    custom = release_dir / "custom.json"

    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--release-dir",
            str(release_dir),
            "--report",
            str(custom),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    first_default = subprocess.run(
        [sys.executable, str(script), "--release-dir", str(release_dir)],
        check=False,
        capture_output=True,
        text=True,
    )
    second_default = subprocess.run(
        [sys.executable, str(script), "--release-dir", str(release_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert rejected.returncode == 2
    assert not custom.exists()
    assert first_default.returncode == 1
    assert second_default.returncode == 1
    assert (release_dir / "mobile-release-report.json").is_file()


def test_failed_report_details_are_root_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = tmp_path / "first-root"
    second = tmp_path / "second-root"
    shutil.copytree(source, first)
    shutil.copytree(source, second)

    def fail_catalog(paths: object, _package: object) -> object:
        catalog_manifest = getattr(paths, "catalog_manifest")
        raise OSError(f"cannot read malformed input {catalog_manifest}")

    monkeypatch.setattr(mobile_release_module, "_inspect_catalog", fail_catalog)

    first_payload = report_payload(verify_mobile_release_directory(first))
    second_payload = report_payload(verify_mobile_release_directory(second))
    first_bytes = json.dumps(first_payload, sort_keys=True).encode()
    second_bytes = json.dumps(second_payload, sort_keys=True).encode()

    assert first_bytes == second_bytes
    assert str(first) not in first_bytes.decode()
    assert str(second) not in second_bytes.decode()
    assert "<release-dir>" in first_bytes.decode()


@pytest.mark.parametrize("candidate_name", ("a", "release"))
def test_failed_report_normalization_canonicalizes_relative_release_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_name: str,
) -> None:
    candidate = tmp_path / candidate_name
    candidate.mkdir()

    def fail_catalog(paths: object, _package: object) -> object:
        manifest = getattr(paths, "catalog_manifest")
        raise OSError(f"catalog data failure in metadata.json at {manifest}")

    monkeypatch.setattr(mobile_release_module, "_inspect_catalog", fail_catalog)

    absolute = report_payload(verify_mobile_release_directory(candidate))
    monkeypatch.chdir(candidate)
    dot = report_payload(verify_mobile_release_directory(Path(".")))
    monkeypatch.chdir(tmp_path)
    relative = report_payload(verify_mobile_release_directory(Path(candidate_name)))
    payloads = tuple(
        json.dumps(payload, sort_keys=True).encode() for payload in (absolute, dot, relative)
    )

    assert payloads[0] == payloads[1] == payloads[2]
    decoded = payloads[0].decode()
    assert "catalog data failure in metadata.json" in decoded
    assert "<release-dir>/catalog/manifest.json" in decoded
    assert str(candidate) not in decoded


@pytest.mark.parametrize(
    "relative_path",
    (
        "rights-attestation.json",
        "catalog/manifest.json",
        "evaluation/parity-pytorch.npy",
        "evaluation/parity-coreml.npy",
        "evaluation/parity.json",
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


@pytest.mark.parametrize("value", (True, False))
@pytest.mark.parametrize("shape_name", ("input_shape", "output_shape"))
def test_directory_verifier_rejects_boolean_export_shapes(
    tmp_path: Path, shape_name: str, value: bool
) -> None:
    release_dir = build_release_fixture(tmp_path)
    path = release_dir / "export-metadata.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[shape_name] = [value, *payload[shape_name][1:]]
    _write_json(path, payload)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert next(g for g in report.gates if g.name == "package").passed is False


@pytest.mark.parametrize("value", (True, False))
@pytest.mark.parametrize(
    "field",
    (
        "schemaVersion",
        "embeddingDimension",
        "referenceCount",
        "catalogCount",
        "scoreThreshold",
        "marginThreshold",
    ),
)
def test_directory_verifier_rejects_boolean_catalog_numeric_fields(
    tmp_path: Path, field: str, value: bool
) -> None:
    release_dir = build_release_fixture(tmp_path)
    _update_json(release_dir / "catalog" / "manifest.json", **{field: value})

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert next(g for g in report.gates if g.name == "catalog").passed is False


@pytest.mark.parametrize("value", (True, False))
@pytest.mark.parametrize(
    ("filename", "field"),
    (
        ("closed-set.json", "schemaVersion"),
        ("closed-set.json", "sampleCount"),
        ("closed-set.json", "top1"),
        ("closed-set.json", "top3"),
        ("open-set.json", "schemaVersion"),
        ("open-set.json", "sampleCount"),
        ("open-set.json", "falseAcceptRate"),
    ),
)
def test_directory_verifier_rejects_boolean_evaluation_numeric_fields(
    tmp_path: Path, filename: str, field: str, value: bool
) -> None:
    release_dir = build_release_fixture(tmp_path)
    _update_json(release_dir / "evaluation" / filename, **{field: value})

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert next(g for g in report.gates if g.name == "required_reports").passed is False


def _rebind_evaluation_catalog_digest(release_dir: Path) -> None:
    digest = sha256_file(release_dir / "catalog" / "manifest.json")
    for path in (release_dir / "evaluation").glob("*.json"):
        _update_json(path, catalogManifestSha256=digest)


def test_directory_verifier_rejects_self_consistent_unnormalized_catalog(
    tmp_path: Path,
) -> None:
    release_dir = build_release_fixture(tmp_path)
    vector_path = release_dir / "catalog" / "references.f16"
    vectors = np.fromfile(vector_path, dtype="<f2")
    vector_path.write_bytes((vectors * 0.5).astype("<f2").tobytes())
    _update_json(
        release_dir / "catalog" / "manifest.json",
        vectorsSha256=sha256_file(vector_path),
    )
    _rebind_evaluation_catalog_digest(release_dir)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert "normalized" in next(g.detail for g in report.gates if g.name == "catalog")


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
