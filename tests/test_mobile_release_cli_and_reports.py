"""Release report, cohort, layout, and production CLI boundary tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_mobile_release import (
    _update_json,
    build_release_fixture,
    verify_mobile_release_directory,
)


def test_directory_verifier_uses_worst_open_set_cohort(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    path = release_dir / "evaluation" / "poor-quality.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["falseAcceptRate"] = 0.06
    _update_json(path, **payload)
    decisions_path = release_dir / "evaluation" / "decisions.json"
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    for record in decisions["records"]:
        if record["fixtureId"] in {"poor-quality-004", "poor-quality-005"}:
            record.update({"accepted": True, "topScore": 0.8, "secondScore": 0.6})
    decisions_path.write_text(
        json.dumps(decisions, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    gate = next(g for g in report.gates if g.name == "false_accept")
    assert gate.observed == pytest.approx(0.06)


def test_directory_verifier_rejects_report_metric_not_supported_by_raw_decisions(
    tmp_path: Path,
) -> None:
    release_dir = build_release_fixture(tmp_path)
    _update_json(release_dir / "evaluation" / "closed-set.json", top1=0.99)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    detail = next(g.detail for g in report.gates if g.name == "required_reports")
    assert "do not match raw decisions" in detail


def test_directory_verifier_rejects_fixture_manifest_tampering(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    path = release_dir / "evaluation" / "fixture-manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["imageSha256"] = "e" * 64
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert "fixtureSetSha256" in next(
        g.detail for g in report.gates if g.name == "required_reports"
    )


def test_directory_verifier_binds_reports_to_approved_evaluation_plan(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    _update_json(
        release_dir / "evaluation" / "evaluation-plan.json",
        provenanceUrl="https://example.invalid/different-approved-plan",
    )

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert "approved evaluation plan" in next(
        g.detail for g in report.gates if g.name == "required_reports"
    )


def test_directory_verifier_requires_exact_report_schema_and_provenance(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    path = release_dir / "evaluation" / "closed-set.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    payload["catalogManifestSha256"] = "0" * 64
    _update_json(path, **payload)

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


@pytest.mark.parametrize(
    "relative_path",
    ("unexpected.txt", "catalog/stale.json", "evaluation/old-results.json"),
)
def test_directory_verifier_rejects_extra_layout_entries(
    tmp_path: Path, relative_path: str
) -> None:
    release_dir = build_release_fixture(tmp_path)
    extra = release_dir / relative_path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text("stale", encoding="utf-8")

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert "exact" in next(g.detail for g in report.gates if g.name == "input_paths")


@pytest.mark.parametrize("content", (b"", b"\x93NUMPY"))
def test_directory_verifier_bounds_corrupt_or_truncated_numpy(
    tmp_path: Path, content: bytes
) -> None:
    release_dir = build_release_fixture(tmp_path)
    (release_dir / "evaluation" / "parity-coreml.npy").write_bytes(content)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False
    assert report.model_package_sha256 is not None


def test_directory_verifier_bounds_huge_integer_metrics(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    _update_json(release_dir / "evaluation" / "closed-set.json", top1=10**1000)

    report = verify_mobile_release_directory(release_dir)

    assert report.ready is False


def test_cli_bounds_verification_failure_and_writes_null_digest_report(tmp_path: Path) -> None:
    release_dir = tmp_path / "invalid"
    release_dir.mkdir()
    (release_dir / "evaluation").mkdir()
    (release_dir / "evaluation" / "closed-set.json").write_text(
        '{"top1":' + "9" * 5000 + "}", encoding="utf-8"
    )
    script = Path(__file__).parent.parent / "scripts" / "verify_mobile_release.py"

    result = subprocess.run(
        [sys.executable, str(script), "--release-dir", str(release_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads((release_dir / "mobile-release-report.json").read_text(encoding="utf-8"))
    assert result.returncode != 0
    assert payload["ready"] is False
    assert payload["modelPackageSha256"] is None
    assert payload["catalogManifestSha256"] is None


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


def test_cli_rejects_literal_synthetic_package_bytes_by_default(tmp_path: Path) -> None:
    release_dir = build_release_fixture(tmp_path)
    script = Path(__file__).parent.parent / "scripts" / "verify_mobile_release.py"

    result = subprocess.run(
        [sys.executable, str(script), "--release-dir", str(release_dir)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ready"] is False
    package_gate = next(gate for gate in payload["gates"] if gate["name"] == "package")
    assert package_gate["passed"] is False
