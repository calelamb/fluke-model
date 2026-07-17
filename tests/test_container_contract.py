"""Static deployment contract for offline weights and real readiness CI."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_container_bakes_verified_weights_and_forces_offline_runtime() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert dockerfile.startswith("FROM python:3.12-slim@sha256:")
    assert "scripts/fetch_model_artifact.py" in dockerfile
    assert "FLUKE_MODEL_ARTIFACT_DIR=/app/model-artifact" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile


def test_ci_builds_real_index_and_requires_ready_200() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "build_ci_reference_index.py" in workflow
    assert "curl --fail --silent http://127.0.0.1:4100/ready" in workflow
    assert 'http://127.0.0.1:4100/ready)" = "503"' not in workflow


def test_ci_proves_container_stays_unready_without_attested_catalog() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "Start fail-closed service without a catalog" in workflow
    assert 'test "$ready_status" = "503"' in workflow
    assert 'assert payload == {"status":"not_ready","reason":"index_unavailable"}' in workflow
