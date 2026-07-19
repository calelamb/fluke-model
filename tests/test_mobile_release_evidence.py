"""Canonical corpus and raw mobile release decision contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from fluke_model.mobile_release_evidence import (
    DecisionRecord,
    FixtureRow,
    canonical_fixture_payload,
    fixture_set_sha256,
    load_corpus_manifest,
    recompute_metrics,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manifest(relative_path: str, digest: str) -> dict[str, object]:
    return {
        "schemaVersion": 1,
        "evidencePurpose": "test",
        "provenanceUrl": "https://example.invalid/evaluation/test-fixtures",
        "rows": [
            {
                "fixtureId": "fixture-1",
                "relativePath": relative_path,
                "imageSha256": digest,
                "roles": ["parity", "reference"],
                "referencePhotoId": "reference-1",
                "whaleId": "whale-1",
                "catalogId": "catalog-1",
                "sourceId": "synthetic-owned-fixture",
            }
        ],
    }


def test_corpus_manifest_hashes_canonical_manifest_and_actual_image_bytes(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    image = corpus / "one.jpg"
    image.write_bytes(b"synthetic-image")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(_manifest("one.jpg", _sha(b"synthetic-image"))), encoding="utf-8"
    )

    manifest = load_corpus_manifest(manifest_path, corpus)

    assert manifest.purpose == "test"
    assert manifest.rows[0].path == image
    assert (
        fixture_set_sha256(manifest.rows)
        == hashlib.sha256(canonical_fixture_payload(manifest.rows)).hexdigest()
    )


@pytest.mark.parametrize("relative_path", ("../escape.jpg", "/tmp/absolute.jpg", "nested/../x.jpg"))
def test_corpus_manifest_rejects_noncanonical_or_traversing_paths(
    tmp_path: Path, relative_path: str
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest(relative_path, "0" * 64)), encoding="utf-8")

    with pytest.raises(ValueError, match="relativePath"):
        load_corpus_manifest(manifest_path, corpus)


def test_corpus_manifest_rejects_symlinked_images(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    external = tmp_path / "external.jpg"
    external.write_bytes(b"synthetic")
    (corpus / "one.jpg").symlink_to(external)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest("one.jpg", _sha(b"synthetic"))), encoding="utf-8")

    with pytest.raises(ValueError, match="symbolic link"):
        load_corpus_manifest(manifest_path, corpus)


def test_corpus_manifest_rejects_duplicate_fixture_and_path(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.jpg").write_bytes(b"synthetic")
    payload = _manifest("one.jpg", _sha(b"synthetic"))
    payload["rows"] = [*payload["rows"], dict(payload["rows"][0])]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        load_corpus_manifest(manifest_path, corpus)


def test_recompute_metrics_uses_rank_and_exact_score_margin_acceptance() -> None:
    decisions = (
        DecisionRecord("closed-1", "closedSetRetrieval", "w1", ("w1", "w2"), 0.9, 0.5, True),
        DecisionRecord("closed-2", "closedSetRetrieval", "w2", ("w1", "w2"), 0.8, 0.7, False),
        DecisionRecord("open-1", "openSet", None, ("w1", "w2"), 0.8, 0.7, False),
        DecisionRecord("open-2", "openSet", None, ("w1", "w2"), 0.8, 0.6, True),
    )

    metrics = recompute_metrics(decisions, score_threshold=0.75, margin_threshold=0.15)

    assert metrics["closedSetRetrieval"] == {"sampleCount": 2, "top1": 0.5, "top3": 1.0}
    assert metrics["openSet"] == {"sampleCount": 2, "falseAcceptRate": 0.5}


def test_fixture_rows_are_immutable() -> None:
    row = FixtureRow(
        fixture_id="x",
        relative_path="x.jpg",
        image_sha256="0" * 64,
        roles=("parity",),
        reference_photo_id=None,
        whale_id=None,
        catalog_id=None,
        source_id=None,
        path=Path("x.jpg"),
    )

    with pytest.raises(AttributeError):
        row.fixture_id = "changed"
