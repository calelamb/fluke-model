"""Contract tests for the rights-gated on-device reference catalog."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fluke_model.mobile_catalog import (
    MobileCatalogManifest,
    MobileCatalogRelease,
    ReferenceRow,
    manifest_payload,
    sha256_file,
    validate_embeddings,
    write_mobile_catalog,
)
from fluke_model.rights import RightsError


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "mobile-catalog"
RIGHTS_FIXTURE = FIXTURE_DIR / "rights-attestation.json"
MODEL_SHA256 = "a" * 64
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
MANIFEST_KEYS = {
    "schemaVersion",
    "manifestVersion",
    "modelId",
    "modelRevision",
    "modelVersion",
    "modelSha256",
    "preprocessingVersion",
    "embeddingDimension",
    "dtype",
    "indexVersion",
    "referenceCount",
    "catalogCount",
    "vectorsSha256",
    "metadataSha256",
    "rightsAttestationSha256",
    "scoreSemantics",
    "scoreThreshold",
    "marginThreshold",
}
METADATA_KEYS = {"referencePhotoId", "whaleId", "catalogId", "sourceId"}


def release_fixture(
    *,
    rights_path: Path = RIGHTS_FIXTURE,
    embedding_dimension: int = 3,
    score_threshold: float = 0.72,
    margin_threshold: float = 0.08,
) -> MobileCatalogRelease:
    return MobileCatalogRelease(
        manifest_version="2026-07-18",
        model_id="facebook/dinov2-small",
        model_revision=MODEL_REVISION,
        model_version="dinov2-small-coreml-v1",
        model_sha256=MODEL_SHA256,
        preprocessing_version="dinov2-imagenet-v1",
        embedding_dimension=embedding_dimension,
        index_version="mobile-reference-v1",
        score_semantics="cosineSimilarity",
        score_threshold=score_threshold,
        margin_threshold=margin_threshold,
        rights_attestation_path=rights_path,
    )


def rows_fixture() -> tuple[ReferenceRow, ...]:
    return (
        ReferenceRow("ref-1", "whale-1", "J35", "synthetic-owned-fixture"),
    )


def _rights_payload() -> dict[str, Any]:
    return json.loads(RIGHTS_FIXTURE.read_text(encoding="utf-8"))


def _with_source_updates(payload: dict[str, Any], **updates: object) -> dict[str, Any]:
    first, *remaining = payload["data_sources"]
    return {
        **payload,
        "data_sources": [{**first, **updates}, *(dict(source) for source in remaining)],
    }


def _write_rights(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "rights.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_catalog_bytes_are_row_major_float16_and_hashed(tmp_path: Path) -> None:
    rows = rows_fixture()
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    manifest = write_mobile_catalog(tmp_path, embeddings, rows, release_fixture())

    assert (tmp_path / "references.f16").read_bytes() == embeddings.astype("<f2").tobytes()
    assert manifest.reference_count == 1
    assert manifest.catalog_count == 1
    assert manifest.vectors_sha256 == sha256_file(tmp_path / "references.f16")
    assert manifest.rights_attestation_sha256 == sha256_file(RIGHTS_FIXTURE)


def test_catalog_uses_exact_camel_case_client_schema(tmp_path: Path) -> None:
    manifest = write_mobile_catalog(
        tmp_path,
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        rows_fixture(),
        release_fixture(),
    )

    raw_manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    raw_metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert set(raw_manifest) == MANIFEST_KEYS
    assert raw_manifest == manifest_payload(manifest)
    assert raw_manifest["scoreSemantics"] == "cosineSimilarity"
    assert len(raw_metadata) == 1
    assert set(raw_metadata[0]) == METADATA_KEYS
    assert raw_metadata[0] == {
        "referencePhotoId": "ref-1",
        "whaleId": "whale-1",
        "catalogId": "J35",
        "sourceId": "synthetic-owned-fixture",
    }
    assert manifest.metadata_sha256 == sha256_file(tmp_path / "metadata.json")


def test_catalog_sorts_rows_and_corresponding_vectors_deterministically(tmp_path: Path) -> None:
    rows = (
        ReferenceRow("ref-z", "whale-z", "Z", "synthetic-owned-fixture"),
        ReferenceRow("ref-a", "whale-a", "A", "synthetic-owned-fixture"),
    )
    embeddings = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_mobile_catalog(first, embeddings, rows, release_fixture())
    write_mobile_catalog(second, embeddings[::-1], rows[::-1], release_fixture())

    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    assert (first / "metadata.json").read_bytes() == (second / "metadata.json").read_bytes()
    assert (first / "references.f16").read_bytes() == (second / "references.f16").read_bytes()
    metadata = json.loads((first / "metadata.json").read_text(encoding="utf-8"))
    assert [row["referencePhotoId"] for row in metadata] == ["ref-a", "ref-z"]


def test_catalog_records_are_immutable() -> None:
    row = rows_fixture()[0]
    release = release_fixture()

    with pytest.raises(FrozenInstanceError):
        row.reference_photo_id = "replacement"
    with pytest.raises(FrozenInstanceError):
        release.model_version = "replacement"


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.ones((3,), dtype=np.float32), "shape"),
        (np.ones((1, 2), dtype=np.float32), "shape"),
        (np.array([[np.nan, 0.0, 0.0]], dtype=np.float32), "finite"),
        (np.array([[1.0, 1.0, 0.0]], dtype=np.float32), "L2 normalized"),
    ],
)
def test_embedding_validation_fails_closed(values: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_embeddings(values, expected_dimension=3)


def test_embedding_validation_returns_an_owned_contiguous_copy() -> None:
    source = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    validated = validate_embeddings(source, expected_dimension=3)
    validated[0, 0] = 0.0

    assert validated.dtype == np.float32
    assert validated.flags.c_contiguous
    assert source[0, 0] == 1.0


def test_embedding_validation_requires_an_integer_dimension() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_embeddings(np.array([[1.0, 0.0, 0.0]]), expected_dimension=3.0)


def test_catalog_fails_when_reference_rights_are_absent(tmp_path: Path) -> None:
    rows = (ReferenceRow("ref-1", "whale-1", "J35", "missing"),)
    with pytest.raises(RightsError, match="not covered"):
        write_mobile_catalog(
            tmp_path,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            rows,
            release_fixture(),
        )
    assert tuple(tmp_path.iterdir()) == ()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("redistribution_allowed", "redistribution"),
        ("mobile_ml_use_allowed", "mobile ML"),
        ("commercial_use_allowed", "commercial production"),
    ],
)
def test_catalog_requires_explicit_source_permissions(
    tmp_path: Path, field: str, message: str
) -> None:
    rights_path = _write_rights(
        tmp_path,
        _with_source_updates(_rights_payload(), **{field: False}),
    )
    output = tmp_path / "catalog"

    with pytest.raises(RightsError, match=message):
        write_mobile_catalog(
            output,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            rows_fixture(),
            release_fixture(rights_path=rights_path),
        )
    assert not output.exists()


def test_catalog_requires_exact_rights_source_coverage(tmp_path: Path) -> None:
    payload = _rights_payload()
    extra = {**payload["data_sources"][0], "source_id": "unused-source"}
    rights_path = _write_rights(
        tmp_path,
        {**payload, "data_sources": [*(dict(row) for row in payload["data_sources"]), extra]},
    )

    with pytest.raises(RightsError, match="unused source"):
        write_mobile_catalog(
            tmp_path / "catalog",
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            rows_fixture(),
            release_fixture(rights_path=rights_path),
        )


def test_catalog_requires_https_rights_evidence(tmp_path: Path) -> None:
    rights_path = _write_rights(
        tmp_path,
        _with_source_updates(
            _rights_payload(), evidence_url="http://example.invalid/rights"
        ),
    )

    with pytest.raises(RightsError, match="evidence"):
        write_mobile_catalog(
            tmp_path / "catalog",
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            rows_fixture(),
            release_fixture(rights_path=rights_path),
        )


@pytest.mark.parametrize(
    "rows",
    [
        (
            ReferenceRow("duplicate", "whale-1", "A", "synthetic-owned-fixture"),
            ReferenceRow("duplicate", "whale-2", "B", "synthetic-owned-fixture"),
        ),
        (
            ReferenceRow("ref-1", "whale-1", "A", "synthetic-owned-fixture"),
            ReferenceRow("ref-2", "whale-1", "B", "synthetic-owned-fixture"),
        ),
        (
            ReferenceRow("ref-1", "whale-1", "A", "synthetic-owned-fixture"),
            ReferenceRow("ref-2", "whale-2", "A", "synthetic-owned-fixture"),
        ),
    ],
)
def test_catalog_rejects_duplicate_or_unstable_identity_rows(
    tmp_path: Path, rows: tuple[ReferenceRow, ...]
) -> None:
    embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="referencePhotoId|stable"):
        write_mobile_catalog(tmp_path, embeddings, rows, release_fixture())


@pytest.mark.parametrize(
    "release",
    [
        release_fixture(score_threshold=float("nan")),
        release_fixture(score_threshold=1.01),
        release_fixture(margin_threshold=-1.01),
        replace(release_fixture(), embedding_dimension=0),
        replace(release_fixture(), embedding_dimension=3.0),
        replace(release_fixture(), model_sha256="not-a-sha"),
        replace(release_fixture(), model_sha256=123),
        replace(release_fixture(), score_semantics="cosine_similarity_not_probability"),
    ],
)
def test_release_contract_rejects_invalid_counts_thresholds_and_hashes(
    tmp_path: Path, release: MobileCatalogRelease
) -> None:
    with pytest.raises(ValueError):
        write_mobile_catalog(
            tmp_path,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            rows_fixture(),
            release,
        )


def test_catalog_rejects_row_count_mismatch_without_partial_output(tmp_path: Path) -> None:
    output = tmp_path / "catalog"
    with pytest.raises(ValueError, match="row count"):
        write_mobile_catalog(
            output,
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            rows_fixture(),
            release_fixture(),
        )
    assert not output.exists()


def test_catalog_refuses_to_replace_published_output(tmp_path: Path) -> None:
    output = tmp_path / "catalog"
    output.mkdir()
    marker = output / "published.txt"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="empty"):
        write_mobile_catalog(
            output,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            rows_fixture(),
            release_fixture(),
        )
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_library_rejects_symlinked_output_ancestor_without_outside_mutation(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-output-parent"
    real_parent.mkdir()
    marker = real_parent / "outside.txt"
    marker.write_text("preserve", encoding="utf-8")
    linked_parent = tmp_path / "linked-output-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    output = linked_parent / "catalog"
    before = {path.name: path.read_bytes() for path in real_parent.iterdir() if path.is_file()}

    with pytest.raises(ValueError, match="symbolic link component"):
        write_mobile_catalog(
            output,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            rows_fixture(),
            release_fixture(),
        )

    after = {path.name: path.read_bytes() for path in real_parent.iterdir() if path.is_file()}
    assert after == before
    assert not output.exists()


def test_library_rejects_symlinked_rights_ancestor_without_input_mutation(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-rights-parent"
    real_parent.mkdir()
    real_rights = _write_rights(real_parent, _rights_payload())
    linked_parent = tmp_path / "linked-rights-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_rights = linked_parent / real_rights.name
    before = real_rights.read_bytes()
    output = tmp_path / "catalog"

    with pytest.raises(ValueError, match="symbolic link component"):
        write_mobile_catalog(
            output,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            rows_fixture(),
            release_fixture(rights_path=linked_rights),
        )

    assert real_rights.read_bytes() == before
    assert not output.exists()


@pytest.mark.parametrize("relation", ("equal", "output-contains-rights", "rights-contains-output"))
def test_library_rejects_rights_output_overlap_without_partial_output(
    tmp_path: Path, relation: str
) -> None:
    rights_parent = tmp_path / "rights-parent"
    rights_parent.mkdir()
    rights_path = _write_rights(rights_parent, _rights_payload())
    if relation == "equal":
        output = rights_path
    elif relation == "output-contains-rights":
        output = rights_parent
    else:
        output = rights_path / "catalog"
    before = rights_path.read_bytes()

    with pytest.raises(ValueError, match="overlap"):
        write_mobile_catalog(
            output,
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            rows_fixture(),
            release_fixture(rights_path=rights_path),
        )

    assert rights_path.read_bytes() == before
    assert not (rights_parent / "manifest.json").exists()
    assert not (rights_parent / "references.f16").exists()
    assert not (rights_parent / "metadata.json").exists()


def test_manifest_payload_does_not_expose_dataclass_field_names() -> None:
    manifest = MobileCatalogManifest(
        schema_version=1,
        manifest_version="v1",
        model_id="model",
        model_revision="revision",
        model_version="version",
        model_sha256=MODEL_SHA256,
        preprocessing_version="preprocessing",
        embedding_dimension=3,
        dtype="float16",
        index_version="index",
        reference_count=1,
        catalog_count=1,
        vectors_sha256="b" * 64,
        metadata_sha256="c" * 64,
        rights_attestation_sha256="d" * 64,
        score_semantics="cosineSimilarity",
        score_threshold=0.7,
        margin_threshold=0.1,
    )

    assert set(manifest_payload(manifest)) == MANIFEST_KEYS
    assert "schema_version" not in manifest_payload(manifest)
