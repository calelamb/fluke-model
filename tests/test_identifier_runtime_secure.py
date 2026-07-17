"""Integration tests for rights-gated atomic reference indexes."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fluke_model.embedders import LoadedEmbedder
from fluke_model import identify_runtime as runtime_module
from fluke_model.identify_runtime import (
    DINO_V2_MODEL_ID,
    DINO_V2_REVISION,
    IdentifierRuntime,
    ReferencePhoto,
    build_reference_index,
)
from fluke_model.index_store import AtomicIndexStore
from fluke_model.rights import DataRights, ModelRights, RightsAttestation


def _embedder() -> LoadedEmbedder:
    def embed(images: list[Image.Image]) -> np.ndarray:
        rows = [[float(image.getpixel((0, 0))[0]), 1.0] for image in images]
        values = np.asarray(rows, dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    return LoadedEmbedder(embed_fn=embed, embed_dim=2, name="dinov2-small")


def _rights() -> RightsAttestation:
    return RightsAttestation(
        schema_version=1,
        approved_by="Launch owner",
        approved_at=datetime.now(timezone.utc),
        commercial_use_allowed=True,
        model=ModelRights(
            model_id=DINO_V2_MODEL_ID,
            revision=DINO_V2_REVISION,
            license_spdx="Apache-2.0",
            evidence_url="https://github.com/facebookresearch/dinov2/blob/main/LICENSE",
            commercial_use_allowed=True,
        ),
        data_sources=(
            DataRights(
                source_id="owner-grant-1",
                license_or_permission="written permission",
                evidence_url="https://fluke.example/grants/1",
                commercial_use_allowed=True,
            ),
        ),
    )


def _references() -> list[ReferencePhoto]:
    return [
        ReferencePhoto(
            "ref-1", "J35", "Tahlequah", "https://images.example/ref-1.jpg", "owner-grant-1"
        ),
        ReferencePhoto("ref-2", "J36", "Alki", "https://images.example/ref-2.jpg", "owner-grant-1"),
    ]


def test_rebuild_publishes_rights_gated_index_and_runtime_is_ready(tmp_path: Path) -> None:
    store = AtomicIndexStore(tmp_path / "indexes")
    result = build_reference_index(
        _references(),
        store=store,
        rights=_rights(),
        embedder=_embedder(),
        image_loader=lambda ref: Image.new(
            "RGB", (8, 8), color=(20 if ref.catalog_id == "J35" else 200, 0, 0)
        ),
    )
    runtime = IdentifierRuntime(index_store=store, embedder=_embedder())

    assert result["ok"] is True
    assert runtime.readiness() == (True, "ready")
    identified = runtime.identify(Image.new("RGB", (8, 8), color=(20, 0, 0)))
    assert identified["matches"][0]["catalogId"] == "J35"
    assert identified["confidenceBand"] == "unavailable"
    assert identified["confidenceSemantics"] == "uncalibrated_similarity_not_probability"


def test_failed_rebuild_does_not_replace_last_published_index(tmp_path: Path) -> None:
    store = AtomicIndexStore(tmp_path / "indexes")
    first = build_reference_index(
        _references(),
        store=store,
        rights=_rights(),
        embedder=_embedder(),
        image_loader=lambda ref: Image.new("RGB", (8, 8), color=(20, 0, 0)),
    )
    current = store.current_version_dir()

    with pytest.raises(ValueError, match="All reference photos"):
        build_reference_index(
            _references(),
            store=store,
            rights=_rights(),
            embedder=_embedder(),
            image_loader=lambda ref: (_ for _ in ()).throw(OSError("download failed")),
        )

    assert store.current_version_dir() == current
    assert first["indexVersion"] == current.name


def test_partial_rebuild_does_not_publish_a_degraded_index(tmp_path: Path) -> None:
    store = AtomicIndexStore(tmp_path / "indexes")
    first = build_reference_index(
        _references(),
        store=store,
        rights=_rights(),
        embedder=_embedder(),
        image_loader=lambda ref: Image.new("RGB", (8, 8), color=(20, 0, 0)),
    )
    current = store.current_version_dir()

    def partial_loader(reference: ReferencePhoto) -> Image.Image:
        if reference.reference_photo_id == "ref-2":
            raise OSError("temporary object-store failure")
        return Image.new("RGB", (8, 8), color=(20, 0, 0))

    with pytest.raises(ValueError, match="All reference photos"):
        build_reference_index(
            _references(),
            store=store,
            rights=_rights(),
            embedder=_embedder(),
            image_loader=partial_loader,
        )

    assert store.current_version_dir() == current
    assert first["indexVersion"] == current.name


def test_rebuild_rejects_unattested_reference_source_before_loading_images(tmp_path: Path) -> None:
    calls = 0

    def loader(ref: ReferencePhoto) -> Image.Image:
        nonlocal calls
        calls += 1
        return Image.new("RGB", (8, 8))

    invalid = [ReferencePhoto("ref-1", "J35", None, "https://images.example/ref.jpg", "unapproved")]
    with pytest.raises(ValueError, match="not covered"):
        build_reference_index(
            invalid,
            store=AtomicIndexStore(tmp_path / "indexes"),
            rights=_rights(),
            embedder=_embedder(),
            image_loader=loader,
        )

    assert calls == 0


def test_readiness_revalidates_persisted_rights_attestation(tmp_path: Path) -> None:
    store = AtomicIndexStore(tmp_path / "indexes")
    build_reference_index(
        _references(),
        store=store,
        rights=_rights(),
        embedder=_embedder(),
        image_loader=lambda ref: Image.new("RGB", (8, 8), color=(20, 0, 0)),
    )
    (store.current_version_dir() / "rights.json").write_text("{}")

    runtime = IdentifierRuntime(index_store=store, embedder=_embedder())

    assert runtime.readiness() == (False, "index_unavailable")


def test_readiness_fails_when_the_pinned_model_cannot_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = AtomicIndexStore(tmp_path / "indexes")
    build_reference_index(
        _references(),
        store=store,
        rights=_rights(),
        embedder=_embedder(),
        image_loader=lambda ref: Image.new("RGB", (8, 8), color=(20, 0, 0)),
    )
    monkeypatch.setattr(
        runtime_module,
        "load_embedder",
        lambda name, **kwargs: (_ for _ in ()).throw(RuntimeError("model unavailable")),
    )

    runtime = IdentifierRuntime(index_store=store)

    assert runtime.readiness() == (False, "model_unavailable")


def test_rebuild_embeds_in_bounded_batches_and_enforces_aggregate_pixels(tmp_path: Path) -> None:
    batch_sizes: list[int] = []

    def embed(images: list[Image.Image]) -> np.ndarray:
        batch_sizes.append(len(images))
        values = np.ones((len(images), 2), dtype=np.float32)
        return values / np.linalg.norm(values, axis=1, keepdims=True)

    embedder = LoadedEmbedder(embed_fn=embed, embed_dim=2, name="dinov2-small")
    references = [
        ReferencePhoto(
            f"ref-{index}",
            f"J{index}",
            None,
            f"https://images.example/ref-{index}.jpg",
            "owner-grant-1",
        )
        for index in range(5)
    ]
    store = AtomicIndexStore(tmp_path / "indexes")
    build_reference_index(
        references,
        store=store,
        rights=_rights(),
        embedder=embedder,
        image_loader=lambda ref: Image.new("RGB", (100, 100)),
        batch_size=2,
        max_total_pixels=50_000,
    )
    assert batch_sizes == [2, 2, 1]

    with pytest.raises(ValueError, match="aggregate pixel"):
        build_reference_index(
            references,
            store=AtomicIndexStore(tmp_path / "overflow"),
            rights=_rights(),
            embedder=embedder,
            image_loader=lambda ref: Image.new("RGB", (100, 100)),
            batch_size=2,
            max_total_pixels=49_999,
        )


def test_crop_cannot_expand_a_validated_source_image_before_budget_accounting(
    tmp_path: Path,
) -> None:
    source_image = Image.new("RGB", (8, 8))
    reference = ReferencePhoto(
        "ref-1",
        "J35",
        None,
        "https://images.example/ref.jpg",
        "owner-grant-1",
        crop={"x": 0, "y": 0, "width": 1_000_000, "height": 1_000_000},
    )

    with pytest.raises(ValueError, match="crop must stay within"):
        build_reference_index(
            [reference],
            store=AtomicIndexStore(tmp_path / "indexes"),
            rights=_rights(),
            embedder=_embedder(),
            image_loader=lambda ref: source_image,
            max_total_pixels=100,
        )
    with pytest.raises(ValueError, match="closed image"):
        source_image.getpixel((0, 0))


@pytest.mark.parametrize(
    "corruption",
    [
        "metadata-count",
        "reference-count",
        "embed-dimension",
        "embedder-info",
        "version-identity",
        "required-metadata",
    ],
)
def test_readiness_rejects_internally_inconsistent_index_artifacts_before_assignment(
    tmp_path: Path,
    corruption: str,
) -> None:
    store = AtomicIndexStore(tmp_path / "indexes")
    build_reference_index(
        _references(),
        store=store,
        rights=_rights(),
        embedder=_embedder(),
        image_loader=lambda ref: Image.new("RGB", (8, 8), color=(20, 0, 0)),
    )
    version_dir = store.current_version_dir()
    info_path = version_dir / "index_info.json"
    metadata_path = version_dir / "metadata.json"
    info = json.loads(info_path.read_text())
    metadata = json.loads(metadata_path.read_text())

    if corruption == "metadata-count":
        metadata["metadata"].pop()
        metadata_path.write_text(json.dumps(metadata))
    elif corruption == "reference-count":
        info["referenceCount"] = 999
        info_path.write_text(json.dumps(info))
    elif corruption == "embed-dimension":
        info["embedDim"] = 999
        info_path.write_text(json.dumps(info))
    elif corruption == "embedder-info":
        info["model"] = "different-embedder"
        info_path.write_text(json.dumps(info))
    elif corruption == "version-identity":
        info["indexVersion"] = "different-version"
        info_path.write_text(json.dumps(info))
    elif corruption == "required-metadata":
        del metadata["metadata"][0]["catalogId"]
        metadata_path.write_text(json.dumps(metadata))

    runtime = IdentifierRuntime(index_store=store, embedder=_embedder())

    assert runtime.readiness() == (False, "index_unavailable")
    assert runtime._bundle is None
