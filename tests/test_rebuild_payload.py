"""Production rebuild request parsing and orchestration tests."""

from __future__ import annotations

from pathlib import Path
from threading import Event, Thread
from typing import Any

import numpy as np
import pytest
from PIL import Image

from fluke_model.embedders import LoadedEmbedder
from fluke_model.deadline import OperationDeadline
from fluke_model.identify_runtime import DINO_V2_MODEL_ID, DINO_V2_REVISION
from fluke_model.index_store import AtomicIndexStore
from fluke_model.rebuild import ProductionRebuilder
from fluke_model import rebuild as rebuild_module
from fluke_model.settings import ServiceSettings


class Runtime:
    embedder = LoadedEmbedder(
        embed_fn=lambda images: np.ones((len(images), 2), dtype=np.float32),
        embed_dim=2,
        name="dinov2-small",
    )


class Fetcher:
    def load(self, url: str, *, deadline: object | None = None) -> Image.Image:
        return Image.new("RGB", (4, 4), color=(1, 2, 3))


def _payload() -> dict[str, Any]:
    return {
        "references": [
            {
                "referencePhotoId": "ref-1",
                "catalogId": "J35",
                "name": "Tahlequah",
                "url": "https://images.example.org/ref.jpg",
                "rightsSourceId": "grant-1",
                "crop": {"x": 0, "y": 0, "width": 4, "height": 4},
            }
        ],
        "rightsAttestation": {
            "schemaVersion": 1,
            "approvedBy": "Launch owner",
            "approvedAt": "2026-07-17T00:00:00Z",
            "commercialUseAllowed": True,
            "model": {
                "modelId": DINO_V2_MODEL_ID,
                "revision": DINO_V2_REVISION,
                "licenseSpdx": "Apache-2.0",
                "evidenceUrl": "https://github.com/facebookresearch/dinov2/blob/main/LICENSE",
                "commercialUseAllowed": True,
            },
            "dataSources": [
                {
                    "sourceId": "grant-1",
                    "licenseOrPermission": "written owner grant",
                    "evidenceUrl": "https://fluke.example/grants/1",
                    "commercialUseAllowed": True,
                }
            ],
        },
    }


def _rebuilder(tmp_path: Path, *, max_references: int = 2) -> ProductionRebuilder:
    settings = ServiceSettings(
        api_key="test-key-that-is-at-least-thirty-two-bytes",
        index_dir=tmp_path,
        allowed_reference_hosts=frozenset({"images.example.org"}),
        max_references=max_references,
    )
    store = AtomicIndexStore(tmp_path)
    return ProductionRebuilder(
        settings=settings,
        runtime=Runtime(),
        store=store,
        fetcher=Fetcher(),
    )


def test_production_rebuilder_parses_camel_case_and_publishes(tmp_path: Path) -> None:
    result = _rebuilder(tmp_path)(_payload())

    assert result["ok"] is True
    assert (tmp_path / "current.json").is_file()


def test_production_rebuilder_rejects_unknown_fields_and_reference_overflow(tmp_path: Path) -> None:
    unknown = _payload()
    unknown["unexpected"] = True
    with pytest.raises(ValueError):
        _rebuilder(tmp_path / "unknown")(unknown)

    overflow = _payload()
    overflow["references"] = overflow["references"] * 2
    with pytest.raises(ValueError, match="reference count"):
        _rebuilder(tmp_path / "overflow", max_references=1)(overflow)


def test_production_rebuilder_rejects_blank_and_duplicate_stable_ids(tmp_path: Path) -> None:
    blank = _payload()
    blank["references"][0]["catalogId"] = "   "
    with pytest.raises(ValueError):
        _rebuilder(tmp_path / "blank")(blank)

    duplicate = _payload()
    duplicate["references"] = duplicate["references"] * 2
    with pytest.raises(ValueError, match="referencePhotoId values must be unique"):
        _rebuilder(tmp_path / "duplicate")(duplicate)


def test_older_rebuild_cannot_publish_after_a_newer_request_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_inside = Event()
    release_first = Event()
    new_registered = Event()
    published: list[str] = []
    failures: list[Exception] = []

    def fake_build(references: list, *, publication_guard: object, **kwargs: object) -> dict:
        catalog_id = references[0].catalog_id
        if catalog_id == "OLD":
            first_inside.set()
            release_first.wait(timeout=1)
        publication_guard()
        published.append(catalog_id)
        return {"ok": True}

    monkeypatch.setattr(rebuild_module, "build_reference_index", fake_build)
    rebuilder = _rebuilder(tmp_path)
    register_request = rebuilder._register_request

    def tracked_register() -> int:
        sequence = register_request()
        if sequence == 2:
            new_registered.set()
        return sequence

    monkeypatch.setattr(rebuilder, "_register_request", tracked_register)
    old_payload = _payload()
    old_payload["references"][0]["catalogId"] = "OLD"
    new_payload = _payload()
    new_payload["references"][0]["catalogId"] = "NEW"

    def run(payload: dict) -> None:
        try:
            rebuilder(payload)
        except Exception as exc:
            failures.append(exc)

    old = Thread(target=run, args=(old_payload,))
    new = Thread(target=run, args=(new_payload,))
    old.start()
    assert first_inside.wait(timeout=1)
    new.start()
    assert new_registered.wait(timeout=1)
    release_first.set()
    old.join(timeout=1)
    new.join(timeout=1)

    assert published == ["NEW"]
    assert len(failures) == 1


def test_latest_sequence_check_and_pointer_swap_share_one_critical_section(tmp_path: Path) -> None:
    rebuilder = _rebuilder(tmp_path)
    first_sequence = rebuilder._register_request()
    pointer_swap_started = Event()
    release_pointer_swap = Event()
    newer_registered = Event()
    published: list[str] = []

    def pointer_swap() -> None:
        pointer_swap_started.set()
        release_pointer_swap.wait(timeout=1)
        published.append("OLD")

    old = Thread(
        target=rebuilder._publish_if_latest,
        args=(first_sequence, OperationDeadline.never(), pointer_swap),
    )
    old.start()
    assert pointer_swap_started.wait(timeout=1)

    def register_newer() -> None:
        rebuilder._register_request()
        newer_registered.set()

    newer = Thread(target=register_newer)
    newer.start()
    assert newer_registered.wait(timeout=0.05) is False

    release_pointer_swap.set()
    old.join(timeout=1)
    newer.join(timeout=1)

    assert published == ["OLD"]
    assert newer_registered.is_set()
