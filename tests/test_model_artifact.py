"""Offline, digest-pinned production model artifact tests."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from io import BytesIO

import pytest

from fluke_model import embedders
from fluke_model.model_artifact import ModelArtifactError, download_artifact, verify_artifact


def test_artifact_verification_accepts_exact_files_and_rejects_tampering(tmp_path: Path) -> None:
    content = b"rights-safe pinned weights"
    expected = {"model.safetensors": hashlib.sha256(content).hexdigest()}
    (tmp_path / "model.safetensors").write_bytes(content)

    verify_artifact(tmp_path, expected)

    (tmp_path / "model.safetensors").write_bytes(content + b"tampered")
    with pytest.raises(ModelArtifactError, match="digest"):
        verify_artifact(tmp_path, expected)


def test_artifact_verification_rejects_missing_and_symlinked_files(tmp_path: Path) -> None:
    expected = {"config.json": hashlib.sha256(b"{}").hexdigest()}
    with pytest.raises(ModelArtifactError, match="missing"):
        verify_artifact(tmp_path, expected)

    outside = tmp_path.parent / "outside-config.json"
    outside.write_bytes(b"{}")
    (tmp_path / "config.json").symlink_to(outside)
    with pytest.raises(ModelArtifactError, match="regular file"):
        verify_artifact(tmp_path, expected)


def test_production_embedder_verifies_local_artifact_and_disables_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    class Processor:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: object) -> object:
            calls.append(("processor", source, kwargs))
            return object()

    class Model:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: object) -> Model:
            calls.append(("model", source, kwargs))
            return cls()

        def to(self, device: object) -> Model:
            return self

        def eval(self) -> Model:
            return self

    verified: list[Path] = []
    monkeypatch.setattr(embedders, "verify_dinov2_artifact", lambda path: verified.append(path))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoImageProcessor=Processor, AutoModel=Model),
    )

    loaded = embedders.load_embedder("dinov2-small", artifact_dir=tmp_path)

    assert loaded.name == "dinov2-small"
    assert verified == [tmp_path]
    assert calls == [
        ("processor", str(tmp_path), {"local_files_only": True}),
        (
            "model",
            str(tmp_path),
            {"local_files_only": True, "use_safetensors": True},
        ),
    ]


def test_artifact_download_uses_exact_urls_and_verifies_before_returning(tmp_path: Path) -> None:
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
    expected = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    requested: list[str] = []

    def opener(url: str, timeout: float) -> BytesIO:
        requested.append(url)
        return BytesIO(files[url.rsplit("/", 1)[-1]])

    download_artifact(
        tmp_path,
        base_url="https://models.example/exact-revision",
        expected_sha256=expected,
        opener=opener,
    )

    assert requested == [
        "https://models.example/exact-revision/config.json",
        "https://models.example/exact-revision/model.safetensors",
    ]
    verify_artifact(tmp_path, expected)
