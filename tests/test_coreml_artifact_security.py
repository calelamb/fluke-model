"""Security regression tests for Core ML publication boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fluke_model.coreml_artifact import (
    CoreMLExportError,
    build_export_metadata,
    package_tree_sha256,
    publish_coreml_export,
)


def _valid_spec() -> object:
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


def test_package_reload_validation_cannot_mutate_staged_export(tmp_path: Path) -> None:
    def exporter(_artifact_dir: Path, staging_dir: Path) -> object:
        package = staging_dir / "FlukeEmbedder.mlpackage"
        package.mkdir()
        (package / "model.bin").write_bytes(b"canonical")
        metadata = build_export_metadata(
            model_sha256="a" * 64,
            package_sha256=package_tree_sha256(package),
            source_artifact_sha256={
                "config.json": "c" * 64,
                "model.safetensors": "a" * 64,
                "preprocessor_config.json": "d" * 64,
            },
            tool_versions={"python": "3.11.9"},
        )
        (staging_dir / "export-metadata.json").write_text(
            json.dumps(metadata.as_json_dict()),
            encoding="utf-8",
        )
        return metadata

    def mutating_loader(package_path: Path) -> object:
        (package_path / "model.bin").write_bytes(b"mutated")
        return _valid_spec()

    result = publish_coreml_export(
        tmp_path / "source",
        tmp_path / "candidate",
        replace=False,
        exporter=exporter,
        package_loader=mutating_loader,
    )

    package = tmp_path / "candidate/FlukeEmbedder.mlpackage"
    assert (package / "model.bin").read_bytes() == b"canonical"
    assert package_tree_sha256(package) == result.package_sha256


def test_publish_rejects_metadata_digest_that_does_not_match_package(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    metadata = build_export_metadata(
        model_sha256="a" * 64,
        package_sha256="b" * 64,
        source_artifact_sha256={
            "config.json": "c" * 64,
            "model.safetensors": "a" * 64,
            "preprocessor_config.json": "d" * 64,
        },
        tool_versions={"python": "3.11.9"},
    )

    def fake_export(_artifact_dir: Path, staging_dir: Path) -> object:
        package = staging_dir / "FlukeEmbedder.mlpackage"
        package.mkdir(parents=True)
        (package / "model.bin").write_bytes(b"different-package")
        (staging_dir / "export-metadata.json").write_text(
            json.dumps(metadata.as_json_dict()),
            encoding="utf-8",
        )
        return metadata

    with pytest.raises(CoreMLExportError, match="digest does not match"):
        publish_coreml_export(
            tmp_path / "source",
            output,
            replace=False,
            exporter=fake_export,
        )

    assert not output.exists()
    assert list(tmp_path.glob(".candidate.*")) == []
