"""Deterministic and fail-closed Core ML artifact export tests."""

from __future__ import annotations

import json
import os
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
import torch

from fluke_model.coreml_artifact import (
    CoreMLExportError,
    FixedShapeDINOv2Embedder,
    _atomic_exchange_directories,
    _canonicalize_coreml_package,
    build_export_metadata,
    export_coreml,
    package_tree_sha256,
    publish_coreml_export,
    validate_preprocessor_config,
)
from fluke_model.embedders import DINO_V2_REVISION
from fluke_model.model_artifact import ModelArtifactError


class _FakePatchEmbeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Conv2d(3, 4, kernel_size=14, stride=14)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.projection(pixels).flatten(2).transpose(1, 2)


class _FakeEmbeddings(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_size = 14
        self.patch_embeddings = _FakePatchEmbeddings()
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, 4))
        self.position_embeddings = torch.nn.Parameter(torch.randn(1, 1370, 4))
        self.dropout = torch.nn.Dropout(0.0)

    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        class_position = self.position_embeddings[:, :1]
        patch_positions = self.position_embeddings[:, 1:].reshape(1, 37, 37, 4)
        patch_positions = patch_positions.permute(0, 3, 1, 2)
        resized = torch.nn.functional.interpolate(
            patch_positions.float(),
            size=(height // self.patch_size, width // self.patch_size),
            mode="bicubic",
            align_corners=False,
        ).to(dtype=patch_positions.dtype)
        resized = resized.permute(0, 2, 3, 1).reshape(1, -1, embeddings.shape[-1])
        return torch.cat((class_position, resized), dim=1)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embeddings(pixels)
        cls_token = self.cls_token.expand(pixels.shape[0], -1, -1)
        embeddings = torch.cat((cls_token, patches), dim=1)
        return self.dropout(embeddings + self.interpolate_pos_encoding(embeddings, 224, 224))


class _FakeEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.GELU()])


class _FakeDINOv2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = _FakeEmbeddings()
        self.encoder = _FakeEncoder()
        self.layernorm = torch.nn.LayerNorm(4)

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        return_dict: bool,
    ) -> tuple[torch.Tensor]:
        assert return_dict is False
        hidden = self.embeddings(pixel_values)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return (self.layernorm(hidden),)


class _WrongPositionShapeEmbeddings(_FakeEmbeddings):
    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        return super().interpolate_pos_encoding(embeddings, height, width)[:, :-1]


class _WrongPositionShapeDINOv2(_FakeDINOv2):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = _WrongPositionShapeEmbeddings()


class _FakeModelMessage:
    def ParseFromString(self, payload: bytes) -> None:
        assert payload.startswith(b"wire-order-")

    def SerializeToString(self, *, deterministic: bool) -> bytes:
        assert deterministic is True
        return b"deterministic-protobuf"


def _write_nondeterministic_package(
    package: Path,
    *,
    model_identifier: str,
    weight_identifier: str,
    model_bytes: bytes,
) -> None:
    data = package / "Data" / "com.apple.CoreML"
    (data / "weights").mkdir(parents=True)
    (data / "model.mlmodel").write_bytes(model_bytes)
    (data / "weights" / "weight.bin").write_bytes(b"stable-weights")
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            weight_identifier: {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Weights",
                "name": "weights",
                "path": "com.apple.CoreML/weights",
            },
            model_identifier: {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Specification",
                "name": "model.mlmodel",
                "path": "com.apple.CoreML/model.mlmodel",
            },
        },
        "rootModelIdentifier": model_identifier,
    }
    (package / "Manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _valid_coreml_spec() -> object:
    input_type = SimpleNamespace(shape=[1, 3, 224, 224], dataType=65568)
    output_type = SimpleNamespace(shape=[1, 384], dataType=65568)
    return SimpleNamespace(
        description=SimpleNamespace(
            input=[
                SimpleNamespace(
                    name="pixels",
                    type=SimpleNamespace(multiArrayType=input_type),
                )
            ],
            output=[
                SimpleNamespace(
                    name="embedding",
                    type=SimpleNamespace(multiArrayType=output_type),
                )
            ],
        )
    )


def _valid_spec_loader(_package_path: Path) -> object:
    return _valid_coreml_spec()


def _source_artifact_sha256(model_sha256: str = "a" * 64) -> dict[str, str]:
    return {
        "config.json": "c" * 64,
        "model.safetensors": model_sha256,
        "preprocessor_config.json": "d" * 64,
    }


def _test_directory_exchange(first: Path, second: Path) -> None:
    temporary = first.with_name(f".{first.name}.test-swap")
    os.replace(first, temporary)
    os.replace(second, first)
    os.replace(temporary, second)


def test_platform_atomic_directory_exchange_swaps_both_trees(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "first.txt").write_text("first", encoding="utf-8")
    (second / "second.txt").write_text("second", encoding="utf-8")

    _atomic_exchange_directories(first, second)

    assert (first / "second.txt").read_text(encoding="utf-8") == "second"
    assert (second / "first.txt").read_text(encoding="utf-8") == "first"


def test_platform_atomic_directory_exchange_failure_preserves_tree(tmp_path: Path) -> None:
    first = tmp_path / "first"
    missing = tmp_path / "missing"
    first.mkdir()
    sentinel = first / "first.txt"
    sentinel.write_text("first", encoding="utf-8")

    with pytest.raises(OSError):
        _atomic_exchange_directories(first, missing)

    assert sentinel.read_text(encoding="utf-8") == "first"
    assert not missing.exists()


def test_export_metadata_records_every_reproducibility_input() -> None:
    metadata = build_export_metadata(
        model_sha256="a" * 64,
        package_sha256="b" * 64,
        source_artifact_sha256=_source_artifact_sha256(),
        tool_versions={
            "coremltools": "9.0",
            "numpy": "2.2.6",
            "python": "3.11.9",
            "torch": "2.13.0",
            "transformers": "5.14.0",
        },
    )

    assert metadata.model_revision == DINO_V2_REVISION
    assert metadata.minimum_deployment_target == "iOS17"
    assert metadata.compute_precision == "FLOAT16"
    assert metadata.input_shape == (1, 3, 224, 224)
    assert metadata.output_shape == (1, 384)
    assert dict(metadata.tool_versions) == {
        "coremltools": "9.0",
        "numpy": "2.2.6",
        "python": "3.11.9",
        "torch": "2.13.0",
        "transformers": "5.14.0",
    }

    with pytest.raises(FrozenInstanceError):
        metadata.package_sha256 = "c" * 64
    with pytest.raises(TypeError):
        metadata.tool_versions["torch"] = "replacement"


@pytest.mark.parametrize(
    ("field_name", "digest"),
    [
        ("model_sha256", "not-a-digest"),
        ("package_sha256", "A" * 64),
        ("package_sha256", "b" * 63),
    ],
)
def test_export_metadata_rejects_invalid_digests(field_name: str, digest: str) -> None:
    arguments = {
        "model_sha256": "a" * 64,
        "package_sha256": "b" * 64,
        "source_artifact_sha256": _source_artifact_sha256(),
        "tool_versions": {"python": "3.11.9"},
    }
    arguments[field_name] = digest

    with pytest.raises(CoreMLExportError, match=field_name):
        build_export_metadata(**arguments)


@pytest.mark.parametrize(
    "tool_versions",
    [{}, {"": "3.11.9"}, {"python": ""}],
)
def test_export_metadata_rejects_invalid_tool_versions(
    tool_versions: dict[str, str],
) -> None:
    with pytest.raises(CoreMLExportError, match="tool_version|tool version"):
        build_export_metadata(
            model_sha256="a" * 64,
            package_sha256="b" * 64,
            source_artifact_sha256=_source_artifact_sha256(),
            tool_versions=tool_versions,
        )


def test_package_tree_digest_is_stable_across_creation_order(tmp_path: Path) -> None:
    first = tmp_path / "first.mlpackage"
    second = tmp_path / "second.mlpackage"
    (first / "Data").mkdir(parents=True)
    (first / "Manifest.json").write_text("manifest\n", encoding="utf-8")
    (first / "Data" / "weights.bin").write_bytes(b"weights")
    (second / "Data").mkdir(parents=True)
    (second / "Data" / "weights.bin").write_bytes(b"weights")
    (second / "Manifest.json").write_text("manifest\n", encoding="utf-8")

    assert package_tree_sha256(first) == package_tree_sha256(second)


def test_package_tree_digest_rejects_symlinks(tmp_path: Path) -> None:
    package = tmp_path / "unsafe.mlpackage"
    package.mkdir()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    (package / "weights.bin").symlink_to(target)

    with pytest.raises(CoreMLExportError, match="symbolic link"):
        package_tree_sha256(package)


@pytest.mark.parametrize("create_empty_directory", [False, True])
def test_package_tree_digest_rejects_missing_or_empty_package(
    tmp_path: Path,
    create_empty_directory: bool,
) -> None:
    package = tmp_path / "empty.mlpackage"
    if create_empty_directory:
        package.mkdir()

    with pytest.raises(CoreMLExportError, match="regular directory|must not be empty"):
        package_tree_sha256(package)


def test_package_tree_digest_rejects_nonregular_entries(tmp_path: Path) -> None:
    package = tmp_path / "unsafe.mlpackage"
    package.mkdir()
    os.mkfifo(package / "named-pipe")

    with pytest.raises(CoreMLExportError, match="non-regular file"):
        package_tree_sha256(package)


def test_coreml_package_canonicalization_is_byte_for_byte_deterministic(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.mlpackage"
    second = tmp_path / "second.mlpackage"
    _write_nondeterministic_package(
        first,
        model_identifier="11111111-1111-4111-8111-111111111111",
        weight_identifier="22222222-2222-4222-8222-222222222222",
        model_bytes=b"wire-order-first",
    )
    _write_nondeterministic_package(
        second,
        model_identifier="AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
        weight_identifier="BBBBBBBB-BBBB-4BBB-8BBB-BBBBBBBBBBBB",
        model_bytes=b"wire-order-second",
    )

    _canonicalize_coreml_package(first, _FakeModelMessage)
    _canonicalize_coreml_package(second, _FakeModelMessage)

    assert package_tree_sha256(first) == package_tree_sha256(second)
    assert (first / "Data/com.apple.CoreML/model.mlmodel").read_bytes() == (
        b"deterministic-protobuf"
    )
    manifest = json.loads((first / "Manifest.json").read_text(encoding="utf-8"))
    root_identifier = manifest["rootModelIdentifier"]
    assert UUID(root_identifier).version == 5
    assert manifest["itemInfoEntries"][root_identifier]["name"] == "model.mlmodel"


def test_coreml_package_canonicalization_rejects_symlink_ancestor(
    tmp_path: Path,
) -> None:
    package = tmp_path / "unsafe.mlpackage"
    data = package / "Data"
    external = tmp_path / "external"
    data.mkdir(parents=True)
    external.mkdir()
    external_model = external / "model.mlmodel"
    external_model.write_bytes(b"wire-order-external")
    (data / "com.apple.CoreML").symlink_to(external, target_is_directory=True)
    manifest = {
        "fileFormatVersion": "1.0.0",
        "itemInfoEntries": {
            "11111111-1111-4111-8111-111111111111": {
                "author": "com.apple.CoreML",
                "description": "CoreML Model Specification",
                "name": "model.mlmodel",
                "path": "com.apple.CoreML/model.mlmodel",
            }
        },
        "rootModelIdentifier": "11111111-1111-4111-8111-111111111111",
    }
    (package / "Manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CoreMLExportError, match="symbolic link|outside package"):
        _canonicalize_coreml_package(package, _FakeModelMessage)

    assert external_model.read_bytes() == b"wire-order-external"


def test_fixed_shape_adapter_matches_original_eager_model() -> None:
    torch.manual_seed(7)
    caller_model = _FakeDINOv2().eval()
    pixels = torch.randn((1, 3, 224, 224))
    expected_hidden = caller_model(pixel_values=pixels, return_dict=False)[0]
    expected = torch.nn.functional.normalize(expected_hidden[:, 0, :], dim=-1)

    adapter = FixedShapeDINOv2Embedder(caller_model).eval()

    torch.testing.assert_close(adapter(pixels), expected, rtol=1e-5, atol=1e-6)
    assert adapter._position_embeddings.shape == (1, 257, 4)


def test_fixed_shape_adapter_owns_copy_without_mutating_caller() -> None:
    caller_model = _FakeDINOv2()
    caller_projection = caller_model.embeddings.patch_embeddings.projection

    adapter = FixedShapeDINOv2Embedder(caller_model)

    assert caller_model.training is True
    assert adapter.training is False
    assert adapter._patch_embeddings.projection is not caller_projection
    assert adapter._position_embeddings.requires_grad is False


def test_fixed_shape_adapter_rejects_every_noncontract_shape() -> None:
    adapter = FixedShapeDINOv2Embedder(_FakeDINOv2()).eval()

    for invalid_shape in ((2, 3, 224, 224), (1, 1, 224, 224), (1, 3, 210, 224)):
        with pytest.raises(CoreMLExportError, match="input shape"):
            adapter(torch.zeros(invalid_shape))


def test_fixed_shape_adapter_rejects_malformed_interpolation_result() -> None:
    with pytest.raises(CoreMLExportError, match="position embedding shape"):
        FixedShapeDINOv2Embedder(_WrongPositionShapeDINOv2())


def test_export_rejects_unverified_source(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError):
        export_coreml(tmp_path / "missing", tmp_path / "out")


def test_preprocessor_config_must_match_mobile_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "preprocessor_config.json"
    config_path.write_text(
        json.dumps(
            {
                "crop_size": {"height": 224, "width": 224},
                "do_center_crop": True,
                "do_convert_rgb": True,
                "do_normalize": True,
                "do_rescale": True,
                "do_resize": True,
                "image_mean": [0.485, 0.456, 0.406],
                "image_std": [0.229, 0.224, 0.225],
                "resample": 3,
                "rescale_factor": 1 / 255,
                "size": {"shortest_edge": 256},
            }
        ),
        encoding="utf-8",
    )

    validate_preprocessor_config(config_path)

    divergent = json.loads(config_path.read_text(encoding="utf-8"))
    divergent["image_mean"] = [0.5, 0.5, 0.5]
    config_path.write_text(json.dumps(divergent), encoding="utf-8")
    with pytest.raises(CoreMLExportError, match="image_mean"):
        validate_preprocessor_config(config_path)


@pytest.mark.parametrize("contents", [None, "not-json"])
def test_preprocessor_config_missing_or_malformed_fails_closed(
    tmp_path: Path,
    contents: str | None,
) -> None:
    config_path = tmp_path / "preprocessor_config.json"
    if contents is not None:
        config_path.write_text(contents, encoding="utf-8")

    with pytest.raises(CoreMLExportError, match="preprocessor config"):
        validate_preprocessor_config(config_path)


def test_preprocessor_config_must_be_json_object(tmp_path: Path) -> None:
    config_path = tmp_path / "preprocessor_config.json"
    config_path.write_text("[]", encoding="utf-8")

    with pytest.raises(CoreMLExportError, match="JSON object"):
        validate_preprocessor_config(config_path)


def test_publish_refuses_nonempty_output_without_replace(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    called = False

    def fake_export(_artifact_dir: Path, _staging_dir: Path) -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(CoreMLExportError, match="non-empty"):
        publish_coreml_export(
            tmp_path / "source",
            output,
            replace=False,
            exporter=fake_export,
        )

    assert called is False
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_publish_rejects_source_as_output(tmp_path: Path) -> None:
    source = tmp_path / "source"

    with pytest.raises(CoreMLExportError, match="overlap"):
        publish_coreml_export(source, source, replace=False)


@pytest.mark.parametrize("output_position", ["parent", "child"])
def test_publish_rejects_source_output_ancestor_overlap(
    tmp_path: Path,
    output_position: str,
) -> None:
    artifact_root = tmp_path / "artifacts"
    source = artifact_root / "dinov2-small"
    source.mkdir(parents=True)
    output = artifact_root if output_position == "parent" else source / "mobile-release"
    called = False

    def fake_exporter(_artifact_dir: Path, _staging_dir: Path) -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(CoreMLExportError, match="overlap"):
        publish_coreml_export(
            source,
            output,
            replace=True,
            exporter=fake_exporter,
        )

    assert called is False


def test_publish_rejects_ambiguous_output_name(tmp_path: Path) -> None:
    with pytest.raises(CoreMLExportError, match="explicit name"):
        publish_coreml_export(tmp_path / "source", Path("."), replace=False)


@pytest.mark.parametrize("destination_kind", ["symlink", "file"])
def test_publish_rejects_unsafe_output_type(
    tmp_path: Path,
    destination_kind: str,
) -> None:
    output = tmp_path / "candidate"
    if destination_kind == "symlink":
        target = tmp_path / "target"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    else:
        output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(CoreMLExportError, match="symbolic link|must be a directory"):
        publish_coreml_export(tmp_path / "source", output, replace=False)


def test_publish_wraps_unexpected_exporter_error_and_cleans_staging(tmp_path: Path) -> None:
    output = tmp_path / "candidate"

    def failing_exporter(_artifact_dir: Path, _staging_dir: Path) -> object:
        raise ValueError("unexpected")

    with pytest.raises(CoreMLExportError, match="failed to publish"):
        publish_coreml_export(
            tmp_path / "source",
            output,
            replace=False,
            exporter=failing_exporter,
        )

    assert list(tmp_path.glob(".candidate.*")) == []


def test_publish_rejects_incomplete_staged_export(tmp_path: Path) -> None:
    metadata = build_export_metadata(
        model_sha256="a" * 64,
        package_sha256="b" * 64,
        source_artifact_sha256=_source_artifact_sha256(),
        tool_versions={"python": "3.11.9"},
    )

    def incomplete_exporter(_artifact_dir: Path, _staging_dir: Path) -> object:
        return metadata

    with pytest.raises(CoreMLExportError, match="missing FlukeEmbedder.mlpackage"):
        publish_coreml_export(
            tmp_path / "source",
            tmp_path / "candidate",
            replace=False,
            exporter=incomplete_exporter,
        )


def test_publish_rejects_staged_package_with_invalid_coreml_interface(
    tmp_path: Path,
) -> None:
    def fake_exporter(_artifact_dir: Path, staging_dir: Path) -> object:
        package = staging_dir / "FlukeEmbedder.mlpackage"
        package.mkdir()
        (package / "model.bin").write_bytes(b"model")
        metadata = build_export_metadata(
            model_sha256="a" * 64,
            package_sha256=package_tree_sha256(package),
            source_artifact_sha256=_source_artifact_sha256(),
            tool_versions={"python": "3.11.9"},
        )
        (staging_dir / "export-metadata.json").write_text(
            json.dumps(metadata.as_json_dict()),
            encoding="utf-8",
        )
        return metadata

    invalid_spec = _valid_coreml_spec()
    invalid_spec.description.output[0].name = "wrong-output"

    with pytest.raises(CoreMLExportError, match="Core ML.*interface"):
        publish_coreml_export(
            tmp_path / "source",
            tmp_path / "candidate",
            replace=False,
            exporter=fake_exporter,
            package_loader=lambda _path: invalid_spec,
        )

    assert not (tmp_path / "candidate").exists()
    with pytest.raises(CoreMLExportError, match="reload failed"):
        publish_coreml_export(
            tmp_path / "source",
            tmp_path / "candidate-reload",
            replace=False,
            exporter=fake_exporter,
            package_loader=lambda _path: (_ for _ in ()).throw(ValueError("unreadable")),
        )


@pytest.mark.parametrize(
    ("failure_mode", "message"),
    [
        ("missing_metadata", "missing export-metadata.json"),
        ("unexpected_entry", "unexpected entries"),
        ("malformed_metadata", "metadata is invalid"),
        ("mismatched_metadata", "does not match exporter result"),
    ],
)
def test_publish_rejects_invalid_staged_metadata(
    tmp_path: Path,
    failure_mode: str,
    message: str,
) -> None:
    def invalid_exporter(_artifact_dir: Path, staging_dir: Path) -> object:
        package = staging_dir / "FlukeEmbedder.mlpackage"
        package.mkdir()
        (package / "model.bin").write_bytes(b"model")
        metadata = build_export_metadata(
            model_sha256="a" * 64,
            package_sha256=package_tree_sha256(package),
            source_artifact_sha256=_source_artifact_sha256(),
            tool_versions={"python": "3.11.9"},
        )
        metadata_path = staging_dir / "export-metadata.json"
        if failure_mode == "missing_metadata":
            return metadata
        if failure_mode == "malformed_metadata":
            metadata_path.write_text("not-json", encoding="utf-8")
        elif failure_mode == "mismatched_metadata":
            metadata_path.write_text("{}", encoding="utf-8")
        else:
            metadata_path.write_text(json.dumps(metadata.as_json_dict()), encoding="utf-8")
            (staging_dir / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        return metadata

    with pytest.raises(CoreMLExportError, match=message):
        publish_coreml_export(
            tmp_path / "source",
            tmp_path / "candidate",
            replace=False,
            exporter=invalid_exporter,
        )


def test_publish_replaces_output_atomically_and_cleans_staging(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    output.mkdir()
    (output / "old.txt").write_text("old", encoding="utf-8")
    metadata = build_export_metadata(
        model_sha256="a" * 64,
        package_sha256="b" * 64,
        source_artifact_sha256=_source_artifact_sha256(),
        tool_versions={"python": "3.11.9"},
    )

    def fake_export(_artifact_dir: Path, staging_dir: Path) -> object:
        package = staging_dir / "FlukeEmbedder.mlpackage"
        package.mkdir(parents=True)
        (package / "model.bin").write_bytes(b"model")
        actual_metadata = build_export_metadata(
            model_sha256=metadata.model_sha256,
            package_sha256=package_tree_sha256(package),
            source_artifact_sha256=metadata.source_artifact_sha256,
            tool_versions=metadata.tool_versions,
        )
        (staging_dir / "export-metadata.json").write_text(
            json.dumps(actual_metadata.as_json_dict()),
            encoding="utf-8",
        )
        return actual_metadata

    result = publish_coreml_export(
        tmp_path / "source",
        output,
        replace=True,
        exporter=fake_export,
        exchange=_test_directory_exchange,
        package_loader=_valid_spec_loader,
    )

    assert result.model_sha256 == metadata.model_sha256
    assert not (output / "old.txt").exists()
    assert (output / "FlukeEmbedder.mlpackage" / "model.bin").read_bytes() == b"model"
    assert list(tmp_path.glob(".candidate.*")) == []


def test_publish_atomic_exchange_failure_preserves_both_trees(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    output.mkdir()
    sentinel = output / "old.txt"
    sentinel.write_text("old", encoding="utf-8")

    def fake_exporter(_artifact_dir: Path, staging_dir: Path) -> object:
        package = staging_dir / "FlukeEmbedder.mlpackage"
        package.mkdir()
        (package / "model.bin").write_bytes(b"new")
        metadata = build_export_metadata(
            model_sha256="a" * 64,
            package_sha256=package_tree_sha256(package),
            source_artifact_sha256=_source_artifact_sha256(),
            tool_versions={"python": "3.11.9"},
        )
        (staging_dir / "export-metadata.json").write_text(
            json.dumps(metadata.as_json_dict()),
            encoding="utf-8",
        )
        return metadata

    def failing_exchange(first: Path, second: Path) -> None:
        assert (first / "FlukeEmbedder.mlpackage/model.bin").read_bytes() == b"new"
        assert (second / "old.txt").read_text(encoding="utf-8") == "old"
        raise OSError("exchange unavailable")

    with pytest.raises(CoreMLExportError, match="atomic directory exchange failed"):
        publish_coreml_export(
            tmp_path / "source",
            output,
            replace=True,
            exporter=fake_exporter,
            exchange=failing_exchange,
            package_loader=_valid_spec_loader,
        )

    assert sentinel.read_text(encoding="utf-8") == "old"
    staging_directories = list(tmp_path.glob(".candidate.*.staging"))
    assert len(staging_directories) == 1
    assert (
        staging_directories[0] / "FlukeEmbedder.mlpackage/model.bin"
    ).read_bytes() == b"new"
