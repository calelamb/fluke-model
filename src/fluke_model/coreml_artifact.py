"""Verified, deterministic Core ML artifact export boundaries."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import tempfile
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import torch

from fluke_model.mobile_export import mobile_model_contract
from fluke_model.model_artifact import verify_dinov2_artifact

COREML_PACKAGE_NAME = "FlukeEmbedder.mlpackage"
EXPORT_METADATA_NAME = "export-metadata.json"
MINIMUM_DEPLOYMENT_TARGET = "iOS17"
COMPUTE_PRECISION = "FLOAT16"
_MODEL_FILENAME = "model.safetensors"
_SHA256_LENGTH = 64
_HASH_CHUNK_BYTES = 1024 * 1024
_PACKAGE_HASH_DOMAIN = b"fluke-coreml-package-v1\0"
_PREPROCESSOR_CONFIG_FILENAME = "preprocessor_config.json"
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_COREML_FLOAT32_DATA_TYPE = 65568
_RENAME_EXCHANGE = 0x00000002
_DARWIN_AT_FDCWD = -2
_LINUX_AT_FDCWD = -100
_COREML_IDENTIFIER_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "https://orcawatch.app/coreml-package/v1",
)


class CoreMLExportError(RuntimeError):
    """The Core ML export cannot be produced without weakening its contract."""


class FixedShapeDINOv2Embedder(torch.nn.Module):
    """Own a fixed-shape DINOv2 path with eager positional encoding frozen."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        owned_model = deepcopy(model).eval()
        embeddings = owned_model.embeddings
        self._patch_embeddings = embeddings.patch_embeddings
        self._cls_token = embeddings.cls_token
        self._dropout = embeddings.dropout
        self._encoder_layers = owned_model.encoder.layer
        self._layernorm = owned_model.layernorm
        position_embeddings = self._fixed_position_embeddings(embeddings)
        self.register_buffer("_position_embeddings", position_embeddings, persistent=True)
        self.eval()

    @staticmethod
    def _fixed_position_embeddings(embeddings: torch.nn.Module) -> torch.Tensor:
        contract = mobile_model_contract()
        batch_size, _, height, width = contract.input_shape
        patch_size = embeddings.patch_size
        token_count = (height // patch_size) * (width // patch_size) + 1
        hidden_size = embeddings.position_embeddings.shape[-1]
        reference = embeddings.position_embeddings
        dummy_embeddings = torch.zeros(
            (batch_size, token_count, hidden_size),
            dtype=reference.dtype,
            device=reference.device,
        )
        with torch.no_grad():
            fixed = embeddings.interpolate_pos_encoding(dummy_embeddings, height, width)
        if tuple(fixed.shape) != (batch_size, token_count, hidden_size):
            raise CoreMLExportError("fixed position embedding shape does not match contract")
        return fixed.detach().clone()

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        contract = mobile_model_contract()
        if tuple(pixels.shape) != contract.input_shape:
            raise CoreMLExportError(
                f"input shape must be {contract.input_shape}, received {tuple(pixels.shape)}"
            )
        target_dtype = self._patch_embeddings.projection.weight.dtype
        patches = self._patch_embeddings(pixels.to(dtype=target_dtype))
        cls_tokens = self._cls_token.expand(pixels.shape[0], -1, -1)
        hidden = torch.cat((cls_tokens, patches), dim=1)
        hidden = self._dropout(hidden + self._position_embeddings)
        for encoder_layer in self._encoder_layers:
            hidden = encoder_layer(hidden)
        normalized_hidden = self._layernorm(hidden)
        return torch.nn.functional.normalize(normalized_hidden[:, 0, :], dim=-1)


@dataclass(frozen=True)
class ExportMetadata:
    """Reproducibility record for one verified Core ML package."""

    model_id: str
    model_revision: str
    preprocessing_version: str
    minimum_deployment_target: str
    compute_precision: str
    input_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int]
    model_sha256: str
    package_sha256: str
    tool_versions: Mapping[str, str]

    def __post_init__(self) -> None:
        immutable_versions = MappingProxyType(dict(sorted(self.tool_versions.items())))
        object.__setattr__(self, "tool_versions", immutable_versions)

    def as_json_dict(self) -> dict[str, object]:
        """Return a JSON-compatible copy with deterministic key/value ordering."""
        return {
            "compute_precision": self.compute_precision,
            "input_shape": list(self.input_shape),
            "minimum_deployment_target": self.minimum_deployment_target,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_sha256": self.model_sha256,
            "output_shape": list(self.output_shape),
            "package_sha256": self.package_sha256,
            "preprocessing_version": self.preprocessing_version,
            "tool_versions": dict(self.tool_versions),
        }


@dataclass(frozen=True)
class _ExportDependencies:
    coremltools: Any
    numpy: Any
    torch: Any
    auto_model: Any
    transformers_version: str


def build_export_metadata(
    *,
    model_sha256: str,
    package_sha256: str,
    tool_versions: Mapping[str, str],
) -> ExportMetadata:
    """Build immutable metadata after validating all external digest/version input."""
    _validate_sha256("model_sha256", model_sha256)
    _validate_sha256("package_sha256", package_sha256)
    normalized_versions = _validate_tool_versions(tool_versions)
    contract = mobile_model_contract()
    return ExportMetadata(
        model_id=contract.model_id,
        model_revision=contract.revision,
        preprocessing_version=contract.preprocessing_version,
        minimum_deployment_target=MINIMUM_DEPLOYMENT_TARGET,
        compute_precision=COMPUTE_PRECISION,
        input_shape=contract.input_shape,
        output_shape=contract.output_shape,
        model_sha256=model_sha256,
        package_sha256=package_sha256,
        tool_versions=normalized_versions,
    )


def package_tree_sha256(package_dir: Path) -> str:
    """Hash a package by sorted relative paths and bytes, independent of traversal order."""
    root = Path(package_dir)
    if root.is_symlink() or not root.is_dir():
        raise CoreMLExportError("Core ML package must be a regular directory")

    entries = tuple(sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix()))
    if not entries:
        raise CoreMLExportError("Core ML package must not be empty")

    digest = hashlib.sha256(_PACKAGE_HASH_DOMAIN)
    for path in entries:
        relative_path = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            raise CoreMLExportError(f"Core ML package contains a symbolic link: {path.name}")
        if path.is_dir():
            _update_framed(digest, b"D", relative_path)
            continue
        if not path.is_file():
            raise CoreMLExportError(f"Core ML package contains a non-regular file: {path.name}")
        _update_framed(digest, b"F", relative_path)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_preprocessor_config(config_path: Path) -> None:
    """Require the pinned processor JSON to match the explicit mobile preprocessing contract."""
    path = Path(config_path)
    if path.is_symlink() or not path.is_file():
        raise CoreMLExportError("preprocessor config must be a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CoreMLExportError(f"preprocessor config is invalid: {error}") from error
    if not isinstance(payload, dict):
        raise CoreMLExportError("preprocessor config must contain a JSON object")

    expected_values = (
        ("crop_size", {"height": 224, "width": 224}),
        ("do_center_crop", True),
        ("do_convert_rgb", True),
        ("do_normalize", True),
        ("do_rescale", True),
        ("do_resize", True),
        ("image_mean", list(_IMAGENET_MEAN)),
        ("image_std", list(_IMAGENET_STD)),
        ("resample", 3),
        ("rescale_factor", 1 / 255),
        ("size", {"shortest_edge": 256}),
    )
    for field_name, expected in expected_values:
        if payload.get(field_name) != expected:
            raise CoreMLExportError(f"preprocessor config field does not match: {field_name}")


def export_coreml(artifact_dir: Path, output_dir: Path) -> ExportMetadata:
    """Convert one digest-verified local DINOv2 artifact to an iOS 17 ML Program."""
    source = Path(artifact_dir)
    destination = Path(output_dir)
    verify_dinov2_artifact(source)
    validate_preprocessor_config(source / _PREPROCESSOR_CONFIG_FILENAME)
    _ensure_empty_export_directory(destination)
    dependencies = _load_export_dependencies()

    try:
        package_path = _convert_coreml_package(source, destination, dependencies)
    except Exception as error:
        raise CoreMLExportError(f"Core ML conversion failed: {error}") from error

    metadata = build_export_metadata(
        model_sha256=_sha256_file(source / _MODEL_FILENAME),
        package_sha256=package_tree_sha256(package_path),
        tool_versions=_tool_versions(dependencies),
    )
    _write_json_atomic(destination / EXPORT_METADATA_NAME, metadata.as_json_dict())
    return metadata


def publish_coreml_export(
    artifact_dir: Path,
    output_dir: Path,
    *,
    replace: bool,
    exporter: Callable[[Path, Path], ExportMetadata] = export_coreml,
    exchange: Callable[[Path, Path], None] | None = None,
    spec_loader: Callable[[Path], Any] | None = None,
) -> ExportMetadata:
    """Stage an export beside its destination, then publish it with an atomic rename."""
    source = Path(artifact_dir)
    destination = Path(output_dir)
    _validate_publish_destination(source, destination, replace=replace)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = _create_staging_directory(destination)
    preserve_staging = False
    try:
        metadata = exporter(source, staging)
        _validate_staged_export(staging, metadata, spec_loader=spec_loader)
        if destination.exists():
            exchange_directories = exchange or _atomic_exchange_directories
            try:
                exchange_directories(staging, destination)
            except Exception as error:
                preserve_staging = True
                raise CoreMLExportError(
                    f"atomic directory exchange failed; staged export retained at {staging}: "
                    f"{error}"
                ) from error
            shutil.rmtree(staging)
        else:
            os.replace(staging, destination)
        return metadata
    except CoreMLExportError:
        raise
    except Exception as error:
        raise CoreMLExportError(f"failed to publish Core ML export: {error}") from error
    finally:
        if staging.exists() and not preserve_staging:
            shutil.rmtree(staging)


def _convert_coreml_package(
    source: Path,
    destination: Path,
    dependencies: _ExportDependencies,
) -> Path:
    model = dependencies.auto_model.from_pretrained(
        source,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="eager",
    ).eval()
    wrapper = FixedShapeDINOv2Embedder(model).eval()
    example = dependencies.torch.zeros(mobile_model_contract().input_shape)
    traced = dependencies.torch.jit.trace(wrapper, example, strict=True)
    package = dependencies.coremltools.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            dependencies.coremltools.TensorType(
                name="pixels",
                shape=example.shape,
                dtype=dependencies.numpy.float32,
            )
        ],
        outputs=[
            dependencies.coremltools.TensorType(
                name="embedding",
                dtype=dependencies.numpy.float32,
            )
        ],
        minimum_deployment_target=dependencies.coremltools.target.iOS17,
        compute_precision=dependencies.coremltools.precision.FLOAT16,
    )
    package_path = destination / COREML_PACKAGE_NAME
    package.save(package_path)
    _canonicalize_coreml_package(
        package_path,
        dependencies.coremltools.proto.Model_pb2.Model,
    )
    return package_path


def _create_staging_directory(destination: Path) -> Path:
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".staging",
            dir=destination.parent,
        )
    )


def _atomic_exchange_directories(first: Path, second: Path) -> None:
    system = platform.system()
    if system == "Darwin":
        _renameatx_np_exchange(first, second)
        return
    if system == "Linux":
        _renameat2_exchange(first, second)
        return
    raise OSError(errno.ENOTSUP, f"atomic directory exchange is unsupported on {system}")


def _renameatx_np_exchange(first: Path, second: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameatx_np = library.renameatx_np
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameatx_np is unavailable") from error
    renameatx_np.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameatx_np.restype = ctypes.c_int
    result = renameatx_np(
        _DARWIN_AT_FDCWD,
        os.fsencode(first),
        _DARWIN_AT_FDCWD,
        os.fsencode(second),
        _RENAME_EXCHANGE,
    )
    _raise_for_exchange_error(result)


def _renameat2_exchange(first: Path, second: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = library.renameat2
    except AttributeError as error:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable") from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _LINUX_AT_FDCWD,
        os.fsencode(first),
        _LINUX_AT_FDCWD,
        os.fsencode(second),
        _RENAME_EXCHANGE,
    )
    _raise_for_exchange_error(result)


def _raise_for_exchange_error(result: int) -> None:
    if result == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number))


def _validate_sha256(field_name: str, value: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CoreMLExportError(f"{field_name} must be a lowercase SHA256 digest")


def _validate_tool_versions(tool_versions: Mapping[str, str]) -> dict[str, str]:
    if not tool_versions:
        raise CoreMLExportError("tool_versions must not be empty")
    normalized = dict(sorted(tool_versions.items()))
    for name, version in normalized.items():
        if not isinstance(name, str) or not name.strip():
            raise CoreMLExportError("tool version names must be non-empty strings")
        if not isinstance(version, str) or not version.strip():
            raise CoreMLExportError(f"tool version must be a non-empty string: {name}")
    return normalized


def _update_framed(digest: Any, entry_type: bytes, relative_path: bytes) -> None:
    digest.update(entry_type)
    digest.update(len(relative_path).to_bytes(8, byteorder="big"))
    digest.update(relative_path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonicalize_coreml_package(
    package_dir: Path,
    model_message_factory: Callable[[], Any],
) -> None:
    """Rewrite generated UUIDs and protobuf map order into canonical package bytes."""
    package = Path(package_dir)
    manifest_path = package / "Manifest.json"
    manifest = _read_package_manifest(manifest_path)
    entries = manifest.get("itemInfoEntries")
    root_identifier = manifest.get("rootModelIdentifier")
    if not isinstance(entries, dict) or not isinstance(root_identifier, str):
        raise CoreMLExportError("Core ML package manifest has invalid identifiers")
    canonical_entries, identifier_map = _canonical_manifest_entries(package, entries)
    if root_identifier not in identifier_map:
        raise CoreMLExportError("Core ML package manifest root identifier is missing")

    root_entry = entries[root_identifier]
    model_path = _package_data_path(package, root_entry)
    if model_path.is_symlink() or not model_path.is_file():
        raise CoreMLExportError("Core ML package model must be a regular file")
    model_message = model_message_factory()
    try:
        model_message.ParseFromString(model_path.read_bytes())
        canonical_model = model_message.SerializeToString(deterministic=True)
    except Exception as error:
        raise CoreMLExportError(f"Core ML package model protobuf is invalid: {error}") from error

    canonical_manifest = {
        **manifest,
        "itemInfoEntries": canonical_entries,
        "rootModelIdentifier": identifier_map[root_identifier],
    }
    _write_bytes_atomic(model_path, canonical_model)
    _write_json_atomic(manifest_path, canonical_manifest)


def _read_package_manifest(manifest_path: Path) -> dict[str, object]:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise CoreMLExportError("Core ML package manifest must be a regular file")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CoreMLExportError(f"Core ML package manifest is invalid: {error}") from error
    if not isinstance(manifest, dict):
        raise CoreMLExportError("Core ML package manifest must contain a JSON object")
    return manifest


def _canonical_manifest_entries(
    package: Path,
    entries: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, str]]:
    canonical_entries: dict[str, object] = {}
    identifier_map: dict[str, str] = {}
    for old_identifier, entry in entries.items():
        if not isinstance(old_identifier, str) or not isinstance(entry, dict):
            raise CoreMLExportError("Core ML package manifest entry is invalid")
        _package_data_path(package, entry)
        stable_identifier = _stable_manifest_identifier(entry)
        if stable_identifier in canonical_entries:
            raise CoreMLExportError("Core ML package manifest entries are not unique")
        canonical_entries[stable_identifier] = entry
        identifier_map[old_identifier] = stable_identifier
    return dict(sorted(canonical_entries.items())), identifier_map


def _stable_manifest_identifier(entry: Mapping[str, object]) -> str:
    labels = tuple(entry.get(field) for field in ("author", "name", "path"))
    if any(not isinstance(label, str) or not label for label in labels):
        raise CoreMLExportError("Core ML package manifest entry labels are invalid")
    stable_label = "\0".join(labels)
    return str(uuid5(_COREML_IDENTIFIER_NAMESPACE, stable_label)).upper()


def _package_data_path(package: Path, entry: Mapping[str, object]) -> Path:
    relative_value = entry.get("path")
    if not isinstance(relative_value, str):
        raise CoreMLExportError("Core ML package manifest entry path is invalid")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise CoreMLExportError("Core ML package manifest entry path is unsafe")
    data_root = package / "Data"
    target = data_root / Path(*relative.parts)
    _reject_symlink_components(package, target)
    try:
        data_root_resolved = data_root.resolve(strict=True)
        target_resolved = target.resolve(strict=True)
    except OSError as error:
        raise CoreMLExportError("Core ML package manifest entry target is missing") from error
    if not target_resolved.is_relative_to(data_root_resolved):
        raise CoreMLExportError("Core ML package manifest entry target is outside package Data")
    if not target.exists():
        raise CoreMLExportError("Core ML package manifest entry target is missing")
    return target


def _reject_symlink_components(package: Path, target: Path) -> None:
    try:
        relative = target.relative_to(package)
    except ValueError as error:
        raise CoreMLExportError("Core ML package target is outside package") from error
    current = package
    if current.is_symlink():
        raise CoreMLExportError("Core ML package path contains a symbolic link")
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise CoreMLExportError("Core ML package path contains a symbolic link")


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_empty_export_directory(destination: Path) -> None:
    if destination.is_symlink():
        raise CoreMLExportError("output directory must not be a symbolic link")
    if destination.exists() and not destination.is_dir():
        raise CoreMLExportError("output path must be a directory")
    if destination.exists() and any(destination.iterdir()):
        raise CoreMLExportError("output directory must be empty")
    destination.mkdir(parents=True, exist_ok=True)


def _load_export_dependencies() -> _ExportDependencies:
    try:
        import coremltools
        import numpy
        import torch
        import transformers
        from transformers import AutoModel
    except ImportError as error:
        raise CoreMLExportError(
            f"Core ML export dependency is unavailable: {error.name}"
        ) from error
    return _ExportDependencies(
        coremltools=coremltools,
        numpy=numpy,
        torch=torch,
        auto_model=AutoModel,
        transformers_version=transformers.__version__,
    )


def _tool_versions(dependencies: _ExportDependencies) -> dict[str, str]:
    return {
        "coremltools": dependencies.coremltools.__version__,
        "numpy": dependencies.numpy.__version__,
        "python": platform.python_version(),
        "torch": dependencies.torch.__version__,
        "transformers": dependencies.transformers_version,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_publish_destination(source: Path, destination: Path, *, replace: bool) -> None:
    source_resolved = source.resolve(strict=source.exists())
    destination_resolved = destination.resolve(strict=destination.exists())
    if _paths_overlap(source_resolved, destination_resolved):
        raise CoreMLExportError("artifact and output directories must not overlap")
    if destination.name in {"", ".", ".."}:
        raise CoreMLExportError("output directory must have an explicit name")
    if destination.is_symlink():
        raise CoreMLExportError("output directory must not be a symbolic link")
    if destination.exists() and not destination.is_dir():
        raise CoreMLExportError("output path must be a directory")
    if destination.exists() and any(destination.iterdir()) and not replace:
        raise CoreMLExportError("output directory is non-empty; pass --replace to replace it")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _validate_staged_export(
    staging: Path,
    metadata: ExportMetadata,
    *,
    spec_loader: Callable[[Path], Any] | None,
) -> None:
    package_path = staging / COREML_PACKAGE_NAME
    metadata_path = staging / EXPORT_METADATA_NAME
    if not package_path.is_dir() or package_path.is_symlink():
        raise CoreMLExportError(f"staged export is missing {COREML_PACKAGE_NAME}")
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise CoreMLExportError(f"staged export is missing {EXPORT_METADATA_NAME}")
    unexpected = {
        path.name
        for path in staging.iterdir()
        if path.name not in {COREML_PACKAGE_NAME, EXPORT_METADATA_NAME}
    }
    if unexpected:
        raise CoreMLExportError(f"staged export contains unexpected entries: {sorted(unexpected)}")
    actual_package_sha256 = package_tree_sha256(package_path)
    if actual_package_sha256 != metadata.package_sha256:
        raise CoreMLExportError("staged package digest does not match export metadata")
    try:
        serialized_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CoreMLExportError(f"staged export metadata is invalid: {error}") from error
    if serialized_metadata != metadata.as_json_dict():
        raise CoreMLExportError("staged export metadata does not match exporter result")
    _validate_coreml_package_interface(package_path, spec_loader=spec_loader)


def _validate_coreml_package_interface(
    package_path: Path,
    *,
    spec_loader: Callable[[Path], Any] | None,
) -> None:
    loader = spec_loader or _load_coreml_package_spec
    try:
        spec = _load_isolated_coreml_spec(package_path, loader)
        inputs = _coreml_feature_contract(spec.description.input)
        outputs = _coreml_feature_contract(spec.description.output)
    except CoreMLExportError:
        raise
    except Exception as error:
        raise CoreMLExportError(f"staged Core ML package reload failed: {error}") from error
    contract = mobile_model_contract()
    expected_inputs = (("pixels", contract.input_shape, _COREML_FLOAT32_DATA_TYPE),)
    expected_outputs = (("embedding", contract.output_shape, _COREML_FLOAT32_DATA_TYPE),)
    if inputs != expected_inputs or outputs != expected_outputs:
        raise CoreMLExportError("staged Core ML package interface does not match contract")


def _load_isolated_coreml_spec(
    package_path: Path,
    loader: Callable[[Path], Any],
) -> Any:
    with tempfile.TemporaryDirectory(
        prefix=".coreml-validation.",
        dir=package_path.parent,
    ) as temporary_root:
        validation_package = Path(temporary_root) / package_path.name
        shutil.copytree(package_path, validation_package)
        return loader(validation_package)


def _coreml_feature_contract(features: Any) -> tuple[tuple[str, tuple[int, ...], int], ...]:
    return tuple(
        (
            feature.name,
            tuple(feature.type.multiArrayType.shape),
            feature.type.multiArrayType.dataType,
        )
        for feature in features
    )


def _load_coreml_package_spec(package_path: Path) -> Any:
    try:
        import coremltools
    except ImportError as error:
        raise CoreMLExportError("Core ML package validation dependency is unavailable") from error
    model = coremltools.models.MLModel(str(package_path), skip_model_load=True)
    return model.get_spec()
