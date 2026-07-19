"""CLI contract and path-safety tests for mobile catalog publication."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import numpy as np
import pytest

from fluke_model.coreml_artifact import CoreMLExportError, package_tree_sha256
from fluke_model.model_artifact import DINOV2_ARTIFACT_SHA256


def _load_cli_module() -> ModuleType:
    script = Path(__file__).parent.parent / "scripts" / "build_mobile_catalog.py"
    specification = importlib.util.spec_from_file_location("build_mobile_catalog", script)
    if specification is None or specification.loader is None:
        raise RuntimeError("mobile catalog CLI module could not be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


cli = _load_cli_module()


MODEL_ID = "facebook/dinov2-small"
MODEL_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
PREPROCESSING_VERSION = "dinov2-imagenet-v1"
TOOL_VERSIONS = {
    "coremltools": "9.0",
    "numpy": "2.2.6",
    "python": "3.11.15",
    "torch": "2.13.0",
    "transformers": "5.14.0",
}


def _export_metadata(package_sha256: str = "a" * 64) -> dict[str, Any]:
    return {
        "compute_precision": "FLOAT16",
        "input_shape": [1, 3, 224, 224],
        "minimum_deployment_target": "iOS17",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_sha256": DINOV2_ARTIFACT_SHA256["model.safetensors"],
        "output_shape": [1, 384],
        "package_sha256": package_sha256,
        "preprocessing_version": PREPROCESSING_VERSION,
        "tool_versions": dict(TOOL_VERSIONS),
    }


def _write_metadata(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "export-metadata.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _replace(payload: dict[str, Any], **updates: object) -> dict[str, Any]:
    return {**payload, **updates}


def _replace_tool(payload: dict[str, Any], **updates: str) -> dict[str, Any]:
    return {**payload, "tool_versions": {**payload["tool_versions"], **updates}}


def _remove_key(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def _add_tool(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "tool_versions": {**payload["tool_versions"], "extra": "1.0"}}


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        (lambda value: {**value, "extra": True}, "fields"),
        (lambda value: _remove_key(value, "model_id"), "fields"),
        (lambda value: _replace(value, model_id="other/model"), "model identity"),
        (lambda value: _replace(value, model_revision="other-revision"), "model identity"),
        (lambda value: _replace(value, preprocessing_version="other"), "preprocessing"),
        (lambda value: _replace(value, model_sha256="b" * 64), "source model digest"),
        (lambda value: _replace(value, package_sha256="not-a-digest"), "package digest"),
        (lambda value: _replace(value, input_shape=[1, 3, 224, True]), "input shape"),
        (lambda value: _replace(value, output_shape=[True, 384]), "output shape"),
        (lambda value: _replace(value, minimum_deployment_target="iOS16"), "deployment"),
        (lambda value: _replace(value, compute_precision="FLOAT32"), "precision"),
        (lambda value: _replace(value, tool_versions={}), "tool version fields"),
        (_add_tool, "tool version fields"),
        (lambda value: _replace_tool(value, coremltools="8.3"), "coremltools"),
        (lambda value: _replace_tool(value, numpy="2.3.0"), "numpy"),
        (lambda value: _replace_tool(value, torch="2.7.1"), "torch"),
        (lambda value: _replace_tool(value, transformers="5.3.0"), "transformers"),
        (lambda value: _replace_tool(value, python="3.12.1"), "python"),
        (lambda value: _replace_tool(value, python=True), "python"),
    ],
)
def test_export_metadata_rejects_malformed_or_tampered_contract(
    tmp_path: Path,
    tamper: Callable[[dict[str, Any]], dict[str, Any]],
    message: str,
) -> None:
    path = _write_metadata(tmp_path, tamper(_export_metadata()))

    with pytest.raises(ValueError, match=message):
        cli._load_export_metadata(path)


def test_export_metadata_accepts_only_the_audited_contract(tmp_path: Path) -> None:
    payload = _export_metadata()
    path = _write_metadata(tmp_path, payload)

    assert cli._load_export_metadata(path) == payload


def _source_paths(tmp_path: Path) -> dict[str, Path]:
    package = tmp_path / "FlukeEmbedder.mlpackage"
    package.mkdir()
    sources = {
        "model package": package,
        "export metadata": tmp_path / "export-metadata.json",
        "embeddings": tmp_path / "embeddings.npy",
        "references": tmp_path / "references.json",
        "rights": tmp_path / "rights.json",
    }
    for path in tuple(sources.values())[1:]:
        path.write_bytes(path.name.encode("utf-8"))
    return sources


@pytest.mark.parametrize(
    "source_name",
    ("model package", "export metadata", "embeddings", "references", "rights"),
)
def test_path_boundary_rejects_each_symlink_source_node(
    tmp_path: Path, source_name: str
) -> None:
    sources = _source_paths(tmp_path)
    original = sources[source_name]
    target = tmp_path / f"real-{source_name.replace(' ', '-')}"
    if original.is_dir():
        original.rmdir()
        target.mkdir()
    else:
        target.write_bytes(original.read_bytes())
        original.unlink()
    original.symlink_to(target, target_is_directory=target.is_dir())

    with pytest.raises(ValueError, match="symbolic link component"):
        cli._validate_path_boundaries(tmp_path / "output", sources)


def test_path_boundary_rejects_symlink_output_node(tmp_path: Path) -> None:
    sources = _source_paths(tmp_path)
    real_output = tmp_path / "real-output"
    real_output.mkdir()
    output = tmp_path / "output"
    output.symlink_to(real_output, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link component"):
        cli._validate_path_boundaries(output, sources)


def test_path_boundary_rejects_existing_symlink_ancestor(tmp_path: Path) -> None:
    sources = _source_paths(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link component"):
        cli._validate_path_boundaries(linked_parent / "catalog", sources)
    assert tuple(real_parent.iterdir()) == ()


def test_path_boundary_rejects_source_symlink_ancestor(tmp_path: Path) -> None:
    sources = _source_paths(tmp_path)
    real_parent = tmp_path / "real-source-parent"
    real_parent.mkdir()
    real_rights = real_parent / "rights.json"
    real_rights.write_bytes(sources["rights"].read_bytes())
    linked_parent = tmp_path / "linked-source-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    linked_sources = {**sources, "rights": linked_parent / "rights.json"}

    with pytest.raises(ValueError, match="symbolic link component"):
        cli._validate_path_boundaries(tmp_path / "output", linked_sources)


@pytest.mark.parametrize("relation", ("equal", "output-contains-source", "source-contains-output"))
def test_path_boundary_rejects_every_source_output_overlap(
    tmp_path: Path, relation: str
) -> None:
    sources = _source_paths(tmp_path)
    package = sources["model package"]
    if relation == "equal":
        output = package
    elif relation == "output-contains-source":
        output = tmp_path
    else:
        output = package / "nested-catalog"

    with pytest.raises(ValueError, match="overlap"):
        cli._validate_path_boundaries(output, sources)


def _valid_cli_inputs(tmp_path: Path) -> tuple[dict[str, Path], list[str]]:
    sources = _source_paths(tmp_path)
    package = sources["model package"]
    (package / "model.mlmodel").write_bytes(b"synthetic-package")
    metadata = _export_metadata(package_tree_sha256(package))
    sources["export metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    embeddings = np.eye(1, 384, dtype=np.float32)
    np.save(sources["embeddings"], embeddings, allow_pickle=False)
    sources["references"].write_text(
        json.dumps(
            [
                {
                    "referencePhotoId": "synthetic-ref-1",
                    "whaleId": "synthetic-whale-1",
                    "catalogId": "SYNTHETIC-1",
                    "sourceId": "synthetic-owned-fixture",
                }
            ]
        ),
        encoding="utf-8",
    )
    rights_fixture = (
        Path(__file__).parent / "fixtures" / "mobile-catalog" / "rights-attestation.json"
    )
    sources["rights"].write_bytes(rights_fixture.read_bytes())
    arguments = [
        "build_mobile_catalog.py",
        "--model-package",
        str(package),
        "--model-metadata",
        str(sources["export metadata"]),
        "--embeddings",
        str(sources["embeddings"]),
        "--references",
        str(sources["references"]),
        "--rights",
        str(sources["rights"]),
        "--manifest-version",
        "synthetic-test",
        "--model-version",
        "synthetic-test",
        "--index-version",
        "synthetic-test",
        "--score-threshold",
        "0.7",
        "--margin-threshold",
        "0.1",
        "--output-dir",
        str(package / "nested-catalog"),
    ]
    return sources, arguments


def _snapshot(paths: dict[str, Path]) -> dict[str, bytes]:
    files = tuple(
        path
        for source in paths.values()
        for path in ((source,) if source.is_file() else tuple(source.rglob("*")))
        if path.is_file()
    )
    return {str(path): path.read_bytes() for path in files}


def _with_output(arguments: list[str], output: Path) -> list[str]:
    return [*arguments[:-1], str(output)]


def test_cli_rejects_nested_output_before_changing_any_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, arguments = _valid_cli_inputs(tmp_path)
    before = _snapshot(sources)
    monkeypatch.setattr(sys, "argv", arguments)

    def reject_read(_: Path) -> dict[str, Any]:
        raise AssertionError("source read occurred before path validation")

    monkeypatch.setattr(cli, "_load_export_metadata", reject_read)

    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 2
    assert _snapshot(sources) == before
    assert not (sources["model package"] / "nested-catalog").exists()


def test_cli_bounds_coreml_export_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sources, arguments = _valid_cli_inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", _with_output(arguments, tmp_path / "output"))

    def fail_export(_: Path) -> dict[str, Any]:
        raise CoreMLExportError("synthetic package failure")

    monkeypatch.setattr(cli, "_load_export_metadata", fail_export)
    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 2
    assert "error: synthetic package failure" in capsys.readouterr().err


def test_cli_rejects_actual_package_digest_mismatch_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sources, arguments = _valid_cli_inputs(tmp_path)
    metadata = _export_metadata("b" * 64)
    sources["export metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "output"
    before = _snapshot(sources)
    monkeypatch.setattr(sys, "argv", _with_output(arguments, output))

    with pytest.raises(SystemExit) as raised:
        cli.main()

    assert raised.value.code == 2
    assert "package digest does not match export metadata" in capsys.readouterr().err
    assert _snapshot(sources) == before
    assert not output.exists()


def test_cli_score_semantics_is_literal(tmp_path: Path) -> None:
    parser = cli.build_parser()
    _, arguments = _valid_cli_inputs(tmp_path)
    parsed = parser.parse_args(arguments[1:])
    assert parsed.score_semantics == "cosineSimilarity"

    with pytest.raises(SystemExit):
        parser.parse_args([*arguments[1:], "--score-semantics", "other"])
