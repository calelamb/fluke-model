"""Atomic index publication tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from fluke_model.index_store import AtomicIndexStore


def _complete(version_dir: Path, marker: str) -> None:
    for filename in ("index.faiss", "metadata.json", "index_info.json", "rights.json"):
        (version_dir / filename).write_text(marker)


def test_publish_switches_a_small_pointer_and_preserves_previous_version(tmp_path: Path) -> None:
    store = AtomicIndexStore(tmp_path)
    first = store.create_version("first")
    _complete(first, "first")
    store.publish(first)
    second = store.create_version("second")
    _complete(second, "second")

    store.publish(second)

    assert store.current_version_dir() == second
    assert (first / "metadata.json").read_text() == "first"


def test_unpublished_or_malformed_versions_never_replace_current(tmp_path: Path) -> None:
    store = AtomicIndexStore(tmp_path)
    current = store.create_version("current")
    _complete(current, "ok")
    store.publish(current)

    staged = store.create_version("staged")
    with pytest.raises(ValueError, match="required index files"):
        store.publish(staged)

    assert store.current_version_dir() == current


def test_pointer_path_traversal_fails_closed(tmp_path: Path) -> None:
    store = AtomicIndexStore(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "current.json").write_text('{"version": "../outside"}')

    with pytest.raises(ValueError, match="invalid index version"):
        store.current_version_dir()
