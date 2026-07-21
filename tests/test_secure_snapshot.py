"""Security contracts for authenticated snapshots and atomic publication."""

from __future__ import annotations

from pathlib import Path

import pytest

from fluke_model.secure_snapshot import (
    publish_directory_no_replace,
    snapshot_relative_file,
    snapshot_tree,
)


def test_snapshot_tree_rejects_symbolic_link_entries(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    external = tmp_path / "external"
    external.write_text("secret", encoding="utf-8")
    (source / "link").symlink_to(external)

    with pytest.raises(OSError):
        snapshot_tree(source, tmp_path / "snapshot")


def test_snapshot_relative_file_rejects_parent_traversal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    with pytest.raises(ValueError, match="canonical and relative"):
        snapshot_relative_file(source, "../outside", tmp_path / "copy")


def test_publication_never_replaces_existing_destination(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    destination.mkdir()
    (staging / "value").write_text("new", encoding="utf-8")
    (destination / "value").write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        publish_directory_no_replace(staging, destination)

    assert (destination / "value").read_text(encoding="utf-8") == "existing"
    assert (staging / "value").read_text(encoding="utf-8") == "new"


def test_publication_atomically_moves_fresh_staging_directory(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    destination = tmp_path / "destination"
    staging.mkdir()
    (staging / "value").write_text("authenticated", encoding="utf-8")

    publish_directory_no_replace(staging, destination)

    assert not staging.exists()
    assert (destination / "value").read_text(encoding="utf-8") == "authenticated"
