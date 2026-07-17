"""Archive traversal regressions for the legacy dataset downloader."""

from __future__ import annotations

import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.download_beluga import extract_archive  # noqa: E402


def test_extract_archive_rejects_zip_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside.txt", "unsafe")

    with pytest.raises(ValueError, match="unsafe archive member"):
        extract_archive(archive, tmp_path / "extract")
    assert not (tmp_path / "outside.txt").exists()


def test_extract_archive_rejects_tar_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        member = tarfile.TarInfo("../outside.txt")
        data = b"unsafe"
        member.size = len(data)
        output.addfile(member, io.BytesIO(data))

    with pytest.raises((tarfile.FilterError, ValueError)):
        extract_archive(archive, tmp_path / "extract")
    assert not (tmp_path / "outside.txt").exists()
