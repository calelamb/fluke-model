"""Descriptor-relative, no-follow snapshots for authenticated release inputs."""

from __future__ import annotations

import hashlib
import ctypes
import errno
import os
import stat
from pathlib import Path, PurePosixPath

_CHUNK_BYTES = 1024 * 1024
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FLAGS = _READ_FLAGS | getattr(os, "O_DIRECTORY", 0)
_AT_FDCWD = -2 if os.uname().sysname == "Darwin" else -100


def snapshot_tree(source: Path, destination: Path) -> None:
    """Copy a tree from one held root descriptor without following links."""
    root = os.open(source, _DIRECTORY_FLAGS)
    try:
        destination.mkdir()
        _snapshot_directory(root, destination)
    finally:
        os.close(root)


def snapshot_relative_file(root: Path, relative: str, destination: Path) -> str:
    """Copy and hash one canonical relative file through held directory descriptors."""
    parts = PurePosixPath(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("snapshot path must be canonical and relative")
    descriptor = os.open(root, _DIRECTORY_FLAGS)
    try:
        for part in parts[:-1]:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        source = os.open(parts[-1], _READ_FLAGS, dir_fd=descriptor)
        try:
            return _copy_file_descriptor(source, destination)
        finally:
            os.close(source)
    finally:
        os.close(descriptor)


def snapshot_regular_file(source: Path, destination: Path) -> str:
    descriptor = os.open(source, _READ_FLAGS)
    try:
        return _copy_file_descriptor(descriptor, destination)
    finally:
        os.close(descriptor)


def publish_directory_no_replace(staging: Path, destination: Path) -> None:
    """Atomically publish only when destination does not exist at the rename instant."""
    library = ctypes.CDLL(None, use_errno=True)
    first = os.fsencode(staging)
    second = os.fsencode(destination)
    if os.uname().sysname == "Darwin":
        rename = library.renameatx_np
        flag = 0x00000004
    else:
        rename = library.renameat2
        flag = 0x00000001
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(_AT_FDCWD, first, _AT_FDCWD, second, flag) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError("publication destination already exists")
        raise OSError(error, os.strerror(error))


def _snapshot_directory(descriptor: int, destination: Path) -> None:
    for name in sorted(os.listdir(descriptor)):
        child = os.open(name, _READ_FLAGS, dir_fd=descriptor)
        try:
            metadata = os.fstat(child)
            target = destination / name
            if stat.S_ISDIR(metadata.st_mode):
                target.mkdir()
                _snapshot_directory(child, target)
            elif stat.S_ISREG(metadata.st_mode):
                _copy_file_descriptor(child, target)
            else:
                raise ValueError("snapshot source contains a non-regular entry")
        finally:
            os.close(child)


def _copy_file_descriptor(descriptor: int, destination: Path) -> str:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("snapshot source must be a regular file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with os.fdopen(os.dup(descriptor), "rb", closefd=True) as source, destination.open("xb") as out:
        for chunk in iter(lambda: source.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())
    return digest.hexdigest()
