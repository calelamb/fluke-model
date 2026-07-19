"""Bounded, fail-closed loading for release parity NumPy arrays."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MAX_PARITY_FILE_BYTES = 32 * 1024 * 1024
MAX_PARITY_SAMPLE_ROWS = 20_000
MAX_NPY_HEADER_BYTES = 10_000
_FLOAT32_BYTES = 4
_SUPPORTED_NPY_VERSIONS = {(1, 0), (2, 0)}


def load_bounded_parity_array(
    path: Path,
    name: str,
    *,
    expected_columns: int,
) -> np.ndarray:
    """Validate size/header bounds before allocating and return an owned array."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{name} file is missing")
    with source.open("rb") as stream:
        file_bytes = os.fstat(stream.fileno()).st_size
        if file_bytes > MAX_PARITY_FILE_BYTES:
            raise ValueError(f"{name} exceeds the maximum file bytes")
        shape, fortran_order, dtype = _read_bounded_header(stream, name)
        _validate_header_contract(
            shape,
            fortran_order,
            dtype,
            name=name,
            expected_columns=expected_columns,
        )
        header_bytes = stream.tell()
        expected_bytes = header_bytes + shape[0] * shape[1] * _FLOAT32_BYTES
        if expected_bytes != file_bytes:
            raise ValueError(f"{name} file length does not match its NumPy header")
        stream.seek(0)
        value = np.load(
            stream,
            allow_pickle=False,
            max_header_size=MAX_NPY_HEADER_BYTES,
        )
    if type(value) is not np.ndarray:
        raise ValueError(f"{name} must contain exactly one NumPy array")
    return np.array(value, dtype=np.float32, order="C", copy=True)


def _read_bounded_header(
    stream: object,
    name: str,
) -> tuple[tuple[int, ...], bool, np.dtype[object]]:
    try:
        version = np.lib.format.read_magic(stream)
        if version not in _SUPPORTED_NPY_VERSIONS:
            raise ValueError(f"{name} uses an unsupported NumPy format version")
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran_order, dtype = reader(
            stream,
            max_header_size=MAX_NPY_HEADER_BYTES,
        )
    except (EOFError, OSError, OverflowError, ValueError) as error:
        raise ValueError(f"{name} has an invalid or oversized NumPy header: {error}") from error
    return shape, fortran_order, dtype


def _validate_header_contract(
    shape: tuple[int, ...],
    fortran_order: bool,
    dtype: np.dtype[object],
    *,
    name: str,
    expected_columns: int,
) -> None:
    exact_shape = (
        type(shape) is tuple
        and len(shape) == 2
        and all(type(dimension) is int for dimension in shape)
    )
    if not exact_shape or shape[0] <= 0 or shape[1] != expected_columns:
        raise ValueError(f"{name} must have positive shape (N, {expected_columns})")
    if shape[0] > MAX_PARITY_SAMPLE_ROWS:
        raise ValueError(f"{name} exceeds the maximum sample rows")
    if type(fortran_order) is not bool or fortran_order:
        raise ValueError(f"{name} must be stored in C order")
    if dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must use exact float32 dtype")
