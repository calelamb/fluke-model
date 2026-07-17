"""SHA256 verification for the exact production DINOv2 artifact."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.request import urlopen


class ModelArtifactError(RuntimeError):
    """The mounted or baked model artifact is absent or differs from its pin."""


DINOV2_ARTIFACT_SHA256 = {
    "config.json": "1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    "model.safetensors": "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
}
DINOV2_ARTIFACT_BASE_URL = (
    "https://huggingface.co/facebook/dinov2-small/resolve/ed25f3a31f01632728cabb09d1542f84ab7b0056"
)


def verify_artifact(root: Path, expected_sha256: Mapping[str, str]) -> None:
    for filename, expected_digest in expected_sha256.items():
        path = root / filename
        if not path.exists():
            raise ModelArtifactError(f"model artifact file is missing: {filename}")
        if path.is_symlink() or not path.is_file():
            raise ModelArtifactError(f"model artifact must be a regular file: {filename}")
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            raise ModelArtifactError(f"model artifact digest mismatch: {filename}")


def verify_dinov2_artifact(root: Path) -> None:
    verify_artifact(root, DINOV2_ARTIFACT_SHA256)


def download_dinov2_artifact(root: Path) -> None:
    download_artifact(
        root,
        base_url=DINOV2_ARTIFACT_BASE_URL,
        expected_sha256=DINOV2_ARTIFACT_SHA256,
    )


def download_artifact(
    root: Path,
    *,
    base_url: str,
    expected_sha256: Mapping[str, str],
    opener: Callable[..., Any] = urlopen,
) -> None:
    if not base_url.startswith("https://"):
        raise ModelArtifactError("model artifact source must use HTTPS")
    root.mkdir(parents=True, exist_ok=True)
    for filename, expected_digest in expected_sha256.items():
        _download_file(
            root / filename,
            url=f"{base_url.rstrip('/')}/{filename}",
            expected_digest=expected_digest,
            opener=opener,
        )
    verify_artifact(root, expected_sha256)


def _download_file(
    target: Path,
    *,
    url: str,
    expected_digest: str,
    opener: Callable[..., Any],
) -> None:
    temporary = target.with_name(f".{target.name}.partial")
    try:
        with opener(url, timeout=30.0) as response, temporary.open("wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
        if _sha256(temporary) != expected_digest:
            raise ModelArtifactError(f"downloaded model artifact digest mismatch: {target.name}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
