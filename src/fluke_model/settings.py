"""Validated runtime configuration for the model service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

MIN_API_KEY_LENGTH = 32


@dataclass(frozen=True)
class ServiceSettings:
    api_key: str
    index_dir: Path
    allowed_reference_hosts: frozenset[str]
    model_artifact_dir: Path = Path("artifacts/models/dinov2-small")
    max_image_bytes: int = 8 * 1024 * 1024
    max_request_bytes: int = 12 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    inference_timeout_seconds: float = 20.0
    rebuild_timeout_seconds: float = 300.0
    max_references: int = 1_000
    reference_batch_size: int = 1
    max_reference_pixels_total: int = 500_000_000
    identify_requests_per_minute: int = 60
    rebuild_requests_per_minute: int = 5
    health_requests_per_minute: int = 120

    def __post_init__(self) -> None:
        if len(self.api_key) < MIN_API_KEY_LENGTH:
            raise ValueError(
                f"FLUKE_MODEL_API_KEY must be at least {MIN_API_KEY_LENGTH} characters"
            )
        if self.max_image_bytes < 1 or self.max_request_bytes < self.max_image_bytes:
            raise ValueError("request byte limit must be at least the image byte limit")
        if self.max_image_pixels < 1:
            raise ValueError("image limits must be positive")
        if self.reference_batch_size < 1 or self.max_reference_pixels_total < 1:
            raise ValueError("reference build limits must be positive")
        if self.inference_timeout_seconds <= 0 or self.rebuild_timeout_seconds <= 0:
            raise ValueError("service timeouts must be positive")
        if (
            min(
                self.identify_requests_per_minute,
                self.rebuild_requests_per_minute,
                self.health_requests_per_minute,
            )
            < 1
        ):
            raise ValueError("rate limits must be positive")

    @classmethod
    def from_env(cls) -> ServiceSettings:
        api_key = os.environ.get("FLUKE_MODEL_API_KEY", "")
        index_dir = Path(os.environ.get("FLUKE_REFERENCE_INDEX_DIR", "artifacts/reference-index"))
        model_artifact_dir = Path(
            os.environ.get("FLUKE_MODEL_ARTIFACT_DIR", "artifacts/models/dinov2-small")
        )
        hosts = frozenset(
            host.strip().lower()
            for host in os.environ.get("FLUKE_REFERENCE_ALLOWED_HOSTS", "").split(",")
            if host.strip()
        )
        return cls(
            api_key=api_key,
            index_dir=index_dir,
            allowed_reference_hosts=hosts,
            model_artifact_dir=model_artifact_dir,
        )
