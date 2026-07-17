"""Validated production rebuild orchestration."""

from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from fluke_model.identify_runtime import (
    IdentifierRuntime,
    ReferencePhoto,
    build_reference_index,
)
from fluke_model.deadline import (
    OperationDeadline,
    OperationSupersededError,
)
from fluke_model.index_store import AtomicIndexStore
from fluke_model.network import ReferenceImageFetcher
from fluke_model.rights import DataRights, ModelRights, RightsAttestation
from fluke_model.settings import ServiceSettings


class _CamelModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class ModelRightsPayload(_CamelModel):
    model_id: str = Field(alias="modelId", min_length=1, max_length=200)
    revision: str = Field(min_length=7, max_length=200)
    license_spdx: str = Field(alias="licenseSpdx", min_length=1, max_length=100)
    evidence_url: str = Field(alias="evidenceUrl", min_length=9, max_length=2_048)
    commercial_use_allowed: bool = Field(alias="commercialUseAllowed")


class DataRightsPayload(_CamelModel):
    source_id: str = Field(alias="sourceId", min_length=1, max_length=200, pattern=r"\S")
    license_or_permission: str = Field(alias="licenseOrPermission", min_length=1, max_length=500)
    evidence_url: str = Field(alias="evidenceUrl", min_length=9, max_length=2_048)
    commercial_use_allowed: bool = Field(alias="commercialUseAllowed")


class RightsPayload(_CamelModel):
    schema_version: int = Field(alias="schemaVersion")
    approved_by: str = Field(alias="approvedBy", min_length=1, max_length=200)
    approved_at: datetime = Field(alias="approvedAt")
    commercial_use_allowed: bool = Field(alias="commercialUseAllowed")
    model: ModelRightsPayload
    data_sources: tuple[DataRightsPayload, ...] = Field(alias="dataSources", min_length=1)


class CropPayload(_CamelModel):
    x: float = Field(ge=0, le=100_000)
    y: float = Field(ge=0, le=100_000)
    width: float = Field(gt=0, le=100_000)
    height: float = Field(gt=0, le=100_000)


class ReferencePayload(_CamelModel):
    reference_photo_id: str = Field(
        alias="referencePhotoId", min_length=1, max_length=200, pattern=r"\S"
    )
    catalog_id: str = Field(alias="catalogId", min_length=1, max_length=200, pattern=r"\S")
    name: str | None = Field(default=None, max_length=20_000)
    url: str = Field(min_length=9, max_length=2_048)
    rights_source_id: str = Field(
        alias="rightsSourceId", min_length=1, max_length=200, pattern=r"\S"
    )
    side: str = Field(default="UNKNOWN", max_length=100)
    quality: str = Field(default="USABLE", max_length=100)
    crop: CropPayload | None = None


class RebuildPayload(_CamelModel):
    references: tuple[ReferencePayload, ...] = Field(min_length=1)
    rights_attestation: RightsPayload = Field(alias="rightsAttestation")


class ProductionRebuilder:
    def __init__(
        self,
        *,
        settings: ServiceSettings,
        runtime: IdentifierRuntime,
        store: AtomicIndexStore,
        fetcher: ReferenceImageFetcher,
    ) -> None:
        self._settings = settings
        self._runtime = runtime
        self._store = store
        self._fetcher = fetcher
        self._lock = Lock()
        self._sequence_lock = Lock()
        self._latest_sequence = 0

    def __call__(
        self,
        raw_payload: dict[str, Any],
        deadline: OperationDeadline | None = None,
    ) -> dict[str, Any]:
        operation_deadline = deadline or OperationDeadline.never()
        sequence = self._register_request()
        with self._lock:
            self._guard_publication(sequence, operation_deadline)
            payload = RebuildPayload.model_validate(raw_payload)
            if len(payload.references) > self._settings.max_references:
                raise ValueError("reference count exceeds the configured limit")
            reference_ids = tuple(reference.reference_photo_id for reference in payload.references)
            if len(reference_ids) != len(set(reference_ids)):
                raise ValueError("referencePhotoId values must be unique")
            rights = _rights_from_payload(payload.rights_attestation)
            references = [_reference_from_payload(reference) for reference in payload.references]
            return build_reference_index(
                references,
                store=self._store,
                rights=rights,
                embedder=self._runtime.embedder,
                image_loader=lambda reference: self._fetcher.load(
                    reference.url, deadline=operation_deadline
                ),
                batch_size=self._settings.reference_batch_size,
                max_total_pixels=self._settings.max_reference_pixels_total,
                deadline=operation_deadline,
                publication_guard=lambda: self._guard_publication(sequence, operation_deadline),
                publish_version=lambda version_dir: self._publish_if_latest(
                    sequence,
                    operation_deadline,
                    lambda: self._store.publish(version_dir),
                ),
            )

    def _register_request(self) -> int:
        with self._sequence_lock:
            sequence = self._latest_sequence + 1
            self._latest_sequence = sequence
            return sequence

    def _guard_publication(self, sequence: int, deadline: OperationDeadline) -> None:
        deadline.check()
        with self._sequence_lock:
            if sequence != self._latest_sequence:
                raise OperationSupersededError("a newer rebuild request superseded this operation")

    def _publish_if_latest(
        self,
        sequence: int,
        deadline: OperationDeadline,
        publish: Callable[[], None],
    ) -> None:
        with self._sequence_lock:
            deadline.check()
            if sequence != self._latest_sequence:
                raise OperationSupersededError("a newer rebuild request superseded this operation")
            publish()


def _rights_from_payload(payload: RightsPayload) -> RightsAttestation:
    return RightsAttestation(
        schema_version=payload.schema_version,
        approved_by=payload.approved_by,
        approved_at=payload.approved_at,
        commercial_use_allowed=payload.commercial_use_allowed,
        model=ModelRights(
            model_id=payload.model.model_id,
            revision=payload.model.revision,
            license_spdx=payload.model.license_spdx,
            evidence_url=payload.model.evidence_url,
            commercial_use_allowed=payload.model.commercial_use_allowed,
        ),
        data_sources=tuple(
            DataRights(
                source_id=source.source_id,
                license_or_permission=source.license_or_permission,
                evidence_url=source.evidence_url,
                commercial_use_allowed=source.commercial_use_allowed,
            )
            for source in payload.data_sources
        ),
    )


def _reference_from_payload(payload: ReferencePayload) -> ReferencePhoto:
    return ReferencePhoto(
        reference_photo_id=payload.reference_photo_id,
        catalog_id=payload.catalog_id,
        name=payload.name,
        url=payload.url,
        rights_source_id=payload.rights_source_id,
        side=payload.side,
        quality=payload.quality,
        crop=payload.crop.model_dump() if payload.crop is not None else None,
    )
