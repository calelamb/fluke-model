"""Fail-closed written-rights attestations for weights and reference data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


class RightsError(ValueError):
    """Raised when production rights are absent, incomplete, or mismatched."""


EXPECTED_MODEL_LICENSE_SPDX = "Apache-2.0"


@dataclass(frozen=True)
class ModelRights:
    model_id: str
    revision: str
    license_spdx: str
    evidence_url: str
    commercial_use_allowed: bool


@dataclass(frozen=True)
class DataRights:
    source_id: str
    license_or_permission: str
    evidence_url: str
    commercial_use_allowed: bool


@dataclass(frozen=True)
class RightsAttestation:
    schema_version: int
    approved_by: str
    approved_at: datetime
    commercial_use_allowed: bool
    model: ModelRights
    data_sources: tuple[DataRights, ...]

    def validate_for(
        self,
        *,
        model_id: str,
        model_revision: str,
        reference_source_ids: tuple[str, ...],
    ) -> None:
        if self.schema_version != 1:
            raise RightsError("unsupported rights attestation schema")
        if not self.commercial_use_allowed or not self.model.commercial_use_allowed:
            raise RightsError("rights attestation does not permit commercial production use")
        if not self.approved_by.strip() or self.approved_at.tzinfo is None:
            raise RightsError(
                "rights attestation requires an approver and timezone-aware approval date"
            )
        if (self.model.model_id, self.model.revision) != (model_id, model_revision):
            raise RightsError("rights attestation does not match the configured model revision")
        if self.model.license_spdx != EXPECTED_MODEL_LICENSE_SPDX:
            raise RightsError("rights attestation does not match the pinned model license")
        if not self.model.evidence_url.startswith("https://"):
            raise RightsError("model rights evidence is incomplete")

        sources = {source.source_id: source for source in self.data_sources}
        uncovered = tuple(
            source_id for source_id in reference_source_ids if source_id not in sources
        )
        if uncovered:
            raise RightsError(f"reference source is not covered by the attestation: {uncovered[0]}")
        denied = tuple(
            source_id
            for source_id in reference_source_ids
            if not sources[source_id].commercial_use_allowed
            or not sources[source_id].license_or_permission.strip()
            or not sources[source_id].evidence_url.startswith("https://")
        )
        if denied:
            raise RightsError(f"reference source lacks commercial production rights: {denied[0]}")


def rights_attestation_from_dict(payload: dict[str, Any]) -> RightsAttestation:
    model = payload["model"]
    sources = payload["data_sources"]
    return RightsAttestation(
        schema_version=int(payload["schema_version"]),
        approved_by=str(payload["approved_by"]),
        approved_at=datetime.fromisoformat(str(payload["approved_at"])),
        commercial_use_allowed=_require_bool(payload["commercial_use_allowed"]),
        model=ModelRights(
            model_id=str(model["model_id"]),
            revision=str(model["revision"]),
            license_spdx=str(model["license_spdx"]),
            evidence_url=str(model["evidence_url"]),
            commercial_use_allowed=_require_bool(model["commercial_use_allowed"]),
        ),
        data_sources=tuple(
            DataRights(
                source_id=str(source["source_id"]),
                license_or_permission=str(source["license_or_permission"]),
                evidence_url=str(source["evidence_url"]),
                commercial_use_allowed=_require_bool(source["commercial_use_allowed"]),
            )
            for source in sources
        ),
    )


def _require_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError("rights commercial-use flags must be booleans")
    return value
