"""Rights attestation and SSRF boundary tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from fluke_model.network import (
    NetworkPolicyError,
    validate_reference_addresses,
    validate_reference_url,
)
from fluke_model.rights import DataRights, ModelRights, RightsAttestation, RightsError


def _attestation(*, commercial: bool = True, license_spdx: str = "Apache-2.0") -> RightsAttestation:
    return RightsAttestation(
        schema_version=1,
        approved_by="Fluke launch owner",
        approved_at=datetime.now(timezone.utc),
        commercial_use_allowed=commercial,
        model=ModelRights(
            model_id="facebook/dinov2-small",
            revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
            license_spdx=license_spdx,
            evidence_url="https://github.com/facebookresearch/dinov2/blob/main/LICENSE",
            commercial_use_allowed=commercial,
        ),
        data_sources=(
            DataRights(
                source_id="owned-catalog",
                license_or_permission="written-owner-permission",
                evidence_url="https://fluke.example/legal/catalog-rights/1",
                commercial_use_allowed=commercial,
            ),
        ),
    )


def test_rights_gate_accepts_only_matching_written_production_attestation() -> None:
    _attestation().validate_for(
        model_id="facebook/dinov2-small",
        model_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        reference_source_ids=("owned-catalog",),
    )

    with pytest.raises(RightsError, match="commercial production use"):
        _attestation(commercial=False).validate_for(
            model_id="facebook/dinov2-small",
            model_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
            reference_source_ids=("owned-catalog",),
        )
    with pytest.raises(RightsError, match="not covered"):
        _attestation().validate_for(
            model_id="facebook/dinov2-small",
            model_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
            reference_source_ids=("unknown-source",),
        )
    with pytest.raises(RightsError, match="license"):
        _attestation(license_spdx="Proprietary").validate_for(
            model_id="facebook/dinov2-small",
            model_revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
            reference_source_ids=("owned-catalog",),
        )


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://images.example.org/ref.jpg",
        "https://127.0.0.1/ref.jpg",
        "https://169.254.169.254/latest/meta-data",
        "https://user:pass@images.example.org/ref.jpg",
        "https://images.example.org:8443/ref.jpg",
        "https://unapproved.example/ref.jpg",
    ],
)
def test_reference_url_policy_blocks_local_non_tls_and_unapproved_targets(url: str) -> None:
    with pytest.raises(NetworkPolicyError):
        validate_reference_url(url, frozenset({"images.example.org"}))


def test_reference_url_policy_accepts_exact_allowlisted_https_host() -> None:
    parsed = validate_reference_url(
        "https://images.example.org/catalog/ref.jpg", frozenset({"images.example.org"})
    )

    assert parsed.hostname == "images.example.org"


def test_dns_resolution_fails_if_any_answer_is_not_public() -> None:
    with pytest.raises(NetworkPolicyError, match="public internet"):
        validate_reference_addresses(("93.184.216.34", "127.0.0.1"))

    validate_reference_addresses(("93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"))
