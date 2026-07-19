"""Canonical corpus evidence and independently recomputable retrieval decisions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import numpy as np

_SCHEMA_VERSION = 1
_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_SHA256_LENGTH = 64
FIXTURE_ROLES = frozenset(
    {
        "reference",
        "parity",
        "closedSetRetrieval",
        "openSet",
        "nonOrca",
        "poorQuality",
        "occlusion",
        "distributionShift",
    }
)
OPEN_EVALUATION_TYPES = (
    "openSet",
    "nonOrca",
    "poorQuality",
    "occlusion",
    "distributionShift",
)
_MANIFEST_KEYS = {"schemaVersion", "evidencePurpose", "provenanceUrl", "rows"}
_ROW_KEYS = {
    "fixtureId",
    "relativePath",
    "imageSha256",
    "roles",
    "referencePhotoId",
    "whaleId",
    "catalogId",
    "sourceId",
}
_DECISIONS_KEYS = {"schemaVersion", "scoreThreshold", "marginThreshold", "records"}
_DECISION_KEYS = {
    "accepted",
    "evaluationType",
    "fixtureId",
    "rankedWhaleIds",
    "secondScore",
    "topScore",
    "truthWhaleId",
}


@dataclass(frozen=True)
class FixtureRow:
    """One digest-bound corpus image and its approved evaluation roles."""

    fixture_id: str
    relative_path: str
    image_sha256: str
    roles: tuple[str, ...]
    reference_photo_id: str | None
    whale_id: str | None
    catalog_id: str | None
    source_id: str | None
    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "roles", tuple(self.roles))


@dataclass(frozen=True)
class CorpusManifest:
    """Validated immutable corpus manifest with locally resolved image paths."""

    purpose: str
    provenance_url: str
    rows: tuple[FixtureRow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))


@dataclass(frozen=True)
class DecisionRecord:
    """Raw per-query retrieval output sufficient to recompute release metrics."""

    fixture_id: str
    evaluation_type: str
    truth_whale_id: str | None
    ranked_whale_ids: tuple[str, ...]
    top_score: float
    second_score: float
    accepted: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranked_whale_ids", tuple(self.ranked_whale_ids))
        _validate_decision(self)


def load_corpus_manifest(path: Path, corpus_root: Path) -> CorpusManifest:
    """Read an exact-schema manifest and verify every named image byte-for-byte."""
    manifest_path = _regular_file(path, "corpus manifest")
    root = Path(corpus_root)
    _reject_symlink_components(root, "corpus root")
    if not root.is_dir():
        raise ValueError("corpus root must be a regular directory")
    payload = _load_bounded_json(manifest_path, "corpus manifest")
    if not isinstance(payload, dict) or set(payload) != _MANIFEST_KEYS:
        raise ValueError("corpus manifest fields do not match the exact schema")
    if payload["schemaVersion"] != _SCHEMA_VERSION or isinstance(payload["schemaVersion"], bool):
        raise ValueError("corpus manifest schemaVersion must be the integer 1")
    purpose = _nonempty_text(payload["evidencePurpose"], "evidencePurpose")
    provenance = _https_url(payload["provenanceUrl"], "provenanceUrl")
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("corpus manifest rows must be a non-empty array")
    rows = tuple(_fixture_row(value, root) for value in raw_rows)
    _reject_duplicate_rows(rows)
    return CorpusManifest(purpose=purpose, provenance_url=provenance, rows=rows)


def canonical_fixture_payload(rows: tuple[FixtureRow, ...]) -> bytes:
    """Serialize identity and image-byte digests independently of host paths/order."""
    ordered = sorted(rows, key=lambda row: row.fixture_id)
    payload = {
        "schemaVersion": _SCHEMA_VERSION,
        "rows": [_fixture_payload(row) for row in ordered],
    }
    return _canonical_json(payload)


def fixture_set_sha256(rows: tuple[FixtureRow, ...]) -> str:
    """Hash the canonical fixture records, which already bind actual image bytes."""
    return hashlib.sha256(canonical_fixture_payload(rows)).hexdigest()


def decision_payload(record: DecisionRecord) -> dict[str, object]:
    """Return the exact client-independent raw decision schema."""
    return {
        "accepted": record.accepted,
        "evaluationType": record.evaluation_type,
        "fixtureId": record.fixture_id,
        "rankedWhaleIds": list(record.ranked_whale_ids),
        "secondScore": record.second_score,
        "topScore": record.top_score,
        "truthWhaleId": record.truth_whale_id,
    }


def canonical_decisions_payload(
    decisions: tuple[DecisionRecord, ...], *, score_threshold: float, margin_threshold: float
) -> bytes:
    """Serialize thresholds and raw decisions deterministically."""
    _threshold(score_threshold, "score threshold")
    _threshold(margin_threshold, "margin threshold")
    ordered = sorted(decisions, key=lambda item: (item.evaluation_type, item.fixture_id))
    if len({(item.evaluation_type, item.fixture_id) for item in ordered}) != len(ordered):
        raise ValueError("raw decisions contain duplicate evaluation fixture identities")
    payload = {
        "schemaVersion": _SCHEMA_VERSION,
        "scoreThreshold": float(score_threshold),
        "marginThreshold": float(margin_threshold),
        "records": [decision_payload(item) for item in ordered],
    }
    return _canonical_json(payload)


def load_published_fixture_rows(path: Path) -> tuple[FixtureRow, ...]:
    """Load the canonical release fixture manifest without requiring source images."""
    source = _regular_file(path, "fixture manifest")
    payload = _load_bounded_json(source, "fixture manifest")
    if not isinstance(payload, dict) or set(payload) != {"schemaVersion", "rows"}:
        raise ValueError("fixture manifest fields do not match the exact schema")
    if payload["schemaVersion"] != _SCHEMA_VERSION or isinstance(payload["schemaVersion"], bool):
        raise ValueError("fixture manifest schemaVersion must be the integer 1")
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("fixture manifest rows must be a non-empty array")
    rows = tuple(_published_fixture_row(value) for value in raw_rows)
    _reject_duplicate_rows(rows)
    if source.read_bytes() != canonical_fixture_payload(rows):
        raise ValueError("fixture manifest must use canonical JSON and row ordering")
    return rows


def load_raw_decisions(
    path: Path,
) -> tuple[tuple[DecisionRecord, ...], float, float]:
    """Load and canonicalize raw decisions and their launch-approved thresholds."""
    source = _regular_file(path, "raw decisions")
    payload = _load_bounded_json(source, "raw decisions")
    if not isinstance(payload, dict) or set(payload) != _DECISIONS_KEYS:
        raise ValueError("raw decision fields do not match the exact schema")
    if payload["schemaVersion"] != _SCHEMA_VERSION or isinstance(payload["schemaVersion"], bool):
        raise ValueError("raw decisions schemaVersion must be the integer 1")
    score = _threshold(payload["scoreThreshold"], "raw decision scoreThreshold")
    margin = _threshold(payload["marginThreshold"], "raw decision marginThreshold")
    raw_records = payload["records"]
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("raw decision records must be a non-empty array")
    decisions = tuple(_decision_record(value) for value in raw_records)
    canonical = canonical_decisions_payload(
        decisions, score_threshold=score, margin_threshold=margin
    )
    if source.read_bytes() != canonical:
        raise ValueError("raw decisions must use canonical JSON and record ordering")
    return decisions, score, margin


def recompute_metrics(
    decisions: tuple[DecisionRecord, ...], *, score_threshold: float, margin_threshold: float
) -> dict[str, dict[str, int | float]]:
    """Recompute closed-set accuracy and each false-accept rate from raw decisions."""
    _threshold(score_threshold, "score threshold")
    _threshold(margin_threshold, "margin threshold")
    grouped: dict[str, list[DecisionRecord]] = {}
    for record in decisions:
        score = np.float32(record.top_score)
        expected_acceptance = bool(score >= np.float32(score_threshold))
        if expected_acceptance and len(record.ranked_whale_ids) > 1:
            margin = score - np.float32(record.second_score) + np.finfo(np.float32).eps
            expected_acceptance = bool(margin >= np.float32(margin_threshold))
        if record.accepted != expected_acceptance:
            raise ValueError(
                f"raw decision acceptance does not match thresholds: {record.fixture_id}"
            )
        grouped.setdefault(record.evaluation_type, []).append(record)
    result: dict[str, dict[str, int | float]] = {}
    closed = grouped.get("closedSetRetrieval", [])
    if closed:
        top1 = sum(
            item.accepted and item.ranked_whale_ids[0] == item.truth_whale_id for item in closed
        )
        top3 = sum(
            item.accepted and item.truth_whale_id in item.ranked_whale_ids[:3] for item in closed
        )
        result["closedSetRetrieval"] = {
            "sampleCount": len(closed),
            "top1": top1 / len(closed),
            "top3": top3 / len(closed),
        }
    for evaluation_type in OPEN_EVALUATION_TYPES:
        cohort = grouped.get(evaluation_type, [])
        if cohort:
            accepted = sum(item.accepted for item in cohort)
            result[evaluation_type] = {
                "sampleCount": len(cohort),
                "falseAcceptRate": accepted / len(cohort),
            }
    return result


def _fixture_row(value: object, root: Path) -> FixtureRow:
    if not isinstance(value, dict) or set(value) != _ROW_KEYS:
        raise ValueError("corpus manifest row fields do not match the exact schema")
    fixture_id = _nonempty_text(value["fixtureId"], "fixtureId")
    relative = _canonical_relative_path(value["relativePath"])
    digest = _sha256(value["imageSha256"], "imageSha256")
    roles = _roles(value["roles"])
    image_path = root / relative
    _regular_file(image_path, f"fixture image {fixture_id}")
    if _sha256_file(image_path) != digest:
        raise ValueError(f"fixture image digest does not match manifest: {fixture_id}")
    identity = {
        name: _optional_text(value[key], key)
        for name, key in (
            ("reference_photo_id", "referencePhotoId"),
            ("whale_id", "whaleId"),
            ("catalog_id", "catalogId"),
            ("source_id", "sourceId"),
        )
    }
    if "reference" in roles and any(item is None for item in identity.values()):
        raise ValueError("reference fixtures require all catalog identity fields")
    if "closedSetRetrieval" in roles and identity["whale_id"] is None:
        raise ValueError("closed-set fixtures require whaleId")
    return FixtureRow(
        fixture_id=fixture_id,
        relative_path=relative,
        image_sha256=digest,
        roles=roles,
        path=image_path,
        **identity,
    )


def _published_fixture_row(value: object) -> FixtureRow:
    if not isinstance(value, dict) or set(value) != _ROW_KEYS:
        raise ValueError("fixture manifest row fields do not match the exact schema")
    fixture_id = _nonempty_text(value["fixtureId"], "fixtureId")
    relative = _canonical_relative_path(value["relativePath"])
    roles = _roles(value["roles"])
    identity = {
        name: _optional_text(value[key], key)
        for name, key in (
            ("reference_photo_id", "referencePhotoId"),
            ("whale_id", "whaleId"),
            ("catalog_id", "catalogId"),
            ("source_id", "sourceId"),
        )
    }
    return FixtureRow(
        fixture_id=fixture_id,
        relative_path=relative,
        image_sha256=_sha256(value["imageSha256"], "imageSha256"),
        roles=roles,
        path=Path(relative),
        **identity,
    )


def _decision_record(value: object) -> DecisionRecord:
    if not isinstance(value, dict) or set(value) != _DECISION_KEYS:
        raise ValueError("raw decision record fields do not match the exact schema")
    ranking = value["rankedWhaleIds"]
    if not isinstance(ranking, list):
        raise ValueError("raw decision rankedWhaleIds must be an array")
    return DecisionRecord(
        fixture_id=value["fixtureId"],
        evaluation_type=value["evaluationType"],
        truth_whale_id=value["truthWhaleId"],
        ranked_whale_ids=tuple(ranking),
        top_score=value["topScore"],
        second_score=value["secondScore"],
        accepted=value["accepted"],
    )


def _fixture_payload(row: FixtureRow) -> dict[str, object]:
    return {
        "catalogId": row.catalog_id,
        "fixtureId": row.fixture_id,
        "imageSha256": row.image_sha256,
        "referencePhotoId": row.reference_photo_id,
        "relativePath": row.relative_path,
        "roles": list(row.roles),
        "sourceId": row.source_id,
        "whaleId": row.whale_id,
    }


def _validate_decision(record: DecisionRecord) -> None:
    _nonempty_text(record.fixture_id, "decision fixtureId")
    if record.evaluation_type not in {"closedSetRetrieval", *OPEN_EVALUATION_TYPES}:
        raise ValueError("raw decision evaluationType is unsupported")
    if not record.ranked_whale_ids or any(
        not isinstance(value, str) or not value.strip() for value in record.ranked_whale_ids
    ):
        raise ValueError("raw decision rankedWhaleIds must contain non-empty strings")
    if len(set(record.ranked_whale_ids)) != len(record.ranked_whale_ids):
        raise ValueError("raw decision rankedWhaleIds must be unique")
    if record.evaluation_type == "closedSetRetrieval":
        _nonempty_text(record.truth_whale_id, "closed-set truthWhaleId")
    elif record.truth_whale_id is not None:
        raise ValueError("open-cohort truthWhaleId must be null")
    _finite_score(record.top_score, "raw decision topScore")
    _finite_score(record.second_score, "raw decision secondScore")
    if record.second_score > record.top_score:
        raise ValueError("raw decision secondScore must not exceed topScore")
    if not isinstance(record.accepted, bool):
        raise ValueError("raw decision accepted must be a boolean")


def _reject_duplicate_rows(rows: tuple[FixtureRow, ...]) -> None:
    identities = (
        ("fixtureId", tuple(row.fixture_id for row in rows)),
        ("relativePath", tuple(row.relative_path for row in rows)),
        ("imageSha256", tuple(row.image_sha256 for row in rows)),
    )
    for name, values in identities:
        if len(set(values)) != len(values):
            raise ValueError(f"corpus manifest contains duplicate {name}")
    reference_ids = tuple(
        row.reference_photo_id for row in rows if row.reference_photo_id is not None
    )
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("corpus manifest contains duplicate referencePhotoId")
    if any("reference" in row.roles and len(row.roles) != 1 for row in rows):
        raise ValueError("reference fixtures must be disjoint from all evaluation roles")


def _canonical_relative_path(value: object) -> str:
    text = _nonempty_text(value, "relativePath")
    candidate = PurePosixPath(text)
    invalid = (
        candidate.is_absolute()
        or text != candidate.as_posix()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    )
    if invalid:
        raise ValueError("relativePath must be a canonical relative POSIX path")
    return text


def _roles(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("roles must be a non-empty array")
    if any(not isinstance(role, str) or role not in FIXTURE_ROLES for role in value):
        raise ValueError("roles contain an unsupported fixture role")
    roles = tuple(sorted(value))
    if len(set(roles)) != len(roles):
        raise ValueError("roles must not contain duplicates")
    return roles


def _regular_file(path: Path, name: str) -> Path:
    candidate = Path(path)
    _reject_symlink_components(candidate, name)
    if not candidate.is_file():
        raise ValueError(f"{name} must be a regular file")
    return candidate


def _reject_symlink_components(path: Path, name: str) -> None:
    absolute = path.absolute()
    for component in (*reversed(absolute.parents), absolute):
        if component.is_symlink():
            raise ValueError(f"{name} path contains a symbolic link component")


def _load_bounded_json(path: Path, name: str) -> object:
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError(f"{name} exceeds the maximum size")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as error:
        raise ValueError(f"{name} is invalid JSON: {error}") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        + "\n"
    ).encode("utf-8")


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_text(value, name)


def _sha256(value: object, name: str) -> str:
    text = _nonempty_text(value, name)
    if len(text) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return text


def _https_url(value: object, name: str) -> str:
    text = _nonempty_text(value, name)
    parsed = urlsplit(text)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username is not None:
        raise ValueError(f"{name} must be an absolute HTTPS URL")
    return text


def _threshold(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and within [-1, 1]")
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and within [-1, 1]")
    return result


def _finite_score(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result
