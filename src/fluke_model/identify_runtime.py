"""Rights-gated reference-index build and retrieval runtime."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Callable
from uuid import uuid4

import numpy as np
from PIL import Image

from fluke_model.deadline import OperationDeadline
from fluke_model.embedders import (
    DINO_V2_MODEL_ID,
    DINO_V2_REVISION,
    LoadedEmbedder,
    load_embedder,
)
from fluke_model.index import IndexBundle, build_index, load_index, save_index, search
from fluke_model.index_store import AtomicIndexStore
from fluke_model.rights import RightsAttestation, rights_attestation_from_dict

DEFAULT_MODEL_NAME = "dinov2-small"
REQUIRED_REFERENCE_METADATA = frozenset(
    {"referencePhotoId", "catalogId", "url", "rightsSourceId", "side", "quality"}
)


@dataclass(frozen=True)
class ReferencePhoto:
    reference_photo_id: str
    catalog_id: str
    name: str | None
    url: str
    rights_source_id: str
    side: str = "UNKNOWN"
    quality: str = "USABLE"
    crop: dict[str, float] | None = None


@dataclass(frozen=True)
class RuntimeMatch:
    catalog_id: str
    name: str | None
    score: float
    rank: int
    matched_reference_photo_ids: tuple[str, ...]
    explanation: str


def maybe_crop(image: Image.Image, crop: dict[str, float] | None) -> Image.Image:
    if not crop:
        return image
    required = ("x", "y", "width", "height")
    if any(crop.get(key) is None for key in required):
        raise ValueError("crop must include x, y, width, and height")
    x = float(crop["x"])
    y = float(crop["y"])
    width = float(crop["width"])
    height = float(crop["height"])
    values = (x, y, width, height)
    if (
        not all(math.isfinite(value) for value in values)
        or x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image.width
        or y + height > image.height
    ):
        raise ValueError("crop must stay within the validated source image")
    return image.crop((x, y, x + width, y + height))


def reference_from_payload(payload: dict[str, Any]) -> ReferencePhoto:
    return ReferencePhoto(
        reference_photo_id=str(payload["referencePhotoId"]),
        catalog_id=str(payload["catalogId"]),
        name=payload.get("name"),
        url=str(payload["url"]),
        rights_source_id=str(payload["rightsSourceId"]),
        side=str(payload.get("side") or "UNKNOWN"),
        quality=str(payload.get("quality") or "USABLE"),
        crop=payload.get("crop"),
    )


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def build_reference_index(
    references: list[ReferencePhoto],
    *,
    store: AtomicIndexStore,
    rights: RightsAttestation,
    embedder: LoadedEmbedder,
    image_loader: Callable[[ReferencePhoto], Image.Image],
    batch_size: int = 1,
    max_total_pixels: int = 500_000_000,
    deadline: OperationDeadline | None = None,
    publication_guard: Callable[[], None] | None = None,
    publish_version: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    operation_deadline = deadline or OperationDeadline.never()
    operation_deadline.check()
    guard = publication_guard or operation_deadline.check
    guard()
    if not references:
        raise ValueError("At least one reference photo is required to build an index")
    rights.validate_for(
        model_id=DINO_V2_MODEL_ID,
        model_revision=DINO_V2_REVISION,
        reference_source_ids=tuple(reference.rights_source_id for reference in references),
    )
    if embedder.name != DEFAULT_MODEL_NAME:
        raise ValueError("configured embedder does not match the production rights gate")

    if batch_size < 1 or max_total_pixels < 1:
        raise ValueError("reference build limits must be positive")

    metadata: list[dict[str, Any]] = []
    embedding_batches: list[np.ndarray] = []
    image_batch: list[Image.Image] = []
    total_pixels = 0
    try:
        for reference in references:
            operation_deadline.check()
            try:
                image = _load_reference_image(reference, image_loader)
            except OSError as exc:
                raise ValueError(
                    "All reference photos must load before an index can be published"
                ) from exc
            total_pixels += image.width * image.height
            if total_pixels > max_total_pixels:
                image.close()
                raise ValueError("reference images exceed the aggregate pixel limit")
            image_batch.append(image)
            metadata.append(_reference_metadata(reference))
            if len(image_batch) >= batch_size:
                batch, image_batch = image_batch, []
                embedding_batches.append(_embed_batch(embedder, batch, operation_deadline))
        if image_batch:
            batch, image_batch = image_batch, []
            embedding_batches.append(_embed_batch(embedder, batch, operation_deadline))
    finally:
        for image in image_batch:
            image.close()

    embeddings = np.concatenate(embedding_batches, axis=0)
    if (
        embeddings.ndim != 2
        or embeddings.shape[0] != len(metadata)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("embedder returned invalid reference vectors")

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
    version_dir = store.create_version(version)
    bundle = build_index(embeddings, metadata, embedder.name)
    save_index(bundle, version_dir)
    created_at = datetime.now(timezone.utc).isoformat()
    info = {
        "indexVersion": version,
        "model": embedder.name,
        "modelId": DINO_V2_MODEL_ID,
        "modelRevision": DINO_V2_REVISION,
        "embedDim": embedder.embed_dim,
        "referenceCount": len(metadata),
        "createdAt": created_at,
    }
    (version_dir / "index_info.json").write_text(json.dumps(info, indent=2))
    (version_dir / "rights.json").write_text(
        json.dumps(asdict(rights), indent=2, default=_json_default)
    )
    with (version_dir / "manifest.jsonl").open("w") as stream:
        for row in metadata:
            stream.write(json.dumps(row) + "\n")
    operation_deadline.check()
    guard()
    publisher = publish_version or store.publish
    publisher(version_dir)
    return {
        "ok": True,
        "indexVersion": version,
        "embeddedReferencePhotoIds": tuple(row["referencePhotoId"] for row in metadata),
        "failedReferencePhotoIds": (),
    }


def _embed_batch(
    embedder: LoadedEmbedder,
    images: list[Image.Image],
    deadline: OperationDeadline,
) -> np.ndarray:
    deadline.check()
    try:
        embeddings = embedder.embed_fn(images)
    finally:
        for image in images:
            image.close()
    deadline.check()
    if (
        embeddings.ndim != 2
        or embeddings.shape != (len(images), embedder.embed_dim)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError("embedder returned invalid reference vectors")
    return embeddings


def _load_reference_image(
    reference: ReferencePhoto,
    image_loader: Callable[[ReferencePhoto], Image.Image],
) -> Image.Image:
    source = image_loader(reference)
    try:
        image = maybe_crop(source, reference.crop)
    except BaseException:
        source.close()
        raise
    if image is not source:
        source.close()
    return image


def _reference_metadata(reference: ReferencePhoto) -> dict[str, Any]:
    return {
        "referencePhotoId": reference.reference_photo_id,
        "catalogId": reference.catalog_id,
        "name": reference.name,
        "url": reference.url,
        "rightsSourceId": reference.rights_source_id,
        "side": reference.side,
        "quality": reference.quality,
    }


def aggregate_hits(
    hits: list[tuple[float, dict[str, Any]]], *, limit: int = 3
) -> list[RuntimeMatch]:
    grouped: dict[str, dict[str, Any]] = {}
    for score, metadata in hits:
        catalog_id = str(metadata["catalogId"])
        group = grouped.setdefault(
            catalog_id,
            {"name": metadata.get("name"), "scores": [], "reference_ids": []},
        )
        group["scores"].append(float(score))
        group["reference_ids"].append(str(metadata["referencePhotoId"]))

    ranked: list[tuple[str, dict[str, Any], float]] = []
    for catalog_id, group in grouped.items():
        score = float(np.mean(sorted(group["scores"], reverse=True)[:3]))
        ranked.append((catalog_id, group, score))
    ranked.sort(key=lambda value: value[2], reverse=True)
    return [
        RuntimeMatch(
            catalog_id=catalog_id,
            name=group["name"],
            score=score,
            rank=rank,
            matched_reference_photo_ids=tuple(group["reference_ids"][:5]),
            explanation="Nearest visual references; human confirmation is required.",
        )
        for rank, (catalog_id, group, score) in enumerate(ranked[:limit], start=1)
    ]


class IdentifierRuntime:
    def __init__(
        self,
        *,
        index_store: AtomicIndexStore,
        embedder: LoadedEmbedder | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        model_artifact_dir: Path | None = None,
    ) -> None:
        self.index_store = index_store
        self.model_name = model_name
        self.model_artifact_dir = model_artifact_dir
        self._embedder = embedder
        self._bundle: IndexBundle | None = None
        self._index_info: dict[str, Any] = {}
        self._loaded_version: str | None = None
        self._validated_model_version: str | None = None
        self._lock = RLock()

    @property
    def embedder(self) -> LoadedEmbedder:
        with self._lock:
            if self._embedder is None:
                self._embedder = load_embedder(
                    self.model_name,
                    artifact_dir=self.model_artifact_dir,
                )
            return self._embedder

    def readiness(self) -> tuple[bool, str]:
        try:
            self.load()
        except (
            FileNotFoundError,
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            KeyError,
        ):
            return (False, "index_unavailable")
        try:
            self._validate_model()
        except (OSError, RuntimeError, ValueError):
            return (False, "model_unavailable")
        return (True, "ready")

    def _validate_model(self) -> None:
        with self._lock:
            if self._loaded_version == self._validated_model_version:
                return
            if self._bundle is None or self._loaded_version is None:
                raise ValueError("reference index is unavailable")
            probe = Image.new("RGB", (224, 224), color=(0, 0, 0))
            try:
                embeddings = self.embedder.embed_fn([probe])
            finally:
                probe.close()
            if embeddings.shape != (1, self._bundle.embed_dim) or not np.isfinite(embeddings).all():
                raise ValueError("configured model failed its readiness probe")
            self._validated_model_version = self._loaded_version

    def load(self) -> None:
        with self._lock:
            version_dir = self.index_store.current_version_dir()
            if self._loaded_version == version_dir.name and self._bundle is not None:
                return
            info = json.loads((version_dir / "index_info.json").read_text())
            if (
                info.get("modelId") != DINO_V2_MODEL_ID
                or info.get("modelRevision") != DINO_V2_REVISION
            ):
                raise ValueError("published index model rights do not match runtime")
            bundle = load_index(version_dir)
            if bundle.embedder_name != self.model_name:
                raise ValueError("published index embedder does not match runtime")
            _validate_index_artifacts(version_dir, bundle, info)
            rights_payload = json.loads((version_dir / "rights.json").read_text())
            rights = rights_attestation_from_dict(rights_payload)
            rights.validate_for(
                model_id=DINO_V2_MODEL_ID,
                model_revision=DINO_V2_REVISION,
                reference_source_ids=tuple(
                    str(metadata["rightsSourceId"]) for metadata in bundle.metadata
                ),
            )
            self._bundle = bundle
            self._index_info = info
            self._loaded_version = version_dir.name

    def identify(self, image: Image.Image, *, limit: int = 3) -> dict[str, Any]:
        if limit < 1 or limit > 10:
            raise ValueError("limit must be between 1 and 10")
        self.load()
        self._validate_model()
        with self._lock:
            if self._bundle is None:
                raise FileNotFoundError("reference index is unavailable")
            embedding = self.embedder.embed_fn([image])[0]
            hits = search(self._bundle, embedding, k=min(25, self._bundle.index.ntotal))
            matches = aggregate_hits(hits, limit=limit)
            info = dict(self._index_info)
            model = self._bundle.embedder_name
        return {
            "matches": [
                {
                    "catalogId": match.catalog_id,
                    "name": match.name,
                    "score": round(match.score, 4),
                    "rank": match.rank,
                    "matchedReferencePhotoIds": match.matched_reference_photo_ids,
                    "explanation": match.explanation,
                }
                for match in matches
            ],
            "confidenceBand": "unavailable",
            "confidenceSemantics": "uncalibrated_similarity_not_probability",
            "model": model,
            "indexVersion": info["indexVersion"],
        }


def _validate_index_artifacts(
    version_dir: Path,
    bundle: IndexBundle,
    info: dict[str, Any],
) -> None:
    reference_count = info.get("referenceCount")
    embed_dim = info.get("embedDim")
    if info.get("indexVersion") != version_dir.name:
        raise ValueError("published index version identity is inconsistent")
    if info.get("model") != bundle.embedder_name:
        raise ValueError("published index embedder identity is inconsistent")
    if not isinstance(reference_count, int) or isinstance(reference_count, bool):
        raise ValueError("published index reference count is invalid")
    if not isinstance(embed_dim, int) or isinstance(embed_dim, bool):
        raise ValueError("published index embedding dimension is invalid")
    if reference_count < 1 or reference_count != len(bundle.metadata):
        raise ValueError("published index reference count is inconsistent")
    if bundle.index.ntotal != reference_count:
        raise ValueError("published FAISS row count is inconsistent")
    if bundle.embed_dim != embed_dim or bundle.index.d != embed_dim:
        raise ValueError("published index embedding dimensions are inconsistent")
    _validate_reference_metadata(bundle.metadata)


def _validate_reference_metadata(metadata: list[dict[str, Any]]) -> None:
    reference_ids: set[str] = set()
    for row in metadata:
        if not isinstance(row, dict) or not REQUIRED_REFERENCE_METADATA.issubset(row):
            raise ValueError("published reference metadata is incomplete")
        required_values = tuple(row[key] for key in REQUIRED_REFERENCE_METADATA)
        if any(not isinstance(value, str) or not value.strip() for value in required_values):
            raise ValueError("published reference metadata is invalid")
        reference_id = row["referencePhotoId"]
        if reference_id in reference_ids:
            raise ValueError("published reference metadata IDs must be unique")
        reference_ids.add(reference_id)
