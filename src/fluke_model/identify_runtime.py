"""MiewID + FAISS runtime for Fluke's V1 identifier.

This is retrieval, not model training. We embed curated whale reference photos
with MiewID-msv3, store those vectors in a FAISS index, then embed an uploaded
query image and aggregate nearest reference-image hits into whale-level matches.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from fluke_model.embedders import LoadedEmbedder, load_embedder
from fluke_model.index import IndexBundle, build_index, load_index, save_index, search

DEFAULT_INDEX_DIR = Path("artifacts/reference-index")
DEFAULT_MODEL_NAME = "miewid-msv3"


@dataclass(frozen=True)
class ReferencePhoto:
    reference_photo_id: str
    catalog_id: str
    name: str | None
    url: str
    side: str = "UNKNOWN"
    quality: str = "USABLE"
    crop: dict[str, float] | None = None


@dataclass(frozen=True)
class RuntimeMatch:
    catalog_id: str
    name: str | None
    score: float
    rank: int
    matched_reference_photo_ids: list[str]
    explanation: str


def load_image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def load_image_from_base64(data: str) -> Image.Image:
    return load_image_from_bytes(base64.b64decode(data))


def load_image_from_url(url: str, timeout: float = 20.0) -> Image.Image:
    """Load an image from a URL.

    Supports `http(s)://` and `file://` schemes. The `file://` path is meant
    for dev/demo mode where reference photos live on the local filesystem;
    production references live behind authenticated object-storage URLs.
    """
    if url.startswith("file://"):
        local_path = url[len("file://") :]
        with open(local_path, "rb") as f:
            return load_image_from_bytes(f.read())
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return load_image_from_bytes(response.content)


def maybe_crop(image: Image.Image, crop: dict[str, float] | None) -> Image.Image:
    if not crop:
        return image
    required = ("x", "y", "width", "height")
    if any(crop.get(key) is None for key in required):
        return image
    x = max(0, float(crop["x"]))
    y = max(0, float(crop["y"]))
    width = max(1, float(crop["width"]))
    height = max(1, float(crop["height"]))
    return image.crop((x, y, x + width, y + height))


def reference_from_payload(payload: dict[str, Any]) -> ReferencePhoto:
    return ReferencePhoto(
        reference_photo_id=str(payload["referencePhotoId"]),
        catalog_id=str(payload["catalogId"]),
        name=payload.get("name"),
        url=str(payload["url"]),
        side=str(payload.get("side") or "UNKNOWN"),
        quality=str(payload.get("quality") or "USABLE"),
        crop=payload.get("crop"),
    )


def build_reference_index(
    references: list[ReferencePhoto],
    *,
    out_dir: Path,
    embedder: LoadedEmbedder | None = None,
) -> dict[str, Any]:
    if not references:
        raise ValueError("At least one reference photo is required to build an index.")

    embedder = embedder or load_embedder(DEFAULT_MODEL_NAME)
    images: list[Image.Image] = []
    metadata: list[dict[str, Any]] = []
    failed: list[str] = []

    for ref in references:
        try:
            image = maybe_crop(load_image_from_url(ref.url), ref.crop)
        except Exception:
            failed.append(ref.reference_photo_id)
            continue
        images.append(image)
        metadata.append(
            {
                "referencePhotoId": ref.reference_photo_id,
                "catalogId": ref.catalog_id,
                "name": ref.name,
                "url": ref.url,
                "side": ref.side,
                "quality": ref.quality,
            }
        )

    if not images:
        raise ValueError("No reference photos could be loaded.")

    embeddings = embedder.embed_fn(images)
    bundle = build_index(embeddings, metadata, embedder.name)
    save_index(bundle, out_dir)

    index_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    info = {
        "indexVersion": index_version,
        "model": embedder.name,
        "embedDim": embedder.embed_dim,
        "referenceCount": len(metadata),
        "failedReferencePhotoIds": failed,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index_info.json").write_text(json.dumps(info, indent=2))
    with (out_dir / "manifest.jsonl").open("w") as f:
        for row in metadata:
            f.write(json.dumps(row) + "\n")

    return {
        "ok": True,
        "indexVersion": index_version,
        "embeddedReferencePhotoIds": [row["referencePhotoId"] for row in metadata],
        "failedReferencePhotoIds": failed,
    }


def aggregate_hits(hits: list[tuple[float, dict[str, Any]]], *, limit: int = 3) -> list[RuntimeMatch]:
    grouped: dict[str, dict[str, Any]] = {}
    for score, meta in hits:
        catalog_id = str(meta["catalogId"])
        group = grouped.setdefault(
            catalog_id,
            {
                "catalog_id": catalog_id,
                "name": meta.get("name"),
                "scores": [],
                "reference_ids": [],
            },
        )
        group["scores"].append(float(score))
        group["reference_ids"].append(str(meta["referencePhotoId"]))

    ranked: list[RuntimeMatch] = []
    for group in grouped.values():
        top_scores = sorted(group["scores"], reverse=True)[:3]
        score = float(np.mean(top_scores))
        ranked.append(
            RuntimeMatch(
                catalog_id=group["catalog_id"],
                name=group["name"],
                score=score,
                rank=0,
                matched_reference_photo_ids=group["reference_ids"][:5],
                explanation=(
                    f"Closest visual match across {len(group['reference_ids'])} reference "
                    f"photo{'s' if len(group['reference_ids']) != 1 else ''}."
                ),
            )
        )

    ranked.sort(key=lambda match: match.score, reverse=True)
    return [
        RuntimeMatch(
            catalog_id=match.catalog_id,
            name=match.name,
            score=match.score,
            rank=index + 1,
            matched_reference_photo_ids=match.matched_reference_photo_ids,
            explanation=match.explanation,
        )
        for index, match in enumerate(ranked[:limit])
    ]


def confidence_band(matches: list[RuntimeMatch]) -> str:
    if not matches:
        return "unavailable"
    top = matches[0].score
    margin = top - matches[1].score if len(matches) > 1 else top
    if top >= 0.78 and margin >= 0.04:
        return "high"
    if top >= 0.58:
        return "medium"
    return "low"


class IdentifierRuntime:
    def __init__(self, index_dir: Path = DEFAULT_INDEX_DIR, model_name: str = DEFAULT_MODEL_NAME):
        self.index_dir = index_dir
        self.model_name = model_name
        self._embedder: LoadedEmbedder | None = None
        self._bundle: IndexBundle | None = None
        self._index_info: dict[str, Any] = {}

    @property
    def embedder(self) -> LoadedEmbedder:
        if self._embedder is None:
            self._embedder = load_embedder(self.model_name)
        return self._embedder

    def load(self) -> None:
        self._bundle = load_index(self.index_dir)
        info_path = self.index_dir / "index_info.json"
        self._index_info = json.loads(info_path.read_text()) if info_path.exists() else {}

    def reload(self) -> None:
        self._bundle = None
        self.load()

    def identify(self, image: Image.Image, *, limit: int = 3) -> dict[str, Any]:
        if self._bundle is None:
            self.load()
        assert self._bundle is not None
        embedding = self.embedder.embed_fn([image])[0]
        hits = search(self._bundle, embedding, k=min(25, self._bundle.index.ntotal))
        matches = aggregate_hits(hits, limit=limit)
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
            "confidenceBand": confidence_band(matches),
            "model": self._bundle.embedder_name,
            "indexVersion": self._index_info.get("indexVersion", "unknown"),
        }
