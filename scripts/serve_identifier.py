#!/usr/bin/env python3
"""Local FastAPI service for Fluke's MiewID-powered identifier V1."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fluke_model.identify_runtime import (  # noqa: E402
    DEFAULT_INDEX_DIR,
    IdentifierRuntime,
    build_reference_index,
    load_image_from_base64,
    load_image_from_bytes,
    reference_from_payload,
)


class IdentifyJsonRequest(BaseModel):
    image_base64: str
    filename: str | None = None
    content_type: str | None = None


class ReferencePayload(BaseModel):
    referencePhotoId: str
    catalogId: str
    name: str | None = None
    url: str
    side: str = "UNKNOWN"
    quality: str = "USABLE"
    crop: dict[str, float | None] | None = None


class RebuildIndexRequest(BaseModel):
    references: list[ReferencePayload] = Field(min_length=1)


INDEX_DIR = Path(os.environ.get("FLUKE_REFERENCE_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
runtime = IdentifierRuntime(index_dir=INDEX_DIR)
app = FastAPI(title="Fluke Identifier", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexDir": str(INDEX_DIR)}


@app.post("/identify")
async def identify_upload(file: Annotated[UploadFile | None, File()] = None) -> dict:
    if file is None:
        raise HTTPException(status_code=400, detail="file is required")
    image = load_image_from_bytes(await file.read())
    try:
        return runtime.identify(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="reference index has not been built") from exc


@app.post("/identify-json")
def identify_json(payload: IdentifyJsonRequest) -> dict:
    image = load_image_from_base64(payload.image_base64)
    try:
        return runtime.identify(image)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="reference index has not been built") from exc


@app.post("/rebuild-index")
def rebuild_index(payload: RebuildIndexRequest) -> dict:
    references = [reference_from_payload(item.model_dump()) for item in payload.references]
    result = build_reference_index(references, out_dir=INDEX_DIR, embedder=runtime.embedder)
    runtime.reload()
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve_identifier:app", host="0.0.0.0", port=4100, reload=False)
