"""Authenticated, bounded FastAPI boundary for the identifier runtime."""

from __future__ import annotations

import asyncio
import secrets
from dataclasses import dataclass
from typing import Annotated, Any, Callable, Protocol

from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

from fluke_model.image_inputs import (
    ImageInputError,
    ImageTooLargeError,
    decode_base64_image,
    load_validated_image,
)
from fluke_model.inference import BoundedInferenceRunner, InferenceBusyError
from fluke_model.deadline import OperationCancelledError, OperationDeadline
from fluke_model.rate_limit import RateLimiter
from fluke_model.request_size import RequestSizeLimitMiddleware
from fluke_model.settings import ServiceSettings


class RuntimeProtocol(Protocol):
    def readiness(self) -> tuple[bool, str]: ...

    def identify(self, image: Any, *, limit: int = 3) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ServiceDependencies:
    runtime: RuntimeProtocol
    rebuild: Callable[[dict[str, Any], OperationDeadline], dict[str, Any]]


class IdentifyJsonRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    image_base64: str = Field(alias="imageBase64", min_length=1)
    content_type: str = Field(alias="contentType")


def create_app(settings: ServiceSettings, dependencies: ServiceDependencies) -> FastAPI:
    app = FastAPI(title="Fluke Identifier", version="1.0.0", docs_url=None, redoc_url=None)
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    api_key_header = APIKeyHeader(name="X-Fluke-Model-Key", auto_error=False)
    limiter = RateLimiter()
    inference_runner = BoundedInferenceRunner(dependencies.runtime)
    app.state.inference_runner = inference_runner

    def authenticate(provided: str | None = Security(api_key_header)) -> None:
        if provided is None or not secrets.compare_digest(provided, settings.api_key):
            raise HTTPException(status_code=401, detail="valid service credential required")

    auth = Depends(authenticate)

    def rate_limit(scope: str, limit: int) -> Callable[[Request], None]:
        def enforce(request: Request) -> None:
            client = request.client.host if request.client is not None else "unknown"
            if not limiter.allow(f"{scope}:{client}", limit=limit):
                raise HTTPException(status_code=429, detail="request rate limit exceeded")

        return enforce

    health_rate = Depends(rate_limit("health", settings.health_requests_per_minute))
    identify_rate = Depends(rate_limit("identify", settings.identify_requests_per_minute))
    rebuild_rate = Depends(rate_limit("rebuild", settings.rebuild_requests_per_minute))

    @app.get("/health", dependencies=[health_rate])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", dependencies=[health_rate])
    def ready() -> JSONResponse:
        is_ready, reason = dependencies.runtime.readiness()
        if not is_ready:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": reason},
            )
        return JSONResponse(content={"status": "ready"})

    async def run_identify(image: Any) -> dict[str, Any]:
        try:
            worker = asyncio.wrap_future(inference_runner.submit(image))
            return await asyncio.wait_for(
                asyncio.shield(worker),
                timeout=settings.inference_timeout_seconds,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="identifier timed out") from exc
        except InferenceBusyError as exc:
            raise HTTPException(status_code=503, detail="identifier is busy") from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail="reference index is unavailable") from exc

    @app.post("/identify-json", dependencies=[auth, identify_rate])
    async def identify_json(payload: IdentifyJsonRequest) -> dict[str, Any]:
        try:
            data = decode_base64_image(payload.image_base64, max_bytes=settings.max_image_bytes)
            image = load_validated_image(
                data,
                content_type=payload.content_type,
                max_bytes=settings.max_image_bytes,
                max_pixels=settings.max_image_pixels,
            )
        except ImageTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ImageInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return await run_identify(image)

    @app.post("/identify", dependencies=[auth, identify_rate])
    async def identify_upload(file: Annotated[UploadFile | None, File()] = None) -> dict[str, Any]:
        if file is None:
            raise HTTPException(status_code=400, detail="file is required")
        data = await file.read(settings.max_image_bytes + 1)
        try:
            image = load_validated_image(
                data,
                content_type=file.content_type,
                max_bytes=settings.max_image_bytes,
                max_pixels=settings.max_image_pixels,
            )
        except ImageTooLargeError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ImageInputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return await run_identify(image)

    @app.post("/rebuild-index", dependencies=[auth, rebuild_rate])
    async def rebuild_index(payload: dict[str, Any]) -> dict[str, Any]:
        deadline = OperationDeadline.after(settings.rebuild_timeout_seconds)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(dependencies.rebuild, payload, deadline),
                timeout=settings.rebuild_timeout_seconds,
            )
        except TimeoutError as exc:
            deadline.cancel()
            raise HTTPException(status_code=504, detail="index rebuild timed out") from exc
        except OperationCancelledError as exc:
            raise HTTPException(status_code=409, detail="index rebuild was cancelled") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
