"""ASGI request-body limit enforced before multipart or JSON parsing."""

from __future__ import annotations

from typing import Any

from starlette.responses import JSONResponse


class RequestTooLargeError(Exception):
    pass


class RequestSizeLimitMiddleware:
    def __init__(self, app: Any, *, max_bytes: int) -> None:
        self._app = app
        self._max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                await self._reject(scope, receive, send, 400, "invalid content length")
                return
            if declared_length < 0:
                await self._reject(scope, receive, send, 400, "invalid content length")
                return
            if declared_length > self._max_bytes:
                await self._reject(scope, receive, send, 413, "request body is too large")
                return

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            received += len(message.get("body", b""))
            if received > self._max_bytes:
                raise RequestTooLargeError
            return message

        try:
            await self._app(scope, limited_receive, send)
        except RequestTooLargeError:
            await self._reject(scope, receive, send, 413, "request body is too large")

    @staticmethod
    async def _reject(
        scope: dict[str, Any], receive: Any, send: Any, status_code: int, detail: str
    ) -> None:
        response = JSONResponse(status_code=status_code, content={"detail": detail})
        await response(scope, receive, send)
