"""Single-flight inference guard for bounded CPU service resource use."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any, Protocol


class InferenceRuntime(Protocol):
    def identify(self, image: Any, *, limit: int = 3) -> dict[str, Any]: ...


class InferenceBusyError(RuntimeError):
    """Raised when an earlier CPU inference is still running."""


class BoundedInferenceRunner:
    """Permit one inference at a time and own every submitted image lifecycle."""

    def __init__(self, runtime: InferenceRuntime) -> None:
        self._runtime = runtime
        self._single_flight = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fluke-inference")

    @property
    def busy(self) -> bool:
        return self._single_flight.locked()

    def run(self, image: Any) -> dict[str, Any]:
        self._claim(image)
        return self._run_owned(image)

    def submit(self, image: Any) -> Future[dict[str, Any]]:
        self._claim(image)
        try:
            return self._executor.submit(self._run_owned, image)
        except RuntimeError:
            try:
                image.close()
            finally:
                self._single_flight.release()
            raise

    def _claim(self, image: Any) -> None:
        if not self._single_flight.acquire(blocking=False):
            try:
                image.close()
            finally:
                raise InferenceBusyError("identifier is busy")

    def _run_owned(self, image: Any) -> dict[str, Any]:
        try:
            return self._runtime.identify(image)
        finally:
            try:
                image.close()
            finally:
                self._single_flight.release()
