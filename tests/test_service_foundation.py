"""Security and launch-contract tests for the identifier service."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Event

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fluke_model.service import ServiceDependencies, create_app
from fluke_model.deadline import OperationCancelledError
from fluke_model.inference import BoundedInferenceRunner, InferenceBusyError
from fluke_model.settings import ServiceSettings


API_KEY = "test-key-that-is-at-least-thirty-two-bytes"


def _jpeg_bytes(size: tuple[int, int] = (12, 12)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size, color=(12, 34, 56)).save(output, format="JPEG")
    return output.getvalue()


@dataclass
class StubRuntime:
    ready: bool = True
    identify_calls: int = 0

    def readiness(self) -> tuple[bool, str]:
        return (self.ready, "ready" if self.ready else "index_unavailable")

    def identify(self, image: Image.Image, *, limit: int = 3) -> dict:
        self.identify_calls += 1
        return {
            "matches": [],
            "confidenceBand": "unavailable",
            "confidenceSemantics": "uncalibrated_similarity_not_probability",
            "model": "dinov2-small",
            "indexVersion": "v1",
        }


def _client(tmp_path: Path, runtime: StubRuntime | None = None) -> TestClient:
    settings = ServiceSettings(
        api_key=API_KEY,
        index_dir=tmp_path / "index",
        allowed_reference_hosts=frozenset({"images.example.org"}),
        max_image_bytes=1_024,
        max_request_bytes=2_048,
        max_image_pixels=10_000,
        inference_timeout_seconds=1.0,
        rebuild_timeout_seconds=1.0,
        identify_requests_per_minute=60,
        rebuild_requests_per_minute=5,
    )
    app = create_app(
        settings,
        ServiceDependencies(
            runtime=runtime or StubRuntime(), rebuild=lambda payload, deadline: {"ok": True}
        ),
    )
    return TestClient(app)


def _rate_limited_client(tmp_path: Path) -> TestClient:
    settings = ServiceSettings(
        api_key=API_KEY,
        index_dir=tmp_path / "index",
        allowed_reference_hosts=frozenset({"images.example.org"}),
        max_image_bytes=1_024,
        max_request_bytes=2_048,
        max_image_pixels=10_000,
        inference_timeout_seconds=1.0,
        rebuild_timeout_seconds=1.0,
        identify_requests_per_minute=2,
        rebuild_requests_per_minute=1,
    )
    return TestClient(
        create_app(
            settings,
            ServiceDependencies(
                runtime=StubRuntime(), rebuild=lambda payload, deadline: {"ok": True}
            ),
        )
    )


def test_liveness_is_public_but_readiness_fails_closed(tmp_path: Path) -> None:
    client = _client(tmp_path, StubRuntime(ready=False))

    assert client.get("/health").status_code == 200
    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready", "reason": "index_unavailable"}


@pytest.mark.parametrize("path", ["/identify-json", "/identify", "/rebuild-index"])
def test_sensitive_endpoints_require_constant_service_credential(tmp_path: Path, path: str) -> None:
    client = _client(tmp_path)

    assert client.post(path).status_code == 401
    assert client.post(path, headers={"X-Fluke-Model-Key": "wrong"}).status_code == 401


def test_identify_json_accepts_a_bounded_valid_image(tmp_path: Path) -> None:
    runtime = StubRuntime()
    client = _client(tmp_path, runtime)
    encoded = base64.b64encode(_jpeg_bytes()).decode("ascii")

    response = client.post(
        "/identify-json",
        headers={"X-Fluke-Model-Key": API_KEY},
        json={"imageBase64": encoded, "contentType": "image/jpeg"},
    )

    assert response.status_code == 200
    assert response.json()["confidenceSemantics"] == "uncalibrated_similarity_not_probability"
    assert runtime.identify_calls == 1


def test_identify_rejects_oversize_and_invalid_image_before_runtime(tmp_path: Path) -> None:
    runtime = StubRuntime()
    client = _client(tmp_path, runtime)
    headers = {"X-Fluke-Model-Key": API_KEY}

    oversize = base64.b64encode(b"x" * 1_025).decode("ascii")
    assert (
        client.post(
            "/identify-json",
            headers=headers,
            json={"imageBase64": oversize, "contentType": "image/jpeg"},
        ).status_code
        == 413
    )
    invalid = base64.b64encode(b"not an image").decode("ascii")
    assert (
        client.post(
            "/identify-json",
            headers=headers,
            json={"imageBase64": invalid, "contentType": "image/jpeg"},
        ).status_code
        == 422
    )
    assert runtime.identify_calls == 0


def test_multipart_reads_only_one_byte_beyond_limit(tmp_path: Path) -> None:
    runtime = StubRuntime()
    client = _client(tmp_path, runtime)

    response = client.post(
        "/identify",
        headers={"X-Fluke-Model-Key": API_KEY},
        files={"file": ("large.jpg", b"x" * 1_025, "image/jpeg")},
    )

    assert response.status_code == 413
    assert runtime.identify_calls == 0


def test_identify_and_rebuild_are_rate_limited_without_external_infrastructure(
    tmp_path: Path,
) -> None:
    client = _rate_limited_client(tmp_path)
    headers = {"X-Fluke-Model-Key": API_KEY}
    encoded = base64.b64encode(_jpeg_bytes()).decode("ascii")
    payload = {"imageBase64": encoded, "contentType": "image/jpeg"}

    assert client.post("/identify-json", headers=headers, json=payload).status_code == 200
    assert client.post("/identify-json", headers=headers, json=payload).status_code == 200
    assert client.post("/identify-json", headers=headers, json=payload).status_code == 429
    assert client.post("/rebuild-index", headers=headers, json={}).status_code == 200
    assert client.post("/rebuild-index", headers=headers, json={}).status_code == 429


def test_request_body_is_rejected_before_parsing_when_content_length_exceeds_limit(
    tmp_path: Path,
) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/identify-json",
        headers={"X-Fluke-Model-Key": API_KEY, "Content-Type": "application/json"},
        content=b"x" * 2_049,
    )

    assert response.status_code == 413


def test_invalid_content_length_fails_closed(tmp_path: Path) -> None:
    client = _client(tmp_path)

    response = client.post(
        "/identify-json",
        headers={"X-Fluke-Model-Key": API_KEY, "Content-Length": "invalid"},
        content=b"{}",
    )

    assert response.status_code == 400


def test_rebuild_has_a_total_request_deadline(tmp_path: Path) -> None:
    settings = ServiceSettings(
        api_key=API_KEY,
        index_dir=tmp_path,
        allowed_reference_hosts=frozenset({"images.example.org"}),
        inference_timeout_seconds=1.0,
        rebuild_timeout_seconds=0.01,
    )

    published = False

    def slow_rebuild(payload: dict, deadline: object) -> dict:
        nonlocal published
        try:
            while True:
                time.sleep(0.005)
                deadline.check()
        except OperationCancelledError:
            return {"ok": False}
        published = True
        return {"ok": True}

    client = TestClient(
        create_app(settings, ServiceDependencies(runtime=StubRuntime(), rebuild=slow_rebuild))
    )
    response = client.post("/rebuild-index", headers={"X-Fluke-Model-Key": API_KEY}, json={})

    assert response.status_code == 504
    time.sleep(0.03)
    assert published is False


class BlockingRuntime(StubRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def identify(self, image: Image.Image, *, limit: int = 3) -> dict:
        self.identify_calls += 1
        self.started.set()
        self.release.wait(timeout=2)
        return super().identify(image, limit=limit)


def test_timed_out_inference_stays_single_flight_and_fails_closed_busy(tmp_path: Path) -> None:
    runtime = BlockingRuntime()
    settings = ServiceSettings(
        api_key=API_KEY,
        index_dir=tmp_path / "index",
        allowed_reference_hosts=frozenset(),
        max_image_bytes=1_024,
        max_request_bytes=2_048,
        max_image_pixels=10_000,
        inference_timeout_seconds=0.01,
    )
    app = create_app(
        settings,
        ServiceDependencies(runtime=runtime, rebuild=lambda payload, deadline: {"ok": True}),
    )
    payload = {
        "imageBase64": base64.b64encode(_jpeg_bytes()).decode("ascii"),
        "contentType": "image/jpeg",
    }
    headers = {"X-Fluke-Model-Key": API_KEY}

    with TestClient(app) as client:
        assert client.post("/identify-json", headers=headers, json=payload).status_code == 504
        assert runtime.started.is_set()
        assert client.post("/identify-json", headers=headers, json=payload).status_code == 503
        assert runtime.identify_calls == 1

        runtime.release.set()
        for _ in range(100):
            if not client.app.state.inference_runner.busy:
                break
            time.sleep(0.005)
        assert client.app.state.inference_runner.busy is False


class CloseTrackingImage:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_bounded_inference_closes_completed_and_busy_images() -> None:
    started = Event()
    release = Event()

    class Runtime:
        def identify(self, image: object, *, limit: int = 3) -> dict:
            started.set()
            release.wait(timeout=2)
            return {"ok": True}

    runner = BoundedInferenceRunner(Runtime())
    first = CloseTrackingImage()
    second = CloseTrackingImage()

    worker = runner.submit(first)
    assert started.wait(timeout=1)
    with pytest.raises(InferenceBusyError):
        runner.submit(second)
    assert second.closed is True

    release.set()
    assert worker.result(timeout=1) == {"ok": True}
    assert first.closed is True
    assert runner.busy is False
