"""Bounded network image fetch tests."""

from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from fluke_model.network import NetworkPolicyError, ReferenceImageFetcher


class FakeResponse:
    def __init__(self, data: bytes, *, status_code: int = 200, location: str | None = None) -> None:
        self.status_code = status_code
        self.headers = {"Content-Length": str(len(data)), "Content-Type": "image/jpeg"}
        if location is not None:
            self.headers["Location"] = location
        self._data = data

    def stream(self, chunk_size: int) -> list[bytes]:
        return [
            self._data[index : index + chunk_size]
            for index in range(0, len(self._data), chunk_size)
        ]

    def release_conn(self) -> None:
        return None


class FakeClient:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def request(self, **kwargs: object) -> FakeResponse:
        self.calls.append(dict(kwargs))
        return self.responses.pop(0)


def _jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color=(1, 2, 3)).save(output, format="JPEG")
    return output.getvalue()


def _resolver(host: str, port: int, **kwargs: object) -> list[tuple]:
    return [(2, 1, 6, "", ("93.184.216.34", port))]


def test_fetcher_streams_bounded_image_without_env_proxy() -> None:
    client = FakeClient([FakeResponse(_jpeg())])
    fetcher = ReferenceImageFetcher(
        allowed_hosts=frozenset({"images.example.org"}),
        max_bytes=1_024,
        max_pixels=1_000,
        resolver=_resolver,
        client=client,
    )

    image = fetcher.load("https://images.example.org/ref.jpg")

    assert image.size == (8, 8)
    assert client.calls[0]["address"] == "93.184.216.34"
    assert client.calls[0]["server_hostname"] == "images.example.org"
    assert client.calls[0]["headers"] == {"Host": "images.example.org"}


def test_fetcher_revalidates_redirect_target_and_rejects_oversize_header() -> None:
    redirect = FakeResponse(b"", status_code=302, location="https://internal.example/ref.jpg")
    fetcher = ReferenceImageFetcher(
        allowed_hosts=frozenset({"images.example.org"}),
        max_bytes=8,
        max_pixels=1_000,
        resolver=_resolver,
        client=FakeClient([redirect]),
    )
    with pytest.raises(NetworkPolicyError, match="allowlisted"):
        fetcher.load("https://images.example.org/ref.jpg")

    oversize = ReferenceImageFetcher(
        allowed_hosts=frozenset({"images.example.org"}),
        max_bytes=8,
        max_pixels=1_000,
        resolver=_resolver,
        client=FakeClient([FakeResponse(b"x" * 9)]),
    )
    with pytest.raises(NetworkPolicyError, match="byte limit"):
        oversize.load("https://images.example.org/ref.jpg")


def test_fetcher_never_resolves_the_validated_hostname_again_before_connecting() -> None:
    resolutions = 0

    def rebinding_resolver(host: str, port: int, **kwargs: object) -> list[tuple]:
        nonlocal resolutions
        resolutions += 1
        address = "93.184.216.34" if resolutions == 1 else "127.0.0.1"
        return [(2, 1, 6, "", (address, port))]

    client = FakeClient([FakeResponse(_jpeg())])
    fetcher = ReferenceImageFetcher(
        allowed_hosts=frozenset({"images.example.org"}),
        max_bytes=1_024,
        max_pixels=1_000,
        resolver=rebinding_resolver,
        client=client,
    )

    fetcher.load("https://images.example.org/ref.jpg")

    assert resolutions == 1
    assert client.calls[0]["address"] == "93.184.216.34"
