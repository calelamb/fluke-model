"""Network boundary policy for reference-photo retrieval."""

from __future__ import annotations

import ipaddress
import socket
import ssl
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable
from urllib.parse import ParseResult, urljoin, urlparse

import certifi
import urllib3
from PIL import Image

from fluke_model.deadline import OperationDeadline
from fluke_model.image_inputs import ImageInputError, ImageTooLargeError, load_validated_image

MAX_URL_LENGTH = 2_048


class NetworkPolicyError(ValueError):
    """Raised when a reference URL violates the production network policy."""


def validate_reference_url(url: str, allowed_hosts: frozenset[str]) -> ParseResult:
    if len(url) > MAX_URL_LENGTH:
        raise NetworkPolicyError("reference URL is too long")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise NetworkPolicyError("reference URL must use HTTPS")
    if parsed.username or parsed.password:
        raise NetworkPolicyError("reference URL credentials are not allowed")
    if parsed.port not in (None, 443):
        raise NetworkPolicyError("reference URL must use the standard HTTPS port")
    host = parsed.hostname.lower().rstrip(".")
    if host not in allowed_hosts:
        raise NetworkPolicyError("reference URL host is not allowlisted")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and not address.is_global:
        raise NetworkPolicyError("reference URL cannot target a private or local address")
    if parsed.fragment:
        raise NetworkPolicyError("reference URL fragments are not allowed")
    return parsed


def validate_reference_addresses(addresses: tuple[str, ...]) -> None:
    if not addresses:
        raise NetworkPolicyError("reference URL host did not resolve")
    for value in addresses:
        try:
            address = ipaddress.ip_address(value)
        except ValueError as exc:
            raise NetworkPolicyError("reference URL host returned an invalid address") from exc
        if not address.is_global:
            raise NetworkPolicyError("reference URL host must resolve only to the public internet")


class ReferenceImageFetcher:
    def __init__(
        self,
        *,
        allowed_hosts: frozenset[str],
        max_bytes: int,
        max_pixels: int,
        resolver: Callable[..., list[tuple[Any, ...]]] = socket.getaddrinfo,
        client: Any | None = None,
        total_timeout_seconds: float = 15.0,
    ) -> None:
        self._allowed_hosts = allowed_hosts
        self._max_bytes = max_bytes
        self._max_pixels = max_pixels
        self._resolver = resolver
        self._client = client or PinnedHttpsClient()
        self._total_timeout_seconds = total_timeout_seconds

    def _validate_target(self, url: str) -> tuple[ParseResult, tuple[str, ...]]:
        parsed = validate_reference_url(url, self._allowed_hosts)
        answers = self._resolver(parsed.hostname, 443, type=socket.SOCK_STREAM)
        addresses = tuple(str(answer[4][0]) for answer in answers)
        validate_reference_addresses(addresses)
        return (parsed, addresses)

    def load(self, url: str, *, deadline: OperationDeadline | None = None) -> Image.Image:
        operation_deadline = deadline or OperationDeadline.never()
        local_expires_at = monotonic() + self._total_timeout_seconds
        current_url = url
        for _ in range(4):
            operation_deadline.check()
            parsed, addresses = self._validate_target(current_url)
            remaining = min(local_expires_at - monotonic(), operation_deadline.remaining(5.0))
            if remaining <= 0:
                raise NetworkPolicyError("reference image download timed out")
            target = parsed.path or "/"
            if parsed.query:
                target = f"{target}?{parsed.query}"
            try:
                response = self._client.request(
                    address=addresses[0],
                    server_hostname=parsed.hostname,
                    target=target,
                    headers={"Host": parsed.hostname},
                    timeout=remaining,
                )
            except urllib3.exceptions.HTTPError as exc:
                raise NetworkPolicyError("reference image request failed") from exc
            try:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("Location")
                    if not location:
                        raise NetworkPolicyError("reference redirect has no location")
                    current_url = urljoin(current_url, location)
                    continue
                if response.status_code != 200:
                    raise NetworkPolicyError("reference server returned an unsuccessful status")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) > self._max_bytes:
                    raise NetworkPolicyError("reference image exceeds the byte limit")
                data = bytearray()
                for chunk in response.stream(64 * 1024):
                    operation_deadline.check()
                    if monotonic() > local_expires_at:
                        raise NetworkPolicyError("reference image download timed out")
                    data.extend(chunk)
                    if len(data) > self._max_bytes:
                        raise NetworkPolicyError("reference image exceeds the byte limit")
                try:
                    return load_validated_image(
                        bytes(data),
                        content_type=response.headers.get("Content-Type", "").split(";", 1)[0],
                        max_bytes=self._max_bytes,
                        max_pixels=self._max_pixels,
                    )
                except (ImageInputError, ImageTooLargeError) as exc:
                    raise NetworkPolicyError(str(exc)) from exc
            finally:
                response.release_conn()
        raise NetworkPolicyError("reference URL exceeded the redirect limit")


class PinnedHttpsClient:
    def request(
        self,
        *,
        address: str,
        server_hostname: str,
        target: str,
        headers: dict[str, str],
        timeout: float,
    ) -> Any:
        pool = urllib3.HTTPSConnectionPool(
            address,
            port=443,
            assert_hostname=server_hostname,
            server_hostname=server_hostname,
            cert_reqs=ssl.CERT_REQUIRED,
            ca_certs=certifi.where(),
            maxsize=1,
            block=True,
        )
        response = pool.urlopen(
            "GET",
            target,
            headers=headers,
            redirect=False,
            preload_content=False,
            retries=False,
            timeout=urllib3.Timeout(connect=min(3.0, timeout), read=min(5.0, timeout)),
        )
        return _PinnedResponse(response=response, pool=pool)


@dataclass(frozen=True)
class _PinnedResponse:
    response: Any
    pool: urllib3.HTTPSConnectionPool

    @property
    def status_code(self) -> int:
        return int(self.response.status)

    @property
    def headers(self) -> Any:
        return self.response.headers

    def stream(self, chunk_size: int) -> Any:
        return self.response.stream(chunk_size)

    def release_conn(self) -> None:
        self.response.release_conn()
        self.pool.close()
