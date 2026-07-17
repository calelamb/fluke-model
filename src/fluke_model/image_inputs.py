"""Bounded and validated image decoding at the service boundary."""

from __future__ import annotations

import base64
import binascii
from io import BytesIO

from PIL import Image, UnidentifiedImageError

ALLOWED_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


class ImageInputError(ValueError):
    """An uploaded image is malformed or outside the configured bounds."""


class ImageTooLargeError(ImageInputError):
    """An uploaded image exceeds byte or decoded-pixel bounds."""


def decode_base64_image(value: str, *, max_bytes: int) -> bytes:
    max_encoded = ((max_bytes + 2) // 3) * 4
    if len(value) > max_encoded:
        raise ImageTooLargeError("image exceeds the byte limit")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageInputError("imageBase64 must be valid base64") from exc
    if len(data) > max_bytes:
        raise ImageTooLargeError("image exceeds the byte limit")
    return data


def load_validated_image(
    data: bytes,
    *,
    content_type: str | None,
    max_bytes: int,
    max_pixels: int,
) -> Image.Image:
    if not data:
        raise ImageInputError("image is empty")
    if len(data) > max_bytes:
        raise ImageTooLargeError("image exceeds the byte limit")
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageInputError("content type must be JPEG, PNG, or WebP")
    try:
        with Image.open(BytesIO(data)) as probe:
            if probe.format not in ALLOWED_FORMATS:
                raise ImageInputError("image format must be JPEG, PNG, or WebP")
            if probe.width * probe.height > max_pixels:
                raise ImageTooLargeError("decoded image exceeds the pixel limit")
            probe.verify()
        with Image.open(BytesIO(data)) as decoded:
            decoded.load()
            return decoded.convert("RGB")
    except ImageInputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageInputError("image could not be decoded") from exc
