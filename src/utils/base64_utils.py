"""Base64 decoding helpers."""
from __future__ import annotations

import base64
import binascii


class Base64DecodeError(ValueError):
    """Raised when a base64 payload cannot be decoded."""


def decode_base64_file(data: str) -> bytes:
    """Decode a base64 string that may contain a ``data:<mime>;base64,`` prefix."""
    if not isinstance(data, str) or not data:
        raise Base64DecodeError("base64 payload is empty")

    payload = data.strip()
    if payload.startswith("data:"):
        comma = payload.find(",")
        if comma == -1:
            raise Base64DecodeError("malformed data URL: missing comma")
        header = payload[:comma]
        if "base64" not in header:
            raise Base64DecodeError("only base64 data URLs are supported")
        payload = payload[comma + 1:]

    # Remove any whitespace/newlines
    payload = "".join(payload.split())

    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise Base64DecodeError(f"invalid base64 payload: {exc}") from exc
