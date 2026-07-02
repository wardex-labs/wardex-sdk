"""Thin codec wrapper — delegates to the native codec submodule.

encode: InternalEnvelope → proto+zstd bytes.
decode: bytes → dict (round-trip verification/debugging). Python has no direct knowledge of proto.
"""

from __future__ import annotations

from typing import Any

from .. import _wardex_native
from .._types import InternalEnvelope


def encode(envelope: InternalEnvelope) -> bytes:
    return _wardex_native.codec.encode_envelope(envelope)


def decode(data: bytes) -> dict[str, Any]:
    return _wardex_native.codec.decode_envelope(data)
