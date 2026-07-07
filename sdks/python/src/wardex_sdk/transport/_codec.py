"""Thin codec wrapper — delegates to the native codec submodule.

encode: InternalEnvelope → proto+zstd bytes.
decode: bytes → dict (round-trip verification/debugging). Python has no direct knowledge of proto.
"""

from __future__ import annotations

from typing import Any

from .. import _wardex_native
from .._types import InternalEnvelope


def encode(
    envelope: InternalEnvelope,
    pii_mode: str = "off",
    pii_disabled: tuple[str, ...] = (),
) -> bytes:
    """Encode to wire bytes. PII policy is applied inside the native call
    (marshal -> mask -> serialize, design §4.2). Transports always pass the
    policy explicitly; the "off" default keeps this usable as a pure
    round-trip fidelity tool in tests.
    SECURITY: any future wire transport MUST pass the policy explicitly (see
    Transport.set_pii_policy) — this "off" default is for local round-trip
    fidelity only, it must never be relied on for an export path."""
    return _wardex_native.codec.encode_envelope(envelope, pii_mode, list(pii_disabled))


def decode(data: bytes) -> dict[str, Any]:
    return _wardex_native.codec.decode_envelope(data)
