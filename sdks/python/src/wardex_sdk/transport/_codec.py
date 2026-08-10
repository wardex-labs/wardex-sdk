"""Thin codec wrapper — delegates to the native codec submodule.

encode: Envelope → proto+zstd bytes.
decode: bytes → dict (round-trip verification/debugging). Python has no direct knowledge of proto.
"""

from __future__ import annotations

from typing import Any

from .. import _wardex_native
from .._types import Envelope


def encode(
    envelope: Envelope,
    pii_mode: str = "off",
    pii_disabled: tuple[str, ...] = (),
    limits: object | None = None,
) -> bytes:
    """Encode to wire bytes. PII policy is applied inside the native call
    (marshal -> mask -> serialize, design §4.2). Transports always pass the
    policy explicitly; the "off" default keeps this usable as a pure
    round-trip fidelity tool in tests.

    `limits` is a native Limits object (LimitsConfig.to_native()); the codec
    reads `zstd_level` from it. None uses the core default.

    SECURITY: any future wire transport MUST pass the policy explicitly (the
    transport's own stored policy, installed by `Transport._set_pii_policy`;
    `Transport.encode()` is the sanctioned path that does so) — this "off"
    default is for local round-trip fidelity only, it must never be relied on
    for an export path."""
    return _wardex_native.codec.encode_envelope(envelope, pii_mode, list(pii_disabled), limits)


def decode(data: bytes) -> dict[str, Any]:
    return _wardex_native.codec.decode_envelope(data)
