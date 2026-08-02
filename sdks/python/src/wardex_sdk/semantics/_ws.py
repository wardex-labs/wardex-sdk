"""WebSocket semantics — close code mapped onto an `error.type`.

RFC 6455 §7.4.1 names the codes; the unnamed ones are rendered rather than
dropped, because `SpanDraft.finish()` deletes an ERROR span that carries no
`error.type` and an unrecognised close code is still a failure worth a span.
"""

from __future__ import annotations


def ws_close_name(code: int) -> str:
    return {
        1002: "protocol_error",
        1003: "unsupported_data",
        1007: "invalid_payload",
        1008: "policy_violation",
        1009: "message_too_big",
        1010: "mandatory_extension",
        1011: "internal_error",
    }.get(code, f"close_{code}")
