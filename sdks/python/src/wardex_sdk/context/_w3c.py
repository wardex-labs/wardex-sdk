"""W3C Trace Context (traceparent) parse/format — pure functions, no SDK state.

Level 1 (https://www.w3.org/TR/trace-context/): we emit version 00 with
sampled=01 always (wardex does not head-sample; retention is decided later
by the RetentionClassifier). Unknown versions are parsed leniently from the
known prefix; version ff and all-zero ids are rejected.
"""

from __future__ import annotations

import re

from .._types import SpanContext, SpanId, TraceId

_FIELD_RE = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})"
    r"-(?P<span_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})(?P<rest>.*)$"
)


def parse_traceparent(value: str) -> tuple[TraceId, SpanId, int] | None:
    m = _FIELD_RE.match(value.strip())
    if m is None:
        return None
    version = m.group("version")
    if version == "ff":
        return None
    rest = m.group("rest")
    if version == "00" and rest:
        return None  # version 00 is exactly 4 fields
    if rest and not rest.startswith("-"):
        return None
    trace_bytes = bytes.fromhex(m.group("trace_id"))
    span_bytes = bytes.fromhex(m.group("span_id"))
    if trace_bytes == bytes(16) or span_bytes == bytes(8):
        return None
    return TraceId(trace_bytes), SpanId(span_bytes), int(m.group("flags"), 16)


def format_traceparent(ctx: SpanContext) -> str:
    return f"00-{ctx.trace_id.hex()}-{ctx.span_id.hex()}-01"
