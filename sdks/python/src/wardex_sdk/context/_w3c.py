"""W3C Trace Context (traceparent) parse/format — pure functions, no SDK state.

Level 1 (https://www.w3.org/TR/trace-context/): we emit version 00 and the
context's own `trace_flags`. Unknown versions are parsed leniently from the
known prefix; version ff and all-zero ids are rejected.

The always-sampled invariant did not go away, it MOVED (design §4.1, V9). It
used to live here as a hardcoded `-01`, which meant an upstream that told us
`-00` was silently promoted to `-01` downstream. It now lives at the source:
`assembly/_parentage.resolve_parentage()` stamps `trace_flags=1` on the branch
that has no parent, because wardex does not head-sample — retention is decided
later by the RetentionClassifier — so a trace wardex ORIGINATES is by definition
sampled. Only once origination says 1 does a `0` arriving here unambiguously
mean "an upstream told us -00", which is the one reading that makes honouring it
correct. The two changes are inseparable: formatting `trace_flags` without
stamping 1 at the source would emit `-00` for every wardex-rooted trace and
silence every downstream OTel service on the default ParentBased(ALWAYS_ON)
sampler.
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
    return f"00-{ctx.trace_id.hex()}-{ctx.span_id.hex()}-{ctx.trace_flags & 0xFF:02x}"
