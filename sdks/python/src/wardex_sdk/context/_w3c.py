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

#: The most list-members a `tracestate` may carry (spec §3.3.1).
_MAX_TRACESTATE_MEMBERS = 32


def _future_fields_ok(rest: str) -> bool:
    """Is the tail of a higher-version traceparent structurally a field list?

    Everything after the four known fields belongs to a version we do not
    implement, and the spec is explicit that we must not try to read it: a
    higher version is parsed by taking the known prefix and checking that the
    56th character is a dash. So this checks the SHAPE and nothing else — the
    tail must be one or more `-field` groups with a non-empty field in each.

    The empty field is the whole point, and it is the one case the previous
    `rest.startswith("-")` admitted: `cc-<32>-<16>-01-` has its dash in the
    right place and no field behind it, and `...-01--x` has a hole in the
    middle. Both are a writer that emitted a separator it had nothing to
    separate, which makes the header truncated rather than futuristic — and a
    truncated traceparent is exactly what the restart rule exists for. Version
    00 never reaches here at all; four fields is its entire grammar.

    What this deliberately does NOT do is look inside a field. Rejecting a
    character we merely find surprising would be reading the rest, which the
    spec tells us not to do, and the cost of being wrong is a dropped trace
    link — the failure direction this SDK does not take.
    """
    if not rest.startswith("-"):
        return False
    return all(rest[1:].split("-"))


def sanitize_tracestate(value: str | None) -> str | None:
    """Vet an INBOUND `tracestate` before it enters the SDK, or drop it.

    Called once, at the edge where the header arrives. That placement is the
    decision: a tracestate does not stop at the scope it lands on — parentage
    copies it onto every unit, units hand it to their children, and the
    injector writes it back out on outbound requests. Sanitizing at emission
    would leave the unvetted string sitting in span state in the meantime, and
    sanitizing at both ends would make two places responsible for one fact.
    Untrusted input is checked where it crosses in.

    Two rejections, and neither is a matter of taste:

    A control character (CR, LF, NUL, anything below 0x20, or DEL) cannot
    appear in a header value at all, and this value is one we later WRITE. An
    inbound `tracestate` carrying CRLF is a request-splitting payload aimed at
    whatever service the host calls next, forwarded by us, in a header the host
    never wrote. There is no partial-credit reading of such a value, so the
    whole header is dropped rather than trimmed to its "safe" prefix — a
    truncated vendor state is not the vendor's state.

    Over 32 list-members is the spec's own ceiling, and the spec's own remedy
    is to drop from the RIGHT: the leftmost member is the most recent writer,
    so the members that survive are the ones a receiver is most likely to act
    on. Truncating rather than dropping keeps propagation working through a
    hop that ran over the limit.

    Empty or whitespace-only collapses to None — "no tracestate", which is what
    a receiver of an empty header would conclude anyway, and what keeps the
    header off the wire entirely on the way out.
    """
    if value is None:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    if any(ch < "\x20" or ch == "\x7f" for ch in trimmed):
        return None
    members = trimmed.split(",")
    if len(members) > _MAX_TRACESTATE_MEMBERS:
        return ",".join(members[:_MAX_TRACESTATE_MEMBERS])
    return trimmed


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
    if rest and not _future_fields_ok(rest):
        return None
    trace_bytes = bytes.fromhex(m.group("trace_id"))
    span_bytes = bytes.fromhex(m.group("span_id"))
    if trace_bytes == bytes(16) or span_bytes == bytes(8):
        return None
    return TraceId(trace_bytes), SpanId(span_bytes), int(m.group("flags"), 16)


def format_traceparent(ctx: SpanContext) -> str:
    return f"00-{ctx.trace_id.hex()}-{ctx.span_id.hex()}-{ctx.trace_flags & 0xFF:02x}"
