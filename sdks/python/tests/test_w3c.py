"""W3C Trace Context Level 1 vectors for parse/format."""

from wardex_sdk._types import SpanContext, SpanId, TraceId
from wardex_sdk.context._w3c import format_traceparent, parse_traceparent

VALID = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def test_parse_valid():
    tid, sid, flags = parse_traceparent(VALID)
    assert tid.hex() == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert sid.hex() == "00f067aa0ba902b7"
    assert flags == 1


def test_parse_rejects_all_zero_ids():
    assert parse_traceparent("00-" + "0" * 32 + "-00f067aa0ba902b7-01") is None
    assert parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-" + "0" * 16 + "-01") is None


def test_parse_rejects_version_ff():
    assert parse_traceparent("ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01") is None


def test_parse_rejects_malformed():
    assert parse_traceparent("") is None
    assert parse_traceparent("not-a-header") is None
    assert parse_traceparent("00-abc-def-01") is None  # wrong lengths
    # uppercase hex is invalid per spec
    assert parse_traceparent(VALID.upper()) is None
    # version 00 must have exactly 4 fields
    assert parse_traceparent(VALID + "-extra") is None


def test_parse_unknown_version_lenient():
    """Future versions may append fields; parse the known prefix."""
    got = parse_traceparent(
        "cc-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-what-the-future-will-be-like"
    )
    assert got is not None
    tid, sid, flags = got
    assert tid.hex() == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_format_roundtrip():
    ctx = SpanContext(trace_id=TraceId.generate(), span_id=SpanId.generate(), trace_flags=1)
    tid, sid, flags = parse_traceparent(format_traceparent(ctx))
    assert tid == ctx.trace_id and sid == ctx.span_id and flags == ctx.trace_flags


def test_format_shape():
    ctx = SpanContext(trace_id=TraceId.generate(), span_id=SpanId.generate(), trace_flags=1)
    out = format_traceparent(ctx)
    assert out.startswith("00-") and out.endswith("-01") and len(out) == 55


def test_format_emits_the_contexts_own_flags_not_a_constant():
    """V9: the always-sampled invariant moved to `resolve_parentage`.

    It used to live here as a hardcoded `-01`, which promoted an upstream's
    `-00` to `-01` on the way out. The formatter is now honest, and the reason
    that is safe is asserted next door: `test_propagation.py`'s
    `test_get_traceparent_inside_and_outside_span` still reads `-01` for a
    wardex-rooted trace, because the parentage core stamps flags=1 at
    origination. Both halves are needed — this one alone would ship `-00`
    everywhere.
    """
    ids = {"trace_id": TraceId.generate(), "span_id": SpanId.generate()}

    assert format_traceparent(SpanContext(**ids, trace_flags=0)).endswith("-00")
    assert format_traceparent(SpanContext(**ids, trace_flags=1)).endswith("-01")
    # a flags byte we never set ourselves still round-trips instead of being
    # rewritten: only bits we were told about reach the wire.
    assert parse_traceparent(format_traceparent(SpanContext(**ids, trace_flags=0xFF)))[2] == 0xFF
