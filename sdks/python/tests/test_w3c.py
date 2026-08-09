"""W3C Trace Context Level 1 vectors for parse/format."""

from wardex_sdk._types import SpanContext, SpanId, TraceId
from wardex_sdk.context._w3c import format_traceparent, parse_traceparent, sanitize_tracestate

VALID = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
FUTURE = "cc-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


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


def test_parse_rejects_a_future_version_with_an_empty_field():
    """A dash with nothing behind it is a truncated header, not a future one.

    The spec's rule for a higher version is "the 56th character must be a
    dash", and a `rest.startswith("-")` check passes all three of these while
    reading them as well-formed. They are a writer that emitted a separator it
    had nothing to separate — which is what the restart rule is for.
    """
    assert parse_traceparent(FUTURE + "-") is None  # trailing dash, no field
    assert parse_traceparent(FUTURE + "--future") is None  # hole at the front
    assert parse_traceparent(FUTURE + "-future-") is None  # hole at the back
    assert parse_traceparent(FUTURE + "-a--b") is None  # hole in the middle


def test_parse_does_not_read_inside_a_future_field():
    """Shape only. The spec says not to parse the rest, so we do not.

    A field whose CONTENT we find surprising still belongs to a version we do
    not implement, and rejecting it would cost a real trace link to enforce a
    grammar nobody published.
    """
    assert parse_traceparent(FUTURE + "-01-ZZ!~") is not None


def test_sanitize_tracestate_drops_the_absent_and_the_empty():
    assert sanitize_tracestate(None) is None
    assert sanitize_tracestate("") is None
    assert sanitize_tracestate("   \t ") is None
    assert sanitize_tracestate("  dd=s:1  ") == "dd=s:1"


def test_sanitize_tracestate_drops_a_value_carrying_control_characters():
    """The header we later WRITE cannot carry a request splitter.

    An inbound tracestate is forwarded on outbound calls, so a CRLF in it is a
    header the host never wrote appearing in a request the host did make. The
    whole value goes rather than its prefix: a truncated vendor state is not
    the vendor's state, and half of an attack string is still not data.
    """
    assert sanitize_tracestate("dd=s:1\r\nx-injected: 1") is None
    assert sanitize_tracestate("dd=s:1\nfoo=bar") is None
    assert sanitize_tracestate("dd=s\x00:1") is None
    assert sanitize_tracestate("dd=s:1\x7f") is None


def test_sanitize_tracestate_truncates_to_the_spec_ceiling_from_the_right():
    """32 list-members, and the ones kept are the most recent writers.

    Truncating rather than dropping keeps propagation alive through a hop that
    ran over the limit; keeping the LEFT is the spec's own rule, because the
    leftmost member was written most recently.
    """
    over = ",".join(f"v{i}=x" for i in range(40))
    kept = sanitize_tracestate(over)
    assert kept is not None
    members = kept.split(",")
    assert len(members) == 32
    assert members[0] == "v0=x" and members[-1] == "v31=x"
    exact = ",".join(f"v{i}=x" for i in range(32))
    assert sanitize_tracestate(exact) == exact


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
