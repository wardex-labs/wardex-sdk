"""`Span.events` (tag 10) and `Span.links` (tag 11) — the encoder that was missing.

`bindings/python/src/codec.rs` contained zero occurrences of `links` or `events`
until `events_to_proto` and `links_to_proto` were written. Both fields were declared
in `span.proto` and neither was filled, so `_types.py`'s `events` and `links` tuples
were dropped whole at encode — and with them `InternalSpanLink.reason`, which made
`LinkReason` dead vocabulary and design §6.3's entire graph model impossible to
transmit:

  * `TRIGGERED_BY` is how a conditional branch in a graph is represented at all
    (the edge that actually fired, which the wire cannot show);
  * `HANDOFF_FROM` is what makes a handoff a marker span with a SIBLING receiver
    instead of a container, and a 5-hop chain rendered as 5 levels of nesting
    makes every duration in the flame graph wrong;
  * `RESUMED_FROM` is the honest representation of a checkpoint resume — a new
    trace, linked, rather than a fabricated parent edge across a process
    boundary.

Declaring the enum without this encoder would have been theatre, so both land in
the same commit and both directions are exercised here.
"""

from __future__ import annotations

import pytest

from wardex_sdk._enums import SpanKind, StatusCode
from wardex_sdk._types import (
    EnvelopeHeader,
    InternalEnvelope,
    InternalSpan,
    InternalSpanEvent,
    InternalSpanLink,
    SdkInfo,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.assembly import LinkReason
from wardex_sdk.transport import _codec

_OTHER_TRACE = TraceId(b"\xaa" * 16)
_OTHER_SPAN = SpanId(b"\xbb" * 8)


def _envelope(span: InternalSpan) -> InternalEnvelope:
    return InternalEnvelope(
        header=EnvelopeHeader(
            event_id="evt-1",
            api_key="k",
            sdk=SdkInfo(
                name="wardex.python",
                version="0.1.0",
                python_version="3.12",
                os="mac",
                arch="arm64",
            ),
            sent_at_ns=42,
        ),
        spans=(span,),
    )


def _span(**kw) -> InternalSpan:
    base = dict(
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
        parent_span_id=None,
        name="execute_step planner",
        kind=SpanKind.INTERNAL,
        start_time_ns=1000,
        end_time_ns=2000,
        status=StatusCode.OK,
    )
    base.update(kw)
    return InternalSpan(**base)


def _roundtrip(span: InternalSpan) -> dict:
    out = _codec.decode(_codec.encode(_envelope(span)))
    return out["items"][0]["span"]


@pytest.mark.parametrize("reason", list(LinkReason))
def test_every_link_reason_round_trips(reason: LinkReason):
    """Every member, not a sample: an enum whose values are declared in proto and
    mapped by hand in `codec.rs` fails one value at a time, and a value that maps
    to UNSPECIFIED is indistinguishable from a link with no reason at all."""
    span = _span(
        links=(InternalSpanLink(trace_id=_OTHER_TRACE, span_id=_OTHER_SPAN, reason=reason.value),)
    )

    (link,) = _roundtrip(span)["links"]

    assert link["trace_id"] == _OTHER_TRACE.value
    assert link["span_id"] == _OTHER_SPAN.value
    assert link["reason"] == reason.value


def test_a_link_without_a_reason_round_trips_as_empty():
    span = _span(links=(InternalSpanLink(trace_id=_OTHER_TRACE, span_id=_OTHER_SPAN),))

    (link,) = _roundtrip(span)["links"]

    assert link["reason"] == ""


def test_several_links_keep_their_order_and_their_reasons():
    """Order is the graph. `TRIGGERED_BY` edges read as a sequence, and a codec
    that reordered them would rewrite the causality it exists to carry."""
    span = _span(
        links=(
            InternalSpanLink(_OTHER_TRACE, _OTHER_SPAN, LinkReason.TRIGGERED_BY),
            InternalSpanLink(_OTHER_TRACE, SpanId(b"\xcc" * 8), LinkReason.HANDOFF_FROM),
            InternalSpanLink(_OTHER_TRACE, SpanId(b"\xdd" * 8), LinkReason.RESUMED_FROM),
        )
    )

    links = _roundtrip(span)["links"]

    assert [link["reason"] for link in links] == [
        "triggered_by",
        "handoff_from",
        "resumed_from",
    ]
    assert [link["span_id"] for link in links] == [_OTHER_SPAN.value, b"\xcc" * 8, b"\xdd" * 8]


def test_events_round_trip_with_timestamps_and_attributes():
    span = _span(
        events=(
            InternalSpanEvent(name="cache_hit", timestamp_ns=1500, attributes=(("key", "k"),)),
            InternalSpanEvent(name="retry", timestamp_ns=1700, attributes=(("attempt", 2),)),
        )
    )

    events = _roundtrip(span)["events"]

    assert [e["name"] for e in events] == ["cache_hit", "retry"]
    assert [e["timestamp_ns"] for e in events] == [1500, 1700]
    assert events[0]["attributes"] == [{"key": "key", "value": "k"}]
    assert events[1]["attributes"] == [{"key": "attempt", "value": 2}]


def test_a_span_with_neither_still_round_trips():
    decoded = _roundtrip(_span())

    assert decoded["events"] == []
    assert decoded["links"] == []


def test_links_and_events_survive_the_pii_pass():
    """PII masking destructures `SpanLink` exhaustively, so a new field forces a
    decision there. `reason` is a closed enum and cannot carry PII — the same
    disposition `CaptureIntegrity.limitations` gets."""
    span = _span(
        links=(InternalSpanLink(_OTHER_TRACE, _OTHER_SPAN, LinkReason.HANDOFF_FROM),),
        events=(
            InternalSpanEvent(
                name="contact", timestamp_ns=1, attributes=(("email", "a@example.com"),)
            ),
        ),
    )

    raw = _codec.encode(_envelope(span), pii_mode="mask")
    decoded = _codec.decode(raw)["items"][0]["span"]

    (link,) = decoded["links"]
    assert link["reason"] == "handoff_from"
    (event,) = decoded["events"]
    assert event["attributes"][0]["value"] != "a@example.com"


# --------------------------------------------------------------------------
# the OTLP surface — the one that actually leaves the process
# --------------------------------------------------------------------------


def _otlp_span(span: InternalSpan) -> dict:
    from wardex_sdk import _wardex_native

    data = _wardex_native.codec.encode_otlp_traces(_envelope(span))
    out = _wardex_native.codec.decode_otlp_traces(data)
    return out["resource_spans"][0]["scope_spans"][0]["spans"][0]


def test_the_otlp_encoder_carries_events_and_links_too():
    """Both encoders or neither.

    `transport/_otlp_http.py` is the only path that puts bytes on a network
    today, so filling `events`/`links` on the wardex envelope alone would leave
    a user configured for the OTLP exporter losing §6.3's whole graph model with
    no counter, no `Limitation` and no failing test — indistinguishable from
    "this agent has no graph edges". That is the silent loss I4 forbids, and it
    is why the two surfaces are asserted to agree rather than assumed to.
    """
    span = _span(
        links=(InternalSpanLink(_OTHER_TRACE, _OTHER_SPAN, LinkReason.HANDOFF_FROM),),
        events=(
            InternalSpanEvent(name="cache_hit", timestamp_ns=1500, attributes=(("key", "k"),)),
        ),
    )

    sp = _otlp_span(span)

    (event,) = sp["events"]
    assert event["name"] == "cache_hit"
    assert event["time_unix_nano"] == 1500
    assert event["attributes"] == {"key": "k"}

    (link,) = sp["links"]
    assert link["trace_id"] == _OTHER_TRACE.value.hex()
    assert link["span_id"] == _OTHER_SPAN.value.hex()
    # `reason` has no OTLP-native home, so it travels as an attribute rather
    # than being dropped — and as the wardex value string, not a raw number.
    assert link["attributes"]["wardex.link.reason"] == "handoff_from"


def test_an_otlp_link_without_a_reason_carries_no_reason_attribute():
    span = _span(links=(InternalSpanLink(_OTHER_TRACE, _OTHER_SPAN),))

    (link,) = _otlp_span(span)["links"]

    assert link["attributes"] == {}


def test_an_otlp_span_with_neither_still_encodes():
    sp = _otlp_span(_span())

    assert sp["events"] == []
    assert sp["links"] == []
