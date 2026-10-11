"""Codec round-trip — Envelope → encode → decode → field preservation."""

from __future__ import annotations

import sys
import types

import pytest

from wardex_sdk import _wardex_native
from wardex_sdk._enums import (
    CaptureSource,
    Direction,
    OperationName,
    Protocol,
    SpanKind,
    StatusCode,
)
from wardex_sdk._types import (
    A2aMeta,
    CaptureIntegrity,
    CorrelationInfo,
    Envelope,
    EnvelopeHeader,
    GenAIAttributes,
    HttpMeta,
    InputRef,
    InternalSpan,
    InternalStateSnapshot,
    SdkInfo,
    SpanContext,
    SpanId,
    ToolDefinition,
    ToolDefinitionSet,
    TraceId,
    TransportAttributes,
    TransportTiming,
)
from wardex_sdk.transport import _codec


def _header() -> EnvelopeHeader:
    return EnvelopeHeader(
        event_id="evt-1",
        sdk=SdkInfo(
            name="wardex.python", version="0.1.0", python_version="3.12", os="mac", arch="arm64"
        ),
        sent_at_ns=42,
    )


def _span(**kw) -> InternalSpan:
    base = dict(
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
        parent_span_id=None,
        name="GET /v1/chat",
        kind=SpanKind.CLIENT,
        start_time_ns=1000,
        end_time_ns=2000,
        status=StatusCode.OK,
    )
    base.update(kw)
    return InternalSpan(**base)


def _env(span: InternalSpan) -> Envelope:
    return Envelope(header=_header(), spans=(span,))


def test_roundtrip_core_span_fields():
    env = _env(
        _span(
            input_data=b"req-bytes",
            output_data=b"resp-bytes",
            error_type=None,
            server_address="api.openai.com",
            server_port=443,
            capture_sources=(CaptureSource.SOCKET,),
            extra=(("network.protocol.version", "2"), ("retry", 3), ("ok", True)),
        )
    )
    out = _codec.decode(_codec.encode(env))
    span = out["items"][0]["span"]
    assert span["name"] == "GET /v1/chat"
    assert span["kind"] == 2  # CLIENT
    assert span["status"]["code"] == 1  # OK
    assert span["start_time_unix_nano"] == 1000
    assert span["end_time_unix_nano"] == 2000
    assert span["server_address"] == "api.openai.com"
    assert span["server_port"] == 443
    assert span["input_data"] == b"req-bytes"
    assert span["output_data"] == b"resp-bytes"
    assert span["trace_id"] == b"\x01" * 16
    assert span["span_id"] == b"\x02" * 8
    assert 7 in span["capture_sources"]  # SOCKET
    # extra passthrough (scalar types preserved)
    kv = {e["key"]: e["value"] for e in span["extra"]}
    assert kv["network.protocol.version"] == "2"
    assert kv["retry"] == 3
    assert kv["ok"] is True


def test_roundtrip_header():
    out = _codec.decode(_codec.encode(_env(_span())))
    h = out["header"]
    assert h["event_id"] == "evt-1"
    # No credential rides in the body, and the project slot leaves the SDK
    # empty: the receiver, not the sender, says which project a batch is.
    assert "api_key" not in h
    assert h["project_id"] == ""
    assert h["sent_at_unix_nano"] == 42
    assert h["sdk"]["name"] == "wardex.python"
    assert h["sdk"]["version"] == "0.1.0"


def test_gen_ai_flattened_into_extra():
    env = _env(
        _span(
            gen_ai=GenAIAttributes(
                operation=OperationName.CHAT,
                request_model="gpt-4o",
                response_model="gpt-4o-2024",
                input_tokens=10,
                output_tokens=5,
                temperature=0.7,
                max_tokens=256,
                stream=True,
                finish_reasons=("stop",),
            )
        )
    )
    out = _codec.decode(_codec.encode(env))
    kv = {e["key"]: e["value"] for e in out["items"][0]["span"]["extra"]}
    assert kv["gen_ai.request.model"] == "gpt-4o"
    assert kv["gen_ai.response.model"] == "gpt-4o-2024"
    assert kv["gen_ai.usage.input_tokens"] == 10
    assert kv["gen_ai.usage.output_tokens"] == 5
    assert kv["gen_ai.request.temperature"] == 0.7
    assert kv["gen_ai.request.max_tokens"] == 256
    assert kv["gen_ai.request.stream"] is True
    assert kv["gen_ai.operation.name"] == "chat"
    # tuple → ArrayValue of strings (semconv array type; CSV was lossy on
    # elements containing commas), decoded back as a list.
    assert kv["gen_ai.response.finish_reasons"] == ["stop"]


def test_encode_is_deterministic():
    env = _env(_span(input_data=b"x" * 100))
    assert _codec.encode(env) == _codec.encode(env)


def test_empty_envelope_roundtrips():
    out = _codec.decode(_codec.encode(Envelope(header=_header())))
    assert out["items"] == []


def test_roundtrip_transport_http():
    tr = TransportAttributes(
        protocol=Protocol.HTTP,
        direction=Direction.OUTBOUND,
        timing=TransportTiming(
            tcp_connect_ms=1.5, tls_handshake_ms=12.0, ttfb_ms=30.0, ttft_ms=25.0, transfer_ms=5.0
        ),
        request_size=100,
        response_size=200,
        http=HttpMeta(method="POST", url="https://api.openai.com/v1/chat", status_code=200),
        is_streaming=True,
        connection_reused=True,
    )
    out = _codec.decode(_codec.encode(_env(_span(transport=tr))))
    t = out["items"][0]["span"]["transport"]
    assert t["protocol"] == 1  # HTTP
    assert t["timing"]["tls_handshake_ms"] == 12.0
    assert t["timing"]["ttft_ms"] == 25.0
    assert t["http"]["method"] == "POST"
    assert t["http"]["url"] == "https://api.openai.com/v1/chat"
    assert t["http"]["status_code"] == 200
    assert t["is_streaming"] is True
    assert t["connection_reused"] is True


def test_roundtrip_capture_integrity_and_correlation():
    span = _span(
        capture_integrity=CaptureIntegrity(
            request_body_captured=True,
            response_body_captured=True,
            truncated=False,
            limitations=("ttft_unavailable_h2",),
        ),
        correlation=CorrelationInfo(strategy="contextvar", confidence=1.0),
    )
    out = _codec.decode(_codec.encode(_env(span)))["items"][0]["span"]
    assert out["capture_integrity"]["request_body_captured"] is True
    assert "ttft_unavailable_h2" in out["capture_integrity"]["limitations"]
    assert out["correlation"]["strategy"] == "contextvar"
    assert out["correlation"]["confidence"] == 1.0


def test_roundtrip_a2a_and_blob_refs_and_tool_definitions():
    tr = TransportAttributes(
        protocol=Protocol.HTTP,
        direction=Direction.OUTBOUND,
        a2a=A2aMeta(task_id="task-1", transport="http"),
        request_blob_ref="blob://req/1",
        response_blob_ref="blob://resp/1",
    )
    snap = InternalStateSnapshot(
        trace_id=TraceId(b"\x05" * 16),
        span_id=SpanId(b"\x06" * 8),
        timestamp_ns=123,
        tool_definitions=ToolDefinitionSet(
            tools=(
                ToolDefinition(
                    name="get_weather",
                    description="get current weather",
                    parameters_schema=b'{"type":"object"}',
                    version="1.0",
                    type="function",
                    hash="sha256:def",
                ),
            ),
            set_hash="sha256:set",
        ),
    )
    env = Envelope(header=_header(), spans=(_span(transport=tr),), state_snapshots=(snap,))
    out = _codec.decode(_codec.encode(env))

    t = out["items"][0]["span"]["transport"]
    assert t["a2a"]["task_id"] == "task-1"
    assert t["a2a"]["transport"] == "http"
    assert t["request_blob_ref"] == "blob://req/1"
    assert t["response_blob_ref"] == "blob://resp/1"

    ss = out["items"][1]["state_snapshot"]
    assert ss["tool_definitions"]["set_hash"] == "sha256:set"
    assert "get_weather" in ss["tool_definitions"]["tool_names"]


def test_roundtrip_state_snapshot():
    snap = InternalStateSnapshot(
        trace_id=TraceId(b"\x03" * 16),
        span_id=SpanId(b"\x04" * 8),
        timestamp_ns=999,
        snapshot_type="turn_start",
        turn_index=2,
        conversation_state=b"history",
        input_refs=(InputRef(key="doc.md", content_hash="sha256:abc"),),
    )
    env = Envelope(header=_header(), state_snapshots=(snap,))
    out = _codec.decode(_codec.encode(env))
    ss = out["items"][0]["state_snapshot"]
    assert ss["trace_id"] == b"\x03" * 16
    assert ss["turn_index"] == 2
    assert ss["conversation_state"] == b"history"
    assert ss["input_refs"][0]["key"] == "doc.md"
    assert ss["input_refs"][0]["content_hash"] == "sha256:abc"


# --------------------------------------------------------------------------
# attribute values the wire cannot hold as given
# --------------------------------------------------------------------------
#
# `Span.set_attribute` promises `str | int | float | bool` and validates
# nothing. An integer outside int64 used to ship as a float that rounded it
# (`2**70` -> `1.18e+21`), and a value no read could take raised out of the
# encoder, where the client's fail-closed drain dropped the WHOLE batch with
# no marker. Both export encoders are pinned, because both are a drain's
# encoder: the OTLP one and the wardex envelope one.

_UNMARSHALLED = "wardex.codec.unmarshalled"


def _batch(value: object) -> Envelope:
    """A good span and one carrying `value`, in ONE batch."""
    good = _span(name="good", extra=(("n", 1),))
    bad = _span(
        name="bad",
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x03" * 8)),
        extra=(("before", "kept"), ("req_id", value), ("after", 2.5)),
    )
    return Envelope(header=_header(), spans=(good, bad))


def _otlp_spans(env: Envelope) -> dict[str, dict]:
    bodies, *_ = _wardex_native.codec.encode_otlp_requests(env, "off", [], None, False)
    assert len(bodies) == 1
    decoded = _wardex_native.codec.decode_otlp_traces(bodies[0])
    return {
        sp["name"]: sp["attributes"]
        for rs in decoded["resource_spans"]
        for ss in rs["scope_spans"]
        for sp in ss["spans"]
    }


def _envelope_spans(env: Envelope) -> dict[str, dict]:
    body, *_ = _wardex_native.codec.encode_envelope_export(env, "off", [], None)
    out: dict[str, dict] = {}
    for item in _codec.decode(body)["items"]:
        sp = item["span"]
        out[sp["name"]] = {kv["key"]: kv["value"] for kv in sp["extra"]}
    return out


_EXPORT_ENCODERS = [
    pytest.param(_otlp_spans, id="otlp"),
    pytest.param(_envelope_spans, id="envelope"),
]


@pytest.mark.parametrize("encode", _EXPORT_ENCODERS)
@pytest.mark.parametrize(
    "value",
    [2**70, sys.maxsize + 1, -(2**63) - 1, 10**400],
    ids=["2**70", "sys.maxsize+1", "-(2**63)-1", "10**400"],
)
def test_an_integer_outside_int64_ships_as_its_exact_digits_and_is_named(encode, value):
    spans = encode(_batch(value))
    assert set(spans) == {"good", "bad"}, "the batch did not survive"
    bad = spans["bad"]
    # The host's digits, exactly -- never a float that rounds them.
    assert bad["req_id"] == str(value)
    assert bad[_UNMARSHALLED] == "req_id"
    # Its neighbours are untouched, and the good span carries no note.
    assert bad["before"] == "kept" and bad["after"] == 2.5
    assert _UNMARSHALLED not in spans["good"]


@pytest.mark.parametrize("encode", _EXPORT_ENCODERS)
def test_the_int64_bounds_themselves_are_still_integers(encode):
    for value in (sys.maxsize, -(2**63)):
        bad = encode(_batch(value))["bad"]
        assert bad["req_id"] == value and isinstance(bad["req_id"], int)
        assert _UNMARSHALLED not in bad


class _LoudInt(int):
    """An int subclass whose own text is not its digits."""

    def __str__(self) -> str:
        return "loud"

    __repr__ = __str__


@pytest.mark.parametrize("encode", _EXPORT_ENCODERS)
def test_an_int_subclass_ships_its_digits_not_its_own_text(encode):
    bad = encode(_batch(_LoudInt(2**70)))["bad"]
    assert bad["req_id"] == str(2**70)


@pytest.mark.parametrize("encode", _EXPORT_ENCODERS)
@pytest.mark.parametrize(
    "value",
    [object(), b"raw bytes", "\ud800", [1, 2]],
    ids=["object()", "bytes", "lone-surrogate", "list"],
)
def test_a_value_the_wire_cannot_hold_is_left_out_named_and_the_batch_ships(encode, value):
    spans = encode(_batch(value))
    assert set(spans) == {"good", "bad"}, "one value cost the whole batch"
    bad = spans["bad"]
    assert "req_id" not in bad
    assert bad[_UNMARSHALLED] == "req_id"
    assert bad["before"] == "kept" and bad["after"] == 2.5


class _Interrupting:
    def __index__(self) -> int:
        raise KeyboardInterrupt


def test_a_keyboard_interrupt_from_a_host_value_reaches_the_host():
    # Only an `Exception` is a value the encoder could not read. A
    # `KeyboardInterrupt` raised from the host's `__index__` is the host's
    # control flow, and becoming one more named attribute would swallow it.
    with pytest.raises(KeyboardInterrupt):
        _wardex_native.codec.encode_otlp_requests(_batch(_Interrupting()), "off", [], None, False)


def test_two_unshippable_values_are_one_note_in_read_order():
    env = Envelope(
        header=_header(),
        spans=(_span(extra=(("a", 2**64), ("ok", 1), ("b", object()))),),
    )
    (attrs,) = _otlp_spans(env).values()
    assert attrs[_UNMARSHALLED] == "a,b"
    assert attrs["a"] == str(2**64) and "b" not in attrs


def test_a_snapshot_attribute_is_named_on_the_snapshot_and_the_batch_ships():
    snap = InternalStateSnapshot(
        trace_id=TraceId(b"\x03" * 16),
        span_id=SpanId(b"\x04" * 8),
        timestamp_ns=999,
        snapshot_type="turn_start",
        turn_index=1,
        attributes=(("big", 2**70), ("odd", object()), ("fine", "yes")),
    )
    env = Envelope(header=_header(), spans=(_span(name="good"),), state_snapshots=(snap,))
    body, *_ = _wardex_native.codec.encode_envelope_export(env, "off", [], None)
    items = _codec.decode(body)["items"]
    assert items[0]["span"]["name"] == "good"
    attrs = {kv["key"]: kv["value"] for kv in items[1]["state_snapshot"]["attributes"]}
    assert attrs == {"big": str(2**70), "fine": "yes", _UNMARSHALLED: "big,odd"}


# An integer that is not an `int`: numpy's, or anything with `__index__`.
# `float()` falls back to `__index__`, so before these were read EXACTLY a
# `np.uint64(2**64 - 1)` shipped as `1.8446744073709552e+19`, rounded and
# unnamed. numpy is not a test dependency; these stand in for its scalars,
# and the real ones join the parametrization wherever numpy is installed.


class _IndexOnly:
    """Only `__index__`, as a numpy integer scalar is to the encoder."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __index__(self) -> int:
        return self.n


class _Uint64Like(_IndexOnly):
    """`__index__` AND `__float__`, so the float read would also take it."""

    def __float__(self) -> float:
        return float(self.n)


def _numpy_scalars() -> list:
    try:
        import numpy
    except ImportError:
        return []
    return [
        pytest.param(numpy.uint64(2**64 - 1), str(2**64 - 1), id="numpy.uint64-max"),
        pytest.param(numpy.uint64(2**63), str(2**63), id="numpy.uint64-2**63"),
        pytest.param(numpy.int64(-5), -5, id="numpy.int64"),
    ]


@pytest.mark.parametrize("encode", _EXPORT_ENCODERS)
@pytest.mark.parametrize(
    ("value", "shipped"),
    [
        pytest.param(_IndexOnly(2**70), str(2**70), id="__index__-2**70"),
        pytest.param(_Uint64Like(2**64 - 1), str(2**64 - 1), id="uint64-like-max"),
        pytest.param(_IndexOnly(-(2**63) - 1), str(-(2**63) - 1), id="__index__-below-int64"),
        pytest.param(_IndexOnly(5), 5, id="__index__-in-range"),
        *_numpy_scalars(),
    ],
)
def test_an_integer_that_is_not_an_int_is_read_exactly(encode, value, shipped):
    bad = encode(_batch(value))["bad"]
    assert bad["req_id"] == shipped and type(bad["req_id"]) is type(shipped)
    if isinstance(shipped, str):
        assert bad[_UNMARSHALLED] == "req_id"
    else:
        assert _UNMARSHALLED not in bad


class _FakeNumpyBool:
    def __init__(self, value: bool) -> None:
        self.value = value

    def __bool__(self) -> bool:
        return self.value

    def __float__(self) -> float:
        return float(self.value)


@pytest.mark.parametrize("encode", _EXPORT_ENCODERS)
def test_a_numpy_bool_ships_as_a_bool_not_as_its_float(encode, monkeypatch):
    # numpy 2's `bool_` is neither an `int` nor an `__index__`, so the float
    # read took it as `1.0`. It is recognized through `sys.modules`, never by
    # importing numpy, which this stand-in module exercises without numpy.
    fake = types.ModuleType("numpy")
    fake.bool_ = _FakeNumpyBool  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "numpy", fake)
    for flag in (True, False):
        bad = encode(_batch(_FakeNumpyBool(flag)))["bad"]
        assert bad["req_id"] is flag
        assert _UNMARSHALLED not in bad


@pytest.mark.parametrize("encode", _EXPORT_ENCODERS)
@pytest.mark.parametrize(
    ("key", "named"),
    [
        pytest.param(123, "<non-str key>", id="int-key"),
        pytest.param(b"req_id", "<non-str key>", id="bytes-key"),
        pytest.param("\ud800", "<unencodable key>", id="lone-surrogate-key"),
    ],
)
def test_a_key_the_wire_cannot_hold_is_left_out_named_and_the_batch_ships(encode, key, named):
    good = _span(name="good", extra=(("n", 1),))
    bad = _span(
        name="bad",
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x03" * 8)),
        extra=(("before", "kept"), (key, "x"), ("after", 2.5)),
    )
    spans = encode(Envelope(header=_header(), spans=(good, bad)))
    assert set(spans) == {"good", "bad"}, "one key cost the whole batch"
    # Named by a fixed placeholder: spelling the key would run its host code.
    assert spans["bad"] == {"before": "kept", "after": 2.5, _UNMARSHALLED: named}


@pytest.mark.parametrize("encode", _EXPORT_ENCODERS)
def test_the_note_replaces_a_host_attribute_of_the_same_name(encode):
    # Pushed beside it, the key would ship twice, and a decoder keeps one of a
    # duplicated key -- the backend's choice which.
    env = Envelope(
        header=_header(),
        spans=(_span(extra=((_UNMARSHALLED, "host value"), ("odd", object()))),),
    )
    if encode is _otlp_spans:
        bodies, *_ = _wardex_native.codec.encode_otlp_requests(env, "off", [], None, False)
        decoded = _wardex_native.codec.decode_otlp_traces(bodies[0])
        assert bodies[0].count(_UNMARSHALLED.encode()) == 1
        (span,) = decoded["resource_spans"][0]["scope_spans"][0]["spans"]
        assert span["attributes"][_UNMARSHALLED] == "odd"
    else:
        body, *_ = _wardex_native.codec.encode_envelope_export(env, "off", [], None)
        (item,) = _codec.decode(body)["items"]
        notes = [kv["value"] for kv in item["span"]["extra"] if kv["key"] == _UNMARSHALLED]
        assert notes == ["odd"]
    # With nothing to name, the host's own attribute is left as it was.
    clean = Envelope(header=_header(), spans=(_span(extra=((_UNMARSHALLED, "host value"),)),))
    (attrs,) = encode(clean).values()
    assert attrs[_UNMARSHALLED] == "host value"
