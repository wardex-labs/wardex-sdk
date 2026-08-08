"""OTLP/HTTP traces marshaling.

InternalEnvelope -> encode_otlp_traces -> decode -> field preservation.
"""

from __future__ import annotations

from wardex_sdk import _wardex_native
from wardex_sdk._enums import (
    Direction,
    OperationName,
    Protocol,
    SpanKind,
    StatusCode,
)
from wardex_sdk._types import (
    EnvelopeHeader,
    GenAIAttributes,
    HttpMeta,
    InputRef,
    InternalEnvelope,
    InternalSpan,
    InternalStateSnapshot,
    SdkInfo,
    SpanContext,
    SpanId,
    TraceId,
    TransportAttributes,
    TransportTiming,
)


def _header() -> EnvelopeHeader:
    return EnvelopeHeader(
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
    )


def _span(**kw) -> InternalSpan:
    base = dict(
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
        parent_span_id=None,
        name="HTTP POST /v1/chat",
        kind=SpanKind.CLIENT,
        start_time_ns=1000,
        end_time_ns=2000,
        status=StatusCode.OK,
    )
    base.update(kw)
    return InternalSpan(**base)


def _envelope_with_span() -> InternalEnvelope:
    return InternalEnvelope(
        header=_header(),
        spans=(
            _span(
                server_address="api.openai.com",
                server_port=443,
                input_data=b"req-bytes",
                output_data=b"resp-bytes",
                transport=TransportAttributes(
                    protocol=Protocol.HTTP,
                    direction=Direction.OUTBOUND,
                    timing=TransportTiming(ttfb_ms=30.0),
                    http=HttpMeta(
                        method="POST",
                        url="https://api.openai.com/v1/chat",
                        status_code=200,
                    ),
                ),
            ),
        ),
    )


def _envelope_with_gen_ai(model: str, input_tokens: int) -> InternalEnvelope:
    return InternalEnvelope(
        header=_header(),
        spans=(
            _span(
                gen_ai=GenAIAttributes(
                    operation=OperationName.CHAT,
                    request_model=model,
                    input_tokens=input_tokens,
                    output_tokens=5,
                    temperature=0.7,
                    finish_reasons=("stop",),
                ),
            ),
        ),
    )


def _envelope_no_spans() -> InternalEnvelope:
    return InternalEnvelope(header=_header())


def _envelope_with_snapshot_only() -> InternalEnvelope:
    snap = InternalStateSnapshot(
        trace_id=TraceId(b"\x03" * 16),
        span_id=SpanId(b"\x04" * 8),
        timestamp_ns=999,
        snapshot_type="turn_start",
        turn_index=1,
        conversation_state=b"history",
        input_refs=(InputRef(key="doc.md", content_hash="sha256:abc"),),
    )
    return InternalEnvelope(header=_header(), state_snapshots=(snap,))


def _first_span(env: InternalEnvelope) -> dict:
    data = _wardex_native.codec.encode_otlp_traces(env)
    d = _wardex_native.codec.decode_otlp_traces(data)
    return d["resource_spans"][0]["scope_spans"][0]["spans"][0]


def test_encode_decode_roundtrip_core_fields():
    env = _envelope_with_span()
    data = _wardex_native.codec.encode_otlp_traces(env)
    assert isinstance(data, bytes) and len(data) > 0
    d = _wardex_native.codec.decode_otlp_traces(data)
    span = d["resource_spans"][0]["scope_spans"][0]["spans"][0]
    assert span["name"] == "HTTP POST /v1/chat"
    assert span["kind"] == 3  # OTLP CLIENT
    assert span["status"]["code"] == 1  # OTLP OK
    assert span["start_time_unix_nano"] == 1000
    assert span["end_time_unix_nano"] == 2000
    assert span["trace_id"] == "01" * 16
    assert span["span_id"] == "02" * 8
    attrs = span["attributes"]
    assert attrs["server.address"] == "api.openai.com"
    assert attrs["server.port"] == 443
    assert attrs["network.protocol.name"] == "http"
    assert attrs["http.request.method"] == "POST"
    assert attrs["http.response.status_code"] == 200
    # Raw I/O leaves as strings, never OTLP bytes_value: backends that
    # re-serialize attributes to JSON (Arize Phoenix) drop the whole span on a
    # bytes attribute — silently, with an HTTP 200.
    assert attrs["wardex.input_data"] == "req-bytes"
    assert attrs["wardex.output_data"] == "resp-bytes"
    assert "wardex.input_data.encoding" not in attrs
    assert "wardex.output_data.encoding" not in attrs


def test_non_utf8_payload_becomes_base64_with_encoding_marker():
    """Binary payloads (gRPC frames, compressed bodies) can't ship verbatim in
    a string attribute; they go base64 with a `.encoding` companion so a
    consumer can tell encoded binary from text that merely looks like base64."""
    import base64
    from dataclasses import replace

    binary = b"\x89PNG\xff\x00binary"
    env = _envelope_with_span()
    span = replace(env.spans[0], input_data=binary, output_data=b"plain text")
    attrs = _first_span(InternalEnvelope(header=env.header, spans=(span,)))["attributes"]
    assert attrs["wardex.input_data"] == base64.b64encode(binary).decode()
    assert attrs["wardex.input_data.encoding"] == "base64"
    assert attrs["wardex.output_data"] == "plain text"
    assert "wardex.output_data.encoding" not in attrs


def test_utf8_with_nul_is_binary_not_text():
    """U+0000 is valid UTF-8, but Postgres-backed ingests (Phoenix-on-Postgres,
    Langfuse) reject any string containing NUL — shipping it verbatim would
    reintroduce the exact silent span loss the string surface exists to prevent.
    Zero-value
    protobuf/gRPC payload bytes are the realistic producer."""
    import base64
    from dataclasses import replace

    payload = b"name: alice\x00\x00\x00\x00"  # valid UTF-8, contains NUL
    env = _envelope_with_span()
    span = replace(env.spans[0], input_data=payload)
    attrs = _first_span(InternalEnvelope(header=env.header, spans=(span,)))["attributes"]
    assert attrs["wardex.input_data"] == base64.b64encode(payload).decode()
    assert attrs["wardex.input_data.encoding"] == "base64"


def test_user_extra_cannot_collide_with_or_spoof_the_encoding_companion():
    """`<key>.encoding` is the debyte pass's namespace for keys it rewrote.
    A user extra sitting on that key would either duplicate the companion
    (duplicate OTLP keys — backend dedup order decides which wins) or claim an
    encoding the verbatim branch never applied. Both get dropped; a user
    `.encoding` suffix on a key that never carried bytes is left alone."""
    from dataclasses import replace

    env = _envelope_with_span()
    span = replace(
        env.spans[0],
        input_data=b"\xff\xfebinary",  # -> base64: companion must win
        output_data=b"plain text",  # -> verbatim: spoofed marker must vanish
        extra=(
            ("wardex.input_data.encoding", "gzip"),
            ("wardex.output_data.encoding", "base64"),
            ("myapp.blob.encoding", "hex"),  # not a rewritten key: untouched
        ),
    )
    data = _wardex_native.codec.encode_otlp_traces(
        InternalEnvelope(header=env.header, spans=(span,))
    )
    d = _wardex_native.codec.decode_otlp_traces(data)
    span_out = d["resource_spans"][0]["scope_spans"][0]["spans"][0]
    keys = [k for k in span_out["attributes"] if k.endswith(".encoding")]
    attrs = span_out["attributes"]
    assert attrs["wardex.input_data.encoding"] == "base64"
    assert "wardex.output_data.encoding" not in attrs
    assert attrs["myapp.blob.encoding"] == "hex"
    assert sorted(keys) == ["myapp.blob.encoding", "wardex.input_data.encoding"]


def test_gen_ai_flattened_to_attributes():
    env = _envelope_with_gen_ai(model="gpt-4o", input_tokens=10)
    attrs = _first_span(env)["attributes"]
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["gen_ai.request.temperature"] == 0.7
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.response.finish_reasons"] == "stop"


def test_an_llm_span_is_named_for_its_operation_and_model():
    """gen_ai semconv names an LLM span `{operation} {model}`.

    The transport name it replaces — `HTTP POST /v1/chat` — is the same string
    for every model, prompt and provider behind one endpoint, and span name is
    the axis every backend groups by. A latency or cost breakdown over an
    agent's LLM calls collapsed into a single bucket that answers no question
    anyone asks, while the two facts a reader groups by sat one level down in
    the attributes.
    """
    assert _first_span(_envelope_with_gen_ai(model="gpt-4o", input_tokens=10))["name"] == (
        "chat gpt-4o"
    )


def test_an_llm_span_with_no_request_model_is_named_for_the_operation_alone():
    """Not `chat unknown`: that invents a model of that name, and a backend
    aggregates it as one."""
    env = InternalEnvelope(
        header=_header(),
        spans=(_span(gen_ai=GenAIAttributes(operation=OperationName.EMBEDDINGS)),),
    )
    assert _first_span(env)["name"] == "embeddings"


def test_a_span_without_llm_semantics_keeps_its_own_name():
    """The negative control for the rename above: it is a rename for LLM spans
    and a no-op for everything else, so a plain HTTP span still says what it
    was."""
    assert _first_span(_envelope_with_span())["name"] == "HTTP POST /v1/chat"


def test_resource_service_name():
    env = _envelope_with_span()
    d = _wardex_native.codec.decode_otlp_traces(_wardex_native.codec.encode_otlp_traces(env))
    res_attrs = d["resource_spans"][0]["resource"]["attributes"]
    assert res_attrs["service.name"] == "wardex.python"
    assert res_attrs["service.version"] == "0.1.0"
    assert res_attrs["telemetry.sdk.name"] == "wardex.python"
    assert res_attrs["telemetry.sdk.version"] == "0.1.0"
    assert res_attrs["telemetry.sdk.language"] == "python"


def test_instrumentation_scope():
    env = _envelope_with_span()
    d = _wardex_native.codec.decode_otlp_traces(_wardex_native.codec.encode_otlp_traces(env))
    scope = d["resource_spans"][0]["scope_spans"][0]["scope"]
    assert scope["name"] == "wardex.python"
    assert scope["version"] == "0.1.0"


def test_determinism():
    env = _envelope_with_span()
    a = _wardex_native.codec.encode_otlp_traces(env)
    b = _wardex_native.codec.encode_otlp_traces(env)
    assert a == b


def test_empty_envelope_no_resource_spans():
    env = _envelope_no_spans()
    d = _wardex_native.codec.decode_otlp_traces(_wardex_native.codec.encode_otlp_traces(env))
    assert d["resource_spans"] == []


def test_state_snapshot_skipped():
    env = _envelope_with_snapshot_only()
    d = _wardex_native.codec.decode_otlp_traces(_wardex_native.codec.encode_otlp_traces(env))
    assert d["resource_spans"] == []


# ==========================================================================
# What wardex knows about its own uncertainty, on the path that leaves the process
# ==========================================================================


def _envelope_with_uncertainty() -> InternalEnvelope:
    from wardex_sdk._types import CaptureIntegrity, CorrelationInfo
    from wardex_sdk.assembly import Limitation
    from wardex_sdk.assembly._parentage import ParentSource

    return InternalEnvelope(
        header=_header(),
        spans=(
            _span(
                correlation=CorrelationInfo(
                    strategy=ParentSource.UNIT_SOLE,
                    confidence=0.5,
                    request_id="toolu_9",
                ),
                capture_integrity=CaptureIntegrity(
                    limitations=(
                        Limitation.UNIT_INFERRED_SOLE,
                        Limitation.BODY_CAP_EXCEEDED,
                    ),
                    request_body_captured=True,
                    response_body_captured=False,
                    truncated=True,
                    dropped_chunk_count=3,
                ),
            ),
        ),
    )


def test_a_guessed_edge_says_so_on_the_otlp_wire():
    """OTLP is the only transport exported from the package root, so a marker
    that reaches the wardex envelope and not this encoder reaches nobody.

    Every marker this SDK spends its design on says what could NOT be
    established. Stripping them here left a 0.5 guess and a 1.0 fact
    indistinguishable for every user on the documented path — the silent-loss
    shape `events_to_otlp` names for a different field in the same file.
    """
    attrs = _first_span(_envelope_with_uncertainty())["attributes"]

    assert attrs["wardex.parent_source"] == "unit_sole"
    assert attrs["wardex.parent_confidence"] == 0.5
    assert attrs["wardex.correlation.request_id"] == "toolu_9"


def test_every_limitation_a_span_carries_reaches_the_otlp_wire():
    attrs = _first_span(_envelope_with_uncertainty())["attributes"]

    assert attrs["wardex.limitations"] == ["unit_inferred_sole", "body_cap_exceeded"]


def test_what_was_and_was_not_captured_reaches_the_otlp_wire():
    """The four capture flags are always present because FALSE is their
    informative reading; truncation and drops are events, so they appear only
    when they happened."""
    attrs = _first_span(_envelope_with_uncertainty())["attributes"]

    assert attrs["wardex.capture.request_body"] is True
    assert attrs["wardex.capture.response_body"] is False
    assert attrs["wardex.capture.truncated"] is True
    assert attrs["wardex.capture.dropped_chunks"] == 3


def test_a_span_with_nothing_to_report_carries_no_uncertainty_attributes():
    """Absence has to stay legible: a span that reports no limitation must not
    be padded with keys that make it look examined and cleared."""
    attrs = _first_span(_envelope_with_span())["attributes"]

    assert "wardex.limitations" not in attrs
    assert "wardex.capture.truncated" not in attrs
    assert "wardex.parent_source" not in attrs
