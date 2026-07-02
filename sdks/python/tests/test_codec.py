"""Codec round-trip — InternalEnvelope → encode → decode → field preservation."""

from __future__ import annotations

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
        api_key="k",
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


def _env(span: InternalSpan) -> InternalEnvelope:
    return InternalEnvelope(header=_header(), spans=(span,))


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
    assert h["api_key"] == "k"
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
    assert kv["gen_ai.response.finish_reasons"] == "stop"  # tuple → CSV


def test_encode_is_deterministic():
    env = _env(_span(input_data=b"x" * 100))
    assert _codec.encode(env) == _codec.encode(env)


def test_empty_envelope_roundtrips():
    out = _codec.decode(_codec.encode(InternalEnvelope(header=_header())))
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
    env = InternalEnvelope(header=_header(), spans=(_span(transport=tr),), state_snapshots=(snap,))
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
    env = InternalEnvelope(header=_header(), state_snapshots=(snap,))
    out = _codec.decode(_codec.encode(env))
    ss = out["items"][0]["state_snapshot"]
    assert ss["trace_id"] == b"\x03" * 16
    assert ss["turn_index"] == 2
    assert ss["conversation_state"] == b"history"
    assert ss["input_refs"][0]["key"] == "doc.md"
    assert ss["input_refs"][0]["content_hash"] == "sha256:abc"
