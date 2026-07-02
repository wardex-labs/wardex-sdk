import dataclasses

import pytest

from wardex_sdk import _types
from wardex_sdk._enums import OperationName, ProviderName, SpanKind


def test_trace_id_generate_is_16_bytes():
    tid = _types.TraceId.generate()
    assert len(tid.value) == 16
    assert len(tid.hex()) == 32


def test_span_id_generate_is_8_bytes():
    sid = _types.SpanId.generate()
    assert len(sid.value) == 8
    assert len(sid.hex()) == 16


def test_internal_span_is_frozen():
    span = _types.InternalSpan(
        context=_types.SpanContext(_types.TraceId.generate(), _types.SpanId.generate()),
        parent_span_id=None,
        name="llm-call",
        kind=SpanKind.CLIENT,
        start_time_ns=1,
        end_time_ns=2,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.name = "x"  # type: ignore[misc]


def test_genai_attributes_open_enum_union():
    a = _types.GenAIAttributes(operation=OperationName.CHAT, provider=ProviderName.ANTHROPIC)
    b = _types.GenAIAttributes(operation="custom_op", provider="custom_provider")
    assert a.operation == OperationName.CHAT
    assert b.provider == "custom_provider"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: _types.ToolDefinitionSet(),
        lambda: _types.TransportAttributes(),
        lambda: _types.CaptureIntegrity(),
        lambda: _types.CorrelationInfo(),
        lambda: _types.RetrievalAttributes(),
        lambda: _types.EmbeddingsAttributes(),
        lambda: _types.EvaluationAttributes(),
        lambda: _types.ClientReport(timestamp_ns=0),
        lambda: _types.SdkInfo(
            name="wardex.python", version="0.1.0", python_version="3.10", os="darwin", arch="arm64"
        ),
    ],
)
def test_all_dataclasses_constructible(factory):
    obj = factory()
    assert obj is not None


def test_inputref_and_snapshot_refs():
    from wardex_sdk import _types

    ref = _types.InputRef(key="POORCODE.md", content_hash="sha256:9f8e")
    assert ref.blob_ref is None
    snap = _types.InternalStateSnapshot(
        trace_id=_types.TraceId.generate(),
        span_id=_types.SpanId.generate(),
        timestamp_ns=1,
        snapshot_type="turn_start",
        attributes=(("code.cwd", "/x"),),
        input_refs=(ref,),
    )
    assert snap.input_refs[0].key == "POORCODE.md"
    assert snap.attributes == (("code.cwd", "/x"),)


def test_transport_attributes_connection_reused_default_and_set():
    from wardex_sdk import _types

    assert _types.TransportAttributes().connection_reused is False
    t = _types.TransportAttributes(connection_reused=True)
    assert t.connection_reused is True


def test_transport_timing_has_ttft_default():
    from wardex_sdk._types import TransportTiming

    t = TransportTiming()
    assert t.ttft_ms == 0.0
