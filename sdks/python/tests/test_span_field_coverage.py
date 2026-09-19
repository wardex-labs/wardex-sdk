"""Every `InternalSpan` field reaches the wire, in the place a reader looks.

The escape this file closes has happened four times. A field is added to
`InternalSpan`, an adapter fills it faithfully, and the encoder — which reads
the span attribute by attribute in Rust — is never taught about it, so the
value is dropped at export with no error and every test green. The
conversation block was the worst of them: it was dropped outright, then
written into `extra` while the typed field the schema declares for it stayed
empty, and a receiver reading that typed field stored an empty conversation
id for every span. `call_site` was filled by the tracing decorator and never
encoded at all. Each was found by a person decoding a real encode and looking
for the value, never by a failing run.

So the check is a round trip with a sentinel per field, and each row looks for
its sentinel in the field's HOME — the typed slot when the schema has one, the
documented `extra` key when the block is flattened. Reading the encoder's
source for the attributes it touches would not do: the conversation escape
read the attribute and wrote it to the wrong place.

A field that is deliberately not encoded goes in `_NOT_ON_THE_WIRE` with the
reason. The table is empty, and the way to keep it empty is to delete a field
nothing ships rather than list it.

WHAT THIS DOES NOT REACH, so that a green run is not read as more than it is:

* One sentinel per TOP-LEVEL field. A block that ships while one of its
  sub-fields is dropped passes: `GenAIAttributes.system_instructions` and
  `tool_definitions_hash` are not encoded today, and no row here fails.
* The ENVELOPE only. A field present on the envelope and absent from the OTLP
  export — which is what `capture_sources` was — cannot fail it; the OTLP side
  is asserted attribute by attribute in `test_otlp_codec.py`.
* What the Python DECODER returns. A value that is on the wire but that
  `decode()` does not surface reads here as missing, not as present.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from test_codec import _env, _span
from wardex_sdk._assembly._parentage import ParentSource
from wardex_sdk._enums import CaptureSource, OperationName, SpanKind, StatusCode
from wardex_sdk._types import (
    AgentAttributes,
    CallSite,
    CaptureIntegrity,
    ConversationContext,
    CorrelationInfo,
    EmbeddingsAttributes,
    EvaluationAttributes,
    GenAIAttributes,
    HttpMeta,
    InternalSpan,
    InternalSpanEvent,
    InternalSpanLink,
    RetrievalAttributes,
    SpanContext,
    SpanId,
    ToolAttributes,
    TraceId,
    TransportAttributes,
)
from wardex_sdk.transport import _codec


def _extra(span: dict) -> dict:
    return {kv["key"]: kv["value"] for kv in span.get("extra", [])}


#: field name -> (the sentinel the span is built with, where it must be found).
#: The check receives the decoded span; it names the field's home explicitly.
_ROWS: dict[str, tuple[Any, Callable[[dict], bool]]] = {
    "context": (
        SpanContext(trace_id=TraceId(b"\xa1" * 16), span_id=SpanId(b"\xa2" * 8)),
        lambda s: s["trace_id"] == b"\xa1" * 16 and s["span_id"] == b"\xa2" * 8,
    ),
    "parent_span_id": (SpanId(b"\xa3" * 8), lambda s: s["parent_span_id"] == b"\xa3" * 8),
    "name": ("sentinel-name", lambda s: s["name"] == "sentinel-name"),
    "kind": (SpanKind.SERVER, lambda s: s["kind"] == 3),
    "start_time_ns": (1234, lambda s: s["start_time_unix_nano"] == 1234),
    "end_time_ns": (5678, lambda s: s["end_time_unix_nano"] == 5678),
    "status": (StatusCode.ERROR, lambda s: s["status"]["code"] == 2),
    "status_message": ("sentinel-status", lambda s: s["status"]["message"] == "sentinel-status"),
    "gen_ai": (
        GenAIAttributes(operation=OperationName.CHAT, request_model="sentinel-model"),
        lambda s: _extra(s)["gen_ai.request.model"] == "sentinel-model",
    ),
    "agent": (
        AgentAttributes(name="sentinel-agent"),
        lambda s: _extra(s)["gen_ai.agent.name"] == "sentinel-agent",
    ),
    "tool": (
        ToolAttributes(name="sentinel-tool"),
        lambda s: _extra(s)["gen_ai.tool.name"] == "sentinel-tool",
    ),
    "transport": (
        TransportAttributes(
            request_size=4321, http=HttpMeta(method="PUT", url="http://sentinel/", status_code=418)
        ),
        lambda s: (
            s["transport"]["request_size"] == 4321 and s["transport"]["http"]["status_code"] == 418
        ),
    ),
    "retrieval": (
        RetrievalAttributes(data_source_id="sentinel-ds"),
        lambda s: _extra(s)["gen_ai.data_source.id"] == "sentinel-ds",
    ),
    "embeddings": (
        EmbeddingsAttributes(dimension_count=4321),
        lambda s: _extra(s)["gen_ai.embeddings.dimension.count"] == 4321,
    ),
    "evaluation": (
        EvaluationAttributes(name="sentinel-eval"),
        lambda s: _extra(s)["gen_ai.evaluation.name"] == "sentinel-eval",
    ),
    "conversation": (
        ConversationContext(
            conversation_id="sentinel-conv", session_id="sentinel-sess", turn_index=7
        ),
        lambda s: (
            s["conversation"]
            == {"conversation_id": "sentinel-conv", "session_id": "sentinel-sess", "turn_index": 7}
        ),
    ),
    "call_site": (
        CallSite(file="/sentinel.py", line=4321, function="sentinel_fn", module="sentinel_mod"),
        lambda s: (
            s["call_site"]
            == {
                "file": "/sentinel.py",
                "line": 4321,
                "function": "sentinel_fn",
                "module": "sentinel_mod",
            }
        ),
    ),
    "error_type": ("SentinelError", lambda s: s["error_type"] == "SentinelError"),
    "server_address": ("sentinel.host", lambda s: s["server_address"] == "sentinel.host"),
    "server_port": (4321, lambda s: s["server_port"] == 4321),
    "workflow_name": ("sentinel-workflow", lambda s: s["workflow_name"] == "sentinel-workflow"),
    "capture_sources": (
        (CaptureSource.MANUAL, CaptureSource.SSL),
        lambda s: s["capture_sources"] == [6, 2],
    ),
    "capture_integrity": (
        CaptureIntegrity(request_body_captured=True, truncated=True),
        lambda s: (
            s["capture_integrity"]["request_body_captured"] is True
            and s["capture_integrity"]["truncated"] is True
        ),
    ),
    "correlation": (
        CorrelationInfo(confidence=0.5, strategy=ParentSource.CONTEXTVAR),
        lambda s: s["correlation"] == {"strategy": "contextvar", "confidence": 0.5},
    ),
    "input_data": (b"sentinel-in", lambda s: s["input_data"] == b"sentinel-in"),
    "output_data": (b"sentinel-out", lambda s: s["output_data"] == b"sentinel-out"),
    "extra": (
        (("sentinel.key", "sentinel-value"),),
        lambda s: _extra(s)["sentinel.key"] == "sentinel-value",
    ),
    "events": (
        (InternalSpanEvent(name="sentinel-event", timestamp_ns=4321),),
        lambda s: (
            [(e["name"], e["timestamp_ns"]) for e in s["events"]] == [("sentinel-event", 4321)]
        ),
    ),
    "links": (
        (InternalSpanLink(trace_id=TraceId(b"\xb1" * 16), span_id=SpanId(b"\xb2" * 8)),),
        lambda s: (
            [(ln["trace_id"], ln["span_id"]) for ln in s["links"]] == [(b"\xb1" * 16, b"\xb2" * 8)]
        ),
    ),
}

#: field name -> why it deliberately never leaves the process. EMPTY: a field
#: nothing ships is deleted, not listed. An entry needs a reason a stranger
#: could act on.
_NOT_ON_THE_WIRE: dict[str, str] = {}

_FIELDS = tuple(f.name for f in dataclasses.fields(InternalSpan))


def test_every_span_field_has_a_row_or_a_stated_reason():
    """A field added to `InternalSpan` without a row here fails on the set
    equality — which is the moment somebody has to say where it goes."""
    assert set(_ROWS) | set(_NOT_ON_THE_WIRE) == set(_FIELDS)
    assert not set(_ROWS) & set(_NOT_ON_THE_WIRE), "a field is both shipped and excused"
    for name, reason in _NOT_ON_THE_WIRE.items():
        assert len(reason.split()) >= 5, f"{name}: a reason a stranger could act on"


@pytest.mark.parametrize("name", [n for n in _FIELDS if n in _ROWS])
def test_the_field_round_trips_into_its_home(name: str):
    sentinel, lands = _ROWS[name]
    span = _codec.decode(_codec.encode(_env(_span(**{name: sentinel}))))["items"][0]["span"]
    try:
        landed = lands(span)
    except KeyError as missing:
        pytest.fail(f"`InternalSpan.{name}` left no {missing} on the decoded span: {span!r}")
    assert landed, f"`InternalSpan.{name}` did not land where its row looks: {span!r}"
