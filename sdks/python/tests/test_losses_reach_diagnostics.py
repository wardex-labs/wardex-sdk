"""Losses that used to end in a counter or a span marker now reach the ledger.

`wardex.diagnostics().losses` and strict mode read the drop ledger and nothing
else. Each case below drives a loss through the code a user's process runs and
reads it back from `diagnostics()`, the way a user would.
"""

from __future__ import annotations

from wardex_sdk import diagnostics
from wardex_sdk._adapters._hook_reach import HookReach
from wardex_sdk._assembly import _ledger
from wardex_sdk._enums import SpanKind, StatusCode
from wardex_sdk._types import (
    Envelope,
    EnvelopeHeader,
    InternalSpan,
    SdkInfo,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.testing.strict import gaps_between
from wardex_sdk.transport import OtlpHttpTransport


def _envelope(payload: bytes) -> Envelope:
    return Envelope(
        header=EnvelopeHeader(
            event_id="evt",
            sdk=SdkInfo(name="wardex.python", version="0", python_version="3", os="x", arch="y"),
            sent_at_ns=1,
        ),
        spans=(
            InternalSpan(
                context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
                name="chat",
                kind=SpanKind.CLIENT,
                start_time_ns=1,
                end_time_ns=2,
                parent_span_id=None,
                status=StatusCode.OK,
                input_data=payload,
            ),
        ),
    )


def _losses() -> dict[str, tuple[str, int]]:
    return {loss.reason: (loss.nature, loss.count) for loss in diagnostics().losses}


def test_an_attribute_cut_on_export_is_a_partial_loss_strict_mode_fails_on():
    import wardex_sdk._wardex_native as native

    transport = OtlpHttpTransport("http://127.0.0.1:1/v1/traces")
    transport._set_limits(native.Limits(max_otlp_attribute_bytes=64))
    before = diagnostics()

    transport.encode(_envelope(b"x" * 4096), compress=False)

    assert _losses()["attribute_cap"] == ("partial", 1)
    gaps = gaps_between(diagnostics(), since=before)
    assert [g.reason for g in gaps] == ["attribute_cap"]
    assert "LimitsConfig(max_otlp_attribute_bytes=...)" in gaps[0].line


def test_an_attribute_inside_the_cap_is_no_loss():
    transport = OtlpHttpTransport("http://127.0.0.1:1/v1/traces")
    transport.encode(_envelope(b"x" * 64), compress=False)
    assert "attribute_cap" not in _losses()


class _Unit:
    def __init__(self) -> None:
        self.notes: list[object] = []

    def note(self, marker: object) -> None:
        self.notes.append(marker)


class _Assembler:
    def __init__(self, unit: _Unit) -> None:
        self._unit = unit

    def unit_for(self, key: int) -> _Unit:
        return self._unit


def test_a_session_started_without_the_hooks_is_a_partial_loss_in_diagnostics():
    """The session still ships, recorded from the CLI's output alone, so it is
    degraded data, not missing data."""
    handshake = '{"type": "control_request", "request": {"subtype": "initialize", "hooks": {}}}'
    reach = HookReach(events=("PreToolUse",))
    unit = _Unit()
    reach.observe(1, handshake, _Assembler(unit))
    reach.observe(1, '{"type": "user"}', _Assembler(unit))

    assert unit.notes, "the root carries its marker as before"
    assert _losses()["hooks_absent"] == ("partial", 1)
    assert _ledger.snapshot().where[
        (_ledger.DropReason.HOOKS_ABSENT, "adapters.anthropic_agent_sdk")
    ]


def test_an_mcp_stdio_line_over_the_cap_is_a_whole_message_lost():
    """A line past max_stream_buffer_bytes is skipped unparsed: the request in it
    never becomes a span, so it is counted as lost, not as shipped short."""
    import json

    from wardex_sdk import LimitsConfig
    from wardex_sdk._interceptors._mcp_stdio import _ProcState

    state = _ProcState(limits=LimitsConfig(max_stream_buffer_bytes=1024).to_native())
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"x": "a" * 5000}}
    state.feed_request(json.dumps(call).encode() + b"\n")

    assert _losses().get("stream_line_over_cap") == ("loss", 1)
    assert "body_cap" not in _losses()
    loss = next(x for x in diagnostics().losses if x.reason == "stream_line_over_cap")
    assert loss.unit == "message"
    assert "LimitsConfig(max_stream_buffer_bytes=...)" in loss.next_step


def test_a_tool_input_that_cannot_be_serialized_is_a_partial_loss():
    from wardex_sdk._adapters._anthropic_agent_sdk import _tool_input

    class _Unserializable:
        pass

    assert _tool_input({"x": _Unserializable()}) == b""
    assert _losses()["payload_unserializable"] == ("partial", 1)
