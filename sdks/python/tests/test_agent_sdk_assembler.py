"""SessionAssembler unit tests — synthetic events, no SDK involved."""

import json

from wardex_sdk._enums import CaptureSource, StatusCode
from wardex_sdk.adapters._assembler import SessionAssembler

INIT = {"type": "system", "subtype": "init", "session_id": "s-1", "model": "claude-sonnet-5"}
ASSISTANT = {
    "type": "assistant",
    "session_id": "s-1",
    "message": {
        "id": "m1",
        "model": "claude-sonnet-5",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 25},
        "content": [
            {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": "ls"}}
        ],
    },
}
TOOL_RESULT = {
    "type": "user",
    "session_id": "s-1",
    "parent_tool_use_id": "toolu_01",
    "message": {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}],
    },
}
RESULT = {
    "type": "result",
    "subtype": "success",
    "session_id": "s-1",
    "is_error": False,
    "num_turns": 1,
    "total_cost_usd": 0.01,
    "duration_ms": 100,
    "duration_api_ms": 80,
}


class FakeClient:
    def __init__(self):
        self.spans = []

    def capture_span(self, span):
        self.spans.append(span)


def _outbound(asm, key, session="s-1", text="go"):
    asm.on_outbound(
        key,
        json.dumps(
            {"type": "user", "session_id": session, "message": {"role": "user", "content": text}}
        ),
    )


def test_happy_path_emits_root_and_chat_spans():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    for msg in (INIT, ASSISTANT, RESULT):
        asm.on_inbound(1, msg)
    asm.on_close(1, None)

    names = [s.name for s in client.spans]
    assert "chat claude-sonnet-5" in names
    root = client.spans[-1]
    assert root.name == "invoke_agent"
    assert root.status is StatusCode.OK
    assert root.conversation.session_id == "s-1"
    assert CaptureSource.ADAPTER in root.capture_sources
    assert "transport_timing_unavailable_subprocess" in root.capture_integrity.limitations
    chat = next(s for s in client.spans if s.name.startswith("chat"))
    assert chat.gen_ai.input_tokens == 10
    assert chat.gen_ai.output_tokens == 25
    assert chat.parent_span_id == root.context.span_id
    assert asm.open_session_count() == 0


def test_tool_span_from_hooks_joins_stream_content():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, ASSISTANT)
    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_input": {"command": "ls"}},
        "toolu_01",
    )
    asm.on_hook(
        "PostToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_response": "ok"}, "toolu_01"
    )
    asm.on_inbound(1, TOOL_RESULT)
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)

    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert tool.tool.call_id == "toolu_01"
    assert tool.correlation.confidence == 1.0
    assert tool.correlation.strategy == "adapter_hook"
    assert b"ls" in tool.input_data


def test_stream_only_degrades_confidence():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    for msg in (INIT, ASSISTANT, TOOL_RESULT, RESULT):
        asm.on_inbound(1, msg)
    asm.on_close(1, None)

    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert tool.correlation.confidence == 0.7
    assert tool.correlation.strategy == "adapter_stream"


def test_subagent_span_attribution():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_hook(
        "SubagentStart", {"session_id": "s-1", "agent_id": "a-1", "agent_type": "researcher"}, None
    )
    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-1", "tool_name": "Read", "tool_input": {}, "agent_id": "a-1"},
        "toolu_02",
    )
    asm.on_hook(
        "PostToolUse",
        {"session_id": "s-1", "tool_name": "Read", "tool_response": "x", "agent_id": "a-1"},
        "toolu_02",
    )
    asm.on_hook(
        "SubagentStop", {"session_id": "s-1", "agent_id": "a-1", "agent_type": "researcher"}, None
    )
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)

    sub = next(s for s in client.spans if s.name == "invoke_agent researcher")
    tool = next(s for s in client.spans if s.name == "execute_tool Read")
    assert sub.agent.id == "a-1"
    assert tool.parent_span_id == sub.context.span_id


def test_abort_closes_open_spans_with_markers():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    asm.on_hook(
        "PreToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_input": {}}, "toolu_09"
    )
    asm.on_close(1, "ProcessError")

    root = next(s for s in client.spans if s.name == "invoke_agent")
    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert root.status is StatusCode.ERROR
    assert "session_aborted" in root.capture_integrity.limitations
    assert "tool_span_unclosed" in tool.capture_integrity.limitations
    assert asm.open_session_count() == 0


def test_open_entry_cap():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1)
    asm.on_inbound(1, INIT)
    for n in range(300):
        asm.on_hook(
            "PreToolUse",
            {"session_id": "s-1", "tool_name": f"T{n}", "tool_input": {}},
            f"toolu_{n}",
        )
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)
    # capped at 256 open entries: overflow force-closed with the unclosed marker
    unclosed = [
        s
        for s in client.spans
        if s.capture_integrity and "tool_span_unclosed" in s.capture_integrity.limitations
    ]
    assert len(unclosed) == 300  # all eventually closed, none leaked
    assert asm.open_session_count() == 0
