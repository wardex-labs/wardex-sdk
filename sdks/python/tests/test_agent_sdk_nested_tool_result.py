"""A tool result produced inside a sub-agent belongs to the call it answers.

The CLI writes ONE `user` line for it and puts two different identifiers on it:
`parent_tool_use_id` says which SUB-AGENT produced the line, and the
`tool_result` block's `tool_use_id` says which CALL the result answers. They are
equal for a main-agent tool and they DIVERGE for every tool a sub-agent runs —
which is the case where reading the wrong one costs the most.

Measured before the split: the result was filed against the `Task` call that
spawned the agent, so `execute_tool Task` shipped carrying the inner tool's
output and the inner call shipped no result at all. One span with bytes that
were never its own, one span missing, and no counter anywhere to find it by.
"""

from __future__ import annotations

import json

import pytest

from wardex_sdk._hub import reset_for_test
from wardex_sdk.adapters._assembler import SessionAssembler
from wardex_sdk.assembly import counters
from wardex_sdk.assembly._units import _ambient_unit
from wardex_sdk.protocol._claude_stream import parse_line


@pytest.fixture(autouse=True)
def _clean_scope():
    reset_for_test()
    token = _ambient_unit.set(None)
    counters.reset()
    yield
    _ambient_unit.reset(token)
    reset_for_test()
    counters.reset()


class _Client:
    def __init__(self) -> None:
        self.spans: list = []

    def capture_span(self, span) -> None:  # noqa: ANN001
        self.spans.append(span)


def _assistant_turn(*calls: tuple[str, str]) -> dict:
    """An assistant turn announcing tool calls, as the CLI writes it."""
    return {
        "type": "assistant",
        "session_id": "s-1",
        "message": {
            "id": "msg_1",
            "model": "claude-opus-4",
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": call_id, "name": name, "input": {}}
                for call_id, name in calls
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }


def _tool_result(call_id: str, output: str, *, produced_by: str | None = None) -> dict:
    line: dict = {
        "type": "user",
        "session_id": "s-1",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": output}],
        },
    }
    if produced_by is not None:
        line["parent_tool_use_id"] = produced_by
    return line


def _drive(assembler: SessionAssembler, key: int, *lines: dict) -> None:
    for line in lines:
        assembler.on_inbound(key, line)


def _tool_spans(client: _Client) -> dict[str, bytes]:
    """Every `execute_tool` span the run produced, by tool name -> its output."""
    out = {}
    for span in client.spans:
        if span.tool is not None:
            out[span.tool.name] = span.output_data or b""
    return out


def test_a_tool_a_subagent_ran_gets_its_own_result_and_the_task_keeps_its_own() -> None:
    client = _Client()
    assembler = SessionAssembler(client)
    key = 1

    # The main agent calls Task; the sub-agent it spawns calls Bash. Both results
    # come back on `user` lines, and the sub-agent's carries BOTH identifiers.
    _drive(
        assembler,
        key,
        {"type": "system", "subtype": "init", "session_id": "s-1", "model": "claude-opus-4"},
        _assistant_turn(("toolu_TASK", "Task"), ("toolu_BASH", "Bash")),
        _tool_result("toolu_BASH", "inner output", produced_by="toolu_TASK"),
        _tool_result("toolu_TASK", "outer output"),
    )
    assembler.on_close(key, None)

    spans = _tool_spans(client)
    assert "Bash" in spans, "the call the sub-agent actually ran produced no span"
    assert b"inner output" in spans["Bash"]
    assert b"inner output" not in spans.get("Task", b""), (
        "the Task call was handed the inner tool's bytes"
    )


def test_the_two_identifiers_on_one_line_are_kept_apart() -> None:
    """The parser-level fact the assembler depends on, asserted where it is read.

    Equal for a main-agent tool, divergent for a sub-agent's — so a single field
    cannot carry both, and any reader that takes `parent_tool_use_id` for "which
    call" is correct only until a sub-agent runs something.
    """
    main = parse_line(json.dumps(_tool_result("toolu_BASH", "x")).encode(), outbound=False)
    assert main is not None
    assert (main.tool_result_id, main.parent_tool_use_id) == ("toolu_BASH", None)

    nested = parse_line(
        json.dumps(_tool_result("toolu_BASH", "x", produced_by="toolu_TASK")).encode(),
        outbound=False,
    )
    assert nested is not None
    assert (nested.tool_result_id, nested.parent_tool_use_id) == ("toolu_BASH", "toolu_TASK")


def test_a_subagents_plain_message_is_not_mistaken_for_a_tool_result() -> None:
    """It carries `parent_tool_use_id` and no result block. Refusing on that field
    admitted it as a tool result whose content was the whole message."""
    plain = {
        "type": "user",
        "session_id": "s-1",
        "parent_tool_use_id": "toolu_TASK",
        "message": {"role": "user", "content": "carry on"},
    }
    assert parse_line(json.dumps(plain).encode(), outbound=False) is None
