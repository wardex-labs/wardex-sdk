"""Events Hermes sends with an id it does not have, asserted on exported spans and on status.

Hermes sends an id it lacks as "" (`model_tools.py::_CallIds.hook_kwargs`). A tool run before the
turn's first model call has no `api_request_id`; a tool that code in `execute_code` calls has no
`tool_call_id` either. Each still ran, so each is a span, and each states only what Hermes said.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import pytest

import wardex_sdk as wardex
from _hermes_payloads import code_calls, compress, dispatch, prelude_tool, turnless_aux
from _hermes_support import (
    exported,
    extras,
    forget_attachments,
    hermes,
    hermes_installed,
    on_loop,
    one,
    outline,
    plugin,
    start_wardex,
    status,
)
from wardex_sdk._assembly import counters

WIRE = "HTTP POST /v1/chat/completions"


@pytest.fixture(autouse=True)
def _installed(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield
    forget_attachments()


def _tree(text: str) -> str:
    return textwrap.dedent(text).strip()


def test_a_tool_run_before_the_first_model_call_is_a_span_that_states_its_turn():
    counters.reset()
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            prelude_tool(h, turn)
            h.model_call(turn)
            h.end(turn)
            return turn

        turn = on_loop(run)
        row = status()
    nodes = exported(transport)
    tool = one(nodes, "execute_tool read_file")
    assert extras(tool)["turn_id"] == turn.turn_id
    assert tool.attributes["gen_ai.tool.call.id"].startswith("call_prelude_")
    assert "api_request_id" not in extras(tool)
    assert outline(nodes) == _tree(f"""
    execute_tool read_file | unresolved 0.0 | OK | parent_unresolved
    invoke_agent hermes | trace_root 1.0 | OK
      chat lab-model | unit_active 1.0 | OK
        {WIRE} | contextvar 1.0 | OK | provider_inferred, reassembled_from_stream
    """)
    assert wardex.diagnostics().losses == ()
    assert row.detail == ""


def test_tools_code_in_execute_code_calls_nest_under_it_with_no_call_id():
    counters.reset()
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            call = h.model_call(turn, tool_calls=["execute_code"])

            def code() -> None:
                dispatch(h, turn, "web_search", inner=lambda: dispatch(h, turn, "read_file"))
                dispatch(h, turn, "read_file")

            code_calls(h, call, code)
            h.model_call(turn)
            h.end(turn)

        on_loop(run)
        row = status()
    nodes = exported(transport)
    # Their reports name no frame, so the frames close at exit, UNSET: a tool that returned has
    # not had Hermes's verdict on it.
    assert outline(nodes) == _tree(f"""
    invoke_agent hermes | trace_root 1.0 | OK
      chat lab-model | unit_active 1.0 | OK
        {WIRE} | contextvar 1.0 | OK | provider_inferred, reassembled_from_stream
      chat lab-model | unit_active 1.0 | OK
        {WIRE} | contextvar 1.0 | OK | provider_inferred, reassembled_from_stream
      execute_tool execute_code | unit_active 1.0 | OK
        execute_tool read_file | unit_active 1.0 | UNSET
        execute_tool web_search | unit_active 1.0 | UNSET
          execute_tool read_file | unit_active 1.0 | UNSET
    """)
    called = [n for n in nodes if n.name in ("execute_tool web_search", "execute_tool read_file")]
    assert len(called) == 3
    for node in called:
        assert "gen_ai.tool.call.id" not in node.attributes
        assert "api_request_id" not in extras(node)
    assert wardex.diagnostics().losses == ()
    assert row.detail == ""


def test_an_auxiliary_call_outside_any_turn_is_marked_and_drops_nothing(caplog):
    counters.reset()
    transport = start_wardex()
    with caplog.at_level(logging.WARNING, logger="wardex_sdk"):
        with plugin() as ctx, hermes(ctx) as h:
            on_loop(turnless_aux, h, name="mcp-loop")
            row = status()
    # Hermes named no turn, so the attempt keeps its own trace and says it has no proven parent.
    assert outline(exported(transport)) == f"{WIRE} | unresolved 0.0 | OK | " + ", ".join(
        ("parent_unresolved", "provider_inferred")
    )
    assert wardex.diagnostics().losses == ()
    assert row.detail == ""
    assert [r.getMessage() for r in caplog.records] == []


def _conversations(nodes) -> set[str]:
    return {n.attributes["gen_ai.conversation.id"] for n in nodes if n.name != WIRE}


@pytest.mark.parametrize("joined", [True, False], ids=["joined", "background"])
def test_a_subagent_states_the_conversation_hermes_gave_it_its_parents(joined: bool):
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.delegate(h.model_call(turn, tool_calls=["delegate_task"]), 2, joined=joined)
            h.model_call(turn)
            h.end(turn)
            h.run_background()
            return turn

        turn = on_loop(run)
    nodes = exported(transport)
    assert len([n for n in nodes if n.name == "invoke_agent hermes"]) == 3
    assert _conversations(nodes) == {turn.session_id}


def test_a_session_hermes_compressed_keeps_its_conversation_and_its_finalize():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            first = h.turn()
            h.model_call(first)
            rotated = compress(first)
            h.model_call(first)
            # Hermes finalizes by the id it rotated to, so this cuts the turn in flight.
            h.finalize(rotated)
            second = h.turn(session_id=rotated)
            second.conversation = first.conversation
            h.model_call(second)
            h.end(second)
            return first

        first = on_loop(run)
    nodes = exported(transport)
    roots = [n for n in nodes if n.name == "invoke_agent hermes"]
    assert [extras(r)["close_cause"] for r in roots] == ["finalize", "end_event"]
    # No tool event stated the conversation: the turn kept it through the rotation.
    assert _conversations(nodes) == {first.conversation}


def test_a_later_turn_takes_the_conversation_hermes_stated_for_its_session():
    """No payload before a turn's first tool event states Hermes's `conversation_id`, so the
    turn that first states one (here a session resumed after an earlier process compressed it)
    opened with its session id; every later turn of that session takes Hermes's."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            first = h.turn()
            first.conversation = "20261121_000000_lineage"
            h.tool(h.model_call(first, tool_calls=["read_file"]), "read_file")
            h.model_call(first)
            h.end(first)
            second = h.turn(session_id=first.session_id)
            h.model_call(second)
            h.end(second)
            return second

        second = on_loop(run)
    nodes = exported(transport)
    mine = [n for n in nodes if extras(n).get("turn_id") == second.turn_id]
    assert [n.attributes["gen_ai.conversation.id"] for n in mine] == ["20261121_000000_lineage"]
