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
from _hermes_payloads import code_calls, dispatch, prelude_tool, turnless_aux
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
