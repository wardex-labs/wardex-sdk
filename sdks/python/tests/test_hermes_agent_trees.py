"""The trees a Hermes turn exports, one test per case, asserted on exported spans only.

A tree is drawn by `outline`: each span's name, where its parent came from and how sure that is,
its status (with `error.type`), and its markers. The wire spans are the socket seam's record of
each HTTP attempt to the scripted provider; the adapter adds the turn, one `chat` per model call
frame (every attempt of a call sits under it), and one `execute_tool` per tool frame.
"""

from __future__ import annotations

import textwrap
import threading
from pathlib import Path

import pytest

from _hermes_fake import Reply
from _hermes_scripted import FakeHermes
from _hermes_support import (
    exported,
    extras,
    forget_attachments,
    hermes,
    hermes_installed,
    links,
    on_loop,
    one,
    outline,
    plugin,
    received,
    start_wardex,
    wait_for,
)
from wardex_sdk._adapters import _hermes_agent_attach as attach_module
from wardex_sdk._assembly import LinkReason

WIRE = "HTTP POST /v1/chat/completions"


@pytest.fixture(autouse=True)
def _installed(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield
    forget_attachments()


def _tree(text: str) -> str:
    return textwrap.dedent(text).strip()


def test_a_single_turn_is_one_trace_named_for_hermes():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.model_call(turn)
            h.end(turn)
            return turn

        turn = on_loop(run)
    nodes = exported(transport)
    assert outline(nodes) == _tree(f"""
    invoke_agent hermes | trace_root 1.0 | OK
      chat lab-model | unit_active 1.0 | OK
        {WIRE} | contextvar 1.0 | OK | provider_inferred, reassembled_from_stream
    """)
    root = one(nodes, "invoke_agent hermes")
    assert root.attributes["gen_ai.agent.name"] == "hermes"
    assert root.attributes["wardex.agent.type"] == "primary"
    assert root.attributes["gen_ai.conversation.id"] == turn.session_id
    assert extras(root) == {
        "turn_id": turn.turn_id,
        "close_cause": "end_event",
        "exit_reason": "text_response(finish_reason=stop)",
    }
    chat = one(nodes, "chat lab-model")
    assert extras(chat)["api_request_id"] == f"{turn.turn_id}:api:1"
    assert chat.attributes["gen_ai.conversation.id"] == turn.session_id


def test_one_tool_round_puts_the_tool_under_the_turn_linked_to_the_call_that_asked():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.tool(h.model_call(turn, tool_calls=["read_file"]), "read_file")
            h.model_call(turn)
            h.end(turn)

        on_loop(run)
    nodes = exported(transport)
    assert outline(nodes) == _tree(f"""
    invoke_agent hermes | trace_root 1.0 | OK
      chat lab-model | unit_active 1.0 | OK
        {WIRE} | contextvar 1.0 | OK | provider_inferred, reassembled_from_stream
      chat lab-model | unit_active 1.0 | OK
        {WIRE} | contextvar 1.0 | OK | provider_inferred, reassembled_from_stream
      execute_tool read_file | unit_active 1.0 | OK
    """)
    tool = one(nodes, "execute_tool read_file")
    asked = next(n for n in nodes if extras(n).get("api_request_id", "").endswith(":api:1"))
    raw = next(s for e in transport.envelopes for s in e.spans if s.context.span_id == tool.span_id)
    assert [(link.reason, link.span_id) for link in raw.links] == [
        (LinkReason.TRIGGERED_BY, asked.span_id)
    ]
    assert tool.attributes["gen_ai.tool.call.arguments"] == "{'input': '<synthetic>'}"
    assert tool.attributes["gen_ai.tool.call.result"] == "<synthetic result>"


def test_a_retried_call_holds_both_attempts_under_one_chat():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.model_call(turn, (Reply.UNAVAILABLE, Reply.OK))
            h.end(turn)

        on_loop(run)
    assert outline(exported(transport)) == _tree(f"""
    invoke_agent hermes | trace_root 1.0 | OK
      chat lab-model | unit_active 1.0 | OK
        {WIRE} | contextvar 1.0 | ERROR(503)
        {WIRE} | contextvar 1.0 | OK | provider_inferred
    """)


def _failed_turn(h: FakeHermes) -> None:
    h.fail(h.turn())


def test_a_failed_turn_closes_unset_aborted_with_the_error_on_the_call():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(_failed_turn, h)
    nodes = exported(transport)
    assert outline(nodes) == _tree(f"""
    invoke_agent hermes | trace_root 1.0 | UNSET | session_aborted
      chat lab-model | unit_active 1.0 | ERROR(InternalServerError)
        {WIRE} | contextvar 1.0 | ERROR(500)
        {WIRE} | contextvar 1.0 | ERROR(500)
    """)
    root = one(nodes, "invoke_agent hermes")
    assert extras(root)["close_cause"] == "retries_spent"
    assert extras(root)["error_reason"] == "server_error"
    assert "exit_reason" not in extras(root), "Hermes sent no end, so no exit reason is stated"


def test_a_failed_turn_is_exported_by_the_wardex_thread_with_no_flush_on_a_hermes_thread(
    monkeypatch,
):
    """Hermes sends no end for a failed turn, and its CLI gateway may exit right after; the
    close only wakes `wardex-hermes`, which flushes. No Hermes thread waits on export."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        flushed_on: list[str] = []
        real = attach_module._EXPORT._flush
        assert real is not None

        def spy(timeout):
            flushed_on.append(threading.current_thread().name)
            real(timeout)

        monkeypatch.setattr(attach_module._EXPORT, "_flush", spy)
        on_loop(_failed_turn, h)
        assert wait_for(
            lambda: any(n.name == "invoke_agent hermes" for n in received(transport))
        ), "the failed turn reached the transport with no flush from the test"
    assert flushed_on and set(flushed_on) == {"wardex-hermes"}


def test_an_interrupted_turn_closes_unset_aborted_with_the_cut_call_errored():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(lambda: h.interrupt(h.turn()))
    nodes = exported(transport)
    assert outline(nodes) == _tree(f"""
    invoke_agent hermes | trace_root 1.0 | UNSET | session_aborted
      chat lab-model | unit_active 1.0 | ERROR(InterruptedError)
        {WIRE} | contextvar 1.0 | UNSET | connect_timing_unavailable, frame_parse_failed
    """)
    root = one(nodes, "invoke_agent hermes")
    assert extras(root)["close_cause"] == "end_event"
    assert extras(root)["exit_reason"] == "interrupted_during_api_call"
    wire = one(nodes, WIRE)
    assert not [k for k in wire.attributes if k.startswith("gen_ai.usage.")]


def test_a_first_turn_carries_no_link():
    """A turn nothing triggered and nothing resumed links to nothing."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.model_call(turn)
            h.end(turn)

        on_loop(run)
    nodes = exported(transport)
    assert links(transport, one(nodes, "invoke_agent hermes"), nodes) == []
