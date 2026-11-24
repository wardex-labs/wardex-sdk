"""The trees a Hermes turn exports, one test per case, asserted on exported spans only.

A tree is drawn by `outline`: each span's name, where its parent came from and how sure that is,
its status (with `error.type`), and its markers. The wire spans are the socket seam's record of
each HTTP attempt to the scripted provider; the adapter adds the turn, one `chat` per model call
frame (every attempt of a call sits under it), and one `execute_tool` per tool frame.
"""

from __future__ import annotations

import json
import textwrap
import threading
from pathlib import Path

import pytest

from _hermes_fake import Reply
from _hermes_scripted import ANSWER, QUESTION, TASK, FakeHermes
from _hermes_support import (
    CUT_STREAM_MARKERS,
    Tree,
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
    render,
    start_wardex,
    wait_for,
)
from wardex_sdk._adapters import _hermes_agent_attach as attach_module
from wardex_sdk._adapters import _hermes_agent_turns as turns_module
from wardex_sdk._assembly import LinkReason, counters

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


def _quiet_net(monkeypatch) -> None:
    """Hermes sends no end for a turn whose retries were spent, so the quiet net closes it. This
    shortens the net so the test need not wait five minutes."""
    monkeypatch.setattr(turns_module, "IDLE_AFTER_ERROR_S", 0.3)
    monkeypatch.setattr(attach_module, "IDLE_SWEEP_S", 0.05)


def _closed_by_the_net(transport) -> None:
    assert wait_for(lambda: any(n.name == "invoke_agent hermes" for n in received(transport)))


def test_a_failed_turn_closes_unset_aborted_on_the_quiet_net_with_the_error_on_the_call(
    monkeypatch,
):
    _quiet_net(monkeypatch)
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(_failed_turn, h)
        _closed_by_the_net(transport)
    nodes = exported(transport)
    assert outline(nodes) == _tree(f"""
    invoke_agent hermes | trace_root 1.0 | UNSET | session_aborted
      chat lab-model | unit_active 1.0 | ERROR(InternalServerError)
        {WIRE} | contextvar 1.0 | ERROR(500)
        {WIRE} | contextvar 1.0 | ERROR(500)
    """)
    root = one(nodes, "invoke_agent hermes")
    assert extras(root)["close_cause"] == "idle"
    assert extras(root)["error_reason"] == "server_error"
    assert "exit_reason" not in extras(root), "Hermes sent no end, so no exit reason is stated"


def test_a_provider_that_never_recovers_gives_one_turn_holding_every_frame_of_the_ladder(
    monkeypatch,
):
    """Hermes's defaults: three frames per round, five more rounds on its recovery ladder, all
    under one `api_request_id`; the first frame's stream and its fallback, then one attempt a
    frame. No frame's error ends the turn; the quiet net does, once."""
    _quiet_net(monkeypatch)
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(lambda: h.fail(h.turn(), retries=3, cycles=5))
        _closed_by_the_net(transport)
    nodes = exported(transport)
    chat = "chat lab-model | unit_active 1.0 | ERROR(InternalServerError)"
    wire = (f"{WIRE} | contextvar 1.0 | ERROR(500)", ())
    assert outline(nodes) == render(
        (
            "invoke_agent hermes | trace_root 1.0 | UNSET | session_aborted",
            ((chat, (wire, wire)), *[(chat, (wire,))] * 17),
        )
    )
    assert extras(one(nodes, "invoke_agent hermes"))["close_cause"] == "idle"


def test_a_failed_turn_is_exported_by_the_wardex_thread_with_no_flush_on_a_hermes_thread(
    monkeypatch,
):
    """Hermes sends no end for a failed turn, so the quiet net on `wardex-hermes` closes it and
    flushes. No Hermes thread waits on export, and the
    flush names no budget: it follows the transport's, since a number would read as the host's."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        flushed_on: list[tuple[str, float | None]] = []
        real = attach_module._EXPORT._flush
        assert real is not None

        def spy(timeout):
            flushed_on.append((threading.current_thread().name, timeout))
            real(timeout)

        monkeypatch.setattr(attach_module._EXPORT, "_flush", spy)
        _quiet_net(monkeypatch)
        on_loop(_failed_turn, h)
        assert wait_for(
            lambda: any(n.name == "invoke_agent hermes" for n in received(transport))
        ), "the failed turn reached the transport with no flush from the test"
    assert flushed_on and set(flushed_on) == {("wardex-hermes", None)}


def test_an_interrupted_turn_closes_unset_aborted_with_the_cut_call_errored():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(lambda: h.interrupt(h.turn()))
    nodes = exported(transport)
    assert outline(nodes, quiet=CUT_STREAM_MARKERS) == _tree(f"""
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


# -- tools ---------------------------------------------------------------------------------------

STREAMED = f"{WIRE} | contextvar 1.0 | OK | provider_inferred, reassembled_from_stream"
CHAT: Tree = ("chat lab-model | unit_active 1.0 | OK", ((STREAMED, ()),))
TURN = "invoke_agent hermes | trace_root 1.0 | OK"
# A call asking for several tools answers without streaming here: the scripted provider's stream
# gives its tool-call deltas no `index`, which the seam rightly reads as unparsed arguments.
ASKING: Tree = (
    "chat lab-model | unit_active 1.0 | OK",
    ((f"{WIRE} | contextvar 1.0 | OK | provider_inferred", ()),),
)


def _tool(name: str, status: str = "OK", *kids: Tree) -> Tree:
    return (f"execute_tool {name} | unit_active 1.0 | {status}", kids)


def test_parallel_tools_are_siblings_under_the_turn_each_linked_to_the_call():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            asked = h.model_call(turn, (Reply.OK,), tool_calls=["read_file", "search", "read_file"])
            h.parallel_tools(asked)
            h.model_call(turn)
            h.end(turn)

        on_loop(run)
    nodes = exported(transport)
    assert outline(nodes) == render(
        (TURN, (ASKING, _tool("read_file"), _tool("search"), _tool("read_file"), CHAT))
    )
    asked = next(n for n in nodes if extras(n).get("api_request_id", "").endswith(":api:1"))
    for tool in [n for n in nodes if n.name.startswith("execute_tool")]:
        assert links(transport, tool, nodes) == [(LinkReason.TRIGGERED_BY, asked.name)]
        assert _link_targets(transport, tool) == [asked.span_id]


def _link_targets(transport, node) -> list:
    raw = next(s for e in transport.envelopes for s in e.spans if s.context.span_id == node.span_id)
    return [link.span_id for link in raw.links]


def test_two_tool_rounds_link_each_tool_to_the_call_that_asked_for_it():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.tool(h.model_call(turn, tool_calls=["read_file"]), "read_file")
            h.tool(h.model_call(turn, tool_calls=["write_file"]), "write_file")
            h.model_call(turn)
            h.end(turn)

        on_loop(run)
    nodes = exported(transport)
    assert outline(nodes) == render(
        (TURN, (CHAT, _tool("read_file"), CHAT, _tool("write_file"), CHAT))
    )
    calls = {
        extras(n)["api_request_id"].rsplit(":", 1)[1]: n
        for n in nodes
        if n.name == "chat lab-model"
    }
    assert _link_targets(transport, one(nodes, "execute_tool read_file")) == [calls["1"].span_id]
    assert _link_targets(transport, one(nodes, "execute_tool write_file")) == [calls["2"].span_id]


def test_a_tool_that_reports_an_error_without_raising_closes_with_hermes_error_type():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.tool(h.model_call(turn, tool_calls=["read_file"]), "read_file", status="error")
            h.end(turn)

        on_loop(run)
        assert ctx.failures == []
    assert outline(exported(transport)) == render(
        (TURN, (CHAT, _tool("read_file", "ERROR(tool_error)")))
    )


def test_a_tool_call_the_model_asked_for_and_hermes_never_ran_is_named_on_the_turn():
    """An invented tool name: Hermes answers the model itself and runs no tool frame."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.model_call(turn, tool_calls=["lab_no_such_tool"])
            h.model_call(turn)
            h.end(turn)

        on_loop(run)
    nodes = exported(transport)
    assert outline(nodes) == render((f"{TURN} | tool_call_not_dispatched", (CHAT, CHAT)))
    assert (
        extras(one(nodes, "invoke_agent hermes"))["undispatched_tool_calls"] == "lab_no_such_tool"
    )
    assert not [n for n in nodes if n.name.startswith("execute_tool")]


def test_a_tool_frame_no_live_turn_owns_is_its_own_span_closed_at_frame_exit():
    """Hermes's MCP child process runs tool frames with no turn of its own in that process."""
    from _hermes_scripted import Call, Turn

    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        stray = Turn("s-elsewhere", "task", "s-elsewhere:task:0")
        on_loop(
            lambda: h.tool(
                Call(stray, "s-elsewhere:task:0:api:1", (("c-1", "read_file"),), 200), "read_file"
            )
        )
        assert ctx.failures == []
    assert outline(exported(transport)) == (
        "execute_tool read_file | unresolved 0.0 | UNSET | parent_unresolved"
    )


# -- MCP -----------------------------------------------------------------------------------------

_MCP_SERVER = """
import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}

async def list_tools(ctx, params):
    return types.ListToolsResult(tools=[
        types.Tool(name="add", description="a + b", input_schema=SCHEMA),
    ])

async def call_tool(ctx, params):
    value = params.arguments["a"] + params.arguments["b"]
    return types.CallToolResult(content=[types.TextContent(type="text", text=str(value))])

server = Server("arith", on_list_tools=list_tools, on_call_tool=call_tool)

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

anyio.run(main)
"""


@pytest.fixture
def recording():
    return start_wardex()


@pytest.fixture
def mcp_loop(recording, tmp_path: Path):
    """An MCP stdio client on its own event loop thread, started from an empty context before
    any turn: Hermes keeps one such loop for every MCP server, so the stdio write a tool makes
    never runs on the tool's own thread. The server starts after wardex, as it does when
    Hermes loads its plugins before connecting its MCP servers."""
    import asyncio
    import contextvars
    import sys

    from fastmcp import Client
    from fastmcp.client.transports import StdioTransport

    script = tmp_path / "arith_server.py"
    script.write_text(_MCP_SERVER)
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=contextvars.Context().run, args=(loop.run_forever,), name="mcp-loop", daemon=True
    )
    thread.start()
    client = Client(StdioTransport(command=sys.executable, args=[str(script)]))
    asyncio.run_coroutine_threadsafe(client.__aenter__(), loop).result(30)

    def call(name: str, args: dict) -> str:
        done = asyncio.run_coroutine_threadsafe(client.call_tool(name, args), loop).result(30)
        return done.content[0].text

    try:
        yield call
    finally:
        asyncio.run_coroutine_threadsafe(client.__aexit__(None, None, None), loop).result(30)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)


def _mcp_tool(h: FakeHermes, call, tool_call_id: str, args: dict, mcp, barrier=None) -> str:
    """One MCP tool frame the way Hermes runs one: the frame's terminal hands the call to the
    MCP loop and waits, and `post_tool_call` lands after the frame."""
    from _hermes_fake import OBSERVER_V1

    turn = call.turn
    ids = {
        "telemetry_schema_version": OBSERVER_V1,
        "session_id": turn.session_id,
        "task_id": turn.task_id,
        "turn_id": turn.turn_id,
        "api_request_id": call.api_request_id,
        "tool_call_id": tool_call_id,
        "tool_name": "mcp__arith__add",
        "conversation_id": turn.session_id,
    }

    def terminal(final_args):
        h.ctx.invoke_hook("pre_tool_call", **ids, args=final_args)
        if barrier is not None:
            barrier.wait()
        return mcp("add", args)

    result = h.ctx.run_execution_chain(
        "tool_execution", terminal, args=args, original_args=args, **ids
    )
    h.ctx.invoke_hook(
        "post_tool_call",
        **ids,
        args=args,
        result=result,
        duration_ms=1,
        status="ok",
        error_type=None,
        error_message=None,
    )
    return result


def _mcp_round(h: FakeHermes, mcp, argsets: list[dict]) -> list[str]:
    import concurrent.futures
    import contextvars

    turn = h.turn()
    call = h.model_call(turn, (Reply.OK,), tool_calls=["mcp__arith__add"] * len(argsets))
    barrier = threading.Barrier(len(argsets), timeout=10) if len(argsets) > 1 else None
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(argsets)) as pool:
        jobs = [
            pool.submit(
                contextvars.copy_context().run, _mcp_tool, h, call, tcid, args, mcp, barrier
            )
            for (tcid, _), args in zip(call.tool_calls, argsets, strict=True)
        ]
        results = [job.result() for job in jobs]
    h.model_call(turn)
    h.end(turn)
    return results


MCP_TOOL = "execute_tool mcp__arith__add | unit_active 1.0 | OK"
MCP_CALL = ("MCP tools/call | contextvar 1.0 | OK", ())


def _in_turns(nodes):
    """The spans of the traces that hold a turn: the client's own session setup (`initialize`,
    `tools/list`) ran before any turn and is a trace of its own."""
    traces = {n.trace_id for n in nodes if n.name == "invoke_agent hermes"}
    return tuple(n for n in nodes if n.trace_id in traces)


def test_an_mcp_tool_is_one_execute_tool_span_with_the_stdio_call_under_it(recording, mcp_loop):
    transport = recording
    with plugin() as ctx, hermes(ctx) as h:
        assert on_loop(_mcp_round, h, mcp_loop, [{"a": 2, "b": 3}]) == ["5"]
    nodes = _in_turns(exported(transport))
    assert outline(nodes) == render((TURN, (ASKING, (MCP_TOOL, (MCP_CALL,)), CHAT)))


def test_parallel_mcp_calls_to_one_tool_each_land_under_their_own_frame(recording, mcp_loop):
    transport = recording
    with plugin() as ctx, hermes(ctx) as h:
        argsets = [{"a": 2, "b": 3}, {"a": 40, "b": 2}]
        assert on_loop(_mcp_round, h, mcp_loop, argsets) == ["5", "42"]
    nodes = _in_turns(exported(transport))
    assert outline(nodes) == render(
        (TURN, (ASKING, (MCP_TOOL, (MCP_CALL,)), (MCP_TOOL, (MCP_CALL,)), CHAT))
    )
    for wire in [n for n in nodes if n.name == "MCP tools/call"]:
        frame = next(n for n in nodes if n.span_id == wire.parent_id)
        sent = json.loads(wire.attributes["wardex.input_data"])
        assert frame.attributes["gen_ai.tool.call.arguments"] == repr(sent), "its own frame"


# -- the turn's input and output ---------------------------------------------------------------

TURN_SPAN = "invoke_agent hermes"
EXPLANATION = "No reply: the turn stopped while a tool result was still pending."


def _io(node) -> tuple[object, object]:
    """What a backend's trace list shows for the span: its input and its output."""
    return node.attributes.get("wardex.input_data"), node.attributes.get("wardex.output_data")


def _turns(script, *, closed_by_the_net: bool = False) -> list:
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(script, h)
        if closed_by_the_net:
            _closed_by_the_net(transport)
    return [n for n in exported(transport) if n.name == TURN_SPAN]


def _answered(h: FakeHermes) -> None:
    turn = h.turn()
    h.model_call(turn)
    h.end(turn)


def test_a_completed_turn_carries_the_users_message_and_hermes_answer():
    (turn,) = _turns(_answered)
    assert _io(turn) == (QUESTION, ANSWER)


def test_a_turn_whose_retries_were_spent_carries_its_input_and_no_output(monkeypatch):
    def script(h: FakeHermes) -> None:
        h.fail(h.turn())

    _quiet_net(monkeypatch)
    (turn,) = _turns(script, closed_by_the_net=True)
    assert extras(turn)["close_cause"] == "idle"
    assert _io(turn) == (QUESTION, None)


def test_an_interrupted_turn_carries_its_input_and_no_output():
    def script(h: FakeHermes) -> None:
        h.interrupt(h.turn())

    (turn,) = _turns(script)
    assert _io(turn) == (QUESTION, None)


def test_a_failed_turn_hermes_finalized_carries_the_explanation_hermes_showed():
    def script(h: FakeHermes) -> None:
        turn = h.turn()
        h.model_call(turn)
        h.end(turn, completed=False, failed=True, reason="pending_tool_result", answer=EXPLANATION)

    (turn,) = _turns(script)
    assert turn.status.name == "ERROR"
    assert _io(turn) == (QUESTION, EXPLANATION)


def test_each_child_turn_carries_its_own_task_and_answer():
    def script(h: FakeHermes) -> None:
        turn = h.turn()
        h.delegate(h.model_call(turn, tool_calls=["delegate_task"]), 2)
        h.model_call(turn)
        h.end(turn)

    turns = _turns(script)
    children = sorted(_io(n) for n in turns if n.attributes["wardex.agent.type"] == "sub_agent")
    assert children == [(f"{TASK} 0", ANSWER), (f"{TASK} 1", ANSWER)]
    (parent,) = [n for n in turns if n.attributes["wardex.agent.type"] == "primary"]
    assert _io(parent) == (QUESTION, ANSWER)


def test_a_message_with_an_attachment_shows_its_text_and_names_the_attachment():
    """A message with an image arrives as content parts. The turn's input is its text, and the
    image is named by its type: the image's bytes would only fill the record up to its cap."""
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 4096}}
    parts = [{"type": "text", "text": "Describe this."}, image]

    def script(h: FakeHermes) -> None:
        turn = h.turn(message=parts)
        h.model_call(turn)
        h.end(turn)

    (turn,) = _turns(script)
    assert _io(turn) == ("Describe this.\n[image_url]", ANSWER)


def test_a_message_that_is_a_list_but_not_content_parts_is_recorded_as_it_came():
    odd = ["not", "content", "parts"]

    def script(h: FakeHermes) -> None:
        turn = h.turn(message=odd)
        h.model_call(turn)
        h.end(turn)

    (turn,) = _turns(script)
    assert _io(turn) == (repr(odd), ANSWER)


def test_with_no_record_budget_the_turn_carries_neither_input_nor_output():
    transport = start_wardex()
    adapter = attach_module._ROUTE
    assert adapter is not None and adapter._ctx is not None
    adapter._ctx._units._max_record_bytes = 0
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(_answered, h)
    (turn,) = [n for n in exported(transport) if n.name == TURN_SPAN]
    assert _io(turn) == (None, None)


def test_an_answer_for_a_closed_turn_is_counted_and_changes_nothing():
    counters.reset()
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def script() -> None:
            turn = h.turn()
            h.model_call(turn)
            h.end(turn)
            h._emit(
                "post_llm_call",
                {"session_id": turn.session_id, "turn_id": turn.turn_id, "assistant_response": "x"},
            )

        on_loop(script)
    (turn,) = [n for n in exported(transport) if n.name == TURN_SPAN]
    assert _io(turn) == (QUESTION, ANSWER)
    assert counters.get("adapters.hermes_agent.ignore_late.turn_answer") == 1


def test_an_answer_for_a_turn_the_quiet_net_closed_is_counted_and_changes_nothing(monkeypatch):
    _quiet_net(monkeypatch)
    counters.reset()
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def script() -> None:
            turn = h.turn()
            h.fail(turn)
            _closed_by_the_net(transport)
            h._emit(
                "post_llm_call",
                {"session_id": turn.session_id, "turn_id": turn.turn_id, "assistant_response": "x"},
            )

        on_loop(script)
    (turn,) = [n for n in exported(transport) if n.name == TURN_SPAN]
    assert _io(turn) == (QUESTION, None)
    assert counters.get("adapters.hermes_agent.ignore_late.turn_answer") == 1
