"""Token usage is counted once per model call, wherever Hermes's own numbers also arrive.

Hermes reports each call's usage in `post_api_request`, and the socket seam reads the same call's
usage off the wire. A backend sums the typed `gen_ai.usage.*` attributes, so they sit on one span
per call: the wire span when there were bytes, else the call's own span, marked
`no_wire_evidence`. Hermes's numbers are kept beside them as `wardex.hermes_agent.reported.*`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from _hermes_fake import Reply, Usage
from _hermes_scripted import FakeHermes, _usage
from _hermes_support import (
    exported,
    extras,
    forget_attachments,
    hermes,
    hermes_installed,
    on_loop,
    one,
    plugin,
    start_wardex,
    typed_usage,
)
from wardex_sdk._assembly import Limitation

WIRE = "HTTP POST /v1/chat/completions"
REPORTED = "reported.gen_ai.usage."


@pytest.fixture(autouse=True)
def _installed(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield
    forget_attachments()


def _reported(node) -> dict:
    return {k[len(REPORTED) :]: v for k, v in extras(node).items() if k.startswith(REPORTED)}


def _summed(nodes, key: str) -> int:
    return sum(typed_usage(n).get(f"gen_ai.usage.{key}", 0) for n in nodes)


def test_with_wire_bytes_usage_is_typed_on_the_wire_span_alone_and_hermes_numbers_are_extras():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.model_call(turn, usage=Usage(1200, 34))
            h.end(turn)

        on_loop(run)
    nodes = exported(transport)
    chat, wire = one(nodes, "chat lab-model"), one(nodes, WIRE)
    assert typed_usage(wire)["gen_ai.usage.input_tokens"] == 1200
    assert typed_usage(wire)["gen_ai.usage.output_tokens"] == 34
    assert typed_usage(chat) == {}
    assert _reported(chat)["input_tokens"] == 1200
    assert _reported(chat)["output_tokens"] == 34
    assert (_summed(nodes, "input_tokens"), _summed(nodes, "output_tokens")) == (1200, 34)
    assert not [n.name for n in nodes if Limitation.NO_WIRE_EVIDENCE in n.limitations]


def test_a_retried_call_sums_to_the_one_answer_that_carried_usage():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.model_call(
                turn, (Reply.UNAVAILABLE, Reply.SERVER_ERROR, Reply.OK), usage=Usage(900, 9)
            )
            h.end(turn)

        on_loop(run)
    nodes = exported(transport)
    assert (_summed(nodes, "input_tokens"), _summed(nodes, "output_tokens")) == (900, 9)


def _call_with_no_bytes(h: FakeHermes, used: Usage) -> None:
    """A model call whose transport the seam cannot read: the frame runs and returns, and
    Hermes reports usage, with no HTTP request in between."""
    turn = h.turn()
    turn.calls += 1
    ids = h._call_ids(turn, f"{turn.turn_id}:api:{turn.calls}")
    h.ctx.run_execution_chain(
        "llm_execution",
        lambda request: SimpleNamespace(tool_calls=()),
        request={},
        original_request={},
        **ids,
    )
    h._emit(
        "post_api_request",
        {
            **ids,
            "response_model": h.model,
            "finish_reason": "stop",
            "assistant_tool_call_count": 0,
            "usage": _usage(used),
        },
    )
    h.end(turn)


def test_with_no_wire_bytes_hermes_numbers_are_promoted_once_and_marked():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(_call_with_no_bytes, h, Usage(500, 7))
    nodes = exported(transport)
    chat = one(nodes, "chat lab-model")
    assert not [n for n in nodes if n.name == WIRE]
    assert typed_usage(chat)["gen_ai.usage.input_tokens"] == 500
    assert typed_usage(chat)["gen_ai.usage.output_tokens"] == 7
    assert Limitation.NO_WIRE_EVIDENCE in chat.limitations
    assert _reported(chat) == {}, "promoted, so not also kept as extras"
    assert (_summed(nodes, "input_tokens"), _summed(nodes, "output_tokens")) == (500, 7)


def test_no_call_carries_both_wire_usage_and_promoted_usage():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.tool(h.model_call(turn, tool_calls=["read_file"], usage=Usage(1100, 11)), "read_file")
            h.model_call(turn, usage=Usage(1300, 13))
            h.end(turn)

        on_loop(run)
        on_loop(_call_with_no_bytes, h, Usage(400, 4))
    nodes = exported(transport)
    for chat in [n for n in nodes if n.name == "chat lab-model"]:
        under = [n for n in nodes if n.parent_id == chat.span_id and n.name == WIRE]
        promoted = Limitation.NO_WIRE_EVIDENCE in chat.limitations
        assert promoted != bool(under), chat
        assert bool(typed_usage(chat)) == promoted
    assert _summed(nodes, "input_tokens") == 1100 + 1300 + 400


def test_a_failed_call_carries_no_usage():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(lambda: h.fail(h.turn()))
    nodes = exported(transport)
    assert _summed(nodes, "input_tokens") == 0
    assert _reported(one(nodes, "chat lab-model")) == {}


def test_the_auxiliary_hook_span_keeps_hermes_numbers_as_extras_and_never_types_them():
    """The title call's wire span is a root of its own (Hermes built its thread before the turn
    ran anything), so its usage is counted there; the hook span under the turn only states
    Hermes's numbers."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn(title=True)
            h.aux(turn)
            h.model_call(turn, usage=Usage(1000, 10))
            h.end(turn)

        on_loop(run)
    nodes = exported(transport)
    root = one(nodes, "invoke_agent hermes")
    hook = one(nodes, "chat lab-aux-title")
    assert hook.parent_id == root.span_id
    assert typed_usage(hook) == {}
    assert _reported(hook)["output_tokens"] == 17
    assert Limitation.NO_WIRE_EVIDENCE not in hook.limitations
    title_wire = next(n for n in nodes if n.name == WIRE and n.trace_id != root.trace_id)
    assert typed_usage(title_wire)["gen_ai.usage.output_tokens"] == 17
    in_turn = [n for n in nodes if n.trace_id == root.trace_id]
    assert (_summed(in_turn, "input_tokens"), _summed(in_turn, "output_tokens")) == (1000, 10)


def test_a_failed_auxiliary_call_has_no_usage_anywhere():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn(title=True)
            h.aux(turn, answer=Reply.BAD_REQUEST)
            h.end(turn)

        on_loop(run)
    nodes = exported(transport)
    hook = one(nodes, "chat lab-aux-title")
    assert typed_usage(hook) == {} and _reported(hook) == {}


def _child(h: FakeHermes, kid) -> None:
    h.tool(
        h.model_call(kid, (Reply.OK,), tool_calls=["read_file"], usage=Usage(300, 3)), "read_file"
    )
    h.model_call(kid, (Reply.OK,), usage=Usage(400, 4))
    h.end(kid)


@pytest.mark.parametrize("in_coroutine", [False, True], ids=["sync", "in_coroutine"])
def test_a_turn_run_from_inside_a_coroutine_is_one_tree_with_usage_counted_once(in_coroutine):
    """An embedder calls Hermes's synchronous `run_conversation` from an `async def` under
    `asyncio.run`, so a task is running on the loop thread while every frame of the turn is
    pinned there. Tools and the joined child run on pool threads, outside any loop. The tree
    and the usage sum are the same as for a call with no loop running."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def turn() -> None:
            t = h.turn()
            asked = ["read_file", "delegate_task"]
            call = h.model_call(t, (Reply.OK,), tool_calls=asked, usage=Usage(100, 1))
            h.tool(call, "read_file")
            h.delegate(call, children=1, child=_child)
            h.model_call(t, (Reply.OK,), usage=Usage(200, 2))
            h.end(t)

        async def embedder() -> None:
            turn()

        on_loop(lambda: asyncio.run(embedder()) if in_coroutine else turn())
    nodes = exported(transport)
    assert len({n.trace_id for n in nodes}) == 1, [(n.name, n.parent_id) for n in nodes]
    turns = [n for n in nodes if n.name == "invoke_agent hermes"]
    root = next(n for n in turns if n.parent_id is None)
    assert len(turns) == 2
    assert not [n.name for n in nodes if Limitation.CORRELATION_CONFLICT in n.limitations]
    assert not [n.name for n in nodes if Limitation.PARENT_UNRESOLVED in n.limitations]
    chats = [n for n in nodes if n.name == "chat lab-model"]
    assert len(chats) == 4
    assert sum(1 for n in chats if n.parent_id == root.span_id) == 2
    assert _summed(nodes, "input_tokens") == 100 + 200 + 300 + 400
    assert not [n.name for n in nodes if Limitation.NO_WIRE_EVIDENCE in n.limitations]
