"""Every recorded Hermes fixture, replayed with Hermes's threads and contexts, exports the tree
the adapter was designed to give it.

The fixtures hold what Hermes delivered to a plugin in the lab: hook payloads, middleware frames
and HTTP attempts, each on the thread that issued it, from the context that thread copied.
`Replay` plays them through the plugin context wardex registered on. The MCP fixtures carry no
MCP wire traffic (their tools answered in Hermes's process), so their transport spans are pinned
in `test_hermes_agent_trees.py` against a real MCP server instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _hermes_fake import ScriptedProvider
from _hermes_replay import SCENARIOS, Replay
from _hermes_support import (
    CUT_STREAM_MARKERS,
    Tree,
    exported,
    extras,
    forget_attachments,
    hermes_installed,
    links,
    on_loop,
    outline,
    plugin,
    render,
    start_wardex,
)
from wardex_sdk._assembly import LinkReason

WIRE = "HTTP POST /v1/chat/completions"
STREAMED = f"{WIRE} | contextvar 1.0 | OK | provider_inferred, reassembled_from_stream"
PLAIN = f"{WIRE} | contextvar 1.0 | OK | provider_inferred"
TURN = "invoke_agent hermes | trace_root 1.0 | OK"


def _chat(*wires: str) -> Tree:
    return ("chat lab-model | unit_active 1.0 | OK", tuple((w, ()) for w in wires or (STREAMED,)))


def _tool(name: str, *kids: Tree) -> Tree:
    return (f"execute_tool {name} | unit_active 1.0 | OK", kids)


def _turn(*kids: Tree, line: str = TURN) -> Tree:
    return (line, kids)


_CHILD = (_chat(), _tool("read_file"), _chat())

EXPECTED: dict[str, tuple[Tree, ...]] = {
    "single_turn": (_turn(_chat()),),
    "one_tool": (_turn(_chat(), _tool("read_file"), _chat()),),
    "one_tool-cli": (_turn(_chat(), _tool("read_file"), _chat()),),
    "parallel_tools": (_turn(_chat(), *[_tool("read_file")] * 3, _chat()),),
    "inline_seq": (_turn(_chat(), _tool("memory"), _chat(), _tool("todo_list"), _chat()),),
    "mixed_batch": (
        _turn(
            _chat(),
            _tool("memory"),
            _tool("todo_list"),
            _tool("read_file"),
            _tool("mcp__labecho__echo"),
            _chat(),
        ),
    ),
    "mcp_seq": (_turn(_chat(), _tool("mcp__labecho__echo"), _chat()),),
    "parallel_mcp": (
        _turn(
            _chat(), _tool("mcp__labecho__echo"), _tool("read_file"), _tool("read_file"), _chat()
        ),
    ),
    "retry": (_turn(_chat(f"{WIRE} | contextvar 1.0 | ERROR(503)", PLAIN)),),
    "two_subagents": (
        _turn(_chat(), _tool("delegate_task"), _chat()),
        _turn(*_CHILD, line=f"{TURN} | correlation_conflict"),
        _turn(*_CHILD, line=f"{TURN} | correlation_conflict"),
    ),
    "two_subagents_joined": (
        _turn(
            _chat(),
            _tool(
                "delegate_task",
                _turn(*_CHILD, line="invoke_agent hermes | unit_active 1.0 | OK"),
                _turn(*_CHILD, line="invoke_agent hermes | unit_active 1.0 | OK"),
            ),
            _chat(),
        ),
    ),
    "auxiliary_title": (
        (f"{WIRE} | unresolved 0.0 | OK | parent_unresolved, provider_inferred", ()),
        _turn(("chat lab-aux-title | unit_active 1.0 | OK | possible_duplicate_chat", ()), _chat()),
    ),
    "interrupted": (
        _turn(
            (
                "chat lab-model | unit_active 1.0 | ERROR(InterruptedError)",
                (
                    (
                        f"{WIRE} | contextvar 1.0 | UNSET"
                        " | connect_timing_unavailable, frame_parse_failed",
                        (),
                    ),
                ),
            ),
            line="invoke_agent hermes | trace_root 1.0 | UNSET | session_aborted",
        ),
    ),
    "failed_turn": (
        _turn(
            (
                "chat lab-model | unit_active 1.0 | ERROR(InternalServerError)",
                ((f"{WIRE} | contextvar 1.0 | ERROR(500)", ()),) * 2,
            ),
            line="invoke_agent hermes | trace_root 1.0 | UNSET | session_aborted",
        ),
    ),
}


@pytest.fixture(autouse=True)
def _installed(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield
    forget_attachments()


def test_every_recorded_fixture_has_an_expected_tree():
    assert sorted(EXPECTED) == sorted(SCENARIOS)


@pytest.mark.parametrize("name", SCENARIOS)
def test_a_recorded_fixture_exports_its_designed_tree_and_hermes_is_never_disturbed(name):
    transport = start_wardex()
    with plugin() as ctx, ScriptedProvider() as provider:
        fixture = on_loop(lambda: Replay(ctx, provider).run(name))
        assert ctx.failures == []
        assert {r.value for r in ctx.returns} <= {None}, "a hook's result is Hermes's to use"
    nodes = exported(transport)
    quiet = CUT_STREAM_MARKERS if name == "interrupted" else frozenset()
    assert outline(nodes, quiet) == render(*EXPECTED[name])
    for tool in [n for n in nodes if n.name.startswith("execute_tool ")]:
        (reason, target), *rest = links(transport, tool, nodes)
        assert (reason, target, rest) == (LinkReason.TRIGGERED_BY, "chat lab-model", [])
    if fixture.header["mode"] == "cli":
        roots = [n for n in nodes if n.name == "invoke_agent hermes"]
        assert {extras(n).get("platform") for n in roots} == {"cli"}


def test_joined_children_are_agent_units_under_the_delegate_call_linked_to_the_parent_turn():
    transport = start_wardex()
    with plugin() as ctx, ScriptedProvider() as provider:
        on_loop(lambda: Replay(ctx, provider).run("two_subagents_joined"))
    nodes = exported(transport)
    turns = [n for n in nodes if n.name == "invoke_agent hermes"]
    parent = next(n for n in turns if n.strategy.name == "TRACE_ROOT")
    children = [n for n in turns if n is not parent]
    assert len(children) == 2
    for child in children:
        assert child.attributes["wardex.agent.type"] == "sub_agent"
        assert extras(child)["child_role"] == "leaf"
        assert extras(child)["child_subagent_id"].startswith("sa-")
        assert links(transport, child, nodes) == [(LinkReason.TRIGGERED_BY, parent.name)]
        assert child.trace_id == parent.trace_id


def test_background_children_are_traces_of_their_own_linked_to_the_turn_and_the_call():
    transport = start_wardex()
    with plugin() as ctx, ScriptedProvider() as provider:
        on_loop(lambda: Replay(ctx, provider).run("two_subagents"))
    nodes = exported(transport)
    turns = [n for n in nodes if n.name == "invoke_agent hermes"]
    parent = next(n for n in turns if not n.limitations)
    children = [n for n in turns if n is not parent]
    assert len({parent.trace_id, *(c.trace_id for c in children)}) == 3
    for child in children:
        assert links(transport, child, nodes) == [
            (LinkReason.TRIGGERED_BY, "invoke_agent hermes"),
            (LinkReason.TRIGGERED_BY, "execute_tool delegate_task"),
        ]
