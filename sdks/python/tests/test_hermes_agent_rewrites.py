"""Tool calls Hermes rewrites between the model call that asked for them and their dispatch.

`post_api_request` states the calls as the provider sent them. Hermes then renames a reused id,
replaces a provider's ids in a parallel batch, repairs a misspelled name, maps a legacy alias,
and splits one `tool_call` bridge call into one call per tool, each run under its tool's own
name. None of that is a call that never ran, so a turn says `tool_call_not_dispatched` only for
an asked call that no frame under its model call explains, and a bridge call explains any
number of frames. Each case is driven on the fake Hermes with the rewrite Hermes v0.23.3 makes
(`_hermes_payloads.py`).

A model request another plugin's `llm_request` middleware replaced is a rewrite of another kind:
the wire carries the replacement, and wardex keeps the user's message and the tool results as
they were. Hermes states each replacement in `middleware_trace`, and the call and its turn carry
`wardex.hermes_agent.request_rewritten`, so a reader knows the two differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import wardex_sdk
from _hermes_fake import FakePluginContext, Reply
from _hermes_payloads import (
    BRIDGE,
    bridge_split,
    connector_batch,
    provider_ids_normalized,
    reused_ids_renamed,
    runs_as,
)
from _hermes_scripted import QUESTION, FakeHermes, Turn
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
)
from wardex_sdk._assembly import Limitation

TURN = "invoke_agent hermes"
CHAT = "chat lab-model"
REWRITTEN = "request_rewritten"
SCRUBBED = "<scrubbed>"


@pytest.fixture(autouse=True)
def _installed(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield
    forget_attachments()


def _turn(script) -> tuple[list[Limitation], dict]:
    """Run `script(h, turn)` between a turn's start and its final answer; the turn's markers and
    attributes."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run() -> None:
            turn = h.turn()
            script(h, turn)
            h.model_call(turn)
            h.end(turn)

        on_loop(run)
    span = one(exported(transport), TURN)
    return list(span.limitations), extras(span)


def _ran_all(markers: list[Limitation], attrs: dict) -> None:
    assert Limitation.TOOL_CALL_NOT_DISPATCHED not in markers
    assert "undispatched_tool_calls" not in attrs
    assert attrs["close_cause"] == "end_event"


def test_a_reused_id_hermes_renamed_before_running_it_is_not_a_call_that_never_ran():
    def script(h: FakeHermes, turn: Turn) -> None:
        h.tool(reused_ids_renamed(h.model_call(turn, tool_calls=["read_file"])), "read_file")

    _ran_all(*_turn(script))


def test_a_parallel_batch_whose_provider_ids_hermes_replaced_ran_in_full():
    def script(h: FakeHermes, turn: Turn) -> None:
        asked = h.model_call(turn, (Reply.OK,), tool_calls=["read_file"] * 3)
        h.parallel_tools(provider_ids_normalized(asked))

    _ran_all(*_turn(script))


@pytest.mark.parametrize(
    ("asked", "runs"), [("Read_File", "read_file"), ("todo", "todo_list"), (BRIDGE, BRIDGE)]
)
def test_a_call_hermes_ran_under_another_tool_name_ran(asked: str, runs: str):
    """A repaired name, a legacy alias, and a bridge call Hermes could not unwrap, which runs
    under the bridge's own name and answers with an error result."""

    def script(h: FakeHermes, turn: Turn) -> None:
        h.tool(runs_as(h.model_call(turn, tool_calls=[asked]), asked, runs), runs)

    _ran_all(*_turn(script))


def _ran(h: FakeHermes, call, skip: str = "lab_no_such_tool") -> None:
    """Run every call of `call` as Hermes dispatched it, except the invented one."""
    for tcid, name in call.tool_calls:
        if name != skip:
            h.tool(call, name, tool_call_id=tcid)


def test_one_bridge_call_hermes_split_in_two_ran_and_hides_no_other_calls_shortfall():
    """The extra frame counts for the call that asked, never for a later one: the next call's
    tool that never ran is still named."""

    def script(h: FakeHermes, turn: Turn) -> None:
        _ran(h, bridge_split(h.model_call(turn, tool_calls=[BRIDGE]), ["todo_list"] * 2))
        h.model_call(turn, tool_calls=["lab_no_such_tool"])

    markers, attrs = _turn(script)
    assert Limitation.TOOL_CALL_NOT_DISPATCHED in markers
    assert attrs["undispatched_tool_calls"] == "lab_no_such_tool"


def test_a_renamed_batch_with_one_call_hermes_dropped_names_only_that_call():
    """A mixed batch: Hermes answers an invented name with an error result and runs the rest,
    under ids it rewrote."""

    def script(h: FakeHermes, turn: Turn) -> None:
        asked = h.model_call(turn, tool_calls=["read_file", "lab_no_such_tool", "search"])
        renamed = reused_ids_renamed(asked)
        h.tool(renamed, "read_file")
        h.tool(renamed, "search")

    markers, attrs = _turn(script)
    assert Limitation.TOOL_CALL_NOT_DISPATCHED in markers
    assert attrs["undispatched_tool_calls"] == "lab_no_such_tool"


@pytest.mark.parametrize("carried", [["todo_list", "todo_list"], ["todo_list"]])
def test_an_invented_tool_beside_a_bridge_call_is_named_however_many_tools_the_bridge_ran(
    carried: list[str],
):
    """One model call asks for a bridge call and a tool that does not exist. Hermes runs the
    bridge's tools under their own names and answers the invented one with an error result:
    the frames that ran are the bridge's, so the invented call is the one named."""

    def script(h: FakeHermes, turn: Turn) -> None:
        asked = h.model_call(turn, tool_calls=[BRIDGE, "lab_no_such_tool"])
        _ran(h, bridge_split(asked, carried))

    markers, attrs = _turn(script)
    assert Limitation.TOOL_CALL_NOT_DISPATCHED in markers
    assert attrs["undispatched_tool_calls"] == "lab_no_such_tool"


def test_an_invented_tool_beside_a_connector_batch_is_named():
    """A bridge call of two connector tools runs as three frames (the wrapper under the
    bridge's name, an entry under each connector's name inside it), all under the one model
    call. The invented call beside it is still named."""

    def script(h: FakeHermes, turn: Turn) -> None:
        asked = h.model_call(turn, tool_calls=[BRIDGE, "lab_no_such_tool"])
        connector_batch(h, asked, ["connectors__lab__get", "connectors__lab__put"])

    markers, attrs = _turn(script)
    assert Limitation.TOOL_CALL_NOT_DISPATCHED in markers
    assert attrs["undispatched_tool_calls"] == "lab_no_such_tool"


def test_a_bridge_call_hermes_ran_in_full_beside_a_tool_that_ran_is_not_marked():
    def script(h: FakeHermes, turn: Turn) -> None:
        asked = h.model_call(turn, tool_calls=["read_file", BRIDGE])
        _ran(h, bridge_split(asked, ["todo_list", "todo_list"]))

    _ran_all(*_turn(script))


def test_a_repaired_name_beside_an_invented_tool_leaves_only_the_invented_one_named():
    """Hermes repairs `Read_File` to `read_file` (case and separators, as its repair compares
    them) and answers the invented name with an error result."""

    def script(h: FakeHermes, turn: Turn) -> None:
        asked = h.model_call(turn, tool_calls=["Read_File", "lab_no_such_tool"])
        h.tool(runs_as(asked, "Read_File", "read_file"), "read_file")

    markers, attrs = _turn(script)
    assert Limitation.TOOL_CALL_NOT_DISPATCHED in markers
    assert attrs["undispatched_tool_calls"] == "lab_no_such_tool"


# -- a model request another plugin rewrote ------------------------------------------------------


def _scrub(**payload: Any) -> dict[str, Any]:
    """A customer-name scrubber: replaces the request Hermes is about to send."""
    request = dict(payload["request"])
    request["messages"] = [{"role": "user", "content": SCRUBBED}]
    return {"request": request, "name": "scrubber"}


def _observe(**payload: Any) -> None:
    """Reads the request and hands nothing back, so Hermes sends it unchanged."""


def _beside(ctx: FakePluginContext, kind: str, callback: Any) -> FakePluginContext:
    """A second plugin on the same Hermes, its middleware in the same table as wardex's."""
    other = FakePluginContext(plugin_id="other", hooks=ctx.hooks, middleware=ctx.middleware)
    other.register_middleware(kind, callback)
    return other


def _run(*callbacks: Any, calls: int = 1) -> tuple:
    """One turn of `calls` model calls with each callback registered as another plugin's
    `llm_request` middleware; every exported span."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        others = [_beside(ctx, "llm_request", cb) for cb in callbacks]

        def run() -> None:
            turn = h.turn()
            for _ in range(calls):
                h.model_call(turn)
            h.end(turn)

        on_loop(run)
        assert [f for c in (ctx, *others) for f in c.failures] == []
    return exported(transport)


def test_a_request_another_plugin_rewrote_is_flagged_on_its_call_and_its_turn():
    """The turn's input stays the message the user sent, and the call and the turn both say the
    provider was sent something else."""
    nodes = _run(_scrub)
    turn = one(nodes, TURN)
    assert extras(one(nodes, CHAT))[REWRITTEN] is True
    assert extras(turn)[REWRITTEN] is True
    assert turn.attributes.get("wardex.input_data") == QUESTION


@pytest.mark.parametrize("callbacks", [(_observe,), ()], ids=["observes-only", "none"])
def test_a_request_no_plugin_replaced_carries_no_flag(callbacks: tuple):
    nodes = _run(*callbacks)
    assert [n.name for n in nodes if REWRITTEN in extras(n)] == []
    assert one(nodes, TURN).attributes.get("wardex.input_data") == QUESTION


def test_only_the_call_whose_request_was_rewritten_is_flagged_and_its_turn_says_so():
    def second_only(**payload: Any) -> dict[str, Any] | None:
        return _scrub(**payload) if payload["api_request_id"].endswith(":api:2") else None

    nodes = _run(second_only, calls=2)
    flagged = {
        extras(n)["api_request_id"][-1]: REWRITTEN in extras(n) for n in nodes if n.name == CHAT
    }
    assert flagged == {"1": False, "2": True}
    assert extras(one(nodes, TURN))[REWRITTEN] is True


@pytest.mark.parametrize("trace", ["scrubber", None, {"source": "plugin"}, []], ids=repr)
def test_a_trace_that_is_not_a_list_with_entries_sets_nothing(trace: object):
    """Only a non-empty list is Hermes saying a plugin replaced the request; nothing else is
    read as yes or as no, and the frame still hands Hermes its own result."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run() -> object:
            turn = h.turn()
            ids = h._call_ids(turn, f"{turn.turn_id}:api:1")
            out = ctx.run_execution_chain(
                "llm_execution", lambda p: "answer", request={}, middleware_trace=trace, **ids
            )
            h.end(turn)
            return out

        assert on_loop(run) == "answer"
        assert ctx.failures == []
    nodes = exported(transport)
    assert CHAT in {n.name for n in nodes}
    assert [n.name for n in nodes if REWRITTEN in extras(n)] == []


def test_a_rewritten_call_no_live_turn_owns_is_flagged_on_its_own_span():
    """A side question is a fork Hermes never announces as a turn, and its request passes the
    same middleware. Its call says it was rewritten; the session's finished turn does not."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        turn = on_loop(h.turn)
        on_loop(h.model_call, turn)
        on_loop(h.end, turn)
        other = _beside(ctx, "llm_request", _scrub)
        h.side_question(turn.session_id)
        assert ctx.failures == other.failures == []
    nodes = exported(transport)
    finished = one(nodes, TURN)
    chats = {
        n.parent_id == finished.span_id: REWRITTEN in extras(n) for n in nodes if n.name == CHAT
    }
    assert chats == {True: False, False: True}
    assert REWRITTEN not in extras(finished)


def test_a_turn_still_open_at_uninstall_says_its_request_was_rewritten():
    """The core closes that turn, not the adapter, and the turn still says the provider was
    sent something other than its input."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        _beside(ctx, "llm_request", _scrub)
        on_loop(lambda: h.model_call(h.turn()))
        wardex_sdk.close()
    turn = one(exported(transport), TURN)
    assert Limitation.ADAPTER_UNINSTALLED in turn.limitations
    assert extras(turn)[REWRITTEN] is True
