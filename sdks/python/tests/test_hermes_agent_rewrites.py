"""Tool calls Hermes rewrites between the model call that asked for them and their dispatch.

`post_api_request` states the calls as the provider sent them. Hermes then renames a reused id,
replaces a provider's ids in a parallel batch, repairs a misspelled name, maps a legacy alias,
and splits one `tool_call` bridge call into one call per tool, each run under its tool's own
name. None of that is a call that never ran, so a turn says `tool_call_not_dispatched` only for
an asked call that no frame under its model call explains, and a bridge call explains any
number of frames. Each case is driven on the fake Hermes with the rewrite Hermes v0.23.3 makes
(`_hermes_payloads.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _hermes_fake import Reply
from _hermes_payloads import (
    BRIDGE,
    bridge_split,
    connector_batch,
    provider_ids_normalized,
    reused_ids_renamed,
    runs_as,
)
from _hermes_scripted import FakeHermes, Turn
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
