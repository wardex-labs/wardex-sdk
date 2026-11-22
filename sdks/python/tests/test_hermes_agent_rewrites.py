"""Tool calls Hermes rewrites between the model call that asked for them and their dispatch.

`post_api_request` states the calls as the provider sent them. Hermes then renames a reused id,
replaces a provider's ids in a parallel batch, repairs a misspelled name, runs a deferred tool
through its bridge, and splits one bridge call into several. None of that is a call that never
ran, so a turn says `tool_call_not_dispatched` only when a model call asked for more tool calls
than ran under it. Each case is driven on the fake Hermes with the rewrite Hermes v0.23.3 makes
(`_hermes_payloads.py`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from _hermes_fake import Reply
from _hermes_payloads import provider_ids_normalized, reused_ids_renamed, runs_as
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
    ("asked", "runs"), [("Read_File", "read_file"), ("lab_deferred", "tool_call")]
)
def test_a_call_hermes_ran_under_another_tool_name_ran(asked: str, runs: str):
    """A repaired name, and a deferred tool run through the `tool_call` bridge."""

    def script(h: FakeHermes, turn: Turn) -> None:
        h.tool(runs_as(h.model_call(turn, tool_calls=[asked]), asked, runs), runs)

    _ran_all(*_turn(script))


def test_one_bridge_call_hermes_split_in_two_ran_and_hides_no_other_calls_shortfall():
    """`agent/tool_call_batches.py::expand_local_tool_batches` runs one `tool_call` bridge call
    as several. The extra frame counts for the call that asked, never for a later one: the
    next call's tool that never ran is still named."""

    def script(h: FakeHermes, turn: Turn) -> None:
        bridge = h.model_call(turn, tool_calls=["tool_call"])
        h.tool(bridge, "tool_call")
        h.tool(bridge, "tool_call", tool_call_id="call_lab_split_1")
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
