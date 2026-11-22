"""What starts on Hermes's loop thread after a turn Hermes did not end, and how long it waits.

Hermes sends no end for a failed turn, nor for one it stops as partial (`gave_up`), so the turn
stays live and pinned on its loop thread until its session's next turn, its finalize or the
quiet net closes it. Nothing Hermes sends says whether such a turn is over: an embedder's
`step_callback` runs an agent between two model calls of a live turn. So a turn that starts on
another turn's bare pin opens where that turn was opened, and leaves it as it is; the
embedder's own span after a failed call is not under it; and a wait Hermes announced after a
failed call keeps the turn one trace when an auxiliary call ends or an agent starts inside it.

Each case is driven on the fake Hermes in the order and on the threads Hermes v0.23.3 uses:
`FakeHermes.fail` raises out of the `llm_execution` frame on the loop thread, as Hermes's retry
loop does (`agent/conversation_loop.py`), and a new `AIAgent` on the same thread fires its
`pre_llm_call` from a hook thread copied from that loop thread's context.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import wardex_sdk
from _hermes_fake import Reply
from _hermes_payloads import gave_up, prelude_tool, slow_tool, tool_starts_agent
from _hermes_scripted import FakeHermes, Turn
from _hermes_support import (
    exported,
    extras,
    forget_attachments,
    hermes,
    hermes_installed,
    on_loop,
    one,
    outline,
    parent_of,
    plugin,
    start_wardex,
    typed_usage,
)
from wardex_sdk._adapters import _hermes_agent_attach as attach_module
from wardex_sdk._adapters import _hermes_agent_turns as turns_module
from wardex_sdk._assembly import Limitation

TURN = "invoke_agent hermes"
WIRE = "HTTP POST /v1/chat/completions"


@pytest.fixture(autouse=True)
def _installed(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield
    forget_attachments()


def _turn_span(nodes, turn: Turn):
    found = [n for n in nodes if n.name == TURN and extras(n)["turn_id"] == turn.turn_id]
    assert len(found) == 1, [outline((n,)) for n in found]
    return found[0]


def _in_trace(nodes, node) -> list[str]:
    return sorted(n.name for n in nodes if n.trace_id == node.trace_id)


def _after_failed_turn(h: FakeHermes, fail) -> tuple[Turn, Turn]:
    first = h.turn()
    fail(first)
    second = h.turn()  # another session, on the same loop thread
    h.model_call(second)
    h.end(second)
    h.finalize(first.session_id, reason="session_close")  # exports a failed turn still open
    return first, second


def test_another_sessions_turn_after_a_failed_turn_on_the_same_thread_is_its_own_root():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        failed, second = on_loop(_after_failed_turn, h, h.fail)
    nodes = exported(transport)
    span = _turn_span(nodes, second)
    assert parent_of(nodes, span) is None
    assert extras(span)["close_cause"] == "end_event"
    assert _in_trace(nodes, span) == sorted([TURN, "chat lab-model", WIRE])
    assert _turn_span(nodes, failed).trace_id != span.trace_id


def test_a_turn_left_open_by_a_refusal_is_left_to_its_own_nets_by_another_sessions_turn():
    """A refusal is an HTTP 200: its frame returns, so the turn stays pinned on the loop thread.
    Another session's turn there opens beside it and does not close it: its own session's
    finalize does."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        failed, second = on_loop(
            _after_failed_turn, h, lambda turn: h.model_call(turn, (Reply.OK,), refused=True)
        )
    nodes = exported(transport)
    span = _turn_span(nodes, second)
    assert parent_of(nodes, span) is None
    closed = _turn_span(nodes, failed)
    assert closed.trace_id != span.trace_id
    assert extras(closed)["close_cause"] == "finalize"
    assert extras(closed)["error_reason"] == "content_policy_blocked"
    assert Limitation.SESSION_ABORTED in closed.limitations


def test_the_embedders_own_span_after_a_failed_turn_is_not_under_it():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run() -> Turn:
            turn = h.turn()
            h.fail(turn)
            with wardex_sdk.span("embedder_fallback"):
                h.provider.post(Reply.OK)
            h.finalize(turn.session_id, reason="session_close")
            return turn

        failed = on_loop(run)
    nodes = exported(transport)
    own = next(n for n in nodes if n.name == "embedder_fallback")
    assert parent_of(nodes, own) is None
    assert own.trace_id != _turn_span(nodes, failed).trace_id
    assert [n.name for n in nodes if n.parent_id == own.span_id] == [WIRE]


def test_an_auxiliary_call_between_a_failed_attempt_and_the_retry_is_marked_and_counted_once():
    """A compression after a context-overflow error runs on the loop thread before Hermes's
    next attempt, when the turn is not pinned there: a hook-built child of the turn, marked
    possible_duplicate_chat, and its wire span an unresolved root. Its usage is typed once."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run() -> Turn:
            turn = h.turn()
            h.fail(turn, (Reply.BAD_REQUEST,))
            h.aux(turn, from_loop=True, aux_task="compression", model="lab-aux-compress")
            h.model_call(turn, (Reply.OK,), again=True)
            h.end(turn)
            return turn

        turn = on_loop(run)
    nodes = exported(transport)
    span = _turn_span(nodes, turn)
    assert extras(span)["close_cause"] == "end_event"
    built = [n for n in nodes if n.name == "chat lab-aux-compress"]
    assert len(built) == 1
    assert parent_of(nodes, built[0]) is span
    assert Limitation.POSSIBLE_DUPLICATE_CHAT in built[0].limitations
    wire = [n for n in nodes if n.name == WIRE and n.parent_id is None]
    assert len(wire) == 1
    assert Limitation.PARENT_UNRESOLVED in wire[0].limitations
    typed = [n for n in (*built, *wire) if typed_usage(n)]
    assert len(typed) == 1, [outline((n,)) for n in typed]


def test_a_failed_call_s_wait_outlasts_an_auxiliary_call_that_ends_inside_it(monkeypatch):
    """The turn-start title call ends after the turn's first call failed with a long
    Retry-After. Its end is no sign that the loop went on, so the quiet net keeps waiting
    IDLE_AFTER_ERROR_S, and the retry that answers stays in the same span."""
    monkeypatch.setattr(turns_module, "IDLE_CLOSE_S", 0.3)
    monkeypatch.setattr(turns_module, "IDLE_AFTER_ERROR_S", 30.0)
    monkeypatch.setattr(attach_module, "IDLE_SWEEP_S", 0.05)
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run() -> Turn:
            turn = h.turn(title=True)
            title = h.aux(turn, answer=Reply.HOLD, wait=False)
            h.fail(turn, (Reply.UNAVAILABLE,))
            h.provider.release()
            assert title is not None
            title.join(10)
            time.sleep(1.0)  # Hermes's wait: past IDLE_CLOSE_S, under the other
            h.model_call(turn, (Reply.OK,), again=True)
            h.end(turn)
            return turn

        turn = on_loop(run)
    nodes = exported(transport)
    spans = [n for n in nodes if n.name == TURN]
    assert len(spans) == 1, [outline((n,)) for n in spans]
    assert extras(spans[0])["close_cause"] == "end_event"
    assert extras(spans[0])["turn_id"] == turn.turn_id
    assert "generation" not in extras(spans[0])


def _calls_of(nodes, turn: Turn) -> list:
    return [n for n in nodes if extras(n).get("api_request_id", "").startswith(turn.turn_id)]


def test_an_agent_run_from_a_step_callback_is_its_own_trace_and_leaves_the_turn_whole():
    """Hermes calls an embedder's `step_callback` on the loop thread before a model call
    (`agent/turn_iteration_prep.py`). An agent run there starts its turn on the live turn's
    bare pin, which looks the same as a turn Hermes stopped: it opens where that turn opened, a
    trace of its own (Hermes states no parent for it), and the live turn stays one span."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run() -> tuple[Turn, Turn]:
            main = h.turn()
            h.tool(h.model_call(main, tool_calls=["read_file"]), "read_file")
            judge = h.turn()  # the step callback, before the main turn's next model call
            h.model_call(judge)
            h.end(judge)
            h.model_call(main)
            h.end(main)
            return main, judge

        main, judge = on_loop(run)
    nodes = exported(transport)
    span = _turn_span(nodes, main)
    assert extras(span)["close_cause"] == "end_event"
    assert Limitation.SESSION_ABORTED not in span.limitations
    other = _turn_span(nodes, judge)
    assert parent_of(nodes, other) is None
    assert other.trace_id != span.trace_id
    calls = _calls_of(nodes, main)
    assert len(calls) == 2
    assert all(parent_of(nodes, n) is span for n in calls)


def test_an_agent_on_a_thread_a_tool_left_running_is_its_own_trace_and_leaves_the_turn_whole():
    """A tool starts an agent on a thread that keeps the tool's context, and returns. The
    agent's turn starts while the turn that called the tool is live: that turn stays one span
    that Hermes ends, and the agent's turn is a root of its own, since the tool it started
    under has closed."""
    transport = start_wardex()
    helper: list[Turn] = []
    with plugin() as ctx, hermes(ctx) as h:

        def agent() -> None:
            turn = h.turn()
            h.model_call(turn)
            h.end(turn)
            helper.append(turn)

        def run() -> Turn:
            main = h.turn()
            call = h.model_call(main, tool_calls=["lab_spawn"])
            thread = tool_starts_agent(h, call, "lab_spawn", agent)
            thread.start()
            thread.join(30)
            h.model_call(main)
            h.end(main)
            return main

        main = on_loop(run)
    nodes = exported(transport)
    span = _turn_span(nodes, main)
    assert extras(span)["close_cause"] == "end_event"
    assert Limitation.SESSION_ABORTED not in span.limitations
    assert all(parent_of(nodes, n) is span for n in _calls_of(nodes, main))
    other = _turn_span(nodes, helper[0])
    assert parent_of(nodes, other) is None
    assert other.trace_id != span.trace_id
    assert Limitation.CORRELATION_CONFLICT in other.limitations


# -- a turn Hermes stopped with no end -------------------------------------------------------------


def test_another_sessions_turn_after_a_turn_hermes_gave_up_on_is_its_own_root():
    """The stopped turn's pin still stands on the loop thread. The next session's turn there
    opens where the stopped turn opened, a root, and ends whole; the stopped turn keeps only
    its own calls and closes at its own session's finalize."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        stopped, second = on_loop(_after_failed_turn, h, lambda turn: gave_up(h, turn))
    nodes = exported(transport)
    span = _turn_span(nodes, second)
    assert parent_of(nodes, span) is None
    assert outline((span,)) == "invoke_agent hermes | trace_root 1.0 | OK"
    assert extras(span)["close_cause"] == "end_event"
    assert _in_trace(nodes, span) == sorted([TURN, "chat lab-model", WIRE])
    left = _turn_span(nodes, stopped)
    assert extras(left)["close_cause"] == "finalize"
    assert _in_trace(nodes, left) == sorted([TURN, *["chat lab-model", WIRE] * 3])


def test_the_quiet_net_of_a_turn_hermes_gave_up_on_closes_that_turn_alone(monkeypatch):
    """The next session's turn is still running a long tool when the stopped turn's quiet net
    fires. The net closes the stopped turn only; the running turn stays one span that Hermes
    ends."""
    monkeypatch.setattr(turns_module, "IDLE_CLOSE_S", 0.5)
    monkeypatch.setattr(attach_module, "IDLE_SWEEP_S", 0.05)
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run() -> tuple[Turn, Turn]:
            stopped = h.turn()
            gave_up(h, stopped)
            second = h.turn()
            slow = h.model_call(second, tool_calls=["lab_slow"])
            slow_tool(h, slow, "lab_slow", 2.0)  # the stopped turn's net fires inside this tool
            h.model_call(second)
            h.end(second)
            return stopped, second

        stopped, second = on_loop(run)
    nodes = exported(transport)
    assert extras(_turn_span(nodes, stopped))["close_cause"] == "idle"
    span = _turn_span(nodes, second)
    assert parent_of(nodes, span) is None
    assert extras(span)["close_cause"] == "end_event"
    assert "generation" not in extras(span)
    assert Limitation.SESSION_ABORTED not in span.limitations


def test_an_embedder_span_around_two_turns_holds_the_turn_after_one_hermes_never_ended():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run() -> tuple[Turn, Turn]:
            with wardex_sdk.span("lab_batch"):
                stopped = h.turn()
                gave_up(h, stopped)
                second = h.turn()
                h.model_call(second)
                h.end(second)
            h.finalize(stopped.session_id, reason="session_close")
            return stopped, second

        stopped, second = on_loop(run)
    nodes = exported(transport)
    batch = one(nodes, "lab_batch")
    assert parent_of(nodes, _turn_span(nodes, stopped)) is batch
    span = _turn_span(nodes, second)
    assert parent_of(nodes, span) is batch
    assert extras(span)["close_cause"] == "end_event"


def test_a_prelude_tool_on_a_thread_still_pinned_to_another_turn_is_a_root_of_its_own_turn():
    """A tool a turn runs before its first model call, on a worker copied from a loop thread
    whose lane is still the stopped turn: what a prelude tool always is, a marked root that
    states its own turn."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run() -> Turn:
            stopped = h.turn()
            gave_up(h, stopped)
            second = h.turn()
            prelude_tool(h, second)
            h.model_call(second)
            h.end(second)
            h.finalize(stopped.session_id, reason="session_close")
            return second

        second = on_loop(run)
    nodes = exported(transport)
    tool = one(nodes, "execute_tool read_file")
    assert outline((tool,)) == "execute_tool read_file | unresolved 0.0 | OK | parent_unresolved"
    assert extras(tool)["turn_id"] == second.turn_id


def test_an_agent_a_tool_started_whose_turn_begins_in_a_retry_wait_leaves_the_turn_whole():
    """The tool's thread still carries the calling turn's lane. That turn's next call failed and
    Hermes waits before its retry; the agent's turn starts in that wait. The calling turn stays
    one span that Hermes ends, and the agent's turn is a trace of its own."""
    transport = start_wardex()
    helper: list[Turn] = []
    with plugin() as ctx, hermes(ctx) as h:

        def agent() -> None:
            turn = h.turn()
            h.model_call(turn)
            h.end(turn)
            helper.append(turn)

        def run() -> Turn:
            main = h.turn()
            spawn = h.model_call(main, tool_calls=["lab_spawn"])
            thread = tool_starts_agent(h, spawn, "lab_spawn", agent)
            h.fail(main, (Reply.UNAVAILABLE,))
            thread.start()  # Hermes's wait before its retry
            thread.join(30)
            h.model_call(main, (Reply.OK,), again=True)
            h.end(main)
            return main

        main = on_loop(run)
    nodes = exported(transport)
    span = _turn_span(nodes, main)
    assert extras(span)["close_cause"] == "end_event"
    assert "generation" not in extras(span)
    other = _turn_span(nodes, helper[0])
    assert parent_of(nodes, other) is None
    assert other.trace_id != span.trace_id
