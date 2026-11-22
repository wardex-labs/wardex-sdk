"""What runs on Hermes's loop thread after a turn failed, and how long a failed turn waits.

Hermes sends no end for a failed turn, so the turn stays live for minutes. Nothing that runs
after it on the same thread may sit under it: another session's turn, the embedder's own span,
an auxiliary call Hermes makes before its next attempt. And a wait Hermes announced after a
failed call (a Retry-After) keeps the turn one trace even when an auxiliary call ends inside it.

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
from _hermes_scripted import FakeHermes, Turn
from _hermes_support import (
    exported,
    extras,
    forget_attachments,
    hermes,
    hermes_installed,
    on_loop,
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


def test_a_turn_left_open_by_a_refusal_is_closed_by_the_next_turn_on_its_loop_thread():
    """A refusal is an HTTP 200: its frame returns, so the turn stays pinned on the loop thread.
    The next root turn there proves the failed turn's loop is over and closes it first."""
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
    assert extras(closed)["close_cause"] == "next_turn"
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
