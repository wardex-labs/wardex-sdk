"""How a Hermes turn ends when Hermes does not say so, and the calls Hermes makes beside a turn.

Hermes sends no end for a failed turn, may start the next turn of a session before ending the
last, and can be stopped with a turn open. Each net closes such a turn UNSET with
`session_aborted` and names its cause in `close_cause`. A net that guessed wrong costs a second
span for the rest of the turn, linked `RESUMED_FROM` the first. Also here: the title call Hermes
makes on its own thread, subagents that outlive their parent, concurrent gateway turns, and the
signal path closing units under a live turn.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
from pathlib import Path

import pytest

import wardex_sdk
from _hermes_fake import OBSERVER_V1, Reply
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
    render,
    start_wardex,
    wait_for,
)
from wardex_sdk._adapters import _hermes_agent_attach as attach_module
from wardex_sdk._adapters import _hermes_agent_turns as turns_module
from wardex_sdk._adapters._hermes_agent_turns import Action
from wardex_sdk._adapters._registry import get_registry
from wardex_sdk._assembly import Limitation, LinkReason, counters

WIRE = "HTTP POST /v1/chat/completions"
STREAMED = f"{WIRE} | contextvar 1.0 | OK | provider_inferred, reassembled_from_stream"
CHAT = ("chat lab-model | unit_active 1.0 | OK", ((STREAMED, ()),))
TURN = "invoke_agent hermes | trace_root 1.0 | OK"
ABORTED = "invoke_agent hermes | trace_root 1.0 | UNSET | session_aborted"
FAILED_CALL = (
    "chat lab-model | unit_active 1.0 | ERROR(InternalServerError)",
    ((f"{WIRE} | contextvar 1.0 | ERROR(500)", ()),) * 2,
)
UNBOUND_LINE = "carries no wardex context"


@pytest.fixture(autouse=True)
def _installed(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield
    forget_attachments()


def _roots(nodes):
    return [n for n in nodes if n.name == "invoke_agent hermes"]


# -- the title call ------------------------------------------------------------------------------


def test_the_turn_start_title_call_is_a_marked_child_and_its_wire_span_an_unresolved_root(caplog):
    transport = start_wardex()
    with caplog.at_level(logging.INFO, logger="wardex_sdk"), plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn(title=True)
            h.aux(turn)
            h.model_call(turn)
            h.end(turn)

        on_loop(run)
    nodes = exported(transport)
    assert outline(nodes) == render(
        (f"{WIRE} | unresolved 0.0 | OK | parent_unresolved, provider_inferred", ()),
        (TURN, (("chat lab-aux-title | unit_active 1.0 | OK | possible_duplicate_chat", ()), CHAT)),
    )
    hook = one(nodes, "chat lab-aux-title")
    assert extras(hook)["aux_task"] == "title_generation"
    assert extras(hook)["aux_request_id"].startswith("aux-")
    said = [r.getMessage() for r in caplog.records]
    assert not [m for m in said if UNBOUND_LINE in m], "the adapter declared this thread"


def test_a_title_call_that_fails_leaves_the_hook_span_as_its_only_record():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn(title=True)
            h.aux(turn, answer=Reply.BAD_REQUEST)
            h.model_call(turn)
            h.end(turn)

        on_loop(run)
    assert outline(exported(transport)) == render(
        (
            TURN,
            (
                (
                    "chat lab-aux-title | unit_active 1.0 | ERROR(BadRequestError)"
                    " | possible_duplicate_chat",
                    (),
                ),
                CHAT,
            ),
        ),
    )


def test_an_auxiliary_call_on_the_turns_own_thread_adds_nothing_its_wire_span_is_in_the_turn():
    """Not measured in the lab (no scenario produced one): compression run on the loop thread
    mid-turn. Its wire span is already under the turn, so a hook span would double it."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.model_call(turn)
            h.aux(turn, from_loop=True, aux_task="compression")
            h.model_call(turn)
            h.end(turn)

        on_loop(run)
    assert outline(exported(transport)) == render(
        (TURN, (CHAT, CHAT, (f"{WIRE} | contextvar 1.0 | OK | provider_inferred", ()))),
    )


def _stray_request(h: FakeHermes) -> None:
    on_loop(h.provider.post, Reply.OK, name="stray")


def _declare_title(h: FakeHermes, turn) -> None:
    """`pre_auxiliary_call` from the title thread, with no `post_auxiliary_call` after it."""
    h.ctx.invoke_hook(
        "pre_auxiliary_call",
        telemetry_schema_version=OBSERVER_V1,
        session_id=turn.session_id,
        task_id=turn.task_id,
        turn_id=turn.turn_id,
        model="lab-aux-title",
        platform="",
        api_request_id="aux-lost",
        api_mode="chat_completions",
        api_call_count=1,
        aux_task="title_generation",
        provider="custom",
        retry_count=0,
        streaming=False,
    )


def _stray_lines(nodes) -> list[str]:
    return [outline((n,)) for n in nodes if n.name == WIRE and n.parent_id is None]


def test_an_unreached_declaration_stands_until_its_turn_closes():
    """While the title call is declared, a request no context reaches is an unresolved root;
    once the turn closes, the declaration is withdrawn and the same request is a plain root."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        turn = on_loop(lambda: h.turn(title=True))
        on_loop(_declare_title, h, turn)
        on_loop(h.model_call, turn)
        _stray_request(h)
        on_loop(h.end, turn)
        _stray_request(h)
    nodes = exported(transport)
    assert _stray_lines(nodes) == [
        f"{WIRE} | unresolved 0.0 | OK | parent_unresolved, provider_inferred",
        f"{WIRE} | trace_root 1.0 | OK | provider_inferred",
    ]
    hook = one(nodes, "chat lab-aux-title")
    assert Limitation.CHILD_SPAN_UNCLOSED in hook.limitations


def test_a_finished_auxiliary_call_withdraws_its_declaration_at_once():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        turn = on_loop(lambda: h.turn(title=True))
        on_loop(lambda: h.aux(turn))
        _stray_request(h)
        on_loop(lambda: (h.model_call(turn), h.end(turn)))
    nodes = exported(transport)
    strays = [line for line in _stray_lines(nodes) if "trace_root" in line]
    assert strays == [f"{WIRE} | trace_root 1.0 | OK | provider_inferred"]


def test_uninstalling_withdraws_every_declaration():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        turn = on_loop(lambda: h.turn(title=True))
        on_loop(_declare_title, h, turn)
        wardex_sdk.close()
        transport = start_wardex()
        _stray_request(h)
    assert _stray_lines(exported(transport)) == [
        f"{WIRE} | trace_root 1.0 | OK | provider_inferred"
    ]


# -- the second turn and the nets ----------------------------------------------------------------


def test_a_second_turn_on_the_same_thread_is_a_clean_root():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            first = h.turn()
            h.model_call(first)
            h.end(first)
            second = h.turn(first.session_id)
            h.model_call(second)
            h.end(second)

        on_loop(run)
    nodes = exported(transport)
    assert outline(nodes) == render((TURN, (CHAT,)), (TURN, (CHAT,)))
    assert len({n.trace_id for n in _roots(nodes)}) == 2


def test_a_turn_with_no_end_is_closed_by_the_next_turn_of_its_session():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            first = h.turn()
            h.model_call(first)
            second = h.turn(first.session_id)
            h.model_call(second)
            h.end(second)

        on_loop(run)
    nodes = exported(transport)
    assert outline(nodes) == render((ABORTED, (CHAT,)), (TURN, (CHAT,)))
    closed = next(n for n in _roots(nodes) if n.limitations)
    assert extras(closed)["close_cause"] == "next_turn"


def test_a_failed_turn_then_a_turn_gives_a_clean_second_root():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            first = h.turn()
            h.fail(first)
            second = h.turn(first.session_id)
            h.model_call(second)
            h.end(second)

        on_loop(run)
    assert outline(exported(transport)) == render((ABORTED, (FAILED_CALL,)), (TURN, (CHAT,)))


def test_a_turn_that_carries_on_after_retries_were_predicted_spent_resumes_in_a_linked_span():
    """Hermes's fallback recovered the call: the prediction that closed the turn was wrong."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.fail(turn)
            h.model_call(turn)
            h.end(turn)
            return turn

        turn = on_loop(run)
    nodes = exported(transport)
    assert outline(nodes) == render((ABORTED, (FAILED_CALL,)), (TURN, (CHAT,)))
    first = next(n for n in _roots(nodes) if n.limitations)
    resumed = next(n for n in _roots(nodes) if not n.limitations)
    assert links(transport, resumed, nodes) == [(LinkReason.RESUMED_FROM, first.name)]
    assert extras(resumed)["turn_id"] == extras(first)["turn_id"] == turn.turn_id
    assert extras(resumed)["generation"] == 1
    assert extras(resumed)["close_cause"] == "end_event"


def test_an_end_that_arrives_after_the_turn_was_presumed_closed_closes_a_resumed_span():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.fail(turn)
            h.end(turn, completed=False, failed=True, reason="failed")

        on_loop(run)
    nodes = exported(transport)
    first = next(n for n in _roots(nodes) if extras(n)["close_cause"] == "retries_spent")
    resumed = next(n for n in _roots(nodes) if n is not first)
    assert (
        outline((resumed,))
        == "invoke_agent hermes | trace_root 1.0 | ERROR(failed) | session_aborted"
    )
    assert links(transport, resumed, nodes) == [(LinkReason.RESUMED_FROM, first.name)]
    assert extras(resumed)["exit_reason"] == "failed"


def test_an_idle_turn_is_closed_and_exported_by_the_wardex_thread(monkeypatch):
    monkeypatch.setattr(turns_module, "IDLE_CLOSE_S", 0.3)
    monkeypatch.setattr(attach_module, "IDLE_SWEEP_S", 0.05)
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(lambda: h.model_call(h.turn()))
        assert wait_for(lambda: bool(_roots(received(transport)))), "no flush from the test"
    nodes = exported(transport)
    assert outline(nodes) == render((ABORTED, (CHAT,)))
    assert extras(_roots(nodes)[0])["close_cause"] == "idle"


def test_an_unreached_declaration_whose_turn_is_gone_expires_with_the_idle_net(monkeypatch):
    monkeypatch.setattr(turns_module, "IDLE_CLOSE_S", 0.3)
    monkeypatch.setattr(attach_module, "IDLE_SWEEP_S", 0.05)
    counters.reset()
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        turn = on_loop(h.turn)
        on_loop(h.model_call, turn)
        on_loop(h.end, turn)
        on_loop(_declare_title, h, turn)  # the turn is closed: declared with no live turn
        on_loop(h.model_call, h.turn())  # keeps the net running
        assert wait_for(lambda: counters.get("adapters.hermes_agent.unreached_expired") == 1)
        _stray_request(h)
    lines = _stray_lines(exported(transport))
    assert lines == [f"{WIRE} | trace_root 1.0 | OK | provider_inferred"]


def test_finalize_cuts_an_open_turn_and_drains_it_on_the_calling_thread_within_its_bound():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            h.model_call(turn)
            started = time.monotonic()
            returned = h.finalize(turn.session_id)
            return returned, time.monotonic() - started, _roots(received(transport))

        returned, took, seen = on_loop(run)
    assert returned == [], "nothing for Hermes to print"
    assert took < 5.0
    assert [extras(n)["close_cause"] for n in seen] == ["finalize"], "on the wire before return"
    assert outline(exported(transport)) == render((ABORTED, (CHAT,)))


def test_a_child_that_never_ended_is_closed_by_its_subagent_stop():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def silent(hermes_, kid):
            hermes_.model_call(kid, (Reply.OK,))

        def run():
            turn = h.turn()
            h.delegate(h.model_call(turn, tool_calls=["delegate_task"]), 1, child=silent)
            h.model_call(turn)
            h.end(turn)

        on_loop(run)
    nodes = exported(transport)
    child = next(n for n in _roots(nodes) if extras(n).get("child_role"))
    assert outline((child,)) == "invoke_agent hermes | unit_active 1.0 | UNSET | session_aborted"
    assert extras(child)["close_cause"] == "subagent_stop"


# -- concurrency and the signal path -------------------------------------------------------------


def test_two_concurrent_gateway_turns_of_different_sessions_stay_apart():
    """Not measured in the lab (gateway messages were not run): each job copies its own context."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        both_live = threading.Barrier(2, timeout=10)

        def job():
            turn = h.turn()
            h.model_call(turn)
            both_live.wait()
            h.model_call(turn)
            h.end(turn)
            return turn.session_id

        threads = [
            threading.Thread(target=contextvars.Context().run, args=(h.gateway_job, job, n))
            for n in (1, 2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
    nodes = exported(transport)
    assert outline(nodes) == render((TURN, (CHAT, CHAT)), (TURN, (CHAT, CHAT)))
    for root in _roots(nodes):
        mine = [n for n in nodes if n.trace_id == root.trace_id and n.name == "chat lab-model"]
        assert {n.attributes["gen_ai.conversation.id"] for n in mine} == {
            root.attributes["gen_ai.conversation.id"]
        }


def test_closing_units_from_a_signal_never_waits_on_an_effect_in_progress():
    """`close_units` runs from a signal handler and takes no lock of the adapter's: here a hook
    thread is parked inside the close effect holding both of them."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        turn = on_loop(h.turn)
        parked, release = threading.Event(), threading.Event()
        adapter = attach_module._ROUTE
        close = adapter._effects[Action.CLOSE]

        def parked_close(*args):
            with adapter._lock, adapter._book._lock:
                parked.set()
                release.wait(10)
            return close(*args)

        adapter._effects[Action.CLOSE] = parked_close
        ender = threading.Thread(target=contextvars.Context().run, args=(h.end, turn))
        ender.start()
        assert parked.wait(10)
        closer = threading.Thread(
            target=get_registry().close_units_all, kwargs={"marker": Limitation.UNIT_INTERRUPTED}
        )
        closer.start()
        closer.join(5)
        finished = not closer.is_alive()
        release.set()
        ender.join(10)
        adapter._effects[Action.CLOSE] = close
        assert finished, "close_units waited on the parked effect"
        assert ctx.failures == []
    root = one(exported(transport), "invoke_agent hermes")
    assert Limitation.UNIT_INTERRUPTED in root.limitations


def test_an_event_for_a_turn_the_signal_path_closed_finds_it_gone_and_resumes_once():
    """The signal path closes the turn's unit and leaves the adapter's record alone; the turn's
    next frame finds the unit gone and carries the rest of the turn in one resumed span."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        turn = on_loop(h.turn)
        on_loop(h.model_call, turn)
        get_registry().close_units_all(marker=Limitation.UNIT_INTERRUPTED)
        on_loop(h.model_call, turn)
        on_loop(h.end, turn)
        assert ctx.failures == []
    nodes = exported(transport)
    assert outline(nodes) == render(
        ("invoke_agent hermes | trace_root 1.0 | UNSET | unit_interrupted", (CHAT,)),
        (TURN, (CHAT,)),
    )
    cut = next(n for n in _roots(nodes) if n.limitations)
    resumed = next(n for n in _roots(nodes) if not n.limitations)
    assert links(transport, resumed, nodes) == [(LinkReason.RESUMED_FROM, cut.name)]
    assert (extras(resumed)["generation"], extras(resumed)["close_cause"]) == (1, "end_event")


def test_the_wardex_thread_starts_with_the_first_turn_and_idles_without_a_live_one(monkeypatch):
    monkeypatch.setattr(attach_module, "IDLE_SWEEP_S", 0.02)
    start_wardex()

    def exporters():
        return [t for t in threading.enumerate() if t.name == "wardex-hermes"]

    with plugin() as ctx, hermes(ctx) as h:
        assert exporters() == [], "registering starts no thread"
        sweeps: list[float] = []
        adapter = attach_module._ROUTE
        real = type(adapter).reap_idle
        monkeypatch.setattr(
            type(adapter), "reap_idle", lambda self, now: (sweeps.append(now), real(self, now))
        )
        turn = on_loop(h.turn)
        assert len(exporters()) == 1
        time.sleep(0.2)
        assert len(sweeps) >= 3, "a live turn is swept"
        on_loop(lambda: (h.model_call(turn), h.end(turn)))
        time.sleep(0.1)
        quiet_from = len(sweeps)
        time.sleep(0.3)
        assert len(sweeps) == quiet_from, "no live turn: the thread waits on its event"
