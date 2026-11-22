"""What the adapter holds back until Hermes says what it meant.

A user message waits for the first model frame, because another plugin's `pre_llm_call` may
veto it and wardex cannot see that callback's answer. A tool report that comes while its worker
still runs waits for the worker, unless Hermes says it gave up on the worker. An auxiliary call
or a declaration whose end never comes stops holding anything after a bound.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
from pathlib import Path

import pytest

from _hermes_fake import OBSERVER_V1, Reply
from _hermes_scripted import QUESTION, FakeHermes
from _hermes_support import (
    exported,
    extras,
    forget_attachments,
    hermes,
    hermes_installed,
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
from wardex_sdk._assembly import Limitation, counters

WIRE = "HTTP POST /v1/chat/completions"
STREAMED = f"{WIRE} | contextvar 1.0 | OK | provider_inferred, reassembled_from_stream"
PLAIN = f"{WIRE} | contextvar 1.0 | OK | provider_inferred"
TURN = "invoke_agent hermes | trace_root 1.0 | OK"
TURN_SPAN = "invoke_agent hermes"
BLOCKED = "This request is not allowed here."


@pytest.fixture(autouse=True)
def _installed(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield
    forget_attachments()


def _io(node) -> tuple[object, object]:
    return node.attributes.get("wardex.input_data"), node.attributes.get("wardex.output_data")


def _anywhere(nodes, text: str) -> list[str]:
    """The spans that carry `text` in any attribute."""
    return [n.name for n in nodes if text in json.dumps(dict(n.attributes), default=str)]


# -- a prompt another plugin may veto -------------------------------------------------------------


def _veto(**_):
    return {"action": "block", "message": BLOCKED}


def test_a_prompt_another_plugin_blocked_never_leaves_the_host():
    """Hermes ends a vetoed turn with no model call. The user's message is in no span, and the
    turn is not marked as one whose model calls ran in a subprocess."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        ctx.register_hook("pre_llm_call", _veto)
        on_loop(lambda: h.end_blocked(h.turn()))
    nodes = exported(transport)
    turn = one(nodes, TURN_SPAN)
    assert _anywhere(nodes, QUESTION) == []
    assert _io(turn) == (None, BLOCKED)
    assert Limitation.SUBPROCESS_MODEL_CALLS_UNOBSERVED not in turn.limitations
    assert extras(turn)["exit_reason"] == "blocked_by_plugin_pre_llm_call"


def test_the_message_of_a_turn_cut_before_its_first_model_frame_is_never_recorded():
    """No frame and no end: wardex never learned whether the prompt was vetoed."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        turn = on_loop(h.turn)
        on_loop(h.finalize, turn.session_id)
    nodes = exported(transport)
    assert _anywhere(nodes, QUESTION) == []
    assert extras(one(nodes, TURN_SPAN))["close_cause"] == "finalize"


def test_the_message_is_recorded_at_the_first_model_frame():
    """A turn cut after its first frame still carries the message: the frame proved that no
    plugin vetoed it."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        turn = on_loop(h.turn)
        on_loop(h.model_call, turn)
        on_loop(h.finalize, turn.session_id)
    assert _io(one(exported(transport), TURN_SPAN)) == (QUESTION, None)


def test_a_turn_that_ended_with_no_model_frame_and_no_block_records_its_message_at_the_end():
    """Hermes under codex app-server: the turn ends completed, its calls ran in a subprocess."""
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(lambda: h.end(h.turn()))
    turn = one(exported(transport), TURN_SPAN)
    assert _io(turn)[0] == QUESTION
    assert Limitation.SUBPROCESS_MODEL_CALLS_UNOBSERVED in turn.limitations


# -- a refusal -------------------------------------------------------------------------------------


def test_a_refused_call_is_an_error_named_by_hermes_and_the_turn_waits_for_the_net(monkeypatch):
    """A content-policy refusal is an HTTP 200 whose frame returned, then api_request_error and
    no post_api_request. The call closes ERROR at the error; Hermes sends no end, so the quiet
    net closes the turn."""
    monkeypatch.setattr(turns_module, "IDLE_AFTER_ERROR_S", 0.3)
    monkeypatch.setattr(attach_module, "IDLE_SWEEP_S", 0.05)
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(lambda: h.model_call(h.turn(), refused=True))
        assert wait_for(lambda: any(n.name == TURN_SPAN for n in received(transport)))
    nodes = exported(transport)
    assert outline(nodes) == render(
        (
            "invoke_agent hermes | trace_root 1.0 | UNSET | session_aborted",
            (
                (
                    "chat lab-model | unit_active 1.0 | ERROR(ContentPolicyBlocked)",
                    ((STREAMED, ()),),
                ),
            ),
        )
    )
    turn = one(nodes, TURN_SPAN)
    assert (extras(turn)["close_cause"], extras(turn)["error_reason"]) == (
        "idle",
        "content_policy_blocked",
    )


# -- a tool Hermes gave up on ----------------------------------------------------------------------


def test_a_tool_hermes_gave_up_on_closes_at_once_and_its_late_worker_changes_nothing():
    """At a tool timeout Hermes stops waiting, reports the tool itself and goes on. The tool's
    span closes ERROR at that report with the message Hermes showed the model; the worker's
    frame exits after the turn closed and adds nothing."""
    counters.reset()
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:

        def run():
            turn = h.turn()
            call = h.model_call(turn, (Reply.OK,), tool_calls=["slow_tool"])
            let_finish = h.abandon_tool(call, "slow_tool")
            h.model_call(turn)
            h.end(turn)
            return let_finish

        on_loop(run)()
    nodes = exported(transport)
    assert outline(nodes) == render(
        (
            TURN,
            (
                ("chat lab-model | unit_active 1.0 | OK", ((PLAIN, ()),)),
                ("chat lab-model | unit_active 1.0 | OK", ((STREAMED, ()),)),
                ("execute_tool slow_tool | unit_active 1.0 | ERROR(tool_timeout)", ()),
            ),
        )
    )
    tool = one(nodes, "execute_tool slow_tool")
    assert tool.attributes["gen_ai.tool.call.result"].startswith("Error executing tool")
    raw = {s.context.span_id: s for e in transport.envelopes for s in e.spans}
    last_chat = max(raw[n.span_id].start_time_ns for n in nodes if n.name == "chat lab-model")
    assert raw[tool.span_id].end_time_ns <= last_chat, "closed at Hermes's report, not later"
    assert counters.get("assembly._units.pin_stale") == 0


def _tool_reported_inside(h: FakeHermes, call, name: str, gate: threading.Event, entered):
    """A tool whose `post_tool_call` fires inside its frame (Hermes's concurrent path), whose
    worker then stays in the frame until `gate` opens."""
    turn = call.turn
    tcid = call.tool_calls[0][0]
    ids = {
        "telemetry_schema_version": OBSERVER_V1,
        "session_id": turn.session_id,
        "task_id": turn.task_id,
        "turn_id": turn.turn_id,
        "api_request_id": call.api_request_id,
        "tool_call_id": tcid,
        "tool_name": name,
    }

    def terminal(args):
        h.ctx.invoke_hook("post_tool_call", **ids, args=args, result="done", duration_ms=1,
                          status="ok", error_type=None, error_message=None)  # fmt: skip
        entered.set()
        gate.wait(10)
        return "done"

    h.ctx.run_execution_chain("tool_execution", terminal, args={}, original_args={}, **ids)


def test_a_report_kept_for_a_running_worker_is_applied_when_its_turn_closes_first():
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        gate, entered = threading.Event(), threading.Event()

        def run():
            turn = h.turn()
            call = h.model_call(turn, (Reply.OK,), tool_calls=["read_file"])
            worker = threading.Thread(
                target=contextvars.copy_context().run,
                args=(_tool_reported_inside, h, call, "read_file", gate, entered),
            )
            worker.start()
            assert entered.wait(10)
            h.end(turn, completed=False, interrupted=True, reason="interrupted")
            return worker

        worker = on_loop(run)
        gate.set()
        worker.join(10)
    tool = one(exported(transport), "execute_tool read_file")
    assert outline((tool,)) == "execute_tool read_file | unit_active 1.0 | OK"
    assert Limitation.CHILD_SPAN_UNCLOSED not in tool.limitations
    assert tool.attributes["gen_ai.tool.call.result"] == "done"


# -- what may never end ----------------------------------------------------------------------------


def _declare_title(h: FakeHermes, turn) -> None:
    """`pre_auxiliary_call` from the title thread, with no `post_auxiliary_call` after it."""
    h.ctx.invoke_hook(
        "pre_auxiliary_call",
        session_id=turn.session_id,
        turn_id=turn.turn_id,
        model="lab-aux",
        api_request_id="aux-lost",
        api_mode="chat_completions",
        api_call_count=1,
        aux_task="title_generation",
        provider="custom",
        retry_count=0,
        streaming=False,
    )


def test_an_auxiliary_call_whose_end_never_came_holds_its_turn_only_for_its_bound(monkeypatch):
    monkeypatch.setattr(turns_module, "IDLE_CLOSE_S", 0.1)
    monkeypatch.setattr(turns_module, "AUX_ATTEMPT_BOUND_S", 0.8)
    monkeypatch.setattr(attach_module, "IDLE_SWEEP_S", 0.05)
    transport = start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        turn = on_loop(lambda: h.turn(title=True))
        declared = time.monotonic()
        on_loop(_declare_title, h, turn)
        on_loop(h.model_call, turn)
        assert wait_for(lambda: any(n.name == TURN_SPAN for n in received(transport)))
        held = time.monotonic() - declared
    assert held >= 0.6, "the attempt held the turn past the quiet net"
    assert extras(one(exported(transport), TURN_SPAN))["close_cause"] == "idle"


def test_a_declaration_with_no_live_turn_wakes_the_wardex_thread_to_expire_it(monkeypatch):
    """The title call outlived its turn and its end was lost. No turn is live, yet the thread
    sweeps until the declaration expires."""
    monkeypatch.setattr(turns_module, "IDLE_CLOSE_S", 0.3)
    monkeypatch.setattr(attach_module, "IDLE_SWEEP_S", 0.05)
    counters.reset()
    start_wardex()
    with plugin() as ctx, hermes(ctx) as h:
        turn = on_loop(h.turn)
        on_loop(h.model_call, turn)
        on_loop(h.end, turn)
        on_loop(_declare_title, h, turn)
        assert wait_for(lambda: counters.get("adapters.hermes_agent.unreached_expired") == 1)
