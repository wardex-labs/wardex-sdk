"""The Hermes scope map as test data: every row is a span, a marker or a notice a user can see.

`SCOPE_ROWS` is the registry. Each row names the scope-map surface it proves, the scenario that
produces it (a recorded fixture replayed through the fake plugin context, or a scripted
`FakeHermes` run for what no fixture recorded), the code that produces it, and what a reader of
the exported trace, of the WARNING lines or of `wardex.diagnostics()` observes. One parametrized
test runs them all; others fail when a surface has no row, a script no use, or a row names code
that does not exist.

`NOT_MEASURED` lists what no lab run produced. Each item has exactly one row, driven by the fake
alone, whose `unmeasured` field names it: that row pins what the adapter does with the fake's
shape of it, and says that Hermes's real behaviour there was never observed.

The complete trees are pinned in `test_hermes_agent_trees.py`, `test_hermes_agent_replays.py` and
`test_hermes_agent_lifecycle.py`; a row here asserts the part of a tree its surface is about.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from _hermes_fake import OBSERVER_V1, Reply, ScriptedProvider, Usage
from _hermes_replay import Replay
from _hermes_scripted import Call, FakeHermes, Turn, _usage
from _hermes_support import (
    exported,
    forget_attachments,
    hermes,
    hermes_installed,
    links,
    on_loop,
    outline,
    parent_of,
    plugin,
    received,
    start_wardex,
    status,
    wait_for,
)
from wardex_sdk._assembly import Limitation
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._diagnostics import AdapterStatus
from wardex_sdk._enums import AdapterState
from wardex_sdk.testing import RecordingTransport, SpanNode

#: The grounding scope map's surfaces, verbatim. A surface with no row in SCOPE_ROWS fails.
SURFACES: tuple[str, ...] = (
    "Turn",
    "Model call attempt, retries included",
    "Bedrock Converse",
    "API modes with no bytes in this process",
    "codex app-server mode",
    "Auxiliary model calls (title, compression)",
    "Tool calls, sequential and parallel",
    "Subagents (delegate_task)",
    "Async delegation (library default)",
    "MCP tools",
    "Interrupted turn; abandoned tool worker",
    "Several profiles in one process",
    "Out of process",
    "plugins.isolation: host",
    "Re-exec and exit",
    "End-user identity",
)

#: What no lab run produced. Each has one fake-driven row naming it in `unmeasured`.
NOT_MEASURED: tuple[str, ...] = (
    "Bedrock Converse wire capture",
    "an auxiliary call from the loop thread",
    "two concurrent gateway sessions",
    "tool timeout with an abandoned worker",
    "plugins.isolation: host",
    "MCP sampling",
    "codex app-server turn hooks",
    "a gateway that exits within milliseconds of a turn close",
    "fallback recovery after a retries_spent close",
)

L = Limitation
WIRE = "HTTP POST /v1/chat/completions"
TURN = "invoke_agent hermes"
CHAT = "chat lab-model"
PREFIX = "wardex.hermes_agent."
SENDER = "lab-sender-4417"


# -- what can be observed ------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """Spans named `name` that carry everything given: at least one, or exactly `n`."""

    name: str
    parent: str | None = None  # the parent span's name; "" means a trace root
    source: str | None = None  # how the parent was resolved: "contextvar", "unit_active", ...
    confidence: float | None = None
    status: str | None = None  # "OK", "ERROR", "UNSET"
    error_type: str | None = None
    markers: tuple[Limitation, ...] = ()
    no_markers: tuple[Limitation, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)
    n: int | None = None


@dataclass(frozen=True)
class Count:
    """Exactly `n` spans named `name` (or whose name starts with it, with `prefix`)."""

    name: str
    n: int
    prefix: bool = False


@dataclass(frozen=True)
class Linked:
    span: str
    reason: str  # "triggered_by", "resumed_from"
    target: str


@dataclass(frozen=True)
class Traces:
    n: int


@dataclass(frozen=True)
class UsageOnce:
    """Every span named `call` has its usage typed on exactly one span of its subtree."""

    call: str


@dataclass(frozen=True)
class Line:
    key: str  # the `wardex_reason` of one WARNING line


@dataclass(frozen=True)
class Status:
    state: AdapterState
    detail: str = ""  # a substring of the row's detail


@dataclass(frozen=True)
class Never:
    """No exported attribute key or value satisfies the predicate."""

    what: str
    test: Callable[[str, object], bool]


@dataclass(frozen=True)
class Early:
    """A span named `name` reached the transport before anything flushed it from outside."""

    name: str


Expect = Span | Count | Linked | Traces | UsageOnce | Line | Status | Never | Early


@dataclass(frozen=True)
class ScopeRow:
    id: str
    surface: str
    scenario: str  # "fixture:<name>" or "script:<name in SCRIPTS>"
    produced_by: tuple[str, ...]  # "<module>::<identifier>", each checked against the source
    observe: tuple[Expect, ...]
    unmeasured: str = ""  # a NOT_MEASURED item: the fake drives it, Hermes was never observed


# -- running a scenario --------------------------------------------------------------------------


@dataclass
class Stage:
    """One scenario's process: wardex started by the script, and what it saw before a flush."""

    monkeypatch: pytest.MonkeyPatch
    home: Path
    transport: RecordingTransport | None = None
    early: tuple[SpanNode, ...] = ()

    def start(self) -> RecordingTransport:
        self.transport = start_wardex()
        return self.transport


Script = Callable[[Stage], None]


def _looped(body: Callable[[FakeHermes], None]) -> Script:
    """wardex started, one plugin context, `body` as one Hermes loop."""

    def run(stage: Stage) -> None:
        stage.start()
        with plugin() as ctx, hermes(ctx) as h:
            on_loop(body, h)

    return run


def _replay(name: str) -> Script:
    def run(stage: Stage) -> None:
        stage.start()
        with plugin() as ctx, ScriptedProvider() as provider:
            on_loop(lambda: Replay(ctx, provider).run(name))

    return run


def _no_bytes_call(h: FakeHermes, turn: Turn, api_mode: str = "chat_completions") -> None:
    """A model frame whose transport this process never writes: the frame runs and returns, and
    Hermes reports usage."""
    turn.calls += 1
    ids = {**h._call_ids(turn, f"{turn.turn_id}:api:{turn.calls}"), "api_mode": api_mode}
    terminal = lambda request: SimpleNamespace(tool_calls=())  # noqa: E731
    h.ctx.run_execution_chain("llm_execution", terminal, request={}, original_request={}, **ids)
    h._emit(
        "post_api_request",
        {**ids, "response_model": h.model, "finish_reason": "stop", "usage": _usage(Usage(500, 7))},
    )


def _tool_ids(call: Call, name: str) -> dict[str, Any]:
    turn = call.turn
    return {
        "telemetry_schema_version": OBSERVER_V1,
        "session_id": turn.session_id,
        "task_id": turn.task_id,
        "turn_id": turn.turn_id,
        "api_request_id": call.api_request_id,
        "tool_call_id": call.tool_calls[0][0],
        "tool_name": name,
        "conversation_id": turn.session_id,
    }


def _on_worker(fn: Callable[[], Any]) -> concurrent.futures.Future:
    """A tool worker, its context copied at submit as Hermes's executor does."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(contextvars.copy_context().run, fn)
    pool.shutdown(wait=False)
    return future


def _no_bytes_mode(h: FakeHermes) -> None:
    turn = h.turn()
    _no_bytes_call(h, turn)
    h.end(turn)


def _bedrock_shape(h: FakeHermes) -> None:
    """Hermes's Converse path reaches the provider through boto3; the fake has no Converse wire,
    so the frame runs with no bytes, as an unmeasured capture would leave it."""
    turn = h.turn()
    _no_bytes_call(h, turn, api_mode="bedrock_converse")
    h.end(turn)


def _codex_app_server(h: FakeHermes) -> None:
    """Turn hooks only: the model calls ran in the codex subprocess, so no frame ran here."""
    h.end(h.turn())


def _orphan_tool(h: FakeHermes) -> None:
    """A tool frame with no turn in this process (Hermes's MCP child under codex app-server)."""
    turn = Turn("s-orphan", "task-orphan", "s-orphan:task-orphan:0")
    h.tool(
        Call(turn, "s-orphan:task-orphan:0:api:1", (("call_orphan", "read_file"),), 200),
        "read_file",
    )


def _aux_in_loop(h: FakeHermes) -> None:
    turn = h.turn()
    h.model_call(turn)
    h.aux(turn, from_loop=True, aux_task="compression")
    h.model_call(turn)
    h.end(turn)


def _invented_tool(h: FakeHermes) -> None:
    turn = h.turn()
    call = h.model_call(turn, (Reply.OK,), tool_calls=["lab_no_such_tool", "read_file"])
    h.tool(call, "read_file")
    h.model_call(turn)
    h.end(turn)


def _abandoned_tool(h: FakeHermes) -> None:
    """Hermes ends the turn at a tool timeout while the worker still runs; the worker's report
    comes after the end."""
    turn = h.turn()
    call = h.model_call(turn, (Reply.OK,), tool_calls=["slow_tool"])
    entered, gate = threading.Event(), threading.Event()

    def body() -> None:
        entered.set()
        gate.wait(10)

    job = _on_worker(
        lambda: h._tool_job(call, call.tool_calls[0][0], "slow_tool", "ok", None, body)
    )
    assert entered.wait(10)
    h.end(turn, completed=False, interrupted=True, reason="tool_timeout")
    gate.set()
    job.result(10)


def _mcp_sampling(h: FakeHermes) -> None:
    """An MCP server asks Hermes for a model call mid-tool. The request leaves from the MCP
    client's loop thread, which copied no context of the turn."""
    turn = h.turn()
    call = h.model_call(turn, (Reply.OK,), tool_calls=["mcp__lab__ask"])
    ids = _tool_ids(call, "mcp__lab__ask")

    def terminal(args: Any) -> str:
        on_loop(h.provider.post, Reply.OK, name="mcp-client-loop")
        return "asked"

    def job() -> None:
        h.ctx.run_execution_chain("tool_execution", terminal, args={}, original_args={}, **ids)
        h.ctx.invoke_hook("post_tool_call", **ids, args={}, result="asked", duration_ms=1,
                          status="ok", error_type=None, error_message=None)  # fmt: skip

    _on_worker(job).result(10)
    h.model_call(turn)
    h.end(turn)


def _fallback_recovered(h: FakeHermes) -> None:
    """Retries spent, then Hermes's fallback provider answered and the turn went on."""
    turn = h.turn()
    h.fail(turn)
    h.model_call(turn)
    h.end(turn)


def _two_gateway_sessions(stage: Stage) -> None:
    stage.start()
    with plugin() as ctx, hermes(ctx) as h:
        both_live = threading.Barrier(2, timeout=10)

        def job() -> None:
            turn = h.turn()
            h.model_call(turn)
            both_live.wait()
            h.model_call(turn)
            h.end(turn)

        threads = [
            threading.Thread(target=contextvars.Context().run, args=(h.gateway_job, job, n))
            for n in (1, 2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)


def _gateway_exit(stage: Stage) -> None:
    """The turn's spans must leave before a gateway that exits right after the turn: the test
    flushes nothing and reads what the wardex thread already sent."""
    transport = stage.start()
    with plugin() as ctx, hermes(ctx) as h:
        on_loop(lambda: h.end(_with_call(h, h.turn())))
        wait_for(lambda: any(n.name == TURN for n in received(transport)))
        stage.early = received(transport)


def _with_call(h: FakeHermes, turn: Turn) -> Turn:
    h.model_call(turn)
    return turn


def _two_profiles(stage: Stage) -> None:
    stage.start()
    for profile in ("work", "home"):
        with plugin(stage.home / profile) as ctx, hermes(ctx) as h:
            on_loop(lambda h=h: h.end(_with_call(h, h.turn())))


def _host_child(stage: Stage) -> None:
    stage.monkeypatch.setenv("HERMES_PLUGIN_HOST_PROCESS", "1")
    stage.start()
    with plugin():
        pass


def _finalize_open_turn(h: FakeHermes) -> None:
    """Shutdown mid-turn: `on_session_finalize`, then nothing (Hermes exits)."""
    turn = h.turn()
    h.model_call(turn)
    h.finalize(turn.session_id)


def _end_user(h: FakeHermes) -> None:
    """A gateway turn: Hermes's hooks name the user who sent the message."""
    turn = h.turn()
    h._emit("pre_llm_call", {**h._turn_ids(turn), "sender_id": SENDER, "is_first_turn": False})
    h.model_call(turn)
    h.end(turn)


SCRIPTS: dict[str, Script] = {
    "no_bytes_mode": _looped(_no_bytes_mode),
    "bedrock_shape": _looped(_bedrock_shape),
    "codex_app_server": _looped(_codex_app_server),
    "orphan_tool": _looped(_orphan_tool),
    "aux_in_loop": _looped(_aux_in_loop),
    "invented_tool": _looped(_invented_tool),
    "abandoned_tool": _looped(_abandoned_tool),
    "mcp_sampling": _looped(_mcp_sampling),
    "fallback_recovered": _looped(_fallback_recovered),
    "two_gateway_sessions": _two_gateway_sessions,
    "gateway_exit": _gateway_exit,
    "two_profiles": _two_profiles,
    "host_child": _host_child,
    "finalize_open_turn": _looped(_finalize_open_turn),
    "end_user": _looped(_end_user),
}

AGENT = "_hermes_agent.py"
FRAMES = "_hermes_agent_frames.py"
ATTACH = "_hermes_agent_attach.py"
TURNS = "_hermes_agent_turns.py"
EVENTS = "_hermes_agent_events.py"
CORE_HOLD = "../_assembly/_usage_hold.py"

# -- the registry --------------------------------------------------------------------------------

_done = {PREFIX + "close_cause": "end_event"}
_aborted = (L.SESSION_ABORTED,)

SCOPE_ROWS: tuple[ScopeRow, ...] = (
    ScopeRow("turn-normal", "Turn", "fixture:one_tool", (f"{AGENT}::_open", f"{AGENT}::_close"),
             (Traces(1),
              Span(TURN, parent="", source="trace_root", confidence=1.0, status="OK",
                   attrs={**_done, PREFIX + "exit_reason": "text_response(finish_reason=stop)",
                          "gen_ai.agent.name": "hermes"}))),
    ScopeRow("turn-failed", "Turn", "fixture:failed_turn", (f"{TURNS}::CLOSE_RULES",),
             (Traces(1),
              Span(TURN, parent="", status="UNSET", markers=_aborted,
                   attrs={PREFIX + "close_cause": "retries_spent",
                          PREFIX + "error_reason": "server_error"}))),
    ScopeRow("turn-concurrent", "Turn", "script:two_gateway_sessions", (f"{FRAMES}::_ensure_lane",),
             (Traces(2), Span(TURN, parent="", status="OK", n=2), Count(CHAT, 4),
              Span(CHAT, parent=TURN, source="unit_active", confidence=1.0, n=4)),
             unmeasured="two concurrent gateway sessions"),
    ScopeRow("turn-fallback", "Turn", "script:fallback_recovered",
             (f"{AGENT}::_continue", f"{TURNS}::TRANSITIONS"),
             (Span(TURN, status="UNSET", markers=_aborted,
                   attrs={PREFIX + "close_cause": "retries_spent"}),
              Span(TURN, status="OK", no_markers=_aborted,
                   attrs={**_done, PREFIX + "generation": 1}),
              Linked(TURN, "resumed_from", TURN)),
             unmeasured="fallback recovery after a retries_spent close"),
    ScopeRow("attempts-retry", "Model call attempt, retries included", "fixture:retry",
             (f"{FRAMES}::_open_call",),
             (Count(CHAT, 1), Span(CHAT, parent=TURN, source="unit_active", confidence=1.0),
              Span(WIRE, parent=CHAT, source="contextvar", status="ERROR", error_type="503"),
              Span(WIRE, parent=CHAT, source="contextvar", status="OK"),
              UsageOnce(CHAT))),
    ScopeRow("attempts-failed", "Model call attempt, retries included", "fixture:failed_turn",
             (f"{FRAMES}::_exit",),
             (Span(CHAT, parent=TURN, status="ERROR", error_type="InternalServerError"),
              Span(WIRE, parent=CHAT, status="ERROR", error_type="500", n=2))),
    ScopeRow("bedrock", "Bedrock Converse", "script:bedrock_shape",
             (f"{FRAMES}::_open_call", f"{CORE_HOLD}::settle_usage"),
             (Span(CHAT, parent=TURN, markers=(L.NO_WIRE_EVIDENCE,),
                   attrs={PREFIX + "api_mode": "bedrock_converse"}),
              UsageOnce(CHAT)),
             unmeasured="Bedrock Converse wire capture"),
    ScopeRow("no-bytes", "API modes with no bytes in this process", "script:no_bytes_mode",
             (f"{FRAMES}::_finish", f"{CORE_HOLD}::settle_usage"),
             (Span(CHAT, parent=TURN, markers=(L.NO_WIRE_EVIDENCE,)), Count(WIRE, 0),
              UsageOnce(CHAT))),
    ScopeRow("codex-app-server", "codex app-server mode", "script:codex_app_server",
             (f"{TURNS}::OBSERVATIONS",),
             (Span(TURN, parent="", status="OK", markers=(L.SUBPROCESS_MODEL_CALLS_UNOBSERVED,)),
              Count("chat ", 0, prefix=True)),
             unmeasured="codex app-server turn hooks"),
    ScopeRow("codex-mcp-child-tool", "codex app-server mode", "script:orphan_tool",
             (f"{FRAMES}::_orphan_frame",),
             (Span("execute_tool read_file", parent="", source="unresolved", confidence=0.0,
                   status="UNSET", markers=(L.PARENT_UNRESOLVED,)),)),
    ScopeRow("aux-title", "Auxiliary model calls (title, compression)", "fixture:auxiliary_title",
             (f"{FRAMES}::_aux_begin", f"{FRAMES}::_aux_finish"),
             (Traces(2),
              Span("chat lab-aux-title", parent=TURN, markers=(L.POSSIBLE_DUPLICATE_CHAT,),
                   attrs={PREFIX + "aux_task": "title_generation"}),
              Span(WIRE, parent="", source="unresolved", confidence=0.0,
                   markers=(L.PARENT_UNRESOLVED,)))),
    ScopeRow("aux-in-loop", "Auxiliary model calls (title, compression)", "script:aux_in_loop",
             (f"{FRAMES}::_aux_begin",),
             (Traces(1), Count("chat lab-aux", 0, prefix=True),
              Span(WIRE, parent=TURN, source="contextvar", confidence=1.0)),
             unmeasured="an auxiliary call from the loop thread"),
    ScopeRow("tools-sequential", "Tool calls, sequential and parallel", "fixture:inline_seq",
             (f"{FRAMES}::_open_tool",),
             (Span("execute_tool memory", parent=TURN, source="unit_active", confidence=1.0),
              Linked("execute_tool todo_list", "triggered_by", CHAT))),
    ScopeRow("tools-parallel", "Tool calls, sequential and parallel", "fixture:parallel_tools",
             (f"{FRAMES}::_open_tool", f"{FRAMES}::_close_frame"),
             (Span("execute_tool read_file", parent=TURN, source="unit_active", confidence=1.0,
                   status="OK", no_markers=(L.CHILD_SPAN_UNCLOSED,), n=3),)),
    ScopeRow("tools-invented", "Tool calls, sequential and parallel", "script:invented_tool",
             (f"{TURNS}::OBSERVATIONS",),
             (Span(TURN, markers=(L.TOOL_CALL_NOT_DISPATCHED,),
                   attrs={PREFIX + "undispatched_tool_calls": "lab_no_such_tool"}),
              Count("execute_tool lab_no_such_tool", 0), Count("execute_tool read_file", 1))),
    ScopeRow("subagents-joined", "Subagents (delegate_task)", "fixture:two_subagents_joined",
             (f"{AGENT}::_open", f"{AGENT}::_tie"),
             (Traces(1),
              Span(TURN, parent="execute_tool delegate_task", source="unit_active",
                   confidence=1.0, attrs={"wardex.agent.type": "sub_agent"}, n=2),
              Linked(TURN, "triggered_by", TURN))),
    ScopeRow("subagents-background", "Async delegation (library default)",
             "fixture:two_subagents", (f"{AGENT}::_open",),
             (Traces(3),
              Span(TURN, parent="", markers=(L.CORRELATION_CONFLICT,), n=2),
              Linked(TURN, "triggered_by", "execute_tool delegate_task"))),
    ScopeRow("mcp-one-span", "MCP tools", "fixture:mcp_seq",
             (f"{AGENT}::_tool_call_of",),
             (Count("execute_tool mcp__labecho__echo", 1), Count("execute_tool echo", 0))),
    ScopeRow("mcp-parallel", "MCP tools", "fixture:parallel_mcp", (f"{AGENT}::_tool_call_of",),
             (Count("execute_tool mcp__labecho__echo", 1), Count("execute_tool echo", 0))),
    ScopeRow("mcp-sampling", "MCP tools", "script:mcp_sampling", (f"{FRAMES}::_open_tool",),
             (Traces(2), Span(WIRE, parent="", source="trace_root", confidence=1.0, n=1)),
             unmeasured="MCP sampling"),
    ScopeRow("interrupted", "Interrupted turn; abandoned tool worker", "fixture:interrupted",
             (f"{TURNS}::CLOSE_RULES", f"{FRAMES}::_exit"),
             (Span(TURN, status="UNSET", markers=_aborted, no_markers=(L.UNIT_INTERRUPTED,),
                   attrs={PREFIX + "exit_reason": "interrupted_during_api_call"}),
              Span(CHAT, parent=TURN, status="ERROR", error_type="InterruptedError"))),
    ScopeRow("abandoned-tool", "Interrupted turn; abandoned tool worker", "script:abandoned_tool",
             (f"{AGENT}::_close", f"{AGENT}::_ignore"),
             (Span(TURN, status="UNSET", markers=_aborted,
                   attrs={PREFIX + "exit_reason": "tool_timeout"}),
              Span("execute_tool slow_tool", parent=TURN, status="UNSET",
                   markers=(L.CHILD_SPAN_UNCLOSED,), n=1)),
             unmeasured="tool timeout with an abandoned worker"),
    ScopeRow("profiles", "Several profiles in one process", "script:two_profiles",
             (f"{ATTACH}::attach",),
             (Traces(2), Span(TURN, parent="", status="OK", n=2),
              Never("profile", lambda k, v: "profile" in k or "/home" in str(v)))),
    ScopeRow("cli", "Out of process", "fixture:one_tool-cli", (f"{ATTACH}::_Exporter",),
             (Traces(1), Span(TURN, attrs={PREFIX + "platform": "cli"}),
              Span("execute_tool read_file", parent=TURN))),
    ScopeRow("gateway-exit", "Out of process", "script:gateway_exit",
             (f"{ATTACH}::_Exporter", f"{ATTACH}::kick"),
             (Early(TURN), Early(CHAT)),
             unmeasured="a gateway that exits within milliseconds of a turn close"),
    ScopeRow("host-isolation", "plugins.isolation: host", "script:host_child",
             (f"{ATTACH}::attach", f"{AGENT}::install"),
             (Line("adapters.hermes_agent.host_isolation"), Traces(0),
              Status(AdapterState.UNSUPPORTED, "plugin host process")),
             unmeasured="plugins.isolation: host"),
    ScopeRow("finalize", "Re-exec and exit", "script:finalize_open_turn",
             (f"{TURNS}::CLOSE_RULES",),
             (Span(TURN, status="UNSET", markers=_aborted,
                   attrs={PREFIX + "close_cause": "finalize"}),
              Span(CHAT, parent=TURN, status="OK"))),
    ScopeRow("end-user", "End-user identity", "script:end_user", (f"{EVENTS}::CONTRACT",),
             (Span(TURN, status="OK"),
              Never("end-user identity",
                    lambda k, v: k.startswith(("enduser.", "user.")) or SENDER in str(v)))),
)  # fmt: skip


# -- checking an observation ---------------------------------------------------------------------


@dataclass(frozen=True)
class Observed:
    nodes: tuple[SpanNode, ...]
    transport: RecordingTransport
    reasons: tuple[str, ...]
    status: AdapterStatus
    early: tuple[SpanNode, ...]


def _fits(node: SpanNode, e: Span, nodes: tuple[SpanNode, ...]) -> bool:
    above = parent_of(nodes, node)
    checks = (
        e.parent is None or (above.name if above is not None else "") == e.parent,
        e.source is None or (node.strategy is not None and node.strategy.name.lower() == e.source),
        e.confidence is None or node.confidence == e.confidence,
        e.status is None or (node.status is not None and node.status.name == e.status),
        e.error_type is None or node.attributes.get("error.type") == e.error_type,
        set(e.markers) <= set(node.limitations),
        not set(e.no_markers) & set(node.limitations),
        all(node.attributes.get(k) == v for k, v in e.attrs.items()),
    )
    return all(checks)


def _check(e: Expect, seen: Observed) -> None:
    nodes = seen.nodes
    if isinstance(e, Span):
        hits = [n for n in nodes if n.name == e.name and _fits(n, e, nodes)]
        assert hits if e.n is None else len(hits) == e.n, e
    elif isinstance(e, Count):
        named = [n for n in nodes if (n.name.startswith(e.name) if e.prefix else n.name == e.name)]
        assert len(named) == e.n, e
    elif isinstance(e, Linked):
        targets = [
            (reason.name.lower(), name)
            for n in nodes
            if n.name == e.span
            for reason, name in links(seen.transport, n, nodes)
        ]
        assert (e.reason, e.target) in targets, (e, targets)
    elif isinstance(e, Traces):
        assert len({n.trace_id for n in nodes}) == e.n, e
    elif isinstance(e, UsageOnce):
        for call in [n for n in nodes if n.name == e.call]:
            typed = [
                n for n in _subtree(call, nodes) if "gen_ai.usage.input_tokens" in n.attributes
            ]
            assert len(typed) == 1, (e, [n.name for n in typed])
    elif isinstance(e, Line):
        assert e.key in seen.reasons, (e, seen.reasons)
    elif isinstance(e, Status):
        assert seen.status.state is e.state and e.detail in seen.status.detail, seen.status
    elif isinstance(e, Never):
        found = [(k, v) for n in nodes for k, v in n.attributes.items() if e.test(k, v)]
        assert not found, (e.what, found)
    else:
        assert any(n.name == e.name for n in seen.early), (e, [n.name for n in seen.early])


def _subtree(top: SpanNode, nodes: tuple[SpanNode, ...]) -> list[SpanNode]:
    out, frontier = [top], [top]
    while frontier:
        frontier = [n for n in nodes for p in frontier if n.parent_id == p.span_id]
        out += frontier
    return out


# -- the tests -----------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _installed(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield
    forget_attachments()


def test_every_scope_surface_has_an_observation() -> None:
    assert len(SURFACES) == 16
    assert {row.surface for row in SCOPE_ROWS} == set(SURFACES)


def test_every_script_is_used() -> None:
    used = {
        row.scenario.split(":", 1)[1] for row in SCOPE_ROWS if row.scenario.startswith("script:")
    }
    assert used == set(SCRIPTS)


def test_every_unmeasured_item_has_exactly_one_row_and_it_is_driven_by_the_fake() -> None:
    rows = {item: [row for row in SCOPE_ROWS if row.unmeasured == item] for item in NOT_MEASURED}
    assert {item: len(found) for item, found in rows.items()} == dict.fromkeys(NOT_MEASURED, 1)
    assert {row.unmeasured for row in SCOPE_ROWS} <= {"", *NOT_MEASURED}
    assert all(found[0].scenario.startswith("script:") for found in rows.values())


def test_every_row_names_code_that_exists() -> None:
    adapters = Path(__file__).resolve().parents[1] / "src" / "wardex_sdk" / "_adapters"
    for row in SCOPE_ROWS:
        for site in row.produced_by:
            module, name = site.split("::")
            source = (adapters / module).resolve().read_text(encoding="utf-8")
            assert re.search(rf"\b{re.escape(name)}\b", source), (row.id, site)


@pytest.mark.parametrize("row", SCOPE_ROWS, ids=[row.id for row in SCOPE_ROWS])
def test_scope_row_is_observable(row: ScopeRow, monkeypatch, tmp_path, caplog) -> None:
    """Run the row's scenario through the fake plugin context, then check each expectation
    against exported spans, WARNING lines and `wardex.diagnostics()`."""
    kind, name = row.scenario.split(":", 1)
    script = _replay(name) if kind == "fixture" else SCRIPTS[name]
    stage = Stage(monkeypatch, tmp_path)
    reset_reports_for_test()  # a line is said once per process; this row must hear its own
    with caplog.at_level(logging.WARNING, logger="wardex_sdk"):
        script(stage)
    assert stage.transport is not None, "the script never started wardex"
    seen = Observed(
        nodes=exported(stage.transport),
        transport=stage.transport,
        reasons=tuple(getattr(r, "wardex_reason", "") for r in caplog.records),
        status=status(),
        early=stage.early,
    )
    for expect in row.observe:
        try:
            _check(expect, seen)
        except AssertionError as exc:
            raise AssertionError(f"{row.id}: {exc}\n{outline(seen.nodes)}") from None
