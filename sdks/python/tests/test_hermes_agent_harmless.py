"""Hermes is never disturbed: whatever wardex does or fails to do, Hermes gets what it would
have got with no plugin, and so do the other plugins beside it.

What Hermes can observe of a plugin is checked on every chain and hook: the result object or the
exception object each middleware frame produced, passed back by identity; the frame's payload
reaching the next callback as Hermes handed it, once; every hook returning None; no callback
exception reaching Hermes's log. On top of that, nothing on stderr, no file in the profile home,
`os.environ` as it was, and OpenTelemetry's global tracer provider untouched.

`_everything` is one Hermes process's worth of events, built so that every action of the turn
lifecycle runs at least once (`test_the_workload_reaches_every_effect` holds it to that). Each
injection test makes one effect, or one step around them, raise every time it runs.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

import wardex_sdk
from _hermes_fake import FakePluginContext, Reply, ScriptedProvider
from _hermes_scripted import FakeHermes, Turn
from _hermes_support import (
    exported,
    forget_attachments,
    hermes_installed,
    on_loop,
    start_wardex,
)
from wardex_sdk._adapters import _hermes_agent_attach as attach_module
from wardex_sdk._adapters._hermes_agent_turns import Action
from wardex_sdk._adapters._registry import get_registry
from wardex_sdk._assembly import Limitation
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk.hermes_agent import register

# The package's registry names a factory `_hermes_agent`, which shadows the module attribute.
adapter_module = importlib.import_module("wardex_sdk._adapters._hermes_agent")

try:  # the assertion runs whenever OpenTelemetry is importable; the SDK does not need it
    from opentelemetry import trace as otel_trace
except ImportError:  # pragma: no cover - the dev environment has it
    otel_trace = None  # type: ignore[assignment]


class _Injected(RuntimeError):
    """Raised by an injected effect; Hermes must never see one."""


class _HostError(RuntimeError):
    """Hermes's own failure inside a frame; Hermes must get this very object back."""


@dataclass
class _Checked(FakePluginContext):
    """A plugin context that checks, on every middleware chain, what Hermes gets back."""

    frames: int = 0
    mismatches: list[str] = field(default_factory=list)

    def run_execution_chain(self, kind: str, terminal: Callable[[Any], Any], **kwargs: Any) -> Any:
        given = kwargs["request" if "request" in kwargs else "args"]
        seen: list[Any] = []
        outcome: dict[str, Any] = {}

        def watched(payload: Any) -> Any:
            seen.append(payload)
            try:
                outcome["result"] = terminal(payload)
            except BaseException as exc:
                outcome["raised"] = exc
                raise
            return outcome["result"]

        self.frames += 1
        try:
            result = super().run_execution_chain(kind, watched, **kwargs)
        except BaseException as exc:
            if exc is not outcome.get("raised"):
                self.mismatches.append(f"{kind}: raised {exc!r}, Hermes's frame raised {outcome}")
            raise
        finally:
            if len(seen) != 1 or seen[0] is not given:
                self.mismatches.append(f"{kind}: the frame ran {len(seen)} times, or on a copy")
        if "result" not in outcome or result is not outcome["result"]:
            self.mismatches.append(f"{kind}: returned {result!r}, Hermes's frame gave {outcome}")
        return result


@contextmanager
def _context(home: Path | None = None) -> Iterator[_Checked]:
    ctx = _Checked(home=home)
    register(ctx)
    try:
        yield ctx
    finally:
        ctx.unload()


def _assert_undisturbed(ctx: _Checked) -> None:
    assert ctx.frames > 0
    assert ctx.mismatches == []
    assert ctx.failures == [], "a callback raised into Hermes"
    assert {r.value for r in ctx.returns} <= {None}, "a hook's result is Hermes's to use"


# -- one process's worth of events ---------------------------------------------------------------


def _late_tool_report(h: FakeHermes, turn: Turn, call_id: str) -> None:
    h._emit(
        "post_tool_call",
        {
            **h._turn_ids(turn),
            "api_request_id": f"{turn.turn_id}:api:1",
            "tool_call_id": call_id,
            "tool_name": "read_file",
            "args": {},
            "result": "late",
            "duration_ms": 1,
            "status": "ok",
            "error_type": None,
            "error_message": None,
        },
    )


def _everything(h: FakeHermes) -> None:
    # A live turn: its title call, a tool, a joined subagent, a failed attempt with retries left.
    first = h.turn(title=True)
    h.aux(first)
    call = h.model_call(first, (Reply.OK,), tool_calls=["read_file", "delegate_task"])
    h.tool(call, "read_file")
    h.delegate(call, 1)
    h._emit("pre_llm_call", {**h._turn_ids(first), "is_first_turn": False})
    h._emit(
        "api_request_error",
        {
            **h._call_ids(first, f"{first.turn_id}:api:9"),
            "retry_count": 0,
            "max_retries": 3,
            "retryable": True,
            "reason": "server_error",
            "status_code": "503",
            "error_type": "InternalServerError",
        },
    )
    h.model_call(first)
    h.end(first)
    # After the end: a late report, an auxiliary call, and a tool frame the turn no longer owns.
    _late_tool_report(h, first, call.tool_calls[0][0])
    h.aux(first, from_loop=True)
    h.tool(call, "read_file", tool_call_id="call_after_the_end")
    # A presumed close that the turn proves wrong, and one that Hermes ends.
    second = h.turn()
    h.fail(second)
    h.model_call(second)
    h.end(second)
    third = h.turn()
    h.fail(third)
    h.end(third, completed=False, failed=True, reason="failed")
    # A turn whose start this process never saw, and an event for a turn it never knew.
    ghost = Turn("s-ghost", "task-ghost", "s-ghost:task-ghost:0")
    h.model_call(ghost)
    h.end(ghost)
    h._emit(
        "post_api_request",
        {**h._call_ids(Turn("s-x", "t-x", "s-x:t-x:0"), "s-x:t-x:0:api:1"), "usage": {}},
    )
    # The signal path closes a live turn under Hermes; the turn goes on.
    fourth = h.turn()
    h.model_call(fourth)
    get_registry().close_units_all(marker=Limitation.UNIT_INTERRUPTED)
    h.model_call(fourth)
    h.end(fourth)
    # A stream cut by an interrupt, then shutdown with a turn open.
    fifth = h.turn()
    h.interrupt(fifth)
    sixth = h.turn()
    h.model_call(sixth)
    h.finalize(sixth.session_id)


def _run(ctx: _Checked) -> None:
    with ScriptedProvider() as provider:
        on_loop(_everything, FakeHermes(ctx, provider))
        ctx.drain_streams()


@pytest.fixture(autouse=True)
def _installed(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield
    forget_attachments()


def _adapter() -> Any:
    adapter = attach_module._ROUTE
    assert adapter is not None, "wardex is started with the Hermes adapter installed"
    return adapter


def test_the_workload_reaches_every_effect(monkeypatch):
    """Without this, an injection test whose effect never ran would pass vacuously."""
    start_wardex()
    adapter = _adapter()
    reached: set[Action] = set()
    for action, effect in list(adapter._effects.items()):

        def spy(*args: Any, _action: Action = action, _effect: Any = effect) -> Any:
            reached.add(_action)
            return _effect(*args)

        monkeypatch.setitem(adapter._effects, action, spy)
    with _context() as ctx:
        _run(ctx)
    assert reached == set(Action)
    _assert_undisturbed(ctx)


#: Steps around the effects table that a raise could come from, as attributes of the adapter.
_STEPS = ("on_hook", "on_frame", "_parse", "_dispatch", "_exit")


@pytest.mark.parametrize("target", [*Action, *_STEPS], ids=lambda t: getattr(t, "value", t))
def test_a_raise_in_any_effect_leaves_hermes_and_its_frames_exactly_as_they_were(
    target, monkeypatch
):
    start_wardex()
    adapter = _adapter()
    fired: list[Any] = []

    def raiser(*args: Any, **kwargs: Any) -> Any:
        fired.append(target)
        raise _Injected(f"injected at {target}")

    if isinstance(target, Action):
        monkeypatch.setitem(adapter._effects, target, raiser)
    else:
        monkeypatch.setattr(adapter, target, raiser)
    with _context() as ctx:
        _run(ctx)
    assert fired, f"the workload never reached {target}"
    _assert_undisturbed(ctx)


def test_a_frame_that_raises_gets_hermes_its_own_exception_object_whatever_wardex_did(
    monkeypatch,
):
    """The model call's own failure, raised from the terminal, with the frame effect failing too."""
    start_wardex()
    monkeypatch.setitem(_adapter()._effects, Action.FRAME, _raise_injected)
    error = _HostError("the provider's own")
    with _context() as ctx, ScriptedProvider() as provider:
        h = FakeHermes(ctx, provider)

        def turn() -> BaseException | None:
            t = h.turn()
            ids = h._call_ids(t, f"{t.turn_id}:api:1")
            try:
                ctx.run_execution_chain(
                    "llm_execution", _raising(error), request={}, original_request={}, **ids
                )
            except _HostError as exc:
                return exc
            return None

        assert on_loop(turn) is error
    _assert_undisturbed(ctx)


def test_wardex_failing_while_the_call_fails_still_gets_hermes_its_own_exception(monkeypatch):
    """wardex failing on the very path that carries the call's own failure out to Hermes: the
    trampoline hands Hermes its exception object, and the call is not run a second time."""
    start_wardex()
    monkeypatch.setattr(adapter_module, "_downstream_name", _raise_injected)
    error = _HostError("the provider's own")
    with _context() as ctx, ScriptedProvider() as provider:
        h = FakeHermes(ctx, provider)
        runs: list[None] = []

        def terminal(payload: Any) -> Any:
            runs.append(None)
            raise error

        def turn() -> BaseException | None:
            t = h.turn()
            ids = h._call_ids(t, f"{t.turn_id}:api:1")
            try:
                ctx.run_execution_chain(
                    "llm_execution", terminal, request={}, original_request={}, **ids
                )
            except _HostError as exc:
                return exc
            return None

        assert on_loop(turn) is error
        assert len(runs) == 1
    _assert_undisturbed(ctx)


def _raise_injected(*args: Any) -> Any:
    raise _Injected("injected")


def _raising(error: BaseException) -> Callable[[Any], Any]:
    def terminal(payload: Any) -> Any:
        raise error

    return terminal


# -- beside other plugins ------------------------------------------------------------------------


def test_other_plugins_keep_their_order_and_their_results(monkeypatch):
    """One plugin registered before wardex and one after, on the same hook and middleware
    tables. Hermes calls them in registration order, nests the chain that way, and uses their
    results; wardex is invisible in both: the outer plugin's `next_call()` returns the inner
    plugin's result object, which wardex passed through between them."""
    start_wardex()
    order: list[str] = []
    got: dict[str, Any] = {}
    before = FakePluginContext(plugin_id="before")
    ctx = FakePluginContext(hooks=before.hooks, middleware=before.middleware)
    after = FakePluginContext(plugin_id="after", hooks=before.hooks, middleware=before.middleware)

    def context_hook(name: str) -> Callable[..., Any]:
        def pre_llm_call(**payload: Any) -> dict[str, str]:
            order.append(f"{name} hook")
            return {"context": f"from {name}"}

        pre_llm_call.__name__ = f"{name}_pre_llm_call"
        return pre_llm_call

    def wrapping(name: str) -> Callable[..., Any]:
        def llm_execution(next_call: Callable[[], Any], **payload: Any) -> Any:
            order.append(f"{name} enters")
            got[f"{name} got"] = result = next_call()
            order.append(f"{name} leaves")
            got[f"{name} gave"] = gave = (name, result)
            return gave

        llm_execution.__name__ = f"{name}_llm_execution"
        return llm_execution

    before.register_hook("pre_llm_call", context_hook("before"))
    before.register_middleware("llm_execution", wrapping("before"))
    register(ctx)
    after.register_hook("pre_llm_call", context_hook("after"))
    after.register_middleware("llm_execution", wrapping("after"))
    try:
        with ScriptedProvider() as provider:
            h = FakeHermes(ctx, provider)

            def turn() -> tuple[list[Any], Any]:
                t = h.turn()
                answers = h._emit("pre_llm_call", {**h._turn_ids(t), "is_first_turn": False})
                ids = h._call_ids(t, f"{t.turn_id}:api:1")
                out = ctx.run_execution_chain(
                    "llm_execution", lambda p: "answer", request={}, original_request={}, **ids
                )
                h.end(t)
                return answers, out

            answers, out = on_loop(turn)
    finally:
        after.unload()
        ctx.unload()
        before.unload()
    assert answers == [{"context": "from before"}, {"context": "from after"}]
    assert out == ("before", ("after", "answer"))
    assert order[-4:] == ["before enters", "after enters", "after leaves", "before leaves"]
    assert got["before got"] is got["after gave"]
    assert before.failures == after.failures == ctx.failures == []
    assert {r.value for r in ctx.returns if r.callback.startswith("wardex_")} == {None}


# -- the process around Hermes -------------------------------------------------------------------


def test_a_recorded_process_leaves_stderr_the_home_the_environment_and_otel_alone(tmp_path, capfd):
    home = tmp_path / "profile-home"
    home.mkdir()
    environment = dict(os.environ)
    provider = otel_trace.get_tracer_provider() if otel_trace is not None else None
    transport = start_wardex()
    capfd.readouterr()
    with _context(home) as ctx, ScriptedProvider() as scripted:
        h = FakeHermes(ctx, scripted)

        def turns() -> None:
            first = h.turn(title=True)
            h.aux(first)
            call = h.model_call(first, tool_calls=["read_file", "delegate_task"])
            h.parallel_tools(call, ["read_file"])
            h.delegate(call, 2)
            h.model_call(first)
            h.end(first)

        on_loop(turns)
    assert [n.name for n in exported(transport) if n.name == "invoke_agent hermes"]
    wardex_sdk.close()
    assert capfd.readouterr().err == ""
    assert sorted(home.rglob("*")) == []
    assert dict(os.environ) == environment
    if otel_trace is not None:
        assert otel_trace.get_tracer_provider() is provider
    _assert_undisturbed(ctx)


def test_a_refused_configuration_is_said_once_and_leaves_the_callbacks_inert(monkeypatch, caplog):
    """`register` starting wardex from a configuration `init()` refuses: Hermes loads the plugin,
    runs its turns, and hears one line about it."""

    def init(*args: Any, **kwargs: Any) -> None:
        raise ValueError("WARDEX_SAMPLE_RATE must be between 0 and 1")

    monkeypatch.setattr(wardex_sdk, "init", init)
    reset_reports_for_test()  # a line is said once per process; this test must hear its own
    with caplog.at_level(logging.WARNING, logger="wardex_sdk"):
        with _context() as ctx:
            _run(ctx)
        with _context():
            pass
    said = [r for r in caplog.records if getattr(r, "wardex_reason", "") == _INIT_REFUSED]
    assert len(said) == 1
    assert "WARDEX_SAMPLE_RATE" in said[0].getMessage()
    assert attach_module._ROUTE is None, "no adapter: the callbacks route nowhere"
    _assert_undisturbed(ctx)


_INIT_REFUSED = "adapters.hermes_agent.init_refused"
