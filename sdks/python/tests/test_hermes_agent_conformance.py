"""The Hermes adapter, run against the shared adapter conformance suite, unchanged.

Everything here is the SUBJECT; the claims are in `wardex_sdk.testing.conformance`. The one seam
is `_hermes_agent_attach._ROUTE`: Hermes's callbacks are registered once per plugin context and
stay registered, so the install is the route they reach, patched from None to the adapter and
handed back by identity. Nothing of Hermes is patched.

The suite installs through a private registry with a recording client and no `wardex.init()`, so
no byte seam is on and the workload makes no HTTP request: each model frame's terminal returns at
once, and Hermes's report carries the usage. Where the wire CHAT would sit under a chat unit, the
terminal opens one stand-in child, as the pydantic-ai subject does. The first call asks for two
tools, run as one parallel batch on two workers that meet at a barrier, and each tool's terminal
opens a stand-in child too: two chains need two different answers to "which unit is current" at
the same moment, on two threads.

`usage_expected` is `"cache_tiers"`: with no wire request under a chat unit, the core promotes
Hermes's numbers into the typed block (marked `no_wire_evidence`), and the report carries a cache
read tier.

Hermes cannot be installed here, so the subject writes the distribution record the registry's
probe reads (`hermes_installed`) and registers the callbacks on a fake plugin context with no
`wardex.init()` of their own.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from _hermes_fake import FakePluginContext, Usage
from _hermes_scripted import Call, FakeHermes, Turn, _usage
from _hermes_support import forget_attachments, hermes_installed, on_loop
from wardex_sdk._adapters import _hermes_agent_attach as attach_module
from wardex_sdk._adapters._context import Placement
from wardex_sdk._adapters._hermes_agent import HermesAgentAdapter
from wardex_sdk._adapters._registry import AdapterRegistry
from wardex_sdk._assembly import PatchSet, SpanIntent, ToolAttributes, UnitKind, _bridge_latch
from wardex_sdk._enums import AdapterName, ToolExecutionType, ToolType
from wardex_sdk.testing import AdapterConformanceSuite, AdapterSubject, RecordingClient, StalledRun
from wardex_sdk.testing.harness import clean_state

ROOT = "invoke_agent hermes"

#: The tree the workload builds, as span-name chains from the root down.
_CHAINS = (
    (ROOT, "chat lab-ask", "execute_tool probe-ask"),
    (ROOT, "execute_tool tool_a", "execute_tool probe-tool_a"),
    (ROOT, "execute_tool tool_b", "execute_tool probe-tool_b"),
    (ROOT, "chat lab-final", "execute_tool probe-final"),
)


def _probe(live: Any, label: str) -> None:
    """A unit opened from inside a frame's terminal, where its child would be. Only when
    installed: the zero-point run opens nothing of wardex's own."""
    if live.ctx is None:
        return
    name = f"probe-{label}"

    def describe(h: Any) -> None:
        h.draft.set_tool(
            ToolAttributes(
                name=name, type=ToolType.FUNCTION, execution_type=ToolExecutionType.IN_PROCESS
            )
        )

    handle = live.ctx.open_run(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        subject=name,
        describe=describe,
    )
    handle.close()


def _model_call(
    h: FakeHermes, turn: Turn, model: str, body: Any, asked: tuple[str, ...] = ()
) -> Call:
    """One model frame with no bytes under it: the terminal runs `body` and returns, and Hermes
    reports usage with a cache tier."""
    turn.calls += 1
    ids = {**h._call_ids(turn, f"{turn.turn_id}:api:{turn.calls}"), "model": model}
    calls = tuple((f"call_{model}_{k}", name) for k, name in enumerate(asked))

    def terminal(request: Any) -> Any:
        body()
        return SimpleNamespace(tool_calls=calls)

    h.ctx.run_execution_chain("llm_execution", terminal, request={}, original_request={}, **ids)
    h._emit(
        "post_api_request",
        {
            **ids,
            "response_model": model,
            "finish_reason": "tool_calls" if calls else "stop",
            "assistant_tool_call_count": len(calls),
            "usage": _usage(Usage(1200, 12, cached_tokens=300)),
            "response": {
                "assistant_message": {
                    "tool_calls": [{"id": i, "name": n, "arguments": "{}"} for i, n in calls]
                },
            },
        },
    )
    return Call(turn, ids["api_request_id"], calls, 200)


def _batch(h: FakeHermes, call: Call, body: Any) -> None:
    """Hermes's parallel tool batch: one job per call on its own worker, each context copied at
    submit, the frames overlapping at a barrier."""
    barrier = threading.Barrier(len(call.tool_calls), timeout=10)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(call.tool_calls)) as pool:
        jobs = [
            pool.submit(
                contextvars.copy_context().run,
                h._tool_job,
                call,
                tcid,
                name,
                "ok",
                barrier,
                lambda name=name: body(name),
            )
            for tcid, name in call.tool_calls
        ]
        for job in jobs:
            job.result()


class _Host:
    """A fake Hermes plugin context with the adapter's callbacks on it, registered the way
    `register` does with wardex already started by somebody else."""

    def __init__(self) -> None:
        self.ctx = FakePluginContext()
        attach_module.attach(
            self.ctx, init=lambda: None, initialized=lambda: True, flush=lambda timeout: None
        )

    def hermes(self) -> FakeHermes:
        return FakeHermes(self.ctx, provider=None)  # type: ignore[arg-type]


def _workload(host: _Host) -> Any:
    def run(live: Any) -> Any:
        h = host.hermes()

        def turn() -> str:
            t = h.turn()
            asking = _model_call(h, t, "lab-ask", lambda: _probe(live, "ask"), ("tool_a", "tool_b"))
            _batch(h, asking, lambda name: _probe(live, name))
            _model_call(h, t, "lab-final", lambda: _probe(live, "final"))
            h.end(t)
            return "done"

        return on_loop(turn)

    return run


def _stall(host: _Host) -> Any:
    def stall(live: Any) -> StalledRun:
        """A turn parked inside a tool frame on a worker, its first call reported."""
        h = host.hermes()
        gate, entered = threading.Event(), threading.Event()
        result: dict[str, Any] = {}

        def wait() -> None:
            entered.set()
            gate.wait(10)

        def turn() -> None:
            t = h.turn()
            call = _model_call(h, t, "lab-stall", lambda: None, ("wait",))
            copied = contextvars.copy_context()
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                tcid = call.tool_calls[0][0]
                pool.submit(copied.run, h._tool_job, call, tcid, "wait", "ok", None, wait).result()
            _model_call(h, t, "lab-resumed", lambda: None)
            h.end(t)

        def drive() -> None:
            try:
                turn()
            except BaseException as exc:  # handed to the check through `resume`
                result["exc"] = exc

        thread = threading.Thread(target=contextvars.Context().run, args=(drive,), daemon=True)
        thread.start()
        assert entered.wait(10), "the tool never started; the stall never happened"

        def resume() -> Any:
            gate.set()
            thread.join(15)
            assert not thread.is_alive(), "the stalled turn never finished"
            assert "exc" not in result, result
            assert host.ctx.failures == [], "a callback raised into Hermes"
            return None

        return StalledRun(root=ROOT, resume=resume)

    return stall


@pytest.fixture
def host(tmp_path: Path):
    forget_attachments()
    with hermes_installed(tmp_path):
        yield _Host()
    forget_attachments()


@pytest.fixture
def subject(host: _Host) -> AdapterSubject:
    return AdapterSubject(
        name=AdapterName.HERMES_AGENT.value,
        module="wardex_sdk._adapters._hermes_agent",
        factory=HermesAgentAdapter,
        seams=lambda: {"route": attach_module._ROUTE},
        workload=_workload(host),
        chains=_CHAINS,
        stall=_stall(host),
        detect_package="hermes_cli",
        usage_expected="cache_tiers",
    )


@pytest.mark.parametrize("check", AdapterConformanceSuite.CHECKS)
def test_conformance(subject, check):
    AdapterConformanceSuite(subject).run(check)


def test_an_install_that_fails_at_the_route_leaves_no_tool_call_follow(host):
    """The suite proves the route is handed back; this proves the other thing the install
    registered is too. A follow left behind outlives its context: every later MCP request walks
    it. No span can show it, so the registered followers are read directly."""
    before = [owner for owner, _ in _bridge_latch._TOOL_RESOLVERS]

    def refuse(self: Any, target: Any, name: str, wrapper: Any) -> bool:
        raise RuntimeError("injected at the route patch")

    with clean_state(), pytest.MonkeyPatch.context() as mp:
        mp.setattr(PatchSet, "patch", refuse)
        registry = AdapterRegistry()
        registry.install(HermesAgentAdapter(), RecordingClient())
    assert not registry.is_installed(AdapterName.HERMES_AGENT.value)
    assert [owner for owner, _ in _bridge_latch._TOOL_RESOLVERS] == before
