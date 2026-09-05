"""The openai-agents adapter — the scenarios, measured against the wire.

Every test here drives REAL `Runner.run` / `run_sync` / `run_streamed` against
the stage-1 fake Responses server (`test_openai_agents_wire`), with wardex
fully initialised, so the tree read back is the one a host would see: the
adapter's `invoke_workflow` / `invoke_agent` / `handoff` / `execute_tool` /
`evaluate` spans AND the interceptor's `chat` spans, parented by context.

The conformance file next door holds the shared invariants. What lives here
is what goes beyond them: the three-turn tree with its call-id and response-id
joins, parallel tools at confidence 1.0, the handoff chain as siblings, the
guardrail and MaxTurns failure mappings, tracing off (INFO once, LLM spans
only), the processor-removed warning, surface probes, fault injection with
nothing reaching the host, and the fork reset.

The framework's default processor is never left registered: it POSTs the run
record to the real API. `agents_env` starts every test from an empty list.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import textwrap
from typing import Any

import pytest
from agents import Agent, RunConfig, Runner
from agents.tracing import (
    TracingProcessor,
    add_trace_processor,
    get_trace_provider,
    set_trace_processors,
    set_tracing_disabled,
)

import wardex_sdk as wardex
from test_openai_agents_wire import _agents, agents_env, fake_openai
from wardex_sdk import _hub
from wardex_sdk._adapters._openai_agents import (
    _PROCESSOR_REMOVED_NOTICE,
    _TRACING_DISABLED_NOTICE,
    OpenAIAgentsAdapter,
)
from wardex_sdk._assembly import Limitation, LinkReason, ParentSource, counters
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._enums import AdapterName, SpanKind, StatusCode
from wardex_sdk.testing import RecordingTransport, installed_adapter

pytestmark = pytest.mark.usefixtures("fresh_counters")

#: The stage-1 fixtures, re-exported into this module so pytest finds them.
_WIRE_FIXTURES = (fake_openai, agents_env)

_ENABLED = AdaptersConfig(enabled=(AdapterName.OPENAI_AGENTS,))


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


class _Records(logging.Handler):
    """Every record the `wardex_sdk` logger emits, whatever handlers a host
    (or pytest) attached: the logger does not propagate to the root, so
    `caplog` alone would see nothing."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def lines(self, level: int) -> list[str]:
        return [r.getMessage() for r in self.records if r.levelno == level]


@pytest.fixture(autouse=True)
def _fresh_framework_http_client(monkeypatch):
    """The framework shares ONE httpx client across providers for the life of
    the process (`agents.models.openai_provider._http_client`), and a client
    first used inside one event loop fails with a connection error from the
    next — `run_sync` after `asyncio.run` in this file, measured. Reset per
    test: the framework builds a fresh one on first use."""
    from agents.models import openai_provider

    monkeypatch.setattr(openai_provider, "_http_client", None)


@pytest.fixture
def wardex_log():
    logger = logging.getLogger("wardex_sdk")
    handler = _Records()
    logger.addHandler(handler)
    level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)


@pytest.fixture
def tracing_enabled():
    """The framework's manual switch, restored to `None` afterwards so the
    next test reads the environment the way a fresh process would."""
    provider = get_trace_provider()
    before = provider._manual_disabled
    yield
    provider._manual_disabled = before
    provider._refresh_disabled_flag()


def _init(**kwargs: Any) -> RecordingTransport:
    t = RecordingTransport()
    wardex.init(transport=t, adapters=_ENABLED, **kwargs)
    return t


def _spans() -> list[Any]:
    client = _hub.get_client()
    client._settle()
    return list(client._spans)


def _adapter_spans(spans: list[Any]) -> list[Any]:
    return [s for s in spans if s.kind is not SpanKind.CLIENT]


def _chat_spans(spans: list[Any]) -> list[Any]:
    return [s for s in spans if s.kind is SpanKind.CLIENT]


def _one(spans: list[Any], name: str) -> Any:
    found = [s for s in spans if s.name == name]
    assert len(found) == 1, f"expected one {name!r}, got {[s.name for s in spans]}"
    return found[0]


def _edge(span: Any) -> tuple[Any, float | None, tuple[Limitation, ...]]:
    corr, integ = span.correlation, span.capture_integrity
    return (
        corr.strategy if corr is not None else None,
        corr.confidence if corr is not None else None,
        tuple(integ.limitations) if integ is not None else (),
    )


def _extra(span: Any) -> dict[str, Any]:
    return dict(span.extra)


def _run(agent: Agent, **kwargs: Any) -> Any:
    return asyncio.run(Runner.run(agent, "hi", **kwargs))


def _print_tree(spans: list[Any]) -> str:
    """The tree as text, for the tracker's record: name, parent, agent, tool
    call id, response id, and the edge tuple. Roots first, children in
    start order."""
    by_id = {s.context.span_id: s for s in spans}
    lines: list[str] = []

    def bits(s: Any) -> str:
        out = []
        if s.agent is not None:
            out.append(f"agent={s.agent.name}")
            if s.agent.parent_agent:
                out.append(f"parent_agent={s.agent.parent_agent}")
        if s.tool is not None and s.tool.call_id:
            out.append(f"tool.call.id={s.tool.call_id}")
        if s.gen_ai is not None and s.gen_ai.response_id:
            out.append(f"gen_ai.response.id={s.gen_ai.response_id}")
        rid = _extra(s).get("wardex.openai_agents.response_id")
        if rid:
            out.append(f"response_id={rid}")
        if s.conversation is not None:
            out.append(f"conv={s.conversation.conversation_id}")
        if s.status is not StatusCode.OK:
            out.append(f"status={s.status.value} error_type={s.error_type}")
        strategy, confidence, markers = _edge(s)
        out.append(
            f"edge=({strategy.value if strategy else None}, {confidence}, "
            f"[{','.join(m.value for m in markers)}])"
        )
        return " ".join(out)

    def walk(s: Any, depth: int) -> None:
        lines.append(f"{'  ' * depth}{s.name}  {bits(s)}")
        kids = [c for c in spans if c.parent_span_id == s.context.span_id]
        for c in sorted(kids, key=lambda c: c.start_time_ns):
            walk(c, depth + 1)

    for s in sorted(spans, key=lambda s: s.start_time_ns):
        if s.parent_span_id not in by_id:
            walk(s, 0)
    return "\n".join(lines)


def _processors() -> tuple[Any, ...]:
    return get_trace_provider()._multi_processor._processors


class _HostProcessor(TracingProcessor):
    def on_trace_start(self, trace: Any) -> None:
        return None

    def on_trace_end(self, trace: Any) -> None:
        return None

    def on_span_start(self, span: Any) -> None:
        return None

    def on_span_end(self, span: Any) -> None:
        return None

    def shutdown(self, timeout: float | None = None) -> None:
        return None

    def force_flush(self) -> None:
        return None


# --------------------------------------------------------------------------
# tracing off: one INFO line, LLM spans only
# --------------------------------------------------------------------------


def test_tracing_disabled_says_so_once_and_ships_only_the_llm_calls(
    agents_env, wardex_log, tracing_enabled
):
    """The hook is silent when the framework's tracing is off. wardex says so
    ONCE at install, at INFO, with the recipe — and the wire still shows the
    three LLM calls, because the interceptor never needed the hook."""
    set_tracing_disabled(True)
    _init()
    try:
        assert _run(_agents()).final_output == "done"
        assert wardex_log.lines(logging.INFO).count(_TRACING_DISABLED_NOTICE) == 1
        spans = _spans()
        assert len(_chat_spans(spans)) == 3
        assert _adapter_spans(spans) == []
        assert counters.get("adapters.openai_agents.active.trace") == 0
        assert counters.get("adapters.openai_agents.tracing_disabled_at_install") == 1
    finally:
        wardex.close()


_PRECEDENCE_SCRIPT = textwrap.dedent(
    """
    import logging, sys
    import agents
    from wardex_sdk._adapters._openai_agents import _TRACING_DISABLED_NOTICE
    import wardex_sdk as wardex
    from wardex_sdk import AdapterName, AdaptersConfig
    from wardex_sdk.testing import RecordingTransport
    if sys.argv[1] == "manual_on":
        agents.set_tracing_disabled(False)
    seen = []
    class H(logging.Handler):
        def emit(self, r):
            seen.append(r.getMessage())
    logging.getLogger("wardex_sdk").addHandler(H(level=logging.INFO))
    wardex.init(transport=RecordingTransport(),
                adapters=AdaptersConfig(enabled=(AdapterName.OPENAI_AGENTS,)))
    wardex.close()
    print("NOTICE" if _TRACING_DISABLED_NOTICE in seen else "SILENT")
    """
)


@pytest.mark.parametrize(
    ("env", "mode", "expected"),
    [
        ("1", "env_only", "NOTICE"),
        ("TRUE", "env_only", "NOTICE"),
        ("1", "manual_on", "SILENT"),
        ("false", "env_only", "SILENT"),
    ],
)
def test_tracing_state_follows_the_frameworks_precedence(env, mode, expected):
    """The variable is read ONCE per process by the framework, so each case is
    its own interpreter: the manual switch wins over the environment, and the
    environment is parsed the way the framework parses it (`TRUE` is true)."""
    proc = subprocess.run(
        [sys.executable, "-c", _PRECEDENCE_SCRIPT, mode],
        env={**os.environ, "OPENAI_AGENTS_DISABLE_TRACING": env, "OPENAI_API_KEY": "sk-test"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected, proc.stderr


def test_a_processor_removed_after_init_is_reported_once_at_uninstall(agents_env, wardex_log):
    """`set_trace_processors` after `wardex.init()` replaces the whole list,
    wardex's processor included. Nothing structural is recorded, and the
    uninstall says exactly that, once, with the fix."""
    _init()
    try:
        set_trace_processors([])
        assert _run(_agents()).final_output == "done"
        spans = _spans()
        assert len(_chat_spans(spans)) == 3
        assert _adapter_spans(spans) == []
    finally:
        wardex.close()
    assert wardex_log.lines(logging.WARNING).count(_PROCESSOR_REMOVED_NOTICE) == 1
    assert counters.get("adapters.openai_agents.processor_removed") == 1


# --------------------------------------------------------------------------
# install / uninstall
# --------------------------------------------------------------------------


def test_install_is_idempotent_and_uninstall_restores_the_tuple_by_identity(agents_env):
    before = _processors()
    with installed_adapter(OpenAIAgentsAdapter) as live:
        during = _processors()
        assert during is not before
        assert during == (*before, live.adapter._processor)
        live.adapter.install(live.client, live.ctx)
        assert _processors() is during, "a second install registered a second processor"
    assert _processors() is before, "the framework's own tuple must come back, not a copy"


def test_a_host_processor_added_after_wardex_survives_the_uninstall(agents_env):
    host = _HostProcessor()
    with installed_adapter(OpenAIAgentsAdapter) as live:
        add_trace_processor(host)
        assert live.adapter._processor in _processors()
        live.teardown()  # inside: the harness resets counters on the way out
        assert _processors() == (host,)
        assert counters.get("adapters.openai_agents.processors_changed_under_us") == 1


def test_a_host_processor_registered_before_init_survives(agents_env):
    host = _HostProcessor()
    set_trace_processors([host])
    before = _processors()
    with installed_adapter(OpenAIAgentsAdapter):
        assert _processors()[0] is host
    assert _processors() is before


def test_an_unrecognized_surface_declines_loudly_and_registers_nothing(
    agents_env, monkeypatch, wardex_log
):
    """A span-data class whose constructor keywords moved is a surface the
    mapping cannot read; the adapter says so once and touches nothing."""
    import agents.tracing as tracing

    class Moved:
        def __init__(self, agent_name: str) -> None:
            self.agent_name = agent_name

    monkeypatch.setattr(tracing, "AgentSpanData", Moved)
    before = _processors()
    with installed_adapter(OpenAIAgentsAdapter) as live:
        assert _processors() is before
        assert not live.adapter._installed
        assert counters.get("adapters.openai_agents.unsupported_surface") == 1
    warnings = wardex_log.lines(logging.WARNING)
    assert len(warnings) == 1 and "surface unrecognized" in warnings[0]


def _fake_agents_package(tmp_path, *, with_tracing: bool):  # noqa: ANN001, ANN202
    """A local package called `agents` whose import leaves a file behind."""
    pkg = tmp_path / "agents"
    pkg.mkdir()
    marker = tmp_path / "imported.txt"
    (pkg / "__init__.py").write_text(f"open({str(marker)!r}, 'w').write('x')\n")
    if with_tracing:
        (pkg / "tracing.py").write_text("")
    return marker


_DECLINE_SCRIPT = textwrap.dedent(
    """
    import importlib.metadata as md, json, logging, sys
    mode = sys.argv[1]
    if mode == "absent":
        real = md.distribution
        def absent(name):
            if name == "openai-agents":
                raise md.PackageNotFoundError(name)
            return real(name)
        md.distribution = absent
    seen = []
    class H(logging.Handler):
        def emit(self, r):
            seen.append([r.levelno, r.getMessage()])
    logging.getLogger("wardex_sdk").addHandler(H(level=logging.DEBUG))
    import wardex_sdk as wardex
    from wardex_sdk import AdapterName, AdaptersConfig
    from wardex_sdk._assembly import counters
    from wardex_sdk.testing import RecordingTransport
    kw = {} if mode == "shadowed_auto" else {
        "adapters": AdaptersConfig(enabled=(AdapterName.OPENAI_AGENTS,))
    }
    wardex.init(transport=RecordingTransport(), **kw)
    wardex.close()
    print(json.dumps({
        "agents_imported": "agents" in sys.modules,
        "lines": seen,
        "shadowed": counters.get("adapters.openai_agents.shadowed"),
        "unsupported": counters.get("adapters.openai_agents.unsupported_surface"),
    }))
    """
)


def _decline_in_a_fresh_process(tmp_path, mode: str) -> dict:  # noqa: ANN001
    """The decline path is only real in a process that has NOT yet imported
    the framework: this suite imported it at collection, so an in-process
    check would judge a module body that already ran against the real
    package. `PYTHONPATH` puts the local package ahead of the wheel the way a
    project folder named `agents` would."""
    proc = subprocess.run(
        [sys.executable, "-c", _DECLINE_SCRIPT, mode],
        env={**os.environ, "PYTHONPATH": str(tmp_path), "OPENAI_API_KEY": "sk-test"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == "", proc.stderr
    import json

    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_an_absent_distribution_declines_before_importing_anything(tmp_path):
    """A local package called `agents` with no `openai-agents` distribution:
    the adapter never imports it, so its side effects never run, and nothing
    is logged — absence is an answer, not a failure."""
    marker = _fake_agents_package(tmp_path, with_tracing=True)
    out = _decline_in_a_fresh_process(tmp_path, "absent")
    assert not marker.exists()
    assert out["agents_imported"] is False
    assert out["lines"] == []
    assert out["shadowed"] == 0 and out["unsupported"] == 0


@pytest.mark.parametrize(
    ("with_tracing", "mode"),
    [
        (True, "shadowed"),
        (False, "shadowed"),
        (True, "shadowed_auto"),
    ],
)
def test_a_shadowed_module_declines_loudly_without_importing_it(tmp_path, with_tracing, mode):
    """The distribution IS installed but `import agents` would answer a local
    package: declined with one line naming both paths, and the local package
    is never imported — with or without a `tracing` submodule of its own, and
    whether the adapter was named or auto-detected."""
    marker = _fake_agents_package(tmp_path, with_tracing=with_tracing)
    out = _decline_in_a_fresh_process(tmp_path, mode)
    assert not marker.exists()
    assert out["agents_imported"] is False
    assert out["shadowed"] == 1 and out["unsupported"] == 0
    warnings = [m for lvl, m in out["lines"] if lvl == logging.WARNING]
    assert len(warnings) == 1
    assert "resolved to" in warnings[0] and str(tmp_path) in warnings[0]
    assert "failed to load" not in warnings[0]


# --------------------------------------------------------------------------
# scenarios on the fake server
# --------------------------------------------------------------------------


def _fc(name: str, call_id: str, args: str = "{}") -> dict:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": args,
        "status": "completed",
    }


_DONE = [
    {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "done", "annotations": []}],
    }
]


def _calls_made(inp: object) -> list[str]:
    items = inp if isinstance(inp, list) else []
    calls = [x for x in items if isinstance(x, dict) and x.get("type") == "function_call"]
    return [x.get("name") for x in calls]


def _outputs_done(inp: object) -> int:
    items = inp if isinstance(inp, list) else []
    return sum(1 for x in items if isinstance(x, dict) and x.get("type") == "function_call_output")


def _decide_chain(inp: object) -> list[dict]:
    """agent_a -> agent_b -> agent_c -> done."""
    made = _calls_made(inp)
    if "transfer_to_agent_b" not in made:
        return [_fc("transfer_to_agent_b", "call_h1")]
    if "transfer_to_agent_c" not in made:
        return [_fc("transfer_to_agent_c", "call_h2")]
    return _DONE


def _decide_single(inp: object) -> list[dict]:
    return _DONE


def _decide_two_handoffs(inp: object) -> list[dict]:
    """Two handoffs requested in ONE response; the framework takes the first."""
    if "transfer_to_agent_b" not in _calls_made(inp):
        return [_fc("transfer_to_agent_b", "call_h1"), _fc("transfer_to_agent_c", "call_h2")]
    return _DONE


def _decide_handoff_once(inp: object) -> list[dict]:
    if "transfer_to_agent_b" not in _calls_made(inp):
        return [_fc("transfer_to_agent_b", "call_h1")]
    return _DONE


@pytest.fixture
def scenario(monkeypatch):
    """Swap the fake server's turn policy: the handler resolves `_decide` by
    name at call time, so one monkeypatch retargets the whole server."""
    import test_openai_agents_wire as wire

    def use(fn) -> None:  # noqa: ANN001
        monkeypatch.setattr(wire, "_decide", fn)

    return use


def _chain_agents() -> Agent:
    agent_c = Agent(name="agent_c", instructions="c", model="gpt-4o-mini")
    agent_b = Agent(name="agent_b", instructions="b", handoffs=[agent_c], model="gpt-4o-mini")
    return Agent(name="agent_a", instructions="a", handoffs=[agent_b], model="gpt-4o-mini")


def test_a_two_hop_handoff_chain_is_three_siblings_not_a_nest(agents_env, scenario):
    """Contract §6.3(d): a handoff is a MARKER, the receiver a SIBLING. Three
    `invoke_agent` spans all parented to the root by id; `parent_agent` names
    the sender; each receiver links `HANDOFF_FROM` to the marker's span id."""
    scenario(_decide_chain)
    _init()
    try:
        res = _run(_chain_agents())
        assert res.last_agent.name == "agent_c"
        spans = _spans()
    finally:
        wardex.close()
    root = _one(spans, "invoke_workflow Agent workflow")
    agents = {n: _one(spans, f"invoke_agent {n}") for n in ("agent_a", "agent_b", "agent_c")}
    for span in agents.values():
        assert span.parent_span_id == root.context.span_id
        assert _edge(span) == (ParentSource.UNIT_ACTIVE, 1.0, ())
    assert [agents[n].agent.parent_agent for n in ("agent_a", "agent_b", "agent_c")] == [
        None,
        "agent_a",
        "agent_b",
    ]
    ab = _one(spans, "handoff agent_a→agent_b")
    bc = _one(spans, "handoff agent_b→agent_c")
    assert ab.parent_span_id == agents["agent_a"].context.span_id
    assert bc.parent_span_id == agents["agent_b"].context.span_id
    assert ab.agent.name == "agent_b" and ab.agent.parent_agent == "agent_a"
    assert [(lk.reason, lk.span_id) for lk in agents["agent_b"].links] == [
        (LinkReason.HANDOFF_FROM, ab.context.span_id)
    ]
    assert [(lk.reason, lk.span_id) for lk in agents["agent_c"].links] == [
        (LinkReason.HANDOFF_FROM, bc.context.span_id)
    ]
    assert len(_chat_spans(spans)) == 3
    assert counters.get("adapters.openai_agents.link_target_unresolved") == 0


def test_two_runs_on_one_task_are_two_clean_roots_and_unpin_is_why(
    agents_env, scenario, monkeypatch
):
    """Pin at start, unpin at end: the second run on the same task opens on a
    clean carrier. The NEGATIVE CONTROL turns `unpin` into a no-op and the
    second root then carries `correlation_conflict` — the failure the verb
    exists to make unnecessary."""
    scenario(_decide_single)

    async def two() -> None:
        await Runner.run(Agent(name="agent_a", instructions="a", model="gpt-4o-mini"), "hi")
        await Runner.run(Agent(name="agent_b", instructions="b", model="gpt-4o-mini"), "hi")

    _init()
    try:
        asyncio.run(two())
        spans = _spans()
        assert counters.get("adapters.openai_agents.unpin") == counters.get(
            "adapters.openai_agents.pin_ok"
        )
    finally:
        wardex.close()
    roots = [s for s in _adapter_spans(spans) if s.parent_span_id is None]
    assert [s.name for s in roots] == ["invoke_workflow Agent workflow"] * 2
    assert all(_edge(s) == (ParentSource.TRACE_ROOT, 1.0, ()) for s in roots)
    assert len({s.context.trace_id for s in roots}) == 2

    from wardex_sdk._adapters._context import RunHandle

    monkeypatch.setattr(RunHandle, "unpin", lambda self: False)
    _init()
    try:
        asyncio.run(two())
        spans = _spans()
    finally:
        wardex.close()
    roots = [s for s in _adapter_spans(spans) if s.parent_span_id is None]
    assert len(roots) == 2
    assert Limitation.CORRELATION_CONFLICT in _edge(roots[1])[2]


def test_a_run_inside_a_host_span_hangs_off_it(agents_env, scenario):
    scenario(_decide_single)
    _init()
    try:
        with wardex.span("outer") as outer:
            _run(Agent(name="agent_a", instructions="a", model="gpt-4o-mini"))
        spans = _spans()
    finally:
        wardex.close()
    root = _one(spans, "invoke_workflow Agent workflow")
    assert root.parent_span_id == outer.context.span_id
    assert _edge(root) == (ParentSource.CONTEXTVAR, 1.0, ())


def test_two_runs_inside_one_framework_trace_share_one_root(agents_env, scenario):
    """`Runner.run` inside a user's `with trace(...)` creates no trace of its
    own, so one `invoke_workflow` carries both agents."""
    from agents.tracing import trace

    scenario(_decide_single)

    async def both() -> None:
        with trace("outer"):
            await Runner.run(Agent(name="agent_a", instructions="a", model="gpt-4o-mini"), "hi")
            await Runner.run(Agent(name="agent_b", instructions="b", model="gpt-4o-mini"), "hi")

    _init()
    try:
        asyncio.run(both())
        spans = _spans()
    finally:
        wardex.close()
    root = _one(spans, "invoke_workflow outer")
    for name in ("agent_a", "agent_b"):
        assert _one(spans, f"invoke_agent {name}").parent_span_id == root.context.span_id
    assert len([s for s in _adapter_spans(spans) if s.parent_span_id is None]) == 1
    assert _extra(root)["wardex.openai_agents.agents"] == 2


def test_without_task_and_turn_spans_only_the_turn_attribute_disappears(agents_env, scenario):
    scenario(_decide_chain)
    _init()
    try:
        _run(_chain_agents(), run_config=RunConfig(tracing={"include_task_and_turn_spans": False}))
        spans = _spans()
    finally:
        wardex.close()
    names = sorted(s.name for s in _adapter_spans(spans))
    assert names == [
        "handoff agent_a→agent_b",
        "handoff agent_b→agent_c",
        "invoke_agent agent_a",
        "invoke_agent agent_b",
        "invoke_agent agent_c",
        "invoke_workflow Agent workflow",
    ]
    marker = _one(spans, "handoff agent_a→agent_b")
    assert "wardex.openai_agents.turn" not in _extra(marker)
    assert _extra(_one(spans, "invoke_workflow Agent workflow"))["wardex.openai_agents.turns"] == 0


def test_no_framework_identifier_can_shape_this_adapters_tree():
    """The product claim as an AST test over the shipped module's own source:
    none of the verbs by which an identifier could shape the tree, and no
    read of the framework's `parent_id`, appears anywhere in it."""
    import ast
    import inspect

    import wardex_sdk._adapters._openai_agents as mod

    source = inspect.getsource(mod)
    assert "parent_id" not in source
    forbidden = {"rejoin", "attach", "claim", "claim_run"}
    used = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    assert used & forbidden == set()


def test_a_handoff_whose_target_never_resolves_is_an_error_marker(agents_env, scenario):
    """`on_handoff` raising: the framework's span has no `to_agent`. The
    marker reads `agent_a→unresolved` at ERROR `handoff_error`, the agent
    fails with the framework's generic error, and the root follows."""
    from agents import handoff

    scenario(_decide_handoff_once)

    def boom(ctx):  # noqa: ANN001, ANN202
        raise RuntimeError("no such target")

    agent_b = Agent(name="agent_b", instructions="b", model="gpt-4o-mini")
    agent_a = Agent(
        name="agent_a",
        instructions="a",
        handoffs=[handoff(agent_b, on_handoff=boom)],
        model="gpt-4o-mini",
    )
    _init()
    try:
        with pytest.raises(RuntimeError):
            _run(agent_a)
        spans = _spans()
    finally:
        wardex.close()
    marker = _one(spans, "handoff agent_a→unresolved")
    assert (marker.status, marker.error_type) == (StatusCode.ERROR, "handoff_error")
    assert marker.agent.name == "unresolved" and marker.agent.parent_agent == "agent_a"
    assert counters.get("adapters.openai_agents.handoff_unresolved") == 1
    agent = _one(spans, "invoke_agent agent_a")
    assert (agent.status, agent.error_type) == (StatusCode.ERROR, "agent_run_error")
    root = _one(spans, "invoke_workflow Agent workflow")
    assert (root.status, root.error_type) == (StatusCode.ERROR, "agent_run_error")
    assert [s.name for s in _adapter_spans(spans) if s.name.startswith("invoke_agent")] == [
        "invoke_agent agent_a"
    ]


def test_two_handoffs_in_one_response_mark_the_handoff_and_nothing_else(agents_env, scenario):
    scenario(_decide_two_handoffs)
    agent_c = Agent(name="agent_c", instructions="c", model="gpt-4o-mini")
    agent_b = Agent(name="agent_b", instructions="b", model="gpt-4o-mini")
    agent_a = Agent(
        name="agent_a", instructions="a", handoffs=[agent_b, agent_c], model="gpt-4o-mini"
    )
    _init()
    try:
        assert _run(agent_a).last_agent.name == "agent_b"
        spans = _spans()
    finally:
        wardex.close()
    marker = _one(spans, "handoff agent_a→agent_b")
    assert (marker.status, marker.error_type) == (StatusCode.ERROR, "multiple_handoffs_requested")
    assert _one(spans, "invoke_agent agent_b").status is StatusCode.OK
    assert _one(spans, "invoke_workflow Agent workflow").status is StatusCode.OK


# --------------------------------------------------------------------------
# the three-turn tree, on every run shape
# --------------------------------------------------------------------------

_ROOT = "invoke_workflow wf"


def _assert_three_turn_tree(spans: list[Any], *, streamed: bool) -> None:
    """The target tree: 8 spans, 1 trace, every edge read from the context,
    chains asserted by span id, and the joins that make the tree navigable."""
    assert len(spans) == 8
    assert len({s.context.trace_id for s in spans}) == 1
    root = _one(spans, _ROOT)
    agent_a = _one(spans, "invoke_agent agent_a")
    agent_b = _one(spans, "invoke_agent agent_b")
    tool = _one(spans, "execute_tool get_weather")
    marker = _one(spans, "handoff agent_a→agent_b")
    chats = sorted(_chat_spans(spans), key=lambda s: s.gen_ai.response_id)
    assert [c.gen_ai.response_id for c in chats] == ["resp_1", "resp_2", "resp_3"]
    assert _edge(root) == (ParentSource.TRACE_ROOT, 1.0, ())
    for s in (agent_a, agent_b, tool, marker):
        assert _edge(s) == (ParentSource.UNIT_ACTIVE, 1.0, ())
    for c in chats:
        markers = _edge(c)[2]
        assert _edge(c)[:2] == (ParentSource.CONTEXTVAR, 1.0)
        assert (Limitation.REASSEMBLED_FROM_STREAM in markers) is streamed
    # chains, BY ID
    assert agent_a.parent_span_id == root.context.span_id
    assert agent_b.parent_span_id == root.context.span_id
    assert tool.parent_span_id == agent_a.context.span_id
    assert marker.parent_span_id == agent_a.context.span_id
    assert [c.parent_span_id for c in chats] == [
        agent_a.context.span_id,
        agent_a.context.span_id,
        agent_b.context.span_id,
    ]
    # the joins
    assert tool.tool.call_id == "call_1"
    assert _extra(tool)["wardex.openai_agents.tool_call_id_source"] == "response_output_match"
    assert _extra(tool)["wardex.openai_agents.response_id"] == chats[0].gen_ai.response_id
    assert _extra(marker)["wardex.openai_agents.response_id"] == chats[1].gen_ai.response_id
    assert agent_b.agent.parent_agent == "agent_a"
    assert [(lk.reason, lk.span_id) for lk in agent_b.links] == [
        (LinkReason.HANDOFF_FROM, marker.context.span_id)
    ]
    assert marker.agent.name == "agent_b" and marker.agent.parent_agent == "agent_a"
    # the conversation, on every adapter span. NOT on the chat spans: the
    # byte seam's tracker latches the span context alone at request time
    # (`_interceptors/_seam.py::_latched`), a documented, pre-existing gap
    # of the wire layer that the adapter cannot close from its side.
    for s in _adapter_spans(spans):
        assert s.conversation is not None and s.conversation.conversation_id == "conv-123"
    for c in chats:
        assert c.conversation is None
    # no usage anywhere but the wire
    for s in _adapter_spans(spans):
        assert s.gen_ai is None
    for s in (root, agent_a, agent_b, tool, marker):
        assert s.capture_integrity is None or s.capture_integrity.limitations == ()
    assert s.status is StatusCode.OK
    assert _extra(root)["wardex.openai_agents.turns"] == 3
    assert _extra(root)["wardex.openai_agents.agents"] == 2


def _assert_counters_clean() -> None:
    snap = counters.snapshot()
    assert {k: v for k, v in snap.items() if k.startswith("assembly.")} == {}
    active = {k: v for k, v in snap.items() if k.startswith("adapters.openai_agents.active.")}
    assert active == {
        "adapters.openai_agents.active.trace": 1,
        "adapters.openai_agents.active.agent": 2,
        "adapters.openai_agents.active.handoff": 1,
        "adapters.openai_agents.active.tool": 1,
    }
    assert counters.get("adapters.openai_agents.pin_refused") == 0


def _run_config() -> RunConfig:
    """Fresh per run: a `RunConfig` owns its model provider, whose OpenAI
    client keeps the base URL of the first server it saw."""
    return RunConfig(workflow_name="wf", group_id="conv-123")


def test_runner_run_three_turns_are_one_tree(agents_env):
    """THE measured record: the tree the tracker's baseline said did not
    exist. Printed, so the text in the tracker is the text this test saw."""
    _init()
    try:
        assert _run(_agents(), run_config=_run_config()).final_output == "done"
        spans = _spans()
        _assert_three_turn_tree(spans, streamed=False)
        _assert_counters_clean()
    finally:
        wardex.close()
    print("\n" + _print_tree(spans))


def test_run_sync_three_turns_are_one_tree(agents_env):
    _init()
    try:
        assert Runner.run_sync(_agents(), "hi", run_config=_run_config()).final_output == "done"
        spans = _spans()
        _assert_three_turn_tree(spans, streamed=False)
        _assert_counters_clean()
    finally:
        wardex.close()


def test_run_streamed_three_turns_are_one_tree_after_the_drain(agents_env):
    """The trace ends when the background loop task ends — after the host
    drained `stream_events()` — so the spans are read after the drain."""

    async def go() -> str:
        result = Runner.run_streamed(_agents(), "hi", run_config=_run_config())
        async for _ in result.stream_events():
            pass
        return result.final_output

    _init()
    try:
        assert asyncio.run(go()) == "done"
        spans = _spans()
        _assert_three_turn_tree(spans, streamed=True)
        _assert_counters_clean()
    finally:
        wardex.close()


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


def _decide_parallel(inp: object) -> list[dict]:
    if _outputs_done(inp) == 0:
        return [_fc("get_weather", "call_1", '{"city":"Seoul"}'), _fc("get_time", "call_2")]
    return _DONE


def _decide_twice_the_same(inp: object) -> list[dict]:
    if _outputs_done(inp) == 0:
        return [
            _fc("get_weather", "call_1", '{"city":"Seoul"}'),
            _fc("get_weather", "call_2", '{"city":"Seoul"}'),
        ]
    return _DONE


def _two_tools() -> Agent:
    from agents import function_tool

    @function_tool
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    @function_tool
    def get_time() -> str:
        return "12:00"

    return Agent(
        name="agent_a", instructions="a", tools=[get_weather, get_time], model="gpt-4o-mini"
    )


def test_parallel_tool_calls_keep_parent_confidence_at_one(agents_env, scenario):
    """Two tools in one turn run in two tasks the framework created AFTER the
    agent span started, so each inherits the agent's pin: both at
    `unit_active` / 1.0 / no marker, with distinct call ids and the source
    label, and not one refused pin."""
    scenario(_decide_parallel)
    _init()
    try:
        assert _run(_two_tools()).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.pin_refused") == 0
    finally:
        wardex.close()
    agent = _one(spans, "invoke_agent agent_a")
    weather = _one(spans, "execute_tool get_weather")
    clock = _one(spans, "execute_tool get_time")
    for s in (weather, clock):
        assert _edge(s) == (ParentSource.UNIT_ACTIVE, 1.0, ())
        assert s.parent_span_id == agent.context.span_id
        assert _extra(s)["wardex.openai_agents.tool_call_id_source"] == "response_output_match"
    assert (weather.tool.call_id, clock.tool.call_id) == ("call_1", "call_2")


def test_two_identical_tool_calls_in_one_response_get_no_guessed_id(agents_env, scenario):
    scenario(_decide_twice_the_same)
    _init()
    try:
        assert _run(_two_tools()).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.tool_call_id_ambiguous") == 2
    finally:
        wardex.close()
    tools = [s for s in spans if s.name == "execute_tool get_weather"]
    assert len(tools) == 2
    for s in tools:
        assert s.tool.call_id is None
        assert Limitation.TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS in _edge(s)[2]
        assert "wardex.openai_agents.tool_call_id_source" not in _extra(s)


def test_an_http_call_inside_a_tool_handler_nests_under_the_tool_span(agents_env, scenario):
    import httpx
    from agents import function_tool

    base, posts = agents_env
    scenario(_decide_handoff_once)

    @function_tool
    def get_weather(city: str) -> str:
        httpx.post(f"{base}/tool-side", json={"city": city})
        return "sunny"

    def decide(inp: object) -> list[dict]:
        if _outputs_done(inp) == 0:
            return [_fc("get_weather", "call_1", '{"city":"Seoul"}')]
        return _DONE

    scenario(decide)
    agent = Agent(name="agent_a", instructions="a", tools=[get_weather], model="gpt-4o-mini")
    _init()
    try:
        assert _run(agent).final_output == "done"
        spans = _spans()
    finally:
        wardex.close()
    tool = _one(spans, "execute_tool get_weather")
    side = _one(spans, "HTTP POST /v1/tool-side")
    assert side.parent_span_id == tool.context.span_id
    assert _edge(side)[:2] == (ParentSource.CONTEXTVAR, 1.0)


def test_an_agent_used_as_a_tool_nests_under_the_tool_span(agents_env, scenario):
    """`agent.as_tool()` runs a nested `Runner.run` inside the tool task: the
    inner `invoke_agent` is the tool span's child by context, and its failure
    is the TOOL's (handled, non-fatal) — the outer agent and the root stay OK."""

    def decide(inp: object) -> list[dict]:
        items = inp if isinstance(inp, list) else []
        if any(isinstance(x, dict) and x.get("content") == "INNER" for x in items):
            return [_fc("no_such_tool", "call_x")]
        if _outputs_done(inp) == 0:
            return [_fc("helper_tool", "call_1", '{"input":"INNER"}')]
        return _DONE

    scenario(decide)
    helper = Agent(name="helper", instructions="inner", model="gpt-4o-mini")
    outer = Agent(
        name="agent_a",
        instructions="outer",
        tools=[helper.as_tool(tool_name="helper_tool", tool_description="helps")],
        model="gpt-4o-mini",
    )
    _init()
    try:
        assert _run(outer).final_output == "done"
        spans = _spans()
    finally:
        wardex.close()
    tool = _one(spans, "execute_tool helper_tool")
    inner = _one(spans, "invoke_agent helper")
    assert inner.parent_span_id == tool.context.span_id
    assert _edge(inner) == (ParentSource.UNIT_ACTIVE, 1.0, ())
    assert (tool.status, tool.error_type) == (StatusCode.ERROR, "tool_error_handled")
    assert (inner.status, inner.error_type) == (StatusCode.ERROR, "model_behavior_error")
    assert _one(spans, "invoke_agent agent_a").status is StatusCode.OK
    assert _one(spans, "invoke_workflow Agent workflow").status is StatusCode.OK


# --------------------------------------------------------------------------
# guardrails
# --------------------------------------------------------------------------


def _guarded(*, where: str) -> Agent:
    from agents import GuardrailFunctionOutput, input_guardrail, output_guardrail

    @input_guardrail
    async def block_input(ctx, agent, inp):  # noqa: ANN001, ANN202
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

    @output_guardrail
    async def block_output(ctx, agent, out):  # noqa: ANN001, ANN202
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

    if where == "input":
        return Agent(
            name="agent_a", instructions="a", input_guardrails=[block_input], model="gpt-4o-mini"
        )
    return Agent(
        name="agent_a", instructions="a", output_guardrails=[block_output], model="gpt-4o-mini"
    )


def _assert_tripwire(spans: list[Any], *, name: str, posts: int, made: int) -> None:
    evaluate = _one(spans, f"evaluate {name}")
    agent = _one(spans, "invoke_agent agent_a")
    root = _one(spans, "invoke_workflow Agent workflow")
    assert evaluate.parent_span_id == agent.context.span_id
    assert _edge(evaluate) == (ParentSource.UNIT_ACTIVE, 1.0, ())
    assert evaluate.evaluation.name == name and evaluate.evaluation.score_label == "tripwire"
    assert _extra(evaluate)["wardex.evaluation.triggered"] is True
    for s in (evaluate, agent, root):
        assert (s.status, s.error_type) == (StatusCode.ERROR, "guardrail_tripwire")
    assert len(_chat_spans(spans)) == posts == made


def test_an_input_guardrail_tripwire_fails_the_agent_and_the_run(agents_env, scenario):
    from agents.exceptions import InputGuardrailTripwireTriggered

    base, posts = agents_env
    scenario(_decide_single)
    _init()
    try:
        with pytest.raises(InputGuardrailTripwireTriggered):
            _run(_guarded(where="input"))
        spans = _spans()
    finally:
        wardex.close()
    # sequential by default: the model call never happened
    _assert_tripwire(spans, name="block_input", posts=len(posts), made=0)


def test_an_output_guardrail_tripwire_fails_the_agent_and_the_run(agents_env, scenario):
    from agents.exceptions import OutputGuardrailTripwireTriggered

    base, posts = agents_env
    scenario(_decide_single)
    _init()
    try:
        with pytest.raises(OutputGuardrailTripwireTriggered):
            _run(_guarded(where="output"))
        spans = _spans()
    finally:
        wardex.close()
    _assert_tripwire(spans, name="block_output", posts=len(posts), made=1)


def test_a_streamed_input_guardrail_tripwire_reaches_the_consumer(agents_env, scenario):
    """The streamed loop runs input guardrails in parallel with the model
    call, so the LLM call DID happen; the exception is raised from the
    stream the host is draining."""
    from agents.exceptions import InputGuardrailTripwireTriggered

    base, posts = agents_env
    scenario(_decide_single)

    async def go() -> None:
        result = Runner.run_streamed(_guarded(where="input"), "hi")
        async for _ in result.stream_events():
            pass

    _init()
    try:
        with pytest.raises(InputGuardrailTripwireTriggered):
            asyncio.run(go())
        spans = _spans()
    finally:
        wardex.close()
    _assert_tripwire(spans, name="block_input", posts=len(posts), made=1)


# --------------------------------------------------------------------------
# MCP list-tools
# --------------------------------------------------------------------------


def test_an_mcp_list_tools_span_carries_a_hash_and_never_a_name(agents_env, scenario):
    from agents.mcp import MCPServer
    from mcp.types import CallToolResult, Tool

    class InProcess(MCPServer):
        def __init__(self) -> None:
            super().__init__()

        @property
        def name(self) -> str:
            return "in-process"

        async def connect(self) -> None:
            return None

        async def cleanup(self) -> None:
            return None

        async def list_tools(self, run_context=None, agent=None):  # noqa: ANN001, ANN202
            return [Tool(name="secret_tool_name", inputSchema={"type": "object"})]

        async def call_tool(self, tool_name, arguments, meta=None):  # noqa: ANN001, ANN202
            return CallToolResult(content=[])

        async def list_prompts(self):  # noqa: ANN202
            raise NotImplementedError

        async def get_prompt(self, name, arguments=None):  # noqa: ANN001, ANN202
            raise NotImplementedError

    scenario(_decide_single)
    agent = Agent(name="agent_a", instructions="a", mcp_servers=[InProcess()], model="gpt-4o-mini")
    _init()
    try:
        assert _run(agent).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.active.mcp_list_tools") == 1
    finally:
        wardex.close()
    step = _one(spans, "execute_step mcp.list_tools")
    extra = _extra(step)
    assert extra["wardex.step.name"] == "mcp.list_tools"
    assert extra["wardex.openai_agents.mcp.server"] == "in-process"
    assert extra["wardex.openai_agents.mcp.tools_count"] == 1
    assert len(extra["wardex.openai_agents.mcp.tools_hash"]) == 16
    for s in _adapter_spans(spans):
        assert "secret_tool_name" not in repr(s.extra)
        assert "secret_tool_name" not in s.name
    # fired before the agent's first turn: the run root is its parent
    assert step.parent_span_id == _one(spans, "invoke_workflow Agent workflow").context.span_id


# --------------------------------------------------------------------------
# extras: fixed arity per span kind
# --------------------------------------------------------------------------


def test_every_span_kind_writes_a_fixed_set_of_extras(agents_env):
    """One more than the adapter wrote: the builder adds
    `gen_ai.operation.name` itself. Every key is under a declared prefix or
    is `wardex.framework` / `wardex.step.name` / `wardex.evaluation.*`."""
    from wardex_sdk._adapters._openai_agents import FRAMEWORK_EXTRA_PREFIXES

    _init()
    try:
        _run(_agents(), run_config=_run_config())
        spans = _spans()
    finally:
        wardex.close()
    expected = {
        _ROOT: {
            "wardex.framework",
            "wardex.openai_agents.trace_id",
            "wardex.openai_agents.turns",
            "wardex.openai_agents.agents",
        },
        "invoke_agent agent_a": {
            "wardex.framework",
            "wardex.openai_agents.turns",
            "wardex.openai_agents.tools_count",
            "wardex.openai_agents.handoffs_count",
            "wardex.openai_agents.last_response_id",
        },
        "handoff agent_a→agent_b": {
            "wardex.framework",
            "wardex.openai_agents.turn",
            "wardex.openai_agents.response_id",
        },
        "execute_tool get_weather": {
            "wardex.framework",
            "wardex.openai_agents.turn",
            "wardex.openai_agents.response_id",
            "wardex.openai_agents.tool_call_id_source",
        },
    }
    for name, keys in expected.items():
        span = _one(spans, name)
        assert set(_extra(span)) == keys | {"gen_ai.operation.name"}, name
        assert len(span.extra) == len(keys) + 1, name
        for key in keys:
            assert key.startswith(FRAMEWORK_EXTRA_PREFIXES) or key in {
                "wardex.framework",
                "wardex.step.name",
            }


# --------------------------------------------------------------------------
# failure mapping
# --------------------------------------------------------------------------


def _decide_loop(inp: object) -> list[dict]:
    """Always one more tool call: the shape that exceeds `max_turns`."""
    n = _outputs_done(inp)
    return [_fc("get_weather", f"call_{n + 1}", '{"city":"Seoul"}')]


def _weather_agent(**tool_kwargs: Any) -> Agent:
    from agents import function_tool

    @function_tool(**tool_kwargs)
    def get_weather(city: str) -> str:
        if city == "boom":
            raise RuntimeError("the tool failed")
        return f"sunny in {city}"

    return Agent(name="agent_a", instructions="a", tools=[get_weather], model="gpt-4o-mini")


def test_max_turns_exceeded_fails_the_agent_and_the_root(agents_env, scenario):
    from agents.exceptions import MaxTurnsExceeded

    scenario(_decide_loop)
    _init()
    try:
        with pytest.raises(MaxTurnsExceeded):
            _run(_weather_agent(), max_turns=2)
        spans = _spans()
    finally:
        wardex.close()
    agent = _one(spans, "invoke_agent agent_a")
    root = _one(spans, "invoke_workflow Agent workflow")
    for s in (agent, root):
        assert (s.status, s.error_type) == (StatusCode.ERROR, "max_turns_exceeded")
    assert _extra(agent)["wardex.openai_agents.max_turns"] == 2
    tools = [s for s in spans if s.name == "execute_tool get_weather"]
    assert len(tools) == 2 and all(t.status is StatusCode.OK for t in tools)


def test_an_unmapped_error_message_is_reported_once_and_never_printed(wardex_log):
    """The framework's error message can be HOST content — an `on_approval`
    rejection reason lands on the function span verbatim — so the report is
    keyed on nothing of it: one WARNING for the whole process, the text
    absent from the line, and the counter carrying the volume."""
    from wardex_sdk._adapters._openai_agents import _classify_error

    with installed_adapter(OpenAIAgentsAdapter) as live:
        adapter = live.adapter
        assert adapter._ctx is not None
        first = _classify_error(adapter, "rejected: my SSN is 123-45-6789")
        second = _classify_error(adapter, "rejected: card 4111 1111 1111 1111")
        assert counters.get("adapters.openai_agents.error_message_unmapped") == 2
    assert first == second == ("openai_agents_error", False)
    warnings = wardex_log.lines(logging.WARNING)
    assert len(warnings) == 1
    assert "does not map" in warnings[0]
    assert "SSN" not in warnings[0] and "4111" not in warnings[0]


def test_a_handled_max_turns_still_reads_as_an_error(agents_env, scenario):
    """DOCUMENTED LIMITATION: the framework marks the agent span before it
    consults `error_handlers`, so a handled max_turns ships ERROR on the
    agent and the root while the host gets its handler's output."""
    scenario(_decide_loop)
    _init()
    try:
        res = _run(
            _weather_agent(), max_turns=2, error_handlers={"max_turns": lambda data: "handled"}
        )
        assert res.final_output == "handled"
        spans = _spans()
    finally:
        wardex.close()
    for name in ("invoke_agent agent_a", "invoke_workflow Agent workflow"):
        s = _one(spans, name)
        assert (s.status, s.error_type) == (StatusCode.ERROR, "max_turns_exceeded")


def _decide_boom(inp: object) -> list[dict]:
    if _outputs_done(inp) == 0:
        return [_fc("get_weather", "call_1", '{"city":"boom"}')]
    return _DONE


def test_a_fatal_tool_failure_fails_the_agent_and_the_root(agents_env, scenario):
    """`failure_error_function=None`: the framework re-raises as `UserError`,
    marks the tool span, then the agent span with its generic error."""
    from agents.exceptions import UserError

    scenario(_decide_boom)
    _init()
    try:
        with pytest.raises(UserError):
            _run(_weather_agent(failure_error_function=None))
        spans = _spans()
    finally:
        wardex.close()
    tool = _one(spans, "execute_tool get_weather")
    assert (tool.status, tool.error_type) == (StatusCode.ERROR, "tool_error")
    for name in ("invoke_agent agent_a", "invoke_workflow Agent workflow"):
        s = _one(spans, name)
        assert (s.status, s.error_type) == (StatusCode.ERROR, "agent_run_error")


def test_a_handled_tool_failure_stays_on_the_tool_span(agents_env, scenario):
    """The default handler turns the exception into a tool output: the tool
    span says `tool_error_handled`, and nothing above it is marked."""
    scenario(_decide_boom)
    _init()
    try:
        assert _run(_weather_agent()).final_output == "done"
        spans = _spans()
    finally:
        wardex.close()
    tool = _one(spans, "execute_tool get_weather")
    assert (tool.status, tool.error_type) == (StatusCode.ERROR, "tool_error_handled")
    assert _one(spans, "invoke_agent agent_a").status is StatusCode.OK
    assert _one(spans, "invoke_workflow Agent workflow").status is StatusCode.OK


# --------------------------------------------------------------------------
# a wardex bug costs a span, never the host
# --------------------------------------------------------------------------


def _inject_open(monkeypatch) -> None:  # noqa: ANN001
    from wardex_sdk._assembly import SpanIntent, UnitRegistry

    original = UnitRegistry.open

    def open_(self, kind, key, **kw):  # noqa: ANN001, ANN202
        if kw.get("intent") is SpanIntent.EXECUTE_TOOL:
            raise RuntimeError("injected: the registry cannot open")
        return original(self, kind, key, **kw)

    monkeypatch.setattr(UnitRegistry, "open", open_)


def _inject_describe(monkeypatch) -> None:  # noqa: ANN001
    from wardex_sdk._assembly import SpanDraft

    def set_tool(self, attrs):  # noqa: ANN001, ANN202
        raise RuntimeError("injected: describe dies")

    monkeypatch.setattr(SpanDraft, "set_tool", set_tool)


def _inject_close(monkeypatch) -> None:  # noqa: ANN001
    from wardex_sdk._assembly import UnitKind, UnitRegistry

    original = UnitRegistry.close

    def close(self, unit, **kw):  # noqa: ANN001, ANN202
        if unit.kind is UnitKind.CALL:
            raise RuntimeError("injected: the close dies")
        return original(self, unit, **kw)

    monkeypatch.setattr(UnitRegistry, "close", close)


def _inject_slot(monkeypatch) -> None:  # noqa: ANN001
    from wardex_sdk._adapters._context import AdapterContext

    original = AdapterContext.slot
    tripped = []

    def slot(self, obj):  # noqa: ANN001, ANN202
        data = getattr(obj, "span_data", None)
        if type(data).__name__ == "FunctionSpanData" and not tripped:
            tripped.append(True)
            raise RuntimeError("injected: the slot lookup dies")
        return original(self, obj)

    monkeypatch.setattr(AdapterContext, "slot", slot)


@pytest.mark.parametrize("inject", [_inject_open, _inject_describe, _inject_close, _inject_slot])
def test_a_wardex_fault_costs_a_span_and_never_reaches_the_host(
    agents_env, monkeypatch, wardex_log, inject
):
    """Four faults in wardex's own machinery, one per test: the host still
    gets `done`, the three LLM calls still ship, exactly one WARNING line
    names the loss, and `instrumentation_degraded` lands on a span that
    shipped — the live agent, or the run root."""
    from wardex_sdk._assembly._diag import reset_reports_for_test

    reset_reports_for_test()
    inject(monkeypatch)
    _init()
    try:
        assert _run(_agents()).final_output == "done"
        spans = _spans()
        assert len(_chat_spans(spans)) == 3
    finally:
        wardex.close()
    warnings = wardex_log.lines(logging.WARNING)
    assert len(warnings) == 1, warnings
    assert "openai_agents" in warnings[0] or "openai-agents" in warnings[0]
    degraded = [
        s.name for s in _adapter_spans(spans) if Limitation.INSTRUMENTATION_DEGRADED in _edge(s)[2]
    ]
    assert degraded, [s.name for s in spans]
    assert set(degraded) <= {
        "invoke_agent agent_a",
        "invoke_workflow Agent workflow",
        "execute_tool get_weather",
    }


def test_shutdown_and_force_flush_during_a_run_leave_the_units_open(agents_env, scenario):
    """The framework's `shutdown()` / `force_flush()` reach the processor
    mid-run (an atexit, a host flush): counted, and nothing closes early."""
    from agents import function_tool

    provider = get_trace_provider()

    @function_tool
    def get_weather(city: str) -> str:
        provider.shutdown()
        provider.force_flush()
        return "sunny"

    def decide(inp: object) -> list[dict]:
        if _outputs_done(inp) == 0:
            return [_fc("get_weather", "call_1", '{"city":"Seoul"}')]
        return _DONE

    scenario(decide)
    agent = Agent(name="agent_a", instructions="a", tools=[get_weather], model="gpt-4o-mini")
    _init()
    try:
        assert _run(agent).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.shutdown") == 1
        assert counters.get("adapters.openai_agents.force_flush") == 1
    finally:
        wardex.close()
    for name in (
        "invoke_workflow Agent workflow",
        "invoke_agent agent_a",
        "execute_tool get_weather",
    ):
        s = _one(spans, name)
        assert s.status is StatusCode.OK and _edge(s)[2] == ()


def test_a_host_processor_registered_before_init_sees_the_run_beside_wardex(agents_env):
    host = _HostProcessor()
    set_trace_processors([host])
    before = _processors()
    _init()
    try:
        assert _processors()[0] is host and len(_processors()) == 2
        assert _run(_agents()).final_output == "done"
        spans = _spans()
        assert len(_adapter_spans(spans)) == 5
        assert len(_chat_spans(spans)) == 3
    finally:
        wardex.close()
    assert _processors() is before


# --------------------------------------------------------------------------
# fork
# --------------------------------------------------------------------------


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
@pytest.mark.filterwarnings(
    "ignore:This process.*is multi-threaded, use of fork:DeprecationWarning"
)
def test_the_fork_child_starts_with_no_run_bookkeeping(agents_env, scenario):
    """A run is parked inside a tool on a background thread; the MAIN thread
    forks. The child holds none of the parent's run bookkeeping, a fresh run
    there is one clean root, and the parent's run finishes as itself."""
    import threading

    from agents import function_tool

    from test_fork_semantics import _run_in_child
    from wardex_sdk._adapters._registry import get_registry

    gate = threading.Event()
    entered = threading.Event()

    @function_tool
    def get_weather(city: str) -> str:
        entered.set()
        gate.wait(20)
        return "sunny"

    def decide(inp: object) -> list[dict]:
        if _outputs_done(inp) == 0:
            return [_fc("get_weather", "call_1", '{"city":"Seoul"}')]
        return _DONE

    scenario(decide)
    parent_agent = Agent(name="agent_a", instructions="a", tools=[get_weather], model="gpt-4o-mini")
    result: dict[str, Any] = {}

    def drive() -> None:
        try:
            result["out"] = Runner.run_sync(
                parent_agent, "hi", run_config=RunConfig(workflow_name="Parent")
            ).final_output
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            result["exc"] = exc

    _init()
    try:
        thread = threading.Thread(target=drive, daemon=True)
        thread.start()
        assert entered.wait(20)

        def child() -> dict[str, Any]:
            from agents.models import openai_provider
            from httpx2 import _utils as httpx_utils

            # A fresh httpx client, and no system-proxy lookup for it: on
            # macOS `SCDynamicStoreCopyProxies` segfaults in a forked child
            # (measured), which is the platform's fork rule and not wardex's.
            openai_provider._http_client = None
            httpx_utils.getproxies = dict
            ctx = get_registry()._contexts["openai_agents"]
            slots = len(ctx._slots)
            starts_before = counters.get("adapters.openai_agents.active.trace")

            @function_tool
            def get_weather(city: str) -> str:  # the parent's copy waits on a gate it never sees
                return "child"

            child_agent = Agent(
                name="agent_c", instructions="c", tools=[get_weather], model="gpt-4o-mini"
            )
            out = Runner.run_sync(
                child_agent, "hi", run_config=RunConfig(workflow_name="Child")
            ).final_output
            spans = _adapter_spans(_spans())
            roots = [
                (s.name, [m.value for m in _edge(s)[2]]) for s in spans if s.parent_span_id is None
            ]
            return {
                "slots": slots,
                "out": out,
                "roots": roots,
                "starts": counters.get("adapters.openai_agents.active.trace") - starts_before,
            }

        code, payload = _run_in_child(child, timeout=60.0)
        assert code == 0, payload
        assert payload["slots"] == 0
        assert payload["out"] == "done"
        assert payload["roots"] == [["invoke_workflow Child", []]]
        assert payload["starts"] == 1

        gate.set()
        thread.join(30)
        assert result.get("out") == "done", result
        spans = _spans()
    finally:
        gate.set()
        wardex.close()
    root = _one(spans, "invoke_workflow Parent")
    assert root.status is StatusCode.OK and _edge(root) == (ParentSource.TRACE_ROOT, 1.0, ())
    tool = _one(spans, "execute_tool get_weather")
    assert tool.parent_span_id == _one(spans, "invoke_agent agent_a").context.span_id
