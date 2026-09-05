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
from agents import Agent, Runner
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
from wardex_sdk._assembly import Limitation, counters
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


def _forget_agents(monkeypatch) -> None:  # noqa: ANN001
    for key in [k for k in sys.modules if k == "agents" or k.startswith("agents.")]:
        monkeypatch.delitem(sys.modules, key)


def test_an_absent_distribution_declines_before_importing_anything(
    tmp_path, monkeypatch, wardex_log
):
    """A local package called `agents` with no `openai-agents` distribution:
    the adapter never imports it, so its side effects never run, and nothing
    is logged — absence is an answer, not a failure."""
    import importlib.metadata as md

    marker = _fake_agents_package(tmp_path, with_tracing=True)
    monkeypatch.syspath_prepend(str(tmp_path))
    _forget_agents(monkeypatch)
    monkeypatch.setattr(md, "packages_distributions", lambda: {})
    with installed_adapter(OpenAIAgentsAdapter) as live:
        assert not live.adapter._installed
    assert not marker.exists()
    assert wardex_log.records == []


def test_a_shadowed_module_declines_loudly(tmp_path, monkeypatch, wardex_log):
    """The distribution IS installed but `import agents` answers a local
    package: declined, with one line naming both paths."""
    _fake_agents_package(tmp_path, with_tracing=True)
    monkeypatch.syspath_prepend(str(tmp_path))
    _forget_agents(monkeypatch)
    with installed_adapter(OpenAIAgentsAdapter) as live:
        assert not live.adapter._installed
        assert counters.get("adapters.openai_agents.shadowed") == 1
    warnings = wardex_log.lines(logging.WARNING)
    assert len(warnings) == 1 and "resolved to" in warnings[0] and str(tmp_path) in warnings[0]
