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
import threading
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
from wardex_sdk._adapters._registry import get_registry
from wardex_sdk._assembly import Limitation, LinkReason, ParentSource, counters
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._enums import AdapterName, SpanKind, StatusCode
from wardex_sdk._semantics import REQUEST_CONVERSATION_KEY
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


@pytest.mark.parametrize("manual", [None, True, False])
@pytest.mark.parametrize("env", [None, "1", "false"])
def test_init_and_close_leave_the_frameworks_tracing_switch_where_they_found_it(
    agents_env, tracing_enabled, monkeypatch, manual, env
):
    """wardex never flips the framework's tracing switch. The framework's
    switch has two halves -- the manual one (`set_tracing_disabled`) and the
    environment variable it falls back to -- and install and uninstall READ
    both and WRITE neither. Nine combinations, so the sentence holds for the
    host that set nothing, the one that disabled tracing on purpose, and the
    one that re-enabled it the way wardex's own notice suggests."""
    provider = get_trace_provider()
    if env is None:
        monkeypatch.delenv("OPENAI_AGENTS_DISABLE_TRACING", raising=False)
    else:
        monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", env)
    provider._manual_disabled = manual
    provider._refresh_disabled_flag()

    _init()
    try:
        assert provider._manual_disabled is manual
        assert os.environ.get("OPENAI_AGENTS_DISABLE_TRACING") == env
    finally:
        wardex.close()
    assert provider._manual_disabled is manual
    assert os.environ.get("OPENAI_AGENTS_DISABLE_TRACING") == env


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


def _otlp_attributes(transport: RecordingTransport) -> dict[str, dict[str, Any]]:
    """`{span name: attributes}` as a RECEIVER decodes them — the export path
    end to end, not the in-process span. What a backend can filter on."""
    from wardex_sdk import _wardex_native

    out: dict[str, dict[str, Any]] = {}
    for env in transport.envelopes:
        decoded = _wardex_native.codec.decode_otlp_traces(
            _wardex_native.codec.encode_otlp_traces(env)
        )
        for rs in decoded["resource_spans"]:
            for ss in rs["scope_spans"]:
                for sp in ss["spans"]:
                    out[sp["name"]] = sp["attributes"]
    return out


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


def test_a_processor_removed_in_a_later_init_cycle_is_still_reported(agents_env, wardex_log):
    """`wardex.close()` leaves the process-global counters alone, so a second
    `init` in the same process starts with `active.trace` already above zero
    from the first cycle's runs. "No run was ever recorded" is judged by the
    ADAPTER INSTANCE's own count, which starts at zero with each install: the
    host that replaces the processor list after the second init has recorded
    nothing since, and must hear so."""
    from wardex_sdk._assembly._diag import reset_reports_for_test

    # The notice's `report_once` key is process-global too, and the removal
    # test above has already spent it by the time this one runs.
    reset_reports_for_test()
    _init()
    try:
        assert _run(_agents()).final_output == "done"
    finally:
        wardex.close()
    assert counters.get("adapters.openai_agents.active.trace") == 1
    _init()
    try:
        set_trace_processors([])
        assert _run(_agents()).final_output == "done"
        assert _adapter_spans(_spans()) == []
    finally:
        wardex.close()
    assert wardex_log.lines(logging.WARNING).count(_PROCESSOR_REMOVED_NOTICE) == 1
    assert counters.get("adapters.openai_agents.processor_removed") == 1
    assert counters.get("adapters.openai_agents.processor_removed_after_runs") == 0


def test_a_counter_reset_between_install_and_uninstall_does_not_change_the_verdict(
    agents_env, wardex_log
):
    """ "No run recorded since this install" is the adapter's OWN count, not a
    process-global counter measured against a baseline: `Runtime.after_in_child`
    resets the counters in a fork child and `wardex_sdk.testing.clean_state()`
    resets them for a test, and either would have made a run this install DID
    record look like none. A run, a reset, then the removal: a change of mind,
    counted, never the blind-spot notice."""
    from wardex_sdk._assembly._diag import reset_reports_for_test

    reset_reports_for_test()
    _init()
    try:
        assert _run(_agents()).final_output == "done"
        counters.reset()
        set_trace_processors([])
        assert _run(_agents()).final_output == "done"
    finally:
        wardex.close()
    assert wardex_log.lines(logging.WARNING).count(_PROCESSOR_REMOVED_NOTICE) == 0
    assert counters.get("adapters.openai_agents.processor_removed") == 0
    assert counters.get("adapters.openai_agents.processor_removed_after_runs") == 1


# --------------------------------------------------------------------------
# install / uninstall
# --------------------------------------------------------------------------


def test_install_is_idempotent_and_uninstall_restores_the_tuple_by_identity(agents_env):
    """One host processor is seeded first: from an empty list `before` would
    be CPython's empty-tuple singleton, which any fresh `tuple([])` also is,
    and the `is` check at the end would pass a copy."""
    set_trace_processors([_HostProcessor()])
    before = _processors()
    assert before != ()
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


class _AgentNameMoved:
    """`AgentSpanData` whose `name` keyword moved."""

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name


class _AgentToolsMoved:
    """`AgentSpanData` with `name` intact but `tools`/`handoffs` renamed —
    the attributes `_agent_end` reads, which a name-only probe never saw."""

    def __init__(self, name: str, tool_names: list | None = None) -> None:
        self.name = name
        self.tool_names = tool_names


class _FunctionMcpMoved:
    """`FunctionSpanData` without `mcp_data`, which `_function_end` reads."""

    def __init__(self, name: str, input: Any, output: Any) -> None:  # noqa: A002
        self.name, self.input, self.output = name, input, output


class _ResponseMoved:
    """`ResponseSpanData` whose `response` keyword moved."""

    def __init__(self, result: Any = None) -> None:
        self.result = result


class _TraceGroupMoved:
    """`TraceImpl` without `group_id`, which `_trace_start` reads. The
    abstract `Trace` never had the attribute — only the concrete class sets
    it — so this is the class the probe has to construct."""

    def __init__(self, name: str, trace_id: Any, metadata: Any, processor: Any) -> None:
        self.name, self.trace_id = name, trace_id


@pytest.mark.parametrize(
    ("module", "symbol", "moved"),
    [
        ("agents.tracing", "AgentSpanData", _AgentNameMoved),
        ("agents.tracing", "AgentSpanData", _AgentToolsMoved),
        ("agents.tracing", "FunctionSpanData", _FunctionMcpMoved),
        ("agents.tracing", "ResponseSpanData", _ResponseMoved),
        ("agents.tracing.traces", "TraceImpl", _TraceGroupMoved),
    ],
    ids=["agent.name", "agent.tools", "function.mcp_data", "response.response", "trace.group_id"],
)
def test_an_unrecognized_surface_declines_loudly_and_registers_nothing(
    agents_env, monkeypatch, wardex_log, module, symbol, moved
):
    """A class whose constructor keywords moved is a surface the mapping
    cannot read; the adapter says so once and touches nothing. Every
    attribute a handler reads is on the list, so a rename declines here
    instead of raising inside a callback mid-run."""
    import importlib

    monkeypatch.setattr(importlib.import_module(module), symbol, moved)
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
    if mode.startswith("absent"):
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
    kw = {} if mode.endswith("_auto") else {
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


def test_an_absent_distribution_declines_auto_detection_before_importing_anything(tmp_path):
    """A local package called `agents` with no `openai-agents` distribution,
    and nothing naming the adapter: auto-detection never imports it, so its
    side effects never run, and nothing is logged — absence is an answer,
    not a failure."""
    marker = _fake_agents_package(tmp_path, with_tracing=True)
    out = _decline_in_a_fresh_process(tmp_path, "absent_auto")
    assert not marker.exists()
    assert out["agents_imported"] is False
    assert out["lines"] == []
    assert out["shadowed"] == 0 and out["unsupported"] == 0


def test_a_named_adapter_installs_a_framework_that_has_no_package_metadata(tmp_path):
    """`enabled=` names the adapter and `import agents` succeeds, but no
    distribution is installed — a PyInstaller bundle or a vendored checkout.
    The user's word wins, as it did before the probe existed: the adapter
    installs on its own import. What the probe could not do (the shadow
    check needs a distribution to compare against) is said once as a
    warning; the empty `tracing` module of this stand-in then declines the
    surface, which is the second warning and the only counter."""
    marker = _fake_agents_package(tmp_path, with_tracing=True)
    out = _decline_in_a_fresh_process(tmp_path, "absent")
    assert marker.exists()
    assert out["agents_imported"] is True
    assert out["shadowed"] == 0 and out["unsupported"] == 1
    warnings = [m for lvl, m in out["lines"] if lvl == logging.WARNING]
    assert len(warnings) == 2
    assert "without package metadata" in warnings[0] and "shadow check" in warnings[0]
    assert "surface unrecognized" in warnings[1]
    assert not any("failed to load" in m for m in warnings)


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


def _dist_info(tmp_path, *, direct_url: str | None) -> Any:  # noqa: ANN001
    """A real `PathDistribution` for `openai-agents`, rooted in `tmp_path` so
    `locate_file("agents")` answers `tmp_path / "agents"` — the site-packages
    shape — with PEP 610's `direct_url.json` written when given."""
    from importlib.metadata import PathDistribution

    info = tmp_path / "site" / "openai_agents-0.22.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text("Metadata-Version: 2.1\nName: openai-agents\nVersion: 0.22.0\n")
    if direct_url is not None:
        (info / "direct_url.json").write_text(direct_url)
    return PathDistribution(info)


@pytest.mark.parametrize(
    ("direct_url", "layout", "shadowed"),
    [
        # pip install -e / uv --editable / a workspace member: the package
        # lives in the project tree, which direct_url.json names.
        ('{{"url": "file://{root}", "dir_info": {{"editable": true}}}}', "src/agents", False),
        ('{{"url": "file://{root}", "dir_info": {{"editable": true}}}}', "agents", False),
        # A non-editable direct install from a local directory: the package
        # was COPIED into site-packages, so a module resolving elsewhere is
        # still a shadow.
        ('{{"url": "file://{root}", "dir_info": {{}}}}', "src/agents", True),
        # A wheel from an index: no direct_url.json at all.
        (None, "src/agents", True),
    ],
)
def test_an_editable_install_of_the_real_framework_is_not_a_shadow(
    tmp_path, monkeypatch, direct_url, layout, shadowed
):
    """`find_spec("agents").origin` under an EDITABLE install is the project
    tree, never `site-packages/agents`, so comparing it with
    `locate_file("agents")` alone declined the adapter on the very machine
    the framework is developed on. The distribution's own record of where
    it was installed from decides: a module under the editable root is the
    installed distribution."""
    import importlib.util
    from types import SimpleNamespace

    from wardex_sdk._adapters._probe import shadow_path

    project = tmp_path / "project"
    module_dir = project / layout
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text("")
    dist = _dist_info(tmp_path, direct_url=direct_url.format(root=project) if direct_url else None)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(module_dir / "__init__.py")),
    )
    answer = shadow_path("agents", dist, where="adapters.openai_agents")
    if shadowed:
        assert answer == (str(module_dir.resolve()), str((tmp_path / "site" / "agents").resolve()))
    else:
        assert answer is None


def test_a_spec_less_stub_module_is_left_to_the_import_step(tmp_path, monkeypatch):
    """A stub placed in `sys.modules` by a test harness or a plugin loader
    carries no `__spec__`, and `find_spec` raises `ValueError` for such a
    name instead of answering. The probe cannot judge it and says so by
    answering `None` -- the import step decides -- rather than by raising,
    which the registry's guard reported as the adapter having failed to
    load."""
    import sys
    import types

    from wardex_sdk._adapters._probe import shadow_path

    stub = types.ModuleType("agents")
    stub.__spec__ = None
    monkeypatch.setitem(sys.modules, "agents", stub)
    dist = _dist_info(tmp_path, direct_url=None)
    assert shadow_path("agents", dist, where="adapters.openai_agents") is None


@pytest.mark.parametrize(
    ("oracle", "url", "expected"),
    [
        # PEP 610 spells a Windows path as `file:///C:/...`; its path part is
        # `/C:/...`, and a `Path()` of that on Windows is `\C:\...`, which
        # `resolve()` roots under the current drive -- every editable install
        # there compared unequal and was declined as a shadow.
        ("nt", "file:///C:/work/agents", "C:\\work\\agents"),
        ("nt", "file:///C:/work/my%20agents", "C:\\work\\my agents"),
        ("posix", "file:///home/me/agents", "/home/me/agents"),
        ("posix", "file:///home/me/my%20agents", "/home/me/my agents"),
    ],
)
def test_an_editable_root_is_read_as_the_platform_spells_paths(monkeypatch, oracle, url, expected):
    """The record's URL is turned into a path by the stdlib's own
    `url2pathname`, exercised here with BOTH platforms' spellings so the
    Windows shape is held on every host rather than only where CI does not
    run."""
    import json
    import nturl2path
    import urllib.request

    from wardex_sdk._adapters import _probe

    monkeypatch.setattr(
        _probe,
        "url2pathname",
        nturl2path.url2pathname if oracle == "nt" else urllib.request.url2pathname,
    )
    raw = json.dumps({"url": url, "dir_info": {"editable": True}})
    assert _probe.editable_root_pathname(raw) == expected
    assert _probe.editable_root_pathname(json.dumps({"url": url, "dir_info": {}})) is None


# --------------------------------------------------------------------------
# scenarios on the fake server
# --------------------------------------------------------------------------


def _fc(name: str, call_id: str, args: str = "{}", *, namespace: str | None = None) -> dict:
    item = {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": name,
        "arguments": args,
        "status": "completed",
    }
    if namespace is not None:
        item["namespace"] = namespace
    return item


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


def test_the_uninstall_sweep_unpins_in_reverse_so_the_host_scope_is_current_again():
    """Run, agent and tool pinned on ONE task, in that order, and then
    `wardex.close()` mid-run: each unpin restores "what was current when
    that pin was installed", so a sweep in pin order puts the RUN's fork
    back last-but-one and leaves the AGENT's fork current -- a dead unit's
    scope standing on the host's task. Reverse order retires them the way a
    stack unwinds, and the host's own scope is current when the sweep ends."""
    import threading

    from wardex_sdk._adapters._context import Placement
    from wardex_sdk._assembly import SpanIntent, UnitKind
    from wardex_sdk._types import SpanContext, SpanId, TraceId

    class _Key:
        """A slot key: `object()` itself cannot be weakly referenced."""

    host = SpanContext(trace_id=TraceId(b"\x0a" * 16), span_id=SpanId(b"\x0b" * 8))
    try:
        with installed_adapter(OpenAIAgentsAdapter) as live:
            # After the harness's own scope reset, so this is the host's scope.
            _hub.get_current_scope().active_span_context = host
            ctx = live.ctx
            driver = threading.current_thread()
            run = ctx.open_run(
                UnitKind.SESSION, intent=SpanIntent.INVOKE_WORKFLOW, placement=Placement.ROOT
            )
            assert run.pin(driver=driver)
            agent = ctx.open_run(
                UnitKind.AGENT, intent=SpanIntent.INVOKE_AGENT, placement=Placement.NESTED
            )
            assert agent.pin(driver=driver)
            tool = ctx.open_run(
                UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.NESTED
            )
            assert tool.pin(driver=driver)
            keys = [_Key() for _ in range(3)]  # weak-referenceable, in pin order
            for key, h in zip(keys, (run, agent, tool), strict=True):
                ctx.slot(key)["handle"] = h
            assert _hub.get_current_scope().active_span_context == tool._unit.context
            live.adapter._unpin_held()
            assert _hub.get_current_scope().active_span_context == host
            assert counters.get("adapters.openai_agents.unpin") == 3
            live.adapter.close_units(marker=Limitation.ADAPTER_UNINSTALLED)
    finally:
        _hub.reset_for_test()


def test_the_uninstall_sweep_still_closes_open_units_when_the_slot_walk_fails():
    """The slot table is a WeakKeyDictionary; on 3.10-3.13 a worker thread's
    `_span_end -> ctx.forget(span)` could pop a key while the sweep copied
    it, and the RuntimeError was swallowed by the registry's guard BEFORE
    `close_all` ran -- every open unit stranded. The copy now takes the same
    lock the slot writes take, and `close_all` sits in a `finally`: even an
    injected failure of the walk leaves no unit open and unmarked."""
    import weakref

    from wardex_sdk._adapters._context import Placement
    from wardex_sdk._assembly import SpanIntent, UnitKind

    class _Broken(weakref.WeakKeyDictionary):
        def values(self):  # noqa: ANN202
            raise RuntimeError("dictionary changed size during iteration")

    with installed_adapter(OpenAIAgentsAdapter) as live:
        ctx = live.ctx
        ctx.open_run(
            UnitKind.SESSION,
            intent=SpanIntent.INVOKE_WORKFLOW,
            placement=Placement.ROOT,
            # The vocabulary refuses an `invoke_workflow` span with no name.
            describe=lambda h: h.draft.set_workflow_name("w"),
        )
        real = ctx._slots
        ctx._slots = _Broken()
        try:
            with pytest.raises(RuntimeError):
                live.adapter.close_units(marker=Limitation.ADAPTER_UNINSTALLED)
        finally:
            ctx._slots = real
        [root] = live.spans
        assert Limitation.ADAPTER_UNINSTALLED in root.capture_integrity.limitations


def test_slot_reads_writes_and_the_sweep_copy_share_one_lock():
    """`slot`, `peek`, `forget` and the sweep's snapshot all take the same
    lock, so a worker thread's forget cannot land inside the sweep's copy."""

    class _Key:
        pass

    class _Counting:
        def __init__(self, inner) -> None:  # noqa: ANN001
            self.inner = inner
            self.entered = 0

        def __enter__(self):  # noqa: ANN204
            self.entered += 1
            return self.inner.__enter__()

        def __exit__(self, *exc):  # noqa: ANN002, ANN204
            return self.inner.__exit__(*exc)

    with installed_adapter(OpenAIAgentsAdapter) as live:
        ctx = live.ctx
        ctx._slots_lock = _Counting(ctx._slots_lock)
        key = _Key()
        ctx.slot(key)["x"] = 1
        assert ctx.peek(key) == {"x": 1}
        assert ctx.slots_snapshot() == [{"x": 1}]
        ctx.forget(key)
        assert ctx.peek(key) is None
        assert ctx._slots_lock.entered == 5


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


def test_two_concurrent_runs_under_one_trace_are_top_level_siblings(agents_env, scenario):
    """The framework's documented parallelization: several `Runner.run`s
    gathered under ONE `with trace(...)`. Each run's agent starts on its own
    task with no agent current there, so both are TOP-LEVEL siblings of the
    root — and agent_b's `max_turns` failure, fatal on a top-level agent,
    reaches the root. Judged by stack emptiness, the second starter read as
    NESTED and the root shipped OK for a run that raised."""
    from agents import function_tool
    from agents.exceptions import MaxTurnsExceeded
    from agents.tracing import trace

    def decide(inp: object) -> list[dict]:
        items = inp if isinstance(inp, list) else []
        user = [x for x in items if isinstance(x, dict) and x.get("role") == "user"]
        if user and user[0].get("content") == "loop":
            return _decide_loop(inp)
        return _DONE

    @function_tool
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    scenario(decide)
    agent_a = Agent(name="agent_a", instructions="a", model="gpt-4o-mini")
    agent_b = Agent(name="agent_b", instructions="b", tools=[get_weather], model="gpt-4o-mini")

    async def both() -> list[Any]:
        with trace("wf"):
            return await asyncio.gather(
                Runner.run(agent_a, "hi", max_turns=10),
                Runner.run(agent_b, "loop", max_turns=1),
                return_exceptions=True,
            )

    _init()
    try:
        results = asyncio.run(both())
        assert results[0].final_output == "done"
        assert isinstance(results[1], MaxTurnsExceeded)
        spans = _spans()
        assert counters.get("adapters.openai_agents.pin_refused") == 0
    finally:
        wardex.close()
    root = _one(spans, "invoke_workflow wf")
    a = _one(spans, "invoke_agent agent_a")
    b = _one(spans, "invoke_agent agent_b")
    for s in (a, b):
        assert s.parent_span_id == root.context.span_id
        assert _edge(s) == (ParentSource.UNIT_ACTIVE, 1.0, ())
    assert a.status is StatusCode.OK
    assert (b.status, b.error_type) == (StatusCode.ERROR, "max_turns_exceeded")
    assert (root.status, root.error_type) == (StatusCode.ERROR, "max_turns_exceeded")
    tool = _one(spans, "execute_tool get_weather")
    assert tool.parent_span_id == b.context.span_id
    assert _extra(root)["wardex.openai_agents.agents"] == 2


def _decide_tool_and_handoff_in_one_response(inp: object) -> list[dict]:
    """agent_a's first response requests `helper_tool` (an agent as a tool)
    AND a handoff to agent_b; the nested helper run and agent_b answer at once."""
    items = inp if isinstance(inp, list) else []
    if any(isinstance(x, dict) and x.get("content") == "INNER" for x in items):
        return _DONE
    if "helper_tool" not in _calls_made(inp):
        return [
            _fc("helper_tool", "call_1", '{"input":"INNER"}'),
            _fc("transfer_to_agent_b", "call_h1"),
        ]
    return _DONE


def test_a_handoff_beside_an_agent_as_tool_carries_the_outer_response_id(agents_env, scenario):
    """The nested run (agent-as-tool) finishes BEFORE the handoff span ends,
    and its own response used to overwrite the run-wide response id — so the
    marker joined the INNER agent's chat span. Turn and response id are the
    SENDER's: read from agent_a's own entry, they are the outer response
    that requested both the tool and the handoff."""
    scenario(_decide_tool_and_handoff_in_one_response)
    helper = Agent(name="helper", instructions="inner", model="gpt-4o-mini")
    agent_b = Agent(name="agent_b", instructions="b", model="gpt-4o-mini")
    agent_a = Agent(
        name="agent_a",
        instructions="a",
        tools=[helper.as_tool(tool_name="helper_tool", tool_description="helps")],
        handoffs=[agent_b],
        model="gpt-4o-mini",
    )
    _init()
    try:
        res = _run(agent_a)
        assert res.final_output == "done" and res.last_agent.name == "agent_b"
        spans = _spans()
    finally:
        wardex.close()
    chats = sorted(_chat_spans(spans), key=lambda c: c.start_time_ns)
    assert [c.gen_ai.response_id for c in chats] == ["resp_1", "resp_2", "resp_3"]
    marker = _one(spans, "handoff agent_a→agent_b")
    assert _extra(marker)["wardex.openai_agents.response_id"] == "resp_1"
    assert _extra(marker)["wardex.openai_agents.turn"] == 1
    tool = _one(spans, "execute_tool helper_tool")
    assert _extra(tool)["wardex.openai_agents.response_id"] == "resp_1"
    assert (
        _extra(_one(spans, "invoke_agent agent_a"))["wardex.openai_agents.last_response_id"]
        == "resp_1"
    )
    assert (
        _extra(_one(spans, "invoke_agent helper"))["wardex.openai_agents.last_response_id"]
        == "resp_2"
    )


def test_agents_finished_under_one_trace_leave_no_bookkeeping_behind(agents_env, scenario):
    """The run's slot holds no per-agent table at all: an agent's state
    lives on its OWN span's slot and is reached through the task-inherited
    current-agent variable, so many differently named agents in sequence
    cost the run nothing that grows. Read while the trace is still open —
    its end clears the whole slot, which would hide exactly this — and on
    the task the runs finished on, where no agent may still be current."""
    from agents.tracing import trace

    from wardex_sdk._adapters import _openai_agents
    from wardex_sdk._adapters._registry import get_registry

    scenario(_decide_single)
    names = [f"agent_{i}" for i in range(6)]
    seen: dict[str, Any] = {}

    async def many() -> None:
        with trace("outer") as t:
            for name in names:
                await Runner.run(Agent(name=name, instructions="x", model="gpt-4o-mini"), "hi")
            run = get_registry()._contexts["openai_agents"].slot(t)
            seen["run"] = dict(run)
            seen["current"] = _openai_agents._CURRENT_AGENT.get()
            seen["pending"] = _openai_agents._PENDING_HANDOFF.get()

    _init()
    try:
        asyncio.run(many())
        spans = _spans()
    finally:
        wardex.close()
    grows = {k: v for k, v in seen["run"].items() if isinstance(v, (dict, list, set))}
    assert grows == {}, f"per-run collections that would grow with the workload: {grows}"
    assert seen["current"] is None and seen["pending"] is None
    root = _one(spans, "invoke_workflow outer")
    assert _extra(root)["wardex.openai_agents.agents"] == len(names)


def test_a_finished_run_leaves_every_slot_empty(agents_env, scenario):
    """Per-span bookkeeping ends with the span and per-run bookkeeping with
    the trace: whatever the framework still holds, no slot the adapter wrote
    keeps a handle, an agent entry or a call table alive afterwards. An
    END-only span (each LLM call's response span) never gets a slot at all."""
    from wardex_sdk._adapters._registry import get_registry

    scenario(_decide_chain)
    _init()
    try:
        assert _run(_chain_agents()).final_output == "done"
        _spans()
        ctx = get_registry()._contexts["openai_agents"]
        leftovers = {type(k).__name__: dict(v) for k, v in ctx._slots.items() if v}
    finally:
        wardex.close()
    assert leftovers == {}, leftovers


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


def test_a_handoff_whose_receiver_never_starts_does_not_leak_into_the_next_run(
    agents_env, scenario
):
    """The sender's end publishes the pending handoff for the receiver that
    starts NEXT on this task. Here the receiver never starts: its tool's
    `is_enabled` raises inside `get_all_tools`, which the framework runs
    BEFORE it opens the receiver's span, so the run fails with the handoff
    still pending. The next run on the same task then opens a TOP-LEVEL
    agent of the receiver's name -- and shipped `parent_agent="agent_a"`
    with a HANDOFF_FROM link into the previous trace, at confidence 1.0 and
    with no marker. The pending handoff belongs to its run: a receiver in
    another run is not that handoff's receiver."""
    from agents import function_tool

    first = [True]

    def decide(inp: object) -> list[dict]:
        if first[0]:
            first[0] = False
            return [_fc("transfer_to_agent_b", "call_h1")]
        return _DONE

    scenario(decide)

    def refuse(ctx, agent) -> bool:  # noqa: ANN001
        raise RuntimeError("tool gate broken")

    @function_tool(is_enabled=refuse)
    def gated() -> str:
        return "never"

    agent_b = Agent(name="agent_b", instructions="b", model="gpt-4o-mini", tools=[gated])
    agent_a = Agent(name="agent_a", instructions="a", handoffs=[agent_b], model="gpt-4o-mini")
    fresh_b = Agent(name="agent_b", instructions="b", model="gpt-4o-mini")

    async def two() -> None:
        with pytest.raises(RuntimeError):
            await Runner.run(agent_a, "hi")
        await Runner.run(fresh_b, "hi")

    _init()
    try:
        asyncio.run(two())
        spans = _spans()
    finally:
        wardex.close()
    marker = _one(spans, "handoff agent_a→agent_b")
    assert marker.status is StatusCode.OK
    agents = [s for s in _adapter_spans(spans) if s.name.startswith("invoke_agent")]
    assert [s.agent.name for s in agents] == ["agent_a", "agent_b"]
    later_b = agents[1]
    assert later_b.agent.parent_agent is None
    assert later_b.links == ()
    roots = [s for s in _adapter_spans(spans) if s.parent_span_id is None]
    assert len(roots) == 2 and later_b.context.trace_id == roots[1].context.trace_id
    assert counters.get("adapters.openai_agents.handoff_receiver_never_started") == 1
    # No agent span was open when the first run raised, so the exception was read where the
    # framework closed that run's own trace: the root is ERROR while both agents are OK.
    assert (roots[0].status, roots[0].error_type) == (StatusCode.ERROR, "RuntimeError")
    assert [s.status for s in agents] == [StatusCode.OK, StatusCode.OK]


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


def _assert_three_turn_tree(
    spans: list[Any], *, streamed: bool, conversation: str = "conv-123"
) -> None:
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
    # the payload, as the framework handed it: the model's JSON string and
    # the handler's string, NOT their Python repr
    assert tool.input_data == b'{"city":"Seoul"}'
    assert tool.output_data == b"sunny in Seoul"
    assert tool.capture_integrity is None or not tool.capture_integrity.truncated
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
    # the conversation, on every span of the run: the adapter's, and the wire
    # `chat` spans, which the byte seam latches at request time beside the
    # parent off the run's pinned carrier
    for s in _adapter_spans(spans):
        assert s.conversation is not None and s.conversation.conversation_id == conversation
    for c in chats:
        assert c.conversation is not None and c.conversation.conversation_id == conversation
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
    transport = _init()
    try:
        assert _run(_agents(), run_config=_run_config()).final_output == "done"
        spans = _spans()
        _assert_three_turn_tree(spans, streamed=False)
        _assert_counters_clean()
    finally:
        wardex.close()
    # What a RECEIVER gets: the conversation id as an OTLP attribute, on
    # every adapter span — measured absent before the
    # codec marshalled the block, while the in-process span carried it.
    attrs = _otlp_attributes(transport)
    for name in (_ROOT, "invoke_agent agent_a", "invoke_agent agent_b", "handoff agent_a→agent_b"):
        assert attrs[name]["gen_ai.conversation.id"] == "conv-123", name
    assert attrs["execute_tool get_weather"]["gen_ai.conversation.id"] == "conv-123"
    # ...and on the LLM calls, the spans that carry the tokens a "what did this
    # conversation cost" query sums.
    assert attrs["chat gpt-4o-mini"]["gen_ai.conversation.id"] == "conv-123"
    # Handoff causality: the receiver's parent agent
    # is a `wardex.*` attribute (the codec flattens `AgentAttributes.parent_agent`
    # there), not a `gen_ai.*` one — the docs once promised the wrong key.
    assert attrs["invoke_agent agent_b"]["wardex.agent.parent"] == "agent_a"
    assert attrs["handoff agent_a→agent_b"]["wardex.agent.parent"] == "agent_a"
    assert "wardex.agent.parent" not in attrs["invoke_agent agent_a"]
    assert attrs["execute_tool get_weather"]["gen_ai.tool.call.arguments"] == '{"city":"Seoul"}'
    assert attrs["execute_tool get_weather"]["gen_ai.tool.call.result"] == "sunny in Seoul"
    print("\n" + _print_tree(spans))


def test_a_host_conversation_wins_over_the_frameworks_group_id(agents_env):
    """HOST WINS. A run inside `with wardex.conversation("chat", id=...)`
    keeps the host's id as `gen_ai.conversation.id` on every adapter span —
    one trace, one conversation — and the framework's `group_id` rides
    along on the root as `wardex.openai_agents.group_id`, counted. Without
    a host conversation the group id is the conversation, as before."""
    transport = _init()
    try:
        with wardex.conversation("chat", id="host-1"):
            assert _run(_agents(), run_config=_run_config()).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.group_id_shadowed_by_host") == 1
    finally:
        wardex.close()
    adapter_spans = [s for s in _adapter_spans(spans) if s.name != "chat"]
    for s in adapter_spans:
        assert s.conversation is not None and s.conversation.conversation_id == "host-1", s.name
    chats = _chat_spans(spans)
    assert len(chats) == 3
    for c in chats:
        assert c.conversation is not None and c.conversation.conversation_id == "host-1"
    root = _one(spans, _ROOT)
    assert _extra(root)["wardex.openai_agents.group_id"] == "conv-123"
    assert "wardex.openai_agents.group_id" not in _extra(_one(spans, "invoke_agent agent_a"))
    attrs = _otlp_attributes(transport)
    assert attrs[_ROOT]["gen_ai.conversation.id"] == "host-1"
    assert attrs[_ROOT]["wardex.openai_agents.group_id"] == "conv-123"
    assert attrs["execute_tool get_weather"]["gen_ai.conversation.id"] == "host-1"
    assert attrs["chat gpt-4o-mini"]["gen_ai.conversation.id"] == "host-1"

    _init()
    try:
        assert _run(_agents(), run_config=_run_config()).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.group_id_shadowed_by_host") == 1  # unchanged
    finally:
        wardex.close()
    root = _one(spans, _ROOT)
    assert root.conversation.conversation_id == "conv-123"
    assert "wardex.openai_agents.group_id" not in _extra(root)


def test_the_runs_conversation_wins_over_the_requests_own(agents_env):
    """`RunConfig(group_id=…)` AND `Runner.run(conversation_id=…)`: the run
    states one conversation, and every request in it names another in its
    body. The run's wins on the `chat` spans too — they are the run's calls,
    and one run ships one conversation id — and the request's rides along
    under its own attribute, counted once per call."""
    transport = _init()
    try:
        result = _run(_agents(), run_config=_run_config(), conversation_id="conv_1")
        assert result.final_output == "done"
        spans = _spans()
        assert counters.get("semantics.request_conversation_shadowed") == 3
    finally:
        wardex.close()
    for s in spans:
        assert s.conversation is not None and s.conversation.conversation_id == "conv-123", s.name
    chats = _chat_spans(spans)
    assert len(chats) == 3
    for c in chats:
        assert _extra(c)["wardex.openai.conversation_id"] == "conv_1"
    for s in _adapter_spans(spans):
        assert "wardex.openai.conversation_id" not in _extra(s)
    attrs = _otlp_attributes(transport)
    assert attrs["chat gpt-4o-mini"]["gen_ai.conversation.id"] == "conv-123"
    assert attrs["chat gpt-4o-mini"]["wardex.openai.conversation_id"] == "conv_1"


def test_concurrent_runs_each_carry_the_conversation_their_call_names(agents_env):
    """`Runner.run(conversation_id=…)` with the adapter ON and no `group_id`,
    two runs at once naming different conversations under `asyncio.gather`.
    The framework hands that id to the model call and never to its trace; the
    entry hook reads it off each call, on each call's own task. Every span of
    each run — root, agents, tool, handoff, LLM calls — carries its own id,
    never the other run's."""

    async def both() -> list[Any]:
        return await asyncio.gather(
            *(
                Runner.run(
                    _agents(), "hi", run_config=RunConfig(workflow_name="wf"), conversation_id=c
                )
                for c in ("convA", "convB")
            )
        )

    _init()
    try:
        assert [r.final_output for r in asyncio.run(both())] == ["done", "done"]
        spans = _spans()
        assert counters.get("semantics.request_conversation_shadowed") == 0
    finally:
        wardex.close()
    by_trace: dict[Any, list[Any]] = {}
    for s in spans:
        by_trace.setdefault(s.context.trace_id, []).append(s)
    named = []
    for trace in by_trace.values():
        chats = _chat_spans(trace)
        assert len(chats) == 3
        (cid,) = {c.conversation.conversation_id for c in chats if c.conversation is not None}
        assert all(s.conversation is not None for s in trace), [s.name for s in trace]
        carried = {s.conversation.conversation_id for s in trace}
        assert carried == {cid}, [(s.name, s.conversation) for s in trace]
        assert len(_adapter_spans(trace)) == 5
        named.append(cid)
    assert sorted(named) == ["convA", "convB"]


def _no_group() -> RunConfig:
    """Fresh per run, like `_run_config`, and naming no `group_id`."""
    return RunConfig(workflow_name="wf")


def _drive(entry: str, **kwargs: Any) -> Any:
    """The three-turn run through one entry point, drained, its final output.

    `run_streamed_positional` passes every argument up to `conversation_id`
    by position, the one entry point whose signature allows it."""
    if entry == "run":
        return asyncio.run(Runner.run(_agents(), "hi", **kwargs)).final_output
    if entry == "run_sync":
        return Runner.run_sync(_agents(), "hi", **kwargs).final_output

    async def go() -> Any:
        if entry == "run_streamed_positional":
            cid = kwargs.pop("conversation_id")
            config = kwargs.pop("run_config")
            result = Runner.run_streamed(_agents(), "hi", None, 10, None, config, None, False, cid)
        else:
            result = Runner.run_streamed(_agents(), "hi", **kwargs)
        async for _ in result.stream_events():
            pass
        return result.final_output

    return asyncio.run(go())


@pytest.mark.parametrize("entry", ["run", "run_sync", "run_streamed", "run_streamed_positional"])
def test_a_runs_conversation_id_is_the_conversation_of_every_span_of_the_run(agents_env, entry):
    """`Runner.run(conversation_id=…)` and no `group_id`: the framework puts
    the id in every request and never in its trace, so the run's own spans
    used to carry none while its LLM calls carried it. The entry hook reads the
    argument, and the id is the run's conversation: on the root, both agents,
    the tool, the handoff marker and the three LLM calls, through each entry
    point. Nothing shadows anything, so nothing rides along."""
    transport = _init()
    try:
        assert _drive(entry, run_config=_no_group(), conversation_id="conv_1") == "done"
        spans = _spans()
        _assert_three_turn_tree(
            spans, streamed=entry.startswith("run_streamed"), conversation="conv_1"
        )
        _assert_counters_clean()
        assert counters.get("semantics.request_conversation_shadowed") == 0
    finally:
        wardex.close()
    for s in spans:
        assert REQUEST_CONVERSATION_KEY not in _extra(s), s.name
    attrs = _otlp_attributes(transport)
    for name in (
        _ROOT,
        "invoke_agent agent_a",
        "invoke_agent agent_b",
        "handoff agent_a→agent_b",
        "execute_tool get_weather",
        "chat gpt-4o-mini",
    ):
        assert attrs[name]["gen_ai.conversation.id"] == "conv_1", name


def test_a_host_conversation_wins_over_the_runs_conversation_id(agents_env):
    """HOST WINS, the rule a `group_id` follows: inside `wardex.conversation`
    every span of the run carries the host's id, and the run's own id rides
    along on each LLM call whose request named it."""
    _init()
    try:
        with wardex.conversation("host", id="host-1"):
            assert _drive("run", run_config=_no_group(), conversation_id="conv_1") == "done"
        spans = _spans()
        shadowed = counters.get("adapters.openai_agents.conversation_id_shadowed_by_host")
        assert shadowed == 1
    finally:
        wardex.close()
    for s in spans:
        assert s.conversation is not None and s.conversation.conversation_id == "host-1", s.name
    chats = _chat_spans(spans)
    assert len(chats) == 3
    for c in chats:
        assert _extra(c)[REQUEST_CONVERSATION_KEY] == "conv_1"
    for s in _adapter_spans(spans):
        assert REQUEST_CONVERSATION_KEY not in _extra(s), s.name


def _decide_weather_then_done(inp: object) -> list[dict]:
    if _outputs_done(inp) == 0:
        return [_fc("get_weather", "call_w", '{"city":"Seoul"}')]
    return _DONE


def _weather(gate: threading.Barrier | None = None) -> Any:
    from agents import function_tool

    @function_tool
    def get_weather(city: str) -> str:
        if gate is not None:
            gate.wait(10)
        return f"sunny in {city}"

    return get_weather


def _by_agent(spans: list[Any]) -> dict[str, set[str | None]]:
    """Each top-level agent's subtree's conversation ids, its LLM calls'
    included: `{agent name: {ids}}`."""
    kids: dict[Any, list[Any]] = {}
    for s in spans:
        kids.setdefault(s.parent_span_id, []).append(s)
    out: dict[str, set[str | None]] = {}
    for agent in [s for s in spans if s.name.startswith("invoke_agent ")]:
        seen: set[str | None] = set()
        todo = [agent]
        while todo:
            s = todo.pop()
            seen.add(s.conversation.conversation_id if s.conversation is not None else None)
            todo += kids.get(s.context.span_id, [])
        out[agent.agent.name] = seen
    return out


def test_runs_gathered_under_the_hosts_own_trace_each_carry_their_call_id(agents_env, scenario):
    """The framework's parallelization pattern: several `Runner.run`s gathered
    under ONE `with trace(...)`. The root is the host's trace and no one call's,
    so it states no id; each call's top-level agent states its own, and its
    tool and LLM calls inherit it. No id crosses from one call to the other."""
    from agents.tracing import trace

    scenario(_decide_weather_then_done)
    gate = threading.Barrier(2)
    agents = {
        name: Agent(name=name, instructions=name, tools=[_weather(gate)], model="gpt-4o-mini")
        for name in ("agent_a", "agent_b")
    }

    async def both() -> list[Any]:
        with trace("wf"):
            return await asyncio.gather(
                Runner.run(agents["agent_a"], "hi", conversation_id="convA"),
                Runner.run(agents["agent_b"], "hi", conversation_id="convB"),
            )

    _init()
    try:
        assert [r.final_output for r in asyncio.run(both())] == ["done", "done"]
        spans = _spans()
    finally:
        wardex.close()
    assert _one(spans, _ROOT).conversation is None
    assert _by_agent(spans) == {"agent_a": {"convA"}, "agent_b": {"convB"}}
    assert len(_chat_spans(spans)) == 4
    assert len([s for s in spans if s.name == "execute_tool get_weather"]) == 2


def test_a_group_id_on_the_hosts_own_trace_wins_over_the_calls_id(agents_env, scenario):
    """`with trace(..., group_id=…)` around `Runner.run(conversation_id=…)`:
    the group id is the run's conversation, as it is on `RunConfig`, so the
    agent states nothing of its own and the call's id rides along on the LLM
    call only."""
    from agents.tracing import trace

    scenario(_decide_single)

    async def one() -> Any:
        with trace("wf", group_id="g-1"):
            agent = Agent(name="agent_a", instructions="a", model="gpt-4o-mini")
            return await Runner.run(agent, "hi", conversation_id="conv_1")

    _init()
    try:
        assert asyncio.run(one()).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.conversation_id_shadowed_by_group_id") == 1
    finally:
        wardex.close()
    for s in spans:
        assert s.conversation is not None and s.conversation.conversation_id == "g-1", s.name
    (chat,) = _chat_spans(spans)
    assert _extra(chat)[REQUEST_CONVERSATION_KEY] == "conv_1"


def test_concurrent_run_sync_calls_on_threads_each_carry_their_own_id(agents_env, scenario):
    """`run_sync` on two threads at once, held inside their tools until both
    are there: each thread's call reads its own id, on its own loop, and every
    span of each run carries it."""
    scenario(_decide_weather_then_done)
    gate = threading.Barrier(2)
    out: dict[str, Any] = {}

    def drive(name: str, cid: str) -> None:
        agent = Agent(name=name, instructions=name, tools=[_weather(gate)], model="gpt-4o-mini")
        try:
            out[name] = Runner.run_sync(
                agent, "hi", conversation_id=cid, run_config=RunConfig(workflow_name=name)
            ).final_output
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            out[name] = exc

    _init()
    try:
        threads = [
            threading.Thread(target=drive, args=(n, c))
            for n, c in (("agent_a", "convA"), ("agent_b", "convB"))
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join(30)
        assert out == {"agent_a": "done", "agent_b": "done"}
        spans = _spans()
    finally:
        wardex.close()
    for name, cid in (("agent_a", "convA"), ("agent_b", "convB")):
        (root,) = [s for s in spans if s.name == f"invoke_workflow {name}"]
        trace = [s for s in spans if s.context.trace_id == root.context.trace_id]
        assert len(trace) == 5, [s.name for s in trace]
        assert {s.conversation.conversation_id for s in trace if s.conversation} == {cid}
        assert all(s.conversation is not None for s in trace)


def test_a_call_whose_entry_surface_moved_still_runs_and_names_the_llm_calls(
    agents_env, monkeypatch, wardex_log
):
    """Group 3 of the probe declines on its own: with `Runner` unrecognized
    the entry points are left untouched, said once, and the run still ships
    its whole tree; the id then reaches only the LLM calls, which read it off
    the request."""
    import wardex_sdk._adapters._openai_agents_entry as entry

    monkeypatch.setattr(entry, "_entry_surface", lambda run_mod: None)
    before = {name: vars(Runner)[name] for name in ("run", "run_sync", "run_streamed")}
    _init()
    try:
        assert {name: vars(Runner)[name] for name in before} == before
        assert _drive("run", run_config=_no_group(), conversation_id="conv_1") == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.unsupported_entry_surface") == 1
    finally:
        wardex.close()
    assert len(spans) == 8
    for s in spans:
        want = "conv_1" if s.kind is SpanKind.CLIENT else None
        assert _conv_of(s) == want, s.name
    assert len([m for m in wardex_log.lines(logging.WARNING) if "entry points" in m]) == 1


def _conv_of(span: Any) -> str | None:
    return span.conversation.conversation_id if span.conversation is not None else None


@pytest.mark.parametrize("entry", ["run", "run_sync"])
def test_a_redacted_error_passes_the_entry_hook_holding_none_of_the_calls_arguments(
    agents_env, monkeypatch, entry
):
    """The framework raises an error whose data it redacted from frames that
    own no payload: it drops the traceback and clears its own locals first,
    so an error tracker that records each frame's locals records no input.
    The wrapper sits in that traceback and must hold no argument either."""
    from agents import run as run_mod
    from agents.exceptions import _mark_error_data_redacted

    secret = "SECRET-INPUT-7f3a"

    def boom() -> RuntimeError:
        err = RuntimeError("redacted")
        _mark_error_data_redacted(err)
        return err

    class _Runner:
        async def run(self, *args: Any, **kwargs: Any) -> Any:
            raise boom()

        def run_sync(self, *args: Any, **kwargs: Any) -> Any:
            raise boom()

    monkeypatch.setattr(run_mod, "DEFAULT_AGENT_RUNNER", _Runner())
    agent = Agent(name="agent_a", instructions="a", model="gpt-4o-mini")
    _init()
    try:
        with pytest.raises(RuntimeError) as info:
            if entry == "run":
                asyncio.run(Runner.run(agent, secret, conversation_id="conv_1"))
            else:
                Runner.run_sync(agent, secret, conversation_id="conv_1")
    finally:
        wardex.close()
    frames = []
    tb = info.value.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next
    # The first frame is this test's, which holds the secret it passed.
    assert frames[0].f_code is sys._getframe().f_code
    assert "entry" in [f.f_code.co_name for f in frames[1:]]
    for f in frames[1:]:
        assert secret not in repr(f.f_locals), (f.f_code.co_name, f.f_locals)


def test_a_resumed_run_carries_the_conversation_its_state_recorded(agents_env, scenario):
    """A run interrupted for a tool approval and resumed with
    `Runner.run(agent, state)` names no id in the second call; the framework
    continues the conversation the state recorded, and so does every span of
    the resumed half — its root, agent, tool and LLM call."""
    from agents import function_tool

    @function_tool(needs_approval=True)
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    # Decided off the tool OUTPUTS in the input: a run the provider holds the
    # conversation for sends only what is new, so the call itself is gone
    # from the resumed half's request.
    scenario(_decide_weather_then_done)
    agent = Agent(name="agent_a", instructions="a", tools=[get_weather], model="gpt-4o-mini")
    _init()
    try:

        async def drive() -> Any:
            first = await Runner.run(agent, "hi", conversation_id="conv_r")
            assert first.interruptions, "the approval interrupt did not happen"
            state = first.to_state()
            for item in first.interruptions:
                state.approve(item)
            return await Runner.run(agent, state)

        assert asyncio.run(drive()).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.run_root_reattached") == 1
    finally:
        wardex.close()
    resumed = [s for s in spans if _extra(s).get("wardex.openai_agents.resumed") is True]
    assert len(resumed) == 1
    assert len(spans) >= 7, [s.name for s in spans]
    for s in spans:
        assert _conv_of(s) == "conv_r", s.name


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


def _decide_nested_tool_then_handoff(inp: object) -> list[dict]:
    """agent_a calls `helper_tool` (an agent as a tool), then hands off to
    agent_b; the inner helper run and agent_b both answer at once."""
    items = inp if isinstance(inp, list) else []
    if any(isinstance(x, dict) and x.get("content") == "INNER" for x in items):
        return _DONE
    made = _calls_made(inp)
    if "helper_tool" not in made:
        return [_fc("helper_tool", "call_1", '{"input":"INNER"}')]
    if "transfer_to_agent_b" not in made:
        return [_fc("transfer_to_agent_b", "call_h1")]
    return _DONE


def test_a_refused_agent_pin_marks_every_child_the_adapter_opens_under_it(
    agents_env, scenario, monkeypatch, wardex_log
):
    """The registry refuses a pin whose declared owner is not the task it
    observes and marks ONLY the refused unit. A framework release that started
    `AgentSpanData` on a task other than the run's would then ship that
    agent's tools, nested agents and handoff marker under the ambient unit of
    the moment — the run root here — at 1.0 with nothing on them. So the
    adapter keeps the refusal on the agent's slot and puts
    `correlation_conflict` on each child it opens while that agent is current;
    agent_b, whose pin is accepted, stays clean, and one WARNING line names
    the agent. Only agent_a's start sees a foreign driver."""
    from wardex_sdk._adapters import _openai_agents
    from wardex_sdk._assembly._diag import reset_reports_for_test

    reset_reports_for_test()
    scenario(_decide_nested_tool_then_handoff)
    real_start = _openai_agents._agent_start

    def foreign_for_agent_a(adapter: Any, run: Any, span: Any) -> None:
        if span.span_data.name != "agent_a":
            real_start(adapter, run, span)
            return
        with monkeypatch.context() as m:
            m.setattr(_openai_agents, "_driver", lambda: object())
            real_start(adapter, run, span)

    monkeypatch.setattr(_openai_agents, "_agent_start", foreign_for_agent_a)
    helper = Agent(name="helper", instructions="inner", model="gpt-4o-mini")
    agent_b = Agent(name="agent_b", instructions="b", model="gpt-4o-mini")
    agent_a = Agent(
        name="agent_a",
        instructions="a",
        tools=[helper.as_tool(tool_name="helper_tool", tool_description="helps")],
        handoffs=[agent_b],
        model="gpt-4o-mini",
    )
    _init()
    try:
        assert _run(agent_a).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.pin_refused") == 1
        assert counters.get("assembly._units.pin_foreign_task") == 1
    finally:
        wardex.close()
    root = _one(spans, "invoke_workflow Agent workflow")
    refused = _one(spans, "invoke_agent agent_a")
    tool = _one(spans, "execute_tool helper_tool")
    nested = _one(spans, "invoke_agent helper")
    marker = _one(spans, "handoff agent_a→agent_b")
    clean = _one(spans, "invoke_agent agent_b")
    # The registry's half: the refused unit itself.
    assert Limitation.CORRELATION_CONFLICT in _edge(refused)[2]
    # The adapter's half: every child it opened while agent_a was current.
    # The tool and the marker are misparented onto the root, because agent_a
    # never became ambient; the nested agent hangs under the tool (its own
    # task's pin held) but the subtree it sits in is the misparented one, so
    # it says so too.
    for s in (tool, nested, marker):
        assert Limitation.CORRELATION_CONFLICT in _edge(s)[2], (s.name, _edge(s))
    assert tool.parent_span_id == root.context.span_id
    assert marker.parent_span_id == root.context.span_id
    assert nested.parent_span_id == tool.context.span_id
    assert _edge(clean) == (ParentSource.UNIT_ACTIVE, 1.0, ())
    assert clean.parent_span_id == root.context.span_id
    notices = [w for w in wardex_log.lines(logging.WARNING) if "carry correlation_conflict" in w]
    assert len(notices) == 1 and "spans under agent_a carry" in notices[0], notices


def test_a_refused_agent_pin_marks_the_mcp_list_tools_step_under_it(
    agents_env, scenario, monkeypatch
):
    """The fifth child site. A nested agent (agent-as-tool) with an MCP
    server lists its tools while the OUTER agent is current, so with the
    outer agent's pin refused the `execute_step mcp.list_tools` it opens
    carries `correlation_conflict` like every other child — the one site
    that used to forget the marker, and the reason all five now open through
    one helper."""
    from wardex_sdk._adapters import _openai_agents
    from wardex_sdk._assembly._diag import reset_reports_for_test

    reset_reports_for_test()

    def decide(inp: object) -> list[dict]:
        items = inp if isinstance(inp, list) else []
        if any(isinstance(x, dict) and x.get("content") == "INNER" for x in items):
            return _DONE
        if "helper_tool" not in _calls_made(inp):
            return [_fc("helper_tool", "call_1", '{"input":"INNER"}')]
        return _DONE

    scenario(decide)
    real_start = _openai_agents._agent_start

    def foreign_for_agent_a(adapter: Any, run: Any, span: Any) -> None:
        if span.span_data.name != "agent_a":
            real_start(adapter, run, span)
            return
        with monkeypatch.context() as m:
            m.setattr(_openai_agents, "_driver", lambda: object())
            real_start(adapter, run, span)

    monkeypatch.setattr(_openai_agents, "_agent_start", foreign_for_agent_a)
    helper = Agent(
        name="helper", instructions="inner", mcp_servers=[_in_process_mcp()], model="gpt-4o-mini"
    )
    agent_a = Agent(
        name="agent_a",
        instructions="a",
        tools=[helper.as_tool(tool_name="helper_tool", tool_description="helps")],
        model="gpt-4o-mini",
    )
    _init()
    try:
        assert _run(agent_a).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.pin_refused") == 1
    finally:
        wardex.close()
    step = _one(spans, "execute_step mcp.list_tools")
    assert Limitation.CORRELATION_CONFLICT in _edge(step)[2], _edge(step)
    for name in ("execute_tool helper_tool", "invoke_agent helper"):
        assert Limitation.CORRELATION_CONFLICT in _edge(_one(spans, name))[2], name


def test_sensitive_data_off_leaves_the_marker_and_no_join_on_the_tool_span(agents_env):
    """`RunConfig(trace_include_sensitive_data=False)`: the framework strips
    the response and the tool arguments from its own spans, so the call id
    cannot be matched and the response id is unavailable. The tool span ships
    the marker and no `response_id` extra rather than a guess, the handoff
    marker carries no join either, the wire span stays the only holder of
    `gen_ai.response.id`, and the host's result is untouched."""
    _init()
    try:
        cfg = RunConfig(trace_include_sensitive_data=False)
        assert _run(_agents(), run_config=cfg).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.response_id_unavailable") == 3
        assert counters.get("adapters.openai_agents.tool_call_id_unmatched") == 1
        assert counters.get("adapters.openai_agents.tool_call_id_ambiguous") == 0
    finally:
        wardex.close()
    tool = _one(spans, "execute_tool get_weather")
    marker = _one(spans, "handoff agent_a→agent_b")
    assert tool.tool.call_id is None
    assert Limitation.TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS in _edge(tool)[2]
    assert "wardex.openai_agents.tool_call_id_source" not in _extra(tool)
    for s in (tool, marker):
        assert "wardex.openai_agents.response_id" not in _extra(s)
    assert tool.input_data == b"" and tool.output_data == b""
    assert [
        c.gen_ai.response_id for c in sorted(_chat_spans(spans), key=lambda c: c.start_time_ns)
    ] == [
        "resp_1",
        "resp_2",
        "resp_3",
    ]
    for name in (
        _ROOT.replace("wf", "Agent workflow"),
        "invoke_agent agent_a",
        "invoke_agent agent_b",
    ):
        assert _one(spans, name).status is StatusCode.OK


def test_a_tool_payload_is_the_frameworks_string_bounded_by_the_handshake():
    """A `str` is recorded as its own bytes; over the budget it comes back as
    exactly `budget + 1` bytes so the storage cap sets the truncated flag; a
    non-string keeps the LangGraph shaping (a dict's repr is a literal)."""
    from wardex_sdk._adapters._payload import _shaped_payload

    assert _shaped_payload('{"city":"Seoul"}', 64) == b'{"city":"Seoul"}'
    assert _shaped_payload("sunny in Seoul", 64) == b"sunny in Seoul"
    assert len(_shaped_payload("x" * 10_000, 64)) == 65
    assert len(_shaped_payload("é" * 10_000, 64)) == 65
    assert _shaped_payload("x" * 64, 64) == b"x" * 64
    assert _shaped_payload({"city": "Seoul"}, 64) == b"{'city': 'Seoul'}"


def test_a_namespaced_tool_recovers_its_call_id(agents_env, scenario):
    """`tool_namespace()` (public in 0.22) names the function span
    `f"{namespace}.{name}"` while the response item keeps the bare `name`
    plus a `namespace` field. The call-id match keys both sides by the
    framework's own trace-name rule, so a namespaced tool joins its turn
    instead of always shipping the marker."""
    from agents import function_tool
    from agents.tool import tool_namespace

    @function_tool
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    def decide(inp: object) -> list[dict]:
        if _outputs_done(inp) == 0:
            return [_fc("get_weather", "call_1", '{"city":"Seoul"}', namespace="ns")]
        return _DONE

    scenario(decide)
    agent = Agent(
        name="agent_a",
        instructions="a",
        tools=tool_namespace(name="ns", description="weather tools", tools=[get_weather]),
        model="gpt-4o-mini",
    )
    _init()
    try:
        assert _run(agent).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.tool_call_id_unmatched") == 0
    finally:
        wardex.close()
    tool = _one(spans, "execute_tool ns.get_weather")
    assert tool.tool.call_id == "call_1"
    assert _extra(tool)["wardex.openai_agents.tool_call_id_source"] == "response_output_match"
    assert Limitation.TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS not in _edge(tool)[2]


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
    transport = _init()
    try:
        with pytest.raises(InputGuardrailTripwireTriggered):
            _run(_guarded(where="input"))
        spans = _spans()
    finally:
        wardex.close()
    # sequential by default: the model call never happened
    _assert_tripwire(spans, name="block_input", posts=len(posts), made=0)
    # and the verdict reaches a receiver under the semconv evaluation keys
    evaluate = _otlp_attributes(transport)["evaluate block_input"]
    assert evaluate["gen_ai.evaluation.name"] == "block_input"
    assert evaluate["gen_ai.evaluation.score.label"] == "tripwire"
    assert evaluate["wardex.evaluation.triggered"] is True


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


def test_a_guardrail_cancelled_by_a_siblings_tripwire_is_not_rendered(agents_env, scenario):
    """The framework assigns `triggered` only AFTER `await guardrail.run(...)`,
    so a guardrail whose sibling tripped first is cancelled mid-body and its
    span exits with `triggered=False` — which used to ship as a completed
    PASS. wardex did not observe a verdict, so it says so: status UNSET,
    `score_label="not_rendered"`, no `wardex.evaluation.triggered`, and a
    counter. The sibling that tripped, the agent and the run read as before."""
    import asyncio as aio

    from agents import GuardrailFunctionOutput, input_guardrail
    from agents.exceptions import InputGuardrailTripwireTriggered

    @input_guardrail
    async def slow_ok(ctx, agent, inp):  # noqa: ANN001, ANN202
        await aio.sleep(5)
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    @input_guardrail
    async def fast_trip(ctx, agent, inp):  # noqa: ANN001, ANN202
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

    scenario(_decide_single)
    agent = Agent(
        name="agent_a",
        instructions="a",
        input_guardrails=[slow_ok, fast_trip],
        model="gpt-4o-mini",
    )
    _init()
    try:
        with pytest.raises(InputGuardrailTripwireTriggered):
            _run(agent)
        spans = _spans()
        assert counters.get("adapters.openai_agents.guardrail_interrupted") == 1
        assert counters.get("adapters.openai_agents.guardrail_failed") == 0
    finally:
        wardex.close()
    cancelled = _one(spans, "evaluate slow_ok")
    assert (cancelled.status, cancelled.error_type) == (StatusCode.UNSET, None)
    assert cancelled.evaluation.score_label == "not_rendered"
    assert "wardex.evaluation.triggered" not in _extra(cancelled)
    tripped = _one(spans, "evaluate fast_trip")
    assert (tripped.status, tripped.error_type) == (StatusCode.ERROR, "guardrail_tripwire")
    assert tripped.evaluation.score_label == "tripwire"
    for name in ("invoke_agent agent_a", "invoke_workflow Agent workflow"):
        s = _one(spans, name)
        assert (s.status, s.error_type) == (StatusCode.ERROR, "guardrail_tripwire")


def test_a_guardrail_whose_body_raises_is_an_error_not_a_pass(agents_env, scenario):
    """A guardrail body that raises exits its span with `triggered=False` and
    no `span.error` (the framework puts the error on the agent). That is the
    GUARDRAIL's failure, not wardex's, so the evaluate span is ERROR with the
    exception's class name and `score_label="not_rendered"`; the exception
    reaches the host untouched."""
    from agents import GuardrailFunctionOutput, input_guardrail

    @input_guardrail
    async def broken(ctx, agent, inp):  # noqa: ANN001, ANN202
        raise RuntimeError("the guardrail failed")
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)  # noqa: B901

    scenario(_decide_single)
    agent = Agent(name="agent_a", instructions="a", input_guardrails=[broken], model="gpt-4o-mini")
    _init()
    try:
        with pytest.raises(RuntimeError, match="the guardrail failed"):
            _run(agent)
        spans = _spans()
        assert counters.get("adapters.openai_agents.guardrail_failed") == 1
        assert counters.get("adapters.openai_agents.guardrail_interrupted") == 0
    finally:
        wardex.close()
    evaluate = _one(spans, "evaluate broken")
    assert (evaluate.status, evaluate.error_type) == (StatusCode.ERROR, "RuntimeError")
    assert evaluate.evaluation.score_label == "not_rendered"
    assert "wardex.evaluation.triggered" not in _extra(evaluate)
    assert evaluate.parent_span_id == _one(spans, "invoke_agent agent_a").context.span_id


def test_a_passing_guardrail_inside_a_hosts_except_block_is_a_pass(agents_env, scenario):
    """The host's OWN in-flight exception is not the guardrail's. A run driven
    from inside a synchronous `except` block — retry-on-error is the common
    shape — carries that exception through `asyncio.run` and `Task.__step`
    into the guardrail span's `__exit__`, where `sys.exc_info()` reports it as
    the exception being handled. A verdict WAS rendered: the span is a pass,
    not an ERROR named after the host's exception class."""
    from agents import GuardrailFunctionOutput, input_guardrail

    @input_guardrail
    async def fine(ctx, agent, inp):  # noqa: ANN001, ANN202
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)

    scenario(_decide_single)
    agent = Agent(name="agent_a", instructions="a", input_guardrails=[fine], model="gpt-4o-mini")
    _init()
    try:
        try:
            raise KeyError("primary path failed")
        except KeyError:
            assert Runner.run_sync(agent, "hi").final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.guardrail_failed") == 0
        assert counters.get("adapters.openai_agents.guardrail_interrupted") == 0
    finally:
        wardex.close()
    evaluate = _one(spans, "evaluate fine")
    assert (evaluate.status, evaluate.error_type) == (StatusCode.OK, None)
    assert evaluate.evaluation.score_label == "pass"
    assert _extra(evaluate)["wardex.evaluation.triggered"] is False


# --------------------------------------------------------------------------
# bookkeeping on the paths that record nothing
# --------------------------------------------------------------------------


def test_spans_that_arrive_without_a_run_leave_no_slot_behind(agents_env, scenario):
    """A processor registered MID-RUN — `wardex.init()` inside a host's open
    `with trace(...)` — sees spans whose run it never opened. They are
    counted as `span_without_run` and recorded nowhere: neither the trace nor
    any span gets an entry allocated just to be read or emptied, so a host
    that holds its spans holds no wardex bookkeeping with them."""
    from agents.tracing import trace

    scenario(_decide_single)
    agent = Agent(name="agent_a", instructions="a", model="gpt-4o-mini")
    with trace("outer"):
        _init()
        try:
            assert Runner.run_sync(agent, "hi").final_output == "done"
            ctx = get_registry()._contexts[AdapterName.OPENAI_AGENTS.value]
            assert counters.get("adapters.openai_agents.span_without_run") >= 1
            assert _adapter_spans(_spans()) == []
            assert dict(ctx._slots) == {}
        finally:
            wardex.close()


def test_the_mcp_tool_list_digest_cannot_be_fooled_by_a_newline_in_a_name():
    """The digest identifies the LIST, not a joined string: `["a\\nb"]` and
    `["a", "b"]` are different catalogues and must not share one."""
    from wardex_sdk._adapters._openai_agents import _tools_digest

    assert _tools_digest(["a\nb"]) != _tools_digest(["a", "b"])
    assert _tools_digest(["a", "b"]) == _tools_digest(["a", "b"])
    assert len(_tools_digest([])) == 16


# --------------------------------------------------------------------------
# the framework's own start instants
# --------------------------------------------------------------------------


def test_a_naive_started_at_is_read_as_utc_and_an_unparsable_one_is_counted():
    """The framework's `time_iso()` is a host-replaceable hook. The default
    is aware UTC; a host's naive string is UTC too — the framework's clock
    is `datetime.now(timezone.utc)` — so it must not be read as LOCAL time,
    which would shift every handoff marker and MCP step by the zone offset.
    A string that is not a timestamp at all is a `None` start (the span
    falls back to wardex's own clock) and a count, never a raise."""
    from types import SimpleNamespace

    from wardex_sdk._adapters._openai_agents import _started_ns

    seen: list[str] = []
    ctx = SimpleNamespace(count=seen.append)
    aware = _started_ns(ctx, SimpleNamespace(started_at="2026-03-01T12:00:00+00:00"))
    naive = _started_ns(ctx, SimpleNamespace(started_at="2026-03-01T12:00:00"))
    assert aware == naive == 1_772_366_400 * 1_000_000_000
    assert _started_ns(ctx, SimpleNamespace(started_at=None)) is None
    assert seen == []
    assert _started_ns(ctx, SimpleNamespace(started_at="yesterday")) is None
    assert seen == ["started_at_unparsed"]


# --------------------------------------------------------------------------
# a pin stranded by wardex.close() mid-run
# --------------------------------------------------------------------------


def _close_mid_run_then_run_again(*, close_from: str) -> list[Any]:
    """`with trace(...)` on the main thread pins the run root there;
    `wardex.close()` INSIDE the block means the trace's own end arrives
    after the uninstall and is ignored. Then a second init and a run with
    its own group id, whose root and children are returned."""
    from agents.tracing import trace

    agent = Agent(name="agent_a", instructions="a", model="gpt-4o-mini")
    _init()
    with trace("outer", group_id="g-first"):
        assert Runner.run_sync(agent, "hi").final_output == "done"
        if close_from == "main":
            wardex.close()
        else:
            worker = threading.Thread(target=wardex.close)
            worker.start()
            worker.join()
    _init()
    try:
        assert (
            Runner.run_sync(agent, "hi", run_config=RunConfig(group_id="g-second")).final_output
            == "done"
        )
        return _spans()
    finally:
        wardex.close()


@pytest.mark.parametrize("close_from", ["main", "worker_thread"])
def test_a_pin_stranded_by_a_close_mid_run_does_not_demote_the_next_runs_group_id(
    agents_env, scenario, close_from
):
    """Two defences, measured one at a time. Closed from the MAIN thread,
    the uninstall takes the stranded pin down itself. Closed from a WORKER
    thread the pin cannot be removed on that thread (a pin comes down only
    on the task that installed it), so the registry retires the dead unit's
    scope fork IN PLACE from where it closes: the main thread reads the
    host's original scope again. Either way the next run is a FRESH root --
    no parent, its own trace -- and not a child of the dead run at 1.0, and
    the next run's `group_id` is its conversation, not the dead run's."""
    scenario(_decide_single)
    spans = _close_mid_run_then_run_again(close_from=close_from)
    assert counters.get("adapters.openai_agents.group_id_shadowed_by_host") == 0
    root = _one(spans, "invoke_workflow Agent workflow")
    assert root.parent_span_id is None
    assert _edge(root) == (ParentSource.TRACE_ROOT, 1.0, ())
    assert root.conversation is not None
    assert root.conversation.conversation_id == "g-second"
    assert "wardex.openai_agents.group_id" not in _extra(root)
    for s in spans:
        assert Limitation.CORRELATION_CONFLICT not in _edge(s)[2], s.name
    if close_from == "main":
        assert counters.get("adapters.openai_agents.pin_stranded") == 0
    else:
        assert counters.get("adapters.openai_agents.pin_stranded") == 1


# --------------------------------------------------------------------------
# resumed runs (tool approval)
# --------------------------------------------------------------------------


def _decide_weather_once(inp: object) -> list[dict]:
    if "get_weather" not in _calls_made(inp):
        return [_fc("get_weather", "call_w1", '{"city": "Seoul"}')]
    return _DONE


def test_a_run_resumed_after_a_tool_approval_ships_under_its_own_root(
    agents_env, scenario, wardex_log
):
    """Human-in-the-loop. A tool with `needs_approval=True` interrupts the
    run; the host approves and calls `Runner.run(agent, state)` in the same
    process. The framework REATTACHES the persisted trace for the second
    half and never announces its start, so no `on_trace_start` arrives —
    the resumed half's agent and tool spans must still ship under a run
    root, opened at the resume, counted and said once. Nothing is dropped
    as `span_without_run`."""
    from agents import function_tool

    @function_tool(needs_approval=True)
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    scenario(_decide_weather_once)
    agent = Agent(name="agent_a", instructions="a", tools=[get_weather], model="gpt-4o-mini")
    _init()
    try:

        async def drive() -> Any:
            first = await Runner.run(agent, "hi")
            assert first.interruptions, "the approval interrupt did not happen"
            state = first.to_state()
            for item in first.interruptions:
                state.approve(item)
            return await Runner.run(agent, state)

        assert asyncio.run(drive()).final_output == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.run_root_reattached") == 1
        # Closed by the framework letting go of its reattached trace, BEFORE
        # the uninstall sweep — the sweep would mark it ADAPTER_UNINSTALLED.
        assert counters.get("adapters.openai_agents.run_root_reattached_closed") == 1
        assert counters.get("adapters.openai_agents.span_without_run") == 0
    finally:
        wardex.close()
    roots = [s for s in spans if s.name == "invoke_workflow Agent workflow"]
    assert len(roots) == 2
    resumed = [r for r in roots if _extra(r).get("wardex.openai_agents.resumed") is True]
    assert len(resumed) == 1
    assert (
        _extra(roots[0])["wardex.openai_agents.trace_id"]
        == _extra(roots[1])["wardex.openai_agents.trace_id"]
    )
    root_id = resumed[0].context.span_id
    # The resumed half: the approved tool runs FIRST (on a subtask of the run
    # task, under the root), then the agent's next turn. The first half's own
    # `execute_tool` is the approval request and hangs under ITS agent.
    resumed_agent = [
        s for s in spans if s.name == "invoke_agent agent_a" and s.parent_span_id == root_id
    ]
    assert len(resumed_agent) == 1
    tools = [s for s in spans if s.name == "execute_tool get_weather"]
    assert len(tools) == 2
    assert [t.parent_span_id == root_id for t in tools].count(True) == 1
    assert (resumed[0].status, resumed[0].error_type) == (StatusCode.OK, None)
    assert Limitation.ADAPTER_UNINSTALLED not in _edge(resumed[0])[2]
    said = [m for m in wardex_log.lines(logging.WARNING) if "resumed" in m]
    assert len(said) == 1


# --------------------------------------------------------------------------
# MCP list-tools
# --------------------------------------------------------------------------


def _in_process_mcp(tools: tuple[str, ...] = ("secret_tool_name",)):  # noqa: ANN202
    """An MCP server that lists `tools` (one by default) and never leaves the
    process."""
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
            return [Tool(name=n, inputSchema={"type": "object"}) for n in tools]

        async def call_tool(self, tool_name, arguments, meta=None):  # noqa: ANN001, ANN202
            return CallToolResult(content=[])

        async def list_prompts(self):  # noqa: ANN202
            raise NotImplementedError

        async def get_prompt(self, name, arguments=None):  # noqa: ANN001, ANN202
            raise NotImplementedError

    return InProcess()


def test_an_mcp_list_tools_span_carries_a_hash_and_never_a_name(agents_env, scenario):
    scenario(_decide_single)
    agent = Agent(
        name="agent_a", instructions="a", mcp_servers=[_in_process_mcp()], model="gpt-4o-mini"
    )
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


def _passes_card_checksum(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch) * (2 if i % 2 else 1)
        total += d - 9 if d > 9 else d
    return total % 10 == 0


def test_a_tool_list_digest_that_reads_as_a_card_number_ships_as_written(agents_env, scenario):
    """The digest is the SDK's own: sixteen hex digits of a SHA-256 of the
    sorted tool names. For about one tool list in eighteen thousand all
    sixteen are decimal and pass the card checksum, and then the default card
    rule rewrote the digest to `****-****-****-NNNN` and wrote `credit_card`
    into the span's record, on both wires and on every run against that
    server. This list is one of those; the real adapter runs it and the real
    codec encodes it with the default rules."""
    from wardex_sdk import _wardex_native
    from wardex_sdk._adapters._openai_agents import _tools_digest
    from wardex_sdk.transport import Transport

    tools = ("get_weather", "search_docs_v28401")
    digest = _tools_digest(sorted(tools))
    assert digest == "8096697742134142" and _passes_card_checksum(digest)

    class Wires(Transport):
        def __init__(self) -> None:
            self.envelopes: list[bytes] = []
            self.otlp: list[bytes] = []

        def export(self, envelope: Any, *, timeout: float | None = None) -> None:
            self.otlp.extend(self.encode(envelope, compress=False))
            self.envelopes.append(
                _wardex_native.codec.encode_envelope(
                    envelope,
                    self._pii_mode,
                    list(self._pii_disabled),
                    self._limits,
                    **self._pii_names(),
                )
            )

    scenario(_decide_single)
    wires = Wires()
    wardex.init(transport=wires, adapters=_ENABLED)
    try:
        agent = Agent(
            name="agent_a",
            instructions="a",
            mcp_servers=[_in_process_mcp(tools)],
            model="gpt-4o-mini",
        )
        assert _run(agent).final_output == "done"
        wardex.flush()
    finally:
        wardex.close()
    (step,) = [
        it["span"]
        for b in wires.envelopes
        for it in _wardex_native.codec.decode_envelope(b)["items"]
        if "span" in it and it["span"]["name"] == "execute_step mcp.list_tools"
    ]
    extra = {kv["key"]: kv["value"] for kv in step["extra"]}
    assert extra["wardex.openai_agents.mcp.tools_hash"] == digest
    assert step.get("capture_integrity", {}).get("redaction_rules", []) == []
    (otlp_step,) = [
        sp
        for b in wires.otlp
        for rs in _wardex_native.codec.decode_otlp_traces(b)["resource_spans"]
        for ss in rs["scope_spans"]
        for sp in ss["spans"]
        if sp["name"] == "execute_step mcp.list_tools"
    ]
    attrs = otlp_step["attributes"]
    assert attrs["wardex.openai_agents.mcp.tools_hash"] == digest
    assert not any(k.startswith("wardex.redact") for k in attrs), attrs


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


def _drive_agent(entry: str, agent: Agent) -> Any:
    """`agent` through one entry point, a streamed run drained to its end."""
    if entry == "run":
        return asyncio.run(Runner.run(agent, "hi")).final_output
    if entry == "run_sync":
        return Runner.run_sync(agent, "hi").final_output

    async def go() -> Any:
        result = Runner.run_streamed(agent, "hi")
        async for _ in result.stream_events():
            pass
        return result.final_output

    return asyncio.run(go())


_ENTRIES = ["run", "run_sync", "run_streamed"]


@pytest.mark.parametrize("entry", _ENTRIES)
def test_a_handled_tool_failure_stays_on_the_tool_span(agents_env, scenario, entry):
    """The default handler turns the exception into a tool output: the tool
    span says `tool_error_handled`, and nothing above it is marked. The run
    went on and no exception left it, so no entry point reads a failure."""
    scenario(_decide_boom)
    _init()
    try:
        assert _drive_agent(entry, _weather_agent()) == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.run_raised_after_root_ok") == 0
    finally:
        wardex.close()
    tool = _one(spans, "execute_tool get_weather")
    assert (tool.status, tool.error_type) == (StatusCode.ERROR, "tool_error_handled")
    assert _one(spans, "invoke_agent agent_a").status is StatusCode.OK
    assert _one(spans, "invoke_workflow Agent workflow").status is StatusCode.OK


def _decide_typed_handoff(inp: object) -> list[dict]:
    """The model hands off with `{}`, which the handoff's input type rejects."""
    return [_fc("transfer_to_agent_b", "call_h1", "{}")]


def _typed_handoff_agent(name: str = "agent_a") -> Agent:
    """An agent whose one handoff declares an input type with a required field."""
    from agents import handoff
    from pydantic import BaseModel

    class Reason(BaseModel):
        reason: str

    async def on_handoff(ctx: Any, data: Reason) -> None:
        return None

    agent_b = Agent(name="agent_b", instructions="b", model="gpt-4o-mini")
    return Agent(
        name=name,
        instructions="a",
        handoffs=[handoff(agent_b, on_handoff=on_handoff, input_type=Reason)],
        model="gpt-4o-mini",
    )


@pytest.mark.parametrize("entry", _ENTRIES)
def test_a_run_that_raises_on_a_typed_handoff_fails_the_agent_and_the_root(
    agents_env, scenario, entry
):
    """Arguments a typed handoff rejects: the framework marks only the handoff
    span, and its generic agent error leaves `ModelBehaviorError` out on
    purpose, so no span the root reads carries an error while the host's call
    raises. The exception leaving the run is the evidence: the agent and the
    root are ERROR, named after its class, through every entry point."""
    from agents.exceptions import ModelBehaviorError

    scenario(_decide_typed_handoff)
    _init()
    try:
        with pytest.raises(ModelBehaviorError):
            _drive_agent(entry, _typed_handoff_agent())
        spans = _spans()
        assert counters.get("adapters.openai_agents.run_raised_after_root_ok") == 0
    finally:
        wardex.close()
    marker = _one(spans, "handoff agent_a→unresolved")
    assert (marker.status, marker.error_type) == (StatusCode.ERROR, "handoff_error")
    for name in ("invoke_agent agent_a", "invoke_workflow Agent workflow"):
        s = _one(spans, name)
        assert (s.status, s.error_type) == (StatusCode.ERROR, "ModelBehaviorError")


def test_a_nested_run_that_raises_fails_its_own_agent_and_not_the_root(agents_env, scenario):
    """`as_tool()` whose nested run raises on a typed handoff. That run did
    fail — the inner agent is ERROR — but its caller is the framework's tool
    wrapper, which handled the failure and went on: the tool span says so, and
    the outer agent and the root stay OK."""

    def decide(inp: object) -> list[dict]:
        items = inp if isinstance(inp, list) else []
        if any(isinstance(x, dict) and x.get("content") == "INNER" for x in items):
            return _decide_typed_handoff(inp)
        if _outputs_done(inp) == 0:
            return [_fc("helper_tool", "call_1", '{"input":"INNER"}')]
        return _DONE

    scenario(decide)
    helper = _typed_handoff_agent("helper")
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
        assert counters.get("adapters.openai_agents.run_raised_after_root_ok") == 0
    finally:
        wardex.close()
    inner = _one(spans, "invoke_agent helper")
    assert (inner.status, inner.error_type) == (StatusCode.ERROR, "ModelBehaviorError")
    tool = _one(spans, "execute_tool helper_tool")
    assert (tool.status, tool.error_type) == (StatusCode.ERROR, "tool_error_handled")
    assert _one(spans, "invoke_agent agent_a").status is StatusCode.OK
    assert _one(spans, "invoke_workflow Agent workflow").status is StatusCode.OK


@pytest.mark.parametrize("entry", ["run", "run_sync"])
@pytest.mark.parametrize("fails", [False, True])
def test_a_run_inside_a_hosts_except_block_reads_its_own_outcome(
    agents_env, scenario, entry, fails
):
    """A run driven from inside the host's own `except` block — retry-on-error
    is the common shape — carries the host's exception into every frame
    underneath, where it reads as in flight. It is the host's, not the run's:
    a run that succeeds there is OK, and one that fails there is named after
    its OWN exception."""
    from agents.exceptions import ModelBehaviorError

    scenario(_decide_typed_handoff if fails else _decide_single)
    _init()
    try:
        try:
            raise KeyError("primary path failed")
        except KeyError:
            if fails:
                with pytest.raises(ModelBehaviorError):
                    _drive_agent(entry, _typed_handoff_agent())
            else:
                assert _drive_agent(entry, _typed_handoff_agent()) == "done"
        spans = _spans()
    finally:
        wardex.close()
    want = (StatusCode.ERROR, "ModelBehaviorError") if fails else (StatusCode.OK, None)
    for name in ("invoke_agent agent_a", "invoke_workflow Agent workflow"):
        s = _one(spans, name)
        assert (s.status, s.error_type) == want


def test_a_run_the_host_cancels_is_not_a_failure(agents_env, scenario):
    """A cancellation leaves the run as `CancelledError`, which is not an
    `Exception`: the host stopped the run and nothing in it failed, so neither
    the agent nor the root says otherwise. The host's own `TimeoutError` is
    raised by `wait_for`, outside the entry point."""
    from agents import function_tool

    @function_tool
    async def get_weather(city: str) -> str:
        await asyncio.Event().wait()
        return "never"

    async def go() -> None:
        await asyncio.wait_for(Runner.run(agent, "hi"), 0.5)

    scenario(_decide_boom)
    agent = Agent(name="agent_a", instructions="a", tools=[get_weather], model="gpt-4o-mini")
    _init()
    try:
        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(go())
        spans = _spans()
        assert counters.get("adapters.openai_agents.run_raised_after_root_ok") == 0
    finally:
        wardex.close()
    for name in ("invoke_agent agent_a", "invoke_workflow Agent workflow"):
        s = _one(spans, name)
        assert (s.status, s.error_type) == (StatusCode.OK, None)


def test_a_raised_run_fails_the_root_even_when_the_message_is_unmapped(
    agents_env, scenario, monkeypatch
):
    """The framework's error sentences are its own and can change. A sentence
    this adapter does not map reads as non-fatal on its own, so a reworded
    max-turns message used to leave the root OK while the host's call raised.
    The exception leaving the run settles it: the root is ERROR, under the
    type the agent span's own error was given."""
    from agents.exceptions import MaxTurnsExceeded

    import wardex_sdk._adapters._openai_agents as mod
    from wardex_sdk._assembly._diag import reset_reports_for_test

    monkeypatch.delitem(mod._ERROR_TABLE, "Max turns exceeded")
    scenario(_decide_loop)
    _init()
    try:
        with pytest.raises(MaxTurnsExceeded):
            _run(_weather_agent(), max_turns=2)
        spans = _spans()
        assert counters.get("adapters.openai_agents.error_message_unmapped") == 1
    finally:
        wardex.close()
        # The unmapped-message line is once per process; give it back.
        reset_reports_for_test()
    for name in ("invoke_agent agent_a", "invoke_workflow Agent workflow"):
        s = _one(spans, name)
        assert (s.status, s.error_type) == (StatusCode.ERROR, "openai_agents_error")


@pytest.mark.parametrize("entry", _ENTRIES)
def test_a_run_that_raises_after_its_root_closed_ok_is_said_once(
    agents_env, scenario, monkeypatch, wardex_log, entry
):
    """The root reads the exception where the framework closes it, on the
    exception's way out. A release that closed it anywhere else would ship an
    OK root for a failed run — simulated here by blinding that read — and the
    call's exit, which does see the exception leave, counts every such run and
    says so once instead of staying silent. `run_streamed` returns before its
    run does, so its exit is the end of the run the stream raises from: a
    check on the entry point's own return would have seen no exception."""
    from agents.exceptions import ModelBehaviorError

    import wardex_sdk._adapters._openai_agents as mod
    from wardex_sdk._assembly._diag import reset_reports_for_test

    monkeypatch.setattr(mod, "failure_leaving", lambda host_inflight: None)
    reset_reports_for_test()
    scenario(_decide_typed_handoff)
    _init()
    try:
        for _ in range(2):
            with pytest.raises(ModelBehaviorError):
                _drive_agent(entry, _typed_handoff_agent())
        spans = _spans()
        assert counters.get("adapters.openai_agents.run_raised_after_root_ok") == 2
    finally:
        wardex.close()
        reset_reports_for_test()
    roots = [s for s in spans if s.name == "invoke_workflow Agent workflow"]
    assert [s.status for s in roots] == [StatusCode.OK, StatusCode.OK]
    lines = [m for m in wardex_log.lines(logging.WARNING) if "under-reports" in m]
    assert len(lines) == 1


def _receiver_never_starts() -> Agent:
    """agent_a hands off to agent_b, whose tool gate raises inside
    `get_all_tools` before the framework opens agent_b's span."""
    from agents import function_tool

    def refuse(ctx, agent) -> bool:  # noqa: ANN001
        raise RuntimeError("tool gate broken")

    @function_tool(is_enabled=refuse)
    def gated() -> str:
        return "never"

    agent_b = Agent(name="agent_b", instructions="b", model="gpt-4o-mini", tools=[gated])
    return Agent(name="agent_a", instructions="a", handoffs=[agent_b], model="gpt-4o-mini")


def _drive_under_host_trace(entry: str, agent: Agent, *, catch: type[BaseException] | None) -> None:
    """`agent` through one entry point inside the host's own `with trace(...)`;
    with `catch`, the host handles that exception inside the trace."""
    from agents import trace

    if entry == "run_sync":
        with trace("host workflow"):
            if catch is None:
                Runner.run_sync(agent, "hi")
                return
            with pytest.raises(catch):
                Runner.run_sync(agent, "hi")
        return

    async def go() -> None:
        if entry == "run":
            await Runner.run(agent, "hi")
            return
        result = Runner.run_streamed(agent, "hi")
        async for _ in result.stream_events():
            pass

    async def hosted() -> None:
        with trace("host workflow"):
            if catch is None:
                await go()
                return
            with pytest.raises(catch):
                await go()

    asyncio.run(hosted())


@pytest.mark.parametrize("entry", _ENTRIES)
@pytest.mark.parametrize("caught", [False, True])
def test_a_run_that_fails_between_two_agents_fails_the_root_the_host_opened(
    agents_env, scenario, entry, caught
):
    """Under the host's own `with trace(...)` the root is the host's, and no
    span of the run is open when it raises after a handoff and before the
    receiver starts, so nothing used to carry the failure there: the host got
    `RuntimeError` and the root read OK. The host's root outlives the call, so
    the call's exit fails it — named after the exception, the agent that
    finished its handoff staying OK — whether the host lets the exception out
    of its trace or handles it inside, as a retry does."""
    scenario(_decide_handoff_once)
    _init()
    try:
        if caught:
            _drive_under_host_trace(entry, _receiver_never_starts(), catch=RuntimeError)
        else:
            with pytest.raises(RuntimeError):
                _drive_under_host_trace(entry, _receiver_never_starts(), catch=None)
        spans = _spans()
        failed_at_exit = counters.get("adapters.openai_agents.root_failed_at_call_exit")
        backstop = counters.get("adapters.openai_agents.run_raised_after_root_ok")
    finally:
        wardex.close()
    root = _one(spans, "invoke_workflow host workflow")
    assert (root.status, root.error_type) == (StatusCode.ERROR, "RuntimeError")
    assert _one(spans, "invoke_agent agent_a").status is StatusCode.OK
    assert (failed_at_exit, backstop) == (1, 0)


def test_a_nested_run_a_hosts_tool_handles_leaves_the_host_root_ok(agents_env, scenario):
    """A tool that runs an agent itself and handles that run's failure, all
    under the host's trace. The nested call raised to ITS caller, the tool,
    which went on: the inner agent is ERROR and nothing above it is. The
    host's trace was opened outside every run, so a call made while an agent
    is current is nested in that agent's run and its exit fails no root."""
    from agents import function_tool, trace

    def decide(inp: object) -> list[dict]:
        items = inp if isinstance(inp, list) else []
        if any(isinstance(x, dict) and x.get("content") == "INNER" for x in items):
            return _decide_typed_handoff(inp)
        if _outputs_done(inp) == 0:
            return [_fc("delegate", "call_1", '{"q":"x"}')]
        return _DONE

    @function_tool
    async def delegate(q: str) -> str:
        try:
            return (await Runner.run(_typed_handoff_agent("helper"), "INNER")).final_output
        except Exception:
            return "fallback"

    async def go() -> Any:
        with trace("host workflow"):
            return (await Runner.run(outer, "hi")).final_output

    scenario(decide)
    outer = Agent(name="agent_a", instructions="outer", tools=[delegate], model="gpt-4o-mini")
    _init()
    try:
        assert asyncio.run(go()) == "done"
        spans = _spans()
        assert counters.get("adapters.openai_agents.root_failed_at_call_exit") == 0
    finally:
        wardex.close()
    inner = _one(spans, "invoke_agent helper")
    assert (inner.status, inner.error_type) == (StatusCode.ERROR, "ModelBehaviorError")
    for name in ("execute_tool delegate", "invoke_agent agent_a", "invoke_workflow host workflow"):
        s = _one(spans, name)
        assert (s.status, s.error_type) == (StatusCode.OK, None)


def _said(inp: object, text: str) -> bool:
    """Whether the request's input carries a message whose content is `text`."""
    items = inp if isinstance(inp, list) else []
    return any(isinstance(x, dict) and x.get("content") == text for x in items)


@pytest.mark.parametrize("failure", ["between_agents", "typed_handoff"])
def test_a_trace_a_tool_opens_does_not_stand_in_for_the_root_of_the_run_it_is_in(
    agents_env, scenario, wardex_log, failure
):
    """A tool that opens its own `with trace(...)` around a run of its own, and
    the outer run, under the host's trace, fails after the tool returned. The
    tool's trace opened and closed OK while the outer call ran, and its close
    was once taken for that call's root closing OK: the host's root then read
    OK for a run that failed between two agents, and a typed-handoff failure
    the agent did report was still counted and said as an under-reported one.
    Only the root the call's framework opens speaks for the call: the host's
    root is ERROR, named after the exception, and nothing is said."""
    from agents import function_tool, trace
    from agents.exceptions import ModelBehaviorError

    from wardex_sdk._assembly._diag import reset_reports_for_test

    then = _decide_handoff_once if failure == "between_agents" else _decide_typed_handoff

    def decide(inp: object) -> list[dict]:
        if _said(inp, "INNER"):
            return _DONE
        if "delegate" not in _calls_made(inp):
            return [_fc("delegate", "call_1", '{"q":"x"}')]
        return then(inp)

    @function_tool
    async def delegate(q: str) -> str:
        with trace("inner workflow"):
            helper = Agent(name="helper", instructions="h", model="gpt-4o-mini")
            return (await Runner.run(helper, "INNER")).final_output

    async def go() -> None:
        with trace("host workflow"):
            await Runner.run(outer, "hi")

    scenario(decide)
    handoffs = (
        _receiver_never_starts() if failure == "between_agents" else _typed_handoff_agent()
    ).handoffs
    outer = Agent(
        name="agent_a",
        instructions="a",
        tools=[delegate],
        handoffs=list(handoffs),
        model="gpt-4o-mini",
    )
    raised = RuntimeError if failure == "between_agents" else ModelBehaviorError
    reset_reports_for_test()  # the line is once per process: let this run say it if it would
    _init()
    try:
        with pytest.raises(raised):
            asyncio.run(go())
        spans = _spans()
        failed_at_exit = counters.get("adapters.openai_agents.root_failed_at_call_exit")
        backstop = counters.get("adapters.openai_agents.run_raised_after_root_ok")
    finally:
        wardex.close()
        reset_reports_for_test()
    root = _one(spans, "invoke_workflow host workflow")
    assert (root.status, root.error_type) == (StatusCode.ERROR, raised.__name__)
    for name in ("invoke_workflow inner workflow", "invoke_agent helper"):
        assert _one(spans, name).status is StatusCode.OK
    # Between two agents no span is open, so the call's exit fails the root; a typed handoff's
    # failure leaves the agent while its span is open, and the agent fails the root first.
    assert (failed_at_exit, backstop) == (1 if failure == "between_agents" else 0, 0)
    assert [m for m in wardex_log.lines(logging.WARNING) if "under-reports" in m] == []


def test_a_trace_opened_inside_a_nested_run_is_not_that_runs_root(agents_env, scenario, wardex_log):
    """A tool runs an agent without a trace of its own, so that nested call
    runs under the outer run's trace and its framework opens none. A tool of
    the nested agent then opens and closes a trace, and the nested run raises
    to the outer tool, which the framework handles. That trace was once taken
    for the nested call's own root closing OK, so a run the outer agent
    handled was counted and said as an under-reported failure. It is no call's
    root: nothing is counted or said, the nested agent is ERROR and the root,
    whose run went on, is OK."""
    from agents import function_tool, trace

    from wardex_sdk._assembly._diag import reset_reports_for_test

    def decide(inp: object) -> list[dict]:
        if _said(inp, "INNER"):
            if "probe" not in _calls_made(inp):
                return [_fc("probe", "call_p1", "{}")]
            return _decide_typed_handoff(inp)
        if _outputs_done(inp) == 0:
            return [_fc("delegate", "call_1", '{"q":"x"}')]
        return _DONE

    @function_tool
    def probe() -> str:
        with trace("tool trace"):
            return "probed"

    @function_tool
    async def delegate(q: str) -> str:
        return (await Runner.run(helper, "INNER")).final_output

    scenario(decide)
    helper = Agent(
        name="helper",
        instructions="h",
        tools=[probe],
        handoffs=list(_typed_handoff_agent().handoffs),
        model="gpt-4o-mini",
    )
    outer = Agent(name="agent_a", instructions="a", tools=[delegate], model="gpt-4o-mini")
    reset_reports_for_test()  # the line is once per process: let this run say it if it would
    _init()
    try:
        assert asyncio.run(Runner.run(outer, "hi")).final_output == "done"
        spans = _spans()
        backstop = counters.get("adapters.openai_agents.run_raised_after_root_ok")
    finally:
        wardex.close()
        reset_reports_for_test()
    assert backstop == 0
    assert [m for m in wardex_log.lines(logging.WARNING) if "under-reports" in m] == []
    inner = _one(spans, "invoke_agent helper")
    assert (inner.status, inner.error_type) == (StatusCode.ERROR, "ModelBehaviorError")
    for name in ("invoke_workflow tool trace", "invoke_workflow Agent workflow"):
        assert _one(spans, name).status is StatusCode.OK


@pytest.mark.parametrize("host_trace", [False, True])
@pytest.mark.parametrize("caught", [False, True])
def test_a_run_that_raises_inside_a_trace_its_tool_opened_fails_that_trace(
    agents_env, scenario, host_trace, caught
):
    """A tool opens its own `with trace(...)` around a run of its own, and that
    run raises to the tool. The tool's trace is a root the host's code opened
    around the call, like the host's own: the call's exit fails it, whether
    the tool lets the exception out of its trace or catches it inside, as a
    fallback does — it used to stay OK then. The outer run handled the tool's
    outcome and went on, so its agent and its root stay OK."""
    from agents import function_tool, trace

    def decide(inp: object) -> list[dict]:
        if _said(inp, "INNER"):
            return _decide_typed_handoff(inp)
        if _outputs_done(inp) == 0:
            return [_fc("delegate", "call_1", '{"q":"x"}')]
        return _DONE

    @function_tool
    async def delegate(q: str) -> str:
        with trace("inner workflow"):
            if not caught:
                return (await Runner.run(_typed_handoff_agent("helper"), "INNER")).final_output
            try:
                return (await Runner.run(_typed_handoff_agent("helper"), "INNER")).final_output
            except Exception:
                return "fallback"

    async def go() -> Any:
        if not host_trace:
            return (await Runner.run(outer, "hi")).final_output
        with trace("host workflow"):
            return (await Runner.run(outer, "hi")).final_output

    scenario(decide)
    outer = Agent(name="agent_a", instructions="a", tools=[delegate], model="gpt-4o-mini")
    _init()
    try:
        assert asyncio.run(go()) == "done"
        spans = _spans()
        backstop = counters.get("adapters.openai_agents.run_raised_after_root_ok")
    finally:
        wardex.close()
    assert backstop == 0
    for name in ("invoke_workflow inner workflow", "invoke_agent helper"):
        s = _one(spans, name)
        assert (s.status, s.error_type) == (StatusCode.ERROR, "ModelBehaviorError")
    outer_root = "invoke_workflow host workflow" if host_trace else "invoke_workflow Agent workflow"
    for name in ("invoke_agent agent_a", outer_root):
        s = _one(spans, name)
        assert (s.status, s.error_type) == (StatusCode.OK, None)


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
