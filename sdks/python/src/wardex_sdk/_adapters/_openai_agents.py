"""Adapter for the OpenAI Agents SDK (PyPI: openai-agents, module `agents`).

THE TREE COMES FROM THE CONTEXT, NOT FROM AN IDENTIFIER — and here the claim
is carried by the framework's OFFICIAL hook rather than by a patch. The
framework's `TracingProcessor` callbacks run synchronously on the task that
opened the span, and every task the framework spawns — the first turn's
model task, one task per parallel tool call, one per guardrail — is created
with `asyncio.create_task` AFTER the enclosing agent span started, so it
inherits whatever wardex made ambient on the opening task. A unit pinned at
`on_span_start` and unpinned at `on_span_end` therefore reaches every child
span, every tool body and every HTTP request underneath at confidence 1.0,
and the framework's own span ids and parent references are never consulted
for the shape of the tree. A test asserts over this file's own source that
the verbs by which an identifier could shape a tree appear nowhere in it.

What the wire already shows, measured on openai-agents 0.22 with no adapter:
a three-turn run (tool call, handoff, final answer) is three parentless
`chat gpt-4o-mini` spans. No agent name, no handoff, no tool span. This
adapter adds the STRUCTURE — one `invoke_workflow` per `Runner.run` /
`run_sync` / `run_streamed`, one `invoke_agent` per agent, a `handoff`
MARKER with the receiving agent as a SIBLING, one `execute_tool` per
function tool with the tool call id recovered so the tool span joins the
turn that requested it, and one `evaluate` per guardrail — while the wire
keeps owning the LLM call: the framework's response span carries usage and
the wire span carries usage, and the adapter DISCARDS the framework's copy
rather than billing a token twice. The join between the two is evidence
only: `gen_ai.response.id` on the wire span, `wardex.openai_agents.
response_id` on the adapter's tool and handoff spans.

Hook decision, recorded rather than revisited. The official processor is a
public, stable surface and its parentage was measured; the internal run loop
(`run_single_turn`, `execute_handoffs`) already moved once between releases
and now exists as two copies — non-streamed and streamed — with subtly
different span placement, so a patch there would break silently on the next
move and would have to be written twice. Internal patching is therefore
REJECTED; the re-examination condition is a release in which the official
hook loses structural information the internals still carry. When the
framework's tracing is disabled the hook is silent; the adapter says so ONCE
at install, as an INFO line with the two-line recipe that enables tracing
without sending anything to OpenAI, and offers no `force_tracing` option —
replacing the host's processor list is a host-behaviour change this SDK's
own rules forbid an adapter to make.

Seams, and the shape each forces:

* `on_trace_start` / `on_trace_end` — the run. `open_run` + `pin` on the
  task that fired the callback; `unpin` + `close` at the end. A `group_id`
  becomes `gen_ai.conversation.id` on every span underneath, handed to the
  registry at the open so children and the pinned carrier inherit it.
* `AgentSpanData` — one `invoke_agent`, pinned for the span's lifetime.
  The receiver of a handoff opens AFTER the sender closed (the framework
  finishes the sender's span before starting the receiver's), so it opens
  under the run's pin and is the sender's sibling; it carries
  `parent_agent` and a `HANDOFF_FROM` link to the marker.
* `HandoffSpanData` — opened at END (the target is known only then), closed
  at once, at the framework's own start instant. A marker, never a container.
* `TurnSpanData` — no span. The turn number is an attribute on what the turn
  contained, and a turn's error is folded onto the agent that ran it.
* `ResponseSpanData` — no span, no usage. The response id and the function
  calls it requested are remembered for the tool spans that follow.
* `FunctionSpanData` — `execute_tool`, pinned on the tool's own task so an
  HTTP call inside the handler nests under it. The call id is recovered by
  an EXACT and UNIQUE `(name, arguments)` match against the response that
  requested it, labelled at the source; anything less than unique ships the
  `tool_call_id_unavailable_in_process` marker instead of a guess. `name`
  is the framework's TRACE name — `namespace.name` under `tool_namespace()`
  — spelled the same on both sides of the match.
* `GuardrailSpanData` — `evaluate`. `score_label` is one of three:
  `pass` (the body returned, no tripwire), `tripwire` (ERROR
  `guardrail_tripwire`), or `not_rendered` — the body never returned a
  verdict, which is read off the exception in flight at the span's exit:
  its own raise is ERROR with that exception's class name, and an
  interruption (a sibling's tripwire cancelling it) is UNSET.
  `wardex.evaluation.triggered` is present only when a verdict was rendered.
* `MCPListToolsSpanData` — `execute_step mcp.list_tools`, carrying a hash and
  a count of the tool names and never the names.

Failure mapping: the framework never marks its trace or task span. The
agent, turn and response spans carry the error, so the run root's status is
derived by this adapter: a TOP-LEVEL agent (one opened directly under the
run) closing with a FATAL type makes the root ERROR with that type, first one
wins; tool span errors never propagate by themselves; a nested agent's
(agent-as-tool) error never reaches the root.

Decisions this adapter records rather than revisits: no `execute_step` per
turn (a turn is an attribute, and a per-turn span would nest what the
framework runs flat); the framework's `TurnSpanData.usage`,
`TaskSpanData.usage` and `ResponseSpanData.usage` are all discarded, not
only the response span's; a handled `max_turns` (`error_handlers`) still
ships ERROR on the agent and the root, because the framework marks the span
before it consults the handler; and `trace_include_sensitive_data=False`
leaves the response id unavailable on the framework side, so the tool span
then carries the marker and no join — the wire span is the only holder.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import threading
from datetime import datetime
from inspect import signature
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from .._assembly import (
    AgentAttributes,
    ConversationContext,
    EvaluationAttributes,
    Limitation,
    LinkReason,
    SpanIntent,
    ToolAttributes,
    UnitKey,
    UnitKind,
    counters,
    diag_info,
    report_once,
)
from .._enums import StatusCode, ToolExecutionType, ToolType
from ._base import AdapterInterface
from ._context import AdapterContext, Placement, RunHandle
from ._langgraph import _shaped_args

_FRAMEWORK = "openai_agents"
_DISTRIBUTION = "openai-agents"
_MODULE = "agents"
_ENV_DISABLED = "OPENAI_AGENTS_DISABLE_TRACING"

#: Every namespaced key this adapter writes falls under this prefix (design
#: §6.5 tier 1). Declared here as documentation and held by a test over the
#: module's source; the vocabulary layer accepts any `wardex.*` key today.
FRAMEWORK_EXTRA_PREFIXES = ("wardex.openai_agents.",)

#: The agent entry CURRENT on this task — set at `AgentSpanData` start on the
#: task that opened it, restored to the enclosing one at its end. Every task
#: the framework spawns underneath (the model task, one per tool call, one
#: per guardrail, a nested run's loop) copies the context and so inherits it:
#: the same mechanism that carries the pin, applied to the adapter's own
#: per-agent bookkeeping. Keyed this way rather than on the trace because a
#: trace is not one agent: the framework's parallelization pattern gathers
#: several `Runner.run`s under one `with trace(...)`, and any "current agent"
#: held on the trace would be whichever started last.
_CURRENT_AGENT: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "wardex_openai_agents_current_agent", default=None
)

#: The `(from, to, marker key)` of the handoff the agent that just ENDED on
#: this task shipped, for the receiver that starts here next. Published at
#: the sender's END rather than at the marker's: the framework runs a turn
#: — its handoff included — in a task of its own (measured: the first turn
#: of a run), and a value set in that child task never reaches the run's
#: task, where the sender's span ends and the receiver's begins.
_PENDING_HANDOFF: contextvars.ContextVar[tuple[str, str, UnitKey] | None] = contextvars.ContextVar(
    "wardex_openai_agents_pending_handoff", default=None
)

_TRACING_DISABLED_NOTICE = (
    "openai-agents tracing is disabled, so wardex will show only the LLM calls "
    "its interceptor captures: no agent, handoff, tool or guardrail spans. To "
    "get them without sending anything to OpenAI, put these two lines BEFORE "
    "wardex.init(): agents.set_tracing_disabled(False); "
    "agents.set_trace_processors([]). Calling set_trace_processors after "
    "wardex.init() removes wardex's processor as well."
)

_PROCESSOR_REMOVED_NOTICE = (
    "openai-agents adapter: wardex's trace processor was removed by a later "
    "agents.set_trace_processors(...) call, so no agent, handoff, tool or "
    "guardrail spans were recorded in this process. Call set_trace_processors "
    "before wardex.init(), or use agents.add_trace_processor for your own "
    "processor."
)


# -- surface probes ------------------------------------------------------
#
# Two groups, probed and declined independently. Group 1 is the processor
# surface plus the six span-data classes the mapping reads; group 2 is the
# MCP list-tools span, which lands in a separately evolving module and is
# only worth reading when it constructs. No version parsing anywhere: the
# attribute set IS the version floor, which is the only spelling that stays
# true when a release moves a symbol without moving its number.

_PROCESSOR_ABSTRACT = frozenset(
    {"on_trace_start", "on_trace_end", "on_span_start", "on_span_end", "shutdown", "force_flush"}
)
_SPAN_ATTRS = ("span_data", "error", "started_at", "span_id", "trace_id")
_TRACE_ATTRS = ("name", "trace_id")
#: What `_trace_start` reads off a trace, probed on the CONCRETE class: the
#: abstract `Trace` never carried `group_id` — only `TraceImpl.__init__` sets
#: it — so a class-level `hasattr` would decline the very release this was
#: measured on. The constructor keywords other than `group_id` are the
#: framework's required positionals; `processor` is stored privately and is
#: not read back, which is why the read set is listed apart.
_TRACE_IMPL_KWARGS: dict[str, Any] = {
    "name": "probe",
    "trace_id": "trace_probe",
    "group_id": "g",
    "metadata": None,
    "processor": None,
}
_TRACE_IMPL_READS = ("name", "trace_id", "group_id")


def _import_agents_tracing() -> Any | None:
    """`agents.tracing`, or `None` when the framework is absent — an ANSWER.

    The ONLY place this module imports the framework, and it runs inside
    `install()` after the distribution and shadow probes — never at module
    import time. That is what keeps the decline path real: the registry
    imports this module on a host whose `agents` may be an unrelated local
    package, and a module-level framework import would execute that
    package's body (the host-behaviour change the decline exists to avoid)
    or surface an ImportError on `_adapters/__init__.py`'s stderr line
    instead of declining silently.
    """
    try:
        import agents.tracing as tracing
    except Exception:  # noqa: BLE001 — an absent framework is the answer, not a failure
        return None
    return tracing


def _installed_distribution() -> importlib.metadata.Distribution | None:
    """The `openai-agents` distribution, found WITHOUT importing anything: a
    host with a local `agents/` package and no distribution answers `None`
    here and nothing of theirs is imported.

    The absence is an ANSWER, spelled as the stdlib spells it. The
    exception-free `packages_distributions()` was tried first and answers
    `None` for this very wheel on the 3.10 floor (it reads `top_level.txt`
    only there), which would have declined the adapter on every 3.10 host.
    """
    try:
        return importlib.metadata.distribution(_DISTRIBUTION)
    except importlib.metadata.PackageNotFoundError:
        return None


def _shadow_path(
    dist: importlib.metadata.Distribution, ctx: AdapterContext
) -> tuple[str, str] | None:
    """`(resolved module path, distribution path)` when the package that
    `import agents` WOULD answer is not the installed distribution's, else None.

    Read from the import system's spec, so the answer costs no import: a
    project-local `agents/` package (an app folder named `agents` is the
    common shape) is named here, with both paths, BEFORE anything of it could
    run — and a local package with no `tracing` submodule is reported the same
    way rather than declined in silence by a failed `import agents.tracing`.
    A module the import system cannot locate answers None and leaves the
    import step to decline; a namespace package has no origin and does the same.

    An EDITABLE install (`pip install -e`, `uv --editable`, a workspace
    member) resolves `agents` to the project tree, never to
    `site-packages/agents`, so the path comparison alone would decline the
    adapter on the very machine the framework is developed on. The
    distribution's own record of where it came from (PEP 610's
    `direct_url.json`) settles it: a module under that editable root IS the
    installed distribution. A corrupt record is counted, not raised.
    """
    spec = importlib.util.find_spec(_MODULE)
    origin = spec.origin if spec is not None else None
    if not origin:
        return None
    found = Path(origin).parent.resolve()
    expected = Path(str(dist.locate_file(_MODULE))).resolve()
    if found == expected:
        return None
    editable = None
    with ctx.guard("direct_url_read"):
        editable = _editable_root(dist)
    if editable is not None and (found == editable or editable in found.parents):
        return None
    return str(found), str(expected)


def _editable_root(dist: importlib.metadata.Distribution) -> Path | None:
    """The project directory an EDITABLE install of `dist` points at, or None
    for a regular install (no `direct_url.json`, or one that records an
    archive, an index, or a non-editable local directory whose package was
    COPIED into site-packages and so must still match `locate_file`)."""
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict) or not (data.get("dir_info") or {}).get("editable"):
        return None
    url = urlsplit(str(data.get("url", "")))
    if url.scheme != "file":
        return None
    return Path(unquote(url.path)).resolve()


def _constructs(cls: Any, attrs: dict[str, Any], reads: tuple[str, ...] | None = None) -> bool:
    """Does `cls(**attrs)` build and expose every attribute in `reads` (by
    default, every keyword in `attrs`) by name?

    The keyword names are checked against the SIGNATURE first, so a renamed
    keyword is an answer (False) rather than a `TypeError` the registry's
    guard would report as a wardex failure — no exception path is needed.
    `reads` exists for a constructor that stores a keyword under another
    name: the probe then passes the keyword and reads only what the handlers
    read.
    """
    if not isinstance(cls, type):
        return False
    accepted = signature(cls).parameters
    if any(key not in accepted for key in attrs):
        return False
    made = cls(**attrs)
    return all(hasattr(made, key) for key in (attrs if reads is None else reads))


def _surface_ok(tracing: Any) -> bool:
    """Is this the processor and span-data surface the mapping was written for?

    Every predicate below is measured True on the pinned release. The span
    data classes are CONSTRUCTED with the keyword names the handlers read,
    because a renamed keyword is a renamed attribute and `hasattr` on the
    class alone would pass a surface whose instances no longer carry it.
    EVERY attribute a handler reads is in a shape below — `tools` and
    `handoffs` (`_agent_end`), `mcp_data` (`_function_end`), `response`
    (`_response_end`), `group_id` (`_trace_start`) included — so a rename
    declines here, once, instead of raising inside a callback mid-run.
    """
    for name in ("TracingProcessor", "add_trace_processor", "set_trace_processors"):
        if not callable(getattr(tracing, name, None)):
            return False
    if not callable(getattr(tracing, "get_trace_provider", None)):
        return False
    processor = tracing.TracingProcessor
    if frozenset(getattr(processor, "__abstractmethods__", ())) != _PROCESSOR_ABSTRACT:
        return False
    span_cls = getattr(tracing, "Span", None)
    trace_cls = getattr(tracing, "Trace", None)
    if not all(hasattr(span_cls, a) for a in _SPAN_ATTRS):
        return False
    if not all(hasattr(trace_cls, a) for a in _TRACE_ATTRS):
        return False
    trace_impl = getattr(getattr(tracing, "traces", None), "TraceImpl", None)
    if not _constructs(trace_impl, _TRACE_IMPL_KWARGS, _TRACE_IMPL_READS):
        return False
    shapes = {
        "AgentSpanData": {"name": "a", "tools": ["t"], "handoffs": ["h"]},
        "FunctionSpanData": {"name": "f", "input": None, "output": None, "mcp_data": None},
        "HandoffSpanData": {"from_agent": "a", "to_agent": "b"},
        "GuardrailSpanData": {"name": "g", "triggered": False},
        "TurnSpanData": {"turn": 1, "agent_name": "a"},
        "ResponseSpanData": {"response": None},
    }
    return all(_constructs(getattr(tracing, cls, None), attrs) for cls, attrs in shapes.items())


def _mcp_surface_ok(tracing: Any) -> bool:
    """Group 2: the list-tools span, read only when it constructs as measured."""
    cls = getattr(tracing, "MCPListToolsSpanData", None)
    return _constructs(cls, {"server": "s", "result": ["t"]})


def _driver() -> object:
    """The task (or thread) a callback is running on — the pin's owner.

    `_get_running_loop` answers `None` instead of raising outside a loop, so
    this needs no exception path: a synchronous host is an ANSWER, not a
    failure, and the thread is then the carrier the registry will observe.
    """
    loop = asyncio._get_running_loop()
    task = asyncio.current_task(loop) if loop is not None else None
    return task if task is not None else threading.current_thread()


# -- the adapter ---------------------------------------------------------


class OpenAIAgentsAdapter(AdapterInterface):
    """Registers ONE `TracingProcessor` with the framework and reads its
    callbacks. Holds no table of its own: per-run state lives in
    `ctx.slot(trace)`, per-agent and per-span state in `ctx.slot(span)`,
    keyed on the framework's own objects by identity, released when the
    framework drops them, and cleared by the context's fork reset. The
    agent a callback belongs to is found through `_CURRENT_AGENT`, a
    task-inherited variable, never through the trace.
    """

    CONTROL_FLOW: tuple[type[BaseException], ...] = ()

    def __init__(self) -> None:
        self._installed = False
        self._ctx: AdapterContext | None = None
        self._processor = _WardexTracingProcessor(self)
        #: The framework's processor tuple BEFORE registration, kept by
        #: identity so `uninstall` can hand the very same object back.
        self._before: tuple[Any, ...] | None = None
        self._mcp = False
        self._tracing: Any = None
        #: `adapters.openai_agents.active.trace` as read at install. The
        #: counter is process-global and `wardex.close()` does not reset it,
        #: so "no run was ever recorded" is judged against THIS install's
        #: starting value, not against zero — or a second init/close cycle in
        #: one process would inherit the first cycle's runs and never report.
        self._trace_baseline = 0

    def name(self) -> str:
        return _FRAMEWORK

    def install(self, client: object | None = None, ctx: object | None = None) -> None:
        """Probe, then register. ORDER IS LOAD-BEARING and is spelled out.

        The distribution and the shadow check both run BEFORE anything is
        imported — the first through `importlib.metadata`, the second through
        the import system's spec — so a host without the framework pays no
        import, and a host with a local package called `agents` (with or
        without a distribution behind it) is declined without that package's
        body ever running. Only then is `agents.tracing` imported and probed.
        `self._installed = True` stays the LAST line.
        """
        if self._installed:
            return
        self._ctx = ctx if isinstance(ctx, AdapterContext) else None
        if self._ctx is None:
            return
        ctx = self._ctx
        dist = _installed_distribution()
        if dist is None:
            return
        shadow = _shadow_path(dist, ctx)
        if shadow is not None:
            report_once(
                f"openai-agents adapter: the module 'agents' resolved to {shadow[0]}, which "
                f"is not the installed openai-agents distribution ({shadow[1]}); the adapter "
                "declined. Rename the local package or fix sys.path",
                key="adapters.openai_agents.shadowed",
            )
            ctx.count("shadowed")
            return
        tracing = _import_agents_tracing()
        if tracing is None:
            return
        if not _surface_ok(tracing):
            report_once(
                "openai-agents adapter: surface unrecognized, adapter declined; agent, "
                "handoff, tool and guardrail spans will be absent",
                key="adapters.openai_agents.unsupported_surface",
            )
            ctx.count("unsupported_surface")
            return
        self._mcp = _mcp_surface_ok(tracing)
        self._tracing = tracing
        type(self).CONTROL_FLOW = ()
        self._notice_if_tracing_disabled(tracing)
        provider = tracing.get_trace_provider()
        self._before = None
        read = False
        with ctx.guard("processors_read"):
            self._before = tuple(provider._multi_processor._processors)
            read = True
        if not read:
            self._before = None
            report_once(
                "openai-agents adapter: could not read the framework's processor list; "
                "uninstall will leave wardex's processor registered but inert",
                key="adapters.openai_agents.processors_read_failed",
            )
            ctx.count("processors_read_failed")
        self._trace_baseline = counters.get("adapters.openai_agents.active.trace")
        tracing.add_trace_processor(self._processor)
        self._installed = True

    def _notice_if_tracing_disabled(self, tracing: Any) -> None:
        """One INFO line when the hook will be silent, following the
        framework's own precedence: the manual switch wins over the
        environment variable, and the environment variable is read the way
        the framework reads it."""
        ctx = self._ctx
        if ctx is None:
            return
        manual = None
        read = False
        with ctx.guard("tracing_state"):
            manual = tracing.get_trace_provider()._manual_disabled
            read = True
        if not read:
            report_once(
                "openai-agents adapter: could not read the framework's manual tracing "
                "switch; the tracing notice below follows the environment variable only",
                key="adapters.openai_agents.tracing_state_unknown",
            )
            ctx.count("tracing_state_unknown")
        if manual is not None:
            disabled = bool(manual)
        else:
            disabled = os.environ.get(_ENV_DISABLED, "false").lower() in ("true", "1")
        if disabled:
            diag_info(_TRACING_DISABLED_NOTICE)
            ctx.count("tracing_disabled_at_install")

    def _current_processors(self) -> tuple[Any, ...] | None:
        """The framework's live processor tuple, or None when unreadable."""
        ctx = self._ctx
        if ctx is None or self._tracing is None:
            return None
        current = None
        with ctx.guard("processors_read"):
            current = tuple(self._tracing.get_trace_provider()._multi_processor._processors)
        return current

    def _check_processor_removed(self) -> None:
        """Say so, once, when a later `set_trace_processors` dropped ours.

        Only when NO run was recorded since this install: a processor removed
        after runs is a change of mind rather than a silent blind spot, and is
        counted under its own name instead of reported. "Since this install"
        is measured against `_trace_baseline`, because the counter outlives
        `wardex.close()` and a later init in the same process would otherwise
        read the earlier cycle's runs as its own.
        """
        ctx = self._ctx
        if ctx is None:
            return
        current = self._current_processors()
        if current is None or self._processor in current:
            return
        if counters.get("adapters.openai_agents.active.trace") == self._trace_baseline:
            report_once(_PROCESSOR_REMOVED_NOTICE, key="adapters.openai_agents.processor_removed")
            ctx.count("processor_removed")
        else:
            ctx.count("processor_removed_after_runs")

    def uninstall(self) -> None:
        """Hand the framework its ORIGINAL tuple back, then close what is open.

        `set_trace_processors(self._before)` passes the very object read at
        install, so the framework's attribute is restored BY IDENTITY — the
        conformance suite's seam check compares with `is`. A host that added
        a processor after wardex keeps it: only wardex's own is filtered out.
        `self._ctx` is NOT nulled; a callback still in flight needs it.
        """
        if not self._installed:
            return
        self._check_processor_removed()
        self._installed = False
        ctx = self._ctx
        tracing = self._tracing
        current = self._current_processors()
        if ctx is None or tracing is None:
            return
        if self._before is not None and current == self._before + (self._processor,):
            tracing.set_trace_processors(self._before)
        elif current is not None:
            tracing.set_trace_processors([p for p in current if p is not self._processor])
            ctx.count("processors_changed_under_us")
        else:
            report_once(
                "openai-agents adapter: uninstall could not remove wardex's trace processor; "
                "it stays registered but inert for the life of this process",
                key="adapters.openai_agents.uninstall_processor_left_inert",
            )
            ctx.count("uninstall_processor_left_inert")
        ctx.close_all(marker=Limitation.ADAPTER_UNINSTALLED)

    def close_units(self, *, marker: Limitation) -> None:
        """Overridden: this adapter holds a run's units open across callbacks."""
        self._check_processor_removed()
        if self._ctx is not None:
            self._ctx.close_all(marker=marker)

    # -- containment -----------------------------------------------------

    def _contained(self, where: str, fn: Any, obj: Any) -> None:
        """Run one callback body under the adapter's guard, and MARK a loss.

        The guard counts and (under debug) logs; this adds the one line a
        person reads and the marker a dashboard shows, on the live agent when
        there is one and on the run root otherwise. A guard around a callback
        with no report would be a span silently missing from a run.
        """
        ctx = self._ctx
        if ctx is None:
            return
        ok = False
        with ctx.guard(where):
            fn(self, obj)
            ok = True
        if not ok:
            report_once(
                f"openai-agents adapter: internal error at {where}; one or more spans of "
                "this run are missing or incomplete (re-run with debug=True for the traceback)",
                key=f"adapters.openai_agents.{where}",
            )
            with ctx.guard("degraded_mark"):
                _mark_degraded(self, obj)


def _mark_degraded(adapter: OpenAIAgentsAdapter, obj: Any) -> None:
    """Best effort: the live agent's handle, else the run's, if reachable."""
    ctx = adapter._ctx
    if ctx is None:
        return
    current = _CURRENT_AGENT.get()
    target = current.get("handle") if current is not None else None
    if target is None:
        trace = obj if hasattr(obj, "group_id") else _trace_of(adapter, obj)
        if trace is None:
            return
        target = ctx.slot(trace).get("handle")
    if target is not None:
        target.note(Limitation.INSTRUMENTATION_DEGRADED)


# -- the framework's hook ------------------------------------------------


class _WardexTracingProcessor:
    """The six callbacks, each a total function of the adapter's state.

    DELIBERATELY NOT a subclass of the framework's `TracingProcessor`. The
    base would have to be imported at class-definition time — at module
    import, before `install()` has probed anything — and that import ran a
    host's unrelated local `agents` package on a host without the framework
    (measured: a package whose `__init__` writes a file wrote it, and one
    with an empty `tracing.py` raised `AttributeError` out of the module
    import, so the shadowed report never fired). The framework dispatches
    by attribute, never by `isinstance` (no such check exists in its
    provider or processor modules), and `_surface_ok` holds the abstract
    method set of the real base equal to the six names below, so a release
    that adds a callback declines instead of registering a partial processor.

    Every callback's first line is the installed check: the framework may
    keep calling a processor that is being uninstalled on another thread,
    and a callback that ran after `uninstall()` would open a unit nothing
    will ever close. The body runs inside `_contained`, so nothing here can
    reach the framework's own error path — which would log wardex's failure
    under the host's logger as if the host had misconfigured tracing.
    """

    def __init__(self, adapter: OpenAIAgentsAdapter) -> None:
        self._adapter = adapter

    def on_trace_start(self, trace: Any) -> None:
        adapter = self._adapter
        if not adapter._installed or adapter._ctx is None:
            return
        adapter._contained("on_trace_start", _trace_start, trace)

    def on_trace_end(self, trace: Any) -> None:
        adapter = self._adapter
        if not adapter._installed or adapter._ctx is None:
            return
        adapter._contained("on_trace_end", _trace_end, trace)

    def on_span_start(self, span: Any) -> None:
        adapter = self._adapter
        if not adapter._installed or adapter._ctx is None:
            return
        adapter._contained("on_span_start", _span_start, span)

    def on_span_end(self, span: Any) -> None:
        adapter = self._adapter
        if not adapter._installed or adapter._ctx is None:
            return
        adapter._contained("on_span_end", _span_end, span)

    def shutdown(self, timeout: float | None = None) -> None:
        """The framework's exit. Units stay open: the run's own end closes them."""
        ctx = self._adapter._ctx
        if ctx is not None:
            ctx.count("shutdown")

    def force_flush(self) -> None:
        ctx = self._adapter._ctx
        if ctx is not None:
            ctx.count("force_flush")


# -- per-run, per-agent and per-span state ---------------------------------
#
# A run's state is a dict in `ctx.slot(trace)`: its handle, the first fatal
# error, and two counts. An AGENT's state — the current turn, the response
# that turn received and the calls it requested, the turn count, the folded
# turn error, whether its pin held — is a dict in `ctx.slot(span)` of its
# agent span, reached from any callback underneath it through
# `_CURRENT_AGENT`. The trace object is found through the framework's OWN
# notion of the current trace — the callback runs in a task that inherited
# it. Nothing here is keyed on a `trace_id` string, so nothing here can
# outlive the objects the framework holds.


def _trace_of(adapter: OpenAIAgentsAdapter, span: Any) -> Any | None:
    """The trace `span` belongs to, remembered at its start or read from the
    framework's current-trace carrier; None when neither answers."""
    ctx = adapter._ctx
    if ctx is None:
        return None
    remembered = ctx.slot(span).get("trace")
    if remembered is not None:
        return remembered
    tracing = adapter._tracing
    current = tracing.get_current_trace() if tracing is not None else None
    if current is None or current.trace_id != span.trace_id:
        ctx.count("trace_lookup_miss")
        return None
    return current


def _run_state(adapter: OpenAIAgentsAdapter, trace: Any) -> dict[str, Any] | None:
    """The run's slot, or None when this run was never opened (a processor
    registered mid-run, or a root whose open failed)."""
    ctx = adapter._ctx
    if ctx is None:
        return None
    run = ctx.slot(trace)
    return run if run.get("handle") is not None else None


def _span_start(adapter: OpenAIAgentsAdapter, span: Any) -> None:
    kind = type(span.span_data).__name__
    handler = _start_handler(kind)
    if handler is None:
        if kind not in _END_ONLY:
            adapter._ctx.count("span_kind_ignored")  # type: ignore[union-attr]
        return
    trace = _trace_of(adapter, span)
    if trace is None:
        return
    run = _run_state(adapter, trace)
    if run is None:
        adapter._ctx.count("span_without_run")  # type: ignore[union-attr]
        return
    adapter._ctx.slot(span)["trace"] = trace  # type: ignore[union-attr]
    handler(adapter, run, span)


def _span_end(adapter: OpenAIAgentsAdapter, span: Any) -> None:
    kind = type(span.span_data).__name__
    handler = _end_handler(kind)
    if handler is None:
        return
    trace = _trace_of(adapter, span)
    if trace is None:
        return
    run = _run_state(adapter, trace)
    if run is None:
        return
    handler(adapter, run, span)


#: Kinds whose span is opened at END (the framework fills them late) and so
#: must not be counted as ignored at start.
_END_ONLY: frozenset[str] = frozenset(
    {"HandoffSpanData", "ResponseSpanData", "MCPListToolsSpanData"}
)


def _start_handler(kind: str) -> Any | None:
    """Span-data class name -> the start handler, an explicit chain rather than
    a table so that one read of this function is the complete mapping."""
    if kind == "AgentSpanData":
        return _agent_start
    if kind == "TurnSpanData":
        return _turn_start
    if kind == "FunctionSpanData":
        return _function_start
    if kind == "GuardrailSpanData":
        return _guardrail_start
    return None


def _end_handler(kind: str) -> Any | None:
    if kind == "AgentSpanData":
        return _agent_end
    if kind == "TurnSpanData":
        return _turn_end
    if kind == "HandoffSpanData":
        return _handoff_end
    if kind == "ResponseSpanData":
        return _response_end
    if kind == "FunctionSpanData":
        return _function_end
    if kind == "GuardrailSpanData":
        return _guardrail_end
    if kind == "MCPListToolsSpanData":
        return _mcp_list_tools_end
    return None


# -- shared pieces ---------------------------------------------------------


def _pin(adapter: OpenAIAgentsAdapter, handle: RunHandle, driver: object, subject: str) -> bool:
    """Pin `handle` on the task the callback arrived on, and say so if refused.

    A degraded handle is not a refusal — its open already reported — so it
    is neither counted nor reported here.
    """
    ctx = adapter._ctx
    if ctx is None or handle.degraded:
        return False
    ok = handle.pin(driver=driver)
    ctx.count("pin_ok" if ok else "pin_refused")
    if not ok:
        report_once(
            "openai-agents adapter: a framework callback arrived on a task other than the "
            f"one that opened its span; spans under {subject} carry correlation_conflict",
            key="adapters.openai_agents.pin_refused",
        )
    return ok


def _open_child(
    ctx: AdapterContext,
    kind: UnitKind,
    *,
    intent: SpanIntent,
    subject: str,
    site: str,
    describe: Any,
    start_ns: int | None = None,
) -> RunHandle:
    """The ONE way a child unit opens under a run — agent, handoff marker,
    tool, guardrail, MCP step alike — so that a sixth site cannot forget
    what every child owes.

    What every child owes is the adapter's half of a refused agent pin: a
    child opened while that agent is current hangs under whatever IS
    ambient — the session, or an earlier agent — at 1.0, so the child says
    the edge is not what it looks like. The registry marks only the refused
    unit itself. This used to be a four-line check copied at four of five
    sites; the fifth (the MCP list-tools step) had none. `confirm_active`
    is here for the same reason: a site that opens is a site that counts.
    """
    h = ctx.open_run(
        kind,
        intent=intent,
        placement=Placement.NESTED,
        subject=subject,
        start_ns=start_ns,
        describe=describe,
    )
    current = _CURRENT_AGENT.get()
    if current is not None and not current.get("pinned", True):
        h.note(Limitation.CORRELATION_CONFLICT)
    ctx.confirm_active(site)
    return h


def _unpin(adapter: OpenAIAgentsAdapter, handle: RunHandle) -> None:
    """Take a handle's pin down, counted so the pin/unpin ledger balances."""
    ctx = adapter._ctx
    if ctx is not None and handle.unpin():
        ctx.count("unpin")


def _started_ns(span: Any) -> int | None:
    """The framework's own start instant, or None when unreadable."""
    raw = span.started_at
    if not isinstance(raw, str):
        return None
    return int(datetime.fromisoformat(raw).timestamp() * 1_000_000_000)


def _error_message(span: Any) -> str | None:
    err = span.error
    if isinstance(err, dict):
        message = err.get("message")
        return str(message) if message is not None else None
    return None


def _error_data(span: Any) -> dict[str, Any]:
    err = span.error
    data = err.get("data") if isinstance(err, dict) else None
    return data if isinstance(data, dict) else {}


# -- the run -----------------------------------------------------------------


def _trace_start(adapter: OpenAIAgentsAdapter, trace: Any) -> None:
    """One `invoke_workflow` per framework trace, pinned on the task that
    started it — the run's own task, or `run_streamed`'s background loop task
    (which copied the caller's context when it was created)."""
    ctx = adapter._ctx
    if ctx is None:
        return
    run = ctx.slot(trace)
    if run.get("handle") is not None:
        ctx.count("trace_start_twice")
        return
    name = str(trace.name)
    group = trace.group_id
    conversation = ConversationContext(conversation_id=str(group)) if group else None
    trace_id = str(trace.trace_id)
    driver = _driver()

    def describe(h: RunHandle) -> None:
        h.draft.set_workflow_name(name)
        h.draft.set_extra("wardex.framework", _FRAMEWORK)
        h.draft.set_extra("wardex.openai_agents.trace_id", trace_id)

    h = ctx.open_run(
        UnitKind.SESSION,
        intent=SpanIntent.INVOKE_WORKFLOW,
        placement=Placement.ROOT,
        subject=name,
        conversation=conversation,
        describe=describe,
    )
    run["handle"] = h
    run["first_error"] = None
    run["agent_count"] = 0
    run["turn_max"] = 0
    _pin(adapter, h, driver, name)
    ctx.confirm_active("trace")


def _trace_end(adapter: OpenAIAgentsAdapter, trace: Any) -> None:
    ctx = adapter._ctx
    if ctx is None:
        return
    run = ctx.slot(trace)
    h = run.get("handle")
    if h is None:
        ctx.count("trace_end_unmatched")
        return
    h.draft.set_extra("wardex.openai_agents.turns", int(run.get("turn_max") or 0))
    h.draft.set_extra("wardex.openai_agents.agents", int(run.get("agent_count") or 0))
    error = run.get("first_error")
    run.clear()
    _unpin(adapter, h)
    if error is not None:
        h.close(status=StatusCode.ERROR, error_type=error)
    else:
        h.close()


# -- agents ------------------------------------------------------------------


def _agent_start(adapter: OpenAIAgentsAdapter, run: dict[str, Any], span: Any) -> None:
    """One `invoke_agent`, pinned for the span's lifetime, and made the
    CURRENT agent on this task for every callback and task underneath.

    A handoff RECEIVER opens after the sender closed — so under the run's
    pin, as the sender's SIBLING — and carries `parent_agent` plus a
    `HANDOFF_FROM` link to the marker the sender's last turn shipped. An
    agent that opens while another is current on its task (agent-as-tool)
    is that agent's child by context, and is remembered as NESTED so its
    failure never reaches the run root. TOP-LEVEL is therefore "no agent is
    current here", which is true of every `Runner.run` gathered under one
    trace and false of every nested run — not "the run's stack is empty",
    which one concurrent sibling was enough to make false.
    """
    ctx = adapter._ctx
    if ctx is None:
        return
    sd = span.span_data
    name = str(sd.name)
    outer = _CURRENT_AGENT.get()
    pending = _PENDING_HANDOFF.get()
    parent_agent = None
    key = None
    if pending is not None and pending[1] == name:
        parent_agent, key = pending[0], pending[2]
        _PENDING_HANDOFF.set(None)
    driver = _driver()

    def describe(h: RunHandle) -> None:
        h.draft.set_agent(AgentAttributes(name=name, parent_agent=parent_agent))
        h.draft.set_extra("wardex.framework", _FRAMEWORK)
        if key is not None:
            h.link(LinkReason.HANDOFF_FROM, key)

    h = _open_child(
        ctx,
        UnitKind.AGENT,
        intent=SpanIntent.INVOKE_AGENT,
        subject=name,
        site="agent",
        describe=describe,
    )
    pinned = _pin(adapter, h, driver, name)
    entry = ctx.slot(span)
    entry["handle"] = h
    entry["pinned"] = pinned
    entry["name"] = name
    entry["turns"] = 0
    entry["error"] = None
    entry["top_level"] = outer is None
    entry["outer"] = outer
    entry["turn"] = None
    entry["response_id"] = None
    entry["calls"] = {}
    entry["handoff_out"] = None
    _CURRENT_AGENT.set(entry)
    run["agent_count"] = int(run.get("agent_count") or 0) + 1


def _agent_end(adapter: OpenAIAgentsAdapter, run: dict[str, Any], span: Any) -> None:
    ctx = adapter._ctx
    if ctx is None:
        return
    entry = ctx.slot(span)
    h = entry.get("handle")
    if h is None:
        ctx.count("agent_end_unmatched")
        return
    sd = span.span_data
    tools = sd.tools
    handoffs = sd.handoffs
    h.draft.set_extra("wardex.openai_agents.turns", int(entry.get("turns") or 0))
    h.draft.set_extra("wardex.openai_agents.tools_count", len(tools) if tools else 0)
    h.draft.set_extra("wardex.openai_agents.handoffs_count", len(handoffs) if handoffs else 0)
    last = entry.get("response_id")
    if last is not None:
        h.draft.set_extra("wardex.openai_agents.last_response_id", str(last))
    error_type = None
    fatal = False
    message = _error_message(span)
    if message is not None:
        error_type, fatal = _classify_error(adapter, message)
        max_turns = _error_data(span).get("max_turns")
        if isinstance(max_turns, int):
            h.draft.set_extra("wardex.openai_agents.max_turns", max_turns)
    elif entry.get("error") is not None:
        error_type, fatal = entry["error"]
    propagate = fatal and bool(entry.get("top_level")) and run.get("first_error") is None
    if error_type is not None and propagate:
        run["first_error"] = error_type
    if error_type is not None:
        h.close(status=StatusCode.ERROR, error_type=error_type)
    else:
        h.close()
    _unpin(adapter, h)
    # The enclosing agent (or none) is current again on this task. A set on
    # a task other than the start's touches only that task's context, so a
    # callback arriving elsewhere cannot corrupt the opening task's view.
    _CURRENT_AGENT.set(entry.get("outer"))
    _PENDING_HANDOFF.set(entry.get("handoff_out"))
    entry["handle"] = None


# -- handoffs ------------------------------------------------------------------


def _handoff_end(adapter: OpenAIAgentsAdapter, run: dict[str, Any], span: Any) -> None:
    """A MARKER, opened at end (the target is known only then) and closed at
    once at the framework's own start instant. The receiving agent is NOT
    nested under it; it links back to this marker by a run-scoped alias."""
    ctx = adapter._ctx
    if ctx is None:
        return
    sd = span.span_data
    frm = str(sd.from_agent) if sd.from_agent is not None else "unresolved"
    to = str(sd.to_agent) if sd.to_agent is not None else None
    start_ns = None
    with ctx.guard("handoff_started_at"):
        start_ns = _started_ns(span)
    message = _error_message(span)
    # The SENDER's own turn and response: the marker ends on the sender's
    # task, where the sender is current, and a nested run (agent-as-tool)
    # that the same response requested wrote its responses to its OWN entry.
    sender = _CURRENT_AGENT.get()
    turn = sender.get("turn") if sender is not None else None
    response_id = sender.get("response_id") if sender is not None else None
    subject = f"{frm}→{to if to is not None else 'unresolved'}"
    receiver = to if to is not None else "unresolved"
    key_holder: list[UnitKey] = []

    def describe(h: RunHandle) -> None:
        h.draft.set_agent(AgentAttributes(name=receiver, parent_agent=frm))
        h.draft.set_extra("wardex.framework", _FRAMEWORK)
        if turn is not None:
            h.draft.set_extra("wardex.openai_agents.turn", int(turn))
        if response_id is not None:
            h.draft.set_extra("wardex.openai_agents.response_id", str(response_id))
        if to is not None:
            token = h.run_token()
            if token is not None:
                key = UnitKey("openai_agents.handoff", f"{token}:{to}")
                h.alias(key, remember=True)
                key_holder.append(key)

    # The marker opens on the sender's task. When the sender's pin was refused
    # it is not ambient there, so the marker hangs under whatever is (the
    # session, or an earlier agent) at 1.0 — the same misparenting the tool
    # and guardrail children get, and `_open_child` gives it the same marker.
    h = _open_child(
        ctx,
        UnitKind.CALL,
        intent=SpanIntent.HANDOFF,
        subject=subject,
        site="handoff",
        describe=describe,
        start_ns=start_ns,
    )
    if to is None:
        h.close(status=StatusCode.ERROR, error_type="handoff_error")
        ctx.count("handoff_unresolved")
    else:
        if message is not None:
            error_type, _fatal = _classify_error(adapter, message)
            h.close(status=StatusCode.ERROR, error_type=error_type)
        else:
            h.close()
        if key_holder and sender is not None:
            # On the sender's ENTRY, which the run's task shares by reference
            # even when this marker ended in a child task; `_agent_end`
            # publishes it on the task the receiver will start on.
            sender["handoff_out"] = (frm, to, key_holder[0])
        elif key_holder:
            ctx.count("handoff_without_agent")


# -- turns -----------------------------------------------------------------------


def _turn_start(adapter: OpenAIAgentsAdapter, run: dict[str, Any], span: Any) -> None:
    """No span. The turn number is what later spans of this turn carry, on
    the agent that is current here — checked by name against the span's
    own `agent_name`, so a turn arriving on a task where another agent is
    current is counted, not misfiled."""
    ctx = adapter._ctx
    if ctx is None:
        return
    sd = span.span_data
    turn = int(sd.turn)
    run["turn_max"] = max(int(run.get("turn_max") or 0), turn)
    agent = _agent_named(adapter, str(sd.agent_name), "turn")
    if agent is None:
        return
    agent["turn"] = turn
    agent["turns"] = int(agent.get("turns") or 0) + 1


def _turn_end(adapter: OpenAIAgentsAdapter, run: dict[str, Any], span: Any) -> None:
    """A turn's error is the agent's: folded onto the agent that ran it."""
    ctx = adapter._ctx
    if ctx is None:
        return
    message = _error_message(span)
    if message is None:
        return
    agent = _agent_named(adapter, str(span.span_data.agent_name), "turn_error_fold")
    if agent is None:
        return
    if agent.get("error") is None:
        agent["error"] = _classify_error(adapter, message)


def _agent_named(adapter: OpenAIAgentsAdapter, name: str, site: str) -> dict[str, Any] | None:
    """The current agent's entry when it IS the agent the framework named,
    else None with the miss counted under `{site}_missed`."""
    ctx = adapter._ctx
    current = _CURRENT_AGENT.get()
    if current is not None and current.get("name") == name:
        return current
    if ctx is not None:
        ctx.count(f"{site}_missed")
    return None


# -- the model call: no span, no usage -------------------------------------------


def _response_end(adapter: OpenAIAgentsAdapter, run: dict[str, Any], span: Any) -> None:
    """The wire owns the LLM call. What the framework's response span
    contributes is the JOIN: the response id, and the function calls that
    response requested, so the tool spans that follow can carry the call id
    the wire span echoes in the next turn's input. `usage` is never read —
    the same tokens are on the wire span, and two sources bill twice."""
    ctx = adapter._ctx
    if ctx is None:
        return
    agent = _CURRENT_AGENT.get()
    if agent is None:
        ctx.count("response_without_agent")
        return
    sd = span.span_data
    response = sd.response
    calls: dict[tuple[str, str], list[str]] = {}
    if response is None:
        agent["response_id"] = None
        ctx.count("response_id_unavailable")
    else:
        rid = response.id
        agent["response_id"] = str(rid) if rid is not None else None
        if rid is None:
            ctx.count("response_id_unavailable")
        for item in response.output or ():
            if getattr(item, "type", None) != "function_call":
                continue
            key = (_tool_trace_name(item), str(item.arguments))
            calls.setdefault(key, []).append(str(item.call_id))
    agent["calls"] = calls


def _tool_trace_name(item: Any) -> str:
    """The framework's own spelling of a tool call's span name
    (`_tool_identity.tool_trace_name`): `namespace.name` for a call under
    `tool_namespace()`, the bare `name` otherwise — and bare when the
    namespace EQUALS the name, the reserved synthetic shape a deferred
    top-level tool arrives in. `FunctionSpanData.name` is this string, so
    the call-id match keys the response side the same way."""
    name = str(item.name)
    namespace = getattr(item, "namespace", None)
    if isinstance(namespace, str) and namespace and namespace != name:
        return f"{namespace}.{name}"
    return name


# -- tools ---------------------------------------------------------------------


def _function_start(adapter: OpenAIAgentsAdapter, run: dict[str, Any], span: Any) -> None:
    """One `execute_tool`, pinned on the tool's OWN task — the framework runs
    each tool call in a task of its own, created after the turn's response
    arrived, so the pin lands where the handler's own HTTP calls will look.

    The arguments are NOT on the span yet: the framework opens the span first
    and stamps `input` on it afterwards, so the call id is recovered at the
    END (`_function_end`). What is remembered here is the response that
    requested this call — the current agent's latest — because the agent's
    next turn may replace it before a slow parallel sibling closes.
    """
    ctx = adapter._ctx
    if ctx is None:
        return
    sd = span.span_data
    name = str(sd.name)
    agent = _CURRENT_AGENT.get()
    turn = agent.get("turn") if agent is not None else None
    response_id = agent.get("response_id") if agent is not None else None
    driver = _driver()

    def describe(h: RunHandle) -> None:
        h.draft.set_tool(
            ToolAttributes(
                name=name, type=ToolType.FUNCTION, execution_type=ToolExecutionType.IN_PROCESS
            )
        )
        h.draft.set_extra("wardex.framework", _FRAMEWORK)
        if turn is not None:
            h.draft.set_extra("wardex.openai_agents.turn", int(turn))
        if response_id is not None:
            h.draft.set_extra("wardex.openai_agents.response_id", str(response_id))

    h = _open_child(
        ctx,
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        subject=name,
        site="tool",
        describe=describe,
    )
    entry = ctx.slot(span)
    entry["handle"] = h
    entry["pinned"] = _pin(adapter, h, driver, name)
    entry["kind"] = "tool"
    entry["name"] = name
    entry["calls"] = (agent.get("calls") if agent is not None else None) or {}


def _function_end(adapter: OpenAIAgentsAdapter, run: dict[str, Any], span: Any) -> None:
    """The call id: an EXACT and UNIQUE `(name, arguments)` match against the
    response that requested the call. One match is the id, labelled at the
    source; two identical requests in one response are AMBIGUOUS and get the
    marker rather than a guess; no match at all (sensitive data off, or a
    tool the response did not request) gets the marker too.
    """
    ctx = adapter._ctx
    if ctx is None:
        return
    entry = ctx.slot(span)
    h = entry.get("handle")
    if h is None:
        ctx.count("tool_end_unmatched")
        return
    sd = span.span_data
    name = str(entry.get("name"))
    raw_input = sd.input
    calls = entry.get("calls") or {}
    ids = calls.get((name, str(raw_input)), []) if raw_input is not None else []
    call_id = ids[0] if len(ids) == 1 else None
    mcp = sd.mcp_data
    execution = ToolExecutionType.IPC if isinstance(mcp, dict) else ToolExecutionType.IN_PROCESS
    h.draft.set_tool(
        ToolAttributes(name=name, call_id=call_id, type=ToolType.FUNCTION, execution_type=execution)
    )
    if isinstance(mcp, dict) and mcp.get("server") is not None:
        h.draft.set_extra("wardex.openai_agents.mcp.server", str(mcp["server"]))
    if call_id is not None:
        h.draft.set_extra("wardex.openai_agents.tool_call_id_source", "response_output_match")
    else:
        h.note(Limitation.TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS)
        ctx.count("tool_call_id_ambiguous" if len(ids) > 1 else "tool_call_id_unmatched")
    budget = ctx.record_budget
    if raw_input is not None:
        h.record_input(_tool_payload(raw_input, budget))
    output = sd.output
    if output is not None:
        h.record_output(_tool_payload(output, budget))
    message = _error_message(span)
    if message is not None:
        error_type, _fatal = _classify_error(adapter, message)
        h.close(status=StatusCode.ERROR, error_type=error_type)
    else:
        h.close()
    _unpin(adapter, h)
    entry["handle"] = None


def _tool_payload(value: Any, budget: int) -> bytes:
    """A tool's arguments and result as the framework hands them, bounded.

    The framework's `FunctionSpanData.input` is the model's JSON STRING and
    its `output` is usually the handler's string, so they are recorded as
    their own bytes: a backend then shows `{"city":"Seoul"}`, not the
    Python repr `'{"city":"Seoul"}'` that `_shaped_args` — written for
    LangGraph's dict of arguments, where the repr layer is free — would add.
    Same budget+1 handshake as that helper: a string that does not fit is
    returned as exactly `budget + 1` bytes so the storage cap flags the cut,
    and the slice is taken on the string BEFORE encoding so the cost stays
    O(budget). Anything that is not exactly a `str` (a handler returning a
    dict, or a subclass with a `__repr__` of its own) keeps `_shaped_args`.
    """
    if type(value) is not str:
        return _shaped_args(value, budget)
    data = value[: budget + 1].encode("utf-8", "replace")
    return data[: budget + 1] if len(data) > budget else data


# -- guardrails ----------------------------------------------------------------


def _guardrail_start(adapter: OpenAIAgentsAdapter, run: dict[str, Any], span: Any) -> None:
    ctx = adapter._ctx
    if ctx is None:
        return
    sd = span.span_data
    name = str(sd.name)
    driver = _driver()

    def describe(h: RunHandle) -> None:
        h.draft.set_evaluation(EvaluationAttributes(name=name))
        h.draft.set_extra("wardex.framework", _FRAMEWORK)

    h = _open_child(
        ctx,
        UnitKind.CALL,
        intent=SpanIntent.EVALUATE,
        subject=name,
        site="guardrail",
        describe=describe,
    )
    entry = ctx.slot(span)
    entry["handle"] = h
    entry["pinned"] = _pin(adapter, h, driver, name)
    entry["kind"] = "guardrail"
    entry["name"] = name


def _guardrail_end(adapter: OpenAIAgentsAdapter, run: dict[str, Any], span: Any) -> None:
    ctx = adapter._ctx
    if ctx is None:
        return
    entry = ctx.slot(span)
    h = entry.get("handle")
    if h is None:
        ctx.count("guardrail_end_unmatched")
        return
    sd = span.span_data
    triggered = bool(sd.triggered)
    name = str(sd.name)
    # The framework assigns `triggered` only AFTER `await guardrail.run(...)`,
    # so a span that exits by exception — a body that raised, or a task a
    # sibling's tripwire cancelled — reads `triggered=False` with `span.error`
    # unset (the error goes to the agent). This callback runs inside the
    # span's `__exit__`, where that exception is the one being handled, so
    # `sys.exc_info()` is the ONLY evidence that no verdict was rendered.
    inflight = sys.exc_info()[1]
    if triggered:
        h.draft.set_evaluation(EvaluationAttributes(name=name, score_label="tripwire"))
        h.draft.set_extra("wardex.evaluation.triggered", True)
        h.close(status=StatusCode.ERROR, error_type="guardrail_tripwire")
    elif inflight is None:
        h.draft.set_evaluation(EvaluationAttributes(name=name, score_label="pass"))
        h.draft.set_extra("wardex.evaluation.triggered", False)
        h.close()
    elif isinstance(inflight, Exception):
        # The guardrail's OWN failure: its body raised. Named after the
        # exception's class, the way the context surface names a host raise.
        h.draft.set_evaluation(EvaluationAttributes(name=name, score_label="not_rendered"))
        ctx.count("guardrail_failed")
        h.close(status=StatusCode.ERROR, error_type=type(inflight).__name__)
    else:
        # Interrupted (`CancelledError` from a sibling's tripwire, or a
        # shutdown): nobody failed and nothing was decided, so no status.
        h.draft.set_evaluation(EvaluationAttributes(name=name, score_label="not_rendered"))
        ctx.count("guardrail_interrupted")
        h.close(status=StatusCode.UNSET)
    _unpin(adapter, h)
    entry["handle"] = None


# -- MCP list-tools --------------------------------------------------------------


def _mcp_list_tools_end(adapter: OpenAIAgentsAdapter, run: dict[str, Any], span: Any) -> None:
    """A step, at the framework's own instants, carrying a HASH and a count of
    the tool names — never the names, which are the server's catalogue and
    not this run's data."""
    ctx = adapter._ctx
    if ctx is None or not adapter._mcp:
        return
    sd = span.span_data
    server = str(sd.server) if sd.server is not None else None
    names = sorted(str(n) for n in (sd.result or ()))
    digest = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]
    count = len(names)
    start_ns = None
    with ctx.guard("mcp_started_at"):
        start_ns = _started_ns(span)

    def describe(h: RunHandle) -> None:
        h.draft.set_extra("wardex.step.name", "mcp.list_tools")
        h.draft.set_extra("wardex.framework", _FRAMEWORK)
        h.draft.set_extra("wardex.openai_agents.mcp.server", server if server is not None else "")
        h.draft.set_extra("wardex.openai_agents.mcp.tools_hash", digest)
        h.draft.set_extra("wardex.openai_agents.mcp.tools_count", count)

    h = _open_child(
        ctx,
        UnitKind.STEP,
        intent=SpanIntent.EXECUTE_STEP,
        subject="mcp.list_tools",
        site="mcp_list_tools",
        describe=describe,
        start_ns=start_ns,
    )
    h.close()


# -- failure mapping ---------------------------------------------------------------

#: The framework's span error message -> (wardex error type, fatal to the run).
_ERROR_TABLE: dict[str, tuple[str, bool]] = {
    "Max turns exceeded": ("max_turns_exceeded", True),
    "Guardrail tripwire triggered": ("guardrail_tripwire", True),
    "Tool execution cancelled": ("tool_cancelled", False),
    "Error running tool": ("tool_error", True),
    "Error running tool (non-fatal)": ("tool_error_handled", False),
    "Multiple handoffs requested": ("multiple_handoffs_requested", False),
    "Error in agent run": ("agent_run_error", True),
    "Error in call_model_input_filter": ("model_behavior_error", True),
    "Invalid JSON provided": ("model_behavior_error", True),
    "Invalid JSON": ("model_behavior_error", True),
}
_MODEL_BEHAVIOR_PREFIXES = ("Program ", "Tool approval ", "Invalid input filter")


def _classify_error(adapter: OpenAIAgentsAdapter, message: str) -> tuple[str, bool]:
    known = _ERROR_TABLE.get(message)
    if known is not None:
        return known
    if message.endswith(" not found") or message.startswith(_MODEL_BEHAVIOR_PREFIXES):
        return ("model_behavior_error", True)
    ctx = adapter._ctx
    if ctx is not None:
        ctx.count("error_message_unmapped")
    # ONE fixed key, and the message stays OFF the line. The framework sets a
    # HOST-supplied string as a function span's error message (the reason an
    # `on_approval` callback returns for a rejected tool call), so a key built
    # from it would grow `report_once`'s process-global table by one entry per
    # distinct reason — the bound the function exists to keep — and printing
    # it would put host content on stderr outside the masking pipeline. The
    # counter above carries the volume; the span carries the type.
    report_once(
        "openai-agents adapter: the framework reported an error message this adapter "
        "does not map; the span carries error_type=openai_agents_error and the counter "
        "adapters.openai_agents.error_message_unmapped counts every occurrence",
        key="adapters.openai_agents.unmapped",
    )
    return ("openai_agents_error", False)


__all__ = ["FRAMEWORK_EXTRA_PREFIXES", "OpenAIAgentsAdapter"]
