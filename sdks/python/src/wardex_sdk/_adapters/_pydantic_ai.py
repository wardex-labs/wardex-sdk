"""Adapter for Pydantic AI (PyPI: pydantic-ai-slim, module `pydantic_ai`) — a BRIDGE.

pydantic-ai already describes its own runs: with instrumentation on it emits OpenTelemetry spans
(`invoke_agent {agent}`, `chat {model}`, `execute_tool {tool}`) at instrumentation schema version
5 by default. This adapter receives those spans LIVE, in-process, and turns each into a wardex
unit while it is open, so the wire-level CHAT wardex already captures from the socket lands under
the framework's `chat` span by ordinary context propagation. It never patches `Agent.run`, a
model or a tool: the framework's spans are the official surface, its internals are not.

What the wire shows with no adapter, measured on 2.66.0: every model call is its own trace root,
and there is no agent run, no tool call and no grouping at all.

Where the spans come from (`_pydantic_ai_otel.py` holds both surfaces; `_pydantic_ai_attrs.py`
holds what each attribute means):

* Mode W — instrumentation was off. The adapter turns it on by patching
  `Agent._instrument_default` (what `Agent.instrument_all` sets) to settings whose tracer provider
  is wardex's own `opentelemetry-api` implementation, with a no-op meter provider. The global
  tracer and meter providers are never set: `set_tracer_provider` is once-only per process, and
  taking it would silently break a later `logfire.configure()`. Uninstall hands the class its
  original value back by identity, and a host that called `instrument_all` after wardex keeps its
  own (the patch is then superseded and left alone).
* Mode U — the host turned it on first (Logfire, or its own SDK provider). The adapter attaches a
  span processor to the provider those settings resolve to and removes it at uninstall where the
  provider allows; where it does not (a provider with only a public `add_span_processor`), the
  processor stays registered and inert. A setup wardex cannot attach to at all is declined as
  UNSUPPORTED, with the reason said once.

Agents given their own instrumentation (`agent.instrument = ...`, an `Instrumentation`
capability, an `InstrumentedModel`) bypass `Agent._instrument_default` and are not seen.

The rules the units follow, each one a measurement or a decision recorded here:

* A run is `invoke_agent`, placed ROOT: a trace root of its own, or under whatever wardex unit
  is ambient — a run started inside another run's tool is that run's SUB_AGENT, opened as an
  AGENT unit. Its conversation is the framework's `gen_ai.conversation.id` unless the host opened
  its own `wardex.conversation(...)` around it, which wins (`_conversation.py`).
* A model call is a CHAT unit for every framework `chat` span, pinned for the span's lifetime on
  the task that opened it, so the HTTP request inside it is its child. In Mode W the open waits
  for the `set_attributes` call that names the model (pydantic opens the span first and sets the
  request attributes right after, before any request byte), because a unit's name cannot change
  after it opens. The chat unit records NO content: the wire span under it carries the bodies, and
  the run records the whole history once.
* Usage is never the framework's to state when the wire saw the call. Every framework usage key
  is DEMOTED under `wardex.pydantic_ai.otel.`; the core promotes a chat unit's numbers back at its
  close only when no wire request was issued under it, and then marks it `NO_WIRE_EVIDENCE`
  (`_assembly/_usage_hold.py`). So one call is never billed twice, and a call with no request
  under it still carries a number, honestly marked; a request the wire parser does not read as a
  model call leaves the number demoted only.
* A streamed model call is sent from the consumer's task, where wardex's ambient unit is the run
  (measured: pydantic copies the run's context back to the caller, but only re-attaches the
  OpenTelemetry context of the `chat` span there). The adapter hands the core OpenTelemetry's
  current-span reader, and the core parents that request to the chat unit when it is a live
  descendant of the ambient run (`_assembly/_bridge_latch.py`).
* A tool call is `execute_tool`, placed NESTED and pinned, so HTTP inside the tool and the MCP
  stdio seam's own `tools/call` span nest under it. Arguments and result go through the capped
  record path.
* `CallDeferred` and `ApprovalRequired` are control flow: the tool and its run close UNSET.
* A run that ends while a child is still open (a cancellation lands on the run's `end` first,
  measured) is HELD and closed when its last child closes, so the child is never force-closed as
  unfinished and never lost.
* Spans pydantic emits that wardex does not record (a host's own span on the shared tracer, the
  decision-model spans) are counted and dropped; work inside them nests under whatever is ambient.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Iterable, Mapping
from typing import Any

from .._assembly import Limitation, SpanDraft, SpanIntent, UnitKind, report_once
from .._enums import CaptureSource
from . import _pydantic_ai_attrs as pa
from ._base import AdapterInterface
from ._context import AdapterContext, Placement, RunHandle
from ._conversation import framework_conversation
from ._payload import _shaped_payload

#: `wardex.framework` on every unit this adapter opens.
_FRAMEWORK = pa.FRAMEWORK

#: The instrumentation schema versions this release reads: 5, pydantic-ai's default across 2.x.
#: A set rather than one number so that measuring another version is one entry here.
_READ_SCHEMAS = frozenset({5})


def _driver() -> object:
    """The task (or thread) a callback is running on — the pin's owner.

    Copied from the openai-agents adapter rather than imported: the two adapters share the rule,
    not a module. `_get_running_loop` answers `None` outside a loop, so a synchronous host is an
    answer, not a failure.
    """
    loop = asyncio._get_running_loop()
    task = asyncio.current_task(loop) if loop is not None else None
    return task if task is not None else threading.current_thread()


class PydanticAiAdapter(AdapterInterface):
    """Receives pydantic-ai's own spans and turns them into units.

    Holds no table of its own: per-span state lives in `ctx.slot(span)`, keyed on the framework's
    span object by identity, released when the framework drops it, cleared by the context's fork
    reset. A child's slot holds its parent span STRONGLY, which keeps the parent's weak slot alive
    for as long as a child can still need it.
    """

    CONTROL_FLOW: tuple[type[BaseException], ...] = ()

    def __init__(self) -> None:
        self._installed = False
        self._ctx: AdapterContext | None = None
        #: Read at install; 0 until then (no schema is version 0).
        self._schema_version = 0
        self._bridge = _Bridge(self)
        #: The provider a Mode U processor was added to through its public method, which has no
        #: remove; the processor is left there inert at uninstall, and said so.
        self._public: object | None = None
        #: Mode U tuple route: `(multi-processor, processor)`, checked at uninstall for a tuple the
        #: host rebuilt meanwhile (the processor is then left in it, inert, and said so).
        self._tuple: tuple[Any, object] | None = None
        #: Mode W: `(Agent, wardex's settings)`, checked at uninstall for a host that replaced them.
        self._owned: tuple[type, object] | None = None

    def name(self) -> str:
        return _FRAMEWORK

    def install(self, client: object | None = None, ctx: object | None = None) -> None:
        """Read how instrumentation stands, then either own it (W) or attach to it (U).

        The registry has already probed `pydantic-ai-slim` and its shadow check, so a host without
        the framework pays no import here. pydantic-ai-slim requires `opentelemetry-api`, so an
        import failure below is a real FAILED, not an absence to swallow. `self._installed = True`
        stays the LAST line: a clean return before it is UNSUPPORTED, a raise is FAILED, and both
        are rolled back through `uninstall()` and the PatchSet.
        """
        if self._installed:
            return
        self._ctx = ctx if isinstance(ctx, AdapterContext) else None
        if self._ctx is None:
            return
        ctx = self._ctx
        import opentelemetry.trace as _otel_trace
        from pydantic_ai import Agent
        from pydantic_ai import exceptions as _exc
        from pydantic_ai.models.instrumented import InstrumentationSettings

        from . import _pydantic_ai_otel as _otel

        type(self).CONTROL_FLOW = (_exc.CallDeferred, _exc.ApprovalRequired)
        # An OWN class attribute: `instrument_all` is the one-liner that assigns it.
        current = vars(Agent).get("_instrument_default", False)
        route, target = "not_enabled", None
        if isinstance(current, InstrumentationSettings) or current is True:
            # The one global READ wardex makes, and only for `instrument_all(True)`, which IS
            # "use the global provider".
            known = _otel_trace.get_tracer_provider() if current is True else None
            route, target = _otel.attach_target(current, known)
        elif current:
            route = "unreachable"
        if route == "unreachable":
            self.decline_reason = (
                "pydantic_ai adapter: pydantic-ai instrumentation is already on, but its tracer "
                "accepts no span processor wardex can attach to, so pydantic-ai's runs are not "
                "recorded; wire-level model calls still are. Pass "
                "`InstrumentationSettings(tracer_provider=<an OpenTelemetry SDK provider>)` to "
                "`Agent.instrument_all`, or turn instrumentation off and let wardex enable it"
            )
            report_once(self.decline_reason, key="adapters.pydantic_ai.unreachable")
            ctx.count("instrumentation_unreachable")
            return
        is_installed = self._is_installed
        processor = None
        if route == "not_enabled":
            ours = InstrumentationSettings(
                tracer_provider=_otel.Provider(self._bridge, is_installed),
                meter_provider=_otel.meter_provider(),
            )
            version: object = ours.version
            # The PatchSet form of `Agent.instrument_all(ours)`: restored by identity, and a later
            # host `instrument_all(x)` supersedes it and is left alone at uninstall.
            if not ctx.patches.patch(Agent, "_instrument_default", ours):
                return
            self._owned = (Agent, ours)
        else:
            processor = _otel.Processor(self._bridge, is_installed)
            if current is True:
                # Reading the signature builds nothing, so the global meter provider is untouched.
                init = inspect.signature(InstrumentationSettings.__init__)
                version = init.parameters["version"].default
            else:
                version = getattr(current, "version", 0)
            if route == "tuple":
                processors = target._span_processors  # type: ignore[union-attr]
                if ctx.patches.patch(target, "_span_processors", (*processors, processor)):
                    self._tuple = (target, processor)
                else:  # a tuple wardex may not replace: the same object's public method
                    route = "public"
        self._note_schema(ctx, version)
        ctx.follow_current_span(_otel_trace.get_current_span)
        if route == "public":
            target.add_span_processor(processor)  # type: ignore[union-attr]
            self._public = target
        self._installed = True

    def _is_installed(self) -> bool:
        return self._installed

    def _note_schema(self, ctx: AdapterContext, version: object) -> None:
        """Say once when the spans are of a schema this release was not written against.

        The adapter still records what matches: every key it does not know is counted as
        `unknown_attribute`, which is where a schema change shows up.
        """
        number = version if isinstance(version, int) and not isinstance(version, bool) else 0
        self._schema_version = number
        if number not in _READ_SCHEMAS:
            read = ", ".join(str(v) for v in sorted(_READ_SCHEMAS))
            # Also the status's detail: the version table reads the package, not this schema.
            self.unmeasured_reason = (
                f"pydantic_ai adapter: pydantic-ai instrumentation schema version {number}; this "
                f"wardex release reads version {read}. Spans are recorded from what matches; "
                "version 2 names nothing this adapter recognises"
            )
            report_once(self.unmeasured_reason, key="adapters.pydantic_ai.schema_version")
            ctx.count("schema_version")

    def uninstall(self) -> None:
        """Inert first, then hand back, then close what is open.

        The resolver is removed BEFORE the installed check, so the rollback of an install that
        failed after registering it removes it too. `self._ctx` is kept: a span still in flight
        ends through it and is counted as late.
        """
        ctx = self._ctx
        if ctx is not None:
            ctx.unfollow_current_span()
        if not self._installed:
            return
        self._installed = False
        if ctx is None:
            return
        owned, self._owned = self._owned, None
        if owned is not None and vars(owned[0]).get("_instrument_default") is not owned[1]:
            report_once(
                "pydantic_ai adapter: pydantic-ai's instrumentation was replaced after "
                "wardex.init() (`Agent.instrument_all` or `logfire.instrument_pydantic_ai`), so "
                "the adapter recorded none of pydantic-ai's runs from then on; call those before "
                "init() and wardex attaches to them",
                key="adapters.pydantic_ai.instrumentation_replaced",
            )
            ctx.count("instrumentation_replaced")
        ctx.patches.restore_all()
        attached, self._tuple = self._tuple, None
        if attached is not None and attached[1] in getattr(attached[0], "_span_processors", ()):
            self._public = attached[0]  # the host rebuilt the tuple: wardex's stays in it
        if self._public is not None:
            self._public = None
            report_once(
                "pydantic_ai adapter: wardex could not take its span processor back from the "
                "OpenTelemetry provider it attached to (the provider has no remove, or the host "
                "added a processor after wardex.init()); it stays registered but inert for the "
                "life of this process",
                key="adapters.pydantic_ai.processor_left_inert",
            )
            ctx.count("processor_left_inert")
        try:
            self._unpin_held()
        finally:
            ctx.close_all(marker=Limitation.ADAPTER_UNINSTALLED)

    def close_units(self, *, marker: Limitation) -> None:
        """Overridden: this adapter holds runs open across callbacks, and stays installed."""
        if self._ctx is not None:
            try:
                self._unpin_held()
            finally:
                self._ctx.close_all(marker=marker)

    def _unpin_held(self) -> None:
        """Take down the pins this task can, newest first, before the units close.

        The openai-agents sweep: a run still open at `wardex.close()` never sees its own end, so
        the pin its start installed would stay on a thread that outlives it. Only a pin on THIS
        task can come down; the rest are counted as stranded.
        """
        ctx = self._ctx
        if ctx is None:
            return
        for entry in reversed(ctx.slots_snapshot()):
            h = entry.get("handle")
            if not isinstance(h, RunHandle) or h.degraded:
                continue
            if h.pinned_here:
                h.unpin()
            elif h.pinned:
                ctx.count("pin_stranded")


class _Bridge:
    """The normalizer both surfaces feed: three calls, each a total function of the state.

    Every body runs under the adapter's guard; a wardex fault costs a span's description and is
    said once and marked, never raised into pydantic.
    """

    def __init__(self, adapter: PydanticAiAdapter) -> None:
        self._adapter = adapter

    def _live(self) -> AdapterContext | None:
        """The context while installed; None (late callback counted) after an uninstall."""
        ctx = self._adapter._ctx
        if ctx is None:
            return None
        if not self._adapter._installed:
            ctx.count("late_callback")
            return None
        return ctx

    # -- the three calls -------------------------------------------------

    def started(
        self,
        span: object,
        *,
        name: str,
        kind: object,
        attrs: Mapping[str, object],
        parent: object | None,
        start_ns: int,
        attrs_follow: bool,
    ) -> None:
        ctx = self._live()
        if ctx is None:
            return
        ok = False
        with ctx.guard("start"):
            self._start(ctx, span, name, attrs, parent, start_ns, attrs_follow)
            ok = True
        if not ok:
            self._tripped(ctx, "start", ctx.peek(span))

    def attributes_set(self, span: object, attrs: Mapping[str, object]) -> None:
        """Mode W only, called on every `set_attribute(s)`: read once, for a pending chat."""
        ctx = self._adapter._ctx
        if ctx is None:
            return
        slot = ctx.peek(span)
        if slot is None or not slot.get("pending_chat") or "gen_ai.request.model" not in attrs:
            return
        if self._live() is None:
            return
        ok = False
        with ctx.guard("attributes"):
            self._open_chat(ctx, slot, attrs, str(attrs["gen_ai.request.model"]))
            ok = True
        if not ok:
            slot["pending_chat"] = False
            self._tripped(ctx, "attributes", slot)

    def ended(
        self,
        span: object,
        *,
        name: str,
        attrs: Mapping[str, object],
        error: bool,
        error_type: str | None,
        end_ns: int,
    ) -> None:
        ctx = self._live()
        if ctx is None:
            return
        slot = ctx.peek(span)
        if slot is None or "role" not in slot:
            ctx.count("end_without_start")
            return
        ok = False
        with ctx.guard("end"):
            self._end(ctx, slot, name, attrs, error, error_type, end_ns)
            ok = True
        if not ok:
            self._tripped(ctx, "end", slot)

    # -- start -------------------------------------------------------------

    def _start(
        self,
        ctx: AdapterContext,
        span: object,
        name: str,
        attrs: Mapping[str, object],
        parent: object | None,
        start_ns: int,
        attrs_follow: bool,
    ) -> None:
        role = pa.classify(attrs)
        slot = ctx.slot(span)
        slot.update(
            role=role,
            parent=parent,
            owner=_owner(ctx, parent),
            start_ns=start_ns,
            start_task=_driver(),
            live_children=0,
            held=None,
            deferred_child=False,
            written=set(),
            reported=set(),
        )
        if role == "other":
            ctx.count("unknown_span")
            return
        self._audit(ctx, slot, attrs)
        if role == "run":
            self._open_run(ctx, slot, name, attrs)
        elif role == "chat":
            slot["pending_chat"] = True
            if not attrs_follow or "gen_ai.request.model" in attrs:
                model = attrs.get("gen_ai.request.model")
                subject = str(model) if model else name.removeprefix("chat ")
                self._open_chat(ctx, slot, attrs, subject)
            elif slot["owner"] is not None:
                # Live from now: a run cut before this call names its model is held for it.
                slot["owner"]["live_children"] += 1
                slot["pending_counted"] = True
        elif role == "tool":
            self._open_tool(ctx, slot, name, attrs)

    def _open_run(
        self, ctx: AdapterContext, slot: dict[str, Any], name: str, attrs: Mapping[str, object]
    ) -> None:
        agent_name = str(attrs.get("gen_ai.agent.name") or name.removeprefix("invoke_agent "))
        slot["agent_name"] = agent_name
        conversation, shadowed = framework_conversation(
            ctx, attrs.get("gen_ai.conversation.id"), shadowed_counter="conversation_shadowed"
        )
        owner = slot["owner"]
        nested = owner is not None
        parent_agent = _enclosing_agent(ctx, slot["parent"])
        schema = self._adapter._schema_version

        def describe(h: RunHandle) -> None:
            h.draft.set_agent(
                pa.agent_block(
                    attrs, nested=nested, parent_agent=parent_agent, fallback_name=agent_name
                )
            )
            _stamp(h.draft)
            h.draft.set_extra("wardex.pydantic_ai.schema_version", schema)
            extras = pa.extras_for("run", attrs)
            if shadowed:
                extras.append((pa.CONVERSATION_SHADOWED_KEY, shadowed))
            _write_extras(h.draft, slot, extras)

        h = ctx.open_run(
            UnitKind.AGENT if nested else UnitKind.SESSION,
            intent=SpanIntent.INVOKE_AGENT,
            placement=Placement.ROOT,
            subject=agent_name,
            start_ns=slot["start_ns"],
            conversation=conversation,
            describe=describe,
        )
        self._adopt(ctx, slot, h, pin=True)
        ctx.confirm_active("agent_run")

    def _open_chat(
        self, ctx: AdapterContext, slot: dict[str, Any], attrs: Mapping[str, object], subject: str
    ) -> None:
        """Open the chat unit, declare its usage held, and pin it — before any request byte.

        Pinned only on the task the span opened on: a pin installed on another task (the consumer
        of a stream) would never be taken down there and would poison that task's next open. Off
        its start task the unit stays unpinned, and the core still finds it for a request issued
        while its span is current.
        """
        slot["pending_chat"] = False
        slot["subject"] = subject

        def describe(h: RunHandle) -> None:
            h.draft.set_gen_ai(pa.chat_block(attrs, subject=subject))
            _set_server(h.draft, attrs)
            _stamp(h.draft)

        h = ctx.open_run(
            UnitKind.CALL,
            intent=SpanIntent.CHAT,
            placement=Placement.ROOT,
            subject=subject,
            start_ns=slot["start_ns"],
            describe=describe,
        )
        if not h.degraded:
            ctx.hold_framework_usage(h, prefix=pa.USAGE_PREFIX)
        on_start_task = _driver() is slot["start_task"]
        if not on_start_task:
            ctx.count("chat_open_off_start_task")
        self._adopt(ctx, slot, h, pin=on_start_task)
        ctx.confirm_active("chat")

    def _open_tool(
        self, ctx: AdapterContext, slot: dict[str, Any], name: str, attrs: Mapping[str, object]
    ) -> None:
        tool_name = str(attrs.get("gen_ai.tool.name") or name.removeprefix("execute_tool "))

        def describe(h: RunHandle) -> None:
            h.draft.set_tool(pa.tool_block(attrs, fallback_name=tool_name))
            _stamp(h.draft)
            _write_extras(h.draft, slot, pa.extras_for("tool", attrs))

        h = ctx.open_run(
            UnitKind.CALL,
            intent=SpanIntent.EXECUTE_TOOL,
            placement=Placement.NESTED,
            subject=tool_name,
            start_ns=slot["start_ns"],
            describe=describe,
        )
        arguments, _ = pa.content_for("tool", attrs)  # the arguments are set at START
        if arguments is not None:
            h.record_input(_shaped_payload(arguments, ctx.record_budget))
        self._adopt(ctx, slot, h, pin=True)
        ctx.confirm_active("tool")

    def _adopt(self, ctx: AdapterContext, slot: dict[str, Any], h: RunHandle, *, pin: bool) -> None:
        """Store the handle where the resolver and the sweep read it, and count it as live."""
        if pin and not h.degraded and not h.pin(driver=_driver()):
            ctx.count("pin_refused")
        slot["handle"] = h
        owner = slot["owner"]
        if owner is not None and not slot.pop("pending_counted", False):
            owner["live_children"] += 1

    # -- end ---------------------------------------------------------------

    def _end(
        self,
        ctx: AdapterContext,
        slot: dict[str, Any],
        name: str,
        attrs: Mapping[str, object],
        error: bool,
        error_type: str | None,
        end_ns: int,
    ) -> None:
        role = slot["role"]
        if role == "other":
            return
        self._audit(ctx, slot, attrs)
        if role == "attempt":
            self._end_attempt(ctx, slot, name, attrs, error_type, end_ns)
            return
        h: RunHandle | None = slot.get("handle")
        if h is None:
            if role == "chat":
                self._chat_at_end(ctx, slot, name, attrs, error, error_type, end_ns)
            return
        described = False
        with ctx.guard("describe_end"):
            _describe_end(ctx, slot, h, role, attrs)
            described = True
        if not described:
            self._tripped(ctx, "describe_end", slot)
        deferred = role == "tool" and pa.is_deferral(attrs, error_type)
        status = pa.outcome(
            role,
            error,
            error_type,
            attrs,
            deferred_child=bool(slot["deferred_child"]),
            held=slot["live_children"] > 0 if role == "run" else _run_was_cut(slot),
        )
        if h.pinned_here:
            h.unpin()
        elif h.pinned:
            ctx.count("pin_stranded")
        if role == "run" and slot["live_children"] > 0:
            # Cancellation reaches the run's end before its child's (measured): close it when the
            # child does, so the child is closed by its own end and not force-closed as unfinished.
            slot["held"] = status
            ctx.count("held_close")
            return
        h.close(status=status[0], error_type=status[1])
        self._release(slot, deferred=deferred)

    def _release(self, slot: dict[str, Any], *, deferred: bool = False) -> None:
        """A child closed: tell its owner, and close the owner if it was only waiting for this."""
        owner = slot["owner"]
        if owner is None:
            return
        if deferred and not owner["deferred_child"]:
            owner["deferred_child"] = True
            owner_h = owner.get("handle")
            if owner.get("role") == "run" and owner_h is not None:
                _write_extras(owner_h.draft, owner, [(pa.RUN_DEFERRED_KEY, True)])
        owner["live_children"] -= 1
        held = owner.get("held")
        if owner["live_children"] > 0 or held is None:
            return
        owner["held"] = None
        owner_h = owner.get("handle")
        if owner_h is not None:
            owner_h.close(status=held[0], error_type=held[1])
        self._release(owner)

    def _chat_at_end(
        self,
        ctx: AdapterContext,
        slot: dict[str, Any],
        name: str,
        attrs: Mapping[str, object],
        error: bool,
        error_type: str | None,
        end_ns: int,
    ) -> None:
        """A chat span that ended before it named its model (a request hook raised, or the run
        was cancelled while one awaited).

        Opened now, as a child of the run with both timestamps: nothing was sent under it, so it
        carries no wire evidence, and a child draft never passes the usage hold, so its usage
        stays demoted. The run was held for it since its start, and is released here.
        """
        slot["pending_chat"] = False
        owner = slot["owner"]
        parent_h: RunHandle | None = owner.get("handle") if owner is not None else None
        unit = getattr(parent_h, "_unit", None)
        try:
            if parent_h is None or unit is None or not unit.is_live:
                ctx.count("chat_lost_no_parent")  # no run open to sit under; nothing was sent
                return
            subject = name.removeprefix("chat ")
            d = parent_h.child_draft(SpanIntent.CHAT, subject=subject, start_ns=slot["start_ns"])
            d.set_gen_ai(pa.chat_block(attrs, subject=subject))
            _set_server(d, attrs)
            _stamp(d)
            d.add_limitation(Limitation.NO_WIRE_EVIDENCE)
            _write_extras(d, slot, pa.extras_for("chat", attrs))
            status, err = pa.outcome("chat", error, error_type, attrs, held=_run_was_cut(slot))
            parent_h.close_child(d, status=status, error_type=err, end_ns=end_ns)
            ctx.count("chat_opened_at_end")
        finally:
            if slot.pop("pending_counted", False):
                self._release(slot)

    def _end_attempt(
        self,
        ctx: AdapterContext,
        slot: dict[str, Any],
        name: str,
        attrs: Mapping[str, object],
        error_type: str | None,
        end_ns: int,
    ) -> None:
        """A FallbackModel's failed attempt: a closed CHAT child of the chat unit, both instants
        the framework's own (`start_span(start_time=)` then `end(t1)`)."""
        owner = slot["owner"]
        chat_h: RunHandle | None = owner.get("handle") if owner is not None else None
        if chat_h is None or owner.get("role") != "chat":
            ctx.count("attempt_without_chat")
            return
        model = attrs.get("gen_ai.request.model")
        subject = str(model) if model else name.removeprefix("model request attempt ")
        d = chat_h.child_draft(SpanIntent.CHAT, subject=subject, start_ns=slot["start_ns"])
        d.set_gen_ai(pa.chat_block(attrs, subject=subject))
        _stamp(d)
        _write_extras(d, slot, pa.extras_for("attempt", attrs))
        status, error_name = pa.outcome("attempt", True, error_type, attrs)
        chat_h.close_child(d, status=status, error_type=error_name, end_ns=end_ns)
        ctx.confirm_active("attempt")

    # -- bookkeeping ---------------------------------------------------------

    def _audit(
        self, ctx: AdapterContext, slot: dict[str, Any], attrs: Mapping[str, object]
    ) -> None:
        """Count, once per key per span, what the tables refused or do not know.

        `unknown_attribute` is the drift signal: the adapter claims the whole 2.x major, and a
        schema change shows up as keys no table names.
        """
        role = slot["role"]
        reported = slot["reported"]
        for key in pa.unknown_keys(role, attrs):
            if ("unknown", key) not in reported:
                reported.add(("unknown", key))
                ctx.count("unknown_attribute")
        for key in pa.rejected_keys(role, attrs):
            if ("rejected", key) not in reported:
                reported.add(("rejected", key))
                ctx.count("attribute_not_scalar")

    def _tripped(self, ctx: AdapterContext, where: str, slot: dict[str, Any] | None) -> None:
        """Say once that wardex's own step failed, and mark the nearest live unit."""
        report_once(
            f"pydantic_ai adapter: wardex's own {where} step failed; the run's tree is incomplete "
            f"here (counted under adapters.pydantic_ai.{where})",
            key=f"adapters.pydantic_ai.degraded.{where}",
        )
        holder = slot
        while holder is not None:
            h = holder.get("handle")
            if isinstance(h, RunHandle) and not h.degraded:
                h.note(Limitation.INSTRUMENTATION_DEGRADED)
                return
            holder = holder.get("owner")


def _owner(ctx: AdapterContext, parent: object | None) -> dict[str, Any] | None:
    """The slot of the nearest ancestor span that holds a unit, through unrecorded spans."""
    seen = 0
    while parent is not None and seen < 64:
        record = ctx.peek(parent)
        if record is None:
            return None
        if record.get("handle") is not None:
            return record
        parent = record.get("parent")
        seen += 1
    return None


def _run_was_cut(slot: dict[str, Any]) -> bool:
    """Whether this child's run already ended and is held for it: the run was cut mid-call."""
    owner = slot["owner"]
    return owner is not None and owner.get("held") is not None


def _enclosing_agent(ctx: AdapterContext, parent: object | None) -> str | None:
    """The agent name of the nearest enclosing run: a nested run's `parent_agent`."""
    seen = 0
    while parent is not None and seen < 64:
        record = ctx.peek(parent)
        if record is None:
            return None
        if record.get("role") == "run":
            return record.get("agent_name")
        parent = record.get("parent")
        seen += 1
    return None


def _describe_end(
    ctx: AdapterContext,
    slot: dict[str, Any],
    h: RunHandle,
    role: str,
    attrs: Mapping[str, object],
) -> None:
    """What only the end knows: the response, the result, the run's history, the usage."""
    if role == "chat":
        h.draft.set_gen_ai(pa.chat_block(attrs, subject=slot.get("subject") or ""))
        _set_server(h.draft, attrs)
    _write_extras(h.draft, slot, pa.extras_for(role, attrs))  # type: ignore[arg-type]
    if role == "chat":
        return
    content_in, content_out = pa.content_for(role, attrs)  # type: ignore[arg-type]
    if role == "run" and content_in is not None:
        h.record_input(_shaped_payload(content_in, ctx.record_budget))
    if content_out is not None:
        h.record_output(_shaped_payload(content_out, ctx.record_budget))


def _stamp(draft: SpanDraft) -> None:
    """What every unit this adapter opens says about where it came from."""
    draft.add_source(CaptureSource.OTEL_BRIDGE)
    draft.set_extra("wardex.framework", _FRAMEWORK)


def _set_server(draft: SpanDraft, attrs: Mapping[str, object]) -> None:
    address, port = pa.server_of(attrs)
    if address is not None or port is not None:
        draft.set_server(address, port)


def _write_extras(
    draft: SpanDraft, slot: dict[str, Any], pairs: Iterable[tuple[str, pa.Scalar]]
) -> None:
    """The one loop that writes framework values as extras, each key once per span.

    `set_extra` appends, so a key written at start and again at end would ship twice; the first
    value stands, and the keys only the end knows (the response, the usage) are new there.
    """
    written = slot["written"]
    for key, value in pairs:
        if key in written:
            continue
        written.add(key)
        draft.set_extra(key, value)


__all__ = ["PydanticAiAdapter"]
