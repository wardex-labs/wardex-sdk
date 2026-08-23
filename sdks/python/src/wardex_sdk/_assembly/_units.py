"""Logical units — design §4.2/§5.2/§5.6, invariants I2, I3, I10, I11.

A *unit* is one logical piece of agent work — a session, a sub-agent, a graph
step, a single call — and it exists so that an adapter has somewhere to put a
parent WITHOUT ever computing one. The registry hands out three things and
nothing else: a `Parentage` (via `Unit.child`), a carrier that makes a unit the
ambient parent for the calling task (`Unit.activate` / `Unit.bind` /
`UnitRegistry.pin_driver`), and a lookup from a framework identifier to a unit
this module already produced (`alias` / `find` / `resolve`).

**Why that list is the whole API.** The product claim is that the causal tree
comes out of in-process context propagation, not out of a framework's callback
identifiers. Competitors reconstruct the tree from `run_id`/`parent_run_id` and
are therefore married to the frameworks that emit them. Here a framework id can
only ever be a `UnitKey` — a lookup ALIAS that selects a unit whose
`SpanContext` was produced by `_assembly/_parentage.py` from a real scope read.
There is deliberately no `create_from_framework_id`, no way to seed a trace id
from a string, and no way to become a parent other than entering the context
(I3). Replace every framework id in a workload with a fresh UUID and the tree
must come out the same shape.

**Lock discipline (I11).** The registry's `RLock` covers map mutation only.
Every method that can emit builds its list of drafts under the lock and calls
the sink AFTER releasing it. This is not hygiene, it is the shape of a live
hazard: today `_adapters/_assembler.py` calls `capture_span` while holding its
own `RLock`, and `Client`'s buffer lock is re-entrant from a same-thread signal
handler — two re-entrant locks taken in a fixed order across an async callback
and a signal path. Reproducing that here would put it under every adapter.

**Bounds (I10).** `max_units` bounds concurrently tracked ROOT units;
`max_entries_per_unit` bounds each per-unit table (children, aliases, claim
keys, open drafts); `max_link_targets` bounds the closed-unit link memory (the
alias-key -> span-context table `resolve_link_target` consults for finished
work, FIFO, evictions counted as `link_memory_full`). All come from
`crates/wardex-limits` and never from a Python literal.

Whether an eviction is VISIBLE ON THE WIRE is decided by one thing: does the
evicted entry own a span? Five tables are bounded — the root table, and per
unit the children, the aliases, the claim keys and the open drafts — and three
of them hold entries that do. Crossing those three CLOSES the oldest entry and
emits it: `UNIT_EVICTED` on a root evicted by `max_units`, `UNIT_TABLE_FULL`
on a child unit or an open draft force-closed by the per-unit breadth bound,
`CHILD_SPAN_UNCLOSED` on what a teardown closes. The one
exception is an open draft that had already LOST a `claim()` arbitration: it is
discarded rather than emitted (`claim_superseded`), because the bound is a
reason to stop tracking a draft and never a reason to promote one the
arbitration rejected.

The other two hold no span, so their evictions emit NOTHING and no marker for
them exists. A dropped lookup alias counts `alias_table_full`; a dropped
de-duplication key counts `claim_table_full`. Neither is undetectable, but
neither is detectable from the exported spans alone: the counters are the
record, and what shows up in the data is only the CONSEQUENCE.

The alias consequence is worth stating exactly, because the obvious reading of
it is backwards. An alias edge is `UNIT_ALIAS` at 0.9 and carries no marker.
Losing the alias makes the id stop resolving, which sends `_edge` back down the
ladder — and the next rung is the ambient scope, not the sole-live guess. So
when the task carries an ambient span the replacement edge is `CONTEXTVAR` at
1.0 with no marker: confidence goes UP while the edge gets WORSE, because a
sub-agent that was hanging off its own unit now hangs off the enclosing
session. That is the flattening `_edge`'s own docstring says most-specific-wins
exists to prevent, arrived at from the other direction. Only when there is no
ambient span at all does the ladder reach `UNIT_INFERRED_SOLE` at 0.5 or
`PARENT_UNRESOLVED`.

An evicted claim key forgets who owned the arbitration, so a lower-ranked
observer of the same event can win it a second time and open a second span for
one logical call.

Every per-unit eviction counts (`child_table_full`, `open_span_table_full`,
`alias_table_full`, `claim_table_full`); the root evictions do not, because the
marked span they emit is already the record. What the emitting half replaces is
an overflow path that dropped the oldest session and its root span outright,
with no marker and no test, so a workload that crossed the cap simply stopped
producing traces and nothing on the wire said why.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import functools
import inspect
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .._enums import CaptureSource, StatusCode
from .._limits import LimitsConfig
from .._scope import Scope
from .._types import ConversationContext, SpanContext
from ..context._contextvar import activate_span, install_span, restore_scope
from ._builder import SpanDraft
from ._diag import counters, guard
from ._integrity import Limitation
from ._parentage import (
    AMBIENT,
    EMPTY_AMBIENT,
    Ambient,
    Evidence,
    Parentage,
    ParentSource,
    child_of,
    latch_ambient,
    resolve_parentage,
)
from ._vocab import SpanIntent


class UnitKind(Enum):
    """CLOSED — design §4.2. What kind of work a unit stands for.

    Framework-neutral on purpose: a LangGraph superstep, a CrewAI task and a
    Claude turn are all `STEP`, and the framework's own word for it rides the
    span's `wardex.framework` attribute rather than splitting this enum.
    """

    SESSION = "session"  # one agent run / graph invocation / conversation
    AGENT = "agent"  # a sub-agent / delegated agent within a session
    STEP = "step"  # a graph node / superstep / turn
    CALL = "call"  # a single tool call / model call bracket


@dataclass(frozen=True, slots=True)
class UnitKey:
    """A framework-supplied lookup ALIAS. NEVER a parentage source (I2).

    `namespace` says whose identifier this is (`"claude.session_id"`,
    `"langgraph.thread_id"`, `"transport.id"`) so that two frameworks using the
    same opaque string cannot collide, and so a reader of a correlation record
    can tell which vocabulary the value came from.

    The type exists to make the invariant visible at every call site: a
    `UnitKey` goes INTO `find`/`alias`/`claim` and comes back as a `Unit` or as
    `None`. One method turns one into a `SpanContext`, and only for a LINK:
    `UnitRegistry.resolve_link_target` answers with a context that feeds
    `SpanDraft.add_link` and nothing else — a link is causality, never
    containment, so the answer cannot become a parent (I2's carve-out).
    """

    namespace: str
    value: str


class SpanSink(Protocol):
    """Where finished drafts go. STRUCTURAL, so that no implementation is named.

    Design §4.4 gives `_assembly/_emit.py` the concrete `SpanSink` — the only
    caller of `Client.capture_span` — and this registry is specified against
    that type. A Protocol is what lets it be specified against a type it cannot
    name, and neither alternative is open: the concrete sink lives in
    `_adapters/`, which `_assembly/` may not import (design §3.1, enforced by
    `tests/test_import_graph.py`), and importing `_client` here instead would
    put a second emit path under the adapters — the exact thing a single funnel
    exists to prevent. That same test asserts no module under `_assembly/` calls
    the sink at all (C-S5), so the shape is all this package needs to know.

    `emit()` MUST NOT raise: it is called from the host's own callbacks. The
    registry wraps each call in `guard()` anyway, because a Protocol cannot
    enforce that on an implementation it does not own.
    """

    def emit(self, draft: SpanDraft, *, agent_semantic: bool) -> bool: ...


#: The evidence a unit hands to its own children. A child unit's parent was not
#: read from a ContextVar — it is the unit itself, whose context this package
#: produced — so recording it as `contextvar` would be the same drift that once
#: left `header` declared in the vocabulary with no producer at all.
_IN_UNIT = Evidence(ParentSource.UNIT_ACTIVE)


@dataclass(frozen=True, slots=True)
class _AmbientUnit:
    """What `activate()` / `pin_driver()` install in the ContextVar."""

    unit: Unit
    owner: object  # the task/thread object observed when this was installed
    pinned: bool


#: The ambient UNIT, alongside (not instead of) the ambient span context that
#: `activate_span` installs in the scope. Two carriers because they answer two
#: questions: `latch_ambient()` asks "what span do I hang off", `current()` asks
#: "which logical unit am I inside" — and the second is what an adapter needs in
#: order to `claim()`, to open a two-phase span, or to record turn I/O.
_ambient_unit: contextvars.ContextVar[_AmbientUnit | None] = contextvars.ContextVar(
    "wardex_ambient_unit", default=None
)


def _current_task() -> object:
    """The identity of the task (or thread) running right now.

    An asyncio Task when there is one, otherwise the thread — the two carriers a
    ContextVar Token is scoped to. Used only for the pin's ownership check;
    nothing about parentage depends on it.

    `suppress` and not an `except` block, and the distinction is not cosmetic:
    C-S4 forbids a handler that leaves no trace of a swallowed FAILURE, and
    there is no failure here. "There is no running event loop" is the ANSWER to
    the question this function asks, and the answer is returned — the thread.
    `asyncio.current_task()` is simply the only public spelling of the question,
    and it reports "no loop" by raising. A counter here would fire on every
    synchronous `activate()` and would be noise, not evidence.
    """
    task: object | None = None
    with contextlib.suppress(RuntimeError):
        task = asyncio.current_task()
    return task if task is not None else threading.current_thread()


class _Carrier:
    """One installed activation: the scope fork plus the ambient-unit entry.

    Deliberately not a context manager. `activate()` is balanced and uses one of
    these under a `try/finally`; `pin_driver()` installs one and never removes
    it on that task, which is the whole point of a pin (design §5.6) — the
    driver task's lifetime is bounded by the unit, so the unbalanced `set()`
    cannot outlive it.

    The two modes therefore use two different scope carriers, and that is
    load-bearing rather than tidy. A pin's fork is installed by `install_span`,
    which HAS NO FINALISER BY DESIGN — do not "simplify" it back into a context
    manager. `activate_span` is a generator-backed CM, so its `finally` runs as
    soon as nothing references the generator; the pin's only strong reference is
    the `PinToken` the registry hands back, and a caller that reads `.installed`
    off a temporary undoes the fork on the very statement that installed it,
    silently, leaving the ambient UNIT installed and the ambient SPAN not. That
    asymmetry ships a wrong parent at confidence 1.0 with no marker.
    """

    __slots__ = ("_prev_scope", "_span_cm", "_token", "owner", "unit")

    def __init__(self, unit: Unit, *, pinned: bool) -> None:
        self.unit = unit
        self.owner = _current_task()
        # The ONE entry point, in both modes. `context/_contextvar.py` installs
        # the conversation identity and the tracestate the unit carries, not
        # only the span context — a carrier that installed only the context
        # would drop the conversation id on every task the unit spans — and
        # `_fork` states that override rule once for both forms.
        self._span_cm: contextlib.AbstractContextManager[None] | None
        self._prev_scope: Scope | None
        if pinned:
            self._span_cm = None
            self._prev_scope = install_span(
                unit.context, conversation=unit.conversation, tracestate=unit.tracestate
            )
        else:
            self._prev_scope = None
            self._span_cm = activate_span(
                unit.context, conversation=unit.conversation, tracestate=unit.tracestate
            )
            self._span_cm.__enter__()
        self._token = _ambient_unit.set(_AmbientUnit(unit, self.owner, pinned))

    def remove(self, *, where: str, debug: bool) -> None:
        """Undo the installation. MUST run on the task that installed it.

        Both halves are attempted, each guarded on its own, and that is not
        belt-and-braces. A cross-task removal raises `ValueError` from the FIRST
        reset — a Token may only be reset in the Context that created it — and
        bailing there would leave `activate_span`'s generator suspended at its
        `yield`. The garbage collector closes such a generator on some later,
        unrelated frame; its `finally` raises the same `ValueError` again, and
        THAT one lands as an unraisable exception in the host's stderr with a
        traceback the host cannot connect to anything it called. Driving the
        generator to completion here — even into its failure — is what keeps a
        misuse of `activate()` a counter instead of noise in someone's logs.

        The pinned half has no generator to drive, so it enforces the same rule
        by hand: a bare `set()` from another task would silently land on THAT
        task's scope, which is worse than the `ValueError` a Token gives for
        free. Raising here keeps a foreign unpin counted rather than silent.

        Two counts for one cross-task exit is therefore the honest tally: two
        installations were left standing, and neither could be taken down.
        """
        with guard(where, debug=debug):
            _ambient_unit.reset(self._token)
        with guard(where, debug=debug):
            if self._span_cm is not None:
                self._span_cm.__exit__(None, None, None)
            elif self.owner is not _current_task():
                raise ValueError("a pin may only be removed on the task that installed it")
            else:
                restore_scope(self._prev_scope)


class PinToken:
    """The handle returned by `UnitRegistry.pin_driver`.

    `installed` is False when the pin was REFUSED — see `pin_driver`. A refused
    token is still a real object with a real `unpin()` so that an adapter's
    teardown path has nothing to branch on; unpinning it is a no-op.
    """

    __slots__ = ("_carrier", "installed", "owner", "unit")

    def __init__(self, unit: Unit, owner: object, carrier: _Carrier | None) -> None:
        self.unit = unit
        self.owner = owner
        self._carrier = carrier
        self.installed = carrier is not None


@dataclass(frozen=True, slots=True)
class _OpenDraft:
    """A two-phase span in flight, plus what it needs to lose an arbitration."""

    draft: SpanDraft
    claim_key: UnitKey | None
    rank_at_open: int


class Unit:
    """One logical piece of work, and the only way an adapter obtains a parent.

    A unit owns its own span: `open()` builds the draft, the adapter fills in
    semantics through `draft`, and `UnitRegistry.close()` materializes it. The
    context children hang off IS that span's context, so anchoring a subtree to
    a span id that is never emitted — an orphaned subtree that looks healthy
    downstream, same trace, confidence 1.0, no marker — is not expressible.
    """

    __slots__ = (
        "_alias_keys",
        "_children",
        "_claims",
        "_draft",
        "_evicted",
        "_input",
        "_input_recorded",
        "_input_truncated",
        "_live",
        "_open",
        "_output",
        "_output_recorded",
        "_output_truncated",
        "_registry",
        "_remembered",
        "conversation",
        "key",
        "kind",
        "owner",
        "parent",
        "parentage",
        "start_ns",
        "tracestate",
    )

    def __init__(
        self,
        registry: UnitRegistry,
        *,
        key: UnitKey,
        kind: UnitKind,
        draft: SpanDraft,
        parentage: Parentage,
        conversation: ConversationContext | None,
        tracestate: str | None,
        start_ns: int,
        parent: Unit | None,
        owner: str | None = None,
    ) -> None:
        self._registry = registry
        self.key = key
        self.kind = kind
        #: Which adapter opened this unit, once one process-wide registry serves
        #: several. None means "not stated", and every lookup that filters on it
        #: treats None as matching nothing rather than everything — an
        #: unattributed unit must not be handed to an adapter as its own.
        self.owner = owner
        self._draft = draft
        self.parentage = parentage
        self.conversation = conversation
        self.tracestate = tracestate
        self.start_ns = start_ns
        self.parent = parent
        self._live = True
        #: Set (under the lock, before the close) by the registry when a
        #: CAPACITY BOUND — not a teardown — closes this unit. Read by
        #: `refused_ambient_marker` to attribute a stranded fork to wardex's
        #: own bound. A why-annotation, never a liveness tier.
        self._evicted = False
        self._children: dict[Unit, None] = {}
        self._alias_keys: list[UnitKey] = []
        #: Alias keys OPTED INTO the registry's closed-unit link memory
        #: (`bind_alias(remember=True)`), mapped to the unit's own span context
        #: as CAPTURED AT BIND TIME — so the detach phase that moves them into
        #: the registry table stays pure dict work and never reads a draft.
        #: Lazy: most units never remember anything. Always a subset of
        #: `_alias_keys`, so the alias breadth bound governs it too.
        self._remembered: dict[UnitKey, SpanContext] | None = None
        self._claims: dict[UnitKey, int] = {}
        self._open: dict[object, _OpenDraft] = {}
        self._input = bytearray()
        self._output = bytearray()
        self._input_recorded = False
        self._output_recorded = False
        self._input_truncated = False
        self._output_truncated = False

    # -- identity --------------------------------------------------------

    @property
    def context(self) -> SpanContext:
        """The unit's own span context — the anchor every child hangs off."""
        return self._draft.context

    @property
    def draft(self) -> SpanDraft:
        """The unit's own span, still under construction.

        NOT in design §4.2's listing, and it has to be: every intent worth
        opening a unit for requires a typed block (`invoke_agent` requires
        `agent`, `execute_step` requires `wardex.step.name`), so without a way
        to reach the draft `finish()` would raise `VocabularyError` and the
        unit's span — the product's exhibit A — would be deleted by `guard()`
        with only a counter left. Exposing the draft rather than mirroring
        fifteen typed setters onto `Unit` also keeps `SpanDraft` the single
        constructor (I5) instead of growing a parallel one here.
        """
        return self._draft

    @property
    def is_live(self) -> bool:
        return self._live

    def enclosing(self, kind: UnitKind) -> Unit | None:
        """The nearest ancestor of `kind`, counting this unit itself.

        Exists for arbitration, not for parentage — nothing here hands out an
        edge. Claims live ON a unit, so two observers of one event have to claim
        on the SAME unit or `claim()` arbitrates nothing: a hook that sees the
        whole run holds the SESSION, while an in-process handler's own scope is
        the CALL it is running inside. Walking up is what puts both claims in
        one table.

        The walk is the registry's rather than a caller's because `parent` is a
        unit-valued attribute, and a caller that walks it is one edit away from
        installing what it found.
        """
        unit: Unit | None = self
        while unit is not None and unit.kind is not kind:
            unit = unit.parent
        return unit

    # -- parentage handout -----------------------------------------------

    def child(self, evidence: Evidence = _IN_UNIT) -> Parentage:
        """The only way an adapter obtains a parent.

        `child_of` is what actually builds the edge; this method exists so that
        the anchor cannot be anything other than a context the registry owns.
        The conversation identity and the tracestate ride along, so a sub-agent
        span does not silently lose the conversation id its session issued.
        """
        return child_of(
            self.context,
            evidence,
            conversation=self.conversation,
            tracestate=self.tracestate,
        )

    # -- carriers --------------------------------------------------------

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Make this unit the ambient parent for the CALLING TASK.

        Enter and exit MUST happen on the same task/thread: a ContextVar Token
        may only be reset in the Context it was created in. This is the ONLY way
        to become a parent (I3) — computing a `SpanContext` and putting it in a
        dict is not parenthood, because it opens neither the nested-manual-span
        path, nor the interceptor latch, nor the `capture_mode=AGENT` gate.

        A cross-task exit is COUNTED rather than raised: the reset would throw
        `ValueError` into the host on a path the host did not ask for, and I6
        forbids that. The leaked fork is bounded by the offending task's life.
        """
        carrier = _Carrier(self, pinned=False)
        try:
            yield
        finally:
            carrier.remove(where="assembly._units.activate_exit", debug=self._registry._debug)

    def bind(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap `fn` so each INVOCATION runs inside a fresh activation.

        NOT `context.bind_context`, and the difference is load-bearing rather
        than stylistic: that helper replays ONE captured `contextvars.Context`,
        and entering the same Context twice concurrently raises
        `RuntimeError: cannot enter context ... is already entered`. For a
        framework callback invoked from two tasks — the ordinary case for a
        thread-pool tool handler — that error surfaces INTO USER CODE. Entering
        a fresh fork per call is re-entrant and concurrency-safe.

        A coroutine function gets an async wrapper so the activation spans the
        whole await rather than only the call that creates the coroutine.
        """
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with self.activate():
                    return await fn(*args, **kwargs)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self.activate():
                return fn(*args, **kwargs)

        return wrapper

    # -- de-duplication, id-keyed, framework-neutral ----------------------

    def claim(self, key: UnitKey, *, rank: int = 0) -> bool:
        """Arbitrate between observers of the same logical event.

        The HIGHEST rank owns the key. NOT first-come: arrival order is a
        property of the framework's event schedule, not of who has the better
        evidence. Claude's `PreToolUse` hook fires BEFORE the handler body runs,
        so first-come would hand every in-process tool to the hook observer —
        the exact inversion of the rule that ownership belongs to the layer that
        wrapped the real execution. Ranks are the adapter's: handler wrapper 10,
        hook 0.

        Because a higher-ranked observer can arrive LATER, a grant is revocable:
        the first claimant of a key is told True (nobody outranks it yet) and
        loses that ownership the moment something outranks it. A claimant that
        opened its span through `open_span(key=...)` needs no discipline for
        that — `close_span` sees the higher rank and DISCARDS the loser's draft,
        so the double emit the arbitration exists to prevent cannot happen even
        if the loser never asks again. A claimant that emits some other way must
        re-`claim()` before it does.

        A losing claimant may still contribute semantics to the winner's span;
        what it may not do is open a second one.
        """
        with self._registry._lock:
            best = self._claims.get(key)
            if best is not None and rank <= best:
                return False
            if best is None:
                self._evict_oldest(self._claims, "claim")
            self._claims[key] = rank
            return True

    def owner_rank(self, key: UnitKey) -> int | None:
        """The rank currently owning `key`, or None if nobody has claimed it."""
        with self._registry._lock:
            return self._claims.get(key)

    # -- two-phase spans -------------------------------------------------

    def open_span(
        self,
        intent: SpanIntent,
        *,
        subject: str | None = None,
        evidence: Evidence = _IN_UNIT,
        key: UnitKey | None = None,
        start_ns: int | None = None,
        source: CaptureSource = CaptureSource.ADAPTER,
    ) -> SpanDraft:
        """Open a child span of this unit. Close it with `close_span`.

        `key` ties the draft to a `claim()` arbitration: the rank owning the key
        at open time is recorded, and `close_span` discards the draft if
        something has outranked it since. That is what makes the arbitration
        mechanical instead of a convention every adapter has to remember.

        A draft opened here is tracked, so `UnitRegistry.close()` can force-close
        it with `CHILD_SPAN_UNCLOSED` rather than leaving it to vanish.
        """
        now = time.time_ns()
        parentage = self.child(evidence)
        draft = SpanDraft(
            parentage,
            intent=intent,
            subject=subject,
            source=source,
            start_ns=start_ns if start_ns is not None else now,
        )

        pending: list[SpanDraft] = []
        with self._registry._lock:
            if self._live:
                evicted = self._evict_oldest(self._open, "open_span")
                if evicted is not None:
                    # An arbitration loser is DISCARDED here rather than
                    # force-closed, exactly as `close_span` would discard it.
                    # The bound is a reason to stop tracking a draft, never a
                    # reason to promote one the arbitration already rejected.
                    if self._superseded_locked(evicted[1]):
                        counters.bump("assembly._units.claim_superseded")
                    else:
                        pending.append(
                            _force_close(evicted[1].draft, Limitation.UNIT_TABLE_FULL, now)
                        )
                # Keyed by the DRAFT, never by `key`. Two observers of one
                # logical event open with the same `key` by design — that is
                # what `claim()` arbitrates — so a key-keyed table would drop
                # the first draft on the floor: unfindable at close, and no
                # longer force-closable by the unit either.
                self._open[id(draft)] = _OpenDraft(
                    draft=draft,
                    claim_key=key,
                    rank_at_open=self._claims.get(key, 0) if key is not None else 0,
                )
            else:
                # A dead unit's span is already emitted, so its context is a
                # perfectly good parent — what it cannot do is force-close this
                # draft later. Counted so "spans opened after their unit closed"
                # is a number rather than a theory.
                counters.bump("assembly._units.open_span_after_close")
        self._registry._flush(pending)
        return draft

    def close_span(
        self,
        draft: SpanDraft,
        *,
        status: StatusCode = StatusCode.OK,
        error_type: str | None = None,
        end_ns: int | None = None,
    ) -> None:
        """Finish a span opened by `open_span` and hand it to the sink.

        Emits OUTSIDE the registry lock (I11). Discards the draft instead when a
        higher-ranked observer took the key over since it was opened.

        A draft that is no longer tracked was ALREADY force-closed and emitted —
        by its unit's teardown, by `close_all`, by the open-table eviction, or by
        an earlier `close_span`. Saying so by name here rather than leaving it to
        `_flush`'s latch: "the handler returned after its session ended" is the
        ordinary shape of that race (it is what `CHILD_SPAN_UNCLOSED` — and
        `UNIT_TABLE_FULL`, for the eviction — exists
        for), and a counter that names it separates it from a genuine double
        emit inside the registry.
        """
        end = end_ns if end_ns is not None else time.time_ns()
        emit = True
        with self._registry._lock:
            entry = self._pop_open(draft)
            if entry is None:
                counters.bump("assembly._units.close_span_after_emit")
                return
            if self._superseded_locked(entry):
                counters.bump("assembly._units.claim_superseded")
                emit = False
        if not emit:
            return
        draft.set_status(status)
        if error_type is not None:
            draft.set_error(error_type)
        draft.set_end_ns(end)
        self._registry._flush([draft])

    def _superseded_locked(self, entry: _OpenDraft) -> bool:
        """Has something outranked this draft's claim since it was opened?

        Caller holds the registry lock. One function rather than the three lines
        inline in `close_span`, because the two FORCE-CLOSE paths — the unit's
        teardown and the open-table eviction — used to skip the question
        entirely, and that is precisely the case `claim()`'s docstring promises
        cannot happen: a hook that opened at rank 0, lost to the handler at rank
        10 and never got its `PostToolUse` (the session aborted) had its losing
        draft force-closed and EMITTED. Two spans for one tool call, and
        `claim_superseded` at zero — downstream it just looks like the agent
        called the tool twice.
        """
        if entry.claim_key is None:
            return False
        owner = self._claims.get(entry.claim_key)
        return owner is not None and owner > entry.rank_at_open

    def _pop_open(self, draft: SpanDraft) -> _OpenDraft | None:
        # Deletes from `self._open` while iterating it, which is legal ONLY
        # because it returns immediately afterwards. Turning this into a
        # "collect every match" loop raises `RuntimeError: dictionary changed
        # size during iteration` at close time, inside the registry lock, on the
        # host's own callback.
        for table_key, entry in self._open.items():
            if entry.draft is draft:
                del self._open[table_key]
                return entry
        return None

    # -- annotation ------------------------------------------------------

    def note(self, marker: Limitation) -> None:
        """Attach a limitation to the unit's OWN span.

        Under the registry lock — re-entrantly, since `_close_locked` notes the
        marker that closed a unit while already holding it. The lock is what
        makes "the marker is on the draft" and "the draft is being finalized"
        two states a reader cannot land between.
        """
        with self._registry._lock:
            if not self._live:
                counters.bump("assembly._units.note_after_close")
                return
            self._draft.add_limitation(marker)

    def record_input(self, data: bytes) -> None:
        """Accumulate task input onto the unit's own span.

        Accumulating rather than assigning is the point: a multi-turn session
        root that keeps only the first prompt is how the adapter lost every turn
        after the first. Bounded by the core's `max_body_bytes`; passing it sets
        `truncated` rather than growing without limit (I10).

        Locked for the same reason as `note`: `_finalize_locked` reads the
        buffer and the truncation flag together, and `flag |= f(...)` is a
        read-modify-write that two framework callbacks on two threads can
        interleave — losing the flag, not the bytes, which is the direction that
        turns a truncated payload into one that claims to be complete.
        """
        with self._registry._lock:
            self._input_recorded = True
            self._input_truncated |= _append_capped(
                self._input, data, self._registry._max_record_bytes
            )

    def record_output(self, data: bytes) -> None:
        """Accumulate task output onto the unit's own span. See `record_input`."""
        with self._registry._lock:
            self._output_recorded = True
            self._output_truncated |= _append_capped(
                self._output, data, self._registry._max_record_bytes
            )

    # -- internal --------------------------------------------------------

    def _evict_oldest(self, table: dict[Any, Any], where: str) -> tuple[Any, Any] | None:
        """Make room in a per-unit table. Caller holds the registry lock.

        FIFO, matching the open-tool eviction this replaces. Returns the evicted
        `(key, value)` — BOTH, because the per-unit tables put the payload on
        different sides: `_children` is an ordered SET of units keyed by the
        unit, while `_open` maps an opaque id to the record. Returning only the
        value silently evicts nothing from `_children`, since every value there
        is `None`.
        """
        if len(table) < self._registry._max_entries_per_unit:
            return None
        oldest = next(iter(table))
        value = table.pop(oldest)
        counters.bump(f"assembly._units.{where}_table_full")
        return oldest, value

    def _evict_alias(self) -> UnitKey | None:
        """The same bound over the alias list. Caller holds the registry lock.

        A separate helper rather than a call into `_evict_oldest` above, and the
        difference is the container, not the policy: `_evict_oldest` pops a
        `(key, value)` pair out of a DICT, because `_children` and `_open` put
        the payload on different sides, while an alias list has no value to
        return. Both read the same `_max_entries_per_unit` and both count, which
        is what keeps them one bound written twice rather than two bounds.
        """
        if len(self._alias_keys) < self._registry._max_entries_per_unit:
            return None
        counters.bump("assembly._units.alias_table_full")
        popped = self._alias_keys.pop(0)
        # An alias lost to the breadth bound is lost from BOTH lookup paths: a
        # key `find()` can no longer answer for must not resurface from the
        # link memory at close, or the bound would govern live lookups only.
        if self._remembered is not None:
            self._remembered.pop(popped, None)
        return popped

    def _finalize_locked(
        self,
        *,
        status: StatusCode,
        error_type: str | None,
        end_ns: int,
    ) -> SpanDraft:
        """Stamp the unit's own span. Caller holds the lock; does not emit.

        Takes no marker argument, deliberately. A `marker=marker` keyword here
        would hand the limitation census a value it cannot follow AND — because
        any function passed a marker-ish name becomes a read-every-argument sink
        for the whole program — would make `status`, `end_ns` and `error_type`
        unreadable holes at every call site of every same-named method in the
        SDK. A marker is attached where it is DECIDED, by `note()`, which is
        also where the literal is spelled.
        """
        draft = self._draft
        if self._input_recorded or self._output_recorded:
            draft.set_io(
                input_data=bytes(self._input),
                output_data=bytes(self._output),
                input_attempted=self._input_recorded,
                output_attempted=self._output_recorded,
            )
            if self._input_truncated or self._output_truncated:
                draft.integrity.truncated(True)
                # The FACT is already on the wire (`CaptureIntegrity.truncated`
                # is a span field), so this is not a silent loss being made
                # visible — it is the AGGREGATE, which the span cannot answer.
                # "I lowered max_body_bytes; how much am I cutting now?" is a
                # question about a rate, and the cap only became configurable
                # in the same change that added this counter.
                counters.bump("assembly._units.record_truncated")
        draft.set_status(status)
        if error_type is not None:
            draft.set_error(error_type)
        draft.set_end_ns(end_ns)
        return draft


def _append_capped(buf: bytearray, data: bytes, cap: int) -> bool:
    """Append what fits. True when something was dropped."""
    room = cap - len(buf)
    if room <= 0:
        return bool(data)
    if len(data) <= room:
        buf += data
        return False
    buf += data[:room]
    return True


def _force_close(draft: SpanDraft, marker: Limitation, end_ns: int) -> SpanDraft:
    """A span ended by someone else's teardown, said so on the span."""
    draft.add_limitation(marker)
    draft.set_status(StatusCode.UNSET)
    draft.set_end_ns(end_ns)
    return draft


class UnitRegistry:
    """The table of live units, and the arbiter of `resolve()`.

    Everything that mutates state takes `_lock`; everything that emits does so
    after releasing it (I11). The two are separated by construction: the
    internal `_*_locked` helpers RETURN the drafts to emit and never touch the
    sink, and `_flush` is the only method that does.
    """

    __slots__ = (
        "_by_alias",
        "_debug",
        "_link_memory",
        "_live_units",
        "_lock",
        "_max_entries_per_unit",
        "_max_link_targets",
        "_max_record_bytes",
        "_max_total_units",
        "_max_units",
        "_roots",
        "_sink",
    )

    def __init__(
        self,
        *,
        sink: SpanSink,
        max_units: int | None = None,
        max_entries_per_unit: int | None = None,
        max_link_targets: int | None = None,
        max_body_bytes: int | None = None,
        debug: bool = False,
    ) -> None:
        """`max_*` default from the CORE, never from a Python literal.

        `LimitsConfig().resolved()` reads `_wardex_native.limits_defaults()`, so
        a default cannot drift from `crates/wardex-limits` — a hardcoded 512 here
        would agree with the core today and silently disagree the moment someone
        changed it there. TWO tests forbid the literal, and each fails on the
        disagreement rather than on the spelling:
        `test_limits.py::test_python_side_fallback_defaults_match_core` compares
        every Python-side fallback in the SDK against `limits_defaults()`, and
        `test_units.py::test_bounds_are_resolved_from_the_core_not_a_python_literal`
        states the same rule locally. Nothing else notices a drifted default:
        every other test that builds a registry either passes both bounds in
        explicitly or never opens enough units to reach one.

        Neither bound is `max_sessions`/`max_session_entries` under a new name:
        those bound one flat table of sessions and three per-session maps, while
        `open()` is the single entry point for all four `UnitKind`s, so one
        session plus 256 sub-agents plus 256 open calls would exhaust 512 on its
        own and "512 concurrent sessions" would quietly become "about one".
        """
        resolved = LimitsConfig().resolved()
        self._sink = sink
        self._debug = debug
        self._max_units = max_units if max_units is not None else resolved["max_units"]
        self._max_entries_per_unit = (
            max_entries_per_unit
            if max_entries_per_unit is not None
            else resolved["max_entries_per_unit"]
        )
        self._max_link_targets = (
            max_link_targets if max_link_targets is not None else resolved["max_link_targets"]
        )
        # Bytes a unit may accumulate through record_input/record_output. Not a
        # knob of its own: the payload ceiling the rest of the SDK already
        # applies to a captured body is the honest ceiling for one assembled
        # from many fragments. The PARAMETER is spelled after the knob rather
        # than after the attribute, because the two mean different things —
        # `max_body_bytes` caps one body, `_max_record_bytes` caps a record
        # accumulated from many — and the delivery table reads better when the
        # keyword and the field it comes from are the same word.
        self._max_record_bytes = (
            max_body_bytes if max_body_bytes is not None else resolved["max_body_bytes"]
        )
        # `max_units` bounds ROOTS, deliberately (a flat LRU would make "evict
        # the oldest unit" pick a long-lived session root nearly every time, so
        # one chatty session would evict OTHER sessions' roots). Children are
        # bounded per parent — which bounds BREADTH but not DEPTH, so a chain of
        # child-of-child units would grow forever under one live root. This
        # derived ceiling closes that without inventing a knob a user would have
        # to discover, and it is safe to enforce by evicting a ROOT because
        # every live unit is reachable from one: a child opened under a closed
        # parent is registered as a root (see `open`), so "live units exist but
        # no root does" is unreachable.
        self._max_total_units = self._max_units * self._max_entries_per_unit
        self._lock = threading.RLock()
        self._roots: dict[Unit, None] = {}
        self._live_units: dict[Unit, None] = {}
        self._by_alias: dict[UnitKey, Unit] = {}
        #: The closed-unit link memory: remembered alias keys -> the span
        #: context their unit owned when it closed. A plain dict, because
        #: insertion order IS the FIFO this file evicts by (`next(iter(...))`),
        #: matching every other bounded table here. Bounded by
        #: `max_link_targets` at the insertion site in `_detach_locked`.
        self._link_memory: dict[UnitKey, SpanContext] = {}

    @property
    def max_record_bytes(self) -> int:
        """The byte ceiling `record_input`/`record_output` enforce.

        The configured `max_body_bytes` (see `__init__`). Exposed so a describe
        function that SHAPES a payload before recording can stop materializing
        at exactly the boundary storage would cut — read at the enforcement
        point, because the two budgets cannot disagree when they are one read.
        Any other source can: reading `AdapterContext.limits` instead would put
        a second source under the same name, and this property spent a release
        being exactly that — the registry resolved the CORE default while
        `ctx.limits` reported the host's, so the shaper and the cap agreed with
        each other and both disagreed with the user.
        """
        return self._max_record_bytes

    # -- lifecycle -------------------------------------------------------

    def open(
        self,
        kind: UnitKind,
        key: UnitKey,
        *,
        ambient: Ambient,
        evidence: Evidence = AMBIENT,
        intent: SpanIntent,
        subject: str | None = None,
        parent_unit: Unit | None = None,
        aliases: Sequence[UnitKey] = (),
        start_ns: int | None = None,
        owner: str | None = None,
    ) -> Unit:
        """Open a unit and the span it owns.

        `owner` names the adapter this unit belongs to, for the day one
        process-wide registry serves several at once. Defaulting it to None
        keeps every caller's meaning unchanged; what it must never become is a
        default that matches — see `sole_live`.

        `ambient` has no default on purpose. It is a snapshot taken by
        `latch_ambient()` at the moment the work was ISSUED, on the task that
        issued it; letting the registry latch it for the caller would move that
        read to whenever the framework happened to call us, which is the single
        most likely way an adapter silently breaks the tree. A unit that hangs
        off another unit passes `EMPTY_AMBIENT` and `parent_unit`.

        A scope left behind by a unit of this registry that has already CLOSED
        — a leaked pin, or an `activate()` fork `close()` could not take down —
        is refused here for the same reason `resolve()` refuses it: it is a
        finished unit's own context, so a unit opened from it would be a fresh
        subtree hanging off a dead session at confidence 1.0 with nothing on the
        wire to say so. The unit becomes a trace root instead and carries
        `CORRELATION_CONFLICT` — or `INSTRUMENTATION_DEGRADED` when the unit
        died by the registry's OWN eviction (`refused_ambient_marker`): the two
        strands have different repairs.
        """
        now = start_ns if start_ns is not None else time.time_ns()
        if parent_unit is not None:
            if evidence is AMBIENT:
                evidence = _IN_UNIT
            parentage = parent_unit.child(evidence)
            conversation = parent_unit.conversation
            tracestate = parent_unit.tracestate
        elif self._poisoned(ambient):
            self._note_refused_ambient()
            parentage = resolve_parentage(EMPTY_AMBIENT, evidence).with_limitation(
                self.refused_ambient_marker()
            )
            conversation = parentage.conversation
            tracestate = parentage.tracestate
        else:
            parentage = resolve_parentage(ambient, evidence)
            conversation = parentage.conversation
            tracestate = parentage.tracestate

        draft = SpanDraft(
            parentage,
            intent=intent,
            subject=subject,
            source=CaptureSource.ADAPTER,
            start_ns=now,
        )

        unit = Unit(
            self,
            key=key,
            kind=kind,
            draft=draft,
            parentage=parentage,
            conversation=conversation,
            tracestate=tracestate,
            start_ns=now,
            parent=parent_unit,
            owner=owner,
        )

        pending: list[SpanDraft] = []
        with self._lock:
            # Make room BEFORE attaching. Attaching first and evicting after
            # lets the arriving unit be closed as part of the very subtree its
            # own arrival evicted — it would be handed to the sink before it was
            # ever returned, and then tracked as live anyway.
            while len(self._live_units) >= self._max_total_units and self._roots:
                pending += self._evict_root_locked(now)
            if parent_unit is not None and parent_unit.is_live and parent_unit._registry is self:
                evicted = parent_unit._evict_oldest(parent_unit._children, "child")
                if evicted is not None:
                    # Only the entry that HIT the bound carries the table-full
                    # fact; its descendants, closed by the same walk below, keep
                    # CHILD_SPAN_UNCLOSED — they truly were closed by their
                    # parent's teardown, which is that member's exact sentence.
                    # The breadcrumb is written BEFORE the close, which is what
                    # makes `refused_ambient_marker`'s lock-free read safe: a
                    # fork that reads its unit as dead already reads it as
                    # evicted when it was.
                    evicted[0]._evicted = True
                    evicted[0].note(Limitation.UNIT_TABLE_FULL)
                    pending += self._close_locked(
                        evicted[0], status=StatusCode.UNSET, error_type=None, end_ns=now
                    )
                parent_unit._children[unit] = None
            else:
                # A child of a CLOSED — or FOREIGN — parent is a root for
                # bookkeeping. Its span still hangs off the parent's context (a
                # closed unit's span was already emitted, so it is a perfectly
                # good parent) but its LIFETIME cannot be managed by a unit this
                # registry does not own or that no longer has a child table:
                # left as a child it would be reachable from no root of EITHER
                # registry, so no bound would ever evict it and `close_all`
                # would never see it. A foreign parent also means the per-parent
                # breadth bound applied would be the OTHER registry's.
                if parent_unit is not None and not parent_unit.is_live:
                    counters.bump("assembly._units.parent_closed")
                elif parent_unit is not None:
                    counters.bump("assembly._units.parent_foreign_registry")
                while len(self._roots) >= self._max_units:
                    pending += self._evict_root_locked(now)
                self._roots[unit] = None
            self._live_units[unit] = None
            # Guarded from the INSIDE, and the placement is the whole point. By
            # this line the unit is already in `_roots` and `_live_units`, so a
            # fault here is not "the open failed" — it is "the open succeeded
            # and the lookup table did not". A guard one level up, around the
            # whole call, would therefore contain the raise and LEAK the unit:
            # registered, live, reachable from no caller, counting against
            # `max_units` until it evicts a real session to make room for a
            # phantom, once per call. Guarded here the fault costs the ALIAS —
            # `find(key)` misses — and the span still ships.
            with guard("assembly._units.open_bind", debug=self._debug):
                self._bind_alias_locked(key, unit)
                for extra in aliases:
                    self._bind_alias_locked(extra, unit)
        self._flush(pending)
        return unit

    def alias(self, key: UnitKey, alias: UnitKey) -> None:
        """Point `alias` at whatever `key` already resolves to.

        A no-op when `key` names nothing live — an alias for a unit that does not
        exist would otherwise be a parent edge waiting to be handed out from a
        table nobody populated.
        """
        with self._lock:
            unit = self._by_alias.get(key)
            if unit is None or not unit.is_live:
                counters.bump("assembly._units.alias_unknown_key")
                return
            self._bind_alias_locked(alias, unit)

    def bind_alias(self, unit: Unit, key: UnitKey, *, remember: bool = False) -> None:
        """Point `key` at `unit` directly — the handed-a-unit twin of `alias()`.

        `remember=True` additionally opts the key into the closed-unit link
        memory: the unit's own span context is captured HERE, at bind time,
        and `_detach_locked` moves it into `_link_memory` when the unit
        closes — which is what keeps the detach phase's "cannot fail"
        property intact, since the context to remember was read while the
        unit was alive and under this same lock.

        A dead unit is refused and counted (`alias_after_close`), exactly as
        `note()` and `open_span()` refuse: its span has already shipped, so a
        fresh alias could neither be found live nor be recorded at a close
        that already happened.
        """
        with self._lock:
            if not unit.is_live:
                counters.bump("assembly._units.alias_after_close")
                return
            self._bind_alias_locked(key, unit)
            if remember:
                if unit._remembered is None:
                    unit._remembered = {}
                unit._remembered[key] = unit.context

    def find(self, alias: UnitKey) -> Unit | None:
        """Resolve a framework id to a live unit, or None.

        Returns `Unit | None` and nothing else. That is the mechanical half of
        I2: on a miss the caller's only options are `sole_live()` (0.5 plus a
        marker) or a new trace (0.0 plus `PARENT_UNRESOLVED`). "Silently
        re-parent an uninterpretable id into the ambient scope, indistinguishable
        from a real attachment" is not expressible.
        """
        with self._lock:
            unit = self._by_alias.get(alias)
            if unit is None:
                return None
            if not unit.is_live:
                del self._by_alias[alias]
                return None
            return unit

    def resolve_link_target(self, selector: UnitKey) -> SpanContext | None:
        """A span context to LINK to — live alias first, then closed memory.

        The ONE sanctioned `UnitKey` -> `SpanContext` conversion in the SDK,
        and its answer feeds `SpanDraft.add_link` and nothing else: a link is
        causality where the parent edge is containment (I2), so a context
        that names a finished predecessor cannot become a parent through it.
        `find()` deliberately stays live-only — every parentage path keeps
        asking it, and a miss there still forces the caller down the honest
        ladder rather than toward a shipped span.

        The memory half answers only for a key's MOST RECENT holder:
        `_bind_alias_locked` pops the entry whenever the key is (re)bound, so
        a stale predecessor cannot resurface after an unremembered successor
        closes, and a cycle re-executing one node name resolves latest-wins.
        """
        with self._lock:
            unit = self._by_alias.get(selector)
            if unit is not None and unit.is_live:
                return unit.context
            return self._link_memory.get(selector)

    def current(self) -> Unit | None:
        """The unit made ambient by `activate()` or a pin, if any.

        ContextVar-backed, so an asyncio task created after the installation
        inherits it by ordinary context copying — which is exactly how a hook
        callback and an in-process tool handler see their session with zero
        framework identifiers and confidence 1.0.

        A carrier whose unit has since CLOSED returns None rather than the dead
        unit. That is the detection the pin restriction asks for: a pin leaked
        onto a pooled worker is self-consistent (same task, same ContextVar) and
        would otherwise parent unrelated later work into a finished unit, which
        no confidence value or marker downstream could reveal.

        Ownership is checked FIRST, ahead of liveness, and the order is not
        cosmetic: a foreign CLOSED unit would otherwise bump `pin_stale` /
        `pin_leaked` and attribute another registry's teardown to this
        registry's pin discipline, polluting the very counters the pin
        restriction is audited by.
        """
        entry = _ambient_unit.get()
        if entry is None:
            return None
        if entry.unit._registry is not self:
            # The ambient carrier is process-wide; a unit is not. A pin installed
            # by a previous `init()`'s registry is not this registry's parent to
            # hand out, and the staleness gate below does NOT catch it: nothing
            # closes a dropped registry's units, so it is still `is_live`.
            counters.bump("assembly._units.ambient_foreign_registry")
            return None
        if not entry.unit.is_live:
            same_task = entry.owner is _current_task()
            if entry.pinned:
                counters.bump(
                    "assembly._units.pin_stale" if same_task else "assembly._units.pin_leaked"
                )
            else:
                counters.bump("assembly._units.ambient_stale")
            return None
        return entry.unit

    def closed_unit_in_scope(self) -> bool:
        """Is the scope on this task the leftover of a unit that has DIED?

        `current()` refusing the dead unit is only half of an activation's
        safety, and the other half is what makes the first half matter. A
        carrier installs TWO things (`_Carrier.__init__`): the ambient UNIT,
        which `current()` gates on liveness, and the ambient SPAN CONTEXT in the
        scope, which nothing can invalidate — `close()` may run on a different
        task, and a ContextVar cannot be reset from one. So after the unit
        closes, the very next `latch_ambient()` still returns the DEAD unit's
        own span context, and an edge built from it reads `contextvar` / 1.0 /
        no marker: the failure `current()`'s docstring says it exists to
        prevent, arriving by the other carrier.

        PINNED OR NOT (design §10.3), and the widening is the whole of (a).
        This gated on `entry.pinned` on the reading that only a pin installs a
        fork nobody can take down. That was wrong about `activate()`, and
        measurably so: a generator-scoped activation (LangGraph's
        `Pregel.stream`) is entered on the carrier that pumps the first
        `next()` and can only be undone when the generator is FINALIZED, on
        whatever carrier finalizes it. A `close_units()` mid-stream, or a
        finalization on a foreign carrier, strands exactly the same corpse — and
        it was not merely uncounted, it was a CONFIDENT PARENT: an entire later
        graph run adopted into an already-shipped span at 1.0 with no marker,
        one trace where two belong. A pin is one way to strand a fork, not the
        definition of one. `entry.pinned` survives below only to NAME a counter,
        which is exactly the role it already plays in `current()`.

        This predicate is what lets `open()` and `resolve()` refuse that scope
        and SAY SO on the span whose parent edge it would have decided — design
        §5.6 asks for a hard `CORRELATION_CONFLICT` rather than an internal
        count (or `INSTRUMENTATION_DEGRADED` when the strand is wardex's own
        eviction — `refused_ambient_marker`), because the consequence is a tree
        shape and a tree shape has to be falsifiable from the data.

        LIVENESS, not §5.6's literal "any read from a different task id". Every
        hook callback and every in-process tool handler runs on a descendant
        task, which is the mechanism the whole design rests on; the literal rule
        would stamp `CORRELATION_CONFLICT` on the product's own exhibit A.

        The NAME is the question the predicate answers: a CLOSED unit of THIS
        registry left its span context in this scope — whether a pin, an
        `activate()` leftover or the registry's own eviction stranded it. The
        counters underneath keep the three-way split (`stale_pin_ambient` /
        `stale_activation_ambient` / `stale_ambient_evicted`, via
        `_note_refused_ambient`) because they name the REPAIR — which task
        was pinned vs the adapter's lifetime vs `max_units` — not the
        predicate. The implementation is `_closed_ambient_context`.
        """
        return self._closed_ambient_context() is not None

    def _closed_ambient_context(self) -> SpanContext | None:
        """The span context a CLOSED unit of THIS registry left in this scope.

        Ownership first, for the reason `current()` gives and one more. The
        ambient carrier is process-wide and a unit is not, so a unit a previous
        `init()`'s registry installed and then closed reads here as THIS
        registry's corpse: `becomes_trace_root` would orphan a nested site and
        `open()` would stamp `CORRELATION_CONFLICT`, both charging a conflict to
        an activation this registry never installed — and doing it while the
        scope in front of us is a perfectly ordinary one.

        The foreign COUNTER stays gated on `entry.pinned` even though the
        refusal no longer is, and that is a decision rather than a leftover.
        `stale_pin_foreign_registry` audits PIN DISCIPLINE — by its name and by
        the test that reads it — while two adapters running side by side put
        each other's perfectly healthy `activate()` on this carrier
        continuously. Counting those would fire the pin audit on every `open()`
        of an ordinary two-adapter process and bury the signal it exists to
        carry; `current()`'s `ambient_foreign_registry` already counts the
        general event.
        """
        entry = _ambient_unit.get()
        if entry is None:
            return None
        if entry.unit._registry is not self:
            if entry.pinned:
                counters.bump("assembly._units.stale_pin_foreign_registry")
            return None
        if entry.unit.is_live:
            return None
        return entry.unit.context

    def _note_refused_ambient(self) -> None:
        """Count a refused leftover, naming which primitive stranded it.

        Three names for one refusal, because they are three repairs — the same
        axis, and the same split, `current()` already makes between
        `pin_stale`/`pin_leaked` and `ambient_stale`. `stale_pin_ambient` says a
        RESTRICTED pin outlived its unit: `pin_driver`'s contract was broken and
        the fix is in which task the adapter pinned. `stale_activation_ambient`
        says an `activate()` scope could not be unwound where it was installed —
        a generator finalized on a foreign carrier, a `close_units()` that ran
        inside one — and the fix is in the adapter's LIFETIME, not its pinning.
        `stale_ambient_evicted` says the registry's own bound stranded the
        scope — nothing the adapter did was wrong, and the repair, if any, is
        `max_units`. An evicted PIN lands there too: the pin's contract was not
        broken, the table was full. One number covering all three leaves an
        operator with three hypotheses and no way to separate them.
        """
        entry = _ambient_unit.get()
        if entry is not None and entry.unit._evicted:
            counters.bump("assembly._units.stale_ambient_evicted")
        elif entry is not None and entry.pinned:
            counters.bump("assembly._units.stale_pin_ambient")
        else:
            counters.bump("assembly._units.stale_activation_ambient")

    def refused_ambient_marker(self) -> Limitation:
        """The word the refusal carries: whose fault is the stranded scope?

        Not a third liveness state — `_poisoned` and every resolution tier are
        unchanged; this chooses only the MARKER after the refusal is already
        decided. A fork stranded because the registry itself EVICTED its unit
        (`Unit._evicted`) reads as wardex's own bound at work, so the refused
        span carries `INSTRUMENTATION_DEGRADED`: the repair is `max_units`, not
        the adapter's pin or lifetime discipline — `resolve_observed` already
        says exactly this for a byte-seam span whose latched parent wardex
        discarded to stay inside a bound (the h2 stream latch). Every other
        strand keeps `CORRELATION_CONFLICT`: two answers to one parent
        question, the disagreement on the wire.

        PUBLIC because the registry is not the only caller with a refusal to
        word: `AdapterContext._open`'s declared fallback takes a parent unit,
        which is exactly what stops `open()` from seeing the poisoned ambient
        for itself, so the adapter surface asks here rather than spelling a
        marker it cannot choose correctly.

        Read without the lock, like `_closed_ambient_context`: `_evicted` is
        written under the lock BEFORE the unit is detached, so a fork that
        reads as dead already reads as evicted when it was.
        """
        # Statement form, not a ternary: the census scanner reads the value of
        # every assignment to a marker-ish name, and a single expression would
        # put the CONDITION in the slot too — each branch here holds exactly
        # the member it decides, which is what keeps this helper a recorded
        # decision rather than a hiding place.
        entry = _ambient_unit.get()
        if entry is not None and entry.unit._evicted:
            marker = Limitation.INSTRUMENTATION_DEGRADED
        else:
            marker = Limitation.CORRELATION_CONFLICT
        return marker

    def _poisoned(self, amb: Ambient) -> bool:
        """Is `amb` the leftover fork of a unit of THIS registry that has died?

        The identity check, not merely `closed_unit_in_scope()`, and the narrowing
        is deliberate. A dead unit in scope says the TASK is descended from one;
        it does not say the scope still holds the dead unit's own span. A host
        that opened its own span inside that task, or a handler inside a nested
        `activate()`, put a real span on top of the leftover — refusing THAT
        would move live work out of the host's trace to escape a ghost that is
        no longer in front of us, which is a wrong tree of a different shape.
        `test_a_real_span_over_a_stale_pin_is_still_a_parent` is the standing
        promise, and the §10.3 widening deliberately does not touch it: what
        changed is which leftovers count as leftovers, not the identity test
        that keeps a real parent a parent.

        What is refused is exactly the measured failure: the fork `activate()`
        or a pin installed, still current after `close()` could not take it
        down, read back as `contextvar` / 1.0 / no marker into a finished unit.
        """
        stale = self._closed_ambient_context()
        return stale is not None and amb.span_context is not None and amb.span_context == stale

    def becomes_trace_root(self, amb: Ambient) -> bool:
        """Would a unit opened from `amb` alone begin a NEW trace?

        The question a caller has to ask BEFORE `open()` in order to declare a
        site that may not legitimately start one — and it is answered here, by
        the object that will answer it again inside `open()`, because asking it
        anywhere else in different words is precisely how the two drift.

        They did. `AdapterContext._evidence` asked `closed_unit_in_scope()`
        (under its earlier, narrower stale-pin spelling), which says this TASK
        descends from a dead driver, while `open()` refuses only
        the dead fork ITSELF. A host that opened its own span inside that task
        put a real parent on top of the leftover, so the two answers disagreed
        exactly there: the adapter surface orphaned live work at confidence 0.0
        to escape a ghost that was no longer in front of it, and dropped the
        `CORRELATION_CONFLICT` that had been the only record of the dead pin.
        `_poisoned`'s own docstring calls that a wrong tree of a different
        shape, and `test_a_real_span_over_a_stale_pin_is_still_a_parent` is the
        registry's standing promise not to build it.
        """
        return self._poisoned(amb) or amb.span_context is None

    def sole_live(self, kind: UnitKind, *, owner: str | None = None) -> Unit | None:
        """A unit ONLY when exactly one of `kind` is live.

        Callers must stamp `ParentSource.UNIT_SOLE` (0.5) and
        `Limitation.UNIT_INFERRED_SOLE`; `resolve()` does. It exists because the
        adapter it replaces DROPS a hook entirely when two sessions are live,
        with no marker at all — and a marked low-confidence edge beats unmarked
        data loss.

        `owner` is the filter that keeps this answer honest once one registry
        serves several adapters. Without it "exactly one SESSION is live" is a
        question about the PROCESS, so an Anthropic run with a LangGraph run
        beside it would either find two and give up, or — worse, with only the
        other framework running — hand the Anthropic adapter a LangGraph
        session as the sole candidate and stamp it 0.5. An unowned unit matches
        nothing rather than everything: guessing across an unstated boundary is
        the failure this filter exists to prevent, so it may not be the way an
        omitted owner degrades.
        """
        with self._lock:
            found: Unit | None = None
            for unit in self._live_units:
                if unit.kind is not kind:
                    continue
                if owner is not None and unit.owner != owner:
                    continue
                if found is not None:
                    return None
                found = unit
            return found

    # -- the whole of I2 -------------------------------------------------

    def resolve(self, alias: UnitKey | None, *, ambient: Ambient | None = None) -> Parentage:
        """Turn an optional framework id plus the live scope into one edge.

        MOST-SPECIFIC-WINS. The earlier rule — "ambient wins whenever the traces
        match" — flattens sub-agent subtrees: under a pin the hook task's ambient
        is always the session root, while an alias carrying an `agent_id`
        resolves to the sub-agent span, a STRICT DESCENDANT of that root and
        therefore the more specific answer. Getting it backwards collapses every
        sub-agent into its session, silently and at confidence 1.0.

        A scope left behind by a CLOSED unit is refused before any of that. It
        is the dead unit's own context, so every branch that consults `amb` would
        re-attach unrelated later work to a finished unit — at 1.0 with no
        marker on the widest branch. The edge is rebuilt from the remaining
        tiers (alias -> sole_live -> UNRESOLVED) and carries
        `CORRELATION_CONFLICT`, which is the same mechanism this method already
        uses for a cross-trace disagreement: a wrong tree is invisible, a marked
        conflict is a shippable bug report. (An evict-origin strand carries
        `INSTRUMENTATION_DEGRADED` instead — see `refused_ambient_marker`.)
        """
        amb = ambient if ambient is not None else latch_ambient()
        if self._poisoned(amb):
            self._note_refused_ambient()
            return self._edge(alias, EMPTY_AMBIENT).with_limitation(self.refused_ambient_marker())
        return self._edge(alias, amb)

    def _edge(self, alias: UnitKey | None, amb: Ambient) -> Parentage:
        """`resolve()`'s precedence ladder, with the ambient already vetted.

        Split out so the staleness refusal is stated once instead of at each of
        the six returns below — and so a later tier added here cannot forget it.
        """
        unit = self.find(alias) if alias is not None else None
        hint = alias.value if alias is not None else None

        if unit is not None and amb.span_context is not None:
            if unit.context.trace_id == amb.span_context.trace_id:
                if self._is_descendant(unit, amb.span_context):
                    # Context established the tree; the id picked the node inside
                    # it. Design §5.2 asks for confidence 1.0 here and does not
                    # get it: `Evidence.confidence` is CLAMPED to the source's
                    # table default (0.9 for an alias) because a hint may only
                    # LOWER trust, and that clamp is what stops any caller
                    # promoting a guess into a fact. Corroboration by descent is
                    # real, but spending the clamp on it would reopen the hole
                    # for every other caller; the alias still WINS the branch,
                    # which is the part that decides the tree's shape.
                    return unit.child(Evidence(ParentSource.UNIT_ALIAS, request_id=hint))
                # Same trace, not a descendant: the live context is the better
                # answer and the id merely corroborates it.
                return resolve_parentage(amb, Evidence(ParentSource.CONTEXTVAR, request_id=hint))
            # Different traces. Context wins and the disagreement is RECORDED —
            # a wrong tree is invisible, a marked conflict is a shippable bug
            # report.
            return resolve_parentage(
                amb, Evidence(ParentSource.CONTEXTVAR, confidence=0.8, request_id=hint)
            ).with_limitation(Limitation.CORRELATION_CONFLICT)

        if unit is not None:
            # Context did not reach us (IPC, a bare thread). Attach to a context
            # captured earlier ON THE CORRECT TASK, addressed by the framework id.
            return unit.child(Evidence(ParentSource.UNIT_ALIAS, request_id=hint))

        if amb.span_context is not None:
            return resolve_parentage(amb, AMBIENT)

        sole = self.sole_live(UnitKind.SESSION)
        if sole is not None:
            return sole.child(Evidence(ParentSource.UNIT_SOLE, request_id=hint)).with_limitation(
                Limitation.UNIT_INFERRED_SOLE
            )

        return resolve_parentage(
            EMPTY_AMBIENT, Evidence(ParentSource.UNRESOLVED, request_id=hint)
        ).with_limitation(Limitation.PARENT_UNRESOLVED)

    def _is_descendant(self, unit: Unit, ancestor: SpanContext) -> bool:
        """Is `unit` strictly below the span `ancestor` names?

        Decided by walking the unit's ANCESTOR CHAIN, which this registry owns —
        not by a reverse index over arbitrary span contexts. A reverse index
        would let any span id that happened to be in the table claim ancestry,
        which is the framework-id-becomes-parent move I2 rules out.
        """
        node = unit.parent
        while node is not None:
            if node.context.span_id == ancestor.span_id:
                return True
            node = node.parent
        return False

    # -- the restricted primitive ----------------------------------------

    def pin_driver(self, unit: Unit, *, owner_task: object) -> PinToken:
        """Install `unit` as ambient for the REMAINDER OF THE CURRENT TASK.

        RESTRICTED. Legal only on a long-lived driver task the adapter itself
        created or patched and whose lifetime is bounded by the unit — a
        transport read loop. It is ILLEGAL on a pooled or shared worker, where a
        leaked pin would silently parent unrelated later work into a dead unit.
        Reachable only through the registry, never exported as a free function,
        because the free-function form put that leak one import away for every
        adapter author (a CrewAI ThreadPoolExecutor worker, a uvicorn/anyio
        worker, an ADK `run_in_executor` thread).

        `owner_task` is the caller's DECLARATION of which task it is pinning, and
        the registry checks it against the task it actually observes. A mismatch
        is refused, not corrected: a `ContextVar.set()` lands on the calling
        task, so pinning "on behalf of" another task would install the unit
        somewhere the caller did not mean and leave the named task unpinned —
        a self-consistent lie. The refusal is recorded as a hard
        `CORRELATION_CONFLICT` on the unit's own span rather than an internal
        counter, because the consequence is a tree shape and a tree shape has to
        be falsifiable from the data.

        The returned token is a HANDLE, not the pin. The pin's scope fork is
        installed by `install_span`, which has no finaliser by design, so
        dropping the token cannot undo it — `_assembler.py` reads `.installed`
        off a temporary and discards the rest, which is the shape a handle whose
        only advertised member is a bool has to survive.

        Staleness is the other half and is enforced in `current()`: a pinned unit
        that has closed stops being ambient.
        """
        observed = _current_task()
        if owner_task is not observed:
            counters.bump("assembly._units.pin_foreign_task")
            unit.note(Limitation.CORRELATION_CONFLICT)
            return PinToken(unit, observed, None)
        return PinToken(unit, observed, _Carrier(unit, pinned=True))

    def unpin(self, token: PinToken) -> None:
        """Remove a pin. MUST run on the task that installed it.

        Counted rather than raised when it does not: `ContextVar.reset` throws
        `ValueError` from another Context, and a teardown path is exactly where
        that would reach the host.
        """
        if token._carrier is None:
            return
        token._carrier.remove(where="assembly._units.unpin", debug=self._debug)
        token._carrier = None
        token.installed = False

    # -- closing ---------------------------------------------------------

    def close(
        self,
        unit: Unit,
        *,
        status: StatusCode = StatusCode.OK,
        error_type: str | None = None,
        end_ns: int | None = None,
    ) -> None:
        """Close a unit, its surviving children and its open spans.

        Children and open drafts are force-closed with `CHILD_SPAN_UNCLOSED`;
        the unit's own span is emitted LAST so a consumer that streams sees the
        subtree before its root. Every span is handed to the sink AFTER the lock
        is released (I11).
        """
        end = end_ns if end_ns is not None else time.time_ns()
        with self._lock:
            pending = self._close_locked(unit, status=status, error_type=error_type, end_ns=end)
        self._flush(pending)

    def close_all(self, *, reason: Limitation, owner: str | None = None) -> None:
        """Close every live root. The teardown BACKSTOP, not the teardown itself.

        A caller that owns state ABOUT a unit — the Agent SDK assembler owns a
        session's model, conversation and still-open tool spans — must finalize
        through its own path first, or this emits a root stripped of everything
        that caller had not yet stamped onto it. What is left afterwards is the
        units no such caller owns: an in-process tool call opened with no
        holder is registered as a root here and is reachable from nowhere else.
        Those are what this closes.

        `reason` is the caller's — `ADAPTER_UNINSTALLED` from an uninstall,
        `UNIT_INTERRUPTED` from a cancelled or shutting-down process — so the
        limitation census records this as a slot it cannot follow rather than
        pretending to have read it.

        `owner` scopes the sweep to one adapter, for the day one registry serves
        several: an Anthropic uninstall must not close a LangGraph run that is
        still being driven. Omitted, it closes everything, which is what a
        process-wide teardown wants and what every caller means today.

        Declines rather than proceeds when this thread already holds the lock.
        The lock is an `RLock`, so re-entering does not deadlock — it does
        something worse, and quietly: a signal handler runs on the main thread
        at an arbitrary bytecode boundary, so it can land INSIDE a half-finished
        mutation, walk `self._roots` while an outer frame is mid-update, and
        emit from state no reader was ever supposed to see. Declining costs a
        teardown that was already racing a dying process; proceeding costs
        correctness at the one moment nothing can be re-run.

        The signal handler is no longer the only shape of that. A weakref
        finalizer — the seams' close hook backstop — runs out of its referent's
        DEALLOCATION, so it lands wherever a reference count reaches zero (and
        at any allocation, through a cyclic collection) and on whatever thread
        dropped that reference: the same half-finished mutation is now reachable
        off the main thread too.
        Nothing routes a finalizer here today, and this guard does not depend on
        that: it asks the lock who owns it, which is true of every re-entry
        route there is or will be.
        """
        # `_is_owned()` and not `acquire(blocking=False)`: on an `RLock` the
        # non-blocking acquire SUCCEEDS for the thread that already owns it, so
        # spelling the guard that way passes every re-entry straight through
        # while looking like it checks something. The attribute is private but
        # not optional — this SDK ships a compiled extension and runs only where
        # that imports, and there `threading.RLock()` is `_thread.RLock`.
        if self._lock._is_owned():
            counters.bump("assembly._units.close_all_reentrant")
            return
        end = time.time_ns()
        with self._lock:
            pending: list[SpanDraft] = []
            # Snapshot, not `while self._roots`, because with an owner filter the
            # loop no longer empties the table and the old form would spin
            # forever on the first root it must not close.
            for root in list(self._roots):
                if owner is not None and root.owner != owner:
                    continue
                # PER ROOT, and then evict the one that failed. One guard around
                # the whole loop would let a single torn subtree cost every root
                # in the table — measured on 3 roots x 4 children with one fault:
                # a whole-sweep guard emits 0 of 15 spans, per-root emits 10.
                # And per-root ALONE leaves the failing root standing, so every
                # later sweep walks back into the same fault and the table never
                # empties; evicting it costs the two units already detached from
                # it and un-wedges the sweep. Both losses are real and neither is
                # a fix — see `_close_locked` for what a real one would take.
                closed = False
                with guard("assembly._units.close_all_root", debug=self._debug):
                    # The marker is its OWN step, and the nesting is not
                    # decoration. `note()` writes to the root's draft, so a draft
                    # this sweep cannot touch would take the close with it —
                    # measured: three roots of four children, every draft broken,
                    # and the twelve children are left in `_live_units` reachable
                    # from no root, because the eviction below drops the root
                    # without walking under it. A marker that cannot be recorded
                    # costs the marker.
                    with guard("assembly._units.close_all_note", debug=self._debug):
                        root.note(reason)
                    pending += self._close_locked(
                        root, status=StatusCode.UNSET, error_type=None, end_ns=end
                    )
                    closed = True
                if not closed:
                    self._roots.pop(root, None)
                    self._live_units.pop(root, None)
        self._flush(pending)

    def _evict_root_locked(self, end_ns: int) -> list[SpanDraft]:
        """Close the oldest ROOT because a bound was reached. Caller holds the lock.

        The eviction EMITS. What this replaces dropped the oldest session and
        its root span outright — no marker, no test — so a user whose workload
        crossed the cap saw traces simply stop appearing (I10).
        """
        # Breadcrumb BEFORE the close (write-before-close is what makes the
        # lock-free read in `refused_ambient_marker` safe: any fork that reads
        # this unit as dead already reads it as evicted when it was).
        oldest = next(iter(self._roots))
        oldest._evicted = True
        oldest.note(Limitation.UNIT_EVICTED)
        return self._close_locked(oldest, status=StatusCode.UNSET, error_type=None, end_ns=end_ns)

    def _close_locked(
        self,
        unit: Unit,
        *,
        status: StatusCode,
        error_type: str | None,
        end_ns: int,
    ) -> list[SpanDraft]:
        """Detach a unit's whole subtree and return the drafts to emit.

        Never touches the sink. That separation is what makes I11 a property of
        the code rather than a rule reviewers have to remember.

        Closing a dead unit is a COUNTED no-op, not a silent one. Every other
        "you asked me to act on a unit that is gone" path here already counts
        (`note_after_close`, `open_span_after_close`, `ambient_stale`,
        `pin_stale`); this is the one that discards a whole ROOT span, because a
        caller holding its own table of sessions finalizes one the registry
        already evicted — it stamps the conversation id, the model name and the
        turn totals onto the draft, hands it here, and gets nothing emitted.
        The counter is the difference between a class of bug that is invisible
        and one that shows up in a counter snapshot.

        TWO PHASES, and the split is the whole reason one broken draft no longer
        costs a subtree. Building a span reads drafts, buffers and framework
        values; unlinking a unit is dict and list operations on wardex's own
        containers. Interleaved — which is what this was — a fault while
        building the third span of a six-child subtree left the parent already
        marked dead, some children already unlinked, the drafts collected so far
        dropped on the floor with the raise, and the rest of the subtree
        reachable from no root of any registry. Measured on three roots of four
        children with one fault: five spans gone and two units permanently
        unreachable.

        So COLLECT first (`_collect_locked`, guarded per unit) and DETACH after
        (`_detach_locked`, which cannot fail). A draft that cannot be built costs
        its own span and nothing else — not its siblings', not its parent's, and
        not the tables' consistency.
        """
        if not unit.is_live:
            counters.bump("assembly._units.close_after_close")
            return []
        pending: list[SpanDraft] = []
        doomed: list[Unit] = []
        self._collect_locked(
            unit,
            status=status,
            error_type=error_type,
            end_ns=end_ns,
            pending=pending,
            doomed=doomed,
        )
        for dead in doomed:
            self._detach_locked(dead)
        return pending

    def _collect_locked(
        self,
        unit: Unit,
        *,
        status: StatusCode,
        error_type: str | None,
        end_ns: int,
        pending: list[SpanDraft],
        doomed: list[Unit],
    ) -> None:
        """PHASE ONE: build every span this subtree owes, mutating no table.

        Guarded PER UNIT and per open draft, which is the point rather than
        belt-and-braces: the alternative is one boundary around the whole walk,
        and one boundary means one broken draft anywhere costs every span in the
        subtree. Here it costs exactly its own.

        `doomed` is appended to BEFORE the recursion, so it comes back
        parent-first — which is the order `_detach_locked` wants, since a parent
        that has already cleared its child table makes every later child's
        unlink a no-op instead of a second search.

        Nothing here touches `_roots`, `_live_units`, `_by_alias` or `_live`. A
        fault therefore leaves the subtree exactly as it was found: still
        reachable, still closable, and the drafts already built are re-buildable
        because `_finalize_locked` only stamps and `note()` is idempotent.
        """
        doomed.append(unit)

        for child in list(unit._children):
            if not child.is_live:
                counters.bump("assembly._units.close_after_close")
                continue
            with guard("assembly._units.close_note_child", debug=self._debug):
                child.note(Limitation.CHILD_SPAN_UNCLOSED)
            self._collect_locked(
                child,
                status=StatusCode.UNSET,
                error_type=None,
                end_ns=end_ns,
                pending=pending,
                doomed=doomed,
            )

        for entry in list(unit._open.values()):
            with guard("assembly._units.close_open_span", debug=self._debug):
                # The loser of an arbitration stays lost. A teardown that emitted
                # it would produce the double emit the arbitration exists to
                # prevent, in the one case `claim()` names as safe: the loser
                # that never asks again because the session was aborted between
                # the two observers.
                if unit._superseded_locked(entry):
                    counters.bump("assembly._units.claim_superseded")
                    continue
                pending.append(_force_close(entry.draft, Limitation.CHILD_SPAN_UNCLOSED, end_ns))

        with guard("assembly._units.close_finalize", debug=self._debug):
            pending.append(
                unit._finalize_locked(status=status, error_type=error_type, end_ns=end_ns)
            )

    def _detach_locked(self, unit: Unit) -> None:
        """PHASE TWO: unlink one unit from every table. Cannot fail.

        Dict and list operations on wardex's own containers, plus counter
        bumps, and nothing else — no draft is read, no span is built, no host
        value is touched (the contexts moved into the link memory below were
        captured at bind time for exactly this reason). That is not a claim
        about care taken here; it is the property the phase split exists to
        create, and it is why this half needs no guard and no partial
        recovery. Whatever phase one managed to build ships; whatever it did
        not is one span, and the tables end consistent either way.
        """
        unit._live = False
        unit._children.clear()
        unit._open.clear()
        self._roots.pop(unit, None)
        self._live_units.pop(unit, None)
        if unit.parent is not None:
            unit.parent._children.pop(unit, None)
        # Remembered aliases move into the closed-unit link memory BEFORE the
        # alias purge, because the ownership test needs `_by_alias` intact: a
        # key rebound to a LIVE unit since is that unit's to remember, not
        # this corpse's. Pop-then-set refreshes insertion order, so the FIFO
        # evicts by recency of CLOSE. The eviction is counted and not marked:
        # the remembered span already shipped, and what an eviction costs is a
        # link on a FUTURE span — there is no span yet to say so on (I10).
        if unit._remembered:
            for key, ctx in unit._remembered.items():
                if self._by_alias.get(key) is unit:
                    self._link_memory.pop(key, None)
                    self._link_memory[key] = ctx
                    while len(self._link_memory) > self._max_link_targets:
                        del self._link_memory[next(iter(self._link_memory))]
                        counters.bump("assembly._units.link_memory_full")
        unit._remembered = None
        for key in unit._alias_keys:
            if self._by_alias.get(key) is unit:
                del self._by_alias[key]
        unit._alias_keys.clear()

    # -- plumbing --------------------------------------------------------

    def _bind_alias_locked(self, key: UnitKey, unit: Unit) -> None:
        existing = self._by_alias.get(key)
        if existing is not None and existing is not unit:
            counters.bump("assembly._units.alias_rebound")
        if key not in unit._alias_keys:
            evicted = unit._evict_alias()
            if evicted is not None and self._by_alias.get(evicted) is unit:
                del self._by_alias[evicted]
            unit._alias_keys.append(key)
        self._by_alias[key] = unit
        # Binding makes the LIVE unit the key's authority, so any closed
        # memory for it is superseded now — kept, it would resurface the
        # moment this holder closed unremembered, answering with a span two
        # holders stale. One O(1) pop; this is `open()`'s hot path.
        self._link_memory.pop(key, None)

    def _flush(self, pending: list[SpanDraft]) -> None:
        """Hand finished drafts to the sink. NEVER called holding `_lock`.

        Each emit is guarded individually so one bad draft cannot take the rest
        of a subtree with it, and so a sink that raises — the Protocol asks it
        not to, but this registry does not own the implementation — cannot reach
        the host.

        THE EMIT-ONCE FUNNEL. Every path that ships a span goes through here —
        `close_span`, `_close_locked`, `close_all`, `_evict_root_locked`,
        `open_span`'s eviction, and any path added after them — so latching the
        draft HERE covers every one of them by construction instead of asking
        each caller to remember. `unit._open` was doing that
        job by accident and enforced only half of it: a draft removed by a
        force-close is no longer findable, and `close_span` read "not in the
        table" as "nothing to arbitrate, emit" rather than "already gone".

        The latch is taken under the lock and the sink is called outside it
        (I11): the window is one attribute test-and-set, never the emit.
        FIRST WRITER WINS, which keeps the force-closed record — `UNSET` plus
        the force-close marker (`CHILD_SPAN_UNCLOSED`, or `UNIT_TABLE_FULL` for
        a breadth eviction), the honest one — and makes the loss countable.
        """
        for draft in pending:
            with self._lock:
                fresh = draft.claim_emit()
            if not fresh:
                counters.bump("assembly._units.emit_duplicate")
                continue
            with guard("assembly._units.emit", debug=self._debug):
                self._sink.emit(draft, agent_semantic=True)


def parent_is_closed_unit(parent: SpanContext | None) -> bool:
    """Was `parent` latched off a unit that had ALREADY CLOSED? — design §10.3(b).

    The registry predicates above answer this for callers that hold a registry.
    The byte seams and MCP stdio hold a `Client` and nothing else, and
    `_interceptors/` may not import `_adapters/`, where the registries are built.
    So the fact is exported as a function of the ambient CARRIER — the one thing
    the two ends share — and travels to `should_capture` and `resolve_observed`
    as a DECLARED input, which is the shape `degraded` already has and for the
    same reason: `_parentage` is imported BY this module, so neither of those
    two may ask the question for itself without a cycle.

    NO REGISTRY, and that is sound rather than a shortcut. "The span I latched
    has already shipped" is registry-independent: whoever opened the unit, its
    span left the process when the unit closed, and hanging later work off it is
    the same wrong tree either way. The ownership guard inside
    `UnitRegistry._closed_ambient_context` exists to stop one registry CHARGING
    a `CORRELATION_CONFLICT` to another registry's teardown; a seam charges
    nothing to anyone. What matters is the guard that IS here: liveness is
    tested first, so a LIVE unit from any registry — the ordinary state of a
    two-adapter process — is never refused.

    IDENTITY, not descent, for the reason `_poisoned` gives: a host span or a
    nested activation standing on top of the leftover is a real parent, and
    moving live work out of the host's trace to escape a ghost that is no longer
    in front of us is a wrong tree of its own shape.

    ASK THIS WHERE YOU LATCH, on the task that ISSUED the work, and store the
    answer beside the parent. Asked on the response path it would be a different
    carrier at a different instant, and it would refuse a request that was
    legitimately issued INSIDE a run that has since finished — real data lost to
    escape a corpse that was not in front of the request when it left. That is
    the same mistake `latch_ambient`'s docstring names, one field over.
    """
    entry = _ambient_unit.get()
    if entry is None or parent is None:
        return False
    if entry.unit.is_live:
        return False
    if parent != entry.unit.context:
        return False
    counters.bump("assembly._units.ambient_closed_at_issue")
    return True


__all__ = [
    "PinToken",
    "SpanSink",
    "Unit",
    "UnitKey",
    "UnitKind",
    "UnitRegistry",
    "parent_is_closed_unit",
]
