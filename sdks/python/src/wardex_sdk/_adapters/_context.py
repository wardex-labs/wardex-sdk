"""What a framework adapter is handed, and the whole of what it may do.

An adapter's job is to know a framework. Deciding parentage is not part of that
job and is not reachable from here: there is no `ambient=`, no `evidence=`, no
`parent=`, no `parent_unit=`, and no value in this module's vocabulary that IS a
parent. `enter()`, `open_run()` and `rejoin()` latch the live scope
INTERNALLY, on the calling carrier, at the instant of the call — so the question
"how did I know this was the parent" has one answer for every adapter that will
ever exist, and an adapter author cannot get it wrong by trying harder.

That is the product claim expressed as a type rather than as a rule. wardex
builds its causal tree from in-process context propagation; competitors rebuild
it from the `run_id`/`parent_run_id` a framework hands their callbacks, which
marries them to the frameworks that emit those callbacks and breaks wherever one
does not. A framework identifier reaches this surface only as a `UnitKey`, only
through `rejoin()` and `attach()`, and `rejoin()` is deliberately its own method
name so that ONE grep lists every place in an adapter where an identifier can
affect the shape of the tree.

**Placement is mandatory, and that is the expensive lesson.** A run entry the
adapter forgot to wrap does not fail loudly — every span underneath simply
becomes its own trace root at confidence 1.0 with no marker, byte-identical to a
legitimate run. The rule, measured on real langgraph rather than counted on one
graph: with no run entry wrapped, **`traces == captured outbound calls`**, run
spans `== 0`, node spans `== 0`, tool spans `== 0`, and every edge is
`trace_root`/1.0 with no marker — so a graph is reported as as many genuine
traces as it happened to make calls, indistinguishable downstream from that many
real ones. (An earlier version of this paragraph cited "four", which was one
graph's call count and not a property of the framework.) Declaring at each site
whether it may begin a trace turns that silence into `PARENT_UNRESOLVED` at
confidence 0.0. The whole of the difference is one argument that cannot be
defaulted.

**Nothing here raises an Exception of wardex's own making.** The host's call
lives inside `enter()`'s `with` body, so a bug in wardex's own work — deciding
the edge, opening the unit, activating the carrier, closing it — would delete a
tool call, a graph node or an LLM request rather than a span. Every one of those
steps is contained, the body always runs, and the scope it is handed is TOTAL:
`degraded` says wardex failed, and every other verb answers anyway. The host's
own exception passes through as the same object, because wardex may not become
the library in the process that eats a real Ctrl-C.

That is why `describe=` exists. Everything the adapter wants to SAY about a span
goes there, and `enter()` runs it inside the SAME guard as the open, so the
observation is atomic: a framework read that moved between releases costs the
whole span, loudly. Described in the `with` body instead, the same fault ships a
span reading `status=OK` with full io, a real duration and an arbitrary suffix
of its markers missing — which is worse than shipping nothing, because nothing
downstream can tell it from a complete observation. The only two things that
belong in the body are the host's own call and `record_output`.
"""

from __future__ import annotations

import weakref
from collections.abc import Callable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from enum import Enum
from itertools import count
from typing import Any

from .._assembly import (
    AMBIENT,
    EMPTY_AMBIENT,
    NULL_DRAFT,
    Ambient,
    Evidence,
    Limitation,
    LinkReason,
    ParentSource,
    PatchSet,
    SpanDraft,
    SpanIntent,
    Unit,
    UnitKey,
    UnitKind,
    UnitRegistry,
    counters,
    degraded_run,
    guard,
    latch_ambient,
    report_once,
)
from .._enums import StatusCode


class Placement(Enum):
    """Whether a site may legitimately begin a trace. CLOSED, and never inferred.

    Not a property of a call — a property of a PATCH SITE. The failure it
    catches is "the adapter did not wrap the framework's run entry", which is a
    bug in the adapter's shape rather than a condition of one invocation, and a
    computed flag would let every site paper over it independently. An AST rule
    requires the literal.
    """

    ROOT = "root"  # a framework run entry: nothing above it is expected
    NESTED = "nested"  # always inside something this adapter already entered


class Fallback(Enum):
    """What a NESTED site may fall back to when the live scope did not reach it.

    CLOSED, never inferred, and never a default — for the same reason
    `Placement` is not: a heuristic that turns itself on is one nobody can find
    later. Declaring it at the site is the adapter saying "this call is reached
    through a carrier the framework may not have propagated to, and one live run
    is a defensible guess there".

    It is NOT a way to name a parent. The candidate comes from the registry's
    own table, filtered to units this adapter owns, and only when there is
    EXACTLY one — so no identifier, argument or adapter state can influence
    which unit is picked. The edge it produces is `unit_sole` at 0.5 carrying
    `UNIT_INFERRED_SOLE`, which is the vocabulary's word for "interpreted, not
    read".

    Reachable only where the site would otherwise become a trace root, which is
    structural rather than a rule: a live scope or a host span always wins, so
    the fallback can never displace something wardex actually read.
    """

    NONE = "none"  # orphan instead of guessing; the honest default
    SOLE_LIVE_RUN = "sole_live_run"  # exactly one live SESSION this adapter owns


class Observer(Enum):
    """Which observer of one event this is, when a framework offers two.

    A framework that announces a tool call through a callback AND lets wardex
    wrap the executor produces two observations of one logical call. Ranks
    arbitrate: the executor wrapped the real work, so it wins however late it
    arrives — the callback usually fires first, and first-come would hand every
    span to the observer that did not run the code.
    """

    EXECUTOR = 10
    CALLBACK = 0


class InstallOutcome(Enum):
    """Why an adapter is or is not running, kept distinct from each other.

    `DECLINED` and `UNSUPPORTED` are the pair today's adapters conflate into a
    bare `return`: the framework being absent is nothing to look at, while the
    framework being present with a surface wardex does not recognize is
    something a human should see.
    """

    INSTALLED = "installed"
    DECLINED = "declined"  # the framework is not present; nothing to do
    UNSUPPORTED = "unsupported"  # present, but its surface is not one we know


class Scope:
    """A unit that is ACTIVE for the duration of a `with`. Valid only inside it.

    Deliberately has no `.context`, `.parentage`, `.parent`, `.unit`, `.child()`
    or `.activate()`. Every one of those is a value an adapter could carry to
    another carrier and install as a parent chosen by something other than the
    live scope, which is the one thing this surface exists to make unsayable.

    TOTAL. Every verb below answers on a DEGRADED scope — one whose unit wardex
    failed to open — and none of them raises. That is what lets the host's own
    code stay in the `with` body: an adapter never has to ask whether wardex is
    working before it may keep describing what it sees.
    """

    __slots__ = ("_ctx", "_failure", "_unit")

    def __init__(self, unit: Unit | None, ctx: AdapterContext) -> None:
        self._unit = unit
        self._ctx = ctx
        #: A failure the HOST reported without raising, declared by the adapter
        #: through `record_failure`. Read by `_run` on the way out, and only
        #: when nothing was raised — see that method and `record_failure`.
        self._failure: str | None = None

    @property
    def degraded(self) -> bool:
        """wardex's own work failed here. The body still ran; nothing was recorded.

        There is nothing for an adapter to branch on — every verb below is
        already total, and an `if scope.degraded: return` deletes the host's own
        call for a wardex bug. This exists for a conformance suite and a human
        at a debugger, in `PinToken.installed`'s spirit.
        """
        return self._unit is None

    @property
    def draft(self) -> SpanDraft:
        """The span this scope owns, still under construction."""
        return NULL_DRAFT if self._unit is None else self._unit.draft  # type: ignore[return-value]

    @property
    def accepted(self) -> bool:
        """False when an arbitration was lost, so this span will not ship.

        NOT a degradation test, and the two must not be confused: this is False
        after a perfectly healthy block — the unit is closed by then — and True
        after a close that failed. Backwards in both directions. `degraded` is
        the question about wardex; this one is about arbitration and liveness.
        """
        return self._unit is not None and self._unit.is_live

    def note(self, marker: Limitation) -> None:
        if self._unit is not None:
            self._unit.note(marker)

    def link(self, reason: LinkReason, target: UnitKey) -> None:
        """Link to another unit BY SELECTOR — resolved here, never handed in.

        A link is not a parent, but it still carries a span context, and letting
        an adapter supply one would put a fabricable `SpanContext` back in its
        vocabulary. So the selector is resolved against units this registry
        produced; a selector naming nothing live adds no link and is counted.
        Linking to a run in ANOTHER PROCESS — a checkpoint resume — needs an
        identity that outlives the registry, and nothing persists one today.
        """
        if self._unit is not None:
            self._ctx._link(self._unit, reason, target)

    def claim(self, selector: UnitKey, *, observer: Observer) -> bool:
        """Take `selector` for this observer, if a higher rank has not.

        A DEGRADED scope refuses. It has nothing to arbitrate with, and
        answering True would tell a rival observer it had been outranked by a
        unit that does not exist — suppressing the one observation of this event
        that could still have survived.
        """
        if self._unit is None:
            return False
        return self._unit.claim(selector, rank=observer.value)

    def claim_run(self, selector: UnitKey, *, observer: Observer) -> bool:
        """`claim()`, but on the RUN this scope sits under rather than on it.

        Two observers of one event have to claim on the SAME unit or `claim()`
        arbitrates nothing — and they rarely stand in the same place. A
        framework's own callback sees the whole run and holds the SESSION; an
        in-process handler's scope is the CALL it is executing, which for a
        nested tool is not even a direct child of the run. Claiming here would
        put the two claims in two tables and let both observers emit.

        No unit is exposed and none is installed: the walk happens inside the
        registry and only its ANSWER — did this claim take — comes back.
        """
        if self._unit is None:
            return False
        run = self._unit.enclosing(UnitKind.SESSION)
        if run is None:
            return False
        return run.claim(selector, rank=observer.value)

    def outranked(self, selector: UnitKey, *, observer: Observer) -> bool:
        """Has a HIGHER-ranked observer taken this selector since we claimed it?

        Not "did claim() refuse". `claim()` also refuses an EQUAL rank, which is
        how one observer is kept from silently replacing another of the same
        standing — but two concurrent calls to one tool share a name selector at
        one rank, and reading that refusal as "someone else owns this" deletes
        the second call's span.
        """
        if self._unit is None:
            return False
        owner = self._unit.owner_rank(selector)
        return owner is not None and owner > observer.value

    def child_draft(
        self,
        intent: SpanIntent,
        *,
        subject: str | None = None,
        selector: UnitKey | None = None,
    ) -> SpanDraft:
        """A child span of this scope, tied to `selector`'s arbitration if given.

        No `observer` argument, because there is nothing here to declare: the
        rank recorded is whichever one already owns the selector, and
        `close_child` discards the draft if something has outranked it since.
        Claim first with `claim()`, then open.
        """
        if self._unit is None:
            return NULL_DRAFT  # type: ignore[return-value]
        return self._unit.open_span(intent, subject=subject, key=selector)

    def close_child(
        self,
        draft: SpanDraft,
        *,
        status: StatusCode = StatusCode.OK,
        error_type: str | None = None,
    ) -> None:
        # Short-circuited on IDENTITY, not on `self._unit`: a null draft can
        # outlive the scope that handed it out, and closing a span that was
        # never opened is the one thing the registry has no honest answer for.
        if self._unit is None or draft is NULL_DRAFT:
            return
        self._unit.close_span(draft, status=status, error_type=error_type)

    def record_input(self, data: bytes) -> None:
        if self._unit is not None:
            self._unit.record_input(data)

    def record_output(self, data: bytes) -> None:
        if self._unit is not None:
            self._unit.record_output(data)

    def record_failure(self, error_type: str) -> None:
        """The host FAILED and did not raise. Say so, or the span reads `OK`.

        Deriving a span's status from the exception that left the body is only
        half a rule: a framework that converts a failure into a return value
        makes the other half unreachable. LangGraph's default
        `handle_tool_errors` does exactly that — the model calling a tool with
        arguments that do not validate becomes a `ToolMessage(status="error")`
        and never raises — so the single most common tool failure there is
        would otherwise ship `status=OK` with the failure legible only to a
        human reading `output_data`.

        The declaration is WEAKER than an exception and `_run` treats it that
        way: it is consulted only when nothing was raised. An exception is
        stronger evidence, carries a real type, and must keep it — an adapter
        must not be able to relabel a crash.

        Total and silent on a degraded scope, exactly like `record_input`: an
        adapter never has to ask whether wardex is working before it may keep
        describing what it saw.
        """
        if self._unit is not None:
            self._failure = error_type


class RunHandle(Scope):
    """A long-lived unit that is NOT installed on the caller's carrier.

    `enter()` creates and installs atomically and hands back something that
    cannot be re-entered, which is what keeps a re-installable parent off the
    ordinary path. A framework run outlives the call that started it, so it
    needs a handle — and the handle is why `pin()` exists and why `pin()` is the
    one restricted verb here.

    Everything a `Scope` can do, plus those two. It was a separate class with
    four copied members; the copies were the same code answering the same
    questions, and a degraded handle would have needed all four written twice.
    """

    __slots__ = ()

    def pin(self, *, driver: object) -> bool:
        """Make this run the ambient parent on the task that DRIVES a generator.

        The one deferred installer, and the one place an adapter can put a
        parent somewhere other than where it stands. An async generator body has
        no context of its own — its frames run in the context of whatever task
        pumps it — so a framework whose whole session arrives through one
        generator has exactly one carrier that every callback inherits from, and
        no way to enter a scope there.

        Restricted rather than removed: the registry refuses a pin declared for
        a task other than the caller's, so it cannot be aimed at an arbitrary
        carrier. It remains the narrowest way an adapter could rebuild an
        identifier-shaped tree, by opening N runs under one real scope and
        pinning one onto each. That residual is visible — a span per forged
        edge, and `pin_installs` proportional to the node count — where the
        alternative was invisible.
        """
        if self._unit is None:
            return False
        return self._ctx._units.pin_driver(self._unit, owner_task=driver).installed

    def close(self, *, status: StatusCode = StatusCode.OK, error_type: str | None = None) -> None:
        if self._unit is None:
            return
        closed = False
        with self._ctx.guard("run_close"):
            self._ctx._units.close(self._unit, status=status, error_type=error_type)
            closed = True
        if not closed:
            self._ctx._degrade(
                "run_close",
                holder=None,
                consequence="this span and everything under it were lost",
            )


class Attachment:
    """An id-selected unit that can be DESCRIBED and CLOSED, and nothing else.

    No `child_draft`, no `link`, no `claim`, no `pin`, no `__enter__`. An
    identifier selected it, so it may not become a parent: reaching a span's
    attributes and its ending is the whole of what a lookup earns.
    """

    __slots__ = ("_ctx", "_unit")

    def __init__(self, unit: Unit, ctx: AdapterContext) -> None:
        self._unit = unit
        self._ctx = ctx

    @property
    def degraded(self) -> bool:
        """Always False, and present so the three handles answer one question.

        `attach()` hands an `Attachment` out only when a unit was found — a
        lookup that failed returns None, which every call site already handles.
        The flag is here so a conformance suite can ask the same question of
        every handle rather than special-casing this one.
        """
        return False

    @property
    def draft(self) -> SpanDraft:
        return self._unit.draft

    def note(self, marker: Limitation) -> None:
        self._unit.note(marker)

    def close(self, *, status: StatusCode = StatusCode.OK, error_type: str | None = None) -> None:
        closed = False
        with self._ctx.guard("attach_close"):
            self._ctx._units.close(self._unit, status=status, error_type=error_type)
            closed = True
        if not closed:
            self._ctx._degrade(
                "attach_close",
                holder=None,
                consequence="this span and everything under it were lost",
            )


#: The edge a NESTED site takes when nothing at all is installed. Spelled once,
#: here, because it is the entire behavioural difference between the two
#: placements — everywhere else `UnitRegistry.open` already decides correctly.
_ORPHAN = Evidence(ParentSource.UNRESOLVED)

#: What `rejoin()` claims when the LOOKUP ITSELF blew up and a live scope was
#: standing. Same strategy the ordinary fall-through would have taken, half the
#: confidence: a failed lookup may only ever LOWER what wardex claims. Reading
#: it as an ordinary miss would be an UPGRADE — the miss path takes the live
#: scope at 1.0, above the 0.9 a real alias hit earns — so a wardex bug would
#: make the edge look more certain than the id it was told to honour.
_LOOKUP_BROKEN = Evidence(ParentSource.UNIT_ACTIVE, confidence=0.5)

#: What `Fallback.SOLE_LIVE_RUN` claims. `resolve_parentage`'s marker table
#: attaches `UNIT_INFERRED_SOLE` on its own, so there is no site here that could
#: forget it — the source's DEFINITION is interpretation, and I4 makes that the
#: table's job rather than the caller's.
_SOLE = Evidence(ParentSource.UNIT_SOLE)


class AdapterContext:
    """The SDK, as an adapter sees it.

    Holds no client. `capture_span` is not reachable from an adapter, so the
    capture-mode gate has one place to live rather than one per emit site.
    """

    __slots__ = (
        "_anon_ns",
        "_anon_seq",
        "_control_flow",
        "_slots",
        "_tripped",
        "_units",
        "debug",
        "limits",
        "name",
        "patches",
    )

    def __init__(
        self,
        name: str,
        *,
        units: UnitRegistry,
        limits: Mapping[str, int],
        debug: bool = False,
        control_flow: Callable[[], tuple[type[BaseException], ...]] | None = None,
    ) -> None:
        self.name = name
        self.patches = PatchSet(f"adapters.{name}", debug=debug)
        self.limits = limits
        self.debug = debug
        self._units = units
        self._tripped = False
        # The selector a site that names none gets. Built HERE, out of wardex's
        # own counter, for two reasons. It is unique per call, where the shared
        # empty key it replaces had every anonymous unit in the process rebinding
        # one alias slot — `find()` on it answered "whichever was last", which is
        # not an answer. And it costs the CALL SITE nothing to compute: an
        # expression in a `with ctx.enter(...)` header runs before any failure
        # boundary exists, so every one an adapter has to write there is a place
        # a framework read can break the host.
        self._anon_ns = f"adapters.{name}"
        self._anon_seq = count()
        self._slots: MutableMapping[Any, dict[str, Any]] = weakref.WeakKeyDictionary()
        # A READER, not a tuple. `AdapterRegistry.install` builds this context
        # BEFORE it calls `adapter.install()`, and an adapter can only import its
        # framework's error classes in there — so a tuple taken here is `()` for
        # the life of the process. `_registry.context_for` is the only caller with
        # an adapter to bind, and it binds a reader over the CLASS attribute;
        # nothing here can hold a snapshot.
        self._control_flow = control_flow

    @property
    def tripped(self) -> bool:
        """Has wardex's own work failed at least once under this adapter?

        For a conformance suite and a test. Not a gate: nothing in the SDK reads
        it to decide whether to keep instrumenting, because a site-local bug
        turning into total blindness is a worse trade than the bug.
        """
        return self._tripped

    # -- housekeeping ----------------------------------------------------

    def guard(self, where: str) -> guard:
        """The one sanctioned swallow, pre-namespaced and pre-bound to debug."""
        return guard(f"adapters.{self.name}.{where}", debug=self.debug)

    def count(self, where: str) -> None:
        counters.bump(f"adapters.{self.name}.{where}")

    def slot(self, obj: object) -> dict[str, Any]:
        """wardex-owned storage keyed on an object's IDENTITY, not its address.

        Replaces keying a table on `id(obj)`, which CPython reuses: once the
        framework drops a transport, a later object can be handed the dead one's
        bookkeeping. A weak key also releases the entry when the host does,
        rather than holding it until something notices.
        """
        existing = self._slots.get(obj)
        if existing is None:
            existing = {}
            self._slots[obj] = existing
        return existing

    def _link(self, unit: Unit, reason: LinkReason, target: UnitKey) -> None:
        """Resolve a selector to a unit and link to its span. Counted on a miss.

        Not a no-op that a caller could mistake for success: a link nobody can
        see is exactly the silent hole this SDK spends its markers on, and the
        counter is the only record available — a span cannot carry a link to a
        span that does not exist.
        """
        found = self._units.find(target)
        if found is None:
            self.count("link_target_unresolved")
            return
        unit.draft.add_link(found.context, reason)

    def _is_control_flow(self, exc: BaseException) -> bool:
        """Is this the host's control flow rather than the host's failure?

        Read HERE, one statement before `_run` re-raises, and not latched at
        construction: the registry builds the context before the adapter runs,
        so the only value a constructor could copy is the empty default.

        GUARDED, and the `isinstance` is inside the guard WITH the read. This
        runs inside `except BaseException`, so an adapter that declared
        something which is not a class of exceptions would otherwise have
        wardex's own `TypeError` replace the host's exception — the one thing
        this module forbids. A trip leaves `matched` False, i.e. today's
        `ERROR`/type-name: a wardex failure may only ever LOWER what wardex
        claims, and must never turn a failure into a success.
        """
        reader = self._control_flow
        if reader is None:
            return False
        matched = False
        with self.guard("control_flow_read"):
            matched = isinstance(exc, reader())
        return matched

    def confirm_active(self, site: str) -> None:
        """Report that a declared patch site actually fired.

        Per SITE, not per install. A framework can move ONE of its entry points
        and leave the rest working, which produces a tree that is wrong only in
        the shape the moved entry governed — measured on langgraph, a run entry
        left unpatched shatters a graph into one trace per captured outbound
        call (the baseline rule this module's header states) while an
        install-level self-check reports success.
        """
        self.count(f"active.{site}")

    # -- the causal surface ----------------------------------------------

    def _evidence(self, placement: Placement, ambient: Ambient) -> Evidence:
        """`AMBIENT`, or the orphan edge when a NESTED site has nothing above it.

        The whole parentage table, once the rows `UnitRegistry.open` already
        gets right are subtracted. Installed wardex unit, host span, remote
        header and dead-pin fork are all its decisions; the single divergence is
        that where a ROOT site legitimately becomes a trace root, a NESTED site
        has lost something it was promised, and must say so.

        Which is why the "would this be a trace root" question is asked of the
        REGISTRY on the very ambient that is about to be handed to it, rather
        than re-derived here. Re-deriving it is what went wrong: a predicate
        about the TASK stood in for one about the SPAN, and a NESTED site under
        a host's own live span was orphaned to escape a dead pin that was not
        in front of it. See `UnitRegistry.becomes_trace_root`.
        """
        if placement is Placement.ROOT:
            return AMBIENT
        if self._units.current() is not None:
            return AMBIENT
        return _ORPHAN if self._units.becomes_trace_root(ambient) else AMBIENT

    def _open(
        self,
        kind: UnitKind,
        *,
        intent: SpanIntent,
        placement: Placement,
        subject: str | None,
        selector: UnitKey | None,
        aliases: Sequence[UnitKey],
        start_ns: int | None,
        parent: Unit | None = None,
        evidence: Evidence | None = None,
        fallback: Fallback = Fallback.NONE,
    ) -> Unit:
        holder = parent if parent is not None else self._units.current()
        # Latched ONCE and used for both the declaration and the open. Two reads
        # would be two different instants on the same carrier, and the whole
        # point of the declaration is that it describes the ambient `open()`
        # actually receives.
        ambient = latch_ambient()
        if evidence is None:
            evidence = self._evidence(placement, ambient)
        conflicted = False
        if holder is None and evidence is _ORPHAN and fallback is Fallback.SOLE_LIVE_RUN:
            # `evidence is _ORPHAN` is the whole gate, and it is structural: only
            # a NESTED site with nothing live above it and no host span reaches
            # this line, so the fallback can never displace something wardex
            # actually read. `owner` keeps the question about THIS adapter's
            # runs — "exactly one session is live" is otherwise a question about
            # the process, and answering it across a framework boundary is the
            # guess `sole_live` refuses to make.
            sole = self._units.sole_live(UnitKind.SESSION, owner=self.name)
            if sole is not None:
                # Asked BEFORE the fallback takes a parent, because taking one
                # is what stops `open()` from seeing the poisoned ambient for
                # itself. Without this the two reasons a guess happened are
                # byte-identical on the wire — "nothing was pinned" and "what
                # was pinned had died" — and the second is a tool call sitting
                # in a run it has nothing to do with.
                conflicted = self._units.stale_pin_in_scope()
                holder, evidence = sole, _SOLE
        unit = self._units.open(
            kind,
            selector if selector is not None else UnitKey(self._anon_ns, str(next(self._anon_seq))),
            ambient=ambient,
            evidence=evidence,
            intent=intent,
            subject=subject,
            parent_unit=holder,
            aliases=aliases,
            start_ns=start_ns,
            owner=self.name,
        )
        if conflicted:
            unit.note(Limitation.CORRELATION_CONFLICT)
        return unit

    # -- containment -----------------------------------------------------

    def _report(self, where: str, consequence: str) -> None:
        """The PRIMARY record. One line, on by default, touching no registry.

        A degradation recorded only in `counters` is, in a production process,
        byte-identical to wardex never having been installed — nothing reads a
        snapshot, `counters` is not exported, and `debug` is off. So this comes
        first, and it is deliberately the half that cannot be taken out by the
        failure it is reporting: it reads no state and calls into no registry.

        The line names the CONSEQUENCE rather than the site, because the person
        reading it is an operator wondering why their dashboard is empty. The
        site name is in the guard's counter and in the `where` on the line.

        NOT counted here: the guard that caught the failure already bumped this
        exact label, and counting again would double every tally.
        """
        self._tripped = True
        report_once(
            f"[wardex] {self.name} adapter: internal error at {where}; {consequence} "
            f"(re-run with debug=True for the traceback)",
            key=f"{self.name}.{where}",
        )

    def _degrade(self, where: str, *, holder: Unit | None, consequence: str) -> None:
        """Report, then BEST EFFORT mark the unit that absorbed the loss.

        The marker is explicitly the part that may be lost: with no holder given
        this reaches for one through the same registry that just failed, so a
        fault wide enough to reach `current()` takes the wire record with it and
        leaves only the line. That is stated rather than defended.

        Used where the loss lands on a span OTHER than the one being opened. A
        site whose own span exists and is the thing that lost something marks it
        directly and calls `_report`, so the marker goes where the reader will
        look for it instead of onto whatever happened to be enclosing.
        """
        self._report(where, consequence)
        target = holder
        if target is None:
            with self.guard("degraded_holder"):
                target = self._units.current()
        if target is None:
            return
        with self.guard("degraded_mark"):
            # Idempotent, and deliberately not accompanied by an `extra` naming
            # the site: `set_extra` appends without a cap, so a site that trips a
            # thousand times would put a thousand entries on one span, while
            # `add_limitation` puts one marker there however often it is called.
            target.note(Limitation.INSTRUMENTATION_DEGRADED)

    def _abandon(self, unit: Unit | None, where: str, *, placement: Placement) -> None:
        """A unit whose open or description failed may not ship looking healthy.

        Closed NOW — `UNSET`, marked, at its TRUE duration rather than at
        teardown. If the description died before the intent's required block,
        `finish()` refuses the draft and the emit funnel drops it, so the
        outcome is "no span" and never a clean-looking one.
        """
        closed = False
        if unit is not None:
            with self.guard("enter_abandon"):
                unit.note(Limitation.INSTRUMENTATION_DEGRADED)
                self._units.close(unit, status=StatusCode.UNSET)
                closed = True
        # PLACEMENT FIRST, and this is a correction. The flag below says the
        # unit was CLOSED, not that it was described — it is True on the
        # ordinary way of getting here (a `describe` that raised after the unit
        # opened), so testing it first made the ROOT branch unreachable and
        # every run entry reported as "one span". A ROOT site is the one place
        # the loss is not one span: the description died before the intent's
        # required block, so `finish()` refuses the draft and NOTHING ships for
        # the whole run.
        if placement is Placement.ROOT:
            # Measured rather than reasoned: 0 spans ship, and the body then
            # runs inside `degraded_run()`. So traffic inside is still captured
            # — that mechanism exists precisely so one failure here does not
            # become total silence — but it arrives ORPHANED and marked instead
            # of attached to a run. (The older wording claimed it would not be
            # captured at all, which was true before `degraded_run` and is not
            # now. An unreachable branch is also an unaudited one.)
            consequence = (
                "this run will produce NO span of its own, and every request and "
                "tool call inside it will arrive orphaned and marked rather than "
                "attached to the run"
            )
        elif closed:
            consequence = "one span is incomplete or missing"
        else:
            consequence = "one span is missing from this run"
        self._degrade(where, holder=None, consequence=consequence)

    def _run(self, unit: Unit, scope: Scope, *, intent: SpanIntent) -> Iterator[Scope]:
        """Activate, hand the scope to the caller's body, close. Shared, guarded.

        `enter()` and `rejoin()` had this block twice, and a guard added to one
        copy is a guard the other does not have.
        """
        status, error_type = StatusCode.OK, None
        activation = None
        # A BARE FLAG, and it is the only valid test of a guarded step. `if
        # activation is None:` is DEAD CODE here — `unit.activate()` binds the
        # name before `__enter__` can raise — and the span it lets through reads
        # byte-identical to a healthy one. Two authors wrote that bug an hour
        # apart, which is why `test_import_graph.py` now checks the shape.
        active = False
        with self.guard(f"enter_activate.{intent.value}"):
            activation = unit.activate()
            activation.__enter__()
            active = True
        if not active:
            # Half-entered at worst, so it must not be exited. The unit itself is
            # live and its own edge is correct; what is lost is that work INSIDE
            # this span will find the enclosing unit instead.
            activation = None
            self._degrade(
                f"enter_activate.{intent.value}",
                holder=unit,
                consequence="work inside this span will be attached one level too high",
            )
        try:
            yield scope
        except BaseException as exc:
            # A framework that implements pausing, handing off or draining by
            # RAISING is not failing, and a span that says otherwise reports
            # every human-in-the-loop pause as a crashed run. The adapter names
            # those classes once, as a classvar; this is the one place that
            # reads it, so no causal path can be the one nobody remembered.
            if self._is_control_flow(exc):
                status, error_type = StatusCode.UNSET, None
            else:
                status, error_type = StatusCode.ERROR, type(exc).__name__
            raise
        finally:
            if status is StatusCode.OK and scope._failure is not None:
                # ONLY when nothing was raised. An exception is stronger
                # evidence than a declaration and keeps its own type, so an
                # adapter cannot relabel a crash — and control flow, which has
                # already resolved to UNSET above, is not a failure to overwrite
                # either. What this reaches is the one case neither can see: the
                # host returned normally and told the adapter it had failed.
                status, error_type = StatusCode.ERROR, scope._failure
            if activation is not None:
                # BEFORE the close, on every path including the ones where the
                # close then fails. Deferring it until the close succeeds leaves
                # a dead unit ambient, so the next sibling call in this task is
                # parented to a corpse.
                with self.guard(f"enter_activate_exit.{intent.value}"):
                    activation.__exit__(None, None, None)
            closed = False
            with self.guard(f"enter_close.{intent.value}"):
                self._units.close(unit, status=status, error_type=error_type)
                closed = True
            if not closed:
                # NO SALVAGE. Hand-detaching the unit to rescue its own span
                # ships one span of a subtree and makes the rest unreachable;
                # leaving it where it is costs the same spans NOW and lets a
                # later `close_all()` recover the whole subtree. Measured on a
                # six-child subtree: salvage 1 of 7, no salvage 0 now and 7 then.
                self._degrade(
                    f"enter_close.{intent.value}",
                    holder=None,
                    consequence="this span and everything under it were lost",
                )

    # -- the causal methods ----------------------------------------------

    @contextmanager
    def enter(
        self,
        kind: UnitKind,
        *,
        intent: SpanIntent,
        placement: Placement,
        subject: str | None = None,
        selector: UnitKey | None = None,
        aliases: Sequence[UnitKey] = (),
        start_ns: int | None = None,
        fallback: Fallback = Fallback.NONE,
        describe: Callable[[Scope], None] | None = None,
    ) -> Iterator[Scope]:
        """Open a unit, make it the ambient parent, and close it on the way out.

        NEVER raises an Exception of wardex's own making, and the body ALWAYS
        runs — with a real scope or with a degraded one, which answers every
        verb the real one does. The host's own exception passes through
        untouched, as the same object.

        Creates and installs atomically on the CURRENT carrier, so there is no
        moment at which an adapter holds a unit it could install somewhere else.

        **Put adapter code in `describe=`, not in the body.** `describe` is
        everything the adapter wants to SAY about this span, and it runs inside
        the SAME guard as the open, before the yield — which is what makes the
        observation atomic. A framework read that moved between releases then
        costs the whole span, loudly. Described in the body instead, under its
        own guard, the same fault ships a span reading `status=OK` with full
        io, a real duration, and an arbitrary suffix of its markers missing.

        The only two things that belong in the body are the host's own call and
        `record_output`, whose absence is self-describing on the wire.
        """
        unit = None
        scope = None
        ok = False
        with self.guard(f"enter.{intent.value}"):
            unit = self._open(
                kind,
                intent=intent,
                placement=placement,
                subject=subject,
                selector=selector,
                aliases=aliases,
                start_ns=start_ns,
                fallback=fallback,
            )
            scope = Scope(unit, self)
            if describe is not None:
                describe(scope)
            ok = True
        # `if scope is None` would be DEAD CODE: `scope` is already bound when
        # `describe` is what raised. Same trap as `_run`'s activation.
        if not ok:
            self._abandon(unit, f"enter.{intent.value}", placement=placement)
            # The host's block runs with nothing ambient, so under
            # `capture_mode=AGENT` every request and every tool call inside it
            # would be dropped at the byte seam — one wardex bug here turning
            # into total silence underneath. The flag says the absence of a
            # parent is wardex's doing, and the gate reads it.
            with degraded_run():
                yield Scope(None, self)
            return
        yield from self._run(unit, scope, intent=intent)

    def open_run(
        self,
        kind: UnitKind,
        *,
        intent: SpanIntent,
        placement: Placement,
        subject: str | None = None,
        selector: UnitKey | None = None,
        start_ns: int | None = None,
        fallback: Fallback = Fallback.NONE,
        describe: Callable[[RunHandle], None] | None = None,
    ) -> RunHandle:
        """A unit that outlives this call. NOT installed; see `RunHandle.pin`.

        NEVER None and NEVER raises; a handle whose open failed answers
        `degraded` and no-ops every verb.
        """
        unit = None
        handle = None
        ok = False
        with self.guard(f"open_run.{intent.value}"):
            unit = self._open(
                kind,
                intent=intent,
                placement=placement,
                subject=subject,
                selector=selector,
                aliases=(),
                start_ns=start_ns,
                fallback=fallback,
            )
            handle = RunHandle(unit, self)
            if describe is not None:
                describe(handle)
            ok = True
        if not ok:
            self._abandon(unit, f"open_run.{intent.value}", placement=placement)
            return RunHandle(None, self)
        return handle

    def attach(self, selector: UnitKey) -> Attachment | None:
        """Find a unit by identifier, to describe or close it. Never to parent.

        A lookup that BLEW UP is reported and returns None, which every call
        site already handles. On the wire a failure and a miss stay conflated —
        stated, not fixed; only the stderr line separates them.
        """
        unit = None
        found = False
        with self.guard("attach_find"):
            unit = self._units.find(selector)
            found = True
        if not found:
            self._degrade(
                "attach_find",
                holder=None,
                consequence=("a span this adapter meant to finish will be left for the teardown"),
            )
        return Attachment(unit, self) if unit is not None else None

    @contextmanager
    def rejoin(
        self,
        selector: UnitKey,
        kind: UnitKind,
        *,
        intent: SpanIntent,
        placement: Placement,
        subject: str | None = None,
        describe: Callable[[Scope], None] | None = None,
    ) -> Iterator[Scope]:
        """`enter()`, but under the unit an identifier selects — if it resolves.

        The ONE method where an identifier influences the shape of the tree, and
        a separate name so that one grep is the complete list. A hit is clamped
        to `UNIT_ALIAS` at 0.9: no argument raises it, because the identifier was
        the framework's word and not a scope wardex read. A miss falls through to
        the ordinary table for the declared placement rather than quietly picking
        a plausible root.

        A lookup that BROKE is neither of those. It is its own step with its own
        guard, and what it may do to the edge is bounded in one direction: see
        `_LOOKUP_BROKEN`.
        """
        target = None
        found = False
        with self.guard("rejoin_find"):
            target = self._units.find(selector)
            found = True
        if not found:
            # REPORTED, not `_degrade`d. The span that lost something is the one
            # about to be opened, and it does not exist yet — so `_degrade` would
            # have marked whatever happened to be enclosing, which lost nothing
            # and would send a reader looking in the wrong place. The marker goes
            # on the new unit below, beside the confidence the same fault lowered.
            self._report(
                "rejoin_find",
                "this span's parent edge is a guess, not the id it was told to honour",
            )
        unit = None
        scope = None
        ok = False
        with self.guard(f"rejoin.{intent.value}"):
            unit = self._open(
                kind,
                intent=intent,
                placement=placement,
                subject=subject,
                selector=selector,
                aliases=(),
                start_ns=None,
                parent=target,
                evidence=self._rejoin_evidence(selector, target, found=found),
            )
            if not found:
                unit.note(Limitation.INSTRUMENTATION_DEGRADED)
            scope = Scope(unit, self)
            if describe is not None:
                describe(scope)
            ok = True
        if not ok:
            self._abandon(unit, f"rejoin.{intent.value}", placement=placement)
            # The host's block runs with nothing ambient, so under
            # `capture_mode=AGENT` every request and every tool call inside it
            # would be dropped at the byte seam — one wardex bug here turning
            # into total silence underneath. The flag says the absence of a
            # parent is wardex's doing, and the gate reads it.
            with degraded_run():
                yield Scope(None, self)
            return
        yield from self._run(unit, scope, intent=intent)

    def _rejoin_evidence(
        self, selector: UnitKey, target: Unit | None, *, found: bool
    ) -> Evidence | None:
        """What a hit, an honest miss and a broken lookup are each worth.

        Evaluated inside `rejoin`'s open guard, because its last branch reads
        the registry that may be the thing that is broken.
        """
        if target is not None:
            return Evidence(ParentSource.UNIT_ALIAS, request_id=selector.value)
        if found:
            return None  # an honest miss: the ordinary table for this placement
        # The lookup broke. Lowering only makes sense where the ordinary path
        # would have claimed something: with nothing live, `UNIT_ACTIVE` would
        # name a unit that is not there, so the honest answer is the same
        # fall-through a miss takes.
        return _LOOKUP_BROKEN if self._units.current() is not None else None

    def close_all(self, *, marker: Limitation) -> None:
        """Close everything this adapter still holds open, and nothing else.

        Guarded because its caller is `uninstall()`, which runs on the host's
        `atexit`: a raise here lands in the host's own shutdown path.
        """
        done = False
        with self.guard("close_all"):
            self._units.close_all(reason=marker, owner=self.name)
            done = True
        if not done:
            self._degrade(
                "close_all",
                holder=None,
                consequence="some spans this adapter still held open were never sent",
            )


__all__ = [
    "AdapterContext",
    "Attachment",
    "Fallback",
    "InstallOutcome",
    "Observer",
    "Placement",
    "RunHandle",
    "Scope",
]

# `EMPTY_AMBIENT` is imported for the type it documents rather than used; the
# ambient is always latched here, never supplied.
_ = EMPTY_AMBIENT
