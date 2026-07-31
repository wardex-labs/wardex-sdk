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
legitimate run. Measured on real langgraph: one graph run reported as four
separate traces, indistinguishable downstream from four genuine ones. Declaring
at each site whether it may begin a trace turns that silence into
`PARENT_UNRESOLVED` at confidence 0.0. The whole of the difference is one
argument that cannot be defaulted.
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from enum import Enum
from typing import Any

from .._enums import StatusCode
from ..assembly import (
    AMBIENT,
    EMPTY_AMBIENT,
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
    guard,
    latch_ambient,
)


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
    """

    __slots__ = ("_ctx", "_unit")

    def __init__(self, unit: Unit, ctx: AdapterContext) -> None:
        self._unit = unit
        self._ctx = ctx

    @property
    def draft(self) -> SpanDraft:
        """The span this scope owns, still under construction."""
        return self._unit.draft

    @property
    def accepted(self) -> bool:
        """False when an arbitration was lost, so this span will not ship."""
        return self._unit.is_live

    def note(self, marker: Limitation) -> None:
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
        self._ctx._link(self._unit, reason, target)

    def claim(self, selector: UnitKey, *, observer: Observer) -> bool:
        return self._unit.claim(selector, rank=observer.value)

    def outranked(self, selector: UnitKey, *, observer: Observer) -> bool:
        """Has a HIGHER-ranked observer taken this selector since we claimed it?

        Not "did claim() refuse". `claim()` also refuses an EQUAL rank, which is
        how one observer is kept from silently replacing another of the same
        standing — but two concurrent calls to one tool share a name selector at
        one rank, and reading that refusal as "someone else owns this" deletes
        the second call's span.
        """
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
        return self._unit.open_span(intent, subject=subject, key=selector)

    def close_child(
        self,
        draft: SpanDraft,
        *,
        status: StatusCode = StatusCode.OK,
        error_type: str | None = None,
    ) -> None:
        self._unit.close_span(draft, status=status, error_type=error_type)

    def record_input(self, data: bytes) -> None:
        self._unit.record_input(data)

    def record_output(self, data: bytes) -> None:
        self._unit.record_output(data)


class RunHandle:
    """A long-lived unit that is NOT installed on the caller's carrier.

    `enter()` creates and installs atomically and hands back something that
    cannot be re-entered, which is what keeps a re-installable parent off the
    ordinary path. A framework run outlives the call that started it, so it
    needs a handle — and the handle is why `pin()` exists and why `pin()` is the
    one restricted verb here.
    """

    __slots__ = ("_ctx", "_unit")

    def __init__(self, unit: Unit, ctx: AdapterContext) -> None:
        self._unit = unit
        self._ctx = ctx

    @property
    def draft(self) -> SpanDraft:
        return self._unit.draft

    @property
    def accepted(self) -> bool:
        return self._unit.is_live

    def note(self, marker: Limitation) -> None:
        self._unit.note(marker)

    def link(self, reason: LinkReason, target: UnitKey) -> None:
        self._ctx._link(self._unit, reason, target)

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
        return self._ctx._units.pin_driver(self._unit, owner_task=driver).installed

    def close(self, *, status: StatusCode = StatusCode.OK, error_type: str | None = None) -> None:
        self._ctx._units.close(self._unit, status=status, error_type=error_type)


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
    def draft(self) -> SpanDraft:
        return self._unit.draft

    def note(self, marker: Limitation) -> None:
        self._unit.note(marker)

    def close(self, *, status: StatusCode = StatusCode.OK, error_type: str | None = None) -> None:
        self._ctx._units.close(self._unit, status=status, error_type=error_type)


#: The edge a NESTED site takes when nothing at all is installed. Spelled once,
#: here, because it is the entire behavioural difference between the two
#: placements — everywhere else `UnitRegistry.open` already decides correctly.
_ORPHAN = Evidence(ParentSource.UNRESOLVED)


class AdapterContext:
    """The SDK, as an adapter sees it.

    Holds no client. `capture_span` is not reachable from an adapter, so the
    capture-mode gate has one place to live rather than one per emit site.
    """

    __slots__ = ("_slots", "_units", "debug", "limits", "name", "patches")

    def __init__(
        self,
        name: str,
        *,
        units: UnitRegistry,
        limits: Mapping[str, int],
        debug: bool = False,
    ) -> None:
        self.name = name
        self.patches = PatchSet(f"adapters.{name}", debug=debug)
        self.limits = limits
        self.debug = debug
        self._units = units
        self._slots: MutableMapping[Any, dict[str, Any]] = weakref.WeakKeyDictionary()

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

    def confirm_active(self, site: str) -> None:
        """Report that a declared patch site actually fired.

        Per SITE, not per install. A framework can move ONE of its entry points
        and leave the rest working, which produces a tree that is wrong only in
        the shape the moved entry governed — measured on langgraph, a run entry
        left unpatched shattered a graph into five traces while an install-level
        self-check reported success.
        """
        self.count(f"active.{site}")

    # -- the causal surface ----------------------------------------------

    def _evidence(self, placement: Placement) -> Evidence:
        """`AMBIENT`, or the orphan edge when a NESTED site has nothing above it.

        The whole parentage table, once the rows `UnitRegistry.open` already
        gets right are subtracted. Installed wardex unit, host span, remote
        header and dead-pin fork are all its decisions; the single divergence is
        that where a ROOT site legitimately becomes a trace root, a NESTED site
        has lost something it was promised, and must say so.
        """
        if placement is Placement.ROOT:
            return AMBIENT
        if self._units.current() is not None:
            return AMBIENT
        if self._units.stale_pin_in_scope():
            return _ORPHAN
        return AMBIENT if latch_ambient().span_context is not None else _ORPHAN

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
    ) -> Unit:
        holder = parent if parent is not None else self._units.current()
        return self._units.open(
            kind,
            selector if selector is not None else UnitKey(f"adapters.{self.name}", ""),
            ambient=latch_ambient(),
            evidence=evidence if evidence is not None else self._evidence(placement),
            intent=intent,
            subject=subject,
            parent_unit=holder,
            aliases=aliases,
            start_ns=start_ns,
            owner=self.name,
        )

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
    ) -> Iterator[Scope]:
        """Open a unit, make it the ambient parent, and close it on the way out.

        Creates and installs atomically on the CURRENT carrier, so there is no
        moment at which an adapter holds a unit it could install somewhere else.
        """
        unit = self._open(
            kind,
            intent=intent,
            placement=placement,
            subject=subject,
            selector=selector,
            aliases=aliases,
            start_ns=start_ns,
        )
        scope = Scope(unit, self)
        status, error_type = StatusCode.OK, None
        try:
            with unit.activate():
                yield scope
        except BaseException as exc:
            status, error_type = StatusCode.ERROR, type(exc).__name__
            raise
        finally:
            self._units.close(unit, status=status, error_type=error_type)

    def open_run(
        self,
        kind: UnitKind,
        *,
        intent: SpanIntent,
        placement: Placement,
        subject: str | None = None,
        selector: UnitKey | None = None,
        start_ns: int | None = None,
    ) -> RunHandle:
        """A unit that outlives this call. NOT installed; see `RunHandle.pin`."""
        return RunHandle(
            self._open(
                kind,
                intent=intent,
                placement=placement,
                subject=subject,
                selector=selector,
                aliases=(),
                start_ns=start_ns,
            ),
            self,
        )

    def attach(self, selector: UnitKey) -> Attachment | None:
        """Find a unit by identifier, to describe or close it. Never to parent."""
        unit = self._units.find(selector)
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
    ) -> Iterator[Scope]:
        """`enter()`, but under the unit an identifier selects — if it resolves.

        The ONE method where an identifier influences the shape of the tree, and
        a separate name so that one grep is the complete list. A hit is clamped
        to `UNIT_ALIAS` at 0.9: no argument raises it, because the identifier was
        the framework's word and not a scope wardex read. A miss falls through to
        the ordinary table for the declared placement rather than quietly picking
        a plausible root.
        """
        target = self._units.find(selector)
        unit = self._open(
            kind,
            intent=intent,
            placement=placement,
            subject=subject,
            selector=selector,
            aliases=(),
            start_ns=None,
            parent=target,
            evidence=(
                Evidence(ParentSource.UNIT_ALIAS, request_id=selector.value)
                if target is not None
                else None
            ),
        )
        scope = Scope(unit, self)
        status, error_type = StatusCode.OK, None
        try:
            with unit.activate():
                yield scope
        except BaseException as exc:
            status, error_type = StatusCode.ERROR, type(exc).__name__
            raise
        finally:
            self._units.close(unit, status=status, error_type=error_type)

    def close_all(self, *, marker: Limitation) -> None:
        """Close everything this adapter still holds open, and nothing else."""
        self._units.close_all(reason=marker, owner=self.name)


__all__ = [
    "AdapterContext",
    "Attachment",
    "InstallOutcome",
    "Observer",
    "Placement",
    "RunHandle",
    "Scope",
]

# `EMPTY_AMBIENT` is imported for the type it documents rather than used; the
# ambient is always latched here, never supplied.
_ = EMPTY_AMBIENT
