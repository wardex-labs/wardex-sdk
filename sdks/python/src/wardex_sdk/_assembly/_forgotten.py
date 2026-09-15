"""The ids a unit's alias bound dropped — design I10's one eviction that could lie.

`UnitRegistry` bounds each unit's lookup aliases at `max_entries_per_unit` and
drops the oldest. An alias owns no span, so the eviction itself has nothing to
mark and only counts (`alias_table_full`). What it costs shows up LATER, on the
next edge that looks the dropped id up, and unmarked that cost reads backwards:
`find()` misses, `_edge` walks down its ladder, and the rung below an alias hit
(`UNIT_ALIAS`, 0.9) is the ambient scope at `CONTEXTVAR` 1.0. A sub-agent that
hung off its own unit would hang off the enclosing session — confidence UP,
edge WORSE, nothing on the wire.

This record is what tells that miss apart from an honest one. It is kept out of
`_units.py` because it is bookkeeping with one reader and one writer, and the
registry module is already the largest in the SDK.

Its own bounds follow from what it shadows. Per unit, FIFO, as long as the alias
table (`max_entries_per_unit`): a record the bound could grow without limit would
be the unbounded table I10 forbids, brought back by the fix for another. Its
overflow counts `alias_forgotten_table_full` and marks nothing — there is no
span yet to say it on — and costs only that an id forgotten that long ago reads
as an honest miss again. And the record dies with its unit: every id a closed
unit held misses for that ordinary reason, and marking those misses would blame
the bound for a run that simply finished.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._diag import counters
from ._integrity import Limitation
from ._parentage import (
    EMPTY_AMBIENT,
    Ambient,
    Evidence,
    Parentage,
    ParentSource,
    cap_at_alias_tier,
    resolve_parentage,
)

if TYPE_CHECKING:
    from ._units import Unit, UnitKey


def forgotten_edge(sole: Unit | None, amb: Ambient, hint: str | None, owner: Unit) -> Parentage:
    """`UnitRegistry._edge`'s ladder for a `find()` miss on a FORGOTTEN id.

    The ordinary miss takes the ambient scope first, and that is exactly the
    rung that may not answer here unmarked. So the rungs that mark themselves go
    first — the sole live session (`UNIT_SOLE`, 0.5), whose marker the parentage
    table attaches — then the ambient scope, capped at the alias tier so the
    replacement cannot outrank the answer the id used to give, then no parent
    at all (`UNRESOLVED`, likewise self-marking). Whichever answers also carries
    `ALIAS_FORGOTTEN`, which is the one fact none of those rungs can say: the
    edge is a fallback because wardex's own bound threw the id away.

    An ambient span in a different trace from `owner` (the unit the id named)
    is a disagreement the bound id reported itself, at 0.8 with
    `CORRELATION_CONFLICT`, so that rung keeps both: capped at the alias tier
    alone it would ship the same parent MORE certain than the id gave, with the
    conflict gone.

    `sole` is asked by the caller, which owns the table it is asked of.
    """
    if sole is not None:
        edge = sole.child(Evidence(ParentSource.UNIT_SOLE, request_id=hint))
    elif amb.span_context is not None:
        cross = amb.span_context.trace_id != owner.context.trace_id
        # 0.8 is `UnitRegistry._edge`'s own number for the bound id's cross-trace branch.
        tier = 0.8 if cross else None
        evidence = Evidence(ParentSource.CONTEXTVAR, request_id=hint, confidence=tier)
        edge = resolve_parentage(amb, cap_at_alias_tier(evidence))
        if cross:
            edge = edge.with_limitation(Limitation.CORRELATION_CONFLICT)
    else:
        edge = resolve_parentage(EMPTY_AMBIENT, Evidence(ParentSource.UNRESOLVED, request_id=hint))
    return edge.with_limitation(Limitation.ALIAS_FORGOTTEN)


def _lost_nothing(owner: Unit, amb: Ambient | None, holder: Unit | None) -> bool:
    """Would the id, still bound to `owner`, have given this same edge anyway?

    The measure is the edge the id gave while it resolved, so the record can
    never make an edge worse than keeping the id would have, and it leaves
    unmarked the edges that stay byte-identical. It is a structural test, not a
    full comparison, so it does not catch every edge the loss left alone: with
    no ambient span and `owner` itself the only live session, the sole-session
    rung reaches the same parent the id gave, at 0.5 and marked where the id
    gave 0.9 unmarked. The parent is unchanged there; the source and the
    number are the lower rung's, and the marker says why that rung was used.
    Two ways the answer is yes:

    - `rejoin`: the live unit `holder` IS `owner` or sits below it. The placement
      edge lands inside the unit the id named, read off the carrier; the id
      could only have pulled it up to `owner`, never somewhere more specific.
    - `UnitRegistry._edge`: the ambient span is in `owner`'s trace and `owner` is
      not below it. That is the branch where a bound id only corroborates the
      context (`CONTEXTVAR`, 1.0) — the ambient span is `owner`'s own, or below
      it, or beside it — so the ordinary ambient rung is the same edge. Only an
      ambient span ABOVE `owner` (the session root under a pin) or in another
      trace is a scope the id used to override, and those stay marked.
    """
    node = holder
    while node is not None:
        if node is owner:
            return True
        node = node.parent
    ctx = amb.span_context if amb is not None else None
    if ctx is None or ctx.trace_id != owner.context.trace_id:
        return False
    node = owner.parent
    while node is not None:
        if node.context.span_id == ctx.span_id:
            return False
        node = node.parent
    return True


class ForgottenAliases:
    """Dropped alias keys, indexed by key for lookup and by unit for bounding.

    Every method runs under the owning registry's lock; this class takes none
    of its own, so it adds no lock to the ordering the registry already keeps.
    """

    __slots__ = ("_bound", "_by_key", "_by_unit")

    def __init__(self, bound: int) -> None:
        self._bound = bound
        self._by_key: dict[UnitKey, Unit] = {}
        self._by_unit: dict[Unit, list[UnitKey]] = {}

    def forget(self, key: UnitKey, unit: Unit) -> None:
        """Record that the bound dropped `key` while `unit` still owned it."""
        ring = self._by_unit.setdefault(unit, [])
        while ring and len(ring) >= self._bound:
            oldest = ring.pop(0)
            if self._by_key.get(oldest) is unit:
                del self._by_key[oldest]
            counters.bump("assembly._units.alias_forgotten_table_full")
        ring.append(key)
        self._by_key[key] = unit

    def recall(
        self, key: UnitKey, *, amb: Ambient | None = None, holder: Unit | None = None
    ) -> Unit | None:
        """Did the bound drop `key` AT A COST to the edge being built? Counted when yes.

        Yes is the unit that owned `key` when it was dropped (still live: the
        record dies with it), so the caller can compare the edge against it.

        The callers ask at the moment they are about to build an edge from a
        `find()` miss, so `alias_forgotten_consumed` says how many edges the
        bound degraded, where `alias_table_full` says how many ids it dropped.
        The record is NOT removed by asking: a second lookup of the same id is
        the same loss, and must say so again.

        A recorded id whose loss changed nothing answers no, unmarked and
        uncounted — see `_lost_nothing` for when that is.
        """
        owner = self._by_key.get(key)
        if owner is None or _lost_nothing(owner, amb, holder):
            return None
        counters.bump("assembly._units.alias_forgotten_consumed")
        return owner

    def rebound(self, key: UnitKey, unit: Unit, *, carries_loss: bool = False) -> None:
        """`unit` binds `key`, so it resolves again: is the record now stale?

        Yes, unless the caller says the binding is built ON the loss. Kept, the
        record would mark a good edge the next time the key missed for an
        unrelated reason: a unit that rebinds the id takes it off the unit that
        lost it, which an unbounded table would do too, so once that unit closes
        the miss is an honest one. One O(1) pop on the common path, which is
        `UnitRegistry.open()`'s.

        `carries_loss` is the one exception, and it is stated by the caller
        rather than inferred here: `AdapterContext._open` passes it for the unit
        `rejoin` opens under the very id `recall` just said was dropped. The unit
        the id named is still live, so once the rejoined unit closes, the next
        lookup of the id is the same loss again, and erasing the record would
        let it ship as an honest miss at 1.0. Nothing is lost while that unit
        holds the key either: `find()` hits, and `recall` is only asked after a
        miss. Inferring it instead (say, from whether a lookup had already
        reported the loss) would make the same final registry ship marked or
        unmarked depending on whether someone happened to look in between.
        """
        owner = self._by_key.get(key)
        if owner is None or (carries_loss and owner is not unit):
            return
        del self._by_key[key]
        ring = self._by_unit.get(owner)
        if ring is not None and key in ring:
            ring.remove(key)

    def drop_unit(self, unit: Unit) -> None:
        """The unit closed: its ids now miss for an ordinary reason."""
        for key in self._by_unit.pop(unit, ()):
            if self._by_key.get(key) is unit:
                del self._by_key[key]

    def _at_fork_reinit(self) -> None:
        """Fork-child reset: the parent's units are not the child's to mark."""
        self._by_key.clear()
        self._by_unit.clear()
