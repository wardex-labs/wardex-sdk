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


def forgotten_edge(sole: Unit | None, amb: Ambient, hint: str | None) -> Parentage:
    """`UnitRegistry._edge`'s ladder for a `find()` miss on a FORGOTTEN id.

    The ordinary miss takes the ambient scope first, and that is exactly the
    rung that may not answer here unmarked. So the rungs that mark themselves go
    first — the sole live session (`UNIT_SOLE`, 0.5), whose marker the parentage
    table attaches — then the ambient scope, capped at the alias tier so the
    replacement cannot outrank the answer the id used to give, then no parent
    at all (`UNRESOLVED`, likewise self-marking). Whichever answers also carries
    `ALIAS_FORGOTTEN`, which is the one fact none of those rungs can say: the
    edge is a fallback because wardex's own bound threw the id away.

    `sole` is asked by the caller, which owns the table it is asked of.
    """
    if sole is not None:
        edge = sole.child(Evidence(ParentSource.UNIT_SOLE, request_id=hint))
    elif amb.span_context is not None:
        edge = resolve_parentage(
            amb, cap_at_alias_tier(Evidence(ParentSource.CONTEXTVAR, request_id=hint))
        )
    else:
        edge = resolve_parentage(EMPTY_AMBIENT, Evidence(ParentSource.UNRESOLVED, request_id=hint))
    return edge.with_limitation(Limitation.ALIAS_FORGOTTEN)


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

    def recall(self, key: UnitKey) -> bool:
        """Was `key` dropped by the bound? Asking is consuming, and counted.

        The callers ask at the moment they are about to build an edge from a
        `find()` miss, so `alias_forgotten_consumed` says how many edges the
        bound degraded, where `alias_table_full` says how many ids it dropped.
        The record is NOT removed by asking: a second lookup of the same id is
        the same loss, and must say so again.
        """
        if key not in self._by_key:
            return False
        counters.bump("assembly._units.alias_forgotten_consumed")
        return True

    def rebound(self, key: UnitKey) -> None:
        """`key` resolves again, so any record of having dropped it is stale.

        Kept, it would mark a good alias edge the next time the key missed for
        an unrelated reason. One O(1) pop on the common path, which is
        `UnitRegistry.open()`'s.
        """
        owner = self._by_key.pop(key, None)
        if owner is not None:
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
