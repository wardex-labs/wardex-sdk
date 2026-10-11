"""The drop ledger: what wardex lost, by reason and unit, on its way to the receiver.

A span can carry a marker for what it is missing. A loss that leaves no span
behind cannot: a batch the receiver refused, items the buffer evicted, a
connection a parser stopped reading, a swallowed internal error. Those used to
end in a counter nobody reads and one stderr line per kind per process, so an
empty stretch of a trace could not say whether nothing happened or wardex
missed it. This module adds them up instead, and the next envelope carries the
sum to the receiver as a `ClientReport`.

THE VOCABULARY IS CLOSED. `DropReason` says why, `DropUnit` says what was
counted, and the wire key is `"<reason>:<unit>"` in
`ClientReport.discarded_events`. Two reasons are not losses at all
(`before_send`, `excluded_by_policy`): they are recorded so a reader can tell
"the host filtered it" from "wardex lost it", and `REASONS` marks them. Where a
name is borrowed from Sentry's client reports it means what it means there.

RECORDING NEVER WAITS. `record()` is one `deque.append`; every `_FOLD_AT`
records it also folds the inbox into the tables, with a lock taken only by
`acquire(blocking=False)`. It is therefore safe to call under the client's
buffer lock, inside the export slot, from a signal handler that interrupted
either, and from a finalizer: none of them can make it wait, so none of them
can deadlock on it. It also calls no Python function on its common path, which
matters in `guard.__exit__` at the recursion limit: a swallowed `RecursionError`
leaves only the frames it unwound, and the counter bump beside it already needs
one.
Every reader folds before it reads, so a record waiting in the inbox is never
missed. The read for the wire (`attach`) never waits either: a drain that
cannot take the fold ships its envelope without a report, and the counts ride
the next one.

The fold lock is an RLock with a `_folding` flag rather than a plain Lock,
because the SDK keeps exactly one non-reentrant lock and a source scan holds it
to that. Reentrancy is the one thing the flag adds back: a signal handler that
re-enters on the thread already folding gets the RLock, sees the flag, and
leaves its entry in the inbox for the next fold.

DELIVERY, and why a count is never sent twice nor dropped. `attach` cuts the
pending counts into a report stamped with a strictly increasing `timestamp_ns`,
and the receiver folds reports by `(project, process pid, timestamp_ns)`. When
the POST carrying it fails, the transport says how (`unsent`):

* DEFINITELY not stored (the receiver answered with a status, or the
  connection never opened): the counts go back into the pending table and ride
  the next report.
* MAYBE stored (no answer came back): the SAME report, same timestamp, is held
  and sent again with the next envelope, so the receiver can fold the repeat.
  At most `_UNACKED_CAP` such reports are held; past that the oldest is dropped
  and counted as `ledger_overflow:report`.

A report is settled at most once per time it is handed out, and a batch's
failure is accounted at most once: a transport subclass that raises after its
own `export` already settled the batch gets `False` back from `unsent` and
counts nothing a second time.

Only a transport that can carry a report gets one: `carries()` asks for
`_carries_client_report`, which the wardex envelope transport sets. Taking the
counts for a transport that cannot ship them would make them vanish.

A drain with nothing else to send ships a ledger-only envelope at most once per
`SOLO_INTERVAL_S` (close's final drain excepted), so a process that only ever
loses data still says so without spending the receiver's request budget.

The ledger is process-global. A fork child starts from zero
(`_at_fork_reinit`): the parent owns what it counted and ships it itself.
"""

from __future__ import annotations

import inspect
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from .._types import ClientReport, Envelope
from ._drop_reasons import REASONS, DropReason, DropUnit, Loss, ReasonInfo

__all__ = [
    "REASONS",
    "DropReason",
    "DropUnit",
    "LedgerSnapshot",
    "Loss",
    "ReasonInfo",
    "attach",
    "carries",
    "due",
    "record",
    "record_failed_send",
    "record_items",
    "snapshot",
    "undelivered",
    "unsent",
]


#: A reason or unit from its wire string. A member is a `str` equal to its
#: value, so the member itself looks up the same entry.
_REASON_OF: dict[str, DropReason] = {m.value: m for m in DropReason}
_UNIT_OF: dict[str, DropUnit] = {m.value: m for m in DropUnit}

#: `ClientReport` counts are int32 on the wire. A count past this rides the
#: next report rather than being cut off.
_INT32_MAX = 2**31 - 1

#: How many reports of unknown fate are held for resending.
_UNACKED_CAP = 8

#: The least time between two ledger-only envelopes.
SOLO_INTERVAL_S = 30.0

#: How many records wait in the inbox before a record folds them itself.
#: Readers fold whatever is waiting, so this only bounds the inbox's memory.
_FOLD_AT = 64

#: Each reason's default unit, for `record`'s one lookup.
_DEFAULT_UNIT: dict[DropReason, DropUnit] = {r: info.unit for r, info in REASONS.items()}

#: How many distinct `where` labels the local view keeps before folding the
#: rest under `"<other>"`. Labels are a fixed set of call sites by convention;
#: this makes the convention a bound.
_WHERE_CAP = 1024

#: Read through the module, so tests can move the clock.
_monotonic = time.monotonic

# --- state, all of it reset at fork and for tests -----------------------------

#: (reason, unit, n, where) per record; `reason is None` is a failed send.
_INBOX: deque[tuple[DropReason | None, DropUnit | None, int, str | None]] = deque()
#: Taken only with `acquire(blocking=False)`; see the module docstring.
_FOLD = threading.RLock()
_folding = False
#: Process totals, monotonic: what `snapshot()` reads.
_TOTALS: dict[tuple[DropReason, DropUnit], int] = {}
_WHERE: dict[tuple[DropReason, str], int] = {}
#: Counted but not yet cut into a report.
_PENDING: dict[tuple[DropReason, DropUnit], int] = {}
_failed_sends = 0
#: Reports whose POST may or may not have arrived, oldest first.
_UNACKED: deque[ClientReport] = deque()
#: (report, definitely_not_stored) from `unsent`, applied at the next fold.
_SETTLED: deque[tuple[ClientReport, bool]] = deque()
#: The reports the last `attach` handed out and nobody has settled since, by
#: timestamp, each mapped to whether it was cut fresh then. Only a fresh one
#: may go back into the pending table when its POST definitely failed; one
#: that is not here any more was settled already, and settling it again would
#: count it twice.
_OUT: dict[int, bool] = {}
#: The last envelope `unsent` settled, held by identity so its id cannot be
#: reused while it is compared. Cleared by the next `attach`: every settle of
#: one batch happens inside the drain that sent it.
_last_unsent: list[object] = [None]
_last_timestamp = 0
_last_solo: float | None = None
_started = _monotonic()


@dataclass(frozen=True)
class LedgerSnapshot:
    """The process's totals so far, never reduced by sending.

    `counts` is keyed by `(reason, unit)` and includes `ledger_overflow`.
    `where` is the local-only breakdown by call-site label (a guard's label,
    `adapters.<name>`); the wire carries only the per-reason sum.
    """

    counts: Mapping[tuple[DropReason, DropUnit], int]
    where: Mapping[tuple[DropReason, str], int]

    def total(self, reason: DropReason) -> int:
        return sum(n for (r, _), n in self.counts.items() if r is reason)


# --- recording ----------------------------------------------------------------


def record(
    reason: DropReason, n: int = 1, unit: DropUnit | None = None, *, where: str | None = None
) -> None:
    """Add `n` lost items to the ledger. Never waits, never raises.

    `unit` defaults to the reason's own (`REASONS[reason].unit`). `where` is a
    low-cardinality call-site label kept in the local view only.
    """
    member = _REASON_OF.get(reason)
    if member is None or n <= 0:
        return
    _INBOX.append((member, _DEFAULT_UNIT[member] if unit is None else unit, n, where))
    if len(_INBOX) >= _FOLD_AT:
        _try_fold()


def record_items(
    reason: DropReason, spans: int, snapshots: int, *, where: str | None = None
) -> None:
    """`record` for a batch: its spans and its state snapshots, each in its own unit."""
    record(reason, spans, DropUnit.SPAN, where=where)
    record(reason, snapshots, DropUnit.SNAPSHOT, where=where)


def record_failed_send() -> None:
    """Count one failed send attempt, for the next report's `failed_sends`."""
    _INBOX.append((None, None, 1, None))
    if len(_INBOX) >= _FOLD_AT:
        _try_fold()


def snapshot() -> LedgerSnapshot:
    """A copy of the process totals. Takes no lock it would wait for."""
    folded = _try_fold()
    counts = dict(_TOTALS)
    where = dict(_WHERE)
    if not folded:
        # Another frame is folding: what it has not reached yet is still
        # in the inbox, and belongs in a read of the totals.
        for reason, unit, n, label in tuple(_INBOX):
            if reason is not None and unit is not None:
                counts[(reason, unit)] = counts.get((reason, unit), 0) + n
                if label is not None:
                    where[(reason, label)] = where.get((reason, label), 0) + n
    return LedgerSnapshot(counts=MappingProxyType(counts), where=MappingProxyType(where))


# --- delivery -----------------------------------------------------------------


def carries(transport: object) -> bool:
    """Whether `transport` ships `ClientReport`s. Read statically, so a
    transport's own `__getattr__` or properties never run here."""
    return inspect.getattr_static(transport, "_carries_client_report", False) is True


def due(transport: object, *, final: bool = False) -> bool:
    """Whether a drain with nothing else to send should send a ledger-only
    envelope now. Claims the interval when it answers yes."""
    global _last_solo
    if not carries(transport):
        return False
    if not (_PENDING or _UNACKED or _INBOX or _SETTLED or _failed_sends):
        return False
    now = _monotonic()
    if not final and _last_solo is not None and now - _last_solo < SOLO_INTERVAL_S:
        return False
    _last_solo = now
    return True


def attach(envelope: Envelope, transport: object, queue_depth: int) -> Envelope:
    """`envelope` with the ledger's reports added, when `transport` carries
    them and the fold is free; otherwise `envelope` unchanged."""
    _last_unsent[0] = None  # a new batch: the last one's settles are over
    if type(envelope) is not Envelope or not carries(transport):
        return envelope
    if not _claim():
        return envelope
    try:
        _fold_held()
        reports = list(_UNACKED)
        _UNACKED.clear()
        _OUT.clear()
        _OUT.update((r.timestamp_ns, False) for r in reports)
        fresh = _cut_held(queue_depth)
        if fresh is not None:
            reports.append(fresh)
            _OUT[fresh.timestamp_ns] = True
    finally:
        _release()
    if not reports:
        return envelope
    return replace(envelope, client_reports=envelope.client_reports + tuple(reports))


def unsent(envelope: Envelope, *, definite: bool) -> bool:
    """The POST carrying `envelope` failed. `definite`: the receiver certainly
    did not store it, so its fresh counts go back into the pending table;
    otherwise every report it carried is held and sent again unchanged.

    Returns whether this is news. False, changing nothing, when this same
    envelope was already settled -- a transport subclass that raised after its
    own `export` settled the batch -- so the caller does not count the batch's
    items twice. Every caller that accounts a failed batch asks this first."""
    if envelope is _last_unsent[0]:
        return False
    _last_unsent[0] = envelope
    if type(envelope) is Envelope:
        for report in envelope.client_reports:
            _SETTLED.append((report, definite))
        _try_fold()
    return True


def undelivered() -> dict[tuple[DropReason, DropUnit], int]:
    """What no receiver has confirmed: the pending counts plus every report
    held for resending."""
    _try_fold()
    out = dict(_PENDING)
    for report in tuple(_UNACKED) + tuple(r for r, _ in tuple(_SETTLED)):
        for key, n in _keys_of(report):
            out[key] = out.get(key, 0) + n
    return out


# --- the fold -----------------------------------------------------------------


def _claim() -> bool:
    """Take the fold for this frame without waiting. False when another
    thread holds it, or this thread is already inside a fold it interrupted."""
    global _folding
    if not _FOLD.acquire(blocking=False):
        return False
    if _folding:
        _FOLD.release()
        return False
    _folding = True
    return True


def _release() -> None:
    global _folding
    _folding = False
    _FOLD.release()


def _try_fold() -> bool:
    if not _claim():
        return False
    try:
        _fold_held()
    finally:
        _release()
    return True


def _add(table: dict, key: object, n: int) -> None:
    table[key] = table.get(key, 0) + n


def _fold_held() -> None:
    """Move the inbox and the settled reports into the tables. Fold held."""
    global _failed_sends
    while _INBOX:
        reason, unit, n, where = _INBOX.popleft()
        if reason is None or unit is None:
            _failed_sends += n
            continue
        _add(_TOTALS, (reason, unit), n)
        _add(_PENDING, (reason, unit), n)
        if where is not None:
            label = where if (reason, where) in _WHERE or len(_WHERE) < _WHERE_CAP else "<other>"
            _add(_WHERE, (reason, label), n)
    while _SETTLED:
        report, definite = _SETTLED.popleft()
        fresh = _OUT.pop(report.timestamp_ns, None)
        if fresh is None:
            continue  # not out (settled already, or never handed out): nothing to settle
        if definite and fresh:
            for key, n in _keys_of(report):
                _add(_PENDING, key, n)
            _failed_sends += report.failed_sends
            continue
        _UNACKED.append(report)
        while len(_UNACKED) > _UNACKED_CAP:
            _UNACKED.remove(min(_UNACKED, key=lambda r: r.timestamp_ns))
            _add(_TOTALS, (DropReason.LEDGER_OVERFLOW, DropUnit.REPORT), 1)
            _add(_PENDING, (DropReason.LEDGER_OVERFLOW, DropUnit.REPORT), 1)


def _cut_held(queue_depth: int) -> ClientReport | None:
    """The pending counts as one new report, or None when there are none."""
    global _failed_sends, _last_timestamp
    if not _PENDING and not _failed_sends:
        return None
    events: list[tuple[str, int]] = []
    for key in sorted(_PENDING, key=_wire_key):
        n = _PENDING.pop(key)
        events.append((_wire_key(key), min(n, _INT32_MAX)))
        if n > _INT32_MAX:
            _PENDING[key] = n - _INT32_MAX
    sends = min(_failed_sends, _INT32_MAX)
    _failed_sends -= sends
    _last_timestamp = max(time.time_ns(), _last_timestamp + 1)
    report = ClientReport(
        timestamp_ns=_last_timestamp,
        discarded_events=tuple(events),
        failed_sends=sends,
        queue_depth=min(max(queue_depth, 0), _INT32_MAX),
        uptime_ms=int((_monotonic() - _started) * 1000),
    )
    return report


def _wire_key(key: tuple[DropReason, DropUnit]) -> str:
    return f"{key[0].value}:{key[1].value}"


def _keys_of(report: ClientReport) -> list[tuple[tuple[DropReason, DropUnit], int]]:
    """A report's counts back as `(reason, unit)` keys; keys this ledger
    did not write are skipped."""
    out = []
    for name, n in report.discarded_events:
        reason_value, _, unit_value = name.partition(":")
        reason, unit = _REASON_OF.get(reason_value), _UNIT_OF.get(unit_value)
        if reason is not None and unit is not None:
            out.append(((reason, unit), n))
    return out


# --- resets -------------------------------------------------------------------


def _at_fork_reinit() -> None:
    """Fork-child reset, run from `Runtime.after_in_child` step 0: start from
    zero with a fresh lock. Acquires nothing, since a parent thread may have
    held the fold at the fork; the parent ships what it counted."""
    global _FOLD, _folding, _failed_sends, _last_timestamp, _last_solo, _started
    _FOLD = threading.RLock()
    _folding = False
    _INBOX.clear()
    _TOTALS.clear()
    _WHERE.clear()
    _PENDING.clear()
    _UNACKED.clear()
    _SETTLED.clear()
    _OUT.clear()
    _last_unsent[0] = None
    _failed_sends = 0
    _last_timestamp = 0
    _last_solo = None
    _started = _monotonic()


def _reset_for_tests() -> None:
    """Test-only: forget everything, as a fresh process would."""
    _at_fork_reinit()
