"""The deferred-parse queue: span finalization off the caller's thread.

A byte seam completes a transaction inside the HOST'S socket call — on an
asyncio host that caller is the event-loop thread — and the LLM-semantic parse
of that transaction's bodies is the one expensive step of turning it into a
span (measured ≈ 5 ms/MB; Python assembly is ~0.03 ms). This queue is where
that work goes instead: the caller seals what only it can read and submits a
`DeferredSpan`; the `wardex-finalize-worker` thread runs the parse (which
releases the GIL) and the assembly, and admits the result into the client's
span buffer.

Bounded on BOTH axes (`max_parse_backlog` jobs / `max_parse_backlog_bytes` of
raw bodies), drop-oldest like the span buffer — but an evicted job is never
silent: it is assembled WITHOUT the parse on the submitting thread (~30 µs)
and ships carrying `PARSE_BACKLOG_FULL`. A single job larger than the byte
bound never enters the queue at all — same fallback, same marker — so
`pending_bytes() <= max_bytes` is literal, not "plus one oversize job".

Fork posture (I-fork-3): a child inherits whatever was pending, and every one
of those jobs was sealed by — and belongs to — the parent, which still owns
and exports them. `_at_fork_reinit` therefore replaces the lock/CV (an
inherited lock can arrive held by a thread that did not cross the fork) and
discards the inherited entries WITHOUT emitting. The PID-owned `in_flight`
count is the backstop for platforms where the hook never ran: a child that
inherited `in_flight == 1` has no thread that will ever decrement it, and
without the reset every `drain_all` in the child would burn its full timeout
waiting on the parent's parse.
"""

from __future__ import annotations

import contextvars
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Protocol

from ._assembly import Limitation, counters, guard, report_once
from ._worker import BatchWorker

if TYPE_CHECKING:
    from ._types import InternalSpan

#: The finalize worker's idle-wakeup interval. This queue is EVENT-DRIVEN —
#: `submit` calls `wake()` — so this number is only how often an idle thread
#: stirs; nothing functional may depend on it, and a test pins that a submit
#: finalizes in milliseconds with this set to an hour.
_IDLE_SWEEP = 60.0


class DeferredSpan(Protocol):
    """One span's worth of work, sealed by the caller and finished elsewhere.

    Structural on purpose: `_interceptors/` must not import this module (the
    seam only ever talks to `Client.capture_deferred`), and the queue needs
    nothing from a job beyond this surface.

    `ctx` is the `contextvars.copy_context()` taken at seal time; `run` and
    `fallback` are ALWAYS executed inside it (`ctx.run(...)`), so every
    immutable-fact ContextVar the gate reads (`in_degraded_run()` and
    friends) answers as it did at the response instant. Mutable Scope state
    (tags/user) is NOT carried here — `Client.capture_deferred` snapshots it
    at submit time and the queue hands the snapshot back at admit (§3.7 of
    the design: `copy_context` preserves bindings, not the contents of a
    mutable object bound in one).
    """

    size: int
    ctx: contextvars.Context

    def run(self) -> InternalSpan | None:
        """Full finalization, parse included. None = the gate refused."""
        ...

    def fallback(self, marker: Limitation) -> InternalSpan | None:
        """Parse-less finalization carrying `marker`. Must stay µs-cheap:
        this is the one thing the queue ever runs on a submitting thread."""
        ...


class Leftover(Enum):
    """What `drain_all` does with jobs its deadline left unfinished."""

    #: flush: leave them queued — the worker (or the next drain) finishes them.
    KEEP = auto()
    #: close/signal: the process is ending, so KEEP would discard them with
    #: it. Every remaining job is assembled as a `PARSE_SKIPPED_AT_SHUTDOWN`
    #: fallback (µs each), so FALLBACK always returns with an empty queue.
    FALLBACK = auto()


@dataclass(frozen=True, slots=True)
class _Entry:
    job: DeferredSpan
    #: the mutable-Scope snapshot (tags, user) `capture_deferred` took on the
    #: submitting thread — the values the span is stamped with at admit.
    scope: tuple[dict[str, str], Any]


class FinalizeQueue:
    """Bounded deferred-finalization queue + its dedicated worker thread.

    Locking: one RLock (the house rule — the SDK's single plain Lock is
    `Client._close_lock`) with a Condition on it for the in-flight wait.
    The lock guards deque/byte/in-flight bookkeeping ONLY: jobs are executed
    and spans admitted strictly outside it (I11/I-F6), so the submit path
    can never block behind a parse. Lock order is queue → buffer (the admit
    callback takes the client's buffer lock); queue and export never nest —
    `flush`/`close` run `drain_all` strictly BEFORE `_drain`.

    The one bounded stall, recorded like `_client.py` records its AB-BA
    sibling: a signal handler that interrupts `capture_span` inside the
    buffer-lock block and then runs `drain_all` can wait on a worker that is
    itself waiting to admit into that same buffer — and survives it because
    the in-flight wait is TIMED, so the handler falls out on its own budget
    with the job left in flight rather than deadlocking.
    """

    def __init__(
        self,
        *,
        max_jobs: int,
        max_bytes: int,
        admit: Callable[..., None],
        debug: bool,
    ) -> None:
        self._max_jobs = max_jobs
        self._max_bytes = max_bytes
        self._admit = admit
        self._debug = debug
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._entries: deque[_Entry] = deque()
        self._bytes = 0
        self._stopping = False
        # In-flight is PID-OWNED (the `_thread_for_pid` idea one layer up): a
        # fork while the worker is inside `_run_one` hands the child an
        # in-flight count no thread of its own will ever decrement.
        self._in_flight = 0
        self._in_flight_pid: int | None = None
        self._worker = BatchWorker(
            self._run_pending,
            interval=_IDLE_SWEEP,
            debug=debug,
            name="wardex-finalize-worker",
        )

    # -- caller-thread surface ------------------------------------------

    def ensure_alive(self) -> None:
        """Respawn the worker if needed; reset a fork-inherited in-flight.

        Called on every `capture_deferred`. The PID reset is the hook-less
        platform's backstop: without it a prefork host that forked mid-parse
        gets a child whose every flush/close burns its full budget waiting
        for a parse that is running in the PARENT.
        """
        with self._lock:
            if self._in_flight_pid is not None and self._in_flight_pid != os.getpid():
                self._in_flight = 0
                self._in_flight_pid = None
        self._worker.ensure_alive()

    def submit(self, job: DeferredSpan, scope: tuple[dict[str, str], Any]) -> None:
        """Enqueue one sealed job; enforce both bounds, never silently.

        Runs on the caller's thread, so the ceiling on what it may do is the
        parse-less fallback (I-F4). Three shapes:

        * an OVERSIZE job (`size > max_bytes`) never enters the queue —
          evicting everything else could not make it fit, and admitting it
          anyway would falsify the byte bound by up to two whole bodies. It
          is assembled inline as a `PARSE_BACKLOG_FULL` fallback;
        * a full queue evicts OLDEST-first (recent spans are worth more,
          `_SpanBuffer`'s own policy), each eviction assembled the same way;
        * otherwise: append, count bytes, wake the worker.

        Every eviction is counted AND reported through `report_once` —
        unconditionally, not debug-gated: overload loss is the one line an
        operator has to be able to see.
        """
        # Direct attribute access, not a defensive getattr: `DeferredSpan`
        # DECLARES `size`, and a job without one is a producer bug the
        # callers' guards surface — a silent 0 would let it bypass the byte
        # bound entirely.
        size = int(job.size)
        if size > self._max_bytes:
            counters.bump("client.finalize.backlog_evicted")
            self._report_backlog(1)
            self._fallback_now(_Entry(job, scope), Limitation.PARSE_BACKLOG_FULL)
            return
        evicted: list[_Entry] = []
        with self._lock:
            while self._entries and (
                len(self._entries) >= self._max_jobs or self._bytes + size > self._max_bytes
            ):
                old = self._entries.popleft()
                self._bytes -= int(old.job.size)
                evicted.append(old)
            self._entries.append(_Entry(job, scope))
            self._bytes += size
        for _ in evicted:
            counters.bump("client.finalize.backlog_evicted")
        if evicted:
            self._report_backlog(len(evicted))
        for old in evicted:
            self._fallback_now(old, Limitation.PARSE_BACKLOG_FULL)
        self._worker.wake()

    def _report_backlog(self, count: int) -> None:
        report_once(
            f"deferred-parse backlog full; {count} transaction(s) shipped without "
            "gen_ai semantics. Raise max_parse_backlog / max_parse_backlog_bytes.",
            key="wardex.finalize.backlog",
        )

    def _fallback_now(self, entry: _Entry, marker: Limitation) -> None:
        """Assemble `entry` parse-less, on THIS thread, inside wardex's guard.

        The guard is load-bearing (review M2-3): `SpanDraft.finish()` raises
        `VocabularyError` on a vocabulary breach, this path runs under
        backlog pressure — i.e. exactly the overload case — and without the
        guard that raise would surface out of the host's own `sock.recv()`.

        The zero-argument closure is for the census scanner as much as for
        `Context.run`: handing `run` a marker-ish argument would register the
        stdlib's `run` as a marker sink and drag every `ctx.run` call in the
        package into the unresolved-slot ledger.
        """
        job = entry.job

        def assemble() -> Any:
            return job.fallback(marker)

        with guard("client.finalize.fallback", debug=self._debug):
            span = job.ctx.run(assemble)
            if span is not None:
                self._admit(span, scope=entry.scope)

    # -- worker-thread surface ------------------------------------------

    def _run_pending(self) -> None:
        """The worker's drain function: finish queued jobs until none remain.

        The lock is held for deque/bookkeeping only; `_run_one` (parse +
        assembly + admit) runs outside it, so submitters never wait behind
        a parse.
        """
        while True:
            with self._lock:
                if self._stopping or not self._entries:
                    return
                entry = self._entries.popleft()
                self._bytes -= int(entry.job.size)
                self._in_flight += 1
                self._in_flight_pid = os.getpid()
            try:
                self._run_one(entry)
            finally:
                with self._lock:
                    self._in_flight -= 1
                    if self._in_flight == 0:
                        self._in_flight_pid = None
                    self._cv.notify_all()

    def _run_one(self, entry: _Entry) -> None:
        """Finish one job. `ctx.run` carries the sealed immutable facts; the
        scope stamp comes from the SNAPSHOT, never from re-reading ContextVars
        on this thread (I-F9)."""
        with guard("client.finalize.run", debug=self._debug):
            span = entry.job.ctx.run(entry.job.run)
            if span is not None:
                self._admit(span, scope=entry.scope)

    # -- drain / shutdown ------------------------------------------------

    def drain_all(self, deadline: float | None, *, leftover: Leftover) -> int:
        """Finish pending jobs on the CALLING thread, up to `deadline`.

        `deadline` is an absolute `time.monotonic()` instant (None: no limit).
        On expiry, `KEEP` leaves the remainder queued and `FALLBACK` assembles
        every remaining job as a `PARSE_SKIPPED_AT_SHUTDOWN` fallback — µs per
        job, so FALLBACK always returns with an empty queue (I-F5): the
        process is ending, and a kept job would die with it.

        Ends by waiting — TIMED, within the same deadline — for the worker's
        in-flight job, so a `close()` that follows can trust the admit
        happened before its final drain. Works after `stop()`: the worker
        loop is what `_stopping` stops, not this method.

        Returns how many jobs were finished (fallbacks included).
        """
        with self._lock:
            if self._in_flight_pid is not None and self._in_flight_pid != os.getpid():
                self._in_flight = 0  # inherited from a pre-fork parent (I-F12)
                self._in_flight_pid = None
        finished = 0
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            with self._lock:
                if not self._entries:
                    break
                entry = self._entries.popleft()
                self._bytes -= int(entry.job.size)
            self._run_one(entry)
            finished += 1
        if leftover is Leftover.FALLBACK:
            while True:
                with self._lock:
                    if not self._entries:
                        break
                    entry = self._entries.popleft()
                    self._bytes -= int(entry.job.size)
                counters.bump("client.finalize.shutdown_fallback")
                marker = Limitation.PARSE_SKIPPED_AT_SHUTDOWN
                self._fallback_now(entry, marker)
                finished += 1
        with self._cv:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            self._cv.wait_for(lambda: self._in_flight == 0, timeout=remaining)
        return finished

    def stop(self, timeout: float) -> None:
        """Stop the worker LOOP and join it; pending jobs stay queued.

        Draining what remains is the caller's job (`close()` runs `drain_all`
        right after), mirroring `BatchWorker.stop`'s own contract. After a
        stop, `ensure_alive` cannot respawn — `BatchWorker._stopped` refuses.
        """
        self._stopping = True
        self._worker.stop(timeout)

    # -- introspection (tests, probes, the pending fixture) ---------------

    def pending(self) -> int:
        with self._lock:
            return len(self._entries)

    def pending_bytes(self) -> int:
        with self._lock:
            return self._bytes

    # -- fork -------------------------------------------------------------

    def _at_fork_reinit(self) -> None:
        """Fork-child reset: fresh lock/CV, inherited jobs discarded UNEMITTED.

        REPLACE, never acquire (I-fork-5): the parent may have been inside
        `submit` or the worker's bookkeeping at the fork instant, so the
        inherited lock can be held by a thread that does not exist here.

        THE BACKLOG IS DISCARDED, NOT DRAINED (I-fork-3, integration S-4):
        every pending job was sealed by the parent, the parent still owns and
        exports it, and a child that finished its inherited copy would ship
        the same transaction twice. No marker and no counter for the same
        reason the span buffer's fork reset has none: nothing was lost — the
        data ships from the process that owns it.

        The worker resets last, into the lazy-respawn posture the next
        `capture_deferred`'s `ensure_alive()` services.
        """
        self._lock = threading.RLock()
        self._cv = threading.Condition(self._lock)
        self._entries = deque()
        self._bytes = 0
        self._in_flight = 0
        self._in_flight_pid = None
        # `_stopping` is inherited as-is, like `BatchWorker._stopped` one
        # layer down: a closed parent's child stays closed.
        self._worker._at_fork_reinit()
