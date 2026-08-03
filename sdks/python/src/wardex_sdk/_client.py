from __future__ import annotations

import inspect
import platform
import sys
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable

from ._config import WardexConfig
from ._types import (
    EnvelopeHeader,
    InternalEnvelope,
    InternalSpan,
    InternalStateSnapshot,
    SdkInfo,
)
from ._version import __version__
from ._worker import BatchWorker
from .transport._base import Transport


def build_sdk_info() -> SdkInfo:
    return SdkInfo(
        name="wardex.python",
        version=__version__,
        python_version=platform.python_version(),
        os=sys.platform,
        arch=platform.machine(),
    )


# Fixed per-span overhead: context, timing, attributes, and the deque slot.
# An exact figure would mean encoding every span on the hot path.
_SPAN_OVERHEAD_BYTES = 512


def _span_size(span: InternalSpan) -> int:
    return _SPAN_OVERHEAD_BYTES + len(span.input_data or b"") + len(span.output_data or b"")


# Handed to Transport.flush() on the periodic path, which carries no deadline of
# its own. Transport.flush() is a no-op for every transport we ship, so this is
# only ever consumed by third-party transports that buffer.
_UNBOUNDED_TRANSPORT_FLUSH_TIMEOUT = 5.0


def _accepts_timeout(fn: Callable[..., object]) -> bool:
    """Whether `fn` can be called with a `timeout=` keyword.

    `Transport.export` gained an optional `timeout` so a bounded flush can bound
    the POST it is waiting on, but `Transport` is exported from the package root
    and subclasses written against the previous two-argument `export(envelope)`
    are already in the wild. Calling one of those with `timeout=` raises
    TypeError inside `_drain`'s fail-closed handler, which would silently drop
    every envelope for that transport -- a stall traded for total data loss.
    So probe, once per transport instance, instead of assuming.

    An unreadable signature (C callables, exotic wrappers) reads as "cannot take
    it". That is the safe direction: the cost of withholding the deadline is a
    slower export, the cost of a wrong guess is the dropped envelope above.
    """
    try:
        parameters = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
    for param in parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "timeout" and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


class _SpanBuffer:
    """A span deque and its approximate byte total, folded into one object so
    _drain() can only ever replace the *whole* pair via a single attribute
    assignment (`self._buffer = ...`).

    A prior design kept the deque and the byte total as two separate Client
    attributes, guarded by an `if self._spans is spans:` identity check
    before every counter update. That check and the update it guarded were
    still two separate statements, and a reentrant drain landing between
    them could invalidate the check's premise: the drain resets the counter
    to 0 out from under a subtraction that already passed the check,
    producing a negative total, or exports the just-appended span and resets
    to 0 out from under an increment that already passed, producing an
    overstated total. No amount of additional checking closes that gap,
    because every check is itself a statement a drain can land after.

    Folding spans+bytes into one object sidesteps the problem instead of
    arguing around it: every mutation here (evict_oldest, append) is
    unconditional and operates on *this* object's own fields. Whether or not
    `self` is still the live `Client._buffer` by the time the mutation
    returns is irrelevant, because nothing here ever depends on that -- an
    eviction subtracts from the same object it popped from, so it can never
    go negative; an append adds to the same object it appended to, so it can
    never overstate. There is no separate counter left for a reentrant swap
    to desynchronize.
    """

    __slots__ = ("spans", "bytes")

    def __init__(self) -> None:
        self.spans: deque[InternalSpan] = deque()
        self.bytes = 0

    def evict_oldest(self) -> InternalSpan:
        evicted = self.spans.popleft()
        self.bytes -= _span_size(evicted)
        return evicted

    def append(self, span: InternalSpan, size: int) -> None:
        self.spans.append(span)
        self.bytes += size


class Client:
    def __init__(self, config: WardexConfig, transport: Transport) -> None:
        self._config = config
        self._transport = transport
        self._sdk_info = build_sdk_info()
        # Lock order is always export lock → buffer lock (one-way; no deadlock).
        # Nothing ever takes the export lock while holding the buffer lock:
        # _drain() acquires the export lock strictly before it opens the buffer
        # lock block, capture_span() calls _worker.wake() outside its block, and
        # _SpanBuffer takes no lock at all.
        # The buffer lock only ever guards an append or a swap — never I/O,
        # encoding, or callbacks (design §5).
        # Both locks are reentrant: the same-thread signal handler may call
        # flush() (and so re-enter _drain, and re-acquire the buffer lock) while
        # the main thread is already mid-append or mid-drain (manual flush() in
        # progress, or install()'s previous.close() during re-init). Every
        # guarded block re-reads self._buffer/self._snapshots fresh each time, so
        # nested reentrant acquisition cannot corrupt or duplicate state — a
        # plain Lock would instead hang forever on that same-thread re-acquire.
        # Cross-thread serialization (the invariant these locks exist for) is
        # unchanged: RLock still blocks other threads until fully released.
        self._buffer_lock = threading.RLock()
        # Named for the guarantee it carries, not for the method that takes it:
        # transport.export()/flush() are never entered by two threads at once,
        # which is the only reason a third-party Transport may hold per-instance
        # mutable state without a lock of its own. It also makes swap order
        # equal export order.
        self._export_lock = threading.RLock()
        # Memo for the export-signature probe, keyed by transport identity
        # because `_transport` is reassignable and is in fact reassigned after
        # construction (tests, and anyone swapping an exporter at runtime).
        # A probe cached once at construction would answer for the transport
        # that is gone and hand `timeout=` to one that cannot take it.
        self._probed_transport: Transport = transport
        self._export_takes_timeout = _accepts_timeout(transport.export)
        self._buffer = _SpanBuffer()
        self._snapshots: deque[InternalStateSnapshot] = deque()
        self._dropped = 0
        self._closed = False
        # Deliberately NOT reentrant, and safe only because the signal handler
        # calls flush() and never close(): a signal landing between the three
        # statements this guards would self-deadlock permanently on re-entry.
        # Anything that routes close() onto the signal path must make this an
        # RLock first.
        self._close_lock = threading.Lock()
        limits = config.limits.resolved()
        self._max_buffer_spans = limits["max_buffer_spans"]
        self._max_buffer_bytes = limits["max_buffer_bytes"]
        self._flush_threshold = max(1, self._max_buffer_spans // 4)
        # None, not a number: the periodic drain is a background daemon that
        # nobody waits on, so it has no deadline to impose. Handing it one would
        # clamp the transport's own configured timeout on the *only* path that
        # ships data without anyone asking -- an OtlpHttpTransport(timeout=10.0)
        # would start abandoning POSTs at 5s and lose those envelopes outright.
        # A deadline exists only where a caller named one: flush(t), close(t),
        # and above all the signal handler's flush(2.0).
        self._worker = BatchWorker(
            lambda: self._drain(None), interval=config.flush_interval, debug=config.debug
        )
        self._worker.start()

    @property
    def config(self) -> WardexConfig:
        return self._config

    # -- test-only internal accessors -----------------------------------
    # `_spans`/`_buffered_bytes` are not part of the public API; several
    # tests read the resident deque and its byte total to assert on
    # reentrancy edge cases. Both are deliberately read-only views onto
    # self._buffer: a setter for either would let a caller replace one half
    # of the pair and leave the other describing something that no longer
    # exists, which is precisely the two-piece state _SpanBuffer exists to
    # rule out. A test that must swap the deque itself reaches through
    # `_buffer.spans` directly, so the hazard has no general route.
    @property
    def _spans(self) -> deque[InternalSpan]:
        return self._buffer.spans

    @property
    def _buffered_bytes(self) -> int:
        return self._buffer.bytes

    def capture_span(self, span: InternalSpan) -> None:
        if self._closed:
            return
        self._worker.ensure_alive()  # fork/thread-death recovery (design §8)
        size = _span_size(span)
        with self._buffer_lock:
            # Drop-oldest on either bound: recent spans are worth more. The byte
            # budget is the backstop that keeps resident memory bounded even
            # when a single span is far larger than the average.
            #
            # Re-entrancy hazard: a same-thread signal handler can call
            # flush() (and so _drain()) between any two statements in this
            # block via the reentrant _buffer_lock (see the class-level
            # comment on the lock). _drain() replaces self._buffer wholesale
            # (see _SpanBuffer's docstring for why the deque and its byte
            # total are folded into one object rather than two separately
            # guarded attributes -- a prior version of this method used an
            # `if self._spans is spans:` identity check before each counter
            # update, but that check and the update were themselves two
            # statements, and a drain landing between them could invalidate
            # a check that had already passed, producing a negative or
            # overstated total; reproduced and fixed, see git history and
            # the tests below). With the fold:
            #   - the walrus below re-reads self._buffer into `buf` on every
            #     loop condition check, and the loop body always evicts from
            #     that same `buf` local -- never a separately re-read
            #     self._buffer -- so a drain can never swap in an empty
            #     buffer between "checked non-empty" and "popped" (which
            #     would otherwise raise IndexError);
            #   - evict_oldest() and append() are unconditional: no identity
            #     check guards them, because none is needed -- each mutates
            #     only the object it was called on, which stays internally
            #     coherent (spans and bytes always agree) whether or not
            #     that object is still the live self._buffer by the time the
            #     call returns. An eviction can never drive a *stale*
            #     buffer's byte total negative, because the total and the
            #     deque it describes are always the same object's own
            #     fields;
            #   - the final append resolves self._buffer fresh, right there
            #     in the call, so the span lands in whatever buffer is live
            #     at that statement, never one read earlier and orphaned by
            #     an intervening drain.
            # What's NOT eliminated, stated plainly because it is worse than
            # a delay: a window narrower than one statement, between
            # resolving self._buffer for that trailing call and
            # _SpanBuffer.append's own first line running. Only a same-thread
            # signal handler can land there -- a drain on another thread
            # blocks on the buffer lock this whole block holds -- and such a
            # handler runs to completion before the interrupted statement
            # resumes. By then it has already swapped self._buffer *and*
            # serialized the old buffer's spans into an envelope (_drain
            # materializes them with tuple(spans) before returning), without
            # ours. The resumed append therefore mutates the pre-drain
            # buffer, which at that point nothing references: self._buffer
            # holds the replacement, _drain's locals died with its frame, and
            # `buf` above is never read again. The span is lost outright --
            # not deferred to the next drain, never on the wire at all -- and
            # it is not counted in self._dropped either, because nothing
            # still alive can observe that it happened. Byte accounting is
            # unaffected (the orphan stays internally consistent and is
            # simply collected), so this is span loss, never counter drift.
            # This is the same class of bytecode-internal, no-second-line gap
            # already present in self._snapshots.append() below and in the
            # pre-byte-budget code. It cannot be closed without giving up
            # same-thread signal-handler reentrancy, and it cannot be counted
            # either: any detection step is itself a statement with a window
            # of the same kind, so it would narrow the silent gap rather than
            # remove it, at the cost of permanent state and a hot-path branch.
            while (buf := self._buffer).spans and (
                len(buf.spans) >= self._max_buffer_spans
                or buf.bytes + size > self._max_buffer_bytes
            ):
                buf.evict_oldest()
                self._dropped += 1
            self._buffer.append(span, size)
            should_wake = len(self._buffer.spans) >= self._flush_threshold
        if should_wake:
            self._worker.wake()

    def capture_snapshot(self, snapshot: InternalStateSnapshot) -> None:
        if self._closed:
            return
        self._worker.ensure_alive()
        with self._buffer_lock:
            # Same reentrancy hazard as capture_span (see its comment): check
            # and pop against the same local reference so a reentrant drain
            # can never swap in an empty deque between "checked non-empty"
            # and "popped".
            snapshots = self._snapshots
            if snapshots and len(snapshots) >= self._max_buffer_spans:
                snapshots.popleft()
                if self._snapshots is snapshots:
                    self._dropped += 1
            self._snapshots.append(snapshot)

    def flush(self, timeout: float = 5.0) -> None:
        self._drain(timeout)

    def _acquire_export_slot(self, timeout: float | None) -> bool:
        """Take the export lock, waiting no longer than `timeout` for it.

        Returns False when the wait ran out, at which point the caller has taken
        nothing and must simply return: the buffer is untouched, so there is no
        envelope to put back and no window in which spans belong to nobody.

        `timeout=None` waits indefinitely, which is what the periodic worker
        wants -- blocking a background daemon costs nothing.
        """
        if timeout is None:
            self._export_lock.acquire()
            return True
        # A timed acquire rejects negatives (ValueError) and cannot represent
        # inf (OverflowError: timestamp out of range). Either would raise out of
        # flush() into the application, which an observability SDK may not do,
        # so clamp into the range the primitive accepts rather than trusting the
        # caller's number.
        budget = min(max(timeout, 0.0), threading.TIMEOUT_MAX)
        # A thread that already owns this RLock -- the signal handler re-entering
        # through before_send or through transport.export -- is granted it
        # immediately even at budget=0, so reentrancy never spuriously declines.
        return self._export_lock.acquire(timeout=budget)

    def _drain(self, timeout: float | None) -> None:
        """Export everything buffered, within `timeout` seconds end to end.

        `timeout` is a wall-clock bound on this whole call, not a per-step one.
        It covers the wait for the export slot, the POST, and the transport
        flush after it, all measured against a single monotonic deadline. This
        is what makes the signal handler's flush(2.0) mean two seconds: before,
        the drain lock was held across the synchronous POST, so a flush arriving
        behind an in-flight export waited out that export's full transport
        timeout, then spent its own, then flushed -- a "2s" bound that measured
        22s on a stalled backend and delayed process exit by that much.

        The export lock is still held across the POST, because serializing
        transport.export() is the guarantee third-party transports were written
        against. What changed is that waiting for it is now bounded: a drain
        that cannot get the slot in time declines and returns. Nothing is lost
        by declining -- the swap happens after the acquire, so a declined drain
        never took the spans -- but on the signal path the process then dies and
        the tail dies with it. That is the deliberate reading of
        _SIGNAL_FLUSH_TIMEOUT's "never delay shutdown": a droppable tail is the
        price of a bounded one.

        Only the buffer lock is released early (marked below); the export lock
        is held to the end of the method.

        Errors from before_send or the export path drop the envelope
        (fail-closed) and never propagate.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        if not self._acquire_export_slot(timeout):
            if self._config.debug:
                print("[wardex] drain skipped (export in progress)", file=sys.stderr)
            return
        try:
            with self._buffer_lock:
                buf, self._buffer = self._buffer, _SpanBuffer()
                spans = buf.spans
                snapshots, self._snapshots = self._snapshots, deque()
                dropped, self._dropped = self._dropped, 0
            # -- buffer lock released here; new captures flow. The EXPORT lock
            # is still held: assembly and I/O below are serialized against every
            # other drain, which is what keeps swap order == wire order.
            if dropped and self._config.debug:
                print(f"[wardex] dropped {dropped} spans (buffer full)", file=sys.stderr)
            if not spans and not snapshots:
                self._flush_transport(deadline)
                return
            header = EnvelopeHeader(
                event_id=str(uuid.uuid4()),
                api_key=self._config.api_key or "",
                sdk=self._sdk_info,
                sent_at_ns=time.time_ns(),
            )
            envelope = InternalEnvelope(
                header=header,
                spans=tuple(spans),
                state_snapshots=tuple(snapshots),
            )
            try:
                if self._config.before_send is not None:
                    maybe = self._config.before_send(envelope)
                    if maybe is None:
                        return
                    envelope = maybe
                self._export(envelope, deadline)
            except Exception as exc:  # fail-closed: drop, never ship half-filtered data
                if self._config.debug:
                    print(f"[wardex] envelope dropped ({exc})", file=sys.stderr)
                return
            self._flush_transport(deadline)
        finally:
            self._export_lock.release()

    def _export(self, envelope: InternalEnvelope, deadline: float | None) -> None:
        """Hand the envelope to the transport with whatever budget is left.

        The client can bound how long it waits; only the transport can bound its
        own I/O. A transport that ignores `timeout` still stalls the process for
        as long as it likes -- this shrinks the blast radius to one drain, it
        does not remove it.
        """
        transport = self._transport
        if deadline is None:
            transport.export(envelope)
            return
        if transport is not self._probed_transport:
            self._probed_transport = transport
            self._export_takes_timeout = _accepts_timeout(transport.export)
        if not self._export_takes_timeout:
            transport.export(envelope)
            return
        transport.export(envelope, timeout=max(0.0, deadline - time.monotonic()))

    def _flush_transport(self, deadline: float | None) -> None:
        remaining = (
            _UNBOUNDED_TRANSPORT_FLUSH_TIMEOUT
            if deadline is None
            else max(0.0, deadline - time.monotonic())
        )
        try:
            self._transport.flush(remaining)
        except Exception as exc:  # fail-silent: never crash the app or the exit path
            if self._config.debug:
                print(f"[wardex] transport flush failed ({exc})", file=sys.stderr)

    def close(self, timeout: float = 5.0) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True  # 1. reject new captures
        # `timeout` is a per-step budget, not a total for close(). Steps 2-4 can
        # each spend it, so the worst case is roughly 3x -- but each step is now
        # bounded, where step 3 previously had no bound at all: it inherited the
        # rest of whatever POST the worker was still inside when step 2's join
        # gave up on it.
        self._worker.stop(timeout)  # 2. worker exits without draining
        self._drain(timeout)  # 3. final drain, owned by the closing thread
        self._transport.close(timeout)  # 4.
