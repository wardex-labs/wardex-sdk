from __future__ import annotations

import platform
import sys
import threading
import time
import uuid
from collections import deque

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
        # Lock order is always drain lock → buffer lock (one-way; no deadlock).
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
        self._drain_lock = threading.RLock()
        self._buffer = _SpanBuffer()
        self._snapshots: deque[InternalStateSnapshot] = deque()
        self._dropped = 0
        self._closed = False
        self._close_lock = threading.Lock()
        limits = config.limits.resolved()
        self._max_buffer_spans = limits["max_buffer_spans"]
        self._max_buffer_bytes = limits["max_buffer_bytes"]
        self._flush_threshold = max(1, self._max_buffer_spans // 4)
        self._worker = BatchWorker(
            lambda: self._drain(5.0), interval=config.flush_interval, debug=config.debug
        )
        self._worker.start()

    @property
    def config(self) -> WardexConfig:
        return self._config

    # -- test-only internal accessors -----------------------------------
    # `_spans`/`_buffered_bytes` are not part of the public API; several
    # tests read (and one, deliberately, swaps out) the resident deque to
    # exercise reentrancy edge cases. Keeping these as thin properties over
    # self._buffer preserves that surface without reintroducing a second,
    # separately-swappable piece of state.
    @property
    def _spans(self) -> deque[InternalSpan]:
        return self._buffer.spans

    @_spans.setter
    def _spans(self, value: deque[InternalSpan]) -> None:
        self._buffer.spans = value

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
            # What's NOT eliminated: a window narrower than one statement,
            # between resolving self._buffer for that trailing call and
            # _SpanBuffer.append's own first line running. A drain landing
            # exactly there still exports without our span, and the append
            # then lands in the (already-exported, now orphaned) pre-drain
            # buffer -- the span is captured by neither self._buffer nor the
            # export that just happened, and only reaches the wire at the
            # *next* drain if nothing else evicts it first. This is the same
            # class of bytecode-internal, no-second-line gap already present
            # in self._snapshots.append() below and in the pre-byte-budget
            # code; it cannot be closed further without giving up per-thread
            # signal-handler reentrancy entirely.
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

    def _drain(self, timeout: float) -> None:
        """Swap the buffer out under the lock, then assemble/export lock-free.

        Serialized by the drain lock so a manual flush() and the periodic
        worker can never interleave envelopes. Errors from before_send or the
        export path drop the envelope (fail-closed) and never propagate.
        """
        with self._drain_lock:
            with self._buffer_lock:
                buf, self._buffer = self._buffer, _SpanBuffer()
                spans = buf.spans
                snapshots, self._snapshots = self._snapshots, deque()
                dropped, self._dropped = self._dropped, 0
            # -- lock-free from here (buffer lock released; new captures flow) --
            if dropped and self._config.debug:
                print(f"[wardex] dropped {dropped} spans (buffer full)", file=sys.stderr)
            if not spans and not snapshots:
                try:
                    self._transport.flush(timeout)
                except Exception as exc:  # fail-silent: never crash the app or the exit path
                    if self._config.debug:
                        print(f"[wardex] transport flush failed ({exc})", file=sys.stderr)
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
                self._transport.export(envelope)
            except Exception as exc:  # fail-closed: drop, never ship half-filtered data
                if self._config.debug:
                    print(f"[wardex] envelope dropped ({exc})", file=sys.stderr)
                return
            try:
                self._transport.flush(timeout)
            except Exception as exc:  # fail-silent: never crash the app or the exit path
                if self._config.debug:
                    print(f"[wardex] transport flush failed ({exc})", file=sys.stderr)

    def close(self, timeout: float = 5.0) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True  # 1. reject new captures
        self._worker.stop(timeout)  # 2. worker exits without draining
        self._drain(timeout)  # 3. final drain, owned by the closing thread
        self._transport.close(timeout)  # 4.
