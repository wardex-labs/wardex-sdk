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
        # guarded block re-reads self._spans/self._snapshots fresh each time, so
        # nested reentrant acquisition cannot corrupt or duplicate state — a
        # plain Lock would instead hang forever on that same-thread re-acquire.
        # Cross-thread serialization (the invariant these locks exist for) is
        # unchanged: RLock still blocks other threads until fully released.
        self._buffer_lock = threading.RLock()
        self._drain_lock = threading.RLock()
        self._spans: deque[InternalSpan] = deque()
        self._snapshots: deque[InternalStateSnapshot] = deque()
        self._dropped = 0
        self._closed = False
        self._close_lock = threading.Lock()
        limits = config.limits.resolved()
        self._max_buffer_spans = limits["max_buffer_spans"]
        self._max_buffer_bytes = limits["max_buffer_bytes"]
        self._buffered_bytes = 0
        self._flush_threshold = max(1, self._max_buffer_spans // 4)
        self._worker = BatchWorker(
            lambda: self._drain(5.0), interval=config.flush_interval, debug=config.debug
        )
        self._worker.start()

    @property
    def config(self) -> WardexConfig:
        return self._config

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
            # comment on the lock). _drain() swaps self._spans for a fresh
            # deque and resets self._buffered_bytes to 0. To stay correct
            # across that swap:
            #   - the walrus below re-reads self._spans into `spans` on
            #     every loop condition check, and the loop body always pops
            #     from that same `spans` local -- never a separately re-read
            #     self._spans -- so a drain can never swap in an empty deque
            #     between "checked non-empty" and "popped" (which would
            #     otherwise raise IndexError on the empty deque);
            #   - every counter update (_buffered_bytes, _dropped) is gated
            #     on `self._spans is spans` -- if a drain interleaved, the
            #     item we just popped belongs to a deque that's already been
            #     handed off (exported), so its delta no longer applies to
            #     the fresh buffer and is skipped rather than corrupting the
            #     reset total.
            #   - the final append always targets self._spans fresh (never
            #     the loop-cached `spans` local) so the span itself is never
            #     lost to an orphaned deque -- only the *counter* update is
            #     gated on identity. If a drain fires before the append, the
            #     span lands in the fresh deque and survives, but the
            #     identity check (comparing against the pre-append `spans`)
            #     correctly sees a mismatch and skips the increment: the
            #     counter understates by one span's size until the next
            #     drain resets it -- bounded and self-healing, never data
            #     loss. If a drain fires after the append but before the
            #     check, the span was already captured in the exported
            #     batch, and skipping the increment is exactly correct (the
            #     fresh buffer doesn't contain it, so 0 is exact). The
            #     counter can therefore only ever understate, never overstate
            #     or go negative, and the span is never dropped silently.
            while (spans := self._spans) and (
                len(spans) >= self._max_buffer_spans
                or self._buffered_bytes + size > self._max_buffer_bytes
            ):
                evicted = spans.popleft()
                evicted_size = _span_size(evicted)
                if self._spans is spans:
                    self._buffered_bytes -= evicted_size
                    self._dropped += 1
            self._spans.append(span)
            if self._spans is spans:
                self._buffered_bytes += size
            should_wake = len(self._spans) >= self._flush_threshold
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
                spans, self._spans = self._spans, deque()
                snapshots, self._snapshots = self._snapshots, deque()
                dropped, self._dropped = self._dropped, 0
                self._buffered_bytes = 0
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
