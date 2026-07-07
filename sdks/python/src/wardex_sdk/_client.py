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


class Client:
    def __init__(self, config: WardexConfig, transport: Transport) -> None:
        self._config = config
        self._transport = transport
        self._sdk_info = build_sdk_info()
        # Lock order is always drain lock → buffer lock (one-way; no deadlock).
        # The buffer lock only ever guards an append or a swap — never I/O,
        # encoding, or callbacks (design §5).
        # _drain_lock is reentrant: the same-thread signal handler may call
        # flush() (and so re-enter _drain) while the main thread is already
        # mid-drain (manual flush() in progress, or install()'s
        # previous.close() during re-init). Each drain operates on its own
        # buffer snapshot swapped out under the buffer lock, so nested
        # reentrant drains cannot corrupt or duplicate data — a plain Lock
        # would instead hang forever on that same-thread re-acquire.
        # Cross-thread serialization (the invariant this lock exists for) is
        # unchanged: RLock still blocks other threads until fully released.
        self._buffer_lock = threading.Lock()
        self._drain_lock = threading.RLock()
        self._spans: deque[InternalSpan] = deque()
        self._snapshots: deque[InternalStateSnapshot] = deque()
        self._dropped = 0
        self._closed = False
        self._close_lock = threading.Lock()
        self._flush_threshold = max(1, config.max_buffer_spans // 4)
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
        with self._buffer_lock:
            if len(self._spans) >= self._config.max_buffer_spans:
                self._spans.popleft()  # drop-oldest: recent spans are worth more
                self._dropped += 1
            self._spans.append(span)
            should_wake = len(self._spans) >= self._flush_threshold
        if should_wake:
            self._worker.wake()

    def capture_snapshot(self, snapshot: InternalStateSnapshot) -> None:
        if self._closed:
            return
        self._worker.ensure_alive()
        with self._buffer_lock:
            if len(self._snapshots) >= self._config.max_buffer_spans:
                self._snapshots.popleft()
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
