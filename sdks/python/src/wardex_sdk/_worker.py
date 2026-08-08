"""Background batch worker — thread & timing only; knows nothing about spans.

Owns the SDK's single daemon thread. It wakes on whichever comes first:
interval elapsed, wake() (buffer size threshold), or stop(). The loop survives
drain exceptions — an observability SDK must never crash the app, and the
worker must never die (design §10).

Fork recovery (design §8, Sentry-style PID check): start() records the PID the
thread was created in; ensure_alive() lazily respawns the thread when the
recorded PID no longer matches (we are in a forked child) or the thread died.
No code runs at fork time, which sidesteps fork-safety traps entirely.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable

from .transport._base import DEFAULT_TIMEOUT


class BatchWorker:
    def __init__(
        self, drain_fn: Callable[[], None], interval: float, *, debug: bool = False
    ) -> None:
        self._drain_fn = drain_fn
        self._interval = interval
        self._debug = debug
        self._wake = threading.Event()
        self._stopped = False
        self._thread: threading.Thread | None = None
        self._thread_for_pid: int | None = None
        # RLock, and the reason is not the signal handler this SDK usually
        # documents. `capture_span` calls `ensure_alive()`, and a WebSocket span
        # can now be shipped from a `weakref.finalize` callback (the byte seams'
        # close hook) — which runs at an arbitrary allocation, on whatever
        # thread dropped the last reference, INCLUDING a thread that is already
        # inside `_spawn_locked` allocating the replacement thread. A plain Lock
        # there is a permanent self-deadlock in the host's own code.
        self._spawn_lock = threading.RLock()
        # What the RLock cannot supply on its own: reentering would otherwise
        # see a not-yet-started thread, spawn a SECOND one and leave the outer
        # frame's thread orphaned. A spawn in progress is a spawn; whoever
        # arrives during it has nothing to do.
        self._spawning = False

    def start(self) -> None:
        with self._spawn_lock:
            if self.is_alive() or self._spawning:
                return  # exactly one SDK thread — start() is idempotent
            self._spawn_locked()

    def wake(self) -> None:
        """Request an immediate drain (buffer threshold reached). Lock-free."""
        self._wake.set()

    def is_alive(self) -> bool:
        return (
            self._thread is not None
            and self._thread_for_pid == os.getpid()
            and self._thread.is_alive()
        )

    def ensure_alive(self) -> None:
        """Respawn the thread if it died or belongs to a pre-fork parent.

        Called on every capture; the happy path costs one os.getpid().
        """
        if self._stopped or self.is_alive():
            return
        with self._spawn_lock:
            if self._stopped or self.is_alive() or self._spawning:
                return  # another thread respawned it while we waited
            if self._debug:
                print("[wardex] batch worker restarted (fork or thread death)", file=sys.stderr)
            self._spawn_locked()

    def stop(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        """Signal the loop to exit and join. The final drain is the caller's job.

        The shared default, not a literal that matched it: this is a shutdown
        step like the two on `Transport`, and it was the fifth copy of that
        number -- one this batch missed by counting, and the source scan in
        `test_timeout_contract` found. `Client.close` always passes its own
        budget, so the default is reached only by callers that have no deadline
        of their own.
        """
        self._stopped = True
        self._wake.set()
        with self._spawn_lock:  # serialize with an in-flight spawn (start/ensure_alive)
            thread = self._thread
            pid = self._thread_for_pid
        if thread is not None and pid == os.getpid() and thread.is_alive():
            thread.join(timeout)

    def _spawn_locked(self) -> None:
        self._spawning = True
        try:
            thread = threading.Thread(target=self._run, daemon=True, name="wardex-batch-worker")
            self._thread = thread
            self._thread_for_pid = os.getpid()
            thread.start()
        finally:
            self._spawning = False

    def _run(self) -> None:
        while True:
            self._wake.wait(timeout=self._interval)
            self._wake.clear()
            if self._stopped:
                break  # no drain here — Client.close() owns the final drain
            try:
                self._drain_fn()
            # Never die. BaseException (SystemExit etc.) is deliberately excluded.
            except Exception as exc:
                if self._debug:
                    print(f"[wardex] background flush failed: {exc}", file=sys.stderr)
