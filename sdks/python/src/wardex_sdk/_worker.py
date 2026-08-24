"""Background batch worker — thread & timing only; knows nothing about spans.

Owns the SDK's single daemon thread. It wakes on whichever comes first:
interval elapsed, wake() (buffer size threshold), or stop(). The loop survives
drain exceptions — an observability SDK must never crash the app, and the
worker must never die (design §10).

Fork posture, two layers. The PRIMARY path is the SDK's
`os.register_at_fork(after_in_child=...)` hook: it calls `_at_fork_reinit()`
in the child, which replaces this worker's lock and Event (an inherited lock
can arrive held by a thread that did not cross the fork) and nulls the thread
slots. Respawn stays LAZY — no thread is started at fork time, so a child
that never captures (the fork+exec shell-out, the short-lived mp worker)
never pays for one; the next capture's `ensure_alive()` brings it back. The
BACKSTOP is the PID check: `start()` records the PID the thread was created
in, and `ensure_alive()` respawns when it no longer matches or the thread
died — which self-heals even where the hook never ran (uWSGI runs only the
child hook; a hypothetical embedding might run none).
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

from ._assembly import diag_warning
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
        # close hook) — which CPython runs out of its referent's deallocation,
        # so wherever a reference count reaches zero and on whatever thread
        # dropped it, INCLUDING a thread that is already inside `_spawn_locked`.
        # The `Thread(...)` construction there is one such site twice over: it
        # allocates, so a cyclic collection can start in it, and it drops the
        # previous thread object when `self._thread` is overwritten. A plain
        # Lock there is a permanent self-deadlock in the host's own code.
        self._spawn_lock = threading.RLock()
        # What the RLock cannot supply on its own: reentering would otherwise
        # see a not-yet-started thread, spawn a SECOND one and leave the outer
        # frame's thread orphaned. A spawn in progress is a spawn; whoever
        # arrives during it has nothing to do. Stored as the PID that owns the
        # spawn rather than as a bool — see `_spawn_in_flight`.
        self._spawning_pid: int | None = None

    def start(self) -> None:
        with self._spawn_lock:
            if self.is_alive() or self._spawn_in_flight():
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

    def _spawn_in_flight(self) -> bool:
        """Is a spawn in progress IN THIS process?

        A PID rather than a bool, and the two differ only after a fork.
        `_spawn_locked` clears the marker in a `finally`, and that frame does
        not exist in the child: fork while another thread is anywhere inside the
        spawn — `Thread(...)` and `start()` allocate heavily, so the window is
        the whole body, not a gap between two statements — and a boolean arrives
        in the child already set with nothing left to clear it. Both `start()`
        and `ensure_alive()` would then decline forever, so the child would
        never respawn its worker and would never recover: no periodic drain for
        the process lifetime, spans accumulating until the buffer cap evicts
        them, and nothing shipped short of an explicit `close()`. Comparing PIDs
        makes an inherited value mean "a spawn in the parent", which is not one
        here. Only ever asked on the locked slow path, so the extra `getpid()`
        never lands on a capture.
        """
        return self._spawning_pid == os.getpid()

    def ensure_alive(self) -> None:
        """Respawn the thread if it died or belongs to a pre-fork parent.

        Called on every capture; the happy path costs one os.getpid().
        """
        if self._stopped or self.is_alive():
            return
        with self._spawn_lock:
            if self._stopped or self.is_alive() or self._spawn_in_flight():
                return  # another thread respawned it while we waited
            if self._debug:
                diag_warning("batch worker restarted (fork or thread death)")
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

    def _at_fork_reinit(self) -> None:
        """Fork-child reset: fresh lock and Event, no thread, nothing spawned.

        Called from the client's own `_at_fork_reinit` on the
        `os.register_at_fork(after_in_child=...)` path. REPLACEMENT, never
        acquisition: the parent may have been inside `_spawn_locked` — which
        holds `_spawn_lock` across a `Thread(...)` construction and a
        `start()` — at the fork instant, so the inherited lock can be
        permanently held by a thread that does not exist in the child.

        The thread slots are nulled rather than left for `is_alive()`'s PID
        check to age out, because the check is the BACKSTOP (uWSGI's C-level
        fork runs `after_in_child` but a hypothetical embedding might not) and
        this is the primary path. Respawn stays LAZY: no thread is started
        here — a child that never captures (the fork+exec shell-out, the mp
        worker that dies young) never pays for one, and `ensure_alive()` on
        the first capture is the tested path that brings the worker back.
        `_stopped` is inherited as-is: a closed parent's child stays closed.
        """
        self._spawn_lock = threading.RLock()
        self._wake = threading.Event()
        self._thread = None
        self._thread_for_pid = None
        self._spawning_pid = None

    def _spawn_locked(self) -> None:
        if self._stopped:
            return
        self._spawning_pid = os.getpid()
        try:
            thread = threading.Thread(target=self._run, daemon=True, name="wardex-batch-worker")
            # `_stopped` again, and this is the check that earns its place. The
            # RLock made a sequence reachable that a plain Lock used to deadlock
            # on: a finalizer landing in the allocation above and reaching
            # `stop()` on this thread now gets through — `stop()` sets
            # `_stopped`, re-enters this lock, reads a `_thread`/`_thread_for_pid`
            # pair this frame has not published yet, finds nothing to join and
            # returns. Starting the thread afterwards would mean `stop()` had
            # returned while a worker was about to begin, with nothing left that
            # could ever join it, and `Client.close()` inherits that contract.
            # Dropping the unstarted thread instead costs one wasted allocation
            # on a path that is already shutting down.
            if self._stopped:
                return
            self._thread = thread
            self._thread_for_pid = os.getpid()
            thread.start()
        finally:
            self._spawning_pid = None

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
                    diag_warning(f"background flush failed: {exc}")
