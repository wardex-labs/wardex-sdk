"""BatchWorker — thread/timing unit tested in isolation with a fake drain_fn."""

import os
import threading
import time

from wardex_sdk._worker import BatchWorker


def _wait_for(predicate, timeout=5.0):
    """Poll a predicate with a generous upper bound — no bare-sleep asserts."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_periodic_drain_fires():
    drained = threading.Event()
    w = BatchWorker(drained.set, interval=0.05)
    w.start()
    try:
        assert drained.wait(timeout=5.0)
    finally:
        w.stop()


def test_wake_fires_immediately_without_waiting_interval():
    drained = threading.Event()
    w = BatchWorker(drained.set, interval=3600.0)  # interval alone can't fire in-test
    w.start()
    try:
        w.wake()
        assert drained.wait(timeout=5.0)
    finally:
        w.stop()


def test_stop_joins_thread():
    w = BatchWorker(lambda: None, interval=0.01)
    w.start()
    w.stop()
    assert not w.is_alive()


def test_stop_exits_without_draining():
    # Final drain is owned by Client.close() (spec §4.1) — stop() must not drain.
    calls = []
    w = BatchWorker(lambda: calls.append(1), interval=3600.0)
    w.start()
    w.stop()
    assert calls == []


def test_drain_exception_does_not_kill_worker():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")

    w = BatchWorker(flaky, interval=0.01)
    w.start()
    try:
        assert _wait_for(lambda: len(calls) >= 3)
    finally:
        w.stop()


def test_ensure_alive_restarts_after_simulated_fork():
    drained = threading.Event()
    w = BatchWorker(drained.set, interval=3600.0)
    w.start()
    first_thread = w._thread
    w._thread_for_pid = os.getpid() - 1  # simulate: thread belongs to a pre-fork parent
    assert not w.is_alive()
    w.ensure_alive()
    try:
        assert w.is_alive()
        assert w._thread is not first_thread
    finally:
        w.stop()


def test_ensure_alive_restarts_dead_thread():
    w = BatchWorker(lambda: None, interval=0.01)
    w.start()
    w.stop()
    w._stopped = False  # mimic an unexpected thread death (not a stop())
    assert not w.is_alive()
    w.ensure_alive()
    try:
        assert w.is_alive()
    finally:
        w.stop()


def test_ensure_alive_is_noop_after_stop():
    w = BatchWorker(lambda: None, interval=0.01)
    w.start()
    w.stop()
    w.ensure_alive()
    assert not w.is_alive()


def test_thread_is_daemon_and_named():
    w = BatchWorker(lambda: None, interval=3600.0)
    w.start()
    try:
        assert w._thread.daemon
        assert w._thread.name == "wardex-batch-worker"
    finally:
        w.stop()


def test_start_twice_keeps_single_thread():
    w = BatchWorker(lambda: None, interval=3600.0)
    w.start()
    first_thread = w._thread
    w.start()  # idempotent — must not spawn a second thread
    try:
        assert w._thread is first_thread
        assert sum(1 for t in threading.enumerate() if t.name == "wardex-batch-worker") == 1
    finally:
        w.stop()


def test_a_respawn_reentered_mid_allocation_neither_deadlocks_nor_doubles(monkeypatch):
    """The spawn lock has a second same-thread re-entry now, and it is not a signal.

    `capture_span` calls `ensure_alive()`, and the byte seams' close hook
    backstops itself with `weakref.finalize` — so a WebSocket span can be
    shipped from a finalizer, which CPython runs at an ARBITRARY allocation on
    whatever thread dropped the last reference. Including this thread, inside
    `_spawn_locked`, while it is allocating the replacement worker.

    Two ways for that to end badly and both are asserted: a plain `Lock` is a
    permanent self-deadlock in the host's own code, and a bare `RLock` lets the
    re-entering call spawn a SECOND worker and orphan the outer frame's.
    """
    spawned: list[threading.Thread] = []
    reentered: list[bool] = []
    w = BatchWorker(lambda: None, interval=3600.0)
    w.start()
    outer = w._thread
    real_thread = threading.Thread

    def hijacked(*args, **kwargs):
        # Stands in for the finalizer: it lands at this allocation, on this
        # thread, with `_spawn_lock` already held by the frame below.
        if not reentered:
            reentered.append(True)
            w.ensure_alive()
        t = real_thread(*args, **kwargs)
        spawned.append(t)
        return t

    try:
        w._thread_for_pid = -1  # what a fork looks like to ensure_alive
        monkeypatch.setattr(threading, "Thread", hijacked)

        finished = threading.Event()
        caller = real_thread(target=lambda: (w.ensure_alive(), finished.set()), daemon=True)
        caller.start()
        assert finished.wait(10.0), "ensure_alive self-deadlocked on its own spawn lock"

        assert reentered == [True], "precondition: the reentrant call never happened"
        assert len(spawned) == 1, "the reentrant call spawned a worker of its own"
        assert w._thread is spawned[0]
    finally:
        monkeypatch.undo()
        # NOT `w.stop()`: it takes the same spawn lock, so a regression here
        # would hang the suite instead of failing this test. The loop reads
        # both of these without one.
        w._stopped = True
        w._wake.set()
        outer.join(5.0)
        for t in spawned:
            t.join(5.0)
