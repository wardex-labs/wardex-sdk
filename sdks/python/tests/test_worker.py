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
