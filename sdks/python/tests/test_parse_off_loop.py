"""Parse-off-loop: the FinalizeQueue, the client's deferred path, and the seam.

The deferred-parse design in one file, layer by layer. The queue section
drives `FinalizeQueue` directly with stub jobs (no seam, no client): the
bounds, the fallback vocabulary, the drain semantics and the fork posture are
all queue properties, and testing them here keeps the failure surface one
class wide.
"""

from __future__ import annotations

import contextvars
import os
import threading
import time

import pytest

from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._finalize import FinalizeQueue, Leftover
from wardex_sdk._limits import LimitsConfig, LimitsConsumer, limits_kwargs

# ==========================================================================
# stub jobs
# ==========================================================================


class _StubJob:
    """A DeferredSpan double whose 'span' is a plain tuple.

    `run` and `fallback` return distinguishable values so a test can tell a
    parsed finalization from a marker fallback by looking at what was
    admitted.
    """

    def __init__(self, name: str = "job", size: int = 64, run_fn=None):
        self.name = name
        self.size = size
        self.ctx = contextvars.copy_context()
        self._run_fn = run_fn
        self.ran_on: list[int] = []

    def run(self):
        self.ran_on.append(threading.get_ident())
        if self._run_fn is not None:
            return self._run_fn()
        return ("parsed", self.name)

    def fallback(self, marker):
        return ("fallback", self.name, marker)


class _Admitted:
    def __init__(self) -> None:
        self.spans: list[tuple] = []
        self.scopes: list[tuple] = []

    def __call__(self, span, *, scope):
        self.spans.append(span)
        self.scopes.append(scope)


def _queue(admitted: _Admitted, **overrides) -> FinalizeQueue:
    """A queue built exactly the way `Client.__init__` builds one."""
    resolved = LimitsConfig(**overrides).resolved()
    return FinalizeQueue(
        admit=admitted,
        debug=False,
        **limits_kwargs(LimitsConsumer.FINALIZE_QUEUE, resolved),
    )


@pytest.fixture(autouse=True)
def _clean_diag():
    counters.reset()
    reset_reports_for_test()
    yield
    counters.reset()
    reset_reports_for_test()


# ==========================================================================
# bounds — §6-6 / §6-7 / §6-7b
# ==========================================================================


def test_backlog_full_evicts_oldest_with_parse_backlog_full(capsys):
    """Job-count bound: the OLDEST pending job ships as a marker fallback,
    and the loss is counted AND reported at debug=False — overload loss is
    the one line an operator has to be able to see."""
    admitted = _Admitted()
    queue = _queue(admitted, max_parse_backlog=2)
    # Worker deliberately never spawned: submits only queue or evict.
    jobs = [_StubJob(f"j{i}") for i in range(3)]
    capsys.readouterr()
    for job in jobs:
        queue.submit(job, ({"k": "v"}, None))

    assert admitted.spans == [("fallback", "j0", Limitation.PARSE_BACKLOG_FULL)]
    assert admitted.scopes == [({"k": "v"}, None)]
    assert queue.pending() == 2
    assert counters.get("client.finalize.backlog_evicted") == 1
    err = capsys.readouterr().err
    assert "deferred-parse backlog full" in err
    assert "max_parse_backlog" in err


def test_backlog_bytes_bound_evicts(capsys):
    """Byte bound, independently of the count bound."""
    admitted = _Admitted()
    queue = _queue(admitted, max_parse_backlog_bytes=150)
    queue.submit(_StubJob("a", size=100), ({}, None))
    queue.submit(_StubJob("b", size=100), ({}, None))

    assert admitted.spans == [("fallback", "a", Limitation.PARSE_BACKLOG_FULL)]
    assert queue.pending() == 1
    assert queue.pending_bytes() == 100


def test_an_oversize_job_never_enters_the_queue():
    """§6-7b: a job over the byte bound could not fit in an EMPTY queue, so
    it never enters at all — the bound stays literal, never 'plus one
    oversize job' — and it still ships, as the same marker fallback."""
    admitted = _Admitted()
    queue = _queue(admitted, max_parse_backlog_bytes=128)
    queue.submit(_StubJob("small"), ({}, None))
    queue.submit(_StubJob("huge", size=1024), ({}, None))

    assert ("fallback", "huge", Limitation.PARSE_BACKLOG_FULL) in admitted.spans
    assert queue.pending() == 1  # only the small job; the oversize never queued
    assert queue.pending_bytes() == 64
    assert counters.get("client.finalize.backlog_evicted") == 1


# ==========================================================================
# the worker — §6-13 / §6-16
# ==========================================================================


def test_submit_finalizes_without_waiting_for_the_idle_sweep(monkeypatch):
    """The queue is EVENT-DRIVEN: `submit` wakes the worker, and the idle
    interval is only how often an idle thread stirs. Pin it by making the
    interval an hour and requiring millisecond finalization."""
    from wardex_sdk import _finalize as finalize_mod

    monkeypatch.setattr(finalize_mod, "_IDLE_SWEEP", 3600.0)
    admitted = _Admitted()
    queue = _queue(admitted)
    try:
        queue.ensure_alive()
        queue.submit(_StubJob("fast"), ({}, None))
        deadline = time.monotonic() + 5.0
        while not admitted.spans and time.monotonic() < deadline:
            time.sleep(0.001)
        assert admitted.spans == [("parsed", "fast")]
        assert queue.pending() == 0
    finally:
        queue.stop(5.0)


def test_finalize_worker_survives_a_raising_job():
    """§6-13/§6-16: a job whose `run` raises is counted under the guard and
    the NEXT job still finalizes — the worker never dies."""

    def boom():
        raise RuntimeError("assembly failed")

    admitted = _Admitted()
    queue = _queue(admitted)
    try:
        queue.ensure_alive()
        queue.submit(_StubJob("bad", run_fn=boom), ({}, None))
        queue.submit(_StubJob("good"), ({}, None))
        deadline = time.monotonic() + 5.0
        while not admitted.spans and time.monotonic() < deadline:
            time.sleep(0.001)
        assert admitted.spans == [("parsed", "good")]
        assert counters.get("client.finalize.run") == 1
    finally:
        queue.stop(5.0)


def test_jobs_run_on_the_worker_thread_not_the_submitter():
    admitted = _Admitted()
    queue = _queue(admitted)
    try:
        queue.ensure_alive()
        job = _StubJob("threaded")
        queue.submit(job, ({}, None))
        deadline = time.monotonic() + 5.0
        while not job.ran_on and time.monotonic() < deadline:
            time.sleep(0.001)
        assert job.ran_on and job.ran_on[0] != threading.get_ident()
    finally:
        queue.stop(5.0)


# ==========================================================================
# drain_all — the shutdown/flush arms, at queue level
# ==========================================================================


def test_drain_all_keep_finishes_pending_on_the_calling_thread():
    admitted = _Admitted()
    queue = _queue(admitted)
    for i in range(3):
        queue.submit(_StubJob(f"j{i}"), ({}, None))
    finished = queue.drain_all(None, leftover=Leftover.KEEP)
    assert finished == 3
    assert admitted.spans == [("parsed", "j0"), ("parsed", "j1"), ("parsed", "j2")]
    assert queue.pending() == 0 and queue.pending_bytes() == 0


def test_drain_all_keep_leaves_the_remainder_when_the_budget_ends():
    admitted = _Admitted()
    queue = _queue(admitted)
    for i in range(3):
        queue.submit(_StubJob(f"j{i}"), ({}, None))
    finished = queue.drain_all(time.monotonic() - 1.0, leftover=Leftover.KEEP)
    assert finished == 0
    assert queue.pending() == 3
    assert admitted.spans == []


def test_drain_all_fallback_always_empties_the_queue_with_marker_47():
    """FALLBACK is the shutdown arm: the process is ending, so a kept job
    would die with it. Every leftover ships as PARSE_SKIPPED_AT_SHUTDOWN and
    the queue is empty on return — whatever the deadline said."""
    admitted = _Admitted()
    queue = _queue(admitted)
    for i in range(3):
        queue.submit(_StubJob(f"j{i}"), ({}, None))
    finished = queue.drain_all(time.monotonic() - 1.0, leftover=Leftover.FALLBACK)
    assert finished == 3
    assert admitted.spans == [
        ("fallback", "j0", Limitation.PARSE_SKIPPED_AT_SHUTDOWN),
        ("fallback", "j1", Limitation.PARSE_SKIPPED_AT_SHUTDOWN),
        ("fallback", "j2", Limitation.PARSE_SKIPPED_AT_SHUTDOWN),
    ]
    assert queue.pending() == 0 and queue.pending_bytes() == 0
    assert counters.get("client.finalize.shutdown_fallback") == 3


def test_drain_all_works_after_stop():
    """`stop()` stops the worker LOOP; draining what remains is the caller's
    job — which is exactly close()'s step order."""
    admitted = _Admitted()
    queue = _queue(admitted)
    queue.submit(_StubJob("tail"), ({}, None))
    queue.stop(1.0)
    finished = queue.drain_all(None, leftover=Leftover.FALLBACK)
    assert finished == 1
    assert admitted.spans == [("parsed", "tail")]


# ==========================================================================
# fork posture — §6-18's queue-level halves
# ==========================================================================


def test_at_fork_reinit_discards_inherited_jobs_without_emitting():
    """I-fork-3 / integration S-4: the parent sealed those jobs, the parent
    exports them; the child discards them unemitted, so one transaction can
    never ship twice."""
    admitted = _Admitted()
    queue = _queue(admitted)
    queue.submit(_StubJob("parent-1"), ({}, None))
    queue.submit(_StubJob("parent-2"), ({}, None))

    queue._at_fork_reinit()

    assert queue.pending() == 0 and queue.pending_bytes() == 0
    assert admitted.spans == []
    # And the replaced lock/CV are usable: a fresh submit still works.
    queue.submit(_StubJob("child"), ({}, None))
    assert queue.pending() == 1


def test_an_inherited_in_flight_count_does_not_stall_the_child():
    """I-F12: `in_flight` is PID-owned. A child forked while the parent's
    worker was mid-parse inherits in_flight=1 with no thread to decrement
    it; without the PID reset every drain in the child would burn its whole
    timeout waiting (measured as a deterministic multi-second close)."""
    admitted = _Admitted()
    queue = _queue(admitted)
    queue._in_flight = 1
    queue._in_flight_pid = os.getpid() + 1  # "some other process"

    started = time.monotonic()
    queue.drain_all(time.monotonic() + 5.0, leftover=Leftover.KEEP)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"the child waited {elapsed:.2f}s on a parent's parse"
    assert queue._in_flight == 0


def test_ensure_alive_resets_a_foreign_pids_in_flight():
    admitted = _Admitted()
    queue = _queue(admitted)
    queue._in_flight = 2
    queue._in_flight_pid = os.getpid() + 1
    try:
        queue.ensure_alive()
        assert queue._in_flight == 0
        assert queue._in_flight_pid is None
    finally:
        queue.stop(5.0)


# ==========================================================================
# scope snapshots travel with the job
# ==========================================================================


def test_the_admit_scope_is_the_submit_time_snapshot():
    """The queue hands back at admit exactly the `(tags, user)` snapshot it
    was given at submit — it never re-reads scope state on its own thread."""
    admitted = _Admitted()
    queue = _queue(admitted)
    queue.submit(_StubJob("a"), ({"tenant": "one"}, None))
    queue.submit(_StubJob("b"), ({"tenant": "two"}, None))
    queue.drain_all(None, leftover=Leftover.KEEP)
    assert admitted.scopes == [({"tenant": "one"}, None), ({"tenant": "two"}, None)]


def test_run_executes_inside_the_sealed_context():
    """`ctx` carries the immutable-fact ContextVars: what `run` reads is the
    seal-time binding, not the drain-time one."""
    var: contextvars.ContextVar[str] = contextvars.ContextVar("probe", default="unset")

    admitted = _Admitted()
    queue = _queue(admitted)

    class _CtxJob(_StubJob):
        def run(self):
            return ("saw", var.get())

    var.set("at-seal")
    job = _CtxJob("ctx")
    job.ctx = contextvars.copy_context()
    var.set("at-drain")
    queue.submit(job, ({}, None))
    queue.drain_all(None, leftover=Leftover.KEEP)
    assert admitted.spans == [("saw", "at-seal")]
