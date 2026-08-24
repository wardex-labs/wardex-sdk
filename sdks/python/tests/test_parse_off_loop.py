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


# ==========================================================================
# the client — capture_deferred, flush/close/signal budgets, _settle
# ==========================================================================

from wardex_sdk import _hub  # noqa: E402
from wardex_sdk._client import Client, _configured_transport_timeout  # noqa: E402
from wardex_sdk._config import BackendConfig, BatchingConfig, WardexConfig  # noqa: E402
from wardex_sdk._enums import SpanKind, StatusCode  # noqa: E402
from wardex_sdk._types import (  # noqa: E402
    CaptureIntegrity,
    Envelope,
    InternalSpan,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport._base import Transport  # noqa: E402


class _RecordingTransport(Transport):
    def __init__(self) -> None:
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)

    @property
    def exported(self) -> list[InternalSpan]:
        return [span for envelope in self.envelopes for span in envelope.spans]


def _real_span(name: str, markers: tuple = ()) -> InternalSpan:
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name=name,
        kind=SpanKind.CLIENT,
        start_time_ns=1,
        end_time_ns=2,
        status=StatusCode.OK,
        capture_integrity=CaptureIntegrity(limitations=tuple(markers)) if markers else None,
    )


class _SpanJob:
    """A DeferredSpan double whose outputs are REAL spans, for client tests."""

    def __init__(self, name: str, *, size: int = 64, delay: float = 0.0, gate=None):
        self.name = name
        self.size = size
        self.delay = delay
        self.gate = gate
        self.ctx = contextvars.copy_context()

    def run(self):
        if self.gate is not None:
            self.gate.wait(10.0)
        if self.delay:
            time.sleep(self.delay)
        return _real_span(f"parsed-{self.name}")

    def fallback(self, marker):
        return _real_span(f"fallback-{self.name}", markers=(marker,))


def _client(transport=None, **config) -> tuple[Client, _RecordingTransport]:
    transport = transport if transport is not None else _RecordingTransport()
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
            **config,
        ),
        transport,
    )
    return c, transport


@pytest.fixture(autouse=True)
def _clean_hub():
    _hub.reset_for_test()
    yield
    _hub.reset_for_test()


def test_scope_mutation_after_capture_does_not_retag_the_deferred_span():
    """§6-4b — the tenant-attribution pin. `set_tag` mutates the Scope OBJECT
    in place and `copy_context` preserves only the binding, so a finalize-time
    re-read would stamp "the dict as it is now": one queued request later,
    tenant B's tags on tenant A's span. The stamp must come from the snapshot
    `capture_deferred` took at submit time."""
    from wardex_sdk import set_tag

    c, t = _client()
    try:
        c._finalize.stop(1.0)  # freeze the worker: the job stays pending
        set_tag("k", "1")
        c.capture_deferred(_SpanJob("tagged"))
        set_tag("k", "2")  # mutates the SAME Scope object the ctx binds
        c._settle()
        (span,) = c._spans
        assert ("k", "1") in span.extra
        assert ("k", "2") not in span.extra
    finally:
        c.close()


def test_set_user_after_capture_does_not_reattribute_the_deferred_span():
    from wardex_sdk import set_user
    from wardex_sdk._scope import UserInfo

    c, t = _client()
    try:
        c._finalize.stop(1.0)
        set_user(UserInfo(id="tenant-a"))
        c.capture_deferred(_SpanJob("owned"))
        set_user(UserInfo(id="tenant-b"))
        c._settle()
        (span,) = c._spans
        assert ("user.id", "tenant-a") in span.extra
    finally:
        c.close()


def test_deferred_span_is_visible_after_flush():
    """§6-2 at client level: the user contract is 'flush and it is all
    visible' — a pending parse is finished BEFORE the export."""
    c, t = _client()
    try:
        c._finalize.stop(1.0)  # nothing finalizes until the flush does it
        c.capture_deferred(_SpanJob("flushed"))
        assert t.exported == []
        c.flush()
        assert [s.name for s in t.exported] == ["parsed-flushed"]
    finally:
        c.close()


def test_close_finalizes_pending_before_the_final_drain():
    """§6-9: close() step 4 parses what is pending; step 5 exports it; and
    the finalize thread is gone afterwards."""
    c, t = _client()
    c._finalize.stop(1.0)
    for i in range(3):
        c.capture_deferred(_SpanJob(f"j{i}"))
    c.close()
    assert sorted(s.name for s in t.exported) == ["parsed-j0", "parsed-j1", "parsed-j2"]
    assert not any(
        th.name == "wardex-finalize-worker" and th.is_alive() for th in threading.enumerate()
    )


def test_close_budget_exhaustion_ships_fallback_with_marker_47():
    """§6-10: when the budget ends mid-backlog, the rest still ships — as
    PARSE_SKIPPED_AT_SHUTDOWN fallbacks, not as losses."""
    c, t = _client()
    c._finalize.stop(1.0)
    for i in range(5):
        c.capture_deferred(_SpanJob(f"j{i}", delay=0.2))
    c.close(0.1)
    assert len(t.exported) == 5, "no pending job may be lost at close"
    skipped = [
        s
        for s in t.exported
        if s.capture_integrity is not None
        and Limitation.PARSE_SKIPPED_AT_SHUTDOWN in s.capture_integrity.limitations
    ]
    assert skipped, "at least one job must have shipped as a shutdown fallback"
    assert c._lost == 0
    assert counters.get("client.finalize.shutdown_fallback") == len(skipped)


def test_an_in_flight_parse_that_outlives_close_is_reported_not_stranded(capsys):
    """§6-10b: not a race — a parse that outlives roughly twice the close
    budget opens this window deterministically. The late span must not sit
    silently in a buffer nothing will drain; it is counted on `_lost` and
    said once."""
    gate = threading.Event()
    c, t = _client()
    c.capture_deferred(_SpanJob("late", gate=gate))  # spawns the worker
    deadline = time.monotonic() + 5.0
    while c._finalize.pending() > 0 and time.monotonic() < deadline:
        time.sleep(0.001)  # wait until the worker holds the job in flight
    capsys.readouterr()
    c.close(0.05)
    assert c._lost == 0  # nothing lost yet — the job is still in flight
    gate.set()  # the parse finishes AFTER close's final drain
    deadline = time.monotonic() + 5.0
    while c._lost == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert c._lost == 1
    err = capsys.readouterr().err
    assert "a deferred parse finished after close()" in err
    assert all(s.name != "parsed-late" for s in t.exported)


def test_capture_deferred_after_close_is_counted_not_silent():
    c, t = _client()
    c.close()
    c.capture_deferred(_SpanJob("rejected"))
    assert counters.get("client.finalize.rejected_closed") == 1


def test_flush_keeps_leftover_when_budget_ends():
    """§6-11: an explicit flush(t) is a wall-clock promise — what the half
    budget could not parse STAYS QUEUED (KEEP) for the worker or the next
    flush, and a later settle finishes it."""
    c, t = _client()
    try:
        c._finalize.stop(1.0)
        for i in range(3):
            c.capture_deferred(_SpanJob(f"slow{i}", delay=0.2))
        c.flush(0.2)
        assert c._finalize.pending() > 0, "KEEP must leave the unfinished tail queued"
        c._settle()
        assert c._finalize.pending() == 0
        c.flush()
        assert len(t.exported) == 3
    finally:
        c.close()


def test_a_bare_flush_still_hands_the_transport_its_own_number(monkeypatch):
    """§6-11b: the bare-flush contract (`test_flush_budget.py`) survives a
    full backlog — the POST budget is the transport's own number, UNMODIFIED
    by however long the parse phase spent."""
    c, t = _client()
    seen: list[float] = []
    real_drain = c._drain

    def spying_drain(timeout, **kw):
        seen.append(timeout)
        return real_drain(timeout, **kw)

    monkeypatch.setattr(c, "_drain", spying_drain)
    try:
        c._finalize.stop(1.0)
        c.capture_deferred(_SpanJob("busy", delay=0.1))
        c.flush()
        assert seen == [_configured_transport_timeout(t)]
    finally:
        c.close()


def test_an_explicit_flush_reserves_half_for_the_export(monkeypatch):
    """I-F11's flush arm: however long the parse phase ran, the export
    receives at least t/2."""
    c, t = _client()
    seen: list[float] = []
    real_drain = c._drain

    def spying_drain(timeout, **kw):
        seen.append(timeout)
        return real_drain(timeout, **kw)

    monkeypatch.setattr(c, "_drain", spying_drain)
    try:
        c._finalize.stop(1.0)
        for i in range(4):
            c.capture_deferred(_SpanJob(f"s{i}", delay=0.08))
        c.flush(0.2)
        assert len(seen) == 1
        assert seen[0] >= 0.1  # the floor: t/2
    finally:
        c.close()


def test_shutdown_flush_empties_the_queue_and_reserves_half_for_export(monkeypatch):
    """§3.11's signal row: FALLBACK (the process is ending), export floor
    t/2 — the buffered spans must still go out even with a hostile backlog."""
    c, t = _client()
    seen: list[float] = []
    real_drain = c._drain

    def spying_drain(timeout, **kw):
        seen.append(timeout)
        return real_drain(timeout, **kw)

    monkeypatch.setattr(c, "_drain", spying_drain)
    try:
        c._finalize.stop(1.0)
        c.capture_span(_real_span("buffered-before-signal"))
        for i in range(3):
            c.capture_deferred(_SpanJob(f"p{i}", delay=0.6))
        c._shutdown_flush(2.0)
        names = [s.name for s in t.exported]
        assert "buffered-before-signal" in names
        assert len(names) == 4, "every pending job leaves — parsed or as a 47 fallback"
        assert c._finalize.pending() == 0
        assert seen and seen[0] >= 1.0
    finally:
        c.close()


def test_no_finalize_thread_after_close():
    """§6-17, the `test_uninstall_isolation` pattern."""
    c, t = _client()
    c.capture_deferred(_SpanJob("spawn"))
    c._settle()
    c.close()
    assert not any(
        th.name == "wardex-finalize-worker" and th.is_alive() for th in threading.enumerate()
    )


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
@pytest.mark.filterwarnings(
    "ignore:This process.*is multi-threaded, use of fork:DeprecationWarning"
)
def test_fork_child_respawns_finalize_worker():
    """§6-18: the PID backstop respawns the finalize worker in a child even
    where no fork hook ran (this is a bare Client, not an init())."""
    c, t = _client()
    c.capture_deferred(_SpanJob("warm"))  # parent spawns + finishes one job
    c._settle()
    pid = os.fork()
    if pid == 0:
        try:
            c.capture_deferred(_SpanJob("child"))
            c._settle()
            ok = any(s.name == "parsed-child" for s in c._spans)
            os._exit(0 if ok else 1)
        except BaseException:
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    c.close()
    assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX-only")
@pytest.mark.filterwarnings(
    "ignore:This process.*is multi-threaded, use of fork:DeprecationWarning"
)
def test_fork_mid_parse_does_not_stall_the_childs_flush():
    """§6-18's fork-mid-parse variant: the child inherits in_flight=1 from a
    worker blocked mid-run in the PARENT; the PID reset keeps the child's
    flush from burning its whole budget waiting on it."""
    gate = threading.Event()
    c, t = _client()
    c.capture_deferred(_SpanJob("blocked", gate=gate))
    deadline = time.monotonic() + 5.0
    while c._finalize.pending() > 0 and time.monotonic() < deadline:
        time.sleep(0.001)  # the worker holds the job in flight now
    pid = os.fork()
    if pid == 0:
        try:
            started = time.monotonic()
            c.flush(5.0)
            elapsed = time.monotonic() - started
            os._exit(0 if elapsed < 2.0 else 1)
        except BaseException:
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    gate.set()
    c.close()
    assert os.waitstatus_to_exitcode(status) == 0


# ==========================================================================
# the public harness — the deferred path on the conformance surface (S2-1)
# ==========================================================================


def test_recording_client_defers_until_settle():
    """The harness double runs a REAL FinalizeQueue: a submitted job is
    sealed but NOT visible until `settle()` — the double's spelling of the
    real client's "visible after flush" contract. An inline double here
    would mean the public conformance surface never drives the deferred
    path at all."""
    from wardex_sdk.testing import RecordingClient

    rc = RecordingClient()
    rc.capture_deferred(_StubJob("harness"))
    assert rc.spans == [], "a deferred job must not be visible before settle()"
    rc.settle()
    assert rc.spans == [("parsed", "harness")]


def test_recording_client_enforces_the_backlog_bounds():
    from wardex_sdk.testing import RecordingClient

    rc = RecordingClient(limits=LimitsConfig(max_parse_backlog=1))
    rc.capture_deferred(_StubJob("first"))
    rc.capture_deferred(_StubJob("second"))
    assert rc.spans == [("fallback", "first", Limitation.PARSE_BACKLOG_FULL)]
    rc.settle()
    assert ("parsed", "second") in rc.spans
