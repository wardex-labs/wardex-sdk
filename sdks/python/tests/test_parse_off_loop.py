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


# ==========================================================================
# the seam — seal on the caller thread, assemble on the worker
# ==========================================================================

import gc  # noqa: E402
import json  # noqa: E402
import weakref  # noqa: E402

from conftest import _FakeSSLSocket  # noqa: E402
from wardex_sdk._assembly import degraded_run  # noqa: E402
from wardex_sdk._config import LimitsConfig as _LimitsConfigAlias  # noqa: E402, F401
from wardex_sdk._enums import CaptureMode  # noqa: E402
from wardex_sdk._interceptors import _seam as seam_mod  # noqa: E402
from wardex_sdk._interceptors._ssl import SSLInterceptor  # noqa: E402

_CHAT_REQ = json.dumps(
    {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
).encode()
_CHAT_RESP = json.dumps(
    {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }
).encode()


def _request_bytes(body: bytes = _CHAT_REQ, path: str = "/v1/chat/completions") -> bytes:
    return (
        f"POST {path} HTTP/1.1\r\nHost: api.openai.com\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode() + body


def _response_bytes(body: bytes = _CHAT_RESP) -> bytes:
    return (
        f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    ).encode() + body


def _seam_client(
    mode: CaptureMode = CaptureMode.ALL, **config
) -> tuple[Client, _RecordingTransport]:
    return _client(capture_mode=mode, **config)


def _live_seam(client: Client) -> SSLInterceptor:
    itc = SSLInterceptor()
    itc._client = client
    itc._load_limits(client)
    return itc


def _sock(host: str = "api.openai.com") -> _FakeSSLSocket:
    sock = _FakeSSLSocket(None)
    sock.server_hostname = host
    return sock


def _exchange(seam: SSLInterceptor, sock, req: bytes | None = None, resp: bytes | None = None):
    seam._on_request_bytes(sock, req if req is not None else _request_bytes())
    seam._on_response_bytes(sock, resp if resp is not None else _response_bytes())


def test_parse_never_runs_on_the_byte_feeding_thread(monkeypatch):
    """§6-1b — I-F4's primary gate, and it is a THREAD-IDENTITY check, not a
    clock: `parse_llm_semantics` must never run on a thread that is feeding
    bytes into the seam, whatever host or protocol that thread serves. An
    explicit flush()/settle() MAY parse on its calling thread (documented),
    so the assertion is scoped to the feed instants."""
    idents: list[int] = []
    real = seam_mod.parse_llm_semantics

    def spy(*args, **kwargs):
        idents.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(seam_mod, "parse_llm_semantics", spy)

    c, t = _seam_client()
    feeders: set[int] = set()
    try:
        seam = _live_seam(c)
        # (a) this thread feeds a TLS HTTP/1 exchange
        feeders.add(threading.get_ident())
        _exchange(seam, _sock())

        # (b) a second thread feeds one — the "event loop thread" shape
        def feed():
            feeders.add(threading.get_ident())
            _exchange(seam, _sock())

        loop_alike = threading.Thread(target=feed, name="loop-alike")
        loop_alike.start()
        loop_alike.join(5.0)
        # (c) the eviction path: the parse-less fallback runs on the feeder,
        # and must not parse there either.
        c2, _t2 = _seam_client(limits=LimitsConfig(max_parse_backlog=1))
        c2._finalize.stop(1.0)
        seam2 = _live_seam(c2)
        feeders.add(threading.get_ident())
        _exchange(seam2, _sock())
        _exchange(seam2, _sock())  # evicts the first as a fallback, right here

        deadline = time.monotonic() + 5.0
        while c._finalize.pending() > 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        settled = threading.Thread(target=c2._settle, name="settler")
        settled.start()
        settled.join(5.0)

        assert idents, "the spy never saw a parse — the fixture went vacuous"
        assert not (set(idents) & feeders), (
            "parse_llm_semantics ran on a byte-feeding thread — the inline path is back"
        )
        c2.close()
    finally:
        c.close()


def test_deferred_span_is_visible_after_flush_seam():
    """§6-2 — the user contract, driven through the real seam."""
    c, t = _seam_client()
    try:
        c._finalize.stop(1.0)  # freeze: only the flush may finalize
        seam = _live_seam(c)
        _exchange(seam, _sock())
        assert t.exported == []
        c.flush()
        (span,) = [s for s in t.exported if s.kind.name == "CLIENT"]
        assert span.gen_ai is not None
        assert span.gen_ai.output_tokens == 5
    finally:
        c.close()


class _JobTrap:
    """A client double that traps the sealed job instead of queueing it."""

    def __init__(self, mode: CaptureMode = CaptureMode.ALL) -> None:
        from wardex_sdk._config import BackendConfig, WardexConfig

        self.config = WardexConfig(capture_mode=mode, backend=BackendConfig(api_key="k"))
        self.jobs: list = []
        self.spans: list = []

    def capture_span(self, span) -> None:
        self.spans.append(span)

    def capture_deferred(self, job) -> None:
        self.jobs.append(job)


def _strip_ids(span):
    from dataclasses import replace

    from wardex_sdk._types import SpanContext, SpanId, TraceId

    zero = SpanContext(trace_id=TraceId(b"\x00" * 16), span_id=SpanId(b"\x00" * 8))
    return replace(span, context=zero, parent_span_id=None)


def test_deferred_assembly_equals_inline_assembly():
    """§6-3 golden — the same sealed job finalizes to the same span on the
    sealing thread and on any other thread (I-F10), ids aside."""
    trap = _JobTrap()
    seam = _live_seam(trap)
    _exchange(seam, _sock())
    (job,) = trap.jobs

    inline = job.ctx.run(job.run)
    box: list = []
    other = threading.Thread(target=lambda: box.append(job.ctx.run(job.run)))
    other.start()
    other.join(5.0)
    (threaded,) = box

    assert inline is not None and threaded is not None
    assert _strip_ids(inline) == _strip_ids(threaded)
    assert inline.gen_ai is not None


def test_request_side_completion_defers_the_same_way():
    """§6-3's request-side case: a transaction the TRACKER completes inside
    `_on_request_bytes` (the h2 tracker's shape — HTTP/1 always completes on
    the response side) takes the identical seal-and-defer path, because both
    entry points converge on `_emit_span`."""
    from wardex_sdk._interceptors._trackers import _Txn as _TxnCls

    trap = _JobTrap()
    seam = _live_seam(trap)
    sock = _sock()
    txn = _TxnCls(
        method="POST",
        path="/v1/chat/completions",
        status=200,
        request_body=_CHAT_REQ,
        response_body=_CHAT_RESP,
        parent=None,
        start_ns=1,
        end_ns=2,
        ttfb_ms=0.0,
        version="2",
    )

    class _RequestSideTracker:
        """A tracker whose transaction completes on the REQUEST call."""

        def on_request_bytes(self, data):
            return [txn]

        def on_response_bytes(self, data):
            return []

    st = seam._state(sock)
    st.tracker = _RequestSideTracker()
    # A request-looking prefix, so the TLS sniff-latch classifies "http";
    # the stub tracker decides what completes, not the bytes.
    seam._on_request_bytes(sock, b"POST tail-of-request")
    (job,) = trap.jobs
    span = job.ctx.run(job.run)
    assert span is not None and span.gen_ai is not None


def test_scope_tags_at_response_time_reach_the_deferred_span():
    """§6-4 — seam-driven: tags ambient at the response instant land on the
    span the worker finalizes later."""
    from wardex_sdk import set_tag

    c, t = _seam_client()
    try:
        c._finalize.stop(1.0)
        seam = _live_seam(c)
        set_tag("who", "here")
        _exchange(seam, _sock())
        set_tag("who", "gone")
        c._settle()
        (span,) = [s for s in c._spans if s.kind.name == "CLIENT"]
        assert ("who", "here") in span.extra
    finally:
        c.close()


def test_degraded_run_at_response_time_marks_the_deferred_span():
    """§6-5 — the immutable-fact ContextVar rides the sealed ctx: a request
    inside a degraded run is admitted and marked, even though the gate runs
    later, on another thread, outside the block."""
    c, t = _seam_client(mode=CaptureMode.AGENT)
    try:
        c._finalize.stop(1.0)
        seam = _live_seam(c)
        with degraded_run():
            _exchange(seam, _sock("internal.example"), resp=_response_bytes(b'{"ok":true}'))
        c._settle()
        (span,) = [s for s in c._spans if s.kind.name == "CLIENT"]
        markers = span.capture_integrity.limitations
        assert Limitation.INSTRUMENTATION_DEGRADED in markers
        assert Limitation.PARENT_UNRESOLVED in markers
    finally:
        c.close()


def test_an_agent_mode_eviction_fallback_is_captured_with_bodies_withheld():
    """§6-6's §3.9 half: a fallback that passed the gate ONLY through
    wardex's own degradation ships WITHOUT payloads — the user's mode
    excluded this traffic, and overload must not be what exports its bodies.
    Transport facts and the marker stay."""
    c, t = _seam_client(mode=CaptureMode.AGENT, limits=LimitsConfig(max_parse_backlog=1))
    try:
        c._finalize.stop(1.0)
        seam = _live_seam(c)
        _exchange(seam, _sock())
        _exchange(seam, _sock())  # evicts the first
        spans = [s for s in c._spans if s.kind.name == "CLIENT"]
        assert len(spans) == 1, "the evicted job must still ship"
        span = spans[0]
        assert Limitation.PARSE_BACKLOG_FULL in span.capture_integrity.limitations
        assert not span.input_data and not span.output_data
        assert span.gen_ai is None
        assert span.transport is not None
    finally:
        c.close()


def test_an_all_mode_eviction_fallback_keeps_its_bodies():
    """The other half of §3.9: when the span would have passed the gate
    anyway (ALL mode here), the fallback keeps its payloads."""
    c, t = _seam_client(mode=CaptureMode.ALL, limits=LimitsConfig(max_parse_backlog=1))
    try:
        c._finalize.stop(1.0)
        seam = _live_seam(c)
        _exchange(seam, _sock())
        _exchange(seam, _sock())
        spans = [s for s in c._spans if s.kind.name == "CLIENT"]
        assert len(spans) == 1
        span = spans[0]
        assert Limitation.PARSE_BACKLOG_FULL in span.capture_integrity.limitations
        assert span.input_data == _CHAT_REQ
        assert span.output_data == _CHAT_RESP
    finally:
        c.close()


def test_a_raising_prefilter_still_captures():
    """§6-6b — the fail-open pin: a raising `_transport_prefilter` reads as
    ALLOW (the old `except: return True`), never as a policy drop."""

    class _RaisingSeam(SSLInterceptor):
        def _transport_prefilter(self, st):
            raise RuntimeError("prefilter broke")

    c, t = _seam_client(mode=CaptureMode.AGENT)
    try:
        seam = _RaisingSeam()
        seam._client = c
        seam._load_limits(c)
        _exchange(seam, _sock("internal.example"), resp=_response_bytes(b'{"ok":true}'))
        c._settle()
        spans = [s for s in c._spans if s.kind.name == "CLIENT"]
        assert len(spans) == 1, "a broken prefilter must fail open, not drop"
        assert counters.get("interceptors.seam.prefilter") == 1
    finally:
        c.close()


def test_an_eviction_fallback_that_raises_stays_inside_wardex(monkeypatch):
    """§6-6c — under backlog pressure (exactly the overload case) an
    assembly raise must never surface out of the host's own socket call."""

    def boom(p, *, parse, extra):
        raise RuntimeError("assembly broke")

    monkeypatch.setattr(seam_mod, "_assemble", boom)
    c, t = _seam_client(limits=LimitsConfig(max_parse_backlog=1))
    try:
        c._finalize.stop(1.0)
        seam = _live_seam(c)
        _exchange(seam, _sock())
        _exchange(seam, _sock())  # evicts the first; its fallback raises
        assert counters.get("client.finalize.fallback") == 1
    finally:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(seam_mod, "_assemble", boom)
        monkeypatch.undo()
        c.close()


def test_parse_exception_ships_span_with_instrumentation_degraded(monkeypatch):
    """§6-12 — the side defect this item closes: a RAISING parser used to be
    swallowed uncounted, and under AGENT with no parent the span vanished
    entirely. Now it ships, marked, and the guard counts."""

    def broken_parse(*args, **kwargs):
        raise RuntimeError("parser broke")

    monkeypatch.setattr(seam_mod, "parse_llm_semantics", broken_parse)
    c, t = _seam_client(mode=CaptureMode.AGENT)
    try:
        c._finalize.stop(1.0)
        seam = _live_seam(c)
        _exchange(seam, _sock())
        c._settle()
        spans = [s for s in c._spans if s.kind.name == "CLIENT"]
        assert len(spans) == 1, "a raising parser must not delete the span"
        span = spans[0]
        assert Limitation.INSTRUMENTATION_DEGRADED in span.capture_integrity.limitations
        assert Limitation.SEMANTIC_PARSE_FAILED not in span.capture_integrity.limitations
        assert span.gen_ai is None
        assert counters.get("interceptors.seam.parse") == 1
    finally:
        c.close()


@pytest.mark.leaves_pending
def test_pending_job_holds_no_socket_reference():
    """§6-14 / I-F1: a queued job must not pin the host's file descriptor."""
    c, t = _seam_client()
    try:
        c._finalize.stop(1.0)
        seam = _live_seam(c)
        sock = _sock()
        _exchange(seam, sock)
        assert c._finalize.pending() == 1
        ref = weakref.ref(sock)
        seam._conns.clear()  # the seam's own table may hold per-connection state
        del sock
        gc.collect()
        assert ref() is None, "the sealed job kept the socket alive"
    finally:
        c.close()


def test_assemble_reads_no_seam_state():
    """§6-15 strengthened (M2-1/S1-1) — three proofs in one:

    (i) the sealed job's output is IMMUTABLE under a re-install: swapping
        `seam._native_limits` after the seal changes nothing, because the
        job holds its own snapshot BY WIRING (`_parse_semantics(p)` reads
        `p.limits`);
    (ii) it is immutable under the seam's DEATH: with the seam
        garbage-collected (weakref proven dead), the job still finalizes to
        the same span — no seam access can exist in code whose referents
        are gone;
    (iii) the self-enforcing field census: every `_PendingTxn` field is
        consumed by the assembly functions or by the queue/client protocol
        surface, so "sealed but never read" (the exact defect the review
        found in the draft) cannot re-enter silently.
    """
    trap = _JobTrap()
    seam = _live_seam(trap)
    _exchange(seam, _sock())
    (job,) = trap.jobs
    golden = _strip_ids(job.ctx.run(job.run))

    seam._native_limits = LimitsConfig(max_decoded_bytes=1024).to_native()  # re-install
    ref = weakref.ref(seam)
    seam.uninstall()
    del seam
    gc.collect()
    assert ref() is None, "something still references the seam; the proof is vacuous"

    again = _strip_ids(job.ctx.run(job.run))
    assert again == golden

    # (iii) — the field census.
    from wardex_sdk._client import Client as _ClientCls
    from wardex_sdk._finalize import FinalizeQueue as _FQ

    consumers = (
        seam_mod._assemble.__code__,
        seam_mod._parse_semantics.__code__,
        seam_mod._PendingTxn.run.__code__,
        seam_mod._PendingTxn.fallback.__code__,
        # the queue/client protocol surface consumes `ctx` and `size`
        _FQ.submit.__code__,
        _FQ._run_pending.__code__,
        _FQ._run_one.__code__,
        _ClientCls.capture_deferred.__code__,
    )
    consumed: set[str] = set()
    for code in consumers:
        consumed |= set(code.co_names)
    fields = set(seam_mod._PendingTxn.__dataclass_fields__)
    unread = fields - consumed
    assert unread == set(), (
        f"_PendingTxn field(s) sealed but never consumed: {sorted(unread)} — "
        "the 'declared snapshot, dead field' defect is back"
    )
    # And the method versions must not return.
    from wardex_sdk._interceptors._seam import ByteSeamInterceptor

    assert not hasattr(ByteSeamInterceptor, "_assemble")
    assert not hasattr(ByteSeamInterceptor, "_parse_semantics")
    assert not hasattr(ByteSeamInterceptor, "_build_span")


def test_grpc_spans_are_assembled_inline():
    """§6-21 — gRPC never defers (§3.6): the span is present synchronously,
    keeps its GRPC label and status, and carries neither deferral marker (a
    skipped-parse marker on a span whose parse never existed is a false
    confession)."""
    from wardex_sdk._interceptors._trackers import _Txn as _TxnCls

    def _msg(payload: bytes) -> bytes:
        return b"\x00" + len(payload).to_bytes(4, "big") + payload

    c, t = _seam_client()
    try:
        seam = _live_seam(c)
        sock = _sock("grpc.example")
        st = seam._state(sock)
        txn = _TxnCls(
            method="POST",
            path="/echo.Echo/Say",
            status=200,
            request_body=_msg(b"abc"),
            response_body=_msg(b"xyz"),
            parent=None,
            start_ns=1,
            end_ns=2,
            ttfb_ms=0.0,
            version="2",
            content_type="application/grpc",
            grpc_status=0,
        )
        seam._emit_span(sock, st, txn)
        spans = [s for s in c._spans if s.kind.name == "CLIENT"]
        assert len(spans) == 1, "gRPC must assemble inline, without a settle"
        assert c._finalize.pending() == 0
        span = spans[0]
        assert span.name == "gRPC /echo.Echo/Say"
        assert ("rpc.grpc.status_code", 0) in span.extra
        markers = span.capture_integrity.limitations if span.capture_integrity else ()
        assert Limitation.PARSE_BACKLOG_FULL not in markers
        assert Limitation.PARSE_SKIPPED_AT_SHUTDOWN not in markers
    finally:
        c.close()


def test_ws_spans_are_assembled_inline():
    """§6-21's WS half, at the retire path: a WS session span exists the
    moment the connection ends — no queue involved."""
    from wardex_sdk._interceptors._trackers import _Txn as _TxnCls

    c, t = _seam_client()
    try:
        seam = _live_seam(c)
        st = seam_mod._ConnectionState(tracker=None, server_address="ws.example", server_port=443)
        txn = _TxnCls(
            method="GET",
            path="/socket",
            status=101,
            request_body=b"",
            response_body=b"",
            parent=None,
            start_ns=1,
            end_ns=2,
            ttfb_ms=0.0,
            version="websocket",
            ws_close_code=1000,
        )
        seam._emit_ws(st, txn)
        spans = [s for s in c._spans if s.kind.name == "CLIENT"]
        assert len(spans) == 1
        assert c._finalize.pending() == 0
    finally:
        c.close()


# ==========================================================================
# §6-1 — the headline: the event loop does not stall on a large response
# ==========================================================================

import asyncio  # noqa: E402
import http.server  # noqa: E402
import ssl  # noqa: E402
from pathlib import Path  # noqa: E402

import wardex_sdk as wardex  # noqa: E402
from wardex_sdk.testing import RecordingTransport  # noqa: E402

_FIXTURES = Path(__file__).parent / "fixtures"


def _big_sse_body(target_bytes: int) -> bytes:
    word = "y" * 512
    chunks: list[bytes] = []
    size = 0
    i = 0
    while size < target_bytes:
        payload = (
            b'data: {"id":"chatcmpl-big","object":"chat.completion.chunk",'
            b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"'
            + f"{word}{i} ".encode()
            + b'"},"finish_reason":null}]}\n\n'
        )
        chunks.append(payload)
        size += len(payload)
        i += 1
    chunks.append(
        b'data: {"id":"chatcmpl-big","object":"chat.completion.chunk",'
        b'"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n'
    )
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)


@pytest.fixture
def big_sse_tls_server():
    """A TLS server streaming an ~8 MB OpenAI SSE body (≈ 40 ms to parse)."""
    body = _big_sse_body(8 * 1024 * 1024)

    class _BigSseHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            for start in range(0, len(body), 256 * 1024):
                self.wfile.write(body[start : start + 256 * 1024])
                self.wfile.flush()

        def log_message(self, *args: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _BigSseHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(_FIXTURES / "cert.pem"), keyfile=str(_FIXTURES / "key.pem"))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    host, port = httpd.socket.getsockname()[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"https://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _verify_ctx() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=str(_FIXTURES / "cert.pem"))


def _max_loop_drift(url: str) -> float:
    """Drive one big SSE request on an asyncio loop while a 1 ms watchdog
    coroutine measures how far its wakeups drift. Returns max drift (s)."""
    import httpx

    async def scenario() -> float:
        drifts: list[float] = []
        stop = asyncio.Event()

        async def watchdog() -> None:
            loop = asyncio.get_running_loop()
            while not stop.is_set():
                before = loop.time()
                await asyncio.sleep(0.001)
                drifts.append(loop.time() - before - 0.001)

        dog = asyncio.ensure_future(watchdog())
        async with httpx.AsyncClient(verify=_verify_ctx()) as client:
            await client.post(f"{url}/v1/chat/completions", json={"model": "gpt-4o-mini"})
        # One extra beat so a stall at response completion is measured too.
        await asyncio.sleep(0.05)
        stop.set()
        await dog
        return max(drifts) if drifts else 0.0

    return asyncio.run(scenario())


def test_loop_thread_is_not_stalled_by_a_large_response(big_sse_tls_server):
    """§6-1 — README's "Capture itself never blocks your coroutines", made
    true and pinned. Asserted against a same-test wardex-off BASELINE, not
    an absolute number (a loaded CI machine drifts on its own): before the
    deferred split the on-minus-off difference measured the whole ~40 ms
    parse; the bound leaves a wide margin under that and far above noise."""
    drift_off = _max_loop_drift(big_sse_tls_server)

    wardex.init(intercept=True, transport=RecordingTransport())
    try:
        drift_on = _max_loop_drift(big_sse_tls_server)
        wardex.flush()
        spans = [s for e in _hub.get_client()._transport.envelopes for s in e.spans]
        assert any(s.gen_ai is not None for s in spans), (
            "the exchange was not captured as an LLM call — the drift measurement went vacuous"
        )
    finally:
        wardex.close()

    assert drift_on - drift_off < 0.010, (
        f"wardex added {1000 * (drift_on - drift_off):.1f} ms of event-loop stall "
        f"(on={1000 * drift_on:.1f} ms, off={1000 * drift_off:.1f} ms) — the parse "
        "is back on the loop thread"
    )
