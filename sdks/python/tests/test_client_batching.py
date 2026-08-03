"""Client batching internals — thread safety, backpressure, drain error paths."""

import threading
import time

from wardex_sdk._client import _UNBOUNDED_TRANSPORT_FLUSH_TIMEOUT, Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._limits import CaptureLimits
from wardex_sdk._types import (
    InternalEnvelope,
    InternalSpan,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport._base import Transport


def _wait_for(predicate, timeout=5.0):
    """Poll a predicate with a generous upper bound — no bare-sleep asserts."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _span(name="s"):
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name=name,
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


def test_concurrent_capture_and_drain_loses_nothing():
    """The core regression guard for the background flusher (spec §11): no loss,
    no dup."""
    t = _Recording()
    c = Client(WardexConfig(api_key="k", limits=CaptureLimits(max_buffer_spans=100_000)), t)
    n_threads, m_spans = 8, 500
    stop_draining = threading.Event()

    def producer():
        for _ in range(m_spans):
            c.capture_span(_span())

    def drainer():
        while not stop_draining.is_set():
            c.flush()

    producers = [threading.Thread(target=producer) for _ in range(n_threads)]
    d = threading.Thread(target=drainer)
    d.start()
    for th in producers:
        th.start()
    for th in producers:
        th.join()
    stop_draining.set()
    d.join()
    c.flush()  # drain whatever the racing drainer didn't take
    total_sent = sum(len(e.spans) for e in t.envelopes)
    assert total_sent == n_threads * m_spans
    c.close()


def test_backpressure_drops_oldest_keeps_newest():
    t = _Recording()
    c = Client(WardexConfig(api_key="k", limits=CaptureLimits(max_buffer_spans=10)), t)
    # This test predates the background worker and asserts on the
    # *manual* flush()'s view of a single overflow burst. With max_buffer_spans=10
    # the wake threshold is max(1, 10 // 4) = 2, so the live worker can (and, on
    # this machine, reliably does) race the tight capture loop and drain part of
    # the burst on its own — splitting it into multiple envelopes and breaking
    # the single-envelope assertion below. Stop the worker so only the explicit
    # flush() drains, which is exactly the invariant this test verifies
    # (flush() bypasses the worker entirely — spec §4.1).
    c._worker.stop()
    for i in range(15):
        c.capture_span(_span(name=f"s{i}"))
    c.flush()
    [env] = t.envelopes
    assert len(env.spans) == 10
    assert [s.name for s in env.spans] == [f"s{i}" for i in range(5, 15)]
    c.close()


def test_dropped_count_reported_once_in_debug(capsys):
    t = _Recording()
    c = Client(WardexConfig(api_key="k", limits=CaptureLimits(max_buffer_spans=2), debug=True), t)
    # max_buffer_spans=2 gives a wake threshold of max(1, 2 // 4) = 1, so the
    # live worker would race this tight burst and drain early,
    # splitting the "dropped 3" report. Stop it so only the explicit flush()
    # below drains (see test_backpressure_drops_oldest_keeps_newest for the
    # same reasoning).
    c._worker.stop()
    for _ in range(5):
        c.capture_span(_span())
    c.flush()
    assert "[wardex] dropped 3 spans (buffer full)" in capsys.readouterr().err
    c.flush()  # counter was reset — no second report
    assert "dropped" not in capsys.readouterr().err
    c.close()


def test_before_send_exception_drops_envelope_and_does_not_propagate():
    def boom(envelope):
        raise RuntimeError("boom")

    t = _Recording()
    c = Client(WardexConfig(api_key="k", before_send=boom), t)
    c.capture_span(_span())
    c.flush()  # must not raise (fail-closed: drop, spec §10)
    assert t.envelopes == []
    c.close()


def test_export_exception_drops_envelope_and_does_not_propagate():
    class _Exploding(Transport):
        def export(self, envelope):
            raise ValueError("encode failed")

    c = Client(WardexConfig(api_key="k"), _Exploding())
    c.capture_span(_span())
    c.flush()  # must not raise
    c.close()


def test_capture_during_export_goes_to_fresh_buffer():
    """Capture must never block on (or leak into) an in-flight drain."""
    entered = threading.Event()
    release = threading.Event()

    class _Blocking(Transport):
        def __init__(self):
            self.envelopes = []

        def export(self, envelope):
            self.envelopes.append(envelope)
            entered.set()
            release.wait(timeout=5.0)

    t = _Blocking()
    c = Client(WardexConfig(api_key="k"), t)
    c.capture_span(_span(name="first"))
    flusher = threading.Thread(target=c.flush)
    flusher.start()
    assert entered.wait(timeout=5.0)
    c.capture_span(_span(name="second"))  # must return instantly, land in new buffer
    release.set()
    flusher.join(timeout=5.0)
    c.flush()
    assert [s.name for e in t.envelopes for s in e.spans] == ["first", "second"]
    c.close()


def test_auto_flush_without_manual_flush():
    """The reason the background flusher exists: data leaves with no flush() call."""
    t = _Recording()
    c = Client(WardexConfig(api_key="k", flush_interval=0.05), t)
    c.capture_span(_span())
    assert _wait_for(lambda: sum(len(e.spans) for e in t.envelopes) >= 1)
    c.close()


def test_threshold_wakes_worker_before_interval():
    t = _Recording()
    # max_buffer_spans=8 → threshold max(1, 8//4)=2; interval too long to fire
    c = Client(
        WardexConfig(api_key="k", flush_interval=3600.0, limits=CaptureLimits(max_buffer_spans=8)),
        t,
    )
    c.capture_span(_span())
    c.capture_span(_span())
    assert _wait_for(lambda: sum(len(e.spans) for e in t.envelopes) >= 2)
    c.close()


def test_close_stops_worker_and_drains_remainder():
    t = _Recording()
    c = Client(WardexConfig(api_key="k", flush_interval=3600.0), t)
    c.capture_span(_span())
    c.close()
    assert not c._worker.is_alive()
    assert sum(len(e.spans) for e in t.envelopes) == 1
    c.close()  # idempotent
    assert sum(len(e.spans) for e in t.envelopes) == 1


def test_capture_after_close_is_rejected():
    t = _Recording()
    c = Client(WardexConfig(api_key="k"), t)
    c.close()
    c.capture_span(_span())
    c.flush()
    assert sum(len(e.spans) for e in t.envelopes) == 0


def test_reentrant_flush_from_before_send_does_not_deadlock():
    """The signal handler re-enters _drain on the same thread; RLock must allow it."""
    t = _Recording()
    holder = {}

    def reenter(envelope):
        holder["client"].flush()  # same-thread nested drain (empty buffer) — must not hang
        return envelope

    c = Client(WardexConfig(api_key="k", before_send=reenter), t)
    holder["client"] = c
    c.capture_span(_span())
    # daemon: on a regression this thread hangs forever; it must not block process exit
    worker = threading.Thread(target=c.flush, daemon=True)
    worker.start()
    worker.join(timeout=5.0)
    assert not worker.is_alive(), "reentrant flush deadlocked"
    assert sum(len(e.spans) for e in t.envelopes) == 1
    c.close()


def test_signal_flush_while_buffer_lock_held_does_not_deadlock():
    """A signal handler may call flush() on a thread that holds _buffer_lock."""
    t = _Recording()
    c = Client(WardexConfig(api_key="k", flush_interval=3600.0), t)
    done = threading.Event()

    def simulate_signal_during_append():
        with c._buffer_lock:  # the frame the signal interrupts
            c.flush()  # handler's flush → same-thread nested buffer-lock acquire
        done.set()

    th = threading.Thread(target=simulate_signal_during_append, daemon=True)
    th.start()
    assert done.wait(timeout=5.0), "flush deadlocked on _buffer_lock"
    c.close()


def test_transport_flush_exception_does_not_propagate():
    class _FlushExploding(Transport):
        def export(self, envelope):
            pass

        def flush(self, timeout: float = 5.0) -> None:
            raise ValueError("stream closed")

    c = Client(WardexConfig(api_key="k"), _FlushExploding())
    c.flush()  # empty-buffer branch must not raise
    c.capture_span(_span())
    c.flush()  # post-export branch must not raise
    c.close()


def test_flush_deadline_is_not_extended_by_an_in_flight_export():
    """flush(t) must return on its own deadline even when another thread is
    already inside a slow export.

    This is the signal handler's flush(2.0) (_lifecycle._SIGNAL_FLUSH_TIMEOUT).
    Before the fix the drain lock was taken unconditionally, so this call waited
    out the in-flight POST in full before starting its own — the "2s" shutdown
    bound was really the transport timeout twice over, and SIGTERM hung for it.
    """
    in_export = threading.Event()
    release = threading.Event()

    class _Stuck(Transport):
        def __init__(self):
            self.envelopes = []
            self.blocked_once = False

        def export(self, envelope, *, timeout=None):
            self.envelopes.append(envelope)
            if not self.blocked_once:  # only the first export blocks, so the
                self.blocked_once = True  # close() below can still finish
                in_export.set()
                release.wait(timeout=10.0)

    t = _Stuck()
    c = Client(WardexConfig(api_key="k", flush_interval=3600.0), t)
    c._worker.stop()  # the slot holder must be a thread this test controls
    try:
        c.capture_span(_span(name="first"))
        holder = threading.Thread(target=c.flush, args=(30.0,), daemon=True)
        holder.start()
        assert in_export.wait(timeout=5.0), "transport never entered export"

        c.capture_span(_span(name="second"))
        start = time.monotonic()
        c.flush(0.2)  # must give up on its own deadline, not on the export's
        elapsed = time.monotonic() - start
    finally:
        release.set()
    holder.join(timeout=5.0)
    assert not holder.is_alive()
    assert elapsed < 2.0, f"flush(0.2) waited {elapsed:.2f}s behind an in-flight export"

    # Declining costs nothing: the swap happens after the slot is taken, so a
    # drain that gave up never held those spans.
    c.close(5.0)
    assert [s.name for e in t.envelopes for s in e.spans] == ["first", "second"]


class _TimeoutRecording(Transport):
    def __init__(self):
        self.timeouts: list[float | None] = []
        self.flush_timeouts: list[float] = []

    def export(self, envelope: InternalEnvelope, *, timeout: float | None = None) -> None:
        self.timeouts.append(timeout)

    def flush(self, timeout: float = 5.0) -> None:
        self.flush_timeouts.append(timeout)


def test_flush_budget_reaches_the_transport():
    """A bound on flush() is worthless if the transport never hears about it:
    only the transport can bound its own I/O."""
    t = _TimeoutRecording()
    c = Client(WardexConfig(api_key="k", flush_interval=3600.0), t)
    c._worker.stop()
    c.capture_span(_span())
    c.flush(2.0)
    assert t.timeouts, "export was never called"
    assert t.timeouts[0] is not None, "flush(2.0) handed the transport no deadline"
    assert 0.0 < t.timeouts[0] <= 2.0
    c.close()


def test_periodic_drain_imposes_no_deadline_on_the_transport():
    """The background worker has no deadline of its own — nobody waits on it.
    Giving it one would clamp the transport's configured timeout on the one path
    that ships data unattended, turning slow-but-working POSTs into lost ones."""
    t = _TimeoutRecording()
    c = Client(WardexConfig(api_key="k", flush_interval=0.05), t)
    c.capture_span(_span())
    assert _wait_for(lambda: bool(t.timeouts))
    assert t.timeouts[0] is None
    # Transport.flush() has no None to express "no deadline", so the periodic
    # path hands it the module's stated constant. Pinned because a buffering
    # third-party transport is the only consumer and would otherwise silently
    # get whatever number someone edited it to.
    assert _wait_for(lambda: bool(t.flush_timeouts))
    # The literal is pinned alongside the constant on purpose: it has to keep
    # matching `Transport.flush`'s own default, so a buffering transport sees
    # the same budget whether the client names one or not.
    assert t.flush_timeouts[0] == _UNBOUNDED_TRANSPORT_FLUSH_TIMEOUT == 5.0
    c.close()


def test_transport_predating_the_timeout_parameter_still_exports(capsys):
    """`Transport` is public API. A subclass written against the old
    `export(self, envelope)` must keep working: calling it with timeout= would
    raise TypeError into _drain's fail-closed handler, which drops the envelope
    silently — a stall traded for total data loss."""

    class _Legacy(Transport):
        def __init__(self):
            self.envelopes = []

        def export(self, envelope):  # pre-timeout signature, on purpose
            self.envelopes.append(envelope)

    t = _Legacy()
    c = Client(WardexConfig(api_key="k", flush_interval=3600.0, debug=True), t)
    c._worker.stop()
    c.capture_span(_span())
    c.flush(2.0)
    assert len(t.envelopes) == 1, "an old-signature transport stopped receiving envelopes"
    assert "envelope dropped" not in capsys.readouterr().err
    c.close()


def test_transport_swapped_after_construction_is_re_probed(capsys):
    """`_transport` is reassignable and is reassigned in practice. A signature
    probe cached once at construction answers for the transport that is gone, so
    it hands `timeout=` to a replacement that cannot take it and drops every
    envelope through the fail-closed path."""
    c = Client(WardexConfig(api_key="k", flush_interval=3600.0, debug=True), _TimeoutRecording())
    c._worker.stop()

    class _LegacyReplacement(Transport):
        def __init__(self):
            self.envelopes = []

        def export(self, envelope):  # pre-timeout signature, on purpose
            self.envelopes.append(envelope)

    replacement = _LegacyReplacement()
    c._transport = replacement
    c.capture_span(_span())
    c.flush(2.0)
    assert len(replacement.envelopes) == 1, "the swapped-in transport received nothing"
    assert "envelope dropped" not in capsys.readouterr().err
    c.close()


class _BlockingExport(Transport):
    """Holds the export slot until `release` is set. The first export blocks;
    every later one returns at once, so a close() behind it can still finish."""

    def __init__(self, in_export, release):
        self.names: list[str] = []
        self.first = True
        self._in_export = in_export
        self._release = release

    def export(self, envelope: InternalEnvelope, *, timeout=None) -> None:
        self.names.extend(s.name for s in envelope.spans)
        if self.first:
            self.first = False
            self._in_export.set()
            self._release.wait(timeout=10.0)


def _client_stuck_in_export(in_export, release, **config):
    """A client whose transport is blocked inside export() on a thread this test
    owns, with two more spans buffered behind it. Returns (client, transport,
    holder-thread)."""
    t = _BlockingExport(in_export, release)
    c = Client(WardexConfig(api_key="k", flush_interval=3600.0, **config), t)
    c._worker.stop()  # the slot holder must be a thread the test controls
    c.capture_span(_span(name="in-flight"))
    holder = threading.Thread(target=c.flush, args=(30.0,), daemon=True)
    holder.start()
    assert in_export.wait(timeout=5.0), "transport never entered export"
    c.capture_span(_span(name="tail-1"))
    c.capture_span(_span(name="tail-2"))
    return c, t, holder


def test_close_ships_the_tail_behind_an_export_it_can_outwait():
    """close() is an orderly shutdown, not the signal path: the process carries
    on afterwards (install() closes the previous client on every re-init), so
    its contract is to ship what is buffered within its own timeout. A POST that
    finishes inside that budget must not cost the tail."""
    in_export, release = threading.Event(), threading.Event()
    c, t, holder = _client_stuck_in_export(in_export, release)
    releaser = threading.Timer(0.3, release.set)
    releaser.start()
    try:
        c.close(5.0)  # step 2 joins nothing, step 3 waits out the POST
    finally:
        release.set()
        releaser.cancel()
    holder.join(timeout=5.0)
    assert not holder.is_alive()
    assert t.names == ["in-flight", "tail-1", "tail-2"], "close() abandoned a tail it could ship"


def test_close_that_cannot_ship_the_tail_reports_it_instead_of_hiding_it(capsys):
    """A declined drain is free everywhere except here: after close() there is
    no next drain, so the same decline is data loss. Bounding close() is the
    point of WAR-40 and stands — losing the tail *silently* is not. The spans
    must leave the buffer counted and reported, not sit in a closed client still
    reporting themselves as pending."""
    in_export, release = threading.Event(), threading.Event()
    c, t, holder = _client_stuck_in_export(in_export, release, debug=True)
    capsys.readouterr()  # discard anything logged during setup
    start = time.monotonic()
    try:
        c.close(0.2)  # POST outlasts every step; the tail is genuinely unshippable
        elapsed = time.monotonic() - start
    finally:
        release.set()
    holder.join(timeout=5.0)
    assert not holder.is_alive()

    assert elapsed < 5.0, f"close(0.2) waited {elapsed:.2f}s behind an in-flight export"
    assert t.names == ["in-flight"], "the blocked export somehow received the tail"
    err = capsys.readouterr().err
    assert "close abandoned 2 buffered spans" in err, (
        f"close() dropped the tail without saying so; stderr was: {err!r}"
    )
    assert list(c._buffer.spans) == [], "unshippable spans left resident in a closed client"


def test_hostile_flush_timeout_never_reaches_the_host():
    """`flush(timeout)` is public API, so `timeout` is application input. Every
    value a host can pass must be handled or ignored — never raised back at it.
    float('nan') used to reach RLock.acquire() as a ValueError and a non-number
    the deadline arithmetic as a TypeError, both outside every handler."""
    t = _TimeoutRecording()
    c = Client(WardexConfig(api_key="k", flush_interval=3600.0), t)
    c._worker.stop()
    for hostile in (float("nan"), float("-nan"), "x", None, object(), -1.0, float("inf")):
        c.capture_span(_span())
        c.flush(hostile)  # must not raise
    assert len(t.timeouts) == 7, "a hostile timeout cost the envelope entirely"
    assert all(v is not None and v == v and v >= 0.0 for v in t.timeouts), t.timeouts
    c.close()


def test_hostile_close_timeout_never_reaches_the_host():
    """Same contract for close(): it also feeds Thread.join() and a third-party
    Transport.close(), neither of which tolerates NaN or a str either."""
    for hostile in (float("nan"), "x", None, object(), -1.0, float("inf")):
        t = _TimeoutRecording()
        c = Client(WardexConfig(api_key="k", flush_interval=3600.0), t)
        c.capture_span(_span())
        c.close(hostile)  # must not raise
        assert len(t.timeouts) == 1, f"close({hostile!r}) shipped nothing"


def test_exhausted_budget_never_reaches_the_transport_as_a_negative():
    """`Transport` is public API and a third-party one will pass `timeout`
    straight to a socket, where a negative is an error rather than "no wait".
    The remaining budget can legitimately go negative — before_send is called
    inside the deadline — so both floors have to hold."""

    class _Recorder(Transport):
        def __init__(self):
            self.export_timeouts: list[float | None] = []
            self.flush_timeouts: list[float] = []

        def export(self, envelope: InternalEnvelope, *, timeout: float | None = None) -> None:
            self.export_timeouts.append(timeout)

        def flush(self, timeout: float = 5.0) -> None:
            self.flush_timeouts.append(timeout)

    def _slow_before_send(envelope):
        time.sleep(0.05)  # outlives the 0.01s budget below
        return envelope

    t = _Recorder()
    c = Client(WardexConfig(api_key="k", flush_interval=3600.0, before_send=_slow_before_send), t)
    c._worker.stop()
    c.capture_span(_span())
    c.flush(0.01)
    assert t.export_timeouts == [0.0], f"export got a negative budget: {t.export_timeouts}"
    assert t.flush_timeouts == [0.0], f"flush got a negative budget: {t.flush_timeouts}"
    c.close()


def test_signal_flush_inside_the_buffer_lock_cannot_deadlock_the_worker():
    """The one path that inverts the export-lock-then-buffer-lock order: a signal
    landing inside capture_span's buffer-lock block runs flush() on that same
    thread, so it reaches for the export lock while holding the buffer lock.
    With the worker holding the export lock and blocked on the buffer lock, that
    is a real AB-BA inversion — survivable only because the handler's acquire is
    timed. An unbounded acquire hangs both threads forever, which is what the
    lock-order comment in Client.__init__ has to keep saying out loud."""

    class _Idle(Transport):
        def export(self, envelope, *, timeout=None):
            pass

    c = Client(WardexConfig(api_key="k", flush_interval=3600.0), _Idle())
    c._worker.stop()  # this test owns both threads
    holding, release, finished = threading.Event(), threading.Event(), threading.Event()

    def interrupted_capture():
        with c._buffer_lock:  # the frame the signal interrupts
            holding.set()
            release.wait(timeout=5.0)
            c.flush(0.3)  # the handler's bounded flush, buffer lock still held
        finished.set()

    def export_slot_taken():
        if c._export_lock.acquire(blocking=False):
            c._export_lock.release()
            return False
        return True

    interrupted = threading.Thread(target=interrupted_capture, daemon=True)
    interrupted.start()
    assert holding.wait(timeout=5.0)

    worker = threading.Thread(target=c.flush, args=(30.0,), daemon=True)
    worker.start()  # takes the export lock, then blocks on the buffer lock
    assert _wait_for(export_slot_taken), "the other thread never took the export slot"
    release.set()

    assert finished.wait(timeout=5.0), "the inverted lock order deadlocked both threads"
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    c.close()
