"""Client batching internals — thread safety, backpressure, drain error paths."""

import threading
import time

from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._client import _UNBOUNDED_TRANSPORT_FLUSH_TIMEOUT, Client
from wardex_sdk._config import BackendConfig, BatchingConfig, WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._types import (
    InternalEnvelope,
    InternalSpan,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport._base import UNDELIVERED, Transport


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
    c = Client(
        WardexConfig(
            limits=LimitsConfig(max_buffer_spans=100_000), backend=BackendConfig(api_key="k")
        ),
        t,
    )
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
    c = Client(
        WardexConfig(limits=LimitsConfig(max_buffer_spans=10), backend=BackendConfig(api_key="k")),
        t,
    )
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
    c = Client(
        WardexConfig(
            limits=LimitsConfig(max_buffer_spans=2), debug=True, backend=BackendConfig(api_key="k")
        ),
        t,
    )
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
    c = Client(WardexConfig(before_send=boom, backend=BackendConfig(api_key="k")), t)
    c.capture_span(_span())
    c.flush()  # must not raise (fail-closed: drop, spec §10)
    assert t.envelopes == []
    c.close()


def test_export_exception_drops_envelope_and_does_not_propagate():
    class _Exploding(Transport):
        def export(self, envelope):
            raise ValueError("encode failed")

    c = Client(WardexConfig(backend=BackendConfig(api_key="k")), _Exploding())
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
    c = Client(WardexConfig(backend=BackendConfig(api_key="k")), t)
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
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=0.05)
        ),
        t,
    )
    c.capture_span(_span())
    assert _wait_for(lambda: sum(len(e.spans) for e in t.envelopes) >= 1)
    c.close()


def test_threshold_wakes_worker_before_interval():
    t = _Recording()
    # max_buffer_spans=8 → threshold max(1, 8//4)=2; interval too long to fire
    c = Client(
        WardexConfig(
            limits=LimitsConfig(max_buffer_spans=8),
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
        ),
        t,
    )
    c.capture_span(_span())
    c.capture_span(_span())
    assert _wait_for(lambda: sum(len(e.spans) for e in t.envelopes) >= 2)
    c.close()


def test_close_stops_worker_and_drains_remainder():
    t = _Recording()
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        t,
    )
    c.capture_span(_span())
    c.close()
    assert not c._worker.is_alive()
    assert sum(len(e.spans) for e in t.envelopes) == 1
    c.close()  # idempotent
    assert sum(len(e.spans) for e in t.envelopes) == 1


def test_capture_after_close_is_rejected():
    t = _Recording()
    c = Client(WardexConfig(backend=BackendConfig(api_key="k")), t)
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

    c = Client(WardexConfig(before_send=reenter, backend=BackendConfig(api_key="k")), t)
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
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        t,
    )
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

    c = Client(WardexConfig(backend=BackendConfig(api_key="k")), _FlushExploding())
    c.flush()  # empty-buffer branch must not raise
    c.capture_span(_span())
    c.flush()  # post-export branch must not raise
    c.close()


def test_flush_deadline_is_not_extended_by_an_in_flight_export():
    """flush(t) must return on its own deadline even when another thread is
    already inside a slow export.

    This is the signal handler's flush(2.0) (_runtime._SIGNAL_FLUSH_TIMEOUT).
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
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        t,
    )
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
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        t,
    )
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
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=0.05)
        ),
        t,
    )
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
    c = Client(
        WardexConfig(
            debug=True,
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
        ),
        t,
    )
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
    c = Client(
        WardexConfig(
            debug=True,
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
        ),
        _TimeoutRecording(),
    )
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
    c = Client(
        WardexConfig(
            **config,
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
        ),
        t,
    )
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


def test_close_that_cannot_ship_the_tail_reports_it_under_the_DEFAULT_config(capsys):
    """A declined drain is free everywhere except here: after close() there is
    no next drain, so the same decline is data loss. Bounding close() is the
    point and stands — losing the tail *silently* is not.

    `debug` is deliberately left at its default (False), because that is where
    the silence lives. The first repair of this defect printed its abandon line
    under `config.debug`, which no production process sets, so off-debug the loss
    stayed exactly as quiet as before — and worse, the spans were no longer in
    the buffer where an operator could find them. A guard that only exercises
    debug=True would have passed against that.
    """
    reset_reports_for_test()  # `report_once` is process-global
    in_export, release = threading.Event(), threading.Event()
    c, t, holder = _client_stuck_in_export(in_export, release)
    assert c.config.debug is False, "this guard is only meaningful off-debug"
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
    assert "could not ship 2 buffered span(s)" in err, (
        f"close() dropped the tail without saying so; stderr was: {err!r}"
    )
    assert "nothing will retry them" in err, (
        f"the line does not name the consequence; stderr was: {err!r}"
    )
    assert list(c._buffer.spans) == [], "unshippable spans left resident in a closed client"
    assert c._lost == 2, f"the abandoned tail was reported but not counted: _lost={c._lost}"
    assert c._dropped == 0, (
        f"a shutdown loss was counted as a buffer overflow: _dropped={c._dropped}"
    )


def test_close_reports_an_abandoned_tail_only_once_per_process(capsys):
    """The bound that makes an ungated print affordable on a shutdown path:
    `install()` closes the previous client on every re-init, so a process that
    re-inits in a loop must not write a line per close."""
    reset_reports_for_test()
    lines = []
    for _ in range(2):
        in_export, release = threading.Event(), threading.Event()
        c, _t, holder = _client_stuck_in_export(in_export, release)
        capsys.readouterr()
        try:
            c.close(0.2)
        finally:
            release.set()
        holder.join(timeout=5.0)
        assert not holder.is_alive()
        lines.append(capsys.readouterr().err)
    assert "could not ship 2 buffered span(s)" in lines[0], lines[0]
    assert lines[1] == "", f"the second close repeated the report: {lines[1]!r}"


class _DeadlineHonouring(Transport):
    """A transport that respects the budget it is handed, as OtlpHttpTransport
    does: a non-positive `timeout` means there is no time to send, so it does
    not — and it SAYS so, by returning `UNDELIVERED`. The recording transports
    elsewhere in this file neither honour the deadline nor say anything, which
    is exactly why they could not see the defects below.

    The return value is the whole mechanism. The client cannot see inside a
    transport, so `UNDELIVERED` is the only way a skip here is distinguishable
    from a delivery — and a client that instead guessed "spent budget, so it
    must have skipped" was wrong about every transport that ignores the budget
    and delivers anyway."""

    def __init__(self):
        self.timeouts: list[float | None] = []
        self.shipped: list[str] = []

    def export(self, envelope: InternalEnvelope, *, timeout=None) -> object | None:
        self.timeouts.append(timeout)
        if timeout is not None and timeout <= 0:
            return UNDELIVERED
        self.shipped.extend(s.name for s in envelope.spans)
        return None


def test_close_with_a_spent_budget_reports_the_tail_it_cannot_send(capsys):
    """The exit `_abandon` cannot see.

    Here the export slot IS free, so the final drain sails past the acquire and
    swaps the spans out — but the deadline is already gone, so the transport is
    handed 0.0 and skips. Same loss as the declined acquire, and it used to be
    reached from the public API (`close(-1.0)` floors to 0.0) with nothing
    counted and stderr empty.

    What close() reacts to is the transport's own verdict on this envelope, not
    a reading of the clock taken before `before_send` ran: see
    `test_a_before_send_that_outlives_close_s_budget_is_still_reported`, which
    is the same loss with the clock check made useless.
    """
    reset_reports_for_test()
    t = _DeadlineHonouring()
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        t,
    )
    c._worker.stop()
    c.capture_span(_span("only"))
    capsys.readouterr()
    c.close(-1.0)  # public API; floors to a 0.0 budget, so it is spent on arrival
    err = capsys.readouterr().err

    assert t.timeouts == [0.0], f"the transport was not handed a spent budget: {t.timeouts}"
    assert t.shipped == [], "a spent budget somehow reached the backend"
    assert list(c._buffer.spans) == [], "unshippable spans left resident in a closed client"
    assert c._lost == 1, f"the lost span was not counted: _lost={c._lost}"
    assert "could not ship 1 buffered span(s)" in err, (
        f"close(-1.0) lost the span without saying so; stderr was: {err!r}"
    )


def test_close_with_budget_left_still_ships_through_a_deadline_honouring_transport(capsys):
    """The control for the test above: reacting to a decline must not have
    turned every close() into an abandonment."""
    reset_reports_for_test()
    t = _DeadlineHonouring()
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        t,
    )
    c._worker.stop()
    c.capture_span(_span("only"))
    capsys.readouterr()
    c.close(5.0)
    assert t.shipped == ["only"], f"an ordinary close() shipped nothing: {t.timeouts}"
    assert c._lost == 0
    assert c._dropped == 0
    assert capsys.readouterr().err == "", "an ordinary close() reported a loss"


class _LegacyIgnoringDeadline(Transport):
    """Pre-`timeout=` signature — the shape `_accepts_timeout` exists for."""

    def __init__(self):
        self.shipped: list[str] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.shipped.extend(s.name for s in envelope.spans)


def test_close_with_a_spent_budget_does_not_claim_a_loss_a_legacy_transport_avoids(capsys):
    """A transport that cannot take `timeout=` is handed the envelope with no
    bound at all, so a spent deadline does not stop it delivering. Reporting a
    loss there would be a false alarm — and a false alarm is not free: it burns
    the process-global `report_once` key, so the next REAL loss prints nothing.
    """
    reset_reports_for_test()
    t = _LegacyIgnoringDeadline()
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        t,
    )
    c._worker.stop()
    c.capture_span(_span("only"))
    capsys.readouterr()
    c.close(-1.0)
    assert t.shipped == ["only"], "a legacy transport was denied an envelope it could ship"
    assert c._lost == 0, "a delivered span was counted as lost"
    assert c._dropped == 0
    assert capsys.readouterr().err == "", "wardex reported a loss that did not happen"


class _IgnoresTheDeadlineAndDelivers(Transport):
    """Takes `timeout=`, ignores it entirely, always delivers. A legal
    transport, and the one every attempt to PREDICT a loss got wrong."""

    def __init__(self):
        self.shipped: list[str] = []

    def export(self, envelope: InternalEnvelope, *, timeout=None) -> None:
        self.shipped.extend(s.name for s in envelope.spans)


def test_a_transport_that_ignores_a_spent_deadline_is_never_treated_as_a_loss(capsys):
    """Delivering unconditionally is legal, so it must cost nothing — on either
    path. On close() a report would be a lie; on flush() putting the spans back
    would be worse than a lie, because the next drain would ship them AGAIN and
    the backend would see the same span twice."""
    reset_reports_for_test()
    t = _IgnoresTheDeadlineAndDelivers()
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        t,
    )
    c._worker.stop()
    c.capture_span(_span("a"))
    capsys.readouterr()
    c.flush(0.0)  # the whole budget is gone before the transport is reached
    assert t.shipped == ["a"], "the envelope never reached a transport that would deliver it"
    assert list(c._buffer.spans) == [], "a delivered batch was put back and will ship twice"
    c.flush(5.0)
    assert t.shipped == ["a"], "the delivered span was shipped a second time"
    c.close(5.0)
    assert c._lost == 0 and c._dropped == 0
    assert capsys.readouterr().err == "", "wardex reported a loss that did not happen"


def test_a_declined_flush_gives_the_spans_back_so_the_next_drain_ships_them(capsys):
    """The contract `_drain`'s docstring states and `flush(0.0)` broke.

    A non-final drain that does not ship costs nothing — that is why declining
    is safe everywhere except close(). But the promise only ever held for the
    declined *acquire*: an acquire that SUCCEEDED with no budget left took the
    slot, swapped the spans out, handed the transport 0.0 and lost them, with
    nothing counted and nothing said. main lost them the same way. The spans
    belong back in the buffer, in order, for the next drain."""
    reset_reports_for_test()
    t = _DeadlineHonouring()
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        t,
    )
    c._worker.stop()
    c.capture_span(_span("a"))
    c.capture_span(_span("b"))
    capsys.readouterr()

    c.flush(0.0)  # acquires the slot, then has nothing left to send with
    assert t.shipped == [], "the transport claimed a delivery it declined"
    assert [s.name for s in c._buffer.spans] == ["a", "b"], (
        f"a declined flush lost the batch: {[s.name for s in c._buffer.spans]}"
    )
    assert c._lost == 0, "a recoverable flush was reported as a shutdown loss"
    assert c._dropped == 0, "a returned batch was counted as a buffer overflow"
    assert capsys.readouterr().err == "", "a recoverable flush reported a loss"

    c.flush(5.0)
    assert t.shipped == ["a", "b"], f"the next drain did not ship the returned batch: {t.shipped}"


def test_a_returned_batch_goes_in_front_of_spans_captured_while_it_was_out():
    """Ordering, which the export lock exists to preserve: the returned batch
    predates everything captured since the swap, so it goes on the FRONT and
    oldest-first. Appending it instead would put the tail on the wire ahead of
    the head."""
    t = _DeadlineHonouring()
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        t,
    )
    c._worker.stop()
    c.capture_span(_span("a"))
    c.capture_span(_span("b"))
    c.flush(0.0)  # declined; a and b come back
    c.capture_span(_span("c"))
    c.flush(5.0)
    assert t.shipped == ["a", "b", "c"], f"capture order was not preserved: {t.shipped}"


def test_a_returned_batch_yields_to_the_buffer_cap_instead_of_overflowing_it():
    """A returned batch must not smuggle the buffer past `max_buffer_spans`:
    the bound would then be one a bounded flush against a declining transport
    could exceed at will. What does not fit is a buffer-full drop and is counted
    as one — on `_dropped`, which is what that word means — never a silent
    disappearance. Drop-oldest, as everywhere else, so the batch (which IS the
    oldest) yields rather than evicting live spans.

    The buffer refills WHILE the batch is out by the one deterministic route
    there is: `before_send` is host code, it runs after the swap and before the
    transport, and nothing stops it capturing. A background thread would race;
    this does not."""
    t = _DeadlineHonouring()
    client_box = {}

    def _captures_while_the_batch_is_out(envelope):
        client_box["c"].capture_span(_span("c"))
        client_box["c"].capture_span(_span("d"))
        return envelope

    c = Client(
        WardexConfig(
            limits=LimitsConfig(max_buffer_spans=3),
            before_send=_captures_while_the_batch_is_out,
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
        ),
        t,
    )
    client_box["c"] = c
    c._worker.stop()
    c.capture_span(_span("a"))
    c.capture_span(_span("b"))

    c.flush(0.0)  # declined; a and b come back to a buffer that now holds c, d
    resident = [s.name for s in c._buffer.spans]
    assert len(resident) <= 3, f"a returned batch pushed the buffer past its cap: {resident}"
    assert resident == ["b", "c", "d"], (
        f"the cap did not take the OLDEST of the returned batch: {resident}"
    )
    assert c._dropped == 1, f"the span that did not fit went uncounted: _dropped={c._dropped}"
    assert c._lost == 0, "a buffer-full drop was labelled a shutdown loss"


def test_a_before_send_that_outlives_close_s_budget_is_still_reported(capsys):
    """The door a clock check taken before `before_send` could never see.

    `before_send` is HOST code and runs INSIDE the deadline, after any check the
    drain could have made and before the transport is reached. So a budget that
    was healthy at the check is spent by the send: the transport is handed 0.0,
    skips, and the tail is out of the buffer, off the wire, uncounted, with
    stderr empty. Byte-for-byte the loss the previous two repairs each announced
    they had closed.

    Nothing about this test's timing is what makes it work — it works because
    the drain stopped predicting and started asking.
    """
    reset_reports_for_test()

    def _slow(envelope):
        time.sleep(0.3)  # outlives the budget below, from inside it
        return envelope

    t = _DeadlineHonouring()
    c = Client(
        WardexConfig(
            before_send=_slow,
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
        ),
        t,
    )
    c._worker.stop()
    c.capture_span(_span("only"))
    capsys.readouterr()
    c.close(0.1)  # ample at the acquire, spent by the time before_send returns
    err = capsys.readouterr().err

    assert t.timeouts == [0.0], f"the transport was not handed a spent budget: {t.timeouts}"
    assert t.shipped == [], "a spent budget somehow reached the backend"
    assert list(c._buffer.spans) == [], "unshippable spans left resident in a closed client"
    assert c._lost == 1, f"the span before_send outlived was not counted: _lost={c._lost}"
    assert "could not ship 1 buffered span(s)" in err, (
        f"a before_send that outlived the budget lost the span in silence: {err!r}"
    )


def test_a_before_send_that_outlives_a_flush_budget_gives_the_spans_back(capsys):
    """The same door on the non-final path, where the answer is different: the
    spans are recoverable, so they go back rather than being announced as lost.
    """
    reset_reports_for_test()

    def _slow(envelope):
        time.sleep(0.3)
        return envelope

    t = _DeadlineHonouring()
    c = Client(
        WardexConfig(
            before_send=_slow,
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
        ),
        t,
    )
    c._worker.stop()
    c.capture_span(_span("only"))
    capsys.readouterr()
    c.flush(0.1)
    assert t.shipped == [], "the transport claimed a delivery it declined"
    assert [s.name for s in c._buffer.spans] == ["only"], "a recoverable flush lost the span"
    assert c._lost == 0 and c._dropped == 0
    assert capsys.readouterr().err == "", "a recoverable flush reported a loss"


class _HostileExportAttribute(Transport):
    """`export` is a property that raises something `inspect.signature` does not
    catch. `Transport` is public and `_transport` is reassignable at runtime, so
    the client's every reach into it has to be inside a handler."""

    export = property(lambda self: (_ for _ in ()).throw(RuntimeError("hostile export")))


def test_a_hostile_transport_attribute_never_raises_into_close_or_flush():
    """wardex may not raise into the host, and reading `.export` off a
    caller-supplied `Transport` is host code: it can be a property, a
    descriptor, or a `__getattr__`, and it can raise anything.

    Every such reach is inside a handler now — the signature probe swallows it
    and answers "cannot take timeout=", the call itself lands in `_drain`'s
    fail-closed block. A probe performed from a SECOND site outside that block,
    which is what deciding in advance whether a send would happen required,
    turned `close(-1.0)` into a RuntimeError in the host's shutdown path.

    Driven through `_transport` reassignment, because that is the route the memo
    in `Client.__init__` says makes this reachable in a live process.

    A span is captured before EVERY call: a drain with an empty buffer returns
    long before it touches the transport, so a version of this test that
    captured once would exercise the escape only on the first call and pass over
    a wide-open close()."""
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        _Recording(),
    )
    c._worker.stop()
    c._transport = _HostileExportAttribute()  # swapped at runtime, as hosts do
    c.capture_span(_span("only"))
    c.flush(0.0)  # must not raise
    c.capture_span(_span("only"))
    c.flush(5.0)  # must not raise
    c.capture_span(_span("only"))
    c.close(-1.0)  # must not raise — the spent-budget shape that used to escape


def test_a_hostile_transport_attribute_never_raises_out_of_the_constructor():
    """The same reach, at the other site: `Client.__init__` probes the signature
    of whatever transport `init()` was handed. An `export` that raises on
    attribute access took `wardex.init()` down with it, which is the one thing an
    observability SDK may never do."""
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        _HostileExportAttribute(),
    )
    c._worker.stop()
    c.capture_span(_span("only"))
    c.close(5.0)  # must not raise on the ordinary budget either


def test_hostile_flush_timeout_never_reaches_the_host():
    """`flush(timeout)` is public API, so `timeout` is application input. Every
    value a host can pass must be handled or ignored — never raised back at it.
    float('nan') used to reach RLock.acquire() as a ValueError and a non-number
    the deadline arithmetic as a TypeError, both outside every handler."""
    t = _TimeoutRecording()
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        t,
    )
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
        c = Client(
            WardexConfig(
                backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
            ),
            t,
        )
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
    c = Client(
        WardexConfig(
            before_send=_slow_before_send,
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
        ),
        t,
    )
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

    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        _Idle(),
    )
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
