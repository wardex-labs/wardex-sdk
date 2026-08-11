"""What a flush() with no argument is allowed to wait for, and what that read costs.

`flush()` and `close()` stopped sharing a default: a bare `flush()` follows the
transport's own configured timeout, `close()` stays bounded at 5s. Three things
have to hold for that to be safe rather than merely convenient, and each one is
watched here:

  * the number a bare `flush()` picks up really is the transport's, end to end,
    all the way into `urlopen` -- otherwise the fix is a docstring;
  * the sentinel that means "ask the transport" is not `None`, which already
    means "unbounded" and belongs to the periodic worker alone;
  * asking the transport is a reach into HOST code, so every way that read can
    go wrong ends in the 5s fallback rather than in an exception the host sees.

Plus the two other host-facing reaches this batch closed: `transport.close()`
inside a handler, and a drain that outlives `close()` reporting its batch
instead of parking it in a client that can never ship again.
"""

from __future__ import annotations

import threading

import pytest

from wardex_sdk import _hub
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._client import (
    _DEFAULT_TIMEOUT,
    _FOLLOW_TRANSPORT_TIMEOUT,
    Client,
    _configured_transport_timeout,
)
from wardex_sdk._config import BackendConfig, BatchingConfig, WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._types import (
    Envelope,
    InternalSpan,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport._base import UNDELIVERED, Transport
from wardex_sdk.transport._otlp_http import OtlpHttpTransport


def _span(name="s"):
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name=name,
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


def _client(transport, **cfg):
    """A client whose only drains are the ones the test asks for."""
    c = Client(
        WardexConfig(
            **cfg,
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0),
        ),
        transport,
    )
    c._worker.stop()
    return c


class _TimeoutRecording(Transport):
    """Records the budget it is handed, and says what it was configured for."""

    def __init__(self, timeout: float = 10.0):
        self.export_timeout = timeout
        self.timeouts: list[float | None] = []

    def export(self, envelope: Envelope, *, timeout: float | None = None) -> object | None:
        self.timeouts.append(timeout)
        return None


# -- 1. a bare flush() follows the transport, end to end ---------------------


class _SevenSecondBackend:
    """A stand-in for `urlopen` against a backend that answers in seven seconds.

    The scenario the old 5s default regressed, expressed as a predicate rather
    than as a seven-second sleep: this backend replies iff the socket was given
    at least seven seconds to wait for it, and raises the same `TimeoutError`
    the real one would otherwise. Wall-clock sleeping would pin the property to
    a stopwatch; what is actually under test is which NUMBER reaches the socket,
    and that is exact.
    """

    def __init__(self) -> None:
        self.granted: list[float | None] = []
        self.delivered = 0

    def __call__(self, req, timeout=None):
        self.granted.append(timeout)
        if timeout is not None and timeout < 7.0:
            raise TimeoutError("timed out")
        self.delivered += 1
        return _Response()


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patch_urlopen(monkeypatch) -> _SevenSecondBackend:
    import urllib.request

    backend = _SevenSecondBackend()
    monkeypatch.setattr(urllib.request, "urlopen", backend)
    return backend


def test_a_bare_flush_delivers_to_a_backend_slower_than_the_old_default(monkeypatch):
    """The regression, stated as the scenario it broke.

    A transport configured for 10s and a backend that answers in 7 shipped
    before the drain was bounded and dropped after: the 5s default budget narrowed the
    POST through `min()`, the socket gave up at 5s, and an attempted POST is
    deliberately never re-queued -- so the spans were gone, silently. A bare
    `flush()` means "send what you have, I will wait", so it must not cap the
    POST below the number the host already chose for exactly this.
    """
    backend = _patch_urlopen(monkeypatch)
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0)
    c = _client(t)
    try:
        c.capture_span(_span("a"))
        c.flush()  # no argument: the transport's own 10s, not a 5s cap on it
        assert backend.delivered == 1, (
            f"a bare flush() dropped an envelope the transport had 10s for; "
            f"the socket was given {backend.granted}"
        )
        assert backend.granted[0] is not None and backend.granted[0] > 7.0, backend.granted
    finally:
        c.close(1.0)


def test_an_explicit_flush_timeout_is_still_a_real_bound(monkeypatch):
    """The bounded-drain win, and the half of it that must survive: naming a number is
    naming a wall-clock bound, whatever the transport was configured for. If
    following the transport leaked into the explicit path, the signal handler's
    `flush(2.0)` would go back to waiting out the transport's 10s."""
    backend = _patch_urlopen(monkeypatch)
    c = _client(OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0))
    try:
        c.capture_span(_span("a"))
        c.flush(2.0)
        assert backend.granted, "the POST was never attempted"
        assert 1.0 < backend.granted[0] <= 2.0, (
            f"flush(2.0) let the socket wait {backend.granted[0]}s"
        )
        assert backend.delivered == 0, "the 7s backend answered a 2s socket"
    finally:
        c.close(1.0)


def test_an_explicit_flush_of_the_old_default_is_taken_at_its_word(monkeypatch):
    """The sentinel is checked by IDENTITY, not by value. A host that passes 5.0
    by hand named a number, and naming one is the whole difference between the
    two readings -- an `==` check here would silently upgrade it to the
    transport's 10s and take the bound away."""
    backend = _patch_urlopen(monkeypatch)
    c = _client(OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0))
    try:
        c.capture_span(_span("a"))
        c.flush(_DEFAULT_TIMEOUT)  # the float 5.0, not the sentinel
        assert backend.granted and backend.granted[0] <= _DEFAULT_TIMEOUT, backend.granted
        assert backend.delivered == 0, "an explicit 5.0 was widened to the transport's 10s"
    finally:
        c.close(1.0)


def test_close_keeps_its_own_tight_default_and_does_not_follow_the_transport(monkeypatch):
    """close() is the other operation. It runs when the process is going away --
    this bound exists because an unbounded one ate a Kubernetes termination
    grace period -- so its default stays 5s even under a transport configured for 10,
    and what it cannot ship it reports."""
    backend = _patch_urlopen(monkeypatch)
    c = _client(OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0))
    c.capture_span(_span("a"))
    c.close()  # no argument: 5s, deliberately not the transport's 10
    assert backend.granted, "close() never attempted the POST"
    assert 4.0 < backend.granted[0] <= _DEFAULT_TIMEOUT, (
        f"close() followed the transport instead of staying bounded: {backend.granted}"
    )


def test_the_module_level_flush_carries_the_sentinel_through_to_the_client(monkeypatch):
    """`wardex.flush()` is the call hosts actually make. Its default has to be
    the same sentinel and has to be forwarded by identity, or the fix stops at
    `Client.flush` and the public entry point keeps capping the POST at 5s."""
    import wardex_sdk

    backend = _patch_urlopen(monkeypatch)
    c = _client(OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0))
    _hub.set_client(c)
    try:
        c.capture_span(_span("a"))
        wardex_sdk.flush()
        assert backend.delivered == 1, (
            f"wardex.flush() capped the POST below the transport's own timeout: {backend.granted}"
        )
    finally:
        _hub.set_client(None)
        c.close(1.0)


# -- 2. the sentinel must not collide with None ------------------------------


def test_the_sentinel_is_not_none_and_degrades_to_the_old_default():
    """`None` already means "unbounded" inside `_drain`, it is the periodic
    worker's contract, and the lock-order note in `Client.__init__` depends on
    no other caller ever passing it: an unbounded acquire from a caller holding
    the buffer lock hangs both threads. So the sentinel is its own type.

    And it is a `float` carrying the old default, so a future branch that
    forgets the identity check degrades to yesterday's behaviour instead of
    handing `RLock.acquire` a bare object."""
    assert _FOLLOW_TRANSPORT_TIMEOUT is not None
    assert isinstance(_FOLLOW_TRANSPORT_TIMEOUT, float)
    assert float(_FOLLOW_TRANSPORT_TIMEOUT) == _DEFAULT_TIMEOUT


def test_none_still_means_unbounded_on_the_periodic_path():
    """The meaning the sentinel exists to leave alone: the background worker
    imposes no deadline, so the transport hears `None` and uses its own
    configured timeout. Handing it a number would clamp the one path that ships
    data with nobody waiting on it."""
    t = _TimeoutRecording(timeout=10.0)
    c = _client(t)
    try:
        c.capture_span(_span("a"))
        c._drain(None)  # the periodic worker's call, verbatim
        assert t.timeouts == [None], (
            f"the unbounded periodic drain imposed a deadline: {t.timeouts}"
        )
    finally:
        c.close(1.0)


def test_following_the_transport_is_a_real_budget_and_never_the_unbounded_one():
    """ "Ask the transport" must be implemented as a NUMBER, not as "no deadline".

    The two look interchangeable through `OtlpHttpTransport`, which would use
    its own 10s either way -- and they are not: `_drain(None)` also takes the
    UNBOUNDED export-slot acquire, and a bare `flush()` is reachable from a
    caller holding the buffer lock (the signal handler), which is the AB-BA
    deadlock the lock-order note in `Client.__init__` says only the periodic
    worker may risk."""
    t = _TimeoutRecording(timeout=10.0)
    c = _client(t)
    try:
        c.capture_span(_span("a"))
        c.flush()
        assert t.timeouts and t.timeouts[0] is not None, (
            "a bare flush() became the periodic worker's unbounded drain"
        )
        assert 9.0 < t.timeouts[0] <= 10.0, (
            f"a bare flush() did not follow the transport's 10s: {t.timeouts}"
        )
    finally:
        c.close(1.0)


def test_a_host_passing_none_gets_a_bounded_flush_not_an_unbounded_one():
    """`None` from application code is not the sentinel and not the worker's
    contract either: it sanitizes to the default budget. If `flush(None)` ever
    reached `_drain` as None it would take the UNBOUNDED acquire -- from a
    caller that may be holding the buffer lock, which is the deadlock the
    lock-order comment is written to prevent."""
    t = _TimeoutRecording(timeout=10.0)
    c = _client(t)
    try:
        c.capture_span(_span("a"))
        c.flush(None)
        assert t.timeouts and t.timeouts[0] is not None, (
            "flush(None) reached the transport as an unbounded drain"
        )
        assert t.timeouts[0] <= _DEFAULT_TIMEOUT, (
            f"flush(None) was read as the sentinel and followed the transport: {t.timeouts}"
        )
    finally:
        c.close(1.0)


# -- 3. reading transport.export_timeout is a reach into host code -----------


class _NoTimeoutAttribute(Transport):
    """A perfectly legal transport: `export_timeout` left undeclared."""

    def export(self, envelope: Envelope, *, timeout: float | None = None) -> object | None:
        return None


def _hostile(value):
    """A transport whose `export_timeout` is `value` -- or, for a callable,
    whatever it raises on attribute access."""

    class _Hostile(_NoTimeoutAttribute):
        export_timeout = property(value) if callable(value) else value

    return _Hostile()


def _raises(exc):
    return lambda self: (_ for _ in ()).throw(exc)


def test_every_unusable_transport_timeout_falls_back_to_the_default():
    """`Transport` is public, so `export_timeout` may be absent, a property
    that raises, a `__getattr__` returning something `float()` chokes on, or a number
    that is no use as a deadline. Rejecting means raising, and wardex may not
    raise into a host over a flush -- so every unusable answer is the 5s
    fallback, and the read, the conversion and the validation all sit inside the
    one try."""
    cases = {
        "no attribute at all": _NoTimeoutAttribute(),
        "a property that raises": _hostile(_raises(RuntimeError("hostile timeout"))),
        "a non-numeric string": _hostile("ten seconds"),
        "an object float() chokes on": _hostile(object()),
        "NaN": _hostile(float("nan")),
        "zero": _hostile(0.0),
        "negative": _hostile(-1.0),
    }
    for label, transport in cases.items():
        assert _configured_transport_timeout(transport) == _DEFAULT_TIMEOUT, (
            f"{label} did not fall back to the default budget"
        )


def test_an_enormous_transport_timeout_is_clamped_to_what_a_timed_acquire_takes():
    """The budget goes on to `RLock.acquire(timeout=...)`, which raises above
    `threading.TIMEOUT_MAX`. A transport configured for a year is not a reason
    to raise into the host."""
    assert _configured_transport_timeout(_hostile(1e18)) == threading.TIMEOUT_MAX


def test_a_usable_transport_timeout_is_taken_as_it_is():
    """The control: the fallback must not have eaten the feature."""
    assert _configured_transport_timeout(_hostile(10.0)) == 10.0
    assert _configured_transport_timeout(_hostile(3)) == 3.0  # ints are numbers too


def test_a_hostile_transport_timeout_never_raises_into_a_bare_flush():
    """The same table at the public boundary, which is where it matters: a bare
    `flush()` is the only caller that reads this attribute, and it is called
    from host code that must never see an exception out of it."""
    for value in (_raises(RuntimeError("boom")), "ten seconds", float("nan"), 0.0, -1.0):
        t = _hostile(value)
        c = _client(t)
        try:
            c.capture_span(_span("a"))
            c.flush()  # must not raise
        finally:
            c.close(1.0)


def test_a_keyboard_interrupt_from_the_timeout_property_still_reaches_the_host():
    """Deliberate, and therefore watched: `KeyboardInterrupt` and
    `CancelledError` are BaseExceptions and pass straight through
    `except Exception`. A host tearing this thread down is not a transport whose
    timeout we failed to read, and swallowing it would leave the host unable to
    interrupt its own process."""
    t = _hostile(_raises(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _configured_transport_timeout(t)

    c = _client(t)
    try:
        c.capture_span(_span("a"))
        with pytest.raises(KeyboardInterrupt):
            c.flush()  # the bare flush is the only caller that makes the read
    finally:
        c.close(1.0)


# -- 5. transport.close() is host code too -----------------------------------


class _CloseRaises(_NoTimeoutAttribute):
    def close(self, timeout: float = 5.0) -> None:
        raise RuntimeError("teardown blew up")


class _HostileCloseAttribute(_NoTimeoutAttribute):
    """`close` is a property that raises on ACCESS, so a `try` wrapped around
    the call alone would not have covered it."""

    close = property(_raises(RuntimeError("hostile close")))


def test_a_transport_whose_close_raises_cannot_reach_the_hosts_shutdown_path(capsys):
    """`wardex.close()` is called from `atexit` hooks and `finally` blocks. A
    third-party transport whose socket teardown throws turned that into a raise
    out of the host's exit path -- the one thing an observability SDK may never
    do. Fail-silent, with the same debug line `_flush_transport` uses."""
    for transport in (_CloseRaises(), _HostileCloseAttribute()):
        c = _client(transport, debug=True)
        c.capture_span(_span("a"))
        capsys.readouterr()
        c.close(1.0)  # must not raise
        assert "transport close failed" in capsys.readouterr().err, (
            f"{type(transport).__name__} closed without a word on debug"
        )


def test_a_keyboard_interrupt_from_transport_close_still_reaches_the_host():
    """The other half of the same rule: fail-silent covers `Exception`, never a
    host tearing the process down."""

    class _CloseInterrupts(_NoTimeoutAttribute):
        def close(self, timeout: float = 5.0) -> None:
            raise KeyboardInterrupt()

    c = _client(_CloseInterrupts())
    c.capture_span(_span("a"))
    with pytest.raises(KeyboardInterrupt):
        c.close(1.0)


# -- 7. a drain that outlives close() ----------------------------------------


class _Declines(_NoTimeoutAttribute):
    """Always says it did not send. The client's answer to that is different on
    a live client (keep them for the next drain) and on a closed one (there is
    no next drain), which is the whole point of the tests below."""

    def __init__(self):
        self.taken: list[str] = []

    def export(self, envelope: Envelope, *, timeout: float | None = None) -> object | None:
        self.taken.extend(s.name for s in envelope.spans)
        return UNDELIVERED


class _BlocksThenDeclines(_Declines):
    def __init__(self, in_export: threading.Event, release: threading.Event):
        super().__init__()
        self._in_export = in_export
        self._release = release
        self.released = False

    def export(self, envelope: Envelope, *, timeout: float | None = None) -> object | None:
        self.taken.extend(s.name for s in envelope.spans)
        self._in_export.set()
        self.released = self._release.wait(timeout=5.0)
        return UNDELIVERED


def test_a_drain_still_running_when_close_lands_does_not_re_seed_the_closed_client(capsys):
    """The race, which needs no post-close flush at all.

    One thread is inside a non-final drain -- spans already swapped OUT of the
    buffer -- when `close()` runs: `_abandon` empties a buffer that is already
    empty, finds nothing to report, and the client is closed. The drain then
    comes back with a batch the transport declined and, on the non-final path,
    hands it to `_return_to_buffer`. Without the check that batch lands in a
    client that will never drain again: resident, uncounted, unreported, with
    `_spans` still listing it as pending -- the exact state `_abandon`'s
    docstring says it exists to prevent.

    The check lives INSIDE the buffer lock because that is what makes this
    decidable: `close()` sets `_closed` strictly before `_abandon` can empty
    anything, so either this batch goes back and `_abandon` collects it, or
    `_abandon` ran first and `_closed` is already True. A check outside the lock
    would sit in the window between the two.
    """
    reset_reports_for_test()
    in_export, release = threading.Event(), threading.Event()
    t = _BlocksThenDeclines(in_export, release)
    c = _client(t)
    c.capture_span(_span("a"))
    capsys.readouterr()

    drain = threading.Thread(target=c.flush, args=(5.0,), daemon=True)
    drain.start()
    assert in_export.wait(timeout=5.0), "the drain never reached the transport"

    c.close(0.1)  # cannot get the slot; _abandon finds the buffer already empty
    release.set()
    drain.join(timeout=5.0)
    assert not drain.is_alive(), "the flush never returned"
    assert t.released, "the transport was never let go"

    err = capsys.readouterr().err
    assert list(c._spans) == [], (
        f"a closed client was re-seeded with spans it can never ship: {[s.name for s in c._spans]}"
    )
    assert c._lost == 1, f"the batch the closed client kept was not counted: _lost={c._lost}"
    assert c._dropped == 0, "a shutdown loss was labelled a buffer overflow"
    assert "could not ship 1 buffered span(s)" in err, f"the batch vanished in silence: {err!r}"
    assert "close() had already run" in err, err
    reset_reports_for_test()


def test_a_flush_that_outlives_close_on_one_thread_reports_rather_than_re_seeding(capsys):
    """The sequential shape of the same door, on a single thread and with no
    race to lose.

    `before_send_envelope` is HOST code and runs INSIDE the drain, after the swap. A host
    that closes wardex from there -- or from anything the drain calls -- returns
    into a flush whose batch now belongs to a closed client, which is `flush()`
    after `close()` with the two calls interleaved rather than merely adjacent.
    `flush` deliberately does not test `_closed`, so nothing upstream stops it;
    the buffer is where it is decided.
    """
    reset_reports_for_test()
    t = _Declines()
    box: dict[str, Client] = {}

    def _closes_wardex(envelope):
        box["c"].close(0.0)  # the host shuts down from inside the drain
        return envelope

    c = _client(t, before_send_envelope=_closes_wardex)
    box["c"] = c
    c.capture_span(_span("a"))
    capsys.readouterr()

    c.flush(5.0)
    err = capsys.readouterr().err

    assert t.taken == ["a"], f"the drain never reached the transport: {t.taken}"
    assert list(c._spans) == [], "a closed client was re-seeded by its own in-flight flush"
    assert c._lost == 1, f"the batch was neither shipped nor counted: _lost={c._lost}"
    assert "could not ship 1 buffered span(s)" in err, f"the batch vanished in silence: {err!r}"
    reset_reports_for_test()


def test_a_declined_flush_on_a_LIVE_client_still_gives_the_spans_back(capsys):
    """The control the two tests above need, and the direction that must not be
    over-corrected: on a client nobody has closed, a declined batch costs
    nothing and the next drain ships it. Reporting it here would be a false
    alarm -- and a false alarm is not free, because it burns the
    one-line-per-process key the real report needs."""
    reset_reports_for_test()
    t = _Declines()
    c = _client(t)
    try:
        c.capture_span(_span("a"))
        capsys.readouterr()
        c.flush(5.0)
        assert [s.name for s in c._spans] == ["a"], "a live client lost a recoverable batch"
        assert c._lost == 0, "a recoverable flush was reported as a shutdown loss"
        assert capsys.readouterr().err == "", "a recoverable flush reported a loss"
    finally:
        c.close(1.0)
        reset_reports_for_test()
