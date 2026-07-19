"""_lifecycle — atexit single registration, signal chaining, re-init teardown."""

import os
import signal
import threading

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub, _lifecycle
from wardex_sdk._client import Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._types import InternalEnvelope, InternalSpan, SpanContext, SpanId, TraceId
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _span():
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name="s",
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


def _client(transport=None, **cfg):
    return Client(WardexConfig(api_key="k", **cfg), transport or _Recording())


@pytest.fixture(autouse=True)
def clean_lifecycle():
    """Restore signal table and module state around every test."""
    _lifecycle._uninstall_signal_handlers()
    _lifecycle._current_client = None
    yield
    client = _lifecycle._current_client
    if client is not None:
        client.close()
    _lifecycle._uninstall_signal_handlers()
    _lifecycle._current_client = None
    _hub.reset_for_test()


def test_install_closes_previous_client_and_flushes_it():
    t1 = _Recording()
    first = _client(t1, flush_interval=3600.0)
    _lifecycle.install(first, first.config)
    first.capture_span(_span())
    second = _client(flush_interval=3600.0)
    _lifecycle.install(second, second.config)
    assert first._closed
    assert not first._worker.is_alive()
    assert sum(len(e.spans) for e in t1.envelopes) == 1
    assert _lifecycle.current_client() is second


def test_atexit_registered_exactly_once(monkeypatch):
    calls = []
    monkeypatch.setattr(_lifecycle.atexit, "register", lambda fn: calls.append(fn))
    monkeypatch.setattr(_lifecycle, "_atexit_registered", False)
    c1 = _client()
    _lifecycle.install(c1, c1.config)
    c2 = _client()
    _lifecycle.install(c2, c2.config)
    assert len(calls) == 1


def test_signal_handler_flushes_then_chains_to_callable_prev():
    seen = []
    prev = lambda signum, frame: seen.append(signum)  # noqa: E731
    old = signal.signal(signal.SIGINT, prev)
    try:
        t = _Recording()
        c = _client(t, flush_interval=3600.0)
        _lifecycle.install(c, c.config)
        c.capture_span(_span())
        _lifecycle._handler(signal.SIGINT, None)  # invoke directly — no real signal
        assert sum(len(e.spans) for e in t.envelopes) == 1  # flushed first
        assert seen == [signal.SIGINT]  # then chained
    finally:
        signal.signal(signal.SIGINT, old)


def test_signal_handler_reraises_default_action(monkeypatch):
    kills = []
    monkeypatch.setattr(_lifecycle.os, "kill", lambda pid, s: kills.append((pid, s)))
    c = _client(flush_interval=3600.0)
    _lifecycle.install(c, c.config)
    _lifecycle._prev_handlers[signal.SIGTERM] = signal.SIG_DFL
    _lifecycle._handler(signal.SIGTERM, None)
    assert kills == [(os.getpid(), signal.SIGTERM)]
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL


def test_signal_handler_respects_sig_ign(monkeypatch):
    kills = []
    monkeypatch.setattr(_lifecycle.os, "kill", lambda pid, s: kills.append((pid, s)))
    c = _client(flush_interval=3600.0)
    _lifecycle.install(c, c.config)
    _lifecycle._prev_handlers[signal.SIGTERM] = signal.SIG_IGN
    _lifecycle._handler(signal.SIGTERM, None)  # must neither kill nor raise
    assert kills == []


def test_flush_on_signals_false_leaves_signal_table_untouched():
    before = signal.getsignal(signal.SIGINT)
    c = _client(flush_on_signals=False)
    _lifecycle.install(c, c.config)
    assert signal.getsignal(signal.SIGINT) is before


def test_signal_install_skipped_off_main_thread():
    c = _client()
    done = threading.Event()

    def run():
        _lifecycle.install(c, c.config)
        done.set()

    threading.Thread(target=run).start()
    assert done.wait(timeout=5.0)
    assert not _lifecycle._signals_installed
    assert signal.getsignal(signal.SIGINT) is not _lifecycle._handler


def test_uninstall_leaves_foreign_handler_alone():
    c = _client()
    _lifecycle.install(c, c.config)
    mine = lambda signum, frame: None  # noqa: E731
    signal.signal(signal.SIGINT, mine)  # app overwrote us after install
    _lifecycle._uninstall_signal_handlers()
    assert signal.getsignal(signal.SIGINT) is mine
    signal.signal(signal.SIGINT, signal.default_int_handler)


def test_partial_signal_install_rolls_back(monkeypatch):
    real_signal = signal.signal

    def failing_signal(signum, handler):
        if signum == signal.SIGTERM and handler is _lifecycle._handler:
            raise OSError("no SIGTERM here")
        return real_signal(signum, handler)

    monkeypatch.setattr(_lifecycle.signal, "signal", failing_signal)
    before = signal.getsignal(signal.SIGINT)
    c = _client()
    _lifecycle.install(c, c.config)
    assert not _lifecycle._signals_installed
    assert _lifecycle._prev_handlers == {}
    assert signal.getsignal(signal.SIGINT) is before  # rolled back, not left as _handler


def test_signal_handler_chains_even_if_flush_raises(monkeypatch):
    seen = []
    prev = lambda signum, frame: seen.append(signum)  # noqa: E731
    old = signal.signal(signal.SIGINT, prev)
    try:
        c = _client()
        _lifecycle.install(c, c.config)

        # Force Client.flush itself to raise — _drain's internal guards must not
        # be what saves the chain; the handler's own try/except must.
        def boom(timeout: float = 5.0) -> None:
            raise ValueError("boom")

        monkeypatch.setattr(c, "flush", boom)
        _lifecycle._handler(signal.SIGINT, None)  # must not raise, must still chain
        assert seen == [signal.SIGINT]
    finally:
        signal.signal(signal.SIGINT, old)


def test_init_twice_closes_previous_client():
    t1 = _Recording()
    wardex.init(transport=t1, api_key="k", flush_interval=3600.0)
    first = _hub.get_client()
    first.capture_span(_span())
    wardex.init(transport=_Recording(), api_key="k", flush_interval=3600.0)
    assert first._closed
    assert sum(len(e.spans) for e in t1.envelopes) == 1


def test_reinit_uninstalls_interceptors_before_closing_previous_client():
    """I2/I3 regression.

    Re-init (_lifecycle.install with a live previous client) must uninstall
    interceptors bound to the previous client before closing it — mirroring
    wardex.close()'s deliberate ordering. Otherwise interceptors stay bound to
    a closed (no-op) client forever (I2), and anything an interceptor's
    uninstall() flushes via capture_span (e.g. a pending WS session) is lost
    because the client already rejects captures (I3).
    """
    from wardex_sdk.interceptors._base import InterceptorInterface
    from wardex_sdk.interceptors._registry import get_registry

    class _FakeInterceptor(InterceptorInterface):
        """Minimal interceptor matching the registry's expected interface."""

        def __init__(self) -> None:
            self.client: Client | None = None

        def name(self) -> str:
            return "fake-lifecycle"

        def install(self, client: Client | None) -> None:
            self.client = client

        def uninstall(self) -> None:
            # Simulates flushing a pending WS session on uninstall: this must
            # succeed, which requires uninstall() to run before the bound
            # client is closed.
            self.client.capture_span(_span())

    registry = get_registry()
    try:
        t1 = _Recording()
        first = _client(t1, flush_interval=3600.0)
        _lifecycle.install(first, first.config)
        fake = _FakeInterceptor()
        registry.install(fake, first)

        second = _client(flush_interval=3600.0)
        _lifecycle.install(second, second.config)  # re-init path

        assert not registry.is_installed("fake-lifecycle")  # uninstall_all ran
        # The span captured inside uninstall() reached client A's transport —
        # proof uninstall() ran (and its capture succeeded) before close().
        assert sum(len(e.spans) for e in t1.envelopes) == 1
    finally:
        registry.uninstall_all()


def test_reinit_uninstalls_adapters_before_closing_previous_client():
    """Regression: re-init (_lifecycle.install with a live previous client)
    must uninstall adapters bound to the previous client, not just
    interceptors. Otherwise a previously installed adapter stays bound to
    the closed client, and since AdapterRegistry.install() is idempotent by
    name, the next init() silently no-ops for that adapter.
    """
    from wardex_sdk.adapters._base import AdapterInterface
    from wardex_sdk.adapters._registry import get_registry

    class _FakeAdapter(AdapterInterface):
        def __init__(self) -> None:
            self.uninstalled = 0

        def name(self) -> str:
            return "fake-lifecycle-adapter"

        def install(self, client: Client | None) -> None:
            pass

        def uninstall(self) -> None:
            self.uninstalled += 1

    registry = get_registry()
    try:
        first = _client(flush_interval=3600.0)
        _lifecycle.install(first, first.config)
        fake = _FakeAdapter()
        registry.install(fake, first)

        second = _client(flush_interval=3600.0)
        _lifecycle.install(second, second.config)  # re-init path

        assert fake.uninstalled == 1
        assert not registry.is_installed("fake-lifecycle-adapter")
    finally:
        registry.uninstall_all()
