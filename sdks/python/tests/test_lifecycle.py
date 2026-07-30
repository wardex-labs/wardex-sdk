"""_lifecycle — atexit single registration, signal chaining, re-init teardown."""

import json
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
from wardex_sdk.adapters._base import AdapterInterface
from wardex_sdk.adapters._registry import get_registry as get_adapter_registry
from wardex_sdk.assembly import Limitation
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


_SIGNALS = (signal.SIGINT, signal.SIGTERM)

# The dispositions this pytest session started with, read at import — i.e. at
# collection, before any test anywhere has run. The audit at the bottom of this
# file compares the table against these.
_DISPOSITIONS_AT_IMPORT = {signum: signal.getsignal(signum) for signum in _SIGNALS}


@pytest.fixture(autouse=True)
def real_signal_table():
    """Put the PROCESS's real SIGINT/SIGTERM dispositions back after every test.

    Separate from `clean_lifecycle`, which restores wardex's *module* state,
    and depended on by it so that this one is set up first and torn down LAST —
    the poisoning below happens *inside* `clean_lifecycle`'s teardown, so a
    fixture that finalized before it would restore the table and then watch it
    be re-broken.

    Several tests here drive `_lifecycle._handler` by writing a disposition
    into `_lifecycle._prev_handlers[SIGTERM]` by hand, which is the only way to
    exercise the SIG_DFL and SIG_IGN branches without actually killing pytest.
    Both then leak into the real table: `_handler` calls
    `signal.signal(signum, SIG_DFL)` itself on the SIG_DFL branch, and on the
    SIG_IGN branch `_uninstall_signal_handlers()` — which restores whatever
    `_prev_handlers` holds — writes SIG_IGN into it at teardown.

    SIG_IGN is the one that hurts, because it survives `exec`: every
    subprocess spawned for the rest of the session inherits an ignored SIGTERM.
    `test_batching_integration.py::test_sigterm_flushes_and_preserves_exit_code`
    sends SIGTERM to a child and waits for it to die, so it hung for 15s and
    failed — in a file that had done nothing wrong, from a fault it could not
    see.

    The snapshot is taken AFTER dropping any wardex handler still in the table,
    not before. Earlier files in the session `init()` without closing, so
    `_lifecycle._handler` is frequently sitting in the table when this file
    starts; snapshotting first would save that and hand it back after every
    test, which is the opposite of the job — and it would break the tests below
    that assert `_handler` is *not* installed.
    """
    _lifecycle._uninstall_signal_handlers()
    saved = {signum: signal.getsignal(signum) for signum in _SIGNALS}
    yield
    for signum, handler in saved.items():
        # None means the disposition was installed from C and cannot be written
        # back through the signal module; nothing to restore it to.
        if handler is not None:
            signal.signal(signum, handler)


@pytest.fixture(autouse=True)
def clean_lifecycle(real_signal_table):
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


# Deliberately the last test in the file: it audits what everything above it
# left behind, so anything appended after it moves out of its view.
def test_this_file_leaves_the_process_signal_table_clean():
    """The signal tests above must not charge the rest of the session for it.

    A test file that installs signal handlers is writing to interpreter-global
    state that no other file can see it holding, and SIG_IGN in particular is
    inherited across `exec` — so a leak here reaches every subprocess any later
    test spawns, and lands as a 15-second timeout in a file that never touched
    signals. That is a test suite reporting a fault in the wrong place, which
    is worse than reporting none.

    Compared against the session's dispositions rather than against the ones
    this file happened to inherit, because a wardex handler left in the table
    by an earlier file is a leak too — this file's fixture clears it, and the
    stricter comparison is what keeps that from silently regressing.

    This asserts the disposition rather than the fixture, because the fixture
    is only the current answer: what must stay true is that the file gives the
    process back a signal table nobody has to work around.
    """
    now = {signum: signal.getsignal(signum) for signum in _SIGNALS}
    assert now == _DISPOSITIONS_AT_IMPORT, (
        f"signal dispositions changed: {_DISPOSITIONS_AT_IMPORT} -> {now}"
    )


# ==========================================================================
# Shutdown reaches the units, not only the buffer
# ==========================================================================
#
# `client.flush()` sends what is already IN the buffer. A run still in flight
# has nothing there — its root span does not exist yet, because the thing that
# creates it is the close. So a shutdown that only flushes is a shutdown that
# loses exactly the span an interrupted run is about.


class _UnitAdapter(AdapterInterface):
    """An adapter holding one live session, on the real assembler."""

    def __init__(self, client):
        from wardex_sdk.adapters._assembler import SessionAssembler

        self._asm = SessionAssembler(client)
        self.close_units_calls: list = []

    def name(self) -> str:
        return "test_unit_adapter"

    def install(self, client) -> None:
        self._asm.on_outbound(
            1,
            json.dumps(
                {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "go"}}
            ),
        )
        self._asm.on_inbound(
            1,
            {"type": "system", "subtype": "init", "session_id": "s-1", "model": "claude-sonnet-5"},
        )

    def uninstall(self) -> None:
        self._asm.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)

    def close_units(self, *, marker: Limitation) -> None:
        self.close_units_calls.append(marker)
        self._asm.close_all_sessions(marker=marker)


def _root_names(transport):
    return [s.name for e in transport.envelopes for s in e.spans]


@pytest.fixture
def unit_adapter():
    """A registered adapter with a live session; always deregistered after."""
    registry = get_adapter_registry()
    made: list = []

    def build(client):
        adapter = _UnitAdapter(client)
        registry.install(adapter, client)
        made.append(adapter)
        return adapter

    yield build
    registry._installed.pop("test_unit_adapter", None)


def test_teardown_closes_units_while_the_client_can_still_send_them(unit_adapter):
    """The ordering inside `_teardown` is what makes this fix reachable at all.

    Adapters are uninstalled BEFORE `client.close()`, so a span the uninstall
    produces still has somewhere to go. Reverse the two lines and the teardown
    still runs, still walks every session, still builds every span — and
    `capture_span` drops each one on the floor, which is the same missing span
    this whole path exists to stop, arriving by a longer route.
    """
    transport = _Recording()
    client = _client(transport, flush_interval=3600.0)
    _lifecycle.install(client, client.config)
    unit_adapter(client)

    _lifecycle._teardown(client)

    assert "invoke_agent" in _root_names(transport)


def test_the_signal_handler_closes_live_units_when_the_process_is_about_to_die(unit_adapter):
    """SIGTERM under the default disposition never reaches atexit.

    The handler ends the process itself — it restores SIG_DFL and re-raises, so
    the interpreter never runs its exit hooks and the adapter is never
    uninstalled. That is the shape `docker stop` and a kubelet send. Closing the
    units here, before the flush that follows, is the only chance the run's span
    gets.
    """
    transport = _Recording()
    client = _client(transport, flush_interval=3600.0)
    _lifecycle.install(client, client.config)
    adapter = unit_adapter(client)

    _lifecycle._prev_handlers[signal.SIGTERM] = signal.SIG_IGN  # do not kill pytest
    _lifecycle._handler(signal.SIGTERM, None)

    assert adapter.close_units_calls == []  # SIG_IGN is not the dying disposition

    _lifecycle._prev_handlers[signal.SIGTERM] = signal.SIG_DFL
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "kill", lambda *a: None)
        _lifecycle._handler(signal.SIGTERM, None)

    assert adapter.close_units_calls == [Limitation.UNIT_INTERRUPTED]
    assert "invoke_agent" in _root_names(transport)


def test_the_signal_handler_leaves_units_open_when_the_app_carries_on(unit_adapter):
    """A callable prior handler means the app decided what a signal means, and
    the program may well keep running. Closing every live unit there would end
    sessions that are still being driven — reporting a run as interrupted while
    it goes on producing spans that now have no parent.
    """
    transport = _Recording()
    client = _client(transport, flush_interval=3600.0)
    _lifecycle.install(client, client.config)
    adapter = unit_adapter(client)

    seen: list = []
    _lifecycle._prev_handlers[signal.SIGTERM] = lambda signum, frame: seen.append(signum)
    _lifecycle._handler(signal.SIGTERM, None)

    assert seen == [signal.SIGTERM], "the chain to the app's handler broke"
    assert adapter.close_units_calls == []
    assert "invoke_agent" not in _root_names(transport)


def test_an_adapter_whose_close_units_raises_cannot_take_the_flush_with_it(unit_adapter):
    """This runs from a signal handler on a dying process. An exception escaping
    into `_handler` would skip the flush and lose the buffer as well as the
    units — one adapter's failure costing every other adapter's spans.
    """
    transport = _Recording()
    client = _client(transport, flush_interval=3600.0)
    _lifecycle.install(client, client.config)
    adapter = unit_adapter(client)

    def boom(*, marker):
        raise RuntimeError("adapter is broken")

    adapter.close_units = boom
    client.capture_span(_span())
    _lifecycle._prev_handlers[signal.SIGTERM] = signal.SIG_DFL
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "kill", lambda *a: None)
        _lifecycle._handler(signal.SIGTERM, None)  # must not raise

    assert transport.envelopes, "the flush was lost with the failing adapter"
