"""One owner for everything wardex installs — reset, reversibility, re-entry.

`test_lifecycle.py` next door asks what the atexit/signal POLICY does. This
file asks the structural questions that policy used to make unanswerable: that
the reset really empties every state the SDK holds, that install and uninstall
are inverses at the process seams, and that a shutdown holding the runtime's
lock cannot stop a signal handler from running.
"""

from __future__ import annotations

import signal
import socket
import ssl
import threading

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub, _runtime
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, BatchingPolicy, PropagationPolicy, WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._types import InternalEnvelope, InternalSpan, SpanContext, SpanId, TraceId
from wardex_sdk.adapters._base import AdapterInterface
from wardex_sdk.assembly import Limitation
from wardex_sdk.transport._base import Transport

_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class _Recording(Transport):
    def __init__(self) -> None:
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _span() -> InternalSpan:
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name="s",
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


def _client(transport: Transport | None = None, **cfg: object) -> Client:
    return Client(
        WardexConfig(backend=BackendConfig(api_key="k"), **cfg), transport or _Recording()
    )


def _ssl_seam():
    from wardex_sdk.interceptors._ssl import SSLInterceptor

    return SSLInterceptor()


class _FakeAdapter(AdapterInterface):
    def __init__(self) -> None:
        self.uninstalls = 0

    def name(self) -> str:
        return "fake-runtime-adapter"

    def install(self, client, ctx=None) -> None:  # noqa: ANN001
        return None

    def uninstall(self) -> None:
        self.uninstalls += 1


@pytest.fixture(autouse=True)
def empty_runtime():
    """A runtime with nothing in it, before and after — and a clean signal table.

    The dispositions are snapshotted after the reset for `test_lifecycle.py`'s
    reason: a wardex handler an earlier file left in the table is a leak, and
    saving it here would hand it back after every test in this one.
    """
    _hub.reset_for_test()
    saved = {signum: signal.getsignal(signum) for signum in _SIGNALS}
    yield
    _hub.reset_for_test()
    for signum, handler in saved.items():
        if handler is not None:
            signal.signal(signum, handler)


# --------------------------------------------------------------------------
# the reset empties EVERY state, not one of five
# --------------------------------------------------------------------------


def test_reset_for_test_empties_every_state_the_runtime_owns():
    """The five module globals this refactor consolidated, asserted one by one.

    Each was reachable only from the module that declared it, and
    `reset_for_test()` reset exactly one of them — the client. A test that
    installed a byte seam therefore charged the seam to every test behind it:
    `ssl.SSLSocket.send` stayed wrapped, the shared connect-timing probe stayed
    on `socket.socket.connect`, the adapter registry kept a name that makes the
    next `install()` of it a silent no-op, and a chained SIGINT handler stayed
    in the process signal table. Asserted individually rather than through one
    "is it clean" helper, because the failure this replaces was precisely that
    four of the five had nobody asking.
    """
    from wardex_sdk.adapters._registry import get_registry as adapter_registry
    from wardex_sdk.interceptors import _conn_timing
    from wardex_sdk.interceptors._registry import get_registry as interceptor_registry

    runtime = _runtime.runtime()
    client = _client(batching=BatchingPolicy(flush_interval=3600.0))
    adapter = _FakeAdapter()

    runtime.install(client, client.config)
    interceptor_registry().install(_ssl_seam(), client)
    adapter_registry().install(adapter, client)

    # 1..5 — every one of them non-empty before the reset, or the assertions
    # underneath measure nothing.
    assert runtime.client is client
    assert interceptor_registry().is_installed("ssl")
    assert adapter_registry().is_installed("fake-runtime-adapter")
    assert runtime._signals_installed and runtime._prev_handlers
    assert _conn_timing._shared_refcount > 0
    assert "connect" in socket.socket.__dict__  # the shared probe is on the host

    _hub.reset_for_test()

    assert runtime.client is None, "the client slot"
    assert not interceptor_registry().is_installed("ssl"), "the interceptor registry"
    assert not adapter_registry().is_installed("fake-runtime-adapter"), "the adapter registry"
    assert not runtime._signals_installed, "the signal handlers"
    assert runtime._prev_handlers == {}, "the chained dispositions"
    assert _conn_timing._shared_refcount == 0, "the shared timing refcount"
    assert _conn_timing._shared_store is None, "the shared timing store"
    assert "connect" not in socket.socket.__dict__, "the shared probe's own patch"
    assert client._closed, "the client was dropped without being closed"
    assert adapter.uninstalls == 1, "the adapter was dropped without being uninstalled"
    for signum in _SIGNALS:
        assert signal.getsignal(signum) is not _runtime._handler


def test_the_hub_and_the_runtime_read_one_client_slot():
    """There were two, set one after the other by `init()` and never together.

    Nothing kept them in step: a `close()` cleared neither, `reset_for_test()`
    cleared one, and the atexit path read the other — so a test that reset the
    hub left `atexit` holding a client the SDK had already forgotten.
    """
    client = _client(batching=BatchingPolicy(flush_interval=3600.0))
    _runtime.runtime().install(client, client.config)
    try:
        assert _hub.get_client() is client
        assert _runtime.current_client() is client

        _hub.set_client(None)
        assert _runtime.current_client() is None
    finally:
        # Clearing the slot is not closing the client, and the reset can no
        # longer find it — a dropped Client leaks its batch-worker thread for
        # the rest of the process.
        client.close()


# --------------------------------------------------------------------------
# install and uninstall are inverses
# --------------------------------------------------------------------------


def _process_seams() -> dict[str, object]:
    """Every host attribute a full `init()` can patch, by identity.

    The shape `testing.harness.AdapterSubject.seams` uses for an adapter, asked
    of the whole runtime: a restore that produced an EQUAL object would leave
    wardex's wrapper welded on for the life of the process, so nothing here is
    compared any other way.
    """
    import httpx

    return {
        "ssl.SSLSocket.send": ssl.SSLSocket.send,
        "ssl.SSLSocket.recv": ssl.SSLSocket.recv,
        "ssl.SSLObject.write": ssl.SSLObject.write,
        "socket.socket.connect": socket.socket.connect,
        "socket.socket.sendall": socket.socket.sendall,
        "httpx.Client.send": httpx.Client.send,
    }


def test_init_then_close_puts_every_patched_attribute_back():
    """Reversibility for the whole runtime, not one component at a time.

    Every seam the SDK owns is installed here at once — the byte seams, the
    shared connect-timing probe underneath them, the adapters, and the
    propagation patches on the host's HTTP clients — because the ordering bug
    this refactor removes is only visible when they are: `close()` and the
    atexit teardown were two hand-written sequences, and only one of them
    dropped the propagation patches.
    """
    before = _process_seams()

    wardex.init(
        intercept=True,
        backend=BackendConfig(api_key="k"),
        propagation=PropagationPolicy(enabled=True),
    )
    during = _process_seams()
    assert any(during[name] is not before[name] for name in before), (
        "init() patched nothing, so putting it back proves nothing"
    )

    wardex.close()

    after = _process_seams()
    for name, original in before.items():
        assert after[name] is original, f"{name} was not restored to the host's own object"


def test_a_default_init_still_behaves_the_way_it_did_before_the_grouping():
    """Defaults END TO END, not only on the dataclass.

    Every per-field default assertion in `test_config.py` would still pass if a
    READ site were left pointing at a field that moved: the config object would
    be right and the runtime would ignore it. This is the other end of each
    default — the flush interval the worker was actually built with, the signal
    handlers `flush_on_signals=True` installs, and the outbound traffic a
    default init must not touch.
    """
    import httpx

    untouched = httpx.Client.send

    wardex.init(transport=_Recording())
    client = _hub.get_client()
    try:
        assert client._worker._interval == 5.0, "batching.flush_interval"
        assert _runtime.runtime()._signals_installed, "batching.flush_on_signals"
        assert httpx.Client.send is untouched, "propagation.enabled must default to off"
        assert client.config.pii.mode.value == "mask", "pii.mode"
        assert client.config.effective_retention.value == "summary_only", "retention.default"
    finally:
        wardex.close()


def test_the_atexit_teardown_drops_the_propagation_patches_too():
    """The half `close()` did and the `atexit` path did not.

    Two hand-written teardowns, and the propagation patches appeared in one of
    them — so a process that exited through `atexit` left `httpx.Client.send`
    wrapped by an SDK that had announced it was gone. (Re-`init()` was never in
    that state: it dropped the previous patches itself, on its own line, which
    is precisely the second copy of the rule that could stop agreeing with the
    first.) One implementation is what makes them agree; this asserts they do.
    """
    import httpx

    original = httpx.Client.send
    wardex.init(backend=BackendConfig(api_key="k"), propagation=PropagationPolicy(enabled=True))
    assert httpx.Client.send is not original, "propagation never installed"

    _runtime.runtime()._at_exit()

    assert httpx.Client.send is original


# --------------------------------------------------------------------------
# transport resolution
# --------------------------------------------------------------------------


def test_a_backend_endpoint_builds_the_default_otlp_transport():
    """`backend.endpoint` alone must reach the wire, not sit inert on the config.

    The field's failure mode is the one the group docstrings keep circling:
    a user sets the address, omits `transport=`, and everything captured is
    discarded by a `NoOpTransport` in silence. So the default exporter must be
    built FROM the field, against exactly the address it names.
    """
    from wardex_sdk.transport import OtlpHttpTransport

    wardex.init(backend=BackendConfig(endpoint="http://collector.invalid/v1/traces"))
    client = _hub.get_client()
    try:
        assert isinstance(client._transport, OtlpHttpTransport)
        assert client._transport._endpoint == "http://collector.invalid/v1/traces"
    finally:
        # Nothing was captured, so the drain on close has no batch to POST —
        # the endpoint above is never contacted.
        wardex.close()


def test_an_explicit_transport_wins_over_the_endpoint_and_debug_says_so(capsys):
    """A `Transport` carries its own address, so `transport=` beats the field —
    and the losing endpoint is announced under `debug`, because a config value
    that loses a precedence fight in silence looks exactly like one that won."""
    transport = _Recording()
    wardex.init(
        transport=transport,
        backend=BackendConfig(endpoint="http://collector.invalid/v1/traces"),
        debug=True,
    )
    client = _hub.get_client()
    try:
        assert client._transport is transport
        assert "endpoint ignored" in capsys.readouterr().err
    finally:
        wardex.close()


def test_no_transport_and_no_endpoint_still_installs_the_noop_default():
    from wardex_sdk.transport._noop import NoOpTransport

    wardex.init(backend=BackendConfig(api_key="k"))
    client = _hub.get_client()
    try:
        assert isinstance(client._transport, NoOpTransport)
    finally:
        wardex.close()


# --------------------------------------------------------------------------
# idempotence
# --------------------------------------------------------------------------


def test_close_twice_is_a_no_op():
    transport = _Recording()
    wardex.init(
        transport=transport,
        backend=BackendConfig(api_key="k"),
        batching=BatchingPolicy(flush_interval=3600.0),
    )
    client = _hub.get_client()
    client.capture_span(_span())

    wardex.close()
    shipped = sum(len(e.spans) for e in transport.envelopes)
    wardex.close()  # must not raise, must not ship anything twice

    assert shipped == 1
    assert sum(len(e.spans) for e in transport.envelopes) == shipped
    assert client._closed


def test_close_after_a_teardown_is_a_no_op():
    """`atexit` and `wardex.close()` reach the same implementation, in either order."""
    transport = _Recording()
    wardex.init(
        transport=transport,
        backend=BackendConfig(api_key="k"),
        batching=BatchingPolicy(flush_interval=3600.0),
    )
    client = _hub.get_client()
    client.capture_span(_span())

    _runtime.runtime()._at_exit()
    wardex.close()  # must not raise

    assert client._closed
    assert sum(len(e.spans) for e in transport.envelopes) == 1


def test_reset_twice_is_a_no_op():
    wardex.init(backend=BackendConfig(api_key="k"))
    _hub.reset_for_test()
    _hub.reset_for_test()  # must not raise on an already-empty runtime
    assert _runtime.runtime().client is None


# --------------------------------------------------------------------------
# the lock, and the signal handler that has to get past it
# --------------------------------------------------------------------------


def test_the_runtime_lock_is_reentrant():
    """`install()` holds it and then calls `_teardown()` and the signal install,
    both of which take it again on the same thread. A plain `Lock` deadlocks the
    SDK's own `init()` — and the deadlock is in the re-init path, so it would
    ship green through every test that inits once.
    """
    runtime = _runtime.runtime()
    with runtime._lock, runtime._lock:
        assert runtime._lock.acquire(blocking=False)
        runtime._lock.release()


def test_the_signal_handler_does_not_wait_for_a_shutdown_on_another_thread():
    """A handler that blocks on the runtime lock turns Ctrl-C into a hang.

    The lock is held while `init()` tears a previous client down, which joins a
    worker thread and can drain to a slow backend. If a host called `init()` off
    the main thread, a SIGINT arriving in that window would queue behind the
    whole shutdown — the SDK stalling the host at the exact moment the host
    asked it to stop (I1). So the handler's read is non-blocking and answers
    from the same attributes either way.
    """
    runtime = _runtime.runtime()
    transport = _Recording()
    client = _client(transport, batching=BatchingPolicy(flush_interval=3600.0))
    runtime.install(client, client.config)
    client.capture_span(_span())

    seen: list[int] = []
    runtime._prev_handlers[signal.SIGTERM] = lambda signum, frame: seen.append(signum)

    holding = threading.Event()
    release = threading.Event()

    def hold_the_lock() -> None:
        with runtime._lock:
            holding.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=hold_the_lock)
    holder.start()
    try:
        assert holding.wait(timeout=5.0)
        done = threading.Event()

        def deliver() -> None:
            _runtime._handler(signal.SIGTERM, None)
            done.set()

        # Delivered from a helper thread only so the test can BOUND it; the
        # code path is the one the main thread runs.
        threading.Thread(target=deliver).start()
        assert done.wait(timeout=2.0), "the handler blocked behind the lock"
    finally:
        release.set()
        holder.join(timeout=5.0)

    assert seen == [signal.SIGTERM], "the chain to the app's handler broke"
    assert sum(len(e.spans) for e in transport.envelopes) == 1, "the flush never happened"


def test_a_signal_landing_inside_a_teardown_still_reaches_the_apps_handler():
    """The re-entrant case: the handler interrupts the thread that holds the lock.

    CPython runs a handler on the main thread between bytecodes, so it can land
    in the middle of `install()`'s own critical section. With a plain `Lock`
    this is a deadlock inside a signal handler, which is unkillable by anything
    short of SIGKILL.
    """
    runtime = _runtime.runtime()
    client = _client(batching=BatchingPolicy(flush_interval=3600.0))
    runtime.install(client, client.config)

    seen: list[int] = []
    runtime._prev_handlers[signal.SIGTERM] = lambda signum, frame: seen.append(signum)

    with runtime._lock:  # the state install()/teardown() are in when a signal lands
        _runtime._handler(signal.SIGTERM, None)

    assert seen == [signal.SIGTERM]


def test_the_handler_closes_live_units_through_the_runtimes_own_registry():
    """`_close_units` is bound at signal-install time and cleared by the reset.

    Bound rather than looked up in the handler because a signal can land in the
    middle of an import; cleared by `reset()` because a stale binding is a
    reference to a registry entry the next test knows nothing about.
    """
    runtime = _runtime.runtime()
    client = _client(batching=BatchingPolicy(flush_interval=3600.0))
    runtime.install(client, client.config)

    assert runtime._close_units is not None
    assert runtime._close_units.__self__ is runtime.adapters

    _hub.reset_for_test()
    assert runtime._close_units is None


def test_close_units_all_is_what_the_handler_reaches_on_the_dying_disposition():
    """The marker the census pins to this module, driven end to end."""
    runtime = _runtime.runtime()
    client = _client(batching=BatchingPolicy(flush_interval=3600.0))
    runtime.install(client, client.config)

    marks: list[Limitation] = []
    runtime._close_units = lambda *, marker: marks.append(marker)
    runtime._prev_handlers[signal.SIGTERM] = signal.SIG_DFL
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_runtime.os, "kill", lambda *a: None)
        _runtime._handler(signal.SIGTERM, None)

    assert marks == [Limitation.UNIT_INTERRUPTED]
