"""The socket close hook — the moment the byte seams never had.

Four defects shared one root: state was created when a connection was first
SEEN and nothing ever destroyed it. These are the tests for the moment that
does, and for each thing it retires.
"""

from __future__ import annotations

import gc
import socket
from functools import partial
from types import SimpleNamespace

import pytest
from hpack import Encoder

from wardex_sdk._enums import CaptureMode
from wardex_sdk._limits import CaptureLimits
from wardex_sdk.assembly import Limitation, counters
from wardex_sdk.interceptors import _close_hook, _seam
from wardex_sdk.interceptors._close_hook import (
    CloseRegistry,
    close_registry,
    install_shared_close_hook,
    uninstall_shared_close_hook,
)
from wardex_sdk.interceptors._trackers import _Http2Tracker, _WebSocketTracker


@pytest.fixture(autouse=True)
def _clean_registry():
    """The registry is a process singleton; a leftover hook is a cross-test bleed."""
    close_registry().clear()
    counters.reset()
    yield
    close_registry().clear()
    counters.reset()


class _Weakrefable:
    """Stands in for a socket: an ordinary object, so it can carry a weak reference."""


# --------------------------------------------------------------------------
# the registry itself
# --------------------------------------------------------------------------


def test_a_hook_fires_once_on_close_and_never_again():
    reg = CloseRegistry()
    obj = _Weakrefable()
    fired = []
    reg.on_close(obj, lambda: fired.append("x"))

    reg.fire(obj)
    reg.fire(obj)  # a socket may be closed twice; the state is gone after the first

    assert fired == ["x"]
    assert reg.tracked() == 0


def test_every_hook_runs_even_when_one_of_them_raises():
    """One seam's broken eviction must not cost another seam its own."""
    reg = CloseRegistry()
    obj = _Weakrefable()
    fired = []

    def boom() -> None:
        raise RuntimeError("eviction went wrong")

    reg.on_close(obj, boom)
    reg.on_close(obj, lambda: fired.append("second"))

    reg.fire(obj)  # must not raise: this runs inside the host's socket.close()

    assert fired == ["second"]
    assert counters.get("interceptors.close_hook.fire") == 1, (
        "a swallowed hook failure left no evidence anywhere"
    )


def test_registering_the_same_hook_twice_registers_it_once():
    """Both callers repeat, and the list they append to has no bound.

    `ssl.SSLSocket.do_handshake` runs in a retry loop on a non-blocking socket
    (eventlet's `GreenSSLSocket` trampolines over exactly that) and again on
    renegotiation, so `_conn_timing._release_at_close` re-registers per attempt.
    The seam re-registers whenever its FIFO cap evicted a state whose socket is
    still live. One `functools.partial` per repeat, held for the socket's whole
    life, and `_fire` then runs N identical hooks inside the host's `close()`.
    """
    reg = CloseRegistry()
    obj = _Weakrefable()
    fired = []
    for _ in range(50):
        reg.on_close(obj, partial(fired.append, "x"))

    assert len(reg._entries[id(obj)].hooks) == 1
    reg.fire(obj)
    assert fired == ["x"]


def test_dedupe_does_not_merge_two_different_hooks_for_one_object():
    """The seam and the timing probe both register on the same socket, and they
    are not each other."""
    reg = CloseRegistry()
    obj = _Weakrefable()
    fired = []
    reg.on_close(obj, partial(fired.append, "state"))
    reg.on_close(obj, partial(fired.append, "timing"))
    reg.on_close(obj, lambda: fired.append("lambda"))
    reg.on_close(obj, lambda: fired.append("another lambda"))

    reg.fire(obj)

    assert fired == ["state", "timing", "lambda", "another lambda"]


def test_collecting_the_object_fires_the_hook_without_the_registry_holding_it():
    """The half that makes the id-reuse bug impossible rather than unlikely.

    CPython runs weakref callbacks during deallocation, before the address can
    be handed to anything else — so there is no instant at which a live object
    shares an id with a registered-but-unfired hook.
    """
    reg = CloseRegistry()
    obj = _Weakrefable()
    fired = []
    reg.on_close(obj, lambda: fired.append("gone"))
    assert reg.tracked() == 1

    del obj
    gc.collect()

    assert fired == ["gone"]
    assert reg.tracked() == 0


def test_a_close_only_hook_is_skipped_by_the_finalizer():
    """`on_finalize=False` is for hooks keyed by something the dead object no
    longer owns — a file descriptor, which the kernel reissues the instant it is
    released."""
    reg = CloseRegistry()
    obj = _Weakrefable()
    fired = []
    reg.on_close(obj, lambda: fired.append("fd"), on_finalize=False)
    reg.on_close(obj, lambda: fired.append("state"))

    del obj
    gc.collect()

    assert fired == ["state"]


def test_a_close_only_hook_still_runs_on_an_explicit_close():
    reg = CloseRegistry()
    obj = _Weakrefable()
    fired = []
    reg.on_close(obj, lambda: fired.append("fd"), on_finalize=False)

    reg.fire(obj)

    assert fired == ["fd"]


def test_forget_drops_the_hooks_without_running_them():
    reg = CloseRegistry()
    obj = _Weakrefable()
    fired = []
    reg.on_close(obj, lambda: fired.append("x"))

    reg.forget(obj)
    reg.fire(obj)
    del obj
    gc.collect()

    assert fired == []


class _DetachThatFires:
    """A finalizer stand-in whose `detach()` reaches back into the registry.

    The real thing has two routes to the same place and neither is easy to time:
    `weakref.finalize.detach` is ordinary Python and allocates, so a cyclic
    collection can start inside it, and any socket in the process closing on
    another thread reaches `_fire` regardless. Modelling the re-entry directly
    keeps the test deterministic instead of racing a collector for it.
    """

    def __init__(self, on_detach) -> None:
        self._on_detach = on_detach

    def detach(self) -> None:
        self._on_detach()


def test_clear_survives_an_entry_disappearing_while_it_walks_the_table():
    """`clear()` is the one method that holds the table open across a re-entry.

    Popping from a dict that is being iterated raises `RuntimeError`, and this
    one would raise it out of `uninstall_shared_close_hook` — after the probes
    above have restored their patches and before they clear their `_installed`
    flags. That exception is counted and swallowed, the flag survives on a
    module singleton, and the next `init()` finds an already-installed probe:
    connection timing silently never instrumented again for the life of the
    process.
    """
    reg = CloseRegistry()
    victims = [_Weakrefable() for _ in range(4)]
    for obj in victims:
        reg.on_close(obj, lambda: None)

    entry = reg._entries[id(victims[0])]
    entry.finalizer = _DetachThatFires(lambda: reg.fire(victims[-1]))

    reg.clear()

    assert reg.tracked() == 0


def test_an_object_that_refuses_a_weak_reference_is_not_a_swallowed_failure():
    """An absence is an answer, not a wardex bug.

    `weakref.finalize` reports "this object cannot carry one" by raising, and
    catching that would put the answer in a `guard()` counter — the signal that
    means a span was deleted. `types.SimpleNamespace` is the shape the seam's
    own tests drive it with, so this is not hypothetical.
    """
    reg = CloseRegistry()
    obj = SimpleNamespace()
    fired = []

    reg.on_close(obj, lambda: fired.append("x"))
    reg.fire(obj)

    assert fired == ["x"], "an explicit close must still reach the hook"
    assert counters.snapshot() == {}, "an absence was recorded as a swallowed failure"


# --------------------------------------------------------------------------
# the probe on socket.socket.close
# --------------------------------------------------------------------------


def test_closing_a_real_socket_fires_its_hooks():
    install_shared_close_hook()
    try:
        sock = socket.socket()
        fired = []
        close_registry().on_close(sock, lambda: fired.append("closed"))
        sock.close()
        assert fired == ["closed"]
    finally:
        uninstall_shared_close_hook()


def test_the_deferred_close_fast_path_still_exists_on_this_interpreter():
    """`socket.socket._real_close` is private, and the probe treats it that way:
    `install()` asks with `getattr` and degrades to the finalizer when the name
    is gone — correct, only late. Users must get that degradation silently;
    MAINTAINERS must not. This is the canary that turns red on the first
    interpreter to rename the fast path, so the new spelling is found in CI
    rather than deduced from `will_close` spans quietly arriving late."""
    assert hasattr(socket.socket, "_real_close"), (
        "this interpreter dropped socket.socket._real_close: the close probe now "
        "retires deferred-close connections via the GC finalizer only — find the "
        "renamed hook and teach install() about it"
    )


def test_the_probe_is_refcounted_and_leaves_no_trace():
    orig = socket.socket.close
    install_shared_close_hook()
    patched = socket.socket.close
    assert patched is not orig
    install_shared_close_hook()
    assert socket.socket.close is patched, "a second acquire re-patched the attribute"
    uninstall_shared_close_hook()
    assert socket.socket.close is patched, "released while another holder was still using it"
    uninstall_shared_close_hook()
    assert socket.socket.close is orig


def test_detach_is_not_a_close():
    """`ssl.SSLContext.wrap_socket` builds the SSLSocket from the plain socket's
    fileno and then calls `sock.detach()`. Firing close hooks there would evict
    the connect timing of every TLS connection in the process at the moment it
    was wrapped — a detach is a handoff, not an end."""
    install_shared_close_hook()
    fd = -1
    try:
        sock = socket.socket()
        fired = []
        close_registry().on_close(sock, lambda: fired.append("closed"))
        fd = sock.detach()
        assert fired == []
    finally:
        if fd != -1:
            socket.socket(fileno=fd).close()  # the detached fd is nobody else's to close
        uninstall_shared_close_hook()


# --------------------------------------------------------------------------
# what the hook retires: the seam's per-connection state
# --------------------------------------------------------------------------


def _frame(ftype: int, flags: int, stream_id: int, payload: bytes) -> bytes:
    return (
        len(payload).to_bytes(3, "big")
        + bytes([ftype, flags])
        + stream_id.to_bytes(4, "big")
        + payload
    )


def _h2_request(stream_id: int) -> bytes:
    """One HEADERS frame that opens `stream_id` and ends it — no response follows."""
    block = Encoder().encode([(b":method", b"POST"), (b":path", b"/v1/messages")])
    return _frame(0x1, 0x4 | 0x1, stream_id, block)  # END_HEADERS | END_STREAM


def test_an_h2_stream_that_never_answers_loses_its_latch_entry_at_close(
    fake_ssl_socket, bare_ssl_interceptor
):
    """A stream that ends without a response never reaches `_mk`, so its latch
    entry used to live as long as the connection did — and a keep-alive h2
    connection to a model provider lives as long as the process."""
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn="h2")
    itc._on_request_bytes(sock, _h2_request(1))
    itc._on_request_bytes(sock, _h2_request(3))

    tracker = itc._conns[id(sock)].tracker
    assert isinstance(tracker, _Http2Tracker)
    assert set(tracker._latch) == {1, 3}, "precondition: both streams are latched"

    close_registry().fire(sock)

    assert tracker._latch == {}
    assert id(sock) not in itc._conns


def test_the_h2_latch_is_capped_for_a_connection_that_never_closes():
    """The path the close hook cannot reach, and the bound that covers it.

    On the async TLS seam the carrier is an `ssl.SSLObject`: no `close()` to
    patch, and asyncio's `SSLProtocol` pins it for the life of the transport, so
    it is neither closed nor collected. A pooled h2 keep-alive to a model
    provider is exactly that shape — and every stream it resets strands a latch
    entry that `on_connection_close` will never be called to release. So the
    latch carries its own cap, sourced from the same `max_streams` the Rust
    parser bounds its own stream table with.
    """
    tracker = _Http2Tracker(CaptureLimits(max_streams=8).to_native())
    for sid in range(1, 2 * 200, 2):
        tracker.on_request_bytes(_h2_request(sid))  # opened, never answered

    assert len(tracker._latch) == 8
    assert max(tracker._latch) == 399, "the cap evicted the newest instead of the oldest"


def test_a_state_rebuilt_on_a_live_socket_does_not_stack_retirement_hooks(
    fake_ssl_socket, bare_ssl_interceptor
):
    """The FIFO cap pops `_conns` but leaves the registry entry, so the next
    byte on an evicted-but-live connection rebuilds the state and re-registers.
    Without a dedupe that is one allocation per `recv`, held until the socket
    finally closes."""
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn=None)
    for _ in range(10):
        itc._state(sock)
        itc._conns.pop(id(sock), None)  # what the cap does to a still-live entry

    assert len(close_registry()._entries[id(sock)].hooks) == 1


def test_the_connection_state_goes_when_the_socket_goes(fake_ssl_socket, bare_ssl_interceptor):
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn=None)
    itc._on_request_bytes(sock, b"POST /v1/messages HTTP/1.1\r\nContent-Length: 0\r\n\r\n")
    assert len(itc._conns) == 1

    del sock
    gc.collect()

    assert itc._conns == {}, "the entry outlived the socket it describes"


def test_a_recycled_id_does_not_inherit_the_dead_connections_verdict(
    monkeypatch, fake_ssl_socket, bare_ssl_interceptor
):
    """The bug the finalizer closes, reproduced without gambling on the allocator.

    `_conns` is keyed by `id(obj)`, and CPython hands a freed address to the
    next object of that size. A TLS-backed Redis connection latched "ignore"
    therefore used to hand its verdict to whatever landed on its address next —
    and an HTTPS connection that inherits `gate == "ignore"` is never captured
    at all, silently, for its whole life.

    The collision is forced by shadowing `id` in the seam's own module rather
    than by allocating until CPython obliges, so the test asserts the fix
    instead of the allocator's mood.
    """
    itc = bare_ssl_interceptor
    # `raising=False`: `id` is a builtin, so the seam module has no attribute of
    # its own to replace — writing one into its globals is what shadows the
    # builtin for that module alone.
    monkeypatch.setattr(_seam, "id", lambda obj: 424242, raising=False)

    dead = fake_ssl_socket(alpn=None)
    itc._on_request_bytes(dead, b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n")
    assert itc._conns[424242].gate == "ignore", "precondition: the Redis connection latched off"

    close_registry().fire(dead)  # what socket.close() does
    assert 424242 not in itc._conns

    live = fake_ssl_socket(alpn=None)
    itc._on_request_bytes(live, b"POST /v1/messages HTTP/1.1\r\nContent-Length: 0\r\n\r\n")

    assert itc._conns[424242].gate == "http", (
        "a new connection was served the dead one's sniff-latch verdict"
    )


def test_a_websocket_session_is_shipped_when_the_socket_closes(
    fake_ssl_socket, bare_ssl_interceptor
):
    """A WS span exists only once the session ends. Before the close hook it
    waited for `uninstall()`, and was lost whenever the process never reached
    one."""
    itc = bare_ssl_interceptor
    itc._client.config.capture_mode = CaptureMode.ALL
    sock = fake_ssl_socket(alpn=None)
    st = itc._state(sock)
    st.gate = "http"
    st.tracker = _WebSocketTracker(path="/chat", deflate=False, parent=None, start_ns=1)

    close_registry().fire(sock)

    (span,) = itc._client.spans
    assert Limitation.WS_NO_CLOSE in span.capture_integrity.limitations
    assert id(sock) not in itc._conns


def test_closing_a_connection_the_seam_never_saw_is_a_no_op(fake_ssl_socket, bare_ssl_interceptor):
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn=None)
    close_registry().fire(sock)
    assert itc._conns == {}


def test_uninstall_still_ships_a_live_websocket_session(fake_ssl_socket, bare_ssl_interceptor):
    """The close hook must not have quietly taken over teardown: a session whose
    socket is still open at `uninstall()` is the case that path exists for."""
    itc = bare_ssl_interceptor
    itc._client.config.capture_mode = CaptureMode.ALL
    sock = fake_ssl_socket(alpn=None)
    st = itc._state(sock)
    st.tracker = _WebSocketTracker(path="/chat", deflate=False, parent=None, start_ns=1)

    itc.uninstall()

    (span,) = itc._client.spans
    assert Limitation.WS_NO_CLOSE in span.capture_integrity.limitations


def test_the_hook_does_not_keep_the_socket_alive(fake_ssl_socket, bare_ssl_interceptor):
    """An eviction mechanism that referenced the socket would hold the host's
    file descriptor open for as long as the seam is installed."""
    import weakref

    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn=None)
    itc._on_request_bytes(sock, b"POST /v1/messages HTTP/1.1\r\nContent-Length: 0\r\n\r\n")
    ref = weakref.ref(sock)

    del sock
    gc.collect()

    assert ref() is None, "the seam or the registry pinned the socket"


def test_uninstalling_the_last_holder_drops_every_registration():
    """Hooks outliving their seam would keep it alive for as long as its
    sockets, and would touch state that no longer exists."""
    install_shared_close_hook()
    obj = _Weakrefable()
    fired = []
    close_registry().on_close(obj, lambda: fired.append("x"))
    assert close_registry().tracked() == 1

    uninstall_shared_close_hook()

    assert close_registry().tracked() == 0
    del obj
    gc.collect()
    assert fired == []


def test_the_module_singleton_is_shared():
    assert _close_hook.close_registry() is close_registry()
