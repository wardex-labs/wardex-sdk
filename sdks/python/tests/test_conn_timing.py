import ssl
from pathlib import Path

import httpx

from wardex_sdk.assembly import PatchSet
from wardex_sdk.interceptors._conn_timing import ConnTimingProbe, ConnTimingStore


def test_store_set_and_pop_returns_pair_then_none():
    s = ConnTimingStore()
    s.set_connect(7, 1.5)
    s.set_handshake(7, 20.0)
    assert s.pop(7) == (1.5, 20.0)
    assert s.pop(7) is None  # removed after being consumed once


def test_store_pop_missing_is_none():
    assert ConnTimingStore().pop(999) is None


def test_store_handshake_without_connect_defaults_connect_zero():
    s = ConnTimingStore()
    s.set_handshake(3, 5.0)
    assert s.pop(3) == (0.0, 5.0)


def test_store_fifo_cap_evicts_oldest():
    s = ConnTimingStore(cap=2)
    s.set_connect(1, 1.0)
    s.set_connect(2, 2.0)
    s.set_connect(3, 3.0)  # evicts 1
    assert s.pop(1) is None
    assert s.pop(3) == (3.0, 0.0)


def _verify_ctx() -> ssl.SSLContext:
    cert = Path(__file__).parent / "fixtures" / "cert.pem"
    return ssl.create_default_context(cafile=str(cert))


def test_probe_install_uninstall_restores_originals():
    import socket as _s

    orig_connect = _s.socket.connect
    orig_hs = ssl.SSLSocket.do_handshake
    probe = ConnTimingProbe(ConnTimingStore())
    probe.install()
    assert _s.socket.connect is not orig_connect
    assert ssl.SSLSocket.do_handshake is not orig_hs
    probe.uninstall()
    assert _s.socket.connect is orig_connect
    assert ssl.SSLSocket.do_handshake is orig_hs


def test_probe_sync_records_connect_and_handshake(tls_server):
    store = ConnTimingStore()
    probe = ConnTimingProbe(store)
    # Captured BEFORE anything is installed, so the assertion at the bottom
    # compares against the interpreter's own method rather than against
    # whatever this test left behind.
    pristine_hs = ssl.SSLSocket.do_handshake
    probe.install()
    captured = {}
    pair = None
    # The spy goes on through a PatchSet of its own, layered over the probe's
    # patch. A raw `ssl.SSLSocket.do_handshake = spy` here is not a shortcut,
    # it is a leak for the whole pytest process: `PatchSet` restores an
    # attribute only while it still holds the exact wrapper that set installed,
    # so `probe.uninstall()` correctly declines to clobber the spy (it counts
    # the supersession and leaves the other "library" alone) and every later
    # test in the session then runs its TLS handshakes through this closure.
    spy_patches = PatchSet("tests.conn_timing.spy")
    try:
        # intercept the fileno at do_handshake time to verify the measured value

        real_hs = ssl.SSLSocket.do_handshake

        def spy(self, *a, **k):
            try:
                captured["fileno"] = self.fileno()
            except Exception:
                pass
            return real_hs(self, *a, **k)

        spy_patches.patch(ssl.SSLSocket, "do_handshake", spy)
        # Read while the connection is still OPEN. Closing it releases the slot
        # now (interceptors/_close_hook.py), which is the whole point of the
        # close hook: the store holds live connections, not dead ones. The
        # measurement this test is about is taken at handshake time either way.
        with httpx.Client(verify=_verify_ctx()) as client:
            resp = client.post(f"{tls_server}/v1/ping", json={})
            assert resp.status_code == 200
            fn = captured.get("fileno")
            assert fn is not None
            pair = store.pop(fn)
    finally:
        # Innermost first, which across two sets is this order and not the other:
        # restoring the probe while the spy is still on top is exactly the
        # supersession that stranded the spy.
        spy_patches.restore_all()
        probe.uninstall()

    assert pair is not None
    connect_ms, handshake_ms = pair
    assert connect_ms >= 0.0
    assert handshake_ms > 0.0  # TLS handshake takes a measurable amount of time

    # The regression guard, and it is two assertions because either one alone
    # can be satisfied by the bug. `superseded` says the spy came off cleanly
    # rather than being left in place; the identity check says the class is back
    # to what the interpreter shipped rather than to some other wrapper.
    assert spy_patches.superseded == 0, "the spy was patched over and stayed installed"
    assert ssl.SSLSocket.do_handshake is pristine_hs, (
        "this test leaked a patch on ssl.SSLSocket into the rest of the session"
    )


def test_store_discard_releases_a_slot_nobody_will_consume():
    s = ConnTimingStore()
    s.set_connect(11, 1.0)
    s.discard(11)
    assert s.pop(11) is None
    s.discard(11)  # idempotent: a socket may be closed twice


def test_a_closed_sockets_slot_no_longer_pushes_a_live_connection_out():
    """`socket.connect` is patched globally, so every non-TLS socket in the
    process leaves an entry here — and nothing pops one that never becomes a
    span. Under the FIFO cap alone, those dead entries push out the OLDEST live
    entry, which is the connection still streaming a response; `_resolve_timing`
    then finds nothing and marks a perfectly measured connection
    `connect_timing_unavailable`.

    The `blocker` is the one thing that looks like a trick and is not. The
    kernel reissues the LOWEST free descriptor, so a short-lived connection
    normally lands on the fd the previous one just released — and a second
    entry under the same key overwrites the dead one's slot instead of adding
    to the table, which masks the bug in a test while doing nothing for a real
    process where the freed fd goes to a file, a pipe or another thread. Taking
    the descriptor out of circulation is what makes each connection cost a
    distinct slot, which is the situation the cap was mismanaging.
    """
    import socket as _s

    store = ConnTimingStore(cap=2)
    probe = ConnTimingProbe(store)
    listener = _s.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    address = listener.getsockname()
    probe.install()
    live = None
    blocker = None
    try:
        live = _s.create_connection(address)  # still streaming; its span is not emitted yet
        live_fileno = live.fileno()

        first = _s.create_connection(address)
        first.close()
        blocker = _s.socket()  # takes the freed descriptor, so the next one is fresh
        second = _s.create_connection(address)
        second.close()

        assert store.pop(live_fileno) is not None, (
            "the live connection's timing was evicted by sockets that had already closed"
        )
    finally:
        probe.uninstall()
        for sock in (live, blocker, listener):
            if sock is not None:
                sock.close()
