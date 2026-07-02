import ssl
from pathlib import Path

import httpx

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
    probe.install()
    captured = {}
    try:
        # intercept the fileno at do_handshake time to verify the measured value

        real_hs = ssl.SSLSocket.do_handshake

        def spy(self, *a, **k):
            try:
                captured["fileno"] = self.fileno()
            except Exception:
                pass
            return real_hs(self, *a, **k)

        # layer a spy on top of what the probe already patched, to observe only the fileno
        ssl.SSLSocket.do_handshake = spy
        resp = httpx.post(f"{tls_server}/v1/ping", json={}, verify=_verify_ctx())
        assert resp.status_code == 200
    finally:
        probe.uninstall()

    fn = captured.get("fileno")
    assert fn is not None
    pair = store.pop(fn)
    assert pair is not None
    connect_ms, handshake_ms = pair
    assert connect_ms >= 0.0
    assert handshake_ms > 0.0  # TLS handshake takes a measurable amount of time
