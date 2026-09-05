"""h2c (cleartext HTTP/2, prior-knowledge) E2E — replays a real h2 exchange over a plaintext socket.

Drives client/server directly with the h2 library to verify that
RawSocketInterceptor captures the sendall/recv bytes. No TLS (no ALPN) →
detection relies solely on the preface. The server-side socket sends SETTINGS
first, so the gate latches to "ignore" → no contamination.
"""

from __future__ import annotations

import json
import socket
import threading

import h2.config
import h2.connection
import h2.events
import pytest

import wardex_sdk as wardex
from conftest import client_spans
from wardex_sdk import ConsoleTransport, _hub
from wardex_sdk._enums import CaptureSource
from wardex_sdk._interceptors._registry import get_registry

_LLM_RESP = json.dumps(
    {
        "id": "c1",
        "model": "gpt-4o",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
).encode()
_PLAIN_RESP = json.dumps({"ok": True}).encode()


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk._interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


def _h2c_server(body: bytes):
    """Plaintext (no TLS) HTTP/2 server. Responds to StreamEnded with body.
    Returns (stop, srv, host, port)."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    srv.settimeout(0.5)
    host, port = srv.getsockname()[:2]
    stop = threading.Event()

    def handle(conn_sock: socket.socket) -> None:
        conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
        conn.initiate_connection()
        conn_sock.sendall(
            conn.data_to_send()
        )  # server SETTINGS first → server socket gate latches "ignore"
        while not stop.is_set():
            data = conn_sock.recv(65535)
            if not data:
                break
            for event in conn.receive_data(data):
                if isinstance(event, h2.events.StreamEnded):
                    sid = event.stream_id
                    conn.send_headers(
                        sid,
                        [(":status", "200"), ("content-type", "application/json")],
                    )
                    conn.send_data(sid, body, end_stream=True)
            out = conn.data_to_send()
            if out:
                conn_sock.sendall(out)

    def serve() -> None:
        while not stop.is_set():
            try:
                client, _ = srv.accept()
            except (TimeoutError, OSError):
                continue
            try:
                handle(client)
            except Exception:
                pass
            finally:
                try:
                    client.close()
                except Exception:
                    pass

    threading.Thread(target=serve, daemon=True).start()
    return stop, srv, host, port


def _h2c_post(host: str, port: int, path: str, body: bytes) -> None:
    """Performs a prior-knowledge h2c POST over a plaintext socket."""
    s = socket.create_connection((host, port))
    s.settimeout(2.0)
    try:
        conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True))
        conn.initiate_connection()
        s.sendall(conn.data_to_send())  # preface + SETTINGS (first send → gate detects the preface)
        sid = conn.get_next_available_stream_id()
        conn.send_headers(
            sid,
            [
                (":method", "POST"),
                (":authority", f"{host}:{port}"),
                (":scheme", "http"),
                (":path", path),
                ("content-type", "application/json"),
            ],
        )
        conn.send_data(sid, body, end_stream=True)
        s.sendall(conn.data_to_send())
        ended = False
        while not ended:
            data = s.recv(65535)
            if not data:
                break
            for event in conn.receive_data(data):
                if isinstance(event, h2.events.StreamEnded):
                    ended = True
            out = conn.data_to_send()
            if out:
                s.sendall(out)
    finally:
        s.close()


def test_h2c_llm_call_captured():
    stop, srv, host, port = _h2c_server(_LLM_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _h2c_post(
            host,
            port,
            "/v1/chat/completions",
            b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}',
        )
        spans = client_spans()
        assert len(spans) == 1
        sp = spans[0]
        assert sp.transport.http.url.startswith("http://")  # plaintext h2c
        assert sp.gen_ai is not None  # LLM identified
        assert ("network.protocol.version", "2") in sp.extra  # HTTP/2
        assert CaptureSource.SOCKET in sp.capture_sources
    finally:
        stop.set()
        srv.close()


def test_h2c_non_llm_dropped_without_allowlist():
    stop, srv, host, port = _h2c_server(_PLAIN_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        # verify the interceptor is actually installed → ensures the drop is a policy outcome
        assert get_registry().is_installed("socket")
        _h2c_post(host, port, "/rpc", b'{"foo":"bar"}')
        # non-LLM + no allowlist → dropped (same emission path as gRPC-over-h2c)
        assert client_spans() == []
    finally:
        stop.set()
        srv.close()


def test_h2c_non_llm_captured_with_allowlist():
    stop, srv, host, port = _h2c_server(_PLAIN_RESP)
    try:
        wardex.init(
            transport=ConsoleTransport(),
            intercept=True,
            intercept_hosts=[f"{host}:{port}"],
        )
        _h2c_post(host, port, "/rpc", b'{"foo":"bar"}')
        spans = client_spans()
        assert len(spans) == 1  # allowlist also emits non-LLM h2c
        assert spans[0].transport.http.url.startswith("http://")
        assert ("network.protocol.version", "2") in spans[0].extra
    finally:
        stop.set()
        srv.close()
