"""Plaintext raw-socket seam E2E — TLS-free http.server."""

from __future__ import annotations

import http.client
import http.server
import json
import socket
import threading

import wardex_sdk as wardex
from wardex_sdk import ConsoleTransport, _hub
from wardex_sdk._enums import CaptureSource, SpanKind
from wardex_sdk.interceptors._socket import RawSocketInterceptor

# The four `socket.socket` methods this seam patches. All four are INHERITED
# from the C base `_socket.socket` — `socket.socket` does not define them — and
# that is what the restore assertions below turn on.
_PATCHED = ("send", "sendall", "recv", "recv_into")

_LLM_RESP = json.dumps(
    {
        "id": "c1",
        "model": "gpt-4o",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
).encode()
_PLAIN_RESP = json.dumps({"ok": True}).encode()


def _server(payload: bytes):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
    host, port = httpd.socket.getsockname()[:2]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, host, port


def _client_spans():
    return [s for s in _hub.get_client()._spans if s.kind == SpanKind.CLIENT]


def _post(host: str, port: int, body: bytes, path: str = "/v1/chat/completions") -> None:
    conn = http.client.HTTPConnection(host, port)  # plaintext (no TLS)
    conn.request("POST", path, body, {})
    conn.getresponse().read()
    conn.close()


def test_uninstall_leaves_no_trace_on_socket_socket():
    """The seam's restore, which nothing checked.

    Deleting `self._patches.restore_all()` from `RawSocketInterceptor.uninstall`
    left the whole suite green: wardex could leave `socket.socket.send`,
    `sendall`, `recv` and `recv_into` patched for the life of the process, in a
    class every network library in it reaches through, and no test noticed. The
    ssl, connection-timing and Agent SDK seams all had an equivalent check;
    this one did not.

    The assertion is on the OWN-attribute namespace, not on identity, because
    identity cannot see the second half of the bug. All four names are
    inherited from `_socket.socket`, so a restore written as
    `setattr(socket.socket, "send", original)` — which is what this seam did
    before `PatchSet` — passes `socket.socket.send is original` while leaving a
    permanent own-attribute shadow: `socket.socket` now carries a frozen copy
    of whatever the C base held at install time, and any later change to the
    base stops reaching it. Nothing is installed, and nothing is visible.
    `"send" not in socket.socket.__dict__` is the only assertion that fails on
    both the missing restore and the shadow.
    """
    for name in _PATCHED:
        assert name not in socket.socket.__dict__, f"precondition: {name} is inherited"
    inherited = {name: getattr(socket.socket, name) for name in _PATCHED}

    itc = RawSocketInterceptor()
    try:
        itc.install(None)
        for name in _PATCHED:
            assert name in socket.socket.__dict__, f"precondition: {name} is patched"
            assert getattr(socket.socket, name) is not inherited[name]
    finally:
        itc.uninstall()

    for name in _PATCHED:
        assert name not in socket.socket.__dict__, (
            f"socket.socket.{name} is still an own attribute after uninstall — "
            "wardex either never restored it, or restored it as a shadow over "
            "the inherited method"
        )
        assert getattr(socket.socket, name) is inherited[name]


def test_plaintext_llm_call_captured():
    httpd, host, port = _server(_LLM_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(host, port, b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}')
        spans = _client_spans()
        assert len(spans) == 1
        assert spans[0].transport.http.url.startswith("http://")  # not https
        assert spans[0].gen_ai is not None  # LLM identified
        # confirm capture_source is tagged as SOCKET
        assert CaptureSource.SOCKET in spans[0].capture_sources
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_plaintext_non_llm_dropped():
    httpd, host, port = _server(_PLAIN_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(host, port, b'{"foo":"bar"}', path="/health")
        assert _client_spans() == []  # non-LLM + no allowlist → dropped
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_non_http_bytes_not_captured():
    # send redis-like bytes over a plaintext socket → sniff-latch ignores it
    httpd, host, port = _server(_PLAIN_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        import socket as _socket

        s = _socket.create_connection((host, port))
        try:
            s.sendall(b"*1\r\n$4\r\nPING\r\n")  # not an HTTP method
        except Exception:
            pass
        s.close()
        assert _client_spans() == []
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_allowlist_non_llm_captured():
    # if the peer ip:port is registered in the allowlist, non-LLM traffic is emitted too
    httpd, host, port = _server(_PLAIN_RESP)
    try:
        wardex.init(
            transport=ConsoleTransport(),
            intercept=True,
            intercept_hosts=[f"{host}:{port}"],
        )
        _post(host, port, b'{"foo":"bar"}', path="/health")
        spans = _client_spans()
        assert len(spans) == 1  # non-LLM emitted too via allowlist
        assert spans[0].transport.http.url.startswith("http://")
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_allowlist_host_only_matches():
    # matches even when only the host is registered, without a port
    httpd, host, port = _server(_PLAIN_RESP)
    try:
        wardex.init(
            transport=ConsoleTransport(),
            intercept=True,
            intercept_hosts=[host],
        )
        _post(host, port, b'{"foo":"bar"}', path="/health")
        assert len(_client_spans()) == 1
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_plaintext_ws_requires_allowlist():
    # without an allowlist, non-LLM traffic (including ws paths) is dropped —
    # verifies the allowlist gate
    httpd, host, port = _server(_PLAIN_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)  # no allowlist
        # a plain response without 101 doesn't trigger the ws tracker — same path as non-LLM drop
        _post(host, port, b'{"x":1}', path="/ws")
        assert _client_spans() == []
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()
