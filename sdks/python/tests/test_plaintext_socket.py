"""Plaintext raw-socket seam E2E — TLS-free http.server."""

from __future__ import annotations

import http.client
import http.server
import json
import threading

import wardex_sdk as wardex
from wardex_sdk import ConsoleTransport, _hub
from wardex_sdk._enums import CaptureSource, SpanKind

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
