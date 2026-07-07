"""Local self-signed TLS HTTP server fixture (zero external dependencies)."""

from __future__ import annotations

import http.server
import socket
import ssl
import threading
from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent / "fixtures"
CERT = _FIXTURES / "cert.pem"
KEY = _FIXTURES / "key.pem"


@pytest.fixture(autouse=True)
def _close_hub_client_after_test():
    """Join the background worker thread (Task 4, Slice C) any test may have started.

    Client now always spawns a daemon "wardex-batch-worker" thread on
    construction. Many tests across the suite reach the SDK through
    `wardex_sdk.init()`/`_hub.set_client()` and predate that thread; they were
    never written to call `close()` because there was previously nothing to
    clean up. Left alone, every one of those clients leaks its worker thread
    for the rest of the pytest process — which trips thread-count assertions
    in test_worker.py. Closing whatever the hub currently holds after each
    test, at this single choke point, joins those threads without touching
    the ~15 individual test files that construct a client through the hub.
    """
    yield
    from wardex_sdk import _hub

    client = _hub.get_client()
    if client is not None:
        try:
            client.close()
        except Exception:
            # Cleanup-only: Client.close() stops+joins the worker thread before
            # draining/closing the transport, so the thread is already reaped
            # by this point regardless. Some tests build a ConsoleTransport
            # against capsys's captured stdout and never intended for it to
            # survive past the test body (e.g. flush()-then-close() after
            # capsys has already restored/closed its buffer); that is a
            # pre-existing transport quirk unrelated to worker-thread cleanup,
            # so we don't let it fail unrelated tests here.
            pass


class _Handler(http.server.BaseHTTPRequestHandler):
    # enables keep-alive (default HTTP/1.0 closes the connection after every response)
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length)
        payload = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:  # keep quiet
        pass


@pytest.fixture
def tls_server():
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(CERT), keyfile=str(KEY))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    host, port = httpd.socket.getsockname()[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"https://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def h2_server():
    """Local self-signed TLS HTTP/2 server (ALPN h2). Uses the h2 package."""
    import h2.config
    import h2.connection
    import h2.events

    def handle(conn_sock):
        config = h2.config.H2Configuration(client_side=False)
        h2conn = h2.connection.H2Connection(config=config)
        h2conn.initiate_connection()
        conn_sock.sendall(h2conn.data_to_send())
        while True:
            data = conn_sock.recv(65535)
            if not data:
                break
            events = h2conn.receive_data(data)
            for event in events:
                if isinstance(event, h2.events.RequestReceived):
                    sid = event.stream_id
                    body = b'{"ok":true}'
                    h2conn.send_headers(
                        sid,
                        [(":status", "200"), ("content-type", "application/json")],
                    )
                    h2conn.send_data(sid, body, end_stream=True)
            out = h2conn.data_to_send()
            if out:
                conn_sock.sendall(out)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(CERT), keyfile=str(KEY))
    ctx.set_alpn_protocols(["h2"])
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(5)
    tls = ctx.wrap_socket(sock, server_side=True)
    host, port = tls.getsockname()[:2]

    stop = threading.Event()
    tls.settimeout(0.5)

    def serve():
        while not stop.is_set():
            try:
                client, _ = tls.accept()
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

    t = threading.Thread(target=serve, daemon=True)
    t.start()
    try:
        yield f"https://{host}:{port}"
    finally:
        stop.set()
        try:
            tls.close()
        except Exception:
            pass


@pytest.fixture
def sse_tls_server():
    """Streams OpenAI-shaped SSE (text/event-stream) as chunked. Delays after
    headers so ttft can be measured."""
    import time as _time

    class _SseHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            _ = self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            # flush headers first -> separates them from the body (for ttft measurement)
            self.wfile.flush()
            _time.sleep(0.02)
            chunks = [
                (
                    b'data: {"id":"chatcmpl-s","model":"gpt-4o-mini",'
                    b'"choices":[{"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
                ),
                (
                    b'data: {"id":"chatcmpl-s","model":"gpt-4o-mini",'
                    b'"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n'
                ),
                b"data: [DONE]\n\n",
            ]
            for c in chunks:
                self.wfile.write(b"%x\r\n" % len(c) + c + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def log_message(self, *args: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _SseHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(CERT), keyfile=str(KEY))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    host, port = httpd.socket.getsockname()[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"https://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
