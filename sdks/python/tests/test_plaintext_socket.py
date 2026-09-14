"""Plaintext raw-socket seam E2E — TLS-free http.server."""

from __future__ import annotations

import http.client
import http.server
import json
import socket
import threading

import pytest

import wardex_sdk as wardex
from conftest import client_spans
from wardex_sdk import ConsoleTransport
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._enums import CaptureSource
from wardex_sdk._interceptors._peer import peer_address
from wardex_sdk._interceptors._seam import _ConnectionState
from wardex_sdk._interceptors._socket import RawSocketInterceptor

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
        spans = client_spans()
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
        assert client_spans() == []  # non-LLM + no allowlist → dropped
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
        assert client_spans() == []
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
        spans = client_spans()
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
        assert len(client_spans()) == 1
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_allowlist_matching_is_case_insensitive():
    """`intercept_hosts` names hosts, and hostnames are case-insensitive.

    Exercised at `_in_allow` rather than end-to-end because the loopback server
    above is reached by IP, where there is no case to get wrong. The names do
    differ on the path that matters: when `getpeername()` fails the seam falls
    back to the connection's `server_hostname`, which is whatever string the
    caller handed to connect.
    """
    seam = RawSocketInterceptor(intercept_hosts=["MyBox.local", "Other.Local:8443"])

    def state(address: str, port: int) -> _ConnectionState:
        return _ConnectionState(None, address, port)

    assert seam._in_allow(state("mybox.local", 80)) is True
    assert seam._in_allow(state("MYBOX.LOCAL", 80)) is True
    assert seam._in_allow(state("other.local", 8443)) is True
    assert seam._in_allow(state("other.local", 80)) is False  # port still counts
    assert seam._in_allow(state("elsewhere.local", 80)) is False
    assert RawSocketInterceptor()._in_allow(state("mybox.local", 80)) is False


# --- a peer wardex cannot name: unix sockets and a failing getpeername() ---

_PEER_UNRESOLVED = "interceptors.seam.peer_unresolved"


def _read_http_response(sock: socket.socket) -> bytes:
    """One Content-Length-framed HTTP/1.1 response, read off a raw socket."""
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(65536)
        if not chunk:
            return buf
        buf += chunk
    head, _, body = buf.partition(b"\r\n\r\n")
    want = 0
    for line in head.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            want = int(line.split(b":", 1)[1])
    while len(body) < want:
        chunk = sock.recv(65536)
        if not chunk:
            break
        body += chunk
    return head + b"\r\n\r\n" + body


@pytest.mark.usefixtures("fresh_counters")
def test_a_unix_socket_llm_call_ships_port_zero_and_says_the_peer_is_unresolved():
    """A local model server behind a unix socket is still an LLM call, and the
    span must not dress it up as a TCP connection to port 443.

    `socket.socketpair()` is `AF_UNIX`, the family httpx's `uds=`, docker-py and
    local model servers ride. Its `getpeername()` answers with a PATH, not a
    `(host, port)` pair, so there is no port to report. The seam used to fall
    back to the literal host `unknown` and port 443 and ship
    `http://unknown:443/...` with no marker and no counter — an invented
    address indistinguishable from a real one. The honest shape is port 0 in
    both `server.port` and the URL, the `PEER_UNRESOLVED` marker on the span,
    and one tick of the counter for the one sealed transaction.
    """
    body = b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}'
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
    )
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(_LLM_RESP)).encode() + b"\r\n\r\n" + _LLM_RESP
    )
    client_end, server_end = socket.socketpair()
    assert client_end.family == socket.AF_UNIX, "precondition: socketpair is a unix socket"

    def serve() -> None:
        buf = b""
        while len(buf) < len(request):
            chunk = server_end.recv(65536)
            if not chunk:
                return
            buf += chunk
        server_end.sendall(response)

    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        server = threading.Thread(target=serve, daemon=True)
        server.start()
        client_end.sendall(request)
        received = _read_http_response(client_end)
        server.join(timeout=5)
        assert received.endswith(_LLM_RESP), "precondition: the whole response arrived"

        spans = client_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.gen_ai is not None, "precondition: the call was identified as LLM traffic"
        assert Limitation.PEER_UNRESOLVED in span.capture_integrity.limitations
        assert span.server_port == 0
        assert span.server_address == "unknown"
        assert span.transport.http.url == "http://unknown:0/v1/chat/completions"
        assert counters.get(_PEER_UNRESOLVED) == 1
    finally:
        wardex.close()
        client_end.close()
        server_end.close()


def test_a_resolved_tcp_peer_carries_no_unresolved_marker():
    """The other half of the contract: a loopback TCP connection has a real
    address and port, so the marker would be a false alarm there."""
    httpd, host, port = _server(_LLM_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(host, port, b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}')
        spans = client_spans()
        assert len(spans) == 1
        assert Limitation.PEER_UNRESOLVED not in spans[0].capture_integrity.limitations
        assert spans[0].server_port == port
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


class _CountingPeerSocket:
    """A socket double whose `getpeername()` counts how often it is asked."""

    def __init__(self, peer: object = ("127.0.0.1", 8080)) -> None:
        self.peer = peer
        self.calls = 0

    def getpeername(self) -> object:
        self.calls += 1
        if isinstance(self.peer, BaseException):
            raise self.peer
        return self.peer

    def fileno(self) -> int:
        return -1


class _InlineClient:
    """Just enough client for a seam driven directly, without `install()`."""

    class config:  # noqa: N801 - mirrors `Client.config`
        debug = False

    def capture_span(self, span: object) -> None:
        pass

    def capture_deferred(self, job: object) -> None:
        pass


def test_the_peer_address_is_asked_once_per_connection_not_once_per_send():
    """`_state` already resolved the address when it built the connection, and
    `_on_request_bytes` used to ask the socket again on every send — a syscall
    per write on the hottest path the seam has, and on a memory-BIO
    `SSLObject` a raised-and-caught `AttributeError` per write. A resolved
    address cannot change while the connection lives, so three sends must cost
    exactly one query."""
    seam = RawSocketInterceptor()
    seam._client = _InlineClient()
    sock = _CountingPeerSocket()

    seam._on_request_bytes(sock, b"POST /v1/chat/completions HTTP/1.1\r\n")
    seam._on_request_bytes(sock, b"Host: localhost\r\nContent-Length: 2\r\n\r\n")
    seam._on_request_bytes(sock, b"{}")

    assert sock.calls == 1


def test_only_the_bare_unknown_placeholder_is_asked_again():
    """The one re-ask the per-send rule keeps, and what it may not claim.

    A connection whose address is the bare `unknown` placeholder is asked again
    on the next send, because a `server_hostname` that appears later is a
    better host for the span. A better host is still not a read peer: the port
    stays 0, which is what keeps the connection marked. Once the host is no
    longer the bare placeholder, nothing is asked again.
    """
    seam = RawSocketInterceptor()
    seam._client = _InlineClient()
    sock = _CountingPeerSocket(OSError("not connected"))
    sock.server_hostname = None

    seam._on_request_bytes(sock, b"POST /v1/chat/completions HTTP/1.1\r\n")
    st = seam._conns[id(sock)]
    assert (st.server_address, st.server_port) == ("unknown", 0)
    asked = sock.calls

    sock.server_hostname = "models.internal"
    seam._on_request_bytes(sock, b"Host: models.internal\r\n")
    assert sock.calls == asked + 1
    assert (st.server_address, st.server_port) == ("models.internal", 0)

    seam._on_request_bytes(sock, b"Content-Length: 2\r\n\r\n{}")
    assert sock.calls == asked + 1


@pytest.mark.parametrize(
    ("peer", "hostname", "expected"),
    [
        # A real INET/INET6 answer is used as-is.
        (("10.0.0.7", 8000), None, ("10.0.0.7", 8000)),
        (("::1", 8000, 0, 0), None, ("::1", 8000)),
        # AF_UNIX answers with a path. The old `int(peer[1])` read a path such
        # as "/9" as host "/" on port 9 — a resolved-looking lie — so the shape
        # is checked rather than indexed.
        ("/9.sock", None, ("unknown", 0)),
        ("", None, ("unknown", 0)),
        (b"\x00abstract", None, ("unknown", 0)),
        # getpeername() raising: the TLS server name is the best host there is,
        # and the port is still not known.
        (OSError("not connected"), "api.example.com", ("api.example.com", 0)),
        (OSError("not connected"), None, ("unknown", 0)),
    ],
)
def test_an_unread_peer_address_is_reported_as_port_zero(peer, hostname, expected):
    """Port 0 is the mark the seam keys `PEER_UNRESOLVED` on, so every shape
    that did not yield an INET address must land on it — and none that did."""
    sock = _CountingPeerSocket(peer)
    sock.server_hostname = hostname
    assert peer_address(sock) == expected


def test_plaintext_ws_requires_allowlist():
    # without an allowlist, non-LLM traffic (including ws paths) is dropped —
    # verifies the allowlist gate
    httpd, host, port = _server(_PLAIN_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)  # no allowlist
        # a plain response without 101 doesn't trigger the ws tracker — same path as non-LLM drop
        _post(host, port, b'{"x":1}', path="/ws")
        assert client_spans() == []
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


# --- `Connection: close` — the socket that closes before its body arrives ---

_WILL_CLOSE_RESP = json.dumps(
    {
        "id": "c1",
        "model": "gpt-4o",
        "choices": [
            {"finish_reason": "stop", "message": {"role": "assistant", "content": "x" * 100_000}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
).encode()


def _will_close_server(payload: bytes):
    """A one-shot HTTP/1.1 server that answers with `Connection: close`.

    Hand-rolled rather than `http.server`, because `BaseHTTPRequestHandler`
    will not emit that header while it is speaking HTTP/1.1 keep-alive, and the
    header is the entire point of the fixture.
    """
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()[:2]

    def serve() -> None:
        conn, _ = srv.accept()
        with conn:
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                buf += chunk
            head, _, body = buf.partition(b"\r\n\r\n")
            want = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    want = int(line.split(b":", 1)[1])
            while len(body) < want:
                body += conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Connection: close\r\n"
                b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
            )
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return host, port


def test_a_connection_close_response_is_captured_after_the_socket_was_closed():
    """`socket.close()` is not the end of the connection, and the seam must agree.

    `http.client.getresponse()` hands the connection to the response when the
    response `will_close` — a `Connection: close` header, HTTP/1.0, or a body
    with no length framing — and calls `sock.close()` as soon as the HEADERS
    are parsed. That call does not release anything: `socket.close()` only
    reaches `_real_close()` once `_io_refs` runs out, and the `makefile()`
    object holding the body still owns a reference. The rest of the body then
    arrives through the seam's own patched `recv_into`, AFTER the close.

    A close hook that fires there retires the connection mid-response: the
    remaining bytes build a fresh state, a response with no request ahead of it
    latches `gate = "ignore"`, and the span is never assembled. Silently — no
    counter, no limitation marker, just a missing span for every `will_close`
    response whose body outgrows one buffered read.
    """
    host, port = _will_close_server(_WILL_CLOSE_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        conn = http.client.HTTPConnection(host, port)
        conn.request(
            "POST",
            "/v1/chat/completions",
            b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}',
            {},
        )
        resp = conn.getresponse()
        assert resp.will_close, "precondition: http.client did not hand over the connection"
        received = resp.read()
        conn.close()

        assert len(received) == len(_WILL_CLOSE_RESP), (
            "precondition: the body must outgrow one buffered read, so that part of "
            "it arrives after http.client already called sock.close()"
        )
        spans = client_spans()
        assert len(spans) == 1, "the connection was retired while its body was still arriving"
        assert spans[0].gen_ai is not None
    finally:
        wardex.close()
