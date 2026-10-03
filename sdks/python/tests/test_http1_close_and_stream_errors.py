"""What the byte seam ships when an HTTP/1 response ends with its connection,
and when a provider fails in the middle of a stream.

Real loopback sockets through the installed seam, under the default capture
mode, read back off a `RecordingTransport` — the spans a user would receive.

Two holes, both silent before:

* A response whose end was the connection's never became a span. A body with
  no Content-Length and no chunking is ENDED by the peer closing its side (that
  is its framing), so a perfectly normal streaming chat call from such a server
  left no trace; a body cut short of the length it declared, or let go by the
  client, left none either.
* A stream whose provider sent an `error` event mid-stream shipped as a
  success: the error object was dropped by the reassembler, so the span had
  status OK, no finish reason, and nothing in its body naming the failure.

The line between "whole" and "cut short" is the peer's EOF: a read that asked
for bytes and got none. A close the CLIENT makes first is not it — that is a
stream let go, and shipping it as whole would claim an end nobody saw.
"""

from __future__ import annotations

import http.client
import json
import socket
import ssl
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

import wardex_sdk as wardex
from conftest import llm_fixture
from wardex_sdk._assembly import Limitation
from wardex_sdk._enums import CaptureMode, StatusCode
from wardex_sdk.testing import RecordingTransport

_CERTS = Path(__file__).parent / "fixtures"

CHAT_REQUEST = json.dumps(
    {"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
).encode()
CHAT_SSE = (
    b'data: {"id":"c1","model":"gpt-4o","choices":[{"index":0,'
    b'"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
    b'data: {"id":"c1","model":"gpt-4o","choices":[{"index":0,'
    b'"delta":{"content":"lo"},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)
SSE_HEAD = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
UNFRAMED = SSE_HEAD + b"Connection: close\r\n\r\n"
#: The first event whole and the second cut: the call is still recognisable as a
#: chat call (the first event names the model), and the stream never finished.
CUT_SSE = CHAT_SSE[: CHAT_SSE.index(b"\n\n") + 12]


def _read_request(conn: socket.socket) -> None:
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = conn.recv(65536)
        if not chunk:
            return
        buf += chunk
    head, _, body = buf.partition(b"\r\n\r\n")
    length = next(
        (
            int(line.split(b":", 1)[1])
            for line in head.split(b"\r\n")
            if line.lower().startswith(b"content-length:")
        ),
        0,
    )
    while len(body) < length:
        body += conn.recv(65536)


def _serve_once(response: bytes, *, tls: bool = False, hold: threading.Event | None = None) -> int:
    """One connection: read one request, write `response`, close.

    `hold`: write `response`, then keep the connection open until the event is
    set — a response still in flight while the test acts.
    """
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    ctx = None
    if tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(_CERTS / "cert.pem"), keyfile=str(_CERTS / "key.pem"))

    def run() -> None:
        conn, _ = srv.accept()
        try:
            if ctx is not None:
                conn = ctx.wrap_socket(conn, server_side=True)
            _read_request(conn)
            conn.sendall(response)
            if hold is not None:
                hold.wait(5)
        except OSError:
            pass  # the client let go first; nothing here is under test
        finally:
            conn.close()
            srv.close()

    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1]


def _post(port: int, path: str, body: bytes) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)  # plaintext: the socket seam
    conn.request("POST", path, body, {"Content-Type": "application/json"})
    try:
        conn.getresponse().read()
    except http.client.IncompleteRead:
        pass  # the cut-short cases: the client sees the cut, and so must the span
    conn.close()


def _captured(drive: Callable[[], None], mode: CaptureMode = CaptureMode.AGENT) -> list:
    """Every span the installed seam shipped while `drive` ran."""
    transport = RecordingTransport()
    wardex.init(transport=transport, intercept=True, capture_mode=mode)
    try:
        drive()
    finally:
        wardex.close()
    return [s for env in transport.envelopes for s in env.spans]


def _one_chat_span(spans: list):
    chat = [s for s in spans if s.gen_ai is not None]
    assert len(chat) == 1, spans
    return chat[0]


def _markers(span) -> set[Limitation]:
    return set(span.capture_integrity.limitations)


def test_a_body_with_no_framing_ships_whole_when_the_server_closes():
    """The peer's EOF IS this body's framing, so the span is an ordinary
    success: whole body, a size, the finish the stream declared, no truncation
    — and, shipped at the EOF while the socket is still there, its timing."""
    port = _serve_once(UNFRAMED + CHAT_SSE)

    span = _one_chat_span(_captured(lambda: _post(port, "/v1/chat/completions", CHAT_REQUEST)))

    assert span.status is StatusCode.OK
    assert span.gen_ai.finish_reasons == ("stop",)
    assert span.transport.http.status_code == 200
    assert span.transport.response_size == len(CHAT_SSE)
    assert span.capture_integrity.truncated is False
    assert Limitation.FRAME_PARSE_FAILED not in _markers(span)
    assert Limitation.CONNECT_TIMING_UNAVAILABLE not in _markers(span)
    assert span.transport.timing.tcp_connect_ms is not None


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            SSE_HEAD + f"Content-Length: {len(CHAT_SSE)}\r\n\r\n".encode() + CUT_SSE,
            id="content-length",
        ),
        pytest.param(
            SSE_HEAD
            + b"Transfer-Encoding: chunked\r\n\r\n"
            + f"{len(CHAT_SSE):x}\r\n".encode()
            + CUT_SSE,
            id="chunked",
        ),
    ],
)
def test_a_body_cut_short_of_its_framing_ships_marked(response: bytes):
    """What arrived ships, and says it is partial; what did not arrive is not
    made up — no finish reason, no size, and no claim the call succeeded."""
    port = _serve_once(response)

    span = _one_chat_span(_captured(lambda: _post(port, "/v1/chat/completions", CHAT_REQUEST)))

    assert Limitation.FRAME_PARSE_FAILED in _markers(span)
    assert span.capture_integrity.truncated is True
    assert span.transport.response_size is None
    assert span.transport.http.status_code == 200  # observed, so kept
    assert span.status is StatusCode.UNSET  # the end was not observed
    assert span.error_type is None
    assert span.gen_ai.finish_reasons is None
    assert b"Hel" in span.output_data


def test_an_error_status_cut_short_stays_an_error():
    """The failure was observed in the status line, so the cut does not hide it.
    (`ALL`: the cut body names no provider, so the default mode would not keep it.)"""
    body = json.dumps({"error": {"type": "server_error", "message": "x" * 64}}).encode()
    response = (
        b"HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body[:20]
    )
    port = _serve_once(response)

    spans = _captured(lambda: _post(port, "/v1/chat/completions", CHAT_REQUEST), CaptureMode.ALL)

    (span,) = spans
    assert span.status is StatusCode.ERROR
    assert span.error_type == "500"
    assert Limitation.FRAME_PARSE_FAILED in _markers(span)


def test_a_close_before_the_response_headers_completed_ships_nothing():
    """No status, no headers: nothing was observed that a span could report."""
    port = _serve_once(b"HTTP/1.1 200 OK\r\nContent-Ty")

    def drive() -> None:
        try:
            _post(port, "/v1/chat/completions", CHAT_REQUEST)
        except http.client.HTTPException:
            pass

    assert _captured(drive) == []


def test_a_stream_the_client_lets_go_ships_marked_not_whole():
    """The client stops reading a body with no framing and closes (a user
    cancelling a stream). The server never ended it, so the span must not say
    it did: it ships as what arrived, marked, with no success claimed."""
    hold = threading.Event()
    port = _serve_once(UNFRAMED + CUT_SSE, hold=hold)

    def drive() -> None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/v1/chat/completions", CHAT_REQUEST, {"Content-Type": "x"})
        resp = conn.getresponse()
        resp.read(10)  # some of the body crossed the seam; its end never will
        resp.close()
        conn.close()

    try:
        span = _one_chat_span(_captured(drive))
    finally:
        hold.set()

    assert Limitation.FRAME_PARSE_FAILED in _markers(span)
    assert span.status is StatusCode.UNSET
    assert span.capture_integrity.truncated is True
    assert span.transport.response_size is None


def test_a_response_still_in_flight_when_wardex_lets_go_ships_nothing():
    """`wardex.close()` while a body is still arriving and the client still
    holds it: the connection goes on, nothing has ended, and wardex stops
    watching without claiming an end either way."""
    hold = threading.Event()
    port = _serve_once(UNFRAMED + CUT_SSE, hold=hold)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    held: list = []

    def drive() -> None:
        conn.request("POST", "/v1/chat/completions", CHAT_REQUEST, {"Content-Type": "x"})
        held.append(conn.getresponse())
        held[0].read(10)  # the headers and a little body crossed the seam

    try:
        spans = _captured(drive)
    finally:
        hold.set()
        for resp in held:
            resp.close()
        conn.close()

    assert spans == []


@pytest.mark.parametrize("peer_closes", [True, False], ids=["peer-eof", "client-lets-go"])
def test_a_tls_response_ending_with_its_connection_names_the_tls_host(peer_closes: bool):
    """Over TLS too, and the URL names the server the client asked for, not the
    peer's address — also when the span is sealed after the socket is gone (the
    client let go, so only the close could end it)."""
    hold = None if peer_closes else threading.Event()
    port = _serve_once(UNFRAMED + (CHAT_SSE if peer_closes else CUT_SSE), tls=True, hold=hold)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: llm.example.test\r\n"
        + f"Content-Length: {len(CHAT_REQUEST)}\r\n\r\n".encode()
        + CHAT_REQUEST
    )

    def drive() -> None:
        raw = socket.create_connection(("127.0.0.1", port), timeout=5)
        with ctx.wrap_socket(raw, server_hostname="llm.example.test") as tls:
            tls.sendall(request)
            if peer_closes:
                while tls.recv(65536):
                    pass
            else:
                tls.recv(65536)  # the headers and part of the body; then the client closes

    try:
        span = _one_chat_span(_captured(drive))
    finally:
        if hold is not None:
            hold.set()

    assert span.transport.http.url.startswith("https://llm.example.test:")
    assert span.status is (StatusCode.OK if peer_closes else StatusCode.UNSET)
    assert (Limitation.FRAME_PARSE_FAILED in _markers(span)) is not peer_closes


@pytest.mark.parametrize(
    ("case", "path", "error_type", "message"),
    [
        (
            "openai_chat_sse_error",
            "/v1/chat/completions",
            "server_error",
            "The server had an error while processing your request.",
        ),
        ("anthropic_messages_sse_error", "/v1/messages", "overloaded_error", "Overloaded"),
    ],
)
def test_a_provider_error_mid_stream_fails_the_chat_span(case, path, error_type, message):
    """The provider declared the failure in its own stream: the span is ERROR
    with the provider's class, the finish says `error`, and the body keeps the
    error object — the only bytes that say what went wrong."""
    stream = llm_fixture(case, "stream.sse")
    response = SSE_HEAD + f"Content-Length: {len(stream)}\r\n\r\n".encode() + stream
    port = _serve_once(response)

    span = _one_chat_span(_captured(lambda: _post(port, path, llm_fixture(case, "request.json"))))

    assert span.status is StatusCode.ERROR
    assert span.error_type == error_type
    assert span.gen_ai.finish_reasons == ("error",)
    body = json.loads(span.output_data)
    assert body["error"]["message"] == message
    assert Limitation.FRAME_PARSE_FAILED not in _markers(span)
