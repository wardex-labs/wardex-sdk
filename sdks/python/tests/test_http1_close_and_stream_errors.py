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
for bytes and got none. A close the CLIENT makes first is not it, nor is the
empty read after the client shut its own read side — that is a stream let go,
and shipping it as whole would claim an end nobody saw. And what counts as the
end is the response's own framing, which for an answer to HEAD or CONNECT the
request decides: no body, so nothing is in flight to be cut.

Over TLS the peer's EOF comes three ways, and all three count: an empty read
on an `ssl.SSLSocket`; on an `ssl.SSLObject` with no close_notify, a raised
`SSLEOFError` (anyio's TLS stream, under httpx's `AsyncClient`) or nothing at
all, the event loop's transport telling asyncio's TLS protocol (aiohttp). And
the default capture mode keeps a call whose reply failed or was cut on the
model its request named, even when no response byte named one.
"""

from __future__ import annotations

import asyncio
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
CHAT_RESPONSE = SSE_HEAD + f"Content-Length: {len(CHAT_SSE)}\r\n\r\n".encode() + CHAT_SSE
#: A whole response to HEAD: the length a GET would have had, and no body.
HEAD_RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: 1234\r\n\r\n"


def _client_tls() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


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


def _serve_once(
    *responses: bytes,
    tls: bool = False,
    hold: threading.Event | None = None,
    tunnel: bool = False,
    close_notify: bool = False,
) -> int:
    """One connection: for each of `responses`, read one request and write it; then close.

    `hold`: write the responses, then keep the connection open until the event
    is set — a response still in flight while the test acts. `tunnel`: be a
    proxy first — answer the CONNECT with a 200 and speak TLS inside it. Over
    TLS the close is a bare TCP close (what a Python `ssl` server does by
    default) unless `close_notify` sends TLS's own end-of-stream alert first.
    """
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    ctx = None
    if tls or tunnel:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(_CERTS / "cert.pem"), keyfile=str(_CERTS / "key.pem"))

    def run() -> None:
        conn, _ = srv.accept()
        try:
            if tunnel:
                _read_request(conn)
                conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            if ctx is not None:
                conn = ctx.wrap_socket(conn, server_side=True)
            for response in responses:
                _read_request(conn)
                conn.sendall(response)
            if hold is not None:
                hold.wait(5)
            if close_notify and isinstance(conn, ssl.SSLSocket):
                conn = conn.unwrap()
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
    ctx = _client_tls()
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


@pytest.mark.parametrize("tls", [False, True], ids=["plaintext", "tls"])
def test_a_client_that_shuts_its_own_read_side_has_let_the_stream_go(tls: bool):
    """After `shutdown(SHUT_RD)` the client's own reads come back empty while the
    server still holds the stream open. That empty read is the client letting
    go, not the server ending the body, so the span says the end was not seen."""
    hold = threading.Event()
    port = _serve_once(UNFRAMED + CUT_SSE, tls=tls, hold=hold)
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: llm.example.test\r\n"
        + f"Content-Length: {len(CHAT_REQUEST)}\r\n\r\n".encode()
        + CHAT_REQUEST
    )

    def drive() -> None:
        raw = socket.create_connection(("127.0.0.1", port), timeout=5)
        sock = _client_tls().wrap_socket(raw, server_hostname="llm.example.test") if tls else raw
        with sock:
            sock.sendall(request)
            sock.recv(65536)  # the headers and part of the body
            sock.shutdown(socket.SHUT_RD)
            while sock.recv(65536):  # ends at the client's own shutdown, not the server's
                pass

    try:
        span = _one_chat_span(_captured(drive))
    finally:
        hold.set()

    assert Limitation.FRAME_PARSE_FAILED in _markers(span)
    assert span.status is StatusCode.UNSET
    assert span.capture_integrity.truncated is True
    assert span.transport.response_size is None


@pytest.mark.parametrize("server_closes", [True, False], ids=["server-closes", "client-closes"])
def test_a_head_response_is_whole_at_its_header_block(server_closes: bool):
    """A response to HEAD has no body whatever its Content-Length says, so the
    exchange is complete when its headers arrive: whichever side then closes,
    nothing was cut short. (`ALL`: a HEAD names no provider.)"""
    hold = None if server_closes else threading.Event()
    port = _serve_once(HEAD_RESPONSE, hold=hold)

    def drive() -> None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("HEAD", "/index.html")
        conn.getresponse().read()
        if server_closes:
            assert conn.sock.recv(1) == b""  # the server's close, after its whole response
        conn.close()

    try:
        (span,) = _captured(drive, CaptureMode.ALL)
    finally:
        if hold is not None:
            hold.set()

    assert span.transport.http.method == "HEAD"
    assert span.status is StatusCode.OK
    assert span.capture_integrity.truncated is False
    assert span.transport.response_size == 0
    assert Limitation.FRAME_PARSE_FAILED not in _markers(span)


def test_the_response_after_a_head_on_a_kept_alive_connection_is_its_own():
    """Read as a body, the HEAD's declared length would swallow the next response
    on the connection. Both exchanges ship, each whole."""
    port = _serve_once(HEAD_RESPONSE, CHAT_RESPONSE)

    def drive() -> None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("HEAD", "/index.html")
        conn.getresponse().read()
        conn.request("POST", "/v1/chat/completions", CHAT_REQUEST, {"Content-Type": "x"})
        conn.getresponse().read()
        conn.close()

    spans = _captured(drive, CaptureMode.ALL)

    assert sorted(s.transport.http.method for s in spans) == ["HEAD", "POST"]
    chat = _one_chat_span(spans)
    assert chat.status is StatusCode.OK
    assert chat.gen_ai.finish_reasons == ("stop",)
    assert chat.transport.response_size == len(CHAT_SSE)
    assert not any(Limitation.FRAME_PARSE_FAILED in _markers(s) for s in spans)


def test_a_proxy_tunnel_ships_the_call_inside_and_no_cut_opening():
    """A 2xx to CONNECT has no body: from its header block on the connection is a
    tunnel. The call inside ships (the TLS seam sees it); the opening exchange is
    not a response cut short when the plain socket is handed over to TLS."""
    port = _serve_once(CHAT_RESPONSE, tunnel=True)

    def drive() -> None:
        conn = http.client.HTTPSConnection("127.0.0.1", port, timeout=5, context=_client_tls())
        conn.set_tunnel("llm.example.test", 443)
        conn.request("POST", "/v1/chat/completions", CHAT_REQUEST, {"Content-Type": "x"})
        conn.getresponse().read()
        conn.close()

    spans = _captured(drive, CaptureMode.ALL)

    assert [s.transport.http.method for s in spans] == ["POST"]
    (span,) = spans
    assert span.transport.http.url.startswith("https://llm.example.test:")
    assert span.status is StatusCode.OK
    assert Limitation.FRAME_PARSE_FAILED not in _markers(span)


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


#: An OpenAI-compatible error chunk that ALSO carries `choices`, after one
#: ordinary chunk. OpenRouter documents its mid-stream error this way (the error
#: at the top level plus a choice finishing with "error"); a choice with no
#: finish, the other shape, must not hide the error either.
ERROR_WITH_CHOICES = {
    "error-and-finishing-choice": (
        {"code": 502, "message": "Provider returned error", "metadata": {"error_type": "upstream"}},
        [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
        "502",
    ),
    "error-and-unfinished-choice": (
        {"message": "boom", "type": "server_error", "param": None, "code": None},
        [{"index": 0, "delta": {}, "finish_reason": None}],
        "server_error",
    ),
}


@pytest.mark.parametrize("shape", sorted(ERROR_WITH_CHOICES))
def test_a_provider_error_that_also_carries_choices_fails_the_chat_span(shape):
    """The top-level `error` is the provider's failure whatever else its chunk
    holds: a choice alongside it does not turn the call into a success, and the
    error object stays in the body."""
    error, choices, error_type = ERROR_WITH_CHOICES[shape]
    first = {"id": "c1", "model": "gpt-4o", "choices": [{"index": 0, "delta": {"content": "Hel"}}]}
    failing = {"id": "c1", "model": "gpt-4o", "error": error, "choices": choices}
    stream = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in (first, failing))
    response = SSE_HEAD + f"Content-Length: {len(stream)}\r\n\r\n".encode() + stream
    port = _serve_once(response)

    span = _one_chat_span(_captured(lambda: _post(port, "/v1/chat/completions", CHAT_REQUEST)))

    assert span.status is StatusCode.ERROR
    assert span.error_type == error_type
    assert span.gen_ai.finish_reasons == ("error",)
    assert json.loads(span.output_data)["error"] == error
    assert Limitation.FRAME_PARSE_FAILED not in _markers(span)


#: A stream whose ONLY event is the provider's failure: no chunk named the model
#: or an id first, so nothing on the response side says "LLM call" except the
#: error itself. Nothing in either stream format promises a model-bearing chunk
#: ahead of an `error` event.
ERROR_ONLY_STREAMS = {
    "openai": (
        "/v1/chat/completions",
        CHAT_REQUEST,
        b'data: {"error":{"message":"The server had an error while processing your request.",'
        b'"type":"server_error","param":null,"code":null}}\n\n',
        "server_error",
    ),
    "anthropic": (
        "/v1/messages",
        json.dumps(
            {
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
        ).encode(),
        b"event: error\n"
        b'data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n',
        "overloaded_error",
    ),
}


@pytest.mark.parametrize("framing", ["content-length", "unframed"])
@pytest.mark.parametrize("provider", sorted(ERROR_ONLY_STREAMS))
def test_a_stream_that_is_only_the_providers_error_fails_the_chat_span(provider, framing):
    """The default mode keeps a call whose 200 stream carried nothing but the
    provider's failure: the request named the model, and the reply is the
    provider failing. It ships ERROR with the provider's class and the error
    object in the body — not dropped as "not an LLM call", and not a success."""
    path, request, stream, error_type = ERROR_ONLY_STREAMS[provider]
    if framing == "content-length":
        response = SSE_HEAD + f"Content-Length: {len(stream)}\r\n\r\n".encode() + stream
    else:
        response = UNFRAMED + stream
    port = _serve_once(response)

    span = _one_chat_span(_captured(lambda: _post(port, path, request)))

    assert span.status is StatusCode.ERROR
    assert span.error_type == error_type
    assert span.gen_ai.finish_reasons == ("error",)
    assert span.gen_ai.request_model == json.loads(request)["model"]
    assert span.gen_ai.response_model is None  # no chunk named one; none is invented
    assert json.loads(span.output_data)["error"]["type"] == error_type
    assert span.capture_integrity.truncated is False
    assert Limitation.FRAME_PARSE_FAILED not in _markers(span)


def _post_tls(port: int, server_name: str, path: str, body: bytes) -> None:
    """One POST over TLS to `server_name` (the name the client asks for, SNI), read to the close."""
    raw = socket.create_connection(("127.0.0.1", port), timeout=5)
    with _client_tls().wrap_socket(raw, server_hostname=server_name) as tls:
        tls.sendall(
            f"POST {path} HTTP/1.1\r\nHost: {server_name}\r\n".encode()
            + f"Content-Length: {len(body)}\r\n\r\n".encode()
            + body
        )
        while tls.recv(65536):
            pass


@pytest.mark.parametrize(
    ("cut", "server_name"),
    [
        # Mid first event: what arrived names no model, but it is the endpoint's stream.
        pytest.param(20, None, id="mid-first-event"),
        # Not one body byte: the status line and headers are all that was observed. Over TLS to
        # the provider's own name, which is what names the provider when no body byte can.
        pytest.param(0, "api.openai.com", id="headers-only"),
    ],
)
def test_a_body_cut_before_its_first_event_still_ships_marked(cut: int, server_name: str | None):
    """The default mode keeps a call whose response was cut before any event
    named the model: the request named it, and the reply never finished. What
    was observed ships (status, headers, the bytes that came), marked; nothing
    else is made up — no response model, no finish, no size, no success."""
    response = SSE_HEAD + f"Content-Length: {len(CHAT_SSE)}\r\n\r\n".encode() + CHAT_SSE[:cut]
    port = _serve_once(response, tls=server_name is not None)

    def drive() -> None:
        if server_name is None:
            _post(port, "/v1/chat/completions", CHAT_REQUEST)
        else:
            _post_tls(port, server_name, "/v1/chat/completions", CHAT_REQUEST)

    span = _one_chat_span(_captured(drive))

    assert span.gen_ai.request_model == "gpt-4o"
    assert span.gen_ai.response_model is None
    assert span.gen_ai.finish_reasons is None
    assert span.transport.http.status_code == 200
    assert span.status is StatusCode.UNSET
    assert span.capture_integrity.truncated is True
    assert span.transport.response_size is None
    assert Limitation.FRAME_PARSE_FAILED in _markers(span)


async def _httpx_async_read(port: int) -> bytes:
    import httpx

    async with httpx.AsyncClient(verify=_client_tls()) as client:
        async with client.stream(
            "POST", f"https://127.0.0.1:{port}/v1/chat/completions", content=CHAT_REQUEST
        ) as resp:
            return b"".join([chunk async for chunk in resp.aiter_raw()])


async def _aiohttp_read(port: int) -> bytes:
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://127.0.0.1:{port}/v1/chat/completions", data=CHAT_REQUEST, ssl=_client_tls()
        ) as resp:
            return await resp.read()


@pytest.mark.parametrize("close_notify", [False, True], ids=["bare-close", "close-notify"])
@pytest.mark.parametrize(
    "read",
    [
        # anyio's TLSStream over an `ssl.SSLObject`: what the async OpenAI and Anthropic SDKs ride.
        pytest.param(_httpx_async_read, id="httpx-async"),
        # asyncio's own TLS protocol over an `ssl.SSLObject`.
        pytest.param(_aiohttp_read, id="aiohttp"),
    ],
)
def test_an_async_tls_body_with_no_framing_ships_whole_when_the_server_closes(read, close_notify):
    """The async TLS clients read through an `ssl.SSLObject`, which reports a
    server's close differently from a socket: without TLS's close_notify (a
    Python `ssl` server's default) it raises instead of reading empty, or says
    nothing at all and the event loop's transport is told. Either way the
    client takes it as the body's end, so the span does too — and it ships at
    that close, not whenever the object happens to be collected."""
    port = _serve_once(UNFRAMED + CHAT_SSE, tls=True, close_notify=close_notify)
    got: list[bytes] = []

    span = _one_chat_span(_captured(lambda: got.append(asyncio.run(read(port)))))

    assert got == [CHAT_SSE]  # what the client received, whole
    assert span.status is StatusCode.OK
    assert span.gen_ai.finish_reasons == ("stop",)
    assert span.transport.response_size == len(CHAT_SSE)
    assert span.capture_integrity.truncated is False
    assert Limitation.FRAME_PARSE_FAILED not in _markers(span)
