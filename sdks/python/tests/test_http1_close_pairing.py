"""Which request a reply that ends with its connection answers, when the seam did
not see every request byte.

Two rules meet on an HTTP/1 connection, and these tests hold them together:

* A reply the connection's close ends ships: whole at the server's close when
  it has no framing, marked as cut when its framing promised more.
* A reply whose request the seam did not see whole is never paired with
  another request: none of it seen, the reply is counted and not shipped; only
  its start seen, it ships as `HTTP ? /` and is counted.

So a reply the close ends is paired exactly as any other final reply is. And the
one thing a reply takes from its request, its framing (a reply to HEAD has no
body, a 2xx to CONNECT opens a tunnel), is taken only from a request the bytes
identify. When the bytes after an unfinished request read as two different
requests equally well, the reply is not framed by either guess: framed by a
wrong HEAD, its body is read as the next reply and the connection's parser
stops; framed by a wrong CONNECT, every later call on the connection is lost
without a count. When both readings hold a request of the same method, the
framing is no guess, and the reply is framed by that method.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading

import aiohttp
import httpx
import pytest

import wardex_sdk as wardex
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._enums import StatusCode
from wardex_sdk._interceptors._trackers import _Http1Tracker
from wardex_sdk.testing import RecordingTransport

pytestmark = pytest.mark.usefixtures("fresh_counters")

_PATH = "/v1/chat/completions"
_UNFINISHED = "protocol.http1.request_unfinished"
_UNOBSERVED = "protocol.http1.request_unobserved"
_EARLY_UNFRAMED = b"HTTP/1.1 413 Payload Too Large\r\nConnection: close\r\n\r\ntoo large"
# Small kernel buffers on both ends, so no single `send()` or `sendmsg()` takes a
# large request whole and asyncio sends the rest with `sendmsg` from Python 3.12.
# Linux doubles the value it is given; both stay far below `_LARGE`.
_SMALL_BUFFER = 32 * 1024
_LARGE = 1024 * 1024 + 4096


def _asyncio_writes_with_sendmsg() -> bool:
    """The test CPython's `asyncio.selector_events` makes when it is imported:
    from Python 3.12, with `socket.sendmsg` present and `SC_IOV_MAX` known to
    `os.sysconf`, the plaintext socket transport sends every `writelines()`, and
    whatever the kernel did not take of a `write()`'s first `send()`, with
    `sendmsg`. Anywhere else it writes with `send` alone."""
    if sys.version_info < (3, 12) or not hasattr(socket.socket, "sendmsg"):
        return False
    try:
        os.sysconf("SC_IOV_MAX")
    except (OSError, ValueError):
        return False
    return True


def _head(n: int, path: str = _PATH) -> bytes:
    return f"POST {path} HTTP/1.1\r\nHost: h\r\nContent-Length: {n}\r\n\r\n".encode()


def _completion(tag: str) -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{tag}",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
    ).encode()


def _reply(tag: str, status: str = "200 OK") -> bytes:
    payload = _completion(tag)
    return f"HTTP/1.1 {status}\r\nContent-Length: {len(payload)}\r\n\r\n".encode() + payload


def _exchange(tracker: _Http1Tracker, tag: str, path: str = _PATH) -> list:
    body = json.dumps({"model": "gpt-4o", "tag": tag}).encode()
    tracker.on_request_bytes(_head(len(body), path) + body)
    return tracker.on_response_bytes(_reply(tag))


# --- a reply the close ends, to a request not seen whole ----------------------


def test_a_reply_the_close_cuts_to_a_request_seen_only_in_part_ships_unpaired_and_marked():
    """The upload's body went out where the seam does not look, the server answered
    it early, and the connection closed before that answer's declared length. The
    cut ships as what arrived, and its request half is the honest `? /`."""
    tracker = _Http1Tracker()
    tracker.on_request_bytes(_head(100, "/upload") + b"x" * 10)
    cut = b"HTTP/1.1 413 Payload Too Large\r\nContent-Length: 64\r\n\r\n0123456789"
    assert tracker.on_response_bytes(cut) == []

    (txn,) = tracker.on_connection_close(Limitation.WS_NO_CLOSE)

    assert (txn.method, txn.path, txn.request_body, txn.request_counted) == ("?", "/", b"", False)
    assert (txn.status, txn.response_body, txn.response_cut) == (413, b"0123456789", True)
    assert Limitation.FRAME_PARSE_FAILED in txn.limitations
    assert counters.get(_UNFINISHED) == 1


@pytest.mark.parametrize("rest_seen", [False, True], ids=["rest-unseen", "rest-seen-before-close"])
def test_an_unframed_early_reply_ships_whole_at_the_servers_close(rest_seen: bool):
    """A reply with no framing is ended by the server's close, and only then does it
    pair. If the rest of the upload it answered reached the seam by then, every
    request byte was seen and the reply pairs with it; if not, it is `? /`."""
    tracker = _Http1Tracker()
    tracker.on_request_bytes(_head(100, "/upload") + b"x" * 10)
    assert tracker.on_response_bytes(_EARLY_UNFRAMED) == []
    if rest_seen:
        tracker.on_request_bytes(b"x" * 90)

    (txn,) = tracker.on_response_eof()

    assert (txn.status, txn.response_body, txn.response_cut) == (413, b"too large", False)
    assert Limitation.FRAME_PARSE_FAILED not in txn.limitations
    if rest_seen:
        assert (txn.method, txn.path, txn.request_body) == ("POST", "/upload", b"x" * 100)
        assert txn.request_counted
        assert counters.get(_UNFINISHED) == 0
    else:
        assert (txn.method, txn.request_body, txn.request_counted) == ("?", b"", False)
        assert counters.get(_UNFINISHED) == 1


@pytest.mark.parametrize("ending", ["server-close", "cut"])
def test_a_reply_the_close_ends_to_a_request_not_seen_at_all_is_counted_not_shipped(ending: str):
    """The next request was written where the seam does not look. Its reply, ended
    by the close, has nothing to pair with: counted, never shipped as `? /` beside
    the call before it."""
    tracker = _Http1Tracker()
    (first,) = _exchange(tracker, "tag0")
    assert first.method == "POST"

    if ending == "server-close":
        payload = _completion("tag1")
        assert tracker.on_response_bytes(b"HTTP/1.1 200 OK\r\n\r\n" + payload) == []
        assert tracker.on_response_eof() == []
    else:
        assert tracker.on_response_bytes(b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\n\r\npart") == []
        assert tracker.on_connection_close(Limitation.WS_NO_CLOSE) == []
    assert counters.get(_UNOBSERVED) == 1


# --- the framing a reply takes from its request --------------------------------


def test_a_head_right_after_an_unfinished_request_is_framed_as_head():
    """Only one reading of the bytes after the unfinished request holds a request,
    so the reply is framed by it: no body, and the next reply is read on its own."""
    tracker = _Http1Tracker()
    tracker.on_request_bytes(_head(5000, "/first"))  # its body went by unseen
    (unpaired,) = tracker.on_response_bytes(_reply("tag1"))
    assert unpaired.method == "?"

    tracker.on_request_bytes(b"HEAD /page HTTP/1.1\r\nHost: h\r\n\r\n")
    (head,) = tracker.on_response_bytes(b"HTTP/1.1 200 OK\r\nContent-Length: 1234\r\n\r\n")
    assert (head.method, head.path, head.response_body) == ("HEAD", "/page", b"")

    (txn,) = _exchange(tracker, "tag3")
    assert (txn.method, txn.path) == ("POST", _PATH)
    assert b"chatcmpl-tag3" in txn.response_body
    assert tracker.disabled_reason() is None


@pytest.mark.parametrize(
    "phantom",
    [
        pytest.param(b"HEAD / HTTP/1.0\r\n\r\n", id="head"),
        pytest.param(b"CONNECT proxy.example:443 HTTP/1.1\r\n\r\n", id="connect"),
    ],
)
def test_a_reply_is_not_framed_by_a_request_that_only_won_a_tie(phantom: bytes):
    """The unfinished request's unseen rest is exactly as long as the next request
    up to its last bytes, and those bytes are a whole request themselves. Read on
    from the unfinished request, the stream holds that request; read fresh, it holds
    the next one. Nothing in the bytes tells them apart, so the reply keeps the
    framing its own headers give it, and the call after it still pairs."""
    second = b"[tag2] " + phantom
    data = _head(len(second), "/second") + second
    tracker = _Http1Tracker()
    tracker.on_request_bytes(_head(len(data) - len(phantom), "/first"))  # its body went unseen
    tracker.on_response_bytes(_reply("tag1"))
    tracker.on_request_bytes(data)

    (reply,) = tracker.on_response_bytes(_reply("tag2"))
    assert b"chatcmpl-tag2" in reply.response_body, "the reply's body was framed away"

    (txn,) = _exchange(tracker, "tag3", "/third")
    assert (txn.method, txn.path) == ("POST", "/third")
    assert b"chatcmpl-tag3" in txn.response_body
    assert tracker.disabled_reason() is None
    assert counters.get(_UNFINISHED) == 1


@pytest.mark.parametrize("writes", [1, 2], ids=["rest-and-head-in-one-write", "separate-writes"])
def test_two_readings_that_tie_on_the_same_head_frame_its_reply_as_head(writes: int):
    """The server answered an upload early, and the upload's late rest is itself a
    whole `HEAD / HTTP/1.0` request; a real HEAD follows. Read on from the upload,
    the stream holds that HEAD; read fresh, it holds the rest-shaped HEAD and then
    the same HEAD. The readings tie, but both say HEAD, so framing the reply as HEAD
    is no guess: its Content-Length promises no body, and the replies after it are
    read on their own instead of being swallowed as that body."""
    rest = b"HEAD / HTTP/1.0\r\n\r\n"
    real_head = b"HEAD /page HTTP/1.1\r\nHost: h\r\n\r\n"
    tracker = _Http1Tracker()
    tracker.on_request_bytes(_head(10 + len(rest), "/upload") + b"x" * 10)
    (early,) = tracker.on_response_bytes(_reply("tag1", "413 Payload Too Large"))
    assert (early.method, early.status) == ("?", 413)
    for chunk in [rest + real_head] if writes == 1 else [rest, real_head]:
        tracker.on_request_bytes(chunk)

    (head,) = tracker.on_response_bytes(b"HTTP/1.1 200 OK\r\nContent-Length: 1234\r\n\r\n")
    assert (head.method, head.path, head.response_body) == ("HEAD", "/page", b"")

    (txn,) = _exchange(tracker, "tag3", "/third")
    assert (txn.method, txn.path) == ("POST", "/third")
    assert b"chatcmpl-tag3" in txn.response_body
    assert tracker.disabled_reason() is None
    assert counters.get(_UNFINISHED) == 1


# --- on a real event loop ------------------------------------------------------


def _serve_unframed_once(reply: bytes) -> int:
    """One connection: read one request whole, answer with a reply the close ends."""
    srv = socket.socket()
    # On the LISTENING socket, so the accepted socket inherits it.
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SMALL_BUFFER)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def run() -> None:
        conn, _ = srv.accept()
        try:
            buf = b""
            while b"\r\n\r\n" not in buf:
                buf += conn.recv(65536)
            head, _, body = buf.partition(b"\r\n\r\n")
            length = next(
                int(line.split(b":", 1)[1])
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            while len(body) < length:
                body += conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                b"Connection: close\r\n\r\n" + reply
            )
        finally:
            conn.close()
            srv.close()

    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1]


async def _post_httpx(port: int, body: bytes) -> None:
    opts = [(socket.SOL_SOCKET, socket.SO_SNDBUF, _SMALL_BUFFER)]
    transport = httpx.AsyncHTTPTransport(socket_options=opts)
    async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=30) as client:
        reply = await client.post(f"http://127.0.0.1:{port}{_PATH}", content=body)
        assert reply.status_code == 200


def _small_send_buffer_socket(addr_info: tuple) -> socket.socket:
    family, type_, proto, _, _ = addr_info
    sock = socket.socket(family=family, type=type_, proto=proto)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SMALL_BUFFER)
    return sock


async def _post_aiohttp(port: int, body: bytes) -> None:
    connector = aiohttp.TCPConnector(socket_factory=_small_send_buffer_socket)
    async with aiohttp.ClientSession(connector=connector) as session:
        headers = {"Content-Type": "application/json"}
        async with session.post(f"http://127.0.0.1:{port}{_PATH}", data=body, headers=headers) as r:
            assert r.status == 200
            await r.read()


@pytest.mark.parametrize(
    "post", [pytest.param(_post_httpx, id="httpx"), pytest.param(_post_aiohttp, id="aiohttp")]
)
def test_a_request_asyncio_sent_with_sendmsg_pairs_with_the_reply_the_server_close_ends(
    monkeypatch, post
):
    """A request asyncio sent at least in part with `sendmsg`, answered by a server
    that ends its reply by closing: one span, carrying the whole request and the
    whole reply, ended at the server's close and not marked as cut.

    Each client writes a 1 MiB request through a socket whose send buffer is far
    smaller, so no kernel takes it in one call. httpx hands asyncio one `write()`:
    its first `send()` takes part and `sendmsg` sends the rest. aiohttp hands it
    `writelines()`, which goes out with `sendmsg` from the first byte, except on
    CPython before 3.12.9 and on 3.13.0 and 3.13.1, where aiohttp avoids
    `writelines()` (CVE-2024-12254) and joins the request into one `write()` that
    goes out as httpx's does. Left to the kernel's own buffer, Linux took that
    `write()` of a 64 KiB request whole in its first `send()`, so `sendmsg` was
    never called.

    Where asyncio has no `sendmsg` path (before Python 3.12), the same exchange
    goes out with `send` alone and must pair the same way."""
    body = json.dumps(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "x" * _LARGE}]}
    ).encode()
    reply = _completion("tag0")
    sendmsg_calls = []
    if hasattr(socket.socket, "sendmsg"):
        real = socket.socket.sendmsg

        def counted(this, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            sendmsg_calls.append(1)
            return real(this, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "sendmsg", counted)
    port = _serve_unframed_once(reply)
    recorder = RecordingTransport()
    wardex.init(transport=recorder, intercept=True)
    try:
        asyncio.run(post(port, body))
        wardex.flush()
    finally:
        wardex.close()

    if _asyncio_writes_with_sendmsg():
        assert sendmsg_calls, "precondition: asyncio sent part of the request with sendmsg"
    else:
        assert not sendmsg_calls, "precondition: asyncio here writes with send alone"
    spans = [s for e in recorder.envelopes for s in e.spans if s.name.startswith("HTTP ")]
    (span,) = spans
    assert span.name == f"HTTP POST {_PATH}"
    assert bytes(span.input_data) == body
    assert span.transport.request_size == len(body)
    assert bytes(span.output_data) == reply
    assert span.transport.response_size == len(reply)
    assert span.status is StatusCode.OK
    assert span.capture_integrity.truncated is False
    assert Limitation.FRAME_PARSE_FAILED not in span.capture_integrity.limitations
