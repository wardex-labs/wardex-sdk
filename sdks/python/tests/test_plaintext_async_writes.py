"""Plaintext requests written outside `send`, on a real event loop.

From Python 3.12 asyncio's plaintext writer sends only the first attempt of a
`write()` through `socket.send`; whatever the kernel did not take, and every
`writelines()`, goes out through `socket.sendmsg`. A seam that watched `send`
alone saw the head of a large request and lost its tail, and the request
parser then took the NEXT request's bytes as that tail: one span carried two
requests and the later one's response, and the calls around it disappeared.

These tests send real bytes over loopback through the clients that take that
path, and assert what a user reads: one span per request, each with its own
request body and its own response. On Python 3.10/3.11 the same tests run
over `send` and must pass unchanged.

The next group covers writes the seam still cannot see (`os.sendfile`,
`os.write` on the descriptor): a response to such a request is counted and
never paired with another request. The last covers a reply that comes before
its request ends, whose rest the seam sees late.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import re
import socket
import tempfile
import threading
import time

import aiohttp
import httpx
import pytest

import wardex_sdk as wardex
from wardex_sdk._assembly import counters
from wardex_sdk._interceptors._trackers import _Http1Tracker
from wardex_sdk.testing import RecordingTransport

_PATH = "/v1/chat/completions"
_TAG = re.compile(rb"\[(tag\d+)\]")
_REPLY_ID = re.compile(rb'"id": "chatcmpl-(tag\d+|none)"')
_REQUEST_LINE = re.compile(rb"(?:GET|POST|PUT|DELETE|HEAD|PATCH|OPTIONS) /\S* HTTP/1\.[01]\r\n")
# Small kernel buffers on both ends, so the first `send()` of a large `write()`
# takes only part of it on any OS and the rest goes through asyncio's buffered
# path (`sendmsg` from 3.12). Linux doubles the value it is given; both stay far
# below one request.
_SMALL_BUFFER = 32 * 1024
_LARGE = 1024 * 1024 + 4096
_UNFINISHED = "protocol.http1.request_unfinished"
_UNOBSERVED = "protocol.http1.request_unobserved"


def _body(tag: str, size: int) -> bytes:
    content = f"[{tag}] " + "x" * size
    return json.dumps(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": content}]}
    ).encode()


def _head(n: int, path: str = _PATH) -> bytes:
    return (
        f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
        f"Content-Type: application/json\r\nContent-Length: {n}\r\n\r\n"
    ).encode()


def _reply(tag: str) -> bytes:
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
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        }
    ).encode()


class _TagEcho(http.server.BaseHTTPRequestHandler):
    """Answers each request with a chat completion whose id names the request's tag."""

    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        body = b""
        while len(body) < n:
            chunk = self.rfile.read(min(65536, n - len(body)))
            if not chunk:
                break
            body += chunk
        found = _TAG.search(body)
        reply = _reply(found.group(1).decode() if found else "none")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *a: object) -> None:
        pass


@pytest.fixture
def tag_server():
    class _Server(http.server.ThreadingHTTPServer):
        daemon_threads = True

        def server_bind(self) -> None:
            # On the LISTENING socket, so every accepted socket inherits it.
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SMALL_BUFFER)
            super().server_bind()

    httpd = _Server(("127.0.0.1", 0), _TagEcho)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _http_spans(transport: RecordingTransport) -> list:
    wardex.flush()
    return [s for e in transport.envelopes for s in e.spans if s.name.startswith("HTTP ")]


def _assert_each_span_is_its_own_exchange(spans: list, bodies: dict[str, bytes]) -> None:
    """One span per request; each carries exactly its own request and its own reply."""
    assert len(spans) == len(bodies), [s.name for s in spans]
    seen = []
    for span in spans:
        sent = bytes(span.input_data or b"")
        got = bytes(span.output_data or b"")
        tags = [t.decode() for t in _TAG.findall(sent)]
        assert len(tags) == 1, f"span {span.name!r} carries the bodies of {tags}"
        (tag,) = tags
        assert not _REQUEST_LINE.search(sent), f"span for {tag} has a request line in its body"
        assert [r.decode() for r in _REPLY_ID.findall(got)] == [tag], (
            f"the request of {tag} is paired with the reply to {_REPLY_ID.findall(got)}"
        )
        assert sent == bodies[tag]
        assert span.transport.request_size == len(bodies[tag])
        seen.append(tag)
    assert sorted(seen) == sorted(bodies)


# --- the clients asyncio writes for -----------------------------------------


def test_large_async_httpx_requests_each_get_a_span_paired_with_their_own_response(tag_server):
    """`httpx.AsyncClient` to a plaintext model server, four requests over 1 MiB.

    The shape behind an `AsyncOpenAI(base_url="http://...")` pointed at a local
    or in-house model server, with a prompt that grows every turn.
    """
    bodies = {f"tag{i}": _body(f"tag{i}", _LARGE) for i in range(4)}

    async def run() -> None:
        opts = [(socket.SOL_SOCKET, socket.SO_SNDBUF, _SMALL_BUFFER)]
        transport = httpx.AsyncHTTPTransport(socket_options=opts)
        url = f"http://127.0.0.1:{tag_server}{_PATH}"
        async with httpx.AsyncClient(transport=transport, trust_env=False, timeout=30) as client:
            for body in bodies.values():
                reply = await client.post(url, content=body)
                assert reply.status_code == 200

    recorder = RecordingTransport()
    wardex.init(transport=recorder, intercept=True)
    try:
        asyncio.run(run())
        _assert_each_span_is_its_own_exchange(_http_spans(recorder), bodies)
    finally:
        wardex.close()


def test_aiohttp_requests_written_with_writelines_each_get_their_own_span(tag_server):
    """aiohttp hands a request of 2 KiB or more to `transport.writelines`, whose
    first attempt on 3.12+ is already `sendmsg`: the seam's first sight of the
    connection used to be the RESPONSE, which latched it off as not-a-request,
    so no call on it was ever captured.
    """
    bodies = {f"tag{i}": _body(f"tag{i}", 64 * 1024) for i in range(4)}

    async def run() -> None:
        url = f"http://127.0.0.1:{tag_server}{_PATH}"
        async with aiohttp.ClientSession() as session:
            for body in bodies.values():
                headers = {"Content-Type": "application/json"}
                async with session.post(url, data=body, headers=headers) as reply:
                    assert reply.status == 200
                    await reply.read()

    recorder = RecordingTransport()
    wardex.init(transport=recorder, intercept=True)
    try:
        asyncio.run(run())
        _assert_each_span_is_its_own_exchange(_http_spans(recorder), bodies)
    finally:
        wardex.close()


def _read_reply(sock: socket.socket) -> bytes:
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += sock.recv(65536)
    head, _, rest = buf.partition(b"\r\n\r\n")
    length = int(re.search(rb"(?i)content-length:\s*(\d+)", head).group(1))
    while len(rest) < length:
        rest += sock.recv(65536)
    return rest


# `sendmsg` exists wherever the seam patches it (not on Windows). `sendto` with an
# address on a connected TCP socket is accepted on Linux and macOS, which is where
# the suite runs; an OS that refused it would fail here rather than skip.
@pytest.mark.parametrize("call", [c for c in ("sendmsg", "sendto") if hasattr(socket.socket, c)])
def test_a_request_written_with_sendmsg_or_sendto_is_captured(tag_server, call):
    """The two other `socket.socket` writers, called directly on a connected TCP socket."""
    body = _body("tag0", 4096)
    recorder = RecordingTransport()
    wardex.init(transport=recorder, intercept=True)
    try:
        with socket.create_connection(("127.0.0.1", tag_server)) as sock:
            if call == "sendmsg":
                sent = sock.sendmsg([_head(len(body)), memoryview(body)])
            else:
                sent = sock.sendto(_head(len(body)) + body, ("127.0.0.1", tag_server))
            assert sent == len(_head(len(body))) + len(body), "precondition: one call wrote it all"
            assert b"chatcmpl-tag0" in _read_reply(sock)
        _assert_each_span_is_its_own_exchange(_http_spans(recorder), {"tag0": body})
    finally:
        wardex.close()


# --- writes the seam does not see --------------------------------------------


@pytest.mark.usefixtures("fresh_counters")
def test_a_reply_to_a_request_the_seam_did_not_see_whole_is_never_paired_with_another(tag_server):
    """One keep-alive connection on a real loop, five requests, two of them
    written where no `socket.socket` method carries them:

      tag0  whole, through `send`
      tag1  head through `send`, body through `os.sendfile` (`loop.sock_sendfile`)
      tag2  whole, through `send` — larger than tag1's unseen body, so it used to fill it
      tag3  whole, through `os.write` on the descriptor
      tag4  whole, through `send`

    tag0, tag2 and tag4 keep their own spans; tag1 and tag3 are counted, and no
    span carries one request with another's reply.
    """
    bodies = {
        "tag0": _body("tag0", 4096),
        "tag1": _body("tag1", 2000),
        "tag2": _body("tag2", 64 * 1024),
        "tag3": _body("tag3", 1000),
        "tag4": _body("tag4", 4096),
    }

    async def reply(loop, sock) -> bytes:
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += await loop.sock_recv(sock, 65536)
        head, _, rest = buf.partition(b"\r\n\r\n")
        length = int(re.search(rb"(?i)content-length:\s*(\d+)", head).group(1))
        while len(rest) < length:
            rest += await loop.sock_recv(sock, 65536)
        return rest

    async def run() -> list[bytes]:
        loop = asyncio.get_running_loop()
        replies = []
        with socket.socket() as sock:
            sock.setblocking(False)
            await loop.sock_connect(sock, ("127.0.0.1", tag_server))
            await loop.sock_sendall(sock, _head(len(bodies["tag0"])) + bodies["tag0"])
            replies.append(await reply(loop, sock))
            with tempfile.TemporaryFile() as f:
                f.write(bodies["tag1"])
                f.flush()
                await loop.sock_sendall(sock, _head(len(bodies["tag1"])))
                # `fallback=False`: the native `os.sendfile` path (Linux, macOS) or an
                # error — never a quiet fallback to `send`, which the seam would see.
                await loop.sock_sendfile(sock, f, 0, len(bodies["tag1"]), fallback=False)
            replies.append(await reply(loop, sock))
            await loop.sock_sendall(sock, _head(len(bodies["tag2"])) + bodies["tag2"])
            replies.append(await reply(loop, sock))
            os.write(sock.fileno(), _head(len(bodies["tag3"])) + bodies["tag3"])
            replies.append(await reply(loop, sock))
            await loop.sock_sendall(sock, _head(len(bodies["tag4"])) + bodies["tag4"])
            replies.append(await reply(loop, sock))
        return replies

    recorder = RecordingTransport()
    wardex.init(transport=recorder, intercept=True, capture_mode=wardex.CaptureMode.ALL)
    try:
        replies = asyncio.run(run())
        assert [_REPLY_ID.findall(r) for r in replies] == [[f"tag{i}".encode()] for i in range(5)]
        spans = _http_spans(recorder)
        paired = [s for s in spans if _TAG.search(bytes(s.input_data or b""))]
        _assert_each_span_is_its_own_exchange(
            paired, {t: bodies[t] for t in ("tag0", "tag2", "tag4")}
        )
        # tag1's reply ships as it always did for a request whose end the seam
        # never saw — no request half, no request size — and tag3's not at all.
        unpaired = [s for s in spans if s not in paired]
        assert [_REPLY_ID.findall(bytes(s.output_data or b"")) for s in unpaired] == [[b"tag1"]]
        assert unpaired[0].transport.request_size is None
        assert counters.get(_UNFINISHED) == 1
        assert counters.get(_UNOBSERVED) == 1
    finally:
        wardex.close()


@pytest.mark.usefixtures("fresh_counters")
def test_the_next_request_after_an_unseen_tail_starts_a_fresh_request():
    """The mechanism under the test above, byte for byte: the reply to a
    request whose body the tracker never saw leaves that request unfinished,
    and the next request line must not be read as its body."""
    tracker = _Http1Tracker()
    first, second = _body("tag1", 2000), _body("tag2", 64 * 1024)
    tracker.on_request_bytes(_head(len(first), "/first"))  # the body went by unseen
    (unpaired,) = tracker.on_response_bytes(_response(_reply("tag1")))
    assert (unpaired.method, unpaired.request_body, unpaired.request_counted) == ("?", b"", False)

    tracker.on_request_bytes(_head(len(second), "/second") + second)
    (txn,) = tracker.on_response_bytes(_response(_reply("tag2")))

    assert (txn.method, txn.path, txn.request_body) == ("POST", "/second", second)
    assert b"chatcmpl-tag2" in txn.response_body
    assert tracker.disabled_reason() is None
    assert counters.get(_UNFINISHED) == 1


@pytest.mark.usefixtures("fresh_counters")
def test_a_request_smaller_than_the_unseen_rest_before_it_still_gets_its_own_reply():
    """The same, with a next request that fits inside the rest that went unseen:
    read as that rest, it leaves the unfinished request still unfinished, and
    nothing in the bytes contradicts it. The reply settles it: a reply means a
    request was sent, and only the other reading holds one."""
    tracker = _Http1Tracker()
    first, second = _body("tag1", 64 * 1024), _body("tag2", 512)
    tracker.on_request_bytes(_head(len(first), "/first"))
    tracker.on_response_bytes(_response(_reply("tag1")))
    tracker.on_request_bytes(_head(len(second), "/second") + second)
    (txn,) = tracker.on_response_bytes(_response(_reply("tag2")))

    assert (txn.method, txn.path, txn.request_body) == ("POST", "/second", second)
    assert txn.request_counted
    assert tracker.disabled_reason() is None
    assert counters.get(_UNFINISHED) == 1
    assert counters.get(_UNOBSERVED) == 0


@pytest.mark.usefixtures("fresh_counters")
def test_a_next_request_whose_first_write_was_cut_inside_its_method_is_still_its_own():
    """A nearly full socket buffer can take only `PO` of the next request. The
    next request begins at the first chunk after the reply, cut or not."""
    tracker = _Http1Tracker()
    first, second = _body("tag1", 64 * 1024), _body("tag2", 512)
    tracker.on_request_bytes(_head(len(first), "/first"))
    tracker.on_response_bytes(_response(_reply("tag1")))
    request = _head(len(second), "/second") + second
    tracker.on_request_bytes(request[:2])
    tracker.on_request_bytes(request[2:])
    (txn,) = tracker.on_response_bytes(_response(_reply("tag2")))

    assert (txn.method, txn.path, txn.request_body) == ("POST", "/second", second)


@pytest.mark.usefixtures("fresh_counters")
def test_one_unseen_byte_does_not_turn_the_next_request_into_another_method():
    """With one byte of the rest unseen, reading on takes the next request's
    `P` as that byte and parses `OST /second HTTP/1.1` as a whole request. No
    client sends `OST`: the reply goes to the reading that began at `POST`."""
    tracker = _Http1Tracker()
    first, second = _body("tag1", 64), _body("tag2", 512)
    tracker.on_request_bytes(_head(len(first), "/first") + first[:-1])
    tracker.on_response_bytes(_response(_reply("tag1")))
    tracker.on_request_bytes(_head(len(second), "/second") + second)
    (txn,) = tracker.on_response_bytes(_response(_reply("tag2")))

    assert (txn.method, txn.path, txn.request_body) == ("POST", "/second", second)
    assert tracker.disabled_reason() is None


@pytest.mark.usefixtures("fresh_counters")
def test_a_reply_with_no_request_bytes_since_the_last_reply_is_counted_not_shipped():
    tracker = _Http1Tracker()
    body = _body("tag0", 64)
    tracker.on_request_bytes(_head(len(body)) + body)
    (first,) = tracker.on_response_bytes(_response(_reply("tag0")))
    assert first.method == "POST"
    # The next request was written entirely where the seam does not look.
    assert tracker.on_response_bytes(_response(_reply("tag1"))) == []
    assert counters.get(_UNOBSERVED) == 1


# --- a reply that comes before its request ends --------------------------------
#
# A server may answer an upload early (a 413, a 401) and still read the rest of
# it, so the connection stays usable. The rest then arrives AFTER the reply, and
# it is the unfinished request's own: neither its content nor the write it rides
# in may decide where the next request starts.

_EARLY = b"HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\n\r\n"


@pytest.mark.usefixtures("fresh_counters")
def test_an_upload_answered_early_keeps_its_late_tail_out_of_the_next_request():
    """That tail is consumed, the request it completes is dropped (its reply
    already shipped), and the next request is latched at its own start rather
    than at the tail's."""
    tracker = _Http1Tracker()
    tracker.on_request_bytes(_head(100, "/upload") + b"x" * 10)
    (early,) = tracker.on_response_bytes(_EARLY)
    assert (early.status, early.request_counted) == (413, False)
    tracker.on_request_bytes(b"x" * 90)  # the rest of the upload, after the reply
    time.sleep(0.01)
    issued = time.time_ns()
    second = _body("tag2", 512)
    tracker.on_request_bytes(_head(len(second), "/second") + second)
    (txn,) = tracker.on_response_bytes(_response(_reply("tag2")))

    assert (txn.method, txn.path, txn.request_body) == ("POST", "/second", second)
    assert txn.start_ns >= issued, "the next request was timed from the previous upload's tail"
    assert tracker.disabled_reason() is None
    assert counters.get(_UNFINISHED) == 1


@pytest.mark.usefixtures("fresh_counters")
def test_a_late_tail_and_the_next_request_in_one_write_keep_the_next_requests_parent_and_start():
    """asyncio queues a `write()` behind bytes still in its buffer and sends
    them together, so the late tail and the whole next request reach the seam
    as ONE chunk that opens with body bytes. The next request was still issued
    where its first byte was seen: under the span that was live, and then."""
    recorder = RecordingTransport()
    wardex.init(transport=recorder)
    try:
        tracker = _Http1Tracker()
        tracker.on_request_bytes(_head(100, "/upload") + b"x" * 10)
        tracker.on_response_bytes(_EARLY)
        second = _body("tag2", 512)
        with wardex.span("agent-turn") as turn:
            issued = time.time_ns()
            tracker.on_request_bytes(b"x" * 90 + _head(len(second), "/second") + second)
            time.sleep(0.01)
            (txn,) = tracker.on_response_bytes(_response(_reply("tag2")))
    finally:
        wardex.close()

    assert (txn.method, txn.path, txn.request_body) == ("POST", "/second", second)
    assert txn.parent is not None, "the next request lost the span it was issued in"
    assert txn.parent.span_id == turn.context.span_id
    assert issued <= txn.start_ns < txn.end_ns, "the next request was never timed"
    assert txn.ttfb_ms is not None
    assert counters.get(_UNFINISHED) == 1


@pytest.mark.parametrize(
    "opening",
    [
        pytest.param(b"GET /index.html was fetched\n", id="a-method-word"),
        pytest.param(b"GET /index.html HTTP/1.1\r\nHost: example\r\n\r\n", id="a-whole-request"),
    ],
)
@pytest.mark.usefixtures("fresh_counters")
def test_a_late_tail_that_opens_like_a_request_is_still_the_uploads_own(opening):
    """An upload of logs or of a captured HTTP exchange can hold a request
    line anywhere, and a write boundary can fall right before one. Read as a
    new request it would stop the parser, and every later call on the keep-alive
    connection would ship as `HTTP ?` paired with nobody's request."""
    tracker = _Http1Tracker()
    upload = b"lines of a log upload\n" + opening + b"z" * 200
    seen = len(b"lines of a log upload\n")
    tracker.on_request_bytes(_head(len(upload), "/upload") + upload[:seen])
    tracker.on_response_bytes(_EARLY)
    tracker.on_request_bytes(upload[seen:])  # the late tail opens with `GET `
    assert tracker.disabled_reason() is None

    for tag in ("tag2", "tag3"):
        body = _body(tag, 512)
        tracker.on_request_bytes(_head(len(body)) + body)
        (txn,) = tracker.on_response_bytes(_response(_reply(tag)))
        assert (txn.method, txn.path, txn.request_body) == ("POST", _PATH, body)
        assert _REPLY_ID.findall(txn.response_body) == [tag.encode()]
    assert counters.get(_UNFINISHED) == 1


@pytest.mark.usefixtures("fresh_counters")
def test_a_late_tail_holding_a_whole_request_does_not_take_the_next_reply():
    """The late tail is exactly a whole request (`HEAD / HTTP/1.0`, from a
    captured exchange), and the next request rides the same write and is
    answered early too. Read as starting at the tail, the stream holds a
    complete `HEAD` and then that next request; the reply answers the LAST
    request begun, which both readings agree is the next request, so it does
    not ship as `HEAD /`."""
    tracker = _Http1Tracker()
    upload = b"captured: " + b"HEAD / HTTP/1.0\r\n\r\n"
    seen = len(b"captured: ")
    tracker.on_request_bytes(_head(len(upload), "/upload") + upload[:seen])
    tracker.on_response_bytes(_EARLY)
    second = _body("tag2", 4096)
    request = _head(len(second)) + second
    tracker.on_request_bytes(upload[seen:] + request[:200])
    (early,) = tracker.on_response_bytes(_EARLY)
    assert (early.method, early.status, early.request_counted) == ("?", 413, False)

    tracker.on_request_bytes(request[200:])  # its rest, read after the reply
    third = _body("tag3", 512)
    tracker.on_request_bytes(_head(len(third)) + third)
    (txn,) = tracker.on_response_bytes(_response(_reply("tag3")))
    assert (txn.method, txn.path, txn.request_body) == ("POST", _PATH, third)
    assert counters.get(_UNFINISHED) == 2


def _answer_the_upload_early(srv: socket.socket) -> None:
    """Answers a large upload with a 413 before reading its body, reads the
    body anyway so the connection stays usable, then answers the next request
    with a completion named by its tag."""
    conn, _ = srv.accept()
    with conn:
        conn.settimeout(30)

        def request(buf: bytes) -> tuple[bytes, bytes]:
            while b"\r\n\r\n" not in buf:
                buf += conn.recv(65536)
            head, _, buf = buf.partition(b"\r\n\r\n")
            length = int(re.search(rb"Content-Length: (\d+)", head).group(1))
            while len(buf) < length:
                buf += conn.recv(65536)
            return buf[:length], buf[length:]

        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += conn.recv(65536)
        conn.sendall(_EARLY)
        _, buf = request(buf)
        body, _ = request(buf)
        found = _TAG.search(body)
        conn.sendall(_response(_reply(found.group(1).decode() if found else "none")))


@pytest.mark.usefixtures("fresh_counters")
def test_the_call_queued_behind_an_early_answered_upload_keeps_its_parent_and_duration(
    monkeypatch,
):
    """The same on a real event loop, through the asyncio transport every
    asyncio HTTP client writes with: a 4 MiB upload answered early, and the
    next call written while the upload's tail is still in the transport's
    buffer. Its span is a child of the span it was issued in, it has a
    duration, and it carries its own request and its own reply."""
    feeds: list[bytes] = []
    feed = _Http1Tracker.on_request_bytes

    def recording(self: _Http1Tracker, data: bytes) -> list:
        feeds.append(bytes(data))
        return feed(self, data)

    monkeypatch.setattr(_Http1Tracker, "on_request_bytes", recording)
    second = _body("tag2", 512)

    class _Client(asyncio.Protocol):
        def __init__(self) -> None:
            loop = asyncio.get_running_loop()
            self.received, self.early, self.done = b"", loop.create_future(), loop.create_future()

        def data_received(self, data: bytes) -> None:
            self.received += data
            if b" 413 " in self.received and not self.early.done():
                self.early.set_result(None)
            if b"chatcmpl-tag2" in self.received and not self.done.done():
                self.done.set_result(None)

    async def run(port: int) -> int:
        loop = asyncio.get_running_loop()
        transport, client = await loop.create_connection(_Client, "127.0.0.1", port)
        sock = transport.get_extra_info("socket")
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SMALL_BUFFER)
        upload = b"x" * (4 * 1024 * 1024)
        transport.write(_head(len(upload)) + upload)
        await asyncio.wait_for(client.early, 30)
        pending = transport.get_write_buffer_size()
        transport.write(_head(len(second)) + second)
        await asyncio.wait_for(client.done, 30)
        transport.close()
        return pending

    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _SMALL_BUFFER)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        server = threading.Thread(target=_answer_the_upload_early, args=(srv,), daemon=True)
        server.start()
        recorder = RecordingTransport()
        wardex.init(transport=recorder, intercept=True)
        try:
            with wardex.span("agent-turn") as turn:
                pending = asyncio.run(run(srv.getsockname()[1]))
            spans = _http_spans(recorder)
        finally:
            wardex.close()
            server.join(30)

    assert pending > 0, "precondition: the next call was queued behind the upload's tail"
    assert any(f.find(b"POST " + _PATH.encode()) > 0 for f in feeds), (
        "precondition: the tail and the next call reached the seam in one write"
    )
    (call,) = [s for s in spans if _TAG.search(bytes(s.input_data or b""))]
    assert bytes(call.input_data) == second
    assert _REPLY_ID.findall(bytes(call.output_data or b"")) == [b"tag2"]
    assert call.parent_span_id == turn.context.span_id, "the call lost the span it was issued in"
    assert call.end_time_ns > call.start_time_ns, "the call was never timed"
    assert counters.get(_UNFINISHED) == 1


def _response(payload: bytes) -> bytes:
    return (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
        b"Content-Length: " + str(len(payload)).encode() + b"\r\n\r\n" + payload
    )
