"""A conversation reaches the spans the byte seam builds.

`wardex.conversation(...)` promises that every span inside the block carries
`gen_ai.conversation.id`. Every span did except the ones that matter most to a
"what did this conversation cost" query: the LLM calls read off the wire, which
carry the tokens. The seam latched the parent span at request time and threw
the ambient conversation away, so a backend grouping by conversation id summed
zero tokens for every conversation.

The rules these tests hold:

* A wire span carries the conversation its request was ISSUED in — latched at
  request time beside the parent, never read on the response side, where the
  ambient scope may already be in the next conversation.
* A span outside every conversation carries none, and concurrent conversations
  — `asyncio.gather`, a thread pool — never lend each other their ids.
* A WebSocket session is ONE span for every call on the socket, so it names a
  conversation only when its handshake and every message were issued in it.
* A Responses request may name its own conversation in its body. With nothing
  ambient that id is the conversation; inside a conversation the ambient one
  wins (the host's word, as over a framework's `group_id`), and the body's id
  rides along under its own attribute, counted. Where wardex could not read the
  conversation the request was issued in (an unproven HTTP/2 stream, a latch
  refused at the edge), the body's id does not stand in for it: it rides along
  the same way, counted apart.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import http.client
import http.server
import json
import pathlib
import threading
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from hpack import Encoder

import wardex_sdk as wardex
from test_close_hook import _h2_answer, _h2_open
from wardex_sdk import _hub, _wardex_native
from wardex_sdk._assembly import (
    EMPTY_AMBIENT,
    Limitation,
    counters,
    resolve_observed,
    resolve_parentage,
)
from wardex_sdk._enums import CaptureMode, SpanKind
from wardex_sdk._interceptors._seam import _latched
from wardex_sdk._interceptors._trackers import _Txn, _WebSocketTracker
from wardex_sdk._semantics import REQUEST_CONVERSATION_KEY
from wardex_sdk._types import ConversationContext
from wardex_sdk.testing import RecordingTransport

pytestmark = pytest.mark.usefixtures("fresh_counters")

_README = pathlib.Path(__file__).resolve().parents[3] / "README.md"
_PROMISE = "# A conversation: every span inside carries gen_ai.conversation.id."


def _response(n: int) -> bytes:
    return json.dumps(
        {
            "id": f"resp_{n}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "id": f"msg_{n}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10 * n, "output_tokens": n, "total_tokens": 11 * n},
        }
    ).encode()


@pytest.fixture
def llm() -> Iterator[tuple[str, int]]:
    """A loopback Responses API: keep-alive HTTP/1.1, a usage per call."""
    lock = threading.Lock()
    seen = {"n": 0}

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with lock:
                seen["n"] += 1
                body = _response(seen["n"])
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: object) -> None:
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    host, port = httpd.socket.getsockname()[:2]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield host, port
    finally:
        httpd.shutdown()
        httpd.server_close()


def _body(tag: str, **extra: Any) -> bytes:
    return json.dumps({"model": "gpt-4o-mini", "input": tag, **extra}).encode()


def _post(host: str, port: int, tag: str, **extra: Any) -> None:
    conn = http.client.HTTPConnection(host, port)
    conn.request("POST", "/v1/responses", _body(tag, **extra), {"Content-Type": "application/json"})
    conn.getresponse().read()
    conn.close()


def _shipped(fn) -> tuple[list[Any], RecordingTransport]:
    t = RecordingTransport()
    wardex.init(transport=t)
    try:
        fn()
        _hub.get_client()._settle()
        wardex.flush()
    finally:
        wardex.close()
    return [sp for env in t.envelopes for sp in env.spans], t


def _conv(span: Any) -> str | None:
    return span.conversation.conversation_id if span.conversation is not None else None


def _tag(span: Any) -> str:
    """The `input` the request was sent with: which call this span is."""
    return json.loads(span.input_data)["input"]


def _chats(spans: list[Any]) -> list[Any]:
    return [s for s in spans if s.kind is SpanKind.CLIENT and s.gen_ai is not None]


def _otlp(t: RecordingTransport) -> list[tuple[str, dict[str, Any]]]:
    """`(name, attributes)` per span as a receiver decodes the OTLP export."""
    out = []
    for env in t.envelopes:
        decoded = _wardex_native.codec.decode_otlp_traces(
            _wardex_native.codec.encode_otlp_traces(env)
        )
        for rs in decoded["resource_spans"]:
            for ss in rs["scope_spans"]:
                out += [(sp["name"], sp["attributes"]) for sp in ss["spans"]]
    return out


def _want(tag: str) -> str | None:
    """'A-3' was issued in conversation 'A'; 'outside-*' / 'after-*' in none."""
    head = tag.split("-")[0]
    return None if head in ("outside", "after") else head


# --------------------------------------------------------------------------
# the README's promise
# --------------------------------------------------------------------------


def test_the_readme_promise_every_span_inside_a_conversation_carries_its_id(llm):
    """README: "every span inside carries gen_ai.conversation.id". Every kind
    of span a block can hold — the conversation's own, a decorated workflow and
    tool, a hand-named span, the wire `chat` calls and a plain HTTP call — and
    the receiver's view of it, the OTLP attribute. A call after the block
    carries none."""
    assert _PROMISE in _README.read_text(), "the README sentence this test asserts moved"
    host, port = llm

    @wardex.tool(name="lookup")
    def lookup() -> str:
        conn = http.client.HTTPConnection(host, port)
        conn.request("GET", "/health")
        conn.getresponse().read()
        conn.close()
        return "found"

    @wardex.workflow(name="turn")
    def turn() -> None:
        _post(host, port, "in-1")
        lookup()
        with wardex.span("rank"):
            _post(host, port, "in-2")

    def run() -> None:
        with wardex.conversation("support-chat", id="session-7"):
            turn()
        _post(host, port, "after-block")

    spans, t = _shipped(run)
    inside = [s for s in spans if not (s.kind is SpanKind.CLIENT and _tag_or(s) == "after-block")]
    names = sorted(s.name for s in inside)
    assert names == sorted(
        [
            "support-chat",
            "turn",
            "lookup",
            "rank",
            "HTTP POST /v1/responses",
            "HTTP POST /v1/responses",
            "HTTP GET /health",
        ]
    ), names
    for s in inside:
        assert _conv(s) == "session-7", s.name
    (after,) = [s for s in _chats(spans) if _tag(s) == "after-block"]
    assert after.conversation is None
    # What a receiver filters on: the attribute, on every exported span but
    # the one call made after the block.
    exported = _otlp(t)
    assert len(exported) == len(spans)
    carried = [attrs["gen_ai.conversation.id"] for _, attrs in exported if _carries(attrs)]
    assert carried == ["session-7"] * len(inside)
    assert [name for name, attrs in exported if not _carries(attrs)] == ["chat gpt-4o-mini"]


def _carries(attrs: dict[str, Any]) -> bool:
    return "gen_ai.conversation.id" in attrs


def _tag_or(span: Any) -> str | None:
    try:
        return _tag(span)
    except (ValueError, KeyError, TypeError):
        return None


def test_the_tokens_of_a_conversation_sum_from_its_spans(llm):
    """The query the promise exists for: group what shipped by conversation id
    and sum the tokens. It was zero for every conversation."""
    host, port = llm

    def run() -> None:
        with wardex.conversation("c", id="conv-a"):
            _post(host, port, "a-1")
            _post(host, port, "a-2")
        with wardex.conversation("c", id="conv-b"):
            _post(host, port, "b-1")

    spans, _ = _shipped(run)
    totals: dict[str | None, int] = {}
    for s in _chats(spans):
        totals[_conv(s)] = totals.get(_conv(s), 0) + s.gen_ai.input_tokens
    assert totals == {"conv-a": 10 + 20, "conv-b": 30}


# --------------------------------------------------------------------------
# no conversation lends its id to another
# --------------------------------------------------------------------------


def _assert_each_carries_its_own(spans: list[Any], *, inside: int, outside: int) -> None:
    chats = _chats(spans)
    got = {_tag(s): _conv(s) for s in chats}
    assert {t: _want(t) for t in got} == got
    assert sum(_want(t) is not None for t in got) == inside
    assert sum(_want(t) is None for t in got) == outside
    for probe in [s for s in spans if s.name == "probe"]:
        assert probe.conversation is None


def test_concurrent_conversations_under_gather_keep_their_own_ids(llm):
    """Two conversations and a task in none, interleaved on one loop under one
    parent span — the shape that shares a scope object across the tasks. A
    sibling outside every conversation used to read whichever block had last
    written the shared scope, and the last block to close restored the OTHER
    block's id onto it, so a call after `gather` carried a conversation too."""
    host, port = llm
    base = f"http://{host}:{port}"

    async def conversation(client: httpx.AsyncClient, cid: str) -> None:
        with wardex.conversation("c", id=cid):
            for i in range(4):
                await client.post(f"{base}/v1/responses", content=_body(f"{cid}-{i}"))
                await asyncio.sleep(0)

    async def outside(client: httpx.AsyncClient) -> None:
        for i in range(4):
            with wardex.span("probe"):
                pass
            await client.post(f"{base}/v1/responses", content=_body(f"outside-{i}"))
            await asyncio.sleep(0)

    async def main() -> None:
        async with httpx.AsyncClient() as client:
            with wardex.span("parent"):
                await asyncio.gather(
                    conversation(client, "A"), conversation(client, "B"), outside(client)
                )
                await client.post(f"{base}/v1/responses", content=_body("after-gather"))
                with wardex.span("probe"):
                    pass

    spans, _ = _shipped(lambda: asyncio.run(main()))
    _assert_each_carries_its_own(spans, inside=8, outside=5)


def test_concurrent_conversations_on_a_thread_pool_keep_their_own_ids(llm):
    """The same on threads carried by `bind_context`, which share the
    submitting thread's scope object; plus a pool task submitted from INSIDE a
    conversation, which belongs to it."""
    host, port = llm

    def conversation(cid: str) -> None:
        with wardex.conversation("c", id=cid):
            for i in range(4):
                _post(host, port, f"{cid}-{i}")

    def outside() -> None:
        for i in range(4):
            with wardex.span("probe"):
                pass
            _post(host, port, f"outside-{i}")

    def run() -> None:
        with wardex.span("parent"):
            with concurrent.futures.ThreadPoolExecutor(3) as pool:
                futures = [
                    pool.submit(wardex.bind_context(conversation), "TA"),
                    pool.submit(wardex.bind_context(conversation), "TB"),
                    pool.submit(wardex.bind_context(outside)),
                ]
                for f in futures:
                    f.result()
                with wardex.conversation("c", id="TC"):
                    pool.submit(wardex.bind_context(_post), host, port, "TC-pool").result()
            _post(host, port, "after-pool")

    spans, _ = _shipped(run)
    _assert_each_carries_its_own(spans, inside=9, outside=5)


def test_a_reused_connection_latches_each_request_where_it_was_issued(llm):
    """One keep-alive connection, three requests: in conversation A, in B, in
    none. And the response half read somewhere else entirely — the id is the
    one the request went out under, never the one ambient when the bytes came
    back."""
    host, port = llm

    def run() -> None:
        conn = http.client.HTTPConnection(host, port)
        with wardex.span("parent"):
            with wardex.conversation("c", id="A"):
                conn.request("POST", "/v1/responses", _body("A-0"))
            with wardex.conversation("c", id="B"):
                conn.getresponse().read()  # A's response, read inside B
                conn.request("POST", "/v1/responses", _body("B-0"))
            conn.getresponse().read()  # B's response, read outside both
            conn.request("POST", "/v1/responses", _body("outside-0"))
            with wardex.conversation("c", id="C"):
                conn.getresponse().read()  # no conversation's response, read inside C
        conn.close()

    spans, _ = _shipped(run)
    _assert_each_carries_its_own(spans, inside=2, outside=1)


# --------------------------------------------------------------------------
# the request body's own conversation
# --------------------------------------------------------------------------


def test_the_request_body_names_the_conversation_when_none_is_ambient(llm):
    host, port = llm

    def run() -> None:
        _post(host, port, "str", conversation="conv_str")
        _post(host, port, "obj", conversation={"id": "conv_obj"})
        _post(host, port, "empty", conversation="")
        _post(host, port, "absent")

    spans, _ = _shipped(run)
    got = {_tag(s): _conv(s) for s in _chats(spans)}
    assert got == {"str": "conv_str", "obj": "conv_obj", "empty": None, "absent": None}
    for s in _chats(spans):
        assert REQUEST_CONVERSATION_KEY not in dict(s.extra)
    assert counters.get("semantics.request_conversation_shadowed") == 0


def test_the_hosts_conversation_wins_over_the_request_body(llm):
    """HOST WINS, the rule a framework's `group_id` already follows: the span
    keeps the conversation it was issued in, and the body's differing id rides
    along under its own attribute, counted. A body that names the SAME id
    shadows nothing."""
    host, port = llm

    def run() -> None:
        with wardex.conversation("c", id="host-9"):
            _post(host, port, "differs", conversation="conv_body")
            _post(host, port, "same", conversation="host-9")

    spans, t = _shipped(run)
    chats = {_tag(s): s for s in _chats(spans)}
    assert _conv(chats["differs"]) == "host-9"
    assert dict(chats["differs"].extra)[REQUEST_CONVERSATION_KEY] == "conv_body"
    assert _conv(chats["same"]) == "host-9"
    assert REQUEST_CONVERSATION_KEY not in dict(chats["same"].extra)
    assert counters.get("semantics.request_conversation_shadowed") == 1
    exported = [attrs for name, attrs in _otlp(t) if REQUEST_CONVERSATION_KEY in attrs]
    assert [(a["gen_ai.conversation.id"], a[REQUEST_CONVERSATION_KEY]) for a in exported] == [
        ("host-9", "conv_body")
    ]


# --------------------------------------------------------------------------
# every tracker latches it: h2 streams, WebSocket sessions
# --------------------------------------------------------------------------


def test_h2_bytes_name_no_conversation_without_a_proven_opener(
    fake_ssl_socket, bare_ssl_interceptor
):
    """Three streams written in A, in B and in none, with no `h2` state machine
    behind them. On a shared HTTP/2 connection any task may write another's
    frames, so the scope a stream's bytes were written in does not prove who
    issued it, and a conversation is never guessed: none is named."""
    itc = bare_ssl_interceptor
    itc._client.config.capture_mode = CaptureMode.ALL
    sock = fake_ssl_socket(alpn="h2")
    client_enc, server_enc = Encoder(), Encoder()
    for sid, cid in ((1, "A"), (3, "B"), (5, None)):
        with _issued_in(cid):
            itc._on_request_bytes(sock, _h2_open(client_enc, sid))
    itc._on_response_bytes(
        sock, _h2_answer(server_enc, 5) + _h2_answer(server_enc, 3) + _h2_answer(server_enc, 1)
    )
    assert [_conv(s) for s in itc._client.spans] == [None, None, None]


_H2_REQUEST = [(":method", "POST"), (":scheme", "https"), (":authority", "a"), (":path", "/v1/m")]


@pytest.fixture
def h2_reads() -> Iterator[Any]:
    """The `h2` state machine with the shared probe's reads on it, as `install()` puts them."""
    import h2.connection

    from wardex_sdk._interceptors._close_hook import (
        install_shared_close_hook,
        uninstall_shared_close_hook,
    )

    real = h2.connection.H2Connection.send_headers
    install_shared_close_hook()
    try:
        assert h2.connection.H2Connection.send_headers is not real
        yield h2.connection.H2Connection
    finally:
        uninstall_shared_close_hook()
    assert h2.connection.H2Connection.send_headers is real, "the probe left its read behind"


def _copy(chunk: bytes) -> bytes:
    """Equal bytes in a different object: what a client that copies its chunks writes."""
    return bytes(bytearray(chunk))


def _flushed_by_another_task(itc: Any, sock: Any, conn: Any, *, link: bool) -> list[Any]:
    """httpcore's shape without the network: the tasks issuing in A and in B
    queue their streams' frames on one connection, and a THIRD task, holding
    the write lock in conversation W, writes every frame queued. With
    `link=False` every chunk is written as a copy, so none proves the link."""
    write = (lambda chunk: chunk) if link else _copy
    conn.initiate_connection()
    itc._on_request_bytes(sock, write(conn.data_to_send()))
    for sid, cid in ((1, "A"), (3, "B")):
        with _issued_in(cid):
            conn.send_headers(sid, _H2_REQUEST, end_stream=True)
    with _issued_in("W"):
        itc._on_request_bytes(sock, write(conn.data_to_send()))
    server_enc = Encoder()
    itc._on_response_bytes(sock, _h2_answer(server_enc, 3) + _h2_answer(server_enc, 1))
    return itc._client.spans


def test_an_h2_stream_carries_the_conversation_of_the_task_that_opened_it(
    fake_ssl_socket, bare_ssl_interceptor, h2_reads
):
    """The connection preface the client wrote IS the object its state machine
    returned, which links the two; from then on each stream's opener is read in
    `send_headers`, where the opening task is still the caller."""
    itc = bare_ssl_interceptor
    itc._client.config.capture_mode = CaptureMode.ALL
    spans = _flushed_by_another_task(itc, fake_ssl_socket(alpn="h2"), h2_reads(), link=True)
    assert [_conv(s) for s in spans] == ["B", "A"]


def test_an_h2_stream_names_no_conversation_when_its_connection_is_not_proven(
    fake_ssl_socket, bare_ssl_interceptor, h2_reads
):
    """The same streams, but every chunk was written as a COPY: equal bytes are
    not proof that this connection carries that state machine's output, so
    nothing links, and neither the openers' ids nor the writer's W is named."""
    itc = bare_ssl_interceptor
    itc._client.config.capture_mode = CaptureMode.ALL
    spans = _flushed_by_another_task(itc, fake_ssl_socket(alpn="h2"), h2_reads(), link=False)
    assert [_conv(s) for s in spans] == [None, None]


def test_a_stream_opened_before_its_connection_was_proven_is_proven_by_the_chunk_that_links_it(
    fake_ssl_socket, bare_ssl_interceptor, h2_reads
):
    """The preface was written as a copy, so the streams opened in A and B are
    opened on a connection nothing has proven yet. The chunk that carries them
    is written as handed out, which proves the link, and the openers read in
    `send_headers` before it still count: a link is about the connection, not
    about the streams opened after it."""
    itc = bare_ssl_interceptor
    itc._client.config.capture_mode = CaptureMode.ALL
    sock, conn = fake_ssl_socket(alpn="h2"), h2_reads()
    conn.initiate_connection()
    itc._on_request_bytes(sock, _copy(conn.data_to_send()))
    for sid, cid in ((1, "A"), (3, "B")):
        with _issued_in(cid):
            conn.send_headers(sid, _H2_REQUEST, end_stream=True)
    with _issued_in("W"):
        itc._on_request_bytes(sock, conn.data_to_send())
    server_enc = Encoder()
    itc._on_response_bytes(sock, _h2_answer(server_enc, 3) + _h2_answer(server_enc, 1))
    assert [_conv(s) for s in itc._client.spans] == ["B", "A"]


def test_connections_whose_chunks_are_written_out_of_order_each_prove_their_own(
    fake_ssl_socket, bare_ssl_interceptor, h2_reads
):
    """anyio's plaintext `SocketStream.send` yields to the event loop BEFORE it
    writes, so with several connections opening under one `asyncio.gather`
    every client hands out its preface before any is written, and they are
    written in whatever order the loop resumes them. One pending offer per
    thread meant the last preface handed out was the only one that could link:
    the first write found another connection's chunk and dropped it, and the
    connection it should have proven named no conversation for its first calls.
    Every connection is proven by its own chunk, in any order."""
    itc = bare_ssl_interceptor
    itc._client.config.capture_mode = CaptureMode.ALL
    conns = {cid: (fake_ssl_socket(alpn="h2"), h2_reads()) for cid in ("A", "B", "C")}
    prefaces = {}
    for cid, (_sock, conn) in conns.items():
        conn.initiate_connection()
        prefaces[cid] = conn.data_to_send()
    for cid in ("A", "C", "B"):  # neither the order handed out nor its reverse
        itc._on_request_bytes(conns[cid][0], prefaces[cid])
    for cid, (sock, conn) in conns.items():
        with _issued_in(cid):
            conn.send_headers(1, _H2_REQUEST, end_stream=True)
        with _issued_in("W"):
            itc._on_request_bytes(sock, conn.data_to_send())
        itc._on_response_bytes(sock, _h2_answer(Encoder(), 1))
    assert [_conv(s) for s in itc._client.spans] == ["A", "B", "C"]


@contextlib.contextmanager
def _h2_serving(*, tls: bool) -> Iterator[tuple[str, Any]]:
    """A loopback Responses API that speaks only HTTP/2: over TLS, or as h2c
    (plaintext, prior knowledge). A call tagged `hN-...` is billed
    `10 * (N + 1)` input tokens. Yields the base URL and the accepted sockets."""
    import socket
    import ssl

    import h2.config
    import h2.connection
    import h2.events

    from conftest import CERT, KEY

    accepted = []

    def handle(sock: Any) -> None:
        conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
        conn.initiate_connection()
        sock.sendall(conn.data_to_send())
        bodies: dict[int, bytes] = {}
        while data := sock.recv(65535):
            for ev in conn.receive_data(data):
                if isinstance(ev, h2.events.DataReceived):
                    bodies[ev.stream_id] = bodies.get(ev.stream_id, b"") + ev.data
                    conn.acknowledge_received_data(ev.flow_controlled_length, ev.stream_id)
                elif isinstance(ev, h2.events.StreamEnded):
                    tag = json.loads(bodies.pop(ev.stream_id))["input"]
                    body = _response(int(tag.split("-")[0][1:]) + 1)
                    conn.send_headers(
                        ev.stream_id,
                        [(":status", "200"), ("content-type", "application/json")],
                    )
                    conn.send_data(ev.stream_id, body, end_stream=True)
            sock.sendall(conn.data_to_send())

    ctx = None
    if tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=str(CERT), keyfile=str(KEY))
        ctx.set_alpn_protocols(["h2"])
    listener = socket.create_server(("127.0.0.1", 0))
    listener.settimeout(0.2)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                raw, _ = listener.accept()
            except (TimeoutError, OSError):
                continue
            accepted.append(raw)
            sock = ctx.wrap_socket(raw, server_side=True) if ctx is not None else raw
            threading.Thread(target=handle, args=(sock,), daemon=True).start()

    threading.Thread(target=serve, daemon=True).start()
    scheme = "https" if tls else "http"
    try:
        yield f"{scheme}://127.0.0.1:{listener.getsockname()[1]}", accepted
    finally:
        stop.set()
        listener.close()


@pytest.fixture
def h2_llm() -> Iterator[tuple[str, Any]]:
    with _h2_serving(tls=True) as served:
        yield served


@pytest.fixture
def h2c_llm() -> Iterator[tuple[str, Any]]:
    with _h2_serving(tls=False) as served:
        yield served


def _assert_one_h2_connection_kept_each_call_to_its_own(spans: list[Any], n: int) -> None:
    chats = _chats(spans)
    assert sorted(_tag(s) for s in chats) == sorted(f"h{i}-0" for i in range(n))
    # The safety property: a call names its own conversation or none, never another's.
    leaked = [(_tag(s), _conv(s)) for s in chats if _conv(s) not in (_want(_tag(s)), None)]
    assert not leaked, f"calls carrying ANOTHER conversation's id (tag, carried): {leaked}"
    # The query: a conversation's token sum is its own calls', none of another's.
    totals: dict[str | None, int] = {}
    for s in chats:
        totals[_conv(s)] = totals.get(_conv(s), 0) + s.gen_ai.input_tokens
    assert totals == {f"h{i}": 10 * (i + 1) for i in range(n)}
    # And the parent: the block the call was issued in, not the one that flushed it.
    blocks = {s.context.span_id: _conv(s) for s in spans if s.name == "c"}
    assert {_tag(s): blocks.get(s.parent_span_id) for s in chats} == {
        f"h{i}-0": f"h{i}" for i in range(n)
    }


@pytest.mark.parametrize("n", [2, 8])
def test_concurrent_conversations_on_one_http2_connection_keep_their_own_ids(h2_llm, n):
    """`asyncio.gather` over ONE HTTP/2 connection — httpx's `http2=True`, as an
    OpenAI or Anthropic client built on it. httpcore queues each call's frames
    from the task that issued it and lets whichever task holds the write lock
    send them all, so the bytes of one conversation's call were routinely
    written by another's task: seven of eight calls carried the wrong id, and
    the per-conversation token sums came out as zero and double."""
    import ssl

    from conftest import CERT

    base, accepted = h2_llm
    verify = ssl.create_default_context(cafile=str(CERT))

    async def one(client: httpx.AsyncClient, i: int) -> None:
        with wardex.conversation("c", id=f"h{i}"):
            r = await client.post(f"{base}/v1/responses", content=_body(f"h{i}-0"))
            assert (r.status_code, r.http_version) == (200, "HTTP/2")

    async def main() -> None:
        async with httpx.AsyncClient(http2=True, verify=verify) as client:
            await asyncio.gather(*(one(client, i) for i in range(n)))

    spans, _ = _shipped(lambda: asyncio.run(main()))
    assert len(accepted) == 1, "the calls did not share one connection"
    _assert_one_h2_connection_kept_each_call_to_its_own(spans, n)


def test_concurrent_conversations_on_one_http2_connection_from_a_thread_pool(h2_llm):
    """The same over a sync `httpx.Client(http2=True)` shared by pool threads,
    whose connection lock hands the write to any of them."""
    import ssl

    from conftest import CERT

    base, accepted = h2_llm
    verify = ssl.create_default_context(cafile=str(CERT))
    n = 8

    def run() -> None:
        with httpx.Client(http2=True, verify=verify) as client:

            def one(i: int) -> None:
                with wardex.conversation("c", id=f"h{i}"):
                    r = client.post(f"{base}/v1/responses", content=_body(f"h{i}-0"))
                    assert (r.status_code, r.http_version) == (200, "HTTP/2")

            with concurrent.futures.ThreadPoolExecutor(n) as pool:
                for f in [pool.submit(wardex.bind_context(one), i) for i in range(n)]:
                    f.result()

    spans, _ = _shipped(run)
    assert len(accepted) == 1, "the calls did not share one connection"
    _assert_one_h2_connection_kept_each_call_to_its_own(spans, n)


@pytest.mark.parametrize("stated", [None, "conv_body"])
def test_http2_connections_opened_together_each_carry_their_own_conversation(h2c_llm, stated):
    """Four h2c clients (`http1=False, http2=True`: plaintext, prior
    knowledge), one connection each, opened together under `asyncio.gather`,
    three calls each inside the client's own conversation. anyio's plaintext
    writer yields before it writes, so every client handed out its preface
    before any was written; only one connection could be proven, the others'
    first calls carried NO conversation, and with a body naming `conv_body` the
    body's id took the call outright, beating the host's.

    Every call must carry its own block's id, and a body that names another
    rides along beside it, as on every other path."""
    base, accepted = h2c_llm
    n, calls = 4, 3
    extra = {} if stated is None else {"conversation": stated}

    async def one(i: int) -> None:
        async with httpx.AsyncClient(http1=False, http2=True) as client:
            for k in range(calls):
                with wardex.conversation("c", id=f"h{i}"):
                    r = await client.post(
                        f"{base}/v1/responses", content=_body(f"h{i}-{k}", **extra)
                    )
                    assert (r.status_code, r.http_version) == (200, "HTTP/2")

    async def main() -> None:
        await asyncio.gather(*(one(i) for i in range(n)))

    spans, _ = _shipped(lambda: asyncio.run(main()))
    assert len(accepted) == n, "each client was to open its own connection"
    chats = _chats(spans)
    assert {_tag(s): _conv(s) for s in chats} == {
        f"h{i}-{k}": f"h{i}" for i in range(n) for k in range(calls)
    }
    assert {_tag(s): dict(s.extra).get(REQUEST_CONVERSATION_KEY) for s in chats} == {
        f"h{i}-{k}": stated for i in range(n) for k in range(calls)
    }
    assert counters.get("semantics.request_conversation_shadowed") == (
        0 if stated is None else n * calls
    )
    assert counters.get("semantics.request_conversation_withheld") == 0
    totals: dict[str | None, int] = {}
    for s in chats:
        totals[_conv(s)] = totals.get(_conv(s), 0) + s.gen_ai.input_tokens
    assert totals == {f"h{i}": calls * 10 * (i + 1) for i in range(n)}


def _h2c_post_copied(base: str, body: bytes) -> None:
    """One POST from a hand-driven `h2` client that writes a COPY of every
    chunk its state machine hands out: equal bytes, never the object, as a
    client that buffers its own writes would. Nothing proves its connection."""
    import socket

    import h2.config
    import h2.connection
    import h2.events

    host, port = base.removeprefix("http://").split(":")
    sock = socket.create_connection((host, int(port)))
    conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True))
    conn.initiate_connection()
    sock.sendall(_copy(conn.data_to_send()))
    headers = [
        (":method", "POST"),
        (":scheme", "http"),
        (":authority", f"{host}:{port}"),
        (":path", "/v1/responses"),
        ("content-type", "application/json"),
    ]
    conn.send_headers(1, headers)
    conn.send_data(1, body, end_stream=True)
    sock.sendall(_copy(conn.data_to_send()))
    ended = False
    while not ended and (data := sock.recv(65535)):
        for ev in conn.receive_data(data):
            ended = ended or (isinstance(ev, h2.events.StreamEnded) and ev.stream_id == 1)
        if out := conn.data_to_send():
            sock.sendall(_copy(out))
    sock.close()


def test_a_request_body_does_not_outrank_a_conversation_wardex_could_not_read(h2c_llm):
    """HOST WINS needs to know the host's conversation. On a stream whose opener
    nothing proved, wardex does not: the call may have been issued inside a
    block, whose id would win. So the body's id is not made the conversation on
    the guess that there was none; it rides along under its own attribute, and
    the withholding is counted. Outside every block the result is the same, for
    the same reason: from this connection the two cannot be told apart."""
    base, _accepted = h2c_llm

    def run() -> None:
        with wardex.conversation("c", id="host-9"):
            _h2c_post_copied(base, _body("h0-in", conversation="conv_body"))
        _h2c_post_copied(base, _body("h0-out", conversation="conv_body"))

    spans, _ = _shipped(run)
    chats = {_tag(s): s for s in _chats(spans)}
    assert set(chats) == {"h0-in", "h0-out"}
    for s in chats.values():
        assert _conv(s) is None
        assert dict(s.extra)[REQUEST_CONVERSATION_KEY] == "conv_body"
    assert counters.get("semantics.request_conversation_withheld") == 2
    assert counters.get("semantics.request_conversation_shadowed") == 0


def test_a_websocket_session_carries_the_conversation_its_handshake_was_issued_in(
    fake_ssl_socket, bare_ssl_interceptor
):
    itc = bare_ssl_interceptor
    itc._client.config.capture_mode = CaptureMode.ALL
    sock = fake_ssl_socket(alpn=None)
    with _hub.new_scope() as scope:
        scope.conversation = ConversationContext(conversation_id="ws-conv")
        itc._on_request_bytes(
            sock,
            b"GET /chat HTTP/1.1\r\nHost: a\r\nUpgrade: websocket\r\n"
            b"Connection: Upgrade\r\nSec-WebSocket-Key: x\r\n\r\n",
        )
    itc._on_response_bytes(
        sock,
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n"
        b"\x88\x02\x03\xe8",  # the server's close frame, code 1000
    )
    (span,) = itc._client.spans
    assert span.name.startswith("WS")
    assert _conv(span) == "ws-conv"


def _ws_frame(opcode: int, payload: bytes) -> bytes:
    return bytes([0x80 | opcode, len(payload)]) + payload  # FIN, unmasked, payload < 126


_WS_CREATE = _ws_frame(0x1, json.dumps({"type": "response.create", "model": "m"}).encode())
_WS_CLIENT_CLOSE = _ws_frame(0x8, (1000).to_bytes(2, "big"))
#: A Text frame header claiming 2 MiB, above `max_ws_frame_bytes`: the client
#: parser dies on it and never yields a message again.
_WS_OVERSIZE = bytes([0x81, 127]) + (2 * 1024 * 1024).to_bytes(8, "big")


@contextlib.contextmanager
def _issued_in(cid: str | None) -> Iterator[None]:
    """A scope whose ambient conversation is `cid`; None is outside every one."""
    with _hub.new_scope() as scope:
        scope.conversation = None if cid is None else ConversationContext(conversation_id=cid)
        yield


def test_a_websocket_session_reused_in_another_conversation_names_neither(
    fake_ssl_socket, bare_ssl_interceptor
):
    """A pooled Responses socket opened lazily in conversation A, its next
    call issued in B, closed outside both. The session is ONE span and
    neither conversation issued all of it, so it names neither: stamping the
    handshake's A on B's call is the cross-conversation leak this rules out."""
    itc = bare_ssl_interceptor
    itc._client.config.capture_mode = CaptureMode.ALL
    sock = fake_ssl_socket(alpn=None)
    with _issued_in("A"):
        itc._on_request_bytes(
            sock,
            b"GET /v1/responses HTTP/1.1\r\nHost: api.openai.com\r\nUpgrade: websocket\r\n"
            b"Connection: Upgrade\r\nSec-WebSocket-Key: x\r\n\r\n",
        )
    itc._on_response_bytes(
        sock,
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    with _issued_in("B"):
        itc._on_request_bytes(sock, _WS_CREATE)
        itc._on_response_bytes(sock, _ws_frame(0x1, b'{"type":"response.created"}'))
    itc._on_response_bytes(sock, _ws_frame(0x8, (1000).to_bytes(2, "big")))
    (span,) = itc._client.spans
    assert dict(span.extra)["ws.messages.sent"] == 1
    assert span.conversation is None


@pytest.mark.parametrize(
    ("handshake", "messages", "closer", "want"),
    [
        # Every message issued in A. The client's close frame, written outside
        # any conversation, is a control frame and not a message: it splits nothing.
        ("A", ["A", "A"], None, "A"),
        ("A", ["B"], "B", None),  # opened in A, every call in B
        ("A", ["A", "B"], "A", None),  # reused across conversations
        ("A", ["A", None], "A", None),  # one call issued outside every conversation
        (None, ["B"], "B", None),  # opened outside, its calls in B
        (None, [None], None, None),  # nothing to name, nothing named
    ],
)
def test_a_websocket_session_names_a_conversation_only_if_every_message_was_issued_in_it(
    handshake, messages, closer, want
):
    t = _WebSocketTracker(
        path="/v1/responses",
        deflate=False,
        parent=None,
        start_ns=1,
        conversation=None if handshake is None else ConversationContext(conversation_id=handshake),
    )
    for cid in messages:
        with _issued_in(cid):
            assert t.on_request_bytes(_WS_CREATE) == []
    with _issued_in(closer):
        (txn,) = t.on_request_bytes(_WS_CLIENT_CLOSE)
    assert txn.ws_messages_sent == len(messages)
    assert (txn.conversation.conversation_id if txn.conversation else None) == want


@pytest.mark.parametrize(("later", "want"), [("A", "A"), ("B", None), (None, None)])
def test_a_websocket_session_whose_parser_died_weighs_every_later_write(later, want):
    """A dead client parser can no longer tell a message from a control frame,
    so every write after it counts, and one issued elsewhere splits the session."""
    t = _WebSocketTracker(
        path="/v1/responses",
        deflate=False,
        parent=None,
        start_ns=1,
        conversation=ConversationContext(conversation_id="A"),
    )
    with _issued_in("A"):
        t.on_request_bytes(_WS_OVERSIZE)
    assert t._sent.is_disabled()
    with _issued_in(later):
        t.on_request_bytes(b"\x81\x02hi")
    (txn,) = t.flush(Limitation.WS_NO_CLOSE)
    assert (txn.conversation.conversation_id if txn.conversation else None) == want


def test_a_request_body_does_not_outrank_a_conversation_refused_with_a_closed_unit(llm):
    """A call issued under an activation its unit's close could not take down
    latches that unit's fork, and with it the host's `host-9`. The edge refuses
    both as stale, so wardex does not know the call's conversation, and the
    body's `conv_body` would otherwise have become it, beating the host's id it
    may well have been issued in. It rides along instead, counted."""
    from test_units import open_session, registry

    host, port = llm

    def run() -> None:
        with wardex.conversation("c", id="host-9"):
            reg = registry()
            unit = open_session(reg)
            cm = unit.activate()
            cm.__enter__()
            try:
                closer = threading.Thread(target=lambda: reg.close(unit))
                closer.start()
                closer.join()
                _post(host, port, "closed", conversation="conv_body")
            finally:
                del cm

    spans, _ = _shipped(run)
    (chat,) = _chats(spans)
    assert _conv(chat) is None
    assert dict(chat.extra)[REQUEST_CONVERSATION_KEY] == "conv_body"
    assert counters.get("semantics.request_conversation_withheld") == 1


def test_a_closed_units_carrier_does_not_lend_its_conversation():
    """A parent whose unit had already closed is refused at the edge, and the
    conversation latched off the same dead carrier goes with it: keeping it
    would stamp a finished run's conversation on unrelated later work."""
    parent = resolve_parentage(EMPTY_AMBIENT).child_context()
    conv = ConversationContext(conversation_id="finished-run")
    txn = _Txn(
        method="POST",
        path="/v1/responses",
        status=200,
        request_body=b"{}",
        response_body=b"{}",
        parent=parent,
        start_ns=1,
        end_ns=2,
        ttfb_ms=0.0,
        parent_closed=True,
        conversation=conv,
    )
    assert resolve_observed(_latched(txn), parent_closed=True).conversation is None
    live = _Txn(**{**txn.__dict__, "parent_closed": False})
    assert resolve_observed(_latched(live), parent_closed=False).conversation is conv
