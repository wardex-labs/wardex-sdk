"""A wire span carries the scope identity its work was ISSUED under.

`wardex.set_tag` / `wardex.set_user` stamp every exported span with the tags
and the user of a scope. The interceptors capture far from where a request was
issued: a WebSocket session's span exists only once its socket is closed or
collected, on whichever thread does that; an HTTP/1 reply cut by a close ships
from the closing context; an HTTP/2 reply is read by whichever task holds the
connection; an MCP reply is read by a reader task the session's first caller
started. Read at capture, the scope was the CAPTURING context's, and in a
process serving many tenants that put one tenant's `user.id` on another's span.

The rules these tests hold:

* A span from a byte seam (HTTP/1, HTTP/2, WebSocket) or from MCP stdio carries
  the tags and user of the scope its request was issued in, wherever and
  whenever it is captured.
* The identity is copied when the request is issued: a later `set_user` in the
  same context does not retag a request already sent.
* Where wardex cannot say who issued a request (an HTTP/2 stream whose opener
  was never proven), the span carries no scope identity rather than a guess.
* A snapshot that cannot be read stamps nothing, and is counted.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from hpack import Encoder

import wardex_sdk as wardex
from conftest import _FakeSSLSocket
from test_close_hook import _h2_answer, _h2_open
from test_conversation_wire import _body, _chats, _h2_serving, _shipped, _tag
from wardex_sdk import UserInfo, _hub
from wardex_sdk._assembly import counters
from wardex_sdk._enums import CaptureMode, SpanKind
from wardex_sdk._interceptors._close_hook import (
    close_registry,
    install_shared_close_hook,
    uninstall_shared_close_hook,
)
from wardex_sdk._interceptors._issue_scope import UNKNOWN_ISSUER, issued_scope
from wardex_sdk._interceptors._mcp_stdio import _ProcState
from wardex_sdk._interceptors._ssl import SSLInterceptor
from wardex_sdk.testing import RecordingTransport

pytestmark = pytest.mark.usefixtures("fresh_counters")


@contextlib.contextmanager
def _tenant(name: str) -> Iterator[None]:
    """One request's isolation scope, the way a multi-tenant host opens it."""
    with wardex.isolation_scope():
        wardex.set_user(UserInfo(id=f"user-{name}"))
        wardex.set_tag("tenant", name)
        yield


def _identity(span: Any) -> tuple[str | None, str | None]:
    extra = dict(span.extra)
    return extra.get("tenant"), extra.get("user.id")


@pytest.fixture
def recorded() -> Iterator[RecordingTransport]:
    """A real client (its scope stamp is what is under test), capturing everything."""
    t = RecordingTransport()
    wardex.init(transport=t, intercept=False, capture_mode=CaptureMode.ALL)
    try:
        yield t
    finally:
        wardex.close()


def _spans(t: RecordingTransport) -> list[Any]:
    _hub.get_client()._settle()
    wardex.flush()
    return [s for env in t.envelopes for s in env.spans if s.kind is SpanKind.CLIENT]


def _seam() -> SSLInterceptor:
    """The TLS byte seam on the real client, driven directly with fake sockets."""
    itc = SSLInterceptor()
    itc._client = _hub.get_client()
    itc._load_limits(itc._client)
    return itc


_UPGRADE = (
    b"GET /chat HTTP/1.1\r\nHost: a\r\nUpgrade: websocket\r\n"
    b"Connection: Upgrade\r\nSec-WebSocket-Key: x\r\n\r\n"
)
_SWITCHED = b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n"
_TEXT = bytes([0x81, 0x02]) + b"hi"  # one unmasked text frame


def _open_websocket(itc: SSLInterceptor, held: list[Any]) -> None:
    """A WebSocket session opened (and used) by the CALLING context. The socket
    lives only in `held`, so whoever empties it is whoever collects it."""
    sock = _FakeSSLSocket(None)
    itc._on_request_bytes(sock, _UPGRADE)
    itc._on_response_bytes(sock, _SWITCHED + _TEXT)
    itc._on_request_bytes(sock, _TEXT)
    held.append(sock)


@pytest.mark.parametrize("ending", ["closed", "collected", "evicted", "uninstalled"])
def test_a_websocket_session_ended_by_another_tenant_carries_the_tenant_that_opened_it(
    recorded, ending
):
    """Tenant A opens the session; tenant B's request is what ends it: B's
    thread closes the socket (a pool reaper), drops its last reference (the
    weakref finalizer), opens connections past a full table, or uninstalls the
    seam. The session's span is built at that moment, in B's context."""
    itc = _seam()
    itc._limits = {**itc._limits, "max_connections": 1}
    held: list[Any] = []
    with _tenant("A"):
        _open_websocket(itc, held)
    with _tenant("B"):
        if ending == "closed":
            close_registry().fire(held.pop())
        elif ending == "collected":
            held.clear()
        elif ending == "evicted":
            held += [_FakeSSLSocket(None), _FakeSSLSocket(None)]  # alive, so they stay tracked
            for sock in held[1:]:
                itc._state(sock)
        else:
            itc.uninstall()
    (span,) = [s for s in _spans(recorded) if s.name.startswith("WS")]
    assert _identity(span) == ("A", "user-A")


def test_an_http1_reply_cut_by_another_tenants_close_carries_the_tenant_that_sent_it(recorded):
    """The reply's headers and part of its body arrived; then tenant B's thread
    closed the socket, which ships the partial span from B's context."""
    itc = _seam()
    sock = _FakeSSLSocket(None)
    with _tenant("A"):
        itc._on_request_bytes(
            sock, b"POST /v1/x HTTP/1.1\r\nHost: a\r\nContent-Length: 2\r\n\r\n{}"
        )
        itc._on_response_bytes(sock, b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{")
    with _tenant("B"):
        close_registry().fire(sock)
    (span,) = _spans(recorded)
    assert span.name == "HTTP POST /v1/x"
    assert _identity(span) == ("A", "user-A")


def test_a_set_user_after_the_request_was_sent_does_not_retag_it(recorded):
    """Copied at the request, not referenced: the scope is mutable, and the
    reply here is read in the very context that changed its user."""
    itc = _seam()
    sock = _FakeSSLSocket(None)
    with _tenant("A"):
        itc._on_request_bytes(sock, b"GET /v1/x HTTP/1.1\r\nHost: a\r\n\r\n")
        wardex.set_user(UserInfo(id="user-A-next"))
        wardex.set_tag("tenant", "A-next")
        itc._on_response_bytes(sock, b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
    (span,) = _spans(recorded)
    assert _identity(span) == ("A", "user-A")


def test_concurrent_tenants_on_one_http2_connection_keep_their_own_identity():
    """`asyncio.gather` over ONE HTTP/2 connection (httpx's `http2=True`, as an
    OpenAI or Anthropic client built on it), one tenant per call. Any task may
    read a reply for any other's stream, so every call must still carry the
    identity of the tenant that issued it, never the reader's."""
    import ssl

    from conftest import CERT

    n = 8
    verify = ssl.create_default_context(cafile=str(CERT))

    with _h2_serving(tls=True) as (base, accepted):

        async def one(client: httpx.AsyncClient, i: int) -> None:
            with _tenant(f"h{i}"):
                r = await client.post(f"{base}/v1/responses", content=_body(f"h{i}-0"))
                assert (r.status_code, r.http_version) == (200, "HTTP/2")

        async def main() -> None:
            async with httpx.AsyncClient(http2=True, verify=verify) as client:
                await asyncio.gather(*(one(client, i) for i in range(n)))

        spans, _ = _shipped(lambda: asyncio.run(main()))
    assert len(accepted) == 1, "the calls did not share one connection"
    chats = _chats(spans)
    assert sorted(_tag(s) for s in chats) == sorted(f"h{i}-0" for i in range(n))
    wrong = {_tag(s): _identity(s) for s in chats if _identity(s)[0] != _tag(s).split("-")[0]}
    assert not wrong, f"calls carrying ANOTHER tenant's identity (or none): {wrong}"
    assert {_tag(s): _identity(s)[1] for s in chats} == {f"h{i}-0": f"user-h{i}" for i in range(n)}


_H2_REQUEST = [(":method", "POST"), (":scheme", "https"), (":authority", "a"), (":path", "/v1/m")]


@pytest.fixture
def h2_state_machine() -> Iterator[Any]:
    """`h2.connection.H2Connection` with the issuer reads on it, as `install()` puts them."""
    import h2.connection

    install_shared_close_hook()
    try:
        yield h2.connection.H2Connection
    finally:
        uninstall_shared_close_hook()


def test_an_h2_stream_carries_the_identity_of_the_tenant_that_opened_it(recorded, h2_state_machine):
    """httpcore's shape without the network: tenants A and B queue their
    streams on one connection, tenant W's task writes every queued frame, and
    tenant R's task reads both replies. The proven openers are A and B."""
    itc = _seam()
    sock = _FakeSSLSocket("h2")
    conn = h2_state_machine()
    conn.initiate_connection()
    itc._on_request_bytes(sock, conn.data_to_send())  # the preface links the state machine
    for sid, name in ((1, "A"), (3, "B")):
        with _tenant(name):
            conn.send_headers(sid, _H2_REQUEST, end_stream=True)
    with _tenant("W"):
        itc._on_request_bytes(sock, conn.data_to_send())
    server = Encoder()
    with _tenant("R"):
        itc._on_response_bytes(sock, _h2_answer(server, 3) + _h2_answer(server, 1))
    assert sorted(_identity(s) for s in _spans(recorded)) == [("A", "user-A"), ("B", "user-B")]


def test_an_h2_stream_whose_opener_was_not_proven_carries_no_identity(recorded):
    """No `h2` state machine behind the bytes: the scope a stream was WRITTEN
    in is any task's on a shared connection, so it is not the issuer's, and
    neither is the reader's. No identity is guessed."""
    itc = _seam()
    sock = _FakeSSLSocket("h2")
    client_enc, server_enc = Encoder(), Encoder()
    with _tenant("A"):
        itc._on_request_bytes(sock, _h2_open(client_enc, 1))
    with _tenant("B"):
        itc._on_response_bytes(sock, _h2_answer(server_enc, 1))
    (span,) = _spans(recorded)
    assert _identity(span) == (None, None)


def test_mcp_replies_read_in_another_context_carry_the_tenant_that_called():
    """The reader that completes a call runs in whatever context started it."""
    state = _ProcState()
    for rid, name in ((1, "A"), (2, "B")):
        with _tenant(name):
            state.feed_request(
                json.dumps({"jsonrpc": "2.0", "id": rid, "method": "tools/call"}).encode() + b"\n"
            )
    with _tenant("R"):
        out = state.feed_response(
            b'{"jsonrpc":"2.0","id":2,"result":{}}\n{"jsonrpc":"2.0","id":1,"result":{}}\n'
        )
    assert [(tags["tenant"], user.id) for _span, (tags, user) in out] == [
        ("B", "user-B"),
        ("A", "user-A"),
    ]


_MCP_SERVER = (
    "import json, sys\n"
    "for line in sys.stdin:\n"
    "    msg = json.loads(line)\n"
    "    out = {'jsonrpc': '2.0', 'id': msg['id'], 'result': {'content': []}}\n"
    "    sys.stdout.write(json.dumps(out) + '\\n'); sys.stdout.flush()\n"
)


def test_an_mcp_reader_started_in_one_tenants_request_does_not_lend_it_to_the_next():
    """End to end on the raw-asyncio seam: the shared reader task is started
    lazily inside tenant A's request, as a shared client is, and then reads
    tenant B's reply too. B's call carried A's user."""
    t = RecordingTransport()
    wardex.init(transport=t, intercept=True, capture_mode=CaptureMode.ALL)
    try:

        async def main() -> None:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _MCP_SERVER,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
            )
            waiting: dict[int, asyncio.Future[Any]] = {}
            readers: list[asyncio.Task[None]] = []

            async def read() -> None:
                while line := await proc.stdout.readline():
                    waiting.pop(json.loads(line)["id"]).set_result(None)

            async def call(name: str, rid: int) -> None:
                with _tenant(name):
                    if not readers:
                        readers.append(asyncio.create_task(read()))
                    waiting[rid] = asyncio.get_running_loop().create_future()
                    req = {"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": {}}
                    proc.stdin.write(json.dumps(req).encode() + b"\n")
                    await proc.stdin.drain()
                    await waiting[rid]

            await call("A", 1)
            await call("B", 2)
            proc.stdin.close()
            await proc.wait()
            await asyncio.gather(*readers)

        asyncio.run(main())
        spans = _spans(t)
    finally:
        wardex.close()
    mcp = [s for s in spans if s.transport is not None and s.transport.mcp is not None]
    assert [(s.transport.mcp.rpc_id, _identity(s)) for s in mcp] == [
        ("1", ("A", "user-A")),
        ("2", ("B", "user-B")),
    ]


def test_a_snapshot_that_cannot_be_read_stamps_nothing_and_is_counted(recorded, monkeypatch):
    """The read runs in the host's send path, so a raise there (a tag dict
    mutated on another thread mid-copy) is swallowed, counted, and stamps
    NOTHING: falling back to the capturing context's scope is the bug."""

    def broken() -> Any:
        raise RuntimeError("dictionary changed size during iteration")

    with monkeypatch.context() as m:
        m.setattr(_hub, "get_merged_tags_and_user", broken)
        assert issued_scope() is UNKNOWN_ISSUER
        itc = _seam()
        sock = _FakeSSLSocket(None)
        with _tenant("A"):
            itc._on_request_bytes(sock, b"GET /v1/x HTTP/1.1\r\nHost: a\r\n\r\n")
    assert counters.get("interceptors.issue_scope") == 2
    with _tenant("B"):
        itc._on_response_bytes(sock, b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
    (span,) = _spans(recorded)
    assert _identity(span) == (None, None)
