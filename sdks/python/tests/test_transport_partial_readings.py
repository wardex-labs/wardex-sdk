"""A count or an interval the seam saw only part of is not a reading.

Every value in a span's transport block is either what the SDK measured or
absent. The cases here each shipped a PART of a measurement, or the wrong
part, as if it were the whole one:

* a non-blocking TLS handshake (an event loop calling `do_handshake()` until
  it stops raising SSLWantReadError) reported the duration of its LAST call;
* a body over its capture cap reported the cap as its size;
* `create_connection` handed a connected socket, or a host name to resolve,
  reported loop overhead or a DNS lookup as the TCP connect;
* an HTTP/2 stream that happened to finish before stream 1 claimed to have
  opened the connection, and stream 1 claimed it opened nothing;
* a response after `100 Continue` timed its first body byte at the arrival of
  its own header block;
* a request still being read when its response arrived reported `0` bytes;
* a WebSocket direction whose frame parser stopped reported the bytes it had
  counted up to then;
* a WebSocket session still open when the SDK let go of it (at
  `wardex.close()`, or when the connection table was full) reported the bytes
  and the length it had seen up to then as the session's.

Each now ships the whole measurement or nothing, on the envelope and on OTLP.
"""

from __future__ import annotations

import asyncio
import http.client
import http.server
import json
import select
import socket
import ssl
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

import test_observed_transport as o
import test_span_class_survival as h
import wardex_sdk as wardex
from test_codec import _header
from wardex_sdk import _hub, _wardex_native
from wardex_sdk._assembly import Limitation
from wardex_sdk._enums import CaptureMode
from wardex_sdk._interceptors._conn_timing import (
    ConnTimingStore,
    _times_only_connect_and_tls,
    _TimingRecord,
    opening_timing,
)
from wardex_sdk._interceptors._seam import ByteSeamInterceptor, _ConnectionState
from wardex_sdk._interceptors._ssl import SSLInterceptor
from wardex_sdk._interceptors._trackers import _Http1Tracker, _Http2Tracker, _WebSocketTracker
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._types import Envelope
from wardex_sdk.testing import RecordingTransport
from wardex_sdk.transport import _codec

_CERT = Path(__file__).parent / "fixtures" / "cert.pem"


def _client_ctx() -> ssl.SSLContext:
    return ssl.create_default_context(cafile=str(_CERT))


def _hostport(url: str) -> tuple[str, int]:
    parts = urlsplit(url)
    assert parts.hostname is not None and parts.port is not None
    return parts.hostname, parts.port


def _request(host: str) -> bytes:
    return (
        f"POST /v1/ping HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\n"
        "Content-Length: 2\r\nConnection: close\r\n\r\n{}"
    ).encode()


def _read_to_eof(sock: socket.socket) -> bytes:
    out = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            return out
        out += chunk


def _spans(transport: RecordingTransport) -> tuple[Envelope, list[dict], list[dict]]:
    """What the envelope and OTLP carry for the captured transport spans."""
    spans = tuple(s for env in transport.envelopes for s in env.spans)
    env = Envelope(header=transport.envelopes[0].header, spans=spans)
    decoded = [i["span"] for i in _codec.decode(_codec.encode(env))["items"] if "span" in i]
    otlp = _wardex_native.codec.decode_otlp_traces(_wardex_native.codec.encode_otlp_traces(env))
    otlp_spans = [
        sp
        for rs in otlp["resource_spans"]
        for ss in rs["scope_spans"]
        for sp in ss["spans"]
        if "wardex.transport.direction" in sp["attributes"]
    ]
    return env, [s for s in decoded if "transport" in s], otlp_spans


def _drive_handshake(tls: ssl.SSLSocket, pause: float) -> int:
    """An event loop's non-blocking handshake: call until it stops raising,
    busy elsewhere for `pause` seconds after the first attempt."""
    attempts = 0
    while True:
        attempts += 1
        try:
            tls.do_handshake()
            return attempts
        except ssl.SSLWantReadError:
            if attempts == 1:
                time.sleep(pause)
            select.select([tls], [], [], 5)
        except ssl.SSLWantWriteError:
            select.select([], [tls], [], 5)


# --- TLS handshake: from the first attempt to the one that completed it ---


def test_a_non_blocking_handshake_is_timed_from_its_first_attempt(tls_server):
    host, port = _hostport(tls_server)
    pause = 0.15
    transport = RecordingTransport()
    try:
        wardex.init(transport=transport, intercept=True, capture_mode=CaptureMode.ALL)
        sock = socket.create_connection((host, port))
        sock.setblocking(False)
        tls = _client_ctx().wrap_socket(sock, server_hostname=host, do_handshake_on_connect=False)
        t0 = time.perf_counter()
        attempts = _drive_handshake(tls, pause)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        tls.setblocking(True)
        tls.sendall(_request(host))
        assert _read_to_eof(tls).startswith(b"HTTP/1.1 200")
        tls.close()
    finally:
        wardex.close()

    assert attempts >= 2, "the handshake must have needed more than one call"
    _env, (span,), (otlp,) = _spans(transport)
    handshake = span["transport"]["timing"]["tls_handshake_ms"]
    # The whole handshake, which the pause after the first attempt is part of —
    # not the last call alone, which took a fraction of a millisecond.
    assert pause * 1000.0 <= handshake <= wall_ms
    assert otlp["attributes"]["wardex.transport.timing.tls_handshake_ms"] == pytest.approx(
        handshake, rel=1e-6
    )
    assert span["transport"]["timing"]["tcp_connect_ms"] is not None
    assert span["transport"]["connection_reused"] is False


def test_a_non_blocking_handshake_begun_before_init_has_no_reading(tls_server):
    """The probe cannot know how many attempts it missed, so no interval it can
    start is the handshake. The connect was before `init` too (so it is marked);
    the TLS session still opened after it, so the connection is a fresh one."""
    host, port = _hostport(tls_server)
    transport = RecordingTransport()
    sock = socket.create_connection((host, port))
    sock.setblocking(False)
    tls = _client_ctx().wrap_socket(sock, server_hostname=host, do_handshake_on_connect=False)
    try:
        with pytest.raises(ssl.SSLWantReadError):
            tls.do_handshake()  # the first attempt, before the probe exists
        wardex.init(transport=transport, intercept=True, capture_mode=CaptureMode.ALL)
        _drive_handshake(tls, 0.05)
        tls.setblocking(True)
        tls.sendall(_request(host))
        assert _read_to_eof(tls).startswith(b"HTTP/1.1 200")
        tls.close()
    finally:
        wardex.close()

    _env, (span,), (otlp,) = _spans(transport)
    t = span["transport"]
    assert t["timing"]["tls_handshake_ms"] is None
    assert t["timing"]["tcp_connect_ms"] is None
    assert "connect_timing_unavailable" in span["capture_integrity"]["limitations"]
    assert t["connection_reused"] is False
    assert "wardex.transport.timing.tls_handshake_ms" not in otlp["attributes"]


def test_the_store_times_a_handshake_once_from_its_first_attempt():
    s = ConnTimingStore()
    s.set_connect(5, 1.0)
    s.handshake_attempt(5, 10.0, 10.001, "pending", blocking=False)
    s.handshake_attempt(5, 10.2, 10.25, "done", blocking=False)
    # A call after completion (OpenSSL answers it at once) is not a handshake.
    s.handshake_attempt(5, 11.0, 11.0001, "done", blocking=False)
    assert s.pop(5) == (1.0, pytest.approx(250.0))


def test_the_store_leaves_an_unattributable_or_failed_handshake_unset():
    s = ConnTimingStore()
    # Non-blocking, no connect seen: the first attempt seen may not be the first.
    s.handshake_attempt(6, 1.0, 1.001, "done", blocking=False)
    assert s.pop(6) == (None, None)
    # A blocking call is the whole handshake on its own, connect seen or not.
    s.handshake_attempt(7, 1.0, 1.02, "done", blocking=True)
    assert s.pop(7) == (None, pytest.approx(20.0))
    s.set_connect(8, 2.0)
    s.handshake_attempt(8, 1.0, 1.02, "failed", blocking=True)
    assert s.pop(8) == (2.0, None)


def test_a_connect_starts_a_new_connection_on_its_fileno():
    """A descriptor is reissued once released; a dead socket's handshake must
    not ride on the next connection that lands on the same number."""
    s = ConnTimingStore()
    s.set_connect(9, 1.0)
    s.handshake_attempt(9, 1.0, 1.01, "done", blocking=True)
    s.set_connect(9, 3.0)
    assert s.pop(9) == (3.0, None)


def test_an_async_handshake_no_observed_call_completed_has_no_reading():
    # OpenSSL can finish a handshake inside the first write; nothing timed it.
    obj = ssl.create_default_context().wrap_bio(
        ssl.MemoryBIO(), ssl.MemoryBIO(), server_hostname="localhost"
    )
    rec = _TimingRecord()
    rec.total_ms = 40.0
    obj._wardex_timing = rec
    st = _ConnectionState(tracker=None, server_address="127.0.0.1", server_port=443)
    assert SSLInterceptor()._resolve_timing(obj, st) == (
        None,
        None,
        False,
        (Limitation.CONNECT_TIMING_UNAVAILABLE,),
    )


# --- TCP connect on the asyncio path: only a wall time that is connect + TLS ---


def test_create_connection_is_timed_only_when_its_wall_time_is_connect_and_tls():
    assert _times_only_connect_and_tls((None, "127.0.0.1", 443), {})
    assert _times_only_connect_and_tls((None,), {"host": "::1", "port": 443})
    assert not _times_only_connect_and_tls((None, "api.openai.com", 443), {})  # DNS inside
    assert not _times_only_connect_and_tls((None,), {"sock": object()})  # connected elsewhere
    assert not _times_only_connect_and_tls((None, None, None), {})


async def _asyncio_tls_call(sni: str, **open_kwargs: object) -> bytes:
    reader, writer = await asyncio.open_connection(
        ssl=_client_ctx(), server_hostname=sni, **open_kwargs
    )
    writer.write(_request(sni))
    await writer.drain()
    response = await reader.read()
    writer.close()
    return response


def _asyncio_tls_span(tls_server: str, open_kwargs: dict) -> dict:
    host, port = _hostport(tls_server)
    transport = RecordingTransport()

    async def call() -> bytes:
        kwargs = dict(open_kwargs)
        if kwargs.pop("connect_first", False):
            raw = socket.socket()
            raw.setblocking(False)
            await asyncio.get_running_loop().sock_connect(raw, (host, port))
            kwargs["sock"] = raw
        else:
            kwargs.setdefault("host", host)
            kwargs["port"] = port
        return await _asyncio_tls_call(host, **kwargs)

    try:
        wardex.init(transport=transport, intercept=True, capture_mode=CaptureMode.ALL)
        assert asyncio.run(call()).startswith(b"HTTP/1.1 200")
    finally:
        wardex.close()
    _env, (span,), (otlp,) = _spans(transport)
    span["otlp"] = otlp["attributes"]
    return span


def test_create_connection_to_an_address_still_times_the_connect(tls_server):
    span = _asyncio_tls_span(tls_server, {})
    t = span["transport"]
    assert t["timing"]["tcp_connect_ms"] is not None
    assert t["timing"]["tls_handshake_ms"] is not None
    assert "connect_timing_unavailable" not in span["capture_integrity"]["limitations"]


def test_create_connection_handed_a_connected_socket_has_no_connect_time(tls_server):
    """aiohttp connects through its own happy-eyeballs dialer and hands asyncio
    the socket. What `create_connection` then times is TLS and loop overhead;
    subtracting the handshake left a fraction of a millisecond, shipped as the
    connect."""
    span = _asyncio_tls_span(tls_server, {"connect_first": True})
    t = span["transport"]
    assert t["timing"]["tcp_connect_ms"] is None
    assert "connect_timing_unavailable" in span["capture_integrity"]["limitations"]
    assert "wardex.transport.timing.tcp_connect_ms" not in span["otlp"]
    assert t["timing"]["tls_handshake_ms"] is not None
    assert t["connection_reused"] is False


def test_create_connection_that_resolved_a_name_has_no_connect_time(tls_server):
    span = _asyncio_tls_span(tls_server, {"host": "localhost"})
    t = span["transport"]
    assert t["timing"]["tcp_connect_ms"] is None
    assert "connect_timing_unavailable" in span["capture_integrity"]["limitations"]
    assert t["timing"]["tls_handshake_ms"] is not None


# --- Sizes: a half whose count is not whole has none ---

_CAP = 4096


class _Echo(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    reply = b""

    def do_POST(self) -> None:  # noqa: N802 — the stdlib's name
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.reply)))
        self.end_headers()
        self.wfile.write(self.reply)

    def log_message(self, *a: object) -> None:
        pass


def test_a_body_over_its_capture_cap_has_no_size():
    """The parser keeps `max_body_bytes` of a body and consumes the rest; the
    kept length used to ship as the body's size."""
    big = json.dumps({"x": "y" * 40000}).encode()
    small = b'{"ok":true}'
    handler = type("_H", (_Echo,), {"reply": big})
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    transport = RecordingTransport()
    try:
        wardex.init(
            transport=transport,
            intercept=True,
            capture_mode=CaptureMode.ALL,
            limits=LimitsConfig(max_body_bytes=_CAP),
        )
        conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1])
        conn.request("POST", "/v1/big", json.dumps({"q": "z" * 30000}), {"X-A": "1"})
        conn.getresponse().read()
        handler.reply = small
        conn.request("POST", "/v1/small", small, {"X-A": "1"})
        conn.getresponse().read()
        conn.close()
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()

    _env, spans, otlp = _spans(transport)
    by_name = {s["name"]: s for s in spans}
    capped = by_name["HTTP POST /v1/big"]
    assert capped["transport"]["request_size"] is None
    assert capped["transport"]["response_size"] is None
    assert "body_cap_exceeded" in capped["capture_integrity"]["limitations"]
    whole = by_name["HTTP POST /v1/small"]
    assert whole["transport"]["request_size"] == len(small)
    assert whole["transport"]["response_size"] == len(small)
    sizes = {
        (a.get("wardex.transport.request_size"), a.get("wardex.transport.response_size"))
        for a in (sp["attributes"] for sp in otlp)
    }
    assert sizes == {(None, None), (len(small), len(small))}


def _h2_seam(client, tracker, opening=(0.0, 0.0, False, ())):
    """The seam harness with a `_resolve_timing` that behaves like the real
    ones: the opening values on the connection's first call, then reuse."""

    class _Seam(h._Seam):
        def _resolve_timing(self, obj, st):
            if st.timing_consumed:
                return (0.0, 0.0, True, ())
            st.timing_consumed = True
            return opening

    seam = _Seam(tracker)
    seam._client = client
    seam._load_limits(client)
    return seam


def _h2_exchange(order: list[int], bodies: dict[int, tuple[bytes, bytes, bytes]]):
    """Requests for every stream in `bodies`, then responses in `order`.
    `bodies[sid] = (path, request body, (response content type, body))`."""
    from hpack import Encoder

    client_enc, server_enc = Encoder(), Encoder()
    requests = b""
    for sid, (path, req, _resp) in bodies.items():
        block = client_enc.encode(
            [
                (b":method", b"POST"),
                (b":path", path),
                (b":authority", b"api.example.com"),
                (b"content-type", b"application/json"),
            ]
        )
        requests += h._h2_frame(0x1, 0x4, sid, block) + h._h2_frame(0x0, 0x1, sid, req)
    responses = []
    for sid in order:
        ctype, body = bodies[sid][2]
        block = server_enc.encode([(b":status", b"200"), (b"content-type", ctype)])
        responses.append(h._h2_frame(0x1, 0x4, sid, block) + h._h2_frame(0x0, 0x1, sid, body))
    return requests, responses


def _drive(seam, requests: bytes, responses: list[bytes]) -> None:
    obj = SimpleNamespace(
        server_hostname="api.example.com", getpeername=lambda: ("203.0.113.9", 443)
    )
    seam._on_request_bytes(obj, requests)
    for chunk in responses:
        seam._on_response_bytes(obj, chunk)


@pytest.fixture
def client() -> h._FakeClient:
    _hub.reset_for_test()
    c = h._FakeClient()
    _hub.set_client(c)
    return c


def test_an_http2_half_over_its_cap_has_no_size_and_the_other_half_keeps_its_own(client):
    tracker = _Http2Tracker(_wardex_native.Limits(max_opaque_body_bytes=4))
    seam = _h2_seam(client, tracker)
    req = b'{"q":1}'
    requests, responses = _h2_exchange(
        [1], {1: (b"/v1/audio/speech", req, (b"audio/mpeg", b"0123456789"))}
    )
    _drive(seam, requests, responses)
    (span,) = client.spans
    assert span.transport.request_size == len(req)
    assert span.transport.response_size is None
    env = Envelope(header=_header(), spans=(span,))
    (decoded,) = [i["span"] for i in _codec.decode(_codec.encode(env))["items"] if "span" in i]
    assert decoded["transport"]["response_size"] is None
    assert decoded["transport"]["request_size"] == len(req)


def test_an_http2_stream_that_finished_before_stream_1_did_not_open_the_connection(client):
    """Streams finish in any order. The connection's opening values used to go
    to whichever finished first: here stream 3, which claimed it opened the
    connection, while stream 1 — the one it was opened for — claimed 0 ms."""
    seam = _h2_seam(client, _Http2Tracker(), opening=(12.5, 34.5, False, ()))
    ok = (b"application/json", b'{"ok":true}')
    requests, responses = _h2_exchange(
        [3, 1], {1: (b"/v1/first", b"{}", ok), 3: (b"/v1/second", b"{}", ok)}
    )
    _drive(seam, requests, responses)
    by_name = {s.name: s.transport for s in client.spans}
    first, second = by_name["HTTP POST /v1/first"], by_name["HTTP POST /v1/second"]
    assert (first.timing.tcp_connect_ms, first.timing.tls_handshake_ms) == (12.5, 34.5)
    assert first.connection_reused is False
    assert (second.timing.tcp_connect_ms, second.timing.tls_handshake_ms) == (0.0, 0.0)
    assert second.connection_reused is True


def _resolver():
    """A `_resolve_timing` stand-in: "open" on the first call, then reuse."""
    calls = iter(["open", "reused", "reused", "reused"])

    def resolve():
        value = next(calls)
        return (value, None, value != "open", ())

    return resolve


def test_opening_timing_holds_the_opening_values_for_stream_1():
    resolve, st = _resolver(), SimpleNamespace(h2_opening=None)
    # Stream 5 finished first: it opened nothing.
    assert opening_timing(resolve(), resolve, st, 5)[:3] == ("reused", None, True)
    # Stream 1, the one the connection was opened for, gets the opening values.
    assert opening_timing(resolve(), resolve, st, 1)[:3] == ("open", None, False)
    assert st.h2_opening is None


def test_opening_timing_gives_http1_the_first_answer_and_reuse_after():
    resolve, st = _resolver(), SimpleNamespace(h2_opening=None)
    assert opening_timing(resolve(), resolve, st, None)[0] == "open"
    assert opening_timing(resolve(), resolve, st, None)[0] == "reused"


def test_an_http1_request_still_being_read_when_its_response_arrived_has_no_size():
    tracker = _Http1Tracker()
    tracker.on_request_bytes(
        b"POST /v1/files HTTP/1.1\r\nHost: a\r\nContent-Length: 100\r\n\r\n" + b"x" * 10
    )
    (txn,) = tracker.on_response_bytes(
        b"HTTP/1.1 413 Payload Too Large\r\nContent-Length: 0\r\n\r\n"
    )
    assert txn.request_counted is False
    assert txn.response_counted is True


def test_ttft_after_an_interim_response_waits_for_the_first_body_byte():
    tracker = _Http1Tracker()
    tracker.on_request_bytes(
        b"POST /v1/x HTTP/1.1\r\nHost: a\r\nExpect: 100-continue\r\nContent-Length: 2\r\n\r\n{}"
    )
    assert tracker.on_response_bytes(b"HTTP/1.1 100 Continue\r\n\r\n") == []
    assert (
        tracker.on_response_bytes(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 11\r\n\r\n"
        )
        == []
    )
    time.sleep(0.1)
    (txn,) = tracker.on_response_bytes(b'{"ok":true}')
    # The header block arrived 100 ms before the first body byte did.
    assert txn.ttft_ms - txn.ttfb_ms >= 100.0


def test_a_websocket_direction_whose_frame_parser_stopped_has_no_size(client):
    limits = _wardex_native.Limits(max_ws_frame_bytes=8)
    tracker = _WebSocketTracker(
        path="/realtime", deflate=False, parent=None, start_ns=1, limits=limits
    )
    seam = h._seam(client, tracker)
    st = h._state(seam, tracker, host="ws.example.com")
    tracker.on_request_bytes(h._ws_frame(True, 0x1, b"hi"))
    # A server frame declaring more than the parser will read: it stops here.
    tracker.on_response_bytes(bytes([0x82, 126]) + (4000).to_bytes(2, "big") + b"z" * 64)
    for txn in tracker.flush(Limitation.WS_NO_CLOSE):
        seam._emit_ws(st, txn)

    (span,) = client.spans
    assert Limitation.FRAME_PARSE_FAILED in span.capture_integrity.limitations
    assert span.transport.request_size == 2
    assert span.transport.response_size is None
    extra = dict(span.extra)
    assert "ws.bytes.received" not in extra and "ws.messages.received" not in extra
    assert extra["ws.bytes.sent"] == 2


def _ws_session_ended_by(client, ending: str):
    """One WebSocket session through the seam's real connection table, ended
    the way `ending` says, before any CLOSE frame crossed."""
    tracker = _WebSocketTracker(
        path="/realtime", deflate=False, parent=None, start_ns=1, limits=_wardex_native.Limits()
    )
    seam = h._seam(client, tracker)
    obj = SimpleNamespace(
        server_hostname="ws.example.com", getpeername=lambda: ("203.0.113.9", 443)
    )
    seam._state(obj)
    tracker.on_request_bytes(h._ws_frame(True, 0x1, b"hello"))
    tracker.on_response_bytes(h._ws_frame(True, 0x1, b"echo:hello"))
    if ending == "socket closed":
        seam._connection_closed(id(obj))  # what the close hook runs
    elif ending == "uninstall":
        ByteSeamInterceptor.uninstall(seam)  # the real one; the harness stubs it out
    else:  # the connection table is full and this session is its oldest entry
        seam._limits = {**seam._limits, "max_connections": 0}
        seam._state(SimpleNamespace(getpeername=lambda: ("203.0.113.10", 443)))
    (span,) = client.spans
    return span


def test_a_websocket_session_whose_socket_closed_keeps_its_length_and_sizes(client):
    # No CLOSE frame, but the socket is gone: nothing more can cross it, so
    # what was counted is the whole session.
    span = _ws_session_ended_by(client, "socket closed")
    assert Limitation.WS_NO_CLOSE in span.capture_integrity.limitations
    assert (span.transport.request_size, span.transport.response_size) == (5, 10)
    assert span.transport.timing.transfer_ms is not None
    assert dict(span.extra)["ws.bytes.received"] == 10


@pytest.mark.parametrize(
    ("ending", "marker"),
    [("uninstall", Limitation.WS_NO_CLOSE), ("evicted", Limitation.CONNECTION_EVICTED)],
)
def test_a_websocket_session_let_go_of_while_open_has_no_length_or_sizes(client, ending, marker):
    span = _ws_session_ended_by(client, ending)
    assert marker in span.capture_integrity.limitations
    assert (span.transport.request_size, span.transport.response_size) == (None, None)
    assert span.transport.timing.transfer_ms is None
    extra = dict(span.extra)
    assert not [k for k in extra if k.startswith(("ws.bytes.", "ws.messages."))], extra


def test_a_websocket_session_open_at_close_ships_no_partial_count_or_length():
    """A long-lived session still open when `wardex.close()` runs, which then
    carries on: the bytes and time seen before the close are part of it."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve() -> None:
        conn, _ = srv.accept()
        with conn:
            o._read_until(conn, b"\r\n\r\n")
            conn.sendall(
                b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                b"Connection: Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n"
            )
            for _ in range(2):
                echo = b"echo:" + o._read_frame(conn)
                conn.sendall(bytes([0x81, len(echo)]) + echo)
            o._read_frame(conn)
            conn.sendall(b"\x88\x02\x03\xe8")
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    transport = RecordingTransport()
    wardex.init(transport=transport, intercept=True, capture_mode=CaptureMode.ALL)
    with socket.create_connection(srv.getsockname()) as sock:
        sock.sendall(
            b"GET /realtime HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\n"
            b"Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            b"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        o._read_until(sock, b"\r\n\r\n")
        sock.sendall(o._masked(0x1, b"before"))
        o._read_frame(sock)
        wardex.close()  # the session is still open
        sock.sendall(o._masked(0x1, b"after the close"))
        o._read_frame(sock)
        sock.sendall(o._masked(0x8, (1000).to_bytes(2, "big")))
        o._read_frame(sock)

    env, decoded, otlp = _spans(transport)
    (ws,) = [s for s in decoded if s["name"] == "WS /realtime"]
    assert "ws_no_close" in ws["capture_integrity"]["limitations"]
    t = ws["transport"]
    assert (t["request_size"], t["response_size"], t["timing"]["transfer_ms"]) == (None, None, None)
    (attrs,) = [sp["attributes"] for sp in otlp if sp["name"] == "WS /realtime"]
    leaked = [
        k
        for k in attrs
        if k.endswith(("_size", "transfer_ms")) or k.startswith(("ws.bytes.", "ws.messages."))
    ]
    assert leaked == [], leaked
