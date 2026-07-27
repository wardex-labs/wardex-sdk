"""Native limits surface: defaults, construction, and parser wiring."""

from __future__ import annotations

import gzip
import tracemalloc

import pytest

from wardex_sdk import CaptureLimits, WardexConfig, _wardex_native


def test_limits_defaults_returns_every_field():
    d = _wardex_native.limits_defaults()
    assert d["max_headers"] == 96
    assert d["max_body_bytes"] == 32 * 1024 * 1024
    assert d["max_opaque_body_bytes"] == 256 * 1024
    assert d["zstd_level"] == 3
    assert len(d) == 16


def test_limits_construction_defaults_unspecified_fields():
    lim = _wardex_native.Limits(max_body_bytes=1024)
    assert lim.max_body_bytes == 1024
    assert lim.max_headers == 96  # untouched field keeps the core default


def test_parser_accepts_limits():
    lim = _wardex_native.Limits(max_headers=1)
    p = _wardex_native.protocol.Http1Parser(False, lim)
    raw = b"HTTP/1.1 200 OK\r\nA: 1\r\nB: 2\r\nContent-Length: 0\r\n\r\n"
    assert p.feed(raw) == []


def test_parser_without_limits_uses_defaults():
    p = _wardex_native.protocol.Http1Parser(False)
    raw = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi"
    assert len(p.feed(raw)) == 1


def test_mirror_field_set_matches_core():
    core = _wardex_native.limits_defaults()
    assert set(CaptureLimits.__dataclass_fields__) == set(core)


def test_mirror_holds_no_values():
    # None means "use the core default". A mirror that carries its own values
    # is exactly the drift this design exists to prevent.
    lim = CaptureLimits()
    assert all(getattr(lim, f) is None for f in CaptureLimits.__dataclass_fields__)


def test_to_native_applies_overrides_only():
    native = CaptureLimits(max_body_bytes=1024).to_native()
    assert native.max_body_bytes == 1024
    assert native.max_headers == 96


def test_config_rejects_non_positive_limit():
    import pytest

    with pytest.raises(ValueError, match="max_body_bytes"):
        WardexConfig(limits=CaptureLimits(max_body_bytes=0))


def test_moved_fields_raise_helpful_error():
    import pytest

    with pytest.raises(TypeError, match="limits=CaptureLimits"):
        WardexConfig(max_buffer_spans=100)


def test_body_cap_reaches_the_parser_end_to_end():
    """A limit set on the config must reach the native parser via _Http1Tracker,
    not just sit in config.

    Uses max_headers rather than max_body_bytes/max_opaque_body_bytes. All
    three are enforced on the HTTP/1 path — crates/wardex-protocol/src/http1.rs
    caps bodies by content type and marks the message it capped — so any of
    them could carry this test. max_headers is kept because its signal is
    binary and cannot be produced any other way: the response parses under the
    default cap and does not parse under the configured one, so a green result
    is only possible if the value actually reached the native parser. The body
    caps are covered where their own failure modes live: the parser's tests in
    http1.rs, and test_http1_body_cap_is_visible_to_the_user below, which
    follows a cap's marker through the seam into CaptureIntegrity.
    """
    import wardex_sdk
    from wardex_sdk import _hub

    wardex_sdk.init(limits=CaptureLimits(max_headers=1))
    try:
        from wardex_sdk.interceptors._trackers import _Http1Tracker

        client = _hub.get_client()
        assert client is not None
        t = _Http1Tracker(client.config.limits.to_native())
        t.on_request_bytes(b"GET / HTTP/1.1\r\nContent-Length: 0\r\n\r\n")
        # Two headers exceeds the configured cap of one — with the default
        # max_headers=96 this response would parse into one transaction, so
        # getting zero here proves the cap reached the native parser.
        raw = b"HTTP/1.1 200 OK\r\nA: 1\r\nB: 2\r\nContent-Length: 0\r\n\r\n"
        txns = t.on_response_bytes(raw)
        assert txns == [], "max_headers=1 must block parsing of a 2-header response"
    finally:
        wardex_sdk.close()


def test_native_parser_exposes_disabled_reason():
    """`Http1Parser.disabled_reason()` (bindings/python/src/lib.rs) must delegate
    to the core Http1Stream, not just exist as a stub — None while parsing
    normally, and the machine-readable reason once the parser latches off.
    """
    p = _wardex_native.protocol.Http1Parser(False)
    assert p.disabled_reason() is None
    # Not HTTP at all → "not_http".
    assert p.feed(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n") == []
    assert p.disabled_reason() == "not_http"


def test_http1_tracker_reports_disabled_reason_from_either_direction():
    """`_Http1Tracker.disabled_reason()` must surface a latch on either the
    request or the response parser — whichever direction actually saw the
    non-HTTP traffic.
    """
    from wardex_sdk.interceptors._trackers import _Http1Tracker

    t = _Http1Tracker()
    assert t.disabled_reason() is None
    t.on_response_bytes(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n")
    assert t.disabled_reason() == "not_http"


def test_python_side_fallback_defaults_match_core():
    """Every Python-side object that accepts a resource-bound parameter with a
    default of None must resolve that default from the core (limits_defaults()),
    never from a hardcoded literal that lives only in Python.

    A hardcoded fallback (e.g. `sample_cap: int = 64 * 1024`) would pass every
    other test in this suite yet silently disagree with crates/wardex-limits the
    moment someone changes the core default without touching Python — exactly
    the drift class this task exists to close. This test fails immediately if
    that happens, because it compares the *effective* default against the core,
    not against another Python literal.
    """
    from wardex_sdk.adapters._assembler import SessionAssembler
    from wardex_sdk.interceptors._conn_timing import ConnTimingStore
    from wardex_sdk.interceptors._mcp_stdio import _ProcState
    from wardex_sdk.interceptors._trackers import _WebSocketTracker

    core = _wardex_native.limits_defaults()

    asm = SessionAssembler(client=None)
    assert asm._max_sessions == core["max_sessions"]
    assert asm._max_session_entries == core["max_session_entries"]

    assert _ProcState.SNIFF_LIMIT == core["mcp_sniff_bytes"]
    assert _ProcState()._sniff_limit == core["mcp_sniff_bytes"]

    tracker = _WebSocketTracker(path="/x", deflate=False, parent=None, start_ns=1)
    assert tracker._sample_cap == core["ws_sample_bytes"]

    assert ConnTimingStore()._cap == core["max_connections"]


# --- End-to-end plumbing + the OOM regression -------------------------------
#
# Everything above proves a single layer in isolation: the Rust side takes a
# Limits object directly, the Python side mocks its neighbours. Both can be
# green while the chain connecting init() to the native parser is severed and
# everything silently runs on core defaults -- that exact failure already
# happened once in this slice, in the opposite direction (a stale native
# module served old behavior while every Rust test passed). The tests below
# drive the real public entry points end to end instead.
#
# Three limits stand in for the three layers a configured value has to cross:
#   - max_headers: enforced inside the native Rust parser. Already pinned end
#     to end by test_body_cap_reaches_the_parser_end_to_end above (it drives
#     init() -> _hub.get_client() -> config.limits.to_native() -> a real
#     _Http1Tracker -> the native Http1Parser), so it is not repeated here.
#   - max_buffer_bytes: enforced by the Python Client's span buffer
#     (Client.capture_span / _SpanBuffer), independent of anything native.
#   - max_connections: enforced by the interceptor's own per-connection
#     eviction (ByteSeamInterceptor._state), a third layer distinct from both
#     the parser and the Client -- this is the "one more of your choosing".
# max_body_bytes/max_opaque_body_bytes are not repeated here. Both are
# enforced on the HTTP/1 path and both are already driven end to end by
# test_http1_body_cap_is_visible_to_the_user, which feeds the byte seam and
# asserts the cap's marker arrives in CaptureIntegrity. What this section
# selects for is one limit per distinct enforcement layer, and the layer a
# body cap exercises -- the native parser -- is the one max_headers already
# stands in for.


def test_max_buffer_bytes_reaches_the_client():
    """max_buffer_bytes must reach Client.capture_span's eviction, not just
    sit in config. With the configured cap far below what 20 spans of 8KB
    payload would need, eviction must have actually run.
    """
    import wardex_sdk
    from wardex_sdk import _hub
    from wardex_sdk._enums import SpanKind
    from wardex_sdk._types import InternalSpan, SpanContext, SpanId, TraceId

    def span(payload: bytes) -> InternalSpan:
        return InternalSpan(
            context=SpanContext(TraceId.generate(), SpanId.generate()),
            parent_span_id=None,
            name="s",
            kind=SpanKind.INTERNAL,
            start_time_ns=1,
            end_time_ns=2,
            output_data=payload,
        )

    wardex_sdk.init(limits=CaptureLimits(max_buffer_bytes=32 * 1024))
    try:
        client = _hub.get_client()
        assert client is not None
        for _ in range(20):
            client.capture_span(span(b"x" * 8192))
        # Unbounded (core default is 64MB), 20 * (~8192 + per-span overhead)
        # would sit well under the cap and nothing would ever be evicted --
        # so this only passes if the 32KB override actually reached the
        # buffer's eviction check.
        assert client._buffered_bytes <= 32 * 1024
        assert len(client._spans) < 20
    finally:
        wardex_sdk.close()


def test_max_connections_reaches_the_seam():
    """max_connections must reach ByteSeamInterceptor._state's per-connection
    eviction. Loads limits the same way SSLInterceptor.install() does
    (_load_limits), without the global ssl.SSLSocket monkeypatch that a full
    install() would perform -- this test only needs the seam's own state
    dict, not real TLS traffic.
    """
    import wardex_sdk
    from wardex_sdk import _hub
    from wardex_sdk.interceptors._ssl import SSLInterceptor

    wardex_sdk.init(limits=CaptureLimits(max_connections=1))
    try:
        client = _hub.get_client()
        assert client is not None
        itc = SSLInterceptor()
        itc._client = client
        itc._load_limits(client)
        assert itc._limits["max_connections"] == 1

        class _FakeSock:
            def selected_alpn_protocol(self) -> str | None:
                return None

            def getpeername(self) -> tuple[str, int]:
                return ("127.0.0.1", 443)

            def fileno(self) -> int:
                return -1

        socks = [_FakeSock() for _ in range(4)]
        for s in socks:
            itc._on_request_bytes(s, b"GET / HTTP/1.1\r\n\r\n")

        # Default max_connections (4096) would never evict after just 4
        # connections -- eviction only fires here because the override of 1
        # reached the seam. Steady-state size is max_connections + 1 (the
        # eviction check runs before the new entry is added), so the two
        # oldest connections must be gone and the two newest must remain.
        assert len(itc._conns) == 2
        assert id(socks[0]) not in itc._conns
        assert id(socks[1]) not in itc._conns
        assert id(socks[2]) in itc._conns
        assert id(socks[3]) in itc._conns
    finally:
        wardex_sdk.close()


def test_non_http_tls_traffic_does_not_grow_memory(fake_ssl_socket, bare_ssl_interceptor):
    """Regression for the production incident: the SDK patches ssl.SSLSocket
    globally, so a TLS-backed Redis/Mongo/Kafka client sharing the process
    streams into the same seam as instrumented HTTP traffic.

    Before the sniff-latch gate (SSLInterceptor._gate) existed, a connection
    that never spoke HTTP still reached _Http1Tracker.on_response_bytes on
    every read. That method appends to a plain Python list
    (_resp_marks) on every single call, unconditionally and without any
    cap -- unlike the raw bytes, which the native parser's own
    max_stream_buffer_bytes latch does eventually bound. Held open for the
    life of a long-running non-HTTP connection (and multiplied across every
    such connection sharing the process), that per-call bookkeeping is what
    grew memory until the host was OOM-killed.

    The gate closes this by classifying a connection once, from its first
    request bytes, and short-circuiting every later call once it is latched
    "ignore" -- so neither direction ever reaches the tracker again. Traced
    allocation must stay flat as far more traffic arrives, not scale with the
    volume of traffic fed.
    """
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn=None)
    # A real Redis client's first write over the connection: enough to
    # classify (and latch) it as non-HTTP, exactly like a live TLS-backed
    # Redis client would.
    itc._on_request_bytes(sock, b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n")
    assert itc._conns[id(sock)].gate == "ignore"

    # A bulk-string reply carrying a multi-KB value, the shape a Redis GET
    # under load would return, repeated as if the connection stayed open and
    # kept serving traffic.
    response = b"$4096\r\n" + b"v" * 4096 + b"\r\n"

    tracemalloc.start()
    try:
        for _ in range(20):
            itc._on_response_bytes(sock, response)
        baseline = tracemalloc.get_traced_memory()[0]
        for _ in range(2000):
            itc._on_response_bytes(sock, response)
        peak = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()

    # 100x more traffic must not retain anywhere close to 100x more memory.
    # The threshold is derived from this test's own chunk size (not a bare
    # literal) so it tracks the test's inputs rather than an arbitrary
    # constant: it sits far above the handful of bytes any legitimate
    # per-call bookkeeping could cost, and far below what even a few retained
    # response chunks would cost, let alone 2000 of them.
    growth = peak - baseline
    assert growth < len(response) * 10, (
        f"traced allocation grew by {growth} bytes over 2000 calls "
        f"(baseline={baseline}, peak={peak}) -- expected it to stay flat"
    )


# --- Structural guard: every advertised limit must actually be enforced ------
#
# The end-to-end tests above pin three fields. The other thirteen were never
# audited, and two of them were inert: the codec encoder hardcoded the default
# limits (so a configured zstd_level was validated and then discarded) and the
# byte seam called the semantic parser without limits (so max_decoded_bytes was
# equally inert) -- while the README and the changelog both promised that every
# resource bound in the SDK is configurable. Nothing in the suite could notice.
#
# _PROBES closes that. Each probe drives the real code path twice, once with an
# override and once on the core defaults, and returns True only if the two
# outcomes differ: a probe that would still pass with the limit ignored proves
# nothing. _NOT_ENFORCED names the fields no probe can cover, each with its
# reason. The two must partition the mirror's field set, so neither adding a
# limit nor quietly unwiring one can pass without this file saying which it is.


def _native(**kw: int):
    return _wardex_native.Limits(**kw)


def _h2_frame(ftype: int, flags: int, stream_id: int, payload: bytes) -> bytes:
    return (
        len(payload).to_bytes(3, "big")
        + bytes([ftype, flags])
        + stream_id.to_bytes(4, "big")
        + payload
    )


def _ws_text_frame(payload: bytes) -> bytes:
    # Unmasked FIN text frame with a 16-bit extended length.
    return b"\x81\x7e" + len(payload).to_bytes(2, "big") + payload


def _http1_body(content_type: str, body: bytes, limits) -> bytes:
    raw = (
        f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    return _wardex_native.protocol.Http1Parser(False, limits).feed(raw)[0].body


def _probe_max_headers() -> bool:
    raw = b"HTTP/1.1 200 OK\r\nA: 1\r\nB: 2\r\nContent-Length: 0\r\n\r\n"
    tight = _wardex_native.protocol.Http1Parser(False, _native(max_headers=1)).feed(raw)
    loose = _wardex_native.protocol.Http1Parser(False, None).feed(raw)
    return tight == [] and len(loose) == 1


def _probe_max_body_bytes() -> bool:
    body = b"0123456789"
    tight = _http1_body("application/json", body, _native(max_body_bytes=4))
    loose = _http1_body("application/json", body, None)
    return tight == b"0123" and loose == body


def _probe_max_opaque_body_bytes() -> bool:
    body = b"0123456789"
    tight = _http1_body("application/octet-stream", body, _native(max_opaque_body_bytes=4))
    loose = _http1_body("application/octet-stream", body, None)
    return tight == b"0123" and loose == body


def _probe_max_stream_buffer_bytes() -> bool:
    # A header block that never reaches its terminating blank line: residue no
    # parse pass can consume, which is exactly what this ceiling bounds.
    data = b"HTTP/1.1 200 OK\r\nX-Padding: " + b"a" * 200 + b"\r\n"
    tight = _wardex_native.protocol.Http1Parser(False, _native(max_stream_buffer_bytes=64))
    loose = _wardex_native.protocol.Http1Parser(False, None)
    tight.feed(data)
    loose.feed(data)
    return tight.disabled_reason() == "stream_buffer_exceeded" and loose.disabled_reason() is None


_OPENAI_REQ = b'{"model":"gpt-4o-mini"}'
_OPENAI_RESP = (
    b'{"id":"chatcmpl-abc","model":"gpt-4o-mini-2024-07-18",'
    b'"choices":[{"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":12,"completion_tokens":3}}'
)


class _StubClient:
    """A client the byte seam can drive without init()/close() and without a
    background worker draining the spans out from under the assertion."""

    class _Config:
        debug = False

        def __init__(self, limits: CaptureLimits) -> None:
            from wardex_sdk._enums import CaptureMode

            self.limits = limits
            # The probe traffic carries no active local span, so AGENT mode
            # would gate out anything without LLM semantics.
            self.capture_mode = CaptureMode.ALL

    def __init__(self, limits: CaptureLimits) -> None:
        self.config = self._Config(limits)
        self.spans: list = []

    def capture_span(self, span) -> None:
        self.spans.append(span)


def _drive_seam(limits: CaptureLimits, request: bytes, response: bytes, host: str):
    """Feed one HTTP/1 exchange through the real TLS byte seam, loading limits
    exactly as SSLInterceptor.install() does, and return the emitted span."""
    from conftest import _FakeSSLSocket
    from wardex_sdk.interceptors._ssl import SSLInterceptor

    client = _StubClient(limits)
    itc = SSLInterceptor()
    itc._client = client
    itc._load_limits(client)

    sock = _FakeSSLSocket(None)
    sock.server_hostname = host
    itc._on_request_bytes(sock, request)
    itc._on_response_bytes(sock, response)
    return client.spans[-1] if client.spans else None


def _openai_exchange(body: bytes, encoding: str | None) -> tuple[bytes, bytes]:
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: api.openai.com\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(_OPENAI_REQ)}\r\n\r\n".encode()
        + _OPENAI_REQ
    )
    headers = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
    if encoding:
        headers += f"Content-Encoding: {encoding}\r\n".encode()
    response = headers + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
    return request, response


def _probe_max_decoded_bytes() -> bool:
    # Driven through the byte seam rather than straight at the binding: the
    # binding has honored `limits` all along, and what was inert was the seam
    # calling it without them. A direct call could never have seen that.
    # The cap bites only on the decompression path, so the body is gzipped.
    request, response = _openai_exchange(gzip.compress(_OPENAI_RESP), "gzip")

    def model(limits: CaptureLimits) -> str | None:
        span = _drive_seam(limits, request, response, "api.openai.com")
        return span.gen_ai.response_model if span is not None and span.gen_ai else None

    # Under the tiny cap the body stays compressed and nothing can be read out
    # of it; under the default it decompresses and parses.
    return (
        model(CaptureLimits(max_decoded_bytes=1)) is None
        and model(CaptureLimits()) == "gpt-4o-mini-2024-07-18"
    )


def _probe_max_streams() -> bool:
    from hpack import Encoder

    def surviving(limits) -> int:
        parser = _wardex_native.protocol.Http2Parser(limits)
        client_enc, server_enc = Encoder(), Encoder()
        sids = [1 + 2 * i for i in range(8)]
        parser.feed(
            True,
            b"".join(
                _h2_frame(
                    0x1, 0x4, sid, client_enc.encode([(b":method", b"POST"), (b":path", b"/s")])
                )
                for sid in sids
            ),
        )
        _opened, txns = parser.feed(
            False,
            b"".join(
                _h2_frame(0x1, 0x4 | 0x1, sid, server_enc.encode([(b":status", b"200")]))
                for sid in sids
            ),
        )
        # A stream evicted by the cap loses the request half it was holding, so
        # its transaction comes back without the method it was opened with.
        return sum(1 for t in txns if t.method == "POST")

    return surviving(_native(max_streams=1)) < 8 and surviving(None) == 8


def _probe_max_ws_frame_bytes() -> bool:
    from wardex_sdk.protocol import WsParser

    frame = _ws_text_frame(b"x" * 300)
    tight, loose = WsParser(_native(max_ws_frame_bytes=8)), WsParser(None)
    tight.feed(frame)
    loose.feed(frame)
    return tight.is_disabled() and not loose.is_disabled()


def _probe_ws_sample_bytes() -> bool:
    from wardex_sdk.protocol import WsParser

    frame = _ws_text_frame(b"x" * 300)
    tight = WsParser(_native(ws_sample_bytes=8)).feed(frame)
    loose = WsParser(None).feed(frame)
    return len(tight.messages[0]) == 8 and len(loose.messages[0]) == 300


def _probe_max_connections() -> bool:
    from wardex_sdk.interceptors._ssl import SSLInterceptor

    class _FakeSock:
        def selected_alpn_protocol(self) -> str | None:
            return None

        def getpeername(self) -> tuple[str, int]:
            return ("127.0.0.1", 443)

    def tracked(limits: CaptureLimits) -> int:
        itc = SSLInterceptor()
        itc._limits = limits.resolved()
        for sock in [_FakeSock() for _ in range(4)]:
            itc._state(sock)
        return len(itc._conns)

    # Steady state is max_connections + 1: the eviction check runs before the
    # new entry is inserted.
    return tracked(CaptureLimits(max_connections=1)) == 2 and tracked(CaptureLimits()) == 4


class _RecordingClient:
    def __init__(self) -> None:
        self.spans: list[object] = []

    def capture_span(self, span: object) -> None:
        self.spans.append(span)


def _assembler(limits: CaptureLimits):
    """Built exactly the way the Agent SDK adapter builds it at install time."""
    from wardex_sdk.adapters._assembler import SessionAssembler

    resolved = limits.resolved()
    return SessionAssembler(
        _RecordingClient(),
        max_sessions=resolved["max_sessions"],
        max_session_entries=resolved["max_session_entries"],
    )


def _probe_max_sessions() -> bool:
    def open_sessions(limits: CaptureLimits) -> int:
        asm = _assembler(limits)
        for key in range(4):
            asm._ensure_session(key, 1)
        return asm.open_session_count()

    return open_sessions(CaptureLimits(max_sessions=1)) == 1 and open_sessions(CaptureLimits()) == 4


def _probe_max_session_entries() -> bool:
    def open_tools(limits: CaptureLimits) -> int:
        asm = _assembler(limits)
        sess = asm._ensure_session(1, 1)
        for i in range(4):
            asm._open_tool(sess, {"tool_name": "t"}, f"id{i}", 1)
        return len(sess.open_tools)

    tight = open_tools(CaptureLimits(max_session_entries=1))
    return tight == 1 and open_tools(CaptureLimits()) == 4


def _probe_mcp_sniff_bytes() -> bool:
    from wardex_sdk.interceptors._mcp_stdio import _ProcState

    def detaches(sniff_limit: int) -> bool:
        state = _ProcState(sniff_limit)
        state.feed_request(b"not json-rpc at all\n" * 8)
        return state.should_detach()

    core = _wardex_native.limits_defaults()["mcp_sniff_bytes"]
    return detaches(16) and not detaches(core)


def _probe_max_buffer_spans() -> bool:
    return _dropped_under(CaptureLimits(max_buffer_spans=2)) > 0


def _probe_max_buffer_bytes() -> bool:
    return _dropped_under(CaptureLimits(max_buffer_bytes=32 * 1024)) > 0


def _dropped_under(limits: CaptureLimits) -> int:
    """Spans evicted by the buffer's caps while capturing 20 fixed-size spans.

    On the core defaults (2048 spans / 64 MiB) 20 small spans evict nothing, so
    a non-zero count can only come from the override reaching
    Client.capture_span.

    Draining is stubbed out first. A small max_buffer_spans also lowers the
    flush threshold, so the background worker would otherwise race the eviction
    it is supposed to prove -- emptying the buffer before it ever fills makes
    the probe report "no eviction" for a limit that is wired correctly.
    """
    import wardex_sdk
    from wardex_sdk import _hub
    from wardex_sdk._enums import SpanKind
    from wardex_sdk._types import InternalSpan, SpanContext, SpanId, TraceId

    wardex_sdk.init(limits=limits)
    try:
        client = _hub.get_client()
        assert client is not None
        client._drain = lambda *a, **kw: None  # type: ignore[method-assign]
        for _ in range(20):
            client.capture_span(
                InternalSpan(
                    context=SpanContext(TraceId.generate(), SpanId.generate()),
                    parent_span_id=None,
                    name="s",
                    kind=SpanKind.INTERNAL,
                    start_time_ns=1,
                    end_time_ns=2,
                    output_data=b"x" * 8192,
                )
            )
        return client._dropped
    finally:
        wardex_sdk.close()


def _probe_zstd_level() -> bool:
    from test_codec import _env, _span
    from wardex_sdk.transport import _codec

    envelope = _env(_span(output_data=bytes(i % 251 for i in range(200_000))))
    fast = _codec.encode(envelope, limits=_native(zstd_level=1))
    dense = _codec.encode(envelope, limits=_native(zstd_level=19))
    return len(dense) < len(fast)


_PROBES = {
    "max_headers": _probe_max_headers,
    "max_body_bytes": _probe_max_body_bytes,
    "max_opaque_body_bytes": _probe_max_opaque_body_bytes,
    "max_stream_buffer_bytes": _probe_max_stream_buffer_bytes,
    "max_decoded_bytes": _probe_max_decoded_bytes,
    "max_streams": _probe_max_streams,
    "max_ws_frame_bytes": _probe_max_ws_frame_bytes,
    "ws_sample_bytes": _probe_ws_sample_bytes,
    "max_connections": _probe_max_connections,
    "max_sessions": _probe_max_sessions,
    "max_session_entries": _probe_max_session_entries,
    "mcp_sniff_bytes": _probe_mcp_sniff_bytes,
    "max_buffer_spans": _probe_max_buffer_spans,
    "max_buffer_bytes": _probe_max_buffer_bytes,
    "zstd_level": _probe_zstd_level,
}

_NOT_ENFORCED = {
    "replay_buffer_size": (
        "Reserved: nothing in the SDK reads it, so setting it changes nothing. "
        "Documented as inert in crates/wardex-limits and in the README's "
        "resource-limits section rather than left for a user to discover. "
        "Delete this entry and add a probe when a replay buffer lands."
    ),
}


def test_every_limit_is_either_probed_or_named_inert():
    """The mirror's field set must be exactly partitioned by the two tables.

    A limit added to the core and mirrored into CaptureLimits without any
    consumer would otherwise be advertised as configurable and silently do
    nothing -- which is how zstd_level and max_decoded_bytes shipped inert.
    """
    fields = set(CaptureLimits.__dataclass_fields__)
    covered = set(_PROBES) | set(_NOT_ENFORCED)
    assert fields == covered, (
        f"unclassified limits: {sorted(fields - covered)}; "
        f"stale entries: {sorted(covered - fields)}. Every field must have a "
        f"probe in _PROBES or a documented reason in _NOT_ENFORCED."
    )
    assert not (set(_PROBES) & set(_NOT_ENFORCED))


@pytest.mark.parametrize("name", sorted(_PROBES))
def test_limit_is_observably_enforced(name):
    """Each probe must show the override changing real behavior."""
    assert _PROBES[name](), (
        f"{name} is advertised as configurable but overriding it did not change "
        f"observable behavior -- it is either unwired or its enforcement site "
        f"reads a hardcoded value instead of the resolved limits."
    )


def test_http1_body_cap_is_visible_to_the_user():
    """A body the HTTP/1 parser capped must say so on the span.

    The cap is applied in Rust and the marker is attached to the parsed
    message there, but a user only ever reads CaptureIntegrity. Without the
    whole chain -- RawHttpMessage.limitations, ParsedMessage, _Txn, the seam --
    an opaque body would be silently sampled to max_opaque_body_bytes while
    HTTP/2 reported the identical truncation, leaving the two protocols
    disagreeing about whether capping is observable at all.
    """
    body = b"x" * 64
    request = b"GET /download HTTP/1.1\r\nHost: files.example\r\n\r\n"
    response = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )

    span = _drive_seam(CaptureLimits(max_opaque_body_bytes=16), request, response, "files.example")
    assert span is not None
    assert span.capture_integrity.truncated
    assert "body_cap_exceeded" in span.capture_integrity.limitations
    assert len(span.output_data) == 16

    # The same exchange under the default cap is neither truncated nor marked,
    # so the assertions above are about the cap and not about this traffic.
    span = _drive_seam(CaptureLimits(), request, response, "files.example")
    assert span is not None
    assert not span.capture_integrity.truncated
    assert "body_cap_exceeded" not in span.capture_integrity.limitations
    assert span.output_data == body


def test_http1_request_body_cap_is_visible_to_the_user():
    """The marker must survive the request half too: a capped upload is
    reported on the same transaction as a capped download, once."""
    body = b"x" * 64
    request = (
        b"POST /upload HTTP/1.1\r\nHost: files.example\r\n"
        b"Content-Type: application/octet-stream\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    response = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )

    span = _drive_seam(CaptureLimits(max_opaque_body_bytes=16), request, response, "files.example")
    assert span is not None
    assert span.capture_integrity.truncated
    # Both halves hit the cap; that is one limitation of the transaction.
    assert span.capture_integrity.limitations.count("body_cap_exceeded") == 1
    assert len(span.input_data) == 16
