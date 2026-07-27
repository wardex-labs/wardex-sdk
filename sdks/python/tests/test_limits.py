"""Native limits surface: defaults, construction, and parser wiring."""

from __future__ import annotations

import tracemalloc

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

    Uses max_headers rather than max_body_bytes/max_opaque_body_bytes: those two
    fields are declared on Limits but are not yet enforced anywhere on the HTTP/1
    path (crates/wardex-protocol/src/http1.rs only wires max_headers today — see
    Task 2's brief; max_body_bytes truncation exists only in http2.rs, and its own
    comment documents the opaque-body-aware cap as a "later change", not yet
    landed). A truncation-based assertion on HTTP/1 would pass or fail identically
    whether or not the config value actually reached the tracker, which is exactly
    the kind of non-discriminating test the task's self-review explicitly warns
    against. max_headers, by contrast, is genuinely enforced by the native
    parser (crates/wardex-protocol/src/http1.rs::try_parse_one), so this test
    still proves the value travels from CaptureLimits through _Http1Tracker into
    the native Http1Parser.
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
# max_body_bytes/max_opaque_body_bytes were considered and rejected: neither
# is wired into the HTTP/1 path yet (see the docstring on
# test_body_cap_reaches_the_parser_end_to_end), so a test built on either
# would pass or fail identically whether or not the value actually reached
# anything -- exactly the non-discriminating shape this file's tests avoid.


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
