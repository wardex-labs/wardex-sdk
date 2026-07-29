import asyncio
import asyncio.unix_events
import socket
import ssl
from pathlib import Path

import httpx
import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._enums import CaptureMode, SpanKind
from wardex_sdk.assembly import Limitation
from wardex_sdk.interceptors._base import InterceptorInterface
from wardex_sdk.interceptors._registry import InterceptorRegistry
from wardex_sdk.interceptors._ssl import SSLInterceptor


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk.interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


def _captured_spans():
    client = _hub.get_client()
    return list(client._spans)  # test-only internal access


class _FakeInterceptor(InterceptorInterface):
    def __init__(self) -> None:
        self.installs = 0
        self.uninstalls = 0

    def name(self) -> str:
        return "fake"

    def install(self, client) -> None:
        self.installs += 1

    def uninstall(self) -> None:
        self.uninstalls += 1


def test_registry_install_is_idempotent():
    reg = InterceptorRegistry()
    fake = _FakeInterceptor()
    reg.install(fake, client=None)
    reg.install(fake, client=None)  # the second call is ignored
    assert fake.installs == 1
    assert reg.is_installed("fake")


def test_registry_uninstall_all_restores():
    reg = InterceptorRegistry()
    fake = _FakeInterceptor()
    reg.install(fake, client=None)
    reg.uninstall_all()
    assert fake.uninstalls == 1
    assert not reg.is_installed("fake")


def _verify_ctx() -> ssl.SSLContext:
    cert = Path(__file__).parent / "fixtures" / "cert.pem"
    return ssl.create_default_context(cafile=str(cert))


def test_sync_https_call_is_captured(tls_server):
    # capture_mode=ALL: targets generic HTTP span assembly, not the AGENT-mode
    # policy gate; no active local span here for AGENT mode to latch onto.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
    resp = httpx.post(
        f"{tls_server}/v1/messages",
        headers={"Authorization": "Bearer sk-secret-xyz"},
        json={"model": "x"},
        verify=_verify_ctx(),
    )
    assert resp.status_code == 200

    spans = _captured_spans()
    client_spans = [s for s in spans if s.kind == SpanKind.CLIENT]
    assert len(client_spans) == 1
    sp = client_spans[0]
    assert sp.transport is not None
    assert sp.transport.http is not None
    assert sp.transport.http.method == "POST"
    assert sp.transport.http.status_code == 200
    assert b'"model"' in sp.input_data
    assert sp.output_data == b'{"ok":true}'
    # secret headers are not stored anywhere
    assert b"sk-secret-xyz" not in sp.input_data
    assert b"sk-secret-xyz" not in sp.output_data


@pytest.mark.asyncio
async def test_async_https_call_is_captured(tls_server):
    # capture_mode=ALL: same rationale as the sync variant above.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
    async with httpx.AsyncClient(verify=_verify_ctx()) as client:
        resp = await client.post(f"{tls_server}/v1/messages", json={"model": "y"})
    assert resp.status_code == 200

    client_spans = [s for s in _captured_spans() if s.kind == SpanKind.CLIENT]
    assert len(client_spans) == 1
    assert client_spans[0].transport.http.status_code == 200


def test_capture_nests_under_active_span(tls_server):
    wardex.init(intercept=True)

    @wardex.agent(name="researcher")
    def run() -> None:
        httpx.post(f"{tls_server}/v1/ping", json={}, verify=_verify_ctx())

    run()

    spans = _captured_spans()
    agent_spans = [s for s in spans if s.kind == SpanKind.INTERNAL]
    client_spans = [s for s in spans if s.kind == SpanKind.CLIENT]
    assert len(agent_spans) == 1
    assert len(client_spans) == 1
    # CLIENT span nests under the agent span (same trace, parent = agent span_id)
    assert client_spans[0].parent_span_id == agent_spans[0].context.span_id
    assert client_spans[0].context.trace_id == agent_spans[0].context.trace_id


def test_sync_capture_populates_connect_and_handshake(tls_server):
    # capture_mode=ALL: targets connect/handshake timing, not the policy gate.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
    resp = httpx.post(f"{tls_server}/v1/ping", json={}, verify=_verify_ctx())
    assert resp.status_code == 200

    sp = [s for s in _captured_spans() if s.kind == SpanKind.CLIENT][0]
    assert sp.transport.timing.tls_handshake_ms > 0.0
    assert sp.transport.timing.tcp_connect_ms >= 0.0
    assert sp.transport.connection_reused is False


def test_sync_keepalive_second_request_is_reused(tls_server):
    # capture_mode=ALL: targets connection-reuse tracking, not the policy gate.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
    with httpx.Client(verify=_verify_ctx()) as client:
        r1 = client.post(f"{tls_server}/v1/ping", json={})
        r2 = client.post(f"{tls_server}/v1/ping", json={})
    assert r1.status_code == r2.status_code == 200

    spans = [s for s in _captured_spans() if s.kind == SpanKind.CLIENT]
    assert len(spans) == 2
    first, second = spans[0], spans[1]
    assert first.transport.connection_reused is False
    assert first.transport.timing.tls_handshake_ms > 0.0
    assert second.transport.connection_reused is True
    assert second.transport.timing.tls_handshake_ms == 0.0
    assert second.transport.timing.tcp_connect_ms == 0.0


@pytest.mark.asyncio
async def test_async_capture_populates_handshake(tls_server):
    # capture_mode=ALL: targets async handshake timing, not the policy gate.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
    async with httpx.AsyncClient(verify=_verify_ctx()) as client:
        resp = await client.post(f"{tls_server}/v1/messages", json={"model": "y"})
    assert resp.status_code == 200

    sp = [s for s in _captured_spans() if s.kind == SpanKind.CLIENT][0]
    assert sp.transport.connection_reused is False
    assert sp.transport.timing.tls_handshake_ms > 0.0
    # On the anyio/httpx path, TCP connect can't be derived by subtraction → connect=0 + marker
    assert sp.transport.timing.tcp_connect_ms == 0.0
    # Census merge (design §6.5.1): `async_connect_unavailable` folded into
    # CONNECT_TIMING_UNAVAILABLE. Both said the same thing — tcp_connect_ms is
    # unknown rather than zero — and differed only in provenance.
    assert Limitation.CONNECT_TIMING_UNAVAILABLE in sp.capture_integrity.limitations


def test_install_uninstall_restores_originals():
    # Patched methods: 5 ssl (send/recv/recv_into/write/read)
    # + 5 connection-timing (connect/do_handshake x2/wrap_bio/create_connection)
    _bel = asyncio.base_events.BaseEventLoop
    originals = {
        (ssl.SSLSocket, "send"): ssl.SSLSocket.send,
        (ssl.SSLSocket, "recv"): ssl.SSLSocket.recv,
        (ssl.SSLSocket, "recv_into"): ssl.SSLSocket.recv_into,
        (ssl.SSLObject, "write"): ssl.SSLObject.write,
        (ssl.SSLObject, "read"): ssl.SSLObject.read,
        # connection-timing patches
        (socket.socket, "connect"): socket.socket.connect,
        (ssl.SSLSocket, "do_handshake"): ssl.SSLSocket.do_handshake,
        (ssl.SSLObject, "do_handshake"): ssl.SSLObject.do_handshake,
        (ssl.SSLContext, "wrap_bio"): ssl.SSLContext.wrap_bio,
        (_bel, "create_connection"): _bel.create_connection,
    }
    # The concrete loop class is patched separately too — only checked when
    # available on this platform
    if hasattr(asyncio.unix_events, "_UnixSelectorEventLoop"):
        originals[(asyncio.unix_events._UnixSelectorEventLoop, "create_connection")] = (
            asyncio.unix_events._UnixSelectorEventLoop.create_connection
        )
    wardex.init(intercept=True)
    # everything is patched
    for (cls, meth), orig in originals.items():
        assert getattr(cls, meth) is not orig

    from wardex_sdk.interceptors._registry import get_registry

    get_registry().uninstall_all()
    # everything is restored to the original
    for (cls, meth), orig in originals.items():
        assert getattr(cls, meth) is orig


class _DebugConfig:
    debug = True


class _DebugRecordingClient:
    """Minimal client stand-in exposing just what the seam's debug-logging
    branch needs (`config.debug` + `capture_span`) — no real Client/transport
    required for a unit test of the log itself."""

    def __init__(self) -> None:
        self.spans: list[object] = []
        self.config = _DebugConfig()

    def capture_span(self, span: object) -> None:
        self.spans.append(span)


def test_disabled_reason_logged_once_per_connection_in_debug(capsys):
    # Pure non-HTTP traffic (Redis/Mongo/Kafka-over-TLS) no longer reaches the
    # parser at all — the seam gate (this task) stops it first, before
    # anything is fed to the tracker. See
    # test_non_http_tls_traffic_produces_no_log below for that property.
    #
    # So reaching the parser's own disable-latch through the seam now requires
    # traffic that *passes* the gate: a real request line classifies the
    # connection "http", and only then does a response the parser refuses to
    # keep parsing latch it off. Here the response carries more headers than
    # the parser will track (`max_headers`, default 96) — real HTTP we
    # deliberately stop parsing rather than let grow unbounded, which is
    # exactly the "headers_exceeded" reason. No span is ever produced to
    # carry the reason (the whole point of the latch is that no message was
    # ever completed), so debug mode logs it instead — exactly once per
    # connection, not once per subsequent read that keeps arriving and keeps
    # being discarded by the already-latched parser.
    interceptor = SSLInterceptor()
    interceptor._client = _DebugRecordingClient()
    obj = object()  # no getpeername/selected_alpn_protocol → falls back in _peer/_select_tracker
    interceptor._on_request_bytes(obj, b"POST /v1/messages HTTP/1.1\r\nContent-Length: 0\r\n\r\n")
    too_many_headers = (
        b"HTTP/1.1 200 OK\r\n" + b"".join(f"X-{i}: v\r\n".encode() for i in range(100)) + b"\r\n"
    )

    interceptor._on_response_bytes(obj, too_many_headers)
    interceptor._on_response_bytes(obj, too_many_headers)
    interceptor._on_response_bytes(obj, too_many_headers)

    err = capsys.readouterr().err
    assert err.count("[wardex] parser disabled for") == 1
    assert "headers_exceeded" in err


def test_non_http_tls_traffic_produces_no_log(capsys):
    """Layering check: pure non-HTTP traffic is stopped by the seam gate
    before it ever reaches the parser, so the parser's own disable-latch
    (exercised above via headers_exceeded, on traffic that passes the gate)
    is never even reached here — no parser-level disabled_reason, and
    therefore no debug log either. This is the property the test above used
    to cover by accident, before the gate existed; now it is pinned directly."""
    interceptor = SSLInterceptor()
    interceptor._client = _DebugRecordingClient()
    obj = object()
    not_http = b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n"

    interceptor._on_response_bytes(obj, not_http)
    interceptor._on_response_bytes(obj, not_http)
    interceptor._on_response_bytes(obj, not_http)

    st = interceptor._conns[id(obj)]
    assert st.gate == "ignore"
    assert st.tracker.disabled_reason() is None  # the tracker was never fed
    assert capsys.readouterr().err == ""


def test_non_http_tls_traffic_is_not_parsed(fake_ssl_socket, bare_ssl_interceptor):
    """Redis over TLS must never reach the HTTP parser."""
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn=None)
    itc._on_request_bytes(sock, b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n")
    st = itc._conns[id(sock)]
    assert st.gate == "ignore"


def test_https_request_is_parsed(fake_ssl_socket, bare_ssl_interceptor):
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn="http/1.1")
    itc._on_request_bytes(sock, b"POST /v1/messages HTTP/1.1\r\nContent-Length: 0\r\n\r\n")
    assert itc._conns[id(sock)].gate == "http"


def test_alpn_h2_is_trusted(fake_ssl_socket, bare_ssl_interceptor):
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn="h2")
    itc._on_request_bytes(sock, b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n")
    assert itc._conns[id(sock)].gate == "h2"


def test_alpn_h2_is_trusted_without_the_preface(fake_ssl_socket, bare_ssl_interceptor):
    """ALPN must be trusted unconditionally, not just when the triggering call
    happens to carry the h2 connection preface.

    `send`/`write` can be called again after the preface has already gone out
    on the wire — the negotiated-ALPN check has to fire before the
    preface/method branches even run, or a healthy h2 connection whose first
    *observed* call is a plain data frame would be misclassified. Unlike
    test_alpn_h2_is_trusted (whose data independently satisfies the preface
    branch and so would pass even with the ALPN-first check deleted), this
    case only passes if ALPN is actually checked first.
    """
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn="h2")
    itc._on_request_bytes(sock, b"\x00\x00\x04\x01\x00\x00\x00\x00\x01arbitrary-h2-frame")
    assert itc._conns[id(sock)].gate == "h2"


def test_connect_proxy_is_recognised(fake_ssl_socket, bare_ssl_interceptor):
    """A proxied connection opens with CONNECT, which was missing from the list."""
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn=None)
    itc._on_request_bytes(sock, b"CONNECT api.anthropic.com:443 HTTP/1.1\r\n\r\n")
    assert itc._conns[id(sock)].gate == "http"


def test_server_first_protocol_is_ignored(fake_ssl_socket, bare_ssl_interceptor):
    """A response arriving before any request means we cannot classify it."""
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn=None)
    itc._on_response_bytes(sock, b"\x00\x00\x00\x08postgres-greeting")
    assert itc._conns[id(sock)].gate == "ignore"


def test_latch_stays_http_once_open(fake_ssl_socket, bare_ssl_interceptor):
    """Once classified "http", later non-HTTP-looking bytes must not flip the
    gate — the decision is made once, from the first request bytes, and never
    revisited."""
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn=None)
    itc._on_request_bytes(sock, b"POST /v1/messages HTTP/1.1\r\nContent-Length: 0\r\n\r\n")
    assert itc._conns[id(sock)].gate == "http"
    itc._on_request_bytes(sock, b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n")
    assert itc._conns[id(sock)].gate == "http"


def test_latch_stays_ignore_once_closed(fake_ssl_socket, bare_ssl_interceptor):
    """Once classified "ignore" (non-HTTP), later bytes that happen to look
    like an HTTP method must not re-arm the gate. This is the direction that
    matters for the OOM path this task closes: a Redis/Mongo/Kafka-over-TLS
    connection latched off must stay off for its whole life, or a coincidental
    later payload resembling a method line would let it start streaming into
    the parser again."""
    itc = bare_ssl_interceptor
    sock = fake_ssl_socket(alpn=None)
    itc._on_request_bytes(sock, b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n")
    assert itc._conns[id(sock)].gate == "ignore"
    itc._on_request_bytes(sock, b"GET / HTTP/1.1\r\n\r\n")
    assert itc._conns[id(sock)].gate == "ignore"


def test_disabled_reason_not_logged_without_debug(capsys):
    # The same latch, without debug=True, must stay silent.
    interceptor = SSLInterceptor()
    client = _DebugRecordingClient()
    client.config.debug = False
    interceptor._client = client
    obj = object()
    not_http = b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n"

    interceptor._on_response_bytes(obj, not_http)

    assert capsys.readouterr().err == ""
