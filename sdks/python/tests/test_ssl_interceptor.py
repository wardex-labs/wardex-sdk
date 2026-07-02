import asyncio
import asyncio.unix_events
import socket
import ssl
from pathlib import Path

import httpx
import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._enums import SpanKind
from wardex_sdk.interceptors._base import InterceptorInterface
from wardex_sdk.interceptors._registry import InterceptorRegistry


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
    wardex.init(intercept=True)
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
    wardex.init(intercept=True)
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
    wardex.init(intercept=True)
    resp = httpx.post(f"{tls_server}/v1/ping", json={}, verify=_verify_ctx())
    assert resp.status_code == 200

    sp = [s for s in _captured_spans() if s.kind == SpanKind.CLIENT][0]
    assert sp.transport.timing.tls_handshake_ms > 0.0
    assert sp.transport.timing.tcp_connect_ms >= 0.0
    assert sp.transport.connection_reused is False


def test_sync_keepalive_second_request_is_reused(tls_server):
    wardex.init(intercept=True)
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
    wardex.init(intercept=True)
    async with httpx.AsyncClient(verify=_verify_ctx()) as client:
        resp = await client.post(f"{tls_server}/v1/messages", json={"model": "y"})
    assert resp.status_code == 200

    sp = [s for s in _captured_spans() if s.kind == SpanKind.CLIENT][0]
    assert sp.transport.connection_reused is False
    assert sp.transport.timing.tls_handshake_ms > 0.0
    # On the anyio/httpx path, TCP connect can't be derived by subtraction → connect=0 + marker
    assert sp.transport.timing.tcp_connect_ms == 0.0
    assert "async_connect_unavailable" in sp.capture_integrity.limitations


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
