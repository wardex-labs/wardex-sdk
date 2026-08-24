import ssl
from pathlib import Path

import httpx
import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._enums import CaptureMode, SpanKind

uvloop = pytest.importorskip("uvloop")


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk._interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


def _verify_ctx() -> ssl.SSLContext:
    cert = Path(__file__).parent / "fixtures" / "cert.pem"
    return ssl.create_default_context(cafile=str(cert))


def test_uvloop_async_capture_populates_handshake(tls_server):
    # capture_mode=ALL: this test targets connect/handshake timing, not the
    # AGENT-mode policy gate, and the server response has no LLM semantics.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)

    async def call() -> int:
        async with httpx.AsyncClient(verify=_verify_ctx()) as client:
            r = await client.post(f"{tls_server}/v1/messages", json={"model": "z"})
            return r.status_code

    code = uvloop.run(call())
    assert code == 200

    client = _hub.get_client()
    client._settle()  # finalization runs on the worker; settle before reading
    sp = [s for s in client._spans if s.kind == SpanKind.CLIENT][0]
    assert sp.transport.connection_reused is False
    assert sp.transport.timing.tls_handshake_ms > 0.0
