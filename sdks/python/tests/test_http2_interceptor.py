from pathlib import Path

import httpx
import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._enums import CaptureMode, SpanKind

_CERT = Path(__file__).parent / "fixtures" / "cert.pem"


def _verify_ctx():
    import ssl

    return ssl.create_default_context(cafile=str(_CERT))


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk._interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


def _client_spans():
    return [s for s in _hub.get_client()._spans if s.kind == SpanKind.CLIENT]


def test_h2_call_is_captured(h2_server):
    # capture_mode=ALL: this test targets HTTP/2 span assembly (method/status/
    # body/headers-stripped), not the AGENT-mode policy gate, and there is no
    # active local span for AGENT mode to latch onto.
    wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
    with httpx.Client(http2=True, verify=_verify_ctx()) as client:
        resp = client.post(
            f"{h2_server}/v1/messages",
            headers={"Authorization": "Bearer sk-secret-xyz"},
            json={"model": "x"},
        )
    assert resp.status_code == 200
    assert resp.http_version == "HTTP/2"

    spans = _client_spans()
    assert len(spans) == 1
    sp = spans[0]
    assert sp.transport.http.method == "POST"
    assert sp.transport.http.status_code == 200
    assert b'"model"' in sp.input_data
    assert sp.output_data == b'{"ok":true}'
    assert b"sk-secret-xyz" not in sp.input_data
    assert b"sk-secret-xyz" not in sp.output_data
    assert ("network.protocol.version", "2") in sp.extra


def test_h2_capture_nests_under_active_span(h2_server):
    wardex.init(intercept=True)

    @wardex.agent(name="researcher")
    def run() -> None:
        with httpx.Client(http2=True, verify=_verify_ctx()) as client:
            client.get(f"{h2_server}/v1/ping")

    run()

    spans = _hub.get_client()._spans
    agent_spans = [s for s in spans if s.kind == SpanKind.INTERNAL]
    client_spans = [s for s in spans if s.kind == SpanKind.CLIENT]
    assert len(agent_spans) == 1
    assert len(client_spans) == 1
    assert client_spans[0].parent_span_id == agent_spans[0].context.span_id
    assert client_spans[0].context.trace_id == agent_spans[0].context.trace_id
