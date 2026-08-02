import ssl
from pathlib import Path

import httpx
import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._enums import SpanKind
from wardex_sdk.assembly import Limitation


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk.interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


def _verify_ctx() -> ssl.SSLContext:
    cert = Path(__file__).parent / "fixtures" / "cert.pem"
    return ssl.create_default_context(cafile=str(cert))


def _client_span():
    spans = list(_hub.get_client()._spans)
    return [s for s in spans if s.kind == SpanKind.CLIENT][0]


def test_sse_stream_extracts_gen_ai_and_synthetic_body(sse_tls_server):
    wardex.init(intercept=True)
    resp = httpx.post(
        f"{sse_tls_server}/v1/chat/completions",
        json={"model": "gpt-4o-mini", "stream": True},
        verify=_verify_ctx(),
    )
    assert resp.status_code == 200

    sp = _client_span()
    assert sp.gen_ai is not None
    assert sp.gen_ai.response_model == "gpt-4o-mini"
    assert sp.gen_ai.finish_reasons == ("stop",)
    # text reassembled into synthetic JSON
    assert b'"content":"Hi!"' in sp.output_data
    # markers
    lims = sp.capture_integrity.limitations
    assert Limitation.REASSEMBLED_FROM_STREAM in lims
    # OpenAI's default stream has no usage
    assert Limitation.STREAM_USAGE_UNAVAILABLE in lims


def test_sse_stream_measures_ttft(sse_tls_server):
    wardex.init(intercept=True)
    httpx.post(
        f"{sse_tls_server}/v1/chat/completions",
        json={"model": "gpt-4o-mini", "stream": True},
        verify=_verify_ctx(),
    )
    sp = _client_span()
    # server delays 20ms after headers → ttft > 0
    assert sp.transport.timing.ttft_ms > 0.0
