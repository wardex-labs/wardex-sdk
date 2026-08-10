"""LLM body parser integration tests — verify _emit_span gen_ai/output_data/marker."""

import gzip
import http.server
import json
import ssl
import threading
from pathlib import Path

import httpx
import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._enums import CaptureMode, SpanKind

_FIX = Path(__file__).parent / "fixtures"


def _make_server(payload: bytes, content_type: str = "application/json", gzip_body: bool = False):
    body = gzip.compress(payload) if gzip_body else payload

    class _H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            if gzip_body:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _H)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(_FIX / "cert.pem"), keyfile=str(_FIX / "key.pem"))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    host, port = httpd.socket.getsockname()[:2]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"https://{host}:{port}"


def _verify_ctx():
    return ssl.create_default_context(cafile=str(_FIX / "cert.pem"))


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk._interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


OPENAI_RESP = json.dumps(
    {
        "id": "chatcmpl-x",
        "model": "gpt-4o-mini",
        "choices": [{"finish_reason": "stop"}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3},
    }
).encode()


def _client_span():
    return [s for s in _hub.get_client()._spans if s.kind == SpanKind.CLIENT][0]


def test_openai_chat_semantics_filled():
    httpd, url = _make_server(OPENAI_RESP)
    try:
        wardex.init(intercept=True)
        httpx.post(
            f"{url}/v1/chat/completions", json={"model": "gpt-4o-mini"}, verify=_verify_ctx()
        )
    finally:
        httpd.shutdown()
    sp = _client_span()
    assert sp.gen_ai is not None
    assert sp.gen_ai.input_tokens == 12
    assert sp.gen_ai.output_tokens == 3
    assert sp.gen_ai.response_model == "gpt-4o-mini"
    assert sp.gen_ai.finish_reasons == ("stop",)


def test_gzip_response_decoded_into_output_data():
    httpd, url = _make_server(OPENAI_RESP, gzip_body=True)
    try:
        wardex.init(intercept=True)
        httpx.post(
            f"{url}/v1/chat/completions", json={"model": "gpt-4o-mini"}, verify=_verify_ctx()
        )
    finally:
        httpd.shutdown()
    sp = _client_span()
    assert sp.gen_ai is not None and sp.gen_ai.input_tokens == 12
    assert sp.output_data == OPENAI_RESP  # stored decompressed (readable)
    assert b'"usage"' in sp.output_data


def test_non_llm_call_no_semantics_no_marker():
    httpd, url = _make_server(b'{"ok":true}')
    try:
        # capture_mode=ALL: this test targets the body parser (no gen_ai marker
        # on non-LLM bodies), not the capture-policy gate, and there is no
        # active local span for the AGENT-mode default to latch onto.
        wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
        httpx.post(f"{url}/v1/widgets", json={}, verify=_verify_ctx())
    finally:
        httpd.shutdown()
    sp = _client_span()
    assert sp.gen_ai is None
    assert "semantic_parse_failed" not in sp.capture_integrity.limitations


def test_unrecognized_host_broken_body_no_gen_ai():
    # unrecognized host (127.0.0.1) + body-shape also doesn't match -> provider
    # unidentified -> gen_ai None, no marker.
    httpd, url = _make_server(b"not-json-at-all")
    try:
        # capture_mode=ALL: same rationale — testing gen_ai==None on an
        # unidentifiable body, not the AGENT-mode policy gate.
        wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
        httpx.post(f"{url}/v1/chat/completions", json={}, verify=_verify_ctx())
    finally:
        httpd.shutdown()
    sp = _client_span()
    assert sp.gen_ai is None  # unidentifiable → no semantics
