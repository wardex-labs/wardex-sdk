"""The Responses API through the PUBLIC seam surface.

Every test drives real bytes through `wardex.init(intercept=True)` and a
local TLS server, because the defect class this file pins was invisible to
parser-level tests: the parser returned SOMETHING for a Responses stream —
a fabricated chat body — and only the capture policy and the seam markers
showed the lie (a dropped span under the default mode, a false
`sse_unknown_provider` on streams).
"""

from __future__ import annotations

import http.server
import json
import pathlib
import ssl
import threading

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._assembly import Limitation
from wardex_sdk._enums import CaptureMode, OperationName, SpanKind
from wardex_sdk.transport import ConsoleTransport

_REPO = pathlib.Path(__file__).resolve().parents[3]
_FIXTURES = _REPO / "crates" / "wardex-protocol" / "tests" / "fixtures" / "llm"
_TLS_FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CERT = _TLS_FIXTURES / "cert.pem"
KEY = _TLS_FIXTURES / "key.pem"


def _fixture(case: str, name: str) -> bytes:
    return (_FIXTURES / case / name).read_bytes()


def _server(payloads: list[tuple[int, bytes]]):
    """A TLS server answering one queued (status, JSON body) per request."""
    queue = list(payloads)

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            status, payload = queue.pop(0)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(CERT), keyfile=str(KEY))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    host, port = httpd.socket.getsockname()[:2]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"https://{host}:{port}"


def _post(url: str, body: bytes, path: str = "/v1/responses") -> int:
    import http.client

    host = url.split("://")[1]
    h, p = host.split(":")
    conn = http.client.HTTPSConnection(h, int(p), context=ssl._create_unverified_context())
    conn.request("POST", path, body, {})
    resp = conn.getresponse()
    resp.read()  # the seam sees the response only as the client consumes it
    conn.close()
    return resp.status


def _client_spans():
    return [s for s in _hub.get_client()._spans if s.kind == SpanKind.CLIENT]


def test_responses_call_is_captured_under_default_agent_mode():
    """T-P1 — the openai-agents default path exists on the wire now: no local
    span, default AGENT mode, one POST /v1/responses -> one CLIENT span with
    chat identity, tokens, and the api-type extra. (It used to be DROPPED:
    no semantics meant no agent_semantic, and the policy gated it out.)
    """
    httpd, url = _server([(200, _fixture("openai_responses", "response.json"))])
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(url, _fixture("openai_responses", "request.json"))
        spans = _client_spans()
        assert len(spans) == 1
        sp = spans[0]
        assert sp.gen_ai is not None
        assert sp.gen_ai.operation == OperationName.CHAT
        assert sp.gen_ai.input_tokens == 52
        assert sp.gen_ai.output_tokens == 17
        assert sp.gen_ai.response_status == "completed"
        assert dict(sp.extra)["openai.api.type"] == "responses"
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_responses_tool_call_correlates_across_two_turns():
    """T-P2 — the correlation contract: turn 1's tool_call part carries the
    call_id that turn 2's tool_call_response quotes, and turn 2 names turn
    1's response id as its previous_response.id.
    """
    turn2_request = json.dumps(
        {
            "model": "gpt-4.1",
            "previous_response_id": "resp_fx2",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_weather_2",
                    "output": "sunny in Busan",
                }
            ],
        }
    ).encode()
    httpd, url = _server(
        [
            (200, _fixture("openai_responses_tools", "response.json")),
            (200, _fixture("openai_responses", "response.json")),
        ]
    )
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(url, _fixture("openai_responses_tools", "request.json"))
        _post(url, turn2_request)
        first, second = _client_spans()

        out = json.loads(dict(first.extra)["gen_ai.output.messages"])
        tool_calls = [p for m in out for p in m["parts"] if p["type"] == "tool_call"]
        assert tool_calls[0]["id"] == "call_weather_2"
        assert first.gen_ai.response_id == "resp_fx2"

        ins = json.loads(dict(second.extra)["gen_ai.input.messages"])
        responses = [p for m in ins for p in m["parts"] if p["type"] == "tool_call_response"]
        assert responses[0]["id"] == "call_weather_2"
        assert second.gen_ai.previous_response_id == first.gen_ai.response_id
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_responses_stream_ships_snapshot_usage_and_markers(responses_sse_tls_server):
    """T-P3 — a Responses stream is a reassembled, identified stream: the
    reassembly marker rides it, the usage marker does NOT (the terminal
    snapshot carried usage), and the false `sse_unknown_provider` is gone —
    as is the fabricated chat body it used to decorate.
    """
    import httpx

    wardex.init(transport=ConsoleTransport(), intercept=True)
    try:
        resp = httpx.post(
            f"{responses_sse_tls_server}/v1/responses",
            json={"model": "gpt-4.1", "input": "Say hello.", "stream": True},
            verify=ssl._create_unverified_context(),
        )
        assert resp.status_code == 200
        sp = _client_spans()[0]
        markers = sp.capture_integrity.limitations
        assert Limitation.REASSEMBLED_FROM_STREAM in markers
        assert Limitation.STREAM_USAGE_UNAVAILABLE not in markers
        assert Limitation.SSE_UNKNOWN_PROVIDER not in markers
        body = json.loads(sp.output_data)
        assert "output" in body and "choices" not in body
        assert sp.gen_ai.input_tokens == 9
    finally:
        wardex.close()


def test_responses_error_response_is_a_chat_span_with_request_identity(fake_ssl_socket):
    """T-P4 — a refused call is still an LLM call: the 429 ships as a chat
    span with ERROR status, the status code as error.type, and the request's
    model as its identity. Driven at the seam with the provider's own host —
    an error envelope names no provider by shape, so the identity here is the
    request's, exactly the widening this pin protects. AGENT mode, no local
    span: the ERROR admission is the gate under test.
    """
    from wardex_sdk import LimitsConfig
    from wardex_sdk._enums import CaptureMode as _CM
    from wardex_sdk._interceptors._ssl import SSLInterceptor

    class _Config:
        debug = False
        limits = LimitsConfig()
        capture_mode = _CM.AGENT

    class _Client:
        config = _Config()

        def __init__(self) -> None:
            self.spans: list = []

        def capture_span(self, span) -> None:
            self.spans.append(span)

    client = _Client()
    itc = SSLInterceptor()
    itc._client = client
    itc._load_limits(client)
    sock = fake_ssl_socket()
    sock.server_hostname = "api.openai.com"

    request_body = _fixture("openai_responses", "request.json")
    error_body = json.dumps(
        {"error": {"type": "rate_limit_exceeded", "message": "slow down"}}
    ).encode()
    itc._on_request_bytes(
        sock,
        b"POST /v1/responses HTTP/1.1\r\nHost: api.openai.com\r\n"
        b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(request_body)}\r\n\r\n".encode()
        + request_body,
    )
    itc._on_response_bytes(
        sock,
        b"HTTP/1.1 429 Too Many Requests\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(error_body)}\r\n\r\n".encode()
        + error_body,
    )
    assert len(client.spans) == 1
    sp = client.spans[0]
    assert sp.status.value == "error"
    assert sp.error_type == "429"
    assert sp.gen_ai is not None
    assert sp.gen_ai.request_model == "gpt-4.1"


def test_count_tokens_is_not_a_chat_call():
    """T-P8 — /v1/messages/count_tokens is token arithmetic, not a chat call:
    no gen_ai block and, critically, no false `semantic_parse_failed` on a
    call the parser correctly declined to parse.
    """
    httpd, url = _server([(200, b'{"input_tokens": 14}')])
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True, capture_mode=CaptureMode.ALL)
        _post(url, b'{"model":"claude-sonnet-4-6","messages":[]}', "/v1/messages/count_tokens")
        sp = _client_spans()[0]
        assert sp.gen_ai is None
        assert Limitation.SEMANTIC_PARSE_FAILED not in sp.capture_integrity.limitations
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()
