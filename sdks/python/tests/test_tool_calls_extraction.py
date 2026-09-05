"""tool_call structured extraction E2E.

Self-signed TLS mock server (zero external dependencies).
"""

from __future__ import annotations

import http.server
import json
import ssl
import threading
import time as _time
from pathlib import Path

import wardex_sdk as wardex
from conftest import client_spans
from wardex_sdk import ConsoleTransport
from wardex_sdk._assembly import Limitation

_FIXTURES = Path(__file__).parent / "fixtures"
CERT = _FIXTURES / "cert.pem"
KEY = _FIXTURES / "key.pem"

_TOOL_RESP = json.dumps(
    {
        "id": "chatcmpl-x",
        "model": "gpt-4o-mini",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "add", "arguments": '{"a": 17, "b": 25}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
).encode()

_TEXT_RESP = json.dumps(
    {
        "id": "chatcmpl-y",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
).encode()


def _server(payload: bytes):
    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
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


def _post(url: str, body: bytes) -> None:
    import http.client

    host = url.split("://")[1]
    h, p = host.split(":")
    ctx = ssl._create_unverified_context()
    conn = http.client.HTTPSConnection(h, int(p), context=ctx)
    # server_hostname=127.0.0.1 → provider_from_host is None → body-shape fallback
    # identifies the provider
    conn.request("POST", "/v1/chat/completions", body, {})
    conn.getresponse().read()
    conn.close()


def test_tool_call_extracted_to_output_messages():
    httpd, url = _server(_TOOL_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(url, b'{"model":"gpt-4o-mini"}')
        spans = client_spans()
        assert len(spans) == 1
        extra = dict(spans[0].extra)
        assert "gen_ai.output.messages" in extra
        msgs = json.loads(extra["gen_ai.output.messages"])
        assert msgs[0]["parts"][0]["name"] == "add"
        assert msgs[0]["parts"][0]["arguments"]["a"] == 17
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_text_response_has_output_messages_with_text_part():
    # A text-only response is still reported: it appears in output_messages as a TextPart.
    httpd, url = _server(_TEXT_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(url, b'{"model":"gpt-4o-mini"}')
        spans = client_spans()
        assert len(spans) == 1
        extra = dict(spans[0].extra)
        assert "gen_ai.output.messages" in extra
        msgs = json.loads(extra["gen_ai.output.messages"])
        assert msgs[0]["parts"][0]["type"] == "text"
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


# --- Fix 1: tool_calls_parse_failed marker ---

# finish_reason=tool_calls but message has no tool_calls array → parse-failure case
_TOOL_CALLS_MISSING_RESP = json.dumps(
    {
        "id": "chatcmpl-z",
        "model": "gpt-4o-mini",
        "choices": [
            {"message": {"role": "assistant", "content": None}, "finish_reason": "tool_calls"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
).encode()


def test_tool_calls_missing_emits_empty_parts_not_failed_marker():
    """finish_reason==tool_calls but there's no tool_calls array → build_output_messages
    assembles a finish_reason-only message, so sem.output_messages becomes Some.
    As a result the else branch never fires; instead, output_messages with empty
    parts is included in extra.

    The branch this pins was written up as `tool_calls_parse_failed`, a string the
    closed vocabulary never adopted (see `_NOT_ADOPTED` in
    test_limitation_census.py — it has zero emitters). The marker the else branch
    actually adds is `Limitation.SEMANTIC_PARSE_FAILED`, so that is what the
    absence assertion names."""
    httpd, url = _server(_TOOL_CALLS_MISSING_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(url, b'{"model":"gpt-4o-mini"}')
        spans = client_spans()
        assert len(spans) == 1
        extra = dict(spans[0].extra)
        # output_messages exists with empty parts (message is assembled even
        # without a tool_calls array)
        assert "gen_ai.output.messages" in extra
        msgs = json.loads(extra["gen_ai.output.messages"])
        assert msgs[0]["parts"] == []
        # the else branch only fires when output_messages=None → so it doesn't fire here
        assert Limitation.SEMANTIC_PARSE_FAILED not in spans[0].capture_integrity.limitations
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


# --- Fix 2: SSE delta.tool_calls reassembly E2E ---

# OpenAI SSE tool_calls delta chunks (same format as the Rust test
# openai_sse_reassembles_tool_call_deltas)
_SSE_TC_CHUNK_1 = (
    b'data: {"id":"chatcmpl-s","model":"gpt-4o-mini","choices":[{"delta":'
    b'{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"add",'
    b'"arguments":"{\\"a\\":1"}}]}}]}\n\n'
)
_SSE_TC_CHUNK_2 = (
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":'
    b'{"arguments":"7}"}}]},"finish_reason":"tool_calls"}]}\n\n'
)
_SSE_TC_DONE = b"data: [DONE]\n\n"


def _sse_tool_server():
    """Self-signed TLS mock server that streams tool_calls deltas as chunked SSE."""

    class _SseTcHandler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.flush()
            _time.sleep(0.01)
            for chunk in (_SSE_TC_CHUNK_1, _SSE_TC_CHUNK_2, _SSE_TC_DONE):
                self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def log_message(self, *a: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _SseTcHandler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(CERT), keyfile=str(KEY))
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
    host, port = httpd.socket.getsockname()[:2]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"https://{host}:{port}"


def test_sse_tool_calls_reassembled():
    """SSE delta.tool_calls chunks → verify output_messages + marker after reassembly."""
    httpd, url = _sse_tool_server()
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(url, b'{"model":"gpt-4o-mini","stream":true}')
        spans = client_spans()
        assert len(spans) == 1
        extra = dict(spans[0].extra)
        assert "gen_ai.output.messages" in extra
        msgs = json.loads(extra["gen_ai.output.messages"])
        assert msgs[0]["parts"][0]["name"] == "add"
        lims = spans[0].capture_integrity.limitations
        assert Limitation.REASSEMBLED_FROM_STREAM in lims
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()
