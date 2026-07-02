"""Input messages (gen_ai.input.messages / gen_ai.system_instructions) E2E
— self-signed TLS mock."""

from __future__ import annotations

import http.server
import json
import ssl
import threading
from pathlib import Path

import wardex_sdk as wardex
from wardex_sdk import ConsoleTransport, _hub
from wardex_sdk._enums import SpanKind

_FIXTURES = Path(__file__).parent / "fixtures"
CERT = _FIXTURES / "cert.pem"
KEY = _FIXTURES / "key.pem"

# minimal response for provider identification (choices shape → OpenAI body-shape fallback)
_MINIMAL_RESP = json.dumps(
    {"id": "c1", "model": "gpt-4o", "choices": [{"finish_reason": "stop"}]}
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
    # server_hostname=127.0.0.1 -> provider_from_host None -> body-shape fallback
    # identifies the provider
    conn.request("POST", "/v1/chat/completions", body, {})
    conn.getresponse().read()
    conn.close()


def _spans():
    return [s for s in _hub.get_client()._spans if s.kind == SpanKind.CLIENT]


def test_input_messages_and_system_instructions_captured():
    """system → gen_ai.system_instructions, user/assistant/tool → gen_ai.input.messages."""
    httpd, url = _server(_MINIMAL_RESP)
    req_body = json.dumps(
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are a weather assistant."},
                {"role": "user", "content": "Seoul weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Seoul"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "18C, sunny"},
            ],
        }
    ).encode()
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(url, req_body)
        sp = _spans()[0]
        extra = dict(sp.extra)
        # verify system_instructions
        si = json.loads(extra["gen_ai.system_instructions"])
        assert si[0]["type"] == "text"
        assert si[0]["content"] == "You are a weather assistant."
        # verify input.messages: user/assistant/tool (system excluded)
        im = json.loads(extra["gen_ai.input.messages"])
        assert [m["role"] for m in im] == ["user", "assistant", "tool"]
        # tool result message → ToolCallResponsePart
        assert im[2]["parts"][0]["type"] == "tool_call_response"
        assert im[2]["parts"][0]["response"] == "18C, sunny"
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_input_media_data_uri_inline_blob():
    """image_url data URI → converted to BlobPart(type=blob)."""
    httpd, url = _server(_MINIMAL_RESP)
    req_body = json.dumps(
        {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                        }
                    ],
                }
            ],
        }
    ).encode()
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(url, req_body)
        sp = _spans()[0]
        im = json.loads(dict(sp.extra)["gen_ai.input.messages"])
        p = im[0]["parts"][0]
        assert p["type"] == "blob"
        assert p["content"] == "iVBORw0KGgo="
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_unknown_input_block_emits_marker():
    """Unrecognized content block → input_messages_unmapped_part marker."""
    httpd, url = _server(_MINIMAL_RESP)
    req_body = json.dumps(
        {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "future_thing", "x": 1}],
                }
            ],
        }
    ).encode()
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(url, req_body)
        sp = _spans()[0]
        assert "input_messages_unmapped_part" in sp.capture_integrity.limitations
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()
