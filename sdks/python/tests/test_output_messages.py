"""output.messages full-parts E2E — self-signed TLS mock (zero external dependencies)."""

from __future__ import annotations

import http.server
import json
import ssl
import threading
from pathlib import Path

import wardex_sdk as wardex
from wardex_sdk import ConsoleTransport, _hub
from wardex_sdk._assembly import Limitation
from wardex_sdk._enums import SpanKind

_FIXTURES = Path(__file__).parent / "fixtures"
CERT = _FIXTURES / "cert.pem"
KEY = _FIXTURES / "key.pem"

_TEXT_RESP = json.dumps(
    {
        "id": "chatcmpl-y",
        "model": "gpt-4o-mini",
        "choices": [
            {"message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
).encode()

_UNKNOWN_RESP = json.dumps(
    {
        "id": "msg_1",
        "model": "claude-3",
        "stop_reason": "end_turn",
        "content": [{"type": "some_future_block", "foo": 1}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
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


def _post(url: str, body: bytes, path: str = "/v1/chat/completions") -> None:
    import http.client

    host = url.split("://")[1]
    h, p = host.split(":")
    ctx = ssl._create_unverified_context()
    conn = http.client.HTTPSConnection(h, int(p), context=ctx)
    # server_hostname=127.0.0.1 → provider_from_host None → body-shape fallback
    # identifies the provider (the endpoint itself is the path's to name)
    conn.request("POST", path, body, {})
    conn.getresponse().read()
    conn.close()


def _spans():
    return [s for s in _hub.get_client()._spans if s.kind == SpanKind.CLIENT]


def test_text_response_emits_text_part_no_toolonly_marker():
    httpd, url = _server(_TEXT_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        _post(url, b'{"model":"gpt-4o-mini"}')
        sp = _spans()[0]
        msgs = json.loads(dict(sp.extra)["gen_ai.output.messages"])
        assert msgs[0]["parts"][0]["type"] == "text"
        assert msgs[0]["parts"][0]["content"] == "hello"
        # `output_messages_tool_calls_only` was retired when full-part mapping
        # landed, so it has no `Limitation` member to name here. Compare against
        # the member *values* — `"str" not in (Limitation...,)` is vacuously
        # true and would pin nothing.
        assert "output_messages_tool_calls_only" not in {
            lim.value for lim in sp.capture_integrity.limitations
        }
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()


def test_unknown_block_sets_unmapped_marker():
    httpd, url = _server(_UNKNOWN_RESP)
    try:
        wardex.init(transport=ConsoleTransport(), intercept=True)
        # An Anthropic-shaped exchange belongs on the Messages endpoint: the
        # endpoint table dispatches by (provider, api), so an Anthropic body
        # on the chat-completions path is no longer parsed as a chat call.
        _post(url, b'{"model":"claude-3"}', path="/v1/messages")
        sp = _spans()[0]
        assert "gen_ai.output.messages" in dict(sp.extra)
        assert Limitation.OUTPUT_MESSAGES_UNMAPPED_PART in sp.capture_integrity.limitations
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()
