"""The byte seam's transport block says what it observed, and nothing else.

Fields the seam never filled used to ship under their zero values as if they
had been read off the connection: every span claimed `is_streaming=false`
(a streaming chat call included, beside its own `gen_ai.request.stream=true`),
a TEXT modality nobody detected, a final-chunk flag nobody tracked, and a
WebSocket session claimed 0 ms for a connect, a handshake, a first byte and a
first token it never timed. A receiver stores exactly what arrives, so it
stored those as facts.

This file drives real loopback traffic through the installed seam — a
streaming chat call (Server-Sent Events), a non-streaming one (JSON) and a
WebSocket session — encodes what was captured with the real encoder, and holds
the result to the contract in `_check`, in two halves:

* FRESH: the traffic above, captured now. Fails when a producer or the encoder
  changes what it says.
* COMMITTED: the same contract over
  `fixtures/transport_envelopes/observed_transport.envelope.zst`, the request
  body a receiver's test suite reads in place of a hand-built envelope, so it
  must keep telling the truth about what this SDK sends.

Regenerate the committed body (its bytes differ per run — ids, clocks, ports —
and are not compared; only the contract is):

    WARDEX_REGEN_ENVELOPES=1 uv run pytest sdks/python/tests/test_observed_transport.py
"""

from __future__ import annotations

import datetime
import http.client
import http.server
import json
import os
import platform
import socket
import threading
from importlib import metadata
from pathlib import Path

import wardex_sdk as wardex
from wardex_sdk._enums import CaptureMode
from wardex_sdk._types import Envelope
from wardex_sdk.testing import RecordingTransport
from wardex_sdk.transport import _codec

_DIR = Path(__file__).parent / "fixtures" / "transport_envelopes"
_BODY = _DIR / "observed_transport.envelope.zst"
_REGEN = os.environ.get("WARDEX_REGEN_ENVELOPES") == "1"

_SSE = (
    b'data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o",'
    b'"choices":[{"index":0,"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
    b'data: {"id":"c1","object":"chat.completion.chunk","model":"gpt-4o",'
    b'"choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
    b"data: [DONE]\n\n"
)
_JSON = json.dumps(
    {
        "id": "c2",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
).encode()


class _Provider(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 — the stdlib's name
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if body.get("stream"):
            payload, kind = _SSE, "text/event-stream"
        else:
            payload, kind = _JSON, "application/json"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a: object) -> None:
        pass


def _chat(port: int, *, stream: bool) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", port)  # plaintext: the raw socket seam
    body = json.dumps(
        {"model": "gpt-4o", "stream": stream, "messages": [{"role": "user", "content": "hi"}]}
    )
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    conn.getresponse().read()
    conn.close()


def _read_until(sock: socket.socket, marker: bytes) -> bytes:
    buf = b""
    while marker not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf


def _read_frame(sock: socket.socket) -> bytes:
    head = sock.recv(2)
    length = head[1] & 0x7F
    masked = head[1] & 0x80
    key = sock.recv(4) if masked else b""
    payload = b""
    while len(payload) < length:
        payload += sock.recv(length - len(payload))
    if masked:
        payload = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
    return payload


def _masked(opcode: int, payload: bytes) -> bytes:
    key = b"\x01\x02\x03\x04"
    body = bytes(b ^ key[i % 4] for i, b in enumerate(payload))
    return bytes([0x80 | opcode, 0x80 | len(payload)]) + key + body


def _ws_server() -> int:
    """One-session WebSocket echo server over a bare socket: upgrade, echo one
    text message, answer the client's close."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)

    def serve() -> None:
        conn, _ = srv.accept()
        with conn:
            _read_until(conn, b"\r\n\r\n")
            conn.sendall(
                b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
                b"Connection: Upgrade\r\nSec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n\r\n"
            )
            message = _read_frame(conn)
            echo = b"echo:" + message
            conn.sendall(bytes([0x81, len(echo)]) + echo)
            _read_frame(conn)  # the client's close
            conn.sendall(b"\x88\x02\x03\xe8")  # close, 1000
        srv.close()

    threading.Thread(target=serve, daemon=True).start()
    return srv.getsockname()[1]


def _ws_session(port: int) -> None:
    with socket.create_connection(("127.0.0.1", port)) as sock:
        sock.sendall(
            b"GET /realtime HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\n"
            b"Connection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
            b"Sec-WebSocket-Version: 13\r\n\r\n"
        )
        _read_until(sock, b"\r\n\r\n")
        sock.sendall(_masked(0x1, b"ping"))
        _read_frame(sock)
        sock.sendall(_masked(0x8, (1000).to_bytes(2, "big")))
        _read_frame(sock)


def _capture() -> bytes:
    """The three calls, captured by the installed seam, as one request body."""
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    ws_port = _ws_server()
    transport = RecordingTransport()
    try:
        # ALL: a WebSocket session carries no LLM semantics, so the default
        # mode would not ship it, and this file is about what it says.
        wardex.init(transport=transport, intercept=True, capture_mode=CaptureMode.ALL)
        _chat(httpd.server_address[1], stream=True)
        _chat(httpd.server_address[1], stream=False)
        _ws_session(ws_port)
    finally:
        wardex.close()
        httpd.shutdown()
        httpd.server_close()
    spans = tuple(s for env in transport.envelopes for s in env.spans)
    return _codec.encode(Envelope(header=transport.envelopes[0].header, spans=spans))


def _extra(span: dict) -> dict:
    return {kv["key"]: kv["value"] for kv in span.get("extra", [])}


def _check(body: bytes) -> None:
    """The contract. One function, so FRESH and COMMITTED cannot drift apart."""
    spans = [i["span"] for i in _codec.decode(body)["items"] if "span" in i]
    chats = [s for s in spans if s["name"] == "HTTP POST /v1/chat/completions"]
    streamed = [s for s in chats if _extra(s).get("gen_ai.request.stream") is True]
    plain = [s for s in chats if _extra(s).get("gen_ai.request.stream") is False]
    ws = [s for s in spans if s["name"] == "WS /realtime"]
    assert len(streamed) == len(plain) == len(ws) == 1, [s["name"] for s in spans]

    # The parser read each response body: one was an SSE stream, one was not.
    assert streamed[0]["transport"]["is_streaming"] is True
    assert plain[0]["transport"]["is_streaming"] is False
    for chat in chats:
        t = chat["transport"]
        # Plaintext: there is no TLS handshake to time.
        assert t["timing"]["tls_handshake_ms"] is None
        assert t["timing"]["ttfb_ms"] is not None
        assert t["timing"]["transfer_ms"] is not None
        assert isinstance(t["connection_reused"], bool)

    # A WebSocket session times its own length and nothing else, and reads no
    # response body as a whole.
    t = ws[0]["transport"]
    for interval in ("tcp_connect_ms", "tls_handshake_ms", "ttfb_ms", "ttft_ms"):
        assert t["timing"][interval] is None, interval
    assert t["timing"]["transfer_ms"] is not None
    assert t["is_streaming"] is None
    assert t["connection_reused"] is None

    # No producer names a modality.
    for span in spans:
        if "transport" in span:
            assert span["transport"]["request_modality"] is None
            assert span["transport"]["response_modality"] is None


def test_fresh_traffic_reports_only_what_was_observed():
    body = _capture()
    _check(body)
    if _REGEN:
        _DIR.mkdir(parents=True, exist_ok=True)
        _BODY.write_bytes(body)
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        (_DIR / "PROVENANCE.md").write_text(
            "# Transport envelope body\n\n"
            "A request body exactly as `WardexTransport` would POST it — a\n"
            "`wardex.v1.Envelope`, protobuf under zstd — produced by driving loopback\n"
            "traffic through the installed byte seam and encoding what it captured with\n"
            "the real encoder (masking off). It carries a streaming chat call (SSE), a\n"
            "non-streaming chat call (JSON) and a WebSocket session. Every value in it is\n"
            "synthetic: a loopback server, canned responses.\n\n"
            f"- Produced: {today}\n"
            f"- wardex-sdk: {metadata.version('wardex-sdk')}, Python {platform.python_version()}\n"
            "- Source: `sdks/python/tests/test_observed_transport.py`, which also holds the\n"
            "  contract the body is checked against on every run\n"
            "- Regenerate: `WARDEX_REGEN_ENVELOPES=1 uv run pytest "
            "sdks/python/tests/test_observed_transport.py`\n"
        )


def test_the_committed_body_still_holds_the_contract():
    assert _BODY.is_file(), (
        f"{_BODY.name} is missing; regenerate with "
        "WARDEX_REGEN_ENVELOPES=1 uv run pytest sdks/python/tests/test_observed_transport.py"
    )
    _check(_BODY.read_bytes())
