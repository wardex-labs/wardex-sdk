"""A request's query string never reaches the wire.

Query strings are where credentials ride (`?api_key=`, `?sig=`, `?code=`).
The OTLP codec always cut them from `url.full`, but the span NAME carried the
raw request target, and so did the envelope's `http.url`: a tool that
authenticates by query string shipped its secret, in plain text, into the
span-name column of every backend. These tests make real requests carrying a
secret and search both encodings of what would be exported — the envelope
(decoded) and the OTLP bytes — for it, anywhere: name, attributes, events.
"""

from __future__ import annotations

import http.client
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

import wardex_sdk as wardex
from wardex_sdk import NoOpTransport, _hub, _wardex_native
from wardex_sdk._enums import CaptureMode
from wardex_sdk._interceptors._trackers import _without_query
from wardex_sdk._types import Envelope
from wardex_sdk.transport import _codec

SECRET = "SECRET123"

_CERT = Path(__file__).parent / "fixtures" / "cert.pem"


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk._interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


class _Ok(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def plain_server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Ok)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()


def _init(mode: CaptureMode | None) -> list[Envelope]:
    """Init with real interceptors; every envelope is kept, none is sent."""
    kept: list[Envelope] = []

    def keep(env: Envelope) -> None:
        kept.append(env)
        return None

    kw: dict = {
        "transport": NoOpTransport(),
        "before_send_envelope": keep,
        "backend": wardex.BackendConfig(api_key="k"),
    }
    if mode is not None:
        kw["capture_mode"] = mode
    wardex.init(**kw)
    return kept


def _exported(kept: list[Envelope]) -> tuple[list[str], int]:
    """Span names, and how often SECRET appears across both wire encodings."""
    wardex.flush()
    names = [s.name for env in kept for s in env.spans]
    hits = 0
    for env in kept:
        hits += repr(_codec.decode(_codec.encode(env))).count(SECRET)
        hits += _wardex_native.codec.encode_otlp_traces(env).count(SECRET.encode())
    return names, hits


def _get(port: int, target: str) -> None:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", target)
        conn.getresponse().read()
    finally:
        conn.close()


def test_a_call_inside_an_active_span_ships_no_query_value(plain_server):
    kept = _init(None)
    with wardex.span("agent-step"):
        _get(plain_server, f"/tool?api_key={SECRET}")
    names, hits = _exported(kept)
    assert "HTTP GET /tool" in names
    assert hits == 0


def test_capture_mode_all_ships_no_query_or_fragment(plain_server):
    kept = _init(CaptureMode.ALL)
    _get(plain_server, f"/tool?x=1&sig={SECRET}#{SECRET}")
    names, hits = _exported(kept)
    assert names == ["HTTP GET /tool"]
    assert hits == 0


def test_an_http2_path_ships_no_query_value(h2_server):
    import ssl

    kept = _init(CaptureMode.ALL)
    with httpx.Client(http2=True, verify=ssl.create_default_context(cafile=str(_CERT))) as c:
        resp = c.get(f"{h2_server}/v1/ping?code={SECRET}")
    assert resp.http_version == "HTTP/2"
    names, hits = _exported(kept)
    assert names == ["HTTP GET /v1/ping"]
    assert hits == 0


class _FakeSSLObj:
    """Only a `server_hostname`, which is all the seam reads for a host."""

    def __init__(self, host: str) -> None:
        self.server_hostname = host

    def selected_alpn_protocol(self) -> None:
        return None


def _frame(opcode: int, payload: bytes) -> bytes:
    return bytes([0x80 | opcode, len(payload)]) + payload


def test_a_websocket_upgrade_path_ships_no_query_value():
    kept = _init(CaptureMode.ALL)
    from wardex_sdk._interceptors._registry import get_registry

    seam = get_registry()._installed["ssl"]  # type: ignore[attr-defined]
    obj = _FakeSSLObj("chat.example.com")
    seam._on_request_bytes(
        obj,
        b"GET /socket?token=" + SECRET.encode() + b" HTTP/1.1\r\nHost: chat.example.com\r\n"
        b"Upgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    seam._on_response_bytes(
        obj,
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    seam._on_request_bytes(obj, _frame(0x1, b"hi"))
    seam._on_response_bytes(obj, _frame(0x8, (1000).to_bytes(2, "big")))
    names, hits = _exported(kept)
    assert names == ["WS /socket"]
    assert hits == 0


@pytest.mark.parametrize(
    ("target", "want"),
    [
        ("/tool?api_key=s", "/tool"),
        ("/tool#frag?x", "/tool"),
        ("/tool?x#y", "/tool"),
        ("/v1/messages", "/v1/messages"),
        ("?only=query", ""),
        ("http://proxy.example/p?q=s", "http://proxy.example/p"),
        ("", ""),
        (None, None),
    ],
)
def test_the_request_target_is_cut_at_the_first_query_or_fragment(target, want):
    assert _without_query(target) == want
