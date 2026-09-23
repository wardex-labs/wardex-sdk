"""Every documented credential argument is masked, and every debugging argument
survives, on every transport an agent can use to send it.

The two corpora in `fixtures/` are the contract: a credential argument named
the way a public API documents it, and an argument a debugger needs to read.
Each entry is sent for real — over a loopback socket, through the installed
interceptors — on seven carriers, then encoded by a transport holding the
policy `init()` installed, on both wires. A secret value found in either wire's
bytes fails; a debugging value missing from either fails; a span name carrying
a query fails.

`code`, `sig` and `key` are credentials only in the `name=value` shape (OAuth,
Azure SAS, Google API keys send them there). In JSON the same names are an
error code, a code interpreter's source, a map entry's key — so on the JSON
carriers this file asserts they are KEPT, which is the documented rule rather
than an exemption from it.
"""

from __future__ import annotations

import http.client
import json
import socket
import threading
import urllib.parse
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import h2.config
import h2.connection
import h2.events
import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub, _wardex_native
from wardex_sdk._enums import CaptureMode
from wardex_sdk.transport import Transport

_FIXTURES = Path(__file__).parent / "fixtures"
SECRETS: list[dict[str, Any]] = json.loads((_FIXTURES / "secret_argument_corpus.json").read_text())[
    "entries"
]
BENIGN: list[dict[str, Any]] = json.loads((_FIXTURES / "benign_argument_corpus.json").read_text())[
    "entries"
]

_JSON_CARRIERS = {"json_top", "json_nested", "tool_args"}


def test_the_corpora_are_the_size_the_contract_names():
    assert len(SECRETS) >= 40
    assert len(BENIGN) >= 40
    values = [e["value"] for e in SECRETS + BENIGN]
    assert len(values) == len(set(values)), "each value must be findable on its own"
    for e in SECRETS:
        assert e["source"].startswith("https://"), e["name"]
    # A short value (`3`, `ko`) is found in any export — in a timestamp, an id
    # — so its presence would prove nothing. Each debugging value carries a
    # unique tail, which is what makes "it survived" a measurement.
    for e in BENIGN:
        assert "-wdxb" in e["value"], e["name"]


# --- loopback servers --------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _reply(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        if self.path.startswith("/v1/chat/completions"):
            args = urllib.parse.unquote(self.headers.get("X-Test-Tool-Args") or "{}")
            body = json.dumps(
                {
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "lookup", "arguments": args},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
                }
            ).encode()
        else:
            body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _reply
    do_POST = _reply

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture(scope="module")
def http_port() -> Iterator[int]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture(scope="module")
def h2_port() -> Iterator[int]:
    """A prior-knowledge cleartext HTTP/2 server answering every stream."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    srv.settimeout(0.5)
    stop = threading.Event()

    def handle(sock: socket.socket) -> None:
        conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=False))
        conn.initiate_connection()
        sock.sendall(conn.data_to_send())
        while not stop.is_set():
            data = sock.recv(65535)
            if not data:
                return
            for event in conn.receive_data(data):
                if isinstance(event, h2.events.StreamEnded):
                    conn.send_headers(
                        event.stream_id, [(":status", "200"), ("content-type", "application/json")]
                    )
                    conn.send_data(event.stream_id, b'{"ok":true}', end_stream=True)
            out = conn.data_to_send()
            if out:
                sock.sendall(out)

    def serve() -> None:
        while not stop.is_set():
            try:
                client, _ = srv.accept()
            except OSError:
                continue
            try:
                handle(client)
            except Exception:  # noqa: BLE001 — a test server; the client side asserts
                pass
            finally:
                client.close()

    threading.Thread(target=serve, daemon=True).start()
    yield srv.getsockname()[1]
    stop.set()
    srv.close()


# --- carriers ----------------------------------------------------------------


def _h1(port: int, method: str, target: str, body: bytes = b"", headers=None) -> None:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        c.request(method, target, body or None, headers or {})
        c.getresponse().read()
    finally:
        c.close()


def _h2_get(port: int, path: str) -> None:
    s = socket.create_connection(("127.0.0.1", port))
    s.settimeout(2.0)
    try:
        conn = h2.connection.H2Connection(h2.config.H2Configuration(client_side=True))
        conn.initiate_connection()
        s.sendall(conn.data_to_send())
        sid = conn.get_next_available_stream_id()
        conn.send_headers(
            sid,
            [
                (":method", "GET"),
                (":authority", f"127.0.0.1:{port}"),
                (":scheme", "http"),
                (":path", path),
            ],
            end_stream=True,
        )
        s.sendall(conn.data_to_send())
        ended = False
        while not ended:
            data = s.recv(65535)
            if not data:
                break
            for event in conn.receive_data(data):
                if isinstance(event, h2.events.StreamEnded):
                    ended = True
            out = conn.data_to_send()
            if out:
                s.sendall(out)
    finally:
        s.close()


class _FakeSSLObj:
    def __init__(self) -> None:
        self.server_hostname = "realtime.example.test"

    def selected_alpn_protocol(self) -> None:
        return None


def _ws_frame(opcode: int, payload: bytes) -> bytes:
    return bytes([0x80 | opcode, len(payload)]) + payload


def _ws_session(target: str) -> None:
    from wardex_sdk._interceptors._registry import get_registry

    interceptor = get_registry()._installed["ssl"]  # type: ignore[attr-defined]
    obj = _FakeSSLObj()
    interceptor._on_request_bytes(
        obj,
        f"GET {target} HTTP/1.1\r\nHost: realtime.example.test\r\n".encode()
        + b"Upgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    interceptor._on_response_bytes(
        obj,
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n",
    )
    interceptor._on_request_bytes(obj, _ws_frame(0x1, b"hi"))
    interceptor._on_response_bytes(obj, _ws_frame(0x8, (1000).to_bytes(2, "big")))


def _carriers(http_port: int, h2_port: int) -> dict[str, Callable[[str, str], None]]:
    # Each call carries one neighbour argument, `wdxprobe`, a name in neither
    # corpus: a neighbour that shared a corpus name (`q`) would overwrite that
    # entry's value in the dict and fake a loss.
    def form_body(name: str, value: str) -> None:
        body = urllib.parse.urlencode({"wdxprobe": "1", name: value}).encode()
        _h1(
            http_port,
            "POST",
            "/tool",
            body,
            {"Content-Type": "application/x-www-form-urlencoded"},
        )

    def tool_args(name: str, value: str) -> None:
        args = urllib.parse.quote(json.dumps({"wdxprobe": "1", name: value}))
        _h1(
            http_port,
            "POST",
            "/v1/chat/completions",
            b'{"model":"gpt-4o","messages":[{"role":"user","content":"look it up"}]}',
            {"Content-Type": "application/json", "X-Test-Tool-Args": args},
        )

    return {
        "get_query": lambda n, v: _h1(http_port, "GET", f"/tool?{n}={v}&wdxprobe=1"),
        "json_top": lambda n, v: _h1(
            http_port,
            "POST",
            "/tool",
            json.dumps({"wdxprobe": "1", n: v}).encode(),
            {"Content-Type": "application/json"},
        ),
        "json_nested": lambda n, v: _h1(
            http_port,
            "POST",
            "/tool",
            json.dumps({"params": {"filters": {n: v}}}).encode(),
            {"Content-Type": "application/json"},
        ),
        "form_body": form_body,
        "http2_query": lambda n, v: _h2_get(h2_port, f"/tool?{n}={v}&wdxprobe=1"),
        "websocket_query": lambda n, v: _ws_session(f"/socket?{n}={v}"),
        "tool_args": tool_args,
    }


# --- capture -----------------------------------------------------------------


class _Capture(Transport):
    """Encodes what it is handed on both wires, with the policy `init()`
    installed on it — the same stored policy the shipped transports read."""

    def __init__(self) -> None:
        self.otlp: list[bytes] = []
        self.envelopes: list[bytes] = []

    def export(self, envelope: Any, *, timeout: float | None = None) -> None:
        self.otlp.extend(self.encode(envelope, compress=False))
        self.envelopes.append(
            _wardex_native.codec.encode_envelope(
                envelope,
                self._pii_mode,
                list(self._pii_disabled),
                self._limits,
                **self._pii_names(),
            )
        )


class _Wires:
    def __init__(self, cap: _Capture) -> None:
        self.otlp = b"".join(cap.otlp)
        self.envelope = "".join(
            repr(_wardex_native.codec.decode_envelope(b)) for b in cap.envelopes
        )
        self.span_names = [
            sp["name"]
            for body in cap.otlp
            for rs in _wardex_native.codec.decode_otlp_traces(body)["resource_spans"]
            for ss in rs["scope_spans"]
            for sp in ss["spans"]
        ]

    def has(self, value: str) -> tuple[bool, bool]:
        return value in self.envelope, value.encode() in self.otlp


def _send_all(carrier: Callable[[str, str], None], entries: list[dict[str, Any]]) -> _Wires:
    _hub.reset_for_test()
    cap = _Capture()
    wardex.init(transport=cap, intercept=True, capture_mode=CaptureMode.ALL)
    try:
        for e in entries:
            carrier(e["name"], e["value"])
        wardex.flush()
    finally:
        from wardex_sdk._interceptors._registry import get_registry

        get_registry().uninstall_all()
        _hub.reset_for_test()
    return _Wires(cap)


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    _hub.reset_for_test()
    yield
    _hub.reset_for_test()


_CARRIER_NAMES = [
    "get_query",
    "json_top",
    "json_nested",
    "form_body",
    "http2_query",
    "websocket_query",
    "tool_args",
]


@pytest.mark.parametrize("carrier", _CARRIER_NAMES)
def test_every_secret_argument_is_masked_on_every_carrier(carrier, http_port, h2_port):
    wires = _send_all(_carriers(http_port, h2_port)[carrier], SECRETS)
    assert len(wires.span_names) >= len(SECRETS), "every request must produce a span"
    leaked, kept_by_rule = [], []
    for e in SECRETS:
        in_env, in_otlp = wires.has(e["value"])
        url_form_only = e.get("shapes") == ["url_form"]
        if url_form_only and carrier in _JSON_CARRIERS:
            # Rule 4: in JSON these names are data, and the rule keeps them.
            if not (in_env and in_otlp):
                kept_by_rule.append(e["name"])
            continue
        if in_env or in_otlp:
            leaked.append(f"{e['name']} (envelope={in_env}, otlp={in_otlp})")
    assert not leaked, f"{carrier}: secret values on the wire: {leaked}"
    assert not kept_by_rule, f"{carrier}: JSON-kept names went missing: {kept_by_rule}"
    assert all("?" not in n for n in wires.span_names), wires.span_names


@pytest.mark.parametrize("carrier", _CARRIER_NAMES)
def test_every_debugging_argument_survives_on_every_carrier(carrier, http_port, h2_port):
    wires = _send_all(_carriers(http_port, h2_port)[carrier], BENIGN)
    missing = []
    for e in BENIGN:
        in_env, in_otlp = wires.has(e["value"])
        if not (in_env and in_otlp):
            missing.append(f"{e['name']} (envelope={in_env}, otlp={in_otlp})")
    assert not missing, f"{carrier}: debugging values lost: {missing}"
    assert all("?" not in n for n in wires.span_names), wires.span_names


def test_every_secret_is_masked_as_a_url_password(http_port):
    """The eighth carrier: a hand-written absolute-form request line with the
    secret as the URL's password. httpx and requests strip userinfo before
    writing the line; a hand-written one does not."""
    wires = _send_all(
        lambda n, v: _h1(http_port, "GET", f"http://user:{v}@127.0.0.1:{http_port}/p"),
        SECRETS,
    )
    leaked = [e["name"] for e in SECRETS if any(wires.has(e["value"]))]
    assert not leaked, leaked
    assert b"REDACTED:REDACTED@" in wires.otlp
    assert all("@" not in n for n in wires.span_names), wires.span_names
