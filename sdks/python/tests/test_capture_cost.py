"""What wardex costs traffic it will not keep, and what it says when it refuses a model call.

The capture decision used to be made once, after a transaction ended: every HTTP body in the
process was stored up to the body cap first, and a parse was queued for it, whether or not the
policy could ever keep it. A download outside any agent work held its whole body, and under load
those parse jobs pushed real model calls out of the backlog. These tests hold the decision to the
request head, the budget across bodies, and the account a refusal leaves.
"""

from __future__ import annotations

import contextlib
import http.client
import http.server
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest

import wardex_sdk as wardex
from conftest import _FakeSSLSocket, client_spans
from wardex_sdk import _hub, _wardex_native
from wardex_sdk._assembly import Limitation
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._enums import CaptureMode
from wardex_sdk._interceptors._ssl import SSLInterceptor
from wardex_sdk._limits import LimitsConfig


@pytest.fixture(autouse=True)
def _fresh_reports() -> Iterator[None]:
    reset_reports_for_test()
    yield
    reset_reports_for_test()


def _held() -> int:
    return _wardex_native.protocol.in_flight_body_bytes()


def _losses() -> dict[str, int]:
    return {loss.reason: loss.count for loss in wardex.diagnostics().losses}


class _Server:
    """A loopback HTTP/1.1 server. GET answers `body` under `get_type`; when `hold` is set, it sends
    all but the last byte and waits for `hold` before the rest, so a client that has read all but
    the last byte is looking mid-transfer. POST answers with `post_reply` under `post_type`."""

    def __init__(self, body: bytes = b"{}") -> None:
        self.body = body
        self.get_type = "application/json"
        self.post_reply = b"{}"
        self.post_type = "application/json"
        self.hold: threading.Event | None = None
        server = self

        class H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", server.get_type)
                self.send_header("Content-Length", str(len(server.body)))
                self.end_headers()
                if server.hold is None:
                    self.wfile.write(server.body)
                    return
                self.wfile.write(server.body[:-1])
                self.wfile.flush()
                server.hold.wait(10)
                self.wfile.write(server.body[-1:])

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(200)
                self.send_header("Content-Type", server.post_type)
                self.send_header("Content-Length", str(len(server.post_reply)))
                self.end_headers()
                self.wfile.write(server.post_reply)

            def log_message(self, *a: object) -> None:
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        if self.hold is not None:
            self.hold.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server() -> Iterator[_Server]:
    s = _Server()
    yield s
    s.close()


def _get(port: int, path: str = "/internal/rows") -> bytes:
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("GET", path)
    body = conn.getresponse().read()
    conn.close()
    return body


def _post(port: int, path: str, body: bytes) -> bytes:
    conn = http.client.HTTPConnection("127.0.0.1", port)
    conn.request("POST", path, body, {"Content-Type": "application/json"})
    reply = conn.getresponse().read()
    conn.close()
    return reply


def _count_parse_jobs(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Every job the seam hands the deferred-parse queue."""
    jobs: list[object] = []
    real = Client.capture_deferred

    def spy(self: Client, job: object) -> None:
        jobs.append(job)
        real(self, job)

    monkeypatch.setattr(Client, "capture_deferred", spy)
    return jobs


_ROWS = json.dumps({"rows": [{"id": i, "name": "x" * 40} for i in range(20_000)]}).encode()

_CHAT_REQUEST = b'{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}'
_CHAT_REPLY = json.dumps(
    {
        "id": "c1",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hey"}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }
).encode()


# --- the request head decides ---------------------------------------------------------------


def test_a_download_outside_agent_work_holds_no_body_and_queues_no_parse(server, monkeypatch):
    server.body = _ROWS
    server.hold = threading.Event()
    jobs = _count_parse_jobs(monkeypatch)
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    before = _held()

    conn = http.client.HTTPConnection("127.0.0.1", server.port)
    conn.request("GET", "/internal/rows")
    resp = conn.getresponse()
    first = resp.read(len(_ROWS) - 1)
    held_mid_transfer = _held() - before
    server.hold.set()
    rest = resp.read()
    conn.close()

    assert first + rest == _ROWS, "the application read its body unchanged"
    assert held_mid_transfer == 0, f"{held_mid_transfer} body bytes held for a download"
    assert jobs == [], "a parse was queued for traffic the policy cannot keep"
    assert client_spans() == []


def test_the_same_download_inside_a_span_keeps_its_body(server):
    server.body = _ROWS
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    with wardex.span("run"):
        assert _get(server.port) == _ROWS
    (span,) = client_spans()
    assert span.transport.response_size == len(_ROWS)
    assert span.capture_integrity.response_body_captured


def test_capture_mode_all_keeps_a_download_outside_any_span(server):
    server.body = _ROWS
    wardex.init(
        transport=wardex.NoOpTransport(),
        adapters=wardex.AdaptersConfig(enabled=()),
        capture_mode=CaptureMode.ALL,
    )
    assert _get(server.port) == _ROWS
    (span,) = client_spans()
    assert span.transport.response_size == len(_ROWS)


def test_a_model_call_outside_any_span_keeps_both_bodies(server):
    server.post_reply = _CHAT_REPLY
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    _post(server.port, "/v1/chat/completions", _CHAT_REQUEST)
    (span,) = client_spans()
    assert span.gen_ai is not None
    assert span.gen_ai.request_model == "gpt-4o"
    assert span.capture_integrity.request_body_captured
    assert span.transport.request_size == len(_CHAT_REQUEST)


_STREAM_REPLY = (
    "data: "
    + json.dumps(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "hey"}, "finish_reason": "stop"}],
        }
    )
    + "\n\ndata: [DONE]\n\n"
).encode()


@pytest.mark.parametrize("reply_type", ["text/event-stream", "application/octet-stream"])
def test_a_streamed_model_call_on_an_unrecognised_path_keeps_both_bodies(server, reply_type):
    """A gateway path the endpoint table does not know. The reply's event grammar names the API
    whether or not its Content-Type declares a stream, and the JSON request is held until the
    reply says so, so the span has the model and the prompt."""
    server.post_type = reply_type
    server.post_reply = _STREAM_REPLY
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    _post(server.port, "/corp/llm/stream", _CHAT_REQUEST)
    (span,) = client_spans()
    assert span.gen_ai is not None
    assert span.gen_ai.request_model == "gpt-4o"
    assert span.gen_ai.response_model == "gpt-4o"
    assert span.capture_integrity.request_body_captured
    assert span.transport.request_size == len(_CHAT_REQUEST)
    assert span.capture_integrity.response_body_captured


def test_a_json_post_whose_reply_is_no_stream_queues_no_parse_and_holds_nothing(
    server, monkeypatch
):
    server.post_reply = _ROWS
    jobs = _count_parse_jobs(monkeypatch)
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    before = _held()
    assert _post(server.port, "/internal/search", _CHAT_REQUEST) == _ROWS
    assert jobs == []
    assert client_spans() == []
    assert _held() == before


# --- one budget for every body in flight ----------------------------------------------------


def test_bodies_in_flight_share_one_budget_and_one_arriving_into_a_full_budget_is_cut(server):
    big = json.dumps({"rows": ["y" * 100] * 3000}).encode()  # ~310 KB
    server.body = big
    server.hold = threading.Event()
    before = _held()
    wardex.init(
        transport=wardex.NoOpTransport(),
        adapters=wardex.AdaptersConfig(enabled=()),
        capture_mode=CaptureMode.ALL,
        # Twice the room: a body that is no model call's may take half of the budget.
        limits=LimitsConfig(max_in_flight_body_bytes=2 * (before + len(big) + 1000)),
    )
    first = http.client.HTTPConnection("127.0.0.1", server.port)
    first.request("GET", "/first")
    first_resp = first.getresponse()
    first_resp.read(len(big) - 1)  # all but one byte of the first body is now held

    second = _Server(big)
    try:
        assert _get(second.port, "/second") == big, "the application's own read is untouched"
    finally:
        second.close()
    server.hold.set()
    assert first_resp.read() == big[-1:]
    first.close()

    spans = {s.transport.http.url.rsplit("/", 1)[-1]: s for s in client_spans()}
    assert spans["first"].transport.response_size == len(big)
    assert Limitation.BODY_CAP_EXCEEDED not in spans["first"].capture_integrity.limitations
    cut = spans["second"]
    assert Limitation.BODY_CAP_EXCEEDED in cut.capture_integrity.limitations
    assert cut.capture_integrity.truncated
    assert cut.transport.response_size is None
    assert _losses().get("body_cap") == 1


@pytest.mark.parametrize("where", ["download inside a span", "event stream outside any span"])
def test_bodies_that_are_no_model_call_leave_room_for_one(server, where):
    """Traffic that cannot be a model call by its target fills its half of the budget, and a model
    call arriving meanwhile is still stored whole: the span keeps its model and is not cut."""
    server.body = (b"data: " + b"y" * 1000 + b"\n\n") * 400  # ~400 KB
    server.get_type = "text/event-stream" if "event stream" in where else "application/json"
    server.hold = threading.Event()
    server.post_reply = _CHAT_REPLY
    before = _held()
    budget = 2 * (before + 150_000)  # the download alone could fill all of it
    wardex.init(
        transport=wardex.NoOpTransport(),
        adapters=wardex.AdaptersConfig(enabled=()),
        limits=LimitsConfig(max_in_flight_body_bytes=budget),
    )
    conn = http.client.HTTPConnection("127.0.0.1", server.port)
    try:
        with wardex.span("tools") if "inside" in where else contextlib.nullcontext():
            conn.request("GET", "/feed")
            resp = conn.getresponse()
            resp.read(len(server.body) - 1)
        assert _held() <= budget // 2, "the download took more than its half"
        with wardex.span("agent"):
            _post(server.port, "/v1/chat/completions", _CHAT_REQUEST)
        server.hold.set()
        resp.read()
    finally:
        conn.close()  # its parser's share of the budget goes with it, whatever failed

    (call,) = [s for s in client_spans() if s.name == "HTTP POST /v1/chat/completions"]
    assert call.gen_ai is not None and call.gen_ai.request_model == "gpt-4o"
    assert Limitation.BODY_CAP_EXCEEDED not in call.capture_integrity.limitations


# --- an overloaded parse queue -------------------------------------------------------------


def test_an_overloaded_parse_queue_ships_no_ordinary_traffic_and_counts_every_model_call(
    server, capsys
):
    """`max_parse_backlog_bytes=1` makes every job too large to wait, so each one takes the
    eviction path at once: the deterministic form of a backlog under load."""
    server.body = _ROWS
    server.post_reply = _CHAT_REPLY
    wardex.init(
        transport=wardex.NoOpTransport(),
        adapters=wardex.AdaptersConfig(enabled=()),
        limits=LimitsConfig(max_parse_backlog_bytes=1),
    )
    for _ in range(3):
        _get(server.port)
        _post(server.port, "/v1/chat/completions", _CHAT_REQUEST)

    spans = client_spans()
    assert [s.name for s in spans] == ["HTTP POST /v1/chat/completions"] * 3
    for span in spans:
        assert Limitation.PARSE_BACKLOG_FULL in span.capture_integrity.limitations
    assert _losses().get("backpressure") == 3
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if "deferred-parse backlog" in line]
    assert len(lines) == 1
    assert "1 transaction" not in lines[0]
    assert "backpressure" in lines[0]


def test_an_overloaded_parse_queue_does_not_count_ordinary_traffic_as_lost(server, capsys):
    """Inside a span every exchange ships, parsed or not; one that could never be a model call
    loses nothing when its parse is skipped, so it is marked but not counted under backpressure."""
    wardex.init(
        transport=wardex.NoOpTransport(),
        adapters=wardex.AdaptersConfig(enabled=()),
        limits=LimitsConfig(max_parse_backlog_bytes=1),
    )
    with wardex.span("tools"):
        for _ in range(3):
            _get(server.port)
    spans = [s for s in client_spans() if s.name == "HTTP GET /internal/rows"]
    assert len(spans) == 3
    for span in spans:
        assert Limitation.PARSE_BACKLOG_FULL in span.capture_integrity.limitations
    assert "backpressure" not in _losses()
    line = next(x for x in capsys.readouterr().err.splitlines() if "deferred-parse" in x)
    assert "A model call among them" in line, line


# --- a model API wardex does not read -------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "api"),
    [
        ("/v1beta/models/gemini-2.0-flash:generateContent", "the Google Gemini API"),
        ("/model/anthropic.claude-3-haiku/converse", "Amazon Bedrock"),
        ("/api/chat", "the Ollama API"),
    ],
)
def test_a_refused_call_to_an_unread_model_api_is_counted_and_said_once(server, capsys, path, api):
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    _post(server.port, path, b'{"contents":[]}')
    _post(server.port, path, b'{"contents":[]}')
    assert client_spans() == []
    assert _losses().get("mode_refused") == 2
    lines = [line for line in capsys.readouterr().err.splitlines() if api in line]
    assert len(lines) == 1, lines
    assert "wardex.span" in lines[0] and "capture_mode=CaptureMode.ALL" in lines[0]


def test_ordinary_refusals_are_not_counted_as_model_calls(server, capsys):
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    _post(server.port, "/internal/search", b"{}")
    _get(server.port)
    assert "mode_refused" not in _losses()
    assert "not recorded" not in capsys.readouterr().err


# --- intercept_hosts names a TLS host too ---------------------------------------------------


class _AgentClient:
    def __init__(self) -> None:
        self.config = WardexConfig(
            capture_mode=CaptureMode.AGENT, backend=BackendConfig(api_key="k")
        )
        self.spans: list[Any] = []

    def capture_span(self, span: Any, *, scope: object = None) -> None:
        self.spans.append(span)

    def capture_deferred(self, job: Any) -> None:
        span = job.ctx.run(job.run)
        if span is not None:
            self.spans.append(span)


@pytest.mark.parametrize(("hosts", "kept"), [(["Gateway.Corp.Example"], 1), ([], 0)])
def test_intercept_hosts_captures_a_tls_host_named_by_its_server_name(hosts, kept):
    _hub.reset_for_test()
    seam = SSLInterceptor(hosts)
    seam._client = _AgentClient()
    seam._load_limits(seam._client)
    sock = _FakeSSLSocket(None)
    sock.server_hostname = "gateway.corp.example"
    seam._on_request_bytes(sock, b"POST /internal/route HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}")
    seam._on_response_bytes(
        sock, b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
    )
    assert len(seam._client.spans) == kept
    if kept:
        assert seam._client.spans[0].capture_integrity.request_body_captured
