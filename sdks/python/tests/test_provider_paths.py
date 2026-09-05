"""Provider paths that are not LLM calls, through the PUBLIC seam surface.

The endpoint table used to answer one question — "is this an LLM
endpoint?" — and every other provider path was ordinary HTTP with no name:
a telemetry upload (`/v1/traces/ingest`, the OpenAI Agents SDK's default
run-record POST) became a span carrying the whole record under
`capture_mode=ALL`, and a Conversations-API call dropped under the default
mode left no trace that it had ever happened. These tests drive plaintext
bytes through `wardex.init(intercept=True)` and a loopback server and pin
what each class of path becomes: excluded and counted, plain HTTP, or a
billable LLM span.
"""

from __future__ import annotations

import http.client
import http.server
import json
import pathlib
import threading

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._enums import CaptureMode, OperationName, ProviderName, SpanKind
from wardex_sdk._protocol import classify_path

_REPO = pathlib.Path(__file__).resolve().parents[3]
_FIXTURES = _REPO / "crates" / "wardex-protocol" / "tests" / "fixtures" / "llm"


def _fixture(case: str, name: str) -> bytes:
    return (_FIXTURES / case / name).read_bytes()


@pytest.fixture(autouse=True)
def _fresh_counters():
    # conftest never resets the process-wide counters; every assertion
    # below is an exact count, so each test starts and ends at zero.
    counters.reset()
    yield
    counters.reset()


def _server(payloads: list[tuple[int, bytes]]):
    """A plaintext server answering one queued (status, JSON body) per
    request, whatever the method or path."""
    queue = list(payloads)

    class H(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _answer(self) -> None:
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            status, payload = queue.pop(0)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_DELETE = _answer

        def log_message(self, *a: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
    host, port = httpd.socket.getsockname()[:2]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, host, port


def _stop(httpd) -> None:
    httpd.shutdown()
    httpd.server_close()


def _request(host: str, port: int, method: str, path: str, body: bytes = b"") -> int:
    conn = http.client.HTTPConnection(host, port)
    conn.request(method, path, body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()  # the seam sees the response only as the client consumes it
    conn.close()
    return resp.status


def _client_spans():
    client = _hub.get_client()
    client._settle()  # finalization runs on the worker; settle before reading
    return [s for s in client._spans if s.kind == SpanKind.CLIENT]


_INGEST = b'{"data":[]}'
_CONVERSATION = b'{"id":"conv_1","object":"conversation","created_at":1,"metadata":{}}'
_ITEMS = b'{"object":"list","data":[],"first_id":null,"last_id":null,"has_more":false}'
_DELETED = b'{"id":"conv_1","object":"conversation.deleted","deleted":true}'


# Mirrors the Rust table test in endpoint.rs by hand; the two move together —
# drift is caught because both assert the same rows.
@pytest.mark.parametrize(
    ("path", "want"),
    [
        ("/v1/responses", "llm_call"),
        ("/v1/responses/compact", "llm_call"),
        ("/v1/conversations", "provider_state"),
        ("/v1/conversations/conv_1", "provider_state"),
        ("/v1/conversations/conv_1/items?limit=20", "provider_state"),
        ("/v1/conversations/conv_1/items/msg_1", "provider_state"),
        ("/openai/v1/conversations/conv_1/items", "provider_state"),
        ("/v1/traces/ingest", "excluded"),
        ("/proxy/v1/traces/ingest", "excluded"),
        ("/conversations/conv_1", None),
        ("/api/traces/ingest", None),
        ("/v1/responses/resp_1", None),
        ("/v1/models", None),
        ("", None),
    ],
)
def test_classify_path_table(path: str, want: str | None):
    assert classify_path(path) == want


def test_traces_ingest_is_excluded_under_agent_mode():
    httpd, host, port = _server([(200, b"{}")])
    try:
        wardex.init(intercept=True)
        assert _request(host, port, "POST", "/v1/traces/ingest", _INGEST) == 200
        assert _client_spans() == []
        assert counters.get("interceptors.seam.path_excluded") == 1
    finally:
        wardex.close()
        _stop(httpd)


def test_traces_ingest_is_excluded_under_all_mode():
    """`capture_mode=ALL` used to ship this as `HTTP POST /v1/traces/ingest`
    with the whole run record as `input_data`; the exclusion beats ALL."""
    httpd, host, port = _server([(200, b"{}")])
    try:
        wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
        assert _request(host, port, "POST", "/v1/traces/ingest", _INGEST) == 200
        assert _client_spans() == []
        assert counters.get("interceptors.seam.path_excluded") == 1
    finally:
        wardex.close()
        _stop(httpd)


def test_traces_ingest_is_excluded_above_intercept_hosts():
    """The exclusion sits above the `intercept_hosts` ALLOW; the allowlist
    still captures every other request to the named host."""
    httpd, host, port = _server([(200, b"{}"), (200, b"{}")])
    try:
        wardex.init(intercept=True, intercept_hosts=[f"{host}:{port}"])
        assert _request(host, port, "POST", "/v1/traces/ingest", _INGEST) == 200
        assert _client_spans() == []
        assert counters.get("interceptors.seam.path_excluded") == 1
        assert _request(host, port, "POST", "/health") == 200
        spans = _client_spans()
        assert [s.name for s in spans] == ["HTTP POST /health"]
    finally:
        wardex.close()
        _stop(httpd)


def test_traces_ingest_inside_workflow_is_still_excluded():
    """Inside a live local span every request is captured as a child — the
    exclusion is the one exception, so the run record never rides along."""
    httpd, host, port = _server([(200, b"{}")])
    try:
        wardex.init(intercept=True)

        @wardex.workflow(name="run")
        def run() -> None:
            assert _request(host, port, "POST", "/v1/traces/ingest", _INGEST) == 200

        run()
        client = _hub.get_client()
        client._settle()
        assert [s.kind for s in client._spans] == [SpanKind.INTERNAL]
        assert counters.get("interceptors.seam.path_excluded") == 1
    finally:
        wardex.close()
        _stop(httpd)


def test_conversation_create_is_dropped_and_counted_under_agent_mode():
    """Provider-owned agent state is not an LLM call: under the default mode
    and outside a local span it is still dropped — but now counted."""
    httpd, host, port = _server([(200, _CONVERSATION), (200, _ITEMS)])
    try:
        wardex.init(intercept=True)
        assert _request(host, port, "POST", "/v1/conversations", b"{}") == 200
        assert _request(host, port, "GET", "/v1/conversations/conv_1/items") == 200
        assert _client_spans() == []
        assert counters.get("interceptors.seam.provider_state_dropped") == 2
        assert counters.get("interceptors.seam.path_excluded") == 0
    finally:
        wardex.close()
        _stop(httpd)


def test_conversation_items_are_plain_http_under_all_mode():
    httpd, host, port = _server([(200, _ITEMS)])
    try:
        wardex.init(intercept=True, capture_mode=CaptureMode.ALL)
        body = b'{"items":[{"role":"user","content":"hello"}]}'
        assert _request(host, port, "POST", "/v1/conversations/conv_1/items", body) == 200
        spans = _client_spans()
        assert [s.name for s in spans] == ["HTTP POST /v1/conversations/conv_1/items"]
        sp = spans[0]
        assert sp.gen_ai is None
        assert Limitation.SEMANTIC_PARSE_FAILED not in sp.capture_integrity.limitations
        assert sp.input_data
        assert counters.get("interceptors.seam.provider_state_dropped") == 0
    finally:
        wardex.close()
        _stop(httpd)


def test_conversation_call_inside_workflow_is_a_child_http_span():
    httpd, host, port = _server([(200, _DELETED)])
    try:
        wardex.init(intercept=True)

        @wardex.workflow(name="run")
        def run() -> None:
            assert _request(host, port, "DELETE", "/v1/conversations/conv_1") == 200

        run()
        client = _hub.get_client()
        client._settle()
        internal = [s for s in client._spans if s.kind == SpanKind.INTERNAL]
        spans = [s for s in client._spans if s.kind == SpanKind.CLIENT]
        assert len(internal) == 1
        assert [s.name for s in spans] == ["HTTP DELETE /v1/conversations/conv_1"]
        assert spans[0].parent_span_id == internal[0].context.span_id
        assert counters.get("interceptors.seam.provider_state_dropped") == 0
    finally:
        wardex.close()
        _stop(httpd)


def test_root_level_conversations_path_is_not_provider_state():
    """The anchor on `v1` is what keeps a non-OpenAI `/conversations`
    (Intercom, Front) out of the count."""
    httpd, host, port = _server([(200, b"{}")])
    try:
        wardex.init(intercept=True)
        assert _request(host, port, "POST", "/conversations", b"{}") == 200
        assert _client_spans() == []
        assert counters.get("interceptors.seam.provider_state_dropped") == 0
    finally:
        wardex.close()
        _stop(httpd)


def test_responses_compact_is_a_billable_llm_call_under_agent_mode():
    """A compaction is billed — model + input in, usage + output out — so it
    is a chat span under the default mode, with the caller's echoed message
    kept as `user` and the compaction item as an unmapped part."""
    httpd, host, port = _server([(200, _fixture("openai_responses_compact", "response.json"))])
    try:
        wardex.init(intercept=True)
        req = _fixture("openai_responses_compact", "request.json")
        assert _request(host, port, "POST", "/v1/responses/compact", req) == 200
        spans = _client_spans()
        assert len(spans) == 1
        sp = spans[0]
        assert sp.gen_ai is not None
        assert sp.gen_ai.operation == OperationName.CHAT
        assert sp.gen_ai.input_tokens == 1200
        assert sp.gen_ai.output_tokens == 300
        assert sp.gen_ai.response_id == "resp_cmp1"
        assert not sp.gen_ai.finish_reasons
        assert sp.gen_ai.response_status is None
        # provider from the body shape: the host here is 127.0.0.1
        assert sp.gen_ai.provider == ProviderName.OPENAI
        extra = dict(sp.extra)
        assert extra["openai.api.type"] == "responses"
        # the one thing that tells a compaction from a chat at export
        assert sp.transport.http.url.endswith("/v1/responses/compact")
        markers = sp.capture_integrity.limitations
        assert Limitation.OUTPUT_MESSAGES_UNMAPPED_PART in markers
        # tokens satisfy `has_core_semantics`, so the `output_messages is
        # None` branch that would mark a parse failure is never reached
        assert Limitation.SEMANTIC_PARSE_FAILED not in markers
        msgs = json.loads(extra["gen_ai.output.messages"])
        assert msgs[0]["role"] == "user"
        assert msgs[0]["parts"][0]["content"] == "hello"
        assert msgs[1]["role"] == "assistant"
        for m in msgs:
            if m["role"] == "assistant":
                assert all(part.get("content") != "hello" for part in m["parts"])
    finally:
        wardex.close()
        _stop(httpd)
