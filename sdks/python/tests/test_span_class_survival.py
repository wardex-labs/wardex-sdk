"""No CLASS of span disappeared when the six emit sites were routed through
`assembly.SpanDraft`.

**Why this file exists, precisely.** `SpanDraft.finish()` raises
`VocabularyError` on a vocabulary breach, and every emit site builds inside
`assembly._diag.guard()` because I6 forbids that exception reaching the host.
The two together mean a breach does not warn, does not log at default settings
and does not produce a partial span: it DELETES the span and increments a
counter. A green test suite is therefore not proof that the routing preserved
anything — a suite that never asserts on a given span class would stay green
while that class vanished entirely. Design §6.5.1 is the same argument written
as a prerequisite: routed against the 15-member `Limitation` enum that predated
the census, this would have silently deleted every gRPC span, every streaming
chat span, every WebSocket span and every Agent-SDK adapter span.

So each test below drives one span class through the REAL path and asserts two
things:

  1. a span came out, with the shape it had before (name, kind, the typed block
     that identifies the class);
  2. `assembly.counters.total() == 0` — nothing was swallowed anywhere while it
     did. This is the half that cannot be skipped. Without it, a class that is
     produced twice and deleted once still passes assertion 1.

`_no_swallows` is autouse so the second half applies to every test here whether
its author remembered it or not.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._enums import CaptureMode, CaptureSource, SpanKind, StatusCode, ToolExecutionType
from wardex_sdk._tracing import span as manual_span
from wardex_sdk._tracing import trace
from wardex_sdk.adapters._assembler import SessionAssembler
from wardex_sdk.assembly import Limitation, ParentSource, counters
from wardex_sdk.interceptors._mcp_stdio import _ProcState
from wardex_sdk.interceptors._seam import ByteSeamInterceptor, _ConnectionState
from wardex_sdk.interceptors._trackers import _Http1Tracker, _Http2Tracker, _WebSocketTracker

# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


class _FakeClient:
    def __init__(self) -> None:
        self.config = WardexConfig(
            capture_mode=CaptureMode.ALL, debug=True, backend=BackendConfig(api_key="k")
        )
        self.spans: list = []
        self.snapshots: list = []

    def capture_span(self, span) -> None:
        self.spans.append(span)

    def capture_snapshot(self, snapshot) -> None:
        self.snapshots.append(snapshot)

    def close(self, timeout: float = 5.0) -> None:
        """conftest's autouse fixture closes whatever the hub holds."""


class _Seam(ByteSeamInterceptor):
    """The byte seam with its two abstract hooks stubbed, nothing else changed.

    The tracker is supplied per test rather than sniffed, so each span class is
    driven through the seam's REAL `_on_request_bytes`/`_on_response_bytes` and
    `_emit_span`/`_emit_ws` — including the gate, the timing resolution and the
    `guard()` that would hide a `VocabularyError`.
    """

    def __init__(self, tracker) -> None:
        super().__init__()
        self._tracker = tracker

    def _select_tracker(self, obj):
        return self._tracker

    def _resolve_timing(self, obj, st):
        return 0.0, 0.0, False, ()

    def name(self):
        return "test-seam"

    def install(self, client, ctx=None):
        self._client = client

    def uninstall(self):
        pass


@pytest.fixture(autouse=True)
def _no_swallows():
    """Every test here asserts nothing was swallowed while it ran.

    `guard()` is the only sanctioned swallow in the SDK and it always counts, so
    this reads the one signal a deleted span leaves behind. Autouse because the
    failure mode this file exists for is precisely the one an author forgets to
    check.
    """
    counters.reset()
    yield
    assert counters.snapshot() == {}, (
        f"a span was swallowed while this test ran: {counters.snapshot()}.\n"
        "That is what a VocabularyError looks like from the outside — the span "
        "is gone and only the counter is left. Run with config.debug=True (all "
        "clients here set it) to get the traceback."
    )


@pytest.fixture
def client() -> _FakeClient:
    _hub.reset_for_test()
    c = _FakeClient()
    _hub.set_client(c)
    return c


def _seam(client: _FakeClient, tracker) -> _Seam:
    s = _Seam(tracker)
    s._client = client
    s._load_limits(client)
    return s


def _state(seam: _Seam, tracker, host="api.openai.com", port=443) -> _ConnectionState:
    return _ConnectionState(tracker=tracker, server_address=host, server_port=port)


def _h1(
    body: bytes,
    content_type: bytes = b"application/json",
    status_line: bytes = b"200 OK",
) -> tuple[bytes, bytes]:
    request = (
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: api.openai.com\r\n"
        b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}"
    )
    response = (
        b"HTTP/1.1 "
        + status_line
        + b"\r\nContent-Type: "
        + content_type
        + b"\r\nContent-Length: "
        + str(len(body)).encode()
        + b"\r\n\r\n"
        + body
    )
    return request, response


def _h2_frame(ftype: int, flags: int, stream_id: int, payload: bytes) -> bytes:
    return (
        len(payload).to_bytes(3, "big")
        + bytes([ftype, flags])
        + stream_id.to_bytes(4, "big")
        + payload
    )


def _grpc_msg(payload: bytes, compressed: int = 0) -> bytes:
    return bytes([compressed]) + len(payload).to_bytes(4, "big") + payload


def _ws_frame(fin: bool, opcode: int, payload: bytes) -> bytes:
    out = bytearray()
    out.append((0x80 if fin else 0) | opcode)
    out.append(len(payload))
    out += payload
    return bytes(out)


def _drive_seam(seam: _Seam, host: str, request: bytes, response: bytes) -> None:
    obj = SimpleNamespace(server_hostname=host)
    seam._on_request_bytes(obj, request)
    seam._on_response_bytes(obj, response)


# --------------------------------------------------------------------------
# 1. HTTP/1 — the plain transport observation
# --------------------------------------------------------------------------


def test_http1_span_survives(client):
    seam = _seam(client, _Http1Tracker())
    request, response = _h1(b'{"ok":true}')

    _drive_seam(seam, "api.openai.com", request, response)

    (sp,) = client.spans
    assert sp.name == "HTTP POST /v1/chat/completions"
    assert sp.kind is SpanKind.CLIENT
    assert sp.status is StatusCode.OK
    assert sp.transport is not None
    assert sp.transport.http.status_code == 200
    assert sp.capture_sources == (CaptureSource.SSL,)
    assert ("network.protocol.version", "1.1") in sp.extra
    # "attempted and succeeded", not "non-empty"
    assert sp.capture_integrity.request_body_captured is True
    assert sp.capture_integrity.response_body_captured is True


@pytest.mark.parametrize(
    ("status_line", "status", "body"),
    [
        (b"400 Bad Request", 400, b'{"error":{"message":"bad"}}'),
        (b"401 Unauthorized", 401, b'{"error":{"message":"no key"}}'),
        (b"404 Not Found", 404, b"{}"),
        (b"429 Too Many Requests", 429, b'{"error":{"message":"Rate limit reached"}}'),
        (b"500 Internal Server Error", 500, b"{}"),
        (b"529 Overloaded", 529, b"{}"),
    ],
)
def test_http1_error_response_span_survives(client, status_line, status, body):
    """The class every one of the tests above misses: a NON-2xx response.

    Every other HTTP fixture in this file answers 200, so the whole
    `status=ERROR` half of the seam was untested — and `finish()` deletes a span
    whose status is ERROR with no `error.type`. For an LLM SDK this is the
    highest-value class there is: the rate limit, the auth failure and the
    provider 5xx are the responses a user opens the dashboard to find.

    The autouse `_no_swallows` fixture is the other half of the assertion here.
    """
    seam = _seam(client, _Http1Tracker())
    request, response = _h1(body, status_line=status_line)

    _drive_seam(seam, "api.openai.com", request, response)

    (sp,) = client.spans
    assert sp.name == "HTTP POST /v1/chat/completions"
    assert sp.status is StatusCode.ERROR
    # the status rendered as a string — OTel's HTTP-client `error.type` when the
    # instrumentation has no richer classification, which the byte seam has not
    assert sp.error_type == str(status)
    assert sp.transport.http.status_code == status


def test_http1_redirect_is_not_an_error(client):
    """The boundary the fix must not move: 3xx stays OK with no error type."""
    seam = _seam(client, _Http1Tracker())
    request, response = _h1(b"", status_line=b"301 Moved Permanently")

    _drive_seam(seam, "api.openai.com", request, response)

    (sp,) = client.spans
    assert sp.status is StatusCode.OK
    assert sp.error_type is None


# --------------------------------------------------------------------------
# 2. HTTP/2
# --------------------------------------------------------------------------


def test_http2_span_survives(client):
    from hpack import Encoder

    seam = _seam(client, _Http2Tracker())
    cenc, senc = Encoder(), Encoder()
    request = _h2_frame(
        0x1,
        0x4,
        1,
        cenc.encode([(b":method", b"POST"), (b":path", b"/v1/messages")]),
    ) + _h2_frame(0x0, 0x1, 1, b'{"x":1}')
    response = _h2_frame(0x1, 0x4, 1, senc.encode([(b":status", b"200")])) + _h2_frame(
        0x0, 0x1, 1, b'{"y":2}'
    )

    _drive_seam(seam, "api.openai.com", request, response)

    (sp,) = client.spans
    assert sp.name == "HTTP POST /v1/messages"
    assert sp.kind is SpanKind.CLIENT
    assert ("network.protocol.version", "2") in sp.extra
    assert sp.output_data == b'{"y":2}'


def test_http2_error_response_span_survives(client):
    """The same non-2xx class over h2. `_build_span` is shared between the two
    protocol trackers and between the TLS and plaintext seams, so this is the
    second of the four surfaces that one line decides."""
    from hpack import Encoder

    seam = _seam(client, _Http2Tracker())
    cenc, senc = Encoder(), Encoder()
    request = _h2_frame(
        0x1,
        0x4,
        1,
        cenc.encode([(b":method", b"POST"), (b":path", b"/v1/messages")]),
    ) + _h2_frame(0x0, 0x1, 1, b'{"x":1}')
    response = _h2_frame(0x1, 0x4, 1, senc.encode([(b":status", b"529")])) + _h2_frame(
        0x0, 0x1, 1, b'{"error":"overloaded"}'
    )

    _drive_seam(seam, "api.openai.com", request, response)

    (sp,) = client.spans
    assert sp.name == "HTTP POST /v1/messages"
    assert sp.status is StatusCode.ERROR
    assert sp.error_type == "529"


# --------------------------------------------------------------------------
# 3. gRPC — the class §6.5.1 names first among the ones the routing could have
#    deleted
# --------------------------------------------------------------------------


def test_grpc_span_survives_with_its_rpc_attributes(client):
    from hpack import Encoder

    seam = _seam(client, _Http2Tracker())
    cenc, senc = Encoder(), Encoder()
    request = _h2_frame(
        0x1,
        0x4,
        1,
        cenc.encode(
            [
                (b":method", b"POST"),
                (b":path", b"/pkg.Svc/Do"),
                (b"content-type", b"application/grpc"),
            ]
        ),
    ) + _h2_frame(0x0, 0x1, 1, _grpc_msg(b"req"))
    response = (
        _h2_frame(
            0x1, 0x4, 1, senc.encode([(b":status", b"200"), (b"content-type", b"application/grpc")])
        )
        + _h2_frame(0x0, 0x0, 1, _grpc_msg(b"resp", compressed=1))
        + _h2_frame(0x1, 0x4 | 0x1, 1, senc.encode([(b"grpc-status", b"5")]))
    )

    _drive_seam(seam, "api.openai.com", request, response)

    (sp,) = client.spans
    assert sp.name == "gRPC /pkg.Svc/Do"
    assert sp.kind is SpanKind.CLIENT
    assert sp.status is StatusCode.ERROR
    assert sp.error_type == "NOT_FOUND"
    assert ("rpc.system", "grpc") in sp.extra
    assert ("rpc.service", "pkg.Svc") in sp.extra
    assert ("rpc.method", "Do") in sp.extra
    # the marker that reaches this span through the closed vocabulary
    assert Limitation.PAYLOAD_COMPRESSED in sp.capture_integrity.limitations


def test_grpc_frame_parse_failure_still_emits_a_span(client):
    """The fallback branch, which is a different construction path.

    A gRPC body wardex cannot frame falls back to plain h2 fields plus
    FRAME_PARSE_FAILED, and the span's NAME follows the same decision — this is
    where those two could disagree, because the name is built by the grammar
    and the marker by the vocabulary.
    """
    from hpack import Encoder

    seam = _seam(client, _Http2Tracker())
    cenc, senc = Encoder(), Encoder()
    # a declared length that runs past the buffer: truncation, not an exception
    request = _h2_frame(
        0x1,
        0x4,
        1,
        cenc.encode(
            [
                (b":method", b"POST"),
                (b":path", b"/pkg.Svc/Do"),
                (b"content-type", b"application/grpc"),
            ]
        ),
    ) + _h2_frame(0x0, 0x1, 1, _grpc_msg(b"hello")[:7])
    response = _h2_frame(
        0x1, 0x4, 1, senc.encode([(b":status", b"200"), (b"content-type", b"application/grpc")])
    ) + _h2_frame(0x0, 0x1, 1, b"")

    _drive_seam(seam, "api.openai.com", request, response)

    (sp,) = client.spans
    assert sp.name == "gRPC /pkg.Svc/Do"
    assert Limitation.GRPC_MESSAGE_TRUNCATED in sp.capture_integrity.limitations


# --------------------------------------------------------------------------
# 4. SSE / streaming chat — the second class §6.5.1 names
# --------------------------------------------------------------------------


_SSE = (
    b'data: {"id":"c1","model":"gpt-4o-mini","choices":[{"index":0,'
    b'"delta":{"role":"assistant","content":"Hi!"},"finish_reason":null}]}\n\n'
    b'data: {"id":"c1","model":"gpt-4o-mini","choices":[{"index":0,'
    b'"delta":{},"finish_reason":"stop"}]}\n\n'
    b"data: [DONE]\n\n"
)


def test_streaming_chat_span_survives_with_gen_ai_and_its_markers(client):
    seam = _seam(client, _Http1Tracker())
    request, response = _h1(_SSE, content_type=b"text/event-stream")

    _drive_seam(seam, "api.openai.com", request, response)

    (sp,) = client.spans
    assert sp.name == "HTTP POST /v1/chat/completions"
    assert sp.gen_ai is not None
    assert sp.gen_ai.response_model == "gpt-4o-mini"
    assert sp.gen_ai.finish_reasons == ("stop",)
    lims = sp.capture_integrity.limitations
    assert Limitation.REASSEMBLED_FROM_STREAM in lims
    assert Limitation.STREAM_USAGE_UNAVAILABLE in lims
    # gen_ai.operation.name is mirrored into the block, not double-carried in
    # `extra` — that unification is exactly what `finish()` decides once
    assert [k for k, _ in sp.extra if k == "gen_ai.operation.name"] == []


# --------------------------------------------------------------------------
# 5. WebSocket — the third class §6.5.1 names
# --------------------------------------------------------------------------


def test_websocket_span_survives(client):
    seam = _seam(client, _WebSocketTracker(path="/realtime", deflate=True, parent=None, start_ns=1))

    _drive_seam(
        seam,
        "ws.example.com",
        _ws_frame(True, 0x1, b'{"a":1}'),
        _ws_frame(True, 0x8, (1000).to_bytes(2, "big")),
    )

    ws = [s for s in client.spans if s.name.startswith("WS ")]
    assert ws, "the WebSocket span class disappeared"
    sp = ws[-1]
    assert sp.name == "WS /realtime"
    assert sp.kind is SpanKind.CLIENT
    assert sp.status is StatusCode.OK
    assert ("network.protocol.version", "websocket") in sp.extra
    assert ("ws.close_code", 1000) in sp.extra
    assert Limitation.PAYLOAD_COMPRESSED in sp.capture_integrity.limitations


def test_websocket_flush_span_survives_with_its_marker(client):
    """The uninstall/eviction path, which builds the span from `flush(marker)`.

    `ws_no_close` and `ws_evicted` were the two markers the census could only
    find by following call sites, and both reach the span through this path.
    """
    tracker = _WebSocketTracker(path="/x", deflate=False, parent=None, start_ns=1)
    seam = _seam(client, tracker)
    tracker.on_request_bytes(_ws_frame(True, 0x1, b"hi"))
    st = _state(seam, tracker, host="ws.example.com")

    for txn in tracker.flush(Limitation.WS_NO_CLOSE):
        seam._emit_ws(st, txn)

    (sp,) = client.spans
    assert sp.name == "WS /x"
    assert Limitation.WS_NO_CLOSE in sp.capture_integrity.limitations


# --------------------------------------------------------------------------
# 6. MCP stdio
# --------------------------------------------------------------------------


def test_mcp_stdio_tool_call_span_survives(client):
    state = _ProcState(debug=True)
    state.feed_request(
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        b'"params":{"name":"search","arguments":{"q":"x"}}}\n'
    )

    spans = state.feed_response(b'{"jsonrpc":"2.0","id":1,"result":{"content":[{"a":1}]}}\n')

    (sp,) = spans
    assert sp.name == "MCP tools/call"
    assert sp.kind is SpanKind.CLIENT
    assert sp.tool is not None
    assert sp.tool.name == "search"
    # IPC, not the NETWORK default this span used to assert about a subprocess pipe
    assert sp.tool.execution_type is ToolExecutionType.IPC
    assert sp.capture_sources == (CaptureSource.STDIO,)
    assert sp.transport.mcp.rpc_method == "tools/call"


def test_mcp_stdio_error_span_survives_and_gains_an_error_type(client):
    """`status=ERROR` with no `error_type` is what `finish()` refuses.

    This span shipped exactly that pair, so it is one of the two places the
    routing had to supply a type or delete the span. The type is derived from
    the JSON-RPC error code rather than invented.
    """
    state = _ProcState(debug=True)
    state.feed_request(b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"t"}}\n')

    spans = state.feed_response(
        b'{"jsonrpc":"2.0","id":7,"error":{"code":-32601,"message":"nope"}}\n'
    )

    (sp,) = spans
    assert sp.name == "MCP tools/call"
    assert sp.status is StatusCode.ERROR
    assert sp.error_type == "json_rpc_-32601"


def test_mcp_stdio_non_tool_method_span_survives(client):
    """`ping` is not a tool call, and the span must not claim it is.

    Design §4.6's site-3 sketch gives every MCP method the `EXECUTE_TOOL`
    intent; this seam sees `initialize`, `ping` and `resources/read` too, and
    naming those `execute_tool` would be a vocabulary lie. The span stays a
    TRANSPORT observation and simply carries no tool block.
    """
    state = _ProcState(debug=True)
    state.feed_request(b'{"jsonrpc":"2.0","id":2,"method":"ping"}\n')

    spans = state.feed_response(b'{"jsonrpc":"2.0","id":2,"result":{}}\n')

    (sp,) = spans
    assert sp.name == "MCP ping"
    assert sp.tool is None


# --------------------------------------------------------------------------
# 7. Adapter — chat, tool, subagent, root (the fourth class §6.5.1 names)
# --------------------------------------------------------------------------

_INIT = {
    "type": "system",
    "subtype": "init",
    "session_id": "s-1",
    "model": "claude-sonnet-5",
}
_ASSISTANT = {
    "type": "assistant",
    "session_id": "s-1",
    "message": {
        "id": "m1",
        "model": "claude-sonnet-5",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 25},
        "content": [
            {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": "ls"}}
        ],
    },
}
_RESULT = {
    "type": "result",
    "subtype": "success",
    "session_id": "s-1",
    "is_error": False,
    "num_turns": 1,
    "total_cost_usd": 0.01,
    "duration_api_ms": 80,
}


def _run_session(client: _FakeClient) -> SessionAssembler:
    asm = SessionAssembler(client)
    asm.on_outbound(
        1,
        json.dumps(
            {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "hi"}}
        ),
    )
    asm.on_inbound(1, _INIT)
    asm.on_inbound(1, _ASSISTANT)
    asm.on_hook(
        "PreToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_input": {"command": "ls"}},
        "toolu_01",
    )
    asm.on_hook(
        "SubagentStart",
        {"session_id": "s-1", "agent_id": "a-1", "agent_type": "researcher"},
        None,
    )
    asm.on_hook("SubagentStop", {"session_id": "s-1", "agent_id": "a-1"}, None)
    asm.on_hook(
        "PostToolUse",
        {"session_id": "s-1", "tool_name": "Bash", "tool_response": "ok"},
        "toolu_01",
    )
    asm.on_inbound(1, _RESULT)
    asm.on_close(1, None)
    return asm


def test_every_adapter_span_class_survives(client):
    _run_session(client)

    names = [s.name for s in client.spans]
    assert "chat claude-sonnet-5" in names
    assert "execute_tool Bash" in names
    assert "invoke_agent researcher" in names
    assert "invoke_agent" in names

    chat = next(s for s in client.spans if s.name == "chat claude-sonnet-5")
    assert chat.kind is SpanKind.CLIENT
    assert chat.gen_ai is not None and chat.gen_ai.input_tokens == 10
    assert chat.conversation.session_id == "s-1"

    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert tool.tool is not None and tool.tool.call_id == "toolu_01"
    # A tool span makes NO parentage claim: its anchor may have come from a
    # silent fallback, so there is no edge here to price. It used to publish a
    # confidence with `strategy=None` beside it, which encodes as
    # `parent_source = UNSPECIFIED` — a number answering a question the span
    # declined to answer. The framework's id survives where it belongs, as the
    # tool's own `call_id` above.
    assert tool.correlation is None

    sub = next(s for s in client.spans if s.name == "invoke_agent researcher")
    assert sub.agent is not None and sub.agent.id == "a-1"

    root = next(s for s in client.spans if s.name == "invoke_agent")
    assert root.agent is not None
    assert ("wardex.agent.num_turns", 1) in root.extra
    for sp in client.spans:
        assert CaptureSource.ADAPTER in sp.capture_sources
        assert (
            Limitation.TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS in sp.capture_integrity.limitations
        )


def test_adapter_tree_is_one_trace_rooted_at_the_session(client):
    """The shape, not just the population: a deleted root would leave the
    children as orphans that still look like spans."""
    _run_session(client)

    root = next(s for s in client.spans if s.name == "invoke_agent")
    for sp in client.spans:
        assert sp.context.trace_id == root.context.trace_id
    for name in ("chat claude-sonnet-5", "execute_tool Bash", "invoke_agent researcher"):
        sp = next(s for s in client.spans if s.name == name)
        assert sp.parent_span_id == root.context.span_id


def test_adapter_session_without_a_session_id_still_emits(client):
    """`conversation_id` may not be the empty string (§6.3), and this session
    never reports one — so it is the case that would have been deleted."""
    asm = SessionAssembler(client)
    asm.on_outbound(
        1,
        json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}}),
    )
    asm.on_close(1, None)

    root = next(s for s in client.spans if s.name == "invoke_agent")
    assert root.conversation.conversation_id
    assert root.conversation.session_id is None


def test_adapter_failed_tool_span_survives_with_an_error_type(client):
    asm = SessionAssembler(client)
    asm.on_outbound(
        1,
        json.dumps(
            {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "hi"}}
        ),
    )
    asm.on_inbound(1, _INIT)
    asm.on_hook(
        "PreToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_input": {}}, "toolu_09"
    )
    asm.on_hook(
        "PostToolUseFailure",
        {"session_id": "s-1", "tool_name": "Bash", "tool_response": "boom"},
        "toolu_09",
    )

    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert tool.status is StatusCode.ERROR
    assert tool.error_type == "tool_error"
    # a tool called with {} attempted its capture and succeeded
    assert tool.capture_integrity.request_body_captured is True


def test_adapter_abort_emits_root_and_forces_children_closed(client):
    asm = SessionAssembler(client)
    asm.on_outbound(
        1,
        json.dumps(
            {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "hi"}}
        ),
    )
    asm.on_inbound(1, _INIT)
    asm.on_hook(
        "PreToolUse", {"session_id": "s-1", "tool_name": "Bash", "tool_input": {}}, "toolu_09"
    )
    asm.on_close(1, "ProcessError")

    root = next(s for s in client.spans if s.name == "invoke_agent")
    tool = next(s for s in client.spans if s.name == "execute_tool Bash")
    assert root.status is StatusCode.ERROR
    assert root.error_type == "session_error"
    assert Limitation.SESSION_ABORTED in root.capture_integrity.limitations
    assert Limitation.CHILD_SPAN_UNCLOSED in tool.capture_integrity.limitations


# --------------------------------------------------------------------------
# 8. Manual spans and snapshots — the last two of the six sites
# --------------------------------------------------------------------------


def test_manual_span_survives_and_gains_the_forensic_fields(client):
    with trace("session") as root:
        with manual_span("inner"):
            pass

    inner = next(s for s in client.spans if s.name == "inner")
    assert inner.parent_span_id == root.context.span_id
    # what the second constructor used to omit (§6.4)
    assert inner.capture_sources == (CaptureSource.MANUAL,)
    assert inner.correlation is not None
    assert inner.correlation.strategy is ParentSource.CONTEXTVAR


def test_manual_span_marked_error_by_the_host_survives(client):
    """`set_status(StatusCode.ERROR)` is published API and ERROR is its only
    non-OK member — so the ordinary way a host reports a failure must not be the
    one thing that deletes the span. With no type given, `finish()` supplies
    OTel's `_OTHER` rather than refusing."""
    with manual_span("checkout") as s:
        s.set_status(StatusCode.ERROR, "payment declined")

    (sp,) = client.spans
    assert sp.name == "checkout"
    assert sp.status is StatusCode.ERROR
    assert sp.status_message == "payment declined"
    assert sp.error_type == "_OTHER"


def test_manual_span_can_name_its_own_error_type(client):
    """The other half: the published surface must be able to express the state
    that keeps the span, not only the state that used to delete it."""
    with trace("root") as s:
        s.set_status(StatusCode.ERROR)
        s.set_error("payment_declined", "card expired")

    (sp,) = client.spans
    assert sp.status is StatusCode.ERROR
    assert sp.error_type == "payment_declined"
    assert sp.status_message == "card expired"


def test_manual_span_with_an_empty_name_degrades_instead_of_disappearing(client):
    """`wardex.span(label or "")` loses its name, not its span. Timing and
    parentage — the two things the caller wanted — do not depend on the name."""
    with manual_span(""):
        pass

    (sp,) = client.spans
    assert sp.name == "span"


def test_manual_span_builder_still_takes_host_attributes(client):
    """The yielded builder is handed to the host's own `with` block, so an
    assignment that used to work may not start raising in the middle of it."""
    with manual_span("x") as s:
        s.my_tag = 1
        s.name = "renamed"
        s.kind = SpanKind.SERVER
        assert s.my_tag == 1

    (sp,) = client.spans
    assert sp.name == "renamed"
    assert sp.kind is SpanKind.SERVER


def test_manual_span_keeps_arbitrary_attribute_keys(client):
    """`set_attribute` is a published API that takes any key. Enforcing the
    `gen_ai.*`/`wardex.*` namespace on it would delete the user's span."""
    with manual_span("inner") as s:
        s.set_attribute("my.company.thing", "v")

    (sp,) = client.spans
    assert ("my.company.thing", "v") in sp.extra


def test_snapshot_survives_and_carries_its_orphan_marker(client):
    wardex_sdk.capture_state_snapshot(conversation_state=b"{}")

    (snap,) = client.snapshots
    assert snap.snapshot_type == "turn_start"
    assert dict(snap.attributes)["wardex.limitations"] == Limitation.PARENT_UNRESOLVED.value


def test_snapshot_with_an_unknown_type_degrades_instead_of_disappearing(client):
    """`SnapshotType` is a closed vocabulary. An unrecognized value used to be
    flattened to UNSPECIFIED inside `codec.rs` with nothing recorded."""
    with trace("session"):
        wardex_sdk.capture_state_snapshot(snapshot_type="not_a_type")

    (snap,) = client.snapshots
    assert snap.snapshot_type == ""
    assert Limitation.SNAPSHOT_TYPE_UNKNOWN.value in dict(snap.attributes)["wardex.limitations"]


# --------------------------------------------------------------------------
# negative control — the harness must FAIL on a span that is actually deleted
# --------------------------------------------------------------------------


def test_the_harness_would_notice_a_deleted_span(client, monkeypatch):
    """A survival suite nobody has watched fail proves nothing.

    This forces the exact failure mode the file exists for — `finish()` raising
    inside `guard()` — and asserts that both halves of every test above react:
    the span is gone AND the counter moved. If someone later wraps an emit site
    in a bare `except Exception: pass`, the counter stops moving and this test
    is what says so.
    """
    from wardex_sdk.assembly import SpanDraft, VocabularyError

    def _boom(self, end_ns=None):
        raise VocabularyError("forced")

    monkeypatch.setattr(SpanDraft, "finish", _boom)
    seam = _seam(client, _Http1Tracker())
    request, response = _h1(b'{"ok":true}')

    _drive_seam(seam, "api.openai.com", request, response)

    assert client.spans == [], "the span should have been deleted by the guard"
    assert counters.snapshot() == {"interceptors.seam.emit_span": 1}
    counters.reset()  # the autouse guard must not fire on the control itself
