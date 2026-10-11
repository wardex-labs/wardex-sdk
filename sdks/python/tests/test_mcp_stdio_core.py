import json
from collections.abc import Iterator

import pytest

from wardex_sdk import _hub
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._enums import CaptureSource, Protocol, SpanKind, StatusCode
from wardex_sdk._interceptors import _mcp_stdio
from wardex_sdk._interceptors._mcp_stdio import _ProcState


@pytest.fixture(autouse=True)
def _no_framework_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whether a framework adapter owns MCP tool spans is decided per process; pin it per test
    so no adapter an earlier test installed decides it here. Tests that need an owner say so."""
    monkeypatch.setattr(_hub, "installed_adapters", frozenset)


@pytest.fixture
def _fresh_reports() -> Iterator[None]:
    reset_reports_for_test()
    yield
    reset_reports_for_test()


def _drive(req: bytes, resp: bytes):
    st = _ProcState()
    st.feed_request(req)
    return [span for span, _scope in st.feed_response(resp)]


def _call(result: object, *, rid: int = 1, name: object = "t") -> tuple[bytes, bytes]:
    """A `tools/call` of tool `name` and the response carrying `result`, as two lines."""
    req = {"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": {"name": name}}
    resp = {"jsonrpc": "2.0", "id": rid, "result": result}
    return (json.dumps(req) + "\n").encode(), (json.dumps(resp) + "\n").encode()


def test_tools_call_extracts_tool_and_io():
    spans = _drive(
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        b'"params":{"name":"create_issue","arguments":{"repo":"acme/web"}}}\n',
        b'{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"Created #42"}],'
        b'"isError":false}}\n',
    )
    assert len(spans) == 1
    sp = spans[0]
    assert sp.kind == SpanKind.CLIENT
    assert sp.name == "execute_tool create_issue"
    assert dict(sp.extra)["gen_ai.operation.name"] == "execute_tool"
    assert sp.transport.protocol == Protocol.MCP_STDIO
    assert sp.transport.mcp.rpc_method == "tools/call"
    assert sp.transport.mcp.rpc_id == "1"
    assert sp.tool is not None
    assert sp.tool.name == "create_issue"
    assert sp.tool.call_id == "1"
    assert b'"repo"' in sp.input_data  # arguments
    assert b"Created #42" in sp.output_data  # result.content
    assert sp.status == StatusCode.OK
    assert CaptureSource.STDIO in sp.capture_sources
    assert sp.parent_span_id is None  # no active span


def test_tools_call_iserror_sets_error_status():
    spans = _drive(
        b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"t","arguments":{}}}\n',
        b'{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"boom"}],"isError":true}}\n',
    )
    assert spans[0].status == StatusCode.ERROR


def test_jsonrpc_error_response_sets_error_status():
    spans = _drive(
        b'{"jsonrpc":"2.0","id":3,"method":"ping"}\n',
        b'{"jsonrpc":"2.0","id":3,"error":{"code":-32601,"message":"no"}}\n',
    )
    assert len(spans) == 1
    assert spans[0].status == StatusCode.ERROR
    assert spans[0].name == "MCP ping"


def test_generic_method_without_tool():
    spans = _drive(
        b'{"jsonrpc":"2.0","id":4,"method":"tools/list","params":{}}\n',
        b'{"jsonrpc":"2.0","id":4,"result":{"tools":[]}}\n',
    )
    assert spans[0].tool is None
    assert spans[0].transport.mcp.rpc_method == "tools/list"


def test_notification_and_unmatched_response_emit_nothing():
    st = _ProcState()
    st.feed_request(b'{"jsonrpc":"2.0","method":"notifications/x"}\n')  # notification (no id)
    out = st.feed_response(b'{"jsonrpc":"2.0","id":999,"result":{}}\n')  # no matching request
    assert out == []


def test_should_detach_after_non_jsonrpc_bytes():
    st = _ProcState()
    st.feed_request(b"x" * (st.SNIFF_LIMIT + 1) + b"\n")  # not JSON-RPC, exceeds the limit
    assert st.should_detach() is True


def test_no_detach_when_jsonrpc_seen():
    st = _ProcState()
    st.feed_request(b'{"jsonrpc":"2.0","id":1,"method":"a"}\n')
    assert st.should_detach() is False


def test_proc_state_reports_disabled_reason_from_either_direction():
    """_ProcState.disabled_reason() must surface a latch on either the
    request or the response parser, mirroring _Http1Tracker's equivalent —
    the JSON-RPC path must not keep the silent-failure mode where a parser
    stops parsing and nothing anywhere says why."""
    from wardex_sdk import LimitsConfig

    limits = LimitsConfig(max_stream_buffer_bytes=64).to_native()

    st_req = _ProcState(limits=limits)
    assert st_req.disabled_reason() is None
    st_req.feed_request(b"x" * 200)  # never a newline, past the 64-byte ceiling
    assert st_req.disabled_reason() == "stream_buffer_exceeded"

    st_resp = _ProcState(limits=limits)
    assert st_resp.disabled_reason() is None
    st_resp.feed_response(b"x" * 200)
    assert st_resp.disabled_reason() == "stream_buffer_exceeded"


def test_proc_state_constructor_default_uses_core_limits():
    # No override → the core default ceiling (16 MiB) applies; 200 bytes must
    # not trip it. Guards against a regression where `limits=None` stopped
    # falling back to `JsonRpcParser`'s own default.
    st = _ProcState()
    st.feed_request(b"x" * 200)
    assert st.disabled_reason() is None


def test_should_detach_after_non_jsonrpc_bytes_on_the_response_side():
    """The asymmetry the trigger used to have.

    A subprocess that writes little to stdin and streams a lot back — a
    compiler, a log follower, a media encoder — counted zero bytes towards the
    sniff budget, so it never detached and paid the tee for its whole life.
    """
    st = _ProcState()
    st.feed_response(b"x" * (st.SNIFF_LIMIT + 1) + b"\n")  # not JSON-RPC, exceeds the limit
    assert st.should_detach() is True


def test_the_two_directions_share_one_sniff_budget():
    """Neither side alone passes the budget; together they do. A per-direction
    budget would let a subprocess split its output across both and stay
    attached forever."""
    st = _ProcState()
    half = b"x" * (st.SNIFF_LIMIT // 2 + 1)
    st.feed_request(half)
    assert st.should_detach() is False
    st.feed_response(half)
    assert st.should_detach() is True


def test_a_jsonrpc_stream_never_detaches_however_much_it_streams():
    """The trigger must stay keyed on "nothing was ever parsed": a real MCP
    server that streams megabytes of tool output is exactly what this seam is
    for."""
    st = _ProcState()
    st.feed_request(b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{}}\n')
    st.feed_response(b"x" * (st.SNIFF_LIMIT + 1))
    assert st.should_detach() is False


def test_a_stream_visible_only_on_stdout_never_detaches():
    """Symmetric budget, symmetric evidence — and only one of the two was.

    `should_detach` spends a budget both directions fill, so `_msgs == 0` has to
    mean "nothing was parsed in EITHER direction". It did not: only
    `feed_request` counted, so a subprocess whose JSON-RPC wardex can observe
    only on stdout detached permanently at the sniff limit while the seam was
    parsing valid messages the whole time. That shape is reachable — a host that
    writes stdin through `StreamWriter.writelines` never reaches the raw-asyncio
    seam's `write` tee, and a notification-only producer never sends a request
    at all.
    """
    st = _ProcState()
    frame = b'{"jsonrpc":"2.0","method":"notifications/progress","params":{}}\n'
    while st._resp_bytes <= st.SNIFF_LIMIT:
        st.feed_response(frame)

    assert st._req_bytes == 0, "precondition: this side was never seen"
    assert st.should_detach() is False


def test_a_subprocess_that_only_streams_noise_still_detaches():
    """The other half of the same claim: counting responses must not make the
    trigger unreachable for the compiler or log follower it exists for."""
    st = _ProcState()
    st.feed_response(b"x" * (st.SNIFF_LIMIT + 1))
    assert st.should_detach() is True


def test_a_dead_server_does_not_strand_its_pending_requests():
    """An MCP server that dies leaves every in-flight request latched with a
    response that is never coming — one `Ambient`, and so one SpanContext, per
    stranded request, held by a correlation table nobody will read again."""
    st = _ProcState()
    st.feed_request(b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{}}\n')
    st.feed_request(b'{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{}}\n')
    assert len(st._latch) == 2

    assert st.on_stream_end() == 2
    assert st._latch == {}
    assert st.on_stream_end() == 0  # idempotent: EOF can be observed more than once


# -- what a tool result reads as ------------------------------------------------------------


def test_a_result_of_structured_content_alone_ships_it_as_the_output():
    """A server answering with `structuredContent` and an empty `content` list used to ship an
    OK span whose output was `[]`: the tool's whole answer was dropped."""
    (sp,) = _drive(*_call({"content": [], "structuredContent": {"temperature": 21.5}}))
    assert json.loads(sp.output_data)["structuredContent"] == {"temperature": 21.5}
    assert sp.status == StatusCode.OK
    assert not sp.capture_integrity or not sp.capture_integrity.limitations

    (sp,) = _drive(*_call({"structuredContent": {"n": 1}}))
    assert json.loads(sp.output_data) == {"structuredContent": {"n": 1}}
    assert sp.status == StatusCode.OK


def test_content_beside_structured_content_and_meta_loses_neither():
    result = {
        "content": [{"type": "text", "text": '{"n": 1}'}],
        "structuredContent": {"n": 1},
        "_meta": {"io.example/trace": "abc"},
        "isError": False,
        "resultType": "complete",
    }
    (sp,) = _drive(*_call(result))
    assert json.loads(sp.output_data) == result
    assert sp.status == StatusCode.OK


def test_a_plain_content_result_still_ships_its_content_list():
    """The common shape keeps the output it always had: the `content` list itself."""
    content = [{"type": "text", "text": "ok"}]
    for result in (
        {"content": content},
        {"content": content, "isError": False, "resultType": "complete"},
    ):
        (sp,) = _drive(*_call(result))
        assert json.loads(sp.output_data) == content


@pytest.mark.parametrize(
    "result",
    [
        # An interim answer: the call goes on in a retry under a new JSON-RPC id.
        {"resultType": "input_required", "inputRequests": {"a": {"method": "elicitation/create"}}},
        # A result type this reader does not know.
        {"content": [], "resultType": "deferred"},
        # A failure flag under another spelling, which may well say the tool failed.
        {"content": [{"type": "text", "text": "boom"}], "is_error": True},
        {"content": [], "IsError": True},
        # A failure flag that is not a boolean.
        {"content": [], "isError": "yes"},
        # A `content` that is not the list of blocks every revision defines.
        {"content": "oops"},
        {"content": None},
        # Neither shape of tool output at all.
        {},
        # Not an object.
        [{"type": "text", "text": "x"}],
        "done",
    ],
)
def test_a_result_the_reader_cannot_vouch_for_ships_raw_unset_and_marked(result: object):
    (sp,) = _drive(*_call(result))
    assert json.loads(sp.output_data) == result
    assert sp.status == StatusCode.UNSET
    assert sp.error_type is None
    assert Limitation.SEMANTIC_PARSE_FAILED in sp.capture_integrity.limitations


def test_is_error_true_is_a_tool_error():
    (sp,) = _drive(*_call({"content": [], "structuredContent": {"e": 1}, "isError": True}))
    assert sp.status == StatusCode.ERROR
    assert sp.error_type == "tool_error"
    assert json.loads(sp.output_data)["structuredContent"] == {"e": 1}


@pytest.mark.parametrize(
    "result",
    [
        {"isError": True},
        {"isError": True, "content": "not a list"},
        {"isError": True, "resultType": "input_required"},
    ],
)
def test_a_server_stated_failure_is_an_error_whatever_else_the_shape_lacks(result: object):
    """`isError: true` is the server saying the tool failed; nothing missing beside it makes that
    a claim wardex cannot vouch for."""
    (sp,) = _drive(*_call(result))
    assert sp.status == StatusCode.ERROR
    assert sp.error_type == "tool_error"
    assert json.loads(sp.output_data) == result
    assert not sp.capture_integrity or not sp.capture_integrity.limitations


# -- which revision the shapes were read against -------------------------------------------


def _version(sp) -> str | None:
    return dict(sp.extra).get("mcp.protocol.version")


def test_spans_carry_the_version_initialize_settled_on():
    st = _ProcState()
    st.feed_request(
        b'{"jsonrpc":"2.0","id":0,"method":"initialize","params":{"protocolVersion":"2025-11-25"}}\n'
    )
    # Before the server answers, the client's proposal is not a version anything was read under.
    req, resp = _call({"content": []}, rid=1)
    st.feed_request(req)
    (init,) = [
        sp
        for sp, _ in st.feed_response(
            b'{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":"2025-06-18"}}\n'
        )
    ]
    assert _version(init) == "2025-06-18"
    (call,) = [sp for sp, _ in st.feed_response(resp)]
    assert _version(call) == "2025-06-18"


def test_a_request_stamped_with_its_own_version_carries_that_one():
    """The 2026-07-28 revision has no `initialize`: every request names its version in `_meta`."""
    meta = {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}
    req = {"jsonrpc": "2.0", "id": 7, "method": "tools/list", "params": {"_meta": meta}}
    (sp,) = _drive(
        (json.dumps(req) + "\n").encode(), b'{"jsonrpc":"2.0","id":7,"result":{"tools":[]}}\n'
    )
    assert _version(sp) == "2026-07-28"


def test_no_version_is_claimed_when_none_was_seen():
    (sp,) = _drive(*_call({"content": []}))
    assert _version(sp) is None


@pytest.mark.usefixtures("fresh_counters", "_fresh_reports")
@pytest.mark.parametrize(
    "version",
    ["v" * (2 << 20), "9999-99-99 spoof", "dave@example.com 4111111111111111", 12345, "", None],
)
def test_a_version_that_does_not_read_as_a_revision_is_refused_counted_and_said(version, capsys):
    """The version is the peer's text. A 2 MiB one rode every later span of the subprocess, and
    free text (an address, a card number) was taken verbatim; neither is a revision."""
    st = _ProcState()
    st.feed_request(b'{"jsonrpc":"2.0","id":0,"method":"initialize","params":{}}\n')
    answer = {"jsonrpc": "2.0", "id": 0, "result": {"protocolVersion": version}}
    (init,) = [sp for sp, _ in st.feed_response((json.dumps(answer) + "\n").encode())]
    req, resp = _call({"content": []}, rid=1)
    st.feed_request(req)
    (call,) = [sp for sp, _ in st.feed_response(resp)]
    assert _version(init) is None and _version(call) is None
    assert counters.get("interceptors.mcp_stdio.protocol_version_unreadable") == 1

    # A request's own stamp is its only claim: refused, it is not replaced by a negotiated one.
    st.feed_request(b'{"jsonrpc":"2.0","id":2,"method":"initialize","params":{}}\n')
    st.feed_response(b'{"jsonrpc":"2.0","id":2,"result":{"protocolVersion":"2025-06-18"}}\n')
    meta = {"io.modelcontextprotocol/protocolVersion": version}
    stamped = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {"_meta": meta}}
    st.feed_request((json.dumps(stamped) + "\n").encode())
    (sp,) = [sp for sp, _ in st.feed_response(b'{"jsonrpc":"2.0","id":3,"result":{"tools":[]}}\n')]
    assert _version(sp) is None
    assert counters.get("interceptors.mcp_stdio.protocol_version_unreadable") == 2
    lines = [x for x in capsys.readouterr().err.splitlines() if "protocol_version_unreadable" in x]
    assert len(lines) == 1 and lines[0].startswith("[wardex] ")


def test_a_revision_shaped_version_of_any_spelling_is_kept():
    """Revisions are dates today; a later one need not be, so the test is a shape, not a date."""
    for version in ("2025-06-18", "2027-01-15-draft", "v3.0_rc1", "x" * 32):
        meta = {"io.modelcontextprotocol/protocolVersion": version}
        req = {"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"_meta": meta}}
        (sp,) = _drive((json.dumps(req) + "\n").encode(), b'{"jsonrpc":"2.0","id":1,"result":{}}\n')
        assert _version(sp) == version


# -- execute_tool, and who owns it ---------------------------------------------------------


def test_an_installed_framework_adapter_keeps_the_call_a_transport_span(monkeypatch):
    """Under openai-agents or LangGraph the adapter records the same call as its own
    `execute_tool` span, in another trace; promoting this one too would count the call twice."""
    for owner in ("openai_agents", "langgraph"):
        monkeypatch.setattr(_hub, "installed_adapters", lambda o=owner: frozenset({o}))
        (sp,) = _drive(*_call({"content": []}))
        assert sp.name == "MCP tools/call"
        assert "gen_ai.operation.name" not in dict(sp.extra)
        assert sp.tool.name == "t"

    monkeypatch.setattr(_hub, "installed_adapters", lambda: frozenset({"anthropic_agent_sdk"}))
    (sp,) = _drive(*_call({"content": []}))
    assert sp.name == "execute_tool t"


@pytest.mark.usefixtures("fresh_counters", "_fresh_reports")
def test_ceding_a_tool_call_to_an_adapter_is_said_once_naming_it(monkeypatch, capsys):
    """An adapter auto-installed because its framework is importable, and never used, still turns
    a raw `mcp` client's tool calls into transport spans; that must not happen in silence."""
    monkeypatch.setattr(
        _hub, "installed_adapters", lambda: frozenset({"openai_agents", "langgraph", "codex_exec"})
    )
    for rid in (1, 2):
        (sp,) = _drive(*_call({"content": []}, rid=rid))
        assert sp.name == "MCP tools/call"
    lines = [x for x in capsys.readouterr().err.splitlines() if "execute_tool_ceded" in x]
    assert len(lines) == 1
    assert "langgraph, openai_agents adapter(s)" in lines[0]
    assert "codex_exec" not in lines[0]
    assert "AdaptersConfig(enabled=" in lines[0]
    assert counters.get("interceptors.mcp_stdio.execute_tool_ceded") == 2

    # A call that is not a tool call cedes nothing, and says nothing more.
    _drive(
        b'{"jsonrpc":"2.0","id":9,"method":"tools/list","params":{}}\n',
        b'{"jsonrpc":"2.0","id":9,"result":{"tools":[]}}\n',
    )
    assert counters.get("interceptors.mcp_stdio.execute_tool_ceded") == 2


def test_only_a_tool_call_is_ever_an_execute_tool():
    spans = _drive(
        b'{"jsonrpc":"2.0","id":4,"method":"tools/list","params":{}}\n',
        b'{"jsonrpc":"2.0","id":4,"result":{"tools":[]}}\n',
    )
    assert spans[0].name == "MCP tools/list"
    assert "gen_ai.operation.name" not in dict(spans[0].extra)
    # A tool call whose tool name is unreadable stays a transport observation.
    (sp,) = _drive(
        b'{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{}}\n',
        b'{"jsonrpc":"2.0","id":5,"result":{"content":[]}}\n',
    )
    assert sp.name == "MCP tools/call"


@pytest.mark.parametrize(
    "name",
    [{"x": 1}, 123, ["a"], True, "", "   ", "a b\n\tc", "x\x00y", "n" * 129],
)
def test_a_tool_name_that_cannot_name_a_span_leaves_it_a_transport_span(name: object):
    """The tool name is the peer's text, and a span name is the label every backend indexes:
    an object, a number or a list was `str()`-ed into the name, a newline split it, and a name of
    thousands of characters became the name."""
    (sp,) = _drive(*_call({"content": []}, name=name))
    assert sp.name == "MCP tools/call"
    assert "gen_ai.operation.name" not in dict(sp.extra)


def test_a_tool_name_at_the_length_bound_still_names_the_span():
    for name in ("n" * 128, "get weather", "tools.search_v2"):
        (sp,) = _drive(*_call({"content": []}, name=name))
        assert sp.name == f"execute_tool {name}"


# -- batches, and what is lost visibly ------------------------------------------------------


def test_a_batch_of_calls_becomes_one_span_per_call():
    """Batches were dropped like lines that are not JSON-RPC at all: 24 calls, 0 spans."""
    calls = [
        {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": f"t{i}"}}
        for i in range(24)
    ]
    answers = [{"jsonrpc": "2.0", "id": i, "result": {"content": []}} for i in range(24)]
    st = _ProcState()
    st.feed_request((json.dumps(calls) + "\n").encode())
    spans = [sp for sp, _ in st.feed_response((json.dumps(answers) + "\n").encode())]
    assert sorted(sp.tool.name for sp in spans) == sorted(f"t{i}" for i in range(24))


@pytest.mark.usefixtures("fresh_counters", "_fresh_reports")
def test_a_request_dropped_at_the_pending_bound_is_counted_and_said_once(monkeypatch, capsys):
    monkeypatch.setattr(_mcp_stdio, "MAX_PENDING_REQUESTS", 2)
    st = _ProcState()
    for i in range(5):
        st.feed_request(_call({"content": []}, rid=i)[0])
    assert len(st._latch) == 2
    assert counters.get("interceptors.mcp_stdio.pending_evicted") == 3
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if "pending_evicted" in line]
    assert len(lines) == 1 and lines[0].startswith("[wardex] ")


@pytest.mark.usefixtures("fresh_counters", "_fresh_reports")
def test_a_message_longer_than_the_line_cap_is_counted_and_said_once(capsys):
    from wardex_sdk import LimitsConfig

    st = _ProcState(limits=LimitsConfig(max_stream_buffer_bytes=128).to_native())
    big = {"content": [{"type": "text", "text": "x" * 500}]}
    for rid in (1, 2):
        req, resp = _call(big, rid=rid)
        st.feed_request(req)
        assert st.feed_response(resp) == []
    assert counters.get("interceptors.mcp_stdio.line_over_cap") == 2
    assert st.disabled_reason() is None, "a skipped line costs only itself"
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if "line_over_cap" in line]
    assert len(lines) == 1 and lines[0].startswith("[wardex] ")
    # The next call that fits is still captured.
    req, resp = _call({"content": []}, rid=3)
    st.feed_request(req)
    assert len(st.feed_response(resp)) == 1
