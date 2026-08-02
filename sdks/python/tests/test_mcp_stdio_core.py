from wardex_sdk._enums import CaptureSource, Protocol, SpanKind, StatusCode
from wardex_sdk.interceptors._mcp_stdio import _ProcState


def _drive(req: bytes, resp: bytes):
    st = _ProcState()
    st.feed_request(req)
    return st.feed_response(resp)


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
    assert sp.name == "MCP tools/call"
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
    from wardex_sdk import CaptureLimits

    limits = CaptureLimits(max_stream_buffer_bytes=64).to_native()

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
