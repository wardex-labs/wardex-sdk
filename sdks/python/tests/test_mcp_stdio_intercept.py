import asyncio
import sys

import anyio
import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._enums import SpanKind, StatusCode


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk.interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


def _client_spans():
    return [s for s in _hub.get_client()._spans if s.kind == SpanKind.CLIENT]


# an echo server that reads one line from stdin (a JSON-RPC request) and writes a
# tools/call response with the same id to stdout
_ECHO = (
    "import sys, json\n"
    "line = sys.stdin.readline()\n"
    "req = json.loads(line)\n"
    "resp = {'jsonrpc':'2.0','id':req['id'],"
    "'result':{'content':[{'type':'text','text':'Created #42'}],'isError':False}}\n"
    "sys.stdout.write(json.dumps(resp)+'\\n'); sys.stdout.flush()\n"
)


@pytest.mark.asyncio
async def test_anyio_stdio_tools_call_is_captured():
    wardex.init(intercept=True)
    proc = await anyio.open_process([sys.executable, "-c", _ECHO])
    req = (
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        b'"params":{"name":"create_issue","arguments":{"repo":"acme/web"}}}\n'
    )
    await proc.stdin.send(req)
    # receive one line of response (until newline-terminated)
    buf = b""
    while b"\n" not in buf:
        chunk = await proc.stdout.receive()
        if not chunk:
            break
        buf += chunk
    await proc.wait()

    spans = _client_spans()
    mcp = [s for s in spans if s.transport and s.transport.mcp is not None]
    assert len(mcp) == 1
    sp = mcp[0]
    assert sp.name == "MCP tools/call"
    assert sp.tool.name == "create_issue"
    assert b'"repo"' in sp.input_data
    assert b"Created #42" in sp.output_data
    assert sp.status == StatusCode.OK
    assert sp.transport.timing.ttfb_ms > 0.0  # measured round-trip latency (request → response)


@pytest.mark.asyncio
async def test_raw_asyncio_stdio_tools_call_is_captured():
    wardex.init(intercept=True)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _ECHO,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    req = (
        b'{"jsonrpc":"2.0","id":5,"method":"tools/call",'
        b'"params":{"name":"read_file","arguments":{"path":"a.py"}}}\n'
    )
    proc.stdin.write(req)
    await proc.stdin.drain()
    await proc.stdout.readline()
    await proc.wait()

    mcp = [s for s in _client_spans() if s.transport and s.transport.mcp is not None]
    assert len(mcp) == 1
    assert mcp[0].tool.name == "read_file"
    assert mcp[0].transport.mcp.rpc_id == "5"


@pytest.mark.asyncio
async def test_anyio_spawn_does_not_double_wrap_via_asyncio_seam():
    """A process spawned via the anyio path must skip the asyncio auxiliary seam
    (no double-wrapping)."""
    wardex.init(intercept=True)
    from wardex_sdk.interceptors._registry import get_registry

    interceptor = get_registry()._installed["mcp_stdio"]
    proc = await anyio.open_process([sys.executable, "-c", _ECHO])
    req = b'{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"t","arguments":{}}}\n'
    await proc.stdin.send(req)
    buf = b""
    while b"\n" not in buf:
        chunk = await proc.stdout.receive()
        if not chunk:
            break
        buf += chunk
    await proc.wait()
    # the anyio path must skip the asyncio auxiliary seam (no double-wrapping)
    assert interceptor._asyncio_wrap_count == 0
    mcp = [s for s in _client_spans() if s.transport and s.transport.mcp is not None]
    assert len(mcp) == 1
