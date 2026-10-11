"""A real `mcp` stdio server, driven by the real `mcp` client, through the stdio seam.

Every other MCP stdio test feeds hand-written JSON or an echo process, so none of them can
say whether the shapes they assume are the shapes the official SDK puts on the pipe. This one
runs both ends of the official SDK: a low-level server whose one tool answers with
`structuredContent` and an empty `content` list, the shape that used to ship as an OK span
whose output was `[]`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anyio
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import wardex_sdk as wardex
from wardex_sdk import _hub
from wardex_sdk._enums import SpanKind, StatusCode

_SERVER = """
import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server


async def list_tools(ctx, params):
    tool = types.Tool(name="weather", input_schema={"type": "object"})
    return types.ListToolsResult(tools=[tool])


async def call_tool(ctx, params):
    return types.CallToolResult(content=[], structured_content={"city": "Seoul", "celsius": 21.5})


server = Server("weather", on_list_tools=list_tools, on_call_tool=call_tool)


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


anyio.run(main)
"""


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    _hub.reset_for_test()


@pytest.mark.asyncio
async def test_a_structured_only_tool_result_from_a_real_server_is_the_spans_output(
    tmp_path: Path,
) -> None:
    server = tmp_path / "weather_server.py"
    server.write_text(_SERVER)
    # No framework adapter, so this seam's span is the call's execute_tool span.
    wardex.init(intercept=True, adapters=wardex.AdaptersConfig(enabled=()))

    with anyio.fail_after(60):
        params = StdioServerParameters(command=sys.executable, args=[str(server)])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                answer = await session.call_tool("weather", {"city": "Seoul"})
    assert answer.structured_content == {"city": "Seoul", "celsius": 21.5}

    client = _hub.get_client()
    assert client is not None
    (sp,) = [s for s in client._spans if s.tool is not None]
    assert sp.name == "execute_tool weather"
    assert sp.kind == SpanKind.CLIENT
    assert sp.status == StatusCode.OK
    extra = dict(sp.extra)
    assert extra["gen_ai.operation.name"] == "execute_tool"
    assert extra["mcp.protocol.version"] == init.protocol_version
    assert sp.transport.mcp.rpc_method == "tools/call"
    output = json.loads(sp.output_data)
    assert output["structuredContent"] == {"city": "Seoul", "celsius": 21.5}
    assert not sp.capture_integrity or not sp.capture_integrity.limitations
