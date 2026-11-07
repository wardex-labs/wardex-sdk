"""An MCP tool a pydantic-ai agent calls is recorded twice, and where the second copy lands.

pydantic-ai wraps each MCP tool call in its own `execute_tool` span exactly like a function tool
(measured on 2.66: call id, arguments and result on the span), so the bridge records it as a tool
unit under the run. The MCP stdio seam ALSO records the JSON-RPC `tools/call` it sees on the pipe,
as it does for any client in the process; the seam is deliberately not ceded to the adapter,
because an agent that bypasses instrumentation, or an MCP client that is not pydantic-ai's, would
then lose its calls in silence.

Where the seam's span lands is decided by the task that WRITES to the server's stdin, and that is
not the tool's task: the MCP client writes from a task it spawned when the toolset connected
(measured, 2026-11-08). So the seam's span is NOT under the bridge's tool unit:

* toolset connected inside the run (no `async with agent`) — the writer task was spawned under
  the run's pin, and the seam's `tools/call` lands under the run, a sibling of the tool unit;
* agent entered before the run (`async with agent:`) — the writer task predates the run, and the
  seam's span is a trace of its own.

This file pins that, so a change that moves either copy is seen; the README says the same.

pydantic-ai 2.x reaches stdio servers through `MCPToolset(fastmcp StdioTransport(...))`
(`MCPServerStdio` was removed), so this file needs `fastmcp-slim` from the test group. The server
is the official SDK's low-level `Server`, as in `test_mcp_real_server.py`.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp.client.transports import StdioTransport
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

import wardex_sdk as wardex
from _pydantic_ai_support import MODES, instrumentation_mode, is_bridge
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._enums import AdapterName, CaptureSource
from wardex_sdk.testing import RecordingTransport

_SERVER = """
import anyio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
}

async def list_tools(ctx, params):
    return types.ListToolsResult(tools=[
        types.Tool(name="add", description="a + b", input_schema=SCHEMA),
        types.Tool(name="mul", description="a * b", input_schema=SCHEMA),
    ])

async def call_tool(ctx, params):
    a, b = params.arguments["a"], params.arguments["b"]
    value = a + b if params.name == "add" else a * b
    return types.CallToolResult(content=[types.TextContent(type="text", text=str(value))])

server = Server("arith", on_list_tools=list_tools, on_call_tool=call_tool)

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

anyio.run(main)
"""


@pytest.fixture(params=MODES)
def instrumentation(request):
    with instrumentation_mode(request.param) as setup:
        yield setup


def _answer(messages: list[Any], info: Any) -> ModelResponse:
    returned = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
    if not returned:
        return ModelResponse(parts=[ToolCallPart("add", {"a": 2, "b": 3}, tool_call_id="call_1")])
    return ModelResponse(parts=[TextPart(f"sum is {returned[-1].content}")])


@pytest.mark.parametrize("entered", [False, True], ids=["connected_in_run", "entered_first"])
def test_an_mcp_tool_call_is_a_tool_unit_and_a_seam_span_beside_it(
    instrumentation, tmp_path: Path, entered: bool
):
    server = tmp_path / "arith_server.py"
    server.write_text(_SERVER)
    transport = RecordingTransport()
    wardex.init(transport=transport, adapters=AdaptersConfig(enabled=(AdapterName.PYDANTIC_AI,)))
    try:
        toolset = MCPToolset(StdioTransport(command=sys.executable, args=[str(server)]), id="arith")
        agent = Agent(FunctionModel(_answer, model_name="fn"), toolsets=[toolset], name="mcp")

        async def run() -> Any:
            return (await asyncio.wait_for(agent.run("add 2 and 3"), 60)).output

        async def main() -> Any:
            if not entered:
                return await run()
            async with agent:
                return await run()

        assert asyncio.run(main()) == "sum is 5"
        wardex.flush()
        spans = [s for e in transport.envelopes for s in e.spans]
    finally:
        wardex.close()

    tools = [s for s in spans if s.name == "execute_tool add"]
    bridge = [s for s in tools if is_bridge(s)]
    seam = [s for s in tools if CaptureSource.STDIO in s.capture_sources]
    assert len(bridge) == 1, sorted(s.name for s in spans)
    assert len(seam) == 1, [s.capture_sources for s in tools]
    unit, call = bridge[0], seam[0]
    run_unit = next(s for s in spans if s.name == "invoke_agent mcp")
    assert unit.tool.call_id == "call_1"
    assert unit.parent_span_id == run_unit.context.span_id
    assert call.parent_span_id != unit.context.span_id, "the writer task is not the tool's"
    if entered:
        assert call.parent_span_id is None
        assert call.context.trace_id != run_unit.context.trace_id
    else:
        assert call.parent_span_id == run_unit.context.span_id
        assert call.context.trace_id == run_unit.context.trace_id
