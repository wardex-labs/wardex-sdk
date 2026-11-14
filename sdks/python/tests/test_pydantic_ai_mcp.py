"""An MCP tool a pydantic-ai agent calls: the framework's `execute_tool`, and the wire under it.

pydantic-ai wraps each MCP tool call in its own `execute_tool` span exactly like a function tool
(measured on 2.66: call id, arguments and result on the span), so the bridge records it as a tool
unit under the run. The MCP stdio seam ALSO records the JSON-RPC `tools/call` it sees on the pipe,
as it does for any client in the process; the seam is deliberately not ceded to the adapter,
because an agent that bypasses instrumentation, or an MCP client that is not pydantic-ai's, would
then lose its calls in silence.

The seam latches its parent on the task that WRITES to the server's stdin, and that is not the
tool's task: the MCP client writes from a task it spawned when the toolset connected (measured,
2026-11-08), and pydantic-ai puts no `traceparent` into the request's `_meta`. Left alone, the
seam's span would land beside the tool unit under the run (toolset connected inside the run) or in
a trace of its own (agent entered with `async with agent:` first). So the adapter names its live
tool units to the core, with the arguments pydantic-ai recorded at START, and the seam parents a
`tools/call` to the ONE live tool unit of that name whose arguments equal the request's
(`_assembly/_bridge_latch.py`, `latch_tool_call`). The tree is then

    invoke_agent -> execute_tool add (framework: call id, arguments, result)
                 -> MCP tools/call   (wire: the JSON-RPC request and response)

and the seam's span stays the transport observation `MCP tools/call`, because the framework's
`execute_tool` above it is the call. Two calls of the same tool in flight at once are told apart by
their arguments. Two with the same arguments cannot be: then the seam keeps its own placement,
promotes its span to `execute_tool` as it would for any client, and the adapter counts the
ambiguity. A tool of the same name that another MCP client calls with other arguments is never
nested under the agent's unit.

One limitation is not closed: an unrelated client calling a tool of the same name with exactly the
same arguments while the agent's call is live is indistinguishable from it, and is nested under it.

pydantic-ai 2.x reaches stdio servers through `MCPToolset(fastmcp StdioTransport(...))`
(`MCPServerStdio` was removed), so this file needs `fastmcp-slim` from the test group. The server
is the official SDK's low-level `Server`, as in `test_mcp_real_server.py`.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from opentelemetry.metrics import NoOpMeterProvider
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings

import wardex_sdk as wardex
from _pydantic_ai_support import MODES, host_provider, instrumentation_mode, is_bridge
from wardex_sdk._assembly import counters
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


def _record(answer: Any, server: Path, *, entered: bool, prefix: str | None = None) -> list[Any]:
    """Run one agent against the arithmetic server and return every shipped span."""
    transport = RecordingTransport()
    wardex.init(transport=transport, adapters=AdaptersConfig(enabled=(AdapterName.PYDANTIC_AI,)))
    try:
        toolset: Any = MCPToolset(
            StdioTransport(command=sys.executable, args=[str(server)]), id="arith"
        )
        if prefix is not None:
            toolset = toolset.prefixed(prefix)
        agent = Agent(FunctionModel(answer, model_name="fn"), toolsets=[toolset], name="mcp")

        async def run() -> Any:
            return (await asyncio.wait_for(agent.run("add 2 and 3"), 60)).output

        async def main() -> Any:
            if not entered:
                return await run()
            async with agent:
                return await run()

        output = asyncio.run(main())
        wardex.flush()
        return [output, *(s for e in transport.envelopes for s in e.spans)]
    finally:
        wardex.close()


@pytest.fixture
def server(tmp_path: Path) -> Path:
    path = tmp_path / "arith_server.py"
    path.write_text(_SERVER)
    return path


@pytest.mark.parametrize("entered", [False, True], ids=["connected_in_run", "entered_first"])
def test_an_mcp_tool_call_nests_the_seam_span_under_the_framework_tool_unit(
    instrumentation, server: Path, entered: bool
):
    output, *spans = _record(_answer, server, entered=entered)
    assert output == "sum is 5"

    tools = [s for s in spans if s.tool is not None and s.tool.name == "add"]
    bridge = [s for s in tools if is_bridge(s)]
    seam = [s for s in tools if CaptureSource.STDIO in s.capture_sources]
    assert [s.name for s in bridge] == ["execute_tool add"], sorted(s.name for s in spans)
    assert [s.name for s in seam] == ["MCP tools/call"], "the framework's span is the call"
    assert len([s for s in spans if s.name.startswith("execute_tool")]) == 1
    unit, call = bridge[0], seam[0]
    run_unit = next(s for s in spans if s.name == "invoke_agent mcp")
    assert unit.tool.call_id == "call_1"
    assert unit.parent_span_id == run_unit.context.span_id
    assert call.parent_span_id == unit.context.span_id
    assert call.context.trace_id == run_unit.context.trace_id
    assert call.transport.mcp.rpc_method == "tools/call"
    assert b'"a": 2' in call.input_data, "the JSON-RPC arguments"
    assert b'"5"' in call.output_data, "the JSON-RPC result"


def _answer_prefixed(messages: list[Any], info: Any) -> ModelResponse:
    returned = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
    if not returned:
        return ModelResponse(
            parts=[ToolCallPart("srv_add", {"a": 2, "b": 3}, tool_call_id="call_1")]
        )
    return ModelResponse(parts=[TextPart(f"sum is {returned[-1].content}")])


def test_a_prefixed_toolsets_call_nests_under_the_framework_tool_unit_by_its_arguments(
    instrumentation, server: Path
):
    """A prefixed toolset (`.prefixed("srv")`, which is what `MCP(tool_prefix="srv")` builds)
    names the framework's tool `srv_add` while the JSON-RPC request names the server's `add`.
    The arguments prove the unit, so the wire call still nests under it."""
    counters.reset()
    output, *spans = _record(_answer_prefixed, server, entered=False, prefix="srv")
    assert output == "sum is 5"

    bridge = [s for s in spans if is_bridge(s) and s.name.startswith("execute_tool")]
    seam = [s for s in spans if CaptureSource.STDIO in s.capture_sources and s.tool is not None]
    assert [s.name for s in bridge] == ["execute_tool srv_add"], sorted(s.name for s in spans)
    assert [s.name for s in seam] == ["MCP tools/call"], "not promoted: the framework's is the call"
    assert len([s for s in spans if s.name.startswith("execute_tool")]) == 1
    unit, call = bridge[0], seam[0]
    assert call.tool.name == "add", "the wire keeps the server's own name"
    assert call.parent_span_id == unit.context.span_id
    assert call.context.trace_id == unit.context.trace_id
    assert counters.get("interceptors.mcp_stdio.nested_under_tool_unit") == 1
    assert counters.get("adapters.pydantic_ai.tool_call_prefix_unproved") == 0


#: The arithmetic server with a second tool whose own name looks prefixed: `srv_add`.
_SELF_SIMILAR_SERVER = _SERVER.replace(
    'types.Tool(name="mul", description="a * b"', 'types.Tool(name="srv_add", description="+1000"'
).replace('a + b if params.name == "add" else a * b', 'a + b + (params.name != "add") * 1000')


def _answer_self_similar(messages: list[Any], info: Any) -> ModelResponse:
    """Under prefix `srv` the framework names the server's `add` `srv_add` and its `srv_add`
    `srv_srv_add`; both called at once with the same arguments."""
    returned = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
    if not returned:
        return ModelResponse(
            parts=[
                ToolCallPart("srv_add", {"a": 2, "b": 3}, tool_call_id="c_add"),
                ToolCallPart("srv_srv_add", {"a": 2, "b": 3}, tool_call_id="c_srvadd"),
            ]
        )
    return ModelResponse(parts=[TextPart(f"got {sorted(p.content for p in returned)}")])


def test_an_exact_name_that_may_be_a_prefixed_one_is_ambiguous_not_a_match(
    instrumentation, tmp_path: Path
):
    """The wire `srv_add` (the framework's `srv_srv_add`) matches the unit named `srv_add` by
    exact name and arguments, but that unit is the framework's call of `add`. Neither wire call
    can be proved to one unit, so neither is nested, and both are counted as ambiguous."""
    server = tmp_path / "self_similar_server.py"
    server.write_text(_SELF_SIMILAR_SERVER)
    counters.reset()
    output, *spans = _record(_answer_self_similar, server, entered=False, prefix="srv")
    assert output == "got ['1005', '5']"

    units = {s.context.span_id: s for s in spans if is_bridge(s) and s.tool is not None}
    assert sorted(s.tool.call_id for s in units.values()) == ["c_add", "c_srvadd"]
    seam = [s for s in spans if CaptureSource.STDIO in s.capture_sources and s.tool is not None]
    assert sorted(s.tool.name for s in seam) == ["add", "srv_add"]
    for call in seam:
        assert call.parent_span_id not in units, (call.tool.name, units[call.parent_span_id].name)
    assert counters.get("interceptors.mcp_stdio.nested_under_tool_unit") == 0
    assert counters.get("adapters.pydantic_ai.tool_call_ambiguous") == 2


@contextlib.contextmanager
def _without_content() -> Iterator[None]:
    """Instrumentation on with `include_content=False`: the tool span records no arguments."""
    provider, _exporter = host_provider()
    settings = InstrumentationSettings(
        tracer_provider=provider, meter_provider=NoOpMeterProvider(), include_content=False
    )
    before = vars(Agent)["_instrument_default"]
    Agent.instrument_all(settings)
    try:
        yield
    finally:
        Agent._instrument_default = before
        provider.shutdown()


def test_a_prefixed_toolsets_call_is_not_nested_by_the_suffix_of_its_name_alone(server: Path):
    """Without content the framework's unit carries only `srv_add`. A name that merely ends in
    `_add` is no proof, so the wire call keeps the seam's own placement and is counted."""
    counters.reset()
    with _without_content():
        output, *spans = _record(_answer_prefixed, server, entered=False, prefix="srv")
    assert output == "sum is 5"

    run_unit = next(s for s in spans if s.name == "invoke_agent mcp")
    unit = next(s for s in spans if is_bridge(s) and s.name == "execute_tool srv_add")
    [call] = [s for s in spans if CaptureSource.STDIO in s.capture_sources and s.tool is not None]
    assert call.name == "execute_tool add", "promoted, as for any client"
    assert call.parent_span_id != unit.context.span_id
    assert call.parent_span_id == run_unit.context.span_id, "the seam's own placement"
    assert counters.get("interceptors.mcp_stdio.nested_under_tool_unit") == 0
    assert counters.get("adapters.pydantic_ai.tool_call_prefix_unproved") == 1


def _answer_twice(a: dict[str, int], b: dict[str, int]) -> Any:
    """Two calls of the same tool in one response, which pydantic-ai runs concurrently."""

    def answer(messages: list[Any], info: Any) -> ModelResponse:
        returned = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        if not returned:
            return ModelResponse(
                parts=[
                    ToolCallPart("add", a, tool_call_id="call_1"),
                    ToolCallPart("add", b, tool_call_id="call_2"),
                ]
            )
        return ModelResponse(parts=[TextPart(f"sums are {sorted(p.content for p in returned)}")])

    return answer


@pytest.mark.parametrize("entered", [False, True], ids=["connected_in_run", "entered_first"])
def test_two_concurrent_calls_of_one_tool_nest_under_their_own_units_by_arguments(
    instrumentation, server: Path, entered: bool
):
    counters.reset()
    answer = _answer_twice({"a": 2, "b": 3}, {"a": 4, "b": 5})
    output, *spans = _record(answer, server, entered=entered)
    assert output == "sums are ['5', '9']"

    bridge = {s.tool.call_id: s for s in spans if is_bridge(s) and s.name == "execute_tool add"}
    seam = [s for s in spans if CaptureSource.STDIO in s.capture_sources and s.tool is not None]
    assert sorted(bridge) == ["call_1", "call_2"]
    assert [s.name for s in seam] == ["MCP tools/call"] * 2
    for call in seam:
        owner = "call_1" if b'"a": 2' in call.input_data else "call_2"
        assert call.parent_span_id == bridge[owner].context.span_id, owner
    assert counters.get("adapters.pydantic_ai.tool_call_ambiguous") == 0
    assert counters.get("interceptors.mcp_stdio.nested_under_tool_unit") == 2


@pytest.mark.parametrize("entered", [False, True], ids=["connected_in_run", "entered_first"])
def test_two_concurrent_identical_calls_of_one_tool_keep_the_seams_own_placement(
    instrumentation, server: Path, entered: bool
):
    counters.reset()
    output, *spans = _record(
        _answer_twice({"a": 2, "b": 3}, {"a": 2, "b": 3}), server, entered=entered
    )
    assert output == "sums are ['5', '5']"

    run_unit = next(s for s in spans if s.name == "invoke_agent mcp")
    bridge = [s for s in spans if is_bridge(s) and s.name == "execute_tool add"]
    seam = [s for s in spans if CaptureSource.STDIO in s.capture_sources and s.tool is not None]
    assert sorted(s.tool.call_id for s in bridge) == ["call_1", "call_2"]
    assert [s.name for s in seam] == ["execute_tool add"] * 2, "promoted, as for any client"
    units = {s.context.span_id for s in bridge}
    for call in seam:
        assert call.parent_span_id not in units
        if entered:
            assert call.parent_span_id is None
        else:
            assert call.parent_span_id == run_unit.context.span_id
    assert counters.get("adapters.pydantic_ai.tool_call_ambiguous") == 2
    assert counters.get("interceptors.mcp_stdio.nested_under_tool_unit") == 0


def test_another_mcp_clients_call_is_never_nested_under_a_function_tool_of_the_same_name(
    instrumentation, server: Path
):
    """A bare `fastmcp.Client` (not pydantic-ai) calls the server's `add` while the agent's own
    plain FUNCTION tool named `add` is live: different arguments, so the wire call is not the
    function tool's and keeps the seam's own placement, promoted to `execute_tool add`."""
    counters.reset()
    started, may_end = asyncio.Event(), asyncio.Event()
    transport = RecordingTransport()
    wardex.init(transport=transport, adapters=AdaptersConfig(enabled=(AdapterName.PYDANTIC_AI,)))
    try:
        agent = Agent(FunctionModel(_answer, model_name="fn"), name="fn-agent")

        @agent.tool_plain
        async def add(a: int, b: int) -> int:
            started.set()
            await may_end.wait()
            return a + b

        async def other_client() -> str:
            await started.wait()
            try:
                async with Client(StdioTransport(command=sys.executable, args=[str(server)])) as c:
                    result = await c.call_tool("add", {"a": 100, "b": 200})
            finally:
                may_end.set()
            return result.content[0].text

        async def main() -> list[Any]:
            return await asyncio.wait_for(
                asyncio.gather(agent.run("add 2 and 3"), other_client()), 60
            )

        run_result, other = asyncio.run(main())
        wardex.flush()
        spans = [s for e in transport.envelopes for s in e.spans]
    finally:
        wardex.close()
    assert (run_result.output, other) == ("sum is 5", "300")

    unit = next(s for s in spans if is_bridge(s) and s.name == "execute_tool add")
    [call] = [s for s in spans if CaptureSource.STDIO in s.capture_sources and s.tool is not None]
    assert call.name == "execute_tool add", "promoted, as for any client"
    assert call.parent_span_id != unit.context.span_id
    assert call.context.trace_id != unit.context.trace_id
    assert counters.get("interceptors.mcp_stdio.nested_under_tool_unit") == 0
