"""The openai-agents adapter on the paths that used to lose structure without a word.

Each section is one path where a run the framework executed left less in wardex than it held, and
nothing said so: a hosted tool, a run whose tracing was turned off after install, a run with
sensitive data off, a Chat Completions model, a custom or voice span, and a span the adapter could
not place. Each test drives REAL `Runner` calls (or the framework's own span helpers) with wardex
initialised, and asserts that the loss is now either recorded as a span or counted and said once.

The fake servers are loopback ones: the Responses fake from `test_openai_agents_wire`, swapped per
scenario, and a Chat Completions fake defined here. Response items are built from the `openai`
SDK's own response types, so a field this file spells wrong fails validation instead of passing.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import logging
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from agents import (
    Agent,
    CodeInterpreterTool,
    FileSearchTool,
    HostedMCPTool,
    ImageGenerationTool,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    ShellTool,
    WebSearchTool,
    function_tool,
)
from agents.tracing import (
    SpanData,
    custom_span,
    function_span,
    get_trace_provider,
    set_tracing_disabled,
    speech_group_span,
    speech_span,
    trace,
    transcription_span,
)
from openai import AsyncOpenAI
from openai.types.responses import (
    ResponseCodeInterpreterToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionShellToolCall,
    ResponseFunctionShellToolCallOutput,
    ResponseFunctionWebSearch,
    ResponseToolSearchCall,
)
from openai.types.responses.response_function_web_search import ActionSearch
from openai.types.responses.response_output_item import ImageGenerationCall, McpCall

import wardex_sdk as wardex
from test_openai_agents_adapter import (
    _DONE,
    _adapter_spans,
    _chat_spans,
    _extra,
    _fc,
    _fresh_framework_http_client,
    _init,
    _one,
    _outputs_done,
    _spans,
    scenario,
    tracing_enabled,
    wardex_log,
)
from test_openai_agents_wire import _agents, agents_env, fake_openai
from wardex_sdk._assembly import counters
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._enums import StatusCode, ToolExecutionType, ToolType

pytestmark = pytest.mark.usefixtures("fresh_counters")

#: Fixtures defined next door, re-exported into this module so pytest finds them.
_FIXTURES = (
    agents_env,
    fake_openai,
    scenario,
    tracing_enabled,
    wardex_log,
    _fresh_framework_http_client,
)

_P = "adapters.openai_agents."


@pytest.fixture(autouse=True)
def _fresh_reports() -> Iterator[None]:
    """Every line asserted here is a `report_once` line, and its key is process-global: a test
    that ran earlier would otherwise have spent the line this one asserts."""
    reset_reports_for_test()
    yield
    reset_reports_for_test()


def _warnings(log: Any, needle: str) -> list[str]:
    return [m for m in log.lines(logging.WARNING) if needle in m]


def _drive(entry: str, agent: Agent, **kwargs: Any) -> Any:
    """`agent` through one entry point, a streamed run drained to its end."""
    if entry == "run":
        return asyncio.run(Runner.run(agent, "hi", **kwargs)).final_output
    if entry == "run_sync":
        return Runner.run_sync(agent, "hi", **kwargs).final_output

    async def go() -> Any:
        result = Runner.run_streamed(agent, "hi", **kwargs)
        async for _ in result.stream_events():
            pass
        return result.final_output

    return asyncio.run(go())


# --------------------------------------------------------------------------
# 1. hosted tools: one execute_tool each, read off the response that ran it
# --------------------------------------------------------------------------


def _hosted_items() -> list[dict]:
    """Six server-executed items and the answer, in ONE response, the way the provider returns a
    turn whose tools it ran itself. The MCP call failed; the rest completed."""
    items = [
        ResponseFunctionWebSearch(
            id="ws_1",
            status="completed",
            type="web_search_call",
            action=ActionSearch(type="search", query="weather in Seoul"),
        ),
        ResponseFileSearchToolCall(
            id="fs_1", queries=["refund policy"], status="completed", type="file_search_call"
        ),
        ResponseCodeInterpreterToolCall(
            id="ci_1",
            code="print(1)",
            container_id="cntr_1",
            outputs=None,
            status="completed",
            type="code_interpreter_call",
        ),
        ImageGenerationCall(
            id="ig_1", result="aGVsbG8=", status="completed", type="image_generation_call"
        ),
        McpCall(
            id="mcp_1",
            arguments='{"q":"x"}',
            name="lookup",
            server_label="docs",
            type="mcp_call",
            error={"type": "http_error", "code": 502, "message": "the server refused"},
        ),
        ResponseToolSearchCall(
            id="ts_1",
            arguments={"query": "calendar"},
            execution="server",
            status="completed",
            type="tool_search_call",
        ),
    ]
    return [i.model_dump(exclude_none=True) for i in items] + _DONE


def _hosted_agent() -> Agent:
    return Agent(
        name="agent_a",
        instructions="a",
        model="gpt-4o-mini",
        tools=[
            WebSearchTool(),
            FileSearchTool(vector_store_ids=["vs_1"]),
            CodeInterpreterTool(tool_config={"type": "code_interpreter", "container": "cntr_1"}),
            ImageGenerationTool(tool_config={"type": "image_generation"}),
            HostedMCPTool(
                tool_config={"type": "mcp", "server_label": "docs", "server_url": "https://x"}
            ),
        ],
    )


_HOSTED_EXPECTED = {
    # span name: (call id, tool type, status, error type)
    "execute_tool web_search": ("ws_1", ToolType.EXTENSION, StatusCode.OK, None),
    "execute_tool file_search": ("fs_1", ToolType.DATASTORE, StatusCode.OK, None),
    "execute_tool code_interpreter": ("ci_1", ToolType.EXTENSION, StatusCode.OK, None),
    "execute_tool image_generation": ("ig_1", ToolType.EXTENSION, StatusCode.OK, None),
    "execute_tool lookup": ("mcp_1", ToolType.EXTENSION, StatusCode.ERROR, "hosted_tool_failed"),
    "execute_tool tool_search": ("ts_1", ToolType.EXTENSION, StatusCode.OK, None),
}


@pytest.mark.parametrize("entry", ["run", "run_streamed"])
def test_hosted_tools_are_tool_spans_joined_to_the_wire_span_that_ran_them(
    agents_env, scenario, entry
):
    """Before: the run read `invoke_agent agent_a` and one `chat` span, and the six tools the
    provider ran inside that call were visible only as parts of the chat span's output. Now each
    is an `execute_tool` under the agent that asked, carrying the item's id, the response id that
    carried it, and the model call's interval, because the provider reports no per-tool timing.
    Five of the six kinds join the wire span by that id today; the tool search is on the wire
    span as a part the wire parser does not map, so it carries no id to join."""
    scenario(lambda inp: _hosted_items())
    _init()
    try:
        assert _drive(entry, _hosted_agent()) == "done"
        spans = _spans()
    finally:
        wardex.close()
    agent = _one(spans, "invoke_agent agent_a")
    (chat,) = _chat_spans(spans)
    for name, (call_id, tool_type, status, error_type) in _HOSTED_EXPECTED.items():
        tool = _one(spans, name)
        assert tool.parent_span_id == agent.context.span_id, name
        assert tool.tool.call_id == call_id, name
        assert tool.tool.type is tool_type, name
        assert tool.tool.execution_type is ToolExecutionType.NETWORK, name
        assert (tool.status, tool.error_type) == (status, error_type), name
        extra = _extra(tool)
        assert extra["wardex.openai_agents.hosted_tool"] is True
        assert extra["wardex.openai_agents.response_id"] == chat.gen_ai.response_id
        assert extra["wardex.openai_agents.tool_call_id_source"] == "response_output_item"
        assert extra["wardex.openai_agents.turn"] == 1
        # The model call's interval: it encloses the wire exchange that carried the tool.
        assert tool.start_time_ns <= chat.start_time_ns
        assert tool.end_time_ns >= chat.end_time_ns
        assert tool.input_data == b"" and tool.output_data == b""
    assert _extra(_one(spans, "execute_tool lookup"))["wardex.openai_agents.mcp.server"] == "docs"
    # The join, from the wire side: every mapped server tool part names a hosted span's id.
    parts = [
        p
        for m in json.loads(_extra(chat)["gen_ai.output.messages"])
        for p in m["parts"]
        if p["type"] == "server_tool_call"
    ]
    assert {p["id"] for p in parts} == {"ws_1", "fs_1", "ci_1", "ig_1", "mcp_1"}
    assert counters.get(_P + "active.hosted_tool") == 6


def test_a_hosted_tool_with_no_outcome_is_unset(agents_env, scenario):
    """An item the provider returned before it finished (`in_progress`, `searching`,
    `incomplete`) reports no outcome: the span says nothing it did not see, neither OK nor
    ERROR."""
    items = [
        ResponseFunctionWebSearch(
            id=f"ws_{status}",
            status=status,
            type="web_search_call",
            action=ActionSearch(type="search"),
        ).model_dump(exclude_none=True)
        for status in ("in_progress", "searching")
    ]
    items.append(
        ResponseFileSearchToolCall(
            id="fs_incomplete", queries=["q"], status="incomplete", type="file_search_call"
        ).model_dump(exclude_none=True)
    )
    scenario(lambda inp: [*items, *_DONE])
    _init()
    try:
        assert _drive("run", _hosted_agent()) == "done"
        spans = _spans()
    finally:
        wardex.close()
    hosted = {s.tool.call_id: s for s in _adapter_spans(spans) if s.tool is not None}
    assert set(hosted) == {"ws_in_progress", "ws_searching", "fs_incomplete"}
    for span in hosted.values():
        assert (span.status, span.error_type) == (StatusCode.UNSET, None), span.tool.call_id


def test_a_tool_search_the_client_must_run_is_not_a_hosted_span(agents_env, scenario):
    """A `tool_search_call` with `execution="client"` is the client's to run, not the
    provider's; the framework's runner refuses it. It is never recorded as a hosted tool."""
    from agents.exceptions import ModelBehaviorError

    item = ResponseToolSearchCall(
        id="ts_1",
        arguments={"query": "x"},
        execution="client",
        status="completed",
        type="tool_search_call",
    ).model_dump(exclude_none=True)
    scenario(lambda inp: [item, *_DONE])
    _init()
    try:
        with pytest.raises(ModelBehaviorError):
            _drive("run", _hosted_agent())
        spans = _spans()
    finally:
        wardex.close()
    assert [s.name for s in _adapter_spans(spans) if s.name.startswith("execute_tool")] == []
    assert counters.get(_P + "active.hosted_tool") == 0


def test_a_local_shell_call_is_the_frameworks_one_span_not_a_hosted_one(agents_env, scenario):
    """A shell call with no output in the same response is the client's: the framework runs it
    through the agent's executor and opens its own function span. It is never doubled by a
    hosted span."""
    from agents.tool import ShellCommandOutput, ShellResult

    call = ResponseFunctionShellToolCall(
        id="sh_1",
        call_id="call_sh",
        action={"commands": ["ls"]},
        status="completed",
        type="shell_call",
    ).model_dump(exclude_none=True)

    def decide(inp: object) -> list[dict]:
        items = inp if isinstance(inp, list) else []
        if any(isinstance(x, dict) and x.get("type") == "shell_call_output" for x in items):
            return _DONE
        return [call]

    def executor(request: Any) -> ShellResult:
        return ShellResult(output=[ShellCommandOutput(stdout="a.txt")])

    scenario(decide)
    agent = Agent(
        name="agent_a", instructions="a", model="gpt-4o-mini", tools=[ShellTool(executor=executor)]
    )
    _init()
    try:
        assert _drive("run", agent) == "done"
        spans = _spans()
    finally:
        wardex.close()
    shells = [s for s in _adapter_spans(spans) if s.name == "execute_tool shell"]
    assert len(shells) == 1
    assert "wardex.openai_agents.hosted_tool" not in _extra(shells[0])
    assert counters.get(_P + "active.hosted_tool") == 0
    assert counters.get(_P + "active.tool") == 1


def test_a_hosted_shell_call_is_a_tool_span_and_a_local_one_is_not_duplicated(agents_env, scenario):
    """A shell call the provider ran comes back WITH its output in the same response: that one
    is hosted and gets a span here. A shell call without its output is the client's to run, gets
    the framework's own function span, and is never doubled."""
    call = ResponseFunctionShellToolCall(
        id="sh_1",
        call_id="call_sh",
        action={"commands": ["ls"]},
        status="completed",
        type="shell_call",
    )
    out = ResponseFunctionShellToolCallOutput(
        id="sho_1",
        call_id="call_sh",
        output=[{"stdout": "a.txt", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}],
        status="completed",
        type="shell_call_output",
    )
    items = [call.model_dump(exclude_none=True), out.model_dump(exclude_none=True), *_DONE]
    scenario(lambda inp: items)
    agent = Agent(
        name="agent_a",
        instructions="a",
        model="gpt-4o-mini",
        tools=[ShellTool(environment={"type": "container_auto"})],
    )
    _init()
    try:
        assert _drive("run", agent) == "done"
        spans = _spans()
    finally:
        wardex.close()
    shell = _one(spans, "execute_tool shell")
    # The item's own id, like every hosted kind; the model's `call_id` keys no wire part.
    assert shell.tool.call_id == "sh_1"
    assert shell.parent_span_id == _one(spans, "invoke_agent agent_a").context.span_id
    assert counters.get(_P + "active.hosted_tool") == 1


def test_a_function_call_in_the_same_response_is_still_matched_not_hosted(agents_env, scenario):
    """Hosted items ride beside function calls in one response. The function call keeps its own
    span and its recovered call id; only the hosted item becomes a hosted span."""
    web = ResponseFunctionWebSearch(
        id="ws_1", status="completed", type="web_search_call", action=ActionSearch(type="search")
    ).model_dump(exclude_none=True)

    def decide(inp: object) -> list[dict]:
        if _outputs_done(inp) == 0:
            return [web, _fc("get_weather", "call_1", '{"city":"Seoul"}')]
        return _DONE

    @function_tool
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    scenario(decide)
    agent = Agent(
        name="agent_a", instructions="a", model="gpt-4o-mini", tools=[WebSearchTool(), get_weather]
    )
    _init()
    try:
        assert _drive("run", agent) == "done"
        spans = _spans()
    finally:
        wardex.close()
    assert _one(spans, "execute_tool get_weather").tool.call_id == "call_1"
    assert _one(spans, "execute_tool web_search").tool.call_id == "ws_1"
    assert counters.get(_P + "active.hosted_tool") == 1
    assert counters.get(_P + "active.tool") == 1


# --------------------------------------------------------------------------
# 2. tracing turned off after install: read at each run's entry
# --------------------------------------------------------------------------


@pytest.fixture
def switch_restored() -> Iterator[Any]:
    """The framework's switch has two cached halves; both come back as they were."""
    provider = get_trace_provider()
    before = (provider._manual_disabled, provider._env_disabled)
    yield provider
    provider._manual_disabled, provider._env_disabled = before
    provider._refresh_disabled_flag()


@pytest.mark.parametrize("entry", ["run", "run_sync", "run_streamed"])
def test_a_run_config_that_disables_tracing_is_counted_and_said_once(agents_env, wardex_log, entry):
    """Before: two runs with `RunConfig(tracing_disabled=True)` left six `chat` spans and nothing
    else, and not a word. Now each such run is counted and the first one says what is missing,
    and a run beside them that keeps tracing on is recorded whole."""
    _init()
    try:
        for _ in range(2):
            assert _drive(entry, _agents(), run_config=RunConfig(tracing_disabled=True)) == "done"
        assert _adapter_spans(_spans()) == []
        assert _drive(entry, _agents(), run_config=RunConfig()) == "done"
        spans = _spans()
    finally:
        wardex.close()
    assert len(_chat_spans(spans)) == 9
    assert {s.name for s in _adapter_spans(spans)} >= {
        "invoke_agent agent_a",
        "invoke_agent agent_b",
    }
    assert counters.get(_P + "tracing_disabled_run") == 2
    [line] = _warnings(wardex_log, "RunConfig(tracing_disabled=True)")
    assert _P + "tracing_disabled_run" in line


def test_a_positional_run_config_is_read_too(agents_env, wardex_log):
    """`run_streamed` takes `run_config` positionally as well; the entry reads its position off
    the signature, so a positional config is seen like a keyword one."""
    _init()
    try:

        async def go() -> None:
            result = Runner.run_streamed(
                _agents(), "hi", None, 10, None, RunConfig(tracing_disabled=True)
            )
            async for _ in result.stream_events():
                pass

        asyncio.run(go())
    finally:
        wardex.close()
    assert counters.get(_P + "tracing_disabled_run") == 1
    assert len(_warnings(wardex_log, "RunConfig(tracing_disabled=True)")) == 1


def test_tracing_switched_off_after_init_is_counted_and_said_once(
    agents_env, wardex_log, switch_restored
):
    """Before: `set_tracing_disabled(True)` after `wardex.init()` silenced the hook for every run
    that followed, and the install-time notice had already passed. Now the first such run says
    so, once, and every one is counted."""
    _init()
    try:
        set_tracing_disabled(True)
        for _ in range(2):
            assert _drive("run", _agents()) == "done"
        spans = _spans()
    finally:
        wardex.close()
    assert _adapter_spans(spans) == []
    assert counters.get(_P + "tracing_disabled_run") == 2
    assert len(_warnings(wardex_log, "turned off after wardex.init()")) == 1


def test_the_environment_switch_read_after_init_is_seen(
    agents_env, wardex_log, switch_restored, monkeypatch
):
    """The framework reads `OPENAI_AGENTS_DISABLE_TRACING` once, when its first trace opens, which
    can be after `wardex.init()`. The entry reads it the same way, so a value set in between is
    seen at the run instead of missed at install."""
    provider = switch_restored
    provider._manual_disabled = None
    provider._env_disabled = None
    monkeypatch.delenv("OPENAI_AGENTS_DISABLE_TRACING", raising=False)
    _init()
    try:
        monkeypatch.setenv("OPENAI_AGENTS_DISABLE_TRACING", "1")
        assert _drive("run", _agents()) == "done"
        spans = _spans()
    finally:
        wardex.close()
    assert _adapter_spans(spans) == []
    assert counters.get(_P + "tracing_disabled_at_install") == 0
    assert counters.get(_P + "tracing_disabled_run") == 1
    assert len(_warnings(wardex_log, "turned off after wardex.init()")) == 1


def test_tracing_off_at_install_is_counted_per_run_but_said_only_once(
    agents_env, wardex_log, switch_restored
):
    """The install-time INFO line already said it; the run that finds the switch still off is
    counted and adds no second line."""
    set_tracing_disabled(True)
    _init()
    try:
        assert _drive("run", _agents()) == "done"
    finally:
        wardex.close()
    assert counters.get(_P + "tracing_disabled_at_install") == 1
    assert counters.get(_P + "tracing_disabled_run") == 1
    assert _warnings(wardex_log, "tracing") == []


def test_a_run_inside_a_disabled_trace_is_said_once(agents_env, wardex_log):
    _init()
    try:
        with trace("host", disabled=True):
            assert _drive("run", _agents()) == "done"
        spans = _spans()
    finally:
        wardex.close()
    assert _adapter_spans(spans) == []
    assert counters.get(_P + "tracing_disabled_run") == 1
    assert len(_warnings(wardex_log, "inside a disabled trace")) == 1


def _web_and_tool(inp: object) -> list[dict]:
    """A web search and a function call in the first response, the answer after."""
    if _outputs_done(inp) == 0:
        web = ResponseFunctionWebSearch(
            id="ws_1",
            status="completed",
            type="web_search_call",
            action=ActionSearch(type="search"),
        ).model_dump(exclude_none=True)
        return [web, _fc("get_weather", "call_1", '{"city":"Seoul"}')]
    return _DONE


def _weather_and_web_agent(name: str = "agent_a") -> Agent:
    @function_tool
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    return Agent(
        name=name, instructions="a", model="gpt-4o-mini", tools=[WebSearchTool(), get_weather]
    )


def test_a_disabling_run_config_inside_an_open_trace_says_exactly_what_is_lost(
    agents_env, scenario, wardex_log
):
    """Inside a trace the host opened, the run's own `tracing_disabled` cannot stop its agent and
    function-tool spans — the framework opens no trace of its own there — but the framework hands
    its model a disabled tracing mode, so no model span ever reaches the adapter: no call id for
    the tool, no response id, no hosted tool span. That is what the line says, and the tool's
    missing call id is counted as this run's loss, never as a failed match."""
    scenario(_web_and_tool)
    _init()
    try:
        with trace("host"):
            cfg = RunConfig(tracing_disabled=True)
            assert _drive("run", _weather_and_web_agent(), run_config=cfg) == "done"
        spans = _spans()
    finally:
        wardex.close()
    tool = _one(spans, "execute_tool get_weather")
    assert tool.tool.call_id is None
    assert counters.get(_P + "tool_call_id_tracing_disabled") == 1
    assert counters.get(_P + "tool_call_id_unmatched") == 0
    assert counters.get(_P + "active.hosted_tool") == 0
    assert "wardex.openai_agents.last_response_id" not in _extra(
        _one(spans, "invoke_agent agent_a")
    )
    assert counters.get(_P + "tracing_disabled_run") == 0
    assert counters.get(_P + "tracing_disabled_run_partial") == 1
    [line] = _warnings(wardex_log, "RunConfig(tracing_disabled=True)")
    for loss in ("no call id", "no response id", "no hosted tool span", "computer, shell"):
        assert loss in line, loss


def test_a_disabled_nested_run_inside_a_traced_one_loses_only_its_own_joins(
    agents_env, scenario, wardex_log
):
    """A tool that runs a nested `Runner.run` with its own `RunConfig(tracing_disabled=True)`,
    inside a traced outer run: the nested run's tool has no call id, counted as that run's
    loss; the outer run's own tool keeps its id."""
    inner = _weather_and_web_agent("inner")

    @function_tool
    async def ask_inner(q: str) -> str:
        result = await Runner.run(inner, q, run_config=RunConfig(tracing_disabled=True))
        return str(result.final_output)

    def decide(inp: object) -> list[dict]:
        items = inp if isinstance(inp, list) else []
        if any(isinstance(x, dict) and x.get("content") == "INNER" for x in items):
            return _web_and_tool(inp)
        if _outputs_done(inp) == 0:
            return [_fc("ask_inner", "call_o", '{"q":"INNER"}')]
        return _DONE

    scenario(decide)
    outer = Agent(name="outer", instructions="o", model="gpt-4o-mini", tools=[ask_inner])
    _init()
    try:
        assert _drive("run", outer) == "done"
        spans = _spans()
    finally:
        wardex.close()
    assert _one(spans, "execute_tool ask_inner").tool.call_id == "call_o"
    assert _one(spans, "execute_tool get_weather").tool.call_id is None
    assert counters.get(_P + "tool_call_id_tracing_disabled") == 1
    assert counters.get(_P + "tool_call_id_unmatched") == 0
    assert counters.get(_P + "tracing_disabled_run_partial") == 1


def test_a_switch_cached_before_init_is_the_installs_notice_not_a_late_change(
    agents_env, wardex_log, switch_restored, monkeypatch
):
    """The framework cached `OPENAI_AGENTS_DISABLE_TRACING=1` when its first trace opened, before
    `wardex.init()`, and the variable is gone since. Install and the run read the same cached
    value: install says tracing is off, and the run is counted without a line claiming the
    switch moved after init."""
    provider = switch_restored
    provider._manual_disabled = None
    provider._env_disabled = True
    provider._refresh_disabled_flag()
    monkeypatch.delenv("OPENAI_AGENTS_DISABLE_TRACING", raising=False)
    _init()
    try:
        assert _drive("run", _agents()) == "done"
        spans = _spans()
    finally:
        wardex.close()
    assert _adapter_spans(spans) == []
    assert counters.get(_P + "tracing_disabled_at_install") == 1
    assert counters.get(_P + "tracing_disabled_run") == 1
    assert _warnings(wardex_log, "tracing") == []


def test_the_switch_turned_off_inside_an_open_host_trace_is_the_switch(
    agents_env, wardex_log, switch_restored
):
    """The framework checks the switch as it creates every span, so a switch turned off while the
    host's own trace is open leaves that trace's root and nothing under it. The reason said is
    the switch, not the trace."""
    _init()
    try:
        with trace("host"):
            set_tracing_disabled(True)
            assert _drive("run", _agents()) == "done"
        spans = _spans()
    finally:
        wardex.close()
    assert [s.name for s in _adapter_spans(spans)] == ["invoke_workflow host"]
    assert counters.get(_P + "tracing_disabled_run") == 1
    assert len(_warnings(wardex_log, "turned off after wardex.init()")) == 1
    assert _warnings(wardex_log, "inside a disabled trace") == []


def _decide_ask_inner(inp: object) -> list[dict]:
    """The outer agent asks its inner agent (a tool) once; both answer at once after that."""
    items = inp if isinstance(inp, list) else []
    if any(isinstance(x, dict) and x.get("content") == "INNER" for x in items):
        return _DONE
    if _outputs_done(inp) == 0:
        return [_fc("ask_inner", "call_1", '{"input":"INNER"}')]
    return _DONE


def test_a_nested_run_inside_an_untraced_run_is_counted_without_a_second_line(
    agents_env, scenario, wardex_log
):
    """An agent used as a tool runs a nested `Runner.run`, inside the outer run's disabled
    trace and under its run config. The outer call's entry already said why nothing is
    recorded; the nested run is counted, and says nothing more."""
    scenario(_decide_ask_inner)
    inner = Agent(name="inner", instructions="i", model="gpt-4o-mini")
    outer = Agent(
        name="outer",
        instructions="o",
        model="gpt-4o-mini",
        tools=[inner.as_tool(tool_name="ask_inner", tool_description="asks")],
    )
    _init()
    try:
        assert _drive("run", outer, run_config=RunConfig(tracing_disabled=True)) == "done"
        spans = _spans()
    finally:
        wardex.close()
    assert _adapter_spans(spans) == []
    assert counters.get(_P + "tracing_disabled_run") == 2
    assert len(_warnings(wardex_log, "tracing")) == 1


def test_a_run_with_tracing_on_says_nothing(agents_env, wardex_log):
    _init()
    try:
        assert _drive("run", _agents()) == "done"
    finally:
        wardex.close()
    assert counters.get(_P + "tracing_disabled_run") == 0
    assert counters.get(_P + "tracing_disabled_run_partial") == 0
    assert wardex_log.lines(logging.WARNING) == []


# --------------------------------------------------------------------------
# 3. sensitive data off: what the framework withheld says it was withheld
# --------------------------------------------------------------------------


def test_sensitive_data_off_marks_the_tool_and_the_agent_as_withheld(agents_env):
    """Before: the tool span's empty payload, its missing call id and the agent's missing response
    id read like a tool that took nothing and a response that never existed. Now the tool and both
    agents say the framework withheld it; the handoff marker and the root, which carry no
    content, do not."""
    _init()
    try:
        cfg = RunConfig(trace_include_sensitive_data=False)
        assert _drive("run", _agents(), run_config=cfg) == "done"
        spans = _spans()
    finally:
        wardex.close()
    key = "wardex.openai_agents.sensitive_data_withheld"
    for name in ("execute_tool get_weather", "invoke_agent agent_a", "invoke_agent agent_b"):
        assert _extra(_one(spans, name))[key] is True, name
    for name in ("handoff agent_a→agent_b", "invoke_workflow Agent workflow"):
        assert key not in _extra(_one(spans, name)), name
    assert counters.get(_P + "model_output_withheld") == 3


def test_sensitive_data_on_marks_nothing(agents_env):
    _init()
    try:
        assert _drive("run", _agents()) == "done"
        spans = _spans()
    finally:
        wardex.close()
    for s in _adapter_spans(spans):
        assert "wardex.openai_agents.sensitive_data_withheld" not in _extra(s), s.name
    assert counters.get(_P + "model_output_withheld") == 0


def test_a_failed_model_call_is_not_read_as_withheld(agents_env, scenario):
    """A model span that failed carries no response either. That is a failure, not the
    framework withholding content, and it is not marked as one."""

    def broken(inp: object) -> list[dict]:
        raise RuntimeError("the fake server fails the call")

    scenario(broken)
    _init()
    try:
        with pytest.raises(Exception):  # noqa: B017 — whichever error the client raises
            _drive("run", Agent(name="agent_a", instructions="a", model="gpt-4o-mini"))
        spans = _spans()
    finally:
        wardex.close()
    agent = _one(spans, "invoke_agent agent_a")
    assert "wardex.openai_agents.sensitive_data_withheld" not in _extra(agent)
    assert counters.get(_P + "model_output_withheld") == 0


def test_a_cancelled_model_call_is_not_read_as_withheld(agents_env, scenario):
    """An input guardrail that trips while the first model call is still in flight cancels that
    call. The framework closes its response span with nothing on it and no error — the same
    shape sensitive data off leaves — but with the cancellation in flight. That is an
    interruption, not content withheld, and it is not marked as one."""
    import time

    from agents import GuardrailFunctionOutput, input_guardrail
    from agents.exceptions import InputGuardrailTripwireTriggered

    @input_guardrail
    async def block_input(ctx: Any, agent: Any, inp: Any) -> GuardrailFunctionOutput:
        await asyncio.sleep(0.3)  # the model call is in flight by now
        return GuardrailFunctionOutput(output_info=None, tripwire_triggered=True)

    def slow(inp: object) -> list[dict]:
        time.sleep(1.0)
        return _DONE

    scenario(slow)
    agent = Agent(
        name="agent_a", instructions="a", input_guardrails=[block_input], model="gpt-4o-mini"
    )
    _init()
    try:
        with pytest.raises(InputGuardrailTripwireTriggered):
            _drive("run", agent)
        spans = _spans()
    finally:
        wardex.close()
    agent_span = _one(spans, "invoke_agent agent_a")
    assert "wardex.openai_agents.sensitive_data_withheld" not in _extra(agent_span)
    assert counters.get(_P + "model_output_withheld") == 0


def _sse_cut(r: dict) -> bytes:
    """A stream that ends after its items with no terminal event: a cut connection, a proxy."""
    created = {**r, "status": "in_progress", "output": [], "usage": None}
    events = [
        (
            "response.created",
            {"type": "response.created", "sequence_number": 0, "response": created},
        )
    ]
    for i, item in enumerate(r["output"]):
        for name in ("response.output_item.added", "response.output_item.done"):
            events.append(
                (name, {"type": name, "sequence_number": 1, "output_index": i, "item": item})
            )
    return "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events).encode()


def test_a_stream_cut_before_its_end_is_not_read_as_withheld(agents_env, scenario, monkeypatch):
    """A streamed call whose stream ends without its terminal event leaves the framework's
    response span with no response and no error, and the framework raises only after that span
    closed. Sensitive data is on, so nothing was withheld, and nothing is marked: the mark is
    read off the run's config, never off a missing response."""
    from agents.exceptions import ModelBehaviorError

    import test_openai_agents_wire as wire

    scenario(lambda inp: _DONE)
    monkeypatch.setattr(wire, "_sse", _sse_cut)
    _init()
    try:
        with pytest.raises(ModelBehaviorError):
            _drive("run_streamed", Agent(name="agent_a", instructions="a", model="gpt-4o-mini"))
        spans = _spans()
    finally:
        wardex.close()
    agent = _one(spans, "invoke_agent agent_a")
    assert "wardex.openai_agents.sensitive_data_withheld" not in _extra(agent)
    assert counters.get(_P + "model_output_withheld") == 0


def test_hosted_tools_under_sensitive_data_off_are_said_on_the_agent(agents_env, scenario):
    """With the response withheld the hosted items cannot be read in process; the agent says
    so, and the wire span still holds them."""
    scenario(lambda inp: _hosted_items())
    _init()
    try:
        cfg = RunConfig(trace_include_sensitive_data=False)
        assert _drive("run", _hosted_agent(), run_config=cfg) == "done"
        spans = _spans()
    finally:
        wardex.close()
    assert counters.get(_P + "active.hosted_tool") == 0
    agent = _one(spans, "invoke_agent agent_a")
    assert _extra(agent)["wardex.openai_agents.sensitive_data_withheld"] is True
    (chat,) = _chat_spans(spans)
    assert "ws_1" in _extra(chat)["gen_ai.output.messages"]


# --------------------------------------------------------------------------
# 4. a Chat Completions model, and the span kinds past the first mapping
# --------------------------------------------------------------------------


def _chat_reply(req: dict) -> dict:
    answered = any(m.get("role") == "tool" for m in req.get("messages") or [])
    if answered:
        message: dict = {"role": "assistant", "content": "done"}
        finish = "stop"
    else:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_cc_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Seoul"}'},
                }
            ],
        }
        finish = "tool_calls"
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def _chat_stream(reply: dict) -> bytes:
    message = reply["choices"][0]["message"]
    delta: dict = {"role": "assistant"}
    if message.get("tool_calls"):
        delta["tool_calls"] = [{"index": 0, **tc} for tc in message["tool_calls"]]
    else:
        delta["content"] = message["content"]
    head = {k: reply[k] for k in ("id", "created", "model")}
    chunks = [
        {**head, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta}]},
        {
            **head,
            "object": "chat.completion.chunk",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": reply["choices"][0]["finish_reason"]}
            ],
        },
        {**head, "object": "chat.completion.chunk", "choices": [], "usage": reply["usage"]},
    ]
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


@pytest.fixture
def fake_chat() -> Iterator[str]:
    """A loopback Chat Completions API: a tool call first, then the answer once the tool's
    result is in the request. HTTP/1.0, for the reason the Responses fake gives."""

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 — the stdlib's handler name
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            reply = _chat_reply(req)
            if req.get("stream"):
                body, ct = _chat_stream(reply), "text/event-stream"
            else:
                body, ct = json.dumps(reply).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _chat_agent(base: str) -> Agent:
    @function_tool
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    model = OpenAIChatCompletionsModel(
        model="gpt-4o-mini", openai_client=AsyncOpenAI(base_url=base, api_key="sk-test")
    )
    return Agent(name="agent_a", instructions="a", tools=[get_weather], model=model)


@pytest.mark.parametrize("entry", ["run", "run_streamed"])
def test_a_chat_completions_tool_span_recovers_its_call_id(agents_env, fake_chat, entry):
    """Before: a Chat Completions model's turns are `GenerationSpanData`, which the adapter
    ignored, so its tool span shipped with no call id and the "unavailable" marker. Now the
    calls are read off the generation's output — the message's `tool_calls`, or a streamed
    run's assembled `Response` — and the tool span carries the id the next request echoes."""
    _init()
    try:
        assert _drive(entry, _chat_agent(fake_chat)) == "done"
        spans = _spans()
    finally:
        wardex.close()
    tool = _one(spans, "execute_tool get_weather")
    assert tool.tool.call_id == "call_cc_1"
    assert _extra(tool)["wardex.openai_agents.tool_call_id_source"] == "generation_output_match"
    assert "wardex.openai_agents.response_id" not in _extra(tool)
    assert tool.capture_integrity is None or tool.capture_integrity.limitations == ()
    assert counters.get(_P + "span_kind_ignored") == 0
    assert counters.get(_P + "tool_call_id_unmatched") == 0
    assert len(_chat_spans(spans)) == 2


def test_a_chat_completions_run_with_sensitive_data_off_is_marked_withheld(agents_env, fake_chat):
    _init()
    try:
        cfg = RunConfig(trace_include_sensitive_data=False)
        assert _drive("run", _chat_agent(fake_chat), run_config=cfg) == "done"
        spans = _spans()
    finally:
        wardex.close()
    tool = _one(spans, "execute_tool get_weather")
    assert tool.tool.call_id is None
    key = "wardex.openai_agents.sensitive_data_withheld"
    assert _extra(tool)[key] is True
    assert _extra(_one(spans, "invoke_agent agent_a"))[key] is True


def test_a_custom_span_inside_a_tool_is_a_step_under_that_tool(agents_env, scenario):
    """Before: `custom_span(...)` — the framework's documented way for a host to add its own
    structure — was counted as an ignored kind and dropped. Now it is an `execute_step` named
    by its opener, under the tool it ran in, with its `data` as the step's input."""

    @function_tool
    def get_weather(city: str) -> str:
        with custom_span("db_query", {"table": "users"}):
            return f"sunny in {city}"

    def decide(inp: object) -> list[dict]:
        if _outputs_done(inp) == 0:
            return [_fc("get_weather", "call_1", '{"city":"Seoul"}')]
        return _DONE

    scenario(decide)
    agent = Agent(name="agent_a", instructions="a", model="gpt-4o-mini", tools=[get_weather])
    _init()
    try:
        assert _drive("run", agent) == "done"
        spans = _spans()
    finally:
        wardex.close()
    step = _one(spans, "execute_step db_query")
    assert step.parent_span_id == _one(spans, "execute_tool get_weather").context.span_id
    assert _extra(step)["wardex.step.name"] == "db_query"
    assert step.input_data == b"{'table': 'users'}"
    assert step.status is StatusCode.OK
    assert counters.get(_P + "span_kind_ignored") == 0


def test_a_failed_custom_span_is_an_error_step(agents_env, wardex_log):
    """The error is its opener's own words, so it is typed generically and nothing reports it as
    a framework sentence this adapter failed to map."""
    from agents.tracing import SpanError

    _init()
    try:
        with trace("wf"):
            with custom_span("risky") as span:
                span.set_error(SpanError(message="it broke", data={}))
        spans = _spans()
    finally:
        wardex.close()
    step = _one(spans, "execute_step risky")
    assert (step.status, step.error_type) == (StatusCode.ERROR, "openai_agents_error")
    assert wardex_log.lines(logging.WARNING) == []
    assert counters.get(_P + "error_message_unmapped") == 0


def test_the_voice_pipelines_spans_are_unpinned_steps(agents_env):
    """Speech, speech-group and transcription spans are steps of their own. They are NOT pinned —
    the voice pipeline finishes some of them on other tasks — so one opened inside another is its
    sibling under the run, not its child."""
    _init()
    try:
        with trace("voice"):
            with speech_group_span():
                with speech_span(model="tts-1"):
                    pass
            with transcription_span(model="whisper-1"):
                pass
        spans = _spans()
    finally:
        wardex.close()
    root = _one(spans, "invoke_workflow voice")
    for word in ("speech_group", "speech", "transcription"):
        step = _one(spans, f"execute_step {word}")
        assert _extra(step)["wardex.step.name"] == word
        assert step.parent_span_id == root.context.span_id, word
    assert counters.get(_P + "span_kind_ignored") == 0


def test_a_runs_task_span_is_folded_into_its_root_not_counted_as_ignored(agents_env):
    """Every `Runner` run opens a task span; it is the run itself, which the root already is."""
    _init()
    try:
        assert _drive("run", _agents()) == "done"
    finally:
        wardex.close()
    assert counters.get(_P + "span_kind_ignored") == 0


class _HostSpanData(SpanData):
    """A span kind no release of the framework has: what a host's own subclass looks like."""

    @property
    def type(self) -> str:
        return "host_kind"

    def export(self) -> dict[str, Any]:
        return {"type": self.type}


def test_a_kind_nothing_maps_is_counted_and_said_once_by_its_class_name(agents_env, wardex_log):
    _init()
    try:
        with trace("wf"):
            for _ in range(2):
                span = get_trace_provider().create_span(_HostSpanData())
                span.start(mark_as_current=True)
                span.finish(reset_current=True)
    finally:
        wardex.close()
    assert counters.get(_P + "span_kind_ignored") == 2
    [line] = _warnings(wardex_log, "does not map")
    assert "_HostSpanData" in line and _P + "span_kind_ignored" in line


# --------------------------------------------------------------------------
# 5. a span this adapter could not place: counted and said once
# --------------------------------------------------------------------------


def _start_elsewhere(span: Any) -> None:
    """Start `span` on a thread of its own: a fresh context, where no trace is current."""
    t = threading.Thread(target=span.start)
    t.start()
    t.join()


def test_a_span_whose_trace_is_not_current_is_counted_and_said_once(agents_env, wardex_log):
    """Before: only the counter moved. Now the first such span says it was not recorded."""
    _init()
    try:
        with trace("wf"):
            for _ in range(2):
                span = custom_span("elsewhere")
                _start_elsewhere(span)
                span.finish()
    finally:
        wardex.close()
    assert counters.get(_P + "trace_lookup_miss") == 2
    assert len(_warnings(wardex_log, _P + "trace_lookup_miss")) == 1


def test_a_tool_span_lost_at_its_start_is_said_once_not_twice(agents_env, wardex_log):
    """The end of a tool span whose start was lost finds nothing to close. That end is counted,
    and the loss is said ONCE — by its start, the place the span was lost."""
    _init()
    try:
        with trace("wf"):
            span = function_span("get_weather")
            _start_elsewhere(span)
            span.finish()
    finally:
        wardex.close()
    assert counters.get(_P + "trace_lookup_miss") == 1
    assert counters.get(_P + "tool_end_unmatched") == 1
    assert len(wardex_log.lines(logging.WARNING)) == 1
    assert len(_warnings(wardex_log, _P + "trace_lookup_miss")) == 1


def test_a_span_from_a_task_the_run_left_behind_is_said_with_its_own_reason(
    agents_env, scenario, wardex_log
):
    """A tool starts a background task that opens a custom span after the run has ended. Its run
    root is closed, so the span cannot be placed; the line says the run had ended, not that wardex
    never saw it start."""

    async def late() -> None:
        await asyncio.sleep(0.3)
        with custom_span("late_audit"):
            pass

    @function_tool
    async def get_weather(city: str) -> str:
        asyncio.get_running_loop().create_task(late())
        return f"sunny in {city}"

    def decide(inp: object) -> list[dict]:
        if _outputs_done(inp) == 0:
            return [_fc("get_weather", "call_1", '{"city":"Seoul"}')]
        return _DONE

    scenario(decide)
    agent = Agent(name="agent_a", instructions="a", model="gpt-4o-mini", tools=[get_weather])
    _init()
    try:

        async def go() -> None:
            assert (await Runner.run(agent, "hi")).final_output == "done"
            await asyncio.sleep(0.6)

        asyncio.run(go())
    finally:
        wardex.close()
    assert counters.get(_P + "span_after_run") == 1
    assert counters.get(_P + "span_without_run") == 0
    assert len(_warnings(wardex_log, _P + "span_after_run")) == 1
    assert _warnings(wardex_log, "never saw") == []


def test_spans_of_a_run_wardex_never_saw_start_are_said_once(agents_env, scenario, wardex_log):
    """`wardex.init()` inside a host's open trace: the run's spans arrive for a root that never
    opened. Before, counted only; now the first one says so."""
    scenario(lambda inp: _DONE)
    agent = Agent(name="agent_a", instructions="a", model="gpt-4o-mini")
    with trace("outer"):
        _init()
        try:
            assert Runner.run_sync(agent, "hi").final_output == "done"
        finally:
            wardex.close()
    assert counters.get(_P + "span_without_run") >= 2
    assert len(_warnings(wardex_log, _P + "span_without_run")) == 1
