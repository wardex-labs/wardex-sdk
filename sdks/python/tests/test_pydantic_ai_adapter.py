"""The pydantic-ai bridge's scenarios beyond the shared conformance invariants.

Driven through the real `AdapterRegistry` (`installed_adapter`) with `TestModel` and
`FunctionModel`, so no HTTP happens and every span shipped is a bridge unit. Most tests run in
both instrumentation modes; the one that cannot happen in Mode U (a chat span that ends before it
named its model, which only Mode W's attribute callback can observe) says so. The shapes are the
measured span dumps in `tests/fixtures/pydantic_ai/`: each test names the scenario it re-runs.
"""

from __future__ import annotations

import asyncio
import gc
import importlib.util
import json
import subprocess
import sys
import textwrap
import threading
from typing import Any

import pytest
from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.capabilities.hooks import Hooks
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelHTTPError, ModelRetry
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from _pydantic_ai_support import (
    MODES,
    extras,
    fake_openai,
    host_provider,
    instrumentation_mode,
    named,
    one,
    openai_model,
    wardex_owned,
)
from wardex_sdk._adapters._pydantic_ai import PydanticAiAdapter
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._enums import AgentType, StatusCode
from wardex_sdk.testing import installed_adapter

_DEMOTED = "wardex.pydantic_ai.otel."

#: Whether this pydantic-ai records a FallbackModel's failed attempts as spans.
_EMITS_ATTEMPTS = importlib.util.find_spec("pydantic_ai._model_request_attempts") is not None


@pytest.fixture(params=MODES)
def instrumentation(request):
    with instrumentation_mode(request.param) as setup:
        yield setup


@pytest.fixture
def live(instrumentation):
    with installed_adapter(PydanticAiAdapter) as installed:
        yield installed


def _limitations(span: Any) -> tuple[Limitation, ...]:
    return tuple(span.capture_integrity.limitations) if span.capture_integrity else ()


def _parent(spans: list[Any], span: Any) -> Any:
    by_id = {s.context.span_id: s for s in spans}
    return by_id.get(span.parent_span_id)


def _seen(messages: list[Any], kind: type) -> bool:
    return any(isinstance(p, kind) for m in messages for p in getattr(m, "parts", ()))


# -- tree shapes -------------------------------------------------------------------


def test_a_run_with_no_tools_is_a_run_over_its_chat(live):
    """`a_run_sync_no_tools`: one root, one chat under it, one trace."""
    Agent(TestModel(), name="plain").run_sync("say hi")
    spans = live.spans
    run = one(spans, "invoke_agent plain")
    chat = one(spans, "chat test")
    assert run.parent_span_id is None
    assert chat.parent_span_id == run.context.span_id
    assert run.agent.name == "plain"
    assert run.agent.agent_type is AgentType.PRIMARY
    assert extras(run)["wardex.framework"] == "pydantic_ai"
    assert extras(run)["wardex.pydantic_ai.schema_version"] == 5
    assert run.conversation is not None, "the framework's conversation id is the run's"
    assert run.status is StatusCode.OK
    assert len({s.context.trace_id for s in spans}) == 1


def test_a_tool_call_is_a_tool_unit_with_its_arguments_and_result(live):
    """`b_one_tool`: the tool sits under the run beside the two chats, with its call id."""
    agent = Agent(TestModel(call_tools=["add"]), name="adder")

    @agent.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    agent.run_sync("add")
    spans = live.spans
    run = one(spans, "invoke_agent adder")
    tool = one(spans, "execute_tool add")
    assert len(named(spans, "chat test")) == 2
    assert tool.parent_span_id == run.context.span_id
    assert tool.tool.name == "add" and tool.tool.call_id
    assert json.loads(tool.input_data) == {"a": 0, "b": 0}
    assert tool.output_data == b"0"
    assert tool.status is StatusCode.OK


def test_parallel_tools_are_siblings_under_the_run(live):
    """`f2_parallel_tools`: two calls in one response run concurrently; each is the run's child."""

    async def answer(messages: list[Any], info: Any) -> ModelResponse:
        if _seen(messages, ToolReturnPart):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(
            parts=[ToolCallPart("left", {}, tool_call_id="c1"), ToolCallPart("right", {}, "c2")]
        )

    agent = Agent(FunctionModel(answer, model_name="m"), name="parallel")

    @agent.tool_plain
    async def left() -> str:
        await asyncio.sleep(0.01)
        return "l"

    @agent.tool_plain
    async def right() -> str:
        await asyncio.sleep(0.01)
        return "r"

    agent.run_sync("go")
    spans = live.spans
    run = one(spans, "invoke_agent parallel")
    for name, call_id in (("left", "c1"), ("right", "c2")):
        tool = one(spans, f"execute_tool {name}")
        assert tool.parent_span_id == run.context.span_id
        assert tool.tool.call_id == call_id


def test_a_run_inside_a_tool_is_a_sub_agent_of_the_outer_run(live):
    """`f_nested_agents`: the inner run sits under the tool that started it."""
    inner = Agent(TestModel(), name="inner")
    outer = Agent(TestModel(call_tools=["delegate"]), name="outer")

    @outer.tool_plain
    async def delegate() -> str:
        return (await inner.run("go")).output

    outer.run_sync("go")
    spans = live.spans
    outer_run = one(spans, "invoke_agent outer")
    inner_run = one(spans, "invoke_agent inner")
    tool = one(spans, "execute_tool delegate")
    assert inner_run.parent_span_id == tool.context.span_id
    assert tool.parent_span_id == outer_run.context.span_id
    assert inner_run.agent.agent_type is AgentType.SUB_AGENT
    assert inner_run.agent.parent_agent == "outer"
    assert outer_run.agent.agent_type is AgentType.PRIMARY
    assert outer_run.agent.parent_agent is None


# -- outcomes ----------------------------------------------------------------------


def test_a_model_retry_is_a_failed_tool_call_and_the_run_goes_on(live):
    """`d1_tool_model_retry`: the first call fails as `ToolRetryError`, the second succeeds."""
    agent = Agent(TestModel(call_tools=["flaky"]), name="retrying")
    calls = []

    @agent.tool_plain
    def flaky() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise ModelRetry("try again")
        return "fine"

    agent.run_sync("go")
    spans = live.spans
    tools = named(spans, "execute_tool flaky")
    assert [(t.status, t.error_type) for t in tools] == [
        (StatusCode.ERROR, "ToolRetryError"),
        (StatusCode.OK, None),
    ]
    assert one(spans, "invoke_agent retrying").status is StatusCode.OK


def test_a_tool_that_raises_fails_the_tool_and_the_run_and_quotes_no_message(live):
    """`d2_tool_runtime_error`: the type travels, the raised message never does."""
    agent = Agent(TestModel(call_tools=["boom"]), name="failing")

    @agent.tool_plain
    def boom() -> str:
        raise RuntimeError("secret=abc")

    with pytest.raises(RuntimeError):
        agent.run_sync("go")
    spans = live.spans
    tool = one(spans, "execute_tool boom")
    run = one(spans, "invoke_agent failing")
    assert (tool.status, tool.error_type) == (StatusCode.ERROR, "RuntimeError")
    assert (run.status, run.error_type) == (StatusCode.ERROR, "RuntimeError")
    for span in spans:
        assert "secret=abc" not in span.status_message
        assert all("secret=abc" not in str(v) for _, v in span.extra)


@pytest.mark.parametrize("raised", [CallDeferred, ApprovalRequired], ids=["deferred", "approval"])
def test_a_deferred_tool_call_is_control_flow_not_a_failure(live, raised):
    """`e1_call_deferred`, `e2_approval_required`: the call is handed back to the caller; the
    tool and its run close UNSET and say why."""
    agent = Agent(
        TestModel(call_tools=["later"]), name="deferring", output_type=[str, DeferredToolRequests]
    )

    @agent.tool_plain
    def later() -> str:
        raise raised()

    assert isinstance(agent.run_sync("go").output, DeferredToolRequests)
    spans = live.spans
    tool = one(spans, "execute_tool later")
    run = one(spans, "invoke_agent deferring")
    assert (tool.status, tool.error_type) == (StatusCode.UNSET, None)
    assert extras(tool)["wardex.pydantic_ai.tool.deferral"] == raised.__name__
    assert (run.status, run.error_type) == (StatusCode.UNSET, None)
    assert extras(run)["wardex.pydantic_ai.run.deferred"] is True


def test_a_tool_that_requires_approval_up_front_emits_no_tool_span(live):
    """`e3_requires_approval_flag`: pydantic never opens a span for it, so nothing is made up."""
    agent = Agent(
        TestModel(call_tools=["gated"]), name="gating", output_type=[str, DeferredToolRequests]
    )

    @agent.tool_plain(requires_approval=True)
    def gated() -> str:
        return "never"

    agent.run_sync("go")
    spans = live.spans
    assert named(spans, "execute_tool gated") == []
    one(spans, "invoke_agent gating")


# -- usage -------------------------------------------------------------------------


def test_with_no_wire_request_the_chat_unit_carries_the_framework_usage_and_says_so(live):
    """No interceptor saw a byte, so the core promotes the demoted numbers into the typed block
    and marks `no_wire_evidence`; the run's aggregate stays demoted, so nothing is summed twice."""
    result = Agent(TestModel(call_tools=["add"]), name="counted")

    @result.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    outcome = result.run_sync("go")
    reported = [
        (m.usage.input_tokens, m.usage.output_tokens)
        for m in outcome.all_messages()
        if isinstance(m, ModelResponse)
    ]
    spans = live.spans
    chats = named(spans, "chat test")
    assert [(c.gen_ai.input_tokens, c.gen_ai.output_tokens) for c in chats] == reported
    for chat in chats:
        assert Limitation.NO_WIRE_EVIDENCE in _limitations(chat)
        left = extras(chat)
        assert _DEMOTED + "gen_ai.usage.input_tokens" not in left
        assert _DEMOTED + "gen_ai.usage.output_tokens" not in left
        assert not any(k.startswith("gen_ai.usage.") for k in left)
    run = one(spans, "invoke_agent counted")
    assert run.gen_ai is None
    total = extras(run)[_DEMOTED + "gen_ai.aggregated_usage.input_tokens"]
    assert total == sum(i for i, _ in reported)
    assert counters.get("assembly._usage_hold.usage_promoted") == len(chats)


# -- lifetimes ---------------------------------------------------------------------


def test_a_cancelled_run_waits_for_its_tool_and_neither_is_force_closed(live):
    """`l_cancel_mid_tool`: the run's end arrives before its tool's. The run is held until the
    tool closes on its own end, so the tool is not marked unfinished and the run ships after."""
    agent = Agent(TestModel(call_tools=["slow"]), name="cancelled")

    @agent.tool_plain
    async def slow() -> str:
        await asyncio.sleep(5)
        return "late"

    async def main() -> None:
        await asyncio.wait_for(agent.run("go"), 0.2)

    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        asyncio.run(main())
    spans = live.spans
    tool = one(spans, "execute_tool slow")
    run = one(spans, "invoke_agent cancelled")
    assert (tool.status, tool.error_type) == (StatusCode.ERROR, "CancelledError")
    assert Limitation.CHILD_SPAN_UNCLOSED not in _limitations(tool)
    assert run.status is StatusCode.UNSET
    assert counters.get("adapters.pydantic_ai.held_close") == 1
    assert run.end_time_ns >= tool.end_time_ns


def test_a_run_cancelled_while_its_chat_awaits_a_request_hook_waits_for_the_chat(live):
    """The run is cancelled while its `chat` span is open but has not named its model yet (a
    `before_model_request` hook is awaiting). In Mode W that chat is not a unit yet, and the run
    still waits for it: the chat ships under the run, and neither claims an outcome."""
    hooks = Hooks()

    @hooks.on.before_model_request
    async def stall(ctx: Any, request_context: Any) -> Any:
        await asyncio.sleep(5)

    agent = Agent(TestModel(), name="pending", capabilities=[hooks])

    async def main() -> None:
        await asyncio.wait_for(agent.run("go"), 0.3)

    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        asyncio.run(main())
    spans = live.spans
    run = one(spans, "invoke_agent pending")
    chat = one(spans, "chat test")
    assert chat.parent_span_id == run.context.span_id
    assert run.status is StatusCode.UNSET
    assert (chat.status, chat.error_type) == (StatusCode.UNSET, None)
    assert counters.get("adapters.pydantic_ai.held_close") == 1
    assert run.end_time_ns >= chat.end_time_ns


def test_a_model_call_cancelled_in_flight_is_not_reported_as_answered(live):
    """`wait_for` cuts the run while the model's HTTP request is waiting for its answer.
    pydantic leaves that chat span UNSET with no exception; the unit must not say OK."""
    with fake_openai(delay=1.0) as (base_url, posts):
        agent = Agent(openai_model(base_url), name="cut")

        async def main() -> None:
            await asyncio.wait_for(agent.run("go"), 0.3)

        with pytest.raises((asyncio.TimeoutError, TimeoutError)):
            asyncio.run(main())
        assert len(posts) == 1, "the request was in flight when the run was cut"
    spans = live.spans
    run = one(spans, "invoke_agent cut")
    chat = one(spans, "chat gpt-4o-mini")
    assert chat.parent_span_id == run.context.span_id
    assert run.status is StatusCode.UNSET
    assert (chat.status, chat.error_type) == (StatusCode.UNSET, None)
    assert counters.get("adapters.pydantic_ai.held_close") == 1


def test_a_stream_dropped_without_closing_still_ships_once(live):
    """`run_stream_sync` without `with`: pydantic ends the spans from the collector, on
    whatever frame drops the last reference. Nothing raises, and each unit ships once."""
    agent = Agent(TestModel(), name="dropped")
    stream = agent.run_stream_sync("go")
    assert "".join(stream.stream_text(delta=True))
    del stream
    gc.collect()
    spans = live.spans
    assert len(named(spans, "invoke_agent dropped")) == 1
    assert len(named(spans, "chat test")) == 1


def test_callbacks_after_uninstall_are_counted_and_ship_nothing(live, instrumentation):
    """A run still in flight when wardex closes keeps calling the tracer it resolved at its
    start (Mode W): every later callback is a counted no-op. In Mode U the processor tuple was
    handed back, so the host's provider no longer calls wardex at all."""
    gate, entered = threading.Event(), threading.Event()
    agent = Agent(TestModel(call_tools=["wait"]), name="outlived")

    @agent.tool_plain
    async def wait() -> str:
        entered.set()
        await asyncio.to_thread(gate.wait, 10)
        return "ok"

    thread = threading.Thread(target=agent.run_sync, args=("go",), daemon=True)
    thread.start()
    assert entered.wait(10)
    live.teardown()
    shipped = len(live.spans)
    gate.set()
    thread.join(15)
    assert len(live.spans) == shipped
    late = counters.get("adapters.pydantic_ai.late_callback")
    assert late >= 1 if instrumentation.mode == "wardex_owned" else late == 0
    run = one(live.spans, "invoke_agent outlived")
    assert Limitation.ADAPTER_UNINSTALLED in _limitations(run)


def test_a_span_wardex_does_not_record_is_counted_and_its_work_nests_under_the_ambient(live):
    """A host's own span on pydantic's tracer (`RunContext.tracer`) is not one of pydantic's
    three; it ships nothing of its own and is counted."""
    agent = Agent(TestModel(call_tools=["mine"]), name="hosting")

    @agent.tool
    def mine(ctx: RunContext[None]) -> str:
        with ctx.tracer.start_as_current_span("mine"):
            return "ok"

    agent.run_sync("go")
    spans = live.spans
    assert named(spans, "mine") == []
    assert counters.get("adapters.pydantic_ai.unknown_span") == 1
    assert counters.get("adapters.pydantic_ai.unknown_attribute") == 0


def test_a_failed_fallback_attempt_is_an_error_child_of_the_chat_unit(live):
    """A FallbackModel's failed attempt is its own span, parented to the `chat` span and opened
    after the fact with both instants: a closed ERROR chat child of the chat unit.

    pydantic-ai emits attempt spans only from the release that added
    `pydantic_ai._model_request_attempts` (present on 2.66, absent on 2.54, the newest 2.x the
    3.10 floor resolves). Before it, the failed attempt is in no span at all, and the bridge
    makes none up: the run is then a run over one OK chat."""

    def bad(messages: list[Any], info: Any) -> ModelResponse:
        raise ModelHTTPError(status_code=500, model_name="bad")

    def good(messages: list[Any], info: Any) -> ModelResponse:
        return ModelResponse(parts=[TextPart("ok")])

    model = FallbackModel(
        FunctionModel(bad, model_name="bad"), FunctionModel(good, model_name="good")
    )
    Agent(model, name="falling").run_sync("go")
    spans = live.spans
    run = one(spans, "invoke_agent falling")
    attempts = [s for s in spans if s.status is StatusCode.ERROR]
    if not _EMITS_ATTEMPTS:
        assert attempts == []
        (chat,) = [s for s in spans if s.parent_span_id == run.context.span_id]
        assert chat.status is StatusCode.OK
        return
    assert len(attempts) == 1
    attempt = attempts[0]
    chat = _parent(spans, attempt)
    assert chat.parent_span_id == run.context.span_id
    assert chat.gen_ai is not None and chat.status is StatusCode.OK
    assert (attempt.name, attempt.error_type) == ("chat bad", "ModelHTTPError")
    assert chat.start_time_ns <= attempt.start_time_ns <= attempt.end_time_ns <= chat.end_time_ns
    assert extras(attempt)["wardex.pydantic_ai.model_request.attempt"] == 0


@pytest.mark.parametrize("instrumentation", ["wardex_owned"], indirect=True)
def test_a_chat_span_that_ends_before_naming_its_model_is_opened_at_its_end(live):
    """A request hook that raises ends the `chat` span before pydantic names the model, which
    only Mode W's attribute callback can see. The chat is opened then, as a closed child of the
    run, and says that nothing was sent under it."""
    hooks = Hooks()

    @hooks.on.before_model_request
    async def refuse(ctx: Any, request_context: Any) -> Any:
        raise RuntimeError("no")

    agent = Agent(TestModel(), name="hooked", capabilities=[hooks])
    with pytest.raises(RuntimeError):
        agent.run_sync("go")
    spans = live.spans
    run = one(spans, "invoke_agent hooked")
    chat = one(spans, "chat test")
    assert chat.parent_span_id == run.context.span_id
    assert (chat.status, chat.error_type) == (StatusCode.ERROR, "RuntimeError")
    assert Limitation.NO_WIRE_EVIDENCE in _limitations(chat)
    assert counters.get("adapters.pydantic_ai.chat_opened_at_end") == 1


@pytest.mark.parametrize("instrumentation", ["user_owned"], indirect=True)
def test_a_span_open_before_the_processor_was_attached_is_counted_at_its_end(instrumentation):
    """Mode U: a pydantic span already open when wardex attached never reached the processor's
    start. Its end still reaches the bridge, once, and is counted as `end_without_start`."""
    tracer = instrumentation.settings.tracer
    early = tracer.start_span("chat early", attributes={"gen_ai.operation.name": "chat"})
    with installed_adapter(PydanticAiAdapter) as installed:
        early.end()
        assert named(installed.spans, "chat early") == []
        assert counters.get("adapters.pydantic_ai.end_without_start") == 1


def test_a_key_written_at_start_and_at_end_is_written_once(live):
    """`gen_ai.agent.call.id` is on the run span from its start to its end; the run unit carries
    it once (`set_extra` appends, so writing it twice would ship it twice)."""
    Agent(TestModel(), name="once").run_sync("go")
    run = one(live.spans, "invoke_agent once")
    assert [k for k, _ in run.extra].count("wardex.pydantic_ai.run_id") == 1


# -- the ids pydantic's spans carry (Mode W) -------------------------------------------


def _otel_id(wardex_id: Any) -> int:
    """A wardex `TraceId`/`SpanId` as the int OpenTelemetry carries."""
    return int.from_bytes(wardex_id.value, "big")


def test_a_span_the_hosts_tracer_opens_in_a_tool_has_the_exported_tool_unit_as_parent():
    """Mode W: pydantic's spans are OpenTelemetry current spans carrying the ids of the units
    wardex exports, so a span the host's own SDK tracer opens inside a tool names, as its parent
    and trace, the `execute_tool` unit that ships. The run's traceparent is the run unit's."""
    from pydantic_ai._utils import get_traceparent

    provider, exporter = host_provider()
    tracer = provider.get_tracer("host")
    with wardex_owned(), installed_adapter(PydanticAiAdapter) as live:
        agent = Agent(TestModel(call_tools=["add"]), name="adder")

        @agent.tool_plain
        def add(a: int, b: int) -> int:
            with tracer.start_as_current_span("host work"):
                return a + b

        result = agent.run_sync("add")
        spans = live.spans
    run = one(spans, "invoke_agent adder")
    tool = one(spans, "execute_tool add")
    (host,) = exporter.get_finished_spans()
    assert host.parent is not None
    assert host.parent.span_id == _otel_id(tool.context.span_id)
    assert host.context.trace_id == _otel_id(tool.context.trace_id)
    assert host.context.trace_id == _otel_id(run.context.trace_id)
    _, trace_id, span_id, _ = get_traceparent(result).split("-")
    assert (trace_id, span_id) == (run.context.trace_id.hex(), run.context.span_id.hex())
    provider.shutdown()


def test_a_chat_span_takes_its_units_ids_once_the_model_is_named():
    """The chat unit opens late (at the attribute write that names the model); the span's
    context changes to the unit's then, and what is current inside the request is the unit."""
    from opentelemetry import trace as otel_trace

    seen: list[Any] = []

    async def answer(messages: list[Any], info: Any) -> ModelResponse:
        seen.append(otel_trace.get_current_span().get_span_context())
        return ModelResponse(parts=[TextPart("done")])

    with wardex_owned(), installed_adapter(PydanticAiAdapter) as live:
        Agent(FunctionModel(answer), name="named").run_sync("hi")
        spans = live.spans
    chat = one(spans, "chat function:answer:")
    (current,) = seen
    assert current.span_id == _otel_id(chat.context.span_id)
    assert current.trace_id == _otel_id(chat.context.trace_id)


# -- import purity -------------------------------------------------------------------

_ABSENT_CHILD = textwrap.dedent(
    """
    import importlib.abc, json, sys

    class Hide(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("pydantic_ai", "opentelemetry"):
                raise ImportError(f"hidden: {name}")
            return None

    sys.meta_path.insert(0, Hide())
    import wardex_sdk as wardex
    from wardex_sdk._adapters import _make_adapter, _probe
    from wardex_sdk._config import AdaptersConfig
    from wardex_sdk._enums import AdapterName
    from wardex_sdk.transport import NoOpTransport

    _probe.installed_distribution = lambda name: None  # the distribution is not installed
    _make_adapter(AdapterName.PYDANTIC_AI)
    wardex.init(
        transport=NoOpTransport(), adapters=AdaptersConfig(enabled=(AdapterName.PYDANTIC_AI,))
    )
    states = {s.name: s.state.value for s in wardex.diagnostics().adapters}
    wardex.close()
    loaded = sorted(m for m in sys.modules if m.split(".")[0] in ("pydantic_ai", "opentelemetry"))
    print(json.dumps({"state": states.get("pydantic_ai"), "loaded": loaded}))
    """
)


def test_without_the_framework_nothing_of_it_or_of_opentelemetry_is_imported(tmp_path):
    """Building the adapter imports nothing of the host's, and a host without pydantic-ai is
    ABSENT before any import is tried: the adapter imports `opentelemetry` only inside its own
    `install()`. Run in a child, where both packages can be hidden from the import system."""
    script = tmp_path / "absent.py"
    script.write_text(_ABSENT_CHILD)
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    assert report == {"state": "absent", "loaded": []}
