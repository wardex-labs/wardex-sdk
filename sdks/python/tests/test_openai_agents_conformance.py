"""The openai-agents adapter, run against the shared adapter conformance suite.

Everything here is the SUBJECT — one seam, one workload, the tree that
workload must produce, and a run left open mid-tool. The claims are in
`wardex_sdk.testing.conformance` and are the same ones the LangGraph and
Agent SDK adapters answer next door.

The seam is the framework's PROCESSOR TUPLE, not a patched attribute: the
adapter registers one `TracingProcessor` and hands the framework its original
tuple back on uninstall, and `tuple(t) is t` is what lets the suite's identity
check hold on the way out. The model is an in-process fake implementing
`agents.models.interface.Model`, so no HTTP happens and no wire span appears —
the declared tree is the adapter's spans alone.

`usage_expected` is `"none"`, and the fake is built so that declaration is a
two-way claim rather than a vacuous one: it opens the framework's own
`response_span` around every answer and hands it a real `Response` whose
`usage` is non-zero, the way the real model does, so `_response_end` runs
under this suite with usage on offer. Usage on any adapter span here is the
adapter reading what it promised to discard. The three-turn scenario in
`test_openai_agents_adapter.py` asserts the same rule against the wire.

The scenario tests that go BEYOND these invariants — the three-turn tree on
the wire, parallel tools, the handoff chain, guardrails, MaxTurns, tracing
off, fault injection, fork — live in `test_openai_agents_adapter.py`.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from agents import Agent, RunConfig, Runner, function_tool
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.tracing import (
    TracingProcessor,
    get_trace_provider,
    response_span,
    set_trace_processors,
)
from agents.usage import Usage
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails

from wardex_sdk._adapters._openai_agents import OpenAIAgentsAdapter
from wardex_sdk._enums import AdapterName
from wardex_sdk.testing import AdapterConformanceSuite, AdapterSubject, StalledRun


def _call(name: str, args: str, call_id: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{call_id}", call_id=call_id, name=name, arguments=args, type="function_call"
    )


def _message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
        role="assistant",
        status="completed",
        type="message",
    )


def _response(response_id: str, output: list[Any]) -> Response:
    """A real `Response` with usage on it — what the framework's response span
    carries when the real model answers, so the adapter's discard is tested
    against the object it would actually be handed."""
    return Response(
        id=response_id,
        created_at=0.0,
        model="fake",
        object="response",
        output=output,
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=ResponseUsage(
            input_tokens=7,
            input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
            output_tokens=3,
            output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
            total_tokens=10,
        ),
    )


class FakeModel(Model):
    """Hands off first when a handoff is offered, calls the first tool once,
    then answers `done`. Decided from the call ids already in the input, so it
    is stateless across turns and honest about what the framework resent.

    Every answer is wrapped in the framework's `response_span`, exactly as the
    real model wraps its HTTP call, and the span is handed a `Response` with
    usage — so the adapter's `_response_end` runs here and the usage it must
    NOT read is on offer.
    """

    def __init__(self) -> None:
        self.responses = 0

    async def get_response(  # noqa: PLR0913
        self,
        system_instructions: Any,
        input: Any,  # noqa: A002
        model_settings: Any,
        tools: Any,
        output_schema: Any,
        handoffs: Any,
        tracing: Any,
        *,
        previous_response_id: Any,
        conversation_id: Any,
        prompt: Any,
    ) -> ModelResponse:
        self.responses += 1
        with response_span(disabled=tracing.is_disabled()) as span:
            response = self._answer(input, tools, handoffs)
            span.span_data.response = response
        return ModelResponse(
            output=list(response.output),
            usage=Usage(requests=1, input_tokens=7, output_tokens=3, total_tokens=10),
            response_id=response.id,
        )

    def _answer(self, input: Any, tools: Any, handoffs: Any) -> Response:  # noqa: A002
        items = input if isinstance(input, list) else []
        done = {
            x.get("call_id")
            for x in items
            if isinstance(x, dict) and x.get("type") == "function_call_output"
        }
        handoff_names = [h.tool_name for h in handoffs]
        tool_names = [t.name for t in tools]
        if handoff_names and "call_h" not in done:
            out: list[Any] = [_call(handoff_names[0], "{}", "call_h")]
        elif tool_names and "call_1" not in done:
            out = [_call(tool_names[0], '{"city":"Seoul"}', "call_1")]
        else:
            out = [_message("done")]
        return _response(f"resp_{self.responses}", out)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("the conformance workload never streams")


def _agents(get_weather: Any) -> Agent:
    agent_b = Agent(name="agent_b", instructions="b", tools=[get_weather], model=FakeModel())
    return Agent(name="agent_a", instructions="a", handoffs=[agent_b], model=FakeModel())


class _HostProcessor(TracingProcessor):
    """A processor the host registered before wardex: inert, and there so the
    tuple the seam check compares by identity is never CPython's empty-tuple
    singleton, which a fresh `tuple([])` would also be."""

    def on_trace_start(self, trace: Any) -> None:
        return None

    def on_trace_end(self, trace: Any) -> None:
        return None

    def on_span_start(self, span: Any) -> None:
        return None

    def on_span_end(self, span: Any) -> None:
        return None

    def shutdown(self, timeout: float | None = None) -> None:
        return None

    def force_flush(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _wardex_only_processors():
    """The framework's default processor would POST the run record to the real
    API. Every run here starts from a list holding ONE inert host processor
    (see `_HostProcessor` for why not none) and the original tuple is put
    back afterwards."""
    provider = get_trace_provider()
    before = provider._multi_processor._processors
    set_trace_processors([_HostProcessor()])
    yield
    provider.set_processors(before)


def seams() -> dict[str, object]:
    """The one attribute the adapter changes: the framework's processor tuple.

    Read through the framework's own provider rather than off the adapter: what
    has to be restored is the FRAMEWORK's tuple, and asking the adapter what it
    registered would take its word for the very thing under test.
    """
    return {"provider.processors": get_trace_provider()._multi_processor._processors}


def workload(live):  # noqa: ANN001, ANN201
    """agent_a hands off to agent_b; agent_b calls `get_weather`, then answers.

    Opens no wardex span of its own: with nothing installed the framework's
    processor list is empty and the run leaves no trace, which is the zero
    point the suite measures against.
    """

    @function_tool
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    return Runner.run_sync(
        _agents(get_weather), "hi", run_config=RunConfig(workflow_name="Conformance")
    ).final_output


def stall(live) -> StalledRun:  # noqa: ANN001
    """A run parked inside a tool on a background thread.

    The run's SESSION unit lives until the framework finishes the run, so this
    is the shape where a shutdown arrives with the run and an agent still
    open — and `resume` is the host carrying on afterwards, which must not
    see an exception wardex invented.
    """
    gate = threading.Event()
    entered = threading.Event()

    @function_tool
    def get_weather(city: str) -> str:
        entered.set()
        gate.wait(10)
        return f"sunny in {city}"

    result: dict[str, Any] = {}

    def drive() -> None:
        try:
            result["out"] = Runner.run_sync(
                _agents(get_weather), "hi", run_config=RunConfig(workflow_name="Stalled")
            ).final_output
        except BaseException as exc:  # noqa: BLE001 — reported to the test
            result["exc"] = exc

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    assert entered.wait(10), "the tool never started; the stall never happened"

    def resume() -> Any:
        gate.set()
        thread.join(15)
        assert "exc" not in result, result
        return result.get("out")

    return StalledRun(root="invoke_workflow Stalled", resume=resume)


@pytest.fixture
def subject() -> AdapterSubject:
    return AdapterSubject(
        name=AdapterName.OPENAI_AGENTS.value,
        module="wardex_sdk._adapters._openai_agents",
        factory=OpenAIAgentsAdapter,
        seams=seams,
        workload=workload,
        chains=(
            ("invoke_workflow Conformance", "invoke_agent agent_a", "handoff agent_a→agent_b"),
            ("invoke_workflow Conformance", "invoke_agent agent_b", "execute_tool get_weather"),
        ),
        stall=stall,
        detect_package="agents",
        # The adapter discards the framework's usage by design and the fake
        # model makes no HTTP call; declaring "none" makes the usage check
        # assert the NEGATIVE — usage appearing here goes red, not unnoticed.
        usage_expected="none",
    )


@pytest.mark.parametrize("check", AdapterConformanceSuite.CHECKS)
def test_conformance(subject, check):
    AdapterConformanceSuite(subject).run(check)
