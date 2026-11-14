"""A pydantic-ai agent on Amazon Bedrock: one tree, with the model call's usage read off the wire.

pydantic-ai's own `BedrockConverseModel` over a real boto3 client pointed at a loopback Bedrock
runtime (`_bedrock_support.fake_bedrock`), the whole `wardex.init()` and the real byte seams, in
both instrumentation modes. boto3 signs each request with SigV4 and sends it through urllib3 from
a worker thread (`anyio.to_thread`), so this is the production path, not a stand-in for it.

What each test pins:

* A plain run (`/converse`): the wire CHAT is a Bedrock model call (the model id from the
  request path, usage with the cache-read tokens counted into the input), parented to
  the bridge's chat unit. The chat unit keeps the framework's numbers demoted and claims no wire
  absence, so the trace bills the call once.
* A streamed run (`/converse-stream`, the AWS event stream encoding): the same, with the usage the
  stream's closing `metadata` event states, and the wire span marked as a reassembled stream. The
  request leaves from a worker thread while the consumer task's ambient unit is the run, so the
  parent is asserted, not assumed.
* A tool call: the model's streamed `toolUse` block reaches the wire span's output as a tool call,
  the tool runs as an `execute_tool` unit under the run, and each of the two model calls is billed
  once.

The fake listens on 127.0.0.1, not a `bedrock-runtime.*.amazonaws.com` host, so the wire span is
read as Converse from the request path alone, names no provider, and says so with
`PROVIDER_INFERRED`; the chat unit names `aws.bedrock`, as pydantic-ai's own span does.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic_ai import Agent

import wardex_sdk as wardex
from _bedrock_support import (
    CONVERSE_USAGE,
    MODEL_ID,
    STREAM_USAGE,
    TOOL_INPUT,
    TOOL_NAME,
    bedrock_model,
    fake_bedrock,
)
from _pydantic_ai_support import (
    MODES,
    extras,
    instrumentation_mode,
    named,
    one,
    spans_by_source,
)
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._enums import AdapterName, OperationName, ProviderName
from wardex_sdk.testing import RecordingTransport

pytestmark = pytest.mark.usefixtures("fresh_counters")

_DEMOTED = "wardex.pydantic_ai.otel.gen_ai.usage."
#: Converse's `inputTokens` excludes the cache tiers; the wire span's input is inclusive.
_PLAIN = (CONVERSE_USAGE[0] + CONVERSE_USAGE[2], CONVERSE_USAGE[1])


@pytest.fixture(params=MODES)
def instrumentation(request):
    with instrumentation_mode(request.param) as setup:
        yield setup


@pytest.fixture
def recorded(instrumentation) -> Iterator[RecordingTransport]:
    transport = RecordingTransport()
    wardex.init(transport=transport, adapters=AdaptersConfig(enabled=(AdapterName.PYDANTIC_AI,)))
    try:
        yield transport
    finally:
        wardex.close()


def _shipped(transport: RecordingTransport) -> list[Any]:
    wardex.flush()
    return [span for envelope in transport.envelopes for span in envelope.spans]


def _streamed(agent: Agent) -> str:
    async def main() -> str:
        async with agent.run_stream("hi") as stream:
            return await stream.get_output()

    return asyncio.run(main())


def _wire_chats(other: list[Any]) -> list[Any]:
    return [s for s in other if s.gen_ai is not None and s.gen_ai.operation is OperationName.CHAT]


def _typed_usage(spans: list[Any]) -> tuple[int, int]:
    def total(field: str) -> int:
        return sum(getattr(s.gen_ai, field) or 0 for s in spans if s.gen_ai is not None)

    return total("input_tokens"), total("output_tokens")


def _limitations(span: Any) -> tuple[Limitation, ...]:
    return span.capture_integrity.limitations if span.capture_integrity else ()


def _assert_bedrock_call(wire: Any, chat: Any, usage: tuple[int, int], *, stream: bool) -> None:
    """`wire` is a Bedrock model call parented to `chat`, owning `usage`; `chat` keeps the
    framework's copy demoted and claims no wire absence."""
    assert wire.parent_span_id == chat.context.span_id
    assert wire.gen_ai.provider is None
    assert wire.gen_ai.request_model == MODEL_ID
    assert (wire.gen_ai.input_tokens, wire.gen_ai.output_tokens) == usage
    assert bool(wire.gen_ai.stream) is stream
    assert (Limitation.REASSEMBLED_FROM_STREAM in _limitations(wire)) is stream
    # The fake answers on 127.0.0.1, which proves no provider; a real
    # bedrock-runtime.<region>.amazonaws.com host names `aws.bedrock` with no marker.
    assert Limitation.PROVIDER_INFERRED in _limitations(wire)

    assert chat.gen_ai.provider == ProviderName.AWS_BEDROCK, "pydantic-ai's own provider name"
    assert (chat.gen_ai.input_tokens, chat.gen_ai.output_tokens) == (None, None)
    demoted = extras(chat)
    assert (demoted[_DEMOTED + "input_tokens"], demoted[_DEMOTED + "output_tokens"]) == usage
    assert Limitation.NO_WIRE_EVIDENCE not in _limitations(chat)


def _assert_one_call_tree(spans: list[Any], usage: tuple[int, int], *, stream: bool) -> None:
    bridge, other = spans_by_source(spans)
    run = one(bridge, "invoke_agent probe")
    chat = one(bridge, f"chat {MODEL_ID}")
    (wire,) = _wire_chats(other)

    assert chat.parent_span_id == run.context.span_id
    assert run.parent_span_id is None
    assert len({s.context.trace_id for s in spans}) == 1, "one run is one trace"
    _assert_bedrock_call(wire, chat, usage, stream=stream)
    assert _typed_usage(spans) == usage, "one call is billed once"
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 1
    assert counters.get("assembly._usage_hold.usage_promoted") == 0


def test_a_plain_bedrock_call_lands_under_its_chat_unit_with_the_wire_usage(recorded):
    with fake_bedrock() as (endpoint, posts):
        assert Agent(bedrock_model(endpoint), name="probe").run_sync("hi").output == "hello there"
    assert posts == [{"model": MODEL_ID, "stream": False, "signed": True, "tool_call": False}]
    _assert_one_call_tree(_shipped(recorded), _PLAIN, stream=False)


def test_a_streamed_bedrock_call_lands_under_its_chat_unit_with_the_wire_usage(recorded):
    with fake_bedrock() as (endpoint, posts):
        assert _streamed(Agent(bedrock_model(endpoint), name="probe")) == "hello there"
    assert posts == [{"model": MODEL_ID, "stream": True, "signed": True, "tool_call": False}]
    _assert_one_call_tree(_shipped(recorded), STREAM_USAGE, stream=True)


def test_a_bedrock_tool_call_reaches_the_wire_span_and_runs_under_the_run(recorded):
    with fake_bedrock(tool_use=True) as (endpoint, posts):
        agent = Agent(bedrock_model(endpoint), name="probe")

        @agent.tool_plain
        def lookup(city: str) -> str:
            return f"sunny in {city}"

        assert _streamed(agent) == "hello there"
    assert [(p["stream"], p["tool_call"]) for p in posts] == [(True, True), (True, False)]

    spans = _shipped(recorded)
    bridge, other = spans_by_source(spans)
    run = one(bridge, "invoke_agent probe")
    tool = one(bridge, f"execute_tool {TOOL_NAME}")
    chats = sorted(named(bridge, f"chat {MODEL_ID}"), key=lambda s: s.start_time_ns)
    wires = sorted(_wire_chats(other), key=lambda s: s.start_time_ns)
    assert len(chats) == len(wires) == 2
    assert tool.parent_span_id == run.context.span_id
    assert all(chat.parent_span_id == run.context.span_id for chat in chats)
    for wire, chat in zip(wires, chats, strict=True):
        _assert_bedrock_call(wire, chat, STREAM_USAGE, stream=True)

    (asked, answered) = (json.loads(extras(w)["gen_ai.output.messages"]) for w in wires)
    assert asked[0]["finish_reason"] == "tool_call"
    assert [(part["type"], part["name"], part["arguments"]) for part in asked[0]["parts"]] == [
        ("tool_call", TOOL_NAME, TOOL_INPUT)
    ]
    assert answered[0]["finish_reason"] == "stop"
    assert wires[0].gen_ai.finish_reasons == ("tool_call",)
    assert _typed_usage(spans) == (2 * STREAM_USAGE[0], 2 * STREAM_USAGE[1]), "each call once"
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 2
    assert counters.get("assembly._usage_hold.usage_promoted") == 0
