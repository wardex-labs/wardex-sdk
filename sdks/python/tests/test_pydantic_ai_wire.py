"""The wire CHAT under the bridge's chat unit, plain and streamed, against a loopback OpenAI.

The whole `wardex.init()`, the real byte seams and pydantic-ai's own OpenAI provider, in both
instrumentation modes. Spans are split by capture source (the bridge's units carry
`OTEL_BRIDGE`; the wire CHAT does not), never by kind: both chat spans are CLIENT.

What each test pins:

* A plain run: pydantic sends the request on the task its `chat` span is current on, so the wire
  CHAT is latched under the pinned chat unit by context (CONTEXTVAR, 1.0). The wire span owns the
  usage; the chat unit keeps the framework's copy demoted and claims no wire absence.
* A streamed run: the request is sent from the CONSUMER task, inside `async with run_stream`,
  where wardex's ambient unit is the run — pydantic copies the run's context back to its caller,
  but only re-attaches the `chat` span's OpenTelemetry context there. The adapter hands the core
  OpenTelemetry's current-span reader, and the core parents the request to the chat unit
  (`assembly._bridge_latch.parent_resolved`).

  Measured for this file (2026-11-08, pydantic-ai-slim 2.66.0): in `run_stream`, the
  `set_attributes` call that names the model runs on the `chat` span's OWN task (the
  `wrap_model_request` task), not the consumer's, so in Mode W the chat unit opens and pins on its
  start task like a plain run's (`chat_open_off_start_task` stays 0), and only the request itself
  needs the resolver.
* The control: the same streamed run with the resolver removed lands the wire CHAT under the run,
  and the chat unit then promotes the framework's usage — the double count the resolver exists
  to prevent, made visible.
* A request to an API wardex does not read as a model call: its plain HTTP span sits under the
  chat unit with no usage, and the chat unit promotes the framework's numbers (`NO_WIRE_EVIDENCE`),
  because only a sealed transaction to a model-call endpoint keeps them demoted.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

import wardex_sdk as wardex
from _pydantic_ai_support import (
    MODES,
    PLAIN_USAGE,
    STREAM_USAGE,
    extras,
    fake_openai,
    instrumentation_mode,
    one,
    openai_model,
    spans_by_source,
)
from wardex_sdk._adapters._registry import get_registry
from wardex_sdk._assembly import Limitation, ParentSource, counters
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._enums import AdapterName, OperationName
from wardex_sdk.testing import RecordingTransport

pytestmark = pytest.mark.usefixtures("fresh_counters")

_DEMOTED = "wardex.pydantic_ai.otel.gen_ai.usage."


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


def _run_plain(base_url: str) -> Any:
    return Agent(openai_model(base_url), name="probe").run_sync("hi").output


def _run_streamed(base_url: str) -> list[str]:
    async def main() -> list[str]:
        parts = []
        async with Agent(openai_model(base_url), name="probe").run_stream("hi") as stream:
            async for text in stream.stream_text():
                parts.append(text)
        return parts

    return asyncio.run(main())


def _wire_chat(other: list[Any]) -> Any:
    chats = [s for s in other if s.gen_ai is not None and s.gen_ai.operation is OperationName.CHAT]
    assert len(chats) == 1, sorted(s.name for s in other)
    return chats[0]


def _typed_usage(spans: list[Any]) -> tuple[int, int]:
    def total(field: str) -> int:
        return sum(getattr(s.gen_ai, field) or 0 for s in spans if s.gen_ai is not None)

    return total("input_tokens"), total("output_tokens")


def _assert_wire_under_chat(spans: list[Any], usage: tuple[int, int]) -> None:
    bridge, other = spans_by_source(spans)
    run = one(bridge, "invoke_agent probe")
    chat = one(bridge, "chat gpt-4o-mini")
    wire = _wire_chat(other)

    assert wire.parent_span_id == chat.context.span_id
    assert (wire.correlation.strategy, wire.correlation.confidence) == (
        ParentSource.CONTEXTVAR,
        1.0,
    )
    assert chat.parent_span_id == run.context.span_id
    assert run.parent_span_id is None
    assert len({s.context.trace_id for s in spans}) == 1, "one run is one trace"

    assert (wire.gen_ai.input_tokens, wire.gen_ai.output_tokens) == usage
    assert (chat.gen_ai.input_tokens, chat.gen_ai.output_tokens) == (None, None)
    demoted = extras(chat)
    assert (demoted[_DEMOTED + "input_tokens"], demoted[_DEMOTED + "output_tokens"]) == usage
    limitations = chat.capture_integrity.limitations if chat.capture_integrity else ()
    assert Limitation.NO_WIRE_EVIDENCE not in limitations
    assert _typed_usage(spans) == usage, "one call is billed once"
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 1
    assert counters.get("assembly._usage_hold.usage_promoted") == 0


def test_a_plain_model_call_lands_under_its_chat_unit(recorded):
    with fake_openai() as (base_url, posts):
        assert _run_plain(base_url) == "hello there"
    assert [p["stream"] for p in posts] == [False]
    _assert_wire_under_chat(_shipped(recorded), PLAIN_USAGE)


def test_a_streamed_model_call_lands_under_its_chat_unit(recorded):
    with fake_openai() as (base_url, posts):
        assert "".join(_run_streamed(base_url)).endswith("there")
    assert [p["stream"] for p in posts] == [True]
    _assert_wire_under_chat(_shipped(recorded), STREAM_USAGE)
    assert counters.get("assembly._bridge_latch.parent_resolved") >= 1
    assert counters.get("adapters.pydantic_ai.chat_open_off_start_task") == 0


def test_without_the_resolver_a_streamed_call_lands_under_the_run_and_is_billed_twice(recorded):
    """The negative control: proves the resolver is what moved the streamed request, and shows
    what a miss costs — the chat unit sees no wire request and promotes the framework's numbers
    while the wire span carries its own."""
    get_registry()._contexts[AdapterName.PYDANTIC_AI.value].unfollow_current_span()
    with fake_openai() as (base_url, _):
        _run_streamed(base_url)
    spans = _shipped(recorded)
    bridge, other = spans_by_source(spans)
    run = one(bridge, "invoke_agent probe")
    chat = one(bridge, "chat gpt-4o-mini")
    wire = _wire_chat(other)

    assert wire.parent_span_id == run.context.span_id
    assert counters.get("assembly._bridge_latch.parent_resolved") == 0
    assert (chat.gen_ai.input_tokens, chat.gen_ai.output_tokens) == STREAM_USAGE
    assert Limitation.NO_WIRE_EVIDENCE in chat.capture_integrity.limitations
    assert _typed_usage(spans) == (2 * STREAM_USAGE[0], 2 * STREAM_USAGE[1])


class _UnreadProvider(http.server.BaseHTTPRequestHandler):
    """A model API whose request and response shapes wardex's parser does not read."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"reply": "hello there", "billing": {"units": 42}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


def test_a_request_wardex_does_not_read_as_a_model_call_promotes_the_frameworks_usage(recorded):
    """A provider wardex sees on the wire but does not parse (Bedrock InvokeModel, a custom model's
    own API): the request went out under the chat unit, but its span is plain HTTP with no usage.
    So the framework's numbers are the only ones there are: promoted into the chat unit's typed
    usage, marked `NO_WIRE_EVIDENCE`, and the tree bills the call exactly once."""
    import httpx

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _UnreadProvider)
    host, port = httpd.socket.getsockname()[:2]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    async def call(messages: list[Any], info: Any) -> ModelResponse:
        async with httpx.AsyncClient() as client:
            answer = await client.post(f"http://{host}:{port}/v1/generate", json={"prompt": "hi"})
        return ModelResponse(parts=[TextPart(answer.json()["reply"])])

    try:
        result = Agent(FunctionModel(call, model_name="custom-1"), name="custom").run_sync("hi")
    finally:
        httpd.shutdown()
        httpd.server_close()
    (response,) = [m for m in result.all_messages() if isinstance(m, ModelResponse)]
    usage = (response.usage.input_tokens, response.usage.output_tokens)
    assert all(usage), "precondition: the framework reported a nonzero usage"
    spans = _shipped(recorded)
    bridge, other = spans_by_source(spans)
    chat = one(bridge, "chat custom-1")
    (wire,) = [s for s in other if s.name.startswith("HTTP POST")]
    assert wire.parent_span_id == chat.context.span_id
    assert wire.gen_ai is None
    assert (chat.gen_ai.input_tokens, chat.gen_ai.output_tokens) == usage
    assert _DEMOTED + "input_tokens" not in extras(chat)
    assert _DEMOTED + "output_tokens" not in extras(chat)
    assert Limitation.NO_WIRE_EVIDENCE in chat.capture_integrity.limitations
    assert _typed_usage(spans) == usage, "one call is billed once"
    assert counters.get("assembly._usage_hold.usage_promoted") == 1
    assert counters.get("assembly._usage_hold.usage_kept_demoted") == 0
