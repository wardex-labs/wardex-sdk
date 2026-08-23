"""`gen_ai.usage.input_tokens` must be the semconv-inclusive total.

The gen-ai semantic conventions say the input total SHOULD include every kind
of input token, cached ones included — and the Anthropic provider document
makes it a MUST, formula spelled out: `gen_ai.usage.input_tokens =
input_tokens + cache_read_input_tokens + cache_creation_input_tokens`, because
Anthropic's wire `input_tokens` EXCLUDES the cache tiers. OpenAI's
`prompt_tokens` already includes `prompt_tokens_details.cached_tokens`, so the
same field is correct there as shipped.

Shipping the Anthropic raw value under the inclusive key makes every backend
that subtracts the cache tiers back out (Langfuse and Phoenix both do)
under-bill by the cache volume — 12-15% on a normal agent loop — and render
the input column as `max(raw - cache, 0) = 0` on every cache-hit turn.

`wardex_sdk._types.GenAIAttributes`'s own field comment has declared the
inclusive contract all along; these tests are the first thing that enforces it.
"""

from __future__ import annotations

import json

import pytest

from wardex_sdk import _wardex_native
from wardex_sdk._adapters._assembler import SessionAssembler
from wardex_sdk._assembly import counters
from wardex_sdk._enums import SpanKind, StatusCode
from wardex_sdk._semantics import build_gen_ai
from wardex_sdk._types import (
    Envelope,
    EnvelopeHeader,
    GenAIAttributes,
    InternalSpan,
    SdkInfo,
    SpanContext,
    SpanId,
    TraceId,
)

# --------------------------------------------------------------------------
# fixtures: one Anthropic exchange with both cache tiers populated
# --------------------------------------------------------------------------

#: Anthropic raw usage: 1000 uncached input, 8000 read from cache, 2000
#: written to cache. The semconv-inclusive input total is 11000.
_ANTHROPIC_REQ = json.dumps(
    {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }
).encode()

_ANTHROPIC_RESP = json.dumps(
    {
        "id": "msg_01",
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "hi"}],
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_input_tokens": 8000,
            "cache_creation_input_tokens": 2000,
        },
    }
).encode()


def _parse_anthropic(resp: bytes = _ANTHROPIC_RESP):
    return _wardex_native.protocol.parse_llm_semantics(
        "api.anthropic.com", "/v1/messages", _ANTHROPIC_REQ, resp
    )


def _header() -> EnvelopeHeader:
    return EnvelopeHeader(
        event_id="evt-1",
        api_key="k",
        sdk=SdkInfo(
            name="wardex.python",
            version="0.1.0",
            python_version="3.12",
            os="mac",
            arch="arm64",
        ),
        sent_at_ns=42,
    )


def _wire_attrs(gen_ai: GenAIAttributes) -> dict:
    """The OTLP attributes the codec emits for a span carrying `gen_ai`."""
    env = Envelope(
        header=_header(),
        spans=(
            InternalSpan(
                context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
                parent_span_id=None,
                name="chat claude-sonnet-4-6",
                kind=SpanKind.CLIENT,
                start_time_ns=1000,
                end_time_ns=2000,
                status=StatusCode.OK,
                gen_ai=gen_ai,
            ),
        ),
    )
    data = _wardex_native.codec.encode_otlp_traces(env)
    decoded = _wardex_native.codec.decode_otlp_traces(data)
    return decoded["resource_spans"][0]["scope_spans"][0]["spans"][0]["attributes"]


class _FakeClient:
    def __init__(self) -> None:
        self.spans: list = []

    def capture_span(self, span) -> None:  # noqa: ANN001
        self.spans.append(span)


@pytest.fixture(autouse=True)
def _fresh_counters():
    counters.reset()
    yield
    counters.reset()


# --------------------------------------------------------------------------
# R1 — P1 (HTTP tee): the parser ships the Anthropic raw (exclusive) value
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="defect A: fill_anthropic ships the provider-raw (exclusive) input_tokens; "
    "semconv and the Anthropic provider doc require input + cache_read + cache_creation",
)
def test_anthropic_input_tokens_include_cache_tiers():
    sem = _parse_anthropic()
    attrs = _wire_attrs(build_gen_ai(sem))
    assert attrs["gen_ai.usage.input_tokens"] == 11000
    # The cache tiers keep their own leaves — the total does not replace them.
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 8000
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 2000
    assert attrs["gen_ai.usage.output_tokens"] == 500


# --------------------------------------------------------------------------
# R1b — P3 (CLI stream-json): a separate Rust type takes the same wrong turn.
# Fixing the HTTP parser alone leaves this path exclusive, which is why the
# reproduction exists per path.
# --------------------------------------------------------------------------

_STREAM_INIT = {
    "type": "system",
    "subtype": "init",
    "session_id": "s-1",
    "model": "claude-sonnet-4-6",
}
_STREAM_ASSISTANT_WITH_CACHE = {
    "type": "assistant",
    "session_id": "s-1",
    "message": {
        "id": "msg_01",
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 500,
            "cache_read_input_tokens": 8000,
            "cache_creation_input_tokens": 2000,
        },
        "content": [{"type": "text", "text": "done"}],
    },
}
_STREAM_RESULT = {
    "type": "result",
    "subtype": "success",
    "session_id": "s-1",
    "is_error": False,
    "num_turns": 1,
    "total_cost_usd": 0.02,
    "duration_ms": 100,
    "duration_api_ms": 80,
}


@pytest.mark.xfail(
    strict=True,
    reason="defect A on P3: the CLI stream-json parser ships Anthropic raw input_tokens "
    "and never went through LlmSemantics, so the P1 fix alone cannot reach it",
)
def test_stream_json_input_tokens_include_cache_tiers():
    client = _FakeClient()
    asm = SessionAssembler(client)
    asm.on_outbound(
        1,
        json.dumps(
            {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "go"}}
        ),
    )
    asm.on_inbound(1, _STREAM_INIT)
    asm.on_inbound(1, _STREAM_ASSISTANT_WITH_CACHE)
    asm.on_inbound(1, _STREAM_RESULT)
    asm.on_close(1, None)
    chat = next(s for s in client.spans if s.name.startswith("chat"))
    assert chat.gen_ai is not None
    assert chat.gen_ai.input_tokens == 11000
    assert chat.gen_ai.cache_read_input_tokens == 8000
    assert chat.gen_ai.cache_creation_input_tokens == 2000
    assert chat.gen_ai.output_tokens == 500
