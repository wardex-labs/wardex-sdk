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
from pathlib import Path

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


# --------------------------------------------------------------------------
# over-correction guards: providers that already report inclusive totals
# --------------------------------------------------------------------------

_OPENAI_REQ = json.dumps(
    {"model": "gpt-5", "messages": [{"role": "user", "content": "x"}]}
).encode()
_OPENAI_RESP = json.dumps(
    {
        "id": "chatcmpl-1",
        "model": "gpt-5",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "y"}}],
        "usage": {
            "prompt_tokens": 11000,
            "completion_tokens": 600,
            "prompt_tokens_details": {"cached_tokens": 8000},
            "completion_tokens_details": {"reasoning_tokens": 100},
        },
    }
).encode()


def _parse_openai():
    return _wardex_native.protocol.parse_llm_semantics(
        "api.openai.com", "/v1/chat/completions", _OPENAI_REQ, _OPENAI_RESP
    )


def test_openai_input_tokens_unchanged():
    """OpenAI `prompt_tokens` already contains `cached_tokens`.

    The one defense against over-correction: adding the cache tier again
    would ship 19000 for an 11000-token prompt — the same class of billing
    error the Anthropic fix removes, manufactured on the other axis.
    """
    sem = _parse_openai()
    assert sem.input_tokens == 11000
    assert sem.cache_read_input_tokens == 8000


def test_reasoning_included_in_output_tokens():
    """`completion_tokens` already contains the reasoning tokens (regression pin)."""
    sem = _parse_openai()
    assert sem.output_tokens == 600
    assert sem.reasoning_output_tokens == 100
    assert sem.output_tokens >= sem.reasoning_output_tokens


# --------------------------------------------------------------------------
# normalization conditions cross the FFI as values, and Python counts them
# --------------------------------------------------------------------------


def test_cache_without_input_tokens_bumps_the_counter():
    """A cache tier without its total: withheld, not invented — and counted.

    This is the reachability proof for the deferred USAGE_TOTALS_UNPAIRED
    limitation: the member is minted only once this counter is seen nonzero
    in real sessions, and a counter no input can reach could never earn it.
    """
    resp = json.dumps(
        {
            "id": "msg_02",
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"output_tokens": 5, "cache_read_input_tokens": 8000},
        }
    ).encode()
    sem = _parse_anthropic(resp)
    assert sem.usage_totals_unpaired is True
    gen_ai = build_gen_ai(sem)
    assert gen_ai.input_tokens is None  # no fabricated total
    assert gen_ai.cache_read_input_tokens == 8000
    assert counters.get("semantics.build_gen_ai.usage_totals_unpaired") == 1
    assert counters.get("semantics.build_gen_ai.usage_overflowed") == 0


def test_stream_json_cache_without_input_tokens_bumps_the_counter():
    """The same condition arrives on P3 and lands in the assembler's tally."""
    line = {
        "type": "assistant",
        "session_id": "s-1",
        "message": {
            "id": "msg_03",
            "model": "claude-sonnet-4-6",
            "stop_reason": "end_turn",
            "usage": {"output_tokens": 5, "cache_read_input_tokens": 8000},
            "content": [{"type": "text", "text": "done"}],
        },
    }
    client = _FakeClient()
    asm = SessionAssembler(client)
    asm.on_outbound(
        1,
        json.dumps(
            {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "go"}}
        ),
    )
    asm.on_inbound(1, _STREAM_INIT)
    asm.on_inbound(1, line)
    asm.on_inbound(1, _STREAM_RESULT)
    asm.on_close(1, None)
    chat = next(s for s in client.spans if s.name.startswith("chat"))
    assert chat.gen_ai.input_tokens is None
    assert chat.gen_ai.cache_read_input_tokens == 8000
    assert counters.get("adapters.assembler.stream_usage_totals_unpaired") == 1
    assert counters.get("adapters.assembler.stream_usage_overflowed") == 0


# --------------------------------------------------------------------------
# G5 — the public-surface choke point counts what the type system cannot see
# --------------------------------------------------------------------------


def test_set_gen_ai_counts_a_non_inclusive_block():
    """A hand-built exclusive block is counted AND shipped, never rejected.

    `GenAIAttributes` is public API and `flatten_gen_ai` encodes whatever it
    is given; the Rust constructor cannot reach a block a third-party adapter
    built by hand. `SpanDraft.set_gen_ai` is the one choke point every span
    passes, so the violation is tallied there — and the span still ships,
    because breaking the host over a diagnostic is not this SDK's trade.
    """
    from wardex_sdk._assembly import AMBIENT, Ambient, SpanDraft, SpanIntent, resolve_parentage
    from wardex_sdk._enums import CaptureSource, OperationName

    draft = SpanDraft(
        resolve_parentage(Ambient(None, None, None), AMBIENT),
        intent=SpanIntent.CHAT,
        subject="claude-sonnet-4-6",
        source=CaptureSource.ADAPTER,
        start_ns=1,
    )
    draft.set_gen_ai(
        GenAIAttributes(
            operation=OperationName.CHAT,
            request_model="claude-sonnet-4-6",
            input_tokens=1000,  # < 8000 + 2000: the pre-fix exclusive shape
            cache_read_input_tokens=8000,
            cache_creation_input_tokens=2000,
        )
    )
    assert counters.get("assembly.builder.gen_ai_usage_not_inclusive") == 1
    span = draft.finish(2)
    assert span is not None and span.gen_ai.input_tokens == 1000  # shipped as given


def test_set_gen_ai_does_not_count_an_inclusive_block():
    """The negative: wardex's own normalized output never bumps the tally."""
    sem = _parse_anthropic()
    attrs = build_gen_ai(sem)
    from wardex_sdk._assembly import AMBIENT, Ambient, SpanDraft, SpanIntent, resolve_parentage
    from wardex_sdk._enums import CaptureSource

    draft = SpanDraft(
        resolve_parentage(Ambient(None, None, None), AMBIENT),
        intent=SpanIntent.CHAT,
        subject="claude-sonnet-4-6",
        source=CaptureSource.ADAPTER,
        start_ns=1,
    )
    draft.set_gen_ai(attrs)
    assert counters.get("assembly.builder.gen_ai_usage_not_inclusive") == 0


# --------------------------------------------------------------------------
# L3 — golden vectors frozen from the Langfuse mapping oracle (CI-permanent)
# --------------------------------------------------------------------------

_GOLDEN_PATH = Path(__file__).parent / "fixtures" / "langfuse_mapping_golden.json"

#: The `??` candidate chains and the pass-through exclusion list, mirrored
#: from Langfuse's `extractGenericGenAiUsageDetails` (v4.16.0). The oracle
#: (`scripts/langfuse-mapping-oracle/run.ts`) re-checks the original on every
#: run; this mirror exists so CI can replay the frozen outputs with no
#: network, no node and no docker.
_LF_INPUT_CHAIN = ("prompt_tokens", "input_tokens", "prompt")
_LF_OUTPUT_CHAIN = ("completion_tokens", "output_tokens", "completion")
_LF_TOTAL_CHAIN = ("total_tokens", "total")
_LF_CACHE_READ_CHAIN = (
    "cache_read.input_tokens",
    "cache_read_input_tokens",
    "cache_read_tokens",
    "details.cache_read_tokens",
    "details.cache_read_input_tokens",
    "prompt_details.cache_read",
    "input_cached_tokens",
)
_LF_CACHE_CREATION_CHAIN = (
    "cache_creation.input_tokens",
    "cache_creation_input_tokens",
    "cache_write_tokens",
    "details.cache_write_tokens",
    "details.cache_creation_input_tokens",
    "prompt_details.cache_write",
    "input_cache_creation",
)
_LF_REASONING_CHAIN = ("reasoning.output_tokens", "completion_details.reasoning")
_LF_AUDIO_CHAIN = ("completion_details.audio",)

#: claude-sonnet-4-6 Standard-tier unit prices (default-model-prices.json).
_LF_UNIT_PRICE = {
    "input": 3e-06,
    "output": 1.5e-05,
    "input_cached_tokens": 3e-07,
    "input_cache_creation": 3.75e-06,
}


def _first(raw: dict, chain: tuple) -> int | None:
    """`??` semantics: the first PRESENT candidate wins, zero included."""
    for key in chain:
        if key in raw:
            return raw[key]
    return None


def _langfuse_generic_usage(attrs: dict) -> dict:
    """`extractGenericGenAiUsageDetails`, in Python, over flattened attrs."""
    raw = {}
    for key, value in attrs.items():
        if (key.startswith("gen_ai.usage.") and key != "gen_ai.usage.cost") or key.startswith(
            "llm.token_count."
        ):
            stripped = key.replace("gen_ai.usage.", "").replace("llm.token_count.", "")
            if isinstance(value, (int, float)):
                raw[stripped] = value
    if not raw:
        return {}
    input_tokens = _first(raw, _LF_INPUT_CHAIN)
    output_tokens = _first(raw, _LF_OUTPUT_CHAIN)
    total_tokens = _first(raw, _LF_TOTAL_CHAIN)
    cache_read = _first(raw, _LF_CACHE_READ_CHAIN)
    cache_creation = _first(raw, _LF_CACHE_CREATION_CHAIN)
    reasoning = _first(raw, _LF_REASONING_CHAIN)
    audio = _first(raw, _LF_AUDIO_CHAIN)

    consumed = set(
        _LF_INPUT_CHAIN
        + _LF_OUTPUT_CHAIN
        + _LF_TOTAL_CHAIN
        + _LF_CACHE_READ_CHAIN
        + _LF_CACHE_CREATION_CHAIN
        + _LF_REASONING_CHAIN
        + _LF_AUDIO_CHAIN
    )
    out = {
        (key.replace("details.", "", 1) if key.startswith("details.") else key): value
        for key, value in raw.items()
        if key not in consumed
    }
    if input_tokens is not None:
        out["input"] = max(input_tokens - (cache_read or 0) - (cache_creation or 0), 0)
    if output_tokens is not None:
        out["output"] = max(output_tokens - (reasoning or 0) - (audio or 0), 0)
    if total_tokens is not None:
        out["total"] = total_tokens
    if cache_read is not None:
        out["input_cached_tokens"] = cache_read
    if cache_creation is not None:
        out["input_cache_creation"] = cache_creation
    if reasoning is not None:
        out["output_reasoning_tokens"] = reasoning
    if audio is not None:
        out["output_audio_tokens"] = audio
    return out


def _langfuse_costs(usage: dict) -> tuple[dict, float]:
    """`IngestionService.calculateUsageCosts`: priced buckets only, then total."""
    cost = {
        key: units * _LF_UNIT_PRICE[key] for key, units in usage.items() if key in _LF_UNIT_PRICE
    }
    total = sum(cost.values())
    if cost:
        cost["total"] = total
    return cost, total


def _golden() -> dict:
    return json.loads(_GOLDEN_PATH.read_text())["results"]


def test_langfuse_mapping_golden_normal():
    """Encoder output -> Langfuse mapping == the frozen L2 result (vector A)."""
    attrs = _wire_attrs(build_gen_ai(_parse_anthropic()))
    usage = _langfuse_generic_usage(attrs)
    cost, total = _langfuse_costs(usage)
    frozen = _golden()["a_normal"]
    assert usage == frozen["usageDetails"]
    assert cost == pytest.approx(frozen["costDetails"])
    assert total == pytest.approx(frozen["totalCost"])
    assert total == pytest.approx(0.0204)


def test_langfuse_mapping_golden_buggy_contrast():
    """The pre-fix shape must keep reproducing the under-billing (vector B)."""
    attrs = _wire_attrs(
        GenAIAttributes(
            operation="chat",
            request_model="claude-sonnet-4-6",
            response_model="claude-sonnet-4-6",
            input_tokens=1000,  # exclusive — the defect being contrasted
            output_tokens=500,
            cache_read_input_tokens=8000,
            cache_creation_input_tokens=2000,
        )
    )
    usage = _langfuse_generic_usage(attrs)
    cost, total = _langfuse_costs(usage)
    frozen = _golden()["b_buggy"]
    assert usage == frozen["usageDetails"]
    assert usage["input"] == 0  # the collapsed input column
    assert cost == pytest.approx(frozen["costDetails"])
    assert total == pytest.approx(0.0174)  # -14.7%


def test_langfuse_mapping_golden_reasoning():
    """Vector D: the reasoning bucket exists and is the one unpriced key."""
    attrs = _wire_attrs(
        GenAIAttributes(
            operation="chat",
            request_model="claude-sonnet-4-6",
            response_model="claude-sonnet-4-6",
            input_tokens=1000,
            output_tokens=500,
            reasoning_output_tokens=200,
        )
    )
    usage = _langfuse_generic_usage(attrs)
    cost, _total = _langfuse_costs(usage)
    frozen = _golden()["d_reasoning"]
    assert usage == frozen["usageDetails"]
    assert cost == pytest.approx(frozen["costDetails"])
    uncovered = set(usage) - {"total"} - set(cost)
    assert uncovered == {"output_reasoning_tokens"}  # the deferred ecosystem finding


def test_langfuse_mapping_golden_conflicted_pair():
    """Vector C: one session, one billed observation — and the demoted keys
    are structurally invisible to the prefix collector."""
    frozen = _golden()["c_pair"]
    carriers = [e for e in frozen if e["usageDetails"]]
    assert len(carriers) == 1 and carriers[0]["name"].startswith("chat")
    assert sum(e["totalCost"] for e in frozen) == pytest.approx(0.0204)
    # The mechanism, replayed on the mirror: a demoted copy maps to NOTHING.
    demoted = {
        "wardex.anthropic_agent_sdk.otel.gen_ai.usage.input_tokens": 1000,
        "wardex.anthropic_agent_sdk.otel.gen_ai.usage.cache_read_input_tokens": 8000,
        "gen_ai.request.model": "claude-sonnet-4-6",
    }
    assert _langfuse_generic_usage(demoted) == {}
