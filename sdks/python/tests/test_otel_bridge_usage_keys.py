"""What the OTel bridge may put on the wire when a chat join is ambiguous.

Two defects live on the conflicted `llm_request` increment, and only there —
it is the single span class in the SDK that can carry `gen_ai.usage.*` extras:

* **Spelling (B)** — the CLI reports the cache tiers under underscore
  spellings (`gen_ai.usage.cache_read_input_tokens`); passing them through
  as top-level attributes makes one SDK state one fact in two spellings,
  and `test_otlp_codec.py`'s "the underscore spellings are gone" assertion
  only ever covered the `GenAIAttributes` path.
* **Double billing (D)** — the unmerged CHAT draft still ships with its own
  gen_ai usage, and the conflicted increment ships the CLI's copy of the
  same tokens next to a model key. A backend that classifies GENERATION on
  the model key alone (Langfuse's `ModelBased` priority-10 fallback) then
  prices ONE LLM call TWICE.

The remedy under test: on a conflicted join the bridge demotes the CLI's
usage keys into the `wardex.anthropic_agent_sdk.otel.*` namespace — kept,
readable, but invisible to any `gen_ai.usage.` prefix collector — while the
identity keys (model, response id) stay under their own names.
"""

from __future__ import annotations

import http.client
import json
import time

import pytest

import _otlp_build
from wardex_sdk._adapters._assembler import SessionAssembler
from wardex_sdk._adapters._otel_receiver import _OtelBridgeReceiver
from wardex_sdk._adapters._session_state import _BridgeBinding
from wardex_sdk._assembly import counters
from wardex_sdk._assembly._diag import reset_reports_for_test

TRACE = "ac" * 16

INIT = {"type": "system", "subtype": "init", "session_id": "s-1", "model": "claude-sonnet-4-6"}
ASSISTANT = {
    "type": "assistant",
    "session_id": "s-1",
    "message": {
        "id": "m1",
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": 30},
        "content": [{"type": "text", "text": "done"}],
    },
}
RESULT = {
    "type": "result",
    "subtype": "success",
    "session_id": "s-1",
    "is_error": False,
    "num_turns": 1,
    "total_cost_usd": 0.01,
    "duration_ms": 100,
    "duration_api_ms": 80,
}

#: The conflicted `llm_request`'s attributes. The model keys are on the span —
#: the CLI telemetry spike confirmed `gen_ai.response.id` and the merge
#: allowlist names both model keys because they were observed — and they are
#: what makes the double-billing scenario concrete: a backend needs nothing
#: else to classify the span as a GENERATION and price it. If the CLI ever
#: stops stamping them, the demotion under test stays correct, just cheaper.
CLI_USAGE_ATTRS = {
    "gen_ai.request.model": "claude-sonnet-4-6",
    "gen_ai.response.id": "req_conflict",
    "gen_ai.usage.input_tokens": 1000,
    "gen_ai.usage.output_tokens": 500,
    "gen_ai.usage.cache_read_input_tokens": 8000,
    "gen_ai.usage.cache_creation_input_tokens": 2000,
}

_DEMOTED_PREFIX = "wardex.anthropic_agent_sdk.otel."


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


@pytest.fixture(autouse=True)
def _fresh_reports():
    reset_reports_for_test()
    yield
    reset_reports_for_test()


@pytest.fixture
def receiver():
    r = _OtelBridgeReceiver(max_body_bytes=64 * 1024, max_spans_per_session=64, max_sessions=8)
    yield r
    r.close()


def _post(receiver, body: bytes) -> int:  # noqa: ANN001
    conn = http.client.HTTPConnection("127.0.0.1", receiver.port, timeout=5)
    try:
        conn.request("POST", "/v1/traces", body=body, headers={"x-wardex-bridge": receiver.token})
        return conn.getresponse().status
    finally:
        conn.close()


def _ambiguous_session(receiver) -> list:  # noqa: ANN001
    """One CHAT draft, two overlapping `llm_request`s: nothing can merge.

    Both requests ship as conflicted sibling increments carrying
    `CLI_USAGE_ATTRS`, and the chat ships unmerged with its own gen_ai block —
    the exact configuration defects B and D live in.
    """
    client = _FakeClient()
    asm = SessionAssembler(client, bridge=receiver)
    receiver.reserve(TRACE)
    t0 = time.time_ns()
    asm.on_outbound(
        1,
        json.dumps(
            {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "go"}}
        ),
        bridge=_BridgeBinding(trace_id_hex=TRACE, confirmed=True),
    )
    asm.on_inbound(1, INIT)
    asm.on_inbound(1, ASSISTANT)
    asm.on_inbound(1, RESULT)
    t1 = time.time_ns()
    body = _otlp_build.request(
        [
            _otlp_build.span(
                name="claude_code.llm_request",
                trace_id=TRACE,
                span_id=f"{i + 1:016x}",
                start_ns=t0,
                end_ns=t1,
                attrs=dict(CLI_USAGE_ATTRS, **{"gen_ai.response.id": f"req_{i}"}),
            )
            for i in range(2)
        ]
    )
    assert _post(receiver, body) == 200
    asm.on_close(1, None)
    return client.spans


def _carries_gen_ai_usage(span) -> bool:  # noqa: ANN001
    """Usage in the gen_ai block OR a `gen_ai.usage.` top-level extra.

    Both channels count, because a backend's usage extraction is a prefix
    rule over the flattened attributes — it cannot tell where a key was born.
    """
    g = span.gen_ai
    from_block = g is not None and any(
        v is not None
        for v in (
            g.input_tokens,
            g.output_tokens,
            g.cache_read_input_tokens,
            g.cache_creation_input_tokens,
            g.reasoning_output_tokens,
        )
    )
    from_extras = any(key.startswith("gen_ai.usage.") for key, _ in span.extra)
    return from_block or from_extras


# --------------------------------------------------------------------------
# R3 — defect B: the underscore spellings ride through to the wire
# --------------------------------------------------------------------------


def test_bridge_never_emits_gen_ai_usage_on_a_conflicted_span(receiver):
    spans = _ambiguous_session(receiver)
    siblings = [s for s in spans if s.name == "execute_step llm_request"]
    assert len(siblings) == 2
    for sibling in siblings:
        extras = dict(sibling.extra)
        leaked = sorted(k for k in extras if k.startswith("gen_ai.usage."))
        assert leaked == [], f"top-level gen_ai usage extras on a conflicted span: {leaked}"
        # The values are kept — demoted into the wardex namespace, not dropped.
        assert extras[_DEMOTED_PREFIX + "gen_ai.usage.cache_read_input_tokens"] == 8000
        assert extras[_DEMOTED_PREFIX + "gen_ai.usage.cache_creation_input_tokens"] == 2000
        assert extras[_DEMOTED_PREFIX + "gen_ai.usage.input_tokens"] == 1000
        assert extras[_DEMOTED_PREFIX + "gen_ai.usage.output_tokens"] == 500
        # Identity keys are NOT demoted: the span really is an LLM call.
        assert extras["gen_ai.request.model"] == "claude-sonnet-4-6"


# --------------------------------------------------------------------------
# R4 — defect D: one LLM call must be reported as usage exactly once
# --------------------------------------------------------------------------


def test_an_ambiguous_join_reports_usage_exactly_once(receiver):
    spans = _ambiguous_session(receiver)
    carriers = [s for s in spans if _carries_gen_ai_usage(s)]
    assert len(carriers) == 1, [s.name for s in carriers]
    assert carriers[0].name.startswith("chat")
    # The demotion is observable, not silent: one bump per demoted span.
    assert counters.get("adapters.anthropic.otel_bridge.usage_demoted_on_conflict") == 2
