"""Produce the wardex-side OTLP vectors the Langfuse mapping oracle consumes.

Each vector is REAL encoder output: spans are built through the SDK's own
capture pieces (the native body parser, `build_gen_ai`, `SpanDraft`, and for
the conflicted-join pair the actual `SessionAssembler` + OTel bridge
receiver), encoded with `_wardex_native.codec.encode_otlp_traces`, and written
as both the wire bytes (`.pb`) and the SDK decoder's JSON view of those same
bytes (`.json` — scope name included, which is the point: Langfuse dispatches
usage extraction on `InstrumentationScope.name`).

Run from the repository root:

    uv run python scripts/langfuse-mapping-oracle/make_vectors.py --out /tmp/oracle-vectors

then feed the directory to `run.ts` (see README.md in this directory).

Vectors:

* ``a_normal``      — one Anthropic exchange with both cache tiers, through
                      the real parse path. Post-fix shape: inclusive 11000.
* ``b_buggy``       — the PRE-fix defect, reproduced by hand: an exclusive
                      ``GenAIAttributes`` block the current code can no longer
                      produce (the G5 counter going to 1 during generation is
                      the fidelity evidence). Drives the regression contrast G.
* ``c_pair``        — a whole session with an AMBIGUOUS chat join: the
                      unmerged CHAT plus two conflicted ``llm_request``
                      increments, straight out of the assembler+bridge. Drives
                      the double-billing check L2-D.
* ``d_reasoning``   — a reasoning-tier span for the price-coverage check
                      (expected finding: ``output_reasoning_tokens`` has no
                      price for Claude models — the deferred ecosystem report).
"""

from __future__ import annotations

import argparse
import http.client
import json
import time
from pathlib import Path

from wardex_sdk import _wardex_native
from wardex_sdk._adapters._assembler import SessionAssembler
from wardex_sdk._adapters._otel_receiver import _OtelBridgeReceiver
from wardex_sdk._adapters._session_state import _BridgeBinding
from wardex_sdk._assembly import (
    AMBIENT,
    Ambient,
    SpanDraft,
    SpanIntent,
    counters,
    resolve_parentage,
)
from wardex_sdk._enums import CaptureSource, OperationName, ProviderName
from wardex_sdk._semantics import build_gen_ai
from wardex_sdk._types import Envelope, EnvelopeHeader, GenAIAttributes, SdkInfo

MODEL = "claude-sonnet-4-6"

_ANTHROPIC_REQ = json.dumps(
    {
        "model": MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }
).encode()

_ANTHROPIC_RESP = json.dumps(
    {
        "id": "msg_oracle_a",
        "model": MODEL,
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


def _header() -> EnvelopeHeader:
    return EnvelopeHeader(
        event_id="oracle",
        api_key="k",
        sdk=SdkInfo(
            name="wardex.python",
            version="0.0.0",
            python_version="3.x",
            os="any",
            arch="any",
        ),
        sent_at_ns=1,
    )


def _span_with(gen_ai: GenAIAttributes, name: str):
    draft = SpanDraft(
        resolve_parentage(Ambient(None, None, None), AMBIENT),
        intent=SpanIntent.CHAT,
        subject=MODEL,
        source=CaptureSource.ADAPTER,
        start_ns=1_000,
    )
    draft.set_gen_ai(gen_ai)
    span = draft.finish(2_000)
    assert span is not None and span.name == name
    return span


def _vector_a():
    sem = _wardex_native.protocol.parse_llm_semantics(
        "api.anthropic.com", "/v1/messages", _ANTHROPIC_REQ, _ANTHROPIC_RESP
    )
    assert sem is not None
    return [_span_with(build_gen_ai(sem), f"chat {MODEL}")]


def _vector_b():
    before = counters.get("assembly.builder.gen_ai_usage_not_inclusive")
    spans = [
        _span_with(
            GenAIAttributes(
                operation=OperationName.CHAT,
                provider=ProviderName.ANTHROPIC,
                request_model=MODEL,
                response_model=MODEL,
                response_id="msg_oracle_b",
                # The pre-fix exclusive shape, by hand — current code cannot
                # produce it any more.
                input_tokens=1000,
                output_tokens=500,
                cache_read_input_tokens=8000,
                cache_creation_input_tokens=2000,
            ),
            f"chat {MODEL}",
        )
    ]
    bumped = counters.get("assembly.builder.gen_ai_usage_not_inclusive") - before
    # The G5 counter noticing this block is the evidence the reproduction is
    # faithful to the defect it reproduces.
    assert bumped == 1, f"expected the non-inclusive counter to bump once, got {bumped}"
    return spans


def _vector_d():
    return [
        _span_with(
            GenAIAttributes(
                operation=OperationName.CHAT,
                provider=ProviderName.ANTHROPIC,
                request_model=MODEL,
                response_model=MODEL,
                response_id="msg_oracle_d",
                input_tokens=1000,
                output_tokens=500,
                reasoning_output_tokens=200,
            ),
            f"chat {MODEL}",
        )
    ]


class _Recorder:
    def __init__(self) -> None:
        self.spans: list = []

    def capture_span(self, span) -> None:
        self.spans.append(span)


def _vector_c():
    """The ambiguous-join session, through the real assembler and receiver."""
    trace = "ad" * 16
    client = _Recorder()
    receiver = _OtelBridgeReceiver(
        max_body_bytes=64 * 1024, max_spans_per_session=64, max_sessions=8
    )
    try:
        asm = SessionAssembler(client, bridge=receiver)
        receiver.reserve(trace)
        t0 = time.time_ns()
        asm.on_outbound(
            1,
            json.dumps(
                {
                    "type": "user",
                    "session_id": "s-oracle",
                    "message": {"role": "user", "content": "go"},
                }
            ),
            bridge=_BridgeBinding(trace_id_hex=trace, confirmed=True),
        )
        asm.on_inbound(
            1,
            {
                "type": "system",
                "subtype": "init",
                "session_id": "s-oracle",
                "model": MODEL,
            },
        )
        asm.on_inbound(
            1,
            {
                "type": "assistant",
                "session_id": "s-oracle",
                "message": {
                    "id": "msg_oracle_c",
                    "model": MODEL,
                    "stop_reason": "end_turn",
                    "usage": {
                        "input_tokens": 1000,
                        "output_tokens": 500,
                        "cache_read_input_tokens": 8000,
                        "cache_creation_input_tokens": 2000,
                    },
                    "content": [{"type": "text", "text": "done"}],
                },
            },
        )
        asm.on_inbound(
            1,
            {
                "type": "result",
                "subtype": "success",
                "session_id": "s-oracle",
                "is_error": False,
                "num_turns": 1,
                "total_cost_usd": 0.02,
                "duration_ms": 100,
                "duration_api_ms": 80,
            },
        )
        t1 = time.time_ns()
        import _otlp_build  # tests/ helper; sys.path is arranged in main()

        body = _otlp_build.request(
            [
                _otlp_build.span(
                    name="claude_code.llm_request",
                    trace_id=trace,
                    span_id=f"{i + 1:016x}",
                    start_ns=t0,
                    end_ns=t1,
                    attrs={
                        "gen_ai.request.model": MODEL,
                        "gen_ai.response.id": f"req_oracle_{i}",
                        "gen_ai.usage.input_tokens": 1000,
                        "gen_ai.usage.output_tokens": 500,
                        "gen_ai.usage.cache_read_input_tokens": 8000,
                        "gen_ai.usage.cache_creation_input_tokens": 2000,
                    },
                )
                for i in range(2)
            ]
        )
        conn = http.client.HTTPConnection("127.0.0.1", receiver.port, timeout=5)
        try:
            conn.request(
                "POST",
                "/v1/traces",
                body=body,
                headers={"x-wardex-bridge": receiver.token},
            )
            status = conn.getresponse().status
        finally:
            conn.close()
        assert status == 200
        asm.on_close(1, None)
    finally:
        receiver.close()
    siblings = [s for s in client.spans if s.name == "execute_step llm_request"]
    assert len(siblings) == 2, "the join must be ambiguous for this vector to mean anything"
    return client.spans


def _write(out: Path, name: str, spans, meta: dict) -> dict:
    env = Envelope(header=_header(), spans=tuple(spans))
    data = _wardex_native.codec.encode_otlp_traces(env)
    decoded = _wardex_native.codec.decode_otlp_traces(data)
    (out / f"{name}.pb").write_bytes(data)
    (out / f"{name}.json").write_text(json.dumps(decoded, indent=2, default=str))
    scope = decoded["resource_spans"][0]["scope_spans"][0]["scope"]["name"]
    assert scope == "wardex.python", scope  # the dispatch fact the oracle asserts
    return {
        "name": name,
        "pb": f"{name}.pb",
        "json": f"{name}.json",
        "spans": len(spans),
        **meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="directory to write the vectors into")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # `_otlp_build` lives in the test tree on purpose (hand-rolled protobuf,
    # NOT wardex's encoder — the CLI side of the pair must not be
    # self-certified by the encoder under test).
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sdks" / "python" / "tests"))

    manifest = {
        "model": MODEL,
        "scope_name": "wardex.python",
        "vectors": [
            _write(out, "a_normal", _vector_a(), {"kind": "normal"}),
            _write(out, "b_buggy", _vector_b(), {"kind": "buggy_regression"}),
            _write(out, "c_pair", _vector_c(), {"kind": "conflicted_join"}),
            _write(out, "d_reasoning", _vector_d(), {"kind": "reasoning_coverage"}),
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest['vectors'])} vectors to {out}")


if __name__ == "__main__":
    main()
