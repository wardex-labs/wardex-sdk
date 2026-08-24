"""Manual E2E (L1): Langfuse must price wardex's usage exactly once, correctly.

The in-process tests prove what the SDK emits (inclusive totals, dotted
spellings, demoted bridge copies) and the L2 oracle proves what Langfuse's
ingestion code computes from those bytes. What neither can prove is the full
stack — OTLP receive, queueing, ClickHouse, and the public-API observation
transform — so this driver sends REAL exports at a REAL Langfuse and reads
back what was STORED.

Opt-in and never part of the suite: the filename does not match pytest's
`python_files`, so nothing collects it, and it refuses to run without
`WARDEX_E2E_LANGFUSE`. It deliberately imports no pytest and needs no test
dependency group — `math.isclose` carries the numeric claims — so it runs on
a bare interpreter. Being uncollected also puts it outside the 3.10 floor
check: after editing, run it once under `.venv-py310/bin/python` (the
`--dry-run` mode exists exactly for that syntax pass — it builds and encodes
every span and never touches the network).

    # stack: langfuse/langfuse v4.16.0 docker compose, seeded via LANGFUSE_INIT_*
    WARDEX_E2E_LANGFUSE=http://127.0.0.1:3000 \
    WARDEX_E2E_LANGFUSE_PK=pk-lf-... WARDEX_E2E_LANGFUSE_SK=sk-lf-... \
        uv run python sdks/python/tests/e2e_usage_pricing_langfuse.py

What it asserts (claude-sonnet-4-6 Standard-tier prices):

  A  usageDetails buckets:  input=1000, output=500,
                            input_cached_tokens=8000, input_cache_creation=2000
  B  sum(input*) == 11000   (Langfuse's own inclusive-input invariant)
  C  no raw spelling survives as a pass-through bucket
  D  every bucket priced at its unit price (keys derived from usageDetails,
     never hardcoded — a renamed bucket must fail in F, legibly, not KeyError)
  E  calculatedTotalCost == 0.0204
  F  every usage bucket has a cost counterpart (no-price == ABSENT, not 0)
  G  the pre-fix defect, reproduced on the same stack: input column 0,
     cost 0.0174 (-14.7%)
  H  (measured, not asserted, except the input column): totalTokens and
     usage.total are 0 because wardex ships no total bucket — the recorded
     cost of the "no gen_ai.usage.total_tokens" decision — while promptTokens
     must be 11000: the input COLUMN is never 0 after the fix.
  L1-C  a reasoning-tier span: expected coverage failure
        (output_reasoning_tokens priced nowhere on Claude models) — recorded
        as the ecosystem finding, not a wardex failure.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sys
import time
import urllib.parse
import urllib.request
from math import isclose

MODEL = "claude-sonnet-4-6"

_ANTHROPIC_REQ = json.dumps(
    {"model": MODEL, "max_tokens": 1024, "messages": [{"role": "user", "content": "hello"}]}
).encode()

_ANTHROPIC_RESP = json.dumps(
    {
        "id": "msg_e2e_a",
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

_UNIT_PRICE = {
    "input": 3e-06,
    "output": 1.5e-05,
    "input_cached_tokens": 3e-07,
    "input_cache_creation": 3.75e-06,
}


def close(a: float, b: float) -> bool:
    return isclose(a, b, rel_tol=1e-9, abs_tol=1e-15)


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.records: dict[str, object] = {}

    def check(self, ok: bool, claim: str, evidence: object = "") -> None:
        if not ok:
            self.failures += 1
        suffix = f"  [{evidence}]" if evidence != "" else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {claim}{suffix}")

    def record(self, key: str, value: object) -> None:
        self.records[key] = value
        print(f"  MEAS  {key} = {value}")


def _normal_gen_ai():
    from wardex_sdk import _wardex_native
    from wardex_sdk._semantics import build_gen_ai

    sem = _wardex_native.protocol.parse_llm_semantics(
        "api.anthropic.com", "/v1/messages", _ANTHROPIC_REQ, _ANTHROPIC_RESP
    )
    assert sem is not None
    return build_gen_ai(sem)


def _buggy_gen_ai():
    """The PRE-fix exclusive shape, by hand — the fixed code cannot make it.

    The G5 counter (`assembly.builder.gen_ai_usage_not_inclusive`) going up by
    one when this block is set is the evidence the reproduction is faithful.
    """
    from wardex_sdk._enums import OperationName, ProviderName
    from wardex_sdk._types import GenAIAttributes

    return GenAIAttributes(
        operation=OperationName.CHAT,
        provider=ProviderName.ANTHROPIC,
        request_model=MODEL,
        response_model=MODEL,
        response_id="msg_e2e_b",
        input_tokens=1000,
        output_tokens=500,
        cache_read_input_tokens=8000,
        cache_creation_input_tokens=2000,
    )


def _reasoning_gen_ai():
    from wardex_sdk._enums import OperationName, ProviderName
    from wardex_sdk._types import GenAIAttributes

    return GenAIAttributes(
        operation=OperationName.CHAT,
        provider=ProviderName.ANTHROPIC,
        request_model=MODEL,
        response_model=MODEL,
        response_id="msg_e2e_c",
        input_tokens=1000,
        output_tokens=500,
        reasoning_output_tokens=200,
    )


def _observation(base: str, auth: str, name: str, timeout: float = 120.0) -> dict | None:
    """The STORED observation of this name, once the worker has priced it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        query = urllib.parse.urlencode({"name": name, "limit": 10})
        request = urllib.request.Request(
            f"{base}/api/public/observations?{query}",
            headers={"Authorization": auth},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                page = json.loads(response.read())
            data = page.get("data") or []
            if data:
                obs = data[0]
                # Priced means processed: costDetails appears when the worker
                # has run the model match, which is the read this driver needs.
                if obs.get("costDetails"):
                    return obs
        except Exception as exc:  # noqa: BLE001 — polling; the deadline reports
            print(f"    (poll: {exc})")
        time.sleep(2.0)
    return None


def _assert_priced(report: Report, obs: dict, *, expect_cached: bool) -> None:
    usage: dict = obs.get("usageDetails") or {}
    cost: dict = obs.get("costDetails") or {}

    report.check(usage.get("input") == 1000, "A: usageDetails.input = 11000 - 8000 - 2000", usage)
    report.check(usage.get("output") == 500, "A: usageDetails.output = 500", usage)
    if expect_cached:
        report.check(usage.get("input_cached_tokens") == 8000, "A: cache read bucket", usage)
        report.check(usage.get("input_cache_creation") == 2000, "A: cache write bucket", usage)

    input_sum = sum(v for k, v in usage.items() if k.startswith("input"))
    report.check(input_sum == 11000, "B: sum of input* buckets is the inclusive total", input_sum)

    survivors = [
        k
        for k in (
            "cache_read.input_tokens",
            "cache_read_input_tokens",
            "cache_creation.input_tokens",
            "cache_creation_input_tokens",
        )
        if k in usage
    ]
    report.check(not survivors, "C: no raw spelling survives as a bucket", survivors)

    # D: keys derived from usageDetails, never hardcoded — an unknown bucket
    # lands in F (legible), not in a KeyError here.
    for key, units in usage.items():
        if key == "total":
            continue
        if key in _UNIT_PRICE:
            report.check(
                close(cost.get(key, float("nan")), units * _UNIT_PRICE[key]),
                f"D: {key} priced at its unit price",
                (units, cost.get(key)),
            )

    total = obs.get("calculatedTotalCost", -1)
    report.check(close(total, 0.0204), "E: calculatedTotalCost 0.0204", total)
    report.check(
        close(cost.get("total", -1), 0.0204), "E: costDetails.total 0.0204", cost.get("total")
    )

    uncovered = sorted(set(usage) - {"total"} - set(cost))
    report.check(
        not uncovered,
        "F: every usage bucket has a cost counterpart (no-price == absent)",
        uncovered,
    )

    # H — the recorded cost of shipping no total bucket, and the headline
    # column that must never be 0 again.
    report.record("totalTokens", obs.get("totalTokens"))
    report.record("usage.total", (obs.get("usage") or {}).get("total"))
    report.record("promptTokens", obs.get("promptTokens"))
    report.record("completionTokens", obs.get("completionTokens"))
    report.check(
        obs.get("promptTokens") == 11000, "H: the input column is NOT zero", obs.get("promptTokens")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="wardex x Langfuse usage/cost E2E driver (L1)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and encode every span, no network — the 3.10 syntax pass",
    )
    args = parser.parse_args()

    import wardex_sdk
    from wardex_sdk._assembly import counters

    run = secrets.token_hex(4)
    names = {
        "a": f"e2e-usage-a-{run}",
        "b": f"e2e-usage-b-{run}",
        "c": f"e2e-usage-c-{run}",
    }

    if args.dry_run:
        from wardex_sdk import _wardex_native
        from wardex_sdk._types import Envelope, EnvelopeHeader, SdkInfo

        # Same builders as the live path, through the real encoder.
        blocks = [_normal_gen_ai(), _buggy_gen_ai(), _reasoning_gen_ai()]
        header = EnvelopeHeader(
            event_id="dry",
            api_key="k",
            sdk=SdkInfo(name="wardex.python", version="0", python_version="3", os="x", arch="y"),
            sent_at_ns=1,
        )
        from wardex_sdk._assembly import AMBIENT, Ambient, SpanDraft, SpanIntent, resolve_parentage
        from wardex_sdk._enums import CaptureSource

        spans = []
        for block in blocks:
            draft = SpanDraft(
                resolve_parentage(Ambient(None, None, None), AMBIENT),
                intent=SpanIntent.CHAT,
                subject=MODEL,
                source=CaptureSource.ADAPTER,
                start_ns=1,
            )
            draft.set_gen_ai(block)
            spans.append(draft.finish(2))
        data = _wardex_native.codec.encode_otlp_traces(Envelope(header=header, spans=tuple(spans)))
        print(f"dry run: encoded {len(spans)} spans, {len(data)} OTLP bytes. No network.")
        return 0

    base = os.environ.get("WARDEX_E2E_LANGFUSE")
    pk = os.environ.get("WARDEX_E2E_LANGFUSE_PK")
    sk = os.environ.get("WARDEX_E2E_LANGFUSE_SK")
    if not (base and pk and sk):
        print(
            "refusing to run: WARDEX_E2E_LANGFUSE / _PK / _SK are unset "
            "(this driver needs a live Langfuse stack; see the module docstring)"
        )
        return 2

    from wardex_sdk.transport import OtlpHttpTransport

    auth = "Basic " + base64.b64encode(f"{pk}:{sk}".encode()).decode()
    wardex_sdk.init(
        transport=OtlpHttpTransport(
            endpoint=f"{base}/api/public/otel/v1/traces",
            headers={"Authorization": auth, "x-langfuse-ingestion-version": "4"},
        ),
        intercept=False,
    )

    g5_before = counters.get("assembly.builder.gen_ai_usage_not_inclusive")
    with wardex_sdk.span(names["a"]) as handle:
        handle.set_gen_ai(_normal_gen_ai())
    with wardex_sdk.span(names["b"]) as handle:
        # The defect reproduction (G): the fixed capture path cannot produce
        # this span, so it is hand-assembled through the public API.
        handle.set_gen_ai(_buggy_gen_ai())
    with wardex_sdk.span(names["c"]) as handle:
        handle.set_gen_ai(_reasoning_gen_ai())
    wardex_sdk.flush(60.0)
    wardex_sdk.close(10.0)

    report = Report()
    print("\nfidelity of the defect reproduction")
    report.check(
        counters.get("assembly.builder.gen_ai_usage_not_inclusive") - g5_before == 1,
        "the G5 counter saw exactly the hand-built exclusive block",
    )

    print(f"\nspan A ({names['a']}): the corrected pipeline, priced")
    obs_a = _observation(base, auth, names["a"])
    if obs_a is None:
        report.check(False, "span A was stored and priced within the deadline")
    else:
        _assert_priced(report, obs_a, expect_cached=True)

    print(f"\nspan B ({names['b']}): the defect, on the same stack (G)")
    obs_b = _observation(base, auth, names["b"])
    if obs_b is None:
        report.check(False, "span B was stored and priced within the deadline")
    else:
        usage_b = obs_b.get("usageDetails") or {}
        report.check(usage_b.get("input") == 0, "G: exclusive input collapses to 0", usage_b)
        report.check(
            close(obs_b.get("calculatedTotalCost", -1), 0.0174),
            "G: the under-billed 0.0174 (-14.7%)",
            obs_b.get("calculatedTotalCost"),
        )

    print(f"\nspan C ({names['c']}): reasoning tier price coverage (L1-C)")
    obs_c = _observation(base, auth, names["c"])
    if obs_c is None:
        report.check(False, "span C was stored and priced within the deadline")
    else:
        usage_c = obs_c.get("usageDetails") or {}
        cost_c = obs_c.get("costDetails") or {}
        uncovered = sorted(set(usage_c) - {"total"} - set(cost_c))
        # EXPECTED ecosystem finding — asserted as such, recorded for the
        # deferred Langfuse price report.
        report.check(
            uncovered == ["output_reasoning_tokens"],
            "L1-C: reasoning is the one unpriced bucket (expected finding)",
            uncovered,
        )
        report.record("reasoning.usageDetails", usage_c)
        report.record("reasoning.costDetails", cost_c)

    print(f"\nmeasurements: {json.dumps(report.records, indent=2)}")
    print(f"\n{'OK' if report.failures == 0 else f'{report.failures} FAILURE(S)'}")
    return 0 if report.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
