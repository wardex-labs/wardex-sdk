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

    # stack: the official langfuse/langfuse docker compose (v4, last run on
    # 4.50.0), seeded headlessly via LANGFUSE_INIT_*
    WARDEX_E2E_LANGFUSE=http://127.0.0.1:3000 \
    WARDEX_E2E_LANGFUSE_PK=pk-lf-... WARDEX_E2E_LANGFUSE_SK=sk-lf-... \
        uv run python sdks/python/tests/e2e_usage_pricing_langfuse.py

How it reads back, and why each choice is forced by Langfuse v4:

  - `GET /api/public/v2/observations`, never the v1 list. A v4 deployment
    runs in events_only mode, where `/api/public/observations` answers 404
    and the single-observation GET is gone too.
  - Looked up by the trace id and span id the SDK minted, inside a start-time
    window around the run. Never by span name: the name this driver gives a
    span is not the name that is stored, because a span carrying a gen_ai
    block is exported as `chat {model}`.
  - `fields=` names the `usage` group explicitly. The v2 default projection
    is `core,basic`, which has no usage or cost at all; `usage` carries
    usageDetails, costDetails and totalCost (there is no separate cost group).
  - v2 field names: `totalCost` (was `calculatedTotalCost`), `inputUsage`,
    `outputUsage`, `totalUsage` (were `promptTokens`, `completionTokens`,
    `totalTokens`).

What it asserts (claude-sonnet-4-6 Standard-tier prices):

  A  usageDetails buckets:  input=1000, output=500,
                            input_cached_tokens=8000, input_cache_creation=2000
  B  sum(input*) == 11000   (Langfuse's own inclusive-input invariant)
  C  no raw spelling survives as a pass-through bucket
  D  every bucket priced at its unit price (keys derived from usageDetails,
     never hardcoded — a renamed bucket must fail in F, legibly, not KeyError)
  E  totalCost == 0.0204
  F  every usage bucket has a cost counterpart (no-price == ABSENT, not 0)
  G  the pre-fix defect, reproduced on the same stack: input column 0,
     cost 0.0174 (-14.7%)
  H  (measured, not asserted, except the input column): totalUsage and
     usageDetails.total. wardex ships no total bucket (there is no
     gen_ai.usage.total_tokens in the semconv registry), so whatever these
     read is Langfuse's own derivation and varies by version: a 4.5.0 stack
     read 0, 4.50.0 fills in the bucket sum. inputUsage must be
     11000: the input COLUMN is never 0 after the fix.
  L1-C  a reasoning-tier span: expected coverage failure — the reasoning
        bucket is the one usage bucket with no price on Claude models —
        recorded as the ecosystem finding, not a wardex failure. Which bucket
        that is comes from usageDetails, never from a fixed key: Langfuse
        spells it `reasoning.output_tokens` before 4.7.0 and
        `output_reasoning_tokens` from 4.7.0 on.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
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


#: The v2 field groups this driver reads. `core` (ids, times, type) is always
#: returned; `usage` is the one that matters — without it the response has no
#: usageDetails, costDetails or totalCost, and every claim below would read an
#: absent key. `basic` is for one measurement: the name the span was stored
#: under.
_FIELDS = "core,basic,usage"

#: Slack on each side of the start-time window. The window is built from the
#: spans' own start times, which this process stamped, so it only has to
#: absorb rounding, not clock skew between the host and the stack.
_WINDOW_SLACK_NS = 60 * 1_000_000_000


def _iso(ns: int) -> str:
    """`ns` since the epoch as the ISO 8601 UTC instant the v2 API filters on."""
    instant = datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)
    return instant.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _observation(
    base: str,
    auth: str,
    *,
    trace_id: str,
    span_id: str,
    window: tuple[int, int],
    timeout: float = 120.0,
) -> dict | None:
    """The STORED observation of this span, once Langfuse has priced it.

    Keyed on the ids the SDK minted, never on the span name: the name a gen_ai
    span is stored under is `chat {model}`, not the one the driver passed in.
    Langfuse uses the OTLP span id as the observation id, so within the trace
    the match is exact.
    """
    query = urllib.parse.urlencode(
        {
            "traceId": trace_id,
            "fromStartTime": _iso(window[0]),
            "toStartTime": _iso(window[1]),
            "fields": _FIELDS,
            "limit": 50,
        }
    )
    url = f"{base}/api/public/v2/observations?{query}"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        request = urllib.request.Request(url, headers={"Authorization": auth})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                page = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # A 4xx other than 408/429 will not change by asking again (wrong
            # endpoint, bad key, rejected filter): say what the stack said now
            # instead of polling into the deadline.
            if 400 <= exc.code < 500 and exc.code not in (408, 429):
                body = exc.read().decode("utf-8", "replace")[:300]
                print(f"    (query refused: HTTP {exc.code} {body})")
                return None
            print(f"    (poll: {exc})")
        except Exception as exc:  # noqa: BLE001 — polling; the deadline reports
            print(f"    (poll: {exc})")
        else:
            obs = next((o for o in page.get("data") or [] if o.get("id") == span_id), None)
            if obs is not None:
                if "usageDetails" not in obs:
                    # Found but projected without the usage group: waiting
                    # cannot add a field the query did not ask for.
                    print(f"    (observation has no usageDetails; fields={_FIELDS!r})")
                    return None
                # Priced means processed: costDetails is filled once Langfuse
                # has matched the model, which is the read this driver needs.
                if obs.get("costDetails"):
                    return obs
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

    total = obs.get("totalCost")
    report.check(close(total if total is not None else -1, 0.0204), "E: totalCost 0.0204", total)
    report.check(
        close(cost.get("total", -1), 0.0204), "E: costDetails.total 0.0204", cost.get("total")
    )

    uncovered = sorted(set(usage) - {"total"} - set(cost))
    report.check(
        not uncovered,
        "F: every usage bucket has a cost counterpart (no-price == absent)",
        uncovered,
    )

    # H — what Langfuse derives for the total wardex does not ship, and the
    # headline column that must never be 0 again.
    report.record("totalUsage", obs.get("totalUsage"))
    report.record("usageDetails.total", usage.get("total"))
    report.record("inputUsage", obs.get("inputUsage"))
    report.record("outputUsage", obs.get("outputUsage"))
    report.check(
        obs.get("inputUsage") == 11000, "H: the input column is NOT zero", obs.get("inputUsage")
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
    # Each span opens with nothing ambient, so each roots its own trace; the
    # ids it was minted with are how it is found again.
    ids: dict[str, tuple[str, str]] = {}
    starts: list[int] = []
    with wardex_sdk.span(names["a"]) as handle:
        handle.set_gen_ai(_normal_gen_ai())
    ids["a"] = (handle.context.trace_id.hex(), handle.context.span_id.hex())
    starts.append(handle.start_time_ns)
    with wardex_sdk.span(names["b"]) as handle:
        # The defect reproduction (G): the fixed capture path cannot produce
        # this span, so it is hand-assembled through the public API.
        handle.set_gen_ai(_buggy_gen_ai())
    ids["b"] = (handle.context.trace_id.hex(), handle.context.span_id.hex())
    starts.append(handle.start_time_ns)
    with wardex_sdk.span(names["c"]) as handle:
        handle.set_gen_ai(_reasoning_gen_ai())
    ids["c"] = (handle.context.trace_id.hex(), handle.context.span_id.hex())
    starts.append(handle.start_time_ns)
    wardex_sdk.flush(60.0)
    wardex_sdk.close(10.0)
    window = (min(starts) - _WINDOW_SLACK_NS, time.time_ns() + _WINDOW_SLACK_NS)
    report = Report()

    def lookup(key: str) -> dict | None:
        trace_id, span_id = ids[key]
        obs = _observation(base, auth, trace_id=trace_id, span_id=span_id, window=window)
        if obs is not None:
            # Measured, so the log shows why a name lookup can never match:
            # the stored name is not the one passed to `wardex.span()`.
            report.record(f"{key}.stored_name", obs.get("name"))
        return obs

    print("\nfidelity of the defect reproduction")
    report.check(
        counters.get("assembly.builder.gen_ai_usage_not_inclusive") - g5_before == 1,
        "the G5 counter saw exactly the hand-built exclusive block",
    )

    print(f"\nspan A ({names['a']}, trace {ids['a'][0]}): the corrected pipeline, priced")
    obs_a = lookup("a")
    if obs_a is None:
        report.check(False, "span A was stored and priced within the deadline")
    else:
        _assert_priced(report, obs_a, expect_cached=True)

    print(f"\nspan B ({names['b']}, trace {ids['b'][0]}): the defect, on the same stack (G)")
    obs_b = lookup("b")
    if obs_b is None:
        report.check(False, "span B was stored and priced within the deadline")
    else:
        usage_b = obs_b.get("usageDetails") or {}
        total_b = obs_b.get("totalCost")
        report.check(usage_b.get("input") == 0, "G: exclusive input collapses to 0", usage_b)
        report.check(
            close(total_b if total_b is not None else -1, 0.0174),
            "G: the under-billed 0.0174 (-14.7%)",
            total_b,
        )

    print(f"\nspan C ({names['c']}, trace {ids['c'][0]}): reasoning tier price coverage (L1-C)")
    obs_c = lookup("c")
    if obs_c is None:
        report.check(False, "span C was stored and priced within the deadline")
    else:
        usage_c = obs_c.get("usageDetails") or {}
        cost_c = obs_c.get("costDetails") or {}
        uncovered = sorted(set(usage_c) - {"total"} - set(cost_c))
        # EXPECTED ecosystem finding — asserted as such, recorded for the
        # deferred Langfuse price report. The bucket is identified by what
        # it holds (the 200 reasoning units, under a key naming reasoning),
        # not by a spelling: Langfuse renamed it in 4.7.0.
        report.check(
            len(uncovered) == 1
            and "reasoning" in uncovered[0]
            and usage_c.get(uncovered[0]) == 200,
            "L1-C: reasoning is the one unpriced bucket (expected finding)",
            {key: usage_c.get(key) for key in uncovered},
        )
        report.record("reasoning.usageDetails", usage_c)
        report.record("reasoning.costDetails", cost_c)

    print(f"\nmeasurements: {json.dumps(report.records, indent=2)}")
    print(f"\n{'OK' if report.failures == 0 else f'{report.failures} FAILURE(S)'}")
    return 0 if report.failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
