/**
 * Langfuse mapping oracle (L2): drive Langfuse's OWN ingestion code over real
 * wardex OTLP bytes — no database, no docker, repeatable.
 *
 * The entry point is the one Langfuse's own servertests use,
 * `OtelIngestionProcessor.processToIngestionEvents`, fed complete
 * ResourceSpans (instrumentation scope included — usage extraction dispatches
 * on the scope NAME, so calling the generic extractor directly would silently
 * skip the dispatch this oracle exists to watch). Costs come from the real
 * `IngestionService.calculateUsageCosts` over the shipped
 * default-model-prices.json.
 *
 * Run (see README.md for the one-time setup of the Langfuse checkout):
 *
 *   LANGFUSE_REPO=/path/to/langfuse VECTORS=/path/to/vectors \
 *     ./node_modules/.bin/tsx run.ts        # from inside the checkout, or
 *   cd $LANGFUSE_REPO && env <dummy-env> ./node_modules/.bin/tsx \
 *     /path/to/wardex-sdk/scripts/langfuse-mapping-oracle/run.ts
 *
 * What it can NOT prove (L1-only): OTLP receive/queueing, the ClickHouse
 * round trip, and the public-API observation transform (the H measurements).
 * Those are recorded as out of scope in the results file.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { createRequire } from "node:module";

const LANGFUSE_REPO = process.env.LANGFUSE_REPO ?? process.cwd();
const VECTORS = process.env.VECTORS;
if (!VECTORS) {
  console.error("VECTORS=<dir produced by make_vectors.py> is required");
  process.exit(2);
}
const OUT = process.env.OUT ?? path.join(VECTORS, "oracle-results.json");

// ---------------------------------------------------------------------------
// tiny assertion ledger — every claim lands in the results file
// ---------------------------------------------------------------------------

type Claim = { id: string; ok: boolean; claim: string; evidence?: unknown };
const claims: Claim[] = [];
function check(id: string, ok: boolean, claim: string, evidence?: unknown) {
  claims.push({ id, ok, claim, evidence });
  console.log(`  ${ok ? "PASS" : "FAIL"}  [${id}] ${claim}` + (ok ? "" : `  ${JSON.stringify(evidence)}`));
}
function close(a: number, b: number): boolean {
  return Math.abs(a - b) <= Math.max(1e-15, 1e-9 * Math.max(Math.abs(a), Math.abs(b)));
}

// ---------------------------------------------------------------------------
// SDK-decoded JSON -> the ResourceSpan shape the processor consumes
// ---------------------------------------------------------------------------

function anyValue(v: unknown): Record<string, unknown> {
  if (typeof v === "boolean") return { boolValue: v };
  if (typeof v === "number") return Number.isInteger(v) ? { intValue: v } : { doubleValue: v };
  if (Array.isArray(v)) return { arrayValue: { values: v.map(anyValue) } };
  return { stringValue: String(v) };
}

function toKvList(attrs: Record<string, unknown>): Array<Record<string, unknown>> {
  return Object.entries(attrs).map(([key, value]) => ({ key, value: anyValue(value) }));
}

function toResourceSpan(decoded: any): any {
  const rs = decoded.resource_spans[0];
  return {
    resource: { attributes: toKvList(rs.resource?.attributes ?? {}) },
    scopeSpans: rs.scope_spans.map((ss: any) => ({
      scope: { name: ss.scope?.name, version: ss.scope?.version, attributes: [] },
      spans: ss.spans.map((sp: any) => ({
        traceId: Buffer.from(sp.trace_id, "hex"),
        spanId: Buffer.from(sp.span_id, "hex"),
        parentSpanId: sp.parent_span_id ? Buffer.from(sp.parent_span_id, "hex") : undefined,
        name: sp.name,
        kind: sp.kind,
        startTimeUnixNano: sp.start_time_unix_nano,
        endTimeUnixNano: sp.end_time_unix_nano,
        attributes: toKvList(sp.attributes ?? {}),
        events: [],
        status: sp.status ?? {},
      })),
    })),
  };
}

// ---------------------------------------------------------------------------
// source watches (L2-S / L2-C / L2-W) — the premises this design hangs on
// ---------------------------------------------------------------------------

const PROCESSOR_SRC = path.join(
  LANGFUSE_REPO,
  "packages/shared/src/server/otel/OtelIngestionProcessor.ts",
);

function scopeBranches(src: string): string[] {
  const out: string[] = [];
  for (const m of src.matchAll(/instrumentationScopeName === "([^"]+)"/g)) {
    if (!out.includes(m[1])) out.push(m[1]);
  }
  return out;
}

function chain(src: string, varName: string): string[] {
  const start = src.indexOf(`const ${varName} =`);
  if (start < 0) return [];
  const body = src.slice(start, src.indexOf(";", start));
  return [...body.matchAll(/rawUsageDetails\["([^"]+)"\]/g)].map((m) => m[1]);
}

// ---------------------------------------------------------------------------
// pricing — the shipped table, matched the way the worker matches it
// ---------------------------------------------------------------------------

type PriceRow = { usageType: string; price: any };

function loadPrices(model: string, DecimalCtor: any): PriceRow[] {
  const table = JSON.parse(
    fs.readFileSync(path.join(LANGFUSE_REPO, "worker/src/constants/default-model-prices.json"), "utf8"),
  );
  // In production the match runs in Postgres (`model ~ match_pattern`), where
  // the inline `(?i)` flag is legal; JS RegExp needs it lifted to the flag
  // argument. Behaviour-preserving for these patterns, which use `(?i)` only
  // as a leading prefix.
  const compile = (pattern: string) =>
    pattern.startsWith("(?i)") ? new RegExp(pattern.slice(4), "i") : new RegExp(pattern);
  const entry = table.find((m: any) => compile(m.matchPattern).test(model));
  if (!entry) throw new Error(`no price entry matches ${model}`);
  const tier = entry.pricingTiers.find((t: any) => t.isDefault) ?? entry.pricingTiers[0];
  return Object.entries(tier.prices).map(([usageType, price]) => ({
    usageType,
    price: new DecimalCtor(price as number),
  }));
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

async function main() {
  const { OtelIngestionProcessor } = await import(
    path.join(LANGFUSE_REPO, "packages/shared/src/server/otel/OtelIngestionProcessor.ts")
  );
  const { IngestionService } = await import(
    path.join(LANGFUSE_REPO, "worker/src/services/IngestionService/index.ts")
  );
  const workerRequire = createRequire(path.join(LANGFUSE_REPO, "worker", "package.json"));
  const { Decimal } = workerRequire("decimal.js");

  const manifest = JSON.parse(fs.readFileSync(path.join(VECTORS!, "manifest.json"), "utf8"));
  const model: string = manifest.model;
  const prices = loadPrices(model, Decimal);
  const src = fs.readFileSync(PROCESSOR_SRC, "utf8");

  // --- L2-S: our scope must fall through to the GENERIC usage extractor. ---
  // Baseline measured on v4.16.0 (the design survey's four plus
  // "@flue/opentelemetry", present in the release). A NEW name here is
  // Langfuse changing the dispatch this oracle's whole reading rests on.
  const branches = scopeBranches(src);
  const baseline = ["genkit-tracer", "ai", "pydantic-ai", "gcp.vertex.agent", "@flue/opentelemetry"];
  check(
    "L2-S",
    branches.every((b) => baseline.includes(b)) && !branches.some((b) => b.startsWith("wardex")),
    "usage extraction scope branches are the known set and none claims wardex.*",
    { branches },
  );

  // --- L2-C: the dotted spellings stay FIRST in the ?? chains (D1). ---
  check(
    "L2-C",
    chain(src, "cacheReadTokens")[0] === "cache_read.input_tokens" &&
      chain(src, "cacheCreationTokens")[0] === "cache_creation.input_tokens",
    "cache_read.input_tokens / cache_creation.input_tokens lead their ?? chains",
    { read: chain(src, "cacheReadTokens"), creation: chain(src, "cacheCreationTokens") },
  );

  // --- L2-W: cache_write has NOT entered the chain (D3 move-condition 1). ---
  check(
    "L2-W",
    !chain(src, "cacheCreationTokens").includes("cache_write.input_tokens"),
    "the semconv cache_write rename has not reached Langfuse — cache_creation stays",
  );

  async function ingest(vectorName: string) {
    const decoded = JSON.parse(fs.readFileSync(path.join(VECTORS!, `${vectorName}.json`), "utf8"));
    const processor = new OtelIngestionProcessor({
      projectId: "oracle-project",
      publicKey: "",
      sdkName: "",
      sdkVersion: "",
    });
    (processor as any).seenTraces = new Set();
    (processor as any).isInitialized = true; // bypass Redis, as the servertests do
    const events = await processor.processToIngestionEvents([toResourceSpan(decoded)]);
    return events.filter((e: any) =>
      ["generation-create", "span-create", "event-create"].includes(e.type),
    );
  }

  function costOf(body: any): { costDetails: Record<string, number>; totalCost: number } {
    // IngestionService.ts gate: model name OR provided usage enables enrichment.
    const enrich =
      Boolean(body.model) || Object.keys(body.usageDetails ?? {}).length > 0;
    if (!enrich) return { costDetails: {}, totalCost: 0 };
    const out = IngestionService.calculateUsageCosts(
      prices,
      { provided_cost_details: {} } as any,
      body.usageDetails ?? {},
    );
    return {
      costDetails: Object.fromEntries(
        Object.entries(out.cost_details ?? {}).map(([k, v]) => [k, Number(v)]),
      ),
      totalCost: Number(out.total_cost ?? 0),
    };
  }

  const results: Record<string, unknown> = {};

  // ---------------- vector A: the normal, post-fix span --------------------
  {
    const [obs] = await ingest("a_normal");
    const usage: Record<string, number> = obs.body.usageDetails;
    const { costDetails, totalCost } = costOf(obs.body);
    results.a_normal = { type: obs.type, usageDetails: usage, costDetails, totalCost };

    check("A", usage.input === 1000, "usageDetails.input = 11000 - 8000 - 2000", usage);
    check("A2", usage.output === 500, "usageDetails.output = 500", usage);
    check("A3", usage.input_cached_tokens === 8000, "cache read lands in its bucket", usage);
    check("A4", usage.input_cache_creation === 2000, "cache write lands in its bucket", usage);
    const inputSum = Object.entries(usage)
      .filter(([k]) => k.startsWith("input"))
      .reduce((a, [, v]) => a + (v ?? 0), 0);
    check("B", inputSum === 11000, "sum of input* buckets equals the inclusive total", inputSum);
    const survivors = [
      "cache_read.input_tokens",
      "cache_read_input_tokens",
      "cache_creation.input_tokens",
      "cache_creation_input_tokens",
    ].filter((k) => k in usage);
    check("C", survivors.length === 0, "no raw spelling survives into the pass-through", survivors);
    const expectedUnit: Record<string, number> = {
      input: 3e-6,
      output: 1.5e-5,
      input_cached_tokens: 3e-7,
      input_cache_creation: 3.75e-6,
    };
    let dOk = true;
    const dEvidence: Record<string, unknown> = {};
    for (const [key, units] of Object.entries(usage)) {
      if (key === "total") continue;
      const ok =
        key in expectedUnit && close(costDetails[key] ?? NaN, (units as number) * expectedUnit[key]);
      if (!ok) dOk = false;
      dEvidence[key] = { units, cost: costDetails[key] };
    }
    check("D", dOk, "every usage bucket is priced at its unit price", dEvidence);
    check("E", close(totalCost, 0.0204), "total cost is the corrected 0.0204", totalCost);
    const uncovered = Object.keys(usage).filter((k) => k !== "total" && !(k in costDetails));
    check("F", uncovered.length === 0, "every usage bucket has a cost counterpart", uncovered);
    check(
      "GEN",
      obs.type === "generation-create",
      "the wardex chat span classifies as a GENERATION",
      obs.type,
    );
  }

  // ---------------- vector B: the pre-fix defect, same stack ---------------
  {
    const [obs] = await ingest("b_buggy");
    const usage: Record<string, number> = obs.body.usageDetails;
    const { costDetails, totalCost } = costOf(obs.body);
    results.b_buggy = { type: obs.type, usageDetails: usage, costDetails, totalCost };
    check("G1", usage.input === 0, "defect A reproduced: input collapses to 0", usage);
    check("G2", close(totalCost, 0.0174), "defect A reproduced: cost -14.7% (0.0174)", totalCost);
  }

  // ---------------- vector C: the conflicted join pair (L2-D) --------------
  {
    const events = await ingest("c_pair");
    const perEvent = events.map((e: any) => ({
      type: e.type,
      name: e.body.name,
      model: e.body.model ?? null,
      usageDetails: e.body.usageDetails,
      ...costOf(e.body),
    }));
    results.c_pair = perEvent;
    const total = perEvent.reduce((a: number, e: any) => a + e.totalCost, 0);
    check(
      "L2-D",
      close(total, 0.0204),
      "one ambiguous-join session bills ONCE (0.0408 would be defect D)",
      { total, perEvent: perEvent.map((e: any) => ({ name: e.name, cost: e.totalCost })) },
    );
    const carriers = perEvent.filter((e: any) => Object.keys(e.usageDetails ?? {}).length > 0);
    check(
      "L2-D2",
      carriers.length === 1 && String(carriers[0].name).startsWith("chat"),
      "exactly one observation carries usage, and it is the chat",
      carriers.map((e: any) => e.name),
    );
  }

  // ---------------- vector D: the reasoning price-coverage finding ---------
  {
    const [obs] = await ingest("d_reasoning");
    const usage: Record<string, number> = obs.body.usageDetails;
    const { costDetails, totalCost } = costOf(obs.body);
    results.d_reasoning = { type: obs.type, usageDetails: usage, costDetails, totalCost };
    const uncovered = Object.keys(usage).filter((k) => k !== "total" && !(k in costDetails));
    // EXPECTED finding, not a failure of wardex: Claude models carry no
    // output_reasoning_tokens price, so the reasoning tier is subtracted from
    // priced output and lands unbilled. Deferred as an ecosystem report.
    check(
      "F2",
      uncovered.length === 1 && uncovered[0] === "output_reasoning_tokens",
      "reasoning tier is the one unpriced bucket (the deferred ecosystem finding)",
      { uncovered, usage, costDetails, totalCost },
    );
  }

  // Optional norm-watch: the semconv Anthropic [26] note (D2's premise).
  const semconv = process.env.SEMCONV_GENAI_REPO;
  if (semconv) {
    const doc = fs.readFileSync(path.join(semconv, "docs/gen-ai/anthropic.md"), "utf8");
    check(
      "NORM",
      /input_tokens.*excludes cached/i.test(doc) || /MUST be added to the Anthropic/i.test(doc),
      "semconv still says Anthropic input_tokens excludes cache (D2 premise)",
    );
  }

  const failed = claims.filter((c) => !c.ok);
  fs.writeFileSync(
    OUT,
    JSON.stringify(
      {
        langfuse_repo: LANGFUSE_REPO,
        note_out_of_scope: "OTLP receive/queueing, ClickHouse round trip, public-API transform (H) — L1 only",
        claims,
        results,
      },
      null,
      2,
    ),
  );
  console.log(`\nresults -> ${OUT}`);
  if (failed.length) {
    console.error(`${failed.length} claim(s) failed`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
