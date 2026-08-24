# Langfuse mapping oracle (L2)

Repeatable, database-free verification that wardex's `gen_ai.usage.*` output
is read, normalized and priced correctly by Langfuse — by running **Langfuse's
own ingestion code** (the same `OtelIngestionProcessor.processToIngestionEvents`
entry point its servertests use, plus the real
`IngestionService.calculateUsageCosts` over the shipped price table) against
**real wardex encoder bytes**.

This is the middle rung of the verification ladder:

* **L1** — a full docker-compose Langfuse stack, driven by
  `sdks/python/tests/e2e_usage_pricing_langfuse.py` (authoritative, run once).
* **L2** — this oracle (repeatable; also re-checks, on every run, the source
  premises the design rests on: scope-branch dispatch, `??` chain order,
  the `cache_write` rename).
* **L3** — golden vectors frozen from an L1/L2 run, in the pytest suite.

## One-time setup

```bash
git clone --depth 1 --branch v4.16.0 https://github.com/langfuse/langfuse "$LF"
cd "$LF"
pnpm install --filter "@langfuse/shared..." --ignore-scripts
pnpm install --filter "worker..." --ignore-scripts
pnpm --filter @langfuse/shared exec prisma generate
(cd packages/shared && ../../node_modules/.bin/tsc)   # build shared's dist for the worker import
pnpm add -w -D tsx --ignore-scripts
```

The shared package validates its environment at import; none of these values
is ever connected to (no DB, no Redis, no S3 — the processor is constructed
the way the servertests construct it):

```bash
export CLICKHOUSE_URL="http://localhost:8123" CLICKHOUSE_USER=default \
       CLICKHOUSE_PASSWORD=x DATABASE_URL="postgresql://u:p@localhost:5432/db" \
       LANGFUSE_S3_EVENT_UPLOAD_BUCKET=dummy
```

## Run

```bash
# 1) vectors, from the wardex repo root (real encoder bytes + decoded JSON):
uv run python scripts/langfuse-mapping-oracle/make_vectors.py --out /tmp/oracle-vectors

# 2) the oracle, from the Langfuse checkout:
cd "$LF" && env LANGFUSE_REPO="$LF" VECTORS=/tmp/oracle-vectors \
    ./node_modules/.bin/tsx /path/to/wardex-sdk/scripts/langfuse-mapping-oracle/run.ts
```

The run writes `oracle-results.json` next to the vectors — the frozen source
for the L3 golden fixtures. A non-zero exit means either a wardex regression
(vectors A/C), a failed defect reproduction (vector B — the same stack must
show the bug the fix removes), or a broken design premise (`L2-S/C/W`).
Vector D's single uncovered bucket (`output_reasoning_tokens`) is the
EXPECTED ecosystem finding, asserted as such.
