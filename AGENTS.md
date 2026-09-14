# AGENTS.md

Conventions for humans and AI agents working in this repository. This is a
constitution (rules and how-to), not a status log.

## Project

wardex-sdk is a multi-language observability SDK for AI agents: a shared Rust
core plus per-language native SDKs (Python first; Node and Java planned). It
captures LLM/tool/agent activity with zero instrumentation and exports it in
OpenTelemetry-native form.

## Build & develop

```bash
uv sync --reinstall-package wardex-sdk  # rebuilds the native module via maturin
cargo build --workspace                 # Rust core
```

Use `--reinstall-package wardex-sdk` whenever Rust changed. A bare `uv sync`
treats the already-installed wheel as up to date and leaves the previous
native module in place, so the Python tests exercise stale code and report a
false green while every Rust test passes.

## Test

```bash
cargo test --workspace                  # Rust
uv run pytest sdks/python/tests -v      # Python (development interpreter)
scripts/check-py310.sh                  # Python on the declared floor, 3.10
```

`cargo test --workspace` passes neither `--ignored` nor `--include-ignored`, so
it leaves all eight `#[ignore]`d Rust checks unrun. The section below names
them and gives the commands that do run them.

Run `scripts/check-py310.sh` before pushing anything under `sdks/python`,
source *or* tests — CI's 3.10 job imports both, so a fixture is as easy a place
to strand the floor as the runtime is. `sdks/python/pyproject.toml` declares
`requires-python >=3.10` while the development venv is the newest interpreter,
so a 3.11+-only API (`asyncio.create_task(..., context=...)` is the one that
actually happened) passes every local run and turns red only in CI's 3.10 job,
after the push.
The script builds its own `.venv-py310` from the `test` dependency group —
which is where every dependency the suite imports is declared — and leaves
`.venv`'s interpreter and dependency set alone. The native module is the one
thing the two share: both venvs install the package editable off
`sdks/python/src`, so the rebuild the floor check forces lands on the same
`_wardex_native.abi3.so` the dev venv imports (harmless while the extension is
abi3, which is why the wheel is). Extra arguments go through to pytest.

### Checks no gate runs

`sdks/python/tests/e2e_split_export_phoenix.py` asserts that an OTLP export
split across several POSTs reassembles into ONE trace at a live receiver. That
property is the receiver's to keep rather than the SDK's, so no in-process test
can reach it — and needing Docker is why neither pytest nor CI may. Its
filename deliberately misses pytest's `python_files`, so nothing collects it and
nothing runs it unless a person does:

```bash
docker run -d --name wardex-e2e-phoenix -p 6006:6006 arizephoenix/phoenix:latest
WARDEX_E2E_PHOENIX=http://127.0.0.1:6006 \
    uv run python sdks/python/tests/e2e_split_export_phoenix.py
```

Run it when you touch the request-splitting encoder, the OTLP transport, or the
caps in `CaptureLimits`. Being uncollected also excludes it from the 3.10 floor
check, which runs pytest — so after editing it, run it once more under
`.venv-py310/bin/python`. Nothing else will catch a 3.11+-only call in there.

Four more files are uncollected for the same reason and run only when a
person does. Each names the change that should trigger it:

| File | Run it when you touch | Needs |
|---|---|---|
| `sdks/python/tests/e2e_agent_sdk_smoke.py` | the Agent SDK adapter or the stream-json parser (it is also the fixture recorder: `WARDEX_RECORD=out.jsonl`) | a working `claude` CLI with auth |
| `sdks/python/tests/e2e_usage_pricing_langfuse.py` | usage fields, the OTel bridge's usage keys, or the OTLP exporter (asserts Langfuse prices the export exactly once; `--dry-run` is the syntax pass for 3.10) | a live Langfuse with `WARDEX_E2E_LANGFUSE*` set |
| `sdks/python/tests/bench_parse_off_loop.py` | the parse-off-loop worker, the GIL release in `parse_llm_semantics`, or anything on the streaming hot path (event-loop drift, throughput, the GIL budget the bench itself prints) | nothing; `--quick` for a smoke run, report the full run |
| `sdks/python/tests/bench_fork_semantics.py` | the fork hook, `Runtime._fork_reinit_us`, or the per-batch pid stamp | nothing |

The Rust core carries eight more. They are marked `#[ignore]`, so `cargo test
--workspace` compiles them and runs none of them — it counts them in the
`ignored` column of its summary and moves on. Exactly one of the eight is gated
anyway, by a second CI step that does pass `--ignored`. The other seven run
only when a person runs them.

| File | Run it when you touch | Needs |
|---|---|---|
| `crates/wardex-pipeline/tests/pii_perf.rs` — `masks_one_mebibyte_under_100ms` | the PII engine, its pattern set, or anything on the masking path | nothing — CI runs this one (`.github/workflows/ci.yml` line 41). Budget: masking 1 MiB of mixed text stays under 100 ms. Locally it must be `--release`; debug builds of the regex crate are ~10x slower |
| `crates/wardex-protocol/src/semantic/mod.rs` — `bench_parse_llm_openai_chat_2k`, `bench_parse_llm_openai_responses_3k`, `bench_parse_llm_anthropic_2k`, `bench_parse_llm_openai_chat_sse_50ev`, `bench_parse_llm_openai_responses_sse_50ev` | the semantic parser, or the usage model on the paths it walks | nothing; run `--release` and report the numbers. Budget: the usage-model additions cost ≤ 10 % on the existing paths |
| `crates/wardex-protocol/src/semantic/mod.rs` — `bench_profile_responses_2k_size_matched` | the same changes — run it alongside the five above, never instead of them | nothing; `--release`. Budget: the Responses path lands within ± 20 % of the Chat path. Only this probe makes that comparison apples-to-apples; the design-named benches differ in body size, 2k against 3k |
| `crates/wardex-protocol/src/usage.rs` — `bench_token_usage_new` | `TokenUsage::new`, or the `Option` plumbing on the usage path | nothing; `--release`. Not a regression gate: the type is new, so there is no "before". A report in whole-digit ns means the `Option` plumbing inlined |

```bash
# the PII guard — the one ignored Rust check a gate also runs
cargo test -p wardex-pipeline --release --test pii_perf -- --ignored

# all seven wardex-protocol probes, the five named benches and the other two
cargo test --release -p wardex-protocol -- --ignored bench_ --nocapture
```

Use `bench_`, not the `bench_parse_llm` filter the source comments give.
Measured with `--list`: `bench_parse_llm` selects five of the seven, leaving
out `bench_profile_responses_2k_size_matched` — the body the ± 20 % comparison
needs — and `bench_token_usage_new`. `bench_` selects all seven.

`sdks/python/tests/test_quality_ratchets.py` fails if an `e2e_*.py` or
`bench_*.py` file under `sdks/python/tests`, or a Rust `#[ignore]`d test under
`crates/` or `bindings/`, has a name this file does not contain somewhere. The
budget each check defends is in its own row above; a check with no budget
written next to it is a check nobody can fail.

That ratchet does not close the class. `scripts/langfuse-mapping-oracle/` is a
hand-run check too — a TypeScript runner plus a vector generator that replays
real encoder bytes through Langfuse's own ingestion code — and it is neither a
file matching those globs nor a Rust test, so nothing turns red if it drops out
of this file. Run it on the same trigger as `e2e_usage_pricing_langfuse.py`; it
needs a Langfuse v4.16.0 checkout with pnpm and tsx, and its README carries the
setup.

## Lint & format

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
uv run ruff check sdks/python examples
uv run ruff format --check sdks/python examples
buf lint
```

## Conventions

- **Language**: all comments, docstrings, and prose are **English**. Code
  identifiers are English.
- **Commits**: Conventional Commits (`feat(scope): ...`, `fix:`, `chore:`,
  `ci:`, `docs:`, `refactor:`). Small and frequent.
- **AI attribution**: commits authored with AI assistance should include a
  `Co-Authored-By:` trailer for the model.
- **License**: Apache-2.0 (SDK). Keep the `LICENSE` and `NOTICE` intact.

## Auto-generated files (do not edit by hand)

- Protobuf Rust types are generated at build time by `crates/wardex-codec/build.rs`
  (prost-build + vendored protoc). Edit `proto/wardex/v1/*.proto`, not generated code.

## Architecture (one mental model)

Thirteen ranked layers with one-way dependency. A unit is a top-level module or
a top-level subpackage of `wardex_sdk`; a unit may import its own rank or any
lower one, never a higher one. Asserted by
`sdks/python/tests/test_import_graph.py`, which reads the block below and fails
if it and the table in that file are not the same fact.

```
rank 0   `_enums.py` · `_hash.py` · `_native.py` · `_suppress.py` · `_version.py` · `_wardex_native`
rank 1   `_limits.py` · `_types.py`
rank 2   `_config.py` · `_hub.py` · `_scope.py`
rank 3   `context/`
rank 4   `_assembly/`
rank 5   `_protocol/`
rank 6   `_semantics/`
rank 7   `_worker.py` · `transport/`
rank 8   `_client.py` · `_finalize.py`
rank 9   `_adapters/` · `_interceptors/`
rank 10  `_runtime.py`
rank 11  `_snapshot_api.py` · `_tracing.py`
rank 12  `__init__.py` · `testing/`
```

- Ranks 2..10 are the order the imports are meant to have, not one they already
  have. Twelve of the twenty-five Python units form a single import cycle, and
  `_hub.py` is its hinge: it hands out the process-global client, so nearly
  every layer below imports it, and to do that it imports `_client.py` and
  `_runtime.py`. No rank is widened to absorb that cycle. The five imports that
  point the wrong way are recorded by name in `_LAYER_RANK_DEBT` and counted
  there as six, because one statement resolves to both a package and a symbol
  re-exported from it. That record may only shrink.
- A rank bounds **imports**, not reachability. `_hub.get_client()` hands a
  rank-8 `Client` to a caller at any rank, and `transport/` and `context/`
  already take it that way without importing `_client.py`. No import-graph rule
  can see that, so do not read the rank table as a closed boundary.
- **Byte-exact / hot-path / heavy** work (protobuf, zstd, gzip, protocol
  parsing) lives in the Rust core: `crates/wardex-protocol`,
  `crates/wardex-codec`, `crates/wardex-pipeline`, `crates/wardex-replay`. Those
  are Rust crates; no Python package answers to any of those names. `_protocol/`
  and `transport/_codec.py` are thin wrappers over the compiled extension
  `_wardex_native`.
- The core is **domain-agnostic**: it never knows the target agent's domain.

## Key paths

| Path | Purpose |
|---|---|
| `crates/wardex-protocol` | HTTP/1·HTTP/2·gRPC·WS·JSON-RPC·SSE parsers |
| `crates/wardex-codec` | Protobuf encode + Zstd/gzip compress (build.rs generates proto) |
| `crates/wardex-core` | Facade rlib re-exporting the core crates |
| `bindings/python` | PyO3 source (built into the wheel by maturin) |
| `sdks/python` | The `wardex-sdk` wheel (Python + native) |
| `proto/wardex/v1` | Wire schema (single source of truth) |
