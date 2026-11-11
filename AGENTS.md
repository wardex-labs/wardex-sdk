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
| `crates/wardex-pipeline/tests/pii_perf.rs` — `masks_one_mebibyte_under_100ms` | the PII engine, its pattern set, or anything on the masking path | nothing — `scripts/check-rust.sh test` runs it, so CI's `rust-test` job and the release gate both do. Budget: masking 1 MiB of mixed text stays under 100 ms. Locally it must be `--release`; debug builds of the regex crate are ~10x slower |
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
scripts/check-rust.sh            # fmt, clippy (with and without features), rustdoc, tests
scripts/check-rust.sh deps msrv  # cargo-deny + cargo-machete; `cargo check` on the declared MSRV
uv run ruff check sdks/python examples
uv run ruff format --check sdks/python examples
buf lint
buf breaking --against '.git#format=git,ref=origin/main'
scripts/check-buf-exceptions.sh '.git#format=git,ref=origin/main'
```

`scripts/check-rust.sh` is the one definition of the Rust gate: CI
(`.github/workflows/ci.yml`) and the release workflow run the same script, so
a check added there is a check every one of them runs. Every lint is `warn` in
`Cargo.toml` and the script passes `-D warnings`, so local `cargo clippy` shows
what CI will refuse without blocking an edit-compile loop. The toolchain is
pinned in `rust-toolchain.toml`; a bump is its own commit, carrying the fixes
for whatever the new release flags.

`buf breaking` fails on an incompatible change to `proto/`. A deliberate one
lands with an `ignore_only` entry under `breaking:` in `buf.yaml`, and the
very next change removes that entry: it excuses a whole file rather than one
field, so left behind it silences the check for that file.
`scripts/check-buf-exceptions.sh` fails CI while a stale entry remains.

## Rust style

The rules live in three files, each entry with its reason: `[workspace.lints]`
in `Cargo.toml` (what the compiler checks), `clippy.toml` (thresholds, the
calls and types the core may not use) and `deny.toml` (what a dependency may
be). Read the reason before working around a rule. The bullets below are the
conventions behind that configuration; where a lint backs one it is named, and
the rest is checked only in review.

Three facts drive every rule. The core runs **inside someone else's process**,
so it never ends, blocks or corrupts the host. It **reports observations**, so
it never publishes a value it did not observe. It **handles captured traffic**,
so that traffic must not leak through a side channel such as a panic message.

- **Panics.** Production code proves an `Option` or `Result` by pattern
  (`let Some(x) = .. else`, a `match` that binds the value). Where an invariant
  truly makes a branch unreachable, `expect("..")` states the invariant ("length
  was just checked"), never the failure, under
  `#[expect(clippy::expect_used, reason = "...")]`. `catch_unwind` exists at
  three sanctioned boundaries (the FFI shield, the fail-closed PII scrub, a
  dependency known to panic on hostile input); a new one is a reviewed decision.
- **Locks.** Recover a poisoned lock with `PoisonError::into_inner` only when
  the protected data cannot be half-updated (a cache of immutable values);
  otherwise return an error. Never panic on poison in a path the host calls
  repeatedly, because the panic then repeats for the life of the process.
- **Indexing on peer-controlled bytes** happens after a bounds check in the
  same function, or through `get`. `str` is sliced with `get`, or the offset is
  provably a char boundary and the `#[expect]` says why.
- **Arithmetic** on lengths read from the wire uses `checked_*` or
  `saturating_*`; a count that cannot fit saturates ("at least this many" is
  true, a wrapped count is not). Plain operators are fine on counters that
  cannot realistically overflow; the release profile keeps overflow checks on.
  A `#[must_use]` result (a parser's `feed`, a `mask_*` call) is never bound to
  `_name` in production code; `let _ =` is caught by `let_underscore_must_use`,
  a named underscore is not.
- **Allocation.** "No change" returns no allocation (`Option<Owned>` or `Cow`).
  Every capacity derived from the wire is clamped by a `Limits` field before
  `with_capacity`. Fixed labels are `&'static str`.
- **Debug.** Production code never formats a value with `{:?}`: the
  `use_debug` lint catches it in code, and `#[error("...")]` format strings,
  which the lint does not see, use `{}` only. So a derived `Debug` on a message
  type is safe and keeps test failures readable. Long-lived parser state that
  buffers captured traffic (the `*Stream`, `*Parser` and connection types)
  writes its own `Debug` that prints lengths and state, never content, and a
  test plants a byte string and asserts that neither the string nor its decimal
  byte form appears in the output.
- **Visibility and docs.** `pub` means a sibling crate uses the item; otherwise
  `pub(crate)`. A doc comment says what the value means: units, caps, who sets
  it, what `None` or empty means (absent on the wire, not yet seen, dropped by
  a limit). A doc that only restates the name is worse than none. These crates
  are workspace-internal, so there is no `#[non_exhaustive]` and no
  `#[deprecated]`: a new variant must break every `match` that has to decide
  what to do with it, and a changed signature is fixed at its callers.
- **Errors** derive `thiserror::Error`; a message is lowercase, has no trailing
  period and names the operation that failed. The cause's text is part of the
  message and is not also exposed through `#[source]`: the Python binding
  flattens errors with `to_string()`, so the message is all a host sees, and a
  chain-walking reporter must not print the cause twice. Never copy a list that
  lives elsewhere into a message; build it from the table the code uses.
- **Comments** state the current contract: why, invariants, units. The history
  of a change ("used to", "moved", "before this commit") belongs in the commit
  message. A comment cites nothing a reader of this public repository cannot
  open; where a pointer would have stood, write the reason itself.
- **Names** are words. Single letters are for closure parameters and short loop
  indices. A helper function exists to name an idea or remove a duplicate, not
  to shorten a function; a long linear state machine is fine as one function.
- **Imports** come after the module doc, inner attributes and `mod`
  declarations and before any other item, in three blank-line-separated groups:
  `std`, external crates, then this workspace's crates together with `crate::`
  and `super::` paths. A `use` scoped inside one function is fine. Prefer
  `crate::` paths over `super::super::`. Multi-file modules are `dir/mod.rs`
  (`self_named_module_files`).
- **Tests** are named as behavior sentences (`redundant_test_prefix`), sit in
  `#[cfg(test)]` modules at the end of the file, and move to a sibling
  `tests.rs` only when they outgrow the code under test. No placeholder
  modules, crates or empty tests.
- **GIL.** Batch-proportional pure-Rust work releases the GIL
  (`py.allow_threads`); per-chunk parser feeds keep it, because a parser shared
  by both directions of a connection must not be re-entered.

**Changing a rule.** A rule changes in the file that holds it, with its reason
rewritten in the same commit, and the commit body says what the codebase gains
and gives up. Adding a lint means fixing or `#[expect]`ing every hit in that
commit; the lint lands at `warn` like every other one except `unsafe_code`.
Removing one means deleting the exceptions written for it. `rustfmt` runs with defaults on purpose: every
non-default option measured on this tree rewrote hundreds of lines for no
reader benefit.

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
rank 0   `_config_checks.py` · `_enums.py` · `_hash.py` · `_native.py` · `_source_paths.py` · `_suppress.py` · `_version.py` · `_wardex_native`
rank 1   `_diagnostics.py` · `_limits.py` · `_types.py`
rank 2   `_config.py` · `_hub.py` · `_scope.py`
rank 3   `context/`
rank 4   `_assembly/`
rank 5   `_protocol/`
rank 6   `_semantics/`
rank 7   `_worker.py` · `transport/`
rank 8   `_client.py` · `_finalize.py`
rank 9   `_adapters/` · `_interceptors/`
rank 10  `_runtime.py`
rank 11  `_decorators.py` · `_snapshot_api.py` · `_tracing.py`
rank 12  `__init__.py` · `testing/`
```

- Ranks 2..10 are the order the imports are meant to have, not one they already
  have. Twelve of the twenty-nine Python units form a single import cycle, and
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
  `crates/wardex-codec`, `crates/wardex-pipeline`. Those
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
