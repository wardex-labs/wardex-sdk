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

## Lint & format

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
uv run ruff check sdks/python
uv run ruff format --check sdks/python
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

Five horizontal layers with one-way dependency, plus a vertical core:

```
Adapters ─┐
          ├─▶ Client ─▶ Pipeline ─▶ Transport ─▶ backend
Interceptors ─┘   (+ Protocol parsers in Rust)
Context (ContextVar) cross-cuts Adapters · Interceptors · Client
```

- **Byte-exact / hot-path / heavy** work (protobuf, zstd, gzip, protocol parsing)
  lives in the Rust core (`crates/`). Python stays a thin layer.
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
