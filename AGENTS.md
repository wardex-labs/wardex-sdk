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
uv sync                                 # builds the native module via maturin
cargo build --workspace                 # Rust core
```

## Test

```bash
cargo test --workspace                  # Rust
uv run pytest sdks/python/tests -v      # Python
```

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

- **Byte-exact / hot-path / heavy** work (protobuf, zstd, protocol parsing)
  lives in the Rust core (`crates/`). Python stays a thin layer.
- The core is **domain-agnostic**: it never knows the target agent's domain.

## Key paths

| Path | Purpose |
|---|---|
| `crates/wardex-protocol` | HTTP/1·HTTP/2·gRPC·WS·JSON-RPC·SSE parsers |
| `crates/wardex-codec` | Protobuf encode + Zstd compress (build.rs generates proto) |
| `crates/wardex-core` | Facade rlib re-exporting the core crates |
| `bindings/python` | PyO3 source (built into the wheel by maturin) |
| `sdks/python` | The `wardex-sdk` wheel (Python + native) |
| `proto/wardex/v1` | Wire schema (single source of truth) |
