# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and versions follow PEP 440.

## [Unreleased]

## [0.1.0b5] - 2026-07-08

### Added
- W3C trace propagation: `continue_trace()`, `get_traceparent()`,
  `get_trace_headers()` (opaque tracestate pass-through), `continue_from_otel()`,
  `WardexMiddleware` (ASGI), `WardexWSGIMiddleware`, and opt-in outbound
  injection via `init(propagate_trace=True, propagate_targets=[...])`
  (httpx/requests/aiohttp).
- `run_in_context()` helper for propagating trace context into threads.

### Changed
- **`capture_mode` defaults to `"agent"`**: LLM-semantic traffic is always
  captured; generic HTTP/gRPC/WS is captured only inside an active local
  wardex span. Set `capture_mode=CaptureMode.ALL` for the previous
  capture-everything behavior.

### Fixed
- Correct span parenting under `asyncio.gather` — spans started concurrently
  under one parent no longer mistake a sibling for their parent (spans now
  fork the current scope via ContextVar). Auto-captured client spans inherit
  the fix.

## [0.1.0b4] - 2026-07-07

### Added
- Background batching: a dedicated daemon worker flushes every 5s
  (`flush_interval`) or when the buffer reaches its threshold; spans are
  flushed at exit (`atexit`) and on SIGINT/SIGTERM via chained signal handlers
  (`flush_on_signals=False` to opt out). Manual `flush()` is no longer required.
- Bounded span buffer (`max_buffer_spans`, default 2048) with drop-oldest
  backpressure; drops are counted and reported in debug mode.
- Fork recovery: the worker respawns lazily in a forked child on first capture
  (Sentry-style PID check).

### Changed
- **`before_send` now runs on the background worker thread** (except during a
  manual `flush()`); callbacks touching shared state must synchronize. If
  `before_send` raises, the envelope is dropped (fail-closed) instead of the
  exception propagating.
- Calling `init()` again now cleanly shuts down the previous client (final
  flush + worker join) before installing the new one.
- PII masking + protobuf/zstd encoding now release the GIL, so application
  threads keep running while the worker encodes.

### Fixed
- `Client` buffer is now thread-safe: concurrent `capture_span` during a flush
  can no longer lose spans (pre-existing race in the copy-then-clear flush).

## [0.1.0b3] - 2026-07-07

### Added
- PII masking: built-in detection for emails, NANP phone numbers, credit cards
  (Luhn-verified), US SSNs, IPv4/IPv6 addresses, ABA routing numbers, IBANs, and
  API-key/token secrets. Masking runs as a single Rust pass on every export path
  (wardex-native and OTLP) right before serialization. Per-category opt-out via
  `pii_disabled_categories`; spans with replacements carry
  `capture_integrity.redacted` (wardex) / `wardex.redacted` (OTLP).

### Changed
- `pii_mode` now **defaults to `PIIMode.MASK`** (secure by default). Set
  `pii_mode=PIIMode.OFF` to restore the previous cleartext behavior.
- `PIIMode.REDACT` / `PIIMode.HASH` now raise `NotImplementedError` at config
  time instead of being silently ignored.

## [0.1.0b2] - 2026-07-04

### Fixed
- `wardex_sdk.__version__` and the SDK version reported in exported telemetry now
  reflect the actual installed version instead of a stale hardcoded string.

## [0.1.0b1] - 2026-07-02

### Added
- Initial public beta.
- Zero-instrumentation interception of LLM HTTP calls (OpenAI, Anthropic) over
  `https`, cleartext `http`, and h2c; `gen_ai` semantics; transport metrics;
  gRPC (grpclib), WebSocket, and MCP stdio capture.
- OpenTelemetry export via `OtlpHttpTransport`.
- Manual span decorators (`@workflow`/`@agent`/`@task`/`@tool`/`@span`).

### Known limitations
- No PII masking (prompts/responses sent in cleartext).
- Manual flush only; no background batching.
- No framework adapters yet.
