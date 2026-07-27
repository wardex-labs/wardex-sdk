# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and versions follow PEP 440.

## [Unreleased]

## [0.2.0b1] - 2026-07-28

### Breaking
- `max_buffer_spans` and `replay_buffer_size` moved from top-level
  `WardexConfig` into `WardexConfig(limits=CaptureLimits(...))`. Passing
  either at the top level now raises a `TypeError` naming the new home; the
  fix is `wardex.init(limits=CaptureLimits(max_buffer_spans=..., replay_buffer_size=...))`.
- Bodies are now capped by content type on both HTTP/1 and HTTP/2: 32 MiB for
  content types carrying extractable meaning (JSON, text, SSE, form-encoded,
  gRPC) and 256 KiB for opaque ones. HTTP/2 previously had a flat 8 MiB cap;
  HTTP/1 had none at all — a large opaque HTTP/1 body (a binary upload,
  say) was captured in full before and is now sampled to 256 KiB. Raise
  `max_opaque_body_bytes` via `CaptureLimits` if you need more. Capping is
  never silent: a capped message reports `body_cap_exceeded` in
  `capture_integrity.limitations` and sets `capture_integrity.truncated`.

### Added
- `CaptureLimits` — every resource bound in the SDK is now configurable via
  `wardex.init(limits=CaptureLimits(...))`, with one exception noted below.
  The core owns the default values; the Python class holds overrides only, and
  a test asserts the two can never drift apart. A second test drives each
  limit through the code path that enforces it, so a bound cannot be
  advertised here while doing nothing.
- `replay_buffer_size` is the exception: it is reserved and currently inert —
  nothing in the SDK reads it, so setting it has no effect.
- `max_buffer_bytes`: a byte budget on the span buffer, bounding resident
  memory independently of span count.

### Fixed
- Non-HTTP traffic over TLS (a Redis, Mongo, or Kafka client sharing the
  process) accumulated in the HTTP/1 parser for the life of the connection —
  an unbounded-memory-growth path. The TLS seam now classifies connections
  the way the plaintext seam always has, and the parser distinguishes
  malformed input from incomplete input instead of treating both the same.
- Chunked HTTP/1 responses were re-parsed from the start on every read: a
  10 MB streaming response scanned roughly 854 MB while holding the GIL. The
  parser is now single-pass — measured at 10.6 MB scanned for the same
  10 MB stream.
- A response with more headers than the parser's fixed-size array could never
  be parsed, and was indistinguishable from traffic that was not HTTP at all.
  The parser now records which of the two happened, and `init(debug=True)`
  prints it once per connection. It is a debug-log line rather than a span
  limitation because no message was ever parsed to attach one to.
- The parser's stream-buffer ceiling measured the whole appended read instead
  of the bytes left unparsed after it, so a request body written in one call —
  what an HTTP client does for a plain `bytes` payload — disabled capture for
  the rest of that connection instead of being governed by the body cap. Every
  later request on a pooled connection was lost, and whether it happened at
  all depended on how the caller wrote the bytes. The same ordering is fixed
  on the JSON-RPC (MCP stdio) parser.
- `CONNECT` and `TRACE` were missing from the HTTP method list used to
  classify traffic, so proxied connections were treated as non-HTTP and
  never captured.
- The adapter's session maps, streamed tool metadata, and subagent maps had
  no bound; only open tools were capped. All four are now bounded by
  `max_session_entries` / `max_sessions`.

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
