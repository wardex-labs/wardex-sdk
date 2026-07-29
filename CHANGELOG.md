# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and versions follow PEP 440.

## [Unreleased]

### Added
- `OperationName` gains three members — `execute_step`, `handoff` and
  `evaluate` — and `ToolExecutionType` gains two, `ipc` and `unknown`. All five
  are also declared in `proto/wardex/v1/common.proto`, together with a new
  `LinkReason` enum (`triggered_by`, `handoff_from`, `resumed_from`,
  `retried_from`, `cache_source`). proto is the source of truth for the span
  vocabulary in a multi-language SDK; the three enums are declared so the Node
  and Java adapters generate them rather than re-deriving them from prose. Two
  of the three intentionally fill no message field yet: `gen_ai.operation.name`
  and `wardex.tool.execution_type` already travel as span attributes, and a
  typed field alongside would carry the same value twice.
- `Span.events` and `Span.links` are now encoded. Both fields have been
  declared in `span.proto` since the first release and neither was ever
  filled, so any events or links on a span — including a link's `reason` —
  were dropped whole when the envelope was encoded. They now round-trip in
  both directions, on the wardex envelope **and** on the OTLP export path
  (a link's `reason` has no OTLP-native home, so it travels there as the
  `wardex.link.reason` link attribute). This is additive: no span the SDK
  builds today carries either, so nothing that used to be exported changes.
- `SpanBuilder.set_error(error_type, message="")`, so a manual span that the
  host marks as failed can name what failed. Marking a span
  `set_status(StatusCode.ERROR)` without one is still valid and records
  `error.type = "_OTHER"`, OpenTelemetry's own "no classification available".

### Changed
- **Wire schema break — `wardex.v1` (`CaptureIntegrity` and `CorrelationInfo`).**
  `CaptureIntegrity.limitations` (field 8, `repeated string`) and
  `CorrelationInfo.strategy` (field 6, `string`) are gone. Both tags are
  `reserved`; the replacements are `repeated Limitation limitation_codes = 9`
  and `ParentSource parent_source = 7`, two enums now declared in
  `common.proto` — 37 values and 7 respectively. There is no compatibility
  shim and no dual-write window.

  This is free exactly once and this is that once: no wardex envelope has ever
  left a user process. The default transport is a no-op, `endpoint` defaults to
  `None`, and the only network egress — OTLP — never read either field. Zero
  bytes are deployed and there are zero consumers, so the "break" renames
  something nobody has. The tags are not reused, because both reuse directions
  are unsafe: `repeated string` → packed enum shares wire type 2 and would
  decode old bytes as one enum value per ASCII byte with no error, and a scalar
  `string` → enum is a hard `DecodeError` that fails the whole envelope, so one
  stale span would kill an entire batch.

  On the Python side the same three fields are typed:
  `CaptureIntegrity.limitations` is `tuple[Limitation, ...]`,
  `CorrelationInfo.strategy` is `ParentSource | None`, and
  `InternalSpanLink.reason` is `LinkReason | None`. If you read
  `span.capture_integrity.limitations`, you now get members rather than strings
  — compare against `Limitation.BODY_CAP_EXCEEDED`, not `"body_cap_exceeded"`.
  This also removes a silent failure mode: a filter written against a
  misspelled marker string used to match nothing and report zero, which reads
  identically to "this never happened".
- The Agent SDK adapter's tool spans no longer report
  `strategy = "adapter_hook"` / `"adapter_stream"`. Those values answered
  "which source observed this event" — already carried by `capture_sources` —
  while sitting in the field that means "how was this span's parent derived".
  A tool span now reports no parentage claim at all, keeping what is actually
  known: the framework's `tool_use_id` as `request_id`, and the trust gap
  between the two paths as `confidence` (1.0 from a hook, 0.7 from stream
  content alone).
- Enum values are now mapped to the wire by deriving the proto value name from
  the schema rather than by hand-written tables in the PyO3 binding. Twelve
  such tables are gone. They were a second declaration of a list the `.proto`
  already owns, with nothing making the compiler compare them, so a value added
  to one and forgotten in the other would have flattened silently to
  `UNSPECIFIED` on the wire.
- A tool span from the Agent SDK adapter now reports
  `wardex.tool.execution_type = "unknown"` instead of `"network"`, and an MCP
  stdio tool span reports `"ipc"`. Both used to say `network`, which was
  simply false: wardex does not observe how a CLI's built-in tool (Bash, Read)
  executes, and an MCP call runs over a subprocess pipe. If you filter or group
  on that attribute, the adapter's tool spans move out of the `network` bucket.
- Four limitation markers changed name, and three more were merged away.
  `ws_evicted` → `connection_evicted`, `grpc_compressed` and `ws_compressed` →
  `payload_compressed`, `grpc_parse_failed` and `ws_parse_failed` →
  `frame_parse_failed`, `tool_span_unclosed` → `child_span_unclosed`, and
  `async_connect_unavailable` → `connect_timing_unavailable`. The merged pairs
  reported one fact under two names — which protocol it was is already carried
  by `TransportAttributes.protocol` — and the markers a user would ACT on
  differently all stayed separate (`connection_evicted` points at
  `max_connections`, `unit_evicted` at `max_units`). `capture_integrity.limitations`
  is now a closed vocabulary end to end: an emitter cannot invent a marker
  string, and a dashboard filtering on the old spellings needs updating.
- A span with `status=ERROR` now always carries `error.type`. Two spans shipped
  the pair `is_error=true` with no type: an MCP stdio call that returned a
  JSON-RPC error (now `json_rpc_<code>`, or `tool_error` for a tool result
  flagged `isError`) and a failed Agent SDK tool span (now `tool_error`, or
  `tool_unclosed` when the session ended with the tool still open). An aborted
  agent session's root span reports `session_error` or `agent_error`. An HTTP
  span with a 4xx or 5xx response now carries the status rendered as a string
  (`"429"`, `"500"`), which is what OpenTelemetry's HTTP-client conventions
  prescribe when the instrumentation observed the failure but not its cause;
  the byte seam is exactly in that position. A manual span the host marked
  ERROR without naming a type carries `"_OTHER"`.
- `capture_integrity.request_body_captured` / `response_body_captured` now mean
  "capture was attempted and succeeded", not "the payload is non-empty". A tool
  invoked with `{}` used to be reported as a capture FAILURE on the field the
  dashboard uses to judge whether a replay is trustworthy.
- Manual spans (`wardex.span`/`trace` and the decorators) now carry
  `capture_sources=("manual",)`. They previously carried an empty tuple, which
  made an `execute_tool` span from the decorator structurally different from
  one the adapter produced.
- An Agent SDK session that has not reported a `session_id` now gets a
  wardex-issued `gen_ai.conversation.id` instead of the empty string. An empty
  conversation id collides across every session in any store that keys on it.
  `session_id` itself is now absent rather than `""` when the CLI has not sent
  one.
- `wardex.capture_state_snapshot(snapshot_type=...)` now validates its
  argument. The signature still takes a `str`, and the three known values are
  unchanged; anything else is recorded as `SNAPSHOT_TYPE_UNSPECIFIED` **and**
  marked `snapshot_type_unknown` in `wardex.limitations`. Previously an
  unrecognized value was flattened to `UNSPECIFIED` inside the codec with
  nothing recorded anywhere.
- An inbound sampling decision is now honoured instead of being overridden.
  wardex used to emit `traceparent` with the sampled flag hardcoded to `01`,
  so a request that arrived with `-00` left with `-01` and every downstream
  service recorded a trace its own upstream had declined to sample. Received
  flags now propagate unchanged, and only traces wardex itself originates
  assert `01` — which is still every trace where wardex is the entry point,
  because wardex does not head-sample (retention is decided later by the
  RetentionClassifier). If you relied on the old promotion to force sampling
  downstream, set the flag upstream instead.
- More spans now carry `correlation`, including the ones that start a new
  trace. Manual spans (`wardex.span`/`trace` and the decorators), the Agent
  SDK adapter's session-root `invoke_agent` span, and any interceptor span
  with no ambient parent previously reported `correlation=None`, which read
  as "a parent was expected and lost" and was indistinguishable from a
  deliberate trace root. The only new `strategy` values are `"trace_root"`,
  when a span starts its own trace, and `"header"`, when the parent was
  joined from a W3C `traceparent`; a joined parent used to be reported as
  `contextvar`. The adapter's `chat` and subagent spans still report no
  `correlation` — their parent is chosen by a lookup that can silently fall
  back to the session root, and a `confidence` those edges have not earned
  would be worse than none.
- `wardex.capture_state_snapshot()` called with no active span now emits the
  snapshot instead of discarding it. The snapshot carries
  `wardex.limitations="parent_unresolved"` in its attributes and the all-zero
  `span_id` (OTel's invalid-span id), because no span existed to name.
  Previously the call returned silently and the data was lost with no counter,
  log or marker. `wardex.limitations` is now the SDK's key on this record: a
  value passed in `attributes=` under that key is dropped rather than emitted
  alongside it.
- Emitted spans now carry the trace's `trace_flags` on their span context
  rather than a hardcoded `0`. This is not visible on the wire yet: the OTLP
  span message has no flags field today.
- The plaintext (non-TLS) seam now obeys `capture_mode`, which it previously
  ignored. Two consequences, both of which mean MORE spans on that seam.
  `capture_mode=CaptureMode.ALL` now captures plaintext HTTP; it used to mean
  "everything except plaintext HTTP", so a user who asked for everything
  silently did not get it. And plaintext traffic issued inside a live wardex
  span (a `wardex.span()`, an adapter's `execute_tool` span) is now captured
  the way the identical request over TLS always was — the two seams used to
  disagree about the same bytes. Link-local addresses are still never
  captured, and an `intercept_hosts` allowlist match still bypasses the mode
  entirely. If the extra plaintext spans are unwanted, the lever is the same
  one it always was: leave `capture_mode` at its `AGENT` default and do not
  wrap the calls in a wardex span.
- A `capture_mode` the SDK cannot read now falls back to the `agent` default
  instead of to `all`. The field is typed `CaptureMode` and is not validated,
  so a value like the string `"agent"` is accepted in silence; it previously
  fell through to the `agent` policy by accident, and only "wardex is not
  configured at all" ever meant "filter nothing". That is now what the code
  says. Nothing changes for a `capture_mode` set to a `CaptureMode` member.

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
  never silent, but it is reported differently per protocol: an HTTP/1
  message sets `capture_integrity.truncated` and adds `body_cap_exceeded` to
  `capture_integrity.limitations`; an HTTP/2 transaction carries no
  `limitations` field and signals the cap through `truncated` alone.

### Added
- Framework adapter for the Anthropic Agent SDK (`claude_agent_sdk`),
  auto-installed at `init()` when the package is importable. It emits an
  `invoke_agent` span per run (and one per subagent), with
  `execute_tool <name>` children correlated back to the turn that issued the
  call — no instrumentation in your code. Spans are assembled from the SDK's
  own stream and hook events, so a tool's input is recorded as it was sent
  rather than re-serialized. Opt out with `wardex.init(adapters=())`, or pin
  an explicit set with `adapters=(AdapterName.ANTHROPIC_AGENT_SDK,)`. An
  adapter that fails to install prints a warning and leaves the rest of the
  SDK running.
- `CaptureLimits` — every resource bound in the SDK is now configurable via
  `wardex.init(limits=CaptureLimits(...))`, with two exceptions named below.
  The core owns the default values; the Python class holds overrides only, and
  a test asserts the two can never drift apart. A second test drives each
  limit through the code path that enforces it, so a bound cannot be
  advertised here while doing nothing — and any bound that cannot be driven
  that way has to be listed as inert instead of quietly skipped.
- Two limits are inert today and have no effect when set: `replay_buffer_size`
  (nothing in the SDK reads it) and `zstd_level` (read only by the envelope
  encoder, which no live export path calls — the OTLP exporter neither takes
  limits nor compresses).
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
