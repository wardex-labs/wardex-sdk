# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and versions follow PEP 440.

## [Unreleased]

### Changed
- **`AdapterContext.enter()`, `open_run()` and `rejoin()` take a `describe=`
  callable, and that is where an adapter's own code belongs.** It runs inside
  the same failure boundary as the open, before the host's block, so opening a
  span and describing it succeed or fail together. Described in the `with` body
  instead, a framework attribute that moved between releases ships a span
  reading `status=OK` with full input and output, a real duration, and an
  arbitrary suffix of its markers silently gone — indistinguishable downstream
  from a complete observation. `RunHandle` is now a `Scope`, which deletes four
  members it had been carrying in duplicate.
- **An `execute_tool` span from the Agent SDK adapter no longer publishes a
  `correlation`.** It used to carry `confidence` 1.0 (hook path) or 0.7
  (stream-only) with no `parent_source` beside it — which encodes on the wire as
  `parent_source = UNSPECIFIED`, indistinguishable from a sender that never set
  the field. The number was never about the parent edge either: it was the
  observation channel wearing a certainty's clothes, so anyone filtering on low
  confidence was selecting spans wardex had watched from a different vantage
  point rather than spans whose place in the tree was a guess. The channel now
  travels where it means something — a stream-reconstructed span carries
  `stdio` in `capture_sources` — and the framework's `tool_use_id` keeps its
  own home as the tool's `call_id`.

### Fixed
- **`wardex.span()` and `wardex.trace()` no longer raise into the block they
  wrap.** The SDK's own published context manager had the same hole as the
  adapter surface and on a shorter path to a user: latching the active scope,
  resolving the parent edge, building the draft, installing the span as the
  active parent, reading the client and handing it the finished span all ran
  outside any failure boundary, and any of them raised out of
  `with wardex.span(...)` into code that has nothing to do with wardex. Each is
  contained now, the block always runs, and the builder it receives is one the
  host can still drive — over a draft nothing will emit, so a lost span costs a
  span. A carrier that could not be installed costs only the attachment of work
  inside the block, which is reported separately, because that span itself
  still ships.
- **A bug in wardex can no longer break the application it is watching.** The
  adapter surface put the host's own call — a tool handler, a graph node, an
  LLM request — inside a `with` block whose open, activation and close were all
  unguarded, so a defect in wardex's own work would delete that call rather
  than a span. Every step is contained now: the body always runs, the scope it
  receives is total (a `degraded` one answers every verb instead of being
  `None`), and the host's own exception reaches its caller as the SAME OBJECT,
  including `KeyboardInterrupt` and `CancelledError` — wardex must not become
  the library in the process that eats a real Ctrl-C. A failure in wardex's
  teardown can no longer replace the failure the host was in the middle of
  reporting, which is the shape that made a wardex bug read as a host bug in
  the host's own logs.
- **A degraded run is now distinguishable from wardex never having been
  installed.** Every containment above writes one line to stderr naming the
  CONSEQUENCE — "this run will produce NO agent span, and under
  capture_mode=AGENT no HTTP or tool traffic inside it will be captured either"
  — bounded to one line per site per process, however many times the site
  trips. This is the same idiom the SDK already used when an adapter fails to
  load. Where a span still ships it also carries `instrumentation_degraded`,
  and that half is best effort by contract: it reaches for a holder through the
  same registry that just failed. On the wire the marker is deliberately not
  accompanied by a span attribute naming the site, because attributes append
  without a cap while a marker is idempotent.
- **One subtree wardex cannot close no longer costs every other one.** The
  teardown sweep closed every live root under a single failure boundary, so a
  fault anywhere in one root's subtree abandoned the whole table — measured on
  three roots of four children each, with one fault: zero of fifteen spans
  reached the sink. The boundary is now per root, and a root that could not be
  closed is dropped from the table rather than left in it, because leaving it
  meant every later sweep walked back into the same fault and the table never
  emptied. Ten of fifteen ship. The five that do not, and the two units left
  permanently unreachable, are a real loss and are recorded as one.
- **A failure in the lookup table no longer leaks the unit it was indexing.**
  A unit is registered before its aliases are bound, so a fault while binding
  arrived after the unit was already live — and containing it at the caller
  left that unit registered, reachable from nothing, and counting against the
  bound on live units until it evicted a real session to make room for a
  phantom, once per call. It is contained where the state is known instead, so
  the fault costs the alias — the id lookup misses — and the span still ships.
- Two agent runs sharing one trace now say so. One CLI subprocess emits
  `system/init` once, so a second one naming a different run — on a transport
  identity this adapter still holds live — means the earlier subprocess went
  away without its close arriving and CPython handed its address to the next
  object. Everything after it was filed under the earlier run's root with
  nothing in the data to show it; the run's span now carries
  `correlation_conflict` and the event is counted under
  `adapters.assembler.session_key_recycled`. The sessions are not split apart:
  the same symptom would follow from a CLI that legitimately re-initialises one
  transport, and splitting a real run in two to fix a merge is the same mistake
  facing the other way.
- A tool call that **failed** no longer ships as a success when its span was
  reconstructed from the CLI's stdout. `is_error` sits in the result block the
  CLI already sends and nothing read it, so the *arrival* of a result was taken
  for the *success* of the call — and status is the first field anyone filters
  an agent run by.
- **`OtlpHttpTransport` now exports what wardex knows about its own uncertainty.**
  `correlation` and `capture_integrity` were encoded on the wardex envelope and
  dropped entirely by the OTLP encoder — and OTLP is the only transport exported
  from the package root, so on the documented path every "this parent edge is a
  guess" and every "this body was truncated" reached nobody, indistinguishable
  from a span that had nothing to report. They now travel as span attributes,
  the same way a link's `reason` already does: `wardex.parent_source`,
  `wardex.parent_confidence`, `wardex.correlation.{request_id,operation_id,attempt_id}`,
  `wardex.limitations` (a string array), `wardex.capture.{request_headers,
  request_body,response_headers,response_body}` and — only when they happened —
  `wardex.capture.{truncated,redacted,dropped_chunks}`. A span with nothing to
  report still carries none of them. List-valued attributes also decode
  correctly now; `decode_otlp_traces` reported them as unset.
- A tool a **sub-agent** ran now gets its own `execute_tool` span with its own
  result. The Agent SDK writes one line for a tool result and puts two different
  identifiers on it — `parent_tool_use_id` says which sub-agent produced the
  line, the `tool_result` block's `tool_use_id` says which call the result
  answers — and the parser folded both into one field. They are equal for a
  main-agent tool and they diverge for every tool a sub-agent runs, so the
  result was filed against the `Task` call that spawned the agent: `execute_tool
  Task` shipped carrying the inner tool's output, the inner call shipped no
  result at all, and nothing recorded either. A sub-agent's ordinary user
  message is also no longer mistaken for a tool result — it carries
  `parent_tool_use_id` and no result block, and refusing on that field admitted
  it as a result whose content was the whole message.

### Added
- A 38th `Limitation`, `instrumentation_degraded`, declared in
  `proto/wardex/v1/common.proto` and in `wardex_sdk.assembly.Limitation`. Every
  other member of that vocabulary describes a limit of what could be
  **observed** — the framework did not say, the protocol does not carry it, a
  bound was reached. This one describes a limit of **wardex**, and it exists
  because without it the two are indistinguishable downstream and the wrong one
  gets blamed: a subtree missing because the SDK's own instrumentation failed
  looks exactly like a subtree that never ran. Declared only; no span carries it
  yet.
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
- Two resource limits, `max_units` (512) and `max_entries_per_unit` (256), on
  `CaptureLimits` and in `crates/wardex-limits`. `max_units` bounds
  concurrently tracked *root* logical units; `max_entries_per_unit` bounds each
  per-unit table (child units, lookup aliases, de-duplication keys, open span
  drafts). Where the evicted entry has a span — a root unit, a child unit, an
  in-flight span — crossing the bound **closes it and exports it**, marked
  `unit_evicted` or `child_span_unclosed`, because a ceiling that dropped state
  silently would be a worse failure than an unenforced one. The other two
  tables hold no span, so evicting a lookup alias or a de-duplication key
  exports nothing and is recorded only in the internal counters
  `assembly._units.alias_table_full` and `assembly._units.claim_table_full`
  (`wardex_sdk.assembly.counters.snapshot()`). `max_entries_per_unit` also
  bounds the adapter's table of wrapped in-process MCP servers, counted under
  `adapters.anthropic.server_table_full`, so lowering it shrinks that too.
  Those evictions still change what you see. A dropped de-duplication key, or a
  dropped server handle, can let one tool call be reported twice. A dropped
  **alias** is subtler: that identifier stops resolving, the parent is decided
  one rung further down, and if the work carries an ambient wardex span the
  edge arrives at confidence **1.0 with no marker** — hanging off the enclosing
  session rather than the sub-agent it belonged to, so a subtree flattens and
  nothing in the data says so. Only with no ambient span does it ship
  `unit_inferred_sole` (0.5) or `parent_unresolved`. Neither limit is a rename of
  `max_sessions` / `max_session_entries`, which keep their present meaning
  and their consumer: a per-session table and a cap over one flat table of
  sessions are not the same quantity as a cap over units of four kinds sharing a
  single entry point, and reusing the number would silently reinterpret what a
  user set it to.
- A logical-unit registry (`wardex_sdk.assembly.UnitRegistry`, `Unit`,
  `UnitKey`, `UnitKind`). A *unit* is one logical piece of agent work — a
  session, a sub-agent, a graph step, a call — and it is where an adapter gets a
  parent from without ever computing one. A framework identifier can reach it
  only as a `UnitKey`: a lookup alias that selects a unit whose span context
  wardex produced from a real scope read. There is no API that turns an
  identifier into a span context, so the causal tree stays a product of
  in-process context propagation rather than of a framework's callback ids.
  `resolve()` is most-specific-wins and records what it could not establish —
  a cross-trace disagreement ships `correlation_conflict`, a sole-live-unit
  guess ships confidence 0.5 with `unit_inferred_sole`, and an edge that could
  not be established at all ships `parent_unresolved` rather than nothing.
  The Agent SDK adapter is its first consumer: its session and each in-process
  tool call are units, and every other span it emits is anchored to the
  session unit's own context.
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
  The tool span **assembled from hook and stream events** now reports no
  parentage claim at all, keeping what is actually known: the framework's
  `tool_use_id` as `request_id`, and the trust gap between the two paths as
  `confidence` (1.0 from a hook, 0.7 from stream content alone). Which
  sub-agent such a span belongs to is still a heuristic, so it publishes no
  `parent_source`. The **in-process** tool span — the one wardex's own handler
  wrapper opens for a `create_sdk_mcp_server` tool — is the exception and does
  publish its edge, because the unit registry resolved that edge from a real
  scope read: `parent_source = unit_active` at confidence 1.0 with no
  `request_id`. When the session did not reach the handler it falls through
  three further tiers, and only two of them leave a marker: the sole live
  session (`unit_sole`, 0.5, marked `unit_inferred_sole`); failing that the
  ambient wardex span, if the task carries one (`contextvar`, 1.0, **no
  marker** — the tool hangs off whatever span enclosed it rather than off a
  session); and finally `unresolved` (0.0, marked `parent_unresolved`).
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
  differently all stayed separate: `connection_evicted` points at
  `max_connections`, while `unit_evicted` points at a *unit* bound — `max_units`
  in the registry, or `max_sessions` in the Agent SDK adapter's own session
  table. (`unit_evicted` also rides the **new** root that continues a run whose
  predecessor was evicted, which is what separates "this run was truncated and
  resumes here" from "a second root appeared from nowhere".)
  `capture_integrity.limitations` is now a closed vocabulary end to end: an
  emitter cannot invent a marker string, and a dashboard filtering on the old
  spellings needs updating.
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

### Fixed
- A call the provider **refused** — a 429 rate limit, a 401, a 5xx — lost the
  identity its own request had already established. `gen_ai.provider.name`,
  `gen_ai.operation.name`, `gen_ai.request.model` and the request parameters
  were all dropped, and the span was marked `semantic_parse_failed`, which was
  false: the body parsed correctly and was an error envelope. Worse, the same
  predicate fed the capture policy, so under the default `capture_mode="agent"`
  a rate-limited call outside a wardex span produced **no span at all** — the
  call an operator goes looking for was the one guaranteed to be missing.
  The two questions are now separate: whether the RESPONSE yielded gen_ai truth
  (tokens, a response model) and whether the REQUEST identified an LLM call
  (provider, operation, model). A refusal answers the second, so it is now
  captured and carries the same gen_ai block and the same
  `gen_ai.input.messages` a successful call carries — **a behaviour change**:
  the response status used to decide, silently, both whether a span existed and
  what it could hold, and one prompt was therefore exported on success and
  dropped on failure. Response-side fields stay empty, including
  `gen_ai.output.type`, which the parser sets unconditionally and which would
  otherwise claim the call produced text. `semantic_parse_failed` now means
  what it says: a SUCCESSFUL response wardex could not read.
  Two limits on the fix, both deliberate. Only a 4xx/5xx is admitted this way —
  the provider gates in the parser are substring matches on host and path, so a
  200 from an internal service at an `anthropic`-ish host is genuinely
  ambiguous and stays dropped exactly as before. And a proxied or self-hosted
  endpoint is still invisible: the parser classifies on the response body, and
  an error envelope carries none of the markers it recognizes.
- An agent run still in flight when the process stopped exported **nothing**.
  A session's root span is created by its close, so a run that never reached
  one left no span at all — not a truncated one, not a marked one, and no
  counter moved. An interrupted run and a run that never started produced
  identical data, which is the shape of loss nothing can find later. Shutdown
  now finalizes live sessions: their still-open tool calls and unstopped
  sub-agents are emitted first, then the root, carrying `adapter_uninstalled`
  or `unit_interrupted` depending on how the process ended. This affects
  Ctrl-C, `SIGTERM` (what `docker stop` and a kubelet send), an explicit
  `wardex.close()`, and a second `wardex.init()` — a re-init flushes the
  previous client's live runs into that client rather than abandoning them.
  Under `SIGTERM` the units are closed inside the signal handler, before the
  flush, because there the process ends in the handler and `atexit` never
  runs. That only happens when the signal was left at its default
  disposition: an app that installed its own handler may well keep running,
  and ending its live sessions would be a worse lie than a missing span.

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
- **An in-process MCP tool's span is now part of the agent's trace.** A tool
  registered with `create_sdk_mcp_server` used to be connected to nothing: the
  handler wrapper opened a span with no ambient parent, so it started a trace of
  its own, invisible from the session — and every HTTP request the tool made
  in-process was captured accurately *into that orphan trace*. What you saw
  alongside that orphan depended on one environment variable, and both outcomes
  were wrong:

  * **Default (the CLI prefixes SDK tool names).** You got the call **twice**.
    The suppression meant to prevent that never fired, because the skip list
    held the tool's bare name (`greet`, all the wrapper knows) and was compared
    against the name the CLI reports (`mcp__tools__greet`). So the hook-driven
    span appeared in the session tree *and* the orphan span appeared outside it.
  * **With `CLAUDE_AGENT_SDK_MCP_NO_PREFIX` set**, the CLI reports the bare
    name, the skip list matched, and the suppression did fire — so the tool node
    was **missing from the session tree** entirely and the orphan was the only
    record of the call.

  The tool span is now a child of the session's `invoke_agent` span at
  confidence 1.0, and the session's own span is what the tool body runs inside —
  so in-process HTTP attaches to the tool rather than to an orphan root. The
  edge carries no framework identifier at all: the session is pinned onto the
  task that drives the SDK's transport read loop, and the handler inherits it by
  ordinary ContextVar copying, because the SDK dispatches `tools/call` from that
  loop. `session_id` is still recorded — as a lookup alias and a correlation
  hint, never as a source of parentage.

  Two observers of one call are now arbitrated on a normalized key rather than
  on a raw name, so the double emit is not expressible: the handler wrapper owns
  the call because it wrapped the execution, and the hook observer stands down.
  Where the two names genuinely cannot be reconciled — an unresolved server
  token, or `CLAUDE_AGENT_SDK_MCP_NO_PREFIX` with two servers exporting the same
  bare name — the span says so with `tool_name_collision` instead of guessing.
  In-process tool spans also carry `tool_call_id_unavailable_in_process`: the
  handler is dispatched with `{name, arguments}` and no id, and matching one by
  name and arguments against the stream would be exactly the framework-id
  heuristic this design removes.
- An Agent SDK tool handler is restored to the host's own function on
  `uninstall()`. The wrapper was written directly onto the `SdkMcpTool` instance
  and was never recorded, so it survived uninstall and re-install for the life
  of the process.
- An Agent SDK session evicted at `max_sessions` now emits its root span marked
  `unit_evicted`. It used to be dropped along with its whole subtree, with no
  marker, no counter and no log — a workload above the cap simply stopped
  producing traces.
- Every swallowed failure in the Agent SDK adapter is now counted and, under
  `init(debug=True)`, logged with its traceback. Twelve `except Exception: pass`
  handlers around the transport tee, the patched entry points and the tool
  wrapper made an SDK bug indistinguishable from wardex not being installed.
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
