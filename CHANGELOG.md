# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and versions follow PEP 440.

## [Unreleased]

### Fixed

- **The OpenAI Responses API was invisible — and streams were worse than
  invisible.** The openai-agents SDK calls `POST /v1/responses` by default
  (one `Runner.run` turn is one Responses call), and wardex had no row for
  it: a non-streaming call yielded no semantics, so under the default
  capture mode the span was DROPPED entirely. A streamed call went through
  the Chat reassembler, which shipped a fabricated empty chat-completion
  JSON as the span's `output_data` under a false `sse_unknown_provider`
  marker. Both are gone: the Responses API parses on both paths (the
  terminal event's snapshot is the streaming truth source; an in-stream
  `error` event's payload is preserved rather than dropped), and the SSE
  reassembler is now selected by the stream's own grammar, never defaulted
  by host.
- **`/v1/messages/count_tokens` and `/v1/messages/batches` are no longer
  chat calls.** Substring path matching classified both as `/v1/messages`,
  and every such span carried a false `semantic_parse_failed` — the parser
  had not failed; the endpoint was never a chat call. Endpoint recognition
  is a last-segment table now: gateway prefixes still match, sub-resources
  no longer do (under the default capture mode such calls are dropped
  outside a local span, which is the pre-existing posture for non-LLM
  traffic).
- **An embeddings span no longer claims text output.** `gen_ai.output.type`
  was stamped `"text"` on every parse; an embeddings response is vectors,
  and the attribute is now absent there. The embeddings REQUEST is parsed
  too: `gen_ai.request.model` (so the span name no longer depends on the
  response-model fallback), `gen_ai.request.encoding_formats`, and
  `gen_ai.embeddings.dimension.count`.
- **Anthropic `output_tokens_details.thinking_tokens` reaches
  `gen_ai.usage.reasoning.output_tokens`.** It was silently discarded by
  the closed usage struct. Streaming too: `message_delta.usage` now
  deep-merges over `message_start.usage`, so the cache tiers,
  `server_tool_use` and the thinking tier survive SSE reassembly instead
  of being reduced to four hand-picked fields.

- **Three resource limits never reached the component that enforces them.**
  `LimitsConfig` accepted all three and reported them back, so the loss was
  invisible from the outside — no exception, no counter, no marker.
  - `max_body_bytes` did not reach the logical-unit registry, which resolved
    the core default instead. A host that lowered the cap still let one unit
    accumulate 32 MiB, and the adapter-side shaping budget derived from that
    cap was 32 MiB too, so a large tool argument was built in full and cut
    afterwards. Observable delta for a host that lowered it: recording 8 MiB
    into a unit configured with `max_body_bytes=64 * 1024` now retains 64 KiB
    where it retained 8192 KiB.
  - `max_entries_per_unit` did not reach the table of wrapped in-process MCP
    servers, which the core documents as one of the tables it sizes. The
    catalog is built before the adapter has a client and resolved the process
    default for itself.
  - `max_connections` was applied to the shared connection-timing store only
    on the FIRST `init()`. `close()` empties that store without dropping it,
    so a host that re-initialised with a different value kept the original for
    the life of the process — and what it saw was a spurious
    `connect_timing_unavailable`, which names no knob.
- **A full per-session table blamed the agent, or said nothing at all.**
  `max_session_entries` bounds four tables inside the Anthropic Agent SDK
  assembler, and three of them reported crossing it wrongly.
  - An evicted OPEN TOOL shipped `child_span_unclosed` with status `ok`. That
    marker names no knob — it says a parent's teardown closed the span — so a
    reader went looking for a close that never happened and concluded the agent
    had abandoned the tool. It now ships `session_entry_table_full`, which
    names `max_session_entries`, with status **UNSET**: the bound stopped the
    observation before the outcome, so `ok` claimed a success nobody watched
    and `error` would report wardex's own full table as a tool failure.
  - An evicted SUB-AGENT was not opened at all — no span, no counter — and
    every span beneath it silently re-parented onto the session root. It is now
    evicted and emitted like a tool, and its span context is remembered so its
    children keep the parent they had.
  - A refused streamed-tool-metadata entry is counted. It owns no span, so
    there is nothing to mark; the refusal direction is unchanged (that table is
    consumed in arrival order, so the oldest entry is the one most likely to be
    read next).
- **A tool that completed after being evicted shipped as a second call.** Its
  span started at the completion instant, so it had a duration of zero that
  dragged tool-latency percentiles down; it carried no marker; on the stream
  path it hung off the session root even when the call belonged to a
  sub-agent; and it was tagged `stdio`, claiming a hook-delivered call had been
  reconstructed from the CLI's stdout. The last of those was wrong
  independently of any eviction — installing the adapter mid-session makes a
  lone `PostToolUse` the normal case — and is fixed at the source.
- **With the OTel bridge on, the CLI's tool duration landed on the wrong
  half.** The merge joins by popping `tool_use_id`, and an evicted call that
  completes puts two records under one id with the truncated half first, so it
  took the CLI's measurement and became the anchor for the CLI's child spans.
  The CLI times the whole call, so its number now goes to the half that
  represents the whole call.

### Added

- **OpenAI Responses API parsing** (non-streaming + SSE): `operation=chat`
  with `openai.api.type=responses`, full request/response semantics —
  `reasoning.effort` -> `gen_ai.request.reasoning.level`,
  `previous_response_id` -> `gen_ai.request.previous_response.id`, `status`
  -> `gen_ai.response.status`, service tiers under
  `openai.request/response.service_tier`, and `output[]` mapped
  item-by-item onto OTel message parts. A `function_call` part's id is the
  `call_id` — the key the next turn's `function_call_output` quotes, so
  tool calls correlate across turns. Streaming takes the terminal event's
  complete snapshot over accumulated deltas; a stream that ends without
  its terminal event is counted
  (`interceptors.seam.stream_unterminated`), and background-mode creates
  (`status=queued`, no usage) are captured as identified, tokenless spans
  by design — their usage exists only on the not-yet-captured
  `GET /v1/responses/{id}` path.
- **The open usage mirror**: every scalar leaf of a provider `usage`
  object rides the span as `wardex.usage.<dotted provider path>`, spelling
  preserved (`wardex.usage.cache_creation.ephemeral_1h_input_tokens`,
  `wardex.usage.server_tool_use.web_search_requests`,
  `wardex.usage.input_tokens_details.cache_write_tokens`, string tiers
  like `wardex.usage.service_tier` included). The normalized
  `gen_ai.usage.*` fields are unchanged and cap-immune; the mirror is what
  a cost pipeline reads without knowing wardex's mapping, and what keeps a
  provider's NEW billing counter from being silently discarded by a typed
  table that predates it.
- **`max_extra_keys`** (core limits table, default 64): bounds the one
  open key family above. Crossing it drops whole leaves and says so —
  marker **`extra_keys_dropped`** (`wardex.v1.Limitation` 44, distinct
  from `otlp_attribute_truncated`, which cuts VALUES on the export
  surface) plus `wardex.usage_leaves.dropped_count` and the
  `interceptors.seam.usage_leaves_dropped` counter, all three gated on an
  identified LLM span.
- Chat Completions: `openai.api.type=chat_completions`, request/response
  `service_tier`, `system_fingerprint`, `reasoning_effort` ->
  `gen_ai.request.reasoning.level`, and `response_format.type` ->
  `gen_ai.output.type=json` for the JSON modes. Anthropic:
  `output_config.effort` -> `gen_ai.request.reasoning.level`.
- **`session_entry_table_full`** (`wardex.v1.Limitation` 43) — a per-session
  table crossed `max_session_entries` and its oldest entry was force-closed and
  emitted to admit a new one. Distinct from `unit_table_full` (40) even though
  the core limits table calls the per-unit knob a generalization of this one:
  they are separate FIELDS, so raising one leaves the other where it was, and a
  reader sent to the wrong knob changes nothing about the marker they are
  looking at.

  **One call, two observations.** An evicted tool call that later completes now
  ships **two** spans with the same `call_id`, both carrying this marker, and
  they OVERLAP: the `UNSET` one is `[start, evicted]` — the window wardex
  actually watched, holding the input bytes — and the other is `[start, end]`,
  the whole call, holding the output bytes. **A latency aggregate must exclude
  the spans carrying this marker AND `UNSET`, or it counts the call twice.**
  What this replaces is a zero-duration phantom that polluted the same
  aggregates with no way to exclude it.
- Nine counters under `adapters.assembler.` for the bound's every site:
  `open_tool_table_full`, `subagent_table_full`, `stream_tool_meta_table_full`,
  `evicted_tool_table_full`, `evicted_subagent_table_full`,
  `tool_completion_after_evict`, `tool_completion_after_evict_duplicate`,
  `tool_close_without_open`, `subagent_stop_after_evict`.
- `assembly._units.record_truncated` counts units whose recorded payload was
  cut by the body cap. The fact was already on each span
  (`capture_integrity.truncated`); the counter answers the aggregate question
  a cap that can now be lowered makes worth asking.


- `wardex_sdk.testing`: `UsageSnapshot` and `read_usage` (a usage reader
  beside `SpanNode`, which stays scoped to causal claims), and the
  `check_usage_totals_are_inclusive` conformance check.
- Diagnostics for usage normalization: `usage_totals_unpaired` /
  `usage_overflowed` getters on the native `LlmSemantics` and
  `ClaudeStreamEvent` (a sub-counter without its total is withheld, never
  invented; overflow withholds and says so), tallied by the SDK as
  `semantics.build_gen_ai.usage_*` and
  `adapters.assembler.stream_usage_*` — plus
  `assembly.builder.gen_ai_usage_not_inclusive`, counted at
  `SpanDraft.set_gen_ai` for hand-built blocks that break the inclusive
  contract (counted, never rejected or rewritten).
- `scripts/langfuse-mapping-oracle/`: a repeatable, database-free oracle
  that runs Langfuse's own ingestion code over real wardex OTLP bytes, plus
  a manual live-stack driver (`sdks/python/tests/e2e_usage_pricing_langfuse.py`)
  and CI-permanent golden vectors frozen from the oracle's outputs.

### Changed

- **`gen_ai.response.finish_reasons` has ONE producer and one spelling per
  fact.** Chat, Anthropic, Responses and the Agent SDK adapter all
  normalize through the same total function into the closed set `stop |
  length | tool_call | content_filter | error`; an UNKNOWN provider value
  passes through in the provider's own spelling instead of being dropped.
  Wire values change accordingly (pre-1.0):

  | provider raw | was emitted | now |
  |---|---|---|
  | anthropic `end_turn` | `end_turn` | `stop` |
  | anthropic `stop_sequence` | `stop_sequence` | `stop` |
  | anthropic `max_tokens` | `max_tokens` | `length` |
  | anthropic `tool_use` | `tool_use` | `tool_call` |
  | anthropic `refusal` | `refusal` | `content_filter` |
  | chat `tool_calls` / `function_call` | raw | `tool_call` |
  | responses `max_output_tokens` (incomplete) | — | `length` |
  | responses `failed` / `cancelled` / `error` event | — | `error` |

  `gen_ai.output.messages[].finish_reason` follows the same function, and
  an unmapped value is now passed through rather than omitted. The OTel
  bridge's passthrough of an external SDK's own `finish_reasons` is
  deliberately NOT normalized — that channel quotes someone else's claim.
- A cross-provider body on another provider's API path (an
  Anthropic-shaped response on `/v1/chat/completions` at an unknown host)
  is no longer cross-parsed as a chat call; dispatch is by
  (provider, API shape).
- Delivery of a resolved limit to its consumer goes through one projection
  (`_limits._LIMIT_DELIVERY`), and five structural guards hold it: every
  consumer parameter is classified, every construction expands the
  projection, the native limits object is never dropped at a parser call,
  every field is observed at its enforcement site through two real `init()`
  round trips, and resolving the process defaults is allowed only where a
  declared row gives a reason.
- The private `SessionAssembler` no longer takes `max_units` /
  `max_entries_per_unit`; the registry owns its own bounds, and an assembler
  that has to build one resolves them from its client's config. A caller that
  supplies a client and omits `max_sessions` now gets the host's value rather
  than the core default.
- The private `McpToolCatalog` no longer takes `max_entries` in its
  constructor; `apply_bound(max_entries=...)` is the single handle.
- `crates/wardex-limits` documents `max_session_entries` the way it already
  documented its sibling: four containers rather than three, which eviction
  reaches the wire and which is only counted, and the marker and counter names
  for each. `max_entries_per_unit`'s own paragraph stops naming
  `child_span_unclosed`, which it stopped emitting when `unit_table_full`
  landed.


- **Anthropic `gen_ai.usage.input_tokens` is now the semconv-inclusive
  total** — a norm-compliance correction, not a behavior preference: the
  gen-ai semantic conventions require `input_tokens = input +
  cache_read_input_tokens + cache_creation_input_tokens` for Anthropic, whose
  wire value excludes the cache tiers, and `GenAIAttributes`' own field
  comment has declared the inclusive contract all along. Shipping the raw
  value made every subtracting backend under-bill and blank the input column
  on cache-hit turns. Measured against Langfuse's shipped claude-sonnet-4-6
  prices: `input=1000, cache_read=8000, cache_creation=2000, output=500`
  displayed input **0** and billed **0.0174** instead of **0.0204** (−14.7%);
  a long-context turn (`input=1200, cache_read=45000`) billed −12.4%. The
  error scales with the fresh (uncached) input of each turn. Applies to all
  three Anthropic parse paths (HTTP body, SSE reassembly, CLI stream-json).
  The cache-tier leaves are unchanged — only the total was wrong. OpenAI
  values are untouched (already inclusive), guarded against over-correction
  by test.
- **`ClaudeStreamEvent.input_tokens` (native module) changed MEANING**, raw →
  inclusive, same correction as above; the field name and type are unchanged.
- **The Agent SDK OTel bridge never re-emits the CLI's `gen_ai.usage.*`
  keys as top-level attributes.** A conflicted `claude_code.llm_request` —
  one the join could not merge — used to re-emit them (in underscore
  spellings) as top-level attributes next to a model key — so one LLM call
  could be priced as two GENERATIONs by any backend that classifies on the
  model attribute, and one SDK stated one fact in two spellings. The
  quantity keys now land under
  `wardex.anthropic_agent_sdk.otel.gen_ai.usage.*` — value preserved, quoted
  rather than asserted — while the identity keys (models, response id) keep
  their names. The demotion applies to EVERY increment the bridge emits, not
  only the conflicted join that exposed it: no increment is ever the
  authoritative reporter of tokens (wardex's own `chat` span is), and a
  conditional demotion rested on "pure increments carry no usage today" —
  unpinned, and false the day the CLI stamps usage on `compaction`. A
  conflicted span thus appears as a GENERATION with no usage and cost 0: the
  honest visual form of an unresolvable correlation, instead of a hidden
  double bill. The guarantee is *billed once*, not *one GENERATION row*: a
  backend may still render the conflicted increments as additional
  GENERATION-typed rows next to the one priced span — token-less and
  cost 0 by construction, not a regression. Watched by the
  `adapters.anthropic.otel_bridge.usage_demoted` counter.
- **BREAKING (`wardex_sdk.testing`): `AdapterSubject` requires
  `usage_expected: Literal["none", "totals", "cache_tiers"]`.** No default,
  deliberately: a default would let a new adapter silently opt out of the
  new inclusive-usage conformance check, and that silent opt-out is the
  exact drift the check exists to stop. Three states rather than a bool
  because two cannot describe an adapter whose runs carry usage totals but
  no cache tier (any OpenAI-path adapter, or an Anthropic workload that
  never caches) — under a bool such a subject failed both branches and the
  suite was unpassable for it. `"none"` is enforced too — an adapter that
  starts shipping usage without declaring its convention fails the suite —
  and `"cache_tiers"` requires a PRESENT tier (`is not None`, so an honestly
  reported 0 on a cold turn counts) so the input half of the inclusivity
  invariant stays non-vacuous.

### Known ecosystem findings (not wardex defects)

- Langfuse's shipped price table has no `output_reasoning_tokens` price for
  any Claude model (or `gpt-5.4-mini`/`gpt-5.4-nano`), so a reported
  reasoning tier is subtracted from priced output and lands unbilled. wardex
  keeps reporting the tier — hiding a normative key to mask a price-table gap
  would be the opposite of what this SDK is for; the oracle asserts the gap
  so its closure is noticed.
- wardex ships no `gen_ai.usage.total_tokens` (the key does not exist in the
  semconv registry). Langfuse's `totalTokens` column therefore reads 0 while
  `promptTokens`, `completionTokens` and every cost figure are exact.
- The "backends subtract the cache tiers back out" claim is live-verified
  for Langfuse only (the mapping oracle ran against a live instance). The
  Phoenix half rests on reading OpenInference/Phoenix cost-tracking source
  and has NOT been confirmed against a live Phoenix ingest — including
  whether Phoenix prices `gen_ai.usage.*` natively, without an OpenInference
  processor in front. The open check is the oracle's Phoenix counterpart
  (the Langfuse e2e driver pointed at a live Phoenix); until someone runs
  it, treat Phoenix cost columns under wardex as unverified.


## [0.5.0b1] - 2026-08-16

This release carries the one deliberate breaking window before the Node and
Java SDKs inherit and freeze the public names. Everything under **BREAKING**
below landed in it; there are no compatibility shims — every moved or removed
spelling is refused with a `TypeError` naming its new home or saying why it is
gone.

### Changed (the breaking batch)

- **BREAKING: internal packages are underscore-private.** `assembly`,
  `protocol`, `semantics`, `adapters` and `interceptors` became `_assembly`,
  `_protocol`, `_semantics`, `_adapters` and `_interceptors`; `pipeline` was
  removed. The public import surface is `wardex_sdk` itself plus the three
  subpackages with a user story — `transport`, `context`, `testing` — and
  internal names that used to leak from the root (`Client`, `NATIVE_OK`,
  stdlib modules) no longer resolve there.
- **BREAKING: `init()` has an explicit keyword-only signature** — its
  parameters are exactly `WardexConfig`'s fields plus `transport=`, and a
  drift test holds the two together. `**config_kwargs` and
  `WardexConfig.from_env()` are gone; environment resolution folded into
  `init()` (explicit argument, then `WARDEX_*`, then the OTel fallbacks).
- **BREAKING: `intercept=True` is the new default.** `init()` is the consent
  and zero-instrumentation capture is the product; `intercept=False` is the
  documented opt-out. Mutating outbound traffic (`propagation`) stays opt-in
  and PII masking stays on.
- **BREAKING: the config groups share one suffix.** `PIIPolicy`,
  `BatchingPolicy`, `PropagationPolicy` and `CaptureLimits` are `PIIConfig`,
  `BatchingConfig`, `PropagationConfig` and `LimitsConfig`. All config
  dataclasses are keyword-only; collection fields accept any iterable and
  canonicalize losslessly; enum-valued fields take enum members, not strings;
  config round-trips as written.
- **BREAKING: `adapters=` is a config group.** `AdaptersConfig(enabled=...,
  anthropic_agent_sdk=AnthropicAgentSdkConfig(...))` replaces the bare
  selection tuple (refused with the new spelling). Selection and per-adapter
  options are separate fields, so configuring an option never disturbs
  auto-detection.
- **BREAKING: the tracing family was renamed for what it does.** `trace()` is
  `conversation()` — it stamps `gen_ai.conversation.id` on everything inside
  and joins the ambient trace, it never rooted one; it gains `id=` (an
  explicit id is used verbatim, so a chat app's turns join one conversation)
  and drops the dead `tags=`. `task()` is `step()`, mapped to `execute_step`
  so its spans appear on operation-keyed dashboards. `SpanBuilder` is `Span`,
  exported; the context manager owns completion. Decorators take
  `attributes=` and support the bare form (`@wardex.tool`); context managers
  take a positional name, decorators an optional keyword name. Using `span()`
  or `conversation()` as a decorator raises a `TypeError` naming the
  decorators.
- **BREAKING: scope writes target the isolation scope.** `set_tag`/`set_user`
  no longer write a process-global scope (cross-tenant bleed under
  concurrency); `isolation_scope()` forks the current scope instead of
  starting empty; `set_user(None)` clears; `set_context()` added. Scope tags
  and user now actually land on exported spans (`user.*`,
  `client.address`) — the whole stratum was write-only before.
- **BREAKING: `before_send` is `before_send_envelope`** — the hook vetoes a
  whole batch, not one event. Frozen contract: synchronous, sees PRE-masking
  data, may run again for a declined batch; return the received envelope to
  send or `None` to drop; raising drops the batch fail-closed with one
  diagnostic line (traceback under `debug=True`).
- **BREAKING: the transport SPI was reshaped around one sanctioned path.**
  `InternalEnvelope` is `Envelope`, exported and opaque (the guaranteed
  surface is `span_count` plus `Transport.encode()`); `Transport.timeout` is
  `export_timeout` (a reflectively-read contract attribute needs a name no
  subclass picks by accident); `set_pii_policy`/`set_limits` demoted to
  private plumbing; `export()`'s return is annotated `Undelivered | None`.
  `wardex_sdk.transport` is the complete implementer home, and the
  export/flush/close threading contract (called from wardex's own worker
  thread, may block up to budget) is recorded on the ABC as the
  cross-language contract.
- **BREAKING: `wardex_sdk.testing` names say what they are.** `Node` is
  `SpanNode`, `read` is `read_spans`, `one` is `exactly_one`, `collapse` is
  `collapse_onto_root`, `bare` is `never_installed`, `installed` is
  `installed_adapter`, `parent_name` is `parent_name_of`, `Live` is
  `LiveAdapter`, `Stalled` is `StalledRun`.
- **BREAKING: wire spellings aligned to OTel semconv.** Cache and reasoning
  token keys use the semconv dot spellings; `execute_tool` payloads ship as
  `gen_ai.tool.call.arguments`/`gen_ai.tool.call.result`; SSE spans map
  `network.protocol.name="http"` with `wardex.transport.protocol="sse"`;
  `url.full` is emitted query-stripped. Backend queries that matched the old
  keys need updating.
- **BREAKING: spans export under YOUR service identity.** Every app used to
  export as `service.name="wardex.python"`; now `service_name=` /
  `WARDEX_SERVICE_NAME` maps to `service.name` (fallback
  `unknown_service:python`), `release` to `service.version`, `environment` to
  `deployment.environment.name`, and the SDK travels only in
  `telemetry.sdk.*` (`telemetry.sdk.name="wardex"`).
- **BREAKING: the middleware pair names its protocols.** `WardexMiddleware`
  (which was the ASGI one, though nothing in the name said so) is
  `WardexAsgiMiddleware`, and `WardexWSGIMiddleware` is `WardexWsgiMiddleware`
  — a symmetric pair with title-cased acronyms.
- **BREAKING: `run_in_context` is `bind_context`.** The function runs
  nothing — it captures the current context at wrap time and returns a bound
  callable for a thread to run later; the old verb promised execution.
- **BREAKING: dead surface was cut.** The `retention=` group
  (`RetentionPolicy`, `RetentionClass`, `CaptureTrigger`) — inert end to end,
  reserved until a backend consumer exists; config `tags=` (no reader;
  `set_tag()` is the tag mechanism and is now wired); `replay_buffer_size`
  and `zstd_level` (inert knobs); `PIIMode.REDACT`/`HASH` (raised
  `NotImplementedError` when selected); `AdapterName.LANGCHAIN`/
  `OPENAI_AGENTS` (selecting one installed nothing); `SessionStatus`,
  `Direction`, `Modality` and `Protocol` left `__all__` (internal
  vocabulary). Each returns when its consumer ships.

### Added (the breaking batch)

- **`py.typed` ships in the wheel**, so the SDK's annotations reach downstream
  type checkers.
- **The frozen environment contract**: `WARDEX_API_KEY`, `WARDEX_ENDPOINT`
  (else `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, else
  `OTEL_EXPORTER_OTLP_ENDPOINT`), `WARDEX_SERVICE_NAME`, `WARDEX_RELEASE`,
  `WARDEX_ENVIRONMENT`, `WARDEX_DEBUG` (can only turn debug ON). A bare
  `wardex.init()` with only `WARDEX_ENDPOINT` set is a working first run, and
  an endpoint with no path gets `/v1/traces` appended at transport
  construction — never written back into the config.
- **`service_name=`**, the app's `service.name` (see the resource-identity
  entry above).
- **`WardexConfigWarning`** — a configuration that is legal but conflicts with
  itself (an endpoint under an explicit `transport=`, PII exemptions under
  `PIIMode.OFF`, an `interceptors=` selection under `intercept=False`, adapter
  options under an `enabled=` that excludes the adapter) is announced as a
  real, filterable warning instead of a debug-gated print.
- **`backend.api_key` is wired**: the default OTLP/HTTP exporter sends it as
  `Authorization: Bearer <key>`. It is excluded from every config `repr` —
  string forms of config objects never contain secret material.
- **`Transport.encode(envelope, *, compress=True) -> tuple[bytes, ...]`** —
  the sanctioned, final path from an envelope to wire bodies: PII masking and
  limits applied, one body per POST, split at `max_otlp_request_bytes`,
  over-cap spans dropped and reported. `OtlpHttpTransport` is written through
  it, so an implementer who copies it copies the masked path.
- **`testing.RecordingTransport`** — the user-facing test double:
  `init(transport=RecordingTransport())`, then assert on `transport.spans`
  as `SpanNode`s.
- **New exports** for every name a public signature mentions: `Span`, `Scope`,
  `Envelope`, `WardexConfig`, `AdaptersConfig`, `AnthropicAgentSdkConfig`,
  `AgentAttributes`, `ToolAttributes`, `ConversationContext`, `CallSite`,
  `ToolDefinition`, `AgentType`, `ToolExecutionType`, `SnapshotType`,
  `BeforeSendEnvelopeCallback`, and `Undelivered` in `wardex_sdk.transport`.
- **`SpanKind.PRODUCER` and `SpanKind.CONSUMER`**, end to end (proto, enum,
  OTLP mapping) — the kinds queue propagation (Celery, Kafka) needs.
- **Diagnostics moved onto the stdlib logger `wardex_sdk`.** Zero-config
  output is byte-identical to the old stderr prints (one line, `[wardex] `
  prefix, resolved against the current `sys.stderr` at emit time), but the
  channel is now routable and silenceable with standard `logging` tools: a
  host handler receives the clean message without the prefix, nothing
  propagates to the root logger, emission never raises into host code, and
  wardex's own diagnostic traffic is excluded from its own capture. A
  `wardex_sdk` logger configured before wardex imports is left untouched.
- **`VERSIONING.md`** — version semantics from 1.0, what 0.x may break, the
  deprecation mechanism, the cross-language contract, and append-only proto
  evolution, recorded as policy and linked from the README.

### Fixed (the breaking batch)

- `propagation.targets` round-trips exactly as written: case-folding for the
  match moved into the header injector (folded once at install), so the
  config no longer reads back lowercased.
- `intercept_hosts=()` no longer collapses into `None` — an empty allowlist
  is a choice ("capture no extra plaintext hosts"), not the absence of one.
- `error.type` is emitted as the span attribute semconv names for it, and the
  status message stays the status message — the exception class used to be
  substituted into `Status.message`, destroying the one field a backend
  renders as "what went wrong".
- `stop_sequences`, `finish_reasons` and `encoding_formats` ship as OTLP
  arrays instead of comma-joined strings, so a backend reads a list, not a
  CSV cell.

### Added

- **Finished work is now a linkable target: the unit registry keeps a bounded
  closed-unit link memory.** An alias opted in with `remember=True` keeps its
  unit's span context after close, and link resolution consults live units
  first, then that memory — so a causal link (`triggered_by`, `resumed_from`)
  can point at a predecessor whose span already shipped, without keeping the
  unit alive and without ever producing a parent edge (a link is causality,
  not containment). The memory is FIFO-bounded by a new core limit,
  `max_link_targets` (default 256, settable via
  `LimitsConfig(max_link_targets=...)`); evictions are counted as
  `assembly._units.link_memory_full` and get no wire marker, because the
  remembered span already shipped — what an eviction can cost is a link on a
  future span. A key answers only for its most recent holder, a breadth-
  evicted alias is forgotten from both lookup paths, and `find()` stays
  live-only. The adapter surface grows `Scope.alias()`, `Scope.run_token()`
  (an opaque per-run string for scoping alias values) and
  `Scope.link(expected=)` — the default keeps every unresolved link counted;
  `expected=False` is for conditional claims whose miss the adapter cannot
  attest as a loss.
- **LangGraph graph edges ship as links where the edge can back it.** A step
  fired by a `join:{a}+{b}:{end}` trigger — the one StateGraph trigger format
  that names its sources — now carries one `TRIGGERED_BY` link per named
  source span, and a run whose config carries a `thread_id` links
  `RESUMED_FROM` to the previous run on the same thread in the same process
  (a new trace, linked — never a fabricated parent across runs). The refusals
  are documented and pinned rather than guessed around: ordinary
  `branch:to:{self}` triggers name only the destination, so plain edges get
  no link; `Send` fan-out copies are structurally ambiguous, so a joined
  push-task source is counted as `link_target_unresolved`, never picked; and
  cross-process resume stays out of scope — nothing persists an identity
  across processes, so a first run on a thread claims nothing and counts
  nothing. One honest side effect to know: a graph joining a CACHED source
  node (its cache hit never crosses the node seam) reports the lost link in
  `adapters.langgraph.link_target_unresolved` — that count is the link that
  genuinely cannot be encoded, not a regression.
- **The Anthropic Agent SDK OTel bridge** — `AnthropicAgentSdkConfig`'s
  `otel_bridge`/`otel_bridge_drain` now gate a real consumer, which is the
  promise their docstring shipped with ("the release that ships these fields
  must contain it") being kept. With `otel_bridge=True` the adapter points
  the Claude CLI's own OpenTelemetry exporter at an in-process loopback
  receiver (127.0.0.1, ephemeral port, secret token, hard body and span caps)
  and merges what arrives into the session tree at close. Merged LLM turns
  take the CLI's own request interval and time-to-first-token and **lose the
  `transport_timing_unavailable_subprocess` marker** — the CLI measured that
  timing inside its own process, so the limitation is genuinely gone; merged
  tool spans keep their IPC times and marker and gain the CLI-measured
  duration as `wardex.anthropic_agent_sdk.otel.tool_duration_ms`. CLI work
  wardex could never see before — hooks, MCP RPCs, bash subprocesses, context
  compaction — appears as `execute_step` spans with CLI-measured times,
  sourced `otel_bridge`. Merged spans carry
  `capture_sources=(adapter, otel_bridge)`; identity PII the CLI stamps on
  every span (`user.*`, `organization.*`) is scrubbed at the receiver
  boundary and the merge admits a NAMED attribute allowlist, never a prefix;
  the CLI's `user_prompt` attribute is dropped (the byte-exact stream keeps
  content authority). An ambiguous join never guesses a parent: the CLI span
  ships as a sibling step span carrying `correlation_conflict`. Off — the
  default — reproduces today's tree byte-for-byte; on-and-failed adds exactly
  one root marker: `otel_bridge_no_data` (41) when a confirmed injection
  produced nothing, `otel_bridge_schema_unknown` (42) when telemetry arrived
  and classified as nothing (the CLI schema is beta). The wire vocabulary
  grows accordingly: `CaptureSource` gains `otel_bridge` (8) — and its first
  Python↔proto parity guard — and the limits table gains
  `max_otel_bridge_body_bytes` / `max_otel_bridge_spans_per_session`. The
  bridge NEVER hijacks: any user `OTEL_*`/`CLAUDE_CODE_ENABLE_TELEMETRY` key
  disables injection for that session with one warning, and the drain runs
  only inside the transport's own async close, only for sessions the bridge
  actually fed — never on the atexit/signal/uninstall paths.
- **`TRACEPARENT` alignment for user-run CLI telemetry.** When the user
  already runs the Claude CLI's telemetry themselves and
  `propagation.enabled=True`, the adapter injects the ambient wardex
  `TRACEPARENT` (and `TRACESTATE` when present) into the subprocess env — and
  nothing else — so the CLI's spans join the host's trace in the *user's*
  backend instead of forming a second, disconnected trace. No ambient
  context, or a `TRACEPARENT` the user set themselves, means no injection.
- **`RemoteGraph` (LangGraph Platform) runs are now visible.** A standalone
  `RemoteGraph.invoke`/`stream`/`ainvoke`/`astream` call used to cross zero
  patched seams and ship nothing; it now ships one `invoke_workflow` span
  carrying `wardex.langgraph.remote: "true"`, named after the remote graph,
  with the platform HTTP request parented underneath it — which is also what
  lets that request pass the default `capture_mode=AGENT` gate instead of
  being dropped. `invoke`/`ainvoke` are covered through their own delegation
  to `stream`/`astream`, the same shape as `Pregel`. The remote run's
  internals execute in another process and remain invisible — the span records
  the call, not the remote tree; `stream_events(version="v3")` uses the
  platform threads API and ships no run span. When `langgraph_sdk` is absent
  the seam declines silently; a moved `RemoteGraph` surface declines with one
  diagnostic line and leaves local run, node and tool spans untouched. Cached
  nodes stay documented-not-instrumented — a cache hit never reaches a seam
  and the absence of a span for work that did not run is honest; the trade-off
  is now stated in the adapter's module docstring.
- **The breadth bound has its own vocabulary word: `unit_table_full`.** When a
  per-unit table crosses `max_entries_per_unit`, the oldest child unit or open
  draft is force-closed and emitted carrying `Limitation.UNIT_TABLE_FULL`
  (`LIMITATION_UNIT_TABLE_FULL = 40` on the wire) instead of
  `CHILD_SPAN_UNCLOSED`. The two facts demanded different next actions from
  one marker: `CHILD_SPAN_UNCLOSED` reports a teardown — go look at what
  closed the session — while a breadth eviction is a capacity knob doing its
  job — raise `max_entries_per_unit` or accept the bound. On the canonical
  300-wide async Send fan-out, the 44 evicted workers now name the knob.
  `CHILD_SPAN_UNCLOSED` keeps every teardown site, and descendants of an
  evicted child keep it too: only the entry that hit the bound has the
  table-full fact to report.
- **An end-of-connection signal for pooled async TLS connections.** wardex now
  patches asyncio's `SSLProtocol.connection_lost` in addition to
  `socket.close`/`_real_close`, which is the only moment a pooled
  `ssl.SSLObject` has: it has no `close()` to observe, and asyncio pins it to
  the protocol for the transport's whole life, so per-connection capture state
  for a keep-alive connection to a model provider used to be released only
  whenever the garbage collector happened to reach it. This is best effort, not
  a guarantee — an event loop that implements TLS itself rather than through
  `asyncio.sslproto` (uvloop), and a protocol subclass that overrides
  `connection_lost` without calling up, both fall back to the finalizer and the
  per-connection caps, which is late but never wrong. Nothing new is held or
  exported; the patch is installed and removed with the rest of the close hook.
- Trace propagation now has a single documented header rule across every
  patched HTTP client (httpx, requests, aiohttp): wardex only ever *adds* a
  header you did not write, and never replaces or duplicates one.
- Inbound `tracestate` headers are vetted at the edge where they arrive:
  values outside printable US-ASCII are dropped, and lists longer than the 32
  members the W3C specification allows are truncated from the right so the
  most recent writers survive.
- Documented what a split OTLP export looks like from your traces. When a
  batch exceeds `max_otlp_request_bytes` it leaves as several POSTs, and the
  README now states why that is still one trace on the other side: the
  requests carry the same trace id and the receiver keys spans by it. Children
  routinely arrive in earlier requests than the parent they name — spans leave
  in completion order, so the root travels last — and a conforming receiver
  resolves the edge when the parent lands.
- Documented the one loss a split cannot absorb: a span too large to fit a
  request even with its payload removed is dropped, the rest of its batch is
  still sent, and the SDK says so on stderr — once per process, at the first
  occurrence, with a count scoped to that batch. This loss cannot appear as a
  `wardex.limitations` marker because the marker would have to ride on the
  span that never reaches the wire.
- An opt-in end-to-end check, `sdks/python/tests/e2e_split_export_phoenix.py`,
  that exports a deliberately over-cap batch through a recording proxy to a
  live OTLP receiver and asserts the reassembly at the receiver: every span
  present, one trace id, parent edges intact across request boundaries, gzip
  accepted as sent, and the oversized-span drop leaving the rest of its batch
  untouched. It requires Docker and an explicit `WARDEX_E2E_PHOENIX` opt-in,
  and its filename is outside pytest's collection patterns, so no suite or CI
  job depends on it.
- **Agent SDK adapter: per-turn prompt capture.** Every main-thread chat span
  now carries its own user turn's prompt (prompts for turns 2+ used to be
  parsed off the wire and discarded), sourced byte-exactly from the stream
  with the `UserPromptSubmit` hook as boundary corroboration and as the
  degraded-content fallback for a write the stream did not record. The
  provenance is published on the span as the `wardex.agent.prompt_source`
  extra (`"stream"` = the outbound message-object JSON slice, `"hook"` = the
  CLI's re-decoded prompt text), because the two shapes differ on the wire
  and a consumer must know which one it is parsing. Assistant turns that had
  no user prompt — the intermediate turns of an agentic loop, and
  subagent-attributed chats — honestly ship `input_attempted=False` instead
  of an empty capture.

### Changed

- **Wire-value change (pre-1.0): an orphan wardex's own eviction caused now
  says so.** A span opened inside the leftover `activate()` scope of a unit
  the registry itself evicted still becomes a marked trace root, but carries
  `INSTRUMENTATION_DEGRADED` instead of `CORRELATION_CONFLICT`: the strand is
  wardex's bound at work, and the repair is `max_units`, not the adapter's pin
  or lifetime discipline. Strands left by ordinary closes keep
  `CORRELATION_CONFLICT`. The refusal itself is unchanged — same predicate,
  same trace-root edge; only the attribution moved, and it moved on every
  reachable path: the registry's own `open()`/`resolve()` refusals and the
  adapter surface's declared sole-live fallback all ask the registry for the
  word (`UnitRegistry.refused_ambient_marker`). The refusal counters gain a
  third name, `assembly._units.stale_ambient_evicted`, so an operator can
  separate "raise the cap" from "fix the adapter".
- **The MCP tool catalogue's internal lock is now reentrant**, which makes
  every lock in the SDK reentrant except one deliberate, documented holdout.
  Since the socket close hook landed, SDK code can be re-entered from a
  weakref finalizer, and CPython runs those wherever a reference count reaches
  zero — on any thread, at a line your application did not write. A
  non-reentrant lock on such a path is a permanent self-deadlock in the host,
  so reentrancy is now the default rather than a per-caller argument. No API
  or behaviour change; the lock is taken per MCP server registration and per
  hook lookup, far below span rate.
- **An HTTP/2 span whose parent wardex itself dropped now says so.** The
  stream correlation latch is capped at `max_streams` so one long-lived h2
  connection cannot accumulate an entry per cancelled stream. A response that
  arrives after its entry was evicted used to ship as a clean trace root at
  confidence 1.0 — indistinguishable from a request the host genuinely issued
  outside any agent work. It now ships `UNRESOLVED` with both
  `parent_unresolved` and `instrumentation_degraded` in `wardex.limitations`,
  so the missing parent reads as wardex's own degradation rather than as a
  fact about your traffic. The claim is bounded at both ends: a stream opened
  before capture attached, or one the cap never reached, is not labelled.
- **The default `AGENT` capture mode no longer drops those spans.** The gate
  reads an absent parent as "not agent work", which would have made the marker
  above unreachable on exactly the traffic that earned it — a silent drop
  instead of a labelled one. A span degraded by an evicted latch entry now
  passes the gate the same way one issued inside a span wardex failed to open
  already did.
- `propagation.targets` glob patterns are matched case-insensitively, since
  hostnames are. (Folding now happens in the injector, once at install — the
  config reads back exactly as written; see the breaking batch above.)
- `intercept_hosts` entries are matched case-insensitively for the same
  reason.
- `get_traceparent()` and `get_trace_headers()` now resolve the same ambient
  scope, so the two can no longer disagree about whether a trace context
  exists. Both read the merged scope, which is the wider of the two
  resolutions the pair used before — no header that was emitted previously
  stops being emitted.
- Both header readers now read only the two propagation fields instead of
  materializing a merged scope. They no longer deep-copy the data you put in
  `set_context()`, which makes them safe to call from a send path when a
  context value holds something that cannot be copied, such as a lock or a
  socket.
- **LangGraph tool-input recording is shaped and bounded at the source.**
  `input_data` is byte-identical to the previous `repr` for plain builtin
  argument shapes, but a framework object inside `call["args"]` (an injected
  `Command`, a message list) now ships as its bare type name instead of a
  full repr, and materialization is bounded by the resolved `max_body_bytes`
  with the `truncated` flag set on overflow. Wire-visible only for
  non-builtin argument values; pre-1.0.
- **The Agent SDK adapter no longer injects the `Stop` hook.** Its payload
  carries `stop_hook_active` and nothing else — no timestamps — so there was
  nothing it could correct that the same control channel had not already
  delivered, and each injected hook costs a blocking control-protocol round
  trip the CLI awaits at the end of every turn. `UserPromptSubmit` is now
  consumed (see per-turn prompt capture above), so every hook event the
  adapter injects has a consumer.
- **Interrupted tool calls now ship `error.type="tool_interrupted"`** (a wire
  value change, pre-1.0), read from the `PostToolUseFailure` payload's
  `is_interrupt` flag; `"tool_error"` remains the fallback for every other
  failure, including stream-only observations, whose result block carries no
  interrupt signal.

### Fixed

- **A forked child could stop draining forever.** If `os.fork()` happened
  while another thread was in the middle of starting the background batch
  worker, the child inherited an "a spawn is in progress" flag that nothing in
  the child was left to clear — the frame that would have cleared it does not
  survive the fork. Every later respawn attempt then declined, so the child
  never restarted its worker: no periodic flush for the life of the process,
  spans accumulating until the buffer cap evicted them, and nothing shipped
  short of an explicit `close()`. The marker now records which process owns
  the spawn, so an inherited value reads as "a spawn in the parent" and the
  child respawns.
- **`close()` no longer leaves a worker thread nobody can join.** A socket
  finalizer landing inside the worker's thread allocation can reach `close()`
  on that same thread; the shutdown would then complete, and the outer frame
  would start a worker afterwards that no `join()` would ever see. A spawn
  that discovers a shutdown has begun is now abandoned instead of started.
- **Uninstalling the connection-close probe no longer risks disabling
  connection timing for the rest of the process.** If a socket closed on
  another thread at the moment the probe was being uninstalled, the teardown
  raised `RuntimeError: dictionary changed size during iteration`. The error
  was swallowed, but it left the probe marked installed on a process-wide
  object, so a later `init()` believed it was already set up and connection
  timing was silently never instrumented again. The teardown now walks a
  snapshot.
- **An MCP tool call could be reported twice.** The MCP tool catalogue walked
  its handle list live while the same list could be trimmed by its own bounded
  cap. A trim mid-walk skipped a handle, and a skipped handle is a server
  whose token is never resolved — which is exactly how one call ends up
  emitted twice. All three walks now iterate a snapshot.
- A `traceparent` set as a session default on an `aiohttp.ClientSession` was
  overwritten on every request through that session. A header you set yourself
  now wins whether you set it per request or once on the session.
- A caller-supplied `tracestate` was silently replaced by httpx and requests,
  and sent twice by aiohttp. It is now left exactly as written by all three.
- Two threads calling `init()` concurrently could stack the propagation
  patches twice, leaving the host permanently patched after `close()`. Install
  and uninstall are now serialized, and a `close()` that re-enters an install
  from a signal handler on the same thread neither deadlocks nor leaves
  patches behind.
- A `traceparent` from a future version with an empty field — a trailing
  hyphen, or a hole between fields — was accepted as well-formed. Such a
  header is truncated rather than futuristic and is now rejected, restarting
  the trace as the specification directs. Fields belonging to versions wardex
  does not implement are still not inspected.
- A non-latin-1 character in an inbound `tracestate` could raise
  `UnicodeEncodeError` out of the host's own outbound request when the value
  was forwarded. Header injection is fail-silent again in that case.

### Documentation

- The LangGraph retry attempt count is a documented limitation: every
  per-attempt signal langgraph exposes today is internal, corner-scoped, or
  process-global. The signal inventory and the re-open trigger live on the
  pinned test `test_the_attempt_count_is_not_recoverable_from_the_span`.
- Abandoned LangGraph streams keep their exception-derived `ERROR` status —
  the interpreter's own exception name, decided by who finalizes the
  generator. No dedicated abandonment marker is minted until field data shows
  a consumer needs one spelling.
- A LangGraph node is not an agent: no `HANDOFF` span is fabricated for
  `Command(goto=...)`. `wardex.langgraph.command_goto` and
  `wardex.step.trigger` are the final vocabulary, and graph-edge causality
  stays in extras and links between step spans.

## [0.4.0b1] - 2026-08-09

### Added

- `BackendConfig`, `RetentionPolicy`, `PIIPolicy`, `BatchingPolicy`, and
  `PropagationPolicy` are exported from `wardex_sdk` and passed to `init()` by
  name. The group names are shared across wardex SDKs, so a Node or Java
  service configured by the same team reads the same way.
- **`init()` without a `transport=` now builds the default OTLP/HTTP exporter
  from `backend=BackendConfig(endpoint=...)`.** An explicit `transport=` still
  wins — a `Transport` carries its own address — and under `debug` the losing
  endpoint is announced on stderr. With neither, `init()` installs
  `NoOpTransport` as before.
- **`max_otlp_attribute_bytes` (default 1 MiB)** — caps one OTLP attribute
  value as it appears **on the wire**, i.e. after binary payloads are rewritten
  to base64. A value over the bound is truncated and the span says so with an
  `otlp_attribute_truncated` marker in `wardex.limitations`. This is not a
  second `max_body_bytes`: that one caps what a parser keeps in raw bytes
  before encoding, this one caps what a value costs on a wire that measures it
  afterwards.
- **`max_otlp_request_bytes` (default 4 MiB)** — caps one OTLP/HTTP request,
  measured both as the compressed body that goes on the wire and as the message
  it decompresses to, because receivers check both. The default is gRPC's own
  receive ceiling, which the OTLP/gRPC receiver inherits and collector HTTP
  deployments commonly mirror. Raise it if your collector accepts more; a
  backend that accepts more will simply never see a split.
- **`OtlpHttpTransport(..., compress=False)`** — sends requests uncompressed,
  for a proxy or receiver that mishandles `Content-Encoding`. The header moves
  with the switch, so an uncompressed body is never declared as gzip.
- **`Transport.set_limits()`** — receives the resolved limits from `init()`,
  mirroring `set_pii_policy`. Relevant only if you have written a custom
  transport that encodes; one that overrides neither keeps working unchanged
  on the core defaults.
- `interceptors.mcp_stdio.stranded_requests` — a diagnostic counter for MCP
  stdio requests that were in flight when the server's stdout ended. A server
  that dies without answering (a crash, a `kill`, an argument it did not like)
  used to leave those requests latched forever; they are now released and
  counted. They are deliberately not shipped as error spans: what the seam
  observed is a pipe closing, not a call failing, and it cannot know whether
  the server answered on a channel wardex does not read.
- `wardex_sdk.interceptors` now imports without the native extension present,
  so shutdown paths that run in degraded mode no longer risk an `ImportError`
  on the way out.

### Changed

- **BREAKING:** twelve flat `WardexConfig` fields moved into five groups. There
  is no compatibility shim — the old spelling raises a `TypeError` naming its
  new home rather than being silently ignored:
  - `api_key=`, `endpoint=` → `backend=BackendConfig(api_key=..., endpoint=...)`
  - `default_retention=`, `retention_triggers=` → `retention=RetentionPolicy(default=..., triggers=...)`
  - `pii_mode=`, `pii_disabled_categories=` → `pii=PIIPolicy(mode=..., disabled_categories=...)`
  - `flush_interval=`, `flush_on_signals=` → `batching=BatchingPolicy(flush_interval=..., flush_on_signals=...)`
  - `propagate_trace=`, `propagate_targets=` → `propagation=PropagationPolicy(enabled=..., targets=...)`

  `intercept`, `intercept_hosts`, `interceptors`, `debug`, `before_send`,
  `capture_mode`, `release`, `environment`, `tags` and `adapters` stay
  top-level.
- Each config group validates its own fields, so an invalid policy fails on the
  line that constructed it instead of at `init()`.
- **OTLP/HTTP requests are now gzipped by default** (`Content-Encoding: gzip`),
  the encoding the OTLP/HTTP specification names. Payload-carrying spans are
  highly compressible — gzip returns the third that the base64 rewrite costs,
  and more. Compression happens in the Rust core, off the GIL. Use
  `compress=False` to opt out.
- **An export larger than `max_otlp_request_bytes` is now split across several
  POSTs** instead of being sent as one request a receiver rejects whole. The
  requests share **one** deadline: a batch that happened to split into several
  chunks cannot multiply the timeout you passed to `flush()` or `close()`.
- **The export timeout now covers encoding as well as the POST.** `timeout` was
  always documented as a wall-clock bound on the whole call, but the clock
  previously started after the encode — which, for a large batch that must be
  serialized, compressed, split and re-measured, made the real budget "the
  encode, plus the time you asked for". A tight budget fully spent by encoding
  now returns undelivered rather than overrunning.
- **`Content-Encoding` supplied through `headers=` is ignored** (with a
  debug-mode notice). It describes bytes only the transport knows how it
  produced, and letting it through shipped a real gzip frame declared as
  something else — a 400 no retry fixes. Use `compress=False` to send
  uncompressed.
- Span losses that were previously visible only in debug mode are now reported
  once per process on the default settings: a span too large to export even
  with its payload removed, a split export that ran out of budget partway, and
  a split export abandoned partway by a failed request.
- Per-connection seam state and connection-timing slots are now tied to the
  lifetime of their socket rather than to a size cap. The `max_connections` and
  connection-timing limits still apply, but they now bound *live* connections
  instead of accumulated dead ones, so a process that opens many short-lived
  connections no longer pushes its active ones out of the tables.

### Fixed

- **A new connection could inherit a dead one's capture verdict and never be
  captured.** Per-connection seam state was keyed by `id()` and destroyed only
  by a size cap, so it outlived its socket — and CPython hands the same address
  to the next object of that size. An HTTPS connection landing on the id of a
  retired Redis one inherited `ignore` and produced no span at all. Connection
  state is now released when the socket closes, so the address can never be
  reused while stale state is still attached to it.
- **Spurious `connect_timing_unavailable` on healthy connections.** Every
  non-TLS socket in the process (a Redis client, a Postgres pool, a health
  check) left an entry in the connection-timing table, and the size cap evicted
  the *oldest* entry — the live TLS connection still streaming a response —
  while long-dead sockets kept their slots. Timing is now released when its
  socket closes, and the cap is only a backstop.
- **Unbounded memory growth on long-lived HTTP/2 connections.** A stream that
  ended without a response (RST_STREAM, GOAWAY, a server that stops) left a
  correlation entry behind for the life of the connection, so a keep-alive h2
  connection to a model provider accumulated one entry per cancelled request
  for as long as the process ran. The table is now cleared at connection close
  and additionally bounded by the existing `max_streams` limit, which covers
  the pooled async-TLS path where no close signal is observable.
- **WebSocket spans lost at shutdown.** A WebSocket span exists only once its
  session ends, and it was previously emitted at `uninstall()` — so any process
  that exited without reaching one lost it. It is now emitted when the
  connection closes.
- **A batch inside every configured capture limit could still be rejected whole
  by an OTLP receiver, losing every span in it.** Because binary payloads are
  encoded as base64 on the OTLP surface, what left was up to a third larger
  than what was captured — a batch inside `max_buffer_bytes` (64 MiB) left as
  ~85 MiB, a body inside `max_body_bytes` (32 MiB) left as ~43 MiB. An OTLP
  request is accepted or rejected whole, so this was a total export failure at
  the size boundary, not a partial one. Oversized values are now capped and
  oversized batches split.
- **Large exports were rejected even when compression brought the body under
  the limit's face value.** Payload attributes are base64 text and gzip
  several-fold, so an export could compress under the cap, pass the only check
  there was, and still be refused by a receiver enforcing its limit on the
  decompressed message. Both sizes are now checked.
- Large exports are also meaningfully faster to encode: compression now runs
  once per body that will actually be sent rather than once per level of the
  split, and splitting picks its fan-out from a measurement already taken
  instead of repeatedly halving and re-encoding the whole batch.
- **Non-MCP subprocesses were instrumented for their whole life.** The stdio
  seam's "this is not an MCP server" check counted only bytes written *to* the
  subprocess and ran only on the write path, so a subprocess that writes little
  and streams a lot back — a compiler, a log follower, a media encoder — never
  detached, and paid a buffer copy and a JSON-RPC parse attempt on every read.
  The check now counts both directions and runs on both paths.
- **An MCP server observable only on stdout could stop being captured
  mid-session.** The same check treated "no JSON-RPC seen" as a write-side
  question, so valid messages parsed on the read side did not count as
  evidence and the seam detached once the sniff budget was spent. Messages are
  now counted in both directions.
- `WardexConfig.from_env()` no longer discards overrides. It previously
  honoured four field names and dropped everything else it was handed.
- `WardexConfig.from_env()` no longer loses `WARDEX_API_KEY` when only part of
  the backend group is overridden. Passing `backend=BackendConfig(endpoint=...)`
  used to skip both backend environment reads, sending every envelope with an
  empty project key. Environment values now resolve per field.
- A normal process exit now removes the trace-propagation patches. `close()`
  unpatched them but the at-exit path did not, so an exiting process could
  leave `httpx.Client.send` wrapped after the SDK had shut down.
- `from wardex_sdk.interceptors import SSLInterceptor` keeps working. The name
  is resolved on first access instead of at package import.

## [0.3.0b5] - 2026-08-09

### Added

- **A reusable adapter conformance suite** under `wardex_sdk.testing`
  (`conformance.py` + `harness.py`). It verifies the invariants every adapter
  must hold — causal parent edges, placement compliance, install/uninstall
  reversibility, unit closing — and detects full causal-tree collapse via
  per-node id chains. Both shipped adapters (Anthropic Agent SDK, LangGraph)
  now run on it.
- `config.interceptors` is honored by `init()` instead of being silently
  ignored. The default remains exactly the previous behavior; a name with no
  implementation behind it now fails loudly at config-validation time.

### Changed

- **LLM spans are named by what they did, not how they traveled.** A span
  with extracted LLM semantics now exports as
  `{gen_ai.operation.name} {gen_ai.request.model}` (e.g. `chat gpt-4.1-mini`;
  the operation alone when the model is absent), following the OpenTelemetry
  gen_ai semantic conventions, instead of `HTTP POST /v1/chat/completions`.
  Spans without LLM semantics keep the HTTP naming. Backend queries, filters
  or alerts that match on the old span names need updating. The wire format
  is unchanged.
- **Traffic wardex will not capture no longer pays for capture.** The byte
  seams now evaluate connection-invariant conditions — no client installed,
  the SDK's own exporter traffic, a connection the first-bytes sniff already
  ruled out — before any send/recv buffer is copied or accumulated and
  before any response body is decompressed or parsed. A TLS connection to a
  non-LLM backend (Redis, Postgres, an internal API) previously paid a
  buffer copy on every send for the life of the connection; it now pays a
  dictionary lookup. Behavior for captured traffic is unchanged.

### Removed

- **Breaking**: the `InterceptorName.GRPC`, `InterceptorName.WEBSOCKET` and
  `InterceptorName.SSE` enum members. They named no interceptor — selecting
  one installed nothing at all — because gRPC, WebSocket and SSE are
  protocols the byte seams parse (see `Protocol`), not interceptors of their
  own. Protocol capture is unchanged. Migration: delete any reference to
  these members; nothing replaces them and nothing is lost.

## [0.3.0b4] - 2026-08-06

### Fixed

- OTLP export no longer emits `bytes_value` attributes, which some backends
  (Arize Phoenix) silently rejected — dropping every span that carried raw
  request/response payloads (the LLM and tool spans) with an HTTP 200.
  `wardex.input_data` / `wardex.output_data` now ship as strings: readable
  UTF-8 text verbatim, binary payloads as base64 with a
  `<key>.encoding = "base64"` companion attribute. PII masking still runs on
  the raw bytes before the rewrite. The wardex envelope protocol is unchanged.

## [0.3.0b3] - 2026-08-06

### Added
- **A LangGraph adapter, auto-detected whenever `langgraph>=1.2` is importable.**
  A graph run now exports a tree instead of a scatter: one `invoke_workflow`
  span per run, one `execute_step` span per node, one `execute_tool` span per
  tool call a `ToolNode` dispatches, and every LLM or HTTP call underneath
  parented to the node or tool that made it. Before this, a graph whose nodes
  called a model produced **one trace per outbound call and no run, node or
  tool span at all** — each of those calls indistinguishable downstream from a
  genuine top-level request; and under the default `capture_mode=AGENT` a graph
  whose tools do pure Python was entirely invisible.

  The tree comes from **in-process context propagation, not from a framework
  identifier**. LangGraph copies the Python context when it submits a task, so
  a unit kept ambient over the run entry reaches every node body, every tool
  body and every request either of them issues. The adapter contains no call to
  `rejoin`, `attach`, `pin`, `open_run`, `claim` or `claim_run` — there is no
  place where a `run_id` can affect the shape of the tree, and a test asserts
  that over the module's own source. Every edge measures `unit_active`/1.0 or
  `contextvar`/1.0 with no integrity marker, on sync and async entries, thread
  and task fan-out, `Send` fan-out, subgraphs, and concurrent runs.

  Covered: `invoke`, `stream`, `ainvoke`, `astream`, `astream_events`,
  `astream_log`, `batch`, `abatch`, the functional API (`@entrypoint`/`@task`),
  subgraphs (which correctly earn both an outer `execute_step` and an inner
  `invoke_workflow`), and agents built with either
  `langgraph.prebuilt.create_react_agent` or `langchain.agents.create_agent`.
  Node retries land in ONE span rather than one per attempt, and so do
  `wrap_tool_call` retries.

- **An adapter can declare which of its framework's exceptions are CONTROL
  FLOW.** `AdapterInterface.CONTROL_FLOW` is a per-adapter classvar that
  `AdapterContext._run` consults on every causal path at the moment it
  classifies an exception. LangGraph implements human-in-the-loop by *raising*:
  before this, every `interrupt()` shipped `status=ERROR
  error_type=GraphInterrupt` on a run the host saw succeed — at every nesting
  level, so one suspension inside a tool produced three false failures. Those
  spans now read `UNSET` with no error type, and the host's exception is
  re-raised as the same object. `GraphBubbleUp` covers interrupts, drains and
  parent commands with one name. An ordinary node failure is unaffected and
  still reads `ERROR`/`RuntimeError`.

- **`Scope.record_failure(error_type)` — an adapter can report a failure the
  host did not raise.** Deriving a span's status from the exception that left
  the body is only half a rule, because a framework that converts a failure
  into a *return value* makes the other half unreachable. LangGraph's default
  `handle_tool_errors` does exactly that: the model calling a tool with
  arguments that do not validate becomes a `ToolMessage(status="error")` and
  never raises — the most common tool failure there is, and it shipped
  `status=OK`. Those spans now read `ERROR`/`tool_error`, while the node and
  the run above them stay `OK`, because the graph really did complete. The
  declaration is deliberately weaker than an exception and is consulted only
  when nothing was raised, so a tool that genuinely crashed keeps its own
  `RuntimeError` and an `interrupt()` stays `UNSET` — an adapter cannot
  relabel a crash or a pause.

- `ToolAttributes` is re-exported from `wardex_sdk.assembly`, which is how an
  adapter is allowed to name it.

### Fixed
- **A span context left behind by a unit that has already CLOSED is no longer
  accepted as a parent.** Every refusal predicate used to gate on
  `entry.pinned`, so only a leaked *pin* was refused — while an `activate()`
  fork that `close()` could not take down (a run closed from a different
  carrier, a generator finalized on a foreign thread, `close_units()`
  mid-stream) survived as a **confident parent at 1.0 with no marker** into a
  span that had already shipped. Measured: an entire later, unrelated graph run
  became a child of a finished one, its own children parented perfectly
  underneath, so the single wrong edge read like wardex's best evidence.

  Those sites now refuse the corpse and say so: a nested site becomes
  `unresolved`/0.0 with `PARENT_UNRESOLVED` + `CORRELATION_CONFLICT`, a run
  entry gets its own trace carrying `CORRELATION_CONFLICT`, and under
  `capture_mode=AGENT` the gate closes rather than opening on a dead parent.
  Two counters name which happened —
  `assembly._units.stale_activation_ambient` (a leftover activation was
  refused, split from the pin's counter because they are two different bugs
  with two different repairs) and `assembly._units.ambient_closed_at_issue` (an
  observing seam latched an already-shipped span).

  The liveness question is asked **when the work is issued**, not when the span
  is emitted, and that is the correctness argument rather than an optimisation:
  a request issued inside a live run whose unit closes before the reply arrives
  is a perfectly good child of that run, and asking on the response side would
  invent a spurious trace for every streaming call that outlives its run.

  A clean run is unaffected — no new counter fires, no edge changes, and the
  no-false-positive case is asserted over every counter in this area.

  **The two hand-written entry points ask the same question.** `wardex.span()`
  and `capture_state_snapshot()` latch a parent they did not open and cannot
  vet, which is exactly what makes them observing sites — so a manual span
  opened after a run's unit had closed used to become a full-confidence child
  of an already-emitted span, in that span's trace. The byte seam beside it got
  this right and the published API did not. Both now refuse the corpse and land
  on `unresolved`, each through the route its own no-parent case already
  documents. A LIVE parent is untouched, which is the ordinary case and is
  asserted as its own control.

- **A `describe` that fails at a RUN ENTRY no longer reports a run-sized loss
  as "one span".** The branch carrying the escalated message was tested after a
  flag meaning *"the unit was closed"* — which is true on exactly the path that
  reaches it — so it was unreachable, and every run entry reported the same
  single-span wording. Measured: at a `ROOT` site the description dies before
  the intent's required block, `finish()` refuses the draft, and **zero spans
  ship for the whole run**. In a production process `counters` is not exported
  and `debug` is off, so that one stderr line is the entire difference between
  "wardex deleted a run" and "wardex was never installed".

  The message is also corrected rather than merely made reachable: it claimed
  traffic inside the run would not be captured either, which was true before
  `degraded_run()` and is not now — that work is still captured, it arrives
  orphaned and marked instead of attached. An unreachable branch is an
  unaudited one.

### Changed
- `wardex.init()` now imports `langgraph.pregel` eagerly for anyone who merely
  has langgraph in the environment — **measured at 310-319 ms** plus 22-24 ms
  for `langgraph.prebuilt.tool_node`, loading 121 `langgraph` and 79
  `langchain_core` modules. That is materially heavier than any other adapter's
  detection, and it is recorded here rather than left to be discovered as a
  startup regression. Opt out with `Config.adapters`.

### Known limitations
- **Under `intercept=False` a LangGraph tree has no LLM leaves.** Run, node and
  tool spans are correct and every `chat` span is absent, because this adapter's
  seams see a node task and never a model call — the LLM spans come from the
  byte seam. Nothing in this release restores them.
- A tool called directly from a hand-written node body produces no tool span:
  the seam observes tool calls the *framework dispatched*, not every LangChain
  tool that ran in the process.
- A bare `threading.Thread` started inside a node or tool body loses the
  context, and with it the parent. This is below the adapter and permanent.
- `RemoteGraph` (LangGraph Platform) implements `PregelProtocol` without
  subclassing `Pregel`, so it is silently uninstrumented.
- A cache hit produces no node span — the seam is never entered, which is
  honest, but a run's tree can legitimately omit nodes.
- `interrupt_before` produces a run span with zero node spans, and nothing on
  the wire distinguishes "paused before its first node" from "ran nothing".
- Breaking early out of `app.stream(...)` reports the run as
  `ERROR`/`GeneratorExit`; abandoning `astream_events` reports
  `CancelledError` on the node span too. This is deliberate rather than
  overlooked — an abandoned run genuinely did not complete, and suppressing it
  would make it indistinguishable from a finished one — but it is not a
  *typed* outcome yet.
- **A node's retry attempts are collapsed into one span, and the attempt count
  is not recoverable from it.** This is the right trade — a node that failed
  twice and then succeeded is one node that worked, and the callback-based
  products render it as three sibling runs with two false errors — but the
  count is genuinely lost, because the retry loop lives inside the seam and
  reaching it would mean substituting an object the framework owns.
- An `interrupt()` inside a subgraph is seen at two levels (the inner node and
  the subgraph-as-node task), so one pause produces two `UNSET` spans rather
  than one.
- A graph the user did not name is called `LangGraph`, which is LangGraph's own
  default — so a parent and its subgraph can both ship `invoke_workflow
  LangGraph`. Name your graphs if you nest them.

## [0.3.0b2] - 2026-08-05

### Changed
- **`Transport.export` takes a keyword-only `timeout`, and what it returns now
  decides what happens to the batch.** The signature is
  `export(self, envelope, *, timeout: float | None = None) -> object | None`,
  and `Transport` is exported from the package root, so every third-party
  implementation is affected. An existing `export(self, envelope)` keeps
  working — the client reads each transport's signature, per transport instance
  and again whenever the transport is swapped at runtime, and withholds the
  keyword from one that cannot take it, because calling such a transport with
  `timeout=` raises `TypeError` inside the drain's fail-closed handler and
  would trade a stall for total data loss. It keeps *stalling*, though: the
  client can bound how long it waits, only the transport can bound its own I/O,
  so a deadline that stops at the drain leaves the process blocked inside the
  transport for its full configured timeout. To honour it, take the keyword and
  NARROW with it, never widen —
  `effective = self._timeout if timeout is None else min(self._timeout, timeout)`
  — because a caller asking for 99s must not get more than the transport was
  configured for. The value is `None` when the caller imposed no deadline, and
  otherwise a non-negative number that may be `0.0`.

  The return value is the other half, and it is an OBSERVATION the client acts
  on rather than something it can work out from outside. Return `UNDELIVERED`
  (`from wardex_sdk.transport import UNDELIVERED`) to say: this envelope was
  not put on the wire, nothing about it was consumed, and an identical attempt
  later with a fresh budget could succeed. A `flush()` then keeps the spans and
  re-sends them; a `close()` counts them and reports them, and so does a
  `flush()` on a client already closed, where there is no next drain either.
  Anything else — most of all `None`, which is what every transport written
  before this returns —
  means "taken", and that default is the safe direction rather than a shrug:
  delivering unconditionally is a legal transport, and a client inferring "not
  delivered" from the outside announces losses that never happened. Do NOT
  return it for a POST that failed partway or a batch that could not be encoded
  at all. Those are attempts of unknown outcome, so handing them back would
  duplicate a batch the backend may already hold, or pin an unencodable one in
  the buffer forever. `ConsoleTransport`, `NoOpTransport` and
  `OtlpHttpTransport` are updated; only the last has bounded I/O to narrow or
  any reason to decline.
- **`wardex.flush(timeout)` is a wall-clock bound on the whole drain, not on
  one step of it — but `wardex.close(timeout)` bounds each of its three
  shutdown steps separately, so the worst case there is roughly 3x.** Take
  `close(5.0)` as "no step waits longer than 5 seconds", not as "returns within
  5 seconds"; the reasoning, and why the steps deliberately do not share one
  deadline, is at the end of the `close()` entry below. The drain held its lock
  across `before_send`, `transport.export` and both `transport.flush` calls, so a
  flush arriving while another drain was inside a synchronous POST waited out
  that POST in full before starting its own, and the argument bounded only its
  own half. The signal handler's `flush(2.0)` — the one whose comment says
  never delay shutdown — measured 10 + 10 + 2 on a stalled backend, so SIGTERM
  hung for twenty-two seconds. One monotonic deadline now covers the wait for
  the export slot, the POST, and the transport flush after it. Exports are
  still serialized, because that is the guarantee third-party transports were
  written against; only *waiting* for one is bounded.

  A drain that cannot get the export slot inside its budget declines and
  returns having taken nothing, so `flush(0.0)` ships nothing where it used to
  block until the slot came free — the spans are untouched in the buffer and
  the next drain ships them. And the deadline now reaches the transport, which
  is where the next paragraph lives.

  **`flush()` and `close()` no longer share a default, and the difference is
  deliberate.** `wardex.flush()` with no argument follows the TRANSPORT's own
  configured timeout: a bare `flush()` means "send what you have, I will wait",
  so it must not cap the POST below a number the host already chose for exactly
  this. An `OtlpHttpTransport(timeout=10.0)` gets its 10 seconds, and a backend
  that reliably answers in 7 is delivered to. An explicit `flush(t)` is still a
  real wall-clock bound on the whole drain whatever the transport was
  configured for, and passing `5.0` by hand is taken at its word: naming a
  number is the whole difference between the two readings. `wardex.close()` keeps the tight 5s default and
  does NOT follow the transport — it runs when the process is going away, which
  is the stall this whole issue started from, and what it cannot ship it
  reports. **`Transport.timeout` is a declared attribute on the base class** —
  `timeout: float = 5.0`, overridable as a plain attribute or as a property —
  which is how a transport says "wait for me this long". Declared rather than
  merely read, so a third-party transport can SEE that the client reads it
  instead of discovering by accident that keeping a `self.timeout` for its own
  bookkeeping changed how long a bare `flush()` waits, or that not having one
  cost it the timeout it was built for. Overriding it stays optional: a
  transport that says nothing inherits the 5s default, and one whose `timeout`
  raises or is not a usable number gets the same 5s, because a declaration is
  not a guarantee when the base class is public and subclassable and that read
  can never raise into the host. Unattended export is unaffected:
  the periodic worker passes no deadline at all, so background batches still
  get the transport's full configured timeout — clamping the one path that
  ships data with nobody watching would turn slow-but-working exports into lost
  ones.

  **What to change: if you relied on a bare `wardex.flush()` returning inside 5
  seconds, pass the number — `wardex.flush(5.0)`.** With no argument it now
  waits as long as the transport was configured to spend, which under the
  default `OtlpHttpTransport(timeout=10.0)` is twice as long, and under a
  transport configured for 60 is a minute. Only the no-argument call moved:
  every call that already named a number behaves exactly as it did, and
  `wardex.close()` is untouched. The signature default is a float carrying 5.0
  whose `repr` is `<the transport's own timeout>`, so `help(wardex.flush)`
  names the behaviour rather than a number that is no longer the whole truth,
  a host that reads the default off the function and passes it straight back
  gets the same reading, and anything that only ever sees a number still sees
  5.0. That reading is matched by TYPE, not by value and not by object
  identity, so it also survives the default being copied, deepcopied or
  pickled on the way — a settings object that carries it through a `deepcopy`
  hands back something that still means "follow the transport", and both public
  defaults survive those three operations rather than raising `TypeError` out
  of whatever host code performed them.
- **An export the caller's own budget cut short now says that delivery could
  not be CONFIRMED.** `flush(1.0)` against a transport configured for 10s can
  leave a POST in flight when the budget expires. The spans are not re-sent —
  the request was already on the wire, so the backend may hold them and a
  resend would duplicate them — and off-debug that was silence byte-identical
  to a successful export. `OtlpHttpTransport` now writes one line per process
  naming the budget, the transport's configured timeout and the span count. It
  is scoped to exactly that event: an ordinary refusal, a reset, an HTTP error
  and a timeout at the transport's OWN configured limit keep their existing
  fail-silent handling, because reporting those would spend the one line on
  "your backend is down" and silence the real one later. The wording is
  deliberate — the outcome is unknown, not lost, and the fix (a larger timeout)
  belongs to whoever chose the budget.

  A budget nobody chose is never reported. A bare `flush()` follows the
  transport's own timeout, a bare `close()` spends wardex's own 5s default, and
  the signal handler installed by `flush_on_signals` spends its own 2s bound;
  all of those arrive at the transport a shade under the configured number once
  the acquire and the encode are paid for — so "shorter than configured" cannot
  be read as "the caller chose it", and is not. The signal handler matters most
  of the three: it is on by default, 2s is under any transport configured for
  more, and there is no knob for it, so "pass a larger timeout" would have been
  advice about a number no host can pass. Third-party transports can make the
  same distinction: the client passes a `wardex_sdk.transport.CallerBudget` (a
  `float` subclass, so a transport that has never heard of it sees exactly the
  number it always did) when and only when the application named the number.
  A stalled backend at SIGTERM stays silent on this channel by design — the
  failed POST is still logged under `debug`, by the transport that saw it.
- **`close(timeout)` now abandons a tail it cannot ship inside its budget, and
  says so on stderr whether or not `debug` is set.** Bounding the drain bounded
  `close()` too, and a declined drain is free everywhere except the last one:
  `close()` sets `_closed`, stops the worker and then closes the transport, so
  spans its final drain declined to take are unreachable forever — with the
  process still running, because `init()` closes the previous client on every
  re-init and `wardex.close(timeout)` is public API. Measured: with the worker
  inside a 30s POST, `close(0.5)` returned in 1.0s having shipped nothing but
  the in-flight envelope and left two buffered spans resident, never to ship;
  unbounded they shipped, after 30s. `close()` keeps its bound — a shutdown
  that cannot be bounded is the failure this started from — but its final drain
  now empties the buffer, counts what it could not ship, and writes one line:

      [wardex] could not ship 2 buffered span(s): <why> They are out of the
      buffer and nothing will retry them. Give wardex.close(timeout=...) a
      larger budget to keep them.

  `<why>` names which exit was taken: an export was already in flight and did
  not finish inside the budget, so the final drain never ran; or the transport
  was handed what was left of the budget and reported back that it did not
  send. A third reason, with its own closing advice, belongs to a drain that
  comes back to a client `close()` already emptied — see *Fixed*, which is also
  why the line no longer opens with the word `close()`. The line is
  deliberately NOT gated on `config.debug`, which defaults to False — a report
  that prints only in the configuration nobody
  runs is worse than none at all here, because the spans are no longer resident
  in the buffer where an operator could at least find them. It is bounded to
  one line per site per process, the same idiom used everywhere else wardex
  will not ship what you expected. The count is kept apart from the buffer-full
  drop counter that `_drain` prints as `dropped N spans (buffer full)`:
  `flush()` does not check `_closed`, so a flush after `close()` was enough to
  have a shutdown loss described as an overflow. Note that `close(timeout)`
  remains a per-STEP budget — the worker join, the final drain and the
  transport close can each spend it, so the worst case is roughly 3x. Steps 2
  and 3 wait on the same in-flight POST, and charging the final drain for what
  the join already spent would abandon tails `close()` can still deliver.
- **An unusable `timeout` is coerced rather than raised on.** `flush()` and
  `close()` take that argument straight from application code, and an
  observability SDK may not raise back into that code, not even on nonsense.
  NaN and anything that is not a number fall back to the 5.0 default; a
  negative value means "do not wait" and floors at 0.0; everything else clamps
  to `threading.TIMEOUT_MAX`. An unusable value is *ignored* rather than
  rejected because rejecting means raising, and refusing a shutdown flush over
  a bad argument loses more than flushing it on the default does. NaN has to be
  tested for by name, since no clamp normalizes it — `max(nan, 0.0)` and
  `min(nan, TIMEOUT_MAX)` are both NaN — and without that, `flush(float("nan"))`
  reaches `RLock.acquire(timeout=nan)` as a `ValueError` and `flush("x")`
  reaches the deadline arithmetic as a `TypeError`, both outside every handler
  in the drain. What a third-party transport is handed is likewise always a
  non-negative float, never a negative remaining budget.

### Fixed
- **A signal arriving inside wardex's own one-line-per-process reporter could
  deadlock the interrupted thread forever.** `report_once` held a plain
  `threading.Lock` across "have I said this already" and "record that I have",
  and the `SIGINT`/`SIGTERM` handler this SDK installs runs `flush()` **on the
  interrupted thread**. A signal delivered inside any of those critical
  sections — one of them is on the span-capture path — reached a handler that
  called `report_once` again and blocked on a lock its own thread was already
  holding. Not a slow shutdown: a permanent stop, in the handler, with the lock
  still held, so every later report in the process would have hung behind it
  too. The lock is an `RLock` now, and the "was I first" claim is made by a
  single `dict.setdefault` rather than by reading membership and then writing —
  a signal landing between those two steps made both callers print, which is
  the bound this function exists to provide. This shipped in 0.3.0b1; reaching
  it needed a signal delivered inside a window of a few instructions, which is
  why nobody saw it.
- **A wheel whose native extension will not load no longer stops the host from
  booting.** `wardex_sdk/__init__.py` opened with a bare
  `from . import _wardex_native`, and two more modules on that same import path
  did the same, so an extension that would not load did not disable wardex — it
  raised out of the host's own `import wardex_sdk` and the application never
  started. A wheel built for the wrong ABI, a `.so` a container build stripped
  out of the image, an sdist installed on a machine with no toolchain: any of
  them turned an observability dependency into a process that will not boot. An
  observability SDK may never be the reason a host fails to start. Both failure
  shapes are covered, because they raise different classes: a `.so` that was
  never installed raises `ModuleNotFoundError`, one that is present and
  unloadable raises the base `ImportError` out of `dlopen`.

  The extension is imported once now and every module reads that one answer.
  Three places answer differently when it is False. `init()` writes one line to
  stderr — `[wardex] native extension unavailable, wardex is disabled: nothing
  will be captured or exported (...)` — and returns before a client,
  interceptor, adapter, propagation patch or atexit hook exists, which is what
  makes this complete rather than partial: every remaining module that touches
  the core is then unreachable by construction. `close()` returns before it
  would import `interceptors/`, which still reaches the extension at import
  time and would otherwise kill the process on the way OUT of a `finally` or an
  atexit hook instead of on the way in. And `CaptureLimits.resolved()` /
  `.to_native()` raise a `RuntimeError` naming the wheel instead of an
  `AttributeError` on `None` — the core owns the limit table, so a Python-side
  fallback would be a second declaration site that drifts. The config is still
  built first, so a caller's bad keyword raises the same `TypeError` it always
  did; degraded mode must not turn a programming error into a shrug.

  Where the line falls is now written down instead of inferred. With the
  extension unimportable, `import wardex_sdk` and every symbol on its `__all__`
  work, and `wardex_sdk.transport`, `.context`, `.assembly`, `.adapters` and
  `.pipeline` import; `wardex_sdk.protocol`, `.semantics`, `.interceptors`,
  `transport._codec` and three `adapters/` modules still raise `ImportError`.
  That stays a boundary rather than a gap: each of those is reachable only
  through `init()`, which returns before importing any of them, so degrading
  them would change nothing a host can observe. The stderr line is not
  optional — a wardex that silently captures nothing looks exactly like a
  backend that is up and receiving no traffic, so nobody goes looking, and that
  is the worse of the two failures rather than the milder one.
- **`OtlpHttpTransport` no longer discards every batch in silence when the
  extension is missing.** It is a published symbol, so a host can construct it
  and call `export()` by hand without ever reaching `init()` and the one line
  printed there. Encoding is impossible either way, so those spans were lost
  with nothing said. It now reports once per process, naming the underlying
  import error, and it reports that BEFORE the debug-gated "deadline exhausted"
  skip which used to be checked first: a degraded process whose budget had also
  run out got the gated diagnosis and never the actionable one. The report is
  bounded to one line rather than written per export, which is what makes an
  unconditional message affordable on a path a host may drive in a loop. It is
  deliberately not an `UNDELIVERED` decline — that word promises a later
  attempt could succeed, and no attempt in this process ever can, so saying it
  would hand the same unencodable batch back to the buffer forever.
- **One interceptor's `install()` no longer takes `wardex.init(intercept=True)`
  down, and a rolled-back install now really removes its patches.**
  `InterceptorRegistry.uninstall_all` has always been total and `install` was
  not, so an interceptor raising out of `install()` propagated straight out of
  `wardex.init()`: a host that added observability got a crash at startup from
  the one component whose whole promise is never to alter the application.
  These seams patch the stdlib and third-party internals — `ssl.SSLSocket.recv`,
  `anyio._backends._asyncio.AsyncIOBackend.open_process` — so a version bump in
  a package the user never chose is an ordinary way for that to happen.

  Containing the raise is only the first of three. The interceptor is recorded
  BEFORE `install()` is called, because an install that raises halfway has
  already patched part of a surface, and one the registry never recorded is one
  nothing can ever reach: those wrappers stayed in front of the host's sockets
  for the life of the process. And the undo has to undo — all three
  interceptors this SDK ships set `self._installed = True` as the LAST
  statement of `install()` while opening `uninstall()` with
  `if not self._installed: return`, so the rollback called an undo that
  declined every time. `uninstall()` is total now: `PatchSet.restore_all()` is
  idempotent and empty before the first patch, `_conns` is empty until a byte
  flows, and the refcounted shared connection-timing probe is tracked by its
  own flag so a rolled-back seam cannot release a reference it never took and
  rip `socket.connect` out from under the seam that is still live. The registry
  also drops the name BEFORE rolling back, so a `KeyboardInterrupt` out of
  `uninstall()` — re-raised by design, because it is the host's control flow —
  can no longer leave behind a name every later `install()` skips. A seam that
  fails is one line on stderr under `init(debug=True)` rather than a counter
  with no reader.
- **anyio is no longer a hard requirement of `init(intercept=True)`.**
  `_mcp_stdio` imported `anyio._backends._asyncio` at module top level — a
  third-party PRIVATE API, from a wheel that declares no runtime dependencies —
  and `init(intercept=True)` imports that module, so on a machine without anyio
  the `ImportError` arrived before any registry could guard it. The import
  degrades to `None` now and only the anyio seam declines, saying so once on
  stderr and naming what still works: `[wardex] mcp_stdio interceptor: anyio is
  not importable, so MCP traffic over anyio subprocesses will not be captured;
  the raw asyncio.create_subprocess_exec path is still intercepted`. An
  environment without anyio therefore loses only the transport it does not
  have, and a reader is not left thinking MCP capture is off altogether. The
  decline is an explicit branch rather than an `AttributeError` swallowed by a
  guard, because reporting an absent optional package as an internal failure in
  a counter nobody reads is the same as saying nothing.
- **A third-party `Transport` whose `close()` raises no longer raises out of
  `wardex.close()`.** `Transport` is public, so `close` can be a socket
  teardown that throws, a property, or a `__getattr__` — and it was the last
  reach into a caller-supplied transport still made outside a handler. Hosts
  call `wardex.close()` from `atexit` hooks and `finally` blocks, so that turned
  someone else's teardown bug into a raise out of the host's exit path. It is
  fail-silent now, with the same debug line `transport.flush()` already had.
  `KeyboardInterrupt` and `CancelledError` still propagate: a host tearing the
  process down must not be swallowed by an observability SDK's cleanup.
- **A drain that comes back to an already-closed client reports its batch
  instead of re-seeding a buffer nobody will drain again.** A non-final drain
  hands a declined batch back to the buffer, which is right on a live client
  and wrong after `close()`: no drain will ever run again, so the spans sat
  resident, uncounted and unreported, with `_spans` still listing them as
  pending — the exact state the abandoned-tail report exists to prevent. Two
  routes reach it and both are closed. `flush()` deliberately does not test
  `_closed`, so a `flush()` after `close()` gets there in a single thread; and
  a drain still in flight when `close()` runs — from another thread, or from
  host code that closes wardex inside `before_send` — hands its batch back to
  the buffer `close()` has just emptied. The decision is made inside the buffer
  lock, the same one `close()` empties the buffer under, because a check
  outside it would sit in the window between `_closed` being set and the buffer
  being emptied. Those
  spans are counted and reported like any other tail a closed client could not
  ship, on the same one-line-per-process channel and off `debug`, and the line
  carries the advice that fits this exit: *Flush before closing, or give
  wardex.close(timeout=...) a larger budget.*

### Added
- `UNDELIVERED`, exported from `wardex_sdk.transport`. It is the sentinel a
  `Transport` returns to say it did not send a batch; the contract it carries
  is in the `Transport` entry above. Published there and not at the package
  root on purpose: it is part of the `Transport` contract, so a third-party
  transport that wants to report a decline must be able to import it by a
  public name, but it is inert — a sentinel with nothing to call and no reach
  into the core — so it does not belong on the surface everyone else reads.
- `CallerBudget`, exported from `wardex_sdk.transport` alongside `UNDELIVERED`
  and published for the same reason: it is part of the `Transport` contract, so
  a third-party transport that wants to diagnose a cut-off export must be able
  to import it by a public name. It is the `timeout` a `Transport` receives
  when — and only when — the APPLICATION named the number, and it carries that
  number as `.requested` so a report can quote what the caller would recognize
  rather than the remainder left after the acquire and the encode. A plain
  `float` means the budget is wardex's own (a bare `flush()` following the
  transport's timeout, the shutdown default, the signal handler's short bound)
  and must not be reported as anyone's fault. `CallerBudget` subclasses `float`,
  so a transport that has never heard of it sees exactly the number it always
  did, and the silent reading is the one a forgotten `isinstance` falls into.
- `Transport.timeout`, a declared attribute on the base class carrying the
  per-export timeout a transport wants — `timeout: float = 5.0`, overridable
  as a plain instance attribute or as a property. It is how a bare
  `wardex.flush()` learns how long the host is willing to wait. Declared rather
  than merely read: the client reads this attribute, and a third-party
  transport needs to be able to SEE that, instead of discovering by accident
  that keeping a `self.timeout` for its own bookkeeping changed how long a bare
  `flush()` waits, or that not having one cost it the timeout it was built for.
  Overriding stays optional — a transport that says nothing inherits the 5s
  default, and one whose `timeout` raises or is not a usable number gets the
  same 5s, because the read stays defensive and can never raise into the host.
  `OtlpHttpTransport` overrides it with a read-only property carrying its
  constructor argument; `ConsoleTransport` and `NoOpTransport` leave the
  default, neither performing bounded I/O worth waiting on.
- The counter `interceptors.mcp_stdio.anyio_unavailable`
  (`wardex_sdk.assembly.counters.snapshot()`), bumped once at import time when
  the optional anyio backend cannot be imported.

## [0.3.0b1] - 2026-08-03

### Changed
- **The Agent SDK adapter's in-process tool wrapper no longer decides its own
  parentage.** It walked a three-tier ladder by hand — the live scope, then the
  one live session, then nothing — and stamped the confidence and the markers at
  the end of it, which is the shape every future adapter would have copied. It
  opens through `AdapterContext.enter()` now, so the edge is decided in the one
  place that decides edges, and the tier markers come from the table that owns
  them: `adapters/` holds no site that can name `PARENT_UNRESOLVED` or
  `UNIT_INFERRED_SOLE` on an edge it chose itself. The tree it produces is
  unchanged, asserted row by row.
- **A site may now DECLARE the one guess it is allowed to make**, with
  `fallback=Fallback.SOLE_LIVE_RUN`. An in-process tool handler is reached
  through a carrier the framework may not have propagated to, and orphaning
  there turns one run into several traces — the failure `Placement` exists to
  prevent, facing the other way. Declared rather than computed, for the same
  reason placement is: a heuristic that turns itself on is one nobody can find
  later. It is reachable only where the site would otherwise become a trace
  root, so a live scope or a host span always wins; the candidate comes from the
  registry filtered to that adapter's own runs, and only when there is exactly
  one; and the edge it produces says `unit_sole` at 0.5 with
  `UNIT_INFERRED_SOLE`. A guess made while a dead pin was standing also carries
  `correlation_conflict`, because "nothing was pinned" and "what was pinned had
  died" are not the same fact.
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
- **Nothing that can raise is left outside a failure boundary in the tool
  wrapper.** Two places in a `with ctx.enter(...)` statement are not contained
  and neither is obvious. Every argument in the HEADER is evaluated before the
  block is entered, so a framework attribute read there breaks the host exactly
  as one in the body would and no guard wardex can add will ever see it; the
  BODY is unguarded on purpose, because a guard there would swallow the host's
  own exception and report a failing tool call as a successful one.

  The one header that existed built a selector out of a framework read, an
  f-string and a counter. It is gone: a site that names no selector now gets a
  unique one the context mints for itself — which is also a better key, since
  the shared one it replaces had every anonymous unit in the process rebinding
  a single alias slot. And `_tool_input`, which runs in the body on the HOST's
  own return value, is total rather than catching what `json.dumps` documents:
  a container whose `items()` raises is not a `TypeError`, and the host was
  losing its result over a span attribute nobody would have missed.

  Both are now shape rules rather than care taken. `test_import_graph.py`
  refuses a header that is anything but names, literals, enum members and a
  `partial` of a name, and a body that is anything but the host's own call and
  verbs on the scope — because neither is a property any other check can see: a
  call-graph rule cannot see an attribute read, and there is no runtime moment
  at which "this expression was in a header" is observable.
- **A run wardex failed to open no longer silences everything inside it.** Under
  `capture_mode=AGENT` — the default — traffic is captured when a local wardex
  span was ambient at the moment the work was issued. A run entry that could not
  be opened leaves nothing ambient, so every HTTP request and every tool call in
  the host's block was dropped at the byte seam with no counter and no marker:
  one bug at the top turned into total silence underneath, and the run read as
  one that never happened rather than one wardex could not follow. Measured
  end-to-end through `wardex.span()` on the default mode: captured before the
  block failed, dropped after.

  The gate now takes "the missing parent is wardex's own doing" as a declared
  input, set on the task for the duration of the block the host was given
  anyway. What comes through is not passed off as ordinary: such a span resolves
  as `unresolved` at confidence 0.0 carrying `parent_unresolved` and
  `instrumentation_degraded`, never as a trace root — shipped as a root it would
  be one run arriving as several, indistinguishable from genuine ones. An edge
  wardex really did read is untouched, because the flag describes an ABSENT
  parent and never a present one.
- **A span built from an interpreted edge now carries the markers that edge
  earned.** `SpanDraft` is built FROM a parentage but did not inherit its
  markers, so a span could ship a confidence below 1.0 with an empty limitation
  list — half of I4 missing, and the half a dashboard renders. Two of the six
  parentage sites copied them across by hand and the rest did not; the draft's
  own constructor does it now, so there is no site left that can forget.
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
- **One span wardex cannot build no longer costs a whole subtree.** Closing a
  subtree interleaved two different kinds of work: building spans, which reads
  drafts and buffers and can fail, and unlinking units, which is dict and list
  operations on wardex's own tables. So a fault partway through left the parent
  already marked dead, some children already unlinked, every draft collected so
  far dropped on the floor with the raise, and the rest of the subtree reachable
  from no root of any registry. The two are separate phases now — collect every
  span first, with its own failure boundary per span, then unlink, which cannot
  fail — and the loss is what it should always have been: the one span whose
  draft is broken.

  Measured on three roots of four children with one broken draft, across the
  four shapes this has had: one boundary around the whole sweep ships 0 of 15,
  one per root ships 10 and wedges the failing root in the table forever, one
  per root plus evicting it ships 10 and leaves 3 units unreachable, and
  collect-then-unlink ships **14 of 15 with none unreachable**.

  The same measurement found one more: the shutdown sweep stamped its marker on
  a root's draft inside the close's own boundary, so a draft that could not take
  the marker skipped the close entirely and the eviction dropped the root
  without walking under it — twelve children left reachable from nothing. The
  marker is its own step now, and a marker that cannot be recorded costs the
  marker.
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
- **The SDK never fully uninstalled, and uninstalling destroyed other
  libraries' patches.** Six hand-rolled patch mechanisms each restored with an
  unconditional `setattr`, which is two bugs and both were live.
  `socket.socket.send/sendall/recv/recv_into` are INHERITED from the base
  socket type, so restoring them by assignment put nothing back — it welded a
  copy of the base's C method into `socket.socket.__dict__` permanently, where
  nothing had one before. No test caught it because the frozen copy IS the
  right object: every identity check passed while the host's process stayed
  mutated after `wardex.close()`. Restore now uses `delattr` where the target
  had no own attribute. And restoring without checking what is actually
  installed overwrote whatever patched the same attribute AFTER wardex did —
  on `httpx.Client.send`, `requests.Session.send` and
  `aiohttp.ClientSession._request`, which is exactly what OpenTelemetry's HTTPX
  instrumentor patches, on a path that runs on every re-init and every
  `close()`. Restore is identity-checked now: an attribute that is no longer
  wardex's wrapper is left alone, counted, and reported as `patch_superseded`
  — until now a declared marker with no emitter.

  `wardex_sdk.assembly.PatchSet` is the one mechanism the six become. Class,
  module and instance targets; LIFO restore; each restore individually guarded
  so one failure cannot abandon the rest; weakly held instances; an `RLock`,
  because teardown can re-enter through the signal handler this SDK installs.
  It cannot raise into the host: `patch()` returns `False` rather than
  propagating on a target that refuses `setattr`, and it refuses a data
  descriptor outright rather than writing through a host's property setter and
  then blaming a third party for the corruption at restore.
- **Installing an adapter is now as total as uninstalling one.** Four
  lifecycle failures of one kind — the loop that sets things up and the loop
  that tears them down did not agree about a misbehaving component. `install()`
  was unguarded, so an adapter raising out of it took `wardex.init()` with it:
  a crash at startup from the one component whose entire promise is never to
  alter the application. An adapter was filed only on SUCCESS, so an
  `install()` that raised halfway had already patched part of a framework's
  surface while the registry held no record — those wrappers stayed in the
  host's classes for the life of the process with nothing able to remove them;
  it is recorded first now and a failed install is rolled back immediately.
  Teardown was FIFO, so two adapters patching one attribute ended with the
  later undo restoring the EARLIER adapter's wrapper as though it were the
  host's original; newest-first now, matching the order `PatchSet` restores in.
  And a `BaseException` cut the sweep short: `guard()` re-raises
  `KeyboardInterrupt` and `CancelledError` deliberately, but re-raising in
  place abandoned every adapter behind the interrupted one — patches left in
  the host's classes and open spans never emitted, from `atexit`, where nothing
  reports why. It is captured, the sweep finishes, and it is re-raised
  afterwards. `InterceptorRegistry.uninstall_all()` and its adapter twin also
  iterated unguarded, so one raising `uninstall()` skipped `client.close()` and
  lost every buffered span.

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
- `bind_context()` helper for propagating trace context into threads.

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
