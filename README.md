# wardex-sdk

[![PyPI](https://img.shields.io/pypi/v/wardex-sdk)](https://pypi.org/project/wardex-sdk/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/wardex-labs/wardex-sdk/blob/main/LICENSE)

Open-source observability SDK for AI agents — zero-instrumentation capture,
OpenTelemetry-native.

> ⚠️ **Beta.** PII masking is on by default (see below), but the
> SDK is still early: review the caveats below before sending sensitive data
> through it. Versioning and stability promises are in
> [VERSIONING.md](VERSIONING.md).

## Install

```bash
pip install wardex-sdk
```

## Quickstart

```python
import wardex_sdk as wardex
from wardex_sdk import BackendConfig

wardex.init(backend=BackendConfig(endpoint="https://<collector>/v1/traces", api_key="..."))

# your app code — LLM calls, tools and agent runs are captured automatically

wardex.close()  # optional — spans auto-flush every 5s, on buffer threshold, and at exit
```

Interception is **on by default**: `init()` is the consent, and
zero-instrumentation capture of LLM traffic is the product (`intercept=False`
is the opt-out). `api_key` is sent as an `Authorization: Bearer <key>` header
by the default exporter. With `WARDEX_ENDPOINT` set in the environment, a bare
`wardex.init()` is a working first run — see
[Environment variables](#environment-variables).

## Configuration

Settings are grouped by concern, and the group names are the same in every
wardex SDK — a Node or Java service configured by the same team reads the same
way.

| Group | What it decides |
|---|---|
| `backend=BackendConfig(...)` | Where the data goes and whose it is: `endpoint`, `api_key` |
| `pii=PIIConfig(...)` | What leaves the process: `mode`, `disabled_categories` |
| `batching=BatchingConfig(...)` | When buffered spans are sent: `flush_interval`, `flush_on_signals`, `shutdown_timeout` |
| `limits=LimitsConfig(...)` | How much is captured — see [Resource limits](#resource-limits) |
| `propagation=PropagationConfig(...)` | Whether wardex touches outbound traffic: `enabled`, `targets` |
| `adapters=AdaptersConfig(...)` | Which framework adapters install, and each adapter's own options |

```python
import wardex_sdk as wardex
from wardex_sdk import BackendConfig, BatchingConfig, PIIConfig, PIIMode

wardex.init(
    backend=BackendConfig(endpoint="https://<collector>/v1/traces", api_key="..."),
    batching=BatchingConfig(flush_interval=2.0),
    pii=PIIConfig(mode=PIIMode.OFF),
)
```

Everything that belongs to no group stays top-level: `service_name`,
`release`, `environment`, `debug`, `before_send_envelope`, `capture_mode`, and
the interception trio `intercept` / `intercept_hosts` / `interceptors`. The
config dataclasses are keyword-only and frozen; each group validates its own
fields, so an invalid value fails on the line that constructed it. Collection
fields accept any iterable and read back canonicalized (tuple/frozenset);
enum-valued fields take enum members, not strings.

### Adapters

Framework adapters auto-detect by default. `adapters=` selects and configures
them:

```python
from wardex_sdk import AdapterName, AdaptersConfig, AnthropicAgentSdkConfig

wardex.init(
    ...,
    adapters=AdaptersConfig(
        enabled=(AdapterName.LANGGRAPH,),  # None auto-detects; () installs none
        anthropic_agent_sdk=AnthropicAgentSdkConfig(...),  # per-adapter options
    ),
)
```

Selection and options are separate fields on purpose: setting an adapter's
options never touches `enabled`, so auto-detection of every other framework
survives. Options set for an adapter that `enabled=` excludes are announced
with a `WardexConfigWarning`.

### Refused, never ignored

The flat spelling of a grouped setting (`api_key=...`, `flush_interval=...`,
the old `before_send=` and the `adapters=(AdapterName.X,)` tuple) is refused
with a `TypeError` naming its new home. Settings that were removed outright
(`retention=`, `tags=`, `replay_buffer_size=`) are refused with a message
saying why they are gone. There is no compatibility shim: a config setting
that is silently ignored is worse than one that stops the program on the line
that set it.

## Environment variables

Per field, an explicit `init()` argument wins; an unset one falls back to its
environment variable; only then does the default apply. The contract:

| Variable | Fills |
|---|---|
| `WARDEX_API_KEY` | `backend.api_key` |
| `WARDEX_ENDPOINT` | `backend.endpoint` — else `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, else `OTEL_EXPORTER_OTLP_ENDPOINT` |
| `WARDEX_SERVICE_NAME` | `service_name` |
| `WARDEX_RELEASE` | `release` |
| `WARDEX_ENVIRONMENT` | `environment` |
| `WARDEX_DEBUG` | `debug` — `true` (case-insensitive) can only turn it ON |

A host already exporting OTLP elsewhere points wardex at the same collector
with zero new variables. **The endpoint rule:** a URL with no path component
(or `/`) gets `/v1/traces` appended when the default transport is built —
`http://collector:4318` exports to `http://collector:4318/v1/traces` — while a
URL with an explicit path is used verbatim. The config always reads back
exactly as written.

With neither a `transport=` nor a resolved `backend.endpoint`, `init()`
installs `NoOpTransport`, captures into nothing, and says so once (see
[Diagnostics](#diagnostics)).

## Identifying your service

Three flat fields name the app in whatever OTel backend receives the data:

| Field | OTLP resource attribute |
|---|---|
| `service_name` | `service.name` (unset: `unknown_service:python`) |
| `release` | `service.version` (unset: not emitted) |
| `environment` | `deployment.environment.name` (unset: not emitted) |

`telemetry.sdk.name` is the constant `"wardex"` in every language — the SDK
travels in `telemetry.sdk.*`, never in your service's identity.

## Tracing

Everything is captured without instrumentation; the tracing API exists to add
**structure** — names, operations, and parents — around your own code.

**Context managers take a positional name; decorators take an optional
keyword name.** That family rule is frozen across SDKs.

```python
import wardex_sdk as wardex
from wardex_sdk import ToolAttributes

# A conversation: every span inside carries gen_ai.conversation.id.
with wardex.conversation("support-chat", id=session_id):  # id=None mints a uuid4
    ...

# A hand-named span over a block, yielding a Span to enrich:
with wardex.span("rank-results") as s:
    s.set_attribute("candidates", 42)
    ...


# Decorators — bare or with keywords; name defaults to the function's name:
@wardex.workflow
def nightly_sync(): ...


@wardex.agent(name="researcher")
def run_agent(): ...


@wardex.step
def plan(): ...


@wardex.tool(name="search", attributes=ToolAttributes(...))
def search(query: str): ...
```

`conversation()` is **not** a trace root: the span it opens joins the ambient
trace as a child, and an explicit `id=` is used verbatim — a multi-turn chat
app passes its own session id so every turn joins one conversation.
`workflow` / `agent` / `step` / `tool` map to the `gen_ai.operation.name`
values `invoke_workflow` / `invoke_agent` / `execute_step` / `execute_tool`,
so decorated spans appear on operation-keyed dashboards. `span()` and
`conversation()` are context managers only — using one as a decorator raises
a `TypeError` naming the decorators (a decorator would silently break async
functions).

## Scope

Ambient data that rides on every span captured under it:

```python
wardex.set_tag("tenant", "acme")  # spans carry the tag
wardex.set_user(wardex.UserInfo(id="u1", email=...))  # spans carry user.*
wardex.set_user(None)  # clears the user
wardex.set_context("job", {"attempt": 3})  # readable back; NOT exported

with wardex.isolation_scope():  # fork: inherits current tags/user, mutations stay inside
    wardex.set_tag("tenant", "other")  # this block only
    ...
```

`set_tag` / `set_user` / `set_context` write the **isolation scope** — inside
an `isolation_scope()` block (one request, one tenant, one job) they stay that
unit's and cannot bleed onto other threads' spans; outside any block they
behave like process-wide values in a simple script. Tags land on exported
spans (a span-local attribute with the same key wins); `UserInfo` maps to
`user.id` / `user.email` / `user.name` and `client.address`. Contexts are for
your own code to read back and are not stamped onto spans.

## Distributed tracing

Trace context propagation is **opt-in** — a plain `wardex.init(...)` never
touches your outbound requests or headers. Turn it on with:

```python
from wardex_sdk import PropagationConfig

wardex.init(
    backend=BackendConfig(endpoint="https://<collector>/v1/traces"),
    propagation=PropagationConfig(
        enabled=True,  # inject W3C headers on outbound calls
        targets=(
            "api.internal.example.com",
            "*.svc.cluster.local",
        ),  # optional glob allowlist; default None = all hosts
    ),
)
```

With `propagation=PropagationConfig(enabled=True)`, outbound calls made through httpx (sync + async),
requests, or aiohttp get a `traceparent` (and `tracestate`, if one was
received) header attached automatically, as long as an active trace context
exists and the request doesn't already carry a `traceparent`. **If
`propagation.targets` is left unset, the trace ID is sent to every host you
call — including third-party LLM providers.** Set it to an allowlist of glob
patterns to scope injection to your own services; patterns are matched
case-insensitively, since hostnames are.

wardex only ever *adds* a header you did not write. A `traceparent` you set
yourself wins whether you set it per request or once as a session default, and
in that case nothing is injected at all; a `tracestate` you set is left exactly
as written rather than replaced or duplicated.

### Joining an inbound trace

Drop the middleware in front of your app to join whatever trace the caller
started:

```python
# ASGI (FastAPI, Starlette, Django ASGI)
app.add_middleware(wardex.WardexAsgiMiddleware)

# WSGI (Flask, Django WSGI)
app.wsgi_app = wardex.WardexWsgiMiddleware(app.wsgi_app)
```

Both extract the incoming `traceparent`/`tracestate` and continue the trace
for the lifetime of the request; a missing or malformed header just starts a
fresh trace (never raises). One WSGI caveat: the joined context covers the
app callable only, so streaming responses (work done while iterating the
returned iterable) run outside it.

### Manual propagation (the universal escape hatch)

The baton is just a string, so it travels over any channel that can carry
one — not just HTTP. `get_traceparent()` and `get_trace_headers()` are plain
functions that return the current trace headers; `continue_trace(headers)`
is a **context manager** — the remote parent is only installed inside the
`with` block, so it must be entered, not merely called. Use them directly
wherever the automatic client patches or ASGI/WSGI middleware don't reach:

```python
# gRPC metadata
stub.Check(req, metadata=[("traceparent", wardex.get_traceparent())])

# WebSocket handshake
websockets.connect(uri, extra_headers=wardex.get_trace_headers())

# Celery: put get_trace_headers() on the task's headers when sending it,
# then inside the worker:
with wardex.continue_trace(task.request.headers):
    ...  # task body

# Kafka: put get_trace_headers() on the message headers when producing,
# then inside the consumer:
with wardex.continue_trace(dict(msg.headers())):
    ...  # process the message
```

**Pass it a `traceparent` a caller actually sent you, and nothing else.**
`continue_trace()` takes a string and takes it at its word — that is what makes
it work over any channel, and it is also its one sharp edge. Any string of the
right shape becomes a parent, so deriving one from something that is not a
propagated trace context (a framework's `run_id`, a request id, a hash of a job
name) manufactures a causal edge that never existed. Spans parented this way are
recorded with `parent_source = header`, which is how they stay distinguishable
from the in-process edges wardex derives itself; what wardex cannot tell you is
whether the header was genuine, because both are just strings. Everywhere else,
the parent comes from real context propagation and is never built from an
identifier.

`with wardex.continue_from_otel():` is a one-line alternative to
`continue_trace()` for code that already runs under an active OpenTelemetry
span — it adopts that span as the remote parent for the duration of the
`with` block (no-op if `opentelemetry` isn't installed or there's no active
span). Like `continue_trace()`, it is a context manager and must be entered
with `with`.

### Propagating into threads

`asyncio` tasks inherit the current trace context automatically; threads do
not. Wrap the target with `wardex.bind_context()` at the point where you
still have the right context:

```python
thread = threading.Thread(target=wardex.bind_context(worker_fn), args=(...,))
thread.start()
```

### `capture_mode`: what gets captured without an active span

`capture_mode` defaults to `CaptureMode.AGENT`: LLM-semantic traffic
(recognized `gen_ai` calls, MCP stdio) is always captured, but generic
HTTP/gRPC/WS traffic is only captured while it happens inside an active
*local* wardex span (a `traceparent` received from an upstream caller doesn't
count on its own — this keeps a service mesh stamping every request with a
traceparent from reviving "capture everything" noise).

A recognized provider's **failed** calls count as LLM-semantic traffic too, so
a 429 or a 401 is captured under the default mode even outside a local span —
the response status decides the span's `status`, never whether it exists.

This means a bare, unwrapped call to an LLM provider wardex doesn't
recognize (or a WS-based provider such as OpenAI Realtime, which carries no
parseable semantics) can be silently dropped if it isn't inside a local
span. Wrap it with `@wardex.workflow` (or any of the span decorators), or set
`capture_mode=wardex.CaptureMode.ALL` to restore capture-everything behavior:

```python
wardex.init(..., capture_mode=wardex.CaptureMode.ALL)
```

Plaintext hosts you've explicitly named via `intercept_hosts` are always
captured regardless of `capture_mode` — a targeted allowlist entry is a
stronger opt-in than the default policy.

## asyncio

`wardex.flush(timeout=None)` and `wardex.close(timeout=None)` are synchronous
and **block the calling thread** — a bare `flush()` waits as long as the
transport's own configured timeout, a bare `close()` follows
`batching.shutdown_timeout` (5s default). From async code, run them in an
executor so the event loop keeps breathing:

```python
await asyncio.get_running_loop().run_in_executor(None, wardex.flush)
```

Capture itself never blocks your coroutines: spans are buffered and exported
from wardex's own background worker thread. `asyncio` tasks inherit the trace
context automatically; only hand-started threads need
`wardex.bind_context()` (above).

## Errors

The exception contract has exactly two tiers:

* **Configuration time raises.** `init()` and the config classes raise
  `TypeError` / `ValueError` on a bad argument, like any Python constructor —
  a config mistake stops the program on the line that made it.
* **Runtime never raises.** Every capture and export path is fail-safe:
  nothing wardex does after `init()` returns raises into host code, and every
  public call (`flush`, `close`, `set_tag`, the tracing helpers, ...) is a
  safe no-op before `init()` was ever called.

There is deliberately no `WardexError` base class — there is no wardex
exception a host is ever expected to catch.

## Diagnostics

Everything wardex says about itself goes through the stdlib logger
**`wardex_sdk`**. Out of the box it carries one pre-attached stderr handler,
so with zero configuration you see one-line messages prefixed `[wardex] ` —
announcements (the NoOp-transport notice, the `debug=True` config dump) at
INFO, losses and failures (a span dropped over the size cap, an export cut
off at shutdown) at WARNING. Some finer-grained lines — a disabled parser, a
buffer-full drop count — additionally sit behind `debug=True`; the gate
changes whether they fire, never their severity.

To route diagnostics into your own logging setup, replace the handler — the
`[wardex] ` prefix lives in wardex's own handler, so yours receives clean
messages:

```python
import logging

logger = logging.getLogger("wardex_sdk")
logger.handlers.clear()
logger.addHandler(my_handler)  # or logging.NullHandler() to silence
```

The logger does not propagate to the root logger, so nothing double-prints
under `logging.basicConfig()`. A `wardex_sdk` logger you configure *before*
importing wardex is left untouched. Configuration conflicts — a setting
another setting disables — are not log lines but real warnings
(`WardexConfigWarning`), filterable with the `warnings` module.

## Testing your instrumentation

`RecordingTransport` is the in-process test double: exports are recorded, not
sent, and read back as structural nodes.

```python
import wardex_sdk as wardex
from wardex_sdk.testing import RecordingTransport

transport = RecordingTransport()
wardex.init(transport=transport)

run_the_code_under_test()
wardex.flush()

names = [node.name for node in transport.spans]
assert "execute_tool search" in names
```

`transport.spans` yields every recorded span in export order as `SpanNode`s
(name, span/parent/trace ids, parent-resolution confidence, limitation
markers). Note it records **pre-masking, in-process data** — what was
captured, not what a backend would have received.

## Bring your own transport

`Transport` is the one advertised extension point: subclass it, implement
`export()`, and pass an instance as `init(transport=...)`. An explicit
`transport=` carries its own address and wins over `backend.endpoint` (the
losing endpoint is announced with a `WardexConfigWarning`) — never configure
both expecting both to apply.

```python
from wardex_sdk.transport import UNDELIVERED, Transport, Undelivered
from wardex_sdk import Envelope


class MyTransport(Transport):
    def export(self, envelope: Envelope, *, timeout: float | None = None) -> Undelivered | None:
        for body in self.encode(envelope):  # masked, capped, split OTLP bodies
            post(body)  # one POST per body, in order
        return None  # or UNDELIVERED if nothing was sent
```

The contract, in brief — `wardex_sdk.transport` is the complete implementer
home (`Transport`, `NoOpTransport`, `ConsoleTransport`, `OtlpHttpTransport`,
`Envelope`, `UNDELIVERED`, `Undelivered`, `CallerBudget`, `DEFAULT_TIMEOUT`):

* **Threading:** `export` / `flush` / `close` are called from wardex's own
  worker thread, never from your event loop or request threads, and may block
  up to their budget. That is the cross-language contract (Node binds
  `export` as async and its pipeline awaits it; Java stays blocking).
* **The envelope is opaque.** Its guaranteed surface is `span_count` and
  `Transport.encode(envelope)`, which returns the wire bodies — protobuf,
  gzipped by default, PII-masked, attribute-capped, and split at
  `max_otlp_request_bytes`. `export()` itself receives **pre-masking** data:
  serialize the envelope yourself and you own PII masking. Do not override
  `encode()`.
* **Declines:** return `UNDELIVERED` only when the envelope was not put on
  the wire and an identical later attempt could succeed; the SDK then keeps
  the spans for the next drain. Anything else (including `None`) means
  "taken".
* **Timeouts:** `timeout` is the remaining budget for this export (`None` =
  no deadline); honour it by narrowing your own configured timeout, never
  widening it. Declare how long a bare `flush()` should wait via the
  `export_timeout` attribute. `CallerBudget` (a `float` subclass marking
  budgets the application named) is a Python-only diagnostic refinement, not
  part of the cross-language SPI.

`ConsoleTransport` prints envelopes **raw — pre-masking — to stdout**; it is a
local debugging tool, never an export path.

### `before_send_envelope`

The last-look hook before an export: `init(before_send_envelope=hook)` with
`hook(envelope) -> Envelope | None`. Return the **received** envelope object
to send (in v1 the only legal non-None return is that same object), or `None`
to drop the batch. It is synchronous, sees **pre-masking** data (masking runs
after it, inside the encoder), may run more than once for a batch a transport
declined, and a raise inside it drops the batch fail-closed with one
diagnostic line (traceback under `debug=True`).

## Status

**Works today**
- Zero-instrumentation capture of LLM HTTP calls (OpenAI, Anthropic) over
  `https`, cleartext `http`, and h2c
- `gen_ai` semantics: model, tokens, parameters, finish reasons, input/output messages
- Failed provider calls (429 rate limits, 401s, 5xx) are captured with the same
  `gen_ai` identity and content as successful ones — only the response-side
  fields are empty
- Transport metrics (TCP/TLS timing, TTFT), gRPC (grpclib), WebSocket (`wss`), MCP stdio
- Export to any OpenTelemetry backend via `OtlpHttpTransport`
- Manual span decorators: `@workflow` / `@agent` / `@step` / `@tool`
- PII masking on by default: emails, phone numbers, credit cards (Luhn-verified),
  US SSNs, IP addresses, bank routing numbers, IBANs, and API-key/token secrets
  are masked before anything leaves the process (`pii=PIIConfig(mode=PIIMode.OFF)`
  to disable, `pii=PIIConfig(disabled_categories={PIICategory.IP_ADDRESS})` for
  per-category opt-out)
- Background batching: automatic flush every 5s / on buffer threshold /
  at exit and on SIGINT/SIGTERM (chained; opt out with
  `batching=BatchingConfig(flush_on_signals=False)`)
- Shutdown closes agent runs that are still in flight, so an interrupted run
  still exports its span — marked `unit_interrupted` or `adapter_uninstalled`
  — instead of vanishing along with its open tool calls
- Framework adapter: Anthropic Agent SDK (`claude_agent_sdk`) — auto-detected,
  zero-instrumentation `invoke_agent`/`chat` spans with tool-call correlation
- Framework adapter: **LangGraph** (`langgraph>=1.2`) — auto-detected, no
  callbacks and no `LangChainTracer`. One `invoke_workflow` span per graph run,
  one `execute_step` span per node (all retries of a node inside ONE span), one
  `execute_tool` span per tool call dispatched by a `ToolNode`, and every LLM
  and HTTP call underneath parented by in-process context propagation rather
  than by a framework `run_id`. Covers `invoke`/`stream`/`ainvoke`/`astream`/
  `batch`/`abatch`, the functional API (`@entrypoint`/`@task`), subgraphs, and
  agents built with either `langgraph.prebuilt.create_react_agent` or
  `langchain.agents.create_agent`. `interrupt()` and `Command(goto=…,
  graph=PARENT)` are recorded as control flow, not as failures. A
  `RemoteGraph` (LangGraph Platform) call ships one `invoke_workflow` span
  marked `wardex.langgraph.remote`, with the platform HTTP request underneath;
  the remote run's internals execute out of process and are not captured. A
  cached node ships no span — no work ran.

**Not yet (see Roadmap)**
- A LangChain adapter for plain LCEL chains (`prompt | model | parser`) and
  tools invoked outside a graph — those produce no structural spans today, and
  a LangChain-built *agent* is covered by the LangGraph adapter above because
  `create_agent` compiles to a `Pregel` graph
- Framework adapter for the OpenAI Agents SDK
- Node/TS and Java SDKs

**Notes**
- After `os.fork()` the worker respawns lazily in the child on first capture;
  spans buffered before the fork may be sent by both processes (duplicates
  are possible; a fork landing mid-export can also strand the child's
  pre-fork buffer — re-init in the child for a clean slate). Under uWSGI
  enable threads (`--enable-threads`).

## Resource limits

Every resource bound in the SDK — body size caps, buffer sizes, connection
and session tracking — is configurable through `limits=LimitsConfig(...)`, but
the defaults suit most workloads and most users never need to touch this. The
body cap is set above the Anthropic Messages API's request size ceiling, so a
request the API itself accepts is never truncated by capture.

```python
import wardex_sdk as wardex
from wardex_sdk import LimitsConfig

wardex.init(
    limits=LimitsConfig(
        max_body_bytes=64 * 1024 * 1024,  # larger multimodal payloads
        max_buffer_bytes=16 * 1024 * 1024,  # tighter memory budget
    )
)
```

**`max_body_bytes` bounds two quantities, and only one of them is a message.**
Besides capping a captured request or response body, it caps the bytes ONE
logical unit accumulates from its adapter-side records — a tool call's input
and output, an agent turn's payload — and the shaping budget an adapter derives
from that cap so it stops building a representation exactly where storage would
cut. Lowering it therefore lowers resident memory per live unit, which is the
reason to lower it. Raising it raises that memory: the worst case is about
`2 x max_body_bytes` per live unit, and the ceiling on live units is
`max_units x max_entries_per_unit`, not `max_units` — `max_buffer_bytes` bounds
the span buffer and does not cover a unit that is still open.

Raising it also raises a TRANSIENT cost that is paid on your own thread. The
LangGraph adapter shapes a tool call's arguments synchronously inside the tool
call, and building that representation peaks near four times the cap for
escape-heavy text: measured at 128 MiB peak / 65 ms for one 64 MiB string
argument at the 32 MiB default, and 256 MiB / 135 ms at the 64 MiB setting
suggested above. Raise it for payloads you want captured whole, not by reflex.

**Two of the bounds are about the wire rather than about capture.** The OTLP
surface encodes binary payloads as base64, so what leaves is up to a third
larger than what was captured, and an OTLP request is accepted or rejected
whole — a batch over the receiver's body limit does not arrive short, it does
not arrive.

* `max_otlp_attribute_bytes` (1 MiB) caps one attribute value as it appears on
  the wire. A value over it is truncated and the span says so with an
  `otlp_attribute_truncated` marker in `wardex.limitations`.
* `max_otlp_request_bytes` (4 MiB, gRPC's own receive ceiling) caps one
  request, measured both as the compressed body that goes on the wire and as
  the message it decompresses to — receivers check both. A batch over either is
  split across several POSTs instead of being sent whole and rejected. Raise it
  if your collector accepts more.

  A split is invisible in your traces, but that is the receiver's doing rather
  than the SDK's: the POSTs carry the same trace id, and a receiver keys spans
  by it, so what was one batch is stored and shown as one trace. Children
  routinely arrive in earlier requests than the parent they name — spans leave
  in completion order, so the root travels last — and a conforming receiver
  resolves the edge when the parent lands. A span so large it would not fit a
  request even with its payload removed is the one loss a split cannot absorb:
  it is dropped, the rest of its batch still goes, and the SDK says so on the
  diagnostic channel — once per process, at the first occurrence, with a count
  that covers that batch and is not a running total. That loss cannot be marked
  in `wardex.limitations` the way a truncation is, because the marker would
  have to ride on the very span that never reaches the wire.

Requests are gzipped by default. `OtlpHttpTransport(..., compress=False)` turns
that off for a proxy or receiver that mishandles `Content-Encoding`.

`max_units` and `max_entries_per_unit` were on that list until the logical-unit
registry landed and became their consumer. What crossing one of them looks like
from your data depends on whether the evicted entry has a span of its own.

**Evictions you can see in your traces.** A root unit evicted at `max_units`,
and a child unit or an in-flight span evicted at `max_entries_per_unit`, are
each closed and **exported**, marked `unit_evicted` or `child_span_unclosed`.
Outgrowing one of these ceilings shows up as marked spans rather than as traces
that quietly stop appearing.

**Evictions you cannot.** `max_entries_per_unit` also bounds bookkeeping tables
whose entries are not spans — the lookup aliases that map a framework's own
identifiers onto units, the keys that de-duplicate two observers of one event,
and the table of in-process MCP servers wardex has wrapped. Evicting from any of
them exports nothing, because there is no span to mark. They are counted
internally instead — `wardex_sdk._assembly.counters.snapshot()` reports them
under `assembly._units.alias_table_full`, `assembly._units.claim_table_full` and
`adapters.anthropic.server_table_full` — and what reaches your data is the
consequence rather than the eviction. A dropped de-duplication key, or a dropped
server handle, can let one tool call be reported twice.

A dropped **alias** is the one to know about, because it does not look like a
loss. That identifier stops resolving, so the parent is decided one rung further
down: if the work carries an ambient wardex span, the span arrives at confidence
**1.0 with no marker** — hanging off the enclosing session instead of the
sub-agent it belonged to. A sub-agent's subtree flattens and nothing in the data
says so. Only when there is no ambient span does it arrive marked
`unit_inferred_sole` (0.5) or `parent_unresolved`.

Raise `max_entries_per_unit` if you see any of these counters move.

## Versioning

Version semantics, what 0.x betas may break, the deprecation mechanism, and
how the SDKs and the wire schema version relative to each other are recorded
in [VERSIONING.md](VERSIONING.md). `wardex_sdk.__version__` is the canonical
runtime version probe.

## Roadmap

1. ~~PII masking (pre-send safety)~~ — shipped
2. ~~Batching & lifecycle (background worker, at-exit/periodic flush, concurrency)~~ — shipped
3. ~~Distributed propagation (W3C)~~ — shipped
4. Framework adapters — ~~Anthropic Agent SDK~~ shipped; ~~LangGraph~~ shipped;
   LangChain (non-graph runnables) and OpenAI Agents SDK next
5. Node/TS and Java SDKs

> PII masking caveats: `before_send_envelope` sees pre-masking data (masking runs inside
> the encoder), the Console transport prints raw (local debugging only), and
> non-UTF-8 binary payloads pass through unmasked.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

"Wardex" is a trademark of Wardex Labs.
