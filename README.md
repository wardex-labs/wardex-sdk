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

## Works with openai-agents

Install wardex next to the OpenAI Agents SDK, set one environment variable,
and every `Runner.run` / `run_sync` / `run_streamed` becomes one trace with
the agents, the tool calls and the handoffs in it — no decorator, no
callback, no processor to register. The framework's own tracing hooks give
wardex the structure; the LLM calls underneath are read from the wire, so
model, messages and token usage are the ones that actually crossed the
socket, and the tool `call_id` the framework echoes into the next turn joins
each `execute_tool` span to the `chat` span that requested it.

From a checkout of this repository (the example script is not in the
wheel), on Python 3.10 or newer, with a Phoenix started as in
[`examples/README.md`](examples/README.md):

```bash
git clone https://github.com/wardex-labs/wardex-sdk && cd wardex-sdk
python3 -m venv .venv-quickstart && source .venv-quickstart/bin/activate
pip install "wardex-sdk>=0.6.0b1" openai-agents          # prebuilt wheel, nothing to compile
export OPENAI_API_KEY=sk-...                             # the framework's own requirement
export WARDEX_ENDPOINT=http://127.0.0.1:6006/v1/traces   # a local Phoenix
python examples/openai_agents_quickstart.py
```

The adapter ships in 0.6.0b1; until that version is on PyPI, `pip install
./sdks/python openai-agents` builds it from the checkout (Rust toolchain
required, and the build reports the checkout's own, lower, version
number — expected). The script makes six short `gpt-4o-mini` calls against
your key. Measured from a fresh clone and virtualenv following the
walkthrough, Phoenix image already pulled: under a minute in the terminal,
18 seconds of it the from-checkout build. That is the measured part; the
clicks in Phoenix to the tree below add an estimated half-minute on top,
paced by hand rather than clocked.

<!-- Absolute URL on purpose: this file is also the PyPI long description
     (readme = "README.md" in sdks/python/pyproject.toml), where a relative
     image path renders as a broken link. The trade-off is that the image
     404s on branches and in PR views until the PNG is on main. -->
![Phoenix showing one openai-agents run: invoke_workflow travel_concierge → invoke_agent concierge (chat, execute_tool lookup_weather, chat, handoff concierge→booking_agent) and its sibling invoke_agent booking_agent (chat)](https://raw.githubusercontent.com/wardex-labs/wardex-sdk/main/examples/openai-agents-phoenix.png)

The receiving agent of a handoff is the sender's **sibling**, not its
child, so a long handoff chain stays one level deep; `wardex.agent.parent`
and a `handoff_from` link record who handed off to whom. Phoenix draws that
indentation only once its trace drawer is widened — the walkthrough in
[`examples/README.md`](examples/README.md) says where to click and what to
drag, covers Langfuse, and explains the framework's own `[non-fatal]
Tracing client error 401` line if you see one.

Two cases where you do **not** get that tree, and what wardex says instead:

- **Responses over WebSocket** (`use_responses_websocket=True`). The LLM
  calls are inside WebSocket frames wardex does not parse, so the `chat`
  spans are replaced by one `WS /v1/responses` span per connection carrying
  the marker `ws_llm_semantics_unread` — no model, tokens or messages. The
  agent, tool and handoff spans are unaffected. Use the framework's default
  HTTP transport for `chat` spans.
- **Framework tracing disabled** (`OPENAI_AGENTS_DISABLE_TRACING=1` or
  `agents.set_tracing_disabled(True)`). There is nothing for the adapter to
  hook, so you get the `chat` spans only, unparented, and one line on stderr
  at `wardex.init()`: `[wardex] openai-agents tracing is disabled, so wardex
  will show only the LLM calls its interceptor captures: no agent, handoff,
  tool or guardrail spans. …` — followed by the two lines that re-enable the
  framework's tracing without sending anything to OpenAI. wardex never flips
  that setting for you.

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
        enabled=(  # None auto-detects; () installs none
            AdapterName.ANTHROPIC_AGENT_SDK,
            AdapterName.LANGGRAPH,
            AdapterName.OPENAI_AGENTS,
        ),
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
app passes its own session id so every turn joins one conversation. It also
**wins over a framework's own conversation id**: an adapter run opened inside
the block (an OpenAI Agents `RunConfig(group_id=…)`, say) keeps your id on
every span and carries the framework's as a separate attribute
(`wardex.openai_agents.group_id`), so one trace never has two conversation ids.
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
recognize — or a WebSocket provider whose path wardex doesn't recognize, such
as OpenAI Realtime — can be silently dropped if it isn't inside a local span.
Wrap that call with `@wardex.workflow` (or any of the span decorators), or set
`capture_mode=wardex.CaptureMode.ALL` to restore capture-everything behavior:

```python
wardex.init(..., capture_mode=wardex.CaptureMode.ALL)
```

**Responses over WebSocket (openai-agents `use_responses_websocket=True`).**
This opt-in transport sends every call over one `wss://…/v1/responses`
connection, and it is the one WebSocket case that ships under the default
mode without a wrapper: once the first call crosses the connection it is
an LLM connection, and when the connection closes wardex counts it
(`interceptors.seam.ws_llm_semantics_unread`) and emits one
`WS /v1/responses` span marked
`ws_llm_semantics_unread` — LLM calls crossed it and wardex read none of
them, because Responses events inside WebSocket frames are not parsed. The
span carries `ws.messages.sent` (about one per call), byte counts and payload
samples (compressed bytes, marked `payload_compressed`, when
permessage-deflate was negotiated), no model or tokens; switch the framework
to its default HTTP transport for `gen_ai` spans. A connection that ends
without a WebSocket close handshake — a server drop, a timeout, process exit,
or wardex uninstalled first — still yields the span, additionally marked
`ws_no_close`. The connection counts as an LLM connection only when the host
is the provider's own — exactly `api.openai.com` or a subdomain of
`openai.com`; a host that merely contains the name, such as
`openai-mock.corp`, is not — or when the first client message carries a
Responses `"type": "response.create"` (anywhere in that message, whatever
the key order) on an uncompressed connection. The decision is made once,
on that first client message, for the life of the connection. A
Responses-path connection
to any other host (localhost, a gateway, a mock) with compression — the
`websockets` client's default — is only counted
(`interceptors.seam.ws_llm_endpoint_unconfirmed`) and yields no span under
the default mode. Under `ALL`, under an `intercept_hosts` entry,
or inside a local span that same connection does ship — but as an ordinary
WebSocket span, `WS /v1/responses` with status OK and no
`ws_llm_semantics_unread` marker (measured against a loopback server: one
span, markers `['payload_compressed']` only) — so the counter is the only
signal that unread LLM calls crossed it.

Plaintext hosts you've explicitly named via `intercept_hosts` are always
captured regardless of `capture_mode` — a targeted allowlist entry is a
stronger opt-in than the default policy.

**One thing is never captured, in any mode and above any allowlist:
telemetry uploads.** Any host, any path ending in `/v1/traces/ingest` — the
OpenAI Agents SDK POSTs its whole run record there by default. The rule is
by path: a custom exporter endpoint (`BackendSpanExporter(endpoint=…)`) that
keeps the `/v1/traces/ingest` path is covered; one on another path is not —
under `ALL`, under `intercept_hosts`, or inside a local span it ships as an
ordinary HTTP span with the run record as its `input_data`. That body is
yours on its way to a tracing backend, not agent activity, so wardex skips the
request before parsing it or attaching it to a span, and counts the skip
under `interceptors.seam.path_excluded`. To be precise about where that
body goes: the bytes pass through wardex's per-connection buffer like any
other request's (capped by `LimitsConfig.max_body_bytes`), and are then
discarded — never parsed, never on a span, never exported. If your own
service exposes that path, its requests are skipped by the same rule.
Server-side conversation state
(`…/v1/conversations/…`) is plain HTTP rather than an LLM call — no model,
no usage — so it follows the non-LLM rule above: captured inside a local
span, under `ALL` or under `intercept_hosts`, otherwise dropped and counted
under `interceptors.seam.provider_state_dropped`. With the OpenAI Agents
SDK's `Runner.run(conversation_id=…)` (measured on 0.22) each turn is still
a `chat` span, but its input is only the items the framework had not sent
yet — the second turn's input is the tool result alone — and the
`conversation` id that joins the turns is not yet surfaced as a span
attribute.

## asyncio

`wardex.flush(timeout=None)` and `wardex.close(timeout=None)` are synchronous
and **block the calling thread** — a bare `flush()` waits as long as the
transport's own configured timeout, a bare `close()` follows
`batching.shutdown_timeout` (5s default). From async code, run them in an
executor so the event loop keeps breathing:

```python
await asyncio.get_running_loop().run_in_executor(None, wardex.flush)
```

Capture itself never blocks your coroutines: spans are buffered, **parsed**
and exported from wardex's own worker threads. The expensive step — the
LLM-semantic parse of a completed response — runs on a dedicated
`wardex-finalize-worker` thread with the GIL released, so a large streamed
completion finishing does not stall the event loop; `flush()` finishes any
pending parses before it sends (which is also why calling it from async code
belongs in an executor, as above). `asyncio` tasks inherit the trace
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

**Counters.** Conditions a span cannot carry — a request wardex skipped, a
WebSocket connection it only counted — are tallied in-process:
`wardex_sdk._assembly.counters.snapshot()` returns `{site: count}`. This
section's neighbours mention `interceptors.seam.path_excluded` (telemetry
uploads skipped), `interceptors.seam.provider_state_dropped` (requests on a
Conversations-API-shaped path the mode did not capture),
`interceptors.seam.ws_llm_semantics_unread` (WebSocket connections that
carried LLM calls wardex did not read) and
`interceptors.seam.ws_llm_endpoint_unconfirmed` (Responses-path WebSocket
connections wardex could not corroborate); table-eviction counters are under
Resource limits.

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
  `https`, cleartext `http`, and h2c — Chat Completions, the **Responses API**
  (the openai-agents SDK's default path, non-streaming and SSE, plus
  `/v1/responses/compact`), Embeddings, and Anthropic Messages
- `gen_ai` semantics: model, tokens, parameters, finish reasons, input/output messages
- **Open usage capture**: every scalar leaf of the provider's `usage` object
  rides the span as `wardex.usage.<provider path>`, spelling preserved — a new
  billing counter (a cache-write tier, a web-search charge, a thinking tier)
  appears in your data the day the provider ships it, without an SDK release.
  Bounded by `max_extra_keys` (default 64; a real usage object has 10–20
  leaves), and a crossed bound says so: marker `extra_keys_dropped` plus
  `wardex.usage_leaves.dropped_count`
- Failed provider calls (429 rate limits, 401s, 5xx) are captured with the same
  `gen_ai` identity and content as successful ones — only the response-side
  fields are empty
- Transport metrics (TCP/TLS timing, TTFT), gRPC (grpclib), WebSocket (`wss`;
  a Responses-over-WebSocket connection is captured at close and marked
  `ws_llm_semantics_unread` when the host is `api.openai.com` or a subdomain
  of `openai.com` — on any other host with compression it is only counted,
  see capture_mode), MCP stdio
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
- Framework adapter: **OpenAI Agents SDK** (`openai-agents>=0.22,<0.23`) —
  auto-detected, hooked through the framework's own `TracingProcessor`,
  nothing internal patched. One `invoke_workflow` per `Runner.run` /
  `run_sync` / `run_streamed`, one `invoke_agent` per agent, a
  `handoff {from}→{to}` marker with the receiving agent as the sender's
  sibling (not nested — `wardex.agent.parent` and a `handoff_from`
  link carry the causality), one `execute_tool` per function tool with the
  call id recovered by a unique match against the response that requested it
  (labelled `wardex.openai_agents.tool_call_id_source`), one `evaluate` per
  guardrail, and `RunConfig(group_id=…)` as `gen_ai.conversation.id` on
  every adapter span — unless the run sits inside the host's own
  `wardex.conversation(...)`, in which case the host's id stays on every
  span and the group id rides along on the root as
  `wardex.openai_agents.group_id`. The LLM calls stay the wire's `chat` spans, parented
  under the agent by context; the adapter discards the framework's usage so
  nothing is billed twice. By default the framework's own upload to
  `api.openai.com/v1/traces/ingest` continues unchanged; wardex does not
  replace it. For the structure without that upload, IN THIS ORDER:
  `agents.set_trace_processors([])` and THEN `wardex.init()` (the reverse
  order removes wardex's processor too). `RunConfig(workflow_name=…)` names
  the root; the default is `Agent workflow`. Runnable end to end in
  [`examples/openai_agents_quickstart.py`](examples/openai_agents_quickstart.py)
  (see [Works with openai-agents](#works-with-openai-agents)). Known limitations: the wire
  `chat` spans carry no `gen_ai.agent.name` — filter by walking up the tree
  to the `invoke_agent` span; a Responses-over-WebSocket run stays the
  counted, marked connection (`ws_llm_semantics_unread`) with no structure
  read from the frames; with the framework's tracing disabled wardex logs one
  INFO line at install and shows only the LLM calls; a `max_turns` handled by
  `error_handlers` still ships ERROR on the agent and the root (the
  framework marks the span before consulting the handler); and with
  `trace_include_sensitive_data=False` the tool span carries the
  `tool_call_id_unavailable_in_process` marker rather than a call id.

**Not yet (see Roadmap)**
- A LangChain adapter for plain LCEL chains (`prompt | model | parser`) and
  tools invoked outside a graph — those produce no structural spans today, and
  a LangChain-built *agent* is covered by the LangGraph adapter above because
  `create_agent` compiles to a `Pregel` graph
- The `conversation` id in the request is not yet surfaced as a span
  attribute on the wire `chat` spans (no `gen_ai.conversation.id` there). It
  is a plain field in the Responses request body — measured, present in
  every POST of a `Runner.run(conversation_id=…)` run — and the byte seam
  latches only the span context at request time, so the adapter's
  `group_id` does not reach the chat spans either; it is a wire-side
  follow-up
- Node/TS and Java SDKs

**Notes**
- After `os.fork()` the SDK reinitializes its per-process state in the child
  via `os.register_at_fork`: the inherited span buffer and any pending parse
  jobs are discarded (the parent still owns and exports them, so each span
  ships exactly once), locks and the worker threads are recreated, per-connection/per-session tracking
  tables are reset (a span assembled on a connection that crossed the fork
  carries the `tracking_reset_at_fork` marker), and every batch stamps the
  live `process.pid`, so a parent and its forked children are distinguishable
  at the backend. `multiprocessing` fork children flush their tail on exit; a
  hand-rolled `os.fork()` + `os._exit()` child should call `wardex.flush()`
  before exiting. `spawn`/`forkserver` start methods launch a fresh
  interpreter and are unaffected. Under uWSGI enable threads
  (`--enable-threads`).

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

The deferred-parse queue has its own pair: `max_parse_backlog` (2048) and
`max_parse_backlog_bytes` (64 MiB) bound how many completed-but-unparsed
transactions may wait for the finalize worker. Over either bound, the OLDEST
waiter ships immediately without its `gen_ai` block, carrying the
`parse_backlog_full` marker — never silently — and a transaction still
pending when a shutdown budget runs out ships the same way under
`parse_skipped_at_shutdown`. Resident memory is therefore at most one
backlog plus one span buffer.

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

**`max_extra_keys` (64) bounds one open key family today: the provider-usage
mirror.** `wardex.usage.*` is the one attribute family whose keys the provider
names rather than wardex, so it is the one place a pathological body could mint
unbounded keys. Lowering the knob prunes usage leaves only — every other
attribute is untouched — and a span that lost leaves carries
`extra_keys_dropped` with the count beside it as
`wardex.usage_leaves.dropped_count`. The normalized `gen_ai.usage.*` totals are
extracted separately and are never subject to this cap. (For scale: the mirror
adds ~5–10 attributes to an LLM span, and the richest span the test corpus
produces carries 47 attributes total against the OTel Collector's default
`attribute_count_limit` of 128.)

`max_units`, `max_entries_per_unit` and `max_session_entries` were on that list
until the logical-unit registry and the Agent SDK assembler became their
consumers. What crossing one of them looks like from your data depends on
whether the evicted entry has a span of its own.

**Evictions you can see in your traces.** A root unit evicted at `max_units`, a
child unit or an in-flight span evicted at `max_entries_per_unit`, and an open
tool call or a sub-agent evicted at `max_session_entries`, are each closed and
**exported**, marked `unit_evicted`, `unit_table_full` or
`session_entry_table_full`. Outgrowing one of these ceilings shows up as marked
spans rather than as traces that quietly stop appearing. Each marker names the
one knob that produced it, so the marker tells you which number to raise.

A span evicted at `max_session_entries` carries status **unset** rather than
`ok` or `error`: wardex stopped watching before the call's outcome, so `ok`
would claim a success it never observed and `error` would report wardex's own
full table as a failure of your agent.

**One call, two observations.** If an evicted tool call later completes, its
completion is reported as a *second* span with the same `gen_ai.tool.call.id`,
also marked `session_entry_table_full`, and the two overlap: the `unset` one is
`[start, evicted]` and holds the call's input, the other is `[start, end]` — the
whole call — and holds its output. **When you aggregate tool latency, exclude
the spans that carry `session_entry_table_full` AND status `unset`,** or you
count that call twice.

**Evictions you cannot.** The two per-table knobs also bound bookkeeping tables
whose entries are not spans — the lookup aliases that map a framework's own
identifiers onto units, the keys that de-duplicate two observers of one event,
the table of
in-process MCP servers wardex has wrapped, and the streamed tool metadata the
assembler holds until a result arrives. Evicting from any of them exports
nothing, because there is no span to mark. They are counted internally instead —
`wardex_sdk._assembly.counters.snapshot()` reports them under
`assembly._units.alias_table_full`, `assembly._units.claim_table_full`,
`adapters.anthropic.server_table_full` and
`adapters.assembler.stream_tool_meta_table_full` — and what reaches your data is
the consequence rather than the eviction. A dropped de-duplication key, or a
dropped server handle, can let one tool call be reported twice. A dropped
streamed metadata entry costs a tool call its byte-exact input, and if no hook
observed that call, its span entirely.

A dropped **alias** is the one to know about, because it does not look like a
loss. That identifier stops resolving, so the parent is decided one rung further
down: if the work carries an ambient wardex span, the span arrives at confidence
**1.0 with no marker** — hanging off the enclosing session instead of the
sub-agent it belonged to. A sub-agent's subtree flattens and nothing in the data
says so. Only when there is no ambient span does it arrive marked
`unit_inferred_sole` (0.5) or `parent_unresolved`.

The evictions that DO reach your traces are counted as well, so you can see one
coming before it is a shape in your data:
`adapters.assembler.open_tool_table_full` and
`adapters.assembler.subagent_table_full` for the two `max_session_entries` sites
that emit, plus `adapters.assembler.tool_completion_after_evict` for the second
half of a call the first one closed.

Which number to raise depends on which counter moved.
`assembly._units.*` and `adapters.anthropic.server_table_full` are
`max_entries_per_unit`; every `adapters.assembler.*_table_full` is
`max_session_entries`. The two are separate fields — the core limits table
calls the first a generalization of the second, but raising it leaves the second
exactly where it was.

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
   ~~OpenAI Agents SDK~~ shipped; LangChain (non-graph runnables) next
5. Node/TS and Java SDKs

> PII masking caveats: `before_send_envelope` sees pre-masking data (masking runs inside
> the encoder), the Console transport prints raw (local debugging only), and
> non-UTF-8 binary payloads pass through unmasked.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

"Wardex" is a trademark of Wardex Labs.
