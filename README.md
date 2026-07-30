# wardex-sdk

[![PyPI](https://img.shields.io/pypi/v/wardex-sdk)](https://pypi.org/project/wardex-sdk/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/wardex-labs/wardex-sdk/blob/main/LICENSE)

Open-source observability SDK for AI agents — zero-instrumentation capture,
OpenTelemetry-native.

> ⚠️ **Beta.** PII masking is on by default (see below), but the
> SDK is still early: review the caveats below before sending sensitive data
> through it.

## Install

```bash
pip install wardex-sdk
```

## Quickstart

```python
import wardex_sdk as wardex
from wardex_sdk import OtlpHttpTransport

wardex.init(
    transport=OtlpHttpTransport(endpoint="https://<your-collector>/v1/traces"),
    intercept=True,  # zero-instrumentation capture of LLM calls
)

# your app code — OpenAI/Anthropic calls are captured automatically

wardex.close()  # optional — spans auto-flush every 5s, on buffer threshold, and at exit
```

## Status

**Works today**
- Zero-instrumentation capture of LLM HTTP calls (OpenAI, Anthropic) over
  `https`, cleartext `http`, and h2c
- `gen_ai` semantics: model, tokens, parameters, finish reasons, input/output messages
- Transport metrics (TCP/TLS timing, TTFT), gRPC (grpclib), WebSocket (`wss`), MCP stdio
- Export to any OpenTelemetry backend via `OtlpHttpTransport`
- Manual span decorators: `@workflow` / `@agent` / `@task` / `@tool` / `@span`
- PII masking on by default: emails, phone numbers, credit cards (Luhn-verified),
  US SSNs, IP addresses, bank routing numbers, IBANs, and API-key/token secrets
  are masked before anything leaves the process (`pii_mode=PIIMode.OFF` to disable,
  `pii_disabled_categories={PIICategory.IP_ADDRESS}` for per-category opt-out)
- Background batching: automatic flush every 5s / on buffer threshold /
  at exit and on SIGINT/SIGTERM (chained; opt out with `flush_on_signals=False`)
- Framework adapter: Anthropic Agent SDK (`claude_agent_sdk`) — auto-detected,
  zero-instrumentation `invoke_agent`/`chat` spans with tool-call correlation

**Not yet (see Roadmap)**
- Framework adapters for LangGraph and OpenAI Agents SDK
- Node/TS and Java SDKs

**Notes**
- After `os.fork()` the worker respawns lazily in the child on first capture;
  spans buffered before the fork may be sent by both processes (duplicates
  are possible; a fork landing mid-export can also strand the child's
  pre-fork buffer — re-init in the child for a clean slate). Under uWSGI
  enable threads (`--enable-threads`).

## Distributed tracing

Trace context propagation is **opt-in** — a plain `wardex.init(...)` never
touches your outbound requests or headers. Turn it on with:

```python
wardex.init(
    transport=OtlpHttpTransport(endpoint="https://<your-collector>/v1/traces"),
    intercept=True,
    propagate_trace=True,  # inject W3C headers on outbound calls
    propagate_targets=[
        "api.internal.example.com",
        "*.svc.cluster.local",
    ],  # optional glob allowlist; default None = all hosts
)
```

With `propagate_trace=True`, outbound calls made through httpx (sync + async),
requests, or aiohttp get a `traceparent` (and `tracestate`, if one was
received) header attached automatically, as long as an active trace context
exists and the request doesn't already carry a `traceparent`. **If
`propagate_targets` is left unset, the trace ID is sent to every host you
call — including third-party LLM providers.** Set it to an allowlist of glob
patterns to scope injection to your own services.

### Joining an inbound trace

Drop the middleware in front of your app to join whatever trace the caller
started:

```python
# ASGI (FastAPI, Starlette, Django ASGI)
app.add_middleware(wardex.WardexMiddleware)

# WSGI (Flask, Django WSGI)
app.wsgi_app = wardex.WardexWSGIMiddleware(app.wsgi_app)
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

`with wardex.continue_from_otel():` is a one-line alternative to
`continue_trace()` for code that already runs under an active OpenTelemetry
span — it adopts that span as the remote parent for the duration of the
`with` block (no-op if `opentelemetry` isn't installed or there's no active
span). Like `continue_trace()`, it is a context manager and must be entered
with `with`.

### Propagating into threads

`asyncio` tasks inherit the current trace context automatically; threads do
not. Wrap the target with `wardex.run_in_context()` at the point where you
still have the right context:

```python
thread = threading.Thread(target=wardex.run_in_context(worker_fn), args=(...,))
thread.start()
```

### `capture_mode`: what gets captured without an active span

`capture_mode` defaults to `"agent"`: LLM-semantic traffic (recognized
`gen_ai` calls, MCP stdio) is always captured, but generic HTTP/gRPC/WS
traffic is only captured while it happens inside an active *local* wardex
span (a `traceparent` received from an upstream caller doesn't count on its
own — this keeps a service mesh stamping every request with a traceparent
from reviving the pre-Phase-4 "capture everything" noise).

This means a bare, unwrapped call to an LLM provider wardex doesn't
recognize (or a WS-based provider such as OpenAI Realtime, which carries no
parseable semantics) can be silently dropped if it isn't inside a local
span. Wrap it with `@wardex.workflow` (or any of the span decorators), or set
`capture_mode=wardex.CaptureMode.ALL` to restore the previous
capture-everything behavior:

```python
wardex.init(..., capture_mode=wardex.CaptureMode.ALL)
```

Plaintext hosts you've explicitly named via `intercept_hosts` are always
captured regardless of `capture_mode` — a targeted allowlist entry is a
stronger opt-in than the default policy.

## Resource limits

Every resource bound in the SDK — body size caps, buffer sizes, connection
and session tracking — is configurable, but the defaults suit most
workloads and most users never need to touch this. The body cap is set
above the Anthropic Messages API's request size ceiling, so a request the
API itself accepts is never truncated by capture.

```python
import wardex_sdk as wardex
from wardex_sdk import CaptureLimits

wardex.init(
    limits=CaptureLimits(
        max_body_bytes=64 * 1024 * 1024,  # larger multimodal payloads
        max_buffer_bytes=16 * 1024 * 1024,  # tighter memory budget
    )
)
```

Two exceptions are inert today, so setting them has no effect:
`replay_buffer_size` (nothing reads it yet) and `zstd_level` (read only by the
envelope encoder, which no live export path calls — the OTLP exporter neither
takes limits nor compresses).

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
internally instead — `wardex_sdk.assembly.counters.snapshot()` reports them
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

## Roadmap

1. ~~PII masking (pre-send safety)~~ — shipped
2. ~~Batching & lifecycle (background worker, at-exit/periodic flush, concurrency)~~ — shipped
3. ~~Distributed propagation (W3C)~~ — shipped
4. Framework adapters — ~~Anthropic Agent SDK~~ shipped; LangGraph and OpenAI
   Agents SDK next
5. Node/TS and Java SDKs

> PII masking caveats: `before_send` sees pre-masking data (masking runs inside
> the encoder), the Console transport prints raw (local debugging only), and
> non-UTF-8 binary payloads pass through unmasked.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

"Wardex" is a trademark of Wardex Labs.
