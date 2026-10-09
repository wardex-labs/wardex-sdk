# wardex-sdk

[![PyPI](https://img.shields.io/pypi/v/wardex-sdk)](https://pypi.org/project/wardex-sdk/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/wardex-labs/wardex-sdk/blob/main/LICENSE)

Open-source observability SDK for AI agents. Install it, call `init()`, and
your agent's model calls are recorded as OpenTelemetry traces, with no changes
to your agent code. On the OpenAI Agents SDK, LangGraph or the Claude Agent
SDK, its agents, tool calls and handoffs are recorded as well.

> **Beta.** Interfaces may change between 0.x releases; the rules are in
> [VERSIONING.md](https://github.com/wardex-labs/wardex-sdk/blob/main/VERSIONING.md).
> PII masking is on by default. Read what it does not catch, under
> [Data safety](#data-safety), before you send sensitive data through it.

## Why agent observability

A conventional service fails loudly: an exception, a 500, a stack trace. An
agent usually fails quietly. A run is a chain of model calls, tool calls and
handoffs between agents, and when one step starts answering differently, the
process still exits cleanly, every HTTP status is 200, and the error tracker
records nothing. The damage surfaces later and somewhere else: a retry loop
that triples the token bill, a tool that returns an empty result the next step
treats as an answer, a handoff to the wrong agent, an answer that has drifted
since the last model upgrade.

Teams usually learn about these failures from a customer complaint or an
invoice, then spend days working out when the behaviour changed and which step
changed it. Every model, prompt or tool change carries the same uncertainty.

Answering "what did the agent actually do, and since when" takes a record with
three properties:

- **Complete.** Every model call, including calls made by libraries you did
  not write, calls that failed, and calls made outside any framework.
- **Causal.** Which agent requested which tool call and which model call,
  across `asyncio` tasks, threads and services.
- **Explicit about gaps.** When part of a run could not be observed, such as
  a streamed response that never reported its token usage or a parent the
  SDK never saw, the record says so, so an empty field is never mistaken for
  a zero.

wardex-sdk is built to produce that record.

## How it works

- **It reads model traffic where it crosses the socket.** Inside your process,
  the SDK observes the traffic your code produces over HTTP/1.1, HTTP/2,
  WebSocket, gRPC (grpclib) and MCP stdio. Calls to recognised model providers
  are always recorded; other traffic is recorded while it runs inside a wardex
  span. For OpenAI Chat Completions, Responses and Embeddings and for Anthropic
  Messages, model, messages, parameters and token usage come from the bytes
  the provider actually received and returned, streaming included. Two cases
  are read another way: the Claude Agent SDK makes its model calls from a CLI
  subprocess, so its adapter rebuilds them from that process's output, and
  Responses over WebSocket is recorded as a marked connection whose calls are
  left unread.
- **It rebuilds the run tree from in-process context.** Parent links come from
  context propagation inside your process, so a model call made deep inside a
  tool lands under that tool even when no framework hook reports it. Each
  link records how it was established and with what confidence, and a link
  the SDK cannot back carries a marker instead.
- **It marks what it could not see.** An incomplete span carries a named
  marker in `wardex.limitations`, for example `stream_usage_unavailable` when
  a streamed response carried no usage block, so its token counts are absent.
  A loss that cannot ride on a span is counted in-process, and export, buffer
  and parser losses are also reported once on the `wardex_sdk` logger.
- **It masks before export.** Secrets and personal data are masked inside your
  process, before a byte is sent, and each masked span records what was
  replaced and by which rule.
- **It speaks OpenTelemetry.** Spans follow the OpenTelemetry `gen_ai`
  semantic conventions and export over OTLP/HTTP to any compatible backend, or
  to the wardex receiver.
- **It stays out of your way.** The core is Rust, the wheel has no runtime
  Python dependencies, parsing runs on the SDK's own worker threads, and
  nothing the SDK does after `init()` raises into your code.

## Next to the tools you are evaluating

**Langfuse, LangSmith, Arize Phoenix.** These platforms store, search and
evaluate traces. Their integrations usually live in your code as a client
wrapper, a decorator, an instrumentation package or a framework callback, and
a trace contains what that integration chose to report. wardex-sdk is a
collector that exports standard OTLP. Langfuse and Phoenix are tested
destinations, so a team can keep the platform it already uses and change only
how the data is collected.

**LangGraph and other agent frameworks.** wardex records LangGraph runs
without callbacks or `LangChainTracer`: one span per graph run, one per node,
one per tool call a `ToolNode` dispatches, with every model and HTTP call
underneath placed by context. The OpenAI Agents SDK and the Claude Agent SDK have adapters as well,
and all three install themselves when `init()` finds the framework. On a
framework with no adapter, model calls to recognised providers are still
captured with their token usage and model ids, and a few decorators add the
structure.

**Building it in-house.** A wrapper around the model client is quick to write.
The work that follows is what this SDK already does: reading streamed responses and
their usage, keeping parent links correct across `asyncio.gather` and shared
HTTP/2 connections, carrying context into worker threads, masking credentials in URLs, JSON bodies and tool
arguments before they leave the process, bounding memory under load, and
reporting explicitly when any of it failed. The SDK is Apache-2.0, so every
one of those decisions is readable in the source.

## Install

```bash
pip install wardex-sdk
```

Python 3.10 or newer. Prebuilt wheels cover Linux with glibc (x86_64,
aarch64), macOS (x86_64, arm64) and Windows (x86_64), so nothing compiles on
install. Other platforms, such as Alpine (musl), have no wheel yet.

## Quickstart

Point the SDK at any OpenTelemetry backend that accepts OTLP/HTTP (Langfuse,
Phoenix, an OTel Collector):

```bash
export WARDEX_ENDPOINT=http://127.0.0.1:6006/v1/traces
```

```python
import wardex_sdk as wardex

wardex.init()

# Your agent code, unchanged.

wardex.close()  # optional: spans also flush every 5 seconds and at exit
```

A host that already exports OTLP needs no new variable:
`OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` are read as
well. Capture starts at `init()` (`intercept=False` turns it off). With no
destination configured, the SDK sends nothing and says so once on stderr.

Where the data goes is decided by what you configured, and nothing else:

| You set | Data goes to |
|---|---|
| `endpoint` (`WARDEX_ENDPOINT`, or the OTel endpoint variables) | that OTLP/HTTP collector, authenticated by `headers` / `OTEL_EXPORTER_OTLP_HEADERS` |
| `api_key` (`WARDEX_API_KEY`) | the hosted wardex receiver the key's region names |
| `api_key` + `base_url` | a self-hosted wardex receiver at `base_url` |
| `transport=` | that transport, always |
| none of these | nowhere, said once on stderr |

When two settings name different destinations, the one that loses is announced
with a `WardexConfigWarning`. The `api_key` rows are on the main branch and
are not available in the current release.

## See a real trace

The [openai-agents example](https://github.com/wardex-labs/wardex-sdk/blob/main/examples/README.md)
runs two agents, a function tool and a handoff against a local Phoenix,
started with one `docker run` as its walkthrough shows. The script makes six
short `gpt-4o-mini` calls against your OpenAI key.

```bash
git clone https://github.com/wardex-labs/wardex-sdk && cd wardex-sdk
python3 -m venv .venv-quickstart && source .venv-quickstart/bin/activate
pip install "wardex-sdk>=0.6.0b1" openai-agents
export OPENAI_API_KEY=sk-...                             # the framework's own requirement
export WARDEX_ENDPOINT=http://127.0.0.1:6006/v1/traces   # a local Phoenix
python examples/openai_agents_quickstart.py
```

<!-- Absolute URL on purpose: this file is also the PyPI long description
     (readme = "README.md" in sdks/python/pyproject.toml), where a relative
     path renders as a broken link. The trade-off is that the image and the
     docs links 404 on branches and in PR views until they are on main. -->
![Phoenix showing one openai-agents run: invoke_workflow travel_concierge → invoke_agent concierge (chat, execute_tool lookup_weather, chat, handoff concierge→booking_agent) and its sibling invoke_agent booking_agent (chat)](https://raw.githubusercontent.com/wardex-labs/wardex-sdk/main/examples/openai-agents-phoenix.png)

No decorator, callback or processor was added to the agent code to get this
tree.

## Frameworks

| Framework | What becomes a span |
|---|---|
| OpenAI Agents SDK (`openai-agents>=0.22,<0.23`) | each run, agent, function tool, handoff and guardrail; model calls read from the wire |
| LangGraph (`langgraph>=1.2`) | each graph run, node and `ToolNode` tool call, including subgraphs and the functional API |
| Claude Agent SDK (`claude_agent_sdk`) | agent turns and model calls, with tool calls correlated |
| Codex CLI (`codex exec`, run as a subprocess) | each run, with its prompt, answer, tool calls and the turn's token usage; with `CodexExecConfig(otel_bridge=True)`, one span per model call with its model and timing |
| Any other framework, or your own loop | model calls with model, messages and token usage; decorators add the structure |

Adapters are detected automatically when the framework is installed (the
Codex adapter when `codex` is on `PATH`); `AdaptersConfig` selects and
configures them.

The Codex adapter only reads what your process already exchanges with
`codex exec` — the prompt on stdin and the `--json` events on stdout — and
never changes the command. That stream names neither the model nor how many
times it was asked, so the run carries the turn's total and says so. With
`otel_bridge=True` the adapter adds a trace exporter override and a
`TRACEPARENT` to each `codex exec` and reads each model call from Codex's own
traces; it leaves alone a run whose command, environment or Codex config
already sets up telemetry. Verified against `codex-cli` 0.160.0.

## Adding structure to your own code

```python
import wardex_sdk as wardex


@wardex.agent(name="researcher")
def research(question: str): ...


@wardex.tool(name="search")
def search(query: str): ...


with wardex.conversation("support-chat", id="session-123"):
    research("Which plan includes SSO?")
```

The decorators produce `invoke_agent` and `execute_tool` spans (`@workflow`
and `@step` produce `invoke_workflow` and `execute_step`). Every span inside a
`conversation()` block, model calls included, carries
`gen_ai.conversation.id`. An exception that leaves a span is recorded on it,
and your code still receives the same exception. Hand-started threads need
`wardex.bind_context(fn)` to carry the context; `asyncio` tasks inherit it.

## Data safety

Masking is on by default and runs in your process before export. It covers
e-mail addresses, phone numbers, card numbers, US SSNs, IP addresses, IBANs,
bank routing numbers, credential formats (`sk-…`, `AKIA…`, `ghp_…`, JWTs,
PEM private keys and more), and any value passed under a secret-looking name
such as `api_key` or `password`, whether it sits in a URL, a JSON body, a form
field or a span attribute. Argument names stay readable, and each masked span
records how many values were replaced and by which rule.

```python
from wardex_sdk import PIIConfig

wardex.init(
    pii=PIIConfig(
        extra_secret_names={"x_corp_auth"},  # also masks xCorpAuth, X-Corp-Auth
        reveal_names={"page_token"},  # kept, unless the value looks like a credential
    ),
)
```

What masking does not catch:

- a secret under an ordinary name with no recognisable shape, such as
  `{"value": "hunter2"}`;
- `name: value` in prose or YAML, and XML such as `<password>…</password>`;
- a credential inside a URL path, such as `/bot<token>/sendMessage`;
- payloads that are not text, and bodies compressed with anything other than
  gzip or zlib;
- anything read before the encoder runs: `before_send_envelope`,
  `ConsoleTransport` and a transport that serializes envelopes itself all see
  data before masking.

## Status

Available today: the Python SDK, the three framework adapters above, OTLP
export to any OpenTelemetry backend, and masking on by default. Not yet: an
adapter for plain LangChain LCEL chains, and SDKs for Node/TypeScript and Java.

`wardex_sdk.__version__` reports the installed version. Version semantics and
the deprecation policy are in
[VERSIONING.md](https://github.com/wardex-labs/wardex-sdk/blob/main/VERSIONING.md);
release notes are in
[CHANGELOG.md](https://github.com/wardex-labs/wardex-sdk/blob/main/CHANGELOG.md).
To build from source or contribute, see
[CONTRIBUTING.md](https://github.com/wardex-labs/wardex-sdk/blob/main/CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](https://github.com/wardex-labs/wardex-sdk/blob/main/LICENSE)
and [NOTICE](https://github.com/wardex-labs/wardex-sdk/blob/main/NOTICE).

"Wardex" is a trademark of Wardex Labs.
