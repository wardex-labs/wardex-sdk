# Examples

Runnable scripts that show what wardex captures from a framework with no
changes to the agent code. Each one is a plain script: install, set an
endpoint, run, open the backend.

| Script | Framework | What you see |
|---|---|---|
| [`openai_agents_quickstart.py`](openai_agents_quickstart.py) | OpenAI Agents SDK (`openai-agents`) | two agents, a function tool, a handoff — one tree per run, for `Runner.run` and `Runner.run_streamed` |

## openai-agents → Phoenix

[Phoenix](https://github.com/Arize-ai/phoenix) is an open-source tracing UI
that accepts OTLP/HTTP directly, so the whole path is one container, two
packages and one environment variable. The steps below take about five
minutes on a machine that already has Docker and Python.

**1. Start Phoenix.**

```bash
docker run -d --name wardex-phoenix -p 6006:6006 arizephoenix/phoenix:latest
```

**2. Install the framework and wardex.**

```bash
python -m venv .venv && source .venv/bin/activate
pip install wardex-sdk openai-agents
```

**3. Point wardex at Phoenix, give the framework its key, run.**

```bash
export OPENAI_API_KEY=sk-...                            # the framework's own requirement
export WARDEX_ENDPOINT=http://127.0.0.1:6006/v1/traces  # the only wardex variable
python examples/openai_agents_quickstart.py
```

`WARDEX_ENDPOINT` is the full OTLP traces URL. Phoenix listens on
`/v1/traces`, and wardex would append that path to a bare
`http://127.0.0.1:6006` anyway, so either spelling works. Nothing else is
configured: the openai-agents adapter is auto-detected because the
`openai-agents` distribution is installed, and PII masking is on by default,
so a real run never sends an email address or an API key to the backend.

**4. Look at the tree.**

Open <http://localhost:6006>, click **Traces** in the left sidebar, and pick
either of the two traces named `invoke_workflow travel_concierge` (one per
`Runner.run` / `Runner.run_streamed`). Expanding it shows:

```
invoke_workflow travel_concierge
├── invoke_agent concierge                       AGENT
│   ├── chat gpt-4o-mini                         LLM   — messages, model, tokens
│   ├── execute_tool lookup_weather              TOOL  — gen_ai.tool.call.id, arguments, result
│   ├── chat gpt-4o-mini                         LLM
│   └── handoff concierge→booking_agent          marker, receiver named in gen_ai.agent.name
└── invoke_agent booking_agent                   AGENT — wardex.agent.parent=concierge
    └── chat gpt-4o-mini                         LLM
```

The receiving agent is the sender's **sibling**, not its child: a chain of
five handoffs stays one level deep, and `wardex.agent.parent` plus a
`handoff_from` span link carry who sent whom. Phoenix assigns the AGENT /
LLM / TOOL badges itself from the OpenTelemetry `gen_ai.*` attributes;
`invoke_workflow` and `handoff` have no OpenInference counterpart and show
without a badge. Click a `chat` span to see the request and response
messages and the token counts read from the wire, and a `execute_tool` span
to see the arguments and the tool's return value.

The framework's own trace upload to `api.openai.com/v1/traces/ingest` keeps
running as it always did; wardex neither replaces it nor captures it. If
you would rather not upload the run record to OpenAI, put
`agents.set_trace_processors([])` **before** `wardex.init()` — the other
order removes wardex's processor as well.

## openai-agents → Langfuse

Langfuse also accepts OTLP/HTTP, on `/api/public/otel/v1/traces`, but it
authenticates with a project's public/secret key pair as HTTP Basic auth
rather than a bearer token, and `WARDEX_ENDPOINT` + `WARDEX_API_KEY` only
know how to send `Authorization: Bearer …`. So this path needs one explicit
transport in place of the bare `wardex.init()` — the agent code is still
untouched.

Step 2 above stays the same; `WARDEX_ENDPOINT` is not used on this path.
Replace the `wardex.init()` call in the example with:

```python
import base64
import os

import wardex_sdk as wardex
from wardex_sdk.transport import OtlpHttpTransport

base = os.environ["LANGFUSE_HOST"]  # e.g. http://127.0.0.1:3000 or your cloud region's host
auth = base64.b64encode(
    f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
).decode()
wardex.init(
    transport=OtlpHttpTransport(
        endpoint=f"{base}/api/public/otel/v1/traces",
        headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
    )
)
```

This is the shape wardex's own Langfuse end-to-end driver uses against a
self-hosted `langfuse/langfuse` v4 docker-compose stack. For the hostname
of a cloud region, or a different self-hosted version, see Langfuse's own
OpenTelemetry ingestion docs — nothing in this repo verifies those.

Once running, open your Langfuse project's traces list; the run appears
under the name `invoke_workflow travel_concierge` with the same tree.
Langfuse's `totalTokens` column reads 0 because wardex ships no
`gen_ai.usage.total_tokens` (the key does not exist in the OpenTelemetry
registry); `promptTokens`, `completionTokens` and the cost figures are exact.

## Known limitations

These are the two cases where the tree above is **not** what you get, and
what wardex says in each case:

- **Responses over WebSocket** (`use_responses_websocket=True` on the
  framework's OpenAI provider). Every LLM call rides one
  `wss://…/v1/responses` connection, and wardex does not parse Responses
  events inside WebSocket frames. You get the agent, tool and handoff spans
  as above, but where the `chat` spans would be there is one
  `WS /v1/responses` span per connection, carrying the marker
  `ws_llm_semantics_unread` — no model, no tokens, no messages. Switch the
  framework back to its default HTTP transport to get the `chat` spans.
- **Framework tracing disabled** (`OPENAI_AGENTS_DISABLE_TRACING=1` or
  `agents.set_tracing_disabled(True)`). The adapter hooks the framework's own
  tracing, so with it off there are no `invoke_workflow`, `invoke_agent`,
  `execute_tool` or `handoff` spans — only the `chat` spans read from the
  wire, unparented. wardex prints exactly one line on stderr at
  `wardex.init()`:

  ```
  [wardex] openai-agents tracing is disabled, so wardex will show only the LLM calls its interceptor captures: no agent, handoff, tool or guardrail spans. To get them without sending anything to OpenAI, put these two lines BEFORE wardex.init(): agents.set_tracing_disabled(False); agents.set_trace_processors([]). Calling set_trace_processors after wardex.init() removes wardex's processor as well.
  ```

  wardex never flips the framework's setting for you; the two lines in the
  message are yours to add.
