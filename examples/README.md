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
packages and one environment variable.

**What you need.** Docker, Python 3.10 or newer (`python3 --version`), an
OpenAI API key, a checkout of this repository — the example script is not
part of the wheel — and, only until wardex-sdk 0.6.0b1 is on PyPI, a
[Rust toolchain](https://rustup.rs) (`cargo --version`), because the
install in step 2 then builds the native module from the checkout instead
of downloading a prebuilt wheel. The walkthrough assumes a fresh clone:

```bash
git clone https://github.com/wardex-labs/wardex-sdk
cd wardex-sdk
```

**How long it takes.** Measured on 2026-09-06 by following exactly the
steps below from a fresh clone and virtualenv, with the Phoenix image
already pulled and pip's and cargo's caches warm: 41 seconds in the
terminal, 34 of them the from-checkout install under step 2, plus the four
clicks and one drag in the UI — about 25 seconds at human pace, 5 when
scripted — so a little over a minute to the tree on screen. Two things
happen only once and are not in that number: the first `docker run` pulls
the Phoenix image (1.1 GB — a few minutes on a typical connection, and the
longest step of a first setup), and a cold Rust build on the from-checkout
path takes minutes rather than 34 seconds; the PyPI wheel skips the build
altogether.

**1. Start Phoenix.**

```bash
docker run -d --name wardex-phoenix -p 6006:6006 arizephoenix/phoenix:latest
```

Phoenix is ready when <http://localhost:6006> opens in a browser, about
fifteen seconds after the image is local. Before moving on, check that the
port is really published to the container you expect:

```bash
docker ps --filter name=phoenix --format '{{.Names}}  {{.Status}}  {{.Ports}}'
# wardex-phoenix  Up 20 seconds  0.0.0.0:6006->6006/tcp, [::]:6006->6006/tcp
```

The `0.0.0.0:6006->6006/tcp` part is the one that matters: it is what
makes `localhost:6006` reach *this* container. Two things can go wrong
here, and they compose badly, so read both before typing:

- `docker run` fails with `Bind for 0.0.0.0:6006 failed: port is already
  allocated`: something else — usually another Phoenix — already listens
  on 6006, and `docker ps` shows which. Either use that one as it is (it
  is a Phoenix; the URL below is the same), or publish this one on another
  port (`-p 6007:6006`) and replace `6006` with `6007` in every URL below.
  The failed `docker run` leaves a container named `wardex-phoenix` behind
  in `Created` state, which is why the *same* command fails a second time
  with `The container name "/wardex-phoenix" is already in use` — remove it
  with `docker rm wardex-phoenix` before running it again.
- **Do not `docker start` that leftover container.** `docker start
  wardex-phoenix` reports success and Phoenix logs "up and running", but
  `docker ps` shows the port as a bare `6006/tcp` with no `0.0.0.0:6006->`
  in front of it — and it stays that way even after the other listener is
  gone (verified: a container whose first start lost the bind never
  publishes the port again). Nothing on `localhost:6006` reaches it, the
  export goes to whatever does hold the port, and no error says so. Remove
  it and re-create it with the `docker run` line above. `docker start` is
  the right verb only for a container you stopped yourself with `docker
  stop wardex-phoenix`; that one comes back with its mapping, and `docker
  ps` is the way to tell the two apart.

**2. Install the framework and wardex.**

```bash
python3 -m venv .venv-quickstart && source .venv-quickstart/bin/activate
pip install "wardex-sdk>=0.6.0b1" openai-agents
```

(The virtualenv is named `.venv-quickstart` rather than `.venv` so that a
contributor running this inside an existing working copy does not replace
the repository's own `uv`-managed `.venv`; it is git-ignored.)

The openai-agents adapter ships in wardex-sdk 0.6.0b1; the wheel is
prebuilt for macOS, Linux and Windows, so there is nothing to compile.
Until that version is on PyPI, `pip` answers `No matching distribution
found for wardex-sdk>=0.6.0b1`, and the install is instead built from the
checkout you are standing in:

```bash
pip install ./sdks/python openai-agents   # needs cargo on the PATH; ~20 s warm, minutes cold
```

That build reports the version number the checkout declares in
`sdks/python/pyproject.toml` — 0.5.0b1 today, because the number is bumped
by the release, not by the commit that adds the adapter. So `pip show
wardex-sdk` printing something *below* the 0.6.0b1 you were just told to
require is expected, not a wrong package: the adapter is in the checkout,
and the next step proves it.

**3. Point wardex at Phoenix, give the framework its key, run.**

```bash
export OPENAI_API_KEY=sk-...                            # the framework's own requirement
export WARDEX_ENDPOINT=http://127.0.0.1:6006/v1/traces  # the only wardex variable
python examples/openai_agents_quickstart.py
```

`WARDEX_ENDPOINT` is the full OTLP traces URL; Phoenix listens on
`/v1/traces`. (A bare `http://127.0.0.1:6006` also works — an endpoint
with no path gets `/v1/traces` appended, the rule stated under
*Environment variables* in the README.) Nothing else is configured: the
openai-agents adapter is auto-detected because the `openai-agents`
distribution is installed, and PII masking is on by default, so a real run
never sends an email address or an API key to the backend.

The script prints the two final answers and then where to click. If the
very last line is

```
[non-fatal] Tracing client error 401. Response data is redacted.
```

that is the framework, not wardex, and not a failure of this quickstart:
openai-agents uploads its own record of every run to OpenAI's trace
dashboard at `api.openai.com`, and a key that dashboard refuses gets a 401
the framework itself labels non-fatal. wardex exported to Phoenix, not to
OpenAI, so the tree is complete either way. To keep the run record off
OpenAI altogether, run with `EXAMPLE_NO_OPENAI_UPLOAD=1`; the script then
calls `agents.set_trace_processors([])` **before** `wardex.init()`, which is
the order that matters — after it, the same call removes wardex's processor
too. (That switch belongs to the example script, not to wardex; wardex never
changes the framework's settings on its own.)

**4. Look at the tree.**

Open <http://localhost:6006>. **Tracing** in the left sidebar lists
projects; open **default** — Phoenix files everything that arrives without a
project name there. The project opens on its **Spans** tab, which lists
root spans newest first; the **Traces** tab next to it lists the same rows
as traces. Either way the top two rows are this run: `invoke_workflow
stre…` (the `Runner.run_streamed` run, which finished last) above
`invoke_workflow trav…` (the `Runner.run` run). The name column clips
long names, which is why the two workflow names differ at the front; older
rows, if any, are earlier runs against the same Phoenix.

Click `invoke_workflow trav…`. The trace opens in a drawer with all eight
spans already expanded — but at the drawer's default width Phoenix draws
them as a **flat list**, without the indentation that shows who is inside
whom. Drag the drawer's left edge to the left until the span list is about
half the window wide; the indentation and the connector lines appear, and
the list reads:

```
invoke_workflow travel_concierge                 unknown
├── invoke_agent concierge                       AGENT
│   ├── chat gpt-4o-mini                         LLM   — messages, model, tokens
│   ├── execute_tool lookup_weather              TOOL  — gen_ai.tool.call.id, arguments, result
│   ├── chat gpt-4o-mini                         LLM
│   └── handoff concierge→booking_agent          unknown — receiver named in gen_ai.agent.name
└── invoke_agent booking_agent                   AGENT — wardex.agent.parent=concierge
    └── chat gpt-4o-mini                         LLM
```

The receiving agent is the sender's **sibling**, not its child: a chain of
five handoffs stays one level deep, and `wardex.agent.parent` plus a
`handoff_from` span link carry who sent whom. The chip on each row is
Phoenix's: it assigns AGENT / LLM / TOOL from the OpenTelemetry `gen_ai.*`
attributes itself, and `invoke_workflow` and `handoff`, which have no
OpenInference kind, get its chip for "no kind", which reads `unknown`.
That is a label, not an error. Click a `chat` span to see the request and
response messages and the token counts read from the wire (the number next
to each `chat` row is Phoenix's total-token chip), and the `execute_tool` span to
see the arguments and the tool's return value.

## openai-agents → Langfuse

Langfuse also accepts OTLP/HTTP, on `/api/public/otel/v1/traces`, but it
authenticates with a project's public/secret key pair as HTTP Basic auth
rather than a bearer token, and `WARDEX_ENDPOINT` + `WARDEX_API_KEY` only
know how to send `Authorization: Bearer …`. So this backend needs one
explicit transport, and the example script builds it when it sees
`LANGFUSE_HOST` — the agent code is still untouched, and nothing in the
script is edited.

Steps 2 and 3 above stay the same except for the variables: in place of
`WARDEX_ENDPOINT`, set the three the script reads and run it again.

```bash
export OPENAI_API_KEY=sk-...
export LANGFUSE_HOST=http://127.0.0.1:3000   # or your cloud region's host
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
python examples/openai_agents_quickstart.py
```

What the script does with them, if you want the same thing in your own
`wardex.init()`:

```python
import base64
import os

import wardex_sdk as wardex
from wardex_sdk.transport import OtlpHttpTransport

base = os.environ["LANGFUSE_HOST"].rstrip("/")
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

Once running, open your Langfuse project's traces list; the two runs
appear under the names `invoke_workflow travel_concierge` and
`invoke_workflow streamed_travel_concierge` with the same tree as above.
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
