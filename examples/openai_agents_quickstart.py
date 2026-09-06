"""Two openai-agents agents, one ``wardex.init()``, one tree in your tracing backend.

A travel concierge checks the weather with a function tool and hands the
booking off to a second agent. The same conversation runs twice — once with
``Runner.run`` and once with ``Runner.run_streamed`` — so you can see that
both shapes produce the same tree. Nothing in the agent code knows wardex
exists; ``wardex.init()`` at the bottom is the only line that is not plain
openai-agents.

Setup, from a checkout of the repository, on Python 3.10 or newer (the full
walkthrough, including Phoenix and Langfuse, is in ``examples/README.md``)::

    pip install "wardex-sdk>=0.6.0b1" openai-agents
    export OPENAI_API_KEY=sk-...                           # the framework's own requirement
    export WARDEX_ENDPOINT=http://127.0.0.1:6006/v1/traces # a local Phoenix
    python examples/openai_agents_quickstart.py

What to expect, per run (span names exactly as wardex emits them)::

    invoke_workflow travel_concierge
    ├── invoke_agent concierge
    │   ├── chat gpt-4o-mini                   the model asks for the weather
    │   ├── execute_tool lookup_weather        gen_ai.tool.call.id joins it to that chat
    │   ├── chat gpt-4o-mini                   the model decides to hand off
    │   └── handoff concierge→booking_agent    a marker; the receiver is NOT nested here
    └── invoke_agent booking_agent             sibling, wardex.agent.parent=concierge
        └── chat gpt-4o-mini                   the booking is confirmed

Eight spans, one trace, twice: the second trace is the streamed run and its
root is ``invoke_workflow streamed_travel_concierge``, so the two are
tellable apart in a backend that truncates long names. The ``chat`` spans
are read from the wire — model, token usage and messages come from the HTTP
exchange itself — and the structure around them comes from the framework's
own tracing hooks. A real model may take a shorter path (skip the tool, or
answer without the handoff); the tree then simply has fewer branches.

Two optional environment variables belong to this script, not to wardex:

``EXAMPLE_NO_OPENAI_UPLOAD=1``
    openai-agents uploads its own record of every run to OpenAI's trace
    dashboard; with a key that dashboard refuses, the framework prints
    ``[non-fatal] Tracing client error 401`` at the end of the run. Setting
    this drops that upload, so nothing leaves the machine but the export to
    your backend. wardex never changes the framework's setting on its own.

``LANGFUSE_HOST`` (with ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY``)
    Export to Langfuse instead of ``WARDEX_ENDPOINT``. Langfuse authenticates
    OTLP with HTTP Basic auth, which ``WARDEX_API_KEY`` (a bearer token) cannot
    express, so this is the one backend that needs an explicit transport.
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys

import agents
from agents import Agent, RunConfig, Runner, function_tool

import wardex_sdk as wardex


@function_tool
def lookup_weather(city: str) -> str:
    """Return a short weather summary for a city."""
    # Canned on purpose: the example is about the trace, not the forecast.
    return f"{city}: 22°C and sunny for the next three days."


booking_agent = Agent(
    name="booking_agent",
    instructions=(
        "You book hotel stays. Confirm the city and the dates back to the "
        "traveller in one sentence and say the booking is done."
    ),
    model="gpt-4o-mini",
)

concierge = Agent(
    name="concierge",
    instructions=(
        "You are a travel concierge. First call lookup_weather for the city "
        "the traveller names. Then hand off to booking_agent so it can book "
        "the stay — do not book anything yourself."
    ),
    model="gpt-4o-mini",
    tools=[lookup_weather],
    handoffs=[booking_agent],
)

PROMPT = (
    "I'd like to spend next weekend in Lisbon. What's the weather like, "
    "and can you book me a hotel?"
)


async def main() -> None:
    # `workflow_name` names the root span; without it the framework's default,
    # `Agent workflow`, is what you would look for in the backend. The two runs
    # get names that differ at the front so a trace list that truncates long
    # names still tells them apart.
    result = await Runner.run(
        concierge, PROMPT, run_config=RunConfig(workflow_name="travel_concierge")
    )
    print("Runner.run:         ", result.final_output)

    streamed = Runner.run_streamed(
        concierge, PROMPT, run_config=RunConfig(workflow_name="streamed_travel_concierge")
    )
    async for _event in streamed.stream_events():
        pass  # draining the stream is what completes the run
    print("Runner.run_streamed:", streamed.final_output)


def init_wardex() -> None:
    """The Phoenix path is the bare ``wardex.init()``; Langfuse needs one transport."""
    host = os.environ.get("LANGFUSE_HOST")
    if not host:
        wardex.init()  # reads WARDEX_ENDPOINT; auto-detects the installed framework
        return

    from wardex_sdk.transport import OtlpHttpTransport

    # Checked rather than indexed: the host alone is a half-configured backend,
    # and a bare KeyError here reads as a bug in the example, not as a missing
    # variable in your shell.
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        sys.exit("LANGFUSE_HOST needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY")

    auth = base64.b64encode(f"{public}:{secret}".encode()).decode()
    wardex.init(
        transport=OtlpHttpTransport(
            endpoint=f"{host.rstrip('/')}/api/public/otel/v1/traces",
            headers={"Authorization": f"Basic {auth}", "x-langfuse-ingestion-version": "4"},
        )
    )


if __name__ == "__main__":
    if os.environ.get("EXAMPLE_NO_OPENAI_UPLOAD") == "1":
        # Must come BEFORE wardex.init(): afterwards this call would remove
        # wardex's own processor along with the framework's exporter.
        agents.set_trace_processors([])
    init_wardex()
    asyncio.run(main())
    wardex.close()  # optional — spans also flush every 5s and at exit
    print()
    print("Now open your backend. Phoenix: http://localhost:6006 → Tracing → project `default`.")
    print("It opens on the Spans tab; the top two rows are this run")
    print("(`invoke_workflow streamed_travel_...` above `invoke_workflow travel_...`).")
    print("Click one, then drag the drawer's left edge wider until the rows indent:")
    print("both agents, the tool call and the handoff.")
