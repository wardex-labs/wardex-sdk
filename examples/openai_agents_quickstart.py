"""Two openai-agents agents, one ``wardex.init()``, one tree in your tracing backend.

A travel concierge checks the weather with a function tool and hands the
booking off to a second agent. The same conversation runs twice — once with
``Runner.run`` and once with ``Runner.run_streamed`` — so you can see that
both shapes produce the same tree. Nothing in the agent code knows wardex
exists; the single ``wardex.init()`` at the bottom is the only line that is
not plain openai-agents.

Setup (the full walkthrough is in ``examples/README.md``)::

    pip install wardex-sdk openai-agents
    export OPENAI_API_KEY=sk-...                          # the framework's own requirement
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

Eight spans, one trace, twice. The ``chat`` spans are read from the wire —
model, token usage and messages come from the HTTP exchange itself — and
the structure around them comes from the framework's own tracing hooks. A
real model may take a shorter path (skip the tool, or answer without the
handoff); the tree then simply has fewer branches.
"""

from __future__ import annotations

import asyncio

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
    # `Agent workflow`, is what you would look for in the backend.
    config = RunConfig(workflow_name="travel_concierge")

    result = await Runner.run(concierge, PROMPT, run_config=config)
    print("Runner.run:         ", result.final_output)

    streamed = Runner.run_streamed(concierge, PROMPT, run_config=config)
    async for _event in streamed.stream_events():
        pass  # draining the stream is what completes the run
    print("Runner.run_streamed:", streamed.final_output)


if __name__ == "__main__":
    wardex.init()  # reads WARDEX_ENDPOINT; auto-detects the installed framework
    asyncio.run(main())
    wardex.close()  # optional — spans also flush every 5s and at exit
    print()
    print("Now open your backend — Phoenix: http://localhost:6006 → Traces — and pick a trace")
    print("named `invoke_workflow travel_concierge`; expand it to see both agents, the tool call")
    print("and the handoff. There are two traces: one per Runner.run / Runner.run_streamed.")
