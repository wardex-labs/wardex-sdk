"""What the OpenAI Agents adapter said about a conversation survives the real encoder.

A real run, encoded as one flush would send it, decoded and held to the
contract in `_adapter_envelopes.py`.
"""

from __future__ import annotations

import asyncio

from agents import RunConfig, Runner

import wardex_sdk as wardex
from _adapter_envelopes import SCENARIOS, deliver
from test_openai_agents_adapter import _init
from test_openai_agents_wire import (
    _agents,
    agents_env,  # noqa: F401 - a fixture, usable here only because it is imported
    fake_openai,  # noqa: F401 - `agents_env` depends on it
)
from wardex_sdk import _hub
from wardex_sdk._types import Envelope
from wardex_sdk.transport import _codec


def test_fresh_openai_agents_envelope(agents_env):  # noqa: F811
    transport = _init()
    try:
        config = RunConfig(workflow_name="wf", group_id=SCENARIOS["openai_agents"][0])
        assert asyncio.run(Runner.run(_agents(), "hi", run_config=config)).final_output == "done"
        _hub.get_client()._settle()
    finally:
        wardex.close()
    # One body, as one flush would send it: the batches the run produced, under
    # the header the client itself wrote.
    envelopes = transport.envelopes
    merged = Envelope(
        header=envelopes[0].header, spans=tuple(s for env in envelopes for s in env.spans)
    )
    deliver("openai_agents", _codec.encode(merged))
