"""What each adapter SAID about a conversation survives the real encoder.

Two suites had been green around a hole. The sender's tests built spans by
hand and looked for the conversation id in `extra`; a receiver's tests built
envelopes by hand with the typed `Span.conversation` field filled. Neither ever
held bytes the real encoder had produced from a real adapter's output, so
nothing noticed that the two were looking in different places and that a
receiver stored an empty conversation id for every span.

This file closes it from the sender's side, per adapter, in two halves:

* FRESH — drive the real adapter with its framework, encode what it emitted
  with the real encoder, decode, and check the contract below. This is the
  half that fails when an adapter or the encoder changes.
* COMMITTED — the same check over the envelope bodies under
  `fixtures/adapter_envelopes/`. Those files are what a receiver's own test
  suite reads in place of hand-built envelopes, so they must keep telling the
  truth about what this SDK sends. They are request bodies exactly as
  `WardexTransport` would POST them: protobuf under zstd.

The contract, for every adapter: an id the host or the framework stated is in
the typed conversation field of every span the adapter opened, nothing about
the conversation is in `extra`, and a span the adapter did not open — a wire
`chat` span — carries no conversation rather than an invented one.

Regenerate the committed files (bytes differ per run — ids and clocks — and
are not compared; only the contract is):

    WARDEX_REGEN_ENVELOPES=1 uv run pytest sdks/python/tests/test_adapter_envelopes.py
"""

from __future__ import annotations

import asyncio
import datetime
import os
import platform
from importlib import metadata
from pathlib import Path
from typing import Any

import pytest
from agents import RunConfig, Runner
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

import wardex_sdk as wardex
from test_agent_sdk_assembler import ASSISTANT, INIT, RESULT, FakeClient, _outbound, _submit
from test_codec import _header
from test_langgraph_adapter import (
    Installed,
    TrailState,
    _clean_scope,  # noqa: F401 — the autouse determinism fixture the langgraph harness needs
)
from test_openai_agents_adapter import _init
from test_openai_agents_wire import (
    _agents,
    agents_env,  # noqa: F401 — a fixture, usable here only because it is imported
    fake_openai,  # noqa: F401 — `agents_env` depends on it
)
from wardex_sdk import _hub
from wardex_sdk._adapters._assembler import SessionAssembler
from wardex_sdk._types import Envelope
from wardex_sdk.transport import _codec

_DIR = Path(__file__).parent / "fixtures" / "adapter_envelopes"
_REGEN = os.environ.get("WARDEX_REGEN_ENVELOPES") == "1"
_CLIENT = 2  # `SPAN_KIND_CLIENT` on the envelope

#: adapter -> (the conversation id its scenario states, the distributions whose
#: versions the provenance records).
_SCENARIOS: dict[str, tuple[str, tuple[str, ...]]] = {
    "openai_agents": ("conv-openai-agents", ("openai-agents",)),
    "langgraph": ("conv-langgraph", ("langgraph",)),
    "agent_sdk": ("s-1", ()),  # the CLI's own session id, read off its stream
}


def _spans_of(body: bytes) -> list[dict]:
    return [item["span"] for item in _codec.decode(body)["items"] if "span" in item]


def _check(adapter: str, body: bytes) -> None:
    """The contract. One function, so FRESH and COMMITTED cannot drift apart."""
    conversation_id, _ = _SCENARIOS[adapter]
    spans = _spans_of(body)
    assert spans, f"{adapter}: the envelope carries no span"
    # A CLIENT span under the OpenAI Agents run is an LLM call read off the wire,
    # not a span the adapter opened. The Agent SDK adapter builds its own chat
    # spans, so there every span is the adapter's.
    wire = [s for s in spans if s["kind"] == _CLIENT and adapter == "openai_agents"]
    stated = [s for s in spans if s not in wire]
    assert stated, f"{adapter}: no adapter-opened span to check"
    for span in stated:
        got = span.get("conversation", {}).get("conversation_id")
        assert got == conversation_id, f"{adapter}: {span['name']!r} carries {got!r}"
    for span in spans:
        keys = [kv["key"] for kv in span.get("extra", [])]
        leaked = [k for k in keys if k == "gen_ai.conversation.id" or k.startswith("wardex.conv")]
        assert not leaked, f"{adapter}: {span['name']!r} still writes {leaked} into extra"
    if adapter == "openai_agents":
        # The LLM calls are read off the wire, and the byte seam does not latch
        # the run's conversation: they carry none. Pinned here so that the day
        # they gain it, this line is where it is noticed — not a store.
        assert wire and all("conversation" not in s for s in wire)
    if adapter == "agent_sdk":
        turns = sorted(
            s["conversation"]["turn_index"] for s in spans if s["conversation"]["turn_index"]
        )
        assert turns and turns == list(range(1, len(turns) + 1)), f"turns run from 1: {turns}"


def _deliver(adapter: str, body: bytes) -> None:
    _check(adapter, body)
    if _REGEN:
        _DIR.mkdir(parents=True, exist_ok=True)
        (_DIR / f"{adapter}.envelope.zst").write_bytes(body)


def test_fresh_openai_agents_envelope(agents_env):  # noqa: F811
    transport = _init()
    try:
        config = RunConfig(workflow_name="wf", group_id=_SCENARIOS["openai_agents"][0])
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
    _deliver("openai_agents", _codec.encode(merged))


def test_fresh_langgraph_envelope():
    graph = StateGraph(TrailState)
    graph.add_node("a", lambda s: {"trail": ["a"]})
    graph.add_node("b", lambda s: {"trail": ["b"]})
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    live = Installed()
    try:
        config: Any = {"configurable": {"thread_id": _SCENARIOS["langgraph"][0]}}
        graph.compile(checkpointer=InMemorySaver()).invoke({"trail": []}, config)
        spans = tuple(live.spans)
    finally:
        live.teardown()
    _deliver("langgraph", _codec.encode(Envelope(header=_header(), spans=spans)))


def test_fresh_agent_sdk_envelope():
    client = FakeClient()
    asm = SessionAssembler(client)
    _outbound(asm, key=1, text="first question")
    asm.on_inbound(1, INIT)
    _submit(asm, "first question")
    asm.on_inbound(1, ASSISTANT)
    asm.on_inbound(1, RESULT)
    asm.on_close(1, None)
    _deliver("agent_sdk", _codec.encode(Envelope(header=_header(), spans=tuple(client.spans))))


@pytest.mark.parametrize("adapter", sorted(_SCENARIOS))
def test_the_committed_envelope_still_holds_the_contract(adapter: str):
    path = _DIR / f"{adapter}.envelope.zst"
    assert path.is_file(), (
        f"{path.name} is missing; regenerate with "
        "WARDEX_REGEN_ENVELOPES=1 uv run pytest sdks/python/tests/test_adapter_envelopes.py"
    )
    _check(adapter, path.read_bytes())


def test_the_provenance_names_every_committed_envelope():
    """A fixture nobody can trace is a fixture nobody can trust: each body is
    named beside the date it was produced and the framework version it was
    produced against."""
    text = (_DIR / "PROVENANCE.md").read_text()
    for adapter in _SCENARIOS:
        assert f"{adapter}.envelope.zst" in text


def test_regeneration_writes_the_provenance_last():
    """Runs after the three FRESH tests in file order, so under
    `WARDEX_REGEN_ENVELOPES=1` it records what they just wrote. Without the
    switch it writes nothing."""
    if not _REGEN:
        return
    rows = []
    for adapter, (conversation_id, dists) in _SCENARIOS.items():
        versions = (
            ", ".join(f"{d} {metadata.version(d)}" for d in dists) or "none (scripted stream)"
        )
        rows.append(f"| `{adapter}.envelope.zst` | `{conversation_id}` | {versions} |")
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    (_DIR / "PROVENANCE.md").write_text(
        "# Adapter envelope bodies\n\n"
        "Request bodies exactly as `WardexTransport` would POST them — a\n"
        "`wardex.v1.Envelope`, protobuf under zstd — each produced by driving the real\n"
        "adapter and encoding what it emitted with the real encoder. A receiver's test\n"
        "suite reads these in place of hand-built envelopes. Every value in them is\n"
        "synthetic: a scripted model, a loopback server, canned tool output.\n\n"
        f"- Produced: {today}\n"
        f"- wardex-sdk: {metadata.version('wardex-sdk')}, Python {platform.python_version()}\n"
        "- Source: `sdks/python/tests/test_adapter_envelopes.py`, which also holds the\n"
        "  contract each body is checked against on every run\n"
        "- Regenerate: `WARDEX_REGEN_ENVELOPES=1 uv run pytest "
        "sdks/python/tests/test_adapter_envelopes.py`\n\n"
        "| File | Conversation id it states | Framework |\n|---|---|---|\n" + "\n".join(rows) + "\n"
    )
