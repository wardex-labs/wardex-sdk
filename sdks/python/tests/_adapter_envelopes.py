"""The contract every adapter's envelope holds, and where the committed bodies live.

Two suites had been green around a hole. The sender's tests built spans by
hand and looked for the conversation id in `extra`; a receiver's tests built
envelopes by hand with the typed `Span.conversation` field filled. Neither ever
held bytes the real encoder had produced from a real adapter's output, so
nothing noticed that the two were looking in different places and that a
receiver stored an empty conversation id for every span.

The check closes it from the sender's side, per adapter, in two halves:

* FRESH: drive the real adapter with its framework, encode what it emitted
  with the real encoder, decode, and check the contract below. This is the
  half that fails when an adapter, the encoder or the framework changes. Each
  adapter's FRESH test sits in a file of its own (`test_<adapter>_envelope.py`)
  that imports that one framework, so it runs wherever that framework alone is
  installed.
* COMMITTED: the same check over the envelope bodies under
  `fixtures/adapter_envelopes/` (`test_adapter_envelopes.py`). Those files are
  what a receiver's own test suite reads in place of hand-built envelopes, so
  they must keep telling the truth about what this SDK sends. They are request
  bodies exactly as `WardexTransport` would POST them: protobuf under zstd.

The contract, for every adapter: an id the host or the framework stated is in
the typed conversation field of every span the adapter opened, nothing about
the conversation is in `extra`, and a span the adapter did not open (a wire
`chat` span issued inside the run) carries the run's conversation too, the
one its request was issued in.

Regenerate the committed files (bytes differ per run, ids and clocks, and are
not compared; only the contract is), with every framework installed:

    WARDEX_REGEN_ENVELOPES=1 uv run pytest \\
        sdks/python/tests/test_openai_agents_envelope.py \\
        sdks/python/tests/test_langgraph_envelope.py \\
        sdks/python/tests/test_agent_sdk_envelope.py
"""

from __future__ import annotations

import datetime
import os
import platform
from importlib import metadata
from pathlib import Path

from wardex_sdk.transport import _codec

DIR = Path(__file__).parent / "fixtures" / "adapter_envelopes"
REGEN = os.environ.get("WARDEX_REGEN_ENVELOPES") == "1"
_CLIENT = 2  # `SPAN_KIND_CLIENT` on the envelope

#: adapter -> (the conversation id its scenario states, the distributions whose
#: versions the provenance records).
SCENARIOS: dict[str, tuple[str, tuple[str, ...]]] = {
    "openai_agents": ("conv-openai-agents", ("openai-agents",)),
    "langgraph": ("conv-langgraph", ("langgraph",)),
    "agent_sdk": ("s-1", ()),  # the CLI's own session id, read off its stream
}


def _spans_of(body: bytes) -> list[dict]:
    return [item["span"] for item in _codec.decode(body)["items"] if "span" in item]


def check(adapter: str, body: bytes) -> None:
    """The contract. One function, so FRESH and COMMITTED cannot drift apart."""
    conversation_id, _ = SCENARIOS[adapter]
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
        # The LLM calls are read off the wire, and the byte seam latches the
        # conversation their requests were issued in: the run's. These are the
        # spans that carry the tokens, so a receiver summing a conversation's
        # cost reads them here.
        assert wire
        for span in wire:
            got = span.get("conversation", {}).get("conversation_id")
            assert got == conversation_id, f"{adapter}: wire {span['name']!r} carries {got!r}"
    if adapter == "agent_sdk":
        turns = sorted(
            s["conversation"]["turn_index"] for s in spans if s["conversation"]["turn_index"]
        )
        assert turns and turns == list(range(1, len(turns) + 1)), f"turns run from 1: {turns}"


def deliver(adapter: str, body: bytes) -> None:
    """Check a FRESH body, and under the regeneration switch commit it."""
    check(adapter, body)
    if REGEN:
        DIR.mkdir(parents=True, exist_ok=True)
        (DIR / f"{adapter}.envelope.zst").write_bytes(body)
        _write_provenance()


def _write_provenance() -> None:
    """Each body named beside the date it was produced and what it was produced against.

    Rewritten whole by every FRESH test that regenerates, so it is right
    whichever of them ran last; a framework that is not installed keeps the row
    it had."""
    path = DIR / "PROVENANCE.md"
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    rows = []
    for adapter, (conversation_id, dists) in SCENARIOS.items():
        name = f"`{adapter}.envelope.zst`"
        try:
            versions = ", ".join(f"{d} {metadata.version(d)}" for d in dists)
        except metadata.PackageNotFoundError:
            kept = [line for line in old.splitlines() if line.startswith(f"| {name} ")]
            if kept:
                rows.append(kept[0])
                continue
            versions = "not installed when this was written"
        rows.append(f"| {name} | `{conversation_id}` | {versions or 'none (scripted stream)'} |")
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    path.write_text(
        "# Adapter envelope bodies\n\n"
        "Request bodies exactly as `WardexTransport` would POST them (a\n"
        "`wardex.v1.Envelope`, protobuf under zstd), each produced by driving the real\n"
        "adapter and encoding what it emitted with the real encoder. A receiver's test\n"
        "suite reads these in place of hand-built envelopes. Every value in them is\n"
        "synthetic: a scripted model, a loopback server, canned tool output.\n\n"
        f"- Produced: {today}\n"
        f"- wardex-sdk: {metadata.version('wardex-sdk')}, Python {platform.python_version()}\n"
        "- Source: `sdks/python/tests/test_<adapter>_envelope.py`, one file per adapter;\n"
        "  `_adapter_envelopes.py` holds the contract each body is checked against on\n"
        "  every run\n"
        "- Regenerate: `WARDEX_REGEN_ENVELOPES=1 uv run pytest` on those three files\n\n"
        "| File | Conversation id it states | Framework |\n|---|---|---|\n"
        + "\n".join(rows)
        + "\n",
        encoding="utf-8",
    )
