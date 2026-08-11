"""What the Anthropic assembler REMEMBERS between two events.

Separated from the machine that acts on it, because the two answer different
questions and change for different reasons. Everything here is shaped by the
Agent SDK's own vocabulary — `tool_use_id`, `agent_id`, the stream's result
event — so it is adapter-specific by construction and says so by living beside
the adapter rather than under `_assembly/`.

The one thing that is NOT specific is the rule these records follow: a span that
spans two events is held as a DRAFT, never as a context plus the fields needed
to rebuild it later. A draft is one object whose `context` children can hang off
immediately and whose eventual span is the same object by construction, where
the rebuild-later shape is a two-phase span written by hand at every site that
uses it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .._assembly import SpanDraft, Unit, UnitKey
from .._protocol._claude_stream import AgentStreamEvent


@dataclass
class _OpenTool:
    tool_use_id: str | None
    name: str
    start_ns: int
    agent_id: str | None
    input_data: bytes
    from_hook: bool
    output_data: bytes = b""
    #: This call's slot in the shared key space (`_anthropic_names`). The hook
    #: observer holds it at rank 0 and re-checks it at emit time, because the
    #: in-process handler wrapper may have taken the key over in between — which
    #: is exactly what happens for every SDK MCP tool.
    claim_key: UnitKey | None = None


@dataclass
class _OpenSubagent:
    """A subagent span opened at `SubagentStart` and finished at `SubagentStop`."""

    draft: SpanDraft
    agent_type: str


@dataclass
class _Session:
    #: The session's logical unit. `unit.draft` is its own two-phase span and
    #: `unit.context` is the P2 anchor every span below it hangs off — the same
    #: context the pin installs on the SDK's reader task, which is how a hook
    #: callback and an in-process tool handler reach it with no framework id.
    unit: Unit
    start_ns: int
    #: The transport key this session is filed under in `_by_key`. Carried on
    #: the record so that a lookup arriving by any other route — the CLI's
    #: `session_id`, the scope a hook runs in — can re-enter the one liveness
    #: check (`_live_session`) instead of growing its own copy of it.
    key: int
    session_id: str | None = None
    model: str | None = None  # from init -> request_model
    turn_start_ns: int = 0
    first_delta_ns: int = 0
    prompt: bytes = b""
    open_tools: dict[str, _OpenTool] = field(default_factory=dict)  # keyed by tool_use_id
    subagents: dict[str, _OpenSubagent] = field(default_factory=dict)  # keyed by agent_id
    turn_index: int = 0
    # Issued by wardex when the CLI has not (yet) reported a session id. §6.3:
    # `conversation_id` may not be the empty string — every span in one session
    # would otherwise collide with every span of every other session in any
    # store that keys on it, and `SpanDraft.finish()` now refuses to ship "".
    issued_conversation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stream_tool_meta: dict[str, tuple[str, bytes]] = field(default_factory=dict)
    # ^ tool_use_id -> (name, input_json) observed on the stream
    result: AgentStreamEvent | None = None
    error: str | None = None
