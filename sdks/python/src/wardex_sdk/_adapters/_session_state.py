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
from typing import Any

from .._assembly import Limitation, SpanDraft, Unit, UnitKey
from .._protocol._claude_stream import AgentStreamEvent


@dataclass
class _BridgeBinding:
    """This session's tie to the OTel bridge, created at injection-correlation.

    EXISTENCE gates pending and the finalize-time merge; ``confirmed`` gates
    the NO_DATA marker alone. The split is I4 at work: a binding exists the
    moment the bridge was live for this session's spawn (its spans are worth
    holding for), but "the CLI was told to send and sent nothing" may only be
    claimed when the subprocess-env read-back CONFIRMED the injection landed.
    A read-back failure leaves ``trace_id_hex`` None and ``confirmed`` False:
    the session still merges whatever routes to it by ``session.id`` — the
    fallback key is what keeps such a session's CLI spans mergeable — and
    never earns the marker.
    """

    trace_id_hex: str | None
    confirmed: bool


@dataclass
class _PendingSpan:
    """An assembler-built draft held for the bridge's finalize-time merge.

    The draft's end instant is stamped via ``set_end_ns`` at pend time, so an
    unmerged flush emits exactly the span that would have shipped immediately
    — plus ``deferred_markers``, the timing markers whose truth the merge is
    what can change. Deferral replaces a limitation-REMOVAL API, which the
    integrity builder deliberately does not grow: a marker, once attached, is
    a fact; a deferred marker is a fact not yet decided.
    """

    draft: SpanDraft
    kind: str  # "chat" | "tool" | "subagent"
    deferred_markers: tuple[Limitation, ...] = ()
    tool_use_id: str | None = None
    agent_id: str | None = None
    #: Chat only: (turn_start_ns, end_ns) — the join window.
    window: tuple[int, int] | None = None
    #: Chat only: the GenAIAttributes block, kept for the ttft rewrite.
    #: Typed Any because `_types` is off-limits in `_adapters/` (C-S1).
    gen_ai: Any = None
    merged: bool = False
    #: Whether this record may be the tool join's target. False for the stub an
    #: eviction ships: `_merge_bridge` joins by POPPING `tool_use_id`, and an
    #: evicted call that later completes puts TWO records under one id with the
    #: stub pended first — so the stub would take the CLI's duration and the
    #: bridge source and become the anchor for the CLI's children, while the
    #: half holding the output and the real interval fell through unmerged. The
    #: CLI measured the WHOLE call, so its duration belongs on the half that
    #: represents the whole call. Expressed as a qualification field rather than
    #: by blanking `tool_use_id`: that field is the call's identity, and a
    #: future consumer would read the blank as a fact.
    mergeable: bool = True


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
class _EvictedTool:
    """What an open-tool record leaves behind when the bound evicts it.

    WAR-76's shape one layer up: the registry writes `Unit._evicted` so a later
    refusal can name wardex's own bound instead of the host's lifecycle, and
    this is the same breadcrumb for a later COMPLETION. Without it a
    `PostToolUse` that arrives after its open record was evicted is
    indistinguishable from a tool wardex never saw open — and the span built
    for it claims a duration of zero and a parent it did not earn.

    Three remembered fields and deliberately not a fourth. `start_ns`, `name`
    and `agent_id` are what the completion CANNOT re-derive: the stream path
    rebuilds the record with `agent_id=None` hardcoded and would hang the two
    halves of one call under two different parents, and the stream metadata
    that would have supplied the name may itself be gone. `claim_key` is
    deliberately RE-DERIVED at completion time rather than remembered —
    `_claim_key` re-runs the arbitration, which is the correct reading at that
    instant. `input_data` is deliberately NOT remembered: it is the bytes, i.e.
    the thing the bound exists to stop holding, and a breadcrumb that carried
    them would leave the bound as a name with no memory behind it.

    `completed` is the third-observation latch. A call can be closed by its
    hook AND by the stream; the first builds the completion half and the second
    is suppressed and counted, because a third span would pollute the very
    aggregates the overlap rule already asks readers to correct for.
    """

    start_ns: int
    name: str
    agent_id: str | None
    completed: bool = False


@dataclass
class _EvictedSubagent:
    """The span CONTEXT an evicted sub-agent leaves behind, so its subtree keeps
    its shape.

    The three anchor lookups (`_tool_draft`, `_resolve_subagent_anchor`,
    `_chat_agent_id`) resolve a sub-agent at EMIT time, not at open time, and
    all three fall silently to the session root on a miss. Evicting a LIVE
    sub-agent without this would re-parent every still-open tool and every later
    chat turn of that sub-agent onto the root and say nothing — trading one
    silent drop for a whole silently flattened subtree. A context stays a valid
    parent after its span ships, so remembering it costs two fields and keeps
    the tree literally identical to the un-evicted one.
    """

    #: SpanContext; typed Any because `_types` is off-limits in `_adapters/`.
    context: Any
    agent_type: str


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
    #: The not-yet-consumed user prompt for the NEXT main-thread chat span. At
    #: most one prompt pends per session — a new observation REPLACES it
    #: (counted), never appends, so the slot is bounded by construction — and
    #: it is consumed exactly once, by the first main-thread assistant turn.
    #: All three `pending_prompt*` fields are cleared together at consumption
    #: and destroyed with this record on close/evict/teardown.
    pending_prompt: bytes = b""
    #: Which channel recorded `pending_prompt`: "stream" (the byte-exact
    #: outbound message-object JSON from the tee) or "hook" (the CLI's
    #: re-decoded prompt text from `UserPromptSubmit`), or None. Doubles as the
    #: consumed/unconsumed flag — None means nothing pends.
    pending_prompt_source: str | None = None
    #: The pending STREAM prompt has been corroborated by one `UserPromptSubmit`
    #: hook. A second submit while the same prompt still pends then reads as a
    #: NEW user turn whose write the stream missed, not as a duplicate.
    pending_prompt_hook_seen: bool = False
    open_tools: dict[str, _OpenTool] = field(default_factory=dict)  # keyed by tool_use_id
    subagents: dict[str, _OpenSubagent] = field(default_factory=dict)  # keyed by agent_id
    #: What the two span-owning tables above leave behind when the bound evicts
    #: an entry, under the SAME bound so the memory cannot outgrow what it
    #: remembers for. Neither holds payload bytes: a call's input and output are
    #: exactly what eviction is for, and a breadcrumb that kept them would make
    #: the bound reclaim nothing. Both die with this record.
    evicted_tools: dict[str, _EvictedTool] = field(default_factory=dict)  # keyed by tool_use_id
    evicted_subagents: dict[str, _EvictedSubagent] = field(default_factory=dict)  # by agent_id
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
    #: The OTel bridge tie, or None for a bridge-off session — and None is the
    #: load-bearing default: every bridge branch in the assembler gates on it,
    #: so a session without a binding walks today's code paths exactly.
    bridge: _BridgeBinding | None = None
    #: Drafts held for the finalize-time merge, in emission order. Bounded by
    #: `max_session_entries` (overflow emits the OLDEST unmerged, immediately)
    #: and flushed on EVERY retirement path — finalize, teardown, and the
    #: registry-eviction retirement — so pending never deletes a span (I10).
    pending: list[_PendingSpan] = field(default_factory=list)
