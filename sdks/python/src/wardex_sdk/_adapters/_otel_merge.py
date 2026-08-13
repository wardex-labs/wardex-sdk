"""Classification and join rules for the Agent SDK OTel bridge.

The Python half of the ratified "mapping in PYTHON, decode in RUST" split:
`crates/wardex-codec` decodes OTLP bytes (a wire format, core-appropriate) and
this module knows what a ``claude_code.*`` span IS — Agent SDK domain
knowledge that deliberately does NOT extend the stream-json-in-Rust precedent
(revisited when a Node SDK exists to share it).

Pure functions over plain dicts: no HTTP, no CLI, no draft building — the
assembler applies what is classified here, which is what keeps every rule in
this file unit-testable without either.

Join rules, and why each is shaped the way it is:

* TOOL — exact join on ``tool_use_id`` (runtime-confirmed on both
  ``claude_code.tool`` and ``.tool.execution``; duplicated as
  ``gen_ai.tool.call.id``). Exact or nothing.
* CHAT — unique-time-window within one agent scope, anchored on the
  ``llm_request``'s START instant. The spec's exact join on
  ``gen_ai.response.id`` has NO left-hand operand: the CLI's stream-json
  carries ``message.id`` (``msg_...``) only, and the OTel side carries the
  Anthropic ``request_id`` (``req_...``), so equality can never hold. A
  chat's window is the interval in which the request that PRODUCED it can
  have started — from the previous same-scope chat's arrival (the assembler
  sequences the windows: chats of one agentic loop share one host write, so
  their raw turn starts collide and full-interval overlap degenerated every
  multi-turn session into ambiguity, measured live) to its own arrival. A
  pair merges iff it is unique in BOTH directions — strict containment
  first, the tolerance pass only over what strictness left, so host-arrival
  lag is absorbed without letting a wide tolerance undo an exact fit. Any
  remaining ambiguity means NO merge for any party — guessing a parent from
  a coin flip is the competitor failure this SDK exists not to repeat.
* everything with no wardex counterpart (``hook``, ``mcp.rpc``,
  ``bash.subprocess``, ``compaction``, ``tool.blocked_on_user``) is a pure
  increment: new EXECUTE_STEP spans, CLI-measured times, no timing marker.
* unrecognized ``claude_code.*`` names are COUNTED, never guessed at — they
  are the schema-drift signal the fail-open marker reads.

Attribute copying is allowlist-based on NAMED keys — never a prefix rule.
``gen_ai.*`` is deliberately not admitted wholesale: the CLI's telemetry is
beta, and a prefix rule is the hole a future content-carrying ``gen_ai.*``
key would walk through (the receiver's identity denylist is the other half).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .._assembly import counters

#: The SECOND-pass tolerance around a chat's window when locating an
#: `llm_request`'s START. The two clocks are one machine's unix-epoch
#: nanoseconds, so this absorbs host-arrival lag (a busy host event loop
#: reads a message late, shifting the window boundaries that message
#: defines), not clock skew. The strict pass runs first, so this widens
#: candidacy only for requests strictness could not place; a request that
#: still fails, fails HONESTLY (no merge, markers kept), never wrongly.
JOIN_EPS_NS = 1_000_000_000

#: CLI span names with no wardex counterpart, mapped to their step name.
_STEP_NAMES = {
    "claude_code.hook": "hook",
    "claude_code.mcp.rpc": "mcp_rpc",
    "claude_code.bash.subprocess": "bash_subprocess",
    "claude_code.compaction": "compaction",
    "claude_code.tool.blocked_on_user": "tool_blocked_on_user",
}

#: Non-gen_ai keys the merge may copy, re-namespaced under
#: ``wardex.anthropic_agent_sdk.otel.<key>``.
_ALLOWED_KEYS = frozenset(
    {
        "tool_name",
        "agent_id",
        "parent_agent_id",
        "stop_reason",
        "request_id",
        "client_request_id",
        "ttft_ms",
    }
)

#: The NAMED gen_ai keys the merge admits, copied under their own names.
#: A closed list, not a prefix rule — see the module docstring.
_ALLOWED_GEN_AI_KEYS = frozenset(
    {
        "gen_ai.response.id",
        "gen_ai.response.model",
        "gen_ai.request.model",
        "gen_ai.response.finish_reasons",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.usage.cache_read_input_tokens",
        "gen_ai.usage.cache_creation_input_tokens",
        "gen_ai.tool.call.id",
    }
)

#: The extras namespace every copied non-gen_ai key lands under.
OTEL_EXTRA_PREFIX = "wardex.anthropic_agent_sdk.otel."


@dataclass
class _OtelSpan:
    """The projection of one decoded CLI span this module consumes."""

    name: str
    span_id: str
    parent_hex: str
    start_ns: int
    end_ns: int
    status_code: int
    attrs: dict


@dataclass
class _OtelTool:
    """One CLI tool call: the ``claude_code.tool`` span plus, when it arrived,
    the inner ``.tool.execution`` interval that prices the body alone."""

    span_ids: list[str] = field(default_factory=list)
    outer: _OtelSpan | None = None
    execution: _OtelSpan | None = None

    @property
    def duration_ms(self) -> float | None:
        span = self.execution or self.outer
        if span is None or span.end_ns <= span.start_ns:
            return None
        return (span.end_ns - span.start_ns) / 1e6


@dataclass
class _OtelLlm:
    """One ``claude_code.llm_request``, with its join inputs precomputed."""

    span: _OtelSpan
    agent_id: str | None
    response_id: str | None
    ttft_ms: float | None


@dataclass
class _BridgeView:
    """Everything one session's slot classified to."""

    tools: dict[str, _OtelTool] = field(default_factory=dict)
    llm: list[_OtelLlm] = field(default_factory=list)
    interactions: list[_OtelSpan] = field(default_factory=list)
    spawns: dict[str, _OtelSpan] = field(default_factory=dict)
    increments: list[tuple[str, _OtelSpan]] = field(default_factory=list)
    by_span_id: dict[str, _OtelSpan] = field(default_factory=dict)
    unknown: int = 0

    @property
    def recognized(self) -> int:
        """How many spans classified as ANYTHING the bridge knows. Zero with a
        non-empty slot is the schema-drift signal."""
        return (
            len(self.tools)
            + len(self.llm)
            + len(self.interactions)
            + len(self.spawns)
            + len(self.increments)
        )


def _project(raw: dict) -> _OtelSpan:
    status = raw.get("status") or {}
    return _OtelSpan(
        name=str(raw.get("name") or ""),
        span_id=str(raw.get("span_id") or ""),
        parent_hex=str(raw.get("parent_span_id") or ""),
        start_ns=int(raw.get("start_time_unix_nano") or 0),
        end_ns=int(raw.get("end_time_unix_nano") or 0),
        status_code=int(status.get("code") or 0),
        attrs=raw.get("attributes") or {},
    )


def _tool_use_id(span: _OtelSpan) -> str | None:
    for key in ("tool_use_id", "gen_ai.tool.call.id"):
        value = span.attrs.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _scalar(value: object) -> bool:
    return isinstance(value, (str, int, float, bool))


def classify(spans: list[dict]) -> _BridgeView:
    """Bucket one slot's decoded spans by what the bridge knows them to be."""
    view = _BridgeView()
    for raw in spans:
        span = _project(raw)
        if span.span_id:
            view.by_span_id[span.span_id] = span
        name = span.name
        if name == "claude_code.interaction":
            view.interactions.append(span)
        elif name == "claude_code.llm_request":
            response_id = span.attrs.get("gen_ai.response.id") or span.attrs.get("request_id")
            ttft = span.attrs.get("ttft_ms")
            view.llm.append(
                _OtelLlm(
                    span=span,
                    agent_id=None,  # resolved after every spawn is known
                    response_id=response_id if isinstance(response_id, str) else None,
                    ttft_ms=float(ttft) if isinstance(ttft, (int, float)) else None,
                )
            )
        elif name in ("claude_code.tool", "claude_code.tool.execution"):
            tid = _tool_use_id(span)
            if tid is None:
                # A tool span with no id cannot join exactly, and a name/time
                # guess is the join this design refuses. Counted, not kept.
                view.unknown += 1
                counters.bump("adapters.anthropic.otel_bridge.tool_without_id")
                continue
            tool = view.tools.setdefault(tid, _OtelTool())
            tool.span_ids.append(span.span_id)
            if name == "claude_code.tool":
                tool.outer = span
            else:
                tool.execution = span
        elif name == "claude_code.subagent.spawn":
            agent_id = span.attrs.get("agent_id")
            if isinstance(agent_id, str) and agent_id:
                view.spawns[agent_id] = span
            else:
                view.unknown += 1
        elif name in _STEP_NAMES:
            view.increments.append((_STEP_NAMES[name], span))
        else:
            view.unknown += 1
            counters.bump("adapters.anthropic.otel_bridge.span_unrecognized")
    for llm in view.llm:
        llm.agent_id = _agent_scope(llm.span, view)
    return view


def _agent_scope(span: _OtelSpan, view: _BridgeView) -> str | None:
    """Which subagent's work this span is, or None for the main thread.

    The span's own ``agent_id`` attribute when the CLI stamped one; otherwise
    the nearest OTel ancestor that is a ``subagent.spawn``. The walk is the
    CLI's own parent chain — reading it is not a guess.
    """
    own = span.attrs.get("agent_id")
    if isinstance(own, str) and own:
        return own
    seen: set[str] = set()
    parent = span.parent_hex
    while parent and parent not in seen:
        seen.add(parent)
        ancestor = view.by_span_id.get(parent)
        if ancestor is None:
            return None
        if ancestor.name == "claude_code.subagent.spawn":
            agent_id = ancestor.attrs.get("agent_id")
            return agent_id if isinstance(agent_id, str) and agent_id else None
        parent = ancestor.parent_hex
    return None


@dataclass
class _ChatWindow:
    """One pended chat draft's join inputs, keyed by its pending index."""

    key: int
    start_ns: int
    end_ns: int
    agent_id: str | None


@dataclass
class _JoinOutcome:
    #: (chat key, llm) pairs that matched uniquely in BOTH directions.
    pairs: list[tuple[int, _OtelLlm]] = field(default_factory=list)
    #: Every llm_request that did not merge — ambiguous or unmatched alike.
    #: The assembler ships each as a SIBLING step span that says so, because
    #: CLI-measured LLM work must not disappear just because the tree could
    #: not place it.
    unjoined: list[_OtelLlm] = field(default_factory=list)


def _start_within(chat: _ChatWindow, llm: _OtelLlm, eps_ns: int) -> bool:
    """Whether this request can have STARTED inside this chat's window.

    The START and not the interval: an `llm_request` ENDS at the moment its
    message is emitted, which is the very instant that opens the NEXT chat's
    window — so any interval-overlap predicate makes every request brush its
    successor's window and re-creates the ambiguity the sequencing removed.
    A request's start has one home.
    """
    return chat.start_ns - eps_ns <= llm.span.start_ns <= chat.end_ns + eps_ns


def _match(chats: list[_ChatWindow], llms: list[_OtelLlm], eps_ns: int) -> list:
    by_chat: dict[int, list[_OtelLlm]] = {}
    by_llm: dict[int, list[_ChatWindow]] = {}
    for chat in chats:
        for llm in llms:
            if chat.agent_id == llm.agent_id and _start_within(chat, llm, eps_ns):
                by_chat.setdefault(chat.key, []).append(llm)
                by_llm.setdefault(id(llm), []).append(chat)
    pairs = []
    for chat in chats:
        candidates = by_chat.get(chat.key, [])
        if len(candidates) == 1 and len(by_llm[id(candidates[0])]) == 1:
            pairs.append((chat.key, candidates[0]))
    return pairs


def join_chats(
    chats: list[_ChatWindow], llms: list[_OtelLlm], eps_ns: int = JOIN_EPS_NS
) -> _JoinOutcome:
    """The unique-start-window join, scoped by agent, in two passes.

    A pair merges iff the llm is the chat's ONLY candidate AND the chat is
    the llm's ONLY candidate. The strict pass (no tolerance) settles every
    request whose start sits cleanly inside one window; the tolerance pass
    runs only over what strictness left, so host-arrival lag is absorbed
    without a wide tolerance manufacturing ambiguity for exact fits. Two
    chats over one llm, or two llms over one chat, unmatch every party
    involved: no merge is the only answer the evidence backs (I4), and the
    drafts keep their timing markers honestly.
    """
    outcome = _JoinOutcome()
    remaining_chats = list(chats)
    remaining_llms = list(llms)
    for tolerance in (0, eps_ns):
        pairs = _match(remaining_chats, remaining_llms, tolerance)
        if not pairs:
            continue
        outcome.pairs.extend(pairs)
        taken_chats = {key for key, _ in pairs}
        taken_llms = {id(llm) for _, llm in pairs}
        remaining_chats = [c for c in remaining_chats if c.key not in taken_chats]
        remaining_llms = [llm for llm in remaining_llms if id(llm) not in taken_llms]
    outcome.unjoined = remaining_llms
    if outcome.unjoined:
        counters.bump("adapters.anthropic.otel_bridge.llm_join_ambiguous")
    return outcome


def allowlisted_extras(attrs: dict) -> list[tuple[str, object]]:
    """The attribute pairs the merge may copy onto a wardex span.

    Scalars only, keys from the two NAMED sets only: unknown keys — including
    unknown ``gen_ai.*`` keys — are dropped and counted like everything else.
    This is the structural PII guarantee on top of the receiver's identity
    denylist: nothing rides through on the strength of its prefix.
    """
    out: list[tuple[str, object]] = []
    for key, value in attrs.items():
        if not _scalar(value):
            counters.bump("adapters.anthropic.otel_bridge.attr_dropped")
            continue
        if key in _ALLOWED_GEN_AI_KEYS:
            out.append((key, value))
        elif key in _ALLOWED_KEYS:
            out.append((OTEL_EXTRA_PREFIX + key, value))
        else:
            counters.bump("adapters.anthropic.otel_bridge.attr_dropped")
    return out
