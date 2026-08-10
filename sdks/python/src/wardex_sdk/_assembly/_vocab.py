"""The closed span vocabulary — design §6.1-§6.4, invariant I9.

Three things live here and they are deliberately in one module, because they
are one decision: **what a span may be called, what it must carry to be called
that, and what may be written next to it.** Splitting them is how a name and its
required block drift apart.

  * `SpanIntent` — the twelve intents (§6.2), CLOSED. Each carries the
    `OperationName` it serializes as, the span kind it implies, and the typed
    block it is not allowed to exist without.
  * `LinkReason` — the five causal edge kinds (§6.3), CLOSED. Parentage is
    containment; a link is causality. `HANDOFF` is a marker span with a
    `HANDOFF_FROM` link and a SIBLING `invoke_agent`, not a container — a 5-hop
    handoff chain rendered as 5 levels of nesting makes every duration in the
    flame graph a lie.
  * the naming grammar and the `extra`-key rule, which is where I9 is actually
    enforced rather than described.

**Three naming modes, not one, and the second two are a correction to §6.1.**
The design states the grammar as `name = f"{operation} {subject}"` with
`operation ∈ OperationName`, full stop. Measured against the tree, that covers
one of the three kinds of span this SDK emits:

  VOCABULARY — an interpreted span: `chat claude-sonnet-5`, `invoke_agent`,
  `execute_tool Bash`. §6.1 exactly.

  TRANSPORT — a byte-seam observation wardex could not interpret as an
  operation: `HTTP POST /v1/messages`, `gRPC /pkg.Svc/Do`, `WS /realtime`,
  `MCP tools/call`. §6.2's twelve intents have no honest home for "an HTTP
  request happened and wardex has no idea what it meant", and inventing one
  would be exactly the protocol-branching §6.2 forbids. So these keep a CLOSED
  protocol label instead of an operation, and the grammar still holds — the
  label set is an enum, not a free string.

  MANUAL — `wardex.span("anything")`. The name is the host's, it is a published
  API, and no vocabulary wardex declares can close it without breaking that API.
  A manual span may still declare an `OperationName` label (the decorators do),
  which is recorded as `gen_ai.operation.name` exactly as before; what it does
  NOT do is trigger the structural requirement, because wardex cannot supply a
  typed block the host never handed it. Closing that gap means changing the
  decorator signatures, which is a public API change this layer cannot make.

`SpanDraft` names its mode at construction, so a site cannot slide from one to
another by accident, and `tests/test_vocabulary.py` pins all three.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .._enums import OperationName, SnapshotType, SpanKind


class VocabularyError(Exception):
    """A span was assembled outside the closed vocabulary.

    Raised only by `SpanDraft.finish()` / `SnapshotDraft.finish()`, and it is an
    SDK bug report, not a host error: it never reaches the host because every
    emit site builds its draft inside `assembly._diag.guard()` (I6). The cost of
    that is real and worth naming — a rejected span is a DELETED span, visible
    only as a counter — which is why the `Limitation` vocabulary had to be
    censused shut (§6.5.1) before any site was routed through here.
    """


class Block(Enum):
    """The typed attribute an intent cannot exist without.

    A `str` naming an `InternalSpan` field, so the requirement is checked by
    reading the draft rather than by a per-intent branch.
    """

    GEN_AI = "gen_ai"
    AGENT = "agent"
    TOOL = "tool"
    RETRIEVAL = "retrieval"
    EMBEDDINGS = "embeddings"
    EVALUATION = "evaluation"
    WORKFLOW_NAME = "workflow_name"


@dataclass(frozen=True, slots=True)
class _Rule:
    operation: OperationName
    kind: SpanKind
    required: tuple[Block, ...]
    required_extra: tuple[str, ...] = ()


class SpanIntent(Enum):
    """CLOSED at twelve — design §6.2. The only operations wardex declares.

    The value is the `OperationName` value, so `SpanIntent.CHAT.value` is what
    reaches `gen_ai.operation.name` and a reader never has to hold two spellings
    of one concept.

    The extension budget is one operation per genuinely new concept, and it is
    spent: `EXECUTE_STEP` (graph node / superstep / task — not an agent, not a
    tool), `HANDOFF` (agent transition), `EVALUATE` (guardrail / judge). A
    protocol variant is an attribute, never an operation: an MCP tool call is
    `EXECUTE_TOOL` + `ToolExecutionType.IPC`, not an `MCP_*` parallel universe.
    """

    INVOKE_WORKFLOW = "invoke_workflow"
    CREATE_AGENT = "create_agent"
    INVOKE_AGENT = "invoke_agent"
    EXECUTE_STEP = "execute_step"
    HANDOFF = "handoff"
    EVALUATE = "evaluate"
    EXECUTE_TOOL = "execute_tool"
    CHAT = "chat"
    TEXT_COMPLETION = "text_completion"
    EMBEDDINGS = "embeddings"
    GENERATE_CONTENT = "generate_content"
    RETRIEVAL = "retrieval"

    @property
    def operation(self) -> OperationName:
        return OperationName(self.value)

    @property
    def default_kind(self) -> SpanKind:
        return _RULES[self].kind

    @property
    def required_blocks(self) -> tuple[Block, ...]:
        return _RULES[self].required

    @property
    def required_extra_keys(self) -> tuple[str, ...]:
        return _RULES[self].required_extra


_RULES: dict[SpanIntent, _Rule] = {
    SpanIntent.INVOKE_WORKFLOW: _Rule(
        OperationName.INVOKE_WORKFLOW, SpanKind.INTERNAL, (Block.WORKFLOW_NAME,)
    ),
    SpanIntent.CREATE_AGENT: _Rule(OperationName.CREATE_AGENT, SpanKind.INTERNAL, (Block.AGENT,)),
    SpanIntent.INVOKE_AGENT: _Rule(OperationName.INVOKE_AGENT, SpanKind.INTERNAL, (Block.AGENT,)),
    # The step's identity is a namespaced extra rather than a typed block, and
    # that is the deliberate choice of §6.4: `wardex.step.*` has no OTel
    # semconv counterpart, and minting typed fields for one framework family's
    # concept is how a domain leaks into a domain-agnostic core.
    SpanIntent.EXECUTE_STEP: _Rule(
        OperationName.EXECUTE_STEP, SpanKind.INTERNAL, (), ("wardex.step.name",)
    ),
    # The RECEIVING agent. A handoff span is a marker, and the receiver's own
    # invoke_agent is its sibling, joined by LinkReason.HANDOFF_FROM.
    SpanIntent.HANDOFF: _Rule(OperationName.HANDOFF, SpanKind.INTERNAL, (Block.AGENT,)),
    SpanIntent.EVALUATE: _Rule(OperationName.EVALUATE, SpanKind.INTERNAL, (Block.EVALUATION,)),
    SpanIntent.EXECUTE_TOOL: _Rule(OperationName.EXECUTE_TOOL, SpanKind.INTERNAL, (Block.TOOL,)),
    SpanIntent.CHAT: _Rule(OperationName.CHAT, SpanKind.CLIENT, (Block.GEN_AI,)),
    SpanIntent.TEXT_COMPLETION: _Rule(
        OperationName.TEXT_COMPLETION, SpanKind.CLIENT, (Block.GEN_AI,)
    ),
    SpanIntent.EMBEDDINGS: _Rule(
        OperationName.EMBEDDINGS, SpanKind.CLIENT, (Block.GEN_AI, Block.EMBEDDINGS)
    ),
    SpanIntent.GENERATE_CONTENT: _Rule(
        OperationName.GENERATE_CONTENT, SpanKind.CLIENT, (Block.GEN_AI,)
    ),
    SpanIntent.RETRIEVAL: _Rule(OperationName.RETRIEVAL, SpanKind.CLIENT, (Block.RETRIEVAL,)),
}


class LinkReason(Enum):
    """CLOSED — design §6.3. Why one span points at another.

    A link is CAUSALITY; the parent edge is CONTAINMENT. Mixing them renders a
    flat sequence as N-deep nesting and makes every parent duration a lie.

    This enum is only half of the feature: `bindings/python/src/codec.rs`
    carried no `links`/`events` handling at all, so `InternalSpanLink.reason`
    was dropped at encode and the whole graph story of §6.3 — `TRIGGERED_BY`
    edges, `HANDOFF_FROM` siblings, `RESUMED_FROM` across a checkpoint — could
    not be transmitted. The encoder landed in the same commit that declared
    this enum, because either half alone is theatre.
    """

    TRIGGERED_BY = "triggered_by"  # a graph edge: the step that scheduled this one
    HANDOFF_FROM = "handoff_from"  # the agent that handed control over
    RESUMED_FROM = "resumed_from"  # a checkpoint resume — a NEW trace, linked
    RETRIED_FROM = "retried_from"  # the attempt this one replaces
    CACHE_SOURCE = "cache_source"  # the span whose result was served from cache


class TransportLabel(Enum):
    """CLOSED. The `operation` slot of a TRANSPORT-mode span name.

    Not an `OperationName`, on purpose: these spans report that bytes moved, not
    what the bytes meant, and promoting a protocol to an operation is the
    Sentry-style branching §6.2 rules out. Keeping the set an enum is what keeps
    the grammar closed in this mode too — the alternative, a free-form name
    argument on `SpanDraft`, would make the byte seams the one place any string
    at all can reach a span name.
    """

    HTTP = "HTTP"
    GRPC = "gRPC"
    WEBSOCKET = "WS"
    MCP = "MCP"


# --------------------------------------------------------------------------
# the name grammar
# --------------------------------------------------------------------------


def vocabulary_name(intent: SpanIntent, subject: str | None) -> str:
    """`"{operation} {subject}"`, or the operation alone — design §6.1.

    A subject that is absent, empty or the string "None" yields the bare
    operation. That third case is the whole point: `f"chat {ev.model}"` over a
    turn whose model the stream never reported produced the literal span name
    `"chat None"` in the shipped adapter, and it is unreachable from here
    because a name is never built by interpolation at a call site again.
    """
    if subject is None:
        return intent.value
    subject = subject.strip()
    if not subject or subject == "None":
        return intent.value
    return f"{intent.value} {subject}"


def transport_name(label: TransportLabel, subject: str | None) -> str:
    """`"{label} {subject}"` for an uninterpreted byte-seam observation."""
    if subject is None or not subject.strip():
        return label.value
    return f"{label.value} {subject.strip()}"


# --------------------------------------------------------------------------
# the `extra` key rule (I9, §6.5 tier 1)
# --------------------------------------------------------------------------

WARDEX_PREFIX = "wardex."
GEN_AI_PREFIX = "gen_ai."

_VOCABULARY_PREFIXES = (GEN_AI_PREFIX, WARDEX_PREFIX)

# I9 says an `extra` key is `gen_ai.*` or `wardex.*`. Measured against the tree,
# the byte seam also ships `network.protocol.version`, `rpc.*` and `ws.*`, and
# they are not a violation of the spirit of the rule: every one is a registered
# OTel semantic-convention key describing the transport, not vocabulary wardex
# invented. Renaming them into `wardex.*` would move standard keys out of the
# namespace every OTel consumer already understands, for no gain.
#
# So the rule is widened to an explicit, CLOSED list of registry namespaces
# rather than left as "anything the seam happens to write". The distinction the
# rule exists to enforce survives: a key nobody standardized has to be
# `wardex.*`, where a dashboard renders it as flat key/value and never as
# structure.
_REGISTRY_PREFIXES = (
    "network.",  # network.protocol.version
    "rpc.",  # rpc.system, rpc.service, rpc.method, rpc.grpc.*
    "ws.",  # ws.messages.*, ws.bytes.*, ws.close_code
    "http.",
    "server.",
    "url.",
    "error.",
    "code.",
)

DECLARED_EXTRA_PREFIXES = _VOCABULARY_PREFIXES + _REGISTRY_PREFIXES


def is_declared_extra_key(key: str) -> bool:
    """Whether `key` may be written onto an SDK-assembled span.

    Per-adapter `FRAMEWORK_EXTRAS` (exact keys plus declared prefixes under
    `wardex.{adapter}.*`, design §6.5) would narrow this further, but no
    adapter declares one yet, so a `wardex.*` key is accepted on the strength
    of its namespace alone.
    """
    return key.startswith(DECLARED_EXTRA_PREFIXES)


__all__ = [
    "DECLARED_EXTRA_PREFIXES",
    "Block",
    "LinkReason",
    "SnapshotType",
    "SpanIntent",
    "TransportLabel",
    "VocabularyError",
    "is_declared_extra_key",
    "transport_name",
    "vocabulary_name",
]
