"""What one pydantic-ai span's attributes mean to wardex — closed tables, pure.

No I/O, no clock, no OpenTelemetry import and no unit call: the adapter hands
this module an attribute mapping and gets back the typed blocks, the extras,
the content to record and the status. That split is what lets every rule below
be tested on pydantic-ai's own measured span dumps with no framework, no
tracer and no wardex registry.

The reading was measured on `pydantic-ai-slim` 2.66.0 at instrumentation
schema version 5 (the default), and versions 5 and 6 differ only inside the
message JSON (a `tool_call_response` message's role), never in a key this
module reads. What it rests on:

* A span is classified by `gen_ai.operation.name` (`invoke_agent`, `chat`,
  `execute_tool`), never by its name: the name is `f"chat {model}"` at open
  and renamed twice afterwards, and output-function and validation-failure
  spans share the `execute_tool` name and operation. The one span with no
  operation name that wardex reads is a FallbackModel's failed attempt,
  recognised by `pydantic_ai.model_request.attempt`.
* Usage is never the framework's to state. pydantic reports per-request
  `gen_ai.usage.*` on `chat` and a run total as `gen_ai.aggregated_usage.*`
  on `invoke_agent` (or `gen_ai.usage.*` there when
  `use_aggregated_usage_attribute_names=False`); the wire CHAT record wardex
  parses from the socket is the authority for the same call. So every usage
  key on every span is DEMOTED under `wardex.pydantic_ai.otel.`, value kept
  verbatim, invisible to a backend's `gen_ai.usage.` collector, and the typed
  block built here never carries a token count. The core promotes a chat's
  demoted numbers back at close when no wire request was issued under it.
* Content never becomes an extra. `set_extra` is uncapped, and the content
  attributes are O(history) — a 50 000-character prompt produced a
  250 000-character `gen_ai.input.messages` on the next chat span, untruncated.
  The run's `pydantic_ai.all_messages` / `final_result` and the tool's
  arguments / result are handed back separately for the capped
  `record_input` / `record_output` path; every other content key is dropped.
* Status descriptions and `exception.message` / `exception.stacktrace` are
  never read: with content on, pydantic writes the raised message into both
  (`'RuntimeError: tool exploded secret=abc'`). Only the exception's type is.
* A key no table names is UNKNOWN and is reported, not guessed at — it is the
  signal a schema change shows up as, since the framework releases minors
  weekly and the adapter claims the whole major.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from .._assembly import AgentAttributes, GenAIAttributes, ToolAttributes
from .._enums import (
    AgentType,
    OperationName,
    ProviderName,
    StatusCode,
    ToolExecutionType,
    ToolType,
)

#: What a span is to the adapter. `other` is a span wardex records nothing
#: for (a user's own span on the shared tracer, a decision-model span).
Role = Literal["run", "chat", "tool", "attempt", "other"]

#: What happens to one attribute key. `typed` feeds a typed block, `extra` is
#: written as an extra, `content` goes through the capped record path,
#: `dropped` is known and deliberately never copied, `unknown` is drift.
Fate = Literal["typed", "extra", "content", "dropped", "unknown"]

Scalar = str | int | float | bool

#: The value the adapter writes as `wardex.framework` on every unit it opens.
FRAMEWORK = "pydantic_ai"

#: Where every renamed key lands. Anything outside `gen_ai.*` / `wardex.*`
#: deletes the whole span at `finish()`, so nothing pydantic names keeps its
#: own spelling unless it is a `gen_ai.*` key a typed block consumes.
EXTRA_PREFIX = "wardex.pydantic_ai."

#: The demoted-usage namespace. The core's promotion rule is told this prefix
#: when the adapter declares a chat unit a framework-usage holder.
USAGE_PREFIX = EXTRA_PREFIX + "otel."

#: Extras the adapter writes from its own state rather than from an attribute.
SCHEMA_VERSION_KEY = EXTRA_PREFIX + "schema_version"
CONVERSATION_SHADOWED_KEY = EXTRA_PREFIX + "conversation_id"
RUN_DEFERRED_KEY = EXTRA_PREFIX + "run.deferred"

_OPERATION_ROLES: dict[object, Role] = {
    "invoke_agent": "run",
    "chat": "chat",
    "execute_tool": "tool",
}

#: `PAI/_model_request_attempts.py`: a failed attempt's span carries the model's
#: attributes plus this index and NO operation name — "deliberately not named
#: `chat`, so model-call views don't count it as a model call".
_ATTEMPT_KEY = "pydantic_ai.model_request.attempt"

#: Usage spellings pydantic emits (`PAI/usage.py` `opentelemetry_attributes`),
#: on every role that can carry them. Demoted by prefix, values kept: the
#: `details.*` suffixes are open-ended (provider-specific token kinds) and a
#: closed list here would drop a billed number silently.
_USAGE_PREFIXES = ("gen_ai.usage.", "gen_ai.aggregated_usage.")

#: Keys pydantic puts on every span it writes, never copied.
_LOGFIRE_PREFIX = "logfire."

#: The three run-identity keys pydantic copies from baggage onto every `chat`
#: and `execute_tool` span. On those spans they repeat the run's own values.
_BAGGAGE = frozenset({"gen_ai.agent.name", "gen_ai.agent.call.id", "gen_ai.conversation.id"})

#: Request settings pydantic maps one-to-one (`PAI/_instrumentation.py`
#: `MODEL_SETTING_ATTRIBUTES`), with the type each block field takes.
_REQUEST_SETTINGS: dict[str, tuple[str, type]] = {
    "gen_ai.request.max_tokens": ("max_tokens", int),
    "gen_ai.request.top_p": ("top_p", float),
    "gen_ai.request.seed": ("seed", int),
    "gen_ai.request.temperature": ("temperature", float),
    "gen_ai.request.presence_penalty": ("presence_penalty", float),
    "gen_ai.request.frequency_penalty": ("frequency_penalty", float),
}

_CHAT_TYPED = frozenset(
    {
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.response.id",
        "gen_ai.response.finish_reasons",
        "gen_ai.client.operation.time_to_first_chunk",
        "server.address",
        "server.port",
        *_REQUEST_SETTINGS,
    }
)

#: Extras renamed from a named source key, per role: `{source: extra key}`.
#: Named, not prefix rules, for the reason the Agent SDK bridge gives for its
#: own allowlist: a prefix rule is the hole a future content-carrying key
#: walks through. `pydantic_ai.cache.*` is listed key by key for the same
#: reason (`collapse_reason` is a closed enum of five words today; a new
#: cache key is reported as unknown until it is read and listed here).
_RENAMED: dict[Role, dict[str, str]] = {
    "run": {
        "gen_ai.agent.call.id": EXTRA_PREFIX + "run_id",
    },
    "chat": {
        "operation.cost": EXTRA_PREFIX + "operation.cost",
        # Written when the user opts into `openai_transport='websocket'`.
        "pydantic_ai.transport": EXTRA_PREFIX + "transport",
        **{
            f"pydantic_ai.cache.{k}": f"{EXTRA_PREFIX}cache.{k}"
            for k in (
                "hit_ratio",
                "established_tokens",
                "missed_tokens",
                "collapsed",
                "collapse_reason",
                "not_enabled",
            )
        },
    },
    "tool": {
        "pydantic_ai.tool.deferral.name": EXTRA_PREFIX + "tool.deferral",
        "pydantic_ai.tool.failure_stage": EXTRA_PREFIX + "tool.failure_stage",
        "pydantic_ai.realtime": EXTRA_PREFIX + "realtime",
        **{
            f"pydantic_ai.{kind}.retry.{k}": f"{EXTRA_PREFIX}{kind}.retry.{k}"
            for kind in ("tool", "output")
            for k in ("cause", "attempt", "max_retries")
        },
    },
    "attempt": {
        _ATTEMPT_KEY: EXTRA_PREFIX + "model_request.attempt",
        # A rejected response's cost is recorded on the attempt "since [it was]
        # billed" (`PAI/_model_request_attempts.py`).
        "operation.cost": EXTRA_PREFIX + "operation.cost",
    },
    "other": {},
}

#: Content keys and where they go: `{source: "in" | "out"}`. Only these
#: reach a unit, and only through the capped record path.
_CONTENT: dict[Role, dict[str, str]] = {
    "run": {"pydantic_ai.all_messages": "in", "final_result": "out"},
    "tool": {"gen_ai.tool.call.arguments": "in", "gen_ai.tool.call.result": "out"},
    "chat": {},
    "attempt": {},
    "other": {},
}

#: Every content-bearing key pydantic writes, as measured: message JSON,
#: instructions, tool schemas, run metadata, deferral metadata.
#: The ones not in `_CONTENT` are known and dropped: the chat unit records no
#: content because the wire span under it carries the bodies, and the run
#: carries the whole history once.
CONTENT_KEYS = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "model_request_parameters",
        "pydantic_ai.all_messages",
        "pydantic_ai.tool.deferral.metadata",
        "final_result",
        "metadata",
    }
)

_TYPED: dict[Role, frozenset[str]] = {
    "run": frozenset({"gen_ai.agent.name", "gen_ai.agent.description", "gen_ai.conversation.id"}),
    "chat": _CHAT_TYPED,
    "tool": frozenset({"gen_ai.tool.name", "gen_ai.tool.call.id"}),
    "attempt": _CHAT_TYPED,
    "other": frozenset(),
}

_DROPPED: dict[Role, frozenset[str]] = {
    "run": frozenset(
        {
            "gen_ai.operation.name",
            "agent_name",  # the legacy duplicate of gen_ai.agent.name
            "model_name",  # the run's model at start; each chat names its own
            "gen_ai.system_instructions",
            "metadata",
            "pydantic_ai.new_message_index",
            "pydantic_ai.variable_instructions",
            "pydantic_ai.workspace.provider",
            "pydantic_ai.workspace.id",
        }
    ),
    "chat": frozenset(
        {
            "gen_ai.operation.name",
            "gen_ai.system",  # the deprecated duplicate of gen_ai.provider.name
            *_BAGGAGE,
            "gen_ai.input.messages",
            "gen_ai.output.messages",
            "gen_ai.system_instructions",
            "gen_ai.tool.definitions",
            "model_request_parameters",
        }
    ),
    "tool": frozenset({"gen_ai.operation.name", *_BAGGAGE, "pydantic_ai.tool.deferral.metadata"}),
    "attempt": frozenset({"gen_ai.system"}),
    "other": frozenset(),
}

#: Every exact key each role knows, whatever its fate: the drift census.
KNOWN_KEYS: dict[Role, frozenset[str]] = {
    role: _TYPED[role] | _DROPPED[role] | frozenset(_RENAMED[role]) | frozenset(_CONTENT[role])
    for role in ("run", "chat", "tool", "attempt", "other")
}

#: The deferral exceptions. At schema version 5 a deferral is an UNSET tool
#: span with `pydantic_ai.tool.deferral.name` (`e1`, `e2`); at versions 3-4 it
#: is an ERROR span whose exception event names the type (`e4`). Both are
#: control flow — the run hands the call back to its caller — not a failure.
_DEFERRALS = frozenset({"CallDeferred", "ApprovalRequired"})


def classify(attrs: Mapping[str, object]) -> Role:
    """The span's role, from its attributes alone."""
    op = attrs.get("gen_ai.operation.name")
    role = _OPERATION_ROLES.get(op) if isinstance(op, str) else None  # a list is unhashable
    if role is not None:
        return role
    if "gen_ai.operation.name" not in attrs and _ATTEMPT_KEY in attrs:
        return "attempt"
    return "other"


def key_fate(role: Role, key: str) -> Fate:
    """What happens to `key` on a span of `role`."""
    if role == "other":
        return "dropped"  # the span itself is not recorded; its keys are not drift
    if key.startswith(_USAGE_PREFIXES):
        return "extra"
    if key in _CONTENT[role]:
        return "content"
    if key in _RENAMED[role]:
        return "extra"
    if key in _TYPED[role]:
        return "typed"
    if key in _DROPPED[role] or key.startswith(_LOGFIRE_PREFIX):
        return "dropped"
    if role == "run" and key.startswith("pydantic_ai.workspace."):
        return "dropped"
    return "unknown"


def _is_scalar(value: object) -> bool:
    return isinstance(value, (str, int, float, bool))


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _extra_key(role: Role, key: str) -> str | None:
    if key.startswith(_USAGE_PREFIXES):
        return USAGE_PREFIX + key
    return _RENAMED[role].get(key)


def extras_for(role: Role, attrs: Mapping[str, object]) -> list[tuple[str, Scalar]]:
    """The `(extra key, value)` pairs this span contributes, in attribute order.

    Only scalars, and only counts for usage: a non-int usage value has no
    meaning the core's promotion could keep, and a non-scalar has no extra
    representation at all. Both are left out here and named by
    `rejected_keys` for the adapter to count.
    """
    out: list[tuple[str, Scalar]] = []
    for key, value in attrs.items():
        extra = _extra_key(role, key)
        if extra is None or key_fate(role, key) != "extra":
            continue
        if key.startswith(_USAGE_PREFIXES):
            if _is_count(value):
                out.append((extra, value))
        elif _is_scalar(value):
            out.append((extra, value))
    return out


def rejected_keys(role: Role, attrs: Mapping[str, object]) -> list[str]:
    """Keys bound for an extra whose value `extras_for` refused to write."""
    out: list[str] = []
    for key, value in attrs.items():
        if key_fate(role, key) != "extra":
            continue
        ok = _is_count(value) if key.startswith(_USAGE_PREFIXES) else _is_scalar(value)
        if not ok:
            out.append(key)
    return out


def unknown_keys(role: Role, attrs: Mapping[str, object]) -> list[str]:
    """Keys no table names, sorted: the schema-drift signal."""
    return sorted(k for k in attrs if key_fate(role, k) == "unknown")


def content_for(role: Role, attrs: Mapping[str, object]) -> tuple[object | None, object | None]:
    """`(input, output)` content for the capped record path; `None` when absent.

    A run's input is `pydantic_ai.all_messages` (the whole history, once) and
    its output `final_result`; a tool's are its call arguments (present at
    START, the only content set then) and its result. Everything else records
    no content.
    """
    found: dict[str, object | None] = {"in": None, "out": None}
    for key, slot in _CONTENT[role].items():
        value = attrs.get(key)
        if value is not None:
            found[slot] = value
    return found["in"], found["out"]


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _typed(value: object, kind: type) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if kind is int:
        return value if isinstance(value, int) else None
    return float(value)


def _provider(value: object) -> ProviderName | str | None:
    text = _text(value)
    if text is None:
        return None
    for member in ProviderName:
        if member.value == text:
            return member
    return text  # pydantic's own names ("test", "function", "google-gla") stay as written


def _reasons(value: object) -> tuple[str, ...] | None:
    # A list on our own span, a tuple once an SDK span has frozen it.
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        return tuple(value)
    return None


def chat_block(attrs: Mapping[str, object], *, subject: str) -> GenAIAttributes:
    """The chat (or failed attempt) unit's `gen_ai` block — never with usage.

    `subject` is the model the unit was opened under; the request model read
    here replaces it once pydantic has set it (deferred to after the span's
    open on the agent path, `PAI/capabilities/instrumentation.py`). Token
    fields are left `None` on purpose: see the module docstring.
    """
    settings = {
        field: _typed(attrs.get(key), kind) for key, (field, kind) in _REQUEST_SETTINGS.items()
    }
    ttft = _typed(attrs.get("gen_ai.client.operation.time_to_first_chunk"), float)
    return GenAIAttributes(
        operation=OperationName.CHAT,
        provider=_provider(attrs.get("gen_ai.provider.name")),
        request_model=_text(attrs.get("gen_ai.request.model")) or subject or None,
        response_model=_text(attrs.get("gen_ai.response.model")),
        response_id=_text(attrs.get("gen_ai.response.id")),
        finish_reasons=_reasons(attrs.get("gen_ai.response.finish_reasons")),
        time_to_first_chunk_s=ttft,
        **settings,
    )


def server_of(attrs: Mapping[str, object]) -> tuple[str | None, int | None]:
    """`(server.address, server.port)` from the model's `base_url`, when stated."""
    port = attrs.get("server.port")
    return _text(attrs.get("server.address")), (
        port if isinstance(port, int) and not isinstance(port, bool) else None
    )


def tool_block(attrs: Mapping[str, object], *, fallback_name: str = "") -> ToolAttributes:
    """The tool unit's block. No call id on an output-function or a
    validation-failure span is `None`, never a made-up id.

    Every tool pydantic runs is IN_PROCESS from the agent's side, MCP tools
    included: the `execute_tool` span wraps the toolset call, and the MCP
    stdio seam records the JSON-RPC `tools/call` as its own span beneath it.
    """
    return ToolAttributes(
        name=_text(attrs.get("gen_ai.tool.name")) or fallback_name,
        call_id=_text(attrs.get("gen_ai.tool.call.id")),
        type=ToolType.FUNCTION,
        execution_type=ToolExecutionType.IN_PROCESS,
    )


def agent_block(
    attrs: Mapping[str, object],
    *,
    nested: bool,
    parent_agent: str | None,
    fallback_name: str = "",
) -> AgentAttributes:
    """The run unit's block. A run whose parent span is another run's tool is a
    sub-agent of that run (`f_nested_agents`: the inner `invoke_agent` sits
    under the outer run's `execute_tool delegate`)."""
    return AgentAttributes(
        name=_text(attrs.get("gen_ai.agent.name")) or fallback_name,
        description=_text(attrs.get("gen_ai.agent.description")),
        agent_type=AgentType.SUB_AGENT if nested else AgentType.PRIMARY,
        parent_agent=parent_agent if nested else None,
    )


def _bare(error_type: str | None) -> str | None:
    # Mode U reads `exception.type`, which pydantic writes fully qualified
    # (`pydantic_ai.exceptions.ToolRetryError`); the type name is the class.
    if not error_type:
        return None
    return str(error_type).rsplit(".", 1)[-1] or None


def is_deferral(attrs: Mapping[str, object], error_type: str | None = None) -> bool:
    """Whether a tool span ended by handing its call back to the caller."""
    return (
        attrs.get("pydantic_ai.tool.deferral.name") is not None or _bare(error_type) in _DEFERRALS
    )


def outcome(
    role: Role,
    status_error: bool,
    error_type: str | None,
    attrs: Mapping[str, object],
    *,
    deferred_child: bool = False,
    held: bool = False,
) -> tuple[StatusCode, str | None]:
    """`(status, error type)` for the unit a span closes.

    * tool — a deferral is UNSET, whichever schema version spelled it; an
      ERROR is a failed call with its exception's type (a `ModelRetry` is
      `ToolRetryError`: one failed attempt, the run goes on, `d1`); else OK.
    * run — ERROR with its type; UNSET when a tool under it deferred (the run
      produced `DeferredToolRequests`, not an answer) or when it ended while a
      child was still live (cancellation: pydantic leaves a cancelled run's
      span UNSET, `l_cancel_mid_tool`, and the run did not finish); else OK.
    * chat, tool — ERROR with its type; UNSET when its run had already ended
      (`held`: the run was cut while this call was in flight, and pydantic
      leaves a cancelled model call's span UNSET with no exception, so OK
      would claim an answer that never came); else OK.
    * attempt — always ERROR: pydantic opens the span only once the attempt
      has failed; a rejected response states no exception, hence
      `FailedAttempt`.
    """
    bare = _bare(error_type)
    if role == "attempt":
        return StatusCode.ERROR, bare or "FailedAttempt"
    if role == "other":
        return StatusCode.UNSET, None
    if role == "tool" and is_deferral(attrs, error_type):
        return StatusCode.UNSET, None
    if status_error:
        return StatusCode.ERROR, bare or "error"
    if held or (role == "run" and deferred_child):
        return StatusCode.UNSET, None
    return StatusCode.OK, None


__all__ = [
    "CONTENT_KEYS",
    "CONVERSATION_SHADOWED_KEY",
    "EXTRA_PREFIX",
    "FRAMEWORK",
    "KNOWN_KEYS",
    "RUN_DEFERRED_KEY",
    "SCHEMA_VERSION_KEY",
    "USAGE_PREFIX",
    "Fate",
    "Role",
    "Scalar",
    "agent_block",
    "chat_block",
    "classify",
    "content_for",
    "extras_for",
    "is_deferral",
    "key_fate",
    "outcome",
    "rejected_keys",
    "server_of",
    "tool_block",
    "unknown_keys",
]
