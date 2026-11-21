"""Hermes's plugin contract as this adapter reads it: names, threads, schema versions, fields.

Owns: `CONTRACT`, the one table registration is driven from (nothing is registered that is not a
row of it); `NOT_REGISTERED`, every other Hermes hook and middleware kind with the reason it is
left alone; `KNOWN_SCHEMAS`; `USAGE_KEYS`; the frozen event records parsed from payloads and the
one generic `parse`. This is the only module that spells a Hermes hook name, middleware kind or
payload field, so a Hermes change lands here and nowhere else.

Knows no wardex verb and no turn state beyond the `Trigger` each event maps to.

What is never read, by design: model message and response content (`user_message`,
`conversation_history`, `request`, `messages`, `response`, `assistant_response`; the wire is the
authority for model content), `sender_id` (end-user identity), `child_goal`, `child_summary`,
cost, and any profile name. `parse` reads only the fields a record's `FIELDS` names, so a field
that is not named never reaches a record. Tool arguments and tool results are read, because no
wire carries an in-process tool's input or output. They are carried through for the core's
capped, masked payload path (`record_input` and `record_output`) only, never for an attribute.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Protocol

from ._hermes_agent_turns import CloseCause, Trigger

__all__ = [
    "CONTRACT",
    "KNOWN_SCHEMAS",
    "MISSING",
    "NOT_REGISTERED",
    "USAGE_KEYS",
    "AuxEnd",
    "AuxStart",
    "CallEnter",
    "CallError",
    "CallReport",
    "CallbackThread",
    "ChildStart",
    "ChildStop",
    "ContractRow",
    "Drift",
    "DriftNote",
    "Event",
    "Field",
    "Finalize",
    "KeyKind",
    "Surface",
    "ToolEnter",
    "ToolReport",
    "TurnEnd",
    "TurnStart",
    "parse",
]

#: The schema versions this release was measured on (Hermes v0.23.3), by the payload field that
#: states each. Any other value is drift: one line, `measured=False`, and recording continues.
KNOWN_SCHEMAS: Mapping[str, str] = {
    "telemetry_schema_version": "hermes.observer.v1",
    "middleware_schema_version": "hermes.middleware.v1",
}

#: Hermes `usage` keys mapped to the key, after the hold prefix, that the core's promotion table
#: reads. `request_count`, `prompt_tokens` and `total_tokens` restate other keys, and
#: `cache_write_by_ttl` has no field to land in, so none of the four is carried.
USAGE_KEYS: Mapping[str, str] = {
    "input_tokens": "gen_ai.usage.input_tokens",
    "output_tokens": "gen_ai.usage.output_tokens",
    "cache_read_tokens": "gen_ai.usage.cache_read.input_tokens",
    "cache_write_tokens": "gen_ai.usage.cache_creation.input_tokens",
    "reasoning_tokens": "gen_ai.usage.reasoning.output_tokens",
}


class Surface(Enum):
    HOOK = "hook"  # ctx.register_hook
    MIDDLEWARE = "middleware"  # ctx.register_middleware, execution kinds only


class CallbackThread(Enum):
    """The thread Hermes runs a callback on, as measured. It decides what the callback may do."""

    HOOK_THREAD = "hook_thread"  # a fresh hook thread holding a COPY of the caller's context
    LOOP = "loop"  # the agent's loop thread itself, no copy: a pin set here stays for the turn
    TOOL_WORKER = "tool_worker"  # a pooled worker, context copied per job
    CALLER = "caller"  # the caller's own thread, no copy (a tool worker, a delegate, shutdown)


class KeyKind(Enum):
    """What an event's `key()` names."""

    TURN = "turn"  # one turn, by turn_id (None when Hermes stated none)
    PARENT_TURN = "parent_turn"  # the parent's turn, by parent_turn_id (subagent_start)
    SESSION = "session"  # every live turn of one session, or of every session for key None


class Event(Protocol):
    """What every parsed event answers. The dispatcher needs nothing else to route it."""

    def trigger(self) -> Trigger: ...
    def key_kind(self) -> KeyKind: ...
    def key(self) -> str | None: ...


class _Missing:
    __slots__ = ()

    def __repr__(self) -> str:
        return "MISSING"


#: What a field reader returns for a value that is absent or of the wrong type.
MISSING: Any = _Missing()


@dataclass(frozen=True, slots=True)
class Field:
    """One payload field: Hermes's name, the record's attribute, and the reader that checks it."""

    payload: str
    attr: str
    read: Callable[[object], object]  # the value, or MISSING when absent or mistyped
    required: bool = True


def _text(v: object) -> object:
    return v if isinstance(v, str) and v else MISSING


def _int(v: object) -> object:
    return v if isinstance(v, int) and not isinstance(v, bool) else MISSING


def _flag(v: object) -> object:
    return v if isinstance(v, bool) else MISSING


def _raw(v: object) -> object:
    return MISSING if v is None else v


def _usage(v: object) -> object:
    """`usage` -> ((core key, count), ...) for the keys in USAGE_KEYS whose value is an int."""
    if not isinstance(v, Mapping):
        return MISSING
    return tuple(
        (core, n)
        for hermes, core in USAGE_KEYS.items()
        if isinstance(n := v.get(hermes), int) and not isinstance(n, bool)
    )


def _member(obj: object, name: str) -> object:
    # Hermes hands a `NormalizedResponse` object; its recorded form is a plain mapping.
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _requested(v: object) -> object:
    """`assistant_message` -> ((tool_call_id, tool name), ...) from its `tool_calls`.

    Ids and names only: `arguments` and every content member are left unread. A call without a
    string id and name is skipped, since the id is what a dispatched frame is matched by.
    """
    if v is None:
        return MISSING
    calls = _member(v, "tool_calls")
    if not isinstance(calls, (list, tuple)):
        return ()
    pairs = ((_member(c, "id"), _member(c, "name")) for c in calls)
    return tuple((i, n) for i, n in pairs if isinstance(i, str) and i and isinstance(n, str) and n)


# -- event records -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TurnStart:
    """`pre_llm_call`. `parent_session_id` is stated for a delegated child only."""

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("session_id", "session_id", _text),
        Field("turn_id", "turn_id", _text),
        Field("parent_session_id", "parent_session_id", _text, required=False),
        Field("platform", "platform", _text, required=False),
    )
    session_id: str
    turn_id: str
    parent_session_id: str | None = None
    platform: str | None = None

    def trigger(self) -> Trigger:
        return Trigger.TURN_START

    def key_kind(self) -> KeyKind:
        return KeyKind.TURN

    def key(self) -> str | None:
        return self.turn_id


@dataclass(frozen=True, slots=True)
class CallEnter:
    """`llm_execution` frame entry. `request` and `next_call` are in the same kwargs, unread."""

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("session_id", "session_id", _text),
        Field("turn_id", "turn_id", _text),
        Field("api_request_id", "api_request_id", _text),
        Field("model", "model", _text, required=False),
        Field("api_mode", "api_mode", _text, required=False),
    )
    session_id: str
    turn_id: str
    api_request_id: str
    model: str | None = None
    api_mode: str | None = None

    def trigger(self) -> Trigger:
        return Trigger.CALL_ENTER

    def key_kind(self) -> KeyKind:
        return KeyKind.TURN

    def key(self) -> str | None:
        return self.turn_id


@dataclass(frozen=True, slots=True)
class CallReport:
    """`post_api_request`: Hermes's usage (held, never promoted here) and the tool calls the
    model asked for, by id and name."""

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("session_id", "session_id", _text),
        Field("turn_id", "turn_id", _text),
        Field("api_request_id", "api_request_id", _text),
        Field("usage", "usage", _usage, required=False),
        Field("assistant_message", "requested", _requested, required=False),
    )
    session_id: str
    turn_id: str
    api_request_id: str
    usage: tuple[tuple[str, int], ...] = ()
    requested: tuple[tuple[str, str], ...] = ()

    def trigger(self) -> Trigger:
        return Trigger.CALL_REPORT

    def key_kind(self) -> KeyKind:
        return KeyKind.TURN

    def key(self) -> str | None:
        return self.turn_id


@dataclass(frozen=True, slots=True)
class CallError:
    """`api_request_error`. One per failed logical call; the attempts inside it are on the wire."""

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("session_id", "session_id", _text),
        Field("turn_id", "turn_id", _text),
        Field("api_request_id", "api_request_id", _text),
        Field("retry_count", "retry_count", _int),
        Field("max_retries", "max_retries", _int),
        Field("retryable", "retryable", _flag),
        Field("reason", "reason", _text, required=False),
    )
    session_id: str
    turn_id: str
    api_request_id: str
    retry_count: int
    max_retries: int
    retryable: bool
    reason: str | None = None

    def trigger(self) -> Trigger:
        """A prediction. Hermes fires this hook before it increments `retry_count` and fails the
        turn when the incremented count reaches `max_retries`. A fallback provider can still
        recover after the hook, which is why the close it causes is PRESUMED_CLOSED."""
        spent = (not self.retryable) or self.retry_count + 1 >= self.max_retries
        return Trigger.CALL_FAILED_FINAL if spent else Trigger.CALL_FAILED

    def key_kind(self) -> KeyKind:
        return KeyKind.TURN

    def key(self) -> str | None:
        return self.turn_id


@dataclass(frozen=True, slots=True)
class ToolEnter:
    """`tool_execution` frame entry. `args` is Hermes's object: read, never changed."""

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("session_id", "session_id", _text),
        Field("turn_id", "turn_id", _text),
        Field("api_request_id", "api_request_id", _text),
        Field("tool_call_id", "tool_call_id", _text),
        Field("tool_name", "tool_name", _text),
        Field("args", "args", _raw, required=False),
    )
    session_id: str
    turn_id: str
    api_request_id: str
    tool_call_id: str
    tool_name: str
    args: object = None  # for record_input only

    def trigger(self) -> Trigger:
        return Trigger.TOOL_ENTER

    def key_kind(self) -> KeyKind:
        return KeyKind.TURN

    def key(self) -> str | None:
        return self.turn_id


@dataclass(frozen=True, slots=True)
class ToolReport:
    """`post_tool_call`. Hermes returns a tool failure as a result, so `status` carries it."""

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("session_id", "session_id", _text),
        Field("turn_id", "turn_id", _text),
        Field("tool_call_id", "tool_call_id", _text),
        Field("status", "status", _text, required=False),  # "ok" measured; others verbatim
        Field("error_type", "error_type", _text, required=False),
        Field("result", "result", _raw, required=False),
    )
    session_id: str
    turn_id: str
    tool_call_id: str
    status: str | None = None
    error_type: str | None = None
    result: object = None  # for record_output only

    def trigger(self) -> Trigger:
        return Trigger.TOOL_REPORT

    def key_kind(self) -> KeyKind:
        return KeyKind.TURN

    def key(self) -> str | None:
        return self.turn_id


@dataclass(frozen=True, slots=True)
class AuxStart:
    """`pre_auxiliary_call`, once per physical attempt. `turn_id` is optional: an auxiliary call
    outside any turn is legal, and keys no turn. `model` names the hook-built span."""

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("session_id", "session_id", _text),
        Field("api_request_id", "aux_request_id", _text),  # "aux-<hex>"
        Field("retry_count", "retry_count", _int),
        Field("turn_id", "turn_id", _text, required=False),
        Field("aux_task", "aux_task", _text, required=False),
        Field("model", "model", _text, required=False),
    )
    session_id: str
    aux_request_id: str
    retry_count: int
    turn_id: str | None = None
    aux_task: str | None = None
    model: str | None = None

    def attempt(self) -> tuple[str, int]:
        """The key one attempt is kept under, from its start to its end."""
        return (self.aux_request_id, self.retry_count)

    def trigger(self) -> Trigger:
        return Trigger.AUX_START

    def key_kind(self) -> KeyKind:
        return KeyKind.TURN

    def key(self) -> str | None:
        return self.turn_id


@dataclass(frozen=True, slots=True)
class AuxEnd:
    """`post_auxiliary_call`. Its usage is written as held extras on the hook-built span only."""

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("session_id", "session_id", _text),
        Field("api_request_id", "aux_request_id", _text),
        Field("retry_count", "retry_count", _int),
        Field("turn_id", "turn_id", _text, required=False),
        Field("error_type", "error_type", _text, required=False),
        Field("usage", "usage", _usage, required=False),
    )
    session_id: str
    aux_request_id: str
    retry_count: int
    turn_id: str | None = None
    error_type: str | None = None
    usage: tuple[tuple[str, int], ...] = ()

    def attempt(self) -> tuple[str, int]:
        return (self.aux_request_id, self.retry_count)

    def trigger(self) -> Trigger:
        return Trigger.AUX_END

    def key_kind(self) -> KeyKind:
        return KeyKind.TURN

    def key(self) -> str | None:
        return self.turn_id


@dataclass(frozen=True, slots=True)
class ChildStart:
    """`subagent_start`, applied to the parent's turn. `child_role` is never an agent name."""

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("parent_session_id", "parent_session_id", _text),
        Field("parent_turn_id", "parent_turn_id", _text),
        Field("child_session_id", "child_session_id", _text),
        Field("child_subagent_id", "child_subagent_id", _text, required=False),
        Field("child_role", "child_role", _text, required=False),
    )
    parent_session_id: str
    parent_turn_id: str
    child_session_id: str
    child_subagent_id: str | None = None
    child_role: str | None = None

    def trigger(self) -> Trigger:
        return Trigger.CHILD_START

    def key_kind(self) -> KeyKind:
        return KeyKind.PARENT_TURN

    def key(self) -> str | None:
        return self.parent_turn_id


@dataclass(frozen=True, slots=True)
class ChildStop:
    """`subagent_stop`: the net for a child session that sent no end of its own."""

    FIELDS: ClassVar[tuple[Field, ...]] = (Field("child_session_id", "child_session_id", _text),)
    child_session_id: str

    def trigger(self) -> Trigger:
        return Trigger.CHILD_STOP

    def key_kind(self) -> KeyKind:
        return KeyKind.SESSION

    def key(self) -> str | None:
        return self.child_session_id


@dataclass(frozen=True, slots=True)
class TurnEnd:
    """`on_session_end`. It usually names its turn. The CLI's interrupted-run path sends only
    the session, so then it applies to every live turn of the session."""

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("session_id", "session_id", _text),
        Field("turn_id", "turn_id", _text, required=False),
        Field("completed", "completed", _flag, required=False),
        Field("failed", "failed", _flag, required=False),
        Field("interrupted", "interrupted", _flag, required=False),
        Field("turn_exit_reason", "exit_reason", _text, required=False),
    )
    session_id: str
    turn_id: str | None = None
    completed: bool = False
    failed: bool = False
    interrupted: bool = False
    exit_reason: str | None = None  # an open set: kept verbatim, never parsed

    def cause(self) -> CloseCause:
        """From Hermes's three flags only. A stated failure outranks a stated interrupt, and
        `completed` counts only when neither is stated."""
        if self.failed:
            return CloseCause.ENDED_FAILED
        if self.interrupted:
            return CloseCause.ENDED_INTERRUPTED
        if self.completed:
            return CloseCause.ENDED_COMPLETED
        return CloseCause.ENDED_INCOMPLETE

    def trigger(self) -> Trigger:
        return Trigger.TURN_END

    def key_kind(self) -> KeyKind:
        return KeyKind.TURN if self.turn_id else KeyKind.SESSION

    def key(self) -> str | None:
        return self.turn_id or self.session_id


@dataclass(frozen=True, slots=True)
class Finalize:
    """`on_session_finalize`. Hermes shows a callback's return value to the user, so the
    callback for this row returns None like every other."""

    FIELDS: ClassVar[tuple[Field, ...]] = (
        Field("session_id", "session_id", _text, required=False),
        Field("reason", "reason", _text, required=False),
    )
    session_id: str | None = None  # None: every live turn in this process
    reason: str | None = None

    def trigger(self) -> Trigger:
        return Trigger.FINALIZE

    def key_kind(self) -> KeyKind:
        return KeyKind.SESSION

    def key(self) -> str | None:
        return self.session_id


# -- the contract table ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ContractRow:
    """One registration. `record` is the event type its payload parses into."""

    surface: Surface
    name: str
    thread: CallbackThread
    schema_field: str | None  # None: the version is not measured on this row, so not checked
    record: type[Any]


_H, _M = Surface.HOOK, Surface.MIDDLEWARE
_T = "telemetry_schema_version"

#: The one table registration is driven from, in registration order.
CONTRACT: tuple[ContractRow, ...] = (
    ContractRow(_H, "pre_llm_call", CallbackThread.HOOK_THREAD, _T, TurnStart),
    ContractRow(_M, "llm_execution", CallbackThread.LOOP, "middleware_schema_version", CallEnter),
    ContractRow(_H, "post_api_request", CallbackThread.HOOK_THREAD, _T, CallReport),
    ContractRow(_H, "api_request_error", CallbackThread.HOOK_THREAD, _T, CallError),
    ContractRow(
        _M, "tool_execution", CallbackThread.TOOL_WORKER, "middleware_schema_version", ToolEnter
    ),
    ContractRow(_H, "post_tool_call", CallbackThread.HOOK_THREAD, _T, ToolReport),
    ContractRow(_H, "pre_auxiliary_call", CallbackThread.HOOK_THREAD, _T, AuxStart),
    ContractRow(_H, "post_auxiliary_call", CallbackThread.HOOK_THREAD, _T, AuxEnd),
    ContractRow(_H, "subagent_start", CallbackThread.CALLER, _T, ChildStart),
    ContractRow(_H, "subagent_stop", CallbackThread.CALLER, _T, ChildStop),
    ContractRow(_H, "on_session_end", CallbackThread.HOOK_THREAD, _T, TurnEnd),
    ContractRow(_H, "on_session_finalize", CallbackThread.CALLER, None, Finalize),
)

_KANBAN = "kanban workers are not measured, and the payload carries profile_name"
_STORE = "session store lifecycle, not a turn"
_IN_TOOL = "approval prompts run inside a tool frame and are outside the scope map"

#: Every other hook name and middleware kind Hermes declares at the measured version, with the
#: reason wardex leaves it alone. CONTRACT names plus these keys equal Hermes's full list.
NOT_REGISTERED: Mapping[str, str] = {
    "llm_request": "request middleware may rewrite the request and costs Hermes a deep copy of "
    "it; llm_execution carries the same ids on the same loop thread",
    "tool_request": "request middleware may rewrite tool arguments; tool_execution carries the "
    "same ids",
    "pre_api_request": "duplicates the llm_execution frame's ids from a hook thread",
    "pre_tool_call": "duplicates the tool_execution frame's ids and can veto a tool",
    "post_llm_call": "carries turn content only, and is absent on a failed turn",
    "on_session_start": "fires per turn before pre_llm_call with no turn_id to key on",
    "transform_llm_output": "a transform: its return value changes what the user sees",
    "transform_tool_result": "a transform: its return value changes what the model sees",
    "transform_terminal_output": "a transform: its return value changes terminal output",
    "transform_api_error_classification": "a transform: its return value changes retry policy",
    "on_stream_start": "the wire is the authority for content and timing",
    "on_stream_delta": "the wire is the authority for content, and it costs a call per delta",
    "on_stream_end": "the wire is the authority for content and timing",
    "on_interim_message": "content",
    "pre_verify": "its return value decides whether the turn continues",
    "on_session_open": _STORE,
    "on_session_reset": _STORE,
    "on_session_delete": _STORE,
    "on_skill_lifecycle": "skill lifecycle facts, not a turn",
    "pre_gateway_dispatch": "can skip or rewrite an incoming message, and carries the sender",
    "post_gateway_admission": "carries the sender, which is end-user identity and not recorded",
    "agent_loop_stopped": "a gateway stop names a session key, not a turn; the interrupted "
    "turn's on_session_end closes the turn",
    "pre_approval_request": _IN_TOOL,
    "post_approval_response": _IN_TOOL,
    "on_human_input_request": "content, inside a tool frame",
    "on_human_input_resolved": "content, inside a tool frame",
    "on_room_member_activity": "hosted group chat, not measured",
    "pre_transcription": "its return value can rewrite transcription parameters",
    "kanban_task_claimed": _KANBAN,
    "kanban_task_completed": _KANBAN,
    "kanban_task_blocked": _KANBAN,
    "on_kanban_worker_spawned": _KANBAN,
    "on_kanban_worker_exited": _KANBAN,
    "on_kanban_worker_stale_claim": _KANBAN,
    "on_kanban_task_updated": _KANBAN,
    "on_kanban_dispatch_tick": _KANBAN,
    "gateway_platform_event": "platform envelopes such as reactions and edits, not a turn",
    "pre_command": "slash commands, not a turn",
    "post_compaction": "the compaction's model calls arrive as auxiliary calls; this adds only "
    "a summary of content",
    "post_curator_run": "curator passes write files, not turns",
}


# -- parsing -----------------------------------------------------------------------------------


class Drift(Enum):
    SCHEMA = "schema"  # a version other than KNOWN_SCHEMAS: the event is kept, measured=False
    FIELD = "field"  # a required field absent or mistyped: the event is dropped and counted


@dataclass(frozen=True, slots=True)
class DriftNote:
    kind: Drift
    name: str  # the contract row's name
    detail: str  # "telemetry_schema_version='hermes.observer.v2'" or "missing turn_id"


def parse(row: ContractRow, payload: Mapping[str, Any]) -> tuple[Event | None, DriftNote | None]:
    """One payload into (event or None, drift or None).

    The caller passes Hermes's keyword arguments without `next_call`. Only the fields the row's
    record names are read, so content fields, `request` and `next_call` are never touched. A
    schema drift keeps the event, because recording goes on. A field drift drops it, because an
    event with no turn_id cannot be placed and an invented one would be a made-up value; when
    both happen, the field drift is the one returned. Never raises on mappings, strings, numbers
    or lists; an object whose attribute access raises propagates to the caller's guard.
    """
    drift = None
    if row.schema_field is not None:
        seen = payload.get(row.schema_field)
        if seen != KNOWN_SCHEMAS[row.schema_field]:
            drift = DriftNote(Drift.SCHEMA, row.name, f"{row.schema_field}={seen!r}")
    values: dict[str, object] = {}
    missing: list[str] = []
    for f in row.record.FIELDS:
        v = f.read(payload.get(f.payload))
        if v is not MISSING:
            values[f.attr] = v
        elif f.required:
            missing.append(f.payload)
    if missing:
        return None, DriftNote(Drift.FIELD, row.name, "missing " + ", ".join(missing))
    return row.record(**values), drift
