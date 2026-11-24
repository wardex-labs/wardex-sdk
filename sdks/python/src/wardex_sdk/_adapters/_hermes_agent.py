"""Hermes Agent: one causal tree per Hermes turn, from inside Hermes's plugin contract.

Hermes is observed through its documented plugin contract only: hooks and execution middleware.
Nothing of Hermes is patched, wrapped or imported. The socket tee (an interceptor) records each HTTP
attempt; this module only decides what those attempts sit under.

Owns: the dispatcher every `CONTRACT` row's callback routes to (`on_hook`, `on_frame`), the turn
effects, the MCP slot reader and `restate` (what `wardex.diagnostics()` says now). The frame and
auxiliary-call effects are `FrameEffects` in `_hermes_agent_frames.py`. Registration, the route
and the export thread are in `_hermes_agent_attach.py`, whose `_ROUTE` `install` patches through
the context's `PatchSet`.

Reading one turn takes four files, in this order: `_hermes_agent_events.py` (what Hermes said),
`_hermes_agent_turns.py` (what that does to the turn), this module and `_hermes_agent_frames.py`
(what wardex records).

Locks. `self._lock` guards the drift facts and the unreached declarations made with no live turn,
and nothing else. The turn, frame and book locks are the turns module's. No wardex verb, `count`
or `report_once` runs while any of them is held: a transition is won under a record's lock, and
the winner runs the effect after `advance` has returned.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .._assembly import (
    AgentAttributes,
    DropReason,
    DropUnit,
    Limitation,
    LinkReason,
    SpanIntent,
    UnitKey,
    UnitKind,
    ambient_restore_point,
    report_once,
)
from .._assembly import record as ledger_record
from .._enums import AdapterState, AgentType
from . import _hermes_agent_attach as _attach
from . import _hermes_agent_turns as _turns
from ._base import AdapterInterface
from ._context import AdapterContext, Placement, RunHandle
from ._conversation import framework_conversation
from ._hermes_agent_attach import HOST_ISOLATION_LINE, NOT_ATTACHED_DETAIL
from ._hermes_agent_events import (
    HOST_PROCESS_ENV,
    KNOWN_SCHEMAS,
    CallEnter,
    CallError,
    ChildStart,
    ContractRow,
    Drift,
    DriftNote,
    Event,
    KeyKind,
    ToolEnter,
    ToolReport,
    TurnAnswer,
    TurnEnd,
    TurnStart,
    parse,
)
from ._hermes_agent_frames import (
    _FRAME,
    HOLD_PREFIX,
    PREFIX,
    TOOL_NS,
    FrameEffects,
    _beside,
    _downstream_name,
    _drop_lane,
    _message_text,
    _opened_on,
    _Out,
)
from ._hermes_agent_turns import (
    CLOSE_RULES,
    CREATES,
    OBSERVATIONS,
    Action,
    ChildTie,
    ExportMode,
    FrameRecord,
    Step,
    Trigger,
    TurnBook,
    TurnKind,
    TurnRecord,
    TurnState,
    step,
)
from ._payload import _shaped_payload

if TYPE_CHECKING:
    from .._assembly import RestorePoint
    from .._diagnostics import AdapterStatus

__all__ = ["HOLD_PREFIX", "PREFIX", "HermesAgentAdapter"]

#: The runtime's name, as the Codex adapter writes `codex`. Hermes states no agent name, and an
#: `invoke_agent` span with no agent block is dropped at emit.
AGENT_NAME = "hermes"

#: Alias namespaces. Every alias is remembered, so a closed unit stays a link target.
TURN_NS = "hermes.turn"  # turn_id

#: The flush bound on the `on_session_finalize` caller at shutdown, inside Hermes's own 10 s
#: bound on it.
FINALIZE_DRAIN_S = 2.0
#: How many undispatched tool names, and how many auxiliary task names, a turn states.
UNDISPATCHED_NAMES = 16
AUX_TASK_NAMES = 16

_UNIT_KIND: Mapping[TurnKind, UnitKind] = {
    TurnKind.ROOT: UnitKind.SESSION,
    TurnKind.CHILD: UnitKind.AGENT,
}
_AGENT_TYPE: Mapping[TurnKind, AgentType] = {
    TurnKind.ROOT: AgentType.PRIMARY,
    TurnKind.CHILD: AgentType.SUB_AGENT,
}


def _tool_call_of(entry: Mapping[str, Any]) -> tuple[str, str | None] | None:
    """`follow_tool_calls` reader: (tool name, canonical arguments) for a live tool frame's slot.

    How an MCP tool gets one span: Hermes names it `mcp__<server>__<tool>`, the MCP seam's wire
    `tools/call` names `<tool>`, and the core accepts a unit named `<prefix>_<wire name>` only
    when the arguments are equal. The transport span then nests under this tool unit.
    """
    name = entry.get("tool_name")
    return (name, entry.get("tool_arguments")) if isinstance(name, str) else None


def _merge(modes: Iterable[_Out]) -> ExportMode | None:
    """DRAIN wins over KICK; frames are not exports."""
    found = None
    for mode in modes:
        if mode is ExportMode.DRAIN:
            return mode
        if mode is ExportMode.KICK:
            found = mode
    return found


class HermesAgentAdapter(FrameEffects, AdapterInterface):
    """The recording half. It patches one attribute of its own module, so `install` cannot
    fail half way, and it holds no Hermes object."""

    def __init__(self) -> None:
        self._installed = False
        self._ctx: AdapterContext | None = None
        self._book = TurnBook()
        self._lock = threading.RLock()
        self._seen: set[str] = set()  # schema fields seen with a known value
        self._schema_drift: dict[str, str] = {}  # schema field -> what Hermes stated, first
        self._dropped: dict[str, str] = {}  # contract row -> the first missing fields
        self._unreached: dict[Hashable, float] = {}  # declarations with no live turn -> when
        self._effects: Mapping[Action, Callable[[TurnRecord, Any, Step, Trigger], _Out]] = {
            Action.OPEN: self._open,
            Action.CONTINUE: self._continue,
            Action.CONTINUE_CLOSE: self._continue_close,
            Action.FRAME: self._frame,
            Action.ORPHAN_FRAME: self._orphan_frame,
            Action.CLOSE_FRAME: self._close_frame,
            Action.AUX_BEGIN: self._aux_begin,
            Action.AUX_FINISH: self._aux_finish,
            Action.AUX_DECLARE_UNREACHED: self._aux_unreached,
            Action.AUX_RELEASE_UNREACHED: self._aux_unreached,
            Action.TIE: self._tie,
            Action.NOTE_ERROR: self._note_error,
            Action.ANSWER: self._answer,
            Action.CLOSE: self._close,
            Action.FORGET: self._forget,
            Action.IGNORE_DUPLICATE: self._ignore,
            Action.IGNORE_LATE: self._ignore,
            Action.IGNORE_UNKNOWN: self._ignore,
        }

    def _at_fork_reinit(self) -> None:
        """Fork-child reset: a fresh lock, no turns, no declarations, no drift facts. A forked
        child's handles point at units the core has already dropped; it learns the rest again
        from its own events."""
        self._lock = threading.RLock()
        self._book._at_fork_reinit()
        self._seen.clear()
        self._schema_drift.clear()
        self._dropped.clear()
        self._unreached.clear()

    def name(self) -> str:
        return "hermes_agent"

    def install(self, client: object | None = None, ctx: AdapterContext | None = None) -> None:
        """Route Hermes's callbacks here. `self._installed = True` stays the last line.

        In Hermes's plugin host process no model call is visible, so the adapter declines with
        HOST_ISOLATION_LINE as its detail. Otherwise: follow tool calls (the MCP seam), then
        patch `_attach._ROUTE`; a refused patch undoes the follow and declines.
        """
        if self._installed or not isinstance(ctx, AdapterContext):
            return
        if os.environ.get(HOST_PROCESS_ENV) == "1":
            report_once(HOST_ISOLATION_LINE, key="adapters.hermes_agent.host_isolation")
            self.decline_reason = HOST_ISOLATION_LINE
            return
        self._ctx = ctx
        ctx.follow_tool_calls(_tool_call_of)
        if not ctx.patches.patch(_attach, "_ROUTE", self):
            ctx.unfollow_tool_calls()
            return
        self._installed = True

    def uninstall(self) -> None:
        """The route before the turns, so Hermes's callbacks go inert before any turn closes.
        Hermes keeps the callbacks; they find `_ROUTE` empty and leave its turns alone.

        The tool-call follow is undone BEFORE the installed check: an install that raised at the
        route patch had registered it with the flag still down, and the registry's rollback is
        this call.

        A turn still live is closed by the core with ADAPTER_UNINSTALLED, not by `_close`, so
        what it asked for and Hermes never ran, and whether another plugin replaced one of its
        requests, is stated here first."""
        ctx = self._ctx
        if ctx is not None:
            ctx.unfollow_tool_calls()
        if not self._installed or ctx is None:
            return
        self._installed = False
        ctx.patches.restore_all()
        ctx.release_unreached_carriers(None)
        with self._lock:
            self._unreached.clear()
        for record in self._book.live():
            with record.lock:
                undispatched = record.undispatched()
                notes = [o.marker for o in OBSERVATIONS if o.applies(record, None)]
                rewritten = record.request_rewritten
            if record.handle is not None:
                self._state_observed(record.handle, undispatched, notes, rewritten)
        self._book.clear()
        ctx.close_all(marker=Limitation.ADAPTER_UNINSTALLED)
        _attach._EXPORT.wake()

    def close_units(self, *, marker: Limitation) -> None:
        """The signal path: takes no lock of this adapter. A record whose unit this closes is
        found gone at its next event (UNIT_GONE)."""
        ctx = self._ctx
        if ctx is not None:
            ctx.close_all(marker=marker)

    def restate(self, filed: AdapterStatus) -> AdapterStatus:
        """What diagnostics() says now: whether a Hermes context attached, which contract
        versions Hermes stated, and what was dropped or refused."""
        refused = _attach.refused()
        if _attach.attached() == 0:
            detail = NOT_ATTACHED_DETAIL
            if refused:
                detail += "; Hermes refused the registration of " + ", ".join(refused)
            return replace(
                filed, state=AdapterState.UNSUPPORTED, measured=None, measured_versions=(),
                detail=detail,
            )  # fmt: skip
        with self._lock:
            seen = bool(self._seen)
            drift = dict(self._schema_drift)
            dropped = dict(self._dropped)
        measured, versions = filed.measured, filed.measured_versions
        clauses = [filed.detail] if filed.detail else []
        if drift:
            measured, versions = False, tuple(KNOWN_SCHEMAS.values())
            stated = ", ".join(drift.values())
            clauses.append(f"Hermes states {stated}, which this release was not measured on")
        elif seen:
            measured, versions = True, tuple(KNOWN_SCHEMAS.values())
        if dropped:
            each = ", ".join(f"{name} ({why})" for name, why in dropped.items())
            clauses.append(
                f"events dropped for missing fields: {each}; upgrade wardex-sdk or report it"
            )
        if refused:
            clauses.append("Hermes refused the registration of " + ", ".join(refused))
        return replace(
            filed, measured=measured, measured_versions=versions, detail="; ".join(clauses)
        )

    def needs_sweep(self) -> bool:
        """A turn is live or a declaration made with no live turn stands: the export thread
        keeps waking for the idle net."""
        with self._lock:
            if self._unreached:
                return True
        return bool(self._book.live())

    def reap_idle(self, now: float) -> None:
        """The IDLE net, from the export thread: every live turn quiet for its `quiet_for()`, and
        every declaration made with no live turn older than IDLE_CLOSE_S (its end was lost)."""
        ctx = self._ctx
        if ctx is None:
            return
        after = _turns.IDLE_CLOSE_S
        modes = [self._advance(r, Trigger.IDLE, None) for r in self._book.idle(now)]
        with self._lock:
            expired = [k for k, at in self._unreached.items() if now - at >= after]
            for key in expired:
                del self._unreached[key]
        for key in expired:
            ctx.release_unreached_carriers(key)
            ctx.count("unreached_expired")
        self._export(_merge(modes))

    # -- dispatch -------------------------------------------------------------------------------

    def on_hook(self, row: ContractRow, payload: Mapping[str, Any]) -> None:
        """Every hook row lands here, on Hermes's thread. Never raises."""
        ctx = self._ctx
        if ctx is None:
            return
        with ctx.guard(f"hook.{row.name}"):
            event = self._parse(row, payload)
            if event is not None:
                out = self._dispatch(event)
                self._export(out if isinstance(out, ExportMode) else None)

    def on_frame(
        self, row: ContractRow, once: Callable[[], Any], payload: Mapping[str, Any]
    ) -> None:
        """Every middleware row lands here: one model call or one tool call, on Hermes's thread.

        The frame's unit is pinned for exactly the call, so the socket tee's attempts (every
        503 and 500 included) and any thread the call starts nest under it. It is unpinned in
        the `finally`, on the same task, so a pooled tool worker carries nothing to its next
        job. `once` is Hermes's `next_call`: its outcome reaches Hermes untouched (`_Once`).
        """
        ctx = self._ctx
        if ctx is None:
            return
        frame = None
        with ctx.guard(f"frame.{row.name}.enter"):
            event = self._parse(row, payload)
            if event is not None:
                out = self._dispatch(event)
                frame = out if isinstance(out, FrameRecord) else None
        error_type = None
        try:
            once()
        except BaseException as exc:
            error_type = _downstream_name(exc)
            raise
        finally:
            if frame is not None:
                with ctx.guard(f"frame.{row.name}.exit"):
                    self._exit(frame, error_type)

    def _parse(self, row: ContractRow, payload: Mapping[str, Any]) -> Event | None:
        event, drift = parse(row, payload)
        if drift is not None:
            self._drift(row, drift)
        elif row.schema_field is not None and row.schema_field not in self._seen:
            with self._lock:
                self._seen.add(row.schema_field)
        return event

    def _drift(self, row: ContractRow, note: DriftNote) -> None:
        """One line per schema field and per dropped row, kept for `restate`. Recording goes
        on for a schema drift; a field drift has already dropped its event."""
        ctx = self._ctx
        if note.kind is Drift.SCHEMA:
            field = row.schema_field or row.name
            with self._lock:
                first = field not in self._schema_drift
                if first:
                    self._schema_drift[field] = note.detail
            if first:
                report_once(
                    f"wardex: Hermes states {note.detail} on {row.name}, which this wardex "
                    "release was not measured on. Hermes turns are still recorded and "
                    "diagnostics() reads measured=False; if they look wrong, upgrade wardex-sdk "
                    "or report it with wardex_sdk.diagnostics().report()",
                    key=f"adapters.hermes_agent.schema_drift.{field}",
                )
            return
        if ctx is not None:
            ctx.count(f"dropped.{row.name}")
            ledger_record(
                DropReason.FRAMEWORK_EVENT_UNMATCHED,
                unit=DropUnit.HOOK_EVENT,
                where=f"adapters.hermes_agent.dropped.{row.name}",
            )
        with self._lock:
            first = row.name not in self._dropped
            if first:
                self._dropped[row.name] = note.detail
        if first:
            report_once(
                f"wardex: Hermes sent {row.name} with {note.detail}, so wardex dropped that "
                "event and its turn may be incomplete. This wardex release does not expect that "
                "payload: upgrade wardex-sdk, or report it with "
                "wardex_sdk.diagnostics().report() and your Hermes version",
                key=f"adapters.hermes_agent.dropped.{row.name}",
            )

    def _export(self, mode: ExportMode | None) -> None:
        if mode is ExportMode.DRAIN:
            _attach._EXPORT.drain(FINALIZE_DRAIN_S)
        elif mode is ExportMode.KICK:
            _attach._EXPORT.kick()

    def _dispatch(self, event: Event) -> _Out:
        """Route one event to the turn or turns it names and run their steps.

        * A session-keyed event (finalize, subagent_stop, a turnless on_session_end): every live
          turn of that session, and none for key None. A finalize at shutdown always drains, so
          a turn that closed just before it is not left in the buffer either; any other
          finalize kicks the export thread, so no session boundary waits on export.
        * pre_llm_call: first the NEXT_TURN net over the session's other live turns, on the
          hook thread Hermes calls before it builds the turn's title thread, then the turn. A
          turn of another session is never closed here, whatever thread it ran on: nothing
          Hermes sends says whether it is over (`_beside`).
        * Everything else: the one turn the key names.

        A tool event's `conversation_id`, the only place Hermes states one, is kept for the
        session's later turns first (`TurnBook.conversation_for`).
        """
        if isinstance(event, (ToolEnter, ToolReport)):
            if event.session_id and event.conversation_id:
                self._book.state_conversation(event.session_id, event.conversation_id)
        trigger = event.trigger()
        if event.key_kind() is KeyKind.SESSION:
            key = event.key()
            live = self._book.live_in_session(key) if key is not None else []
            mode = _merge([self._advance(r, trigger, event) for r in live])
            return ExportMode.DRAIN if trigger is Trigger.SHUTDOWN else mode
        if isinstance(event, TurnStart):
            others = [
                r
                for r in self._book.live_in_session(event.session_id)
                if r.turn_id != event.turn_id
            ]
            swept = _merge([self._advance(r, Trigger.NEXT_TURN, event) for r in others])
            return _merge([swept, self._by_key(event, trigger)])
        return self._by_key(event, trigger)

    def _by_key(self, event: Event, trigger: Trigger) -> _Out:
        key = event.key()
        if key is None:  # Hermes named no turn: an auxiliary call or a tool outside any turn
            return self._run(None, event, step(None, trigger), trigger)
        record = self._book.find(key)
        if record is None:
            return self._resolve(key, event, trigger)
        if isinstance(event, CallEnter) and event.session_id not in (None, record.session_id):
            # Only a model call's frame tells: Hermes compresses right before one, and an
            # auxiliary call's session id is the one its turn was admitted with.
            self._book.rotate(record, event.session_id)
        return self._advance(record, trigger, event)

    def _advance(self, record: TurnRecord, trigger: Trigger, event: Event | None) -> _Out:
        """Step a live record. One whose unit the core no longer holds (closed with its parent,
        evicted, closed by the signal path) is retired first, so the event resolves against its
        PRESUMED_CLOSED tombstone: a loop event continues the turn, anything else is late."""
        handle = record.handle
        if handle is not None and not handle.degraded and not handle.accepted:
            gone = record.advance(Trigger.UNIT_GONE)
            if gone.action is Action.FORGET:
                self._run(record, event, gone, trigger)
            return self._resolve(record.turn_id, event, trigger)
        s = record.advance(trigger)
        if s.action in CREATES:  # retired by another thread between its find and this step
            return self._resolve(record.turn_id, event, trigger)
        return self._run(record, event, s, trigger)

    def _resolve(self, turn_id: str, event: Event | None, trigger: Trigger) -> _Out:
        """A turn with no live record: its tombstone's step, and a new record when the step
        creates one (generation 0, or the tombstone's plus one)."""
        tomb = self._book.tombstone(turn_id)
        s = step(None if tomb is None else tomb.state, trigger)
        if s.action not in CREATES or event is None or s.next is None:
            return self._run(None, event, s, trigger)
        session_id = getattr(event, "session_id", None) or ""  # "" on a record Hermes sent none
        parent = getattr(event, "parent_session_id", None)
        tie = self._book.tie_for(session_id)
        child = parent is not None or tie is not None
        if tie is not None:
            parent = parent or tie.parent_session_id
        record = TurnRecord(
            session_id=session_id,
            turn_id=turn_id,
            generation=0 if tomb is None else tomb.generation + 1,
            kind=TurnKind.CHILD if child else TurnKind.ROOT,
            state=s.next,
            conversation=self._book.conversation_for(
                session_id, parent if child else None, tie.parent_turn_id if tie else None
            ),
            model_frames=0 if tomb is None else tomb.model_frames,
        )
        live = self._book.admit(record)
        if live is None:  # the live table is full: this event, and so its turn, go unrecorded
            self._ctx.count("book_full")  # type: ignore[union-attr]
            ledger_record(
                DropReason.TABLE_FULL, unit=DropUnit.HOOK_EVENT, where="adapters.hermes_agent.turns"
            )
            return None
        if live is not record:  # another thread admitted it first: step the winner instead
            s = live.advance(trigger)
            if s.action in CREATES:  # and the winner was retired meanwhile: the event is lost
                self._ctx.count("admit_race")  # type: ignore[union-attr]
                ledger_record(
                    DropReason.FRAMEWORK_EVENT_UNMATCHED,
                    unit=DropUnit.HOOK_EVENT,
                    where="adapters.hermes_agent.admit_race",
                )
                return None
            return self._run(live, event, s, trigger)
        return self._run(record, event, s, trigger)

    def _run(self, record: TurnRecord | None, event: Any, s: Step, trigger: Trigger) -> _Out:
        return self._effects[s.action](record, event, s, trigger)  # type: ignore[arg-type]

    # -- effects: turns -------------------------------------------------------------------------

    def _open(self, record: TurnRecord, event: TurnStart, s: Step, trigger: Trigger) -> _Out:
        """pre_llm_call, on a hook thread whose context is a copy of the loop thread's. ROOT
        takes whatever is ambient in that copy: a first turn or a gateway job, a root; the
        next turn on the same thread after the previous one closed, a clean root, since a
        closed turn's pin is retired; a joined child, the live delegate_task unit, so nested
        under it; a background child, or an agent a tool started on a thread it left running,
        the tool's unit already closed, which the core refuses, so a root with
        CORRELATION_CONFLICT. When the unit current there is another turn's own (a turn Hermes
        stopped with no end, or a live turn whose `step_callback` runs this agent) the turn
        opens beside it (`_beside`), and that turn is left as it is.

        With synchronous hooks (`plugins.hook_callback_timeout: 0`) this runs on the loop
        thread itself, where the previous turn's pin still stands: `_drop_lane` takes it down
        first, on the task that holds it, and that turn's next frame pins it again. On a hook
        thread it only clears the copy.

        The user's message is held, not recorded: another plugin's `pre_llm_call` may block the
        prompt, and Hermes then keeps it off every store and provider. It becomes the turn's
        input at the turn's first model-call frame or at a non-block end (`_release_input`).
        A continuation has no `pre_llm_call`, so no input."""
        within = _beside(record)
        _drop_lane()
        self._open_turn(record, event.platform, resumed=False, within=within)
        if event.user_message is not None:
            with record.lock:
                record.held_input = event.user_message
        return None

    def _answer(self, record: TurnRecord, event: TurnAnswer, s: Step, trigger: Trigger) -> _Out:
        """post_llm_call, on a hook thread before the end event: Hermes's final response
        becomes the turn's output, through the capped payload path. The turn stays open."""
        handle = record.handle
        if handle is not None and event.assistant_response is not None:
            budget = self._ctx.record_budget  # type: ignore[union-attr]
            handle.record_output(_shaped_payload(_message_text(event.assistant_response), budget))
        return None

    def _continue(self, record: TurnRecord, event: Any, s: Step, trigger: Trigger) -> _Out:
        """A guess was wrong: the loop is still running a presumed-closed turn. Generation n+1
        opens on this loop thread, linked RESUMED_FROM the closed generation, then the frame."""
        within = _beside(record)
        _drop_lane()
        self._open_turn(record, None, resumed=True, within=within)
        return self._frame(record, event, s, trigger)

    def _continue_close(
        self, record: TurnRecord, event: TurnEnd, s: Step, trigger: Trigger
    ) -> _Out:
        """Hermes ended a turn wardex presumed closed (the idle net closed it during a wait
        longer than its threshold, and no model frame came after). The earlier span cannot be
        reopened, so a short generation n+1 states Hermes's own outcome, linked RESUMED_FROM,
        and closes at once."""
        self._open_turn(record, None, resumed=True, within=_beside(record))
        return self._close(record, event, s, trigger)

    def _open_turn(
        self, record: TurnRecord, platform: str | None, *, resumed: bool, within: RestorePoint
    ) -> None:
        """Open the turn's unit here, or on the carriers `within` (`_beside`). The carriers it
        opened on are kept (`TurnRecord.beneath`): what later opens beside this turn goes there."""
        ctx = self._ctx
        assert ctx is not None
        tie = self._book.tie_for(record.session_id) if record.kind is TurnKind.CHILD else None
        delegate = _FRAME.get() if record.kind is TurnKind.CHILD else None
        session_id = record.session_id
        unit_kind = _UNIT_KIND[record.kind]
        shadowed = False

        def describe(h: RunHandle) -> None:
            d = h.draft
            d.set_agent(AgentAttributes(name=AGENT_NAME, agent_type=_AGENT_TYPE[record.kind]))
            d.set_extra(PREFIX + "turn_id", record.turn_id)
            if record.generation:
                d.set_extra(PREFIX + "generation", record.generation)
            if platform:
                d.set_extra(PREFIX + "platform", platform)
            if shadowed:  # the host's conversation wins; Hermes's session id is kept beside it
                d.set_extra(PREFIX + "session_id", session_id)
            if tie is not None:
                if tie.child_role:
                    d.set_extra(PREFIX + "child_role", tie.child_role)
                if tie.child_subagent_id:
                    d.set_extra(PREFIX + "child_subagent_id", tie.child_subagent_id)
                if tie.parent_turn_id:
                    h.link(LinkReason.TRIGGERED_BY, UnitKey(TURN_NS, tie.parent_turn_id))
            if delegate is not None and not delegate.pinned():  # the delegate call has returned
                h.link(LinkReason.TRIGGERED_BY, UnitKey(TOOL_NS, delegate.key))
            if resumed:  # before the alias, so the key still names the closed generation
                h.link(LinkReason.RESUMED_FROM, UnitKey(TURN_NS, record.turn_id))
            h.alias(UnitKey(TURN_NS, record.turn_id), remember=True)

        def opened() -> RunHandle:
            nonlocal shadowed  # the host's conversation is read where the unit opens
            record.beneath = ambient_restore_point()
            conversation, shadowed = framework_conversation(
                ctx, record.conversation, shadowed_counter="conversation_shadowed"
            )
            return ctx.open_run(
                unit_kind,
                intent=SpanIntent.INVOKE_AGENT,
                placement=Placement.ROOT,
                subject=AGENT_NAME,
                conversation=conversation,
                describe=describe,
            )

        record.handle = _opened_on(within, opened)
        _attach._EXPORT.start()

    def _close(self, record: TurnRecord, event: Any, s: Step, trigger: Trigger) -> _Out:
        """The one close. Everything it states comes from CLOSE_RULES[cause] and OBSERVATIONS.

        The record is retired first, so an event that races the close resolves against the
        tombstone. Then: the auxiliary declarations still standing are released; a held input
        is recorded when Hermes's end says the prompt was not blocked, and dropped otherwise;
        each frame still pinned whose report is kept is closed with it; extras close_cause,
        exit_reason (verbatim), error_reason (the `reason` of an api_request_error that is the
        turn's last event), undispatched_tool_calls, aux_tasks (the auxiliary tasks Hermes
        named for this turn) and request_rewritten (another plugin replaced the request of one
        of its model calls); the rule's marker and each applying observation's; the close.
        """
        cause = s.cause or event.cause()
        rule = CLOSE_RULES[cause]
        self._book.retire(record, rule.next_state)
        record.beneath = ()  # only a live turn is opened beside; let its carriers go
        ended = isinstance(event, TurnEnd)
        with record.lock:
            record.blocked = ended and event.blocked()
            held, record.held_input = record.held_input, None
            error_reason = record.error_reason if record.after_error else None
            undispatched = record.undispatched()
            notes = [o.marker for o in OBSERVATIONS if o.applies(record, cause)]
            aux_tasks = record.aux_tasks[:AUX_TASK_NAMES]
            rewritten = record.request_rewritten
            pending = list(record.aux.items())
            record.aux.clear()
            kept = list(record.frames.values())
        ctx = self._ctx
        assert ctx is not None
        for key, attempt in pending:
            if attempt.draft is not None:
                ctx.release_unreached_carriers(("aux", *key))
        for frame in kept:
            if frame.take_kept():
                self._finish(frame)
        handle = record.handle
        if handle is None:
            return rule.export
        if ended and held is not None and not record.blocked:
            self._release_input(record, held)
        _drop_lane(record)
        d = handle.draft
        d.set_extra(PREFIX + "close_cause", rule.literal)
        if ended and event.exit_reason:
            d.set_extra(PREFIX + "exit_reason", event.exit_reason)
        if error_reason:
            d.set_extra(PREFIX + "error_reason", error_reason)
        if aux_tasks:
            d.set_extra(PREFIX + "aux_tasks", ",".join(aux_tasks))
        if rule.marker is not None:
            handle.note(rule.marker)
        self._state_observed(handle, undispatched, notes, rewritten)
        handle.close(status=rule.status, error_type=rule.error_type)
        return rule.export

    @staticmethod
    def _state_observed(
        handle: RunHandle, undispatched: list[str], notes: list[Limitation], rewritten: bool
    ) -> None:
        """What a turn about to close states however it closes: each marker `OBSERVATIONS`
        found, the names of the tool calls its model asked for that no frame explains, and that
        another plugin replaced the request of one of its model calls."""
        if rewritten:
            handle.draft.set_extra(PREFIX + "request_rewritten", True)
        if undispatched:
            handle.draft.set_extra(
                PREFIX + "undispatched_tool_calls", ",".join(undispatched[:UNDISPATCHED_NAMES])
            )
        for marker in notes:
            handle.note(marker)

    def _forget(self, record: TurnRecord, event: Any, s: Step, trigger: Trigger) -> _Out:
        """The core already closed the record's unit: retire it and call no verb on it. Its
        auxiliary declarations are withdrawn, since no close of ours will do it."""
        self._book.retire(record, TurnState.PRESUMED_CLOSED)
        with record.lock:
            pending = list(record.aux)
            record.aux.clear()
        for key in pending:
            self._ctx.release_unreached_carriers(("aux", *key))  # type: ignore[union-attr]
        return None

    def _tie(self, record: TurnRecord | None, event: ChildStart, s: Step, trigger: Trigger) -> _Out:
        """subagent_start, on the parent's tool worker: remember child session -> parent turn."""
        self._book.tie(
            ChildTie(
                event.child_session_id,
                event.parent_turn_id,
                event.child_subagent_id,
                event.child_role,
                event.parent_session_id,
            )
        )
        return None

    def _note_error(self, record: TurnRecord, event: CallError, s: Step, trigger: Trigger) -> _Out:
        """api_request_error: the turn stays open, because Hermes may retry, park it on its
        recovery ladder or fall back. Its reason is kept for the close, and the quiet net waits
        IDLE_AFTER_ERROR_S while this is the turn's last event. The call's frame gets it as its
        report (`_close_frame`)."""
        with record.lock:
            record.after_error = True
            record.error_reason = event.reason
        return self._close_frame(record, event, s, trigger)

    def _ignore(self, record: TurnRecord | None, event: Any, s: Step, trigger: Trigger) -> _Out:
        self._ctx.count(f"{s.action.value}.{trigger.value}")  # type: ignore[union-attr]
        return None
