"""The Hermes turn lifecycle as data: what each contract event does to a turn.

Owns: the closed alphabets of the lifecycle (`TurnState`, `Trigger`, `Action`, `CloseCause`), the
two tables that are the lifecycle (`TRANSITIONS`, `CLOSE_RULES`), the close-time observations
(`OBSERVATIONS`), the frame lifecycle (`FrameState`), the per-turn and per-frame records, and
`TurnBook`, the adapter's one shared table.

Knows no Hermes field name (that is `_hermes_agent_events.py`, the boundary) and calls no wardex
verb (that is the effects module). A transition is a dictionary lookup, so the whole lifecycle,
late, duplicated, missing and out-of-order events included, is testable without threads, without
Hermes and without a client.

Concurrency, stated once. Three locks, never two held at once:

* `TurnBook._lock` guards the book's three dictionaries and nothing else. No record lock is
  taken while it is held, and no record lock is held while it is taken.
* `TurnRecord.lock` guards that record's state and fields. `TurnRecord.advance` is the one
  transition: it looks the step up, writes the new state, and releases the lock before it
  returns. The thread that got the step back is the winner and runs the step's effect after the
  lock is released, so a wardex verb never runs under it. Two threads cannot win the same
  transition, because the second one reads the state the first one wrote.
* `FrameRecord.lock` decides one race: a frame's exit (its own task) against its report (a hook
  thread). `on_exit` and `on_report` return True to exactly one of the two, which then closes
  the unit after the lock is released.

Nothing is counted or reported while any of these locks is held. That is what keeps a signal
handler safe: an embedder's `init()` installs handlers on the main thread, the main thread can be
Hermes's loop thread, and a handler that waited on a lock the interrupted frame holds would hang.

Nets, stated once. Hermes sends no end event for a failed turn (measured), and a killed process
sends nothing. Only `pre_llm_call` opens a turn: Hermes runs internal agents under the user's
session id that announce no start and no end (the background memory review, side questions),
and a frame of a turn never seen open is an orphan frame, not a turn. An `api_request_error`
closes nothing, because Hermes may still recover (its retry loop, its recovery ladder, a fallback
provider). Every close that wardex decides on a guess lands in PRESUMED_CLOSED, never in a final
state, so a loop event that proves the guess wrong continues the turn instead of being lost.
Only `on_session_finalize` for the turn's session cuts a turn for good.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypeVar

from .._assembly import Limitation
from .._enums import StatusCode

if TYPE_CHECKING:
    from .._assembly import SpanDraft
    from ._context import RunHandle

__all__ = [
    "CLOSE_RULES",
    "AUX_ATTEMPT_BOUND_S",
    "CREATES",
    "IDLE_AFTER_ERROR_S",
    "IDLE_CLOSE_S",
    "LIVE",
    "OBSERVATIONS",
    "TRANSITIONS",
    "Action",
    "AuxAttempt",
    "ChildTie",
    "CloseCause",
    "CloseRule",
    "ExportMode",
    "FrameKind",
    "FrameRecord",
    "FrameState",
    "Observation",
    "Step",
    "Tombstone",
    "Trigger",
    "TurnBook",
    "TurnKind",
    "TurnRecord",
    "TurnState",
    "step",
]

#: A live turn with no pinned frame, no auxiliary attempt in flight and no event for this long
#: is presumed dead. It is below Hermes's auto-started gateway idle exit (600 s by default), so
#: the CLI's gateway exports a turn Hermes never ended before it exits. Long work is safe: model
#: calls and tools hold a frame pinned for as long as they run.
IDLE_CLOSE_S = 120.0
#: The same net while the turn's loop has done nothing since an `api_request_error`: Hermes waits
#: before its next attempt (Hermes v0.23.3 source): an ordinary retry at most 90 s (2 s doubling
#: to 60 s, plus up to half again), a provider overload backoff at most 144 s, a recovery-ladder
#: cycle at most 120 s (the provider's Retry-After up to 120 s, else 15/30/60/60/60 s plus up to
#: a fifth). An ordinary retry honours a Retry-After up to 600 s, so one above this still splits
#: the turn into two linked generations. Under the gateway's 600 s idle exit, after which nothing
#: is exported.
IDLE_AFTER_ERROR_S = 300.0
#: An auxiliary attempt older than this no longer keeps its turn from being quiet: Hermes skips
#: a hook callback while the previous one for the same key still runs, so an end can be lost.
#: Above Hermes's 300 s floor on a compression call's timeout (other auxiliary calls default to
#: 30 s), under the gateway's 600 s idle exit.
AUX_ATTEMPT_BOUND_S = 360.0


# -- alphabets ---------------------------------------------------------------------------------


class TurnState(Enum):
    """Where one Hermes turn generation stands. `None` (no record and no tombstone) is the sixth
    state and is spelled `None` in the table."""

    OPEN = "open"  # span open since pre_llm_call, not yet ambient on its loop task
    AMBIENT = "ambient"  # pinned on its loop task by the first llm_execution frame
    CLOSED = "closed"  # Hermes said it ended (on_session_end). Final.
    PRESUMED_CLOSED = "presumed_closed"  # closed on a guess; a loop event continues it
    CUT = "cut"  # closed by shutdown (on_session_finalize). Final.


class Trigger(Enum):
    """The lifecycle's input alphabet: one member per thing that can happen to a turn.

    NEXT_TURN, IDLE and UNIT_GONE are derived by the caller, not sent by Hermes. NEXT_TURN comes
    from the session's next turn (a newer `pre_llm_call` of the same session) or the loop
    thread's (a root turn's `pre_llm_call` on the thread the turn is pinned on, while Hermes's
    last word on it is a failed call), IDLE from the clock. FINALIZE and SHUTDOWN
    are both `on_session_finalize` for the turn's session, told apart by Hermes's reason.
    UNIT_GONE says that the record's unit is no longer live in the core (closed with its parent,
    evicted, or closed by the signal path): the caller steps it before the event that found the
    record, so that event resolves against PRESUMED_CLOSED.
    """

    TURN_START = "turn_start"  # pre_llm_call
    CALL_ENTER = "call_enter"  # llm_execution frame entered (loop task)
    CALL_REPORT = "call_report"  # post_api_request
    CALL_FAILED = "call_failed"  # api_request_error: Hermes may still retry or recover
    TOOL_ENTER = "tool_enter"  # tool_execution frame entered (tool worker)
    TOOL_REPORT = "tool_report"  # post_tool_call
    AUX_START = "aux_start"  # pre_auxiliary_call
    AUX_END = "aux_end"  # post_auxiliary_call
    CHILD_START = "child_start"  # subagent_start, applied to the parent's turn
    TURN_ANSWER = "turn_answer"  # post_llm_call
    TURN_END = "turn_end"  # on_session_end
    NEXT_TURN = "next_turn"  # net: the session's or the loop thread's next turn
    CHILD_STOP = "child_stop"  # net: subagent_stop for the child's session
    FINALIZE = "finalize"  # net: on_session_finalize at a session boundary (/new, reset, close)
    SHUTDOWN = "shutdown"  # net: on_session_finalize at process shutdown
    IDLE = "idle"  # net: no event for IDLE_CLOSE_S (IDLE_AFTER_ERROR_S after an error) while quiet
    UNIT_GONE = "unit_gone"  # derived: the core no longer holds the record's unit


#: Events that prove Hermes's loop went on after a failed model call: its next attempt's frame,
#: a tool it ran, a call's report, the turn's answer and its end. Only these end the wait after
#: an error. An auxiliary call, a child's start and a late tool report run on other threads while
#: the loop sleeps out a Retry-After, and a net is not an event of the turn at all.
_LOOP_WENT_ON: frozenset[Trigger] = frozenset(
    {
        Trigger.CALL_ENTER,
        Trigger.TOOL_ENTER,
        Trigger.CALL_REPORT,
        Trigger.TURN_ANSWER,
        Trigger.TURN_END,
    }
)


class Action(Enum):
    """What the effects module does for a step: one effect per member."""

    OPEN = "open"  # open_run the turn (the NEXT_TURN sweep has already run)
    CONTINUE = "continue"  # open the next generation, linked RESUMED_FROM the presumed-closed one
    CONTINUE_CLOSE = "continue_close"  # Hermes ended a presumed-closed turn: say so, then close
    FRAME = "frame"  # open and pin a frame unit; pin the turn on this task first if it is not
    #: A frame no live turn owns: open on whatever is ambient, close at frame exit. A model-call
    #: frame of a turn never seen open counts `turn_start_missed` and opens no turn.
    ORPHAN_FRAME = "orphan_frame"
    CLOSE_FRAME = "close_frame"  # hand a frame its post hook's report (`FrameRecord.on_report`)
    #: A live turn's auxiliary attempt starts. Keep one `AuxAttempt` under `attempt()`; a second
    #: start under the same key is a duplicate and does nothing. When the call ran outside the
    #: turn's lane, open a hook-built `chat <model>` child draft of the turn, marked
    #: POSSIBLE_DUPLICATE_CHAT, and declare the unreached carrier so the call's own wire span is
    #: marked unresolved. In the lane, keep the attempt with no draft: its wire span is already
    #: under the turn.
    AUX_BEGIN = "aux_begin"
    #: A live turn's auxiliary attempt ends. Pop its `AuxAttempt`; if it holds a draft, close the
    #: draft with Hermes's usage as held extras (ERROR with Hermes's `error_type` when stated) and
    #: release the attempt's unreached carrier.
    AUX_FINISH = "aux_finish"
    #: An auxiliary attempt starts for a turn that is closed, presumed closed or unknown. Open no
    #: span: declare the unreached carrier only, so the call's wire span is marked, and count.
    AUX_DECLARE_UNREACHED = "aux_declare_unreached"
    #: An auxiliary attempt ends for a turn that is closed, presumed closed or unknown. Release
    #: its unreached carrier, and count. Releasing a key that was never declared does nothing,
    #: so an end whose start was seen while the turn was live is safe here.
    AUX_RELEASE_UNREACHED = "aux_release_unreached"
    TIE = "tie"  # remember child session -> parent turn
    NOTE_ERROR = "note_error"  # keep Hermes's reason; the quiet net waits IDLE_AFTER_ERROR_S
    ANSWER = "answer"  # record the turn's final response as the turn's output
    CLOSE = "close"  # close the turn per CLOSE_RULES[cause]
    #: The core already closed the record's unit. Retire the record to a PRESUMED_CLOSED
    #: tombstone and call no verb, so the event that found it continues the turn or is ignored.
    FORGET = "forget"
    IGNORE_DUPLICATE = "ignore_duplicate"  # counted
    IGNORE_LATE = "ignore_late"  # counted; a middleware frame still calls next_call
    IGNORE_UNKNOWN = "ignore_unknown"  # counted: a turn this process never saw open


#: Actions whose effect needs a new `TurnRecord` (generation 0, or the tombstone's plus one).
CREATES: frozenset[Action] = frozenset({Action.OPEN, Action.CONTINUE, Action.CONTINUE_CLOSE})


class CloseCause(Enum):
    """Why a turn closed. Each member is one row of `CLOSE_RULES`."""

    ENDED_COMPLETED = "ended_completed"
    ENDED_INTERRUPTED = "ended_interrupted"
    ENDED_FAILED = "ended_failed"
    ENDED_INCOMPLETE = "ended_incomplete"
    NEXT_TURN = "next_turn"
    CHILD_STOPPED = "child_stopped"
    IDLE = "idle"
    FINALIZE = "finalize"
    SHUTDOWN = "shutdown"


class ExportMode(Enum):
    KICK = "kick"  # wake the export thread; the calling hook returns at once
    DRAIN = "drain"  # flush with a bound on the calling thread (process shutdown only)


@dataclass(frozen=True, slots=True)
class CloseRule:
    """Everything a close states, in one row: no close site decides any of it by itself."""

    status: StatusCode
    error_type: str | None
    marker: Limitation | None
    next_state: TurnState
    export: ExportMode
    literal: str  # the value written as `wardex.hermes_agent.close_cause`


def _rule(next_state: TurnState, literal: str, export: ExportMode = ExportMode.KICK) -> CloseRule:
    # A close rule's marker is always passed as `marker=`: the limitation census reads keyword
    # slots, so the member is recorded here, where it is decided, not only where a close forwards
    # it.
    return CloseRule(
        StatusCode.UNSET,
        None,
        marker=Limitation.SESSION_ABORTED,
        next_state=next_state,
        export=export,
        literal=literal,
    )


_S = TurnState

#: How a turn ends. Only Hermes's own verdict gives OK or ERROR: UNSET means Hermes said nothing
#: about the outcome and wardex will not make one up. Every turn that did not complete carries
#: SESSION_ABORTED. A guess lands in PRESUMED_CLOSED; a finalize for the session cuts, and only
#: the one at process shutdown waits on export.
CLOSE_RULES: dict[CloseCause, CloseRule] = {
    CloseCause.ENDED_COMPLETED: CloseRule(
        StatusCode.OK,
        None,
        marker=None,
        next_state=_S.CLOSED,
        export=ExportMode.KICK,
        literal="end_event",
    ),
    CloseCause.ENDED_FAILED: CloseRule(
        StatusCode.ERROR,
        "failed",
        marker=Limitation.SESSION_ABORTED,
        next_state=_S.CLOSED,
        export=ExportMode.KICK,
        literal="end_event",
    ),
    CloseCause.ENDED_INTERRUPTED: _rule(_S.CLOSED, "end_event"),
    CloseCause.ENDED_INCOMPLETE: _rule(_S.CLOSED, "end_event"),
    CloseCause.NEXT_TURN: _rule(_S.PRESUMED_CLOSED, "next_turn"),
    CloseCause.CHILD_STOPPED: _rule(_S.PRESUMED_CLOSED, "subagent_stop"),
    CloseCause.IDLE: _rule(_S.PRESUMED_CLOSED, "idle"),
    CloseCause.FINALIZE: _rule(_S.CUT, "finalize"),
    CloseCause.SHUTDOWN: _rule(_S.CUT, "finalize", ExportMode.DRAIN),
}


@dataclass(frozen=True, slots=True)
class Step:
    next: TurnState | None
    action: Action
    cause: CloseCause | None = None  # None on CLOSE or CONTINUE_CLOSE: from the TURN_END event


# -- the table ---------------------------------------------------------------------------------

_T = Trigger
_A = Action

#: The states a turn is open in.
LIVE: tuple[TurnState, ...] = (_S.OPEN, _S.AMBIENT)
_NOT_LIVE: tuple[TurnState | None, ...] = (None, _S.PRESUMED_CLOSED, _S.CLOSED, _S.CUT)

#: (state, trigger) -> step. A pair missing from the table falls to `_DEFAULT[state]`, which is
#: always an IGNORE_* action: an event the table does not name is counted, never acted on.
TRANSITIONS: dict[tuple[TurnState | None, Trigger], Step] = {
    (None, _T.TURN_START): Step(_S.OPEN, _A.OPEN),
    # a run Hermes never announced (a review or side-question fork, a register mid-turn)
    (None, _T.CALL_ENTER): Step(None, _A.ORPHAN_FRAME),
    # a guess closed it; the loop proves the guess wrong by coming back
    (_S.PRESUMED_CLOSED, _T.CALL_ENTER): Step(_S.AMBIENT, _A.CONTINUE),
    (_S.PRESUMED_CLOSED, _T.TURN_END): Step(_S.CLOSED, _A.CONTINUE_CLOSE),
}
for _live in LIVE:
    TRANSITIONS.update(
        {
            (_live, _T.TURN_START): Step(_live, _A.IGNORE_DUPLICATE),
            (_live, _T.CALL_ENTER): Step(_S.AMBIENT, _A.FRAME),
            (_live, _T.TOOL_ENTER): Step(_live, _A.FRAME),
            (_live, _T.CALL_REPORT): Step(_live, _A.CLOSE_FRAME),
            (_live, _T.TOOL_REPORT): Step(_live, _A.CLOSE_FRAME),
            (_live, _T.CALL_FAILED): Step(_live, _A.NOTE_ERROR),
            (_live, _T.AUX_START): Step(_live, _A.AUX_BEGIN),  # the turn-start title lands here
            (_live, _T.AUX_END): Step(_live, _A.AUX_FINISH),
            (_live, _T.CHILD_START): Step(_live, _A.TIE),
            (_live, _T.TURN_ANSWER): Step(_live, _A.ANSWER),
            (_live, _T.TURN_END): Step(_S.CLOSED, _A.CLOSE),
            (_live, _T.UNIT_GONE): Step(_S.PRESUMED_CLOSED, _A.FORGET),
            # the nets close a live turn the same way, whatever it was doing
            (_live, _T.NEXT_TURN): Step(_S.PRESUMED_CLOSED, _A.CLOSE, CloseCause.NEXT_TURN),
            (_live, _T.CHILD_STOP): Step(_S.PRESUMED_CLOSED, _A.CLOSE, CloseCause.CHILD_STOPPED),
            (_live, _T.IDLE): Step(_S.PRESUMED_CLOSED, _A.CLOSE, CloseCause.IDLE),
            (_live, _T.FINALIZE): Step(_S.CUT, _A.CLOSE, CloseCause.FINALIZE),
            (_live, _T.SHUTDOWN): Step(_S.CUT, _A.CLOSE, CloseCause.SHUTDOWN),
        }
    )
# For a turn that is not live: a child's tie is a fact about the parent, kept in any state. A
# tool that starts after its turn ended still ran, so it gets a span of its own, marked by the
# core. An auxiliary call still has a wire span to mark. A model frame of an ended turn needs
# nothing: the socket tee records its attempts wherever the context puts them.
for _gone in _NOT_LIVE:
    TRANSITIONS.update(
        {
            (_gone, _T.CHILD_START): Step(_gone, _A.TIE),
            (_gone, _T.TOOL_ENTER): Step(_gone, _A.ORPHAN_FRAME),
            (_gone, _T.AUX_START): Step(_gone, _A.AUX_DECLARE_UNREACHED),
            (_gone, _T.AUX_END): Step(_gone, _A.AUX_RELEASE_UNREACHED),
        }
    )

_DEFAULT: dict[TurnState | None, Step] = {
    None: Step(None, _A.IGNORE_UNKNOWN),
    _S.OPEN: Step(_S.OPEN, _A.IGNORE_DUPLICATE),
    _S.AMBIENT: Step(_S.AMBIENT, _A.IGNORE_DUPLICATE),
    _S.PRESUMED_CLOSED: Step(_S.PRESUMED_CLOSED, _A.IGNORE_LATE),
    _S.CLOSED: Step(_S.CLOSED, _A.IGNORE_LATE),
    _S.CUT: Step(_S.CUT, _A.IGNORE_LATE),
}


def step(state: TurnState | None, trigger: Trigger) -> Step:
    """The one lookup every event goes through. Pure, total, never raises."""
    return TRANSITIONS.get((state, trigger), _DEFAULT[state])


# -- frames ------------------------------------------------------------------------------------


class FrameKind(Enum):
    MODEL_CALL = "model_call"  # one llm_execution frame: one logical call, its retries inside
    TOOL = "tool"  # one tool_execution frame


class FrameState(Enum):
    """A frame unit's own lifecycle. Its report can come either side of the frame's exit:
    `post_tool_call` lands after the exit on Hermes's sequential path and inside the frame on
    the concurrent path (measured). A pinned unit is never closed from another thread, so a
    report that comes early is kept and applied at exit.

    PINNED -> (frame returns) AWAITING_REPORT -> (post hook) CLOSED
    PINNED -> (post hook, concurrent path) REPORTED -> (frame returns) CLOSED with that report
    REPORTED -> (its turn closes first) CLOSED with that report, from the closing thread
    PINNED -> (post hook saying Hermes abandoned the worker) CLOSED at once with that report;
              the worker's late exit does nothing
    PINNED or REPORTED -> (frame raises) CLOSED, ERROR with the downstream exception's type; a
              turn's model-call frame also takes the turn's pin down on its task, since a
              turn that failed for good sends no end, and Hermes's next attempt pins it again
    AWAITING_REPORT -> (its turn closes first) closed by the core with CHILD_SPAN_UNCLOSED
    A frame no report can reach (`reportable()` False) goes PINNED -> CLOSED at frame exit,
    UNSET unless it raised.
    """

    PINNED = "pinned"
    REPORTED = "reported"  # still pinned, its report already kept
    AWAITING_REPORT = "awaiting_report"
    CLOSED = "closed"


_PINNED = (FrameState.PINNED, FrameState.REPORTED)


@dataclass(eq=False)
class FrameRecord:
    """One frame unit, keyed in its turn by (kind, api_request_id or tool_call_id).

    No `slots=True` and `eq=False`, both on purpose: the record is the identity key of the
    frame's `AdapterContext.slot`, a weak-keyed table, which is how the MCP seam finds a live
    tool unit. Python 3.10 cannot weak-reference a slotted dataclass.

    `lock` guards `state`, `error_type` and `kept_report`. Once `on_exit` or `on_report` has
    returned True, the frame is CLOSED, nothing writes those fields again, and the winner reads
    them without the lock while it closes the unit.
    """

    kind: FrameKind
    key: str  # "" when Hermes named no id for it
    handle: RunHandle | None = None  # published by the effect that opened the unit
    orphan: bool = False  # no live turn owns it
    token: object | None = None  # the frame ContextVar token, reset at exit on the same task
    state: FrameState = FrameState.PINNED
    error_type: str | None = None  # the downstream exception's type when the frame raised
    kept_report: object | None = None  # the post hook's event, when it came before the exit
    #: Reentrant like every lock in the SDK: a weakref finalizer can run wardex code on a thread
    #: that holds it.
    lock: threading.RLock = field(default_factory=threading.RLock)

    def on_exit(self, error_type: str | None) -> bool:
        """The frame returned (`error_type` None) or raised. True: the caller closes the unit."""
        with self.lock:
            if self.state is FrameState.CLOSED or self.state is FrameState.AWAITING_REPORT:
                return False
            if error_type is None and self.reportable() and self.state is FrameState.PINNED:
                self.state = FrameState.AWAITING_REPORT
                return False
            self.error_type = error_type
            self.state = FrameState.CLOSED
            return True

    def on_report(self, report: object, *, abandoned: bool = False) -> bool:
        """The post hook reported. True: the caller closes the unit with `report`.

        While pinned the report is kept and False is returned: the exit closes it. When the
        report says Hermes abandoned the worker, its exit may never come, so True at once. A
        second report, or one after the close, is False and changes nothing.
        """
        with self.lock:
            if self.state is FrameState.PINNED and self.reportable():
                self.kept_report = report
                self.state = FrameState.CLOSED if abandoned else FrameState.REPORTED
                return abandoned
            if self.state is FrameState.AWAITING_REPORT:
                self.kept_report = report
                self.state = FrameState.CLOSED
                return True
            return False

    def take_kept(self) -> bool:
        """The frame's turn is closing while the frame is still pinned with its report kept.
        True: the caller closes the unit with that report, from its own thread."""
        with self.lock:
            if self.state is FrameState.REPORTED:
                self.state = FrameState.CLOSED
                return True
            return False

    def pinned(self) -> bool:
        return self.state in _PINNED

    def reportable(self) -> bool:
        """A post hook can find it: a live turn keeps it under the id Hermes named. Otherwise
        nothing will report on it, and its exit closes it."""
        return not self.orphan and bool(self.key)


# -- turns -------------------------------------------------------------------------------------


class TurnKind(Enum):
    ROOT = "root"  # a turn a user or a gateway message started: UnitKind.SESSION
    CHILD = "child"  # a delegated subagent's turn (`parent_session_id` stated): UnitKind.AGENT


@dataclass(frozen=True, slots=True)
class ChildTie:
    """What `subagent_start` said, kept until the child's own `pre_llm_call` reads it."""

    child_session_id: str
    parent_turn_id: str | None  # None: the parent had no turn running, so nothing to link to
    child_subagent_id: str | None = None
    child_role: str | None = None
    parent_session_id: str | None = None  # whose conversation the child's turns are in


@dataclass(frozen=True, slots=True)
class AuxAttempt:
    """One auxiliary attempt in flight, from `pre_auxiliary_call` to `post_auxiliary_call`.

    `draft` is the hook-built child span when the call ran outside the turn's lane, and None
    when it ran in the lane, where its wire span is already under the turn. `started` is when
    its start was seen: past AUX_ATTEMPT_BOUND_S the attempt no longer keeps its turn busy.
    """

    task: str | None
    draft: SpanDraft | None = None
    started: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class Tombstone:
    """What is kept of a retired record: enough to continue the turn or to ignore it."""

    state: TurnState
    generation: int
    model_frames: int  # carried into a continuation: a turn's frames span its generations


@dataclass(eq=False)
class TurnRecord:
    """One live turn generation. `lock` guards every field after it.

    Mutate the containers only under `lock`; `quiet()` and `undispatched()` iterate them and
    are called under it too. `session_id` is the one field before `lock` that changes: only
    `TurnBook.rotate` writes it, and only under the book's lock, which `live_in_session`
    reads it under.
    """

    session_id: str
    turn_id: str
    generation: int  # 0; each continuation adds one
    kind: TurnKind
    state: TurnState
    #: The conversation the turn's span states: Hermes's `conversation_id` when an earlier
    #: event stated it for this session, a delegated child's parent's, else the session id.
    conversation: str = ""
    lock: threading.RLock = field(default_factory=threading.RLock)
    handle: RunHandle | None = None  # published by the effect that opened the unit
    frames: dict[tuple[FrameKind, str], FrameRecord] = field(default_factory=dict)
    #: api_request_id -> the tool names that model call asked for, in its order, and the tool
    #: names whose frame named that call. Compared per call by count, see `undispatched`.
    requested: dict[str, tuple[str, ...]] = field(default_factory=dict)
    dispatched: dict[str, list[str]] = field(default_factory=dict)
    model_frames: int = 0
    aux: dict[tuple[str, int], AuxAttempt] = field(default_factory=dict)  # in flight
    aux_tasks: list[str] = field(default_factory=list)  # task names Hermes stated, first seen first
    #: The turn's last loop event was an api_request_error: Hermes may be waiting to retry, so
    #: the quiet net waits IDLE_AFTER_ERROR_S. `error_reason` is that event's `reason`. Only an
    #: event in `_LOOP_WENT_ON` clears both; any other still refreshes `last_event`.
    after_error: bool = False
    error_reason: str | None = None
    #: The user's message from `pre_llm_call`, held until it is known to leave the host: the
    #: turn's first model-call frame, or an end event that is not another plugin's block.
    held_input: object = None
    blocked: bool = False  # Hermes ended the turn because another plugin blocked its prompt
    last_event: float = field(default_factory=time.monotonic)

    def advance(self, trigger: Trigger, now: float | None = None) -> Step:
        """The transition: look up, write the new state, release, return. The caller that gets
        the step runs its effect after this returns, with no lock of this record held."""
        with self.lock:
            s = step(self.state, trigger)
            if s.next is not None:
                self.state = s.next
            self.last_event = time.monotonic() if now is None else now
            if trigger in _LOOP_WENT_ON:
                self.after_error = False
                self.error_reason = None
            return s

    def undispatched(self) -> list[str]:
        """Names of tool calls a model call asked for that never reached a frame, by count per
        call: Hermes rewrites ids between report and dispatch (a reused id, a provider's ids in a
        parallel batch) and names too (a repair, the deferred-tool bridge). A shortfall names the
        asked-for names no frame ran, in asked order, cut to the shortfall: a repaired name can
        make it name the wrong one of two, never the wrong number."""
        missing: list[str] = []
        for call, asked in self.requested.items():
            ran = self.dispatched.get(call, [])
            if len(ran) >= len(asked):
                continue
            left = Counter(ran)
            unmatched = []
            for name in asked:
                if left[name]:
                    left[name] -= 1
                else:
                    unmatched.append(name)
            missing += unmatched[: len(asked) - len(ran)]
        return missing

    def quiet(self, now: float) -> bool:
        """No frame pinned and no auxiliary attempt in flight within its bound: what the idle
        net reads."""
        bound = AUX_ATTEMPT_BOUND_S
        if any(now - a.started < bound for a in self.aux.values()):
            return False
        return not any(f.pinned() for f in self.frames.values())

    def quiet_for(self) -> float:
        """How long the idle net waits on this turn, read at call time so a test can shorten it."""
        return IDLE_AFTER_ERROR_S if self.after_error else IDLE_CLOSE_S


# -- what a close observes ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    """A fact a close states about the whole turn, computed from the record and the close's
    cause, never guessed."""

    name: str
    marker: Limitation
    applies: Callable[[TurnRecord, CloseCause], bool]


def _completed_without_a_seen_call(r: TurnRecord, cause: CloseCause) -> bool:
    # Hermes says the turn completed and no llm_execution frame ran: the codex app-server
    # runtime, whose model calls happen in its own subprocess. A turn another plugin blocked
    # made no model call at all, so nothing went unobserved.
    return cause is CloseCause.ENDED_COMPLETED and r.model_frames == 0 and not r.blocked


def _asked_for_a_tool_call_that_never_ran(r: TurnRecord, cause: CloseCause) -> bool:
    # True for any cause: an invented tool name, a malformed batch Hermes answered with error
    # results, and a call an interrupt cut before dispatch all leave the same trace, a model
    # call that asked for more tool calls than ran under it.
    return bool(r.undispatched())


#: Close-time observations, in the order their markers are noted. `marker=` by keyword, for the
#: same reason as in `_rule`.
OBSERVATIONS: tuple[Observation, ...] = (
    Observation(
        "model_calls_unobserved",
        marker=Limitation.SUBPROCESS_MODEL_CALLS_UNOBSERVED,
        applies=_completed_without_a_seen_call,
    ),
    Observation(
        "tool_calls_not_dispatched",
        marker=Limitation.TOOL_CALL_NOT_DISPATCHED,
        applies=_asked_for_a_tool_call_that_never_ran,
    ),
)


# -- the book ----------------------------------------------------------------------------------


class TurnBook:
    """The adapter's only shared table: live records, tombstones, child ties, conversations.

    Bounded by `cap` in each of the four. A full live table admits nothing: `admit` returns
    None and the caller counts it. Tombstones and ties drop their oldest write first. A record
    leaves the live table when its turn closes (`retire`), when the adapter uninstalls
    (`clear`), or at fork (`_at_fork_reinit`: a forked child's handles point at units the core
    has already reset).

    Every method takes `_lock` for its own dictionary work only and returns a snapshot, so no
    caller ever holds it while it takes a record lock or runs an effect.
    """

    def __init__(self, *, cap: int = 4096) -> None:
        self._cap = cap
        self._lock = threading.RLock()
        self._live: dict[str, TurnRecord] = {}
        self._tombs: OrderedDict[str, Tombstone] = OrderedDict()
        self._ties: OrderedDict[str, ChildTie] = OrderedDict()
        #: Session id -> the conversation id Hermes stated for it, or one a turn of it took
        #: from its parent or kept through a rotation. Only an id that differs is kept.
        self._conversations: OrderedDict[str, str] = OrderedDict()

    def find(self, turn_id: str) -> TurnRecord | None:
        with self._lock:
            return self._live.get(turn_id)

    def tombstone(self, turn_id: str) -> Tombstone | None:
        """The retired generation's state, generation and frame count, if one is kept."""
        with self._lock:
            return self._tombs.get(turn_id)

    def admit(self, record: TurnRecord) -> TurnRecord | None:
        """Publish a new record and return the live one: `record` itself when it won, the
        record another thread admitted first when it lost, None when the table is full. A loser
        runs no effect of its own; it steps the winner instead."""
        with self._lock:
            live = self._live.get(record.turn_id)
            if live is not None:
                return live
            if len(self._live) >= self._cap:
                return None
            self._live[record.turn_id] = record
            return record

    def retire(self, record: TurnRecord, final: TurnState) -> None:
        """Move a closed record to a tombstone. Call it once the record's state is not live,
        when nothing writes its counters any more."""
        tomb = Tombstone(final, record.generation, record.model_frames)
        with self._lock:
            if self._live.get(record.turn_id) is record:
                del self._live[record.turn_id]
            _put(self._tombs, record.turn_id, tomb, self._cap)

    def live(self) -> list[TurnRecord]:
        """Every live record, in admission order."""
        with self._lock:
            return list(self._live.values())

    def live_in_session(self, session_id: str) -> list[TurnRecord]:
        """Live records of one session, in admission order."""
        with self._lock:
            return [r for r in self._live.values() if r.session_id == session_id]

    def idle(self, now: float) -> list[TurnRecord]:
        """Live records quiet for their `quiet_for()`. Each record is read under its own lock,
        after the book's lock is released."""
        quiet = []
        for r in self.live():
            with r.lock:
                if r.state in LIVE and r.quiet(now) and now - r.last_event >= r.quiet_for():
                    quiet.append(r)
        return quiet

    def tie(self, tie: ChildTie) -> None:
        with self._lock:
            _put(self._ties, tie.child_session_id, tie, self._cap)

    def tie_for(self, child_session_id: str) -> ChildTie | None:
        """Not popped: every turn of a child session links to the same parent turn."""
        with self._lock:
            return self._ties.get(child_session_id)

    def conversation_for(
        self, session_id: str, parent_session_id: str | None, parent_turn_id: str | None
    ) -> str:
        """The conversation a new turn of `session_id` opens in, never one Hermes contradicted.

        Hermes's `conversation_id` (`agent/prompt_cache_scope.py`) is a delegated child's
        parent's, else the root of the session's compression lineage. No payload at or before
        `pre_llm_call` states it, so a turn takes, in order: the id Hermes stated on an earlier
        tool event of this session; for a child, the id stated for its parent's session, else
        the conversation its parent turn's span states, else the parent's session id; else the
        session id, which is the lineage root until Hermes compresses the session."""
        with self._lock:
            known = self._conversations.get(session_id)
            if known is not None:
                return known
            if parent_session_id is None:
                return session_id
            found = self._conversations.get(parent_session_id)
            if found is None:
                parent = self._live.get(parent_turn_id) if parent_turn_id else None
                found = parent.conversation if parent is not None else parent_session_id
            if found and found != session_id:
                _put(self._conversations, session_id, found, self._cap)
            return found or session_id

    def state_conversation(self, session_id: str, conversation_id: str) -> None:
        """Hermes stated `conversation_id` for `session_id`: later turns of it take that."""
        with self._lock:
            if self._conversations.get(session_id, session_id) != conversation_id:
                _put(self._conversations, session_id, conversation_id, self._cap)

    def rotate(self, record: TurnRecord, session_id: str) -> None:
        """Hermes moved a live turn to a new session id mid-turn (a context compression,
        `agent/conversation_compression.py`): the session's finalize and its next turn name
        the new id, and the conversation stays the lineage's."""
        with self._lock:
            record.session_id = session_id
            if record.conversation and record.conversation != session_id:
                _put(self._conversations, session_id, record.conversation, self._cap)

    def clear(self) -> None:
        """Uninstall: forget everything. The core closes the units themselves."""
        with self._lock:
            self._live.clear()
            self._tombs.clear()
            self._ties.clear()
            self._conversations.clear()

    def _at_fork_reinit(self) -> None:
        # The child has one thread, and the parent's lock may have been held at the fork.
        self._lock = threading.RLock()
        self._live.clear()
        self._tombs.clear()
        self._ties.clear()
        self._conversations.clear()


_V = TypeVar("_V")


def _put(table: OrderedDict[str, _V], key: str, value: _V, cap: int) -> None:
    table[key] = value
    table.move_to_end(key)
    while len(table) > cap:
        table.popitem(last=False)
