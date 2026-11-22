"""Hermes Agent: what a frame and an auxiliary call record. The frame half of the effects.

Owns: the effects for model-call and tool frames (`FRAME`, `ORPHAN_FRAME`, `CLOSE_FRAME`) and for
auxiliary calls (`AUX_BEGIN`, `AUX_FINISH`, `AUX_DECLARE_UNREACHED`, `AUX_RELEASE_UNREACHED`), the
frame exit, the two task-local variables (`_LANE`, `_FRAME`) and the turn pin on a loop thread.
`HermesAgentAdapter` in `_hermes_agent.py` inherits them; the dispatcher, the turn effects and
`restate` are there.

A frame unit is pinned for exactly its call, on the thread Hermes runs the call on, and closed
once, by its exit or by its post hook's report, whichever `FrameRecord` says won. Nothing here
runs a wardex verb while a turn, frame or book lock is held.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Hashable
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from .._assembly import (
    GenAIAttributes,
    Limitation,
    LinkReason,
    SpanIntent,
    ToolAttributes,
    UnitKey,
    UnitKind,
    canonical_arguments,
)
from .._enums import OperationName, StatusCode
from . import _hermes_agent_attach as _attach
from ._context import AdapterContext, Placement, RunHandle
from ._hermes_agent_events import (
    AuxEnd,
    AuxStart,
    CallEnter,
    CallError,
    CallReport,
    ToolEnter,
    ToolReport,
)
from ._hermes_agent_turns import (
    Action,
    AuxAttempt,
    ExportMode,
    FrameKind,
    FrameRecord,
    Step,
    Trigger,
    TurnRecord,
)
from ._payload import _shaped_args, _shaped_payload

if TYPE_CHECKING:
    from .._assembly import SpanDraft

__all__ = ["CALL_NS", "HOLD_PREFIX", "PREFIX", "TOOL_NS", "FrameEffects"]

#: This adapter's extras namespace.
PREFIX = "wardex.hermes_agent."
#: Hermes-reported usage, demoted under this prefix. The core promotes it, with NO_WIRE_EVIDENCE,
#: only when no wire attempt was witnessed under the call, so one call is never counted twice.
HOLD_PREFIX = PREFIX + "reported."

#: Alias namespaces of the frame units. Every alias is remembered, so a closed unit stays a
#: link target.
CALL_NS = "hermes.api_request"  # api_request_id
TOOL_NS = "hermes.tool_call"  # tool_call_id

#: The turn pinned on THIS task. Set on the loop thread by a turn's first model frame. Hermes
#: copies it into every thread it starts from the loop thread afterwards, which is how a hook
#: thread answers "did this auxiliary call run inside its turn?" by identity, never by a guess.
_LANE: ContextVar[TurnRecord | None] = ContextVar("wardex_hermes_lane", default=None)
#: The tool frame running on THIS task. A delegated child's loop is started inside the
#: delegate_task frame, so the child's first event names the exact delegate call it came from.
_FRAME: ContextVar[FrameRecord | None] = ContextVar("wardex_hermes_frame", default=None)

#: What an effect hands the dispatcher: the frame it opened, the export its close asks for, or
#: nothing.
_Out = FrameRecord | ExportMode | None


def _downstream_name(exc: BaseException) -> str:
    """Hermes wraps a downstream failure as its own error raised from the original; the
    original's type is the one a reader needs. No Hermes class is imported to find it."""
    return type(exc.__cause__ or exc).__name__


def _chat_block(model: str | None) -> GenAIAttributes:
    """The block a `chat` span needs to export at all, and the core's usage promotion needs."""
    return GenAIAttributes(operation=OperationName.CHAT, request_model=model)


def _aux_key(event: AuxStart | AuxEnd) -> Hashable:
    return ("aux", *event.attempt())


def _message_text(value: object) -> object:
    """A turn's message as its text. Hermes hands a string, or a list of content parts when the
    message has attachments. Of the parts only the text is kept and any other part (an image, a
    file) is named by its type, so a base64 image does not fill the record up to its cap. A
    list that is not content parts is returned as it came, for the general shaper."""
    if type(value) is not list:
        return value
    lines = []
    for part in value:
        kind = part.get("type") if type(part) is dict else None
        text = part.get("text") if kind == "text" else None
        if type(text) is str:
            lines.append(text)
        elif type(kind) is str:
            lines.append(f"[{kind}]")
        else:
            return value
    return "\n".join(lines)


class FrameEffects:
    """The frame half of `HermesAgentAdapter`'s effects, one method per frame or auxiliary
    `Action`. It reads three attributes the adapter owns: `_ctx`, and `_lock` with `_unreached`,
    the declarations made with no live turn, which the idle sweep releases."""

    _ctx: AdapterContext | None
    _lock: threading.RLock
    _unreached: dict[Hashable, float]

    # -- effects: auxiliary calls ---------------------------------------------------------------

    def _aux_begin(self, record: TurnRecord, event: AuxStart, s: Step, trigger: Trigger) -> _Out:
        """pre_auxiliary_call for a live turn, on a hook thread copied from the calling thread.

        In the turn's lane (the copy carries this turn's `_LANE`, set by its first model frame
        on its loop thread) the call's wire span is already under the turn: keep the attempt
        and open nothing. Outside it (the turn-start title thread, built before any frame) open
        a hook-built `chat <model>` child of the turn, marked POSSIBLE_DUPLICATE_CHAT, and
        declare the unreached carrier, so the call's own wire span is marked unresolved.
        """
        key = event.attempt()
        with record.lock:
            duplicate = key in record.aux
            if not duplicate:
                record.aux[key] = AuxAttempt(event.aux_task)
                # A call in the turn's lane gets no span of ours, and its wire span cannot say
                # which task it served, so the turn names the tasks it ran.
                if event.aux_task and event.aux_task not in record.aux_tasks:
                    record.aux_tasks.append(event.aux_task)
        handle = record.handle
        if duplicate or handle is None or _LANE.get() is record:
            if duplicate:
                self._ctx.count("ignore_duplicate.aux_start")  # type: ignore[union-attr]
            return None
        draft = handle.child_draft(SpanIntent.CHAT, subject=event.model)
        draft.set_gen_ai(_chat_block(event.model))
        draft.set_extra(PREFIX + "aux_request_id", event.aux_request_id)
        if event.aux_task:
            draft.set_extra(PREFIX + "aux_task", event.aux_task)
        draft.add_limitation(Limitation.POSSIBLE_DUPLICATE_CHAT)
        self._ctx.expect_unreached_carriers(_aux_key(event))  # type: ignore[union-attr]
        with record.lock:
            attempt = record.aux.get(key)
            if attempt is not None:
                record.aux[key] = replace(attempt, draft=draft)
        return None

    def _aux_finish(self, record: TurnRecord, event: AuxEnd, s: Step, trigger: Trigger) -> _Out:
        """post_auxiliary_call for a live turn: close its hook-built span, if it has one, with
        Hermes's usage as held extras only (never typed: the wire span, when there is one,
        carries the usage), and withdraw the declaration."""
        with record.lock:
            attempt = record.aux.pop(event.attempt(), None)
        handle = record.handle
        if attempt is None or attempt.draft is None or handle is None:
            return None
        draft: SpanDraft = attempt.draft
        for key, n in event.usage:
            draft.set_extra(HOLD_PREFIX + key, n)
        if event.error_type:
            handle.close_child(draft, status=StatusCode.ERROR, error_type=event.error_type)
        else:
            handle.close_child(draft, status=StatusCode.OK)
        self._ctx.release_unreached_carriers(_aux_key(event))  # type: ignore[union-attr]
        return None

    def _aux_unreached(self, record: None, event: Any, s: Step, trigger: Trigger) -> _Out:
        """An auxiliary call for a turn that is not live (closed, presumed closed, unknown, or
        none at all): no span, only the declaration that marks its wire span, and its release.
        Kept with its time, so the idle sweep releases one whose end was lost."""
        ctx = self._ctx
        assert ctx is not None
        key = _aux_key(event)
        if s.action is Action.AUX_DECLARE_UNREACHED:
            ctx.expect_unreached_carriers(key)
            with self._lock:
                self._unreached[key] = time.monotonic()
            _attach._EXPORT.start()  # the idle sweep releases one whose end is lost
        else:
            ctx.release_unreached_carriers(key)
            with self._lock:
                self._unreached.pop(key, None)
        ctx.count(f"{s.action.value}.{trigger.value}")
        return None

    # -- effects: frames ------------------------------------------------------------------------

    def _frame(self, record: TurnRecord, event: Any, s: Step, trigger: Trigger) -> _Out:
        """Open, describe and pin one frame unit of a live turn on the calling task. The turn's
        first model-call frame is where its held input leaves the host, so it is recorded here.

        A tool frame Hermes named no `tool_call_id` for (one that code in `execute_code` called)
        is kept by no turn: no report can name it, so it closes at its exit."""
        if isinstance(event, ToolEnter):
            call_id = event.tool_call_id
            frame = FrameRecord(FrameKind.TOOL, call_id or "")
            if call_id:
                with record.lock:
                    record.dispatched.add(call_id)
                    record.frames[(FrameKind.TOOL, call_id)] = frame
            self._open_tool(frame, event)
            return frame
        _ensure_lane(record)
        frame = FrameRecord(FrameKind.MODEL_CALL, event.api_request_id)
        with record.lock:
            record.model_frames += 1
            record.frames[(FrameKind.MODEL_CALL, event.api_request_id)] = frame
            held, record.held_input = record.held_input, None
        if held is not None:
            self._release_input(record, held)
        self._open_call(frame, event)
        return frame

    def _orphan_frame(
        self, record: None, event: CallEnter | ToolEnter, s: Step, trigger: Trigger
    ) -> _Out:
        """A frame no live turn owns. It ran, so it gets a span on whatever is ambient
        (PARENT_UNRESOLVED from the core when nothing is), closed at frame exit, ERROR when it
        raised and UNSET otherwise: Hermes's report on it comes after the exit or never, and a
        frame that returned is not a verdict.

        A model call here belongs to a run Hermes never announced (its background memory review
        or a side question: forks that share the user's session id and send no `pre_llm_call`
        and no end) or to a turn that started before wardex registered. It is counted, and no
        turn is made for it. A tool here ran in Hermes's MCP child process for codex app-server
        mode, after a register mid-turn, or after a presumed close."""
        if isinstance(event, ToolEnter):
            frame = FrameRecord(FrameKind.TOOL, event.tool_call_id or "", orphan=True)
            self._open_tool(frame, event)
            return frame
        self._ctx.count("turn_start_missed")  # type: ignore[union-attr]
        frame = FrameRecord(FrameKind.MODEL_CALL, event.api_request_id, orphan=True)
        self._open_call(frame, event)
        return frame

    def _release_input(self, record: TurnRecord, message: object) -> None:
        """The turn's input: the user's message from `pre_llm_call`, through the capped payload
        path, once it is known to leave the host."""
        handle = record.handle
        if handle is not None:
            budget = self._ctx.record_budget  # type: ignore[union-attr]
            handle.record_input(_shaped_payload(_message_text(message), budget))

    def _open_call(self, frame: FrameRecord, event: CallEnter) -> None:
        """The logical call: CALL, CHAT, NESTED under the turn pinned on this loop thread. Every
        HTTP attempt of the frame, retries included, nests under it. An orphan states the turn
        id Hermes gave it, so a reader can group one fork's calls."""
        ctx = self._ctx
        assert ctx is not None
        model, request_id, api_mode = event.model, event.api_request_id, event.api_mode
        turn_id = event.turn_id if frame.orphan else None

        def describe(h: RunHandle) -> None:
            h.draft.set_gen_ai(_chat_block(model))
            h.draft.set_extra(PREFIX + "api_request_id", request_id)
            if turn_id:
                h.draft.set_extra(PREFIX + "turn_id", turn_id)
            if api_mode:
                h.draft.set_extra(PREFIX + "api_mode", api_mode)
            h.alias(UnitKey(CALL_NS, request_id), remember=True)

        handle = ctx.open_run(
            UnitKind.CALL,
            intent=SpanIntent.CHAT,
            placement=Placement.NESTED,
            subject=model,
            describe=describe,
        )
        if not handle.degraded:
            ctx.hold_framework_usage(handle, prefix=HOLD_PREFIX)
        handle.pin(driver=threading.current_thread())
        frame.handle = handle

    def _open_tool(self, frame: FrameRecord, event: ToolEnter) -> None:
        """`execute_tool <name>`: CALL, EXECUTE_TOOL, NESTED, linked TRIGGERED_BY the model call
        that asked for it, whether it ran alone, in a round or in a parallel batch. Its input
        goes through the capped payload path only, and its slot is what the MCP seam reads.

        Placed by context alone, never by an id: a tool that code in `execute_code` called runs
        on a thread Hermes starts with the code tool's context, so it nests under that tool; a
        prelude tool runs before the turn is ambient on any thread, so it is a root the core
        marks PARENT_UNRESOLVED. With no `api_request_id` there is no link, and the span states
        the turn id Hermes gave it instead, so a reader can still find its turn. With no
        `tool_call_id` there is no call id and no alias."""
        ctx = self._ctx
        assert ctx is not None
        name, call_id, asked_by = event.tool_name, event.tool_call_id, event.api_request_id
        turn_id = None if asked_by else event.turn_id
        args = event.args
        budget = ctx.record_budget
        expected = not frame.orphan

        def describe(h: RunHandle) -> None:
            h.draft.set_tool(ToolAttributes(name=name, call_id=call_id))
            if asked_by:
                h.link(LinkReason.TRIGGERED_BY, UnitKey(CALL_NS, asked_by), expected=expected)
            elif turn_id:
                h.draft.set_extra(PREFIX + "turn_id", turn_id)
            if call_id:
                h.alias(UnitKey(TOOL_NS, call_id), remember=True)

        handle = ctx.open_run(
            UnitKind.CALL,
            intent=SpanIntent.EXECUTE_TOOL,
            placement=Placement.NESTED,
            subject=name,
            describe=describe,
        )
        if args is not None:
            handle.record_input(_shaped_args(args, budget))
        slot = ctx.slot(frame)
        slot["tool_name"] = name
        slot["tool_arguments"] = canonical_arguments(args) if args is not None else None
        slot["handle"] = handle
        handle.pin(driver=threading.current_thread())
        frame.handle = handle
        frame.token = _FRAME.set(frame)

    def _exit(self, frame: FrameRecord, error_type: str | None) -> None:
        """Unpin and reset `_FRAME` on the frame's own task, then close it if this side won the
        race against its report (`FrameRecord.on_exit`)."""
        handle = frame.handle
        if handle is not None:
            handle.unpin()
        token = frame.token
        if token is not None:
            frame.token = None
            _FRAME.reset(token)  # type: ignore[arg-type]
        if frame.on_exit(error_type):
            self._finish(frame)

    def _close_frame(
        self,
        record: TurnRecord,
        event: CallReport | CallError | ToolReport,
        s: Step,
        trigger: Trigger,
    ) -> _Out:
        """post_api_request, api_request_error or post_tool_call, on a hook thread: hand the
        frame its report.

        An api_request_error after a frame that raised changes nothing; after one that returned
        (a refusal is an HTTP 200) it closes the call ERROR. A report while the frame is still
        pinned (the concurrent tool path) is kept and applied at exit, or at its turn's close if
        that comes first. A report that says Hermes abandoned the tool's worker (a timeout or a
        cancel, sent from the loop thread while the worker still runs) closes the frame at once,
        from this thread: the core retires the worker's pin from afar, and the worker's late
        exit then does nothing. A report with no frame (a missing middleware, a frame that raised
        and closed, or a report that names no `tool_call_id`) is counted."""
        if isinstance(event, ToolReport):
            key = (FrameKind.TOOL, event.tool_call_id or "")  # "" is never a kept frame's key
        else:
            key = (FrameKind.MODEL_CALL, event.api_request_id)
        with record.lock:
            if isinstance(event, CallReport):
                record.requested.update(event.requested)
            frame = record.frames.get(key)
        abandoned = isinstance(event, ToolReport) and event.abandoned()
        if frame is None:
            self._ctx.count("report_without_frame")  # type: ignore[union-attr]
        elif frame.on_report(event, abandoned=abandoned):
            self._finish(frame)
        return None

    def _finish(self, frame: FrameRecord) -> None:
        """Close a frame unit once, by whichever side won: a raise is ERROR with the downstream
        type; a frame no report can reach (an orphan, or a tool Hermes named no id for), or one
        whose report never came, is UNSET, since a frame that returned is not Hermes's verdict;
        otherwise the report decides, and an api_request_error is ERROR with the type Hermes
        stated."""
        handle = frame.handle
        if handle is None:
            return
        report = frame.kept_report
        if frame.error_type is not None:
            handle.close(status=StatusCode.ERROR, error_type=frame.error_type)
        elif isinstance(report, CallReport):
            for key, n in report.usage:
                handle.draft.set_extra(HOLD_PREFIX + key, n)
            handle.close(status=StatusCode.OK)
        elif isinstance(report, CallError):
            error_type = report.error_type or report.reason or "api_request_error"
            handle.close(status=StatusCode.ERROR, error_type=error_type)
        elif isinstance(report, ToolReport):
            self._close_tool(handle, report)
        else:
            handle.close(status=StatusCode.UNSET)
        if frame.kind is FrameKind.TOOL:
            self._ctx.forget(frame)  # type: ignore[union-attr]

    def _close_tool(self, handle: RunHandle, report: ToolReport) -> None:
        """Hermes returns a tool failure as a result and does not raise, so its own status
        decides: an error type is ERROR, "ok" is OK, anything else is UNSET with the status
        kept verbatim."""
        if report.result is not None:
            handle.record_output(_shaped_payload(report.result, self._ctx.record_budget))  # type: ignore[union-attr]
        if report.error_type:
            handle.close(status=StatusCode.ERROR, error_type=report.error_type)
        elif report.status == "ok":
            handle.close(status=StatusCode.OK)
        else:
            if report.status:
                handle.draft.set_extra(PREFIX + "tool_status", report.status)
            handle.close(status=StatusCode.UNSET)


def _ensure_lane(record: TurnRecord) -> None:
    """Make `record` the turn pinned on this loop thread: where a turn becomes ambient, and how
    a later turn on the same thread takes the ended turn's pin down on its owning task."""
    if _LANE.get() is record:
        return
    _drop_lane()
    handle = record.handle
    if handle is not None:
        handle.pin(driver=threading.current_thread())
    _LANE.set(record)


def _drop_lane(only: TurnRecord | None = None) -> None:
    """Take down the previous lane's pin if this task holds it, so nothing opened next on this
    thread nests under a turn that is over. With `only`, only when that turn is the lane."""
    lane = _LANE.get()
    if lane is None or (only is not None and lane is not only):
        return
    handle = lane.handle
    if handle is not None and handle.pinned_here:
        handle.unpin()
    _LANE.set(None)
