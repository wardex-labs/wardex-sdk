"""The Hermes turn lifecycle tables, the records and the turn book.

These are table tests: the tables are the lifecycle, so the tables are what is asserted. No
Hermes, no client, no exported span. What a user sees from these tables is asserted on exported
traces by the adapter's scope tests.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable

import pytest

from wardex_sdk._adapters._hermes_agent_turns import (
    CLOSE_RULES,
    CREATES,
    IDLE_CLOSE_S,
    LIVE,
    OBSERVATIONS,
    TRANSITIONS,
    Action,
    AuxAttempt,
    ChildTie,
    CloseCause,
    ExportMode,
    FrameKind,
    FrameRecord,
    FrameState,
    Step,
    Trigger,
    TurnBook,
    TurnKind,
    TurnRecord,
    TurnState,
    step,
)
from wardex_sdk._assembly import Limitation
from wardex_sdk._enums import StatusCode

STATES: tuple[TurnState | None, ...] = (None, *TurnState)
FINAL = (TurnState.CLOSED, TurnState.CUT)
NOT_LIVE = (None, TurnState.PRESUMED_CLOSED, *FINAL)
_T = Trigger
_A = Action


def _record(turn_id: str = "t-1", session_id: str = "s-1", **kw: object) -> TurnRecord:
    fields: dict[str, object] = {"generation": 0, "kind": TurnKind.ROOT, "state": TurnState.OPEN}
    fields.update(kw)
    return TurnRecord(session_id, turn_id, **fields)  # type: ignore[arg-type]


def _free_elsewhere(lock: threading.RLock, wait: float = 1.0) -> bool:
    """Whether another thread can take `lock` now. A reentrant lock is always free to the thread
    that holds it, so the probe runs on a thread of its own."""
    got: list[bool] = []

    def probe() -> None:
        taken = lock.acquire(timeout=wait)
        if taken:
            lock.release()
        got.append(taken)

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join(wait + 5)
    return got == [True]


def _walk(start: TurnState | None, *triggers: Trigger) -> list[tuple[Action, TurnState | None]]:
    """Feed triggers through the table the way the effects module does, state by state."""
    state, path = start, []
    for t in triggers:
        s = step(state, t)
        path.append((s.action, s.next))
        state = s.next
    return path


# -- the tables are total ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "trigger"),
    list(itertools.product(STATES, Trigger)),
    ids=lambda v: getattr(v, "value", "none"),
)
def test_every_state_answers_every_trigger(state: TurnState | None, trigger: Trigger) -> None:
    s = step(state, trigger)
    assert isinstance(s, Step)
    if s.action is Action.CLOSE and trigger is not Trigger.TURN_END:
        assert s.cause is not None
    if s.action in (Action.CLOSE, Action.CONTINUE_CLOSE) and s.cause is None:
        assert trigger is Trigger.TURN_END


def test_a_record_never_steps_to_no_state() -> None:
    for state, trigger in itertools.product(TurnState, Trigger):
        assert step(state, trigger).next is not None


def test_a_close_lands_where_its_rule_says() -> None:
    for (state, trigger), s in TRANSITIONS.items():
        if s.action is Action.CLOSE and s.cause is not None:
            assert CLOSE_RULES[s.cause].next_state is s.next, (state, trigger)
    ended = [c for c in CloseCause if c.value.startswith("ended_")]
    for live in LIVE:
        assert step(live, Trigger.TURN_END).next is TurnState.CLOSED
    assert {CLOSE_RULES[c].next_state for c in ended} == {TurnState.CLOSED}


def test_every_cause_has_a_rule() -> None:
    assert set(CLOSE_RULES) == set(CloseCause)


def test_every_action_is_reachable() -> None:
    reached = {s.action for s in TRANSITIONS.values()}
    reached |= {step(state, t).action for state, t in itertools.product(STATES, Trigger)}
    assert reached == set(Action)


def test_new_records_are_made_only_where_no_live_record_exists() -> None:
    for state, trigger in itertools.product(STATES, Trigger):
        if step(state, trigger).action in CREATES:
            assert state in (None, TurnState.PRESUMED_CLOSED), (state, trigger)


def test_final_states_never_reopen() -> None:
    """A turn Hermes ended or shutdown cut never becomes live again; only a guess is undone."""
    for state, trigger in itertools.product(FINAL, Trigger):
        assert step(state, trigger).next in FINAL


def test_every_guess_is_undoable() -> None:
    guesses = (_T.CALL_FAILED_FINAL, _T.NEXT_TURN, _T.CHILD_STOP, _T.IDLE, _T.UNIT_GONE)
    for state, trigger in itertools.product(LIVE, guesses):
        assert step(state, trigger).next is TurnState.PRESUMED_CLOSED
    assert step(TurnState.PRESUMED_CLOSED, _T.CALL_ENTER).action is Action.CONTINUE
    assert step(TurnState.PRESUMED_CLOSED, _T.TURN_END).action is Action.CONTINUE_CLOSE


# -- close rules -------------------------------------------------------------------------------

#: The design's close table: status, error.type, marker, next state, close_cause literal.
_EXPECTED_RULES = {
    CloseCause.ENDED_COMPLETED: (StatusCode.OK, None, None, TurnState.CLOSED, "end_event"),
    CloseCause.ENDED_FAILED: (
        StatusCode.ERROR,
        "failed",
        Limitation.SESSION_ABORTED,
        TurnState.CLOSED,
        "end_event",
    ),
    CloseCause.ENDED_INTERRUPTED: (
        StatusCode.UNSET,
        None,
        Limitation.SESSION_ABORTED,
        TurnState.CLOSED,
        "end_event",
    ),
    CloseCause.ENDED_INCOMPLETE: (
        StatusCode.UNSET,
        None,
        Limitation.SESSION_ABORTED,
        TurnState.CLOSED,
        "end_event",
    ),
    CloseCause.RETRIES_SPENT: (
        StatusCode.UNSET,
        None,
        Limitation.SESSION_ABORTED,
        TurnState.PRESUMED_CLOSED,
        "retries_spent",
    ),
    CloseCause.NEXT_TURN: (
        StatusCode.UNSET,
        None,
        Limitation.SESSION_ABORTED,
        TurnState.PRESUMED_CLOSED,
        "next_turn",
    ),
    CloseCause.CHILD_STOPPED: (
        StatusCode.UNSET,
        None,
        Limitation.SESSION_ABORTED,
        TurnState.PRESUMED_CLOSED,
        "subagent_stop",
    ),
    CloseCause.IDLE: (
        StatusCode.UNSET,
        None,
        Limitation.SESSION_ABORTED,
        TurnState.PRESUMED_CLOSED,
        "idle",
    ),
    CloseCause.FINALIZE: (
        StatusCode.UNSET,
        None,
        Limitation.SESSION_ABORTED,
        TurnState.CUT,
        "finalize",
    ),
}


@pytest.mark.parametrize("cause", list(CloseCause), ids=lambda c: c.value)
def test_close_rule_states_only_what_hermes_stated(cause: CloseCause) -> None:
    r = CLOSE_RULES[cause]
    assert (r.status, r.error_type, r.marker, r.next_state, r.literal) == _EXPECTED_RULES[cause]


def test_only_shutdown_drains_on_the_calling_thread() -> None:
    drains = {c for c, r in CLOSE_RULES.items() if r.export is ExportMode.DRAIN}
    assert drains == {CloseCause.FINALIZE}


def test_no_close_claims_an_interrupt_marker() -> None:
    """UNIT_INTERRUPTED belongs to the runtime's own teardown; a Hermes interrupt is a turn that
    did not complete."""
    assert all(r.marker is not Limitation.UNIT_INTERRUPTED for r in CLOSE_RULES.values())


# -- required lifecycle paths ------------------------------------------------------------------


def test_normal_end() -> None:
    assert _walk(None, _T.TURN_START, _T.CALL_ENTER, _T.CALL_REPORT, _T.TURN_END) == [
        (_A.OPEN, TurnState.OPEN),
        (_A.FRAME, TurnState.AMBIENT),
        (_A.CLOSE_FRAME, TurnState.AMBIENT),
        (_A.CLOSE, TurnState.CLOSED),
    ]


def test_interrupt_closes_by_the_end_event() -> None:
    path = _walk(TurnState.AMBIENT, _T.TOOL_ENTER, _T.TURN_END, _T.TOOL_REPORT)
    assert path == [
        (_A.FRAME, TurnState.AMBIENT),
        (_A.CLOSE, TurnState.CLOSED),
        (_A.IGNORE_LATE, TurnState.CLOSED),
    ]


def test_failed_turn_with_no_end_event_is_presumed_closed() -> None:
    path = _walk(TurnState.AMBIENT, _T.CALL_FAILED, _T.CALL_FAILED_FINAL)
    assert path == [(_A.NOTHING, TurnState.AMBIENT), (_A.CLOSE, TurnState.PRESUMED_CLOSED)]
    assert step(TurnState.AMBIENT, _T.CALL_FAILED_FINAL).cause is CloseCause.RETRIES_SPENT


@pytest.mark.parametrize(
    ("trigger", "cause"),
    [
        (_T.NEXT_TURN, CloseCause.NEXT_TURN),
        (_T.IDLE, CloseCause.IDLE),
        (_T.CHILD_STOP, CloseCause.CHILD_STOPPED),
        (_T.FINALIZE, CloseCause.FINALIZE),
    ],
    ids=["next-turn", "idle", "child-stop", "finalize"],
)
def test_nets_close_a_live_turn(trigger: Trigger, cause: CloseCause) -> None:
    for live in LIVE:
        s = step(live, trigger)
        assert (s.action, s.cause, s.next) == (_A.CLOSE, cause, CLOSE_RULES[cause].next_state)


def test_nets_leave_a_turn_that_is_not_live_alone() -> None:
    for state, trigger in itertools.product(
        (TurnState.PRESUMED_CLOSED, *FINAL), (_T.NEXT_TURN, _T.IDLE, _T.CHILD_STOP, _T.FINALIZE)
    ):
        assert step(state, trigger) == Step(state, _A.IGNORE_LATE)


def test_continuation_after_a_presumed_close() -> None:
    path = _walk(TurnState.AMBIENT, _T.CALL_FAILED_FINAL, _T.CALL_ENTER, _T.TURN_END)
    assert path == [
        (_A.CLOSE, TurnState.PRESUMED_CLOSED),
        (_A.CONTINUE, TurnState.AMBIENT),
        (_A.CLOSE, TurnState.CLOSED),
    ]


def test_end_event_after_a_presumed_close() -> None:
    assert _walk(TurnState.AMBIENT, _T.NEXT_TURN, _T.TURN_END) == [
        (_A.CLOSE, TurnState.PRESUMED_CLOSED),
        (_A.CONTINUE_CLOSE, TurnState.CLOSED),
    ]


def test_duplicated_start_changes_nothing() -> None:
    assert _walk(None, _T.TURN_START, _T.TURN_START) == [
        (_A.OPEN, TurnState.OPEN),
        (_A.IGNORE_DUPLICATE, TurnState.OPEN),
    ]
    assert step(TurnState.AMBIENT, _T.TURN_START) == Step(TurnState.AMBIENT, _A.IGNORE_DUPLICATE)


@pytest.mark.parametrize("state", [TurnState.PRESUMED_CLOSED, *FINAL], ids=lambda s: s.value)
def test_late_report_is_ignored(state: TurnState) -> None:
    for trigger in (_T.TOOL_REPORT, _T.CALL_REPORT, _T.CALL_FAILED, _T.TURN_START):
        assert step(state, trigger) == Step(state, _A.IGNORE_LATE)


def test_report_for_a_turn_never_seen_is_unknown() -> None:
    for trigger in (_T.TOOL_REPORT, _T.CALL_REPORT, _T.TURN_END, _T.CALL_FAILED_FINAL):
        assert step(None, trigger) == Step(None, _A.IGNORE_UNKNOWN)


@pytest.mark.parametrize("state", NOT_LIVE, ids=lambda s: getattr(s, "value", "none"))
def test_tool_frame_with_no_live_turn_still_gets_a_span(state: TurnState | None) -> None:
    assert step(state, _T.TOOL_ENTER) == Step(state, _A.ORPHAN_FRAME)


def test_the_answer_is_kept_by_a_live_turn_and_moves_it_nowhere() -> None:
    for live in LIVE:
        assert step(live, _T.TURN_ANSWER) == Step(live, _A.ANSWER)
    for state in (TurnState.PRESUMED_CLOSED, *FINAL):
        assert step(state, _T.TURN_ANSWER) == Step(state, _A.IGNORE_LATE)
    assert step(None, _T.TURN_ANSWER) == Step(None, _A.IGNORE_UNKNOWN)


def test_a_missing_start_opens_late() -> None:
    assert step(None, _T.CALL_ENTER) == Step(TurnState.AMBIENT, _A.OPEN_LATE)


def test_a_gone_unit_is_presumed_closed_at_its_next_event() -> None:
    for live in LIVE:
        assert step(live, _T.UNIT_GONE) == Step(TurnState.PRESUMED_CLOSED, _A.FORGET)
    assert _walk(TurnState.AMBIENT, _T.UNIT_GONE, _T.CALL_ENTER) == [
        (_A.FORGET, TurnState.PRESUMED_CLOSED),
        (_A.CONTINUE, TurnState.AMBIENT),
    ]
    assert _walk(TurnState.OPEN, _T.UNIT_GONE, _T.TURN_END)[-1] == (
        _A.CONTINUE_CLOSE,
        TurnState.CLOSED,
    )
    assert _walk(TurnState.AMBIENT, _T.UNIT_GONE, _T.TOOL_REPORT)[-1] == (
        _A.IGNORE_LATE,
        TurnState.PRESUMED_CLOSED,
    )


def test_a_gone_unit_on_a_record_steps_through_advance() -> None:
    r = _record(state=TurnState.AMBIENT)
    assert r.advance(_T.UNIT_GONE).action is _A.FORGET
    assert r.state is TurnState.PRESUMED_CLOSED
    assert r.advance(_T.IDLE).action is _A.IGNORE_LATE


def test_auxiliary_attempts_by_turn_state() -> None:
    for live in LIVE:
        assert step(live, _T.AUX_START) == Step(live, _A.AUX_BEGIN)
        assert step(live, _T.AUX_END) == Step(live, _A.AUX_FINISH)
    for state in NOT_LIVE:
        assert step(state, _T.AUX_START) == Step(state, _A.AUX_DECLARE_UNREACHED)
        assert step(state, _T.AUX_END) == Step(state, _A.AUX_RELEASE_UNREACHED)


def test_a_child_tie_is_kept_in_every_state() -> None:
    for state in STATES:
        assert step(state, _T.CHILD_START) == Step(state, _A.TIE)


# -- observations ------------------------------------------------------------------------------


def _applying(record: TurnRecord, cause: CloseCause) -> set[Limitation]:
    return {o.marker for o in OBSERVATIONS if o.applies(record, cause)}


def test_completed_turn_with_no_model_frame_says_its_calls_were_unobserved() -> None:
    marker = Limitation.SUBPROCESS_MODEL_CALLS_UNOBSERVED
    assert marker in _applying(_record(), CloseCause.ENDED_COMPLETED)
    assert marker not in _applying(_record(model_frames=1), CloseCause.ENDED_COMPLETED)
    for cause in set(CloseCause) - {CloseCause.ENDED_COMPLETED}:
        assert marker not in _applying(_record(), cause), cause


def test_turn_that_asked_for_a_tool_call_that_never_ran_says_so_whatever_closed_it() -> None:
    marker = Limitation.TOOL_CALL_NOT_DISPATCHED
    asked = _record()
    asked.requested.update({"call_1": "read_file", "call_2": "lab_no_such_tool"})
    asked.dispatched.add("call_1")
    ran_all = _record()
    ran_all.requested["call_1"] = "read_file"
    ran_all.dispatched.add("call_1")
    for cause in CloseCause:
        assert marker in _applying(asked, cause), cause
        assert marker not in _applying(ran_all, cause), cause
        assert marker not in _applying(_record(), cause), cause


# -- records -----------------------------------------------------------------------------------


def test_undispatched_compares_tool_call_ids_not_names() -> None:
    r = _record()
    r.requested.update({"call_1": "read_file", "call_2": "lab_no_such_tool", "call_3": "tool_call"})
    r.dispatched.update({"call_1", "call_3"})  # call_3 ran under another name, same id
    assert r.undispatched() == {"call_2": "lab_no_such_tool"}
    r.dispatched.add("call_2")
    assert r.undispatched() == {}


def test_quiet_only_with_nothing_pinned_and_no_auxiliary_attempt_in_flight() -> None:
    r = _record()
    assert r.quiet()
    frame = FrameRecord(FrameKind.TOOL, "call_1")
    r.frames[(FrameKind.TOOL, "call_1")] = frame
    assert not r.quiet()
    assert not frame.on_report("report")  # concurrent path: reported, still pinned
    assert not r.quiet()
    assert frame.on_exit(None)
    assert r.quiet()
    r.aux[("aux-1", 0)] = AuxAttempt(task="title_generation")
    assert not r.quiet()
    del r.aux[("aux-1", 0)]
    assert r.quiet()


def test_a_frame_awaiting_its_report_is_quiet() -> None:
    r = _record()
    frame = FrameRecord(FrameKind.MODEL_CALL, "s-1:api:1")
    r.frames[(FrameKind.MODEL_CALL, "s-1:api:1")] = frame
    assert not frame.on_exit(None)
    assert frame.state is FrameState.AWAITING_REPORT
    assert r.quiet()


def test_advance_writes_state_and_time_and_releases_the_lock() -> None:
    r = _record()
    s = r.advance(_T.CALL_ENTER, now=42.0)
    assert (s.action, r.state, r.last_event) == (_A.FRAME, TurnState.AMBIENT, 42.0)
    assert _free_elsewhere(r.lock)


def _race(n: int, fn: Callable[[], object]) -> list[object]:
    barrier = threading.Barrier(n)
    results: list[object] = []
    results_lock = threading.Lock()

    def run() -> None:
        barrier.wait()
        value = fn()
        with results_lock:
            results.append(value)

    threads = [threading.Thread(target=run) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_exactly_one_thread_wins_a_close() -> None:
    r = _record(state=TurnState.AMBIENT)
    actions = _race(8, lambda: r.advance(_T.TURN_END).action)
    assert actions.count(_A.CLOSE) == 1
    assert actions.count(_A.IGNORE_LATE) == 7


@pytest.mark.parametrize("report_first", [True, False], ids=["report-first", "exit-first"])
def test_a_frame_closes_once_whichever_side_comes_second(report_first: bool) -> None:
    frame = FrameRecord(FrameKind.TOOL, "call_1")
    if report_first:
        assert not frame.on_report("r")
        assert frame.on_exit(None)
    else:
        assert not frame.on_exit(None)
        assert frame.on_report("r")
    assert (frame.state, frame.kept_report) == (FrameState.CLOSED, "r")
    assert not frame.on_report("again")
    assert not frame.on_exit(None)
    assert frame.kept_report == "r"


def test_a_raised_frame_closes_at_exit_with_its_error() -> None:
    frame = FrameRecord(FrameKind.MODEL_CALL, "s-1:api:1")
    assert frame.on_exit("APIConnectionError")
    assert (frame.state, frame.error_type) == (FrameState.CLOSED, "APIConnectionError")
    assert not frame.on_report("late")
    assert frame.kept_report is None


def test_an_orphan_frame_closes_at_exit_and_never_by_report() -> None:
    frame = FrameRecord(FrameKind.TOOL, "call_1", orphan=True)
    assert not frame.on_report("r")
    assert frame.on_exit(None)
    assert frame.kept_report is None


def test_exit_and_report_race_has_one_winner() -> None:
    def race_once() -> tuple[list[object], FrameRecord]:
        frame = FrameRecord(FrameKind.TOOL, "call_1")
        sides = [lambda: frame.on_exit(None), lambda: frame.on_report("r")]
        return _race(2, lambda: sides.pop()()), frame

    for _ in range(50):
        wins, frame = race_once()
        assert wins.count(True) == 1
        assert frame.state is FrameState.CLOSED


# -- the book ----------------------------------------------------------------------------------


def test_admit_returns_the_winner() -> None:
    book = TurnBook()
    first, second = _record(), _record()
    assert book.admit(first) is first
    assert book.admit(second) is first
    assert book.find("t-1") is first
    assert book.find("t-2") is None


def test_admit_refuses_when_the_live_table_is_full() -> None:
    book = TurnBook(cap=2)
    assert book.admit(_record("t-1")) is not None
    assert book.admit(_record("t-2")) is not None
    assert book.admit(_record("t-3")) is None
    assert book.find("t-3") is None


def test_concurrent_admits_of_one_turn_publish_one_record() -> None:
    book = TurnBook()
    winners = _race(8, lambda: book.admit(_record()))
    assert len({id(w) for w in winners}) == 1


def test_retire_moves_a_record_to_its_tombstone() -> None:
    book = TurnBook()
    r = _record(generation=2, model_frames=3)
    book.admit(r)
    book.retire(r, TurnState.PRESUMED_CLOSED)
    assert book.find("t-1") is None
    tomb = book.tombstone("t-1")
    assert tomb is not None
    assert (tomb.state, tomb.generation, tomb.model_frames) == (TurnState.PRESUMED_CLOSED, 2, 3)


def test_last_generation_is_the_latest_retired_one() -> None:
    book = TurnBook()
    first = _record()
    book.admit(first)
    book.retire(first, TurnState.PRESUMED_CLOSED)
    tomb = book.tombstone("t-1")
    assert tomb is not None
    second = _record(generation=tomb.generation + 1)
    book.admit(second)
    book.retire(second, TurnState.CLOSED)
    tomb = book.tombstone("t-1")
    assert tomb is not None
    assert (tomb.generation, tomb.state) == (1, TurnState.CLOSED)


def test_retiring_a_stale_record_leaves_the_live_one() -> None:
    book = TurnBook()
    live, stale = _record(generation=1), _record(generation=0)
    book.admit(live)
    book.retire(stale, TurnState.PRESUMED_CLOSED)
    assert book.find("t-1") is live


def test_session_snapshot() -> None:
    book = TurnBook()
    a, b, c = _record("t-1", "s-1"), _record("t-2", "s-2"), _record("t-3", "s-1")
    for r in (a, b, c):
        book.admit(r)
    assert book.live_in_session("s-1") == [a, c]
    assert book.live_in_session(None) == [a, b, c]
    snapshot = book.live_in_session("s-1")
    book.retire(a, TurnState.CLOSED)
    assert snapshot == [a, c]
    assert book.live_in_session("s-1") == [c]


def test_idle_lists_only_quiet_live_turns_past_the_limit() -> None:
    book = TurnBook()
    quiet = _record("t-1", last_event=0.0)
    busy = _record("t-2", last_event=0.0)
    busy.aux[("aux-1", 0)] = AuxAttempt(task="title_generation")
    recent = _record("t-3", last_event=IDLE_CLOSE_S)
    for r in (quiet, busy, recent):
        book.admit(r)
    assert book.idle(IDLE_CLOSE_S) == [quiet]
    assert book.idle(IDLE_CLOSE_S - 1) == []


def test_ties_are_kept_until_evicted() -> None:
    book = TurnBook()
    tie = ChildTie("child-s", "parent-t", "sub-1", "worker")
    book.tie(tie)
    assert book.tie_for("child-s") is tie
    assert book.tie_for("child-s") is tie
    assert book.tie_for("other") is None


def test_tombstones_and_ties_drop_the_oldest_write_first() -> None:
    book = TurnBook(cap=2)
    records = [_record(f"t-{i}") for i in range(3)]
    for r in records[:2]:
        book.admit(r)
        book.retire(r, TurnState.CLOSED)
    book.retire(records[0], TurnState.CLOSED)  # a rewrite makes t-0 the newest
    book.retire(records[2], TurnState.CLOSED)
    assert book.tombstone("t-1") is None
    assert book.tombstone("t-0") is not None
    assert book.tombstone("t-2") is not None
    for i in range(3):
        book.tie(ChildTie(f"c-{i}", "p"))
    assert [book.tie_for(f"c-{i}") is not None for i in range(3)] == [False, True, True]


def test_clear_forgets_everything() -> None:
    book = TurnBook()
    r = _record()
    book.admit(r)
    book.admit(_record("t-2"))
    book.retire(r, TurnState.CLOSED)
    book.tie(ChildTie("c", "p"))
    book.clear()
    assert book.live_in_session(None) == []
    assert book.tombstone("t-1") is None
    assert book.tie_for("c") is None


def test_fork_reset_frees_a_lock_held_at_the_fork() -> None:
    book = TurnBook()
    book.admit(_record())
    # Held by another thread, as a fork child finds a lock a parent thread held. The holder stays
    # alive: a new thread can reuse a dead one's ident, and a reentrant lock then reads as its.
    held, done = threading.Event(), threading.Event()
    parent_lock = book._lock  # noqa: SLF001 - the parent's lock, as a fork would copy it

    def hold() -> None:
        parent_lock.acquire()
        held.set()
        done.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    try:
        assert not _free_elsewhere(parent_lock, wait=0.2)
        book._at_fork_reinit()  # noqa: SLF001 - the fork-reset protocol itself
        assert _free_elsewhere(book._lock)  # noqa: SLF001 - the child's lock is a fresh one
    finally:
        done.set()
        holder.join(5)
    assert book.find("t-1") is None
    assert book.admit(_record()) is not None
