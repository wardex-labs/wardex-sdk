"""`assembly/_units.py` — the logical-unit registry (design §4.2, §5.2, §5.6).

What this file is defending, in one sentence per section:

  * `resolve()` — the precedence table, branch by branch. The branch that
    matters most is the one an earlier draft of the design had BACKWARDS: an
    alias that resolves to a STRICT DESCENDANT of the live context wins over the
    live context. Getting it wrong collapses every sub-agent into its session,
    silently, at confidence 1.0, with no marker to find it by.
  * `claim()` — the rank inversion. Claude's `PreToolUse` hook fires BEFORE the
    handler body, so first-come hands every in-process tool to the observer that
    did not wrap the execution.
  * eviction — a bound that drops data silently is worse than an unenforced one,
    so an evicted root's span is EMITTED and carries `UNIT_EVICTED`, and its
    whole subtree is closed with it.
  * `pin_driver()` — a pin is only legal on the task the caller is running.
  * lock discipline — the sink is never called while the registry lock is held.

Every test here fails against a plausible wrong implementation, not merely
against a missing one; the docstrings say which one.
"""

from __future__ import annotations

import asyncio
import gc
import threading

import pytest

from wardex_sdk._enums import AgentType, CaptureSource, StatusCode
from wardex_sdk._hub import reset_for_test
from wardex_sdk._types import AgentAttributes, ToolAttributes
from wardex_sdk.assembly import (
    EMPTY_AMBIENT,
    Ambient,
    Evidence,
    Limitation,
    ParentSource,
    SpanIntent,
    UnitKey,
    UnitKind,
    UnitRegistry,
    counters,
    latch_ambient,
)
from wardex_sdk.assembly._units import _ambient_unit
from wardex_sdk.context import activate_span


class RecordingSink:
    """A `SpanSink`-shaped double that keeps the drafts it was handed.

    `finish()` is deliberately NOT called on receipt: a draft's markers and
    parentage are what most of these tests assert, and materializing here would
    make every test also depend on the closed vocabulary being satisfied.
    `spans()` is for the cases that genuinely need the `InternalSpan`.
    """

    def __init__(self) -> None:
        self.drafts: list = []
        self.on_emit = None

    def emit(self, draft, *, agent_semantic: bool) -> bool:
        if self.on_emit is not None:
            self.on_emit()
        self.drafts.append(draft)
        return True

    def spans(self) -> list:
        return [d.finish() for d in self.drafts]


@pytest.fixture(autouse=True)
def _clean_scope():
    """Each test starts with no ambient scope, no ambient unit and no counters."""
    reset_for_test()
    token = _ambient_unit.set(None)
    counters.reset()
    yield
    _ambient_unit.reset(token)
    reset_for_test()
    counters.reset()


def registry(**kw) -> UnitRegistry:
    return UnitRegistry(sink=kw.pop("sink", RecordingSink()), **kw)


def open_session(reg: UnitRegistry, value: str = "s1", **kw):
    """A SESSION unit whose span can actually be materialized.

    `invoke_agent` requires an `agent` block, so a unit opened without one
    would be deleted by `finish()` — which is exactly why `Unit.draft` exists.
    """
    unit = reg.open(
        UnitKind.SESSION,
        UnitKey("test.session", value),
        ambient=kw.pop("ambient", EMPTY_AMBIENT),
        intent=SpanIntent.INVOKE_AGENT,
        subject=kw.pop("subject", "agent"),
        **kw,
    )
    unit.draft.set_agent(AgentAttributes(name="agent", id=value))
    return unit


def open_subagent(reg: UnitRegistry, parent, value: str = "a1"):
    unit = reg.open(
        UnitKind.AGENT,
        UnitKey("test.agent_id", value),
        ambient=EMPTY_AMBIENT,
        parent_unit=parent,
        intent=SpanIntent.INVOKE_AGENT,
        subject="sub",
    )
    unit.draft.set_agent(AgentAttributes(name="sub", id=value, agent_type=AgentType.SUB_AGENT))
    return unit


# ==========================================================================
# The bounds come from the core
# ==========================================================================


def test_bounds_are_resolved_from_the_core_not_a_python_literal():
    """A Python default here would pass every other test and disagree with
    crates/wardex-limits the moment someone changed it there. `test_limits.py`
    owns the drift assertion; this is the local statement of the same rule.
    """
    from wardex_sdk import _wardex_native

    core = _wardex_native.limits_defaults()
    reg = registry()
    assert reg._max_units == core["max_units"]
    assert reg._max_entries_per_unit == core["max_entries_per_unit"]


# ==========================================================================
# Rule P: a parent comes from the scope, never from an identifier
# ==========================================================================


def test_a_root_unit_takes_its_edge_from_the_latched_scope():
    reg = registry()
    with activate_span(_a_context()) as _:
        parent = latch_ambient()
        unit = open_session(reg, ambient=parent)
    assert unit.parentage.parent_span_id == parent.span_context.span_id
    assert unit.parentage.trace_id == parent.span_context.trace_id
    assert unit.parentage.correlation.strategy is ParentSource.CONTEXTVAR


def test_a_unit_with_no_scope_starts_a_trace_and_says_so():
    reg = registry()
    unit = open_session(reg)
    assert unit.parentage.parent_span_id is None
    assert unit.parentage.correlation.strategy is ParentSource.TRACE_ROOT
    assert unit.parentage.correlation.confidence == 1.0


def test_the_tree_does_not_depend_on_the_framework_ids(monkeypatch):
    """C-3 in miniature: replace every identifier and the shape is identical.

    The registry may use a key to FIND a unit; if a key could ever seed a trace
    id or a span id this assertion would fail, because the two runs share no
    identifier at all.
    """

    def shape(prefix: str):
        reg = registry()
        root = open_session(reg, f"{prefix}-session")
        sub = open_subagent(reg, root, f"{prefix}-agent")
        return (
            sub.parentage.parent_span_id == root.context.span_id,
            sub.context.trace_id == root.context.trace_id,
        )

    assert shape("alpha") == shape("zulu") == (True, True)


# ==========================================================================
# resolve() — design §5.2, branch by branch
# ==========================================================================


def test_resolve_alias_to_a_strict_descendant_beats_the_ambient_context():
    """THE branch the design had backwards.

    Under a pin the hook task's ambient is always the session root, while the
    alias carrying an agent_id resolves to the sub-agent — a strict descendant,
    and the more specific answer. An implementation that keeps the earlier rule
    ("ambient wins whenever the traces match") returns the SESSION here, and
    every sub-agent subtree silently flattens into its session.
    """
    reg = registry()
    root = open_session(reg)
    sub = open_subagent(reg, root)

    with root.activate():
        p = reg.resolve(UnitKey("test.agent_id", "a1"))

    assert p.parent_span_id == sub.context.span_id, "the alias must win over the ambient root"
    assert p.correlation.strategy is ParentSource.UNIT_ALIAS
    assert p.correlation.request_id == "a1"
    # 0.9, not the 1.0 design §5.2 writes: `Evidence.confidence` is clamped to
    # the source's table default so that a hint can only LOWER trust. See the
    # comment at this branch in `_units.resolve` for why the clamp wins.
    assert p.correlation.confidence == 0.9


def test_resolve_prefers_the_live_context_when_the_alias_is_not_below_it():
    """Same trace, not a descendant: the live context is the better answer and
    the id merely corroborates. Asserting the strategy is what separates this
    from the branch above — both return the same span id when the alias IS the
    ambient unit, so only the recorded evidence tells them apart.
    """
    reg = registry()
    root = open_session(reg)
    open_subagent(reg, root)

    with root.activate():
        p = reg.resolve(UnitKey("test.session", "s1"))

    assert p.parent_span_id == root.context.span_id
    assert p.correlation.strategy is ParentSource.CONTEXTVAR
    assert p.correlation.request_id == "s1"


def test_resolve_records_a_cross_trace_conflict_instead_of_hiding_it():
    """Two live traces disagreeing is a shippable bug report, not a tree to
    guess at. The context wins, the confidence drops, and the marker rides.
    """
    reg = registry()
    other = open_session(reg, "other")

    with activate_span(_a_context()):
        p = reg.resolve(UnitKey("test.session", "other"))

    assert p.trace_id != other.context.trace_id
    assert p.correlation.strategy is ParentSource.CONTEXTVAR
    assert p.correlation.confidence == 0.8
    assert Limitation.CORRELATION_CONFLICT in p.limitations


def test_resolve_falls_back_to_the_alias_when_no_context_reached_us():
    """The IPC / bare-thread case: attach to a context captured earlier ON THE
    CORRECT TASK, addressed by the framework id. 0.9, because an alias table can
    be stale — but it is an exact match, so it carries no marker.
    """
    reg = registry()
    root = open_session(reg)

    p = reg.resolve(UnitKey("test.session", "s1"), ambient=EMPTY_AMBIENT)

    assert p.parent_span_id == root.context.span_id
    assert p.correlation.strategy is ParentSource.UNIT_ALIAS
    assert p.correlation.confidence == 0.9
    assert p.limitations == ()


def test_resolve_uses_the_ambient_context_when_the_alias_misses():
    reg = registry()
    open_session(reg)
    ctx = _a_context()

    p = reg.resolve(UnitKey("test.session", "nope"), ambient=Ambient(ctx, None, None))

    assert p.parent_span_id == ctx.span_id
    assert p.correlation.strategy is ParentSource.CONTEXTVAR


def test_resolve_falls_to_the_sole_live_session_and_marks_the_guess():
    """The adapter this replaces DROPS the hook entirely when it cannot decide.
    A marked 0.5 edge beats unmarked data loss — but it must be marked, or it is
    indistinguishable from a real attachment.
    """
    reg = registry()
    root = open_session(reg)

    p = reg.resolve(UnitKey("test.hook", "unknown"), ambient=EMPTY_AMBIENT)

    assert p.parent_span_id == root.context.span_id
    assert p.correlation.strategy is ParentSource.UNIT_SOLE
    assert p.correlation.confidence == 0.5
    assert Limitation.UNIT_INFERRED_SOLE in p.limitations


def test_resolve_refuses_to_guess_when_two_sessions_are_live():
    reg = registry()
    open_session(reg, "one")
    open_session(reg, "two")

    p = reg.resolve(UnitKey("test.hook", "unknown"), ambient=EMPTY_AMBIENT)

    assert p.parent_span_id is None
    assert p.correlation.strategy is ParentSource.UNRESOLVED
    assert p.correlation.confidence == 0.0
    assert Limitation.PARENT_UNRESOLVED in p.limitations


def test_resolve_reads_the_live_scope_when_no_ambient_is_supplied():
    """`resolve(alias)` with no `ambient=` must latch, not treat it as empty."""
    reg = registry()
    ctx = _a_context()
    with activate_span(ctx):
        p = reg.resolve(None)
    assert p.parent_span_id == ctx.span_id


def test_an_interpreted_edge_carries_its_marker_onto_the_span():
    """Half of I4 is the confidence and the other half is the marker. A draft is
    built FROM a parentage but does not inherit its markers, so a registry that
    forgets to copy them ships a 0.5 edge with an empty `limitations` — the half
    a dashboard renders.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)

    draft = root.open_span(
        SpanIntent.EXECUTE_TOOL,
        subject="Bash",
        evidence=Evidence(ParentSource.UNIT_SOLE),
    )
    draft.set_tool(ToolAttributes(name="Bash"))
    root.close_span(draft)

    assert Limitation.UNIT_INFERRED_SOLE in sink.drafts[0].integrity.markers


# ==========================================================================
# claim() — the rank inversion
# ==========================================================================


def test_a_higher_rank_takes_a_key_that_a_lower_rank_claimed_first():
    """First-come would hand every in-process tool to the `PreToolUse` observer,
    which fires BEFORE the handler body runs. Ownership belongs to the layer
    that wrapped the real execution, whenever it happens to arrive.
    """
    reg = registry()
    root = open_session(reg)
    key = UnitKey("mcp.tool", "srv/greet")

    assert root.claim(key, rank=0) is True  # hook, first
    assert root.claim(key, rank=10) is True  # handler, later, higher
    assert root.claim(key, rank=0) is False  # hook asks again: it lost
    assert root.owner_rank(key) == 10


def test_an_equal_rank_does_not_take_a_key_from_the_incumbent():
    reg = registry()
    root = open_session(reg)
    key = UnitKey("mcp.tool", "srv/greet")
    assert root.claim(key, rank=10) is True
    assert root.claim(key, rank=10) is False


def test_the_loser_of_an_arbitration_never_emits_its_span():
    """The mechanical half. The hook opens its span at `PreToolUse`, the handler
    outranks it while the body runs, and the hook's span is DISCARDED at close —
    without the loser having to remember to ask again. Without this the two
    observers emit one span each and every in-process tool is double-counted.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)
    key = UnitKey("mcp.tool", "srv/greet")

    assert root.claim(key, rank=0) is True
    hook_draft = root.open_span(SpanIntent.EXECUTE_TOOL, subject="greet", key=key)
    hook_draft.set_tool(ToolAttributes(name="greet"))

    assert root.claim(key, rank=10) is True
    handler_draft = root.open_span(SpanIntent.EXECUTE_TOOL, subject="greet", key=key)
    handler_draft.set_tool(ToolAttributes(name="greet"))

    root.close_span(hook_draft)
    root.close_span(handler_draft)

    assert sink.drafts == [handler_draft]
    assert counters.get("assembly._units.claim_superseded") == 1


def test_a_loser_that_never_closes_is_not_emitted_by_the_units_teardown():
    """The case `claim()`'s docstring names as safe, and the one that was not.

    "close_span sees the higher rank and DISCARDS the loser's draft, so the
    double emit the arbitration exists to prevent cannot happen even if the
    loser never asks again" — but the rank re-check lived ONLY in `close_span`,
    and a loser that never asks again is precisely the one whose draft the UNIT
    force-closes. In the Claude adapter that is "the session aborted between
    `PreToolUse` and `PostToolUse`", i.e. exactly when things go wrong: two
    `execute_tool` spans for one tool call, `claim_superseded` at zero, and
    nothing downstream to say the agent did not call the tool twice.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)
    key = UnitKey("mcp.tool", "srv/greet")

    assert root.claim(key, rank=0) is True
    hook_draft = _tool_draft(root, "greet", key=key)
    assert root.claim(key, rank=10) is True
    handler_draft = _tool_draft(root, "greet", key=key)
    root.close_span(handler_draft)  # the winner ships

    reg.close(root)  # the hook never gets its PostToolUse

    tool_spans = [d for d in sink.drafts if d is not root.draft]
    assert tool_spans == [handler_draft]
    assert hook_draft not in sink.drafts
    assert counters.get("assembly._units.claim_superseded") >= 1


def test_a_loser_evicted_by_the_open_table_bound_is_discarded_not_emitted():
    """The same question at the other force-close. A bound is a reason to stop
    TRACKING a draft, never a reason to promote one the arbitration rejected.
    """
    sink = RecordingSink()
    reg = registry(sink=sink, max_entries_per_unit=1)
    root = open_session(reg)
    key = UnitKey("mcp.tool", "srv/greet")

    assert root.claim(key, rank=0) is True
    hook_draft = _tool_draft(root, "greet", key=key)
    assert root.claim(key, rank=10) is True
    handler_draft = _tool_draft(root, "greet", key=key)  # evicts the hook's draft
    root.close_span(handler_draft)

    assert sink.drafts == [handler_draft]
    assert hook_draft not in sink.drafts
    assert counters.get("assembly._units.claim_superseded") == 1


def test_the_winner_of_an_arbitration_still_emits():
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)
    key = UnitKey("mcp.tool", "srv/greet")

    assert root.claim(key, rank=10) is True
    draft = root.open_span(SpanIntent.EXECUTE_TOOL, subject="greet", key=key)
    draft.set_tool(ToolAttributes(name="greet"))
    root.close_span(draft)

    assert sink.drafts == [draft]


# ==========================================================================
# activate() / bind() / pin_driver() — becoming a parent
# ==========================================================================


def test_activate_makes_the_unit_the_ambient_parent_and_restores_after():
    reg = registry()
    root = open_session(reg)
    assert reg.current() is None
    with root.activate():
        assert reg.current() is root
        assert latch_ambient().span_context == root.context
    assert reg.current() is None
    assert latch_ambient().span_context is None


def test_activate_carries_the_conversation_identity():
    """A carrier that installs only the span context leaves the conversation id
    behind on every task the unit spans.
    """
    from wardex_sdk._types import ConversationContext

    reg = registry()
    conv = ConversationContext(conversation_id="c-1")
    root = reg.open(
        UnitKind.SESSION,
        UnitKey("test.session", "s1"),
        ambient=Ambient(None, conv, None),
        intent=SpanIntent.INVOKE_AGENT,
        subject="agent",
    )
    root.draft.set_agent(AgentAttributes(name="agent"))
    with root.activate():
        assert latch_ambient().conversation == conv


def test_bind_enters_a_fresh_fork_per_invocation():
    """`context.run_in_context` replays ONE captured Context and raises
    `RuntimeError: cannot enter context ... is already entered` when the same
    Context is entered twice concurrently — into USER CODE, for any framework
    callback invoked from two tasks. Recursion is the cheapest reproduction of
    the concurrent case and fails the same way.
    """
    reg = registry()
    root = open_session(reg)
    seen: list = []

    def work(depth: int) -> None:
        seen.append(latch_ambient().span_context)
        if depth:
            wrapped(depth - 1)

    wrapped = root.bind(work)
    wrapped(2)

    assert seen == [root.context] * 3


def test_bind_wraps_a_coroutine_for_the_whole_await():
    reg = registry()
    root = open_session(reg)
    seen: list = []

    async def work() -> None:
        seen.append(latch_ambient().span_context)
        await asyncio.sleep(0)
        seen.append(latch_ambient().span_context)

    asyncio.run(root.bind(work)())
    assert seen == [root.context, root.context]


def test_a_pin_is_refused_when_it_names_a_task_other_than_the_caller():
    """`ContextVar.set()` lands on the CALLING task, so pinning "on behalf of"
    another task installs the unit somewhere the caller did not mean and leaves
    the named task unpinned — self-consistent and undetectable downstream. The
    refusal is recorded on the unit's own span, not merely counted.
    """
    reg = registry()
    root = open_session(reg)

    token = reg.pin_driver(root, owner_task=object())

    assert token.installed is False
    assert reg.current() is None
    assert Limitation.CORRELATION_CONFLICT in root.draft.integrity.markers
    assert counters.get("assembly._units.pin_foreign_task") == 1


def test_a_pin_on_the_calling_task_installs_and_survives_the_call():
    reg = registry()
    root = open_session(reg)

    token = reg.pin_driver(root, owner_task=threading.current_thread())

    assert token.installed is True
    assert reg.current() is root
    assert latch_ambient().span_context == root.context
    reg.unpin(token)
    assert reg.current() is None


def test_a_pin_survives_the_token_being_discarded():
    """The shipped call site reads `.installed` and throws the token away.

    That is the correct contract for a handle whose only advertised member is a
    bool, so the pin has to survive it — and BOTH halves have to, not one. A
    generator-backed carrier does not: the `PinToken` is the only strong
    reference to it, so CPython finalizes the suspended generator by refcount on
    the very statement that installed it, its `finally` resets the scope, and
    the ambient UNIT is left standing while the ambient SPAN is gone. The unit
    half surviving is what makes that shape look healthy: `current()` answers,
    the tool tree comes out right, and every task descended from the driver
    silently hangs off whatever scope preceded the pin — a wrong parent at
    confidence 1.0 with no marker.

    The test above holds the token in a local and therefore proves only
    "survives while you hold the handle".
    """
    reg = registry()
    root = open_session(reg)

    assert reg.pin_driver(root, owner_task=threading.current_thread()).installed is True
    gc.collect()

    assert reg.current() is root
    assert latch_ambient().span_context == root.context


def test_a_pin_is_inherited_by_a_task_spawned_after_it():
    """The mechanism the whole design rests on: a hook callback and an
    in-process tool handler run in tasks whose context was COPIED from the
    driver task after the pin, so they see the session with zero framework
    identifiers and confidence 1.0.
    """
    reg = registry()
    root = open_session(reg)
    seen: list = []

    async def driver() -> None:
        token = reg.pin_driver(root, owner_task=asyncio.current_task())
        assert token.installed is True

        async def spawned() -> None:
            seen.append(reg.current())
            seen.append(reg.resolve(None).parent_span_id)

        await asyncio.create_task(spawned())
        reg.unpin(token)

    asyncio.run(driver())
    assert seen == [root, root.context.span_id]


def test_unpinning_from_a_foreign_task_is_counted_not_raised():
    """A ContextVar Token may only be reset in the Context that created it, and
    a teardown path is exactly where a `ValueError` would reach the host.
    """
    reg = registry()
    root = open_session(reg)
    token = reg.pin_driver(root, owner_task=threading.current_thread())

    failed: list = []

    def elsewhere() -> None:
        try:
            reg.unpin(token)
        except BaseException as exc:  # noqa: BLE001 — the point of the test
            failed.append(exc)

    thread = threading.Thread(target=elsewhere)
    thread.start()
    thread.join()

    assert failed == []
    # Two, not one: both installations were left standing and neither could be
    # taken down. The second attempt is what drives `activate_span`'s generator
    # to completion — bailing after the first leaves it suspended for the
    # garbage collector to close later, where the same ValueError lands as an
    # unraisable exception in the HOST's stderr.
    assert counters.get("assembly._units.unpin") == 2


def test_exiting_an_activation_on_another_task_is_counted_not_raised():
    """Same rule from the other side. `activate()` is documented as same-task
    only; a caller that breaks it gets a counter, not an exception in a `finally`
    block it did not write.
    """
    reg = registry()
    root = open_session(reg)
    manager = root.activate()
    manager.__enter__()

    failed: list = []

    def elsewhere() -> None:
        try:
            manager.__exit__(None, None, None)
        except BaseException as exc:  # noqa: BLE001 — the point of the test
            failed.append(exc)

    thread = threading.Thread(target=elsewhere)
    thread.start()
    thread.join()

    assert failed == []
    assert counters.get("assembly._units.activate_exit") == 2  # see the unpin test


def test_a_pin_stops_being_ambient_once_its_unit_closes():
    """The leaked-pin detection. A pin on a pooled worker is self-consistent —
    same task, same ContextVar — so the only thing that can reveal it is the
    unit being dead. Without this, later unrelated work is parented into a
    finished unit and nothing downstream can tell.
    """
    reg = registry()
    root = open_session(reg)
    reg.pin_driver(root, owner_task=threading.current_thread())
    assert reg.current() is root
    # The token was discarded, exactly as the shipped call site discards it, so
    # this line is also the span half of that contract.
    assert latch_ambient().span_context == root.context

    reg.close(root)

    assert reg.current() is None
    assert counters.get("assembly._units.pin_stale") == 1


def test_a_stale_pins_scope_is_refused_and_the_edge_says_so():
    """`current()` refusing the dead unit is only half of the pin's safety.

    A pin installs TWO things and `close()` can invalidate neither — it runs on
    another task, and a ContextVar cannot be reset from one. The unit half is
    gated by liveness; the SPAN half is still standing, so the very next
    `latch_ambient()` returns the DEAD unit's own context and the edge built
    from it reads `contextvar` / 1.0 / no marker: unrelated later work filed
    into a finished session, byte-for-byte indistinguishable from a real
    attachment. Design §5.6 asks for a hard `CORRELATION_CONFLICT` here rather
    than an internal count, because the consequence is a tree shape and a tree
    shape has to be falsifiable from the data.
    """
    reg = registry()
    root = open_session(reg)
    reg.pin_driver(root, owner_task=threading.current_thread())
    reg.close(root)
    assert latch_ambient().span_context == root.context, "the leftover fork is the precondition"

    p = reg.resolve(None)

    assert p.parent_span_id is None
    assert p.correlation.strategy is not ParentSource.CONTEXTVAR
    assert p.correlation.confidence < 1.0
    assert Limitation.CORRELATION_CONFLICT in p.limitations
    assert counters.get("assembly._units.stale_pin_ambient") == 1


def test_a_stale_pin_does_not_capture_work_while_other_sessions_are_live():
    """The branch that is silent. With more than one session live the ladder's
    ambient tier wins outright, so the dead unit would be handed back at 1.0
    with an empty `limitations` — the one outcome no downstream reader can
    question. Refusing the poisoned scope drops it to a marked tier.
    """
    reg = registry()
    root = open_session(reg, "dead")
    reg.pin_driver(root, owner_task=threading.current_thread())
    reg.close(root)
    open_session(reg, "live-one")
    open_session(reg, "live-two")

    p = reg.resolve(None)

    assert p.parent_span_id != root.context.span_id
    assert p.correlation.confidence < 1.0
    assert Limitation.CORRELATION_CONFLICT in p.limitations


def test_a_unit_opened_from_a_stale_pins_scope_is_a_root_that_says_so():
    """`open()` is the other place an edge is built from a latched scope. A
    session that starts after the previous one died is a NEW run, not a subtree
    of the corpse the reader task is still carrying.
    """
    reg = registry()
    root = open_session(reg, "dead")
    reg.pin_driver(root, owner_task=threading.current_thread())
    reg.close(root)

    fresh = open_session(reg, "next", ambient=latch_ambient())

    assert fresh.parentage.parent_span_id is None
    assert fresh.context.trace_id != root.context.trace_id
    assert Limitation.CORRELATION_CONFLICT in fresh.draft.integrity.markers


def test_a_real_span_over_a_stale_pin_is_still_a_parent():
    """The refusal is the leftover FORK, not the task. A stale pin says this
    task descends from a dead driver; it does not say the scope still holds the
    dead unit's span. A host that opened its own span inside that task put a
    real parent on top, and moving live work out of the host's trace to escape
    a ghost that is no longer in front of us is a wrong tree of its own shape.
    """
    reg = registry()
    root = open_session(reg)
    reg.pin_driver(root, owner_task=threading.current_thread())
    reg.close(root)
    host = _a_context()

    with activate_span(host):
        p = reg.resolve(None)

    assert p.parent_span_id == host.span_id
    assert p.correlation.strategy is ParentSource.CONTEXTVAR
    assert Limitation.CORRELATION_CONFLICT not in p.limitations
    assert counters.get("assembly._units.stale_pin_ambient") == 0


# ==========================================================================
# cross-registry ownership — the ambient carrier is process-wide, a unit is not
# ==========================================================================


def test_a_pin_from_another_registry_is_not_this_registrys_parent():
    """`_ambient_unit` is one module-level ContextVar for the whole process, so
    a second `init()` — a new adapter, a new registry — reads the previous
    registry's pin verbatim: same object, `is_live` True, no counter.

    The staleness gate below it never fires on that path, because nothing in
    production closes a dropped registry's units (`close_all` has no production
    caller). So the leaked unit is handed out at `UNIT_ACTIVE`/1.0 and the work
    is filed into a registry that no longer exists.
    """
    reg_a = registry()
    unit_a = open_session(reg_a, "a")
    reg_a.pin_driver(unit_a, owner_task=threading.current_thread())

    reg_b = registry()

    assert reg_b.current() is None
    assert counters.get("assembly._units.ambient_foreign_registry") == 1
    assert reg_a.current() is unit_a, "the owning registry must be unaffected"


def test_a_foreign_parent_unit_makes_the_child_a_root_of_this_registry():
    """Ownership at the second entry point. `open()` is public and a foreign
    parent is what actually breaks the BOUNDS: filed into the other registry's
    `_children`, the child is reachable from no root of either — `close_all`
    cannot see it, so its span is never emitted, and the total-unit bound
    short-circuits on `and self._roots` and stops bounding anything.

    Registering it as a root of THIS registry is the conservative outcome: the
    span still hangs off the foreign context, but the lifetime is managed here.
    """
    sink_b = RecordingSink()
    reg_a = registry()
    unit_a = open_session(reg_a, "a")
    reg_b = registry(sink=sink_b)

    child = open_subagent(reg_b, unit_a, "cross")

    assert child in reg_b._roots
    assert child not in unit_a._children
    assert counters.get("assembly._units.parent_foreign_registry") == 1

    reg_b.close_all(reason=Limitation.ADAPTER_UNINSTALLED)

    assert child.is_live is False
    assert child.draft in sink_b.drafts


# ==========================================================================
# find / alias / sole_live
# ==========================================================================


def test_aliases_registered_at_open_and_afterwards_both_resolve():
    reg = registry()
    root = reg.open(
        UnitKind.SESSION,
        UnitKey("transport.id", "42"),
        ambient=EMPTY_AMBIENT,
        intent=SpanIntent.INVOKE_AGENT,
        subject="agent",
        aliases=(UnitKey("claude.session_id", "sid"),),
    )
    reg.alias(UnitKey("transport.id", "42"), UnitKey("claude.session_id", "later"))

    assert reg.find(UnitKey("claude.session_id", "sid")) is root
    assert reg.find(UnitKey("claude.session_id", "later")) is root


def test_a_closed_unit_is_no_longer_findable():
    reg = registry()
    root = open_session(reg)
    reg.close(root)
    assert reg.find(UnitKey("test.session", "s1")) is None
    assert reg.sole_live(UnitKind.SESSION) is None


def test_a_full_alias_table_drops_the_oldest_alias():
    """Aliases are bounded per unit like every other table. Losing one costs a
    lookup, not a span: the next `resolve()` degrades to `sole_live` or to
    `unresolved`, and both of those mark themselves.
    """
    reg = registry(max_entries_per_unit=2)
    root = open_session(reg)  # its own key takes the first slot
    reg.alias(UnitKey("test.session", "s1"), UnitKey("extra", "one"))
    reg.alias(UnitKey("extra", "one"), UnitKey("extra", "two"))

    assert reg.find(UnitKey("test.session", "s1")) is None
    assert reg.find(UnitKey("extra", "two")) is root
    assert counters.get("assembly._units.alias_table_full") == 1


def test_aliasing_an_unknown_key_is_a_counted_no_op():
    reg = registry()
    reg.alias(UnitKey("test.session", "ghost"), UnitKey("other", "x"))
    assert reg.find(UnitKey("other", "x")) is None
    assert counters.get("assembly._units.alias_unknown_key") == 1


# ==========================================================================
# eviction and closing — I10
# ==========================================================================


def test_closing_a_unit_emits_the_subtree_with_the_root_last():
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)
    sub = open_subagent(reg, root)
    open_draft = root.open_span(SpanIntent.EXECUTE_TOOL, subject="Bash")
    open_draft.set_tool(ToolAttributes(name="Bash"))

    reg.close(root, status=StatusCode.OK)

    assert [d.context for d in sink.drafts][-1] == root.context
    assert sub.draft in sink.drafts
    assert open_draft in sink.drafts
    assert Limitation.CHILD_SPAN_UNCLOSED in sub.draft.integrity.markers
    assert Limitation.CHILD_SPAN_UNCLOSED in open_draft.integrity.markers


def test_eviction_over_max_units_closes_the_whole_oldest_subtree():
    """A bound whose enforcement drops data silently is worse than an unenforced
    one. The evicted root's span is EMITTED and says which knob evicted it, and
    its children go with it rather than being orphaned into a table nobody owns.
    """
    sink = RecordingSink()
    reg = registry(sink=sink, max_units=1)
    first = open_session(reg, "first")
    sub = open_subagent(reg, first)

    second = open_session(reg, "second")

    assert first.is_live is False
    assert sub.is_live is False
    assert second.is_live is True
    assert sink.drafts == [sub.draft, first.draft]
    assert Limitation.UNIT_EVICTED in first.draft.integrity.markers
    assert Limitation.CHILD_SPAN_UNCLOSED in sub.draft.integrity.markers
    assert reg.find(UnitKey("test.session", "first")) is None


def test_the_evicted_root_span_reaches_the_wire_shape():
    """Not just "a draft was handed over": it has to be a span. An eviction that
    produced a draft `finish()` rejects would be the same silent drop with extra
    steps, and `guard()` would leave only a counter.
    """
    sink = RecordingSink()
    reg = registry(sink=sink, max_units=1)
    open_session(reg, "first")
    open_session(reg, "second")

    span = sink.spans()[0]
    assert span.name == "invoke_agent agent"
    assert Limitation.UNIT_EVICTED in span.capture_integrity.limitations
    assert span.capture_sources == (CaptureSource.ADAPTER,)


def test_a_full_child_table_evicts_the_oldest_child_and_emits_it():
    sink = RecordingSink()
    reg = registry(sink=sink, max_entries_per_unit=1)
    root = open_session(reg)
    first = open_subagent(reg, root, "a1")

    open_subagent(reg, root, "a2")

    assert first.is_live is False
    assert sink.drafts == [first.draft]
    assert Limitation.CHILD_SPAN_UNCLOSED in first.draft.integrity.markers
    assert counters.get("assembly._units.child_table_full") == 1


def test_a_full_open_span_table_evicts_the_oldest_draft_and_emits_it():
    sink = RecordingSink()
    reg = registry(sink=sink, max_entries_per_unit=1)
    root = open_session(reg)
    first = root.open_span(SpanIntent.EXECUTE_TOOL, subject="a")
    first.set_tool(ToolAttributes(name="a"))

    root.open_span(SpanIntent.EXECUTE_TOOL, subject="b")

    assert sink.drafts == [first]
    assert Limitation.CHILD_SPAN_UNCLOSED in first.integrity.markers


def test_deeply_nested_units_stay_bounded():
    """`max_units` bounds roots and `max_entries_per_unit` bounds breadth;
    neither bounds DEPTH on its own, so a chain of child-of-child units would
    grow without limit. The derived ceiling closes that without inventing a knob.
    """
    reg = registry(max_units=1, max_entries_per_unit=2)
    root = open_session(reg)
    node = root
    for i in range(6):
        node = open_subagent(reg, node, f"a{i}")

    assert reg._max_total_units == 2
    assert len(reg._live_units) <= reg._max_total_units
    assert root.is_live is False


def test_close_all_marks_every_root_with_the_callers_reason():
    sink = RecordingSink()
    reg = registry(sink=sink)
    one = open_session(reg, "one")
    two = open_session(reg, "two")

    reg.close_all(reason=Limitation.ADAPTER_UNINSTALLED)

    assert one.is_live is False and two.is_live is False
    for draft in (one.draft, two.draft):
        assert Limitation.ADAPTER_UNINSTALLED in draft.integrity.markers
    assert len(sink.drafts) == 2


def test_closing_twice_emits_once():
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)
    reg.close(root)
    reg.close(root)
    assert len(sink.drafts) == 1
    assert counters.get("assembly._units.close_after_close") == 1


# ==========================================================================
# emit-once — one draft is one span id, and one span id is one span
# ==========================================================================


def _tool_draft(unit, subject: str, **kw):
    draft = unit.open_span(SpanIntent.EXECUTE_TOOL, subject=subject, **kw)
    draft.set_tool(ToolAttributes(name=subject))
    return draft


def _one_span_per_id(sink: RecordingSink) -> None:
    """A draft emitted twice is not a duplicate downstream, it is a
    CONTRADICTION: both copies carry one span id, and the late one keeps the
    `CHILD_SPAN_UNCLOSED` a force-close added while claiming `status=OK` and a
    later end. Last-write-wins then shows a span that says both.
    """
    ids = [d.context.span_id for d in sink.drafts]
    assert len(set(ids)) == len(ids), "one span id reached the sink twice"


def test_a_handler_that_returns_after_its_unit_died_does_not_emit_a_second_span():
    """The ordinary race, not a pathological one: a session ends on the
    transport's `on_close` while an in-process tool handler is still awaiting,
    and the handler closes its span afterwards. `CHILD_SPAN_UNCLOSED` exists
    BECAUSE this happens.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)
    draft = _tool_draft(root, "greet")

    reg.close(root)  # force-closes the open draft and emits it
    root.close_span(draft)  # the handler returns afterwards

    _one_span_per_id(sink)
    assert sink.drafts.count(draft) == 1
    assert counters.get("assembly._units.close_span_after_emit") == 1
    # The force-closed record is the one that shipped, and the late close did
    # not rewrite it into a success after the fact.
    assert Limitation.CHILD_SPAN_UNCLOSED in draft.integrity.markers
    assert draft.finish().status is StatusCode.UNSET


def test_an_evicted_draft_is_not_emitted_again_by_the_owner_that_closes_it():
    """The bound-triggered force-close, same hole. `open_span`'s eviction hands
    the oldest draft to the sink and forgets it; its owner still holds the
    object and closes it normally.
    """
    sink = RecordingSink()
    reg = registry(sink=sink, max_entries_per_unit=1)
    root = open_session(reg)
    first = _tool_draft(root, "a")

    _tool_draft(root, "b")  # evicts and emits `first`
    root.close_span(first)

    _one_span_per_id(sink)
    assert sink.drafts.count(first) == 1
    assert counters.get("assembly._units.close_span_after_emit") == 1


def test_closing_the_same_span_twice_emits_it_once():
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)
    draft = _tool_draft(root, "greet")

    root.close_span(draft)
    root.close_span(draft)

    _one_span_per_id(sink)
    assert sink.drafts == [draft]
    assert counters.get("assembly._units.close_span_after_emit") == 1


def test_a_straggler_close_after_close_all_does_not_emit_a_second_span():
    """`close_all` is the uninstall/teardown path, so its stragglers arrive from
    code the SDK no longer controls — the one place a second emit would be
    hardest to notice.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)
    draft = _tool_draft(root, "greet")

    reg.close_all(reason=Limitation.ADAPTER_UNINSTALLED)
    root.close_span(draft)

    _one_span_per_id(sink)
    assert sink.drafts.count(draft) == 1
    assert counters.get("assembly._units.close_span_after_emit") == 1


def test_the_emit_funnel_ships_a_draft_once_however_it_arrives():
    """The backstop under the four paths above. Each of them is also gated at
    its own door, and that is the point: `_flush` is the ONE funnel every emit
    goes through — `close_span`, `_close_locked`, `close_all`,
    `_evict_root_locked`, the open-table eviction — so latching the draft here
    is what keeps the next path added from re-opening the hole without a test
    of its own.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)
    draft = _tool_draft(root, "greet")
    draft.set_end_ns(1)

    reg._flush([draft, draft])

    _one_span_per_id(sink)
    assert sink.drafts == [draft]
    assert counters.get("assembly._units.emit_duplicate") == 1


# ==========================================================================
# I/O accumulation
# ==========================================================================


def test_recorded_io_accumulates_onto_the_units_own_span():
    """Assigning instead of accumulating is how the adapter kept only the first
    prompt of a multi-turn session.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)
    root.record_input(b"turn-1 ")
    root.record_input(b"turn-2")
    root.record_output(b"done")

    reg.close(root)
    span = sink.spans()[0]

    assert span.input_data == b"turn-1 turn-2"
    assert span.output_data == b"done"
    assert span.capture_integrity.request_body_captured is True


def test_recorded_io_is_bounded_and_says_when_it_truncated():
    sink = RecordingSink()
    reg = registry(sink=sink)
    reg._max_record_bytes = 4
    root = open_session(reg)
    root.record_input(b"0123456789")

    reg.close(root)
    span = sink.spans()[0]

    assert span.input_data == b"0123"
    assert span.capture_integrity.truncated is True


def test_a_span_with_no_recorded_io_says_nothing_about_integrity():
    """`set_io` would report seven `False`s as "we tried everything and failed",
    which is a different claim from "there was nothing to say".
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)
    reg.close(root)
    assert sink.spans()[0].capture_integrity is None


# ==========================================================================
# I11 — nothing is emitted while the registry lock is held
# ==========================================================================


def _lock_is_free(reg: UnitRegistry) -> bool:
    """Can ANOTHER thread take the registry lock right now?

    It has to be another thread: `_lock` is an RLock, so the emitting thread
    could re-acquire it and report a false green — which is precisely the
    re-entrancy that makes the hazard this rule exists for so hard to see.
    """
    result: list[bool] = []

    def probe() -> None:
        acquired = reg._lock.acquire(blocking=False)
        result.append(acquired)
        if acquired:
            reg._lock.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return result[0]


@pytest.mark.parametrize(
    "drive",
    [
        pytest.param(lambda reg, root: reg.close(root), id="close"),
        pytest.param(
            lambda reg, root: reg.close_all(reason=Limitation.UNIT_INTERRUPTED), id="close_all"
        ),
        pytest.param(lambda reg, root: open_session(reg, "evictor"), id="evict"),
        pytest.param(
            lambda reg, root: root.close_span(root.open_span(SpanIntent.EXECUTE_TOOL, subject="t")),
            id="close_span",
        ),
    ],
)
def test_the_sink_is_never_called_while_the_registry_lock_is_held(drive):
    """I11. `Client.capture_span` can be re-entered from a same-thread signal
    handler and can run the host's `before_send` on the calling thread; the
    assembler this replaces calls it holding its own RLock. Two re-entrant locks
    in a fixed order across an async callback and a signal path is a deadlock
    waiting for a schedule, and it is invisible to every other test here.
    """
    sink = RecordingSink()
    reg = registry(sink=sink, max_units=1)
    root = open_session(reg)
    observations: list[bool] = []
    sink.on_emit = lambda: observations.append(_lock_is_free(reg))

    drive(reg, root)

    assert observations, "the drive did not emit anything, so it proved nothing"
    assert all(observations), "the sink was called while the registry lock was held"


def test_a_sink_that_raises_cannot_reach_the_host():
    sink = RecordingSink()
    sink.on_emit = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    reg = registry(sink=sink)
    root = open_session(reg)

    reg.close(root)  # must not raise

    assert counters.get("assembly._units.emit") == 1


# ==========================================================================
# helpers
# ==========================================================================


def _a_context():
    """A span context that came from the parentage core, as a parent must."""
    from wardex_sdk.assembly import resolve_parentage

    return resolve_parentage(EMPTY_AMBIENT).child_context()
