"""`_assembly/_units.py` — the logical-unit registry (design §4.2, §5.2, §5.6).

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

from wardex_sdk._assembly import (
    EMPTY_AMBIENT,
    Ambient,
    Evidence,
    Limitation,
    ParentSource,
    SpanIntent,
    Unit,
    UnitKey,
    UnitKind,
    UnitRegistry,
    counters,
    latch_ambient,
    parent_is_closed_unit,
)
from wardex_sdk._assembly._units import _ambient_unit
from wardex_sdk._enums import AgentType, CaptureSource, StatusCode
from wardex_sdk._hub import reset_for_test
from wardex_sdk._types import AgentAttributes, ToolAttributes
from wardex_sdk.context._contextvar import activate_span


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
    assert reg._max_link_targets == core["max_link_targets"]


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
    """`context.bind_context` replays ONE captured Context and raises
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


def test_a_unit_closed_on_another_thread_retires_its_pin_fork_in_place():
    """The pin stands on the main thread; the unit dies on a worker (a
    `wardex.close()` from another thread while the run's block is still
    open). No task but the main one can take the pin down, and the dead
    unit's scope fork stayed current there: every later `restore_scope` put
    it back and, after a re-init, the new registry could not judge it as its
    own corpse, so every later root on the main thread opened UNDER the dead
    unit at 1.0 with no marker. The registry now retires the fork in place
    from the closing thread: the main thread reads the host's scope again --
    the host's own span context and conversation, not the dead unit's."""
    from wardex_sdk._hub import get_current_scope
    from wardex_sdk._types import ConversationContext, SpanContext, SpanId, TraceId

    reset_for_test()
    host = SpanContext(trace_id=TraceId(b"\x0a" * 16), span_id=SpanId(b"\x0b" * 8))
    get_current_scope().active_span_context = host
    get_current_scope().conversation = ConversationContext(conversation_id="host-chat")
    reg = registry()
    root = open_session(reg)
    token = reg.pin_driver(root, owner_task=threading.current_thread())
    assert token.installed is True
    assert get_current_scope().active_span_context == root.context

    worker = threading.Thread(target=reg.close, args=(root,))
    worker.start()
    worker.join()

    assert root.is_live is False
    assert get_current_scope().active_span_context == host
    assert get_current_scope().conversation.conversation_id == "host-chat"
    assert counters.get("assembly._units.pin_retired_from_afar") == 1
    # After a re-init the entry is another registry's corpse on this task's
    # carrier, and nothing but this task can clear it: the first read that
    # finds it dead counts it as foreign ONCE and retires it, so a process
    # that re-inits is not counting a ghost on every read for its lifetime.
    later = registry()
    assert later.current() is None
    assert _ambient_unit.get() is None
    assert later.current() is None
    assert counters.get("assembly._units.ambient_foreign_registry") == 1
    assert counters.get("assembly._units.ambient_foreign_retired") == 1
    # The owning task's own unpin still restores the same host scope.
    reg.unpin(token)
    assert get_current_scope().active_span_context == host
    reset_for_test()


def test_a_unit_closed_on_its_own_thread_leaves_its_pin_to_unpin():
    """The negative control of the retirement: a pin whose unit closes on
    the pinning task is that task's to take down, and until it does the dead
    fork is a leftover the staleness gate refuses WITH a marker -- the audit
    signal pin discipline is measured by. Retiring it here would hide that."""
    from wardex_sdk._hub import get_current_scope

    reset_for_test()
    reg = registry()
    root = open_session(reg)
    token = reg.pin_driver(root, owner_task=threading.current_thread())
    reg.close(root)
    assert get_current_scope().active_span_context == root.context
    assert counters.get("assembly._units.pin_retired_from_afar") == 0
    reg.unpin(token)
    assert get_current_scope().active_span_context is None
    reset_for_test()


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

    The staleness gate below it never fires on that path. Teardown does now
    close a registry's units, but it is reached from the adapter's `uninstall`,
    and a re-`init()` uninstalls the PREVIOUS adapter — not a registry someone
    dropped on the floor without one. So the leaked unit is still handed out at
    `UNIT_ACTIVE`/1.0 and the work is filed into a registry that no longer
    exists; the foreign-registry check below is what catches it, and it is the
    only thing that does.
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
# the closed-unit link memory — resolve_link_target
# ==========================================================================


def test_a_remembered_alias_still_resolves_for_linking_after_close():
    """`find()` stays live-only — parentage must never see a corpse — while the
    LINK path can still name the finished predecessor, because a link is
    causality rather than containment and the target's span already shipped.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    unit = open_session(reg)
    key = UnitKey("test.thread", "t-1")
    reg.bind_alias(unit, key, remember=True)
    reg.close(unit)

    assert reg.find(key) is None
    resolved = reg.resolve_link_target(key)
    assert resolved is not None
    assert resolved.trace_id == sink.drafts[-1].context.trace_id
    assert resolved.span_id == sink.drafts[-1].context.span_id


def test_an_unremembered_alias_leaves_no_link_memory():
    """Memory is opt-in PER KEY, not a registry-wide recording of every alias
    namespace: the open()-time key, an `aliases=` entry and a plain
    `bind_alias` all vanish at close exactly as they always have.
    """
    reg = registry()
    unit = open_session(reg, aliases=(UnitKey("extra", "at-open"),))
    reg.bind_alias(unit, UnitKey("extra", "bound"))
    reg.close(unit)

    assert reg.resolve_link_target(UnitKey("test.session", "s1")) is None
    assert reg.resolve_link_target(UnitKey("extra", "at-open")) is None
    assert reg.resolve_link_target(UnitKey("extra", "bound")) is None


def test_the_link_memory_is_fifo_bounded_and_counts_evictions():
    """I10 accounting: the bound holds, and the eviction is a number rather
    than a marker — the remembered span already shipped, so what the eviction
    costs is a link on a FUTURE span, and there is no span yet to say so on.
    """
    reg = registry(max_link_targets=2)
    for i in range(3):
        unit = open_session(reg, f"s{i}")
        reg.bind_alias(unit, UnitKey("mem", str(i)), remember=True)
        reg.close(unit)

    assert reg.resolve_link_target(UnitKey("mem", "0")) is None
    assert reg.resolve_link_target(UnitKey("mem", "1")) is not None
    assert reg.resolve_link_target(UnitKey("mem", "2")) is not None
    assert counters.get("assembly._units.link_memory_full") == 1


def test_a_live_unit_wins_over_memory_and_a_rebind_supersedes_it():
    """The latest-holder rule, both halves. While B holds the key live, B is
    the answer; and because BINDING popped A's memory entry, B closing
    unremembered leaves nothing — a stale predecessor must not resurface once
    a successor owned the key, or a cycle's second iteration would link to its
    first.
    """
    reg = registry()
    key = UnitKey("test.thread", "t-1")
    a = open_session(reg, "a")
    reg.bind_alias(a, key, remember=True)
    reg.close(a)
    assert reg.resolve_link_target(key) == a.context

    b = open_session(reg, "b")
    reg.bind_alias(b, key)
    assert reg.resolve_link_target(key) == b.context

    reg.close(b)
    assert reg.resolve_link_target(key) is None


def test_a_breadth_evicted_alias_is_forgotten_not_remembered():
    """The alias breadth bound governs BOTH lookup paths: a key `find()` can no
    longer answer for must not resurface from the link memory at close.
    (`alias_table_full` already counts the loss.)
    """
    reg = registry(max_entries_per_unit=2)
    unit = open_session(reg)  # its own key takes the first slot
    for i in range(3):
        reg.bind_alias(unit, UnitKey("mem", str(i)), remember=True)
    reg.close(unit)

    # Binding "2" evicted "0" — the oldest remembered key — from both paths.
    assert reg.resolve_link_target(UnitKey("mem", "0")) is None
    assert reg.resolve_link_target(UnitKey("mem", "1")) is not None
    assert reg.resolve_link_target(UnitKey("mem", "2")) is not None


def test_an_evicted_roots_remembered_alias_still_answers():
    """Eviction goes through the same close path as any teardown, so the
    memory records — and linking to an evicted-but-shipped span is honest
    causality: the span is on the wire, marked with what ended it.
    """
    sink = RecordingSink()
    reg = registry(sink=sink, max_units=1)
    first = open_session(reg, "first")
    key = UnitKey("test.thread", "t-1")
    reg.bind_alias(first, key, remember=True)

    open_session(reg, "second")  # evicts `first`; its span ships marked

    assert Limitation.UNIT_EVICTED in first.draft.integrity.markers
    assert reg.resolve_link_target(key) == first.context


def test_bind_alias_on_a_closed_unit_is_refused_and_counted():
    """The same honest refusal `note()` and `open_span()` give a corpse: its
    span already shipped, so the alias could neither be found live nor be
    recorded at a close that already happened.
    """
    reg = registry()
    unit = open_session(reg)
    reg.close(unit)
    key = UnitKey("mem", "late")
    reg.bind_alias(unit, key, remember=True)

    assert reg.find(key) is None
    assert reg.resolve_link_target(key) is None
    assert counters.get("assembly._units.alias_after_close") == 1


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
    # Descendants keep the teardown marker; only a bound-hit entry carries the
    # table-full fact — the root crossed `max_units`, not the breadth knob, so
    # nothing here says UNIT_TABLE_FULL.
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
    # A swap, not an augment: the marker names the knob (`max_entries_per_unit`)
    # rather than claiming a teardown that never happened.
    assert Limitation.UNIT_TABLE_FULL in first.draft.integrity.markers
    assert Limitation.CHILD_SPAN_UNCLOSED not in first.draft.integrity.markers
    assert counters.get("assembly._units.child_table_full") == 1


def test_a_full_open_span_table_evicts_the_oldest_draft_and_emits_it():
    sink = RecordingSink()
    reg = registry(sink=sink, max_entries_per_unit=1)
    root = open_session(reg)
    first = root.open_span(SpanIntent.EXECUTE_TOOL, subject="a")
    first.set_tool(ToolAttributes(name="a"))

    root.open_span(SpanIntent.EXECUTE_TOOL, subject="b")

    assert sink.drafts == [first]
    assert Limitation.UNIT_TABLE_FULL in first.integrity.markers


def test_the_breadth_evicted_span_reaches_the_wire_shape():
    """The `UNIT_EVICTED` twin, for the member this bound now names.

    Not just "a draft was handed over": `finish()` has to accept the marker,
    or the eviction would be the same silent drop with extra steps — a new
    member that is a Python enum entry but not legal vocabulary is exactly
    what this catches.
    """
    sink = RecordingSink()
    reg = registry(sink=sink, max_entries_per_unit=1)
    root = open_session(reg)
    open_subagent(reg, root, "a1")
    open_subagent(reg, root, "a2")

    span = sink.spans()[0]
    assert Limitation.UNIT_TABLE_FULL in span.capture_integrity.limitations
    assert span.capture_sources == (CaptureSource.ADAPTER,)


def test_an_evicted_childs_descendants_keep_the_teardown_marker():
    """Only the entry that HIT the bound reports the table-full fact.

    A descendant closed by the same walk truly was closed by someone else's
    teardown — `CHILD_SPAN_UNCLOSED`'s exact sentence — and stamping the knob's
    name on it would claim a bound it never crossed. The split also keeps the
    root-eviction symmetry: `UNIT_EVICTED` on the root, teardown marker below.
    """
    sink = RecordingSink()
    reg = registry(sink=sink, max_entries_per_unit=1)
    root = open_session(reg)
    c1 = open_subagent(reg, root, "c1")
    g1 = open_subagent(reg, c1, "g1")

    open_subagent(reg, root, "c2")  # evicts c1's whole subtree

    assert c1.is_live is False and g1.is_live is False
    assert Limitation.UNIT_TABLE_FULL in c1.draft.integrity.markers
    assert Limitation.CHILD_SPAN_UNCLOSED not in c1.draft.integrity.markers
    assert Limitation.CHILD_SPAN_UNCLOSED in g1.draft.integrity.markers
    assert Limitation.UNIT_TABLE_FULL not in g1.draft.integrity.markers


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


def test_lowered_body_cap_bounds_what_a_unit_retains():
    """The user-visible end of the body cap: what a unit KEEPS, and the tally.

    Deliberately not a `tracemalloc` measurement. `_append_capped` copies
    `data[:room]`, so the caller's own 8 MiB argument is alive for as long as
    the assertion is, and a total taken here reads ~8 MiB whether the cap is
    honoured or not. What the cap decides is the length of the bytes on the
    span, and that is what is asserted — together with the flag that says
    something was dropped and the counter that makes "how much am I losing
    since I lowered it" answerable at all.
    """
    sink = RecordingSink()
    reg = UnitRegistry(sink=sink, max_body_bytes=64 * 1024)
    root = open_session(reg)
    before = counters.get("assembly._units.record_truncated")
    root.record_input(b"x" * (8 * 1024 * 1024))

    reg.close(root)
    span = sink.spans()[0]

    assert len(span.input_data) == 64 * 1024
    assert span.capture_integrity.truncated is True
    assert counters.get("assembly._units.record_truncated") - before == 1


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
    handler and can run the host's `before_send_envelope` on the calling thread; the
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
    from wardex_sdk._assembly import resolve_parentage

    return resolve_parentage(EMPTY_AMBIENT).child_context()


def test_close_all_declines_when_this_thread_already_holds_the_lock():
    """The registry lock is an `RLock`, so the failure this guards is not a hang.

    A signal handler runs on the main thread wherever the interpreter happened
    to be, which includes the middle of a registry mutation. Re-entering an
    RLock succeeds, so `close_all` would walk `_roots` while an outer frame is
    partway through updating it and emit spans built from state nobody meant to
    be readable. Declining costs a teardown that was already racing a dying
    process.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    root = open_session(reg)

    with reg._lock:
        reg.close_all(reason=Limitation.UNIT_INTERRUPTED)
        assert sink.drafts == []
        assert root.is_live

    assert counters.get("assembly._units.close_all_reentrant") == 1

    reg.close_all(reason=Limitation.UNIT_INTERRUPTED)
    assert not root.is_live
    assert len(sink.drafts) == 1


def test_a_non_blocking_acquire_could_not_be_that_guard():
    """Pins the SPELLING, because the obvious simplification is silently broken.

    `acquire(blocking=False)` reads like "is the lock free?" and on an `RLock`
    it is not: the owning thread's non-blocking acquire SUCCEEDS. A guard
    written that way passes every re-entry straight through while looking, in
    review and in a green suite, exactly like a guard.
    """
    reg = registry()

    with reg._lock:
        acquired = reg._lock.acquire(blocking=False)
        if acquired:
            reg._lock.release()
        assert acquired is True, "an RLock stopped re-admitting its owner"
        assert reg._lock._is_owned() is True

    assert reg._lock._is_owned() is False


def test_the_registry_lock_must_stay_reentrant():
    """`close_all` re-enters its own lock through `Unit.note()`, with no signal
    involved. Spelling the lock as a plain `Lock` — the reflex when a comment
    says "held from a signal handler" — deadlocks the ordinary teardown.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    open_session(reg)

    done: list[bool] = []

    def drive():
        reg.close_all(reason=Limitation.UNIT_INTERRUPTED)
        done.append(True)

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    thread.join(timeout=5.0)

    assert done == [True], "close_all did not return — the lock is no longer reentrant"
    assert len(sink.drafts) == 1


# ==========================================================================
# owner — whose unit is it, once one registry serves several adapters
# ==========================================================================


def test_sole_live_without_an_owner_answers_about_the_whole_process():
    """Today's meaning, pinned so the filter cannot change it by accident."""
    reg = registry()
    a = open_session(reg, "a")

    assert reg.sole_live(UnitKind.SESSION) is a

    open_session(reg, "b")
    assert reg.sole_live(UnitKind.SESSION) is None


def test_sole_live_filters_to_one_adapters_units():
    """The whole reason the filter exists.

    "Exactly one SESSION is live" is a question about the PROCESS, and once two
    frameworks share a registry the answer stops being about the asker. Two live
    runs make the unfiltered question give up; filtered, each adapter still sees
    its own.
    """
    reg = registry()
    mine = open_session(reg, "mine", owner="anthropic")
    theirs = open_session(reg, "theirs", owner="langgraph")

    assert reg.sole_live(UnitKind.SESSION) is None
    assert reg.sole_live(UnitKind.SESSION, owner="anthropic") is mine
    assert reg.sole_live(UnitKind.SESSION, owner="langgraph") is theirs


def test_an_unowned_unit_matches_no_owner_rather_than_every_owner():
    """The direction an omitted owner degrades in, and it is not the tidy one.

    Treating `owner is None` as "matches anything" would hand an adapter a unit
    nobody said was its own — and this method's caller stamps that guess at 0.5
    and ships it. Guessing across an unstated boundary is precisely what the
    filter exists to stop, so it may not be how the filter fails open.
    """
    reg = registry()
    open_session(reg, "unowned")

    assert reg.sole_live(UnitKind.SESSION) is not None, "unfiltered still sees it"
    assert reg.sole_live(UnitKind.SESSION, owner="anthropic") is None


def test_close_all_with_an_owner_leaves_another_adapters_run_alone():
    """An Anthropic uninstall must not end a LangGraph run that is still driven."""
    sink = RecordingSink()
    reg = registry(sink=sink)
    mine = open_session(reg, "mine", owner="anthropic")
    theirs = open_session(reg, "theirs", owner="langgraph")

    reg.close_all(reason=Limitation.ADAPTER_UNINSTALLED, owner="anthropic")

    assert not mine.is_live
    assert theirs.is_live
    assert len(sink.drafts) == 1


def test_close_all_with_an_owner_terminates_when_the_first_root_is_not_its_own():
    """`while self._roots` was correct only while the loop emptied the table.

    With a filter it does not: the first root may be one this call must NOT
    close, and `while` would take it again on every pass. A hang, not a wrong
    answer — and one that only appears once a second adapter exists, which is
    after the code shipped.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    first = open_session(reg, "theirs", owner="langgraph")  # inserted FIRST
    second = open_session(reg, "mine", owner="anthropic")

    done: list[bool] = []

    def drive():
        reg.close_all(reason=Limitation.ADAPTER_UNINSTALLED, owner="anthropic")
        done.append(True)

    thread = threading.Thread(target=drive, daemon=True)
    thread.start()
    thread.join(timeout=5.0)

    assert done == [True], "close_all did not return — the owner filter spins"
    assert first.is_live
    assert not second.is_live


def test_another_registrys_dead_pin_is_not_this_registrys_conflict():
    """`_ambient_unit` is process-wide; a unit belongs to one registry.

    A re-`init()` builds a new registry while the previous one's pin stays
    reachable on the carrier. Once that older run closes, its corpse read back
    here as THIS registry's stale pin — so an ordinary scope in front of us
    would have been refused and the span stamped `CORRELATION_CONFLICT`,
    charging a conflict to a pin this registry never installed.
    """
    old = registry()
    root = open_session(old, "old")
    old.pin_driver(root, owner_task=threading.current_thread())
    old.close(root)

    fresh = registry()

    assert fresh.closed_unit_in_scope() is False
    assert fresh.becomes_trace_root(latch_ambient()) is False
    # One per question asked, not one per foreign pin: both predicates above
    # consult it, and each refusal is a real refusal to record.
    assert counters.get("assembly._units.stale_pin_foreign_registry") >= 1

    # And the ordinary scope survives: a unit opened here takes the ambient the
    # older run left standing rather than becoming an orphan that blames it.
    unit = open_session(fresh, "new", ambient=latch_ambient())
    assert Limitation.CORRELATION_CONFLICT not in unit.draft.integrity.markers


# ==========================================================================
# A wardex bug costs the smallest thing it can
# ==========================================================================


def test_a_fault_binding_an_alias_costs_the_alias_and_not_the_unit():
    """The guard's PLACEMENT is the assertion, not its presence.

    By the time `open()` binds aliases the unit is already in `_roots` and
    `_live_units`, so a fault there is not "the open failed" — it is "the open
    succeeded and the lookup table did not". Guarded from OUTSIDE, the raise is
    contained one level up and the unit is LEAKED: registered, live, reachable
    from no caller, counting against `max_units` until it evicts a real session
    to make room for a phantom — once per call, and invisible until traces stop
    appearing. Guarded from inside, the fault costs the alias and nothing else.
    """

    class Broken(UnitRegistry):
        __slots__ = ()

        def _bind_alias_locked(self, key, unit):
            raise RuntimeError("the alias table is gone")

    sink = RecordingSink()
    reg = Broken(sink=sink)

    unit = open_session(reg, "s1")

    # The unit came back live and fully registered: the span is not lost.
    assert unit.is_live
    reg.close(unit)
    assert len(sink.spans()) == 1

    # What WAS lost is the lookup, and only the lookup.
    assert reg.find(UnitKey("test.session", "s1")) is None
    assert counters.get("assembly._units.open_bind") >= 1

    # And nothing leaked: the counterfactual an outside guard produces is a
    # live root per call that no later sweep can reach.
    assert len(reg._live_units) == 0
    assert len(reg._roots) == 0


def _fifteen_owed(reg) -> list:
    """3 roots x 4 children. Fifteen spans, three subtrees, one shared table."""
    roots = [open_session(reg, f"root{i}", subject=f"root{i}") for i in range(3)]
    for i, root in enumerate(roots):
        for j in range(4):
            open_subagent(reg, root, f"a{i}{j}")
    return roots


def test_one_span_that_cannot_be_built_costs_only_that_span():
    """15 owed, ONE child whose span cannot be built. Fourteen ship.

    Four shapes have been measured on exactly this fixture, and the numbers are
    the argument:

      one boundary around the whole sweep        0 of 15
      one per root                              10 of 15, root #0 wedged forever
      one per root, plus evicting the poisoned  10 of 15
      collect-then-detach, one boundary per span 14 of 15   <- this

    The first three all lose a SUBTREE for one bad draft, because building spans
    and unlinking units were interleaved: a fault partway left the parent marked
    dead, some children unlinked, and the drafts already built dropped on the
    floor with the raise. Splitting the phases makes the loss what it should
    always have been — the one span whose draft is broken.
    """
    victim: list = []

    class BrokenDraft(UnitRegistry):
        __slots__ = ()

    real = Unit._finalize_locked

    def blow(self, **kw):
        if victim and self is victim[0]:
            raise RuntimeError("this draft cannot be stamped")
        return real(self, **kw)

    sink = RecordingSink()
    reg = BrokenDraft(sink=sink)
    roots = _fifteen_owed(reg)
    victim.append(next(iter(roots[0]._children)))

    Unit._finalize_locked = blow
    try:
        reg.close_all(reason=Limitation.ADAPTER_UNINSTALLED)
    finally:
        Unit._finalize_locked = real

    assert len(sink.drafts) == 14
    names = {d.name for d in sink.drafts}
    # Its PARENT ships, its siblings ship, the other two roots ship. Under the
    # interleaved shape the parent was the first thing lost.
    assert "invoke_agent root0" in names
    assert "invoke_agent root1" in names
    assert "invoke_agent root2" in names
    assert counters.get("assembly._units.close_finalize") == 1

    # And the tables end consistent: nothing left standing, nothing unreachable.
    assert len(reg._roots) == 0
    assert len(reg._live_units) == 0
    reg.close_all(reason=Limitation.ADAPTER_UNINSTALLED)
    assert len(sink.drafts) == 14


def test_a_subtree_the_sweep_cannot_close_at_all_still_costs_only_itself():
    """The backstop under the backstop.

    `_close_locked` is now two phases, neither of which can take a subtree down
    — but `close_all` keeps its per-root boundary anyway, because "this function
    cannot fail" is a property of today's code and not of the next edit. Faulted
    wholesale, one root's failure still costs one root: the other two sweep
    normally, and the one that could not be closed is dropped from the table
    rather than left in it for every later sweep to re-trip on.
    """
    victim: list = []

    class Broken(UnitRegistry):
        __slots__ = ()

        def _close_locked(self, unit, **kw):
            if victim and unit is victim[0]:
                raise RuntimeError("this subtree is torn")
            return super()._close_locked(unit, **kw)

    sink = RecordingSink()
    reg = Broken(sink=sink)
    roots = _fifteen_owed(reg)
    victim.append(roots[0])

    reg.close_all(reason=Limitation.ADAPTER_UNINSTALLED)

    assert len(sink.drafts) == 10
    names = {d.name for d in sink.drafts}
    assert "invoke_agent root1" in names
    assert "invoke_agent root2" in names
    assert "invoke_agent root0" not in names

    assert len(reg._roots) == 0
    assert counters.get("assembly._units.close_all_root") == 1

    # A second sweep is quiet: it finds nothing and does not re-trip.
    reg.close_all(reason=Limitation.ADAPTER_UNINSTALLED)
    assert len(sink.drafts) == 10
    assert counters.get("assembly._units.close_all_root") == 1


def test_the_unlink_phase_does_not_depend_on_any_draft():
    """Phase two claims it cannot fail, and the claim rests entirely on it
    touching nothing but wardex's own dicts and lists.

    Asserted by breaking EVERY draft in the subtree rather than by reading the
    code: no span can be built and no marker can be recorded, so nothing ships —
    and the tables still end empty and consistent, which is the half that keeps
    a registry usable after a bad teardown instead of leaving it holding fifteen
    units nothing can reach.

    Written this way it found one: `close_all` stamped its shutdown marker on
    the root INSIDE the per-root boundary and before the close, so a draft that
    could not take the marker skipped the close entirely and the eviction
    dropped the root without walking under it. Twelve children, reachable from
    no root of any registry. The marker is its own step now.
    """
    sink = RecordingSink()
    reg = registry(sink=sink)
    _fifteen_owed(reg)

    real_finalize, real_note = Unit._finalize_locked, Unit.note

    def blow(self, *a, **kw):
        raise RuntimeError("every draft in this subtree is broken")

    Unit._finalize_locked, Unit.note = blow, blow
    try:
        reg.close_all(reason=Limitation.ADAPTER_UNINSTALLED)
    finally:
        Unit._finalize_locked, Unit.note = real_finalize, real_note

    assert sink.drafts == [], "a span was built on a path that must not build one"
    assert len(reg._roots) == 0
    assert len(reg._live_units) == 0
    assert len(reg._by_alias) == 0
    assert counters.get("assembly._units.close_finalize") == 15
    assert counters.get("assembly._units.close_all_note") == 3


# ==========================================================================
# design §10.3(a) — a CLOSED unit's leftover carrier, PINNED OR NOT
# ==========================================================================


def _stranded(reg: UnitRegistry, unit):
    """A unit closed from ANOTHER carrier while its `activate()` is still entered.

    The LangGraph shape without LangGraph. `Pregel.stream` installs the fork on
    the carrier that pumps the first `next()`; the fork can only come down when
    the generator is FINALIZED, on whatever carrier finalizes it. A
    `close_units()` mid-stream, or a finalization on a foreign thread, closes
    the unit from a task that cannot reset that Token — so the fork is left
    standing, holding the context of a span that has already SHIPPED.

    The activation CM is RETURNED and the caller must keep it alive.
    `activate_span` is generator-backed, so dropping the last reference runs its
    `finally` and takes the fork down — which is the clean case, not this one.
    In production the reference is the suspended generator the framework holds.
    """
    cm = unit.activate()
    cm.__enter__()
    closer = threading.Thread(target=lambda: reg.close(unit))
    closer.start()
    closer.join()
    return cm


def test_a_closed_activations_fork_is_refused_exactly_like_a_dead_pins():
    """`pinned` was never what made the leftover dangerous.

    A pin and an `activate()` install the SAME two carriers through the same
    `_Carrier.__init__`, branching only on which of them `close()` can normally
    take down — and the failure is the case where NEITHER was taken down.
    Gating every refusal on `entry.pinned` therefore passed one of two identical
    corpses straight through: `contextvar` / 1.0 / no marker into a span that
    had already been emitted, which is one trace where two belong and nothing on
    the wire to find it by.
    """
    reg = registry()
    unit = open_session(reg)
    _fork = _stranded(reg, unit)

    assert not unit.is_live
    assert latch_ambient().span_context == unit.context, "the standing fork is the precondition"

    assert reg.closed_unit_in_scope() is True
    assert reg.becomes_trace_root(latch_ambient()) is True

    p = reg.resolve(None)

    assert p.parent_span_id is None
    assert p.correlation.strategy is not ParentSource.CONTEXTVAR
    assert p.correlation.confidence < 1.0
    assert Limitation.CORRELATION_CONFLICT in p.limitations


def test_a_stranded_activation_and_a_stranded_pin_are_counted_apart():
    """Two bugs, two repairs, so two names — the same split `current()` already
    makes between `pin_stale`/`pin_leaked` and `ambient_stale`.

    `stale_pin_ambient` says `pin_driver`'s contract was broken and the fix is
    in WHICH TASK the adapter pinned. `stale_activation_ambient` says a scope
    could not be unwound where it was installed and the fix is in the adapter's
    LIFETIME. One number for both leaves an operator two hypotheses and no way
    to separate them.
    """
    reg = registry()
    unit = open_session(reg)
    _fork = _stranded(reg, unit)

    reg.resolve(None)

    assert counters.get("assembly._units.stale_activation_ambient") == 1
    assert counters.get("assembly._units.stale_pin_ambient") == 0


def test_a_unit_opened_over_a_dead_activation_is_a_root_that_says_so():
    """The measured failure as a tree: an ENTIRE later run adopted by a
    finished one. A run that starts after the previous one died is a NEW run,
    not a subtree of the corpse the carrier is still holding.
    """
    reg = registry()
    dead = open_session(reg, "dead")
    _fork = _stranded(reg, dead)

    later = open_session(reg, "next", ambient=latch_ambient())

    assert later.parentage.parent_span_id is None
    assert later.context.trace_id != dead.context.trace_id
    assert Limitation.CORRELATION_CONFLICT in later.draft.integrity.markers


def test_a_real_span_over_a_dead_activation_is_still_a_parent():
    """The no-false-positive rule, restated for the branch the widening added.

    The refusal is the leftover FORK, not the task. A host that opened its own
    span inside that task put a real parent on top, and moving live work out of
    the host's trace to escape a ghost that is no longer in front of us is a
    wrong tree of its own shape. `_poisoned` stayed an IDENTITY test through
    the widening, and this is why.
    """
    reg = registry()
    unit = open_session(reg)
    _fork = _stranded(reg, unit)
    host = _a_context()

    with activate_span(host):
        p = reg.resolve(None)
        assert reg.becomes_trace_root(latch_ambient()) is False

    assert p.parent_span_id == host.span_id
    assert p.correlation.strategy is ParentSource.CONTEXTVAR
    assert Limitation.CORRELATION_CONFLICT not in p.limitations
    assert counters.get("assembly._units.stale_activation_ambient") == 0
    assert counters.get("assembly._units.stale_pin_ambient") == 0


def test_a_clean_activation_leaves_nothing_to_refuse():
    """THE constraint: the ordinary sequential shape must produce ZERO signals.

    Opened, activated, exited, closed on ONE carrier — the fork comes down with
    the `with`. A widening that refused on "this task once descended from a unit
    that is now dead" would fire here, on the shape that is the product working
    correctly, and a fix that marks healthy runs is worse than the bug.
    """
    reg = registry()
    first = open_session(reg)
    with first.activate():
        pass
    reg.close(first)

    assert latch_ambient().span_context is None
    assert reg.closed_unit_in_scope() is False

    later = open_session(reg, "next", ambient=latch_ambient())

    assert later.parentage.parent_span_id is None
    assert later.parentage.limitations == ()
    for name in (
        "stale_pin_ambient",
        "stale_activation_ambient",
        "stale_pin_foreign_registry",
        "ambient_stale",
        "ambient_closed_at_issue",
        "pin_stale",
        "pin_leaked",
    ):
        assert counters.get(f"assembly._units.{name}") == 0, name


def test_a_live_activation_from_another_registry_is_neither_refused_nor_counted():
    """Two adapters in one process is the ORDINARY state, not an incident.

    `_ambient_unit` is process-wide, so registry B sees registry A's perfectly
    healthy `activate()` on every `open()`. Nothing may be refused — and the
    foreign bump stays gated on `entry.pinned` precisely so that
    `stale_pin_foreign_registry`, which audits PIN DISCIPLINE, does not start
    firing once per open in a two-adapter process and bury the signal it exists
    to carry.
    """
    reg_a = registry()
    reg_b = registry()
    live = open_session(reg_a, "theirs", owner="langgraph")

    with live.activate():
        assert reg_b.closed_unit_in_scope() is False
        assert reg_b.becomes_trace_root(latch_ambient()) is False
        mine = open_session(reg_b, "mine", ambient=latch_ambient(), owner="anthropic")

    assert Limitation.CORRELATION_CONFLICT not in mine.draft.integrity.markers
    assert counters.get("assembly._units.stale_pin_foreign_registry") == 0


def test_another_registrys_dead_activation_is_not_this_registrys_conflict():
    """Ownership still comes FIRST, for the non-pinned leftover too.

    The twin of `test_another_registrys_dead_pin_is_not_this_registrys_conflict`
    for the shape the widening added. The ambient carrier is process-wide and a
    unit is not, so charging THIS registry's spans with a `CORRELATION_CONFLICT`
    for a corpse it never installed reports a conflict nobody can act on — while
    the scope in front of us is somebody else's problem.
    """
    old = registry()
    dead = open_session(old, "old")
    _fork = _stranded(old, dead)

    fresh = registry()

    assert fresh.closed_unit_in_scope() is False
    assert fresh.becomes_trace_root(latch_ambient()) is False

    unit = open_session(fresh, "new", ambient=latch_ambient())
    assert Limitation.CORRELATION_CONFLICT not in unit.draft.integrity.markers


def test_an_eviction_that_strands_its_own_activation_orphans_what_follows():
    """A DELIBERATE consequence of the widening, pinned so it cannot go silent.

    A root evicted by table pressure while its `activate()` is still entered is
    closed and emitted (with `UNIT_EVICTED`) — the fork stands, and what
    follows is refused and becomes a marked orphan. NOT a third liveness
    state: the refusal is the one every stranded fork gets. What the
    breadcrumb (`Unit._evicted`) changes is the WORD: this strand is wardex's
    own bound at work, so the orphan carries `INSTRUMENTATION_DEGRADED` — the
    repair is `max_units` — where a pin or lifetime bug carries
    `CORRELATION_CONFLICT` and sends the reader to the adapter. One marker for
    both was this test's previous pin, and it filed a capacity decision under
    adapter discipline.
    """
    reg = registry(max_units=1)
    evicted = open_session(reg, "A")
    fork = evicted.activate()
    fork.__enter__()
    open_session(reg, "B")  # evicts A: closed, span emitted with UNIT_EVICTED

    assert evicted.is_live is False
    assert latch_ambient().span_context == evicted.context

    after = open_session(reg, "C", ambient=latch_ambient())

    assert after.parentage.parent_span_id != evicted.context.span_id
    assert Limitation.INSTRUMENTATION_DEGRADED in after.draft.integrity.markers
    assert Limitation.CORRELATION_CONFLICT not in after.draft.integrity.markers
    assert counters.get("assembly._units.stale_ambient_evicted") == 1
    assert counters.get("assembly._units.stale_activation_ambient") == 0


def test_resolve_after_an_eviction_strand_reports_wardex_fault():
    """Pins that BOTH refusal sites route through `refused_ambient_marker`.

    `open()` is covered above; a `resolve()` over the same stranded fork must
    say the same word, or the attribution would depend on which entry point
    the adapter happened to use.
    """
    reg = registry(max_units=1)
    evicted = open_session(reg, "A")
    fork = evicted.activate()
    fork.__enter__()
    open_session(reg, "B")

    assert evicted.is_live is False
    assert latch_ambient().span_context == evicted.context

    p = reg.resolve(None)

    # The edge is rebuilt from the remaining tiers — here the sole-live
    # session B — never from the corpse in the scope.
    assert p.parent_span_id != evicted.context.span_id
    assert Limitation.INSTRUMENTATION_DEGRADED in p.limitations
    assert Limitation.CORRELATION_CONFLICT not in p.limitations
    assert counters.get("assembly._units.stale_ambient_evicted") == 1


def test_an_ordinary_close_leftover_still_reads_as_a_conflict():
    """The negative control: the breadcrumb never fires on a teardown.

    An `activate()` scope stranded by an ordinary `close()` keeps
    `CORRELATION_CONFLICT` and the lifetime counter — the swap cannot dilute
    that member's meaning, or every stranded fork would read as wardex's
    fault and the adapter bug it points at would go unhunted.
    """
    reg = registry()
    unit = open_session(reg, "A")
    _fork = _stranded(reg, unit)

    after = open_session(reg, "B", ambient=latch_ambient())

    assert Limitation.CORRELATION_CONFLICT in after.draft.integrity.markers
    assert Limitation.INSTRUMENTATION_DEGRADED not in after.draft.integrity.markers
    assert counters.get("assembly._units.stale_activation_ambient") == 1
    assert counters.get("assembly._units.stale_ambient_evicted") == 0


def test_a_child_eviction_that_strands_its_fork_is_wardex_fault_too():
    """The breadcrumb covers BOTH capacity-bound close paths.

    A CHILD unit evicted by the breadth bound (`max_entries_per_unit`) while
    its `activate()` is entered strands exactly the same corpse as a root
    evicted by `max_units` — without the second write site, this strand would
    keep the misattributed `CORRELATION_CONFLICT` the swap exists to remove.
    """
    reg = registry(max_entries_per_unit=1)
    root = open_session(reg)
    c1 = open_subagent(reg, root, "c1")
    fork = c1.activate()
    fork.__enter__()
    open_subagent(reg, root, "c2")  # evicts c1 through the child-table bound

    assert c1.is_live is False
    assert latch_ambient().span_context == c1.context

    after = open_session(reg, "after", ambient=latch_ambient())

    assert Limitation.INSTRUMENTATION_DEGRADED in after.draft.integrity.markers
    assert Limitation.CORRELATION_CONFLICT not in after.draft.integrity.markers
    assert counters.get("assembly._units.stale_ambient_evicted") == 1


# ==========================================================================
# design §10.3(b) — the same fact, for callers that hold no registry
# ==========================================================================


def test_a_closed_units_span_is_nameable_without_a_registry():
    """`parent_is_closed_unit` is what the byte seams and MCP stdio can ask.

    They hold a `Client` and nothing else, and `_interceptors/` may not import
    `_adapters/`, where the registries are built. It takes no registry on
    purpose: "the span I latched has already shipped" does not depend on who
    opened the unit, and a seam charges no `CORRELATION_CONFLICT` to anyone —
    the ownership guard exists to stop one registry blaming another's teardown,
    which is a registry-path concern and stays on the registry path.
    """
    reg = registry()
    unit = open_session(reg)
    cm = unit.activate()
    cm.__enter__()
    try:
        assert parent_is_closed_unit(unit.context) is False, "a LIVE unit is never refused"
        assert counters.get("assembly._units.ambient_closed_at_issue") == 0

        closer = threading.Thread(target=lambda: reg.close(unit))
        closer.start()
        closer.join()

        assert parent_is_closed_unit(unit.context) is True
        assert counters.get("assembly._units.ambient_closed_at_issue") == 1
    finally:
        del cm


def test_a_real_span_over_a_dead_activation_is_not_a_closed_unit_either():
    """The identity narrowing, for the registry-free predicate.

    Same rule as `_poisoned`: a host span or a nested activation standing on top
    of the leftover is a real parent, and a seam that refused it would move live
    traffic out of the host's trace. And `None` claims nothing — a request
    issued with no ambient at all is an honest trace root, not a corpse.
    """
    reg = registry()
    unit = open_session(reg)
    _fork = _stranded(reg, unit)
    host = _a_context()

    assert parent_is_closed_unit(host) is False
    assert parent_is_closed_unit(None) is False
    assert counters.get("assembly._units.ambient_closed_at_issue") == 0


# --------------------------------------------------------------------------
# design §10.3(b) — the two HAND-WRITTEN sites, not just the byte seams
# --------------------------------------------------------------------------


def test_a_manual_span_does_not_become_a_child_of_a_span_that_already_shipped():
    """`wardex.span()` is issued on the same carriers an adapter runs on.

    It latches a parent it did not open and cannot vet, which is the whole
    definition of an OBSERVING site — so it asks the same question the byte
    seams ask. Before it did, a hand-written span opened after a graph run's
    unit had closed became a full-confidence child of an already-emitted span,
    in that span's own trace, with nothing on the wire to say so. The byte seam
    beside it got this right and the published API did not, which is the
    inconsistency this closes.
    """
    from wardex_sdk import _hub, span

    reg = registry()
    unit = open_session(reg)
    _fork = _stranded(reg, unit)
    client = _RecordingClient()
    _hub.set_client(client)
    try:
        with span("after-the-run"):
            pass
    finally:
        _hub.reset_for_test()

    emitted = [s for s in client.spans if s.name == "after-the-run"]
    assert len(emitted) == 1, "the span must still SHIP — refusing a parent is not dropping work"
    manual = emitted[0]
    assert manual.parent_span_id != unit.context.span_id
    assert manual.parent_span_id is None
    assert manual.context.trace_id != unit.context.trace_id
    assert manual.correlation.strategy is ParentSource.UNRESOLVED
    assert manual.correlation.confidence == 0.0
    assert counters.snapshot().get("assembly._units.ambient_closed_at_issue") == 1


def test_a_state_snapshot_does_not_describe_the_state_of_a_run_that_ended():
    """The same refusal, and the same `unresolved` — reached by its own route.

    A snapshot's no-parent answer is ALREADY `unresolved` rather than a trace
    root, because a snapshot describes the state of a span and one without a
    span to describe has lost something. So a corpse collapses onto that branch
    instead of through `resolve_observed`, whose no-parent answer is a root.
    Same decision function, same outcome, one place each states its own
    consequence.
    """
    from wardex_sdk import _hub, capture_state_snapshot

    reg = registry()
    unit = open_session(reg)
    _fork = _stranded(reg, unit)
    client = _RecordingClient()
    _hub.set_client(client)
    try:
        capture_state_snapshot(snapshot_type="turn_start", conversation_state=b"{}")
    finally:
        _hub.reset_for_test()

    assert len(client.snapshots) == 1
    snap = client.snapshots[0]
    # A snapshot carries the span it DESCRIBES, so "refused" means it no longer
    # names the dead unit's span or sits in its trace.
    assert snap.span_id != unit.context.span_id
    assert snap.trace_id != unit.context.trace_id


def test_a_LIVE_parent_is_still_the_parent_of_both_hand_written_sites():
    """The control, and the one that would catch an over-eager refusal.

    Neither site may refuse a unit that is simply still running — that is the
    ordinary case and the whole product. Without this, a predicate that always
    said "closed" would pass every assertion above.
    """
    from wardex_sdk import _hub, span

    reg = registry()
    unit = open_session(reg)
    client = _RecordingClient()
    _hub.set_client(client)
    try:
        with unit.activate():
            with span("inside-the-run"):
                pass
    finally:
        _hub.reset_for_test()

    manual = [s for s in client.spans if s.name == "inside-the-run"][0]
    assert manual.parent_span_id == unit.context.span_id
    assert manual.context.trace_id == unit.context.trace_id
    assert manual.correlation.confidence == 1.0
    assert "ambient_closed_at_issue" not in str(counters.snapshot())


class _Config:
    debug = False


class _RecordingClient:
    """Enough client for the two published entry points above."""

    config = _Config()

    def __init__(self) -> None:
        self.spans: list = []
        self.snapshots: list = []

    def capture_span(self, span) -> None:
        self.spans.append(span)

    def capture_snapshot(self, snapshot) -> None:
        self.snapshots.append(snapshot)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None
