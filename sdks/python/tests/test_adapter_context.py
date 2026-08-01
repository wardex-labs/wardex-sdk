"""`adapters/_context.py` — the surface a framework adapter is handed.

Two things are under test and they are not the same thing.

The PARENTAGE TABLE is the product claim written as a pure function of (which
method) x (declared placement) x (what was installed on the carrier). Every row
is asserted, because the rows are what an adapter author never gets to decide.

The SHAPE is the other half, and it is what makes the table hold for adapters
nobody has written yet. `Scope` and `Attachment` deliberately expose no value
that IS a parent — no context, no parentage, no unit, no activate — so an
adapter cannot carry one to another carrier and install it there. A rule saying
"don't do that" is defeated by a careless author; a vocabulary with no word for
it is not.

The bold row is `NESTED` with nothing installed. Measured on real langgraph: a
run entry the adapter forgot to wrap turns one graph run into four separate
traces, every orphan reading `trace_root` / 1.0 / no marker — byte-identical to
a legitimate run, so a backend cannot tell a shattered run from four real ones.
Declaring placement per site is what converts that silence into a span that says
it lost something.
"""

from __future__ import annotations

import threading

import pytest

from wardex_sdk._enums import AgentType, StatusCode
from wardex_sdk._hub import reset_for_test
from wardex_sdk._types import AgentAttributes, ToolAttributes
from wardex_sdk.adapters._context import (
    AdapterContext,
    Attachment,
    Fallback,
    InstallOutcome,
    Observer,
    Placement,
    RunHandle,
    Scope,
)
from wardex_sdk.assembly import (
    EMPTY_AMBIENT,
    Limitation,
    LinkReason,
    ParentSource,
    SpanIntent,
    UnitKey,
    UnitKind,
    UnitRegistry,
    counters,
)
from wardex_sdk.assembly._builder import NULL_DRAFT
from wardex_sdk.assembly._diag import reset_reports_for_test
from wardex_sdk.assembly._units import _ambient_unit
from wardex_sdk.context import activate_span


class RecordingSink:
    def __init__(self) -> None:
        self.drafts: list = []

    def emit(self, draft, *, agent_semantic: bool) -> bool:
        self.drafts.append(draft)
        return True


@pytest.fixture(autouse=True)
def _clean_scope():
    reset_for_test()
    token = _ambient_unit.set(None)
    counters.reset()
    # `_REPORTED` is process-global, so without this the ORDER of tests decides
    # what a later one sees on stderr — a one-line-per-key channel is silent for
    # every test after the first that trips the same site.
    reset_reports_for_test()
    yield
    _ambient_unit.reset(token)
    reset_for_test()
    counters.reset()
    reset_reports_for_test()


def context(name: str = "test", **kw) -> tuple[AdapterContext, RecordingSink]:
    sink = kw.pop("sink", RecordingSink())
    return AdapterContext(name, units=UnitRegistry(sink=sink), limits={}, **kw), sink


def broken(where: str, name: str = "probe", **kw) -> tuple[AdapterContext, RecordingSink]:
    """A context whose registry raises at exactly one method.

    SUBCLASSED, not monkeypatched on the instance: `UnitRegistry` uses
    `__slots__`, so `reg.open = ...` is `AttributeError: read-only`. That
    constraint is why every fault in this file arrives this way.
    """
    sink = kw.pop("sink", RecordingSink())

    def blow(self, *a, **k):
        raise RuntimeError(f"wardex is broken at {where}")

    cls = type("Broken", (UnitRegistry,), {"__slots__": (), where: blow})
    return AdapterContext(name, units=cls(sink=sink), limits={}, **kw), sink


def _agent(handle) -> None:
    """`invoke_agent` needs an agent block or the closed vocabulary refuses it."""
    handle.draft.set_agent(AgentAttributes(name="a", agent_type=AgentType.PRIMARY))


def _tool(handle) -> None:
    """`execute_tool` requires a tool block, for the same reason `_step` does."""
    handle.draft.set_tool(ToolAttributes(name="t"))


def _step(handle) -> None:
    """`execute_step` requires `wardex.step.name` — the vocabulary says a step
    span that cannot say WHICH step is not worth shipping."""
    handle.draft.set_extra("wardex.step.name", "n")


def _edge(sink):
    """The edge as it SHIPS, read off the emitted span.

    Not off the handle, and that is the point rather than an inconvenience:
    `Scope` exposes no parentage, because a value that describes a parent is one
    step from a value that IS a parent. The wire is the only place the edge is
    legible, which is also the only place it matters.
    """
    return _edge_of(sink.drafts[-1].finish())


def _edge_of(span):
    c = span.correlation
    # `capture_integrity` is None on a span with nothing to report, which is the
    # shape most of these assertions are looking for.
    integrity = span.capture_integrity
    return c.strategy, c.confidence, tuple(integrity.limitations) if integrity else ()


def _emitted(sink):
    return [d.finish() for d in sink.drafts]


def _a_context():
    from wardex_sdk._types import SpanContext, SpanId, TraceId

    return SpanContext(TraceId.generate(), SpanId.generate())


# ==========================================================================
# The parentage table
# ==========================================================================


def test_a_root_site_with_nothing_above_it_starts_a_trace():
    """The legitimate case: a framework run entry, called from ordinary code."""
    ctx, sink = context()

    with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT) as s:
        _agent(s)

    assert _edge(sink) == (ParentSource.TRACE_ROOT, 1.0, ())


def test_a_nested_site_with_nothing_above_it_says_so_instead_of_inventing_a_root():
    """THE graft, and the reason placement cannot be defaulted or inferred.

    Same call, same carrier, one word different — and the difference is between
    a span that quietly claims to be a whole run and one that reports the run it
    belonged to is missing. Measured on real langgraph, the first shape turned
    one graph invocation into four traces that no consumer could tell apart from
    four genuine ones.
    """
    ctx, sink = context()

    with ctx.enter(UnitKind.STEP, intent=SpanIntent.EXECUTE_STEP, placement=Placement.NESTED) as st:
        _step(st)

    source, confidence, markers = _edge(sink)
    assert source is ParentSource.UNRESOLVED
    assert confidence == 0.0
    assert Limitation.PARENT_UNRESOLVED in markers


def test_a_nested_site_inside_another_scope_takes_that_scope():
    ctx, sink = context()

    with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT) as r:
        _agent(r)
        with ctx.enter(
            UnitKind.STEP, intent=SpanIntent.EXECUTE_STEP, placement=Placement.NESTED
        ) as st:
            _step(st)
        step = _emitted(sink)[-1]

    run = _emitted(sink)[-1]
    assert step.parent_span_id == run.context.span_id
    assert step.context.trace_id == run.context.trace_id
    assert (step.correlation.strategy, step.correlation.confidence) == (
        ParentSource.UNIT_ACTIVE,
        1.0,
    )
    assert _edge_of(step)[2] == ()


def test_a_nested_site_inside_a_host_span_takes_the_host_span():
    """A wardex span the HOST opened — a `@workflow` decorator — is a real
    parent read from the live scope, so a nested site is not an orphan there.
    """
    ctx, sink = context()
    parent = _a_context()

    with activate_span(parent):
        with ctx.enter(
            UnitKind.STEP, intent=SpanIntent.EXECUTE_STEP, placement=Placement.NESTED
        ) as st:
            _step(st)

    span = _emitted(sink)[-1]
    assert span.parent_span_id == parent.span_id
    assert _edge(sink) == (ParentSource.CONTEXTVAR, 1.0, ())


@pytest.mark.parametrize("placement", [Placement.ROOT, Placement.NESTED])
def test_placement_changes_nothing_when_a_real_parent_is_installed(placement):
    """The two placements differ ONLY where a parent is absent. Anywhere a scope
    was genuinely read, declaring the site wrong costs nothing — which is what
    keeps the declaration about the site's shape rather than about its luck.
    """
    ctx, sink = context()

    with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT) as r:
        _agent(r)
        with ctx.enter(UnitKind.STEP, intent=SpanIntent.EXECUTE_STEP, placement=placement) as st:
            _step(st)
        assert _edge(sink) == (ParentSource.UNIT_ACTIVE, 1.0, ())


def test_a_nested_site_under_a_dead_pin_is_an_orphan_that_names_the_conflict():
    """The leftover fork a closed pin cannot take down is not a parent.

    `close()` runs on another task and a ContextVar cannot be reset from one, so
    the dead unit's own span context is still standing in this task's scope. An
    edge built from it reads `contextvar` / 1.0 / no marker into a finished run,
    which is the failure the pin restriction exists to prevent arriving by the
    other carrier. NESTED turns it into 0.0 as well, because a site that
    declared it is always inside something has lost what it was promised.
    """
    ctx, sink = context()
    units = ctx._units

    dead = units.open(
        UnitKind.SESSION,
        UnitKey("test.session", "dead"),
        ambient=EMPTY_AMBIENT,
        intent=SpanIntent.INVOKE_AGENT,
        subject="a",
    )
    dead.draft.set_agent(AgentAttributes(name="a", agent_type=AgentType.PRIMARY))
    units.pin_driver(dead, owner_task=threading.current_thread())
    units.close(dead)

    with ctx.enter(
        UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.NESTED, subject="t"
    ) as call:
        _tool(call)

    source, confidence, markers = _edge(sink)
    assert (source, confidence) == (ParentSource.UNRESOLVED, 0.0)
    assert set(markers) == {Limitation.PARENT_UNRESOLVED, Limitation.CORRELATION_CONFLICT}


def test_a_real_host_span_over_a_dead_pin_is_still_the_parent():
    """The refusal is the leftover FORK, not the task.

    A stale pin says this task descends from a dead driver; it does not say the
    scope still holds the corpse's span. A host that opened its own span inside
    that task put a real parent on top, and orphaning THAT to escape a ghost no
    longer in front of us is a wrong tree of a different shape — the one
    `UnitRegistry._poisoned` narrows to avoid and
    `test_a_real_span_over_a_stale_pin_is_still_a_parent` promises against.

    Asking the registry the same question `open()` will ask is what keeps the
    two from disagreeing: a predicate about the TASK once stood in here for one
    about the SPAN, and this row orphaned live work at 0.0 while dropping the
    only record that a pin had died.
    """
    ctx, sink = context()
    units = ctx._units

    dead = units.open(
        UnitKind.SESSION,
        UnitKey("test.session", "dead"),
        ambient=EMPTY_AMBIENT,
        intent=SpanIntent.INVOKE_AGENT,
        subject="a",
    )
    dead.draft.set_agent(AgentAttributes(name="a", agent_type=AgentType.PRIMARY))
    units.pin_driver(dead, owner_task=threading.current_thread())
    units.close(dead)

    host = _a_context()
    with activate_span(host):
        with ctx.enter(
            UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.NESTED, subject="t"
        ) as call:
            _tool(call)

    span = _emitted(sink)[-1]
    assert span.parent_span_id == host.span_id
    assert span.context.trace_id == host.trace_id
    assert _edge(sink) == (ParentSource.CONTEXTVAR, 1.0, ())


# ==========================================================================
# rejoin — the one method where an identifier shapes the tree
# ==========================================================================


def test_rejoin_on_a_hit_is_clamped_to_the_alias_tier():
    """An identifier is the framework's word, not a scope wardex read. There is
    no argument that raises it above 0.9, because there is nothing an adapter
    could learn that would justify raising it.
    """
    ctx, sink = context()
    key = UnitKey("test.agent", "a1")

    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, selector=key
    ) as r:
        _agent(r)
        with ctx.rejoin(
            key, UnitKind.STEP, intent=SpanIntent.EXECUTE_STEP, placement=Placement.NESTED
        ) as st:
            _step(st)
        step = _emitted(sink)[-1]

    run = _emitted(sink)[-1]
    assert step.parent_span_id == run.context.span_id
    assert step.correlation.strategy is ParentSource.UNIT_ALIAS
    assert step.correlation.confidence == 0.9
    assert step.correlation.request_id == "a1"


def test_rejoin_on_a_miss_falls_to_the_placement_table_rather_than_guessing():
    """The failure mode this replaces re-parented a miss to "some plausible
    root" in silence. A selector that resolves to nothing is a fact about the
    data, and the span says it.
    """
    ctx, sink = context()

    with ctx.rejoin(
        UnitKey("test.agent", "gone"),
        UnitKind.STEP,
        intent=SpanIntent.EXECUTE_STEP,
        placement=Placement.NESTED,
    ) as st:
        _step(st)

    source, confidence, markers = _edge(sink)
    assert source is ParentSource.UNRESOLVED
    assert confidence == 0.0
    assert Limitation.PARENT_UNRESOLVED in markers


# ==========================================================================
# The shape — what an adapter has no word for
# ==========================================================================


@pytest.mark.parametrize(
    "forbidden", ["context", "parentage", "parent", "unit", "child", "activate", "bind"]
)
@pytest.mark.parametrize("handle", [Scope, RunHandle, Attachment])
def test_no_handle_exposes_anything_that_is_a_parent(handle, forbidden):
    """The table above holds for the adapters that exist. THIS is what makes it
    hold for the ones nobody has written: a value that is a parent can be
    carried to another carrier and installed there, and an adapter doing that
    rebuilds an identifier-shaped tree while every edge still reads 1.0. There
    is no word for it here.
    """
    assert not hasattr(handle, forbidden), (
        f"{handle.__name__}.{forbidden} is a parent-valued member"
    )


@pytest.mark.parametrize("forbidden", ["child_draft", "link", "claim", "pin", "__enter__"])
def test_an_id_selected_attachment_can_describe_and_close_and_nothing_else(forbidden):
    """`attach()` is keyed on an identifier, so what it returns may reach a
    span's attributes and its ending — never its children.
    """
    assert not hasattr(Attachment, forbidden)


def test_the_context_offers_no_way_to_supply_or_read_a_parent():
    """No `ambient=`, no `evidence=`, no `parent=` on any causal method, and no
    client — `capture_span` is unreachable, so the capture gate has one home.
    """
    import inspect

    for name in ("enter", "open_run", "rejoin", "attach"):
        params = set(inspect.signature(getattr(AdapterContext, name)).parameters)
        assert not params & {"ambient", "evidence", "parent", "parent_unit", "confidence"}, name
    assert not hasattr(AdapterContext, "client")


# ==========================================================================
# Housekeeping the contract promises
# ==========================================================================


def test_a_scope_is_removed_when_its_with_block_ends():
    ctx, sink = context()

    with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT) as s:
        _agent(s)
        assert ctx._units.current() is not None

    assert ctx._units.current() is None


def test_a_scope_that_raises_closes_its_span_as_an_error_and_re_raises():
    sink = RecordingSink()
    ctx, sink = context(sink=sink)

    with pytest.raises(ValueError):
        with ctx.enter(
            UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT
        ) as s:
            _agent(s)
            raise ValueError("host code failed")

    assert ctx._units.current() is None
    span = sink.drafts[-1].finish()
    assert span.status is StatusCode.ERROR
    assert span.error_type == "ValueError"


def test_every_unit_a_context_opens_is_stamped_with_its_adapter():
    """Which is what lets one registry serve two adapters without either being
    handed the other's run as a sole-live guess.
    """
    ctx, sink = context("anthropic")

    with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT) as s:
        _agent(s)
        assert ctx._units.sole_live(UnitKind.SESSION, owner="anthropic") is not None
        assert ctx._units.sole_live(UnitKind.SESSION, owner="langgraph") is None


def test_a_link_to_a_selector_that_names_nothing_is_counted_not_faked():
    """A checkpoint resume names a run in another PROCESS. Nothing persists an
    identity that outlives the registry, so the link cannot be built — and a
    silent no-op would be the hole this SDK spends its markers on.
    """
    ctx, sink = context("lg")

    with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT) as s:
        _agent(s)
        s.link(LinkReason.RESUMED_FROM, UnitKey("lg.thread_id", "not-in-this-process"))

    assert _emitted(sink)[-1].links == ()
    assert counters.get("adapters.lg.link_target_unresolved") == 1


def test_a_slot_survives_an_address_being_reused():
    """`id(obj)` keying hands a new object the dead one's bookkeeping, because
    CPython reuses addresses. Identity keying cannot.
    """

    class Transport:
        pass

    ctx, sink = context()
    first = Transport()
    ctx.slot(first)["session"] = "one"
    assert ctx.slot(first) == {"session": "one"}

    second = Transport()
    assert ctx.slot(second) == {}


def test_confirm_active_counts_per_site_not_per_install():
    """A framework can move ONE entry point and leave the rest working, which
    produces a tree wrong only in the shape that entry governed — measured, a
    run entry left unpatched shattered a graph while an install-level check
    still reported success.
    """
    ctx, sink = context("lg")
    ctx.confirm_active("pregel.astream")

    assert counters.get("adapters.lg.active.pregel.astream") == 1
    assert counters.get("adapters.lg.active.pregel.stream") == 0


def test_install_outcome_separates_absent_from_unrecognized():
    assert InstallOutcome.DECLINED is not InstallOutcome.UNSUPPORTED
    assert {o.value for o in InstallOutcome} == {"installed", "declined", "unsupported"}


def test_the_executor_outranks_the_callback():
    """`PreToolUse`-style callbacks fire BEFORE the body they describe, so
    first-come hands every span to the observer that did not run the code.
    """
    assert Observer.EXECUTOR.value > Observer.CALLBACK.value

    ctx, sink = context()
    key = UnitKey("test.tool", "Bash")

    with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT) as s:
        _agent(s)
        assert s.claim(key, observer=Observer.CALLBACK) is True
        assert s.outranked(key, observer=Observer.CALLBACK) is False
        assert s.claim(key, observer=Observer.EXECUTOR) is True
        assert s.outranked(key, observer=Observer.CALLBACK) is True
        assert s.outranked(key, observer=Observer.EXECUTOR) is False


def test_a_run_handle_pins_onto_the_carrier_that_asked():
    """The one deferred installer, and the registry refuses it for a task other
    than the caller's — which is what keeps it from being aimed anywhere.
    """
    ctx, sink = context()
    run = ctx.open_run(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT)
    _agent(run)

    assert run.pin(driver=threading.current_thread()) is True
    assert ctx._units.current() is not None

    assert run.pin(driver=object()) is False
    run.close()


# ==========================================================================
# A wardex bug costs a span. It never costs the host.
# ==========================================================================
#
# Every test below installs ONE fault in wardex's own machinery and asks three
# questions of the result: did the host's code still run, did the host get its
# own value or its own exception back, and is what wardex shipped honest about
# what it lost. The `with` body stands for the host's call, which is where a
# framework adapter has to put it — a tool handler, a graph node, an LLM
# request — so a guard that swallowed an exception there would turn a failing
# call into a successful one on the wire.


def test_a_bug_deciding_the_parent_edge_still_runs_the_hosts_code_and_returns_its_value():
    """The parentage decision is wardex's most opinionated code and its most
    likely to break. `_evidence` asks the registry two questions, and both are
    reached before the host's body has run — so an unguarded one means a wardex
    bug about tree SHAPE deletes a tool call.
    """
    for fault in ("becomes_trace_root", "current"):
        ctx, sink = broken(fault)
        ran = []

        with ctx.enter(
            UnitKind.STEP, intent=SpanIntent.EXECUTE_STEP, placement=Placement.NESTED
        ) as scope:
            ran.append(scope)
            value = {"the host's own": fault}

        assert len(ran) == 1, f"{fault}: the body ran {len(ran)} times"
        assert isinstance(ran[0], Scope), f"{fault}: the body was handed {ran[0]!r}"
        assert ran[0].degraded
        assert value == {"the host's own": fault}
        assert ctx.tripped


@pytest.mark.parametrize("exc", [ValueError("host"), KeyboardInterrupt(), Exception("host")])
def test_the_hosts_own_exception_reaches_its_caller_as_the_same_object(exc):
    """IDENTITY, not type. A wrapper that rebuilt the exception would preserve
    the type, the message and the status on the span, and would still break a
    host whose `except` clause matches on the instance — a retry loop holding
    the object it raised, an `ExceptionGroup` member, a cause chain.

    `KeyboardInterrupt` is here because wardex must not become the first library
    in the process that can eat a real Ctrl-C.
    """
    ctx, sink = context()

    with pytest.raises(type(exc)) as caught:
        with ctx.enter(
            UnitKind.SESSION,
            intent=SpanIntent.INVOKE_AGENT,
            placement=Placement.ROOT,
            describe=_agent,
        ):
            raise exc

    assert caught.value is exc
    span = _emitted(sink)[-1]
    assert span.status is StatusCode.ERROR
    assert span.error_type == type(exc).__name__


def test_a_wardex_bug_in_the_close_does_not_supersede_the_hosts_own_exception():
    """The most dangerous shape there is: a failure in wardex's TEARDOWN
    replacing the failure the host was in the middle of reporting. The host's
    `ValueError` is the thing its operator needs; wardex's `RuntimeError`
    arriving instead makes a wardex bug look like a host bug, in the host's own
    logs, with wardex nowhere in the traceback's first frames.
    """
    ctx, sink = broken("close")
    mine = ValueError("the host's own failure")

    with pytest.raises(ValueError) as caught:
        with ctx.enter(
            UnitKind.SESSION,
            intent=SpanIntent.INVOKE_AGENT,
            placement=Placement.ROOT,
            describe=_agent,
        ):
            raise mine

    assert caught.value is mine


def test_a_wardex_bug_in_the_close_costs_the_span_and_not_the_hosts_return_value():
    ctx, sink = broken("close")

    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ) as scope:
        assert not scope.degraded  # the OPEN was fine; only the close will fail
        value = "the host's own"

    assert value == "the host's own"
    assert sink.drafts == []
    assert ctx.tripped


def test_a_close_that_fails_leaves_its_subtree_for_the_teardown_to_recover():
    """No salvage, and the recovery is what buys that.

    A close that fails could be answered by hand-detaching the unit and pushing
    its own span to the sink. Measured on this exact shape, that ships 1 of 7
    spans — with `status=OK`, which it has no basis for — and destroys the
    reachability that lets the teardown find the other 6. Leaving the subtree
    where it is costs the same 7 spans NOW and recovers all 7 later.
    """
    faulty = [True]

    class Broken(UnitRegistry):
        __slots__ = ()

        def _close_locked(self, unit, **kw):
            if faulty[0]:
                raise RuntimeError("this subtree is torn")
            return super()._close_locked(unit, **kw)

    sink = RecordingSink()
    ctx = AdapterContext("probe", units=Broken(sink=sink), limits={})

    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ):
        for _ in range(6):
            ctx.open_run(
                UnitKind.CALL,
                intent=SpanIntent.EXECUTE_TOOL,
                placement=Placement.NESTED,
                describe=_tool,
            )

    assert sink.drafts == [], "a partial subtree shipped — that is the salvage this rejects"

    faulty[0] = False
    ctx.close_all(marker=Limitation.ADAPTER_UNINSTALLED)
    assert len(sink.drafts) == 7


@pytest.mark.parametrize("when", ["before the required block", "after it"])
def test_a_description_that_fails_never_ships_a_span_that_reads_healthy(when):
    """`describe=` runs inside the OPEN's guard, and that is the whole argument.

    Described in the `with` body under its own guard instead, the same fault
    ships a span with `status=OK`, full io, a real duration and an arbitrary
    suffix of its markers missing — which is worse than shipping nothing,
    because nothing downstream can tell it from a complete observation.

    Two arms, because the vocabulary decides the outcome. A description that
    died before the intent's required block leaves a draft `finish()` refuses,
    so the span is dropped at the emit funnel. One that died after it ships,
    `UNSET` and marked. Neither reads healthy.
    """
    ctx, sink = context()

    def describe(scope):
        if when == "after it":
            _tool(scope)
        raise AttributeError("the framework moved this attribute")

    ran = []
    with ctx.enter(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        describe=describe,
    ) as scope:
        ran.append(scope)
        value = "the host's own"

    assert len(ran) == 1
    assert value == "the host's own"
    assert ctx.tripped

    if when == "before the required block":
        from wardex_sdk.assembly import VocabularyError

        with pytest.raises(VocabularyError):
            sink.drafts[-1].finish()
        return

    span = sink.drafts[-1].finish()
    assert span.status is StatusCode.UNSET
    assert Limitation.INSTRUMENTATION_DEGRADED in span.capture_integrity.limitations

    # And at its TRUE duration: the unit is closed the instant the description
    # failed, not left for the teardown to force-close at an inflated end.
    ctx.close_all(marker=Limitation.ADAPTER_UNINSTALLED)
    assert len(sink.drafts) == 1


_DEGRADED_VERB_ARGS = {
    "note": ((Limitation.PARENT_UNRESOLVED,), {}),
    "link": ((LinkReason.HANDOFF_FROM, UnitKey("k", "v")), {}),
    "claim": ((UnitKey("k", "v"),), {"observer": Observer.EXECUTOR}),
    "claim_run": ((UnitKey("k", "v"),), {"observer": Observer.EXECUTOR}),
    "outranked": ((UnitKey("k", "v"),), {"observer": Observer.EXECUTOR}),
    "child_draft": ((SpanIntent.EXECUTE_TOOL,), {}),
    "close_child": ((NULL_DRAFT,), {}),
    "record_input": ((b"in",), {}),
    "record_output": ((b"out",), {}),
    "pin": ((), {"driver": threading.current_thread()}),
    "close": ((), {}),
}
_DEGRADED_READERS = ("draft", "accepted", "degraded")


def test_every_verb_on_a_degraded_scope_is_answerable_and_none_of_them_raises(monkeypatch):
    """Enumerated BY REFLECTION, so a verb added later cannot quietly go missing.

    The fault is installed on `SpanDraft.__init__` itself — the construction
    whose failure produced the degraded scope in the first place — so a null
    draft that lazily built a real one would recur here rather than in a host's
    process. A hand-written list of verbs would have been written from the same
    memory that produced a partial null draft.
    """
    from wardex_sdk.assembly import _builder

    members = {n for n in dir(Scope) if not n.startswith("_")}
    members |= {n for n in dir(RunHandle) if not n.startswith("_")}
    uncovered = members - set(_DEGRADED_VERB_ARGS) - set(_DEGRADED_READERS)
    assert uncovered == set(), (
        f"{sorted(uncovered)} is reachable on a degraded handle and untested here.\n"
        "Add it to _DEGRADED_VERB_ARGS, and give it a total answer in _context.py."
    )

    def blow(self, *a, **k):
        raise RuntimeError("a draft cannot be built")

    monkeypatch.setattr(_builder.SpanDraft, "__init__", blow)

    ctx, sink = context()
    handle = ctx.open_run(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT
    )
    assert handle is not None and handle.degraded

    ran = []
    with ctx.enter(
        UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.NESTED
    ) as scope:
        ran.append(scope)
        for target in (scope, handle):
            for name in _DEGRADED_READERS:
                getattr(target, name)
            for name, (args, kwargs) in _DEGRADED_VERB_ARGS.items():
                verb = getattr(target, name, None)
                if verb is not None:
                    verb(*args, **kwargs)
            assert target.draft is NULL_DRAFT
            # Every verb of the draft AND of what its integrity returns.
            for holder in (target.draft, target.draft.integrity):
                for name in dir(holder):
                    if name.startswith("_"):
                        continue
                    member = getattr(holder, name)
                    if not callable(member):
                        continue
        value = "the host's own"

    assert len(ran) == 1
    assert value == "the host's own"
    assert sink.drafts == []


def test_a_degraded_scope_writes_nothing_onto_the_span_that_encloses_it():
    """The tempting fix is to reuse the enclosing unit so "at least something is
    recorded". What that records is the CHILD's tool block, io and markers on
    the PARENT's span — a fabricated observation with nothing saying so. The
    only thing a degraded scope may add to its enclosing span is the marker that
    says wardex failed under it.
    """
    faulty = [False]

    class Broken(UnitRegistry):
        __slots__ = ()

        def open(self, *a, **k):
            if faulty[0]:
                raise RuntimeError("wardex is broken at open")
            return super().open(*a, **k)

    sink = RecordingSink()
    ctx = AdapterContext("probe", units=Broken(sink=sink), limits={})

    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ) as root:
        root.record_input(b"the root's own input")
        faulty[0] = True
        with ctx.enter(
            UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.NESTED
        ) as child:
            child.draft.set_tool(ToolAttributes(name="the child's tool"))
            child.record_input(b"the child's own input")
            child.note(Limitation.TOOL_NAME_COLLISION)
        faulty[0] = False

    span = _emitted(sink)[-1]
    assert span.tool is None, "the child's tool block landed on its parent"
    assert span.input_data == b"the root's own input"
    assert set(span.capture_integrity.limitations) == {Limitation.INSTRUMENTATION_DEGRADED}


@pytest.mark.parametrize(
    "fault", ["open", "close", "_close_locked", "becomes_trace_root", "current"]
)
def test_the_carrier_is_empty_after_a_scope_that_failed_at_any_step(fault):
    """The activation must be exited on EVERY path, including the ones where the
    close then fails. Deferring it until the close succeeds leaves a dead unit
    ambient, so the next sibling call in this task is parented to a corpse — and
    a corpse is a perfectly good-looking parent on the wire.
    """
    ctx, sink = broken(fault)

    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ):
        pass

    assert _ambient_unit.get() is None
    # The BASE implementation, called explicitly: for the `current` fault the
    # subclass's own method is the thing that raises, and the question here is
    # about the carrier rather than about that method.
    assert UnitRegistry.current(ctx._units) is None


def test_a_span_whose_activation_failed_does_not_read_like_a_healthy_one(monkeypatch):
    """The activation is the only fault whose span still SHIPS: the unit is live
    and its own edge is correct, and what is lost is that work INSIDE it finds
    the enclosing unit instead. That loss has to be legible, and the trap is
    that `if activation is None:` — the obvious test — is dead code, because
    `unit.activate()` binds the name before `__enter__` can raise. A span that
    reads byte-identical to a healthy one is the failure this asserts against.
    """
    from wardex_sdk.assembly import _units

    healthy_ctx, healthy_sink = context()
    with healthy_ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ):
        pass
    healthy = _emitted(healthy_sink)[-1]

    def blow(self, *a, **k):
        raise RuntimeError("the carrier is broken")

    monkeypatch.setattr(_units._Carrier, "__init__", blow)

    ctx, sink = context()
    ran = []
    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ) as scope:
        ran.append(scope)
        assert not scope.degraded

    assert len(ran) == 1
    span = _emitted(sink)[-1]
    assert span.status is StatusCode.OK
    assert span.capture_integrity is not None
    assert set(span.capture_integrity.limitations) != set(
        healthy.capture_integrity.limitations if healthy.capture_integrity else ()
    )
    assert Limitation.INSTRUMENTATION_DEGRADED in span.capture_integrity.limitations


def test_a_lookup_that_blows_up_lowers_the_edge_it_would_have_claimed():
    """A failed lookup may only ever LOWER what wardex claims about the edge.

    Read as an ordinary miss it would be an UPGRADE: the miss path falls through
    to the live scope at 1.0, which is ABOVE the 0.9 a real alias hit earns. So
    a bug in the lookup would make the edge look more certain than the
    identifier it was told to honour, which is I4 exactly backwards.
    """
    key = UnitKey("test.run", "r1")

    # HIT — the identifier resolved, and an exact match is worth 0.9.
    ctx, sink = context()
    with ctx.enter(
        UnitKind.SESSION,
        intent=SpanIntent.INVOKE_AGENT,
        placement=Placement.ROOT,
        selector=key,
        describe=_agent,
    ):
        with ctx.rejoin(
            key,
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            describe=_step,
        ):
            pass
        assert _edge(sink)[:2] == (ParentSource.UNIT_ALIAS, 0.9)

    # MISS — the documented fall-through, unchanged, and NOT a degradation.
    ctx, sink = context()
    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ):
        with ctx.rejoin(
            UnitKey("test.run", "nobody"),
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            describe=_step,
        ):
            pass
        source, confidence, markers = _edge(sink)
    assert (source, confidence) == (ParentSource.UNIT_ACTIVE, 1.0)
    assert markers == ()
    assert not ctx.tripped

    # BROKEN — same strategy, half the confidence, and it says wardex failed.
    ctx, sink = broken("find")
    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ):
        with ctx.rejoin(
            key,
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            describe=_step,
        ):
            pass
        source, confidence, markers = _edge(sink)
    assert (source, confidence) == (ParentSource.UNIT_ACTIVE, 0.5)
    assert confidence < 0.9, "a wardex bug outranked a real alias hit"
    assert Limitation.INSTRUMENTATION_DEGRADED in markers


def test_a_thousand_trips_of_one_site_leave_a_bounded_record_on_the_wire(capsys):
    """One line, one marker, zero extras — however many times the site trips.

    Recording the site name as a span attribute is the shape this rejects:
    `set_extra` appends with no dedup and no cap, so a site failing in a loop
    puts one entry per trip on the enclosing span, while `add_limitation` puts
    one marker there however often it is called. The site name lives on stderr,
    where the person who needs it is already looking.
    """
    healthy_ctx, healthy_sink = context()
    with healthy_ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ):
        pass
    healthy = _emitted(healthy_sink)[-1]

    faulty = [False]

    class Broken(UnitRegistry):
        __slots__ = ()

        def open(self, *a, **k):
            if faulty[0]:
                raise RuntimeError("wardex is broken at open")
            return super().open(*a, **k)

    sink = RecordingSink()
    ctx = AdapterContext("probe", units=Broken(sink=sink), limits={})

    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ):
        capsys.readouterr()
        faulty[0] = True
        for _ in range(1000):
            with ctx.enter(
                UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.NESTED
            ):
                pass
        faulty[0] = False

    lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(lines) == 1, f"1000 trips wrote {len(lines)} lines"

    span = _emitted(sink)[-1]
    assert list(span.capture_integrity.limitations) == [Limitation.INSTRUMENTATION_DEGRADED]
    # Compared against the SAME span built healthily: the assertion is that the
    # degraded path added nothing to `extra`, not that `extra` happens to be
    # empty — this span already carries its operation name there.
    assert span.extra == healthy.extra
    assert counters.get("adapters.probe.enter.execute_tool") == 1000


def test_a_degraded_run_is_told_apart_from_wardex_never_having_been_installed(capsys):
    """The question that decides whether any of this is worth anything.

    An operator whose dashboard is empty has to be able to tell "wardex is
    broken here" from "wardex was never installed". Recorded only in `counters`
    the two are byte-identical in a production process — nothing reads a
    snapshot, `counters` is not exported, and `debug` is off by default. The
    stderr line is the difference, and it names the CONSEQUENCE rather than a
    site label, because the reader is asking why their traces stopped.
    """
    ctx, sink = context()
    capsys.readouterr()
    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ):
        pass
    assert len(sink.drafts) == 1
    assert capsys.readouterr().err == ""

    ctx, sink = broken("open")
    capsys.readouterr()
    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ):
        pass
    degraded = capsys.readouterr().err
    assert sink.drafts == []
    assert len([line for line in degraded.splitlines() if line.strip()]) == 1
    assert "probe adapter" in degraded
    assert "NO agent span" in degraded, f"the line does not name the consequence: {degraded!r}"
    assert "capture_mode=AGENT" in degraded

    # And the third arm: no adapter at all. Zero spans, and nothing said,
    # because nothing went wrong.
    capsys.readouterr()
    assert capsys.readouterr().err == ""


def test_a_degraded_scope_refuses_the_claim_it_cannot_honour():
    """A True claim tells the rival observer it was outranked by a unit that
    does not exist — so the ONE observation of this event that could still have
    survived is suppressed too. Refusing costs a weaker span; claiming costs
    every span.
    """
    ctx, sink = broken("open")

    with ctx.enter(
        UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.NESTED
    ) as scope:
        assert scope.degraded
        assert scope.claim(UnitKey("k", "v"), observer=Observer.EXECUTOR) is False
        assert scope.outranked(UnitKey("k", "v"), observer=Observer.CALLBACK) is False


def test_describe_is_handed_the_same_scope_the_body_is_handed():
    """The new keyword must not become a back door for a privileged value.

    `describe` runs inside wardex's own guard, where a `Unit` — which exposes
    `child()`, `activate()` and `context` — is right there to hand over. Doing
    so would put a parent-valued object in every adapter author's hands through
    a keyword nobody thinks of as part of the causal surface.
    """
    ctx, sink = context()
    seen = []

    with ctx.enter(
        UnitKind.SESSION,
        intent=SpanIntent.INVOKE_AGENT,
        placement=Placement.ROOT,
        describe=lambda s: (seen.append(s), _agent(s)),
    ) as scope:
        assert seen[0] is scope

    for forbidden in ("context", "parentage", "parent", "unit", "child", "activate", "bind"):
        assert not hasattr(seen[0], forbidden)


def test_a_teardown_that_fails_never_reaches_the_hosts_uninstall():
    """`close_all` runs from `uninstall()`, which runs on the host's `atexit`.
    A raise there lands in the host's shutdown path, after its own code is done
    and where nobody is catching anything.
    """
    ctx, sink = broken("close_all")

    ctx.close_all(marker=Limitation.ADAPTER_UNINSTALLED)  # must not raise

    assert ctx.tripped


def test_a_lost_subtree_is_recorded_on_a_span_that_still_ships():
    """The activation is exited BEFORE the close, never after a successful one.

    On CPython the carrier comes down by refcounting either way, so the cost of
    the wrong order is not a leak — it is that the degradation is recorded in
    the wrong place. `_degrade` asks the registry which unit should carry the
    marker, and with the dying unit still ambient the answer is the unit whose
    span was just lost. The record then goes down with the thing it describes,
    and the span that DOES ship says nothing about the subtree missing under it.
    """
    faulty = [False]

    class Broken(UnitRegistry):
        __slots__ = ()

        def close(self, unit, **kw):
            if faulty[0]:
                raise RuntimeError("wardex is broken at close")
            return super().close(unit, **kw)

    sink = RecordingSink()
    ctx = AdapterContext("probe", units=Broken(sink=sink), limits={})

    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ):
        faulty[0] = True
        with ctx.enter(
            UnitKind.CALL,
            intent=SpanIntent.EXECUTE_TOOL,
            placement=Placement.NESTED,
            describe=_tool,
        ):
            pass
        faulty[0] = False

    span = _emitted(sink)[-1]
    assert Limitation.INSTRUMENTATION_DEGRADED in span.capture_integrity.limitations, (
        "the run's own span does not say a subtree was lost under it"
    )


# ==========================================================================
# The one guess a site may declare
# ==========================================================================


def _session(ctx, value="s1"):
    """A live SESSION owned by this adapter — what `sole_live(owner=)` looks for."""
    from wardex_sdk._types import AgentAttributes as _AA

    unit = ctx._units.open(
        UnitKind.SESSION,
        UnitKey("test.session", value),
        ambient=EMPTY_AMBIENT,
        intent=SpanIntent.INVOKE_AGENT,
        owner=ctx.name,
    )
    unit.draft.set_agent(_AA(name="a", agent_type=AgentType.PRIMARY))
    return unit


def test_an_undeclared_fallback_orphans_instead_of_guessing():
    """The default, and it has to be the honest one. A site that says nothing
    gets `unresolved`/0.0 even with a run standing right there — a heuristic
    that turns itself on is one nobody can find later.
    """
    ctx, sink = context()
    _session(ctx)

    with ctx.enter(
        UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.NESTED, describe=_tool
    ):
        pass

    source, confidence, markers = _edge(sink)
    assert (source, confidence) == (ParentSource.UNRESOLVED, 0.0)
    assert Limitation.PARENT_UNRESOLVED in markers


def test_a_declared_fallback_takes_the_one_live_run_and_says_it_guessed():
    """Half the confidence and a marker naming the interpretation. The
    alternative is a tool call that becomes its own trace root — one run
    reported as several, which is the shape nothing downstream can detect.
    """
    ctx, sink = context()
    run = _session(ctx)

    with ctx.enter(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        fallback=Fallback.SOLE_LIVE_RUN,
        describe=_tool,
    ):
        pass

    span = _emitted(sink)[-1]
    assert span.parent_span_id == run.draft.context.span_id
    assert (span.correlation.strategy, span.correlation.confidence) == (
        ParentSource.UNIT_SOLE,
        0.5,
    )
    assert Limitation.UNIT_INFERRED_SOLE in span.capture_integrity.limitations


def test_a_second_live_run_makes_the_fallback_decline_rather_than_pick_one():
    """`sole_live` means SOLE. Two runs and it answers nothing, because picking
    one would be a coin flip presented on the wire as a 0.5 edge — and 0.5 says
    "interpreted", not "one of two".
    """
    ctx, sink = context()
    _session(ctx, "s1")
    _session(ctx, "s2")

    with ctx.enter(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        fallback=Fallback.SOLE_LIVE_RUN,
        describe=_tool,
    ):
        pass

    assert _edge(sink)[:2] == (ParentSource.UNRESOLVED, 0.0)


def test_a_run_another_adapter_owns_is_not_a_candidate():
    """ "Exactly one session is live" is a question about the PROCESS unless it
    is scoped. Unscoped, an Anthropic tool call with only a LangGraph run beside
    it would be handed that run and stamped 0.5 — a guess across a boundary
    nobody stated.
    """
    ctx, sink = context()
    other = ctx._units.open(
        UnitKind.SESSION,
        UnitKey("test.session", "theirs"),
        ambient=EMPTY_AMBIENT,
        intent=SpanIntent.INVOKE_AGENT,
        owner="some_other_adapter",
    )
    assert other.owner == "some_other_adapter"

    with ctx.enter(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        fallback=Fallback.SOLE_LIVE_RUN,
        describe=_tool,
    ):
        pass

    assert _edge(sink)[:2] == (ParentSource.UNRESOLVED, 0.0)


@pytest.mark.parametrize("above", ["a live scope", "a host span"])
def test_a_fallback_never_displaces_something_wardex_actually_read(above):
    """Structural, not a rule: the fallback is reachable only where the site
    would otherwise orphan, so a real read always wins and the declaration
    cannot quietly downgrade an edge from 1.0 to 0.5.
    """
    ctx, sink = context()
    _session(ctx, "the sole live one")

    def nested():
        with ctx.enter(
            UnitKind.CALL,
            intent=SpanIntent.EXECUTE_TOOL,
            placement=Placement.NESTED,
            fallback=Fallback.SOLE_LIVE_RUN,
            describe=_tool,
        ):
            pass

    if above == "a live scope":
        with ctx.enter(
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            describe=_step,
        ):
            nested()
        expected = ParentSource.UNIT_ACTIVE
    else:
        with activate_span(_a_context()):
            nested()
        expected = ParentSource.CONTEXTVAR

    tool = next(d.finish() for d in sink.drafts if d.name.startswith("execute_tool"))
    assert (tool.correlation.strategy, tool.correlation.confidence) == (expected, 1.0)
    assert Limitation.UNIT_INFERRED_SOLE not in _edge_of(tool)[2]


def test_a_guess_a_dead_pin_caused_says_which_kind_of_guess_it_was():
    """The two reasons a fallback fires are not the same fact.

    "Nothing was pinned" and "what was pinned had died" produce byte-identical
    spans otherwise — and the second is a call filed inside a run it has nothing
    to do with. `open()` marks the poisoned ambient itself, but taking a parent
    is exactly what stops it from seeing one, so the fallback carries it.
    """
    ctx, sink = context()
    dead = _session(ctx, "the one that died")
    ctx._units.pin_driver(dead, owner_task=threading.current_thread())
    ctx._units.close(dead)
    _session(ctx, "an unrelated run")

    with ctx.enter(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        fallback=Fallback.SOLE_LIVE_RUN,
        describe=_tool,
    ):
        pass

    source, confidence, markers = _edge(sink)
    assert (source, confidence) == (ParentSource.UNIT_SOLE, 0.5)
    assert Limitation.CORRELATION_CONFLICT in markers, (
        "a guess caused by a dead pin reads the same as a guess caused by nothing"
    )


def test_a_claim_is_taken_on_the_run_and_not_on_the_scope_that_took_it():
    """Two observers of one event have to claim on the SAME unit or `claim()`
    arbitrates nothing — and they never stand in the same place. A framework's
    own callback sees the whole run; an in-process handler's scope is the CALL
    it is executing, which for a nested tool is not even a direct child.
    """
    ctx, sink = context()
    run = _session(ctx)
    selector = UnitKey("tool.call", "greet")

    with ctx.enter(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        fallback=Fallback.SOLE_LIVE_RUN,
    ) as outer:
        with ctx.enter(
            UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.NESTED
        ) as inner:
            # From a call nested TWO levels under the run, and it still lands
            # on the run — a plain `claim()` would land on `inner`.
            assert inner.claim_run(selector, observer=Observer.EXECUTOR) is True
            assert inner.claim(selector, observer=Observer.EXECUTOR) is True
            _tool(inner)
        _tool(outer)

    # The lower-ranked observer, arriving at the run where the hook stands.
    assert run.claim(selector, rank=Observer.CALLBACK.value) is False
    assert run.owner_rank(selector) == Observer.EXECUTOR.value
