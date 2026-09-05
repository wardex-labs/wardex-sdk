"""`_adapters/_context.py` — the surface a framework adapter is handed.

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

from wardex_sdk._adapters._context import (
    AdapterContext,
    Attachment,
    Fallback,
    InstallOutcome,
    Observer,
    Placement,
    RunHandle,
    Scope,
)
from wardex_sdk._assembly import (
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
from wardex_sdk._assembly._builder import NULL_DRAFT
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._assembly._units import _ambient_unit
from wardex_sdk._assembly._vocab import VocabularyError
from wardex_sdk._enums import AgentType, StatusCode
from wardex_sdk._hub import reset_for_test
from wardex_sdk._types import AgentAttributes, ConversationContext, ToolAttributes
from wardex_sdk.context._contextvar import activate_span


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


def test_a_link_to_a_closed_but_remembered_selector_is_built_from_memory():
    """The same-process half of a resume: the predecessor FINISHED, its unit
    closed and every live lookup for it is gone — and the link is still built,
    from the registry's bounded closed-unit memory rather than from a unit
    kept alive. A new trace linked to the old one's span, never a parent edge
    across the two.
    """
    ctx, sink = context("lg")
    key = UnitKey("lg.thread_id", "t-1")

    with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT) as a:
        _agent(a)
        a.alias(key, remember=True)

    with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT) as b:
        _agent(b)
        b.link(LinkReason.RESUMED_FROM, key)

    first, second = _emitted(sink)
    (link,) = second.links
    assert link.trace_id == first.context.trace_id
    assert link.span_id == first.context.span_id
    assert link.reason is LinkReason.RESUMED_FROM
    assert counters.get("adapters.lg.link_target_unresolved") == 0


def test_an_opportunistic_link_miss_is_silent_but_a_claimed_one_counts():
    """`expected=False` never dilutes the counted-not-faked default: the same
    unknown selector is silent while the claim is conditional ("IF this ever
    ran here, link it") and counted the moment it is stated as fact.
    """
    ctx, sink = context("lg")
    unknown = UnitKey("lg.thread_id", "never-opened")

    with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT) as s:
        _agent(s)
        s.link(LinkReason.RESUMED_FROM, unknown, expected=False)
        assert counters.get("adapters.lg.link_target_unresolved") == 0
        s.link(LinkReason.RESUMED_FROM, unknown)

    assert _emitted(sink)[-1].links == ()
    assert counters.get("adapters.lg.link_target_unresolved") == 1


def test_a_scope_can_never_link_to_its_own_span():
    """The guard that backs a resume site's link-before-alias ordering: even
    when a selector resolves to the very unit asking — here the scope aliased
    itself before linking — no self-edge ships, and the refused claim is
    counted like any other expected miss.
    """
    ctx, sink = context("lg")
    key = UnitKey("lg.thread_id", "t-1")

    with ctx.enter(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT) as s:
        _agent(s)
        s.alias(key, remember=True)
        s.link(LinkReason.RESUMED_FROM, key)

    assert _emitted(sink)[-1].links == ()
    assert counters.get("adapters.lg.link_target_unresolved") == 1


def test_run_token_is_stable_within_a_run_and_distinct_across_runs():
    """The token exists to make alias VALUES run-scoped, so its whole contract
    is here: same run, same string (a SESSION's own token equals its steps' —
    the enclosing walk counts self); different run, different string; no run
    above, None.
    """
    ctx, sink = context("lg")

    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT
    ) as run:
        _agent(run)
        own = run.run_token()
        with ctx.enter(
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            describe=_step,
        ) as s1:
            first = s1.run_token()
        with ctx.enter(
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            describe=_step,
        ) as s2:
            second = s2.run_token()

    assert first is not None
    assert first == second == own

    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT
    ) as other:
        _agent(other)
        with ctx.enter(
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            describe=_step,
        ) as s3:
            third = s3.run_token()

    assert third is not None
    assert third != first

    with ctx.enter(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        describe=_tool,
    ) as lone:
        assert lone.run_token() is None


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
        from wardex_sdk._assembly import VocabularyError

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
    "alias": ((UnitKey("k", "v"),), {"remember": True}),
    "run_token": ((), {}),
    "claim": ((UnitKey("k", "v"),), {"observer": Observer.EXECUTOR}),
    "claim_run": ((UnitKey("k", "v"),), {"observer": Observer.EXECUTOR}),
    "outranked": ((UnitKey("k", "v"),), {"observer": Observer.EXECUTOR}),
    "child_draft": ((SpanIntent.EXECUTE_TOOL,), {}),
    "close_child": ((NULL_DRAFT,), {}),
    "record_input": ((b"in",), {}),
    "record_output": ((b"out",), {}),
    "record_failure": (("tool_error",), {}),
    "pin": ((), {"driver": threading.current_thread()}),
    "unpin": ((), {}),
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
    from wardex_sdk._assembly import _builder

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
    from wardex_sdk._assembly import _units

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
    assert "NO span of its own" in degraded, f"the line does not name the consequence: {degraded!r}"
    # It names what happens to the work INSIDE the run too, and the wording is
    # measured rather than inherited: `degraded_run()` keeps that traffic being
    # captured — the older line claimed it would not be — so what it costs is
    # the attachment, not the capture.
    assert "orphaned and marked" in degraded

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
    assert Limitation.INSTRUMENTATION_DEGRADED not in markers, (
        "an ordinary close is the adapter's strand to repair, not wardex's"
    )


def test_a_guess_an_eviction_caused_reads_as_wardexs_own_bound():
    """The evict-origin twin of the dead-pin case above.

    `max_units=1`: root A is evicted while its `activate()` fork is still
    entered, so the fork stands holding A's already-shipped context; a NESTED
    site then opens under sole-live B through the declared fallback. That
    strand is wardex's own table at work, so the guess carries
    `INSTRUMENTATION_DEGRADED` — the repair is `max_units` — not the
    `CORRELATION_CONFLICT` that would send the reader hunting an adapter bug
    that does not exist. The WORD comes from the registry
    (`refused_ambient_marker`), the same one `open()` uses when it sees the
    corpse itself — taking a parent unit is what stops it from seeing this
    one, and this path must not fall out of step with the registry's.
    """
    sink = RecordingSink()
    ctx = AdapterContext("test", units=UnitRegistry(sink=sink, max_units=1), limits={})
    dead = _session(ctx, "A")
    fork = dead.activate()
    fork.__enter__()
    _session(ctx, "B")  # evicts A: closed, emitted with UNIT_EVICTED, breadcrumbed

    assert dead.is_live is False

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
    assert Limitation.INSTRUMENTATION_DEGRADED in markers
    assert Limitation.CORRELATION_CONFLICT not in markers


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


def test_a_degraded_scope_says_so_on_the_carrier_the_capture_gate_reads():
    """The block runs with nothing ambient, and under `capture_mode=AGENT` that
    is indistinguishable — to the byte seam three layers away — from a host
    doing ordinary non-agent work. So every request and every tool call inside
    it would be dropped.

    The flag is a `ContextVar` and not a value on the scope for the same reason:
    the seam shares nothing with this module except the TASK the host's code
    runs on, which is the carrier the whole tree is built from anyway.
    """
    from wardex_sdk._assembly import in_degraded_run

    ctx, sink = broken("open")

    assert in_degraded_run() is False
    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT
    ) as scope:
        assert scope.degraded
        assert in_degraded_run() is True
    assert in_degraded_run() is False, "the flag outlived the block it was set for"


def test_a_healthy_scope_leaves_the_carrier_alone():
    """It answers one question — "is the missing parent wardex's doing" — and a
    healthy run never asks it. A flag set on a working scope would widen the
    capture gate for every host whose wardex is fine.
    """
    from wardex_sdk._assembly import in_degraded_run

    ctx, sink = context()

    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, describe=_agent
    ):
        assert in_degraded_run() is False


# --------------------------------------------------------------------------
# a failure the host reported without raising
# --------------------------------------------------------------------------


def test_a_declared_failure_reaches_the_wire_when_nothing_was_raised():
    """The half `_run`'s exception handler structurally cannot see.

    A framework that converts a failure into a RETURN VALUE — LangGraph's
    default `handle_tool_errors` is the case this was built for — leaves
    nothing for an exception handler to classify, so the span shipped `OK` for
    work the host itself had already called failed.
    """
    ctx, sink = context()

    with ctx.enter(
        UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.ROOT, describe=_tool
    ) as scope:
        scope.record_failure("tool_error")

    span = _emitted(sink)[-1]
    assert span.status is StatusCode.ERROR
    assert span.error_type == "tool_error"


def test_an_exception_outranks_a_declared_failure_and_keeps_its_own_type():
    """A declaration may not relabel a crash.

    An exception is stronger evidence and carries a real type. If a declaration
    could overwrite it, an adapter reading a framework's error field would
    replace `RuntimeError` with whatever string it happened to have — and the
    one thing on a span a consumer trusts to be mechanical would become an
    adapter's opinion.
    """
    ctx, sink = context()

    with pytest.raises(RuntimeError, match="the host crashed"):
        with ctx.enter(
            UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.ROOT, describe=_tool
        ) as scope:
            scope.record_failure("tool_error")
            raise RuntimeError("the host crashed")

    span = _emitted(sink)[-1]
    assert span.status is StatusCode.ERROR
    assert span.error_type == "RuntimeError"


def test_a_declared_failure_does_not_turn_control_flow_back_into_a_failure():
    """Control flow resolves to UNSET before the declaration is consulted.

    Both mechanisms exist to stop `type(exc).__name__` being the only source of
    a status, and they push in opposite directions — so the one case where they
    could collide is pinned rather than left to ordering.
    """

    from wardex_sdk._adapters._base import AdapterInterface
    from wardex_sdk._adapters._registry import context_for

    class Bubble(Exception):
        pass

    class Adapter(AdapterInterface):
        CONTROL_FLOW = (Bubble,)

        def name(self) -> str:
            return "cf"

        def install(self, client, ctx=None) -> None:
            return None

        def uninstall(self) -> None:
            return None

    sink = RecordingSink()
    ctx = context_for("cf", None, Adapter())
    ctx._units._sink = sink  # type: ignore[attr-defined]

    with pytest.raises(Bubble):
        with ctx.enter(
            UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.ROOT, describe=_tool
        ) as scope:
            scope.record_failure("tool_error")
            raise Bubble()

    span = _emitted(sink)[-1]
    assert span.status is StatusCode.UNSET
    assert span.error_type is None


def test_a_declared_failure_on_a_degraded_scope_is_silent():
    """Total, exactly like `record_input`: an adapter never has to ask whether
    wardex is working before it may keep describing what it saw."""
    ctx, sink = broken("open")

    with ctx.enter(
        UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.ROOT, describe=_tool
    ) as scope:
        assert scope.degraded
        scope.record_failure("tool_error")


def test_a_root_site_whose_describe_dies_reports_the_WHOLE_RUN_not_one_span(capsys):
    """The operator's only record must not understate a run-sized loss.

    In production `counters` is not exported and `debug` is off, so this one
    stderr line is the entire difference between "wardex deleted a run" and
    "wardex was never installed". Measured here rather than asserted from the
    code: the draft is refused by the vocabulary (the required block never
    landed), so ZERO spans ship for the run — and the branch that says so was
    unreachable, because the flag guarding it means "the unit was closed" and
    is True on exactly this path.
    """
    ctx, sink = context()

    def boom(scope):
        raise RuntimeError("a framework read moved")

    with ctx.enter(
        UnitKind.SESSION, intent=SpanIntent.INVOKE_WORKFLOW, placement=Placement.ROOT, describe=boom
    ) as scope:
        assert scope.degraded

    # ZERO spans for the run, and it is the VOCABULARY that refuses it: the
    # description died before `INVOKE_WORKFLOW`'s required block, so there is
    # nothing to ship rather than something incomplete.
    assert len(sink.drafts) == 1
    with pytest.raises(VocabularyError, match="workflow_name"):
        sink.drafts[0].finish()

    err = capsys.readouterr().err
    assert "this run will produce NO span of its own" in err
    assert "one span is incomplete or missing" not in err, (
        "a run entry reporting a single-span loss is the understatement this fixes"
    )


def test_a_nested_site_whose_describe_dies_still_reports_one_span(capsys):
    """The control. Reordering the branches must not escalate every site."""
    ctx, sink = context()

    def boom(scope):
        raise RuntimeError("a framework read moved")

    with ctx.enter(
        UnitKind.STEP, intent=SpanIntent.EXECUTE_STEP, placement=Placement.NESTED, describe=boom
    ):
        pass

    err = capsys.readouterr().err
    assert "one span is incomplete or missing" in err
    assert "NO span of its own" not in err


# ==========================================================================
# A pin can be taken down again — on the task that installed it
# ==========================================================================


def test_unpin_on_the_same_task_restores_what_was_current():
    """The enclosing pin — or nothing — is current again after `unpin()`."""
    ctx, sink = context()
    outer = ctx.open_run(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT)
    _agent(outer)
    assert outer.pin(driver=threading.current_thread()) is True
    inner = ctx.open_run(UnitKind.AGENT, intent=SpanIntent.INVOKE_AGENT, placement=Placement.NESTED)
    _agent(inner)
    assert inner.pin(driver=threading.current_thread()) is True
    assert ctx._units.current() is inner._unit

    assert inner.unpin() is True
    assert ctx._units.current() is outer._unit
    assert inner.unpin() is False, "a second unpin has nothing to remove"
    inner.close()

    assert outer.unpin() is True
    assert ctx._units.current() is None
    outer.close()
    assert counters.get("assembly._units.unpin") == 0


def test_unpin_from_another_task_is_counted_and_never_raises():
    ctx, sink = context()
    run = ctx.open_run(UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT)
    _agent(run)
    assert run.pin(driver=threading.current_thread()) is True

    answer: list[object] = []
    th = threading.Thread(target=lambda: answer.append(run.unpin()))
    th.start()
    th.join()
    assert answer == [True], "the question was answered, and the registry did the counting"
    assert counters.get("assembly._units.unpin") >= 1
    run.close()


def _two_runs(ctx, *, unpin: bool) -> list:  # noqa: ANN001
    spans = []
    for name in ("first", "second"):
        run = ctx.open_run(
            UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT, subject=name
        )
        _agent(run)
        run.pin(driver=threading.current_thread())
        run.close()
        if unpin:
            run.unpin()
        spans.append(run)
    return spans


def test_two_sequential_pinned_runs_on_one_task_are_two_clean_roots(monkeypatch):
    """Pin, close, unpin, repeat: the second run opens on a clean carrier.

    The NEGATIVE CONTROL is the same sequence with `unpin` turned into a no-op:
    the first run's dead pin is then still standing when the second opens,
    the registry refuses it, and the second root carries the conflict marker.
    That is the failure `unpin()` exists to make unnecessary, shown here so the
    two outcomes are distinguishable by the test and not only by the prose.
    """
    ctx, sink = context()
    _two_runs(ctx, unpin=True)
    spans = [d.finish() for d in sink.drafts]
    assert [s.name for s in spans] == ["invoke_agent first", "invoke_agent second"]
    for span in spans:
        assert span.parent_span_id is None
        assert span.capture_integrity is None or (
            Limitation.CORRELATION_CONFLICT not in span.capture_integrity.limitations
        )

    ctx2, sink2 = context()
    monkeypatch.setattr(RunHandle, "unpin", lambda self: False)
    _two_runs(ctx2, unpin=True)
    second = sink2.drafts[-1].finish()
    assert second.name == "invoke_agent second"
    assert Limitation.CORRELATION_CONFLICT in second.capture_integrity.limitations


def test_a_stated_conversation_reaches_a_nested_child_and_the_ambient_under_the_pin():
    """`open_run(conversation=)` is the framework's group id: the run's own
    span, a child unit opened under it, and the AMBIENT under the run's pin
    all carry it. What this does NOT prove — and the last assertion pins as
    the documented gap — is that a wire span carries it: the byte seam
    builds its own `Ambient(conversation=None)` from the latched span
    context, so `resolve_parentage` over what the seam actually latches
    answers no conversation."""
    from wardex_sdk._assembly import Ambient, latch_ambient, resolve_parentage

    ctx, sink = context()
    conv = ConversationContext(conversation_id="conv-123")
    run = ctx.open_run(
        UnitKind.SESSION,
        intent=SpanIntent.INVOKE_AGENT,
        placement=Placement.ROOT,
        conversation=conv,
    )
    _agent(run)
    assert run.pin(driver=threading.current_thread()) is True
    with ctx.enter(
        UnitKind.CALL, intent=SpanIntent.EXECUTE_TOOL, placement=Placement.NESTED, subject="t"
    ) as s:
        s.draft.set_tool(ToolAttributes(name="t"))
        wire = resolve_parentage(latch_ambient())
        # The seam's own latch, as `_interceptors/_seam.py::_latched` builds
        # it: the span context alone. A wire span parented through it has no
        # conversation — the gap the README and CHANGELOG name.
        seam = resolve_parentage(Ambient(latch_ambient().span_context, None, None))
    assert wire.parent_span_id == s.draft.context.span_id
    assert wire.conversation is not None and wire.conversation.conversation_id == "conv-123"
    assert seam.parent_span_id == s.draft.context.span_id
    assert seam.conversation is None
    run.unpin()
    run.close()
    spans = [d.finish() for d in sink.drafts]
    assert {s.name for s in spans} == {"execute_tool t", "invoke_agent"}
    for span in spans:
        assert span.conversation is not None
        assert span.conversation.conversation_id == "conv-123"


def test_the_fork_reset_drops_every_slot():
    """A slot holds a run's in-flight bookkeeping; the child must start empty."""
    ctx, _ = context()

    class Holder:
        pass

    holder = Holder()
    ctx.slot(holder)["handle"] = "parent's"
    ctx._at_fork_reinit()
    assert ctx.slot(holder) == {}
