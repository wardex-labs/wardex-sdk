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
from wardex_sdk._types import AgentAttributes
from wardex_sdk.adapters._context import (
    AdapterContext,
    Attachment,
    InstallOutcome,
    Observer,
    Placement,
    RunHandle,
    Scope,
)
from wardex_sdk.assembly import (
    Limitation,
    LinkReason,
    ParentSource,
    SpanIntent,
    UnitKey,
    UnitKind,
    UnitRegistry,
    counters,
)
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
    yield
    _ambient_unit.reset(token)
    reset_for_test()
    counters.reset()


def context(name: str = "test", **kw) -> tuple[AdapterContext, RecordingSink]:
    sink = kw.pop("sink", RecordingSink())
    return AdapterContext(name, units=UnitRegistry(sink=sink), limits={}, **kw), sink


def _agent(handle) -> None:
    """`invoke_agent` needs an agent block or the closed vocabulary refuses it."""
    handle.draft.set_agent(AgentAttributes(name="a", agent_type=AgentType.PRIMARY))


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
