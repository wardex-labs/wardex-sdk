"""The conformance suite's own proof of life.

A gate nobody has watched fail is not a gate, and the suite makes exactly one
claim that is hard to believe: that it catches a causal tree which has totally
collapsed. So this file builds a real adapter that produces one — the step span
opened with an empty body and the work run AFTER the `with`, which is what an
adapter that wraps a framework's PRODUCER instead of its consumer does — and
watches the tier half wave it through while the id half fails.

The fake framework is deliberately not a mock of the adapter surface. It is a
tiny library with a run entry, a step entry and a streaming entry, and the fake
adapters below patch it exactly the way the shipped ones patch theirs, so the
suite is exercised through the same `AdapterContext` the real subjects use.
"""

from __future__ import annotations

import ast
from functools import partial

import pytest

from wardex_sdk._adapters._base import AdapterInterface
from wardex_sdk._adapters._context import AdapterContext, Placement
from wardex_sdk._assembly import Limitation, SpanIntent, ToolAttributes, UnitKind
from wardex_sdk.testing import (
    AdapterConformanceSuite,
    AdapterSubject,
    StalledRun,
    installed_adapter,
)
from wardex_sdk.testing.conformance import _declares_placement

_MODULE = __name__


# --------------------------------------------------------------------------
# a framework to instrument
# --------------------------------------------------------------------------


class FakeFramework:
    """Three entry points, mirroring the shapes a real adapter has to wrap.

    `run` delegates to `run_step` through the CLASS rather than through a local
    reference, so a patch on `run_step` is live for calls `run` makes — which is
    the relationship every real seam pair has and the thing a fake built out of
    two independent functions quietly loses.
    """

    @staticmethod
    def run(steps):  # noqa: ANN001, ANN205
        return [FakeFramework.run_step(step) for step in steps]

    @staticmethod
    def run_step(step):  # noqa: ANN001, ANN205
        return step()

    @staticmethod
    def stream(steps):  # noqa: ANN001, ANN205
        for step in steps:
            yield FakeFramework.run_step(step)


def seams() -> dict[str, object]:
    return {
        "run": FakeFramework.run,
        "run_step": FakeFramework.run_step,
        "stream": FakeFramework.stream,
    }


# --------------------------------------------------------------------------
# an adapter for it
# --------------------------------------------------------------------------


def _describe_run(run) -> None:  # noqa: ANN001
    run.draft.set_workflow_name("Fake")


def _describe_step(name: str, step) -> None:  # noqa: ANN001
    step.draft.set_extra("wardex.step.name", name)


def _describe_leaf(name: str, leaf) -> None:  # noqa: ANN001
    leaf.draft.set_tool(ToolAttributes(name=name))


def leaf_span(ctx: AdapterContext, subject: str) -> None:
    """A stand-in for a byte-seam span, opened inside a step body.

    A unit opened through `ctx.enter` resolves its parent through the same
    ambient a byte seam latches, so the edge under test is the edge a real
    outbound request would get.
    """
    describe = partial(_describe_leaf, subject)
    with ctx.enter(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        subject=subject,
        describe=describe,
    ):
        pass


def _mk_run(original, adapter):  # noqa: ANN001, ANN202
    def wrapper(steps):  # noqa: ANN001, ANN202
        ctx = adapter._ctx
        if ctx is None:
            return original(steps)
        with ctx.enter(
            UnitKind.SESSION,
            intent=SpanIntent.INVOKE_WORKFLOW,
            placement=Placement.ROOT,
            subject="Fake",
            describe=_describe_run,
        ):
            return original(steps)

    return wrapper


def _mk_stream(original, adapter):  # noqa: ANN001, ANN202
    """A GENERATOR FUNCTION, so the run's lifetime is the iteration's — which is
    what gives the shutdown checks a run that is open when they arrive."""

    def wrapper(steps):  # noqa: ANN001, ANN202
        ctx = adapter._ctx
        if ctx is None:
            return (yield from original(steps))
        with ctx.enter(
            UnitKind.SESSION,
            intent=SpanIntent.INVOKE_WORKFLOW,
            placement=Placement.ROOT,
            subject="Stalled",
            describe=_describe_run,
        ):
            return (yield from original(steps))

    return wrapper


def _mk_step(original, adapter):  # noqa: ANN001, ANN202
    def wrapper(step):  # noqa: ANN001, ANN202
        ctx = adapter._ctx
        if ctx is None:
            return original(step)
        subject = step.__name__
        describe = partial(_describe_step, subject)
        with ctx.enter(
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            subject=subject,
            describe=describe,
        ):
            return original(step)

    return wrapper


def _mk_collapsed_step(original, adapter):  # noqa: ANN001, ANN202
    """THE defect. The span exists, is the run's child, and holds no work.

    Every step span is real, correctly parented and decorative; the step body
    then runs with the RUN ambient, so every leaf latches the run. One trace,
    right counts, `unit_active` at 1.0 on every edge, no marker anywhere.
    """

    def wrapper(step):  # noqa: ANN001, ANN202
        ctx = adapter._ctx
        if ctx is None:
            return original(step)
        subject = step.__name__
        describe = partial(_describe_step, subject)
        with ctx.enter(
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            subject=subject,
            describe=describe,
        ):
            pass
        return original(step)

    return wrapper


class _FakeAdapter(AdapterInterface):
    """The healthy adapter. `_step_seam` is what the collapsed twin overrides."""

    _step_seam = staticmethod(_mk_step)

    def __init__(self) -> None:
        self._installed = False
        self._ctx: AdapterContext | None = None

    def name(self) -> str:
        return "fake_conformance"

    def install(self, client=None, ctx=None) -> None:  # noqa: ANN001
        if self._installed:
            return
        self._ctx = ctx if isinstance(ctx, AdapterContext) else None
        if self._ctx is None:
            return
        patches = self._ctx.patches
        patches.patch(FakeFramework, "run", _mk_run(FakeFramework.run, self))
        patches.patch(FakeFramework, "stream", _mk_stream(FakeFramework.stream, self))
        patches.patch(
            FakeFramework, "run_step", type(self)._step_seam(FakeFramework.run_step, self)
        )
        self._installed = True

    def uninstall(self) -> None:
        ctx = self._ctx
        self._installed = False
        if ctx is None:
            return
        ctx.patches.restore_all()
        ctx.close_all(marker=Limitation.ADAPTER_UNINSTALLED)

    def close_units(self, *, marker: Limitation) -> None:
        if self._ctx is not None:
            self._ctx.close_all(marker=marker)


class _CollapsedAdapter(_FakeAdapter):
    _step_seam = staticmethod(_mk_collapsed_step)


# --------------------------------------------------------------------------
# the subject
# --------------------------------------------------------------------------


def _steps(live):  # noqa: ANN001, ANN202
    """Two step callables, each opening one leaf span when there is a context."""

    def make(name: str):  # noqa: ANN202
        def step():  # noqa: ANN202
            if live.ctx is not None:
                leaf_span(live.ctx, f"leaf-{name}")
            return name

        step.__name__ = name
        return step

    return [make("s0"), make("s1")]


def _workload(live):  # noqa: ANN001, ANN202
    return FakeFramework.run(_steps(live))


def _stall(live) -> StalledRun:  # noqa: ANN001
    it = FakeFramework.stream(_steps(live))
    next(it)
    return StalledRun(root="invoke_workflow Stalled", resume=lambda: list(it))


def _subject(factory) -> AdapterSubject:  # noqa: ANN001
    return AdapterSubject(
        name="fake_conformance",
        module=_MODULE,
        factory=factory,
        seams=seams,
        workload=_workload,
        chains=(
            ("invoke_workflow Fake", "execute_step s0", "execute_tool leaf-s0"),
            ("invoke_workflow Fake", "execute_step s1", "execute_tool leaf-s1"),
        ),
        stall=_stall,
        detect_package="wardex_sdk",
        usage_expected=False,
    )


@pytest.fixture
def healthy() -> AdapterSubject:
    return _subject(_FakeAdapter)


@pytest.fixture
def collapsed() -> AdapterSubject:
    return _subject(_CollapsedAdapter)


# --------------------------------------------------------------------------
# the suite passes a correct adapter
# --------------------------------------------------------------------------

_TREE_CHECKS = (
    "check_the_declared_tree_is_one_a_collapse_could_break",
    "check_install_replaces_every_seam_and_uninstall_restores_it",
    "check_a_restored_seam_captures_nothing",
    "check_the_workload_ships_one_read_tree",
    "check_the_causal_chain_holds_by_span_id",
    "check_a_collapsed_tree_fails_this_suite",
    "check_usage_totals_are_inclusive",
    "check_close_units_ships_the_open_run_and_stays_installed",
    "check_uninstall_ships_the_open_run_and_never_raises_into_the_host",
)


@pytest.mark.parametrize("check", _TREE_CHECKS)
def test_a_correct_adapter_passes_every_check(healthy, check):
    AdapterConformanceSuite(healthy).run(check)


def test_the_check_list_is_covered_here_or_named_as_uncovered(healthy):
    """`check_the_adapter_names_itself_after_its_enum_member` and the placement
    rule are the two checks a synthetic subject cannot drive honestly: the fake
    adapter is not in the factory table, and the placement rule reads THIS
    module's source rather than an adapter's. They are covered by the real
    subjects and by `test_the_placement_rule_sees_...` below.
    """
    uncovered = set(AdapterConformanceSuite.CHECKS) - set(_TREE_CHECKS)
    assert uncovered == {
        "check_the_adapter_names_itself_after_its_enum_member",
        "check_every_span_site_declares_its_placement",
    }


# --------------------------------------------------------------------------
# THE negative case: a really collapsed tree must fail the suite
# --------------------------------------------------------------------------


def test_the_tier_half_alone_cannot_see_a_collapsed_tree(collapsed):
    """Both halves, on one deliberately broken adapter. This is the whole file.

    The tier half passes: every span exists, the counts are right, one trace,
    `unit_active` at 1.0 everywhere, no marker anywhere. A suite built out of
    those assertions certifies an adapter whose causal tree is one level deep.
    """
    suite = AdapterConformanceSuite(collapsed)
    suite.run("check_the_workload_ships_one_read_tree")

    with pytest.raises(AssertionError, match="is parented to"):
        suite.run("check_the_causal_chain_holds_by_span_id")


def test_the_collapse_is_a_real_one_and_not_an_absence(collapsed):
    """The broken adapter still emits every span, or the test above proves
    nothing: an adapter that emitted no step spans at all would also fail the id
    half, for a reason a reader would then attribute to the wrong defect."""
    with installed_adapter(collapsed.factory) as live:
        collapsed.workload(live)
        names = sorted(span.name for span in live.spans)
    assert names == sorted({name for chain in collapsed.chains for name in chain})


# --------------------------------------------------------------------------
# the suite refuses a subject that could not fail it
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chains", "why"),
    [
        ((("r", "n", "l"),), "at least two chains"),
        ((("r", "n0", "l0"), ("r", "n1")), "shallower than root"),
        ((("r", "n", "l0"), ("r", "n", "l1")), "share the intermediate"),
        ((("r", "n0", "l0"), ("other", "n1", "l1")), "one declared tree has one root"),
        ((("r", "n0", "n0"), ("r", "n1", "l1")), "names a span twice"),
    ],
)
def test_a_subject_whose_tree_could_not_collapse_is_refused(healthy, chains, why):
    """Each of these declares a tree on which the id half and the tier half
    agree for every workload — which is a suite that certifies without
    measuring."""
    import dataclasses

    subject = dataclasses.replace(healthy, chains=chains)
    with pytest.raises(AssertionError, match=why):
        AdapterConformanceSuite(subject).run(
            "check_the_declared_tree_is_one_a_collapse_could_break"
        )


def test_an_unknown_check_name_is_refused(healthy):
    with pytest.raises(AssertionError, match="is not a conformance check"):
        AdapterConformanceSuite(healthy).run("check_nothing_at_all")


# --------------------------------------------------------------------------
# the placement rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "with ctx.enter(K, intent=I, placement=placement) as s:\n    pass\n",
        "with ctx.enter(K, intent=I, placement=_PLACEMENT) as s:\n    pass\n",
        "with ctx.enter(K, intent=I, placement=_pick(task)) as s:\n    pass\n",
        "with ctx.enter(K, intent=I) as s:\n    pass\n",
        "handle = ctx.open_run(K, intent=I)\n",
    ],
)
def test_the_placement_rule_sees_a_site_that_does_not_declare_a_literal(source):
    """A computed placement is the failure the rule exists for: it lets one site
    decide at runtime that it may begin a trace, which is how a run entry nobody
    wrapped stops being visible."""
    site = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) in ("enter", "open_run")
    )
    assert not _declares_placement(site)


def test_the_placement_rule_accepts_the_literal():
    source = "with ctx.enter(K, intent=I, placement=Placement.NESTED) as s:\n    pass\n"
    site = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "enter"
    )
    assert _declares_placement(site)


# --------------------------------------------------------------------------
# the shipped surface
# --------------------------------------------------------------------------


def test_the_package_exports_a_sorted_resolvable_all():
    import wardex_sdk.testing as testing

    assert testing.__all__ == sorted(testing.__all__)
    for name in testing.__all__:
        assert hasattr(testing, name)
