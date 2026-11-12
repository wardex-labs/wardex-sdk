"""A unit's close at the recursion limit leaves the host's exception alone.

`AdapterContext._run` closes a unit on the way out of the adapter's body. A host
that lowers `sys.setrecursionlimit` inside the body can leave that close too few
frames, and the guards let a `RecursionError` through rather than count it, so the
close used to raise a fresh one chained onto whatever the body was leaving with.
The exit now drops its own work there: it writes back the ambient carriers it
replaced (C calls only), tallies the drop with `counters.tally`, and the host's
exception, or the host's return, leaves the block as it would without wardex.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from wardex_sdk import _hub
from wardex_sdk._adapters._context import AdapterContext, Placement, Scope
from wardex_sdk._adapters._sink import _ClientSink
from wardex_sdk._assembly import Limitation, SpanIntent, UnitKind, UnitRegistry, _ledger, counters
from wardex_sdk._assembly._ledger import DropReason
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._enums import StatusCode
from wardex_sdk._types import Envelope
from wardex_sdk.transport._base import Transport

_LABEL = "adapters.context.close_at_limit"


class _Recording(Transport):
    def __init__(self) -> None:
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)


class HostFailed(Exception):
    pass


def _stack_depth() -> int:
    frame, n = sys._getframe(1), 0
    while frame is not None:
        n += 1
        frame = frame.f_back
    return n


def _lowest_recursion_limit() -> int:
    """The lowest limit `sys.setrecursionlimit` accepts from the caller's depth.

    Copied from `test_span_exceptions.py`: 3.10 counts C-level calls in the depth
    it checks the new limit against, so a limit computed from the Python frame
    count is refused there, while 3.12+ counts Python frames alone.
    """
    n = _stack_depth()
    while True:
        try:
            sys.setrecursionlimit(n)
        except RecursionError:
            n += 1
        else:
            return n


def _chain_length(exc: BaseException) -> int:
    n = 0
    seen: BaseException | None = exc
    while seen is not None:
        n += 1
        seen = seen.__cause__ if seen.__cause__ is not None else seen.__context__
    return n


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[AdapterContext, _Recording, list]]:
    _hub.reset_for_test()
    recording = _Recording()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), recording))
    counters.reset()
    _ledger._reset_for_tests()
    unraisable: list[Any] = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    units = UnitRegistry(sink=_ClientSink(_hub.get_client()))
    yield AdapterContext("probe", units=units, limits={}), recording, unraisable
    _hub.reset_for_test()


def _describe(scope: Scope) -> None:
    scope.draft.set_extra("wardex.step.name", "n")


def _block(ctx: AdapterContext, headroom: int, *, raises: bool) -> BaseException | None:
    """Enter a unit, lower the limit inside its body, and leave by raising or returning."""
    limit = sys.getrecursionlimit()
    caught = None
    try:
        with ctx.enter(
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            describe=_describe,
        ):
            sys.setrecursionlimit(_lowest_recursion_limit() + headroom)
            if raises:
                raise HostFailed("host")
    except BaseException as exc:  # noqa: BLE001 -- whatever leaves the block is the subject
        caught = exc
    finally:
        sys.setrecursionlimit(limit)
    return caught


def _steps(recording: _Recording) -> list:
    _hub.get_client().flush()
    return [sp for env in recording.envelopes for sp in env.spans]


# The close needs about eleven free frames here with a real client sink, so every
# headroom below that cuts it. From 2, because below it the drop cannot run either:
# at 0 the interpreter has no room to resume the `enter` generator at all, so `_run`'s
# exit never starts and the suspended activation is finalized with no room either;
# and 3.10 counts C calls against the limit too, so at 1 the drop's own `var.set`
# raises there. No exit written in Python reaches further down.
@pytest.mark.parametrize("headroom", range(2, 9))
@pytest.mark.parametrize("raises", [True, False], ids=["raise", "return"])
def test_a_unit_that_closes_without_room_leaves_the_host_alone(setup, headroom, raises):
    """The host's exception arrives alone, or the block returns; the scope and the
    ambient unit are what they were before the unit; nothing counts as wardex's own
    failure; and the dropped close is tallied once."""
    ctx, recording, unraisable = setup
    scope = _hub.get_current_scope()

    caught = _block(ctx, headroom, raises=raises)

    if raises:
        assert isinstance(caught, HostFailed)
        assert caught.__context__ is None
        assert _chain_length(caught) == 1
    else:
        assert caught is None
    assert _hub.get_current_scope() is scope
    assert ctx._units.current() is None
    assert _ledger.snapshot().total(DropReason.INTERNAL_SDK_ERROR) == 0
    assert counters.get(_LABEL) == 1
    assert unraisable == []
    assert _steps(recording) == []
    # The span did not ship now. Whether it ships later depends on how far the close got
    # before the limit bit: still registered, the parent's teardown ends it; already
    # deregistered, it is lost and the tally above is its only record.
    registered = len(ctx._units._roots)
    assert registered in (0, 1)
    ctx._units.close_all(reason=Limitation.UNIT_INTERRUPTED)
    assert len(_steps(recording)) == registered


def test_the_next_sibling_after_a_dropped_close_is_not_parented_to_it(setup):
    """The dead unit is not left ambient: a unit opened next on this task is a root
    of its own, not a child of a span that never ships."""
    ctx, recording, _ = setup
    _block(ctx, 3, raises=True)

    with ctx.enter(
        UnitKind.STEP,
        intent=SpanIntent.EXECUTE_STEP,
        placement=Placement.ROOT,
        describe=_describe,
    ):
        pass

    (sibling,) = _steps(recording)
    assert sibling.parent_span_id is None


@pytest.mark.parametrize("raises", [True, False], ids=["raise", "return"])
def test_a_unit_with_room_to_close_closes_as_before(setup, raises):
    """With room to spare nothing changes: the span ships with the host's outcome,
    and nothing is tallied."""
    ctx, recording, unraisable = setup

    caught = _block(ctx, 200, raises=raises)

    (step,) = _steps(recording)
    if raises:
        assert isinstance(caught, HostFailed)
        assert _chain_length(caught) == 1
        assert (step.status, step.error_type) == (StatusCode.ERROR, "HostFailed")
    else:
        assert caught is None
        assert step.status != StatusCode.ERROR
    assert ctx._units.current() is None
    assert counters.get(_LABEL) == 0
    assert _ledger.snapshot().total(DropReason.INTERNAL_SDK_ERROR) == 0
    assert unraisable == []


# 3.10 counts C calls against the limit, and on a second thread the exit there runs out
# below 3 whether it writes anything or not; 3.12+ reaches 1. From 3 on both.
@pytest.mark.parametrize("headroom", range(3, 9))
def test_a_unit_closed_at_the_limit_on_another_thread_leaves_that_thread_alone(setup, headroom):
    """A generator-shaped host can enter a unit on one thread and close it on another.
    At the limit the exit must not write the first thread's pre-unit carriers onto the
    second: the second thread keeps its own unit ambient and its own scope, its next
    unit is that unit's child, and nothing reaches stderr."""
    ctx, recording, unraisable = setup

    def host() -> Iterator[None]:
        with ctx.enter(
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.ROOT,
            describe=_describe,
        ):
            yield

    gen = host()
    entered, release = threading.Event(), threading.Event()

    def first() -> None:
        next(gen)
        entered.set()
        release.wait()

    seen: dict[str, Any] = {}

    def second() -> None:
        with ctx.enter(
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.ROOT,
            describe=_describe,
        ):
            own_unit, own_scope = ctx._units.current(), _hub.get_current_scope()
            limit = sys.getrecursionlimit()
            try:
                sys.setrecursionlimit(_lowest_recursion_limit() + headroom)
                try:
                    gen.close()
                except BaseException as exc:  # noqa: BLE001 -- whatever leaves is the subject
                    seen["left"] = exc
            finally:
                sys.setrecursionlimit(limit)
            seen["unit_kept"] = ctx._units.current() is own_unit
            seen["scope_kept"] = _hub.get_current_scope() is own_scope
            with ctx.enter(
                UnitKind.STEP,
                intent=SpanIntent.EXECUTE_STEP,
                placement=Placement.NESTED,
                describe=_describe,
            ):
                pass

    a = threading.Thread(target=first)
    a.start()
    entered.wait()
    b = threading.Thread(target=second)
    b.start()
    b.join()
    release.set()
    a.join()

    assert "left" not in seen
    assert seen["unit_kept"] and seen["scope_kept"]
    assert counters.get(_LABEL) == 1
    # The refusal is on record once: in the drop's own tally when the limit bit before the
    # activation's exit ran, or as that exit's two refused resets when it got that far.
    refused = (
        counters.get(_LABEL + "_not_restored"),
        counters.get("assembly._units.activate_exit"),
    )
    assert refused in ((1, 0), (0, 2))
    assert unraisable == []
    roots = [sp for sp in _steps(recording) if sp.parent_span_id is None]
    children = [sp for sp in _steps(recording) if sp.parent_span_id is not None]
    assert (len(roots), len(children)) == (1, 1)
    assert children[0].parent_span_id == roots[0].context.span_id


def test_a_control_flow_read_that_meets_the_limit_keeps_the_error_and_is_counted(
    setup, monkeypatch
):
    """The adapter's control-flow question, asked at the limit, cannot be answered.
    The span keeps the host's error rather than a guess, the host's exception leaves
    alone, and the unanswered read is counted."""
    ctx, recording, unraisable = setup

    def at_limit(*_args: Any) -> bool:
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(AdapterContext, "_is_control_flow", at_limit)

    caught = _block(ctx, 200, raises=True)

    assert isinstance(caught, HostFailed)
    assert _chain_length(caught) == 1
    (step,) = _steps(recording)
    assert (step.status, step.error_type) == (StatusCode.ERROR, "HostFailed")
    assert counters.get("adapters.context.control_flow_at_limit") == 1
    assert unraisable == []
