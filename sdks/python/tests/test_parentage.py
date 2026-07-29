"""Unit tests for the assembly core landed by migration step 0 — design §4.1, §7.6.

Nothing in the SDK calls this code yet; these tests are what makes step 0 a
landing rather than a deposit. The one that matters most is
`test_framework_ids_do_not_change_the_edge`: it is the in-the-small form of
conformance C-3, the product claim that the causal tree comes from in-process
context propagation and never from a framework's identifiers.
"""

from __future__ import annotations

import sys
from uuid import uuid4

import pytest

from wardex_sdk import _hub
from wardex_sdk._types import ConversationContext, SpanContext, SpanId, TraceId
from wardex_sdk.assembly import (
    AMBIENT,
    EMPTY_AMBIENT,
    Ambient,
    Evidence,
    Limitation,
    ParentSource,
    child_of,
    counters,
    guard,
    latch_ambient,
    resolve_parentage,
)
from wardex_sdk.assembly._diag import LOG_FAILED
from wardex_sdk.assembly._parentage import _CONFIDENCE, _MARKER


def setup_function():
    _hub.reset_for_test()
    counters.reset()


def _ctx(*, flags: int = 0, remote: bool = False) -> SpanContext:
    return SpanContext(
        trace_id=TraceId.generate(),
        span_id=SpanId.generate(),
        trace_flags=flags,
        is_remote=remote,
    )


# --- root vs unresolved: "no parent" is two different facts (I4) -------------


def test_no_ambient_parent_starts_a_declared_trace_root():
    p = resolve_parentage(EMPTY_AMBIENT)

    assert p.parent_span_id is None
    assert p.joined is False
    assert p.correlation is not None
    assert p.correlation.strategy == ParentSource.TRACE_ROOT.value
    assert p.correlation.confidence == 1.0
    # wardex does not head-sample, so a trace it ORIGINATES is sampled (V9).
    # This is the value `_w3c.format_traceparent` hardcodes as `-01` today; step
    # 1 deletes that hardcode and reads the field, and if this is 0 then every
    # wardex-rooted trace goes out `-00` and every downstream OTel service on the
    # default ParentBased(ALWAYS_ON) sampler silently stops recording.
    assert p.trace_flags == 1


def test_an_upstream_not_sampled_flag_survives_being_joined():
    """The other half of V9: 0 must keep meaning "an upstream told us -00"."""
    p = resolve_parentage(Ambient(_ctx(flags=0, remote=True), None, None))

    assert p.joined is True
    assert p.trace_flags == 0
    assert p.child_context().trace_flags == 0


def test_expected_parent_that_is_missing_is_unresolved_not_root():
    p = resolve_parentage(EMPTY_AMBIENT, Evidence(ParentSource.UNRESOLVED))

    assert p.parent_span_id is None
    assert p.correlation.strategy == ParentSource.UNRESOLVED.value
    assert p.correlation.confidence == 0.0


def test_correlation_is_never_none_on_any_path():
    parent = _ctx()
    for parentage in (
        resolve_parentage(EMPTY_AMBIENT),
        resolve_parentage(EMPTY_AMBIENT, Evidence(ParentSource.UNRESOLVED)),
        resolve_parentage(Ambient(parent, None, None)),
        child_of(parent, Evidence(ParentSource.UNIT_ACTIVE)),
    ):
        assert parentage.correlation is not None


# --- joining an ambient parent ---------------------------------------------


def test_ambient_parent_is_joined_with_its_trace():
    parent = _ctx()

    p = resolve_parentage(Ambient(parent, None, None))

    assert p.joined is True
    assert p.trace_id == parent.trace_id
    assert p.parent_span_id == parent.span_id
    assert p.correlation.strategy == ParentSource.CONTEXTVAR.value
    assert p.correlation.active_span_id_at_capture == parent.span_id


def test_trace_flags_propagate_to_the_child_context():
    parent = _ctx(flags=1)

    p = resolve_parentage(Ambient(parent, None, None))
    child = p.child_context()

    assert p.trace_flags == 1
    assert child.trace_flags == 1
    assert child.trace_id == parent.trace_id
    assert child.is_remote is False


def test_child_context_allocates_a_fresh_span_id_each_time():
    p = resolve_parentage(Ambient(_ctx(), None, None))

    first, second = p.child_context(), p.child_context()

    assert first.span_id != second.span_id
    assert first.trace_id == second.trace_id


def test_conversation_and_tracestate_ride_along():
    conv = ConversationContext(conversation_id="c-1", turn_index=3)

    p = resolve_parentage(Ambient(_ctx(), conv, "vendor=1"))

    assert p.conversation is conv
    assert p.tracestate == "vendor=1"


# --- a remote parent is a joined parent, not a contextvar parent -------------


def test_remote_parent_is_recorded_as_header_not_contextvar():
    p = resolve_parentage(Ambient(_ctx(remote=True), None, None), AMBIENT)

    assert p.correlation.strategy == ParentSource.HEADER.value
    assert p.correlation.confidence == 1.0


def test_remote_correction_does_not_override_an_explicit_source():
    p = resolve_parentage(Ambient(_ctx(remote=True), None, None), Evidence(ParentSource.UNIT_ALIAS))

    assert p.correlation.strategy == ParentSource.UNIT_ALIAS.value


def test_ambient_is_local_only_for_a_non_remote_parent():
    assert Ambient(_ctx(), None, None).is_local is True
    assert Ambient(_ctx(remote=True), None, None).is_local is False
    assert EMPTY_AMBIENT.is_local is False


# --- confidence: a guess reports itself (I4) --------------------------------


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (ParentSource.CONTEXTVAR, 1.0),
        (ParentSource.UNIT_ACTIVE, 1.0),
        (ParentSource.UNIT_ALIAS, 0.9),
        (ParentSource.UNIT_SOLE, 0.5),
    ],
)
def test_each_strategy_carries_its_declared_confidence(source, expected):
    p = child_of(_ctx(), Evidence(source))

    assert p.correlation.confidence == expected


def test_supplied_confidence_may_lower_but_never_raise_the_default():
    lowered = child_of(_ctx(), Evidence(ParentSource.UNIT_ALIAS, confidence=0.2))
    inflated = child_of(_ctx(), Evidence(ParentSource.UNIT_SOLE, confidence=1.0))

    assert lowered.correlation.confidence == 0.2
    assert inflated.correlation.confidence == 0.5  # clamped to the UNIT_SOLE default


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        (-0.5, 0.0),  # an honest decay expression that ran past its floor
        (-1e300, 0.0),
        (0.0, 0.0),
        (float("nan"), 0.9),  # "I don't know" == confidence=None, NOT 0.9-by-luck
        (float("inf"), 0.9),
        (float("-inf"), 0.0),
    ],
)
def test_confidence_never_leaves_its_declared_zero_to_one_domain(supplied, expected):
    """`CorrelationInfo.confidence` is documented 0.0~1.0 and validated nowhere else.

    `confidence=1.0 - staleness_s / 60.0` is the natural way an adapter lowers
    confidence honestly, and it goes negative the moment the alias is a minute
    old. NaN is the same accident from `matched / total` with `total == 0`; since
    `nan < x` is False an unclamped `min()` would let it through as the FULL
    table default — the one direction the clamp exists to prevent.
    """
    p = child_of(_ctx(), Evidence(ParentSource.UNIT_ALIAS, confidence=supplied))

    assert p.correlation.confidence == expected


def test_nan_confidence_is_not_promoted_to_a_full_confidence_edge():
    p = child_of(_ctx(), Evidence(ParentSource.CONTEXTVAR, confidence=float("nan")))

    assert p.correlation.confidence == 1.0  # the default, because NaN said nothing


def test_every_parent_source_has_a_declared_confidence_and_marker():
    """Adding a `ParentSource` member without the tables is a KeyError in the host.

    `resolve_parentage` is the chokepoint under every span the SDK emits and is
    deliberately NOT wrapped in `guard()` — it is the core, not an adapter. So
    the omission has to fail here, at collection time, rather than in a user's
    process the first time the new member is passed.
    """
    assert set(_CONFIDENCE) == set(ParentSource)
    assert set(_MARKER) == set(ParentSource)


# --- framework ids are hints, never parentage (I2 — the product claim) -------


def test_framework_ids_do_not_change_the_edge():
    parent = _ctx()
    ambient = Ambient(parent, None, None)

    a = resolve_parentage(
        ambient, Evidence(ParentSource.UNIT_ALIAS, request_id="req-1", operation_id="run-1")
    )
    b = resolve_parentage(
        ambient,
        Evidence(ParentSource.UNIT_ALIAS, request_id="totally-different", operation_id="also"),
    )

    # The edge is identical; only the recorded hint differs.
    assert (a.trace_id, a.parent_span_id) == (b.trace_id, b.parent_span_id)
    assert a.correlation.request_id != b.correlation.request_id


def test_a_framework_id_alone_never_produces_a_parent():
    """C-3 in the small, on the branch where it can actually fail.

    The test above only exercises the branch where an ambient parent already
    exists — where the hints are ignored BY CONSTRUCTION, so it asserts a
    tautology. The dangerous branch is `parent is None`: that is where a lookup
    keyed on `operation_id` would be added, and where it would silently start
    manufacturing edges. Ten distinct id sets must produce ten parentless roots
    in ten distinct traces; anything that selects or joins on an id fails here.
    """
    seen: set[str] = set()
    for n in range(10):
        p = resolve_parentage(
            EMPTY_AMBIENT,
            Evidence(
                ParentSource.UNIT_ALIAS,
                request_id=f"req-{n}",
                operation_id=f"run-{n}",
                attempt_id=f"try-{n}",
            ),
        )

        assert p.parent_span_id is None, "a framework id manufactured a parent edge"
        assert p.joined is False, "a framework id joined an existing trace"
        seen.add(p.trace_id.hex())

    assert len(seen) == 10, "framework ids selected a shared trace"


def test_replacing_every_framework_id_leaves_the_tree_identical():
    """The whole product claim, reduced to one assertion (conformance C-3)."""
    parent = _ctx()
    real = Evidence(ParentSource.UNIT_ALIAS, request_id="run_01H8X", operation_id="sess-42")
    scrambled = Evidence(ParentSource.UNIT_ALIAS, request_id=uuid4().hex, operation_id=uuid4().hex)

    a = resolve_parentage(Ambient(parent, None, None), real)
    b = resolve_parentage(Ambient(parent, None, None), scrambled)

    assert (a.trace_id, a.parent_span_id, a.joined) == (b.trace_id, b.parent_span_id, b.joined)
    assert a.correlation.confidence == b.correlation.confidence
    assert a.limitations == b.limitations


def test_hints_are_recorded_on_both_the_root_and_the_joined_path():
    ev = Evidence(ParentSource.UNIT_ALIAS, request_id="r", operation_id="o", attempt_id="a")

    root = resolve_parentage(EMPTY_AMBIENT, ev)
    joined = resolve_parentage(Ambient(_ctx(), None, None), ev)

    for p in (root, joined):
        assert (p.correlation.request_id, p.correlation.operation_id) == ("r", "o")
        assert p.correlation.attempt_id == "a"

    # The root path must stay a ROOT while carrying those hints — the mutation
    # that reads them as a parent lookup passes every other assertion here.
    assert root.parent_span_id is None
    assert root.joined is False


# --- child_of: the Unit path ------------------------------------------------


def test_child_of_parents_to_the_supplied_anchor_not_the_ambient_scope():
    anchor = _ctx()
    with _hub.new_scope() as scope:
        scope.active_span_context = _ctx()

        p = child_of(anchor, Evidence(ParentSource.UNIT_ACTIVE))

    assert p.trace_id == anchor.trace_id
    assert p.parent_span_id == anchor.span_id


# --- limitation markers -----------------------------------------------------


def test_with_limitation_is_additive_idempotent_and_ordered():
    p = resolve_parentage(Ambient(_ctx(), None, None), AMBIENT)

    marked = p.with_limitation(Limitation.UNIT_INFERRED_SOLE)
    again = marked.with_limitation(Limitation.UNIT_INFERRED_SOLE)
    both = marked.with_limitation(Limitation.PARENT_UNRESOLVED)

    assert p.limitations == ()  # the original is untouched
    assert marked.limitations == (Limitation.UNIT_INFERRED_SOLE,)
    assert again is marked
    assert both.limitations == (Limitation.UNIT_INFERRED_SOLE, Limitation.PARENT_UNRESOLVED)


@pytest.mark.parametrize(
    ("source", "marker"),
    [
        (ParentSource.UNIT_SOLE, Limitation.UNIT_INFERRED_SOLE),
        (ParentSource.UNRESOLVED, Limitation.PARENT_UNRESOLVED),
    ],
)
def test_an_interpreted_edge_marks_itself_without_being_asked(source, marker):
    """I4's second half, as a mechanism rather than caller discipline.

    A sub-1.0 confidence alone is not enough: nothing on the shipping OTLP path
    carries `CorrelationInfo` at all (`bindings/python/src/codec.rs` `span_to_otlp`
    does not map it), so an unmarked guess is structurally indistinguishable from
    a 1.0 contextvar edge by the time it reaches a dashboard. The marker is what
    the dashboard actually renders, so it cannot depend on a caller remembering
    `.with_limitation()`.
    """
    for p in (
        resolve_parentage(EMPTY_AMBIENT, Evidence(source)),
        child_of(_ctx(), Evidence(source)),
    ):
        assert marker in p.limitations
        assert p.correlation.confidence < 1.0

    # ...and it is idempotent with the design's own explicit call (§5.5).
    p = resolve_parentage(EMPTY_AMBIENT, Evidence(source))
    assert p.with_limitation(marker) is p


@pytest.mark.parametrize("source", list(ParentSource))
def test_an_interpreted_source_never_ships_an_empty_limitations_tuple(source):
    """The conjunction itself: interpretation implies BOTH halves, on every path."""
    interpreted = {ParentSource.UNIT_SOLE, ParentSource.UNRESOLVED}

    for p in (
        resolve_parentage(EMPTY_AMBIENT, Evidence(source)),
        child_of(_ctx(), Evidence(source)),
    ):
        if source in interpreted:
            assert p.limitations != (), f"{source} is a guess that does not report itself"
            assert p.correlation.confidence < 1.0
        else:
            # An exact match — including UNIT_ALIAS, whose 0.9 says the alias
            # table can be stale, not that the edge was guessed.
            assert p.limitations == ()


# --- latch_ambient ----------------------------------------------------------


def test_latch_ambient_reads_the_current_scope():
    parent = _ctx()
    conv = ConversationContext(conversation_id="c-2")
    with _hub.new_scope() as scope:
        scope.active_span_context = parent
        scope.conversation = conv
        scope.tracestate = "vendor=2"

        ambient = latch_ambient()

    assert ambient.span_context == parent
    assert ambient.conversation is conv
    assert ambient.tracestate == "vendor=2"


def test_latch_ambient_outside_any_span_is_empty_but_usable():
    ambient = latch_ambient()

    assert ambient.span_context is None
    assert resolve_parentage(ambient).correlation.strategy == ParentSource.TRACE_ROOT.value


# --- guard: swallow, but never in silence (I6) ------------------------------


def test_guard_swallows_exceptions_and_counts_them():
    with guard("test.site"):
        raise ValueError("boom")

    assert counters.get("test.site") == 1
    assert counters.total() == 1
    assert counters.snapshot() == {"test.site": 1}


def test_guard_lets_control_flow_exceptions_through_uncounted():
    class Bubble(Exception):
        pass

    with pytest.raises(Bubble):
        with guard("test.site", ignored=(Bubble,)):
            raise Bubble

    assert counters.get("test.site") == 0


def test_guard_does_not_catch_base_exceptions():
    with pytest.raises(KeyboardInterrupt):
        with guard("test.site"):
            raise KeyboardInterrupt

    assert counters.get("test.site") == 0


def test_guard_logs_a_traceback_only_in_debug(capsys):
    with guard("test.quiet"):
        raise ValueError("hidden")
    assert capsys.readouterr().err == ""

    with guard("test.loud", debug=True):
        raise ValueError("shown")
    err = capsys.readouterr().err
    assert "test.loud" in err
    assert "ValueError" in err
    assert "Traceback" in err


def test_guard_does_nothing_when_the_block_succeeds():
    result = []
    with guard("test.site"):
        result.append(1)

    assert result == [1]
    assert counters.total() == 0


# --- the reporting path may not itself become a throw (I6) ------------------


def test_guard_survives_an_exception_whose_repr_raises():
    """`repr(exc)`/`str(exc)` are HOST code, and wardex is here because the host
    already broke. Lazily-formatted exceptions — an ORM error over a detached
    session, an httpx `ResponseNotRead`, a pydantic error over a torn-down model
    — raise from inside the log line, replacing the exception guard() swallowed
    with a NEW one thrown into the application."""

    class Nasty(Exception):
        def __repr__(self):
            raise RuntimeError("repr blew up")

        def __str__(self):
            raise RuntimeError("str blew up")

    with guard("test.repr", debug=True):
        raise Nasty

    assert counters.get("test.repr") == 1
    assert counters.get(LOG_FAILED) == 0  # degraded gracefully, did not give up


def test_guard_survives_an_unwritable_stderr(monkeypatch):
    """A daemonized worker that closed fd 2, `pythonw`, a closed redirect target,
    or a pytest capture teardown racing a background thread."""

    class Closed:
        def write(self, _s):
            raise ValueError("I/O operation on closed file")

        def flush(self):
            raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(sys, "stderr", Closed())

    with guard("test.stderr", debug=True):
        raise ValueError("boom")

    assert counters.get("test.stderr") == 1
    assert counters.get(LOG_FAILED) == 1  # the failure to report is itself reported


def test_guard_debug_output_names_the_type_without_calling_its_repr(capsys):
    with guard("test.header", debug=True):
        raise ValueError("shown")

    err = capsys.readouterr().err
    assert "builtins.ValueError" in err
    assert "Traceback" in err


def test_guard_works_as_a_decorator():
    @guard("test.decorated")
    def boom():
        raise ValueError("x")

    boom()

    assert counters.get("test.decorated") == 1
