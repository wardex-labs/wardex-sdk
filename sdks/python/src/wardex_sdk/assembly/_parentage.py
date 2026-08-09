"""The single source of span parentage — design §4.1, invariants I1-I4.

Nothing else in the SDK may compute a trace_id or a parent_span_id. The rule:
a parent is a SpanContext that came out of the wardex scope -- either read live
(`latch_ambient`) or held by a Unit whose own context was produced by this
module. Framework identifiers select which unit; they never become a span id.

That last sentence is the product claim in code form. A run id, a session id, a
checkpoint id can only ever reach `Evidence` as a *hint* that is recorded on
the span; there is deliberately no function here that turns one into a
`SpanContext`. Replace every framework id in a workload with a fresh UUID and
the tree must come out the same shape (conformance C-3).

All six parentage sites — `_seam._emit_span`, `_seam._emit_ws`,
`_mcp_stdio._build_mcp_span`, the Agent SDK assembler, `_tracing._begin` and
`capture_state_snapshot` — get their edge from `resolve_parentage()` or
`child_of()`, and nowhere else (I1). There is no second answer left
in the SDK: `tests/test_import_graph.py` asserts a single `TraceId.generate()`
call site as a hard rule rather than a budget.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum

from .. import _hub
from .._types import ConversationContext, CorrelationInfo, SpanContext, SpanId, TraceId
from ._integrity import Limitation


class ParentSource(Enum):
    """CLOSED. The only legal values of CorrelationInfo.strategy.

    This axis answers exactly one question — *how was the parent derived* — and
    nothing else. "Which source observed the event" is already carried by
    `InternalSpan.capture_sources`; mixing the two into one string is the drift
    that produced today's `adapter_hook` / `adapter_stream` values (I9).
    """

    CONTEXTVAR = "contextvar"  # ambient scope, latched in-task by the initiator
    HEADER = "header"  # parent joined from a W3C traceparent (is_remote)
    UNIT_ACTIVE = "unit_active"  # a Unit was ambient via activate()/pin
    UNIT_ALIAS = "unit_alias"  # a Unit resolved by an exact framework-id alias
    UNIT_SOLE = "unit_sole"  # exactly one live unit; heuristic, last resort
    TRACE_ROOT = "trace_root"  # deliberately started a new trace, no parent existed
    UNRESOLVED = "unresolved"  # a parent was expected and not found


_CONFIDENCE: dict[ParentSource, float] = {
    ParentSource.CONTEXTVAR: 1.0,
    ParentSource.UNIT_ACTIVE: 1.0,
    ParentSource.HEADER: 1.0,
    ParentSource.UNIT_ALIAS: 0.9,
    ParentSource.UNIT_SOLE: 0.5,
    ParentSource.TRACE_ROOT: 1.0,
    ParentSource.UNRESOLVED: 0.0,
}

# I4's second half, as a mechanism rather than caller discipline. A source whose
# DEFINITION is interpretation -- "one unit was live so it must be the parent",
# "a parent was expected and never found" -- carries its marker automatically,
# so there is no path that produces an interpreted edge with an empty
# `limitations` tuple and only a float to give it away.
#
# `UNIT_ALIAS` is deliberately absent. I4 governs edges decided "by
# interpretation rather than exact match", and an alias IS an exact match: the
# framework id selected a unit whose context this module produced. Its 0.9 says
# the alias table can be stale, not that the edge was guessed. `with_limitation`
# stays for the CONDITIONAL markers a caller alone can observe -- design §5.5
# attaches `CORRELATION_CONFLICT` that way, when an alias and the live context
# disagree about which trace they are in.
_MARKER: dict[ParentSource, Limitation | None] = {
    ParentSource.CONTEXTVAR: None,
    ParentSource.UNIT_ACTIVE: None,
    ParentSource.HEADER: None,
    ParentSource.UNIT_ALIAS: None,
    ParentSource.UNIT_SOLE: Limitation.UNIT_INFERRED_SOLE,
    ParentSource.TRACE_ROOT: None,
    ParentSource.UNRESOLVED: Limitation.PARENT_UNRESOLVED,
}


@dataclass(frozen=True, slots=True)
class Evidence:
    """How the caller knows this parentage. Supplied, never guessed."""

    source: ParentSource
    request_id: str | None = None  # framework id -- HINT ONLY
    operation_id: str | None = None  # framework run/session id -- HINT ONLY
    attempt_id: str | None = None
    confidence: float | None = None  # None -> table default; may only LOWER it


AMBIENT = Evidence(ParentSource.CONTEXTVAR)


def _confidence_for(source: ParentSource, evidence: Evidence) -> float:
    """Table default, which a caller may lower but never raise. Always 0.0~1.0.

    The upper clamp is the enforcement of "may only LOWER it" on `Evidence`: a
    caller that knows its match is shakier than the strategy implies (a unit
    alias it only half trusts) says so, but no caller can promote a 0.5
    heuristic into a 1.0 fact and hide a guess from I4.

    The lower clamp and the NaN rejection are the enforcement of the field's
    declared domain (`span.proto` `CorrelationInfo.confidence`, "0.0~1.0").
    Nothing between here and the wire validates that float, so an honest decay
    expression -- `confidence=1.0 - staleness_s / 60.0` on a stale alias -- would
    otherwise put a negative number on a dashboard's confidence bar the moment
    the alias passed a minute old. NaN is the other end of the same accident
    (`matched / total` with `total == 0`); since `nan < x` is False, `min()`
    would let it through as the FULL default -- the one direction the clamp
    exists to prevent -- so it is treated as "the caller does not know", which is
    what `confidence=None` already means.

    Raising here would be the wrong answer even though the input is wrong:
    `Evidence` is built on the adapter path, and I6 forbids throwing into the
    host from there.
    """
    default = _CONFIDENCE[source]
    supplied = evidence.confidence
    if supplied is None or supplied != supplied:  # None, or NaN
        return default
    return max(0.0, min(default, supplied))


def _markers_for(source: ParentSource) -> tuple[Limitation, ...]:
    marker = _MARKER[source]
    return () if marker is None else (marker,)


@dataclass(frozen=True, slots=True)
class Parentage:
    """A resolved edge: which trace this span belongs to and what it hangs off.

    `correlation` is never None — "started a new trace" and "expected a parent
    and did not find one" must stay distinguishable downstream (I4).
    """

    trace_id: TraceId
    parent_span_id: SpanId | None
    trace_flags: int
    tracestate: str | None
    conversation: ConversationContext | None
    correlation: CorrelationInfo  # NEVER None
    joined: bool
    limitations: tuple[Limitation, ...] = ()

    def child_context(self) -> SpanContext:
        """ALLOCATE a new span context under this parentage. Carries trace_flags.

        This is a factory, not an accessor: every call mints a fresh span id and
        therefore names a DIFFERENT span. That is deliberate -- one latched
        parent serves many sibling transactions on the byte seam -- but it means
        a caller that needs the context both to emit a span and to anchor that
        span's children must call this ONCE and hold the result:

            ctx = parentage.child_context()          # right
            unit = registry.open(kind, anchor=ctx)
            sink.emit(InternalSpan(context=ctx, ...))

        Calling it twice there would anchor the children to a span id that is
        never emitted, and nothing downstream could detect it: same trace id,
        confidence 1.0, no marker -- an orphaned subtree that looks healthy.
        """
        return SpanContext(
            trace_id=self.trace_id,
            span_id=SpanId.generate(),
            trace_flags=self.trace_flags,
            is_remote=False,
        )

    def with_limitation(self, marker: Limitation) -> Parentage:
        """Return a copy carrying `marker`. Idempotent, order-preserving."""
        if marker in self.limitations:
            return self
        return replace(self, limitations=self.limitations + (marker,))


@dataclass(frozen=True, slots=True)
class Ambient:
    """A snapshot of the causally-relevant scope, taken on the originating task."""

    span_context: SpanContext | None
    conversation: ConversationContext | None
    tracestate: str | None

    @property
    def is_local(self) -> bool:
        return self.span_context is not None and not self.span_context.is_remote


EMPTY_AMBIENT = Ambient(None, None, None)


def latch_ambient() -> Ambient:
    """Read the active parent NOW, on the caller's task/thread.

    Call this at the moment the outbound work is ISSUED (socket write, stdin
    send, framework entry) -- never on the response/completion path, by which
    time the ambient context has moved on. Naming this is the point: reading
    the context on the response path is the single most likely way a new
    adapter silently breaks the tree.
    """
    scope = _hub.get_current_scope()
    return Ambient(
        span_context=scope.active_span_context,
        conversation=scope.conversation,
        tracestate=scope.tracestate,
    )


def resolve_parentage(ambient: Ambient, evidence: Evidence = AMBIENT) -> Parentage:
    """Turn a latched scope plus the caller's evidence into one resolved edge.

    The only function in the SDK that decides a trace_id or a parent_span_id.
    """
    parent = ambient.span_context
    if parent is None:
        src = (
            ParentSource.TRACE_ROOT
            if evidence.source is ParentSource.CONTEXTVAR
            else evidence.source
        )
        return Parentage(
            trace_id=TraceId.generate(),  # THE ONLY CALL SITE IN THE SDK
            parent_span_id=None,
            # SAMPLED. wardex does not head-sample -- retention is decided later
            # by the RetentionClassifier -- so a trace wardex ORIGINATES is by
            # definition sampled, which is what `_w3c.format_traceparent` has
            # always emitted (its hardcoded `-01`). Setting it here is what let
            # that hardcode go: once origination says 1, a `0` on the
            # wire unambiguously means "an upstream told us -00" and we may
            # honour it. Shipping 0 here instead would make every wardex-rooted
            # trace emit `-00`, and every downstream OTel service on the default
            # ParentBased(ALWAYS_ON) sampler would stop recording -- silently.
            # NOTE: design §4.1 (line 332) writes 0 here; the V9 resolution
            # (line 1577) says wardex-originated traces emit 01. The doc
            # contradicts itself and 1 is the resolution it reached. Correct the
            # snippet, not this line.
            trace_flags=1,
            tracestate=ambient.tracestate,
            conversation=ambient.conversation,
            joined=False,
            correlation=CorrelationInfo(
                # The MEMBER, not `src.value`. Unwrapping here was left over
                # from when the field was a free-form string, and it made the
                # closed retype a lie on the one path that produces almost
                # every span: `isinstance(strategy, ParentSource)` was False
                # everywhere, so the closed vocabulary bought nothing and an
                # assertion written in the member form failed while the old
                # string form kept passing.
                strategy=src,
                confidence=_confidence_for(src, evidence),
                request_id=evidence.request_id,
                operation_id=evidence.operation_id,
                attempt_id=evidence.attempt_id,
            ),
            limitations=_markers_for(src),
        )

    src = evidence.source
    if src is ParentSource.CONTEXTVAR and parent.is_remote:
        # A joined W3C parent is not a ContextVar parent. Recording it as one is
        # the drift that left `header` unused in the declared vocabulary.
        src = ParentSource.HEADER
    return Parentage(
        trace_id=parent.trace_id,
        parent_span_id=parent.span_id,
        trace_flags=parent.trace_flags,  # fixes the universal trace_flags drop
        tracestate=ambient.tracestate,
        conversation=ambient.conversation,
        joined=True,
        correlation=CorrelationInfo(
            strategy=src,  # the member — see the no-parent branch above
            confidence=_confidence_for(src, evidence),
            active_span_id_at_capture=parent.span_id,
            request_id=evidence.request_id,
            operation_id=evidence.operation_id,
            attempt_id=evidence.attempt_id,
        ),
        limitations=_markers_for(src),
    )


def child_of(
    anchor: SpanContext,
    evidence: Evidence,
    *,
    conversation: ConversationContext | None = None,
    tracestate: str | None = None,
) -> Parentage:
    """Parent explicitly to a scope-derived anchor (the Unit path).

    `anchor` must be a context this module produced and a Unit is holding. It is
    never reconstructed from a framework identifier — that is I2, and there is
    no API here that would let a caller try.
    """
    return resolve_parentage(Ambient(anchor, conversation, tracestate), evidence)


# -- work issued inside a span wardex FAILED to open -----------------------
#
# A ContextVar and not a flag on any object, because the thing that has to know
# is a byte seam three layers away that shares nothing with the site that
# failed except the TASK the host's code is running on. That is the same
# carrier the whole tree is built from, so it is the one thing the two ends
# reliably agree about.

_degraded_run: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "wardex_degraded_run", default=False
)

#: What an edge is worth when the span above it is one wardex cannot stand
#: behind: one it FAILED to open, or one it opened and then CLOSED while an
#: activation fork went on handing the finished span out (design §10.3).
#: `UNRESOLVED`, so `_MARKER` attaches `PARENT_UNRESOLVED` on its own — a parent
#: was expected here, and the expectation is exactly what makes this not a
#: trace root.
_ORPHANED_BY_WARDEX = Evidence(ParentSource.UNRESOLVED)


@contextmanager
def degraded_run() -> Iterator[None]:
    """Mark this task as running inside a span wardex FAILED to open.

    Set for the duration of the block the host was given anyway. It says one
    thing and only one: *the absence of a parent here is wardex's doing, not the
    host's*. Two consequences follow, and both are losses this closes rather
    than behaviour it adds.

    First, the CAPTURE GATE. Under `capture_mode=AGENT` — the default — traffic
    is captured when a local wardex span was ambient at the moment the work was
    issued. A run entry that failed to open leaves nothing ambient, so every
    HTTP request and every tool call inside the host's block is dropped at the
    seam, with no counter and no marker: one wardex bug at the top turns into
    total silence underneath it, and the user sees a run that looks like it
    never happened rather than one wardex could not follow.

    Second, the EDGE. Captured anyway, such a span would resolve as a trace root
    at confidence 1.0 — the shattered-run shape `Placement` exists to prevent,
    arriving through a different door. `resolve_observed` is what stops that.

    Deliberately not reset by the seam or read anywhere else. A caller that
    yields a degraded scope owns the block, so the flag's lifetime is the
    block's, and a `ContextVar` restores itself on every task that forked from
    this one without anyone remembering to.
    """
    token = _degraded_run.set(True)
    try:
        yield
    finally:
        _degraded_run.reset(token)


def in_degraded_run() -> bool:
    """Is this task inside a span wardex failed to open? See `degraded_run`."""
    return _degraded_run.get()


def resolve_observed(
    ambient: Ambient, *, parent_closed: bool = False, parent_evicted: bool = False
) -> Parentage:
    """`resolve_parentage` for an OBSERVED byte seam, which has three extra cases.

    A seam latches whatever the host's carrier held. With nothing there the edge
    is an honest TRACE ROOT: the host issued this request outside any agent
    work, and under `capture_mode=AGENT` the gate would have dropped it long
    before this line. The one way it arrives here anyway is that wardex failed
    to open the span that would have been above it — and that is not a trace
    root. It is a span whose parent exists and is missing, which is what
    `PARENT_UNRESOLVED` means, with `INSTRUMENTATION_DEGRADED` saying whose
    fault that is: not the host's threading, not a run entry the adapter forgot
    to wrap, but wardex.

    The distinction is the whole value of capturing it at all. Shipped as a
    trace root the span is indistinguishable from a legitimate one and quietly
    inflates the trace count; shipped like this it is one row a consumer can
    filter, count, and file a bug about.

    `parent_closed` is the second case and the same argument one step earlier:
    the seam DID latch a local parent, and that parent's unit had ALREADY closed
    — an activation fork the adapter could not take down (design §10.3). This
    module cannot see a unit and must not learn to (`_units` imports this one),
    so the caller latches the fact where it latches the parent
    (`assembly._units.parent_is_closed_unit`) and declares it here, which is the
    shape `degraded` already has.

    Refused to `EMPTY_AMBIENT`, exactly as `UnitRegistry.open()` and
    `UnitRegistry.resolve()` refuse the same carrier: the conversation and the
    tracestate came off the dead unit's fork too, and keeping either would stamp
    a finished run's identity on unrelated later work — the same adoption in a
    different field. The edge is `UNRESOLVED` and NOT `TRACE_ROOT`, because a
    parent was expected here — one was latched — so `PARENT_UNRESOLVED` is the
    literal truth and `_MARKER` attaches it without this function naming a
    marker it is not on the census for. It is not `INSTRUMENTATION_DEGRADED`
    either: nothing failed to open. What it must NOT do is keep the parent — an
    already-shipped span adopting later, unrelated traffic at confidence 1.0
    with no marker is the one shape no consumer can detect downstream.

    `parent_evicted` is the third and it is the FIRST case above, reached by a
    different road: the seam latched a parent and then threw it away itself, to
    stay inside a bound of its own (`interceptors/_trackers.py`, the h2 stream
    latch). So it earns the same pair of markers as a failed open, for the same
    reason — the parent existed, wardex lost it, and the alternative is a span
    that ships as a legitimate trace root at confidence 1.0. It reaches the same
    pair of markers because there is nothing to tell apart downstream: "wardex
    dropped what belongs on this span" is the whole content of both, and the
    knob a user reaches for is different only in that one of them has a knob at
    all.

    It is not an over-reach on a stream that had no parent to lose. The dropped
    record held the request instant too, so the span's own start and duration
    are invented on this path whether or not a parent was in it — the pair says
    "wardex lost bookkeeping that belongs here", which is true in both cases,
    and the alternative is a span that looks measured and rooted and is neither.

    Refused to `EMPTY_AMBIENT` for the same reason `parent_closed` is, and in
    code rather than as a rule for callers: the eviction IS the statement that
    nothing usable was resolved, so an ambient arriving beside it can only be
    one the caller could not vet. Left to flow through, a non-empty one takes
    `resolve_parentage`'s JOIN branch and ships `joined=True` with a parent
    while `strategy=UNRESOLVED` claims there is none — a span simultaneously
    adopting and disowning, which is the exact undetectable shape this whole
    module exists to prevent. Nothing is lost by the substitution: the byte
    seam's carrier on this path holds no conversation and no tracestate either.
    """
    if parent_closed:
        return resolve_parentage(EMPTY_AMBIENT, _ORPHANED_BY_WARDEX)
    if parent_evicted:
        return resolve_parentage(EMPTY_AMBIENT, _ORPHANED_BY_WARDEX).with_limitation(
            Limitation.INSTRUMENTATION_DEGRADED
        )
    if ambient.span_context is None and in_degraded_run():
        return resolve_parentage(ambient, _ORPHANED_BY_WARDEX).with_limitation(
            Limitation.INSTRUMENTATION_DEGRADED
        )
    return resolve_parentage(ambient)
