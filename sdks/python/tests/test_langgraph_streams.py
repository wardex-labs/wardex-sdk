"""LangGraph adapter — the leftover scope, teardown mid-stream, and the tool seam.

Three groups that share one mechanism and one file.

**The leftover scope.** `ctx.enter` inside a generator installs the run on the
carrier that pumps the FIRST `next()` and can only take it down when the
generator is FINALIZED, on the carrier that finalizes it. Every case below
asserts the EDGES OF LATER TRAFFIC rather than a run-span count, because the
damage a leftover does is not that one span is missing — it is that spans which
have nothing to do with the run get shipped as its children, and a backend
cannot tell them from real ones.

**Three of the four leftover shapes are DEFECTIVE today, and every assertion
about them says so with a `§10.3` comment.** A separate fix will make the
registry refuse an ambient span context whose unit has CLOSED; until it lands,
a finished run keeps adopting later traffic at `contextvar`/1.0 with no marker.
Each affected assertion carries the exact edge it becomes afterwards —
measured, not predicted: the fix was briefly in this working tree and every
post-fix value below was read off a real run. Flipping this file when the fix
lands is a one-line edit per assertion, and until then it RUNS.

The fourth shape is the one that fix cannot help with, and its test says so: a
retained iterator's unit is genuinely LIVE, so nothing dead exists to refuse
and the counter gap is the only record there will ever be.

**The tool seam.** `ToolNode._run_one` is one tool CALL — below the fan-out and
above `wrap_tool_call`'s retries. The two assertions that pin that choice are
here: a retrying wrapper yields exactly one span while the body runs three
times, and the tool spans come out identical across all four ways langgraph and
langchain can build a tool-calling agent, including the two where the NODE seam
sees one task for N calls.

**Housekeeping that is load-bearing.** A test that abandons a stream must
FINALIZE it before it returns. `_clean_scope` resets the ambient unit but not
the hub scope, so an abandoned generator's fork otherwise stands for the rest
of the pytest process and every later test inherits it — which is the same
adoption these tests are about, aimed at the suite. Measured: without
`carrier_is_clean()` here, four unrelated cases in this file fail.
"""

from __future__ import annotations

import asyncio
import contextvars
import gc
import threading
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from test_langgraph_adapter import (  # noqa: F401
    Installed,
    _clean_scope,
    adapter_counters,
    assembly_counters,
    call,
    chain,
    edge_of,
    extra_of,
    leaf_span,
    named,
    parent_name,
    pure_add,
    runs,
    steps,
    tool_graph,
    tools,
    traces,
)
from wardex_sdk.assembly import Limitation, ParentSource, counters, latch_ambient

#: TODAY'S edge for any site issued on a carrier whose run has already closed
#: and shipped: a full-confidence child of a span that has ENDED, with nothing
#: on the wire to say so. Spelled once because four cases assert it and because
#: it is the single value §10.3 changes — measured post-fix it becomes
#: `ORPHANED_ON_A_CORPSE` below at every one of those sites.
ADOPTED_BY_A_CORPSE = (ParentSource.CONTEXTVAR, 1.0, ())

#: §10.3's replacement, measured while the fix was in this tree: no parent, no
#: confidence, and TWO markers — one saying a parent was expected and not found,
#: one saying what was in front of us was a corpse. Referenced by the comments
#: rather than asserted, so that flipping this file is a name change per site.
ORPHANED_ON_A_CORPSE = (
    ParentSource.UNRESOLVED,
    0.0,
    (Limitation.PARENT_UNRESOLVED, Limitation.CORRELATION_CONFLICT),
)

#: A NESTED site with nothing at all on the carrier — no corpse, so no conflict.
#: Unchanged by §10.3, which is what makes it the control for the two above.
CLEAN_ORPHAN = (ParentSource.UNRESOLVED, 0.0, (Limitation.PARENT_UNRESOLVED,))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class Carrier:
    """A thread that OUTLIVES each job, so its ContextVar context can be asked.

    A `Thread(target=...)` that returns takes its context with it, which would
    make every leftover-scope question unanswerable by construction: the leak
    under test lives on the carrier, so the carrier has to still be there to be
    interrogated afterwards.
    """

    def __init__(self) -> None:
        self._jobs: list = []
        self._results: dict = {}
        self._go = threading.Event()
        self._done = threading.Event()
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while True:
            self._go.wait()
            self._go.clear()
            with self._lock:
                job = self._jobs.pop(0) if self._jobs else None
            if job is None:
                return
            key, fn = job
            try:
                self._results[key] = fn()
            except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
                self._results[key] = exc
            self._done.set()

    def run(self, key: str, fn) -> Any:
        with self._lock:
            self._jobs.append((key, fn))
        self._done.clear()
        self._go.set()
        assert self._done.wait(10), "the carrier thread never finished its job"
        out = self._results[key]
        if isinstance(out, BaseException):
            raise out
        return out

    def shutdown(self) -> None:
        self._go.set()
        self.thread.join(5)


def carrier_is_clean() -> None:
    """Assert this thread holds no fork, after a test has dropped its generator.

    Called from the `finally` of every case that abandons a stream. The drop has
    to happen in the TEST's frame (it holds the only reference) and the collect
    has to happen on the carrier that pumped, or the fork is still standing when
    the next test starts.
    """
    gc.collect()
    assert latch_ambient().span_context is None, (
        "an abandoned stream left its fork on this thread; every later test would inherit it"
    )


def refused_as_a_corpse(span, corpse) -> None:
    """The edge at a site whose carrier still holds a CLOSED unit's fork.

    This is the shape §10.3 landed. Before it, every one of these sites read
    `parent=<the corpse> / (CONTEXTVAR, 1.0, ())` — a full-confidence child of a
    span that had already ENDED, with nothing on the wire to say so, which is
    the one failure mode this SDK exists to make impossible. Now the leftover is
    refused: no parent, no confidence, and two markers saying both what was
    expected and what was actually in front of us.

    `corpse` is still taken and still asserted against, because "not the corpse"
    is the whole claim — a site that merely lost its parent for some other
    reason would satisfy a bare `parent_span_id is None`.
    """
    assert span.parent_span_id != corpse, (
        "the shipped, closed run adopted this span — the §10.3 refusal did not fire"
    )
    assert (span.parent_span_id, edge_of(span)) == (None, ORPHANED_ON_A_CORPSE)


def refused_root_after_a_corpse(span, corpse) -> None:
    """A ROOT site whose carrier held a corpse: its own trace, and MARKED.

    The counterpart of `refused_as_a_corpse` for a site that may legitimately
    begin a trace. Refusing the leftover does not orphan it — a run entry is
    allowed to be a root — so what is left is the disagreement itself, and
    `CORRELATION_CONFLICT` is where it goes. Asserting the marker rather than
    just the trace count is the point: an unmarked trace root here is
    byte-identical to a genuine independent run, which is the exact ambiguity
    §10.3 exists to remove.
    """
    assert span.parent_span_id != corpse, (
        "the shipped, closed run adopted this whole run — the §10.3 refusal did not fire"
    )
    assert edge_of(span) == (
        ParentSource.TRACE_ROOT,
        1.0,
        (Limitation.CORRELATION_CONFLICT,),
    )


def one(spans, name: str):
    """Exactly one span with this name, or the assertion says which it was."""
    found = named(spans, name)
    assert len(found) == 1, f"expected one {name!r}, got {[s.name for s in found]}"
    return found[0]


def tool_signature(spans) -> list[tuple]:
    """Every tool span as a comparable tuple, with the span ids taken out.

    Span ids differ between two runs of the same workload by design, so the
    equality this file needs — "the four ways to build a tool-calling agent all
    produce the SAME tool spans" — has to be spelled over everything except
    them, and must include the PARENT'S NAME or a tree collapse compares equal.

    Sorted by `repr`, not by the tuples themselves. `StatusCode` is a plain
    `Enum` and does not define `<`, so a natural sort raises `TypeError` the
    moment two tool spans agree on name, call id and description — two attempts
    at one call, or a fan-out of identical calls. That is a crash in the
    comparison helper rather than a failure of the claim, and it would land on
    whoever next points this at a workload with repeats.
    """
    return sorted(
        (
            (
                s.name,
                s.tool.name if s.tool else None,
                s.tool.call_id if s.tool else None,
                s.tool.description if s.tool else None,
                s.status,
                s.output_data,
                parent_name(spans, s),
                edge_of(s),
            )
            for s in tools(spans)
        ),
        key=repr,
    )


class StubModel(BaseChatModel):
    """A chat model that asks for tools once, then answers.

    STATELESS — it reads the transcript instead of counting its own calls,
    because `create_agent`, `create_react_agent` and a hand-built graph invoke
    it a different number of times, and a call counter would make the workloads
    incomparable, which is exactly what the tool-span equality tests.
    """

    calls: list = []

    @property
    def _llm_type(self) -> str:
        return "stub"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        answered = any(isinstance(m, ToolMessage) for m in messages)
        msg = (
            AIMessage(content="done")
            if answered
            else AIMessage(content="", tool_calls=list(self.calls))
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # noqa: A002 — framework's name
        return self


def two_tool_graph(ctx, *, name: str):
    """One pure-Python tool and one that issues a wire call, in one message.

    The second tool stands in for a tool that makes an HTTP request: §9.1
    forbids an in-process server, and `leaf_span` opens its unit through the
    same `ctx.enter` the byte seam's parent resolution goes through.
    """

    @tool
    def fetch(q: str) -> str:
        """Fetch a thing over the wire."""
        leaf_span(ctx, "leaf-fetch")
        return "fetched:" + q

    return tool_graph(
        [call("pure_add", {"a": 1, "b": 2}, "c1"), call("fetch", {"q": "x"}, "c2")],
        [pure_add, fetch],
        name=name,
    )


# --------------------------------------------------------------------------
# 1. the leftover scope, four shapes
# --------------------------------------------------------------------------


def test_a_retained_stream_iterator_adopts_every_later_span_on_that_carrier():
    """(a) The one shape §10.3 cannot help with, because nothing here is closed.

    The host called `stream()`, pumped it once and kept the iterator. Its
    SESSION unit is therefore standing, ambient, and will never ship — and
    every later span on that carrier, including an entire unrelated graph run,
    is parented to a span id that is NOT IN THE EXPORTED SET. Downstream that is
    a subtree hanging off a parent nobody ever sent: `unit_active` at 1.0, no
    marker, and one trace where there should be two.

    No marker is possible here and that is not an oversight: a marker can only
    be attached to a span, this run's span never ships, and the unit is
    genuinely alive — so a refusal keyed on "the unit has closed" has nothing to
    refuse. The gap between `active.pregel.stream` and `stream_finalized` IS the
    assertion, and it is the only handle an operator will ever have on a stream
    the host never finished.
    """
    live = Installed()
    it = None
    try:
        it = chain(live.ctx, 3, name="Retained", leaves=False).stream({"trail": []})
        next(it)
        leaf_span(live.ctx, "after")
        chain(live.ctx, 1, name="Unrelated", leaves=False).invoke({"trail": []})

        exported = {s.context.span_id for s in live.spans}
        leaf = one(live.spans, "execute_tool after")
        later_run = one(live.spans, "invoke_workflow Unrelated")
        assert leaf.parent_span_id is not None
        assert leaf.parent_span_id not in exported, (
            "the leaf hangs off the retained run's span, which never ships"
        )
        assert later_run.parent_span_id == leaf.parent_span_id, (
            "a whole unrelated run was adopted by the retained one"
        )
        assert edge_of(leaf) == (ParentSource.UNIT_ACTIVE, 1.0, ())
        assert edge_of(later_run) == (ParentSource.UNIT_ACTIVE, 1.0, ())
        assert len(traces(live.spans)) == 1, "two runs, one trace"
        assert runs(live.spans) == [later_run], "the retained run emits nothing at all"

        snap = adapter_counters()
        assert snap["adapters.langgraph.active.pregel.stream"] == 2
        assert snap["adapters.langgraph.stream_finalized"] == 1
        assert assembly_counters() == {}, (
            "nothing is stale and nothing is refused — the unit is LIVE, which "
            "is why only the counter gap can report this shape"
        )
    finally:
        it = None
        carrier_is_clean()
        live.teardown()


def test_a_stream_finalized_off_its_carrier_leaves_a_corpse_that_is_refused():
    """(b) Two counters fire, two ContextVar resets fail, and the carrier keeps a corpse.

    The generator was pumped on thread T and finalized on the main thread, so
    `_Carrier.remove` runs where the Token was not created: BOTH halves raise
    `ValueError` and both are counted, which is why `activate_exit` is 2 and not
    1. What is left standing on T is an `activate()` fork over a unit that has
    since closed and SHIPPED.

    T's next span is therefore built from a dead unit's own span context, and
    the two halves of the fork disagree about it: the UNIT half notices
    (`ambient_stale`, twice — once for the evidence and once for the holder)
    while the SPAN half is still handed out as a confident parent. The main
    thread, which never held the fork, is untouched — the same site there is a
    plain orphan, which is what proves the difference is the carrier and not the
    placement.
    """
    live = Installed()
    carrier = Carrier()
    try:
        app = chain(live.ctx, 3, name="Foreign", leaves=False)
        held: dict = {}

        def pump() -> None:
            it = app.stream({"trail": []})
            next(it)
            held["it"] = it  # the only reference, and it is not on T's stack

        carrier.run("pump", pump)
        assert "adapters.langgraph.stream_finalized" not in adapter_counters()

        del held["it"]
        gc.collect()

        snap = adapter_counters()
        assert snap["adapters.langgraph.stream_finalized"] == 1
        assert snap["adapters.langgraph.stream_finalized_off_carrier"] == 1
        assert assembly_counters()["assembly._units.activate_exit"] == 2, (
            "the ambient-unit Token and the scope fork each failed to come down"
        )

        run = one(live.spans, "invoke_workflow Foreign")
        carrier.run("leaf", lambda: leaf_span(live.ctx, "on-T"))
        on_t = one(live.spans, "execute_tool on-T")
        # §10.3: `parent_span_id is None` and `ORPHANED_ON_A_CORPSE`, with
        # `stale_pin_ambient == 1` beside the two `ambient_stale`s. Today the
        # span half of the fork is still handed out.
        refused_as_a_corpse(on_t, run.context.span_id)
        assert on_t.context.trace_id != run.context.trace_id, (
            "refusing the corpse is what puts this work in its own trace; adopting it "
            "is how a finished run keeps collecting traffic from the carrier it leaked onto"
        )
        assert assembly_counters()["assembly._units.stale_activation_ambient"] == 1
        assert assembly_counters()["assembly._units.ambient_stale"] == 2, (
            "the unit half of the fork is asked twice — for the evidence and for the holder"
        )

        leaf_span(live.ctx, "on-main")
        on_main = one(live.spans, "execute_tool on-main")
        assert on_main.parent_span_id is None, "the leak is confined to the carrier that held it"
        assert edge_of(on_main) == CLEAN_ORPHAN
    finally:
        carrier.shutdown()
        live.teardown()


def test_an_abandoned_astream_ships_nothing_until_the_loop_finalizes_asyncgens():
    """(c) The ORDINARY async abandon: dropping the generator is not enough.

    An abandoned async generator is closed by the LOOP, on its own finalizer
    task, so `gc.collect()` alone produces no span and no counter — the run is
    held until `shutdown_asyncgens()` (or `asyncio.run`'s exit) gets to it. A
    loop torn down without that step loses the run span entirely, which is what
    makes `astream_finalized` worth counting.

    The finalizer task is not the task that pumped, so `off_carrier` fires on
    this ordinary shape rather than on a corner case. Nothing leaks to the
    caller: a Task owns its context copy, so the fork died with the task and the
    trace count stays at one — the async twin of (b) with the blast radius the
    loop's own task discipline already bounds.
    """
    live = Installed()
    loop = asyncio.new_event_loop()
    try:
        app = chain(live.ctx, 3, name="AsyncAbandon", leaves=False)
        held: dict = {}

        async def pump() -> None:
            agen = app.astream({"trail": []})
            await agen.__anext__()
            held["agen"] = agen

        loop.run_until_complete(pump())
        del held["agen"]
        gc.collect()
        assert "adapters.langgraph.astream_finalized" not in adapter_counters()
        assert runs(live.spans) == [], "the run span is still being held by the loop"

        loop.run_until_complete(loop.shutdown_asyncgens())
        snap = adapter_counters()
        assert snap["adapters.langgraph.astream_finalized"] == 1
        assert snap["adapters.langgraph.astream_finalized_off_carrier"] == 1
        assert assembly_counters()["assembly._units.activate_exit"] == 2

        run = one(live.spans, "invoke_workflow AsyncAbandon")
        node = one(live.spans, "execute_step n0")
        assert node.parent_span_id == run.context.span_id
        assert run.parent_span_id is None
        assert len(traces(live.spans)) == 1
        assert latch_ambient().span_context is None, "the caller never held the fork"
    finally:
        loop.close()
        live.teardown()


def test_the_clean_control_leaves_nothing_on_any_carrier():
    """(d) Without this, §10.3 could land false positives and every case above would pass.

    Every other case in this group asserts a leftover; this one asserts its
    ABSENCE on the two shapes that must stay clean — runs driven to completion,
    sequentially on one carrier and concurrently on two. Three runs are three
    trace roots, two are two, no `assembly.*` counter fires at all (no
    staleness, no conflict), and a span opened afterwards on any of those
    carriers is a plain orphan rather than a child.

    The orphan is the load-bearing half, and it must carry `PARENT_UNRESOLVED`
    ALONE: a `CORRELATION_CONFLICT` beside it would mean a refusal had started
    firing on carriers that were never dirty.
    """
    live = Installed()
    workers = [Carrier(), Carrier()]
    try:
        app = chain(live.ctx, 2, name="Clean", leaves=False)
        for _ in range(3):
            app.invoke({"trail": []})
        assert len(runs(live.spans)) == 3
        assert len({r.context.trace_id for r in runs(live.spans)}) == 3
        assert all(r.parent_span_id is None for r in runs(live.spans))
        assert assembly_counters() == {}

        leaf_span(live.ctx, "after-sequential")
        after = one(live.spans, "execute_tool after-sequential")
        assert after.parent_span_id is None
        assert edge_of(after) == CLEAN_ORPHAN

        live.client.spans.clear()
        counters.reset()
        for w in workers:
            w.run("start", lambda: app.invoke({"trail": []}))
        assert len(runs(live.spans)) == 2
        assert len({r.context.trace_id for r in runs(live.spans)}) == 2
        assert all(r.parent_span_id is None for r in runs(live.spans))
        assert assembly_counters() == {}

        for i, w in enumerate(workers):
            w.run("leaf", lambda i=i: leaf_span(live.ctx, f"after-w{i}"))
            later = one(live.spans, f"execute_tool after-w{i}")
            assert later.parent_span_id is None, "a completed run must not adopt later work"
            assert edge_of(later) == CLEAN_ORPHAN
    finally:
        for w in workers:
            w.shutdown()
        live.teardown()


# --------------------------------------------------------------------------
# 2. close_units() and uninstall() mid-stream, in both pumping shapes
# --------------------------------------------------------------------------


def test_close_units_mid_stream_orphans_the_stragglers_rather_than_faking_a_parent():
    """The host keeps pumping, and the nodes that follow are children of a corpse.

    `close_units` shipped the run span and left the patches in place, so the
    remaining supersteps still reach the node seam — but the unit they would
    have hung off is closed and its fork cannot be taken down from here.
    `current()` refuses the dead unit, so `Fallback.SOLE_LIVE_RUN` is consulted
    and finds no live SESSION either; what survives is the SCOPE half of the
    same fork, and the edge is built from the dead run's own span context.

    The result on the wire is a parent that ENDED before its children STARTED —
    asserted here as timestamps, because that is the one thing about this shape
    a consumer could in principle detect, and it is not a marker.
    """
    live = Installed()
    try:
        it = chain(live.ctx, 3, name="Straggler", leaves=False).stream({"trail": []})
        next(it)
        live.adapter.close_units(marker=Limitation.UNIT_INTERRUPTED)
        list(it)  # the host keeps pumping; this must not raise into it

        run = one(live.spans, "invoke_workflow Straggler")
        assert Limitation.UNIT_INTERRUPTED in run.capture_integrity.limitations
        for name in ("execute_step n1", "execute_step n2"):
            span = one(live.spans, name)
            # §10.3: `parent_span_id is None`, `ORPHANED_ON_A_CORPSE`, a trace of
            # its own (so `traces == 3`), and `stale_pin_ambient == 2`.
            refused_as_a_corpse(span, run.context.span_id)
            assert span.start_time_ns > run.end_time_ns, (
                "these children start after their parent has already ended"
            )
        assert assembly_counters()["assembly._units.ambient_stale"] == 4  # two per straggler
        assert assembly_counters()["assembly._units.stale_activation_ambient"] == 2
        # Three, not one. The drained run keeps its own; the stragglers no
        # longer hang off a span that has already shipped. That is the honest
        # count — those supersteps ran after their run was closed, so there is
        # no live span they can belong to, and inventing one is the whole
        # failure mode. Each carries `CORRELATION_CONFLICT`, so a consumer can
        # find them rather than reading them as independent work.
        assert len(traces(live.spans)) == 3
    finally:
        live.teardown()


def test_uninstall_mid_stream_stops_the_straggler_spans_altogether():
    """The same host, the same pumping, and a different outcome — by design.

    `uninstall()` restores the patches FIRST, so the supersteps that follow do
    not reach the node seam at all: three nodes run and one node span exists.
    That is the bounded teardown the ordering buys, and it is why this shape
    cannot produce the orphans the `close_units` case above measures — there is
    nothing left to open them, and nothing asks the registry anything.
    """
    live = Installed()
    try:
        it = chain(live.ctx, 3, name="TornDown", leaves=False).stream({"trail": []})
        first = next(it)
        live.teardown()
        rest = list(it)  # must complete

        untouched = "the graph itself is untouched by the teardown"
        assert list(first) == ["n0"], untouched
        assert [next(iter(chunk)) for chunk in rest] == ["n1", "n2"], untouched

        run = one(live.spans, "invoke_workflow TornDown")
        assert Limitation.ADAPTER_UNINSTALLED in run.capture_integrity.limitations
        # THREE nodes ran (asserted above) and ONE node span exists. Without the
        # first half this reads as "the graph stopped", which is the one outcome
        # a teardown must never cause — and the assertion would be satisfied by
        # an uninstall that broke the host's own generator.
        assert [s.name for s in steps(live.spans)] == ["execute_step n0"], (
            "the two supersteps after the uninstall reach no seam"
        )
        assert adapter_counters()["adapters.langgraph.active.runner.run_with_retry"] == 1
        assembly = assembly_counters()
        assert "assembly._units.ambient_stale" not in assembly
        assert "assembly._units.stale_pin_ambient" not in assembly
    finally:
        # `teardown()` already ran in the body; this is for the paths where an
        # assertion above it did not. A second uninstall is a no-op, and the
        # alternative is that ONE failure here leaves six patches welded on for
        # the rest of the session and every later file measures a haunted tree.
        live.teardown()


def test_close_units_then_abandoning_the_stream_no_longer_swallows_later_spans():
    """The host stops pumping — so nothing ever takes the fork down.

    This is the worst of the four shapes and the one §10.3 is for: the run has
    SHIPPED, the caller's carrier still holds its fork, and it will hold it
    until the process ends. Everything issued on that carrier afterwards joins
    the finished run's trace at confidence 1.0 with no marker — a bare leaf
    span, and then an entire unrelated graph run whose OWN nodes are correctly
    parented to it, so the wrong edge is one level up and every span below it
    looks perfect.

    That is the shape that makes this worth a fix rather than a note: nothing in
    the subtree is detectably wrong, and the one bad edge reads like the best
    kind of evidence wardex produces.
    """
    live = Installed()
    it = None
    try:
        it = chain(live.ctx, 3, name="Abandoned", leaves=False).stream({"trail": []})
        next(it)
        live.adapter.close_units(marker=Limitation.UNIT_INTERRUPTED)
        leaf_span(live.ctx, "after-close")
        chain(live.ctx, 1, name="Later", leaves=False).invoke({"trail": []})

        dead = one(live.spans, "invoke_workflow Abandoned")
        leaf = one(live.spans, "execute_tool after-close")
        later = one(live.spans, "invoke_workflow Later")
        # §10.3: the leaf becomes `ORPHANED_ON_A_CORPSE` with no parent, and the
        # ROOT-placed run becomes `(TRACE_ROOT, 1.0, (CORRELATION_CONFLICT,))` —
        # a trace of its own, marked. `traces` goes 1 -> 3 and
        # `stale_pin_ambient` goes 0 -> 2.
        refused_as_a_corpse(leaf, dead.context.span_id)
        refused_root_after_a_corpse(later, dead.context.span_id)
        under_later = [s for s in steps(live.spans) if s.parent_span_id == later.context.span_id]
        assert len(under_later) == 1, "the later run's own node is parented perfectly"
        assert len(traces(live.spans)) == 3, (
            "the drained run, the refused leaf and the later run each get their own — "
            "before the refusal all three were one trace, and the later run's own "
            "children were parented perfectly under it, so the single wrong edge read "
            "like wardex's best evidence"
        )
        assert assembly_counters()["assembly._units.stale_activation_ambient"] == 2
    finally:
        it = None
        carrier_is_clean()
        live.teardown()


def test_uninstall_then_abandoning_the_stream_leaks_the_carrier_but_marks_it():
    """Same abandon, after `uninstall()` — the fork survives, the reach does not.

    `self._ctx` is deliberately NOT nulled, so a span opened through the context
    an adapter still holds lands on the same leftover fork and is adopted the
    same way. What DOES change is that a later graph run produces nothing at
    all, because the patches went back before the drain: the blast radius of an
    abandoned stream is bounded by what is still instrumented, not by what is
    still ambient. That bound is the reason `uninstall()` restores first.
    """
    live = Installed()
    ctx = live.ctx
    it = None
    try:
        it = chain(ctx, 3, name="AbandonedTorn", leaves=False).stream({"trail": []})
        next(it)
        live.teardown()
        leaf_span(ctx, "after-uninstall")
        before = len(live.spans)
        chain(ctx, 1, name="LaterTorn", leaves=False).invoke({"trail": []})

        dead = one(live.spans, "invoke_workflow AbandonedTorn")
        leaf = one(live.spans, "execute_tool after-uninstall")
        # §10.3: `ORPHANED_ON_A_CORPSE` with no parent, its own trace (2 in
        # total), and `stale_pin_ambient == 1`.
        refused_as_a_corpse(leaf, dead.context.span_id)
        assert len(live.spans) == before, "an uninstalled adapter emits nothing for a new run"
        assert len(traces(live.spans)) == 2
        assert assembly_counters()["assembly._units.ambient_stale"] == 2
        assert assembly_counters()["assembly._units.stale_activation_ambient"] == 1
    finally:
        it = None
        carrier_is_clean()
        live.teardown()  # a no-op on the happy path; insurance on every other


# --------------------------------------------------------------------------
# 3. the tool seam end to end
# --------------------------------------------------------------------------


def test_the_tool_seam_adds_one_span_per_call_and_chains_a_wire_call_to_the_run():
    """The whole point of the seam, as a delta and as a chain of ids.

    The delta is measured against the same workload with the tool half declined
    rather than against a literal, so a future release that changes the node or
    run span count cannot make this pass for the wrong reason.

    The chain is asserted BY ID — `leaf < execute_tool < execute_step tools <
    invoke_workflow` — because a tier assertion cannot see the collapse where
    every wire call is parented to the RUN and the tool spans survive as
    decoration. That collapse is what a seam reading the node's INPUT instead of
    wrapping the call would produce, and with the seam declined it is exactly
    what the baseline shows: the wire call hangs off the node.
    """
    live = Installed()
    try:
        two_tool_graph(live.ctx, name="TwoTools").invoke({"messages": []})
        with_seam = len(live.spans)

        leaf = one(live.spans, "execute_tool leaf-fetch")
        fetch = one(live.spans, "execute_tool fetch")
        node = one(live.spans, "execute_step tools")
        run = one(live.spans, "invoke_workflow TwoTools")
        assert leaf.parent_span_id == fetch.context.span_id
        assert fetch.parent_span_id == node.context.span_id
        assert node.parent_span_id == run.context.span_id
        assert run.parent_span_id is None
        assert len(traces(live.spans)) == 1
        assert {s.tool.call_id for s in tools(live.spans) if s.tool} == {"c1", "c2", None}
        assert adapter_counters()["adapters.langgraph.active.toolnode.run_one"] == 2
    finally:
        live.teardown()

    with pytest.MonkeyPatch.context() as mp:
        import wardex_sdk.adapters._langgraph as mod

        mp.setattr(mod, "_tool_surface_ok", lambda *a: False)
        base = Installed()
        try:
            two_tool_graph(base.ctx, name="TwoTools").invoke({"messages": []})
            without_seam = len(base.spans)
            base_leaf = one(base.spans, "execute_tool leaf-fetch")
            assert tools(base.spans) == [base_leaf], "the wire call is all that is left"
            assert base_leaf.parent_span_id == one(base.spans, "execute_step tools").context.span_id
        finally:
            base.teardown()

    assert with_seam - without_seam == 2, "one span per tool CALL, and nothing else"


def test_a_retrying_wrap_tool_call_is_one_span_and_not_one_per_attempt():
    """The granularity claim: one logical tool call is one span, retries included.

    `wrap_tool_call` calls `execute` three times, so the tool BODY runs three
    times and `_execute_tool_sync` runs three times — and the wire carries one
    `execute_tool` span at `status=OK` holding the last outcome. A seam one
    layer down would emit three siblings for a call the model asked for once,
    and downstream that reads as an agent that called the tool three times.
    """
    live = Installed()
    try:
        body: list[int] = []

        @tool
        def flaky(x: int) -> int:
            """Double a number, unreliably."""
            body.append(1)
            return x * 2

        def wrap(request, execute):
            out = None
            for _ in range(3):
                out = execute(request)
            return out

        wants = [call("flaky", {"x": 3}, "r1")]
        g = StateGraph(MessagesState)
        g.add_node("model", lambda s: {"messages": [AIMessage(content="", tool_calls=wants)]})
        g.add_node("tools", ToolNode([flaky], wrap_tool_call=wrap))
        g.add_edge(START, "model")
        g.add_edge("model", "tools")
        g.add_edge("tools", END)
        app = g.compile()
        app.name = "Retrying"
        app.invoke({"messages": []})

        assert len(body) == 3, "the wrapper really did run the tool three times"
        span = one(live.spans, "execute_tool flaky")
        assert span.status.value == "ok"
        assert span.output_data == b"6"
        assert span.parent_span_id == one(live.spans, "execute_step tools").context.span_id
        assert adapter_counters()["adapters.langgraph.active.toolnode.run_one"] == 1
    finally:
        live.teardown()


def test_a_handled_tool_error_ships_as_a_failure_the_framework_never_raised():
    """The framework's single most ordinary tool failure, and the counter that must be absent.

    `handle_tool_errors` defaults to catching a `ToolInvocationError` — the model
    calling a tool with arguments that do not validate — and returning a
    `ToolMessage(status="error")` instead of raising. Deriving a span's status
    from the exception that left the body is therefore only half a rule here:
    nothing leaves the body at all, and `record_failure` is what keeps the span
    from reading `OK` with the failure legible only to a human reading
    `output_data`.

    The error type is the CONSTANT `tool_error`, not a synthesized exception
    name: by the time `_run_one` returns, the original type is gone — the
    handler replaced it with a content string — so any class name here would be
    an invention.

    `tool_outcome` is the other assertion, and it is about wardex rather than
    the framework. It is the guard around reading that result, and its PRESENCE
    would mean wardex tripped on the commonest thing a tool call does wrong.
    """
    live = Installed()
    try:
        out = tool_graph(
            [call("pure_add", {"a": "not-an-int"}, "bad")], [pure_add], name="BadArgs"
        ).invoke({"messages": []})

        assert out["messages"][-1].status == "error", "the framework converted, not raised"
        span = one(live.spans, "execute_tool pure_add")
        assert span.status.value == "error"
        assert span.error_type == "tool_error"
        assert b"Input should be a valid integer" in span.output_data
        assert span.tool.call_id == "bad"
        assert span.tool.description == "Add two numbers."
        assert "adapters.langgraph.tool_outcome" not in adapter_counters()
    finally:
        live.teardown()


def test_an_unregistered_tool_name_still_ships_a_span_with_no_description():
    """A hallucinated tool name is told from a REGISTERED one by ONE absence.

    The framework answers it the same way it answers a validation failure — an
    error `ToolMessage` rather than a raise — so this span and the handled-error
    span above are identical on every field that carries a failure: same intent,
    same parent, `status=error`, `tool_error`, a real `call_id`. `description is
    None` is the whole of the difference, which is why `_tool_description`
    returns None rather than a placeholder: a placeholder would make "the model
    invented this tool" and "the model got this tool's arguments wrong"
    indistinguishable on the wire.
    """
    live = Installed()
    try:
        tool_graph([call("hallucinated", {"a": 1}, "h1")], [pure_add], name="Ghost").invoke(
            {"messages": []}
        )
        span = one(live.spans, "execute_tool hallucinated")
        assert span.tool.name == "hallucinated"
        assert span.tool.call_id == "h1"
        assert span.tool.description is None, "the one field that says it was never registered"
        assert span.status.value == "error"
        assert b"is not a valid tool" in span.output_data
        assert span.parent_span_id == one(live.spans, "execute_step tools").context.span_id
    finally:
        live.teardown()


def test_one_tool_node_task_fans_out_to_one_child_span_per_call():
    """Three calls, ONE node task, three tool spans — the reason the seam sits here.

    A hand-built `ToolNode` on a plain edge gives the NODE seam a single task for
    all three calls, so a tool span read off the node's input would be one span
    with three names in it. Counting the node's children by parent id is what
    makes the fan-out visible; counting node spans is what shows there was only
    ever one task to read.
    """
    live = Installed()
    try:
        tool_graph(
            [
                call("pure_add", {"a": 1, "b": 2}, "f1"),
                call("pure_add", {"a": 3, "b": 4}, "f2"),
                call("pure_add", {"a": 5, "b": 6}, "f3"),
            ],
            [pure_add],
            name="FanTools",
        ).invoke({"messages": []})

        node = one(live.spans, "execute_step tools")
        kids = [s for s in live.spans if s.parent_span_id == node.context.span_id]
        assert len(kids) == 3
        assert {s.tool.call_id for s in kids} == {"f1", "f2", "f3"}
        assert len(steps(live.spans)) == 2, "one model task and ONE tools task"
        snap = adapter_counters()
        assert snap["adapters.langgraph.active.toolnode.run_one"] == 3
        assert snap["adapters.langgraph.active.runner.run_with_retry"] == 2
    finally:
        live.teardown()


# --------------------------------------------------------------------------
# 4. both agent constructors reach the tool seam
# --------------------------------------------------------------------------


def _agent_shapes():
    """The four ways a tool-calling agent is built on this langgraph/langchain pair."""
    from langchain.agents import create_agent
    from langgraph.prebuilt import create_react_agent

    calls = [call("pure_add", {"a": 1, "b": 2}, "t1"), call("pure_add", {"a": 3, "b": 4}, "t2")]
    return calls, {
        "create_react_agent_v2": lambda m: create_react_agent(m, [pure_add]),
        "create_react_agent_v1": lambda m: create_react_agent(m, [pure_add], version="v1"),
        "create_agent": lambda m: create_agent(m, [pure_add]),
        "hand_built_toolnode": lambda m: tool_graph(calls, [pure_add], name="HandBuilt"),
    }


def test_every_agent_constructor_drives_the_tool_seam_identically():
    """Four constructors, two node-seam shapes, ONE set of tool spans.

    `create_react_agent`'s default `version="v2"` sends one task per tool call
    while `version="v1"` and a hand-built `ToolNode` give the node seam a single
    task for both — measured here as 4 node spans against 3 and 2. The tool
    spans compare EQUAL across all four anyway, and that equality is the whole
    argument for putting the seam BELOW the fan-out: the shape of the node layer
    is a framework decision that keeps changing, and a tool span must not
    inherit it.

    `langchain.agents.create_agent` is included because it is the successor
    `create_react_agent` now emits a deprecation for, so its coverage is a
    measured claim here rather than an assumed one.
    """
    calls, builders = _agent_shapes()
    signatures: dict[str, list] = {}
    node_counts: dict[str, int] = {}
    for label, build in builders.items():
        live = Installed()
        try:
            counters.reset()
            build(StubModel(calls=calls)).invoke({"messages": [("user", "add")]})
            assert adapter_counters()["adapters.langgraph.active.toolnode.run_one"] == 2, (
                f"{label}: the tool seam must fire once per CALL"
            )
            assert len(tools(live.spans)) == 2, f"{label}: one execute_tool span per call"
            signatures[label] = tool_signature(live.spans)
            node_counts[label] = len(steps(live.spans))
        finally:
            live.teardown()

    assert node_counts["create_react_agent_v2"] == 4
    assert node_counts["create_agent"] == 4
    assert node_counts["create_react_agent_v1"] == 3, "v1 gives the node seam ONE tools task"
    assert node_counts["hand_built_toolnode"] == 2
    assert len({tuple(sig) for sig in signatures.values()}) == 1, (
        f"the tool spans differ between shapes: {signatures}"
    )


# --------------------------------------------------------------------------
# 5. batch([1 input]) — the control every leak test needs
# --------------------------------------------------------------------------

_PROBE: contextvars.ContextVar[str | None] = contextvars.ContextVar("g7_probe", default=None)


def _reducer(a: list, b: list) -> list:
    """Runs on the DRIVING carrier — the one `Pregel.stream` is entered on."""
    _PROBE.set("reducer")
    return a + b


class ProbeState(TypedDict):
    trail: Annotated[list, _reducer]


def _probe_graph():
    def node(state: ProbeState) -> ProbeState:
        _PROBE.set("node")
        return {"trail": ["n0"]}

    g = StateGraph(ProbeState)
    g.add_node("n0", node)
    g.add_edge(START, "n0")
    g.add_edge("n0", END)
    app = g.compile()
    app.name = "Batchy"
    return app


def test_batch_of_one_is_the_only_shape_where_a_carrier_leak_would_be_visible():
    """Why a leak test must be written on `batch([1])` and never on `batch([2+])`.

    langchain-core skips the executor for a single input ("If there's only one
    input, don't bother with the executor"), so the run is driven in the
    CALLER's context — proved here by the reducer, which runs on the driving
    carrier and whose ContextVar write comes back to the caller. With two inputs
    that same write is invisible, so every "the carrier is clean afterwards"
    assertion is green for free and measures nothing.

    A probe placed in a NODE body is worthless in both shapes: langgraph submits
    every task under `copy_context()` (`pregel/_executor.py`), so a node's write
    never reaches the caller even when the whole batch ran on the caller's own
    thread. That is the trap this case exists to document.

    What survives both shapes is the COUNTER pair — `stream_finalized` against
    `active.pregel.stream` — because `counters` is process-global rather than
    carrier-scoped. That is the handle a future leak test should reach for.
    """
    live = Installed()
    try:
        app = _probe_graph()

        _PROBE.set(None)
        app.batch([{"trail": []}])
        assert _PROBE.get() == "reducer", (
            "batch of one runs on the caller's carrier — a leak here WOULD show"
        )
        assert len(runs(live.spans)) == 1
        snap = adapter_counters()
        assert (
            snap["adapters.langgraph.stream_finalized"]
            == snap["adapters.langgraph.active.pregel.stream"]
        ), "the run seam took down what it installed"
        leaf_span(live.ctx, "after-one")
        assert one(live.spans, "execute_tool after-one").parent_span_id is None
        assert edge_of(one(live.spans, "execute_tool after-one")) == CLEAN_ORPHAN

        live.client.spans.clear()
        counters.reset()
        _PROBE.set(None)
        app.batch([{"trail": []}, {"trail": []}])
        assert _PROBE.get() is None, (
            "two inputs are driven in copied contexts, so this same probe — and "
            "any carrier assertion built on it — proves nothing"
        )
        assert len(runs(live.spans)) == 2
        snap = adapter_counters()
        assert snap["adapters.langgraph.stream_finalized"] == 2
        assert snap["adapters.langgraph.active.pregel.stream"] == 2
    finally:
        live.teardown()


# --------------------------------------------------------------------------
# 6. a hostile config
# --------------------------------------------------------------------------


class _HostileConfigurable:
    """A `configurable` whose every read raises. Not a dict, deliberately."""

    def get(self, *a: Any, **k: Any) -> Any:
        raise RuntimeError("hostile get")


def _rebind_name(app, exc: BaseException):
    """Give a REAL compiled graph a `.name` that raises. No fake `Pregel`."""

    def boom(self):
        raise exc

    app.__class__ = type(f"{type(app).__name__}Hostile", (type(app),), {"name": property(boom)})
    return app


def test_a_graph_whose_name_is_gone_still_ships_one_run_span_and_one_trace():
    """`_graph_name` degrades to a literal, and a degraded SUBJECT costs nothing.

    `INVOKE_WORKFLOW` requires `workflow_name`, so the one thing this read may
    not do is come back empty: a missing required key DELETES the span, and with
    the run span gone every node under it orphans at 0.0. The `or "LangGraph"`
    is what keeps that from being one attribute's problem.

    No counter fires, and that is the point of `getattr`'s default: an absent
    attribute is an ANSWER, not a failure worth reporting.
    """
    live = Installed()
    try:
        app = _rebind_name(chain(live.ctx, 2, name="Named", leaves=False), AttributeError("gone"))
        app.invoke({"trail": []})

        run = one(live.spans, "invoke_workflow LangGraph")
        assert run.workflow_name == "LangGraph"
        assert extra_of(run)["wardex.framework"] == "langgraph"
        assert len(traces(live.spans)) == 1
        assert len(steps(live.spans)) == 2
        assert all(s.parent_span_id == run.context.span_id for s in steps(live.spans))
        assert "adapters.langgraph.stream_prologue" not in adapter_counters()
    finally:
        live.teardown()


def test_a_hostile_configurable_reaches_the_host_as_the_frameworks_own_error():
    """The exception the host sees must be langgraph's, never one wardex made.

    Nothing is computed in the `enter` header, so the only thing wardex does
    with this config is decline to read it: `type(conf) is not dict` refuses a
    `.get` that raises WITHOUT calling it, so no guard has to catch anything and
    no `describe_run_extras` counter appears. langgraph then rejects the same
    object on its own terms, and the run span still ships — one trace, the
    workflow name present, the framework's error type on it.

    The failure this pins is the measured one: with those reads in the `enter`
    header instead, `graph.invoke()` became a wardex-shaped `KeyError` and the
    run emitted zero spans.
    """
    live = Installed()
    try:
        app = chain(live.ctx, 2, name="Hostile", leaves=False)
        with pytest.raises(AttributeError) as caught:
            app.invoke({"trail": []}, config={"configurable": _HostileConfigurable()})
        assert "_HostileConfigurable" in str(caught.value), "langgraph's own read, not wardex's"

        run = one(live.spans, "invoke_workflow Hostile")
        assert run.workflow_name == "Hostile"
        assert run.error_type == "AttributeError"
        assert len(traces(live.spans)) == 1
        assert steps(live.spans) == [], "the graph never ran, so there is nothing to orphan"
        snap = adapter_counters()
        assert "adapters.langgraph.describe_run_extras" not in snap
        assert "adapters.langgraph.stream_prologue" not in snap
    finally:
        live.teardown()


def test_a_thread_id_that_is_not_a_scalar_is_skipped_without_a_guard_trip():
    """The optional half declines on TYPE, so the ordinary hostile value costs nothing.

    A non-scalar `thread_id` is the shape a user's own config most easily takes,
    and `isinstance(thread_id, str | int)` is what keeps it from reaching
    `set_extra`. No guard fires, the key is simply absent, and the rest of the
    run — the run span, both node spans, one trace — is untouched. An optional
    enrichment key that could cost the run's whole tree is the failure the split
    inside `_describe_run` exists to prevent.
    """
    live = Installed()
    try:
        app = chain(live.ctx, 2, name="Cfg", leaves=False)
        app.invoke({"trail": []}, config={"configurable": {"thread_id": ["not", "scalar"]}})

        run = one(live.spans, "invoke_workflow Cfg")
        assert run.workflow_name == "Cfg"
        assert "wardex.langgraph.thread_id" not in extra_of(run)
        assert len(traces(live.spans)) == 1
        assert len(steps(live.spans)) == 2
        assert "adapters.langgraph.describe_run_extras" not in adapter_counters()
    finally:
        live.teardown()


def test_a_name_that_raises_something_other_than_attributeerror_still_ships_the_run(capsys):
    """MEASURED. `getattr(graph, "name", None)` absorbs `AttributeError` ALONE.

    A `.name` implemented as a property that raises anything else propagates,
    and there are two places this adapter reads it. Both are guarded, and the
    reason is blast radius rather than tidiness: an unguarded read in
    `_describe_run`'s MANDATORY half abandons the unit, and the host's body then
    runs with nothing ambient — so the run span is DELETED and every node under
    it orphans at confidence 0.0, for one attribute.

    The rule the guarded spelling follows: the mandatory half may be guarded
    exactly where a TOTAL fallback exists. It does here — LangGraph's own
    default graph name — and a run span reading `LangGraph` is indistinguishable
    from one whose graph genuinely has that name, so nothing false is published.
    An unknown framework read has no such answer, which is why the rest of the
    mandatory half is still unguarded.

    What ships, measured: one run span, one trace, `workflow_name="LangGraph"`,
    the operation name bare (the SUBJECT degraded, which costs a suffix and not
    a span), and TWO counters naming both reads. The host still sees its own
    `RuntimeError`, because langgraph reads `.name` too and dies on the same
    attribute — which is also why there are no node spans to orphan here.
    """
    live = Installed()
    try:
        app = _rebind_name(chain(live.ctx, 2, name="Named", leaves=False), RuntimeError("boom"))
        with pytest.raises(RuntimeError, match="boom"):
            app.invoke({"trail": []})

        snap = adapter_counters()
        assert snap["adapters.langgraph.stream_prologue"] == 1
        assert snap["adapters.langgraph.describe_run_name"] == 1
        assert "adapters.langgraph.enter.invoke_workflow" not in snap, (
            "the unit must not be abandoned — that is what deletes the run span "
            "and orphans everything under it"
        )

        run = one(live.spans, "invoke_workflow")
        assert run.workflow_name == "LangGraph"
        assert extra_of(run)["wardex.framework"] == "langgraph"
        assert len(traces(live.spans)) == 1
        assert steps(live.spans) == [], "langgraph died on the same attribute, so no node ran"

        # Nothing was abandoned, so there is nothing for `_abandon` to report.
        # Those two counters are the whole record, which is the right amount of
        # noise for one read that moved.
        assert "internal error at enter.invoke_workflow" not in capsys.readouterr().err
    finally:
        live.teardown()
