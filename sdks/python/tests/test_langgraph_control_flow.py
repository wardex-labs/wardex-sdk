"""LangGraph adapter — control flow, interrupts, fan-out and the functional API.

Every graph here is a real compiled `StateGraph`; see `test_langgraph_adapter`'s
module docstring for why a `Pregel` double would be the one test shape this
adapter exists to rule out.

The thread running through the file is that LangGraph implements PAUSING,
HANDING OFF and DRAINING by raising, so the same `except BaseException` that
records a crashed node also sees a human-in-the-loop pause. `CONTROL_FLOW` is
what keeps those apart, and the assertions below are written on `status` and
`error_type` as they SHIP rather than on the classification function, because
the classification is only correct if the registry's reader is wired to it —
which is why everything here installs through `AdapterRegistry`.
"""

from __future__ import annotations

import asyncio
import gc
import sys
from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Send, interrupt

import wardex_sdk._adapters._langgraph as lg_mod
from test_langgraph_adapter import (  # noqa: F401 — `_clean_scope` is an autouse fixture
    Installed,
    _clean_scope,
    adapter_counters,
    assembly_counters,
    call,
    chain,
    edge_of,
    extra_of,
    named,
    parent_name,
    runs,
    steps,
    tools,
    traces,
)
from test_langgraph_adapter import installed as _installed_fixture
from wardex_sdk._assembly import Limitation
from wardex_sdk._enums import StatusCode
from wardex_sdk._limits import CaptureLimits

#: The harness's fixture, rebound so that a test may take `installed` as an
#: argument. A parameter shadowing a bare `from … import installed` reads to the
#: linter as a redefined import, and one suppression comment per test is a worse
#: trade than one alias; pytest registers a fixture under the name it is bound
#: to here, so the rebinding costs nothing.
installed = _installed_fixture


# --------------------------------------------------------------------------
# local builders — everything the shared harness does not already have
# --------------------------------------------------------------------------


def _append(a: list, b: list) -> list:
    return a + b


class Trail(TypedDict):
    trail: Annotated[list, _append]


def send_fanout(width: int, *, node, name: str = "SendFan"):
    """A `Send` fan-out: ONE node name, `width` sibling tasks.

    Not the harness's `fanout`, which gives every worker its own node name and
    therefore its own `wardex.step.name`. The whole question here is what
    distinguishes siblings that a framework made indistinguishable, so the
    graph has to be the one that produces them.
    """
    g = StateGraph(Trail)
    g.add_node("fan", lambda s: {"trail": ["fan"]})
    g.add_node("worker", node)
    g.add_edge(START, "fan")
    g.add_conditional_edges(
        "fan", lambda s: [Send("worker", {"trail": [f"s{i}"]}) for i in range(width)], ["worker"]
    )
    g.add_edge("worker", END)
    app = g.compile()
    app.name = name
    return app


def markers_of(span) -> tuple:
    """The shipped limitations, read off the emitted span via the harness."""
    return edge_of(span)[2]


def marked(spans, marker: Limitation) -> list:
    return [s for s in spans if marker in markers_of(s)]


def outcomes(spans) -> list[tuple[str, StatusCode, str | None]]:
    return [(s.name, s.status, s.error_type) for s in spans]


# --------------------------------------------------------------------------
# `Send` fan-out: the vocabulary, and the breadth bound
# --------------------------------------------------------------------------


def test_send_siblings_share_a_name_and_an_index_and_differ_only_by_task_id(installed):
    """The task id is the SOLE discriminator, and that is a decision, not luck.

    A `Send` fan-out is the shape where one node name yields N spans that
    LangGraph itself makes indistinguishable — same name, same superstep, same
    trigger. Without `wardex.step.task_id` a consumer grouping by name would
    fold three concurrent workers into one, and nothing on the wire would say
    it had.
    """
    send_fanout(3, node=lambda s: {"trail": ["w"]}).invoke({"trail": []})
    workers = [s for s in steps(installed.spans) if s.name == "execute_step worker"]
    assert len(workers) == 3
    extras = [extra_of(s) for s in workers]
    assert {e["wardex.step.name"] for e in extras} == {"worker"}
    assert {e["wardex.step.index"] for e in extras} == {2}, "one superstep, one index"
    assert len({e["wardex.step.task_id"] for e in extras}) == 3
    # By EDGE, not by tier: three siblings of one run is the claim, and a tier
    # assertion would read the same if each had become its own root.
    assert {parent_name(installed.spans, s) for s in workers} == {"invoke_workflow SendFan"}
    assert len(traces(installed.spans)) == 1


def test_send_siblings_carry_no_ordinal_key(installed):
    """Siblings are UNORDERED on the wire, and this is the test that pins it.

    Nothing in the adapter derives from `task.path`, whose Send entries carry
    the fan-out index — so a future author adding `task.path[1]` would give
    these three otherwise-identical spans three different values here. Asserted
    as "every key but the two the task id feeds is equal across siblings",
    which catches that shape whatever the new key is called.
    """
    send_fanout(3, node=lambda s: {"trail": ["w"]}).invoke({"trail": []})
    extras = [extra_of(s) for s in steps(installed.spans) if s.name == "execute_step worker"]
    assert len(extras) == 3
    # `wardex.step.namespace` embeds the task id, so it is the second key that
    # legitimately varies; everything else must be byte-identical.
    varying = {"wardex.step.task_id", "wardex.step.namespace"}
    shared = [{k: v for k, v in e.items() if k not in varying} for e in extras]
    assert shared[0] == shared[1] == shared[2], (
        "an ordinal key would be the one thing that differs between siblings"
    )
    assert [k for e in extras for k in e if "path" in k] == []


def test_an_async_send_fanout_past_the_breadth_bound_evicts_the_oldest_children():
    """300 children of one run, all live at once, against a 256-entry table.

    The eviction is not a bug — it is the bound doing its job — but it must be
    VISIBLE, because the alternative is a run whose first 44 workers are simply
    absent with nothing to say so. This asserts the three things a reader has:
    `CHILD_SPAN_UNCLOSED` on the evicted spans, which still ship, the
    `child_table_full` counter that makes the loss countable in aggregate, and
    WHICH spans were evicted — the oldest, as a prefix of start time. A count
    alone passes on an eviction policy that dropped 44 arbitrary children.

    44 rather than 45 because the `fan` node closes — and is unlinked from the
    parent's child table — before the first worker opens.

    A BARRIER, not a sleep. "All 300 are live at once" is the precondition the
    whole arithmetic rests on, and a `sleep` only makes it likely — on a loaded
    machine the early workers finish, unlink, and the count comes out under 44
    for a reason that has nothing to do with the bound. The barrier makes the
    precondition an assertion (`arrived == width`) instead of a hope.
    """
    width = 300
    bound = CaptureLimits().resolved()["max_entries_per_unit"]
    assert bound == 256, "the arithmetic below is this limit's, not a literal"
    arrived = 0
    gate: asyncio.Event | None = None

    async def worker(state):
        nonlocal arrived
        arrived += 1
        if arrived == width:
            gate.set()
        await gate.wait()  # nobody proceeds until every sibling has opened its unit
        return {"trail": ["w"]}

    async def drive(app):
        nonlocal gate
        gate = asyncio.Event()
        return await app.ainvoke({"trail": []})

    live = Installed()
    try:
        asyncio.run(drive(send_fanout(width, node=worker, name="WideAsync")))
        assert arrived == width, "the premise: every sibling was live simultaneously"
        assert len(steps(live.spans)) == width + 1, "every step still ships"
        assert len(marked(live.spans, Limitation.CHILD_SPAN_UNCLOSED)) == width - bound == 44
        assert assembly_counters()["assembly._units.child_table_full"] == 44

        # WHICH 44. Sorted by start time, the marked ones are a prefix — the
        # table evicts its oldest entry, so the spans that lose their link are
        # the ones that opened first.
        workers = sorted(
            (s for s in steps(live.spans) if s.name == "execute_step worker"),
            key=lambda s: s.start_time_ns,
        )
        flags = [Limitation.CHILD_SPAN_UNCLOSED in markers_of(s) for s in workers]
        assert flags == [True] * 44 + [False] * (width - 44), (
            "the evicted children must be the OLDEST, not 44 arbitrary ones"
        )

        # And the eviction is about the LINK, not the edge: all 301 steps are
        # still the run's children by id, in one trace.
        run = runs(live.spans)[0]
        assert {s.parent_span_id for s in steps(live.spans)} == {run.context.span_id}
        assert len(traces(live.spans)) == 1
    finally:
        live.teardown()


def test_the_same_width_run_synchronously_stays_clean():
    """The CONTROL for the case above, and the reason it had to be async.

    The sync path runs its node tasks on a BOUNDED thread pool
    (`BackgroundExecutor` over `get_executor_for_config`), so a closed child is
    unlinked from the run's table long before the pool gets round to the 257th
    worker and the same 300-wide fan-out evicts nothing. The async path submits
    all 300 at once, which is the whole difference: a breadth test written only
    on `invoke()` passes while `ainvoke()` is dropping 44 spans.
    """
    live = Installed()
    try:
        send_fanout(300, node=lambda s: {"trail": ["w"]}, name="WideSync").invoke({"trail": []})
        assert len(steps(live.spans)) == 301
        assert marked(live.spans, Limitation.CHILD_SPAN_UNCLOSED) == []
        assert "assembly._units.child_table_full" not in assembly_counters()
        # By id, so that "301 clean spans" cannot be satisfied by 301 spans that
        # became their own roots — which is the other way a wide fan-out breaks.
        run = runs(live.spans)[0]
        assert {s.parent_span_id for s in steps(live.spans)} == {run.context.span_id}
        assert len(traces(live.spans)) == 1
    finally:
        live.teardown()


# --------------------------------------------------------------------------
# abandonment: three entries, three different wire signals
# --------------------------------------------------------------------------


def test_abandoning_a_sync_stream_ships_the_run_as_a_generator_exit(installed):
    """A host that stops iterating is a run that never finished, and it SHIPS.

    The span is the only record that the run existed at all — nothing else
    fires — so what matters is that `stream_finalized` says the generator was
    finalized and that the run carries the interpreter's own word for why.
    """
    app = chain(installed.ctx, 3, name="SyncAbandon", leaves=False)
    for _ in app.stream({"trail": []}):
        break
    shipped = runs(installed.spans)
    assert len(shipped) == 1
    assert (shipped[0].status, shipped[0].error_type) == (StatusCode.ERROR, "GeneratorExit")
    assert len(steps(installed.spans)) == 1, "only the first node ever ran"
    snap = adapter_counters()
    assert snap["adapters.langgraph.stream_finalized"] == 1
    assert "adapters.langgraph.stream_finalized_off_carrier" not in snap


def test_abandoning_an_astream_is_finalized_off_carrier_and_reads_cancelled():
    """The async twin does NOT produce the sync signal, and the counter says why.

    `async for ... break` leaves the generator suspended; the LOOP finalizes it
    at shutdown, on its own finalizer task. So the exception that reaches the
    wrapper is the loop's `CancelledError` rather than a `GeneratorExit`, and
    `astream_finalized_off_carrier` is the handle an operator has on exactly
    that — the run's scope was taken down somewhere other than where it was
    installed.
    """

    async def drive(live):
        app = chain(live.ctx, 3, name="AsyncAbandon", leaves=False)
        async for _ in app.astream({"trail": []}):
            break

    live = Installed()
    try:
        asyncio.run(drive(live))
        # Collected HERE, deliberately. The loop's own `aclose` task outlives
        # the loop and asyncio logs "Task was destroyed but it is pending" when
        # it is finally collected — on stderr, from a `__del__`, i.e. inside
        # whichever test happens to allocate next. Forcing it inside the case
        # that created it keeps a neighbour's `capsys.readouterr().err == ""`
        # from failing for a reason that has nothing to do with it.
        gc.collect()
        shipped = runs(live.spans)
        assert len(shipped) == 1
        assert (shipped[0].status, shipped[0].error_type) == (StatusCode.ERROR, "CancelledError")
        snap = adapter_counters()
        assert snap["adapters.langgraph.astream_finalized"] == 1
        assert snap["adapters.langgraph.astream_finalized_off_carrier"] == 1
    finally:
        live.teardown()


def test_closing_an_abandoned_astream_on_its_own_task_reads_generator_exit():
    """The same abandon, finalized by the HOST — and the wire signal changes.

    Written beside the case above because the pair is the finding: "an
    abandoned async run" has two spellings, and which exception the run span
    carries is decided by WHO finalizes the generator, not by the adapter. A
    consumer alerting on `CancelledError` would miss this one entirely.
    """

    async def drive(live):
        app = chain(live.ctx, 3, name="AcloseSameTask", leaves=False)
        it = app.astream({"trail": []})
        async for _ in it:
            break
        await it.aclose()

    live = Installed()
    try:
        asyncio.run(drive(live))
        shipped = runs(live.spans)
        assert len(shipped) == 1
        assert (shipped[0].status, shipped[0].error_type) == (StatusCode.ERROR, "GeneratorExit")
        snap = adapter_counters()
        assert snap["adapters.langgraph.astream_finalized"] == 1
        assert "adapters.langgraph.astream_finalized_off_carrier" not in snap
    finally:
        live.teardown()


def test_abandoning_astream_events_cancels_the_NODE_as_well_as_the_run():
    """The third entry, and the only one where a node span is collateral.

    `astream_events` pumps the graph from an internal consumer task that
    langchain-core CANCELS when the caller stops iterating, so the cancellation
    lands inside the node body rather than only at the run boundary. That is
    why abandonment has to be measured per entry point instead of asserted once
    on `stream`: the node span reads ERROR here and OK through the other two.
    """

    async def drive(live):
        app = chain(live.ctx, 3, name="EventsAbandon", leaves=False)
        async for _ in app.astream_events({"trail": []}, version="v2"):
            break

    live = Installed()
    try:
        asyncio.run(drive(live))
        assert outcomes(live.spans) == [
            ("execute_step n0", StatusCode.ERROR, "CancelledError"),
            ("invoke_workflow EventsAbandon", StatusCode.ERROR, "CancelledError"),
        ]
        assert adapter_counters()["adapters.langgraph.astream_finalized"] == 1
    finally:
        live.teardown()


# --------------------------------------------------------------------------
# cancellation amplification
# --------------------------------------------------------------------------


def test_one_failing_node_ships_its_cancelled_siblings_as_errors_too():
    """Three ERROR node spans for ONE failure, deliberately, in one trace.

    A `CancelledError` is a `BaseException` and not a `GraphBubbleUp`, so it
    does not take the control-flow branch — and it should not: the siblings
    were abandoned mid-flight and genuinely did not complete, which is what
    `ERROR` means. Classifying them as control flow would be the adapter
    claiming a run finished cleanly when two thirds of it was torn down.

    The trace assertion is the other half. Amplification is only readable if
    the three failures sit under one run; three roots would look like three
    unrelated crashes.

    Ordered by EVENTS, not by two sleeps chosen 500x apart. The failing node
    waits until both siblings are demonstrably inside their bodies, and the
    siblings wait on something that only cancellation can end — so "two spans
    were cancelled mid-flight" is arranged rather than raced, and the test
    cannot pass by having measured two nodes that had already finished.
    """
    inside: asyncio.Event | None = None
    started = 0

    async def boom(state):
        await inside.wait()  # both siblings are suspended in their bodies
        raise RuntimeError("boom")

    def slow(i):
        async def node(state):
            nonlocal started
            started += 1
            if started == 2:
                inside.set()
            await asyncio.Event().wait()  # only a cancellation ever ends this
            return {"trail": [f"w{i}"]}

        return node

    g = StateGraph(Trail)
    g.add_node("boom", boom)
    g.add_edge(START, "boom")
    g.add_edge("boom", END)
    for i in range(2):
        g.add_node(f"w{i}", slow(i))
        g.add_edge(START, f"w{i}")
        g.add_edge(f"w{i}", END)
    app = g.compile()
    app.name = "Amplified"

    async def drive():
        nonlocal inside
        inside = asyncio.Event()
        return await app.ainvoke({"trail": []})

    live = Installed()
    try:
        try:
            asyncio.run(drive())
            raise AssertionError("the host was supposed to see the RuntimeError")
        except RuntimeError:
            pass
        assert started == 2, "both siblings really were in flight when the failure landed"
        node_spans = steps(live.spans)
        assert len(node_spans) == 3
        assert all(s.status is StatusCode.ERROR for s in node_spans)
        assert sorted(s.error_type for s in node_spans) == [
            "CancelledError",
            "CancelledError",
            "RuntimeError",
        ]
        assert len(traces(live.spans)) == 1, "amplification is only readable under one run"
        # BY ID as well as by trace: one trace is also what a chain of wrongly
        # nested siblings produces, and the claim is that all three are the
        # RUN's children.
        run = runs(live.spans)[0]
        assert {s.parent_span_id for s in node_spans} == {run.context.span_id}
    finally:
        live.teardown()


# --------------------------------------------------------------------------
# `interrupt()` at three levels
# --------------------------------------------------------------------------


def test_an_interrupt_inside_a_node_is_control_flow_and_passes_through_unchanged(installed):
    """A human-in-the-loop pause is not a crashed run, at either level.

    `status is UNSET` with `error_type is None` is the whole of the claim: the
    span says "this did not complete" without saying "this failed", which is
    exactly what a pause is. `ERROR`/`GraphInterrupt` here would report every
    approval prompt in a production graph as an incident.

    The identity half is the other invariant `_run` owes the host — wardex may
    classify an exception but may never REPLACE it. Read across two wrappers:
    the object leaves the inner node body, crosses `execute_step ask` and
    `invoke_workflow Inner`, and arrives in the outer node's `except` as the
    same object.
    """
    seen: dict[str, BaseException] = {}

    def ask(state):
        try:
            interrupt("who?")
        except BaseException as exc:
            seen["raised"] = exc
            raise
        return {"trail": ["never"]}

    inner = StateGraph(Trail)
    inner.add_node("ask", ask)
    inner.add_edge(START, "ask")
    inner.add_edge("ask", END)
    sub = inner.compile()
    sub.name = "Inner"

    def child(state):
        try:
            return sub.invoke(state)
        except BaseException as exc:
            seen["caught"] = exc
            raise

    outer = StateGraph(Trail)
    outer.add_node("child", child)
    outer.add_edge(START, "child")
    outer.add_edge("child", END)
    app = outer.compile(checkpointer=InMemorySaver())
    app.name = "Outer"

    out = app.invoke({"trail": []}, {"configurable": {"thread_id": "n1"}})
    assert "__interrupt__" in out
    assert seen["caught"] is seen["raised"], "wardex may classify, never replace"
    assert outcomes(installed.spans) == [
        ("execute_step ask", StatusCode.UNSET, None),
        ("invoke_workflow Inner", StatusCode.UNSET, None),
        ("execute_step child", StatusCode.UNSET, None),
        ("invoke_workflow Outer", StatusCode.OK, None),
    ]


def test_an_interrupt_inside_a_subgraph_reads_as_control_flow_at_both_levels(installed):
    """A subgraph pauses the OUTER node too, and both spans have to say so.

    The outer `execute_step child` never sees the `interrupt()` call — it sees
    a compiled subgraph raising out of it — so its classification is a second,
    independent trip through `CONTROL_FLOW`. A test that only asserted the
    inner node would pass with the outer one reading `ERROR`, which is the
    shape where one paused approval step turns a whole nested graph red.
    """

    def ask(state):
        return {"trail": [interrupt("who?")]}

    inner = StateGraph(Trail)
    inner.add_node("ask", ask)
    inner.add_edge(START, "ask")
    inner.add_edge("ask", END)
    sub = inner.compile()
    sub.name = "Inner"

    outer = StateGraph(Trail)
    outer.add_node("child", sub)
    outer.add_edge(START, "child")
    outer.add_edge("child", END)
    app = outer.compile(checkpointer=InMemorySaver())
    app.name = "Outer"

    out = app.invoke({"trail": []}, {"configurable": {"thread_id": "s1"}})
    assert "__interrupt__" in out
    assert outcomes(installed.spans) == [
        ("execute_step ask", StatusCode.UNSET, None),
        ("invoke_workflow Inner", StatusCode.UNSET, None),
        ("execute_step child", StatusCode.UNSET, None),
        ("invoke_workflow Outer", StatusCode.OK, None),
    ]


def test_an_interrupt_inside_a_tool_reads_as_control_flow_at_the_tool_and_its_node(installed):
    """The third seam. The tool wrapper classifies through the same reader.

    An approval tool is the canonical human-in-the-loop shape, and it crosses
    `_run` twice — once at `execute_tool` and once at the `execute_step tools`
    that contains it. The RUN meanwhile reads OK and hands the caller
    `__interrupt__`, because from LangGraph's side nothing failed: the graph
    paused and said where.
    """

    @tool
    def approve(what: str) -> str:
        """Ask a human to approve something."""
        return interrupt(f"approve {what}?")

    asked = call("approve", {"what": "x"}, "c1")

    def model_node(state):
        return {"messages": [AIMessage(content="", tool_calls=[asked])]}

    g = StateGraph(MessagesState)
    g.add_node("model", model_node)
    g.add_node("tools", ToolNode([approve]))
    g.add_edge(START, "model")
    g.add_edge("model", "tools")
    g.add_edge("tools", END)
    app = g.compile(checkpointer=InMemorySaver())
    app.name = "ToolInterrupt"

    out = app.invoke({"messages": []}, {"configurable": {"thread_id": "t1"}})
    assert "__interrupt__" in out
    paused = tools(installed.spans) + [
        s for s in steps(installed.spans) if s.name == "execute_step tools"
    ]
    assert [s.name for s in paused] == ["execute_tool approve", "execute_step tools"]
    for span in paused:
        assert (span.status, span.error_type) == (StatusCode.UNSET, None)
    shipped = runs(installed.spans)
    assert [(s.status, s.error_type) for s in shipped] == [(StatusCode.OK, None)]


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason=(
        "LangGraph cannot run interrupt() inside an ASYNC node before 3.11, with or "
        "without wardex: langgraph/config.py's get_config() has an explicit "
        "`sys.version_info < (3, 11)` guard and then finds no "
        "`var_child_runnable_config`, because Task context propagation is 3.11+. "
        "Verified on a bare venv with no wardex installed — 3.10 raises "
        "'Called get_config outside of a runnable context', 3.11 returns "
        "'__interrupt__' — so skipping here declines to assert a behaviour the "
        "FRAMEWORK does not have, rather than hiding one of ours. The sync twins "
        "above run on every supported version and are what pin the classification."
    ),
)
def test_an_interrupt_through_the_async_seams_is_control_flow_too():
    """The same classification, through `astream` and `arun_with_retry`.

    Every other control-flow assertion in this file drives `invoke()`, so all of
    them exercise `_mk_stream` and `_mk_run_with_retry`. The async twins reach
    `_run`'s `except BaseException` by a different route — an `await` inside the
    `with`, and an async generator's `athrow` at the run seam rather than a
    plain `yield from` — and `CONTROL_FLOW` being a CLASSVAR populated at
    install means a wiring that only reached the sync pair would leave every
    async interrupt shipping as `ERROR`/`GraphInterrupt`.

    That is the shape this pins: a production graph on `ainvoke` reporting every
    human-in-the-loop pause as a crashed node, while the sync tests stay green.
    """

    def ask(state):
        return {"trail": [str(interrupt("who?"))]}

    g = StateGraph(Trail)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    app = g.compile(checkpointer=InMemorySaver())
    app.name = "AsyncPause"

    live = Installed()
    try:
        out = asyncio.run(app.ainvoke({"trail": []}, {"configurable": {"thread_id": "as1"}}))
        assert "__interrupt__" in out
        assert outcomes(live.spans) == [
            ("execute_step ask", StatusCode.UNSET, None),
            ("invoke_workflow AsyncPause", StatusCode.OK, None),
        ]
        # The async sites, not the sync ones — otherwise this could be green on a
        # graph that quietly ran through `stream`.
        snap = adapter_counters()
        assert snap["adapters.langgraph.active.pregel.astream"] == 1
        assert snap["adapters.langgraph.active.runner.arun_with_retry"] == 1
        assert "adapters.langgraph.active.pregel.stream" not in snap
        assert steps(live.spans)[0].parent_span_id == runs(live.spans)[0].context.span_id
    finally:
        live.teardown()


# --------------------------------------------------------------------------
# `Command`
# --------------------------------------------------------------------------


def test_a_parent_command_from_a_subgraph_leaves_no_error_status_anywhere(installed):
    """`ParentCommand` is a jump the host SAW SUCCEED, so nothing may read ERROR.

    It reaches the seams as an exception like every other piece of LangGraph
    control flow, but unlike an interrupt the run continues and completes — the
    outer graph routes to the named sibling and returns a value. A span reading
    ERROR anywhere here would put a red node in the middle of a run that worked.
    """

    def hop(state):
        return Command(goto="sibling", graph=Command.PARENT, update={"trail": ["hop"]})

    inner = StateGraph(Trail)
    inner.add_node("hop", hop)
    inner.add_edge(START, "hop")
    sub = inner.compile()
    sub.name = "Inner"

    outer = StateGraph(Trail)
    outer.add_node("child", sub)
    outer.add_node("sibling", lambda s: {"trail": ["sibling"]})
    outer.add_edge(START, "child")
    outer.add_edge("sibling", END)
    app = outer.compile(checkpointer=InMemorySaver())
    app.name = "ParentCmd"

    out = app.invoke({"trail": []}, {"configurable": {"thread_id": "p1"}})
    assert out["trail"] == ["hop", "sibling"]
    assert [s for s in installed.spans if s.status is StatusCode.ERROR] == []
    assert [s for s in installed.spans if s.error_type is not None] == []
    assert outcomes(installed.spans) == [
        ("execute_step hop", StatusCode.UNSET, None),
        ("invoke_workflow Inner", StatusCode.UNSET, None),
        ("execute_step child", StatusCode.OK, None),
        ("execute_step sibling", StatusCode.OK, None),
        ("invoke_workflow ParentCmd", StatusCode.OK, None),
    ]


def test_a_command_goto_from_an_ordinary_node_emits_no_handoff_span(installed):
    """A routing decision is not an agent transition, and gets no span of its own.

    `SpanIntent.HANDOFF` exists and is deliberately NOT used here: a `Command`
    is one graph deciding its next node, where a handoff is one agent giving
    work to another. Emitting one per `goto` would put a marker span between
    every pair of nodes in every routed graph, and the routing is already
    legible from the two `execute_step` spans and their order.
    """

    def router(state):
        return Command(goto="b", update={"trail": ["router"]})

    g = StateGraph(Trail)
    g.add_node("router", router)
    g.add_node("b", lambda s: {"trail": ["b"]})
    g.add_edge(START, "router")
    g.add_edge("b", END)
    app = g.compile()
    app.name = "Goto"

    assert app.invoke({"trail": []})["trail"] == ["router", "b"]
    assert [s.name for s in installed.spans] == [
        "execute_step router",
        "execute_step b",
        "invoke_workflow Goto",
    ]
    assert named(installed.spans, "handoff") == []
    assert [ln for s in installed.spans for ln in s.links] == []


# --------------------------------------------------------------------------
# static interrupts
# --------------------------------------------------------------------------


def _two_node_graph(**compile_kwargs):
    g = StateGraph(Trail)
    g.add_node("n0", lambda s: {"trail": ["n0"]})
    g.add_node("n1", lambda s: {"trail": ["n1"]})
    g.add_edge(START, "n0")
    g.add_edge("n0", "n1")
    g.add_edge("n1", END)
    app = g.compile(checkpointer=InMemorySaver(), **compile_kwargs)
    app.name = "Static"
    return app


def test_interrupt_before_ships_a_clean_run_with_no_node_spans_at_all(installed):
    """A KNOWN limitation, pinned so it stays known rather than surprising.

    `interrupt_before` pauses the graph before its first node executes, so no
    task ever reaches the node seam — `active.runner.run_with_retry` is absent
    from the snapshot entirely, not zero-valued — and the run ships `OK` with a
    normal duration. Nothing on the wire distinguishes "paused before its first
    node" from "ran nothing", because LangGraph raises nothing and returns a
    state. Recording that here is cheaper than rediscovering it from a
    dashboard.
    """
    out = _two_node_graph(interrupt_before=["n0"]).invoke(
        {"trail": []}, {"configurable": {"thread_id": "b1"}}
    )
    assert out == {"trail": []}
    shipped = runs(installed.spans)
    assert len(shipped) == 1
    assert (shipped[0].status, shipped[0].error_type) == (StatusCode.OK, None)
    assert steps(installed.spans) == []
    assert markers_of(shipped[0]) == (), "nothing was lost — there was nothing to lose"
    assert "adapters.langgraph.active.runner.run_with_retry" not in adapter_counters()


def test_interrupt_after_is_an_ordinary_two_pass_resume(installed):
    """The CONTRAST, and what makes the case above a limitation rather than a bug.

    `interrupt_after` lets the node run before pausing, so both passes are
    ordinary runs with ordinary node spans — two traces, one per entry into
    `stream`, which is the honest shape for two separate host calls. The pair
    is what shows the silence above belongs to the framework's pause point and
    not to the seam.
    """
    app = _two_node_graph(interrupt_after=["n0"])
    cfg = {"configurable": {"thread_id": "a1"}}
    assert app.invoke({"trail": []}, cfg)["trail"] == ["n0"]
    assert app.invoke(None, cfg)["trail"] == ["n0", "n1"]
    assert outcomes(installed.spans) == [
        ("execute_step n0", StatusCode.OK, None),
        ("invoke_workflow Static", StatusCode.OK, None),
        ("execute_step n1", StatusCode.OK, None),
        ("invoke_workflow Static", StatusCode.OK, None),
    ]
    assert len(traces(installed.spans)) == 2, "two host calls are two traces"
    assert adapter_counters()["adapters.langgraph.active.runner.run_with_retry"] == 2

    # Each node under ITS OWN pass, by id. Two traces is also what "both nodes
    # landed under pass one and pass two shipped an empty run" produces, and the
    # outcome list above reads identically either way — the resume's node has to
    # be shown hanging off the resume's run.
    n0, run1, n1, run2 = installed.spans
    assert n0.parent_span_id == run1.context.span_id
    assert n1.parent_span_id == run2.context.span_id
    assert run1.context.trace_id != run2.context.trace_id
    assert n1.context.trace_id == run2.context.trace_id, (
        "the resumed node belongs to the resumed run's trace, not the first pass's"
    )


# --------------------------------------------------------------------------
# the functional API
# --------------------------------------------------------------------------


def _functional_workflow():
    @task
    def double(x: int) -> int:
        return x * 2

    @entrypoint(checkpointer=InMemorySaver())
    def wf(x: int) -> list:
        futures = [double(i) for i in (1, 2)]
        return [f.result() for f in futures]

    return wf


def test_the_functional_api_reaches_the_same_node_seam(installed):
    """`@entrypoint` and `@task` are Pregel nodes, so they need no second seam.

    The functional API looks nothing like a `StateGraph` from the outside and
    is the shape most likely to be assumed unsupported. It is the same seam:
    the entrypoint arrives as a node, each `@task` arrives as a node, and the
    two sibling tasks are again identical in name and index with the task id
    as their only discriminator — the third distinct workload after `Send` and
    `create_react_agent` that produces that shape.
    """
    _functional_workflow().invoke(3, {"configurable": {"thread_id": "f1"}})
    node_spans = steps(installed.spans)
    assert [s.name for s in node_spans] == [
        "execute_step double",
        "execute_step double",
        "execute_step wf",
    ]
    tasks = [extra_of(s) for s in node_spans[:2]]
    assert {t["wardex.step.name"] for t in tasks} == {"double"}
    assert {t["wardex.step.index"] for t in tasks} == {0}, "`0` is a real index here"
    assert len({t["wardex.step.task_id"] for t in tasks}) == 2
    # The tree, by id. `@task` submits through the executor's `copy_context()`,
    # so its span nests under the entrypoint that called it rather than landing
    # flat under the run — which is the whole claim this adapter makes and the
    # thing a name-only assertion cannot see.
    assert {parent_name(installed.spans, s) for s in node_spans[:2]} == {"execute_step wf"}
    assert parent_name(installed.spans, node_spans[2]) == "invoke_workflow LangGraph"


def test_nothing_derives_from_a_task_path_because_a_task_path_is_nested(installed):
    """The reason `_node_extras` reads four keys and none of them is `path`.

    A `@task`'s `task.path` is a NESTED tuple, so `",".join(task.path)` — the
    obvious spelling next to the one `wardex.step.trigger` uses — raises
    `TypeError` for every `@task` while working for every ordinary node. Inside
    `describe` that is not a missing key: the guard catches it, `enter`
    abandons the unit, and the span DISAPPEARS. This asserts the outcome
    (spans present, guard never tripped) beside the fact that makes it fragile.
    """
    seen: list[tuple[str, tuple]] = []
    original = lg_mod._node_extras

    def spy(task_obj, step):
        seen.append((task_obj.name, task_obj.path))
        return original(task_obj, step)

    lg_mod._node_extras = spy
    try:
        _functional_workflow().invoke(3, {"configurable": {"thread_id": "f2"}})
    finally:
        lg_mod._node_extras = original

    paths = {name: path for name, path in seen}
    assert paths["wf"] == ("__pregel_pull", "wf")
    assert paths["double"][:2] == ("__pregel_push", ("__pregel_pull", "wf"))
    try:
        ",".join(paths["double"])
        raise AssertionError("a nested path was supposed to defeat a join")
    except TypeError:
        pass
    assert len(steps(installed.spans)) == 3, "all three spans survived describe"
    assert "adapters.langgraph.describe_node_extras" not in adapter_counters()
