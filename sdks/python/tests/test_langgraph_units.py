"""§9.2 — the merge gate. The one file that must fail on a collapsed causal tree.

The SHARED half of this is a conformance suite now
(`wardex_sdk.testing.conformance`, driven for this adapter by
`test_langgraph_conformance.py`), and what stays here is what that suite does
not reach: the async twin of every claim, the `submit`-hook negative control
that separates a read from a guess, and the counter cross-checks that are
specific to this adapter's four `confirm_active` sites.

It is deliberately two halves that fail independently.

The TIER half reads every edge's `(strategy, confidence, limitations)` off the
shipped span. It catches a run entry nobody wrapped, a node seam patched on
`pregel._retry` instead of `pregel._runner`, and a guess that quietly replaced a
read.

The ID half pairs each leaf with the node that made it and asserts
`parent_span_id` by identity. It exists because the tier half CANNOT see a total
collapse: a tree in which every `execute_step` span is real, is the run's child,
and is decorative — every leaf parented straight to the run — reads
`unit_active` / 1.0 / no marker on every single span, ships one trace, and
counts correctly. `test_the_tier_half_alone_cannot_see_a_collapsed_tree` builds
that tree on purpose and proves the id half is what catches it. A gate nobody has
seen fail is not a gate.
"""

from __future__ import annotations

import asyncio
import concurrent.futures

import pytest
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from test_langgraph_adapter import (
    Installed,
    TrailState,
    _clean_scope,  # noqa: F401 — the autouse determinism fixture this module needs too
    adapter_counters,
    assembly_counters,
    call,
    chain,
    edge_of,
    fanout,
    installed,  # noqa: F401 — a FIXTURE; importing it is what makes it usable here,
    # and every test below re-flags it as F811 because its own parameter shadows it
    leaf_span,
    parent_name,
    pure_add,
    runs,
    steps,
    tool_graph,
    tools,
    traces,
)
from wardex_sdk._assembly import Limitation, ParentSource, counters

#: What an edge wardex READ looks like on the wire. Spelled once: every place it
#: appears below is asserting the same three-part fact, and a gate that spells it
#: per site is a gate whose sites can drift apart.
_READ = (ParentSource.UNIT_ACTIVE, 1.0, ())
_ROOT = (ParentSource.TRACE_ROOT, 1.0, ())
_GUESSED = (ParentSource.UNIT_SOLE, 0.5, (Limitation.UNIT_INFERRED_SOLE,))


def exactly(spans, name):
    """The one span with this EXACT name.

    Exact rather than `named()`'s prefix test: `execute_step n1` is a prefix of
    `execute_step n10`, so a prefix match would silently pair a leaf with the
    wrong node the first time this gate is pointed at a graph wide enough to
    matter.
    """
    found = [s for s in spans if s.name == name]
    assert len(found) == 1, f"expected exactly one {name!r}, got {[s.name for s in found]}"
    return found[0]


def leaves(spans):
    """The stand-in wire spans, told apart from real tool spans by subject.

    `leaf_span` opens a CALL at `EXECUTE_TOOL`, which is what the byte seam's
    parent resolution sees — and which also means `tools()` cannot distinguish
    it from an `execute_tool` span the adapter itself opened.
    """
    return [s for s in spans if s.name.startswith("execute_tool leaf-")]


# --------------------------------------------------------------------------
# 1. the baseline
# --------------------------------------------------------------------------


def _seams():
    from langgraph.prebuilt.tool_node import ToolNode as TN
    from langgraph.pregel import _runner
    from langgraph.pregel import main as pregel_mod

    return (
        pregel_mod.Pregel.stream,
        pregel_mod.Pregel.astream,
        _runner.run_with_retry,
        _runner.arun_with_retry,
        TN._run_one,
        TN._arun_one,
    )


def test_the_same_graphs_produce_nothing_without_the_adapter():
    """The gate's zero point: every number below is the adapter's doing.

    Without this the whole file could be measuring spans some other installed
    adapter, or an earlier test's leaked patch, happens to emit for the same
    workload — and the counts would still come out right.

    The client is driven through a LIVE install first, and the emptiness is
    asserted on that same object afterwards. A bare `RecordingClient()` nobody
    ever wired up is empty whatever the adapter does — the assertion reads as if
    it measured something and cannot fail — so the wiring has to be proven
    before its silence means anything.
    """
    before = _seams()
    live = Installed()
    try:
        chain(live.ctx, 3, name="Wired", leaves=False).invoke({"trail": []})
        assert len(runs(live.spans)) == 1, "this client really does receive spans"
        assert len(steps(live.spans)) == 3
    finally:
        live.teardown()

    assert _seams() == before, "teardown must put all six seams back by identity"
    live.client.spans.clear()
    counters.reset()

    # the identical workloads, now with nothing installed
    chain(None, 3, name="Bare", leaves=False).invoke({"trail": []})
    tool_graph([call("pure_add", {"a": 1, "b": 2}, "c1")], name="BareTools").invoke(
        {"messages": []}
    )
    assert _seams() == before, "no seam may be patched with no adapter installed"
    assert runs(live.spans) == []
    assert steps(live.spans) == []
    assert tools(live.spans) == []
    assert live.spans == [], "a restored seam emits NOTHING, not merely nothing named"
    assert adapter_counters() == {}, (
        "a confirm_active site firing with nothing installed would mean a patch "
        "leaked out of an earlier test and this file is not measuring what it thinks"
    )


# --------------------------------------------------------------------------
# 2. the gate itself, split into the two halves that fail independently
# --------------------------------------------------------------------------


#: The four `confirm_active` sites the run and node seams declare, sync and async.
#: A workload drives exactly one pair; the other two must stay absent.
_ENTRY_SITES = (
    "pregel.stream",
    "pregel.astream",
    "runner.run_with_retry",
    "runner.arun_with_retry",
)


def assert_tier(
    spans,
    *,
    n_nodes,
    run_name,
    entry="pregel.stream",
    node_site="runner.run_with_retry",
):
    """Every edge, as a tier. Blind to a collapse — see the module docstring.

    The two site names are parameters rather than literals so that the ASYNC
    twin of the gate asserts the identical facts about the other three seams.
    Three of the six patch sites are async, and a gate that only ever drives
    `invoke()` leaves half the adapter measured by nothing at all.
    """
    assert len(traces(spans)) == 1, "one graph run is one trace"
    assert len(runs(spans)) == 1
    assert len(steps(spans)) == n_nodes
    assert len(leaves(spans)) == n_nodes, "one wire span per node body"
    assert len(spans) == 2 * n_nodes + 1, "and nothing else shipped"

    for node in steps(spans):
        assert edge_of(node) == _READ, f"{node.name}: a node edge must be READ, not guessed"
    for leaf in leaves(spans):
        assert edge_of(leaf) == _READ, f"{leaf.name}: the context reached the node body"
    assert edge_of(exactly(spans, f"invoke_workflow {run_name}")) == _ROOT

    for span in spans:
        assert span.capture_integrity is None, (
            f"{span.name} carries {span.capture_integrity}; a healthy run has "
            "nothing to report, and a marker here means wardex lost something"
        )
    assert assembly_counters() == {}, (
        "an assembly counter is only ever bumped when something failed to resolve"
    )

    snap = adapter_counters()
    assert snap[f"adapters.langgraph.active.{entry}"] == 1
    # EQUAL to the span count, not a second literal: the counter is a cross-check
    # on `len(steps)`, and two literals drift apart the first time the workload
    # changes.
    assert snap[f"adapters.langgraph.active.{node_site}"] == len(steps(spans))
    # And the OTHER twin's sites stayed at zero. A wrapper installed on both the
    # sync and the async spelling of one seam would double-count here while every
    # span above still read correctly.
    for site in _ENTRY_SITES:
        if site not in (entry, node_site):
            assert f"adapters.langgraph.active.{site}" not in snap, (
                f"{site} fired on a workload that never touches it"
            )


def assert_chain_by_id(spans, *, n_nodes, run_name):
    """The half that matters most: `leaf-n{i}` hangs off `n{i}`, by span id.

    THAT node's id, not "some step's". A tree where every leaf is parented to
    the run and the node spans survive as decoration passes `assert_tier`
    completely — same tier, same confidence, same marker set, same trace, same
    counts — and is the shape a langgraph release that stops copying the context
    at task submit would produce.
    """
    run = exactly(spans, f"invoke_workflow {run_name}")
    for i in range(n_nodes):
        node = exactly(spans, f"execute_step n{i}")
        leaf = exactly(spans, f"execute_tool leaf-n{i}")
        assert leaf.parent_span_id == node.context.span_id, (
            f"leaf-n{i} parented to {parent_name(spans, leaf)!r}, not to its own node"
        )
        assert node.parent_span_id == run.context.span_id, (
            f"n{i} parented to {parent_name(spans, node)!r}, not to the run"
        )


def test_a_three_node_chain_ships_one_read_tree(installed):  # noqa: F811 — see the import
    """THE MERGE GATE. Both halves, on the workload the adapter exists for."""
    chain(installed.ctx, 3, name="Gate").invoke({"trail": []})
    assert_tier(installed.spans, n_nodes=3, run_name="Gate")
    assert_chain_by_id(installed.spans, n_nodes=3, run_name="Gate")


def async_chain(ctx, n_nodes: int, *, name: str, leaves: bool = True):
    """The harness's `chain`, with `async def` node bodies. Same names, same leaves.

    Deliberately identical in shape so that `assert_tier` and
    `assert_chain_by_id` are reused verbatim: the claim is that the async half
    of the adapter produces the SAME tree, and re-spelling the assertions for it
    is how the two drift apart.
    """

    def make(i: int):
        async def node(state: TrailState) -> TrailState:
            if leaves:
                leaf_span(ctx, f"leaf-n{i}")
            return {"trail": [f"n{i}"]}

        return node

    g = StateGraph(TrailState)
    prev = START
    for i in range(n_nodes):
        g.add_node(f"n{i}", make(i))
        g.add_edge(prev, f"n{i}")
        prev = f"n{i}"
    g.add_edge(prev, END)
    app = g.compile()
    app.name = name
    return app


def test_an_async_chain_ships_the_same_read_tree(installed):  # noqa: F811 — see the import
    """THE GATE'S ASYNC HALF. `Pregel.astream` + `arun_with_retry`, both halves.

    Three of the six patch sites are async, and every other assertion in this
    suite drives `invoke()`. Their coverage was that `test_six_patches_...`
    sees them REPLACED — which a wrapper that opens the node span outside the
    body, forwards the wrong original, or loses the context across
    `asyncio.create_task` passes without a murmur. The async executor propagates
    through `context=copy_context()` at `_executor.py:164` rather than through
    the sync `ctx = copy_context()` at `:64`, so it is a genuinely different
    mechanism carrying the same claim, and nothing was measuring it.

    Identical expectations to the sync gate, down to the helper: same tiers,
    same per-node id chain, same counter/span-count equality.
    """
    asyncio.run(async_chain(installed.ctx, 3, name="AsyncGate").ainvoke({"trail": []}))
    assert_tier(
        installed.spans,
        n_nodes=3,
        run_name="AsyncGate",
        entry="pregel.astream",
        node_site="runner.arun_with_retry",
    )
    assert_chain_by_id(installed.spans, n_nodes=3, run_name="AsyncGate")


def test_an_async_tool_call_adopts_the_wire_span_its_own_body_issued(installed):  # noqa: F811
    """`ToolNode._arun_one` — the LAST of the six seams with no tree assertion.

    `_mk_arun_one` is the one wrapper whose whole body is unmeasured by every
    other test in this suite: a version that opened the CALL unit and awaited
    the host OUTSIDE it would ship two tool spans with the right names, the
    right ids, the right counts and the right tiers, while the request the tool
    body issued landed on the `tools` node. Only the id chain sees that, and
    only an async workload reaches this wrapper.
    """
    ctx = installed.ctx

    @tool
    async def deep_add(a: int, b: int) -> int:
        """Add two numbers, issuing one request on the way."""
        leaf_span(ctx, "leaf-deep")
        return a + b

    app = tool_graph(
        [call("deep_add", {"a": 1, "b": 2}, "c1"), call("pure_add", {"a": 3, "b": 4}, "c2")],
        tools_list=[deep_add, pure_add],
        name="AsyncTools",
    )
    asyncio.run(app.ainvoke({"messages": []}))

    spans = installed.spans
    called = [s for s in tools(spans) if not s.name.startswith("execute_tool leaf-")]
    assert len(called) == 2, "one span per tool call, not per executor attempt"
    for span in called:
        assert edge_of(span) == _READ
    assert len(traces(spans)) == 1

    leaf = exactly(spans, "execute_tool leaf-deep")
    deep = exactly(spans, "execute_tool deep_add")
    pure = exactly(spans, "execute_tool pure_add")
    node = exactly(spans, "execute_step tools")
    run = exactly(spans, "invoke_workflow AsyncTools")
    assert leaf.parent_span_id == deep.context.span_id, (
        f"leaf-deep parented to {parent_name(spans, leaf)!r}; the async tool span "
        "exists but did not adopt the request its own body issued"
    )
    assert deep.parent_span_id == node.context.span_id
    assert pure.parent_span_id == node.context.span_id
    assert node.parent_span_id == run.context.span_id
    assert run.parent_span_id is None

    snap = adapter_counters()
    assert snap["adapters.langgraph.active.toolnode.arun_one"] == 2
    assert "adapters.langgraph.active.toolnode.run_one" not in snap, (
        "the async path must not reach the sync tool wrapper"
    )


# --------------------------------------------------------------------------
# 3. the gate's own proof of life
# --------------------------------------------------------------------------


def test_the_tier_half_alone_cannot_see_a_collapsed_tree(installed):  # noqa: F811 — see the import
    """Build the collapse deliberately and watch exactly one half catch it.

    The wrapper opens the node's span with an EMPTY body and runs the node
    AFTER the `with` has closed — so `execute_step` spans still exist, are still
    the run's children, and carry no work. Every leaf then latches the RUN,
    because the run is what is ambient by the time the node body runs.

    This is not a hypothetical shape. It is what the adapter's own docstring
    names as the failure it exists to avoid — the node seam patched on
    `pregel._retry` (the producer) instead of `pregel._runner` (the consumer) —
    and it is why the pristine function is reachable at `_retry` to build it.
    """
    from langgraph.pregel import _retry, _runner

    import wardex_sdk._adapters._langgraph as mod

    def _empty(task, retry_policy, *a, **k):
        return None

    decorative = mod._mk_run_with_retry(_empty, installed.adapter, START)

    def collapsed(task, retry_policy, *args, **kwargs):
        decorative(task, retry_policy, *args, **kwargs)  # the span, with nothing in it
        return _retry.run_with_retry(task, retry_policy, *args, **kwargs)  # the work, outside

    saved = _runner.run_with_retry
    _runner.run_with_retry = collapsed
    try:
        chain(installed.ctx, 3, name="Collapsed").invoke({"trail": []})
    finally:
        _runner.run_with_retry = saved

    spans = installed.spans
    assert_tier(spans, n_nodes=3, run_name="Collapsed")  # every tier assertion still passes
    for leaf in leaves(spans):
        assert parent_name(spans, leaf) == "invoke_workflow Collapsed", (
            "the collapse this test builds must actually be a collapse"
        )
    with pytest.raises(AssertionError, match="not to its own node"):
        assert_chain_by_id(spans, n_nodes=3, run_name="Collapsed")


# --------------------------------------------------------------------------
# 4. the tool half
# --------------------------------------------------------------------------


def test_a_tool_call_adopts_the_wire_span_its_own_body_issued(installed):  # noqa: F811 — see the import
    """The full chain by id: leaf < execute_tool < execute_step < invoke_workflow.

    Counting tool spans is not enough. A tool span that exists, sits under the
    right step and does NOT adopt the request its own body issued is a broken
    tree that passes every count and every tier — the request lands on the
    `tools` node instead, and a consumer reading "which call did this cost" gets
    the wrong answer with full confidence.
    """
    ctx = installed.ctx

    @tool
    def deep_add(a: int, b: int) -> int:
        """Add two numbers, issuing one request on the way."""
        leaf_span(ctx, "leaf-deep")
        return a + b

    tool_graph(
        [call("deep_add", {"a": 1, "b": 2}, "c1"), call("pure_add", {"a": 3, "b": 4}, "c2")],
        tools_list=[deep_add, pure_add],
        name="Tools",
    ).invoke({"messages": []})

    spans = installed.spans
    called = [s for s in tools(spans) if not s.name.startswith("execute_tool leaf-")]
    assert len(called) == 2, "one span per tool call, not per executor attempt"
    for span in called:
        assert edge_of(span) == _READ
    assert len(traces(spans)) == 1

    leaf = exactly(spans, "execute_tool leaf-deep")
    deep = exactly(spans, "execute_tool deep_add")
    pure = exactly(spans, "execute_tool pure_add")
    node = exactly(spans, "execute_step tools")
    run = exactly(spans, "invoke_workflow Tools")
    assert leaf.parent_span_id == deep.context.span_id, (
        f"leaf-deep parented to {parent_name(spans, leaf)!r}; the tool span exists "
        "but did not adopt the request its own body issued"
    )
    assert deep.parent_span_id == node.context.span_id
    assert pure.parent_span_id == node.context.span_id
    assert node.parent_span_id == run.context.span_id
    assert run.parent_span_id is None


# --------------------------------------------------------------------------
# 5. the negative control
# --------------------------------------------------------------------------


def test_a_submit_hook_that_drops_the_context_degrades_to_the_sole_run(installed):  # noqa: F811 — see the import
    """Without this the two tiers are distinguishable only by the spec.

    LangGraph reads a custom task submitter out of the config —
    `CONFIG_KEY_RUNNER_SUBMIT = "__pregel_runner_submit"`
    (`langgraph/_internal/_constants.py:66`), read at `pregel/main.py:2925`
    (sync) and `:3379` (async) as `config[CONF].get(...)`. It is stored as a
    ZERO-ARG callable returning the submitter — the default is
    `weakref.WeakMethod(loop.submit)` and `_runner.py` derefs it with
    `self.submit()(...)` — and the submitter the default resolves to is
    `_executor.BackgroundExecutor.submit`, whose `ctx = copy_context()` at
    `_executor.py:64` is the ONE line the whole `unit_active` tier rests on.

    Supply one that omits it and the node seam finds nothing ambient. It does
    not become a trace root and it does not lie: `Fallback.SOLE_LIVE_RUN` takes
    the one live SESSION this adapter owns, and the edge says so in all three
    fields at once — 0.5 with `UNIT_INFERRED_SOLE` — where the honest read is
    1.0 with nothing.

    Two shapes are excluded on purpose. The 1-task FAST PATH
    (`_runner.py:203`) calls `run_with_retry` inline and never touches `submit`,
    which is why the `fan` node below keeps its 1.0 edge and why a naive version
    of this test measures nothing at all. And a SUBGRAPH would open a second
    live SESSION, making `sole_live` return None and the whole run degrade to
    orphans — passing for the wrong reason.
    """
    width = 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pool:

        def submit(
            fn,
            *args,
            __name__=None,
            __cancel_on_exit__=False,
            __reraise_on_exit__=True,
            __next_tick__=False,
            **kwargs,
        ):
            return pool.submit(fn, *args, **kwargs)  # no copy_context()

        fanout(installed.ctx, width, name="Leaky").invoke(
            {"trail": []},
            config={"configurable": {"__pregel_runner_submit": lambda: submit}},
        )

    spans = installed.spans
    assert len(traces(spans)) == 1, (
        "the guess still lands inside the run, so the run does not shatter — "
        "which is the whole reason 0.5 and a marker are the only difference"
    )
    assert len(steps(spans)) == width + 1  # the `fan` node plus the fanned-out ones

    for i in range(width):
        node = exactly(spans, f"execute_step w{i}")
        assert edge_of(node) == _GUESSED, (
            f"w{i} must say it GUESSED: 0.5 and UNIT_INFERRED_SOLE, not a bare 1.0"
        )
        assert node.parent_span_id == exactly(spans, "invoke_workflow Leaky").context.span_id
        # The guess happens once, at the seam that lost the context. Everything
        # opened INSIDE the node body reads the tier again at full confidence.
        leaf = exactly(spans, f"execute_tool leaf-w{i}")
        assert edge_of(leaf) == _READ
        assert leaf.parent_span_id == node.context.span_id

    assert edge_of(exactly(spans, "execute_step fan")) == _READ, (
        "the 1-task fast path bypasses `submit` entirely, so this node is "
        "unaffected — and a test written on a graph with only this shape would "
        "assert nothing"
    )
