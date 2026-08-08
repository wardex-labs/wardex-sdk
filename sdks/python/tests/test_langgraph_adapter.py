"""LangGraph adapter — install, teardown, and the shape rules the module lives by.

The tree-shape and vocabulary assertions are in `test_langgraph_units.py`, which
imports this file's harness. The split mirrors the Agent SDK adapter's two files
and exists for the same reason: the builders below are the expensive part and
both files need them.

**Real graphs, never a fake `Pregel`.** langgraph is a dev dependency, a real
compiled `StateGraph` is cheaper to write than a double, and the failure shape
this adapter exists to avoid — a patch installed on `pregel._retry` instead of
`pregel._runner`, which passes every install-level assertion while emitting zero
node spans in production — is *precisely* a test double that differs from
production. So nothing here asserts that a patch was installed; the tests assert
the spans that only a live seam can produce.
"""

from __future__ import annotations

import ast
import pathlib
from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from wardex_sdk._enums import StatusCode
from wardex_sdk._hub import reset_for_test
from wardex_sdk.adapters._context import Placement
from wardex_sdk.adapters._langgraph import LangGraphAdapter
from wardex_sdk.adapters._registry import AdapterRegistry
from wardex_sdk.assembly import SpanIntent, UnitKind, counters
from wardex_sdk.assembly._diag import reset_reports_for_test
from wardex_sdk.assembly._units import _ambient_unit

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "wardex_sdk"
_MODULE = _SRC / "adapters" / "_langgraph.py"


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------


class RecordingClient:
    """A client double that keeps every span the registry emits."""

    config = None

    def __init__(self) -> None:
        self.spans: list[Any] = []

    def capture_span(self, span) -> None:
        self.spans.append(span)

    def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _clean_scope():
    """Determinism, and three things are process-global rather than one.

    `_REPORTED` is, so without the reset the ORDER of tests decides what a later
    `report_once` assertion sees. And `CONTROL_FLOW` is a CLASSVAR — that is its
    whole design, so that one value serves every path — which means one
    successful `install()` populates it for every instance and every later test
    in the process. Benign in production (the value is only ever read through an
    installed adapter's own context, and an adapter that declined has no `ctx`
    and therefore no spans), but it would let a decline test pass on a value an
    earlier test wrote.

    `reset_for_test()` is the third, and it is the one a langgraph suite needs
    that others do not: a test that ABANDONS a `stream()` leaves the run's fork
    installed on this carrier, because the generator's `finally` can only take
    it down when the generator is finalized. Without the hub reset that fork
    outlives the test and becomes the ambient parent of everything after it.
    """
    reset_for_test()
    token = _ambient_unit.set(None)
    counters.reset()
    reset_reports_for_test()
    LangGraphAdapter.CONTROL_FLOW = ()
    yield
    _ambient_unit.reset(token)
    reset_for_test()
    counters.reset()
    reset_reports_for_test()
    LangGraphAdapter.CONTROL_FLOW = ()


class Installed:
    """An installed adapter plus the client it emits into.

    Installed through `AdapterRegistry.install`, never by hand: `CONTROL_FLOW`
    reaches `_run` through a reader the REGISTRY binds, so an adapter given a
    hand-built `AdapterContext` passes control-flow assertions under a wiring
    that is dead in production.
    """

    def __init__(self) -> None:
        self.client = RecordingClient()
        self.registry = AdapterRegistry()
        self.adapter = LangGraphAdapter()
        self.registry.install(self.adapter, self.client)

    @property
    def ctx(self):
        return self.adapter._ctx

    @property
    def spans(self):
        return self.client.spans

    def teardown(self) -> None:
        self.registry.uninstall_all()


@pytest.fixture
def installed():
    live = Installed()
    try:
        yield live
    finally:
        live.teardown()


# -- reading the wire -------------------------------------------------------


def edge_of(span):
    """The edge as it SHIPS, off the emitted span rather than off a handle.

    `capture_integrity` is None on a span with nothing to report, which is the
    shape most assertions here are looking for — so it is normalized to `()`
    and the *presence* of a marker is asserted separately where it matters.
    """
    c = span.correlation
    integrity = span.capture_integrity
    return c.strategy, c.confidence, tuple(integrity.limitations) if integrity else ()


def named(spans, prefix):
    return [s for s in spans if s.name.startswith(prefix)]


def runs(spans):
    return named(spans, "invoke_workflow")


def steps(spans):
    return named(spans, "execute_step")


def tools(spans):
    return named(spans, "execute_tool")


def traces(spans):
    return {s.context.trace_id for s in spans}


def extra_of(span) -> dict:
    return dict(span.extra)


def parent_name(spans, span) -> str | None:
    by_id = {s.context.span_id: s for s in spans}
    parent = span.parent_span_id
    if parent is None:
        return None
    found = by_id.get(parent)
    return found.name if found is not None else "<MISSING>"


def adapter_counters() -> dict:
    return {k: v for k, v in counters.snapshot().items() if k.startswith("adapters.langgraph.")}


def assembly_counters() -> dict:
    return {k: v for k, v in counters.snapshot().items() if k.startswith("assembly.")}


# -- graph builders ---------------------------------------------------------


def _append(a: list, b: list) -> list:
    return a + b


class TrailState(TypedDict):
    trail: Annotated[list, _append]


def leaf_describe(scope) -> None:
    from wardex_sdk.assembly import ToolAttributes

    scope.draft.set_tool(ToolAttributes(name="leaf"))


def leaf_span(ctx, subject: str) -> None:
    """A stand-in for a byte-seam span, opened inside a node or tool body.

    §9.1 forbids an in-process HTTP server of any kind — `interceptors/_seam.py`
    keys its connection table on `id(obj)` and never removes an entry, so a
    recycled CPython address hands a live socket a dead socket's latched gate
    and requests vanish with no counter. A unit opened through `ctx.enter`
    resolves its parent through the same ambient the byte seam latches, so the
    edge under test is the same one.
    """
    with ctx.enter(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        subject=subject,
        describe=leaf_describe,
    ):
        pass


def chain(ctx, n_nodes: int, *, name: str = "Chain", leaves: bool = True):
    """A linear graph of `n_nodes`, each optionally opening one leaf span.

    The leaf's subject carries the node's own name so a test can pair each leaf
    with the node that made it and assert the chain BY ID — a tier assertion
    (`unit_active`/1.0) cannot see a tree collapse in which every leaf is
    parented to the RUN and the node spans survive as decoration.
    """

    def make(i: int):
        def node(state: TrailState) -> TrailState:
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


def fanout(ctx, width: int, *, name: str = "FanOut", leaves: bool = True):
    """A FLAT fan-out — deliberately not a subgraph.

    A subgraph opens a second live `SESSION`, which makes `sole_live(SESSION)`
    return None for its whole duration; a fallback test written on one would
    silently measure the `Fallback.NONE` behaviour and pass for the wrong
    reason.
    """

    def make(i: int):
        def node(state: TrailState) -> TrailState:
            if leaves:
                leaf_span(ctx, f"leaf-w{i}")
            return {"trail": [f"w{i}"]}

        return node

    g = StateGraph(TrailState)
    g.add_node("fan", lambda s: {"trail": ["fan"]})
    g.add_edge(START, "fan")
    for i in range(width):
        g.add_node(f"w{i}", make(i))
        g.add_edge("fan", f"w{i}")
        g.add_edge(f"w{i}", END)
    app = g.compile()
    app.name = name
    return app


@tool
def pure_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def tool_graph(tool_calls, tools_list=None, *, name: str = "HandBuilt"):
    """A model node emitting `tool_calls`, then a hand-built `ToolNode`.

    A hand-built `ToolNode` on a plain edge is one of the two shapes where the
    NODE seam sees ONE task for N tool calls — which is exactly why the tool
    seam sits below the fan-out rather than reading the node's input.
    """

    def model_node(state):
        return {"messages": [AIMessage(content="", tool_calls=list(tool_calls))]}

    g = StateGraph(MessagesState)
    g.add_node("model", model_node)
    g.add_node("tools", ToolNode(tools_list if tools_list is not None else [pure_add]))
    g.add_edge(START, "model")
    g.add_edge("model", "tools")
    g.add_edge("tools", END)
    app = g.compile()
    app.name = name
    return app


def call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# --------------------------------------------------------------------------
# install / uninstall
# --------------------------------------------------------------------------
#
# Registration, seam identity, install/uninstall idempotence and the two
# shutdown paths are NOT here any more. They are shared invariants — every
# adapter owes them and each was being proved twice, in two styles — so they
# live in `wardex_sdk.testing.conformance` and this adapter answers them in
# `test_langgraph_conformance.py`. What stays below is what is true of THIS
# adapter and no other: a six-patch surface that declines in two independent
# groups, and a probe with a specific idea of what it will patch.


def _originals():
    from langgraph.prebuilt.tool_node import ToolNode as TN
    from langgraph.pregel import _runner
    from langgraph.pregel import main as pregel_mod

    return {
        (pregel_mod.Pregel, "stream"): pregel_mod.Pregel.stream,
        (pregel_mod.Pregel, "astream"): pregel_mod.Pregel.astream,
        (_runner, "run_with_retry"): _runner.run_with_retry,
        (_runner, "arun_with_retry"): _runner.arun_with_retry,
        (TN, "_run_one"): TN._run_one,
        (TN, "_arun_one"): TN._arun_one,
    }


def test_control_flow_is_populated_only_after_a_successful_install():
    """Before `install()` the classvar is empty — an uninstalled adapter has no
    spans, so the value is unreachable — and after it, one name covers
    interrupts, drains and parent commands."""
    from langgraph.errors import GraphBubbleUp

    LangGraphAdapter.CONTROL_FLOW = ()
    live = Installed()
    try:
        assert type(live.adapter).CONTROL_FLOW == (GraphBubbleUp,)
    finally:
        live.teardown()


def test_installing_without_a_context_patches_nothing():
    """Every span this adapter emits goes through `ctx.enter`, so a `None` ctx
    installs NOTHING rather than installing patches that could only pass
    through."""
    before = _originals()
    adapter = LangGraphAdapter()
    adapter.install(None, None)
    assert _originals() == before
    assert adapter._installed is False


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------


def test_the_real_surface_passes_both_probes():
    from langgraph.prebuilt.tool_node import ToolNode as TN
    from langgraph.pregel import _runner
    from langgraph.pregel import main as pregel_mod
    from langgraph.types import PregelExecutableTask

    from wardex_sdk.adapters._langgraph import _surface_ok, _tool_surface_ok

    assert _surface_ok(pregel_mod, _runner, PregelExecutableTask)
    assert _tool_surface_ok(TN)


def test_a_reordered_signature_fails_the_probe():
    """Ordered positional NAMES, not set containment. The node wrapper forwards
    `task` and `retry_policy` positionally, so a reordered signature would hand
    the host its `retry_policy` as a `task` while set containment reported the
    surface intact."""
    from langgraph.pregel import main as pregel_mod
    from langgraph.types import PregelExecutableTask

    from wardex_sdk.adapters._langgraph import _surface_ok

    class Reordered:
        @staticmethod
        def run_with_retry(retry_policy, task, configurable=None): ...

        @staticmethod
        async def arun_with_retry(retry_policy, task, configurable=None): ...

    assert not _surface_ok(pregel_mod, Reordered, PregelExecutableTask)


def test_an_inherited_run_one_fails_the_tool_probe():
    """`in __dict__`, not `hasattr`: patching the class would otherwise shadow a
    base-class attribute the restore must not delete."""
    from wardex_sdk.adapters._langgraph import _tool_surface_ok

    class Base:
        def _run_one(self, call, *a, **k): ...
        async def _arun_one(self, call, *a, **k): ...

    class Derived(Base):
        pass

    assert _tool_surface_ok(Base)
    assert not _tool_surface_ok(Derived)


def test_a_pydantic_shaped_class_fails_the_probe():
    """`PatchSet` restores with `setattr`, which a Pydantic model would defeat."""
    from wardex_sdk.adapters._langgraph import _tool_surface_ok

    class BaseModel:
        pass

    class Modelled(BaseModel):
        def _run_one(self, call, *a, **k): ...
        async def _arun_one(self, call, *a, **k): ...

    assert not _tool_surface_ok(Modelled)


def test_an_unrecognized_surface_declines_loudly_and_patches_nothing(monkeypatch, capsys):
    import wardex_sdk.adapters._langgraph as mod

    before = _originals()
    monkeypatch.setattr(mod, "_surface_ok", lambda *a: False)
    live = Installed()
    try:
        assert _originals() == before
        assert adapter_counters().get("adapters.langgraph.unsupported_surface") == 1
        assert "surface unrecognized" in capsys.readouterr().err
        assert type(live.adapter).CONTROL_FLOW == (), (
            "a declined install must leave the classvar empty"
        )
    finally:
        live.teardown()


def test_a_partial_install_keeps_the_run_and_node_seams(monkeypatch, capsys):
    """PARTIAL: group 2 declines on its own. Run and node spans are correct
    without tool spans; the reverse is not true, so there is no
    tool-without-node mode."""
    from langgraph.prebuilt.tool_node import ToolNode as TN

    import wardex_sdk.adapters._langgraph as mod

    monkeypatch.setattr(mod, "_tool_surface_ok", lambda *a: False)
    before_tool = (TN._run_one, TN._arun_one)
    live = Installed()
    try:
        assert (TN._run_one, TN._arun_one) == before_tool
        assert adapter_counters().get("adapters.langgraph.unsupported_tool_surface") == 1
        assert "tool surface unrecognized" in capsys.readouterr().err
        tool_graph([call("pure_add", {"a": 1, "b": 2}, "c1")]).invoke({"messages": []})
        assert steps(live.spans), "the node seam must still be live"
        assert runs(live.spans), "the run seam must still be live"
        assert tools(live.spans) == [], "the tool seam declined, so no tool spans"
    finally:
        live.teardown()


def test_an_absent_tool_distribution_declines_silently(monkeypatch, capsys):
    import wardex_sdk.adapters._langgraph as mod

    monkeypatch.setattr(mod, "_import_toolnode", lambda: None)
    live = Installed()
    try:
        assert "unsupported_tool_surface" not in str(adapter_counters())
        assert capsys.readouterr().err == ""
    finally:
        live.teardown()


# --------------------------------------------------------------------------
# a failure the host reported without raising
# --------------------------------------------------------------------------


def test_a_converted_tool_failure_reads_error_and_costs_no_counter(installed):
    """The most common tool failure there is, and it never raises.

    LangGraph's DEFAULT `handle_tool_errors` turns a `ToolInvocationError` —
    the model calling a tool with arguments that do not validate — into a
    `ToolMessage(status="error")`. Nothing reaches `_run`'s exception handler,
    so before `record_failure` this span read `status=OK` with the failure
    legible only to a human reading `output_data`.

    Three things are asserted together, and the third is the one that matters
    most: the failing call is ERROR, its healthy SIBLING in the same fan-out is
    still OK, and `adapters.langgraph.tool_outcome` is ABSENT — a wardex
    degradation counter firing on the framework's single most ordinary
    behaviour would be worse than the gap it closes.
    """

    @tool
    def strict(a: int) -> int:
        """Needs an int."""
        return a * 2

    out = tool_graph(
        [
            call("strict", {"a": "not-an-int"}, "bad"),
            call("strict", {"a": 21}, "good"),
        ],
        tools_list=[strict],
        name="Converted",
    ).invoke({"messages": []})

    assert "messages" in out, "the host's graph run completed normally"
    tool_spans = {s.error_type: s for s in tools(installed.spans)}
    assert set(tool_spans) == {"tool_error", None}
    assert tool_spans["tool_error"].status is StatusCode.ERROR
    assert b"Error invoking tool" in tool_spans["tool_error"].output_data
    assert tool_spans[None].status is StatusCode.OK
    assert tool_spans[None].output_data == b"42"
    # the node and the run really did succeed, and must not be relabelled
    assert steps(installed.spans)[-1].status is StatusCode.OK
    assert runs(installed.spans)[0].status is StatusCode.OK
    assert "adapters.langgraph.tool_outcome" not in adapter_counters()


def test_a_raised_tool_failure_keeps_its_own_exception_type(installed):
    """A declaration is WEAKER than an exception and must never overwrite one.

    The raising path is the control for the case above: `record_failure` is
    consulted only when nothing was raised, so a tool that really crashed keeps
    `RuntimeError` rather than being relabelled `tool_error` by an adapter.
    """

    @tool
    def explode(x: int) -> int:
        """Always fails."""
        raise RuntimeError("tool exploded")

    with pytest.raises(RuntimeError):
        tool_graph([call("explode", {"x": 1}, "c1")], tools_list=[explode], name="Raised").invoke(
            {"messages": []}
        )

    span = tools(installed.spans)[0]
    assert span.status is StatusCode.ERROR
    assert span.error_type == "RuntimeError"


def test_control_flow_out_of_a_tool_is_not_overwritten_by_a_declared_failure(installed):
    """`interrupt()` inside a tool resolves to UNSET before the `finally` runs.

    The ordering is load-bearing: a suspension is not a failure, and the
    declaration path must not be able to turn one back into one. Nothing was
    returned on this path at all, so there is nothing to declare — this test
    exists to pin that the two mechanisms cannot collide.
    """
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import interrupt

    @tool
    def ask(q: str) -> str:
        """Ask a human."""
        return str(interrupt(q))

    g = StateGraph(MessagesState)
    g.add_node(
        "model",
        lambda s: {
            "messages": [AIMessage(content="", tool_calls=[call("ask", {"q": "ok?"}, "c1")])]
        },
    )
    g.add_node("tools", ToolNode([ask]))
    g.add_edge(START, "model")
    g.add_edge("model", "tools")
    g.add_edge("tools", END)
    app = g.compile(checkpointer=InMemorySaver())
    app.name = "AskFirst"
    out = app.invoke({"messages": []}, {"configurable": {"thread_id": "T-1"}})

    assert "__interrupt__" in out
    span = tools(installed.spans)[0]
    assert span.status is StatusCode.UNSET
    assert span.error_type is None


# --------------------------------------------------------------------------
# retries — the granularity claim that decided the node seam
# --------------------------------------------------------------------------


def test_a_retrying_node_is_ONE_span_at_status_ok():
    """`run_with_retry` wraps ALL attempts of one node task.

    This is the assertion that would catch a slip to any seam below it —
    `RunnableSeq.invoke`, `RunnableCallable.invoke`, the user function — every
    one of which fires once per ATTEMPT. It also states the difference from the
    callback-tree products in a form a reader can check: LangGraph starts a
    fresh langchain run per attempt with a fresh id, so a node that failed twice
    and then succeeded renders there as THREE sibling runs, two of them carrying
    `on_chain_error`, for one logical node that worked. Here it is one span, at
    `status=OK`, and the seam counter agrees with the span count.
    """
    from langgraph.types import RetryPolicy

    attempts = {"n": 0}

    def flaky(state: TrailState) -> TrailState:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("not yet")
        return {"trail": ["ok"]}

    live = Installed()
    try:
        g = StateGraph(TrailState)
        g.add_node(
            "flaky",
            flaky,
            retry_policy=RetryPolicy(
                max_attempts=3,
                retry_on=ValueError,
                initial_interval=0.001,
                backoff_factor=1.0,
            ),
        )
        g.add_edge(START, "flaky")
        g.add_edge("flaky", END)
        app = g.compile()
        app.name = "Flaky"
        out = app.invoke({"trail": []})

        assert attempts["n"] == 3, "the node body really did run three times"
        assert out["trail"] == ["ok"], "and the host's run really did succeed"
        node_spans = steps(live.spans)
        assert len(node_spans) == 1
        assert node_spans[0].status is StatusCode.OK
        assert node_spans[0].error_type is None
        assert adapter_counters()["adapters.langgraph.active.runner.run_with_retry"] == 1
        assert runs(live.spans)[0].status is StatusCode.OK
    finally:
        live.teardown()


def test_the_attempt_count_is_not_recoverable_from_the_span():
    """The cost of the choice above, pinned so it is known rather than assumed.

    Collapsing retries into one span buys a truthful `status` and loses the
    attempt count: the retry loop lives INSIDE `run_with_retry`, so the only way
    to observe attempts from this seam would be to substitute the host's own
    `task.proc` — mutating an object the framework owns, which this adapter does
    not do for a span attribute. There is deliberately no `wardex.step.attempts`
    key today; this test fails the moment someone adds one, which is the point at
    which the trade-off should be re-argued rather than quietly reversed.
    """
    from langgraph.types import RetryPolicy

    attempts = {"n": 0}

    def flaky(state: TrailState) -> TrailState:
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("not yet")
        return {"trail": ["ok"]}

    live = Installed()
    try:
        g = StateGraph(TrailState)
        g.add_node(
            "flaky",
            flaky,
            retry_policy=RetryPolicy(
                max_attempts=2,
                retry_on=ValueError,
                initial_interval=0.001,
                backoff_factor=1.0,
            ),
        )
        g.add_edge(START, "flaky")
        g.add_edge("flaky", END)
        app = g.compile()
        app.invoke({"trail": []})

        keys = set(extra_of(steps(live.spans)[0]))
        assert "wardex.step.attempts" not in keys
        assert not any("attempt" in k for k in keys)
    finally:
        live.teardown()


# --------------------------------------------------------------------------
# signature totality
# --------------------------------------------------------------------------


def test_an_unexpected_keyword_at_the_node_seam_reaches_the_host_untouched():
    """A langgraph release that adds a parameter must not make the wrapper raise
    a `TypeError` INTO the host's graph run. Both wrappers are signature-total
    and forward verbatim."""
    from langgraph.pregel import _runner

    live = Installed()
    try:
        seen = {}

        class FakeTask:
            name = "n"

        def spy(task, retry_policy, *args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return "host-value"

        patched = _runner.run_with_retry
        # the wrapper is what is installed; drive it with the extra keyword a
        # future release would add, standing the original up as a spy
        import wardex_sdk.adapters._langgraph as mod

        wrapper = mod._mk_run_with_retry(spy, live.adapter, "__start__")
        assert wrapper(FakeTask(), None, "positional", brand_new_kwarg=7) == "host-value"
        assert seen["args"] == ("positional",)
        assert seen["kwargs"] == {"brand_new_kwarg": 7}
        assert patched is _runner.run_with_retry
    finally:
        live.teardown()


def test_a_wrapper_with_no_context_passes_straight_through():
    """The `if ctx is None` branch is reachable AFTER `uninstall()`, because the
    wrappers read `adapter._ctx` per call rather than latching it at install."""
    import wardex_sdk.adapters._langgraph as mod

    adapter = LangGraphAdapter()  # never installed: `_ctx` is None
    wrapper = mod._mk_run_with_retry(lambda task, rp, *a, **k: "host", adapter, "__start__")

    class FakeTask:
        name = "n"

    assert wrapper(FakeTask(), None) == "host"


# --------------------------------------------------------------------------
# confirm_active
# --------------------------------------------------------------------------


def test_confirm_active_counts_node_spans_not_seam_entries(installed):
    """BELOW the `__start__` filter. Above it the site counts tasks that reached
    the seam, which is `nodes + 1` on a fresh run — a number with no stable
    relation to anything a gate can name."""
    for n in (1, 2, 3):
        counters.reset()
        installed.client.spans.clear()
        chain(installed.ctx, n, name=f"C{n}", leaves=False).invoke({"trail": []})
        snap = adapter_counters()
        # Asserted as EQUAL to the span count, not as two literals: the counter
        # is a cross-check on `len(steps)`, not a second number to maintain.
        counted = snap["adapters.langgraph.active.runner.run_with_retry"]
        assert counted == len(steps(installed.spans))
        assert len(steps(installed.spans)) == n


def test_the_sync_entry_does_not_bump_the_async_counter(installed):
    chain(installed.ctx, 1, name="SyncOnly", leaves=False).invoke({"trail": []})
    snap = adapter_counters()
    assert snap["adapters.langgraph.active.pregel.stream"] == 1
    assert "adapters.langgraph.active.pregel.astream" not in snap


def test_a_graph_with_no_tools_leaves_the_tool_counter_at_zero(installed):
    """The shape that must NOT be read as a broken patch."""
    chain(installed.ctx, 2, name="NoTools", leaves=False).invoke({"trail": []})
    snap = adapter_counters()
    assert "adapters.langgraph.active.toolnode.run_one" not in snap
    assert snap["adapters.langgraph.active.runner.run_with_retry"] == 2


def test_there_is_no_pregel_invoke_confirm_site(installed):
    """`invoke` DELEGATES to `stream`, so a site declared there would sit at 0
    forever and read as a broken patch."""
    chain(installed.ctx, 1, name="Delegated", leaves=False).invoke({"trail": []})
    assert "adapters.langgraph.active.pregel.invoke" not in adapter_counters()


# --------------------------------------------------------------------------
# uninstall while the host is mid-run
# --------------------------------------------------------------------------


def test_uninstall_mid_stream_leaves_the_context_standing_for_the_straggler():
    """`self._ctx` is NOT nulled, which is the half the shared suite cannot see.

    That the run span ships carrying `ADAPTER_UNINSTALLED` is a conformance
    invariant and lives there. What is specific to this adapter is WHY the
    generator the host is still pumping can finish at all: its `with` needs the
    context to close, and an adapter that nulled it on teardown — which the
    Agent SDK adapter does, correctly, because it holds a table instead — would
    turn a straggler into an `AttributeError` inside the host's own generator.
    """
    live = Installed()
    app = chain(live.ctx, 3, name="TornDown", leaves=False)
    it = app.stream({"trail": []})
    next(it)
    live.teardown()
    assert live.adapter._ctx is not None
    trail = list(it)  # must complete, and the host's own values must arrive
    assert trail


# --------------------------------------------------------------------------
# the shape rules the module itself must satisfy
# --------------------------------------------------------------------------

#: The one grep the adapter surface was designed around. Every one of these is a
#: way for a FRAMEWORK IDENTIFIER to affect the shape of the tree, and this
#: adapter's whole claim is that none of them is reachable from it.
_FORBIDDEN_VERBS = frozenset({"rejoin", "attach", "pin", "open_run", "claim", "claim_run"})


def test_no_framework_identifier_can_shape_this_adapters_tree():
    """The product claim, as an AST test over the shipped module's own source.

    It is the only assertion that survives a future author reintroducing
    identifier-rebuilt parentage in a way that happens to produce the right
    tier on the workloads the other tests drive.
    """
    tree = ast.parse(_MODULE.read_text())
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    assert used & _FORBIDDEN_VERBS == set()


def test_the_module_exports_a_sorted_resolvable_all():
    import wardex_sdk.adapters._langgraph as mod

    assert mod.__all__ == sorted(mod.__all__)
    assert mod.__all__
    for name in mod.__all__:
        assert hasattr(mod, name)


def test_tool_attributes_is_reachable_without_naming_a_forbidden_module():
    """`wardex_sdk._types` is C-S1-forbidden for `adapters/`, and this adapter
    carries no debt entry — so the re-export is what makes the tool half legal
    to write at all."""
    import wardex_sdk.assembly as assembly

    assert "ToolAttributes" in assembly.__all__
    assert assembly.__all__ == sorted(assembly.__all__)
