"""RemoteGraph — the run boundary of a LangGraph Platform call.

A standalone `RemoteGraph` call crosses ZERO local seams: `remote.py` defines
its own `stream`/`astream`/`invoke`/`ainvoke` rather than inheriting Pregel's,
which is exactly why the adapter patches it as its own probe group. These
tests drive the FRAMEWORK'S OWN `remote.py` code and stub only the network
client, through the constructor's documented injection surface (`client=` /
`sync_client=` are stored verbatim when `url` is None). The no-fakes policy in
`test_langgraph_adapter` targets faking the FRAMEWORK; here the framework runs
for real up to the exact boundary a live platform would occupy.

The chunks are real `langgraph_sdk.schema.StreamPart` named tuples — the type
the platform client yields and the type `remote.py` pattern-matches on
(`chunk.event` split, `chunk._replace`). The canned parts deliberately include
a `metadata` event: `remote.py` filters it out of the requested modes itself,
so the pass-through assertions below measure the framework's fidelity, not the
fake's.
"""

from __future__ import annotations

import asyncio

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.pregel.remote import RemoteException, RemoteGraph
from langgraph_sdk.schema import StreamPart

from test_langgraph_adapter import (
    TrailState,
    _clean_scope,  # noqa: F401 — the autouse determinism fixture this module needs too
    adapter_counters,
    chain,
    edge_of,
    extra_of,
    installed,  # noqa: F401 — a FIXTURE; importing it is what makes it usable here,
    # and every test below re-flags it as F811 because its own parameter shadows it
    leaf_span,
    parent_name,
    runs,
    steps,
    tools,
)
from wardex_sdk._assembly import ParentSource
from wardex_sdk._enums import StatusCode

#: One metadata part the framework filters out itself, then two values parts.
VALUES = [
    StreamPart("metadata", {"run_id": "r-1"}),
    StreamPart("values", {"messages": ["step1"]}),
    StreamPart("values", {"done": True}),
]


class FakeRuns:
    """`runs.stream(**kwargs)` — the one platform surface `RemoteGraph.stream`
    touches. Records its kwargs so a test can assert what the framework sent."""

    def __init__(self, parts, before_yield=None):
        self.parts = parts
        self.calls = []
        self.before_yield = before_yield

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        if self.before_yield is not None:
            self.before_yield()
        yield from self.parts


class FakeSyncClient:
    def __init__(self, parts, before_yield=None):
        self.runs = FakeRuns(parts, before_yield)


class FakeAsyncRuns:
    def __init__(self, parts):
        self.parts = parts
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)

        async def gen():
            for part in self.parts:
                yield part

        return gen()


class FakeAsyncClient:
    def __init__(self, parts):
        self.runs = FakeAsyncRuns(parts)


def remote(parts=VALUES, *, name="Remote", before_yield=None) -> RemoteGraph:
    return RemoteGraph("asst", sync_client=FakeSyncClient(parts, before_yield), name=name)


def aremote(parts=VALUES, *, name="Remote") -> RemoteGraph:
    return RemoteGraph("asst", client=FakeAsyncClient(parts), name=name)


# --------------------------------------------------------------------------
# the run boundary
# --------------------------------------------------------------------------


def test_a_remote_invoke_ships_one_run_span_marked_remote(installed):  # noqa: F811
    """`invoke` drains `self.stream`, so the class-level stream patch is the
    whole coverage — one span, not two, and the host's return value intact."""
    out = remote().invoke({"q": 1})

    assert out == {"done": True}, "the final values chunk — host behaviour intact"
    assert len(installed.spans) == 1
    span = installed.spans[0]
    assert span.name == "invoke_workflow Remote"
    assert extra_of(span)["wardex.langgraph.remote"] == "true"
    assert extra_of(span)["wardex.framework"] == "langgraph"
    assert (span.status, span.error_type) == (StatusCode.OK, None)
    assert edge_of(span) == (ParentSource.TRACE_ROOT, 1.0, ())
    assert adapter_counters()["adapters.langgraph.active.remote.stream"] == 1


def test_a_remote_ainvoke_ships_one_run_span_via_astream(installed):  # noqa: F811
    """The async twin: `ainvoke` drains `self.astream`, and the delegation did
    not double the run."""

    async def drive():
        return await aremote().ainvoke({"q": 1})

    out = asyncio.run(drive())

    assert out == {"done": True}
    shipped = runs(installed.spans)
    assert len(shipped) == 1
    assert extra_of(shipped[0])["wardex.langgraph.remote"] == "true"
    snap = adapter_counters()
    assert snap["adapters.langgraph.active.remote.astream"] == 1
    assert "adapters.langgraph.active.remote.stream" not in snap


def test_remote_thread_id_lands_as_an_extra(installed):  # noqa: F811
    """The wrapper reads the CALLER's config. The framework pops `thread_id`
    from its own sanitized COPY (`merge_configs`/`_sanitize_config`), so the
    read survives the pop."""
    remote().invoke({"q": 1}, config={"configurable": {"thread_id": "t-1"}})

    span = runs(installed.spans)[0]
    assert extra_of(span)["wardex.langgraph.thread_id"] == "t-1"


def test_a_remote_error_event_ships_error_with_the_hosts_type(installed):  # noqa: F811
    """An `error` event makes `remote.py` raise `RemoteException` — not a
    `GraphBubbleUp`, so it is a real failure, not control flow."""
    with pytest.raises(RemoteException):
        remote([StreamPart("error", {"message": "boom"})]).invoke({"q": 1})

    span = runs(installed.spans)[0]
    assert (span.status, span.error_type) == (StatusCode.ERROR, "RemoteException")


# --------------------------------------------------------------------------
# the ambient value — what the span buys beyond detection
# --------------------------------------------------------------------------


def test_a_remote_run_nests_under_the_local_step_that_called_it(installed):  # noqa: F811
    """`RemoteGraph` used as a node: `Placement.ROOT` resolves AMBIENT
    evidence, so the remote run is a child of the live step — the exact shape
    RemoteGraph-as-node needs."""

    def call_remote(state: TrailState) -> TrailState:
        remote().invoke({"q": 1})
        return {"trail": ["called"]}

    g = StateGraph(TrailState)
    g.add_node("call_remote", call_remote)
    g.add_edge(START, "call_remote")
    g.add_edge("call_remote", END)
    app = g.compile()
    app.name = "Parent"
    app.invoke({"trail": []})

    names = {s.name for s in installed.spans}
    assert {"invoke_workflow Parent", "execute_step call_remote", "invoke_workflow Remote"} <= names
    remote_run = next(s for s in runs(installed.spans) if s.name == "invoke_workflow Remote")
    assert parent_name(installed.spans, remote_run) == "execute_step call_remote"
    assert edge_of(remote_run) == (ParentSource.UNIT_ACTIVE, 1.0, ())
    assert len({s.context.trace_id for s in installed.spans}) == 1


def test_platform_traffic_runs_inside_the_ambient_remote_unit(installed):  # noqa: F811
    """The `capture_mode=AGENT` value claim, without an HTTP server (§9.1).

    `leaf_span` is the sanctioned byte-seam stand-in — it resolves its parent
    through the same ambient the byte seam latches. Opened while the platform
    client produces chunks, it lands under the remote run span, which is what
    lets real platform HTTP pass the AGENT gate and nest instead of being
    dropped.
    """
    rg = remote(before_yield=lambda: leaf_span(installed.ctx, "platform-post"))
    rg.invoke({"q": 1})

    leaf = tools(installed.spans)[0]
    assert leaf.name == "execute_tool platform-post"
    run_span = runs(installed.spans)[0]
    assert leaf.parent_span_id == run_span.context.span_id
    assert edge_of(leaf) == (ParentSource.UNIT_ACTIVE, 1.0, ())


# --------------------------------------------------------------------------
# lifecycle parity with the local seam
# --------------------------------------------------------------------------


def test_an_abandoned_remote_stream_counts_like_a_local_one(installed):  # noqa: F811
    """Same wire shape as the pinned local abandon, on the remote seam's own
    counters — disjoint from the pregel ones."""
    it = remote().stream({"q": 1}, stream_mode="values")
    next(it)
    it.close()

    span = runs(installed.spans)[0]
    assert (span.status, span.error_type) == (StatusCode.ERROR, "GeneratorExit")
    snap = adapter_counters()
    assert snap["adapters.langgraph.active.remote.stream"] == 1
    assert snap["adapters.langgraph.remote_stream_finalized"] == 1
    assert "adapters.langgraph.remote_stream_finalized_off_carrier" not in snap


def test_a_local_run_carries_no_remote_extra(installed):  # noqa: F811
    """The extra is minted by `_describe_remote_run` only, never by the shared
    `_describe_run` both entries delegate to."""
    chain(installed.ctx, 1, leaves=False).invoke({"trail": []})

    span = runs(installed.spans)[0]
    assert "wardex.langgraph.remote" not in extra_of(span)
    assert steps(installed.spans), "and the local chain really did run"


def test_the_wrapped_stream_yields_the_hosts_chunks_untouched(installed):  # noqa: F811
    """Chunk-for-chunk what an unpatched `RemoteGraph.stream` yields for these
    parts: the metadata part filtered by the FRAMEWORK's own requested-mode
    logic, the values data unwrapped by its `req_single` branch. wardex
    altered nothing the host sees."""
    chunks = list(remote().stream({"q": 1}, stream_mode="values"))

    assert chunks == [{"messages": ["step1"]}, {"done": True}]
