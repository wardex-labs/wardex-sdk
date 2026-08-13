"""LangGraph adapter — graph-edge causality as LINKS between spans.

Two edges become links here and every other candidate is a documented refusal.
`TRIGGERED_BY` ships only for a `join:{a}+{b}:{end}` trigger — the one
StateGraph trigger format that NAMES its sources, with the barrier channel
guaranteeing each of them wrote — one link per named source, resolved through
the registry's closed-unit link memory because the sources' spans are finished
by the time the joined step opens. `RESUMED_FROM` ships for the same-process
half of a checkpoint resume: runs sharing a `thread_id` alias themselves under
it, and a successor links to its predecessor's span — a NEW trace linked,
never a fabricated parent across runs.

The negatives are pinned as deliberately as the positives, because each one is
a decision a future author could quietly reverse into a guess: a
`branch:to:{self}` trigger names only its DESTINATION, so ordinary chains
carry no links; a `Send` fan-out produces same-named sibling copies no
selector could pick between, so push tasks never become link targets and the
named-but-unresolvable join source is COUNTED, never faked; and a fresh
thread is indistinguishable from a cross-process resume, so a first run on a
thread claims nothing and counts nothing.

Real compiled `StateGraph`s throughout — the harness is
`test_langgraph_adapter.py`'s, for the reason that file states. The trigger
formats these graphs produce were verified by executing them against the
installed langgraph before the adapter code was written; the shapes here are
that census, kept live.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from test_codec import _header  # the established envelope helper
from test_langgraph_adapter import (
    TrailState,
    _clean_scope,  # noqa: F401 — the autouse determinism fixture this module needs too
    adapter_counters,
    installed,  # noqa: F401 — a fixture, usable here only because it is imported
    runs,
    steps,
)
from wardex_sdk._assembly import LinkReason
from wardex_sdk._types import Envelope
from wardex_sdk.transport import _codec

# --------------------------------------------------------------------------
# local builders — only the shapes the harness does not already own
# --------------------------------------------------------------------------


def _node(label: str):
    def node(state):
        return {"trail": [label]}

    return node


def _join_graph(name: str = "Join"):
    """`a` and `b` in one parallel superstep, joined into `c`.

    The one StateGraph shape whose trigger names its sources: `c` fires on
    `('branch:to:c', 'join:a+b:c')` — verified by executing this graph.
    """
    g = StateGraph(TrailState)
    g.add_node("a", _node("a"))
    g.add_node("b", _node("b"))
    g.add_node("c", _node("c"))
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge(["a", "b"], "c")
    g.add_edge("c", END)
    app = g.compile()
    app.name = name
    return app


def _thread_graph(name: str = "Threaded"):
    """One node, checkpointed — the resume shape. Threads live in the config."""
    g = StateGraph(TrailState)
    g.add_node("only", _node("only"))
    g.add_edge(START, "only")
    g.add_edge("only", END)
    app = g.compile(checkpointer=InMemorySaver())
    app.name = name
    return app


def _send_fanout(name: str = "Fan"):
    """Three `Send` copies of one worker, joined into a reducer.

    Verified shape: every worker fires on `('__pregel_push',)` — no source in
    the string and no alias registered — while `reduce` fires on
    `'join:worker:reduce'`, naming a source that never became a link target.
    """
    g = StateGraph(TrailState)
    g.add_node("fan", _node("fan"))
    g.add_node("worker", _node("w"))
    g.add_node("reduce", _node("reduce"))
    g.add_edge(START, "fan")
    g.add_conditional_edges("fan", lambda s: [Send("worker", {"trail": []}) for _ in range(3)])
    g.add_edge(["worker"], "reduce")
    g.add_edge("reduce", END)
    app = g.compile()
    app.name = name
    return app


def _step_named(spans, name: str):
    return next(s for s in steps(spans) if s.name == f"execute_step {name}")


def _steps_named(spans, name: str):
    found = [s for s in steps(spans) if s.name == f"execute_step {name}"]
    return sorted(found, key=lambda s: s.start_time_ns)


# --------------------------------------------------------------------------
# TRIGGERED_BY — the join format links, everything else refuses
# --------------------------------------------------------------------------


def test_a_join_edge_names_its_sources_and_each_gets_a_triggered_by_link(installed):  # noqa: F811
    """One link PER NAMED SOURCE: the barrier fired because every one of them
    wrote, so each edge is individually backed — refusing them would leave
    honest, encodable causality off the wire.
    """
    _join_graph().invoke({"trail": []})

    spans = installed.spans
    a, b, c = (_step_named(spans, n) for n in "abc")

    assert a.links == ()
    assert b.links == ()
    assert len(c.links) == 2
    assert all(link.reason is LinkReason.TRIGGERED_BY for link in c.links)
    assert {link.span_id for link in c.links} == {a.context.span_id, b.context.span_id}
    assert all(link.trace_id == c.context.trace_id for link in c.links)
    assert "adapters.langgraph.link_target_unresolved" not in adapter_counters()


def test_a_plain_chain_carries_no_triggered_by_links(installed):  # noqa: F811
    """Every ordinary edge fires on `branch:to:{self}` — verified destination-
    named, source-anonymous — so a plain chain ships NO links and no unresolved
    count. Pinned so the source-anonymity decision cannot be reversed into a
    graph-introspection guess without failing a test that says why.
    """
    g = StateGraph(TrailState)
    g.add_node("a", _node("a"))
    g.add_node("b", _node("b"))
    g.add_node("c", _node("c"))
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "c")
    g.add_edge("c", END)
    app = g.compile()
    app.name = "Plain"
    app.invoke({"trail": []})

    assert [s.links for s in steps(installed.spans)] == [(), (), ()]
    assert "adapters.langgraph.link_target_unresolved" not in adapter_counters()


def test_a_send_fanout_source_is_never_guessed(installed):  # noqa: F811
    """The brief's required negative. `reduce`'s trigger NAMES `worker`, but
    three identical push copies ran and none of them may own the name — so the
    link is counted as unresolvable, and no span anywhere links to any worker.
    A fabricated link to "whichever copy" would be confidence the edge cannot
    back.
    """
    _send_fanout().invoke({"trail": []})

    spans = installed.spans
    workers = _steps_named(spans, "worker")
    assert len(workers) == 3
    assert _step_named(spans, "reduce").links == ()
    worker_ids = {w.context.span_id for w in workers}
    assert all(link.span_id not in worker_ids for s in spans for link in s.links)
    assert adapter_counters()["adapters.langgraph.link_target_unresolved"] == 1


def test_a_cycle_links_each_iteration_to_the_latest_prior_execution(installed):  # noqa: F811
    """Latest-wins across supersteps: re-binding the node alias supersedes the
    previous iteration's memory entry, so each `b` links to the `a` that
    actually triggered it — one link each, never an accumulation and never a
    stale first-iteration target.
    """
    g = StateGraph(TrailState)
    g.add_node("a", _node("a"))
    g.add_node("b", _node("b"))
    g.add_edge(START, "a")
    g.add_edge(["a"], "b")  # the joint-edge spelling, so the trigger names `a`
    g.add_conditional_edges("b", lambda s: "a" if len(s["trail"]) < 4 else END)
    app = g.compile()
    app.name = "Cycle"
    app.invoke({"trail": []})

    spans = installed.spans
    a_spans = _steps_named(spans, "a")
    b_spans = _steps_named(spans, "b")
    assert len(a_spans) == 2
    assert len(b_spans) == 2
    (first_link,) = b_spans[0].links
    (second_link,) = b_spans[1].links
    assert first_link.reason is LinkReason.TRIGGERED_BY
    assert second_link.reason is LinkReason.TRIGGERED_BY
    assert first_link.span_id == a_spans[0].context.span_id
    assert second_link.span_id == a_spans[1].context.span_id


# --------------------------------------------------------------------------
# RESUMED_FROM — the same-process half of a resume
# --------------------------------------------------------------------------


def test_a_second_run_on_the_same_thread_links_resumed_from(installed):  # noqa: F811
    """A resume is a NEW trace linked to the old one's span — never a parent
    edge across runs, which would splice two invocations into one flame graph.
    """
    app = _thread_graph()
    config = {"configurable": {"thread_id": "t1"}}
    app.invoke({"trail": []}, config)
    app.invoke({"trail": []}, config)

    run1, run2 = sorted(runs(installed.spans), key=lambda s: s.start_time_ns)
    assert run1.context.trace_id != run2.context.trace_id
    assert run1.links == ()
    (link,) = run2.links
    assert link.trace_id == run1.context.trace_id
    assert link.span_id == run1.context.span_id
    assert link.reason is LinkReason.RESUMED_FROM
    assert "adapters.langgraph.link_target_unresolved" not in adapter_counters()


def test_the_first_run_on_a_thread_claims_no_resume_and_counts_nothing(installed):  # noqa: F811
    """A fresh thread and a cross-process resume are indistinguishable at this
    seam, so the miss is SILENT (`expected=False`): counting every first run
    would make `link_target_unresolved` meaningless, and it would itself be a
    fabricated loss claim.
    """
    _thread_graph().invoke({"trail": []}, {"configurable": {"thread_id": "fresh"}})

    (run,) = runs(installed.spans)
    assert run.links == ()
    assert "adapters.langgraph.link_target_unresolved" not in adapter_counters()


def test_runs_on_different_threads_do_not_cross(installed):  # noqa: F811
    """Memory keys on the THREAD alias, not on recency: t1's third run links to
    t1's first, never to the more recent t2 run in between.
    """
    app = _thread_graph()
    app.invoke({"trail": []}, {"configurable": {"thread_id": "t1"}})
    app.invoke({"trail": []}, {"configurable": {"thread_id": "t2"}})
    app.invoke({"trail": []}, {"configurable": {"thread_id": "t1"}})

    run1, run2, run3 = sorted(runs(installed.spans), key=lambda s: s.start_time_ns)
    assert run2.links == ()  # t2's first and only
    (link,) = run3.links
    assert link.span_id == run1.context.span_id
    assert link.reason is LinkReason.RESUMED_FROM


def test_a_run_without_a_thread_id_neither_links_nor_aliases(installed):  # noqa: F811
    """No thread identity means no resume claim in EITHER direction: nothing to
    link to, and nothing left behind in the link memory for a later run to
    find — runs without threads must not grow registry state.
    """
    g = StateGraph(TrailState)
    g.add_node("only", _node("only"))
    g.add_edge(START, "only")
    g.add_edge("only", END)
    app = g.compile()
    app.name = "NoThread"
    app.invoke({"trail": []})
    app.invoke({"trail": []})

    assert [s.links for s in runs(installed.spans)] == [(), ()]
    assert "adapters.langgraph.link_target_unresolved" not in adapter_counters()
    memory = installed.ctx._units._link_memory
    assert not any(key.namespace == "langgraph.thread_id" for key in memory)


# --------------------------------------------------------------------------
# the wire — links as they actually ship
# --------------------------------------------------------------------------


def test_triggered_by_and_resumed_from_survive_the_native_codec(installed):  # noqa: F811
    """Everything above is measured in Python; this measures the WIRE, on the
    two link shapes this adapter emits. A link the codec dropped or re-keyed
    would pass every draft assertion and ship a graph with no edges.
    """
    _join_graph().invoke({"trail": []})
    app = _thread_graph()
    config = {"configurable": {"thread_id": "t-wire"}}
    app.invoke({"trail": []}, config)
    app.invoke({"trail": []}, config)

    spans = installed.spans
    a, b, c = (_step_named(spans, n) for n in "abc")
    run1, run2 = sorted(
        (s for s in runs(spans) if s.name == "invoke_workflow Threaded"),
        key=lambda s: s.start_time_ns,
    )

    envelope = Envelope(header=_header(), spans=(c, run2))
    decoded = _codec.decode(_codec.encode(envelope))
    out = {item["span"]["name"]: item["span"] for item in decoded["items"]}

    wire_step = out["execute_step c"]
    assert [link["reason"] for link in wire_step["links"]] == ["triggered_by", "triggered_by"]
    assert {link["span_id"] for link in wire_step["links"]} == {
        a.context.span_id.value,
        b.context.span_id.value,
    }

    wire_run = out["invoke_workflow Threaded"]
    (wire_link,) = wire_run["links"]
    assert wire_link["reason"] == "resumed_from"
    assert wire_link["trace_id"] == run1.context.trace_id.value
    assert wire_link["span_id"] == run1.context.span_id.value
