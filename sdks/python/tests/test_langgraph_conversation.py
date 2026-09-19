"""A LangGraph run's `thread_id` is its conversation.

`thread_id` is the identifier a host hands LangGraph to continue one chat
across runs. The adapter read it — it links a resumed run to its predecessor by
it — and recorded it only as `wardex.langgraph.thread_id`, so a host that had
already said which conversation a run belongs to shipped spans with no
conversation at all, and a backend grouping by `gen_ai.conversation.id` put
every LangGraph run in one unnamed bucket. The rule these tests hold is the one
the OpenAI Agents adapter already kept for its `group_id`: what the host or the
framework SAID is carried, what nobody said stays empty, and the host's own
`wardex.conversation(...)` wins over the framework's id.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from test_langgraph_adapter import (
    TrailState,
    _clean_scope,  # noqa: F401 — the autouse determinism fixture this module needs too
    adapter_counters,
    extra_of,
    installed,  # noqa: F401 — a fixture, usable here only because it is imported
    runs,
    steps,
)
from wardex_sdk import _hub
from wardex_sdk._types import ConversationContext


def _app():
    g = StateGraph(TrailState)
    g.add_node("a", lambda s: {"trail": ["a"]})
    g.add_node("b", lambda s: {"trail": ["b"]})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile(checkpointer=InMemorySaver())


def _conversation_ids(spans):
    return {s.conversation.conversation_id if s.conversation else None for s in spans}


def test_the_thread_id_is_the_conversation_of_the_run_and_of_every_span_under_it(installed):  # noqa: F811
    _app().invoke({"trail": []}, {"configurable": {"thread_id": "T-1"}})
    (run,) = runs(installed.spans)
    assert run.conversation == ConversationContext(conversation_id="T-1")
    assert len(steps(installed.spans)) == 2
    assert _conversation_ids(installed.spans) == {"T-1"}
    # The framework's own spelling stays where it was: readers keyed on it keep working.
    assert extra_of(run)["wardex.langgraph.thread_id"] == "T-1"


def test_an_integer_thread_id_is_its_text(installed):  # noqa: F811
    _app().invoke({"trail": []}, {"configurable": {"thread_id": 7}})
    assert _conversation_ids(installed.spans) == {"7"}


def test_a_run_without_a_thread_id_has_no_conversation(installed):  # noqa: F811
    """Nobody said, so nothing is carried. wardex does not mint an id here: a
    minted one would call a run a conversation, and a store would key on it."""
    g = StateGraph(TrailState)
    g.add_node("a", lambda s: {"trail": ["a"]})
    g.add_edge(START, "a")
    g.add_edge("a", END)
    g.compile().invoke({"trail": []})
    assert _conversation_ids(installed.spans) == {None}


def test_the_hosts_own_conversation_wins_over_the_thread_id(installed):  # noqa: F811
    """The same scope write `wardex.conversation(...)` performs. The host's id
    stays on every span, the thread id keeps its own attribute, and the
    shadowing is counted — once, not once per span."""
    scope = _hub.get_current_scope()
    scope.conversation = ConversationContext(conversation_id="host-1")
    try:
        _app().invoke({"trail": []}, {"configurable": {"thread_id": "T-1"}})
    finally:
        scope.conversation = None
    assert _conversation_ids(installed.spans) == {"host-1"}
    (run,) = runs(installed.spans)
    assert extra_of(run)["wardex.langgraph.thread_id"] == "T-1"
    assert adapter_counters()["adapters.langgraph.thread_id_shadowed_by_host"] == 1


def test_a_subgraph_run_under_the_hosts_conversation_keeps_the_hosts_id(installed):  # noqa: F811
    """Found by review, by running it: the inner run of a subgraph-as-node
    carried the THREAD id under a host conversation, so one trace shipped two
    conversation ids. The rule had asked who OWNS the ambient unit, and the
    outer run — this adapter's own unit — was carrying the host's id. What
    matters is whether this adapter STATED the ambient conversation."""
    inner = StateGraph(TrailState)
    inner.add_node("s1", lambda s: {"trail": ["s1"]})
    inner.add_edge(START, "s1")
    inner.add_edge("s1", END)
    sub = inner.compile()

    outer = StateGraph(TrailState)
    outer.add_node("a", lambda s: {"trail": ["a"]})
    outer.add_node("sub", lambda s: sub.invoke(s, {"configurable": {"thread_id": "T-SUB"}}))
    outer.add_edge(START, "a")
    outer.add_edge("a", "sub")
    outer.add_edge("sub", END)

    scope = _hub.get_current_scope()
    scope.conversation = ConversationContext(conversation_id="host-1")
    try:
        outer.compile(checkpointer=InMemorySaver()).invoke(
            {"trail": []}, {"configurable": {"thread_id": "T-1"}}
        )
    finally:
        scope.conversation = None
    assert len(runs(installed.spans)) == 2
    assert _conversation_ids(installed.spans) == {"host-1"}
    # Counted per RUN that yielded, and two did: the counter answers "how many
    # runs had a thread id the host overruled", not "how many host blocks".
    assert adapter_counters()["adapters.langgraph.thread_id_shadowed_by_host"] == 2


def test_a_nested_run_with_no_host_conversation_keeps_its_own_thread(installed):  # noqa: F811
    """The other half of the same rule: nothing the host said is ambient, the
    outer run STATED its thread, so the inner run's own thread stays the inner
    run's conversation."""
    inner = StateGraph(TrailState)
    inner.add_node("s1", lambda s: {"trail": ["s1"]})
    inner.add_edge(START, "s1")
    inner.add_edge("s1", END)
    sub = inner.compile()

    outer = StateGraph(TrailState)
    outer.add_node("sub", lambda s: sub.invoke(s, {"configurable": {"thread_id": "INNER"}}))
    outer.add_edge(START, "sub")
    outer.add_edge("sub", END)
    outer.compile(checkpointer=InMemorySaver()).invoke(
        {"trail": []}, {"configurable": {"thread_id": "OUTER"}}
    )
    by_run = {r.conversation.conversation_id for r in runs(installed.spans)}
    assert by_run == {"OUTER", "INNER"}


def test_a_uuid_thread_id_is_carried_as_its_text(installed):  # noqa: F811
    """Found by review: the shared rule took only `str | int`, so a host that
    keys its threads by `uuid.UUID` — which LangGraph accepts — had stated a
    conversation and shipped none."""
    import uuid

    thread = uuid.UUID("12345678-1234-5678-1234-567812345678")
    _app().invoke({"trail": []}, {"configurable": {"thread_id": thread}})
    assert _conversation_ids(installed.spans) == {str(thread)}


def test_a_host_conversation_opened_inside_a_node_does_not_reach_the_inner_run_yet(installed):  # noqa: F811
    """A KNOWN GAP, pinned so that closing it is a visible edit here.

    The outer run STATED its thread, and then the host opens a conversation of
    its own inside a node before calling a subgraph. The rule in
    `framework_conversation` does its half: it sees a conversation on the scope
    that no unit of this adapter stated, and the inner run yields — it does NOT
    take its own thread id. What it inherits instead is wrong: the unit
    registry hands a nested unit its PARENT UNIT's conversation and never
    consults the scope, so the inner run ships the outer thread, not the
    host's word. That precedence predates this adapter stating conversations
    at all, and it is the registry's to change, for every adapter at once.
    """
    inner = StateGraph(TrailState)
    inner.add_node("s1", lambda s: {"trail": ["s1"]})
    inner.add_edge(START, "s1")
    inner.add_edge("s1", END)
    sub = inner.compile()

    def node(state):
        scope = _hub.get_current_scope()
        before = scope.conversation
        scope.conversation = ConversationContext(conversation_id="host-in-node")
        try:
            return sub.invoke(state, {"configurable": {"thread_id": "INNER"}})
        finally:
            scope.conversation = before

    outer = StateGraph(TrailState)
    outer.add_node("sub", node)
    outer.add_edge(START, "sub")
    outer.add_edge("sub", END)
    outer.compile(checkpointer=InMemorySaver()).invoke(
        {"trail": []}, {"configurable": {"thread_id": "OUTER"}}
    )
    by_run = sorted(r.conversation.conversation_id for r in runs(installed.spans))
    assert "INNER" not in by_run  # the half that holds: the inner thread yielded
    assert by_run == ["OUTER", "OUTER"]  # the gap: "host-in-node" is what it should carry
