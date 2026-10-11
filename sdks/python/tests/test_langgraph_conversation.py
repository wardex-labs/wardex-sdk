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

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import wardex_sdk as wardex
from test_langgraph_adapter import (
    TrailState,
    _clean_scope,  # noqa: F401 — the autouse determinism fixture this module needs too
    adapter_counters,
    extra_of,
    installed,  # noqa: F401 — a fixture, usable here only because it is imported
    pure_add,
    runs,
    steps,
    tools,
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


def test_a_thread_bound_with_with_config_is_the_conversation(installed):  # noqa: F811
    """`graph.with_config(configurable={"thread_id": ...})` states the thread as
    surely as a call config — LangGraph checkpoints under it — and the run read
    only the call, so it shipped no conversation at all."""
    _app().with_config(configurable={"thread_id": "T-BOUND"}).invoke({"trail": []})
    (run,) = runs(installed.spans)
    assert run.conversation == ConversationContext(conversation_id="T-BOUND")
    assert _conversation_ids(installed.spans) == {"T-BOUND"}
    assert extra_of(run)["wardex.langgraph.thread_id"] == "T-BOUND"


def test_the_call_configs_thread_is_the_conversation_over_the_bound_one(installed):  # noqa: F811
    """Both directions on one graph, so that neither reading alone passes: the
    bound thread is the conversation until a call names its own, and the
    call's wins for that run only."""
    app = _app().with_config(configurable={"thread_id": "T-BOUND"})
    app.invoke({"trail": []})
    app.invoke({"trail": []}, {"configurable": {"thread_id": "T-CALL"}})
    in_order = sorted(runs(installed.spans), key=lambda s: s.start_time_ns)
    assert [_conversation_ids([r]) for r in in_order] == [{"T-BOUND"}, {"T-CALL"}]


def test_a_thread_in_an_enclosing_runnables_config_is_the_conversation(installed):  # noqa: F811
    """LangGraph merges the ambient runnable config into a call that names no
    thread and runs on the thread it carries, so that thread is the
    conversation too; the call and bound configs alone said nothing."""
    from langchain_core.runnables import RunnableLambda

    app = _app()
    RunnableLambda(lambda x: app.invoke(x)).invoke(
        {"trail": []}, {"configurable": {"thread_id": "T-AMB"}}
    )
    (run,) = runs(installed.spans)
    assert _conversation_ids(installed.spans) == {"T-AMB"}
    assert extra_of(run)["wardex.langgraph.thread_id"] == "T-AMB"


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


def _leaf(name, node=None):
    g = StateGraph(TrailState)
    g.add_node("s1", node or (lambda s: {"trail": ["s1"]}))
    g.add_edge(START, "s1")
    g.add_edge("s1", END)
    app = g.compile()
    app.name = name
    return app


def _outer_calling(node):
    outer = StateGraph(TrailState)
    outer.add_node("sub", node)
    outer.add_edge(START, "sub")
    outer.add_edge("sub", END)
    app = outer.compile(checkpointer=InMemorySaver())
    app.name = "Outer"
    return app


def _by_name(spans):
    return {s.name: _conversation_ids([s]).pop() for s in spans}


def test_a_host_conversation_opened_inside_a_node_reaches_the_inner_run(installed):  # noqa: F811
    """The outer run STATED its thread, and then the host opens a conversation
    of its own inside a node before calling a subgraph. The inner run yields its
    own thread id to it (`framework_conversation`), and what it carries instead
    is the host's id — it used to inherit the OUTER thread, because the unit
    registry handed a nested unit its parent unit's conversation and never
    consulted the scope. The node's own step was opened before the block and
    keeps the outer thread."""
    inner = _leaf("Inner")

    def node(state):
        with wardex.conversation("chat", id="host-in-node"):
            return inner.invoke(state, {"configurable": {"thread_id": "INNER"}})

    _outer_calling(node).invoke({"trail": []}, {"configurable": {"thread_id": "OUTER"}})
    assert _by_name(runs(installed.spans)) == {
        "invoke_workflow Outer": "OUTER",
        "invoke_workflow Inner": "host-in-node",
    }
    assert _by_name(steps(installed.spans)) == {
        "execute_step sub": "OUTER",
        "execute_step s1": "host-in-node",
    }


def test_a_host_conversation_opened_inside_a_node_reaches_every_depth_below_it(installed):  # noqa: F811
    """A subgraph two levels down yields to the host too. The middle run carries
    the host's id as INHERITED, not stated, so the deeper run reads it as the
    host's word rather than as this adapter's — the rule that keeps one block
    one conversation however deep the graphs nest."""
    deep = _leaf("Deep")
    mid = _leaf("Mid", lambda s: deep.invoke(s, {"configurable": {"thread_id": "DEEP"}}))

    def node(state):
        with wardex.conversation("chat", id="host-in-node"):
            return mid.invoke(state, {"configurable": {"thread_id": "MID"}})

    _outer_calling(node).invoke({"trail": []}, {"configurable": {"thread_id": "OUTER"}})
    assert _by_name(runs(installed.spans)) == {
        "invoke_workflow Outer": "OUTER",
        "invoke_workflow Mid": "host-in-node",
        "invoke_workflow Deep": "host-in-node",
    }
    assert adapter_counters()["adapters.langgraph.thread_id_shadowed_by_host"] == 2


def test_a_tool_in_a_subgraph_under_a_host_conversation_carries_it(installed):  # noqa: F811
    def model(state):
        call = {"name": "pure_add", "args": {"a": 1, "b": 2}, "id": "c1", "type": "tool_call"}
        return {"messages": [AIMessage(content="", tool_calls=[call])]}

    g = StateGraph(MessagesState)
    g.add_node("model", model)
    g.add_node("tools", ToolNode([pure_add]))
    g.add_edge(START, "model")
    g.add_edge("model", "tools")
    g.add_edge("tools", END)
    sub = g.compile()

    def node(state):
        with wardex.conversation("chat", id="host-in-node"):
            sub.invoke({"messages": []}, {"configurable": {"thread_id": "INNER"}})
        return {"trail": ["sub"]}

    _outer_calling(node).invoke({"trail": []}, {"configurable": {"thread_id": "OUTER"}})
    (tool,) = tools(installed.spans)
    assert _conversation_ids([tool]) == {"host-in-node"}


def _two_steps(name):
    g = StateGraph(TrailState)
    g.add_node("s1", lambda s: {"trail": ["s1"]})
    g.add_node("s2", lambda s: {"trail": ["s2"]})
    g.add_edge(START, "s1")
    g.add_edge("s1", "s2")
    g.add_edge("s2", END)
    app = g.compile()
    app.name = name
    return app


def test_a_run_started_inside_a_host_block_keeps_its_conversation_after_the_block(installed):  # noqa: F811
    """A run's conversation is fixed when it starts. A subgraph streamed from
    inside the block and pumped to its end after the block closed is one run
    in one conversation: its later steps used to take the outer thread id,
    which the scope showed again once the block was gone."""
    inner = _two_steps("Inner")

    def node(state):
        with wardex.conversation("chat", id="host-in-node"):
            stream = inner.stream(state, {"configurable": {"thread_id": "INNER"}})
            next(stream)
        list(stream)
        return {"trail": ["sub"]}

    _outer_calling(node).invoke({"trail": []}, {"configurable": {"thread_id": "OUTER"}})
    assert _by_name(runs(installed.spans))["invoke_workflow Inner"] == "host-in-node"
    inner_steps = [s for s in steps(installed.spans) if s.name.endswith(("s1", "s2"))]
    assert _conversation_ids(inner_steps) == {"host-in-node"}


def test_a_host_block_opened_while_a_run_streams_does_not_relabel_its_steps(installed):  # noqa: F811
    """The mirror: the run started before the block, so its steps keep the
    run's conversation (here its own thread id) even where one of them runs
    inside the block."""
    inner = _two_steps("Inner")

    def node(state):
        stream = inner.stream(state, {"configurable": {"thread_id": "INNER"}})
        next(stream)
        with wardex.conversation("chat", id="host-in-node"):
            list(stream)
        return {"trail": ["sub"]}

    _outer_calling(node).invoke({"trail": []}, {"configurable": {"thread_id": "OUTER"}})
    assert _by_name(runs(installed.spans))["invoke_workflow Inner"] == "INNER"
    inner_steps = [s for s in steps(installed.spans) if s.name.endswith(("s1", "s2"))]
    assert _conversation_ids(inner_steps) == {"INNER"}
