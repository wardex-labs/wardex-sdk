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
