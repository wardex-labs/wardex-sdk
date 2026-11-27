"""What the LangGraph adapter said about a conversation survives the real encoder.

A real graph run under a checkpointer, encoded, decoded and held to the
contract in `_adapter_envelopes.py`.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from _adapter_envelopes import SCENARIOS, deliver
from test_codec import _header
from test_langgraph_adapter import (
    Installed,
    TrailState,
    _clean_scope,  # noqa: F401 - the autouse determinism fixture the langgraph harness needs
)
from wardex_sdk._types import Envelope
from wardex_sdk.transport import _codec


def test_fresh_langgraph_envelope():
    graph = StateGraph(TrailState)
    graph.add_node("a", lambda s: {"trail": ["a"]})
    graph.add_node("b", lambda s: {"trail": ["b"]})
    graph.add_edge(START, "a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    live = Installed()
    try:
        config: Any = {"configurable": {"thread_id": SCENARIOS["langgraph"][0]}}
        graph.compile(checkpointer=InMemorySaver()).invoke({"trail": []}, config)
        spans = tuple(live.spans)
    finally:
        live.teardown()
    deliver("langgraph", _codec.encode(Envelope(header=_header(), spans=spans)))
