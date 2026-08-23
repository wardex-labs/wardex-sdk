"""LangGraph adapter — §9.3, the span vocabulary as it reaches the wire.

Everything here is measured off an EMITTED span rather than off a draft or a
handle, because the vocabulary is enforced at `SpanDraft.finish()` and every
emit site runs inside `assembly._diag.guard()`: a breach is a DELETED span and a
counter, never an exception. So an assertion written against a draft would pass
on exactly the shape that ships nothing.

The harness — real `StateGraph`s, the recording client, the readers — is
`test_langgraph_adapter.py`, imported rather than re-invented for the reason
that file states: a test double that differs from production is the failure this
adapter exists to avoid.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, Send, interrupt

from test_codec import _header  # the established envelope helper; see test_codec_typed_blocks
from test_langgraph_adapter import (
    Installed,
    RecordingClient,
    TrailState,
    _clean_scope,  # noqa: F401 — the autouse determinism fixture this module needs too
    adapter_counters,
    assembly_counters,
    call,
    chain,
    edge_of,
    extra_of,
    installed,  # noqa: F401 — a fixture, usable here only because it is imported
    pure_add,
    runs,
    steps,
    tool_graph,
    tools,
)
from wardex_sdk._adapters._langgraph import LangGraphAdapter
from wardex_sdk._adapters._registry import AdapterRegistry
from wardex_sdk._assembly import Limitation, ParentSource
from wardex_sdk._enums import ToolExecutionType, ToolType
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._types import Envelope
from wardex_sdk.transport import _codec

#: One tool call, spelled once. Every payload and block assertion below reads
#: its expectation out of THIS dict rather than out of a literal, so a test
#: cannot go green against an id or an argument the adapter invented.
_ARGS = {"a": 1, "b": 2}
_CALL = call("pure_add", _ARGS, "toolu_01")


# --------------------------------------------------------------------------
# local builders — only the shapes the harness does not already own
# --------------------------------------------------------------------------


def _plain_graph():
    """A compiled graph whose `.name` nobody touched. Deliberately not `chain`.

    `chain` assigns `app.name`, which is the case the named test covers; the
    unnamed case has to reach `_graph_name` with whatever langgraph itself left
    there.
    """
    g = StateGraph(TrailState)
    g.add_node("only", lambda s: {"trail": ["only"]})
    g.add_edge(START, "only")
    g.add_edge("only", END)
    return g.compile()


def _subgraph_app():
    """A compiled graph used as a node inside another — two namespace levels."""
    inner = StateGraph(TrailState)
    inner.add_node("inner", lambda s: {"trail": ["inner"]})
    inner.add_edge(START, "inner")
    inner.add_edge("inner", END)

    outer = StateGraph(TrailState)
    outer.add_node("outer", inner.compile(name="Sub"))
    outer.add_edge(START, "outer")
    outer.add_edge("outer", END)
    return outer.compile(name="Outer")


def _one_tool_graph(fn, call_dict, *, name: str = "OneTool", extra_node: str | None = None):
    """A model node that emits `call_dict`, a `ToolNode` holding `fn`, done.

    `extra_node` exists because a tool-returned `Command(goto=...)` names a node
    that has to exist: without it langgraph raises before the run finishes, and
    the assertion under test would be measuring an aborted graph.
    """

    def model_node(state):
        return {"messages": [AIMessage(content="", tool_calls=[call_dict])]}

    g = StateGraph(MessagesState)
    g.add_node("model", model_node)
    g.add_node("tools", ToolNode([fn]))
    g.add_edge(START, "model")
    g.add_edge("model", "tools")
    if extra_node is None:
        g.add_edge("tools", END)
    else:
        g.add_node(extra_node, lambda s: {"messages": []})
        g.add_edge(extra_node, END)
    app = g.compile()
    app.name = name
    return app


def _ancestors(spans, span) -> list[str]:
    """The names above `span`, root last. Walks parent IDS, never names."""
    by_id = {s.context.span_id: s for s in spans}
    out: list[str] = []
    cursor = span
    while cursor.parent_span_id is not None:
        cursor = by_id[cursor.parent_span_id]
        out.append(cursor.name)
    return out


def _one_tool_span(live, fn, call_dict, **kw):
    """Drive one tool call through a real `ToolNode` and hand back its span."""
    _one_tool_graph(fn, call_dict, **kw).invoke({"messages": []})
    shipped = tools(live.spans)
    assert len(shipped) == 1, f"expected exactly one tool span, got {[s.name for s in shipped]}"
    return shipped[0]


class _CappedClient(RecordingClient):
    """A recording client that also carries a config, which the base one does not.

    `context_for` reads `client.config.limits`, so a double with `config = None`
    can only ever exercise the core defaults — which is precisely the state a
    test about a CONFIGURED bound must not be in.
    """

    class _Config:
        debug = False

        def __init__(self, limits: LimitsConfig) -> None:
            self.limits = limits

    def __init__(self, limits: LimitsConfig) -> None:
        super().__init__()
        self.config = self._Config(limits)


class _CappedInstalled(Installed):
    """`Installed`, but through a client with a limits config. Same wiring."""

    def __init__(self, limits: LimitsConfig) -> None:
        self.client = _CappedClient(limits)
        self.registry = AdapterRegistry()
        self.adapter = LangGraphAdapter()
        self.registry.install(self.adapter, self.client)


def _capped_install(limits: LimitsConfig) -> _CappedInstalled:
    return _CappedInstalled(limits)


def _bytes(value) -> bytes:
    """`output_data`/`input_data` as the codec sees them: absent is `b""`."""
    return b"" if value is None else value


# --------------------------------------------------------------------------
# workflow_name — the required block of INVOKE_WORKFLOW
# --------------------------------------------------------------------------


def test_an_unnamed_graph_still_ships_a_run_span_with_a_workflow_name(installed):  # noqa: F811
    """`workflow_name` is INVOKE_WORKFLOW's required block, and a missing
    required block DELETES the span — so the span's EXISTENCE is half the
    assertion. A `_graph_name` that returned "" or None would not degrade the
    run span, it would remove it, and every node underneath would be reporting
    into a run that never shipped."""
    _plain_graph().invoke({"trail": []})
    shipped = runs(installed.spans)
    assert len(shipped) == 1, "no run span at all is what an empty workflow_name looks like"
    assert shipped[0].workflow_name == "LangGraph"


def test_a_graph_whose_name_is_empty_falls_back_to_the_literal(installed):  # noqa: F811
    """The adapter's OWN `or "LangGraph"`, which the test above cannot reach.

    langgraph's `compile()` already defaults the name to "LangGraph", so an
    ordinary unnamed graph never exercises the fallback; clearing `.name` is
    what puts a falsy value in front of `_graph_name`, and it pins that the
    fallback is that same literal rather than "" or the string "None" — either
    of which `vocabulary_name` and `_check_structure` treat as no name at all."""
    app = _plain_graph()
    app.name = None
    app.invoke({"trail": []})
    shipped = runs(installed.spans)
    assert len(shipped) == 1
    assert shipped[0].workflow_name == "LangGraph"


def test_a_named_graph_carries_its_own_name(installed):  # noqa: F811
    """The graph's name, not the framework's. A fallback that fired
    unconditionally would file every workflow in a process under one name and
    make the run span useless for grouping."""
    chain(installed.ctx, 1, name="Pipeline", leaves=False).invoke({"trail": []})
    shipped = runs(installed.spans)
    assert len(shipped) == 1
    assert shipped[0].workflow_name == "Pipeline"
    assert shipped[0].name == "invoke_workflow Pipeline"


# --------------------------------------------------------------------------
# wardex.step.name — EXECUTE_STEP's required extra key
# --------------------------------------------------------------------------


def test_every_step_span_carries_the_step_name(installed):  # noqa: F811
    """`EXECUTE_STEP` declares `wardex.step.name` as a REQUIRED extra key, and
    the step's identity lives nowhere else — §6.4 keeps it a namespaced extra
    rather than a typed block, so one step span without it is one step gone."""
    chain(installed.ctx, 3, name="Named", leaves=False).invoke({"trail": []})
    shipped = steps(installed.spans)
    assert len(shipped) == 3
    assert {extra_of(s)["wardex.step.name"] for s in shipped} == {"n0", "n1", "n2"}


def test_a_describe_that_drops_the_step_name_deletes_the_span(installed, monkeypatch):  # noqa: F811
    """The NEGATIVE, and the only signal a describe-side breach ever produces.

    A vocabulary breach is raised by `finish()` inside `assembly._units.emit`'s
    guard, so the span does not degrade — it disappears, with no marker on
    anything and nothing on the wire to notice. `assembly._units.emit` is the
    entire record. Without this test a future author folds `_describe_node` into
    its optional half, every other assertion in this file still passes, and the
    node layer of the tree is silently gone.
    """
    import wardex_sdk._adapters._langgraph as mod

    def _without_the_name(adapter, task_, step):
        step.draft.set_extra("wardex.framework", "langgraph")

    monkeypatch.setattr(mod, "_describe_node", _without_the_name)
    chain(installed.ctx, 1, name="Blind", leaves=False).invoke({"trail": []})

    assert steps(installed.spans) == []
    assert assembly_counters().get("assembly._units.emit") == 1
    # The loss is confined to the step: the run seam still shipped and the node
    # seam still ENTERED, so "deleted at emit" stays distinguishable from "never
    # opened" — which are two different bugs with two different fixes.
    assert len(runs(installed.spans)) == 1
    assert adapter_counters()["adapters.langgraph.active.runner.run_with_retry"] == 1


# --------------------------------------------------------------------------
# wardex.step.index — the falsy real value
# --------------------------------------------------------------------------


def test_a_functional_api_task_carries_step_index_zero(installed):  # noqa: F811
    """`0` is a REAL index — the entrypoint's, and every `@task`'s — and it is
    falsy, so `if index:` drops it where `isinstance(index, int)` keeps it. An
    absent index and an index of zero are different facts about a step, and the
    truthiness spelling is the one way to conflate them."""

    @task
    def double(x: int) -> int:
        return x * 2

    @entrypoint()
    def wf(x: int) -> int:
        return double(x).result()

    wf.invoke(3)
    by_name = {extra_of(s)["wardex.step.name"]: extra_of(s) for s in steps(installed.spans)}
    assert set(by_name) == {"wf", "double"}
    for name, extra in by_name.items():
        assert "wardex.step.index" in extra, f"{name} lost a falsy-but-real index"
        assert extra["wardex.step.index"] == 0
        assert isinstance(extra["wardex.step.index"], int)


# --------------------------------------------------------------------------
# wardex.step.namespace — the same nesting the tree claims
# --------------------------------------------------------------------------


def test_the_namespace_chain_agrees_with_the_parentage_chain(installed):  # noqa: F811
    """Two ways of saying where a step sits, and they may not disagree.

    The namespace is langgraph's own word (`langgraph_checkpoint_ns`), joined
    with `|` per subgraph level and each segment spelled `name:task_id`; the
    parent edge is the one wardex read from the context. If a patch-site change
    ever reparents a subgraph's nodes, this is where the two stop agreeing — and
    it is the only place a merely PLAUSIBLE tree can be caught, because every
    span in one still reads `unit_active` at confidence 1.0.
    """
    _subgraph_app().invoke({"trail": []})
    spans = installed.spans
    # Counted before the dict, because `{s.name: s}` silently keeps the LAST of
    # any duplicate — a seam that opened two spans per task would collapse into
    # one entry here and every assertion below would pass on the survivor.
    assert len(steps(spans)) == 2, [s.name for s in steps(spans)]
    by_name = {s.name: s for s in steps(spans)}
    outer, inner = by_name["execute_step outer"], by_name["execute_step inner"]

    ns_outer = extra_of(outer)["wardex.step.namespace"]
    ns_inner = extra_of(inner)["wardex.step.namespace"]
    # `|` is the real separator, verified against a live two-level graph rather
    # than assumed: each SEGMENT is `name:task_id`, so a test that split on `:`
    # would pass on one segment and never see the nesting at all.
    assert ns_outer.split("|") == [f"outer:{extra_of(outer)['wardex.step.task_id']}"]
    assert ns_inner.split("|") == [ns_outer, f"inner:{extra_of(inner)['wardex.step.task_id']}"]

    assert _ancestors(spans, outer) == ["invoke_workflow Outer"]
    assert _ancestors(spans, inner) == [
        "invoke_workflow Sub",
        "execute_step outer",
        "invoke_workflow Outer",
    ]
    for step in (outer, inner):
        depth = len(extra_of(step)["wardex.step.namespace"].split("|"))
        enclosing = sum(a.startswith("execute_step") for a in _ancestors(spans, step))
        assert depth == enclosing + 1


# --------------------------------------------------------------------------
# wardex.framework
# --------------------------------------------------------------------------


def test_every_span_this_adapter_produces_names_its_framework(installed):  # noqa: F811
    """One key on all three span kinds, so a consumer can select this adapter's
    output without pattern-matching span names. It is written in the MANDATORY
    half of every `describe`, beside the key or block the intent requires, so
    losing it costs the span rather than leaving an unattributable one behind."""
    tool_graph([_CALL]).invoke({"messages": []})
    spans = installed.spans
    assert len(runs(spans)) == 1
    assert len(steps(spans)) == 2  # model, tools
    assert len(tools(spans)) == 1
    assert len(spans) == 4, "these three kinds are the whole of what this adapter emits"
    for span in spans:
        assert extra_of(span)["wardex.framework"] == "langgraph", span.name


# --------------------------------------------------------------------------
# arity — how many entries each span kind puts in `extra`
# --------------------------------------------------------------------------


def test_a_node_span_carries_exactly_seven_extra_entries(installed):  # noqa: F811
    """SIX from `_describe_node` plus ONE the builder writes itself.

    `SpanDraft.finish()` inserts `gen_ai.operation.name` for any intent with no
    `gen_ai` block, so the count is always one above what the adapter wrote — a
    fact worth pinning, because an author adding a key counts their own and gets
    an off-by-one that reads as the builder having changed under them.

    `wardex.step.trigger` is final vocabulary for the scheduling edge, per the
    same node-is-not-an-agent decision the adapter module's docstring records.
    """
    chain(installed.ctx, 1, name="Arity", leaves=False).invoke({"trail": []})
    extra = extra_of(steps(installed.spans)[0])
    assert len(extra) == 7, extra
    assert set(extra) == {
        "gen_ai.operation.name",
        "wardex.framework",
        "wardex.step.name",
        "wardex.step.task_id",
        "wardex.step.index",
        "wardex.step.trigger",
        "wardex.step.namespace",
    }
    assert extra["gen_ai.operation.name"] == "execute_step"


def test_a_tool_span_carries_two_extras_and_the_tool_block_adds_none(installed):  # noqa: F811
    """TWO, and `ToolAttributes` is not among them.

    The tool block is a TYPED FIELD on `InternalSpan`; the Rust codec flattens
    it into `Span.extra` on the WIRE and Python's `extra` never sees it. A test
    that expected seven here would be reading the wire's shape off the wrong
    side of the codec — and would go green on an adapter that had abandoned the
    typed field for five hand-written keys, which is the domain leak §6.4 rules
    out. The flattening is asserted where it happens, in the codec test below.
    """
    span = _one_tool_span(installed, pure_add, _CALL)
    extra = extra_of(span)
    assert len(extra) == 2, extra
    assert set(extra) == {"gen_ai.operation.name", "wardex.framework"}
    assert span.tool is not None
    assert [k for k in extra if k.startswith("gen_ai.tool.")] == []


def test_a_tool_span_carries_three_extras_when_a_command_named_a_node(installed):  # noqa: F811
    """The one optional tool key, and the only shape that adds an entry."""

    @tool
    def hop(x: int) -> Command:
        """Hop somewhere."""
        return Command(
            update={"messages": [ToolMessage(content="hopped", tool_call_id="toolu_02")]},
            goto="somewhere",
        )

    span = _one_tool_span(
        installed, hop, call("hop", {"x": 1}, "toolu_02"), name="Goto", extra_node="somewhere"
    )
    extra = extra_of(span)
    assert len(extra) == 3, extra
    assert extra["wardex.langgraph.command_goto"] == "somewhere"


def test_a_run_span_carries_two_extras_and_three_with_a_thread_id(installed):  # noqa: F811
    """The optional half of `_describe_run` is one key, and it is there only when
    the CALLER supplied a config — so these two counts are the two shapes a run
    span legitimately has, and a third would mean a key grew without a decision."""
    app = chain(installed.ctx, 1, name="RunArity", leaves=False)
    app.invoke({"trail": []})
    bare = extra_of(runs(installed.spans)[0])
    assert len(bare) == 2, bare
    assert set(bare) == {"gen_ai.operation.name", "wardex.framework"}

    installed.client.spans.clear()
    app.invoke({"trail": []}, config={"configurable": {"thread_id": "t-1"}})
    with_thread = extra_of(runs(installed.spans)[0])
    assert len(with_thread) == 3, with_thread
    assert with_thread["wardex.langgraph.thread_id"] == "t-1"


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------


def test_both_run_spans_of_a_resume_carry_the_thread_id_and_no_checkpoint_key(installed):  # noqa: F811,E501
    """A CORRECTED claim, pinned here so it cannot silently regrow.

    The thread id is the only identifier the RUN seam can read: `Pregel.stream`
    is handed the caller's `configurable`, and a checkpoint id is not in it —
    langgraph mints one inside the run. So `wardex.langgraph.checkpoint_id` and
    `wardex.langgraph.checkpoint_ns` genuinely do not exist here, and someone
    pairing the two runs of an interrupt/resume should find the thread id doing
    that job rather than go looking for a key that was never written.
    (`wardex.step.namespace` IS a checkpoint namespace, but it is a STEP key and
    comes from the task's own metadata, which is why this asserts over run spans
    only.)
    """

    def ask(state: TrailState) -> TrailState:
        return {"trail": [str(interrupt("what?"))]}

    g = StateGraph(TrailState)
    g.add_node("ask", ask)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    app = g.compile(checkpointer=InMemorySaver(), name="Resumable")

    config = {"configurable": {"thread_id": "th-9"}}
    app.invoke({"trail": []}, config=config)
    app.invoke(Command(resume="yes"), config=config)

    shipped = runs(installed.spans)
    assert len(shipped) == 2, "one span per graph run; a resume is its own run"
    for span in shipped:
        extra = extra_of(span)
        assert extra["wardex.langgraph.thread_id"] == "th-9"
        assert [k for k in extra if "checkpoint" in k] == []


# --------------------------------------------------------------------------
# the tool block
# --------------------------------------------------------------------------


def test_the_tool_block_is_typed_and_complete_for_a_registered_tool(installed):  # noqa: F811
    """`call_id` reaches the wire as a PAYLOAD field and never as a selector: it
    joins the tool span to the assistant turn that asked for the call, while the
    parent edge came from the context like everything else here. Asserted
    against the call's own id so a later `describe` cannot quietly publish some
    other identifier in that slot."""
    span = _one_tool_span(installed, pure_add, _CALL)
    block = span.tool
    assert block.name == _CALL["name"]
    assert block.call_id == _CALL["id"]
    assert block.type is ToolType.FUNCTION
    assert block.execution_type is ToolExecutionType.IN_PROCESS
    assert block.description == "Add two numbers."


def test_an_unregistered_tool_name_leaves_the_description_absent(installed):  # noqa: F811
    """`None` is INFORMATION: it is how a hallucinated tool name is told apart
    from a real one on a span that is otherwise identical. A `""` default, or a
    placeholder string, erases the only difference there is."""
    span = _one_tool_span(installed, pure_add, call("no_such_tool", {"a": 1}, "toolu_03"))
    assert span.tool.name == "no_such_tool"
    assert span.tool.description is None


def test_the_tool_call_id_limitation_is_absent_because_the_id_is_here(installed):  # noqa: F811
    """An ABSENCE that carries information. The Agent SDK's in-process tools are
    dispatched with `{name, arguments}` and mark every span
    `TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS`; this seam is handed the whole `call`
    dict, so the id is always available and the marker must never appear —
    otherwise a consumer filtering on it drops joins that are perfectly good."""
    span = _one_tool_span(installed, pure_add, _CALL)
    _, _, limitations = edge_of(span)
    assert Limitation.TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS not in limitations
    # The whole tuple, not just this member. A `not in` passes on a span with no
    # integrity block at all, which is also what a tool span looks like when
    # `record_input`/`record_output` never ran — so the narrow assertion would
    # go green on a seam that had stopped capturing io entirely.
    assert limitations == (), f"a healthy in-process tool call reports nothing: {limitations}"
    assert span.capture_integrity is not None, (
        "the tool seam records input AND output, so this span has an integrity "
        "block — its emptiness is the claim, not its absence"
    )
    assert span.capture_integrity.request_body_captured is True
    assert span.capture_integrity.response_body_captured is True
    assert span.tool.call_id == _CALL["id"]


# --------------------------------------------------------------------------
# the tool payload — four shapes, and no `repr` among them
# --------------------------------------------------------------------------


def test_tool_input_is_the_repr_of_the_arguments(installed):  # noqa: F811
    """Repr-identical for exact-builtin shapes under `max_body_bytes`, which is
    what a dict of model-JSON arguments always is. The old justification — "a
    repr of them is bounded by what the model asked for" — is false in general:
    `ToolNode` injects `InjectedState`/`Command` values into `call["args"]`
    before `_run_one`, so the dict's size is set by graph state. `_shaped_args`
    is what makes this assertion safe to keep: foreign objects ship as bare
    type names and materialization is bounded at the source, while the healthy
    shape asserted here stays byte-identical to `repr`."""
    span = _one_tool_span(installed, pure_add, _CALL)
    assert span.input_data == repr(_CALL["args"]).encode()
    assert span.input_data == b"{'a': 1, 'b': 2}"


def test_a_command_argument_ships_as_its_type_name_not_graph_state(installed):  # noqa: F811
    """The input-side twin of `test_a_command_carrying_graph_state_never_
    materializes_it`: 200 KB of channel state inside `call["args"]` ships as
    the 16-byte spelling `{'cmd': Command}` — the state was DECLINED, not cut,
    so the span must not claim truncation either."""

    @tool
    def take_cmd(cmd: Any) -> str:
        """Accept a command-shaped argument."""
        return "took it"

    span = _one_tool_span(
        installed,
        take_cmd,
        call("take_cmd", {"cmd": Command(update={"blob": "x" * 200_000})}, "toolu_20"),
    )
    assert span.input_data == b"{'cmd': Command}"
    assert len(span.input_data) < 100, f"{len(span.input_data)} bytes of channel state shipped"
    assert span.capture_integrity is not None
    assert span.capture_integrity.truncated is False, (
        "nothing recorded was dropped — declined is not truncated"
    )


def test_an_over_budget_tool_input_ships_truncated_and_flagged(installed):  # noqa: F811
    """End-to-end proof of the +1 handshake: the shaper returned 65 bytes,
    `_append_capped` kept 64 and set the flag. The budget is lowered at the
    registry's own attribute — the enforcement point `record_budget` reads —
    so the shaper's budget and the storage cap share one source by
    construction and cannot be lowered apart."""
    installed.ctx._units._max_record_bytes = 64

    over = {"a": "y" * 500}
    span = _one_tool_span(installed, pure_add, call("pure_add", over, "toolu_21"))
    assert len(span.input_data) == 64
    assert span.input_data == repr(over).encode()[:64], "a prefix, not garbage"
    assert span.capture_integrity is not None
    assert span.capture_integrity.truncated is True


@pytest.mark.xfail(
    strict=True,
    reason="the registry the context builds keeps the core body cap, so the budget does too",
)
def test_shaped_args_uses_the_configured_budget():
    """The source side of the same bound: what the shaper MATERIALIZES.

    The test above lowers the cap at the registry attribute, which proves the
    +1 handshake but says nothing about where the number came from. This one
    configures the host the way a host does — `LimitsConfig(max_body_bytes=...)`
    on the client — and drives a real tool call through a real `ToolNode`. The
    protection this bound is bought for is the one a host that lowered it wants
    most: a 200 KB argument must not be spelled out in full inside the host's
    own tool-call thread just to be cut afterwards.
    """
    live = _capped_install(LimitsConfig(max_body_bytes=4096))
    try:
        big = {"a": "y" * 200_000}
        span = _one_tool_span(live, pure_add, call("pure_add", big, "toolu_budget"))
    finally:
        live.teardown()

    assert live.ctx.record_budget == 4096
    assert len(span.input_data) == 4096
    assert span.capture_integrity is not None
    assert span.capture_integrity.truncated is True


def test_a_string_result_is_recorded_as_the_message_content(installed):  # noqa: F811
    """Shape one: `ToolMessage.content` is a `str`, and it is the tool's own
    words, so it ships as itself."""

    @tool
    def speak(word: str) -> str:
        """Say a word back."""
        return f"the tool said {word}"

    span = _one_tool_span(installed, speak, call("speak", {"word": "hello"}, "toolu_04"))
    assert span.output_data == b"the tool said hello"


def test_a_block_list_result_is_the_block_text_and_never_the_dicts(installed):  # noqa: F811
    """Shape two, and the reason it is not `str(content)`: a multimodal result's
    blocks are dicts whose repr carries a base64 image, so the obvious spelling
    puts the whole picture on the wire under an attribute nobody sized for it."""

    @tool
    def multimodal(x: int) -> ToolMessage:
        """Return two content blocks, one of them an image."""
        return ToolMessage(
            content=[
                {"type": "text", "text": "part one"},
                {"type": "image", "source": {"type": "base64", "data": "QUJD" * 64}},
            ],
            tool_call_id="toolu_05",
        )

    span = _one_tool_span(installed, multimodal, call("multimodal", {"x": 1}, "toolu_05"))
    assert span.output_data == b"part one"
    assert b"base64" not in span.output_data
    assert b"{" not in span.output_data


def test_a_command_result_is_recorded_as_its_type_name(installed):  # noqa: F811
    """Shape three: a `Command` is graph control flow, not a tool result, and the
    honest record of one is that a `Command` came back."""

    @tool
    def hop(x: int) -> Command:
        """Hop somewhere."""
        return Command(
            update={"messages": [ToolMessage(content="hopped", tool_call_id="toolu_06")]},
            goto="somewhere",
        )

    span = _one_tool_span(
        installed, hop, call("hop", {"x": 1}, "toolu_06"), name="Cmd", extra_node="somewhere"
    )
    assert span.output_data == b"Command"


def test_a_command_carrying_graph_state_never_materializes_it(installed):  # noqa: F811
    """THE assertion the payload rule exists for.

    A one-line `repr` fallback satisfies every other payload test in this file
    and, measured, put 200 131 bytes of graph channel state on the wire with
    `truncated` UNSET — the default body cap is two orders of magnitude larger
    than the payload, so nothing downstream flags it either. `_tool_payload` has
    three cases and no fallback precisely so no shape can reach a spelling that
    builds something unbounded, and this is the only test that fails when the
    fallback comes back.
    """

    class BlobState(TypedDict):
        messages: Annotated[list, lambda a, b: a + b]
        blob: str

    @tool
    def bloat(x: int) -> Command:
        """Return a Command whose update is 200 KB of channel state."""
        return Command(
            update={
                "messages": [ToolMessage(content="ok", tool_call_id="toolu_07")],
                "blob": "x" * 200_000,
            }
        )

    def model_node(state):
        calls = [call("bloat", {"x": 1}, "toolu_07")]
        return {"messages": [AIMessage(content="", tool_calls=calls)]}

    g = StateGraph(BlobState)
    g.add_node("model", model_node)
    g.add_node("tools", ToolNode([bloat]))
    g.add_edge(START, "model")
    g.add_edge("model", "tools")
    g.add_edge("tools", END)
    app = g.compile()
    app.name = "Bloat"
    app.invoke({"messages": [], "blob": ""})

    span = tools(installed.spans)[0]
    assert len(span.output_data) < 100, f"{len(span.output_data)} bytes of channel state shipped"
    assert span.output_data == b"Command"


# --------------------------------------------------------------------------
# wardex.langgraph.command_goto
# --------------------------------------------------------------------------


def test_command_goto_is_published_for_a_name_and_withheld_for_a_send(installed):  # noqa: F811
    """`Command.goto` is typed `Send | Sequence[Send | str] | str`, and only the
    string shapes ARE names. A `Send` carries a node name PLUS a payload, which
    is a second decision this slice does not make, so it is omitted rather than
    guessed — and the omission is asserted so that a later `str(goto)` cannot
    slip a repr in under a key that promises a destination.

    DECISION: this extra IS the handoff vocabulary. No `HANDOFF` span is
    minted, because a node has no honest agent identity to put in the AGENT
    block that intent requires — the adapter module's docstring records the
    trigger for revisiting."""

    @tool
    def by_name(x: int) -> Command:
        """Go to a named node."""
        return Command(
            update={"messages": [ToolMessage(content="a", tool_call_id="toolu_08")]},
            goto="somewhere",
        )

    @tool
    def by_send(x: int) -> Command:
        """Go to a node with a payload."""
        return Command(
            update={"messages": [ToolMessage(content="b", tool_call_id="toolu_09")]},
            goto=Send("somewhere", {"messages": []}),
        )

    named = _one_tool_span(
        installed, by_name, call("by_name", {"x": 1}, "toolu_08"), name="N", extra_node="somewhere"
    )
    assert extra_of(named)["wardex.langgraph.command_goto"] == "somewhere"

    installed.client.spans.clear()
    sent = _one_tool_span(
        installed, by_send, call("by_send", {"x": 1}, "toolu_09"), name="S", extra_node="somewhere"
    )
    assert "wardex.langgraph.command_goto" not in extra_of(sent)


# --------------------------------------------------------------------------
# the codec
# --------------------------------------------------------------------------


def test_a_run_a_node_and_a_tool_span_survive_the_native_codec(installed):  # noqa: F811
    """Everything above is measured in Python; this measures the WIRE.

    Three things can be lost between the two and not one of them raises: the
    parent edge (a subtree silently reparented to nothing), the
    strategy/confidence pair (every guessed edge becoming indistinguishable from
    a read one), and the int-ness of `wardex.step.index` (a `0` arriving as
    `"0"` breaks any consumer that orders on it).

    The tool block shows up here as five `gen_ai.tool.*`/`wardex.tool.*` keys it
    does not have on the Python side — the flattening the arity test refuses to
    count, asserted on the side of the codec where it actually happens.
    """
    tool_graph([_CALL]).invoke({"messages": []})
    spans = installed.spans
    run = runs(spans)[0]
    node = next(s for s in steps(spans) if s.name == "execute_step tools")
    tool_span = tools(spans)[0]

    envelope = Envelope(header=_header(), spans=(run, node, tool_span))
    decoded = _codec.decode(_codec.encode(envelope))
    out = {item["span"]["name"]: item["span"] for item in decoded["items"]}
    assert set(out) == {"invoke_workflow HandBuilt", "execute_step tools", "execute_tool pure_add"}

    wire_run = out["invoke_workflow HandBuilt"]
    wire_node = out["execute_step tools"]
    wire_tool = out["execute_tool pure_add"]

    # the edge, end to end — by id, because a reparented subtree keeps every
    # other field intact
    assert wire_run["parent_span_id"] == b""
    assert wire_node["parent_span_id"] == wire_run["span_id"]
    assert wire_tool["parent_span_id"] == wire_node["span_id"]

    assert wire_run["correlation"]["strategy"] == ParentSource.TRACE_ROOT.value
    assert wire_run["correlation"]["confidence"] == 1.0
    for wire in (wire_node, wire_tool):
        assert wire["correlation"]["strategy"] == ParentSource.UNIT_ACTIVE.value
        assert wire["correlation"]["confidence"] == 1.0

    assert wire_run["workflow_name"] == "HandBuilt"

    node_extra = {kv["key"]: kv["value"] for kv in wire_node["extra"]}
    assert node_extra == extra_of(node), "every key, same value, same type"
    index = node_extra["wardex.step.index"]
    assert isinstance(index, int) and not isinstance(index, bool)

    tool_extra = {kv["key"]: kv["value"] for kv in wire_tool["extra"]}
    assert tool_extra["gen_ai.tool.name"] == _CALL["name"]
    assert tool_extra["gen_ai.tool.call.id"] == _CALL["id"]
    assert tool_extra["gen_ai.tool.description"] == "Add two numbers."
    assert tool_extra["gen_ai.tool.type"] == ToolType.FUNCTION.value
    assert tool_extra["wardex.tool.execution_type"] == ToolExecutionType.IN_PROCESS.value
    assert wire_tool["input_data"] == _bytes(tool_span.input_data)
    assert wire_tool["output_data"] == _bytes(tool_span.output_data)
