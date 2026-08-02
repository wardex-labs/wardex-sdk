"""In-process tool parentage — the product claim, end to end (design §5.3-iii, §5.4).

Competitors reconstruct the causal tree from a framework's own callback
identifiers. wardex derives it from in-process context propagation, and this
file is where that stops being an architecture diagram: an in-process MCP tool
handler is dispatched from a task the SDK spawns inside its transport read loop,
and the ONLY thing connecting it to the agent run is the ContextVar the reader
task carries. No `session_id`, no `tool_use_id`, nothing the framework issued.

The mechanism, stated once:

  * an async generator body has no context of its own — its frames run in the
    context of the task that DRIVES it;
  * the SDK drives the transport's `read_messages()` from one long-lived reader
    task per `Query`, cancelled and awaited at close;
  * so a `ContextVar.set()` performed inside that loop installs the session unit
    on the reader task and cannot outlive it;
  * and every hook callback and `tools/call` dispatch is spawned FROM that loop,
    so each inherits the session by ordinary context copying.

`_ReaderDispatchTransport` below is that dispatch, modelled at the one point
that matters (a task created from inside the read loop's own body) so the tests
do not need a Claude Code subprocess to exercise it.

What these tests prove, and what they do not
--------------------------------------------

The parentage assertions were built by writing rival implementations that
reconstruct the tree some other way and keeping only the assertions those rivals
fail. Six shapes are ruled out, each by a specific construction, and no single
test rules out all six:

  * ANSWERING FROM A FRAMEWORK IDENTIFIER. Two runs dispatch a tool each at the
    same instant and both edges are asserted by identity, so no value read out of
    either stream can be right twice. The nested case adds the half an identifier
    cannot reach at all: the enclosing CALL has no id to be keyed on.
  * A PROCESS-GLOBAL REGISTER — a module-level "current run", a "current span", a
    LIFO of "the call we are inside". Every scenario keeps two brackets open
    simultaneously and asserts BOTH, and one register holds one value at a time.
    The reader-task test adds a probe on a task the pin never reached, taken
    while the run is live: a global register answers it, a fork does not.
  * A WALK OF THE PYTHON CALL STACK. The nested call is dispatched from a fresh
    task, so no chain of frames joins the inner handler to the enclosing one.
  * PER-ASYNCIO-TASK STATE — a task-keyed table, or a task lineage graph walked
    retrospectively. This is the shape that agrees with propagation everywhere
    the work stays on the task graph, which is everywhere above; the two part
    company at the thread offload, because a copied Context follows work into a
    pool thread and an edge in a task graph does not. That test asserts through
    BOTH carriers for a related reason: a build that forks the scope correctly
    and reconstructs the unit from the task graph gets the manual span right and
    silently attributes the tool call one level too high.
  * A LINEAGE GRAPH OVER SPAWN POINTS — the same idea extended until it covers
    threads too: an edge recorded at `create_task`, at `run_in_executor`, at
    `ThreadPoolExecutor.submit`, at `Thread.start`, the node set being tasks
    and threads together, and both questions answered by walking it. That build
    passes every guard above, the thread offload included, because it patches
    the offload as well. Adding one more scenario only teaches it one more
    spelling to patch, so what is asserted instead is the two things that
    separate an ENUMERATION OF MECHANISMS from propagation:

    IT NEEDS A SPAWN POINT TO EXIST, and one transfer has none.
    `contextvars.Context.run` runs work under a copy of a context with no task
    created, no thread started and nothing submitted anywhere — the copy IS the
    transfer — so there is no primitive to hang an edge on, and no number of
    additional spellings ever produces one. That is the replayed-callback test.
    The reused anyio worker is the same fact in production dress:
    `anyio.to_thread.run_sync` starts a worker ONCE and then hands it job after
    job with `Context.run`, so the single edge available at `Thread.start` names
    whoever needed a worker first and every later job inherits that stranger.

    IT READS LIVE STATE WHERE INHERITANCE READS A SNAPSHOT. A copied context is
    frozen at the moment the work was ISSUED; a graph walk arrives at an
    ancestor and reads what that ancestor is inside NOW. Two tests make those
    disagree — work issued inside one call whose task then moves on into a
    second call, and a task the reader created before the session was pinned to
    it — and both times the walk answers with state that did not exist when the
    work was issued.

  * PROPAGATION, REIMPLEMENTED BY HAND — the answer to both structural facts
    above, and the closest rival there is. A snapshot register keyed by task and
    thread, copied BY VALUE at every spawn point so a child is frozen at what
    was ambient when it was issued, plus a hook on `contextvars.copy_context`
    so the register rides along with a captured context and `Context.run`
    reinstalls it. That build passes every guard listed so far, the replayed
    callback and the reused anyio worker included. What it cannot do is see the
    copies the INTERPRETER makes in C, where there is no Python call to hook at
    either end: `Future.add_done_callback` captures the caller's context at
    registration in C, and the loop replays it through a `Handle` that was
    HANDED that context and so never calls `copy_context()` at all.
    `loop.call_soon(cb, context=ctx)` and `Task(coro, context=ctx)` are the same
    family. That is the done-callback test, and it is the guard no hand-rolled
    register can be extended to pass, because the thing it would have to
    intercept is not written in Python. Measured on such a build: the callback's
    work gets no parent and a trace of its own, while everything else stays
    green.

NOT proven here: anything about work reaching a thread that was never handed a
copied context — a pre-existing worker draining a queue, a callback invoked from
a foreign thread, an executor submission that passes no context. wardex has no
edge to that work either, and nothing below claims one.

Also not proven, and written down because a guard that implies more than it
shows is the defect this list exists to remove:

  * that any of this is measured against `claude_agent_sdk`'s own dispatch.
    Every guard here drives `_ReaderDispatchTransport` or a subclass of it — a
    MODEL of the read loop, reproducing the one structural fact the claim rests
    on: the handler task is created from inside the body that drives
    `read_messages()`, and therefore copies that body's context. The fact is
    real today — `Query._read_messages` routes every `control_request` into
    `spawn_detached`, and both a `tools/call` and a hook callback arrive as one
    — and the tee under test is the shipped one, but the dispatch is ours and
    only the shape of it is asserted. Were the library to move that
    spawn off the reader task, every parentage assertion below would stay green
    while the shipped adapter lost the edge. A reader must not mistake the model
    for the library.
  * that any of it holds on a backend other than asyncio. `claude_agent_sdk` is
    an anyio program and supports trio; every guard here runs `anyio.run` on its
    default backend and reaches for `asyncio.create_task`, `asyncio.to_thread`
    and asyncio futures by name. The mechanism is `contextvars`, which is
    backend-independent, but that is an argument and not a measurement.
  * that any single construction here rules out much. Three of them — the
    replayed callback, the reused anyio worker and the done callback — have one
    run live and one call open at a time, so a process-global "current run"
    register with no per-task state at all passes all three; it is ruled out
    only by the scenarios that hold two brackets open simultaneously and by the
    probes taken on tasks the pin never reached. The file rules things out; the
    tests mostly do not.
  * that a REPLAYED context is a TRUE one. Inheritance is faithful, not
    clairvoyant: a host that captures one call's context and replays it under
    unrelated work gets that work attributed to that call, at confidence 1.0.
    The carrier reports what it was handed, and nothing here second-guesses it.
  * that the pre-pin task's edge is wrong-looking by SHAPE. It is not. The sole
    live session is the same node a live graph walk hands over, and only the
    0.5 and `UNIT_INFERRED_SOLE` say it was a guess — a consumer that reads the
    tree and ignores the confidence sees no difference at all.
  * one offload spelling, one worker. The anyio test drives two sequential jobs
    through a single reused worker, and says nothing about several workers in
    flight at once.

No rival implementation written against this file has passed all of it, which is
a measured floor and not a proof. What the file rules out is answering from a
framework identifier, from a process-global register, from the interpreter's
frames, from a graph of spawn points read after the fact, and from a register
that copies those snapshots by hand at every spawn point Python can see. What
still passes is an implementation that carries its answer INSIDE the context it
hands to the work — and that one has adopted this mechanism rather than replaced
it.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import threading

import anyio
import claude_agent_sdk
import pytest
from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk._internal.transport.subprocess_cli import SubprocessCLITransport

from test_agent_sdk_adapter_install import INIT_LINE, RESULT_LINE, FakeTransport
from wardex_sdk import _hub
from wardex_sdk._enums import StatusCode, ToolExecutionType
from wardex_sdk.adapters._anthropic_agent_sdk import AnthropicAgentSdkAdapter
from wardex_sdk.adapters._anthropic_names import ServerHandle
from wardex_sdk.adapters._context import AdapterContext, Placement
from wardex_sdk.adapters._registry import context_for
from wardex_sdk.assembly import (
    Limitation,
    ParentSource,
    SpanIntent,
    UnitKind,
    UnitRegistry,
    counters,
)


class RecordingClient:
    """A client double. `close()` exists because two tests put this on the hub —
    `wardex.span()` inside a tool handler has to reach the same sink as the
    adapter — and both the hub reset and conftest's teardown close what they find.
    """

    config = None

    def __init__(self):
        self.spans = []

    def capture_span(self, span):
        self.spans.append(span)

    def close(self):
        return None


@pytest.fixture(autouse=True)
def _clean_scope():
    """No ambient span from a neighbouring test — the session latches the scope."""
    _hub.reset_for_test()
    yield
    _hub.reset_for_test()


def _tool(name="greet", handler=None):
    async def default(args):
        return {"content": [{"type": "text", "text": f"hi {args.get('name')}"}]}

    return claude_agent_sdk.SdkMcpTool(
        name=name, description="d", input_schema={}, handler=handler or default
    )


class _ReaderDispatchTransport(FakeTransport):
    """A transport whose read loop dispatches a tool call, as the SDK's does.

    The one structural fact being reproduced: `claude_agent_sdk` spawns the
    `tools/call` handler (and every hook callback) with `spawn_detached` from
    inside `_read_messages`, i.e. from the task driving the transport's message
    generator. `asyncio.create_task` copies the current context exactly as
    `spawn_detached` does, so a unit pinned by the tee is visible to the spawned
    task here for the same reason it is there.

    `probe` runs inside that spawned task, before the handler, so a test can ask
    what the dispatch actually sees rather than inferring it from the span tree.
    """

    def __init__(self, adapter, tool, *, session_id="s-1", tool_use_id="toolu_1", hook_name=None):
        super().__init__([])
        self.adapter = adapter
        self.tool = tool
        self.session_id = session_id
        self.tool_use_id = tool_use_id
        self.hook_name = hook_name
        self.tool_result = None
        self.probe = None
        self.probed = None

    def read_messages(self):
        async def gen():
            await self._init_seen.wait()
            yield {
                "type": "control_response",
                "response": {"subtype": "success", "request_id": self._init_request_id},
            }
            yield dict(INIT_LINE, session_id=self.session_id)
            self.tool_result = await asyncio.create_task(self._dispatch())
            yield dict(RESULT_LINE, session_id=self.session_id)

        return gen()

    async def _dispatch(self):
        if self.probe is not None:
            self.probed = self.probe()
        payload = {
            "session_id": self.session_id,
            "tool_name": self.hook_name,
            "tool_input": {"name": "world"},
        }
        if self.hook_name is not None:
            self.adapter._on_hook("PreToolUse", payload, self.tool_use_id)
        result = await self.tool.handler({"name": "world"})
        if self.hook_name is not None:
            self.adapter._on_hook(
                "PostToolUse",
                {**payload, "tool_response": "hi world"},
                self.tool_use_id,
            )
        return result


def _run(adapter, transport, *, server_key="tools", tools=None, before_close=None):
    """Drive one `query()` with a wrapped in-process server attached."""
    server = claude_agent_sdk.create_sdk_mcp_server("srv", tools=tools or [transport.tool])
    options = ClaudeAgentOptions(mcp_servers={server_key: server})

    async def main():
        received = []
        async for msg in claude_agent_sdk.query(prompt="x", options=options, transport=transport):
            received.append(msg)
            if before_close is not None:
                await before_close()
        return received

    return anyio.run(main)


def _named(spans, name):
    return next(s for s in spans if s.name == name)


def _two_live_sessions(*, sid_a, sid_b, host_labels=None):
    """Two `query()` runs live at once, and EACH dispatches its own tool.

    SYMMETRY IS THE MECHANISM. A one-sided version of this scenario — only A
    dispatches — is not a test of context propagation at all, because A is
    registered first and announced first, so "the oldest live run wins" and "the
    first id the stream announced wins" are both right by construction. Those
    are exactly the fallbacks a tracer with no per-task state has to use once it
    has run out of framework identifiers, and a one-sided scenario waves them
    through.

    With both readers dispatching while both runs are live, no process-global
    answer to "which run is this call part of?" can be right, whatever its
    tie-break: one register holds one value at a time, and the two calls need
    two different ones. `a_tool` must land under A and `b_tool` under B, and any
    implementation that answers from anything but the dispatching TASK gives
    both calls the same parent.

    The choreography is enforced by events and never by timing. The tee calls
    `_on_inbound` and installs the pin BEFORE it yields each message, and an
    async generator only resumes when its consumer asks for the next item, so:

      * `a_up` / `b_up` are set after the corresponding init line has been fully
        observed. Neither reader dispatches until both are set, so at both
        dispatches both sessions are registered, pinned and live — and the
        framework's "most recently announced run" is the SAME one for both,
        which is what stops it from being accidentally right for either.
      * the two handlers rendezvous, so each call's parent is resolved while the
        other call is open. The two calls genuinely overlap; the returned
        entry/exit log is how a caller asserts that rather than assuming it.
      * neither reader yields its result line until both dispatches returned, so
        neither session closes early and drains the other's evidence.
    """
    a_up, b_up = anyio.Event(), anyio.Event()
    a_done, b_done = anyio.Event(), anyio.Event()
    open_a, open_b = anyio.Event(), anyio.Event()
    order: list[str] = []

    def _rendezvous(tag, mine, theirs):
        async def handler(args):
            order.append(f"enter:{tag}")
            mine.set()
            await theirs.wait()  # the other call's parent is decided in here
            order.append(f"exit:{tag}")
            return {"content": [{"type": "text", "text": tag}]}

        return handler

    tool_a = _tool("a_tool", handler=_rendezvous("a", open_a, open_b))
    tool_b = _tool("b_tool", handler=_rendezvous("b", open_b, open_a))

    class _A(FakeTransport):
        def read_messages(self):
            async def gen():
                await self._init_seen.wait()
                yield {
                    "type": "control_response",
                    "response": {"subtype": "success", "request_id": self._init_request_id},
                }
                yield dict(INIT_LINE, session_id=sid_a)
                a_up.set()
                await b_up.wait()  # both runs announced, both pinned
                await asyncio.create_task(tool_a.handler({"name": "world"}))
                a_done.set()
                await b_done.wait()  # stay live across B's dispatch
                yield dict(RESULT_LINE, session_id=sid_a)

            return gen()

    class _B(FakeTransport):
        def read_messages(self):
            async def gen():
                await self._init_seen.wait()
                yield {
                    "type": "control_response",
                    "response": {"subtype": "success", "request_id": self._init_request_id},
                }
                yield dict(INIT_LINE, session_id=sid_b)
                b_up.set()
                await a_up.wait()  # both runs announced, both pinned
                await asyncio.create_task(tool_b.handler({"name": "world"}))
                b_done.set()
                await a_done.wait()  # stay live across A's dispatch
                yield dict(RESULT_LINE, session_id=sid_b)

            return gen()

    server = claude_agent_sdk.create_sdk_mcp_server("srv", tools=[tool_a, tool_b])
    options = ClaudeAgentOptions(mcp_servers={"tools": server})

    async def drive(transport, label):
        if label is None:
            async for _ in claude_agent_sdk.query(prompt="x", options=options, transport=transport):
                pass
            return
        import wardex_sdk

        with wardex_sdk.trace(label):
            async for _ in claude_agent_sdk.query(prompt="x", options=options, transport=transport):
                pass

    label_a, label_b = host_labels if host_labels else (None, None)

    async def main():
        async with anyio.create_task_group() as tg:
            tg.start_soon(drive, _A([]), label_a)
            tg.start_soon(drive, _B([]), label_b)

    anyio.run(main)
    return order


def _assert_calls_overlapped(order):
    """Both calls were open at once, so both runs were live at both resolutions.

    Without this the scenario can silently degrade into two consecutive calls,
    where a single process-global "current run" is free to be right about each
    in turn — which is the exact weakness the symmetric construction removes.
    """
    assert [entry.split(":")[0] for entry in order] == ["enter", "enter", "exit", "exit"]


# ==========================================================================
# The claim: an in-process tool span parented by CONTEXT, not by an identifier
# ==========================================================================


def test_a_tool_is_parented_by_the_task_that_dispatched_it_not_the_id_in_the_stream():
    """The product claim, made falsifiable.

    Two agent runs share one process and each dispatches a tool from its OWN
    reader task, both calls open at the same time. Context propagation answers
    "A" for one and "B" for the other, because each handler's task descends from
    one specific reader. Nothing else can: `tools/call` hands the handler
    `{name, arguments}` and nothing else, so an implementation without per-task
    state has only a process-global notion of "the current run" to fall back on —
    and one register cannot hold the two different answers this scenario needs at
    the same instant.

    That is why BOTH edges are asserted, and asserted by IDENTITY. Asserting one
    of them tests the tie-break rather than the mechanism: the run that dispatches
    is then also the oldest, the first-announced and the first-pinned, so "the
    oldest live run wins" — the ordinary fallback of a tracer with no context
    propagation — is right for the wrong reason. Asserting both leaves no
    ordering, in either direction, that is right twice.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        order = _two_live_sessions(sid_a="s-A", sid_b="s-B")
    finally:
        adapter.uninstall()

    _assert_calls_overlapped(order)

    roots = [s for s in client.spans if s.name == "invoke_agent"]
    assert len(roots) == 2
    root_a = next(r for r in roots if r.conversation.session_id == "s-A")
    root_b = next(r for r in roots if r.conversation.session_id == "s-B")
    tool_a = _named(client.spans, "execute_tool a_tool")
    tool_b = _named(client.spans, "execute_tool b_tool")

    # IDENTITY, not shape: each parent IS its own session's root span object.
    # Shape survives every way of getting this wrong; identity does not.
    assert tool_a.parent_span_id == root_a.context.span_id
    assert tool_b.parent_span_id == root_b.context.span_id
    # The two calls disagree about the answer, which no single register can.
    assert tool_a.parent_span_id != tool_b.parent_span_id
    assert tool_a.context.trace_id == root_a.context.trace_id
    assert tool_b.context.trace_id == root_b.context.trace_id
    # And both are full-confidence edges, so no marker can excuse a wrong one.
    for tool in (tool_a, tool_b):
        assert tool.correlation.confidence == 1.0
        assert Limitation.UNIT_INFERRED_SOLE not in tool.capture_integrity.limitations
        assert Limitation.PARENT_UNRESOLVED not in tool.capture_integrity.limitations


def test_an_in_process_tool_span_is_a_child_of_the_session_at_full_confidence():
    """The broken tree this closes. The handler used to open a manual span with
    no ambient parent, so `TraceId.generate()` fired and the tool — plus every
    HTTP request it made, captured accurately — landed in a trace of its own,
    invisible from the session. The tool node was then ALSO missing from the
    session tree, because the skip list suppressed the hook-driven span for it.

    Every assertion here is about provenance, not shape: the edge carries
    `unit_active` at confidence 1.0 and NO framework identifier, which is what
    distinguishes it from a tree reconstructed out of `session_id`.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        transport = _ReaderDispatchTransport(adapter, _tool())
        _run(adapter, transport)
    finally:
        adapter.uninstall()

    root = _named(client.spans, "invoke_agent")
    tool = _named(client.spans, "execute_tool greet")

    assert tool.parent_span_id == root.context.span_id
    assert tool.context.trace_id == root.context.trace_id
    assert tool.correlation.strategy is ParentSource.UNIT_ACTIVE
    assert tool.correlation.confidence == 1.0
    # The whole point: nothing the framework issued took part in deciding this.
    assert tool.correlation.request_id is None
    assert tool.correlation.operation_id is None
    assert tool.tool.execution_type is ToolExecutionType.IN_PROCESS
    assert tool.status is StatusCode.OK
    # `tools/call` hands the handler `{name, arguments}` and nothing else, so the
    # id genuinely is not available — recorded rather than guessed at.
    assert Limitation.TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS in tool.capture_integrity.limitations
    assert Limitation.TOOL_NAME_COLLISION not in tool.capture_integrity.limitations


def test_a_nested_in_process_tool_is_a_child_of_the_enclosing_call_not_the_session():
    """The edge no identifier can forge, and no stack can guess.

    An in-process handler is handed `{name, arguments}` and nothing else — the
    span says so, with `TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS` — so the enclosing
    CALL has no framework key to be looked up by. Only the carrier the outer call
    activated can name it. A tree built out of identifiers can reach the SESSION
    and stop there, which is depth 1 where the truth is depth 2.

    INTERLEAVED, and that is the second half of the guard. A single nested call
    on one task is a lexically nested bracket, and a process-global LIFO of "the
    call we are inside" reproduces it exactly — push, push, pop, pop — with no
    propagation at all. Claude issues tool calls in PARALLEL, so the real shape is
    two brackets open at once on two tasks, and a stack has one top: whichever
    nested call asks second is handed the OTHER one, because that is what was
    opened most recently. Per-task propagation hands each nested call the outer
    call on ITS OWN task, and is unmoved by what the other task is doing.

    OFF THE ENCLOSING CALL STACK, and that is the third half. Awaited inline, a
    nested handler runs in a frame whose `f_back` chain reaches the enclosing
    handler's own frame — and every asyncio task has a stack of its own, so
    walking the interpreter's frames for the enclosing call is per-task, needs no
    state at all, and reproduces this edge exactly. Dispatched from a FRESH TASK
    there is no stack bridge left: the loop steps the new task from its own
    frame, and the only thing still joining inner to outer is the context that
    task copied at creation.

    Both nested edges are asserted, so no push order is right twice. The
    rendezvous is what forces the interleave, and the entry/exit log is asserted
    rather than assumed — run sequentially this degrades back into two lexically
    nested brackets, which is the case that proves nothing.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))

    outer_in_a, outer_in_b = anyio.Event(), anyio.Event()
    inner_in_a, inner_in_b = anyio.Event(), anyio.Event()
    order: list[str] = []

    def _inner_handler(tag, mine, theirs):
        async def handler(args):
            order.append(f"inner_enter:{tag}")
            mine.set()
            await theirs.wait()  # the other nested call is being resolved in here
            order.append(f"inner_exit:{tag}")
            return {"content": []}

        return handler

    inner_a = _tool("inner_a", handler=_inner_handler("a", inner_in_a, inner_in_b))
    inner_b = _tool("inner_b", handler=_inner_handler("b", inner_in_b, inner_in_a))

    def _outer_handler(tag, inner, mine, theirs):
        async def handler(args):
            order.append(f"outer_enter:{tag}")
            mine.set()
            await theirs.wait()  # both outer calls are open before either nests
            # A NEW TASK rather than an inline await, so the Python call stack no
            # longer bridges the outer handler to the inner one — see the third
            # paragraph above. `inner.handler` is already wardex's wrapper by the
            # time this runs, so the nested call still goes through `_run_tool`,
            # inside the outer call's `activate()` that the new task inherits.
            return await asyncio.create_task(inner.handler({"name": tag}))

        return handler

    outer_a = _tool("outer_a", handler=_outer_handler("a", inner_a, outer_in_a, outer_in_b))
    outer_b = _tool("outer_b", handler=_outer_handler("b", inner_b, outer_in_b, outer_in_a))

    class _TwoNested(_ReaderDispatchTransport):
        async def _dispatch(self):
            return await asyncio.gather(
                asyncio.create_task(outer_a.handler({"name": "a"})),
                asyncio.create_task(outer_b.handler({"name": "b"})),
            )

    try:
        transport = _TwoNested(adapter, outer_a)
        _run(adapter, transport, tools=[outer_a, outer_b, inner_a, inner_b])
    finally:
        adapter.uninstall()

    # The interleave, asserted: both outer calls were open before either nested,
    # and the two nested calls were open at the same time. So whichever nested
    # call resolved second saw the OTHER one as the most recently opened bracket,
    # never its own enclosing call — which is the answer a stack would give it.
    def _at(prefix):
        return [i for i, entry in enumerate(order) if entry.startswith(prefix)]

    assert len(_at("outer_enter")) == len(_at("inner_enter")) == len(_at("inner_exit")) == 2
    assert max(_at("outer_enter")) < min(_at("inner_enter"))
    assert max(_at("inner_enter")) < min(_at("inner_exit"))

    root = _named(client.spans, "invoke_agent")
    outer_span_a = _named(client.spans, "execute_tool outer_a")
    outer_span_b = _named(client.spans, "execute_tool outer_b")
    inner_span_a = _named(client.spans, "execute_tool inner_a")
    inner_span_b = _named(client.spans, "execute_tool inner_b")

    # Depth 2, twice, and the identity of each parent is the outer CALL that
    # actually made the nested call.
    assert inner_span_a.parent_span_id == outer_span_a.context.span_id
    assert inner_span_b.parent_span_id == outer_span_b.context.span_id
    assert inner_span_a.parent_span_id != inner_span_b.parent_span_id
    assert inner_span_a.parent_span_id != root.context.span_id
    assert inner_span_b.parent_span_id != root.context.span_id
    assert outer_span_a.parent_span_id == root.context.span_id
    assert outer_span_b.parent_span_id == root.context.span_id
    assert inner_span_a.context.trace_id == root.context.trace_id
    # Full confidence, and nothing the framework issued took part in it.
    for inner_span in (inner_span_a, inner_span_b):
        assert inner_span.correlation.confidence == 1.0
        assert inner_span.correlation.request_id is None
        assert inner_span.correlation.operation_id is None
    # The reason no id could have produced these edges, asserted rather than
    # assumed: the enclosing calls have no id to be keyed on.
    assert (
        Limitation.TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS in outer_span_a.capture_integrity.limitations
    )


def test_the_tree_is_identical_when_every_framework_id_is_replaced():
    """C-3, on the real install path. If any identifier were load-bearing — a
    `session_id` seeding a trace, a `tool_use_id` picking a parent — these two
    runs would not agree, because they share no identifier at all.
    """

    def shape(session_id, tool_use_id, server_key):
        client = RecordingClient()
        adapter = AnthropicAgentSdkAdapter()
        adapter.install(client, context_for(adapter.name(), client))
        try:
            transport = _ReaderDispatchTransport(
                adapter, _tool(), session_id=session_id, tool_use_id=tool_use_id
            )
            _run(adapter, transport, server_key=server_key)
        finally:
            adapter.uninstall()
        root = _named(client.spans, "invoke_agent")
        tool = _named(client.spans, "execute_tool greet")
        return (
            tool.parent_span_id == root.context.span_id,
            tool.context.trace_id == root.context.trace_id,
            tool.correlation.confidence,
        )

    assert shape("s-1", "toolu_1", "tools") == shape("zzz", "call_9", "other") == (True, True, 1.0)


def test_two_sessions_the_cli_gave_one_id_do_not_share_a_subtree():
    """When the identifier is ambiguous, only the context can be right.

    The CLI reports the same `session_id` for two live runs — a resume, a replay,
    a CLI that reuses ids. A lookup keyed on that string cannot tell them apart
    even in principle; the ContextVar can, because the handler's task descends
    from one specific reader.

    The two roots are told apart by the host span each `query()` was issued
    under, which the session latches at `_ensure_session` on the writing task —
    not by the id, which is the thing under test.

    Both runs dispatch, and both edges are asserted, for the reason spelled out
    in `_two_live_sessions`: with only one dispatcher the run that dispatches is
    also the oldest and the first-pinned, so a process-global "current run" is
    right by accident and the ambiguous id never gets tested against anything.
    Two simultaneous calls need two different answers, and a duplicate id can
    supply neither.
    """
    client = RecordingClient()
    _hub.set_client(client)
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        order = _two_live_sessions(sid_a="dup", sid_b="dup", host_labels=("host_A", "host_B"))
    finally:
        adapter.uninstall()

    _assert_calls_overlapped(order)

    host_a = _named(client.spans, "host_A")
    host_b = _named(client.spans, "host_B")
    roots = [s for s in client.spans if s.name == "invoke_agent"]
    assert len(roots) == 2
    root_a = next(r for r in roots if r.parent_span_id == host_a.context.span_id)
    root_b = next(r for r in roots if r.parent_span_id == host_b.context.span_id)
    tool_a = _named(client.spans, "execute_tool a_tool")
    tool_b = _named(client.spans, "execute_tool b_tool")

    # Deliberately NOT a trace comparison. With the carrier removed a tool
    # attaches straight to its host span, skipping the session root, and the
    # trace ids still agree — the tree has lost a node and a relation assertion
    # would call it correct. Only identity bites.
    assert tool_a.parent_span_id == root_a.context.span_id
    assert tool_b.parent_span_id == root_b.context.span_id
    assert tool_a.parent_span_id != tool_b.parent_span_id
    assert tool_a.correlation.confidence == 1.0
    assert tool_b.correlation.confidence == 1.0


def test_the_tool_span_hangs_under_the_hosts_own_span():
    """The session is latched on the task that ISSUED the run, so a host that
    wraps `query()` in its own span gets the agent — and everything under it —
    inside that span rather than beside it.
    """
    import wardex_sdk

    client = RecordingClient()
    _hub.set_client(client)
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        transport = _ReaderDispatchTransport(adapter, _tool())
        with wardex_sdk.trace("caller") as outer:
            _run(adapter, transport)
        outer_ctx = outer.context
    finally:
        adapter.uninstall()

    root = _named(client.spans, "invoke_agent")
    tool = _named(client.spans, "execute_tool greet")
    assert root.parent_span_id == outer_ctx.span_id
    assert tool.context.trace_id == outer_ctx.trace_id


def test_the_handler_body_runs_inside_the_tool_span_not_merely_beside_it():
    """I3: computing a `SpanContext` is not parenthood — being IN the context is.

    A nested manual span is the cheapest observer of that, and it stands in for
    the thing that actually matters: an HTTP request the tool makes in-process is
    seen by the byte seam, passes the `capture_mode=AGENT` gate because a wardex
    span is active, and attaches HERE instead of under an orphan root.
    """
    import wardex_sdk

    client = RecordingClient()
    _hub.set_client(client)
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))

    async def handler(args):
        with wardex_sdk.span("inner"):
            pass
        return {"content": []}

    try:
        transport = _ReaderDispatchTransport(adapter, _tool(handler=handler))
        _run(adapter, transport)
    finally:
        adapter.uninstall()

    tool = _named(client.spans, "execute_tool greet")
    inner = _named(client.spans, "inner")
    assert inner.parent_span_id == tool.context.span_id


def test_work_the_handler_offloads_to_a_thread_stays_inside_the_tool_span():
    """A synchronous tool, which is the ordinary shape of an in-process one.

    A handler that wraps blocking code — a database driver, a file walk, an SDK
    with no async surface — hands the real work to a thread and awaits it:
    `asyncio.to_thread`, or the `run_in_executor` spelling underneath it.
    `to_thread` copies the caller's `Context` into the worker, so the tool's
    activation travels WITH the work and everything the thread then does stays
    inside the tool span and inside the run's trace. The manual span here is the
    cheapest observer of that; what it stands in for is an HTTP request the tool
    makes from the thread, seen by the byte seam and attached HERE.

    It sits with the guards above, and not merely beside them, because this is
    where PER-TASK and PROPAGATED stop being the same claim. Every other guard
    in this file can be satisfied by answering "which run am I in, and what am I
    inside" from the asyncio task — a task-keyed table, a task lineage graph
    walked after the fact, anything at all that is per-task. There is no task
    here: the work runs on a pool thread that no task graph has an edge to, so
    the same question answered per-task returns nothing, the work roots a trace
    of its own, and the tool's activity becomes accurate content hanging off
    nothing and invisible from the run. Copying the context is the whole
    difference, and an assertion made OFF the event loop is the only one that
    can see it.

    TWO OBSERVERS run in the thread, because the carrier has two halves and an
    implementation can get one right while getting the other wrong. The manual
    span reads the SCOPE — the half an HTTP request the byte seam sees also
    reads. The nested in-process tool call reads the UNIT REGISTRY — the half
    every adapter observation reads. Measured on a build that forks the scope
    properly and reconstructs the unit from the task graph: the manual span is
    correct and the nested call is attributed to the SESSION instead, at
    confidence 1.0 with no marker, which is a tree that has silently lost a
    level. One observer cannot see that; both can.

    `asyncio.run` is how synchronous code calls an async handler, and it copies
    the calling context into the task it creates — so the nested call is
    dispatched from the thread and still arrives inside the enclosing tool.
    """
    import wardex_sdk

    client = RecordingClient()
    _hub.set_client(client)
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))

    nested = _tool("in_thread_tool")
    idents: dict[str, int] = {}

    def blocking():
        idents["worker"] = threading.get_ident()
        with wardex_sdk.span("in_thread"):
            pass
        # `nested.handler` is wardex's wrapper by now, so this goes through the
        # same in-process tool path as any other call — from a thread.
        asyncio.run(nested.handler({"name": "world"}))

    async def handler(args):
        idents["loop"] = threading.get_ident()
        await asyncio.to_thread(blocking)
        return {"content": []}

    try:
        transport = _ReaderDispatchTransport(adapter, _tool(handler=handler))
        _run(adapter, transport, tools=[transport.tool, nested])
    finally:
        adapter.uninstall()

    # The precondition. Had the offload silently run on the loop thread this
    # would degrade into the nested-span case above — which a per-task answer
    # satisfies, and which is exactly what this test exists to go beyond.
    assert idents["worker"] != idents["loop"]

    root = _named(client.spans, "invoke_agent")
    tool = _named(client.spans, "execute_tool greet")
    in_thread = _named(client.spans, "in_thread")
    in_thread_tool = _named(client.spans, "execute_tool in_thread_tool")

    # All three of the ways the scope half fails when the context does not
    # travel: a different parent, no parent at all, and a different trace.
    assert in_thread.parent_span_id == tool.context.span_id
    assert in_thread.parent_span_id is not None
    assert in_thread.context.trace_id == tool.context.trace_id == root.context.trace_id

    # And the registry half, which is the one a correct scope fork does not
    # imply: depth 2 rather than the session's depth 1, at full confidence and
    # with no inference marker to excuse a guess.
    assert in_thread_tool.parent_span_id == tool.context.span_id
    assert in_thread_tool.parent_span_id != root.context.span_id
    assert in_thread_tool.correlation.strategy is ParentSource.UNIT_ACTIVE
    assert in_thread_tool.correlation.confidence == 1.0
    assert Limitation.UNIT_INFERRED_SOLE not in in_thread_tool.capture_integrity.limitations


def test_a_callback_replayed_under_a_captured_context_lands_inside_the_call_that_issued_it():
    """A handler that hands its dispatcher a callback AND the context to run it in.

    This is an ordinary shape, not a curiosity. `contextvars.Context` is the
    standard way to say "run this later, as if it were still here":
    `loop.call_soon(cb, context=ctx)` stores one, `anyio`'s worker threads
    replay one, and any in-process dispatcher — an event bus, a completion
    callback, a deferred continuation — that wants a callback to observe its
    issuer's state captures one and calls `Context.run`. The handler below does
    exactly that: it captures `copy_context()` and stays open while its
    dispatcher replays the callback.

    It sits here because it is the one transfer with NO SPAWN POINT. Every other
    way work moves in this file announces itself to whoever is watching the
    concurrency primitives — `create_task`, `to_thread`, `run_in_executor`,
    `Thread.start`. `Context.run` announces nothing: no task is created, no
    thread is started, nothing is submitted anywhere. The copy IS the transfer.
    So an implementation that reconstructs "which run am I in, and what am I
    inside" by recording an edge at each concurrency primitive has nothing to
    record here, however many spellings it learns to recognise — while an
    implementation that reads the ambient context is right without doing
    anything at all. The replay is deliberately performed on the call's PARENT
    task, so any answer taken from the live state of the running task is the
    SESSION and the enclosing call is lost.

    TWO OBSERVERS, for the reason the thread test gives: the manual span reads
    the scope half, the nested in-process call reads the unit registry half, and
    a build can get one right while getting the other wrong.
    """
    import wardex_sdk

    client = RecordingClient()
    _hub.set_client(client)
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))

    nested = _tool("replayed_tool")
    captured: list[contextvars.Context] = []
    handed_over, replayed = anyio.Event(), anyio.Event()
    spawned: list[asyncio.Task] = []

    async def handler(args):
        captured.append(contextvars.copy_context())
        handed_over.set()
        await replayed.wait()  # the call stays open across the replay
        return {"content": []}

    def callback():
        with wardex_sdk.span("replayed_span"):
            pass
        # Dispatched from INSIDE the replayed context, so the task copies that
        # context and not the dispatcher's.
        spawned.append(asyncio.create_task(nested.handler({"name": "world"})))

    class _ReplayFromTheDispatcher(_ReaderDispatchTransport):
        async def _dispatch(self):
            call = asyncio.create_task(self.tool.handler({"name": "world"}))
            await handed_over.wait()
            # No task created, no thread started, nothing submitted: the copied
            # context is the whole mechanism, and this is the call's PARENT task.
            captured[0].run(callback)
            await spawned[0]
            replayed.set()
            return await call

    try:
        transport = _ReplayFromTheDispatcher(adapter, _tool(handler=handler))
        _run(adapter, transport, tools=[transport.tool, nested])
    finally:
        adapter.uninstall()

    root = _named(client.spans, "invoke_agent")
    tool = _named(client.spans, "execute_tool greet")
    replayed_span = _named(client.spans, "replayed_span")
    replayed_tool = _named(client.spans, "execute_tool replayed_tool")

    # The scope half: inside the call that captured the context, not beside it
    # and not one level up under the session.
    assert replayed_span.parent_span_id == tool.context.span_id
    assert replayed_span.parent_span_id != root.context.span_id
    assert replayed_span.context.trace_id == tool.context.trace_id == root.context.trace_id

    # The registry half, which a correct scope fork does not imply.
    assert replayed_tool.parent_span_id == tool.context.span_id
    assert replayed_tool.parent_span_id != root.context.span_id
    assert replayed_tool.context.trace_id == root.context.trace_id
    assert replayed_tool.correlation.strategy is ParentSource.UNIT_ACTIVE
    assert replayed_tool.correlation.confidence == 1.0
    assert Limitation.UNIT_INFERRED_SOLE not in replayed_tool.capture_integrity.limitations
    assert Limitation.PARENT_UNRESOLVED not in replayed_tool.capture_integrity.limitations


def test_work_issued_inside_a_call_keeps_that_call_after_the_task_moves_on():
    """A SNAPSHOT taken when the work was issued, not the issuer's state now.

    A tool handler starts a piece of work and does not wait for it — a fire-and-
    forget notification, a background fetch it collects later — and then the same
    handler goes on to make a second, nested tool call. The spawned work asks its
    question while that second call is open.

    The two answers differ, and only one of them is causal. The spawned work was
    issued BY the outer call, so the outer call is what it belongs to; the inner
    call did not exist yet when it was issued and never touched it. A copied
    context says so by construction, because the copy froze at `create_task` and
    carries what was ambient THEN. An implementation that walks a lineage graph
    reads the ancestor's state AT ANSWER TIME instead: it arrives at the same
    task, sees whatever that task is inside NOW, and hands back the inner call —
    at confidence 1.0, with no marker, one level off in the wrong subtree.

    The interleave is enforced by events and asserted rather than assumed: run
    without the rendezvous the spawned work would ask before the inner call ever
    opened, both answers would agree, and the test would prove nothing.

    Both observers again — the manual span for the scope half, a nested
    in-process call for the registry half.
    """
    import wardex_sdk

    client = RecordingClient()
    _hub.set_client(client)
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))

    issued_tool = _tool("issued_tool")
    inner_open, answered = anyio.Event(), anyio.Event()
    order: list[str] = []
    seen: list[object] = []

    async def issued_work():
        await inner_open.wait()  # ask while the OTHER call is what the task is in
        order.append("issued_asks")
        seen.append(adapter._assembler.units.current())
        with wardex_sdk.span("issued_span"):
            pass
        await issued_tool.handler({"name": "world"})
        answered.set()

    async def inner_handler(args):
        order.append("inner_enter")
        inner_open.set()
        await answered.wait()
        order.append("inner_exit")
        return {"content": []}

    inner = _tool("inner_call", handler=inner_handler)

    async def outer_handler(args):
        order.append("outer_enter")
        # Issued from inside THIS call, and collected much later.
        issued = asyncio.create_task(issued_work())
        # The same task then moves on into a DIFFERENT call, awaited inline so
        # the activation lands on this very task and shadows the outer one.
        await inner.handler({"name": "world"})
        await issued
        order.append("outer_exit")
        return {"content": []}

    outer = _tool("outer_call", handler=outer_handler)

    try:
        transport = _ReaderDispatchTransport(adapter, outer)
        _run(adapter, transport, tools=[outer, inner, issued_tool])
    finally:
        adapter.uninstall()

    # The divergence is only reachable if the spawned work asked while the inner
    # call was open. Asserted, because without it both answers coincide.
    assert order == ["outer_enter", "inner_enter", "issued_asks", "inner_exit", "outer_exit"]

    root = _named(client.spans, "invoke_agent")
    outer_span = _named(client.spans, "execute_tool outer_call")
    inner_span = _named(client.spans, "execute_tool inner_call")
    issued_span = _named(client.spans, "issued_span")
    issued_tool_span = _named(client.spans, "execute_tool issued_tool")

    # Read directly off the registry, not inferred from the tree: the unit the
    # spawned work resolves to IS the outer call's.
    assert seen[0] is not None
    assert seen[0].context.span_id == outer_span.context.span_id

    for span in (issued_span, issued_tool_span):
        assert span.parent_span_id == outer_span.context.span_id
        assert span.parent_span_id != inner_span.context.span_id
        assert span.parent_span_id != root.context.span_id
        assert span.context.trace_id == root.context.trace_id
    assert issued_tool_span.correlation.strategy is ParentSource.UNIT_ACTIVE
    assert issued_tool_span.correlation.confidence == 1.0
    assert Limitation.UNIT_INFERRED_SOLE not in issued_tool_span.capture_integrity.limitations
    assert Limitation.PARENT_UNRESOLVED not in issued_tool_span.capture_integrity.limitations


def test_a_reused_anyio_worker_thread_serves_each_call_its_own_context():
    """The offload spelling `claude_agent_sdk` itself runs on.

    `claude_agent_sdk` is an anyio program, so a handler that offloads blocking
    work idiomatically writes `anyio.to_thread.run_sync`, not `asyncio.to_thread`
    — and the two are not the same mechanism underneath. `asyncio.to_thread`
    goes through `loop.run_in_executor`. anyio does not: it keeps its own pool of
    LONG-LIVED worker threads, hands each job to an idle one through a queue, and
    runs it with `context.run(func, ...)` on a `copy_context()` taken at submit
    time. The thread is started ONCE and then serves job after job for unrelated
    callers.

    That reuse is what this test is for. Because the context is copied per JOB,
    each call's work is inside that call — the second call's thread work is in
    the second call's span even though the worker was first started for the
    first, and the first call has since closed. Anything that instead records an
    edge when the WORKER IS CREATED gets one edge for the lifetime of the thread
    and attributes every later job to the caller that happened to need a worker
    first; here that is a closed call, so the work lands under the session, at
    confidence 1.0 and unmarked.

    The two preconditions are asserted, since the scenario silently degrades
    without them: the job really ran off the event loop, and the SAME worker
    served both calls.
    """
    import wardex_sdk

    client = RecordingClient()
    _hub.set_client(client)
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))

    idents: dict[str, int] = {}
    nested_1, nested_2 = _tool("in_worker_1"), _tool("in_worker_2")

    def _job(tag, nested):
        idents[tag] = threading.get_ident()
        with wardex_sdk.span(f"worker_span_{tag}"):
            pass
        # `nested.handler` is wardex's wrapper by now, so this is an ordinary
        # in-process tool call made from the worker thread.
        asyncio.run(nested.handler({"name": tag}))

    def _handler_for(tag, nested):
        async def handler(args):
            await anyio.to_thread.run_sync(_job, tag, nested)
            return {"content": []}

        return handler

    tool_1 = _tool("worker_tool_1", handler=_handler_for("1", nested_1))
    tool_2 = _tool("worker_tool_2", handler=_handler_for("2", nested_2))

    class _TwoSequentialOffloads(_ReaderDispatchTransport):
        async def _dispatch(self):
            idents["loop"] = threading.get_ident()
            # Each call on its OWN task, which is how the SDK dispatches them,
            # and the first call is CLOSED before the second one offloads.
            await asyncio.create_task(tool_1.handler({"name": "1"}))
            await asyncio.create_task(tool_2.handler({"name": "2"}))

    try:
        transport = _TwoSequentialOffloads(adapter, tool_1)
        _run(adapter, transport, tools=[tool_1, tool_2, nested_1, nested_2])
    finally:
        adapter.uninstall()

    # Off the loop, and the SAME worker twice — the reuse IS the scenario.
    assert idents["1"] != idents["loop"]
    assert idents["2"] == idents["1"]

    root = _named(client.spans, "invoke_agent")
    call_1 = _named(client.spans, "execute_tool worker_tool_1")
    call_2 = _named(client.spans, "execute_tool worker_tool_2")

    for tag, call, other in (("1", call_1, call_2), ("2", call_2, call_1)):
        worker_span = _named(client.spans, f"worker_span_{tag}")
        worker_tool = _named(client.spans, f"execute_tool in_worker_{tag}")
        for span in (worker_span, worker_tool):
            assert span.parent_span_id == call.context.span_id
            assert span.parent_span_id != other.context.span_id
            assert span.parent_span_id != root.context.span_id
            assert span.context.trace_id == root.context.trace_id
        assert worker_tool.correlation.strategy is ParentSource.UNIT_ACTIVE
        assert worker_tool.correlation.confidence == 1.0
        assert Limitation.UNIT_INFERRED_SOLE not in worker_tool.capture_integrity.limitations


def test_a_future_done_callback_lands_inside_the_call_that_registered_it():
    """A handler that schedules follow-up work on the loop.

    This is an ordinary supported shape and the first thing asserted is that it
    simply works. An in-process handler that starts something and wants to react
    when it finishes registers `fut.add_done_callback(cb)`; the callback's work
    belongs to the call that registered it, and lands there — in the call's span,
    in the run's trace, at full confidence, with no marker.

    It sits with the product-claim guards because of WHERE the copy is taken.
    `add_done_callback` captures the caller's context AT REGISTRATION, in C, and
    the loop later replays it through a `Handle` that was HANDED that context and
    so never calls `copy_context()` at all. Both halves happen below Python:
    there is no call to intercept at either end. `loop.call_soon(cb, context=ctx)`
    and `Task(coro, context=ctx)` are the same family.

    Every other transfer in this file announces itself somewhere a Python hook
    can sit — `create_task`, `to_thread`, `run_in_executor`, `Thread.start`, or
    `contextvars.copy_context` itself for the case with no spawn point at all.
    This one announces itself nowhere, and no number of additional hooks reaches
    it, because the interpreter is doing the copying. So a build that replaced
    the ContextVar carrier with a hand-rolled snapshot register — copying by
    value at every spawn point it knows AND wrapping `Context.run`, which is
    enough to satisfy every other guard here — is blind exactly here. Measured on
    such a build: the callback's work gets NO PARENT and a trace of its own,
    while every other test in this file stays green.

    The future is resolved from the call's PARENT task, so anything answered from
    the live state of the task driving the replay is the SESSION and the
    enclosing call is lost. That precondition is recorded on that task at that
    moment rather than assumed.

    TWO OBSERVERS, for the reason the thread test gives: the manual span reads
    the scope half, the nested in-process call reads the unit registry half, and
    a build can get one right while getting the other wrong.
    """
    import wardex_sdk

    client = RecordingClient()
    _hub.set_client(client)
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))

    nested = _tool("callback_tool")
    registered, replayed, release = anyio.Event(), anyio.Event(), anyio.Event()
    futures: list[asyncio.Future] = []
    spawned: list[asyncio.Task] = []
    seen: list[object] = []
    at_replay: list[object] = []

    def on_done(_future):
        # Runs under the context the interpreter captured at registration and
        # replayed through the `Handle`. Nothing Python-visible copied anything,
        # at either end.
        seen.append(adapter._assembler.units.current())
        with wardex_sdk.span("callback_span"):
            pass
        # Dispatched from INSIDE the replayed context, so the task copies that
        # and not whatever the loop was last inside.
        spawned.append(asyncio.create_task(nested.handler({"name": "world"})))
        replayed.set()

    async def handler(args):
        future = asyncio.get_running_loop().create_future()
        future.add_done_callback(on_done)  # the C-level context capture
        futures.append(future)
        registered.set()
        await release.wait()  # the call stays open across the replay
        return {"content": []}

    class _ResolveFromTheDispatcher(_ReaderDispatchTransport):
        async def _dispatch(self):
            call = asyncio.create_task(self.tool.handler({"name": "world"}))
            await registered.wait()
            at_replay.append(adapter._assembler.units.current())
            # Resolving schedules the callback with the context captured at
            # registration — and this is the call's PARENT task, so the live
            # state here is the session and not the call.
            futures[0].set_result(None)
            await replayed.wait()
            await spawned[0]
            release.set()
            return await call

    try:
        transport = _ResolveFromTheDispatcher(adapter, _tool(handler=handler))
        _run(adapter, transport, tools=[transport.tool, nested])
    finally:
        adapter.uninstall()

    root = _named(client.spans, "invoke_agent")
    tool = _named(client.spans, "execute_tool greet")
    callback_span = _named(client.spans, "callback_span")
    callback_tool = _named(client.spans, "execute_tool callback_tool")

    # The precondition that makes the divergence reachable: the task that drove
    # the replay was inside the SESSION, not inside the call. Without it a live
    # read and a replayed snapshot would agree and the guard would prove nothing.
    assert at_replay[0] is not None
    assert at_replay[0].kind is UnitKind.SESSION
    assert at_replay[0].context.span_id == root.context.span_id

    # Read straight off the registry rather than inferred from the tree: the unit
    # the replayed callback resolves to IS the call's.
    assert seen[0] is not None
    assert seen[0].context.span_id == tool.context.span_id

    # The scope half: inside the call that registered the callback, not beside it
    # and not one level up under the session.
    assert callback_span.parent_span_id == tool.context.span_id
    assert callback_span.parent_span_id != root.context.span_id
    assert callback_span.context.trace_id == tool.context.trace_id == root.context.trace_id

    # The registry half, which a correct scope fork does not imply.
    assert callback_tool.parent_span_id == tool.context.span_id
    assert callback_tool.parent_span_id != root.context.span_id
    assert callback_tool.context.trace_id == root.context.trace_id
    assert callback_tool.correlation.strategy is ParentSource.UNIT_ACTIVE
    assert callback_tool.correlation.confidence == 1.0
    assert Limitation.UNIT_INFERRED_SOLE not in callback_tool.capture_integrity.limitations
    assert Limitation.PARENT_UNRESOLVED not in callback_tool.capture_integrity.limitations


# ==========================================================================
# The pin
# ==========================================================================


def test_the_pin_reads_the_returned_object_and_not_the_function():
    """The trap that disables the pin on exactly the transport it was written for.

    `SubprocessCLITransport.read_messages` is a plain `def` returning a
    generator, and the Transport ABC declares the same signature — so
    `isasyncgenfunction` is FALSE for it and for every user transport that
    follows the ABC. A guard asserting on the FUNCTION therefore refuses to pin
    anywhere, silently, while every other test still passes.

    So the shape facts are recorded AND the consequence is driven: the same
    transport whose function the trap misreads is run through the real tee, and
    the pin has to take on the reader task anyway. Recording the shape alone
    would leave the guard itself — `pinnable = inspect.isasyncgen(inner)` — with
    no test at all, which is the state this replaces.
    """
    assert inspect.isasyncgenfunction(SubprocessCLITransport.read_messages) is False
    assert inspect.isasyncgenfunction(FakeTransport.read_messages) is False
    assert inspect.isasyncgen(FakeTransport([]).read_messages()) is True

    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        transport = _ReaderDispatchTransport(adapter, _tool())
        # This very transport has the shape the trap gets wrong: its
        # `read_messages` is a plain `def`, and only the OBJECT it returns is an
        # async generator.
        assert inspect.isasyncgenfunction(type(transport).read_messages) is False
        units = adapter._assembler.units
        transport.probe = lambda: units.current()
        _run(adapter, transport)
        seen = transport.probed
    finally:
        adapter.uninstall()

    # Pinned regardless — which is only true if the guard read the returned
    # object. A guard on the function refuses this transport and `seen` is None.
    assert seen is not None
    assert seen.kind is UnitKind.SESSION


def test_the_dispatch_task_sees_the_session_unit_itself():
    """Directly, rather than inferred from a span tree: the task the SDK spawns
    out of its read loop resolves `units.current()` to the SESSION. That is the
    pin, and it is the single fact every parentage assertion in this file rests
    on.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        transport = _ReaderDispatchTransport(adapter, _tool())
        units = adapter._assembler.units
        transport.probe = lambda: units.current()
        _run(adapter, transport)
        seen = transport.probed
    finally:
        adapter.uninstall()

    assert seen is not None
    assert seen.kind is UnitKind.SESSION


def test_the_reader_task_carries_the_sessions_span_and_not_only_its_unit():
    """Both halves of the pin, or it is not a pin.

    The carrier installs two things: the ambient UNIT, which parents tool spans,
    and the ambient SPAN fork, which parents everything ELSE the reader task
    touches — a host's own `wardex.span()` inside a hook callback, an HTTP
    request the byte seam sees, the `capture_mode=AGENT` gate. Tool parentage
    goes through `parent_unit.child()` and never reads the second one, so every
    other test in this file passes with it missing. This is the one that looks at
    it.

    The probe runs on the task the read loop spawned, before the tool wrapper —
    i.e. exactly where a hook callback runs, outside any CALL activation.

    INSTALLED IS NOT SCOPED, so there is a second probe. The first one on its own
    proves only that SOME span was reachable where the reader runs; a
    process-global "current span", which is what a tracer with no context
    propagation has instead of a scope, satisfies it exactly — with one run live
    there is nothing to tell the two apart. The second probe runs on the USER's
    own task, a SIBLING of the reader, while the same run is live, and it must be
    a trace ROOT: not merely a different parent, a different TRACE. That is the
    difference between a fork that only descendants inherit and a global register
    every caller in the process reads.

    RED until `SessionAssembler.pin_reader` stops discarding the `PinToken`:
    evaluating `pin_driver(...).installed` drops the carrier on the floor, so it
    is refcount-collected the instant the statement returns, the generator behind
    `activate_span` is closed and the span fork is popped — while the ambient
    unit entry survives. Measured on the unfixed tree: unit ambient, span None.
    """
    import wardex_sdk

    client = RecordingClient()
    _hub.set_client(client)
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    outside = []
    try:
        transport = _ReaderDispatchTransport(adapter, _tool())

        def _probe():
            with wardex_sdk.span("host_probe"):
                pass

        async def _probe_from_outside():
            # Runs on the task driving `query()`, which the pin never touched —
            # the reader is its descendant, not its ancestor. `before_close` runs
            # per message; one probe is the scenario. The live-run count is
            # recorded WITH it, because "there was a session to leak" is the
            # precondition: probing after the run ended would make a trace root
            # the right answer for a global register too, and the guard would
            # prove nothing while still passing.
            if outside:
                return
            outside.append(adapter._assembler.open_session_count())
            with wardex_sdk.span("outside_probe"):
                pass

        transport.probe = _probe
        _run(adapter, transport, before_close=_probe_from_outside)
    finally:
        adapter.uninstall()

    root = _named(client.spans, "invoke_agent")
    probe = _named(client.spans, "host_probe")
    tool = _named(client.spans, "execute_tool greet")

    assert probe.parent_span_id == root.context.span_id
    assert probe.context.trace_id == root.context.trace_id
    # Kept deliberately: the tool edge survives a half-installed pin, which is
    # exactly why the two assertions above cannot be folded into another test.
    assert tool.parent_span_id == root.context.span_id

    # And the same span, opened on a task the pin did not reach while that run
    # was still live, is a trace root. A global "current span" hands it the
    # session instead, and the first probe cannot see the difference.
    assert outside == [1]
    outside_probe = _named(client.spans, "outside_probe")
    assert outside_probe.parent_span_id is None
    assert outside_probe.context.trace_id != root.context.trace_id


def test_the_pin_does_not_leak_to_a_task_outside_the_reader():
    """A pin is installed on ONE task and inherited only by its descendants. A
    naive implementation — a module-level "current session", or a pin installed
    on whatever task happened to call in — would hand the session to unrelated
    work in the same process and nothing downstream could tell.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    seen = []
    try:
        transport = _ReaderDispatchTransport(adapter, _tool())
        units = adapter._assembler.units
        server = claude_agent_sdk.create_sdk_mcp_server("srv", tools=[transport.tool])
        options = ClaudeAgentOptions(mcp_servers={"tools": server})

        async def unrelated():
            seen.append(units.current())

        async def main():
            async for _ in claude_agent_sdk.query(prompt="x", options=options, transport=transport):
                # Spawned from the USER's task while the run is live. The reader
                # task is a sibling, not an ancestor, so no context reaches here.
                await asyncio.create_task(unrelated())

        anyio.run(main)
    finally:
        adapter.uninstall()

    assert seen and all(unit is None for unit in seen)


def test_a_task_the_reader_spawned_before_the_pin_never_joins_the_run():
    """A DESCENDANT of the reader that still does not belong to the run.

    The test above rules out a sibling. This one rules out the harder case: a
    task the reader itself created, before the session existed. A transport that
    starts a keep-alive, a watchdog, a queue drainer at the top of its read loop
    produces exactly this — work that outlives the moment it was issued and asks
    its question much later, once a run IS live on its parent.

    Inheritance is a SNAPSHOT, so it answers correctly without knowing anything
    about time: the context this task copied was taken before the pin existed
    and does not contain it, and no later `set()` on the parent can reach
    backwards into a copy already taken. An implementation that walks a lineage
    graph instead reads the parent's state at ANSWER time — which by then holds
    the pin — and hands the task a run it was never part of.

    The parent NODE the two produce is the same, and that is exactly why this
    guard asserts provenance rather than shape. Falling through to the
    sole-live-session tier attaches the call to that session too. The difference
    is what the span CLAIMS: 0.5 and `UNIT_INFERRED_SOLE` — "nothing was ambient,
    there was exactly one run, we guessed" — against a graph walk's 1.0 with no
    marker, which is unfalsifiable downstream. Both preconditions are recorded on
    the reader's own task, because a guess is only the right answer here if the
    pin really was installed and its session really was live and sole.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    units = adapter._assembler.units

    early_tool = _tool("early_tool")
    pinned = anyio.Event()
    seen: list[object] = []
    reader_view: list[object] = []

    async def early_work():
        await pinned.wait()  # ask only once the run IS live on the parent task
        seen.append(units.current())
        await early_tool.handler({"name": "world"})

    class _SpawnBeforeThePin(_ReaderDispatchTransport):
        def read_messages(self):
            async def gen():
                # On the reader task, before the tee has observed a single
                # message — so before any pin exists to be inherited.
                early = asyncio.create_task(early_work())
                await self._init_seen.wait()
                yield {
                    "type": "control_response",
                    "response": {"subtype": "success", "request_id": self._init_request_id},
                }
                yield dict(INIT_LINE, session_id=self.session_id)
                reader_view.append(units.current())
                reader_view.append(adapter._assembler.open_session_count())
                pinned.set()
                await early
                yield dict(RESULT_LINE, session_id=self.session_id)

            return gen()

    try:
        transport = _SpawnBeforeThePin(adapter, early_tool)
        _run(adapter, transport, tools=[early_tool])
    finally:
        adapter.uninstall()

    # The preconditions: the pin IS on the reader task, and its session is the
    # one live run — so the guess below lands on the same node a graph walk
    # would have handed over as fact.
    assert reader_view[0] is not None
    assert reader_view[0].kind is UnitKind.SESSION
    assert reader_view[1] == 1
    # The snapshot the early task copied never contained the pin.
    assert seen == [None]

    root = _named(client.spans, "invoke_agent")
    early = _named(client.spans, "execute_tool early_tool")
    assert early.parent_span_id == root.context.span_id
    assert early.correlation.strategy is ParentSource.UNIT_SOLE
    assert early.correlation.strategy is not ParentSource.UNIT_ACTIVE
    assert early.correlation.confidence == 0.5
    assert Limitation.UNIT_INFERRED_SOLE in early.capture_integrity.limitations


def test_a_closed_session_stops_being_ambient_on_the_reader_task():
    """The other half of the pin's safety, observed where the pin actually is.

    The pin is deliberately never unpinned — the reader task's life bounds it —
    so what must hold is that a session which has closed stops being handed out
    as a parent, AND that the work it stops parenting says so.

    Both facts live in the reader's context and nowhere else. The pin is
    installed inside the transport's message loop, so on the MAIN task
    `_ambient_unit` is unset and `units.current()` returns None at its first
    line, for a reason that has nothing to do with staleness. Asserting it there
    is vacuous, and was: the entire liveness branch of `current()` could be
    deleted and this file stayed green. So the dispatch task captures its own
    `Context` and the late call is made inside it.

    The late call is made while a SECOND, unrelated run is live, because that is
    the case with no other tell. `current()` refuses the dead session, the next
    tier finds exactly one live one and attaches to it at 0.5 with
    `UNIT_INFERRED_SOLE` — a span byte-identical to a tool call that never had a
    session of its own, sitting in a run it has nothing to do with.
    `CORRELATION_CONFLICT` is the whole difference between "we guessed because
    nothing was pinned" and "we guessed because what was pinned had died".

    `pin_leaked`, not `pin_stale`, and the counter names are the misleading
    half: they mean "stale read by the pin's OWNER task" and "stale read by a
    task descended from it". Every hook callback and every `tools/call` dispatch
    is a descendant, so the descendant spelling is the one a real run produces
    and `pin_leaked` is not the anomaly its name suggests.
    """
    counters.reset()
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        units = adapter._assembler.units
        first = _ReaderDispatchTransport(adapter, _tool())
        reader_context = []

        def _probe():
            # The dispatch task's own Context, captured rather than a closure: a
            # closure re-reads on whatever task later calls it, and that task is
            # the main one, which never held the pin.
            reader_context.append(contextvars.copy_context())
            return units.current()

        first.probe = _probe
        _run(adapter, first)
        live_view = first.probed
        assert adapter._assembler.open_session_count() == 0

        second = _ReaderDispatchTransport(adapter, _tool("other_tool"), session_id="s-2")
        late_handler = first.tool.handler  # wardex's wrapper, as the host holds it
        called = []

        async def _call_from_the_finished_readers_context():
            # `before_close` runs per message; one late call is the scenario.
            if called:
                return
            called.append(None)
            # `create_task(context=...)` is 3.11+, and this package supports
            # 3.10. Creating the task INSIDE the captured context reaches the
            # same place on every version: `Task.__init__` copies whatever
            # context is current at construction, and under `Context.run` that
            # is the reader's. The task gets a copy rather than the Context
            # object itself, which is what a real descendant task gets anyway —
            # and this test only reads the pin.
            await reader_context[0].run(asyncio.create_task, late_handler({"name": "late"}))

        _run(adapter, second, before_close=_call_from_the_finished_readers_context)
    finally:
        adapter.uninstall()

    # (a) the pin did reach the dispatch task while the session was live — the
    # precondition without which everything below would pass for no reason.
    assert live_view is not None
    assert live_view.kind is UnitKind.SESSION
    # (b) and it stopped being handed out once that session closed. This counter
    # is the only externally visible effect of the liveness branch itself.
    assert counters.get("assembly._units.pin_leaked") >= 1
    assert called == [None]

    roots = [s for s in client.spans if s.name == "invoke_agent"]
    root_a = next(r for r in roots if r.conversation.session_id == "s-1")
    root_b = next(r for r in roots if r.conversation.session_id == "s-2")
    # Emission order: the first run's tool span closes inside the first run.
    in_session, late = [s for s in client.spans if s.name == "execute_tool greet"]

    assert in_session.parent_span_id == root_a.context.span_id
    assert Limitation.CORRELATION_CONFLICT not in in_session.capture_integrity.limitations
    # (c) the late call is guessed into the wrong run — and says so. Without the
    # marker this span is indistinguishable from a tool call that never had a
    # pin at all, which is the state this test exists to make impossible.
    assert late.parent_span_id == root_b.context.span_id
    assert late.correlation.strategy is ParentSource.UNIT_SOLE
    assert late.correlation.confidence == 0.5
    assert Limitation.UNIT_INFERRED_SOLE in late.capture_integrity.limitations
    assert Limitation.CORRELATION_CONFLICT in late.capture_integrity.limitations


# ==========================================================================
# Arbitration — one call, one span
# ==========================================================================


def test_the_hook_and_the_handler_do_not_both_open_a_span():
    """Through the REAL install path, which is the part that matters.

    The mechanism this replaces was hand-fed in its own test: `skip_tool_names`
    was populated with the CLI's namespaced spelling, which the wrapper never
    produced — it inserted the bare name. Built through `create_sdk_mcp_server`
    the set never matched, and every in-process SDK MCP tool was emitted twice.
    Here both observers normalize into one key and the handler's rank 10 takes it
    from the hook's 0, however late the handler arrives.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        transport = _ReaderDispatchTransport(adapter, _tool(), hook_name="mcp__tools__greet")
        _run(adapter, transport)
    finally:
        adapter.uninstall()

    tools = [s for s in client.spans if s.name.startswith("execute_tool")]
    assert len(tools) == 1
    assert tools[0].name == "execute_tool greet"
    # The winner is the layer that wrapped the execution (§8.4), so the span
    # reports IN_PROCESS rather than the hook's "not observed".
    assert tools[0].tool.execution_type is ToolExecutionType.IN_PROCESS


def test_a_builtin_tool_still_gets_its_hook_driven_span():
    """The arbitration must not swallow tools nobody wrapped: a `Bash` call has
    exactly one observer, and a key space shared with in-process tools would be
    the way to lose it.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        transport = _ReaderDispatchTransport(adapter, _tool(), hook_name="Bash")
        _run(adapter, transport)
    finally:
        adapter.uninstall()

    assert [s.name for s in client.spans].count("execute_tool Bash") == 1


def test_two_concurrent_calls_to_one_tool_both_produce_a_span():
    """`claim()` refuses an equal rank, and reading that refusal as "someone else
    owns this" would delete the second of two parallel calls to the same tool —
    which Claude issues routinely. Ownership is decided by being OUTRANKED, not
    by a failed claim.

    The two calls really do overlap, and the recorded entry/exit order proves it
    rather than assuming it: the first call is held open until the second has
    entered — and therefore claimed — so both spans are in flight at once. Run
    sequentially the second claim would meet a key nobody holds and the test
    would pass without ever exercising the refusal it is named for.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))

    first_in, second_in = anyio.Event(), anyio.Event()
    order = []

    async def handler(args):
        name = args.get("name")
        order.append(f"enter:{name}")
        if name == "first":
            first_in.set()
            await second_in.wait()  # held open across the second call
        else:
            await first_in.wait()
            second_in.set()
        order.append(f"exit:{name}")
        return {"content": []}

    tool = _tool(handler=handler)

    class _TwoCalls(_ReaderDispatchTransport):
        async def _dispatch(self):
            async def call(name):
                return await self.tool.handler({"name": name})

            return await asyncio.gather(
                asyncio.create_task(call("first")), asyncio.create_task(call("second"))
            )

    try:
        transport = _TwoCalls(adapter, tool)
        _run(adapter, transport)
    finally:
        adapter.uninstall()

    # The overlap, asserted: the first call had not returned when the second
    # entered, so the second claimed a key the first still held.
    assert order.index("enter:second") < order.index("exit:first")
    assert [s.name for s in client.spans].count("execute_tool greet") == 2


# ==========================================================================
# Degradation — marked, never silent
# ==========================================================================


def test_a_handler_with_no_unit_available_returns_the_hosts_result_untouched():
    """The first invariant: never break the host. A wrapper the host still holds
    after `uninstall()` has no registry to attach to, so it does exactly what the
    unwrapped handler would have done — and emits nothing rather than inventing a
    parent for a run that no longer exists.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    tool = _tool()
    claude_agent_sdk.create_sdk_mcp_server("srv", tools=[tool])
    wrapped = tool.handler  # the host's long-held reference
    adapter.uninstall()
    counters.reset()

    result = anyio.run(wrapped, {"name": "world"})

    assert result == {"content": [{"type": "text", "text": "hi world"}]}
    assert client.spans == []
    # Handled, not merely survived. Falling into `guard()` would produce the same
    # return value and the same empty span list while counting an internal error
    # on every call the host makes — so the counter is what separates "this path
    # is designed" from "this path happens to be caught".
    assert counters.snapshot() == {}
    # And the host got its own function back, which is I7 rather than luck.
    assert tool.handler is not wrapped


def test_a_handler_exception_propagates_and_still_closes_the_span():
    """The user's exception is the host's control flow: it reaches the caller
    unchanged, and the span it ends says how it ended.

    Called directly rather than through `query()` on purpose — the SDK's own
    reader turns any exception out of a tool into a generic transport error, so
    routing this through it would be testing the SDK's error plumbing instead of
    the wrapper's. What must hold here is that wardex adds nothing between the
    handler's `raise` and its caller: the original type, the original message.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))

    async def handler(args):
        raise ValueError("tool blew up")

    try:
        tool_def = _tool(handler=handler)
        claude_agent_sdk.create_sdk_mcp_server("srv", tools=[tool_def])
        with pytest.raises(ValueError, match="tool blew up"):
            anyio.run(tool_def.handler, {"name": "world"})
    finally:
        adapter.uninstall()

    span = _named(client.spans, "execute_tool greet")
    assert span.status is StatusCode.ERROR
    assert span.error_type == "ValueError"


def test_a_handler_called_with_no_session_at_all_says_the_parent_is_missing():
    """A tool invoked outside any agent run — a unit test, a host calling its own
    tool directly. The span still exists (deleting the observation would be the
    silent drop this design refuses) and it reports confidence 0.0 with
    `PARENT_UNRESOLVED`, so it can never be mistaken for a real attachment.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        tool = _tool()
        claude_agent_sdk.create_sdk_mcp_server("srv", tools=[tool])
        anyio.run(tool.handler, {"name": "world"})
    finally:
        adapter.uninstall()

    span = _named(client.spans, "execute_tool greet")
    assert span.parent_span_id is None
    assert span.correlation.strategy is ParentSource.UNRESOLVED
    assert span.correlation.confidence == 0.0
    assert Limitation.PARENT_UNRESOLVED in span.capture_integrity.limitations


def test_an_unresolved_server_token_marks_the_span_it_cannot_key():
    """The server was never named in any options, so the handler keys on the
    server's own name while the hook keys on the dict key. That is a key SPLIT —
    `claim()` cannot arbitrate between two different keys — so the span says so
    instead of pretending the arbitration happened.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        tool = _tool()
        claude_agent_sdk.create_sdk_mcp_server("srv", tools=[tool])
        anyio.run(tool.handler, {"name": "world"})
    finally:
        adapter.uninstall()

    span = _named(client.spans, "execute_tool greet")
    assert Limitation.TOOL_NAME_COLLISION in span.capture_integrity.limitations


def test_two_servers_exporting_one_bare_name_mark_the_span_the_hook_abandons(monkeypatch):
    """The other half of the same marker, and the harder half to reach.

    With the CLI shipping SDK tools unprefixed and two wrapped servers exporting
    `search`, the hook cannot tell which server ran: `key_for_hook` returns None
    and the observation stands down. So the handler's span is the ONLY record of
    the call, and it has to say why the other one is missing — a dropped
    observation with nothing on the wire explaining it is exactly the failure
    this marker exists for.

    Both tokens must be RESOLVED before the handler runs. Otherwise the key-SPLIT
    emitter supplies the marker instead, the assertion below passes for the wrong
    reason, and the branch under test can be deleted without anything noticing.
    """
    monkeypatch.setenv("CLAUDE_AGENT_SDK_MCP_NO_PREFIX", "1")
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        alpha_search, beta_search = _tool(name="search"), _tool(name="search")
        alpha = claude_agent_sdk.create_sdk_mcp_server("alpha", tools=[alpha_search])
        beta = claude_agent_sdk.create_sdk_mcp_server("beta", tools=[beta_search])
        # Constructing the client is only a way to reach `_prepare_options`,
        # which is the one moment a wrapped server's CLI token is knowable.
        claude_agent_sdk.ClaudeSDKClient(
            options=ClaudeAgentOptions(mcp_servers={"alpha": alpha, "beta": beta})
        )
        # Both handles, and both resolved. `all()` over the handle list alone is
        # the vacuous form of this precondition: it is loudest exactly when the
        # list is empty, which is the case where no token was resolved at all and
        # the marker below comes from the key-SPLIT emitter instead.
        handles = adapter._names._handles
        assert sorted(handle.name for handle in handles) == ["alpha", "beta"]
        assert all(handle.token_resolved for handle in handles)
        anyio.run(alpha_search.handler, {"q": "x"})
    finally:
        adapter.uninstall()

    span = _named(client.spans, "execute_tool search")
    assert Limitation.TOOL_NAME_COLLISION in span.capture_integrity.limitations


def test_the_owner_this_assembler_stamps_is_the_adapters_own_name():
    """Two spellings of one identity, and the whole of `owner` scoping rests on
    them agreeing.

    The assembler stamps `owner` on every session it opens; the tool wrapper
    asks `sole_live(..., owner=ctx.name)` for one. `ctx.name` is
    `adapter.name()`. If the two ever drift, the fallback finds nothing — and
    the failure is not an error, it is every in-process tool call quietly
    becoming its own trace root while the pin is what covers for it.
    """
    from wardex_sdk.adapters._assembler import _OWNER

    assert _OWNER == AnthropicAgentSdkAdapter().name()

    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    try:
        assert adapter._ctx is not None
        assert adapter._ctx.name == _OWNER
        # And one table, not two: the assembler's registry IS the context's, so
        # `owner` is a filter over units the tool wrapper can actually see.
        assert adapter._assembler.units is adapter._ctx._units
    finally:
        adapter.uninstall()


def test_a_tool_wrapper_that_outlives_its_adapter_still_runs_the_hosts_tool():
    """`uninstall()` drops the context in the same latch as the assembler, and a
    wrapper the host still holds a reference to must read that. A context left
    standing would open a unit in a registry the teardown has already swept —
    a span live in a table nothing will ever close.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    handle = ServerHandle(name="srv")

    async def handler(args):
        return {"echoed": args}

    from wardex_sdk.adapters._anthropic_agent_sdk import _run_tool

    adapter.uninstall()
    assert adapter._ctx is None

    result = asyncio.run(_run_tool(adapter, handle, "greet", handler, {"a": 1}))
    assert result == {"echoed": {"a": 1}}
    assert [s for s in client.spans if s.name.startswith("execute_tool")] == []


def test_a_framework_read_that_breaks_costs_the_tool_span_and_never_the_tool_call(monkeypatch):
    """The whole argument for `describe=`, on the real adapter.

    `_names.ambiguous_bare` is a framework read — the kind that breaks when an
    SDK moves an attribute between releases — and the marker it would have
    earned is genuinely earned here. Described inside `enter()`'s own guard, the
    fault costs the WHOLE span, loudly. Described in the `with` body under a
    guard of its own, the same fault ships `status=OK` with full input, full
    output, a real duration and the earned marker simply gone: a span nothing
    downstream can tell from a complete observation.

    Either way the host's tool runs exactly once and returns its own value.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))

    def blow(self, name):
        raise AttributeError("the catalog moved")

    monkeypatch.setattr(type(adapter._names), "ambiguous_bare", blow)

    try:
        tool = _tool()
        claude_agent_sdk.create_sdk_mcp_server("srv", tools=[tool])
        result = anyio.run(tool.handler, {"name": "world"})
    finally:
        adapter.uninstall()

    assert result == {"content": [{"type": "text", "text": "hi world"}]}
    tools = [s for s in client.spans if s.name.startswith("execute_tool")]
    for span in tools:
        assert span.status is not StatusCode.OK, (
            "a span whose description died reads exactly like a complete one"
        )


def test_a_tool_that_raises_reaches_its_caller_even_while_wardex_is_degraded():
    """The adapter-level restatement of the module-level guarantee, because this
    is the shape a third-party adapter author copies.

    A guard around the whole `with` would swallow the host's own exception,
    turning a failing tool into a silently successful one for its caller AND on
    the wire. There is nothing left in the block that needs protecting, so there
    is no reason to reach for one.
    """

    class Broken(UnitRegistry):
        __slots__ = ()

        def open(self, *a, **k):
            raise RuntimeError("wardex is broken at open")

    for registry_is_broken in (False, True):
        client = RecordingClient()
        adapter = AnthropicAgentSdkAdapter()
        ctx = context_for(adapter.name(), client)
        if registry_is_broken:
            ctx = AdapterContext(
                adapter.name(), units=Broken(sink=ctx._units._sink), limits=ctx.limits
            )
        adapter.install(client, ctx)
        mine = ValueError("the host's own failure")

        async def handler(args, exc=mine):
            raise exc

        try:
            tool_def = _tool(handler=handler)
            claude_agent_sdk.create_sdk_mcp_server("srv", tools=[tool_def])
            with pytest.raises(ValueError) as caught:
                anyio.run(tool_def.handler, {"name": "world"})
        finally:
            adapter.uninstall()

        assert caught.value is mine, f"broken={registry_is_broken}: not the same object"


def test_a_result_wardex_cannot_serialize_still_reaches_the_hosts_caller():
    """The last unguarded expression in the `with` body, and it runs on the
    HOST's own return value.

    `json.dumps` documents `TypeError` for an unserializable value, so that is
    what the narrow except caught — but a container whose `items()` raises, a
    `__getattr__` that throws, a lazy proxy over a closed session are none of
    them `TypeError`. There is nothing left inside the body to contain a raise,
    so the host would have lost its result over a span attribute nobody would
    have missed.
    """

    class Hostile(dict):
        def items(self):
            raise RuntimeError("this object cannot be walked")

    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    adapter.install(client, context_for(adapter.name(), client))
    mine = Hostile(a=1)

    async def handler(args):
        return mine

    try:
        tool_def = _tool(handler=handler)
        claude_agent_sdk.create_sdk_mcp_server("srv", tools=[tool_def])
        result = anyio.run(tool_def.handler, {"name": "world"})
    finally:
        adapter.uninstall()

    assert result is mine
    span = _named(client.spans, "execute_tool greet")
    # The span still ships, and says the output capture was attempted and empty
    # rather than claiming a body it never had.
    assert span.output_data == b""
    assert counters.get("adapters.anthropic.tool_input_unserializable") >= 1


def test_two_concurrent_calls_get_their_own_slot_in_the_lookup_table():
    """The selector this site no longer computes for itself.

    A site that names none used to get one shared key — `adapters.<name>` with
    an empty value — so every anonymous unit in the process rebound one alias
    slot and `find()` on it answered "whichever was last", which is not an
    answer. The context mints a unique one now, which is both a better key and
    one the CALL SITE does not have to build in a header that runs before any
    failure boundary exists.
    """
    client = RecordingClient()
    adapter = AnthropicAgentSdkAdapter()
    ctx = context_for(adapter.name(), client)
    adapter.install(client, ctx)
    try:
        keys = []
        for _ in range(3):
            unit = ctx._open(
                UnitKind.CALL,
                intent=SpanIntent.EXECUTE_TOOL,
                placement=Placement.NESTED,
                subject="t",
                selector=None,
                aliases=(),
                start_ns=None,
            )
            keys.append(unit.key)
        assert len({k.value for k in keys}) == 3, keys
        assert {k.namespace for k in keys} == {f"adapters.{adapter.name()}"}
    finally:
        adapter.uninstall()
