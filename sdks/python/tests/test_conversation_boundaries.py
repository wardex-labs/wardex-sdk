"""A conversation id stays inside its block — across every way a block can end.

`wardex.conversation(...)` promises the id on every span inside the block and
on none outside it. Three boundaries broke that promise, each in a way a
backend grouping by `gen_ai.conversation.id` turned into wrong numbers:

* An incoming trace continued INSIDE a conversation (`continue_trace`,
  `continue_from_otel`) started from a blank scope, so every call inside the
  request lost the id. `isolation_scope()` still does, on purpose: it is how a
  host steps out of a conversation.
* A block inside an async generator that the consumer abandoned with `break`
  is closed by the event loop on a task of its own. The consumer's context
  was never handed back the scope from before the block, so every later call
  on it carried the dead block's id, parented under the dead block's span. A
  task the block started keeps what it was started with either way.
* An openai-agents trace pinned inside a block and finished outside it (or the
  other way round) put back a scope from a block that was already over: the
  block's id after the block, or the finished trace's id after the block it
  had ended in.
"""

from __future__ import annotations

import asyncio
import contextvars
import gc
import sys
import threading
import weakref
from collections.abc import Callable
from typing import Any

import pytest
from agents import trace as agents_trace

import wardex_sdk as wardex
from test_conversation_wire import _chats, _conv, _post, _tag, llm  # noqa: F401
from test_openai_agents_wire import agents_env, fake_openai  # noqa: F401
from wardex_sdk import _hub
from wardex_sdk._assembly import counters
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._enums import AdapterName
from wardex_sdk.testing import RecordingTransport

pytestmark = pytest.mark.usefixtures("fresh_counters")

_REMOTE_TRACE = "0af7651916cd43dd8448eb211c80319c"
_TRACEPARENT = f"00-{_REMOTE_TRACE}-b7ad6b7169203331-01"


def _shipped(fn: Callable[[], Any], **init: Any) -> list[Any]:
    """Run `fn` in a context of its own, with wardex installed, and return what shipped.

    A context of its own because the failures here are LEAKS into the caller's
    context: run on the test's own, a leak would outlive the test.
    """
    t = RecordingTransport()
    wardex.init(transport=t, **init)
    try:
        contextvars.Context().run(fn)
        _hub.get_client()._settle()
        wardex.flush()
    finally:
        wardex.close()
    return [sp for env in t.envelopes for sp in env.spans]


def _named(spans: list[Any], name: str) -> Any:
    (span,) = [s for s in spans if s.name == name]
    return span


def _call(spans: list[Any], tag: str) -> Any:
    (span,) = [s for s in _chats(spans) if _tag(s) == tag]
    return span


def _ambient_conversation() -> str | None:
    c = _hub.get_merged_scope().conversation
    return c.conversation_id if c is not None else None


# --------------------------------------------------------------------------
# a continued trace inside a conversation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "enter",
    [
        pytest.param(lambda: wardex.continue_trace({}), id="continue_trace-no-headers"),
        pytest.param(
            lambda: wardex.continue_trace({"traceparent": _TRACEPARENT}),
            id="continue_trace-traceparent",
        ),
        pytest.param(wardex.continue_from_otel, id="continue_from_otel"),
    ],
)
def test_a_trace_continued_inside_a_conversation_keeps_the_conversation(llm, enter):  # noqa: F811
    """The request's spans are the conversation's: the hand-named span, the
    decorated tool and the LLM call read off the wire all carry its id. The
    call after the block carries none."""
    host, port = llm

    @wardex.tool(name="lookup")
    def lookup() -> str:
        return "found"

    def run() -> None:
        with wardex.conversation("chat", id="X"):
            with enter():
                with wardex.span("handle"):
                    _post(host, port, "in-request")
                lookup()
        _post(host, port, "after-block")

    spans = _shipped(run)
    for span in (_named(spans, "handle"), _named(spans, "lookup"), _call(spans, "in-request")):
        assert _conv(span) == "X", span.name
    assert _conv(_call(spans, "after-block")) is None


def test_a_continued_trace_still_starts_over_while_the_conversation_carries_on():
    """What `continue_trace` resets is the TRACE: the request joins the remote
    parent the headers name, not the span the conversation opened. Keeping the
    conversation must not keep the parent too."""

    def run() -> None:
        with wardex.conversation("chat", id="X"):
            with wardex.continue_trace({"traceparent": _TRACEPARENT}):
                with wardex.span("joined"):
                    pass
            with wardex.continue_trace({}):
                with wardex.span("restarted"):
                    pass

    spans = _shipped(run)
    chat, joined, restarted = (_named(spans, n) for n in ("chat", "joined", "restarted"))
    assert joined.context.trace_id.hex() == _REMOTE_TRACE
    assert restarted.parent_span_id is None
    assert restarted.context.trace_id != chat.context.trace_id
    assert _conv(joined) == _conv(restarted) == "X"


def test_isolation_scope_steps_out_of_a_conversation(llm):  # noqa: F811
    """`isolation_scope()` is a fresh current scope, and the way to run work
    that must not be counted in the conversation around it: nothing inside it
    carries the id, and the block's own spans after it do again."""
    host, port = llm

    def run() -> None:
        with wardex.conversation("chat", id="X"):
            with wardex.isolation_scope():
                with wardex.span("shared-cache-refresh"):
                    _post(host, port, "isolated")
            with wardex.span("back-in-the-conversation"):
                pass

    spans = _shipped(run)
    assert _conv(_named(spans, "shared-cache-refresh")) is None
    assert _conv(_call(spans, "isolated")) is None
    assert _conv(_named(spans, "back-in-the-conversation")) == "X"


# --------------------------------------------------------------------------
# a block that ends in another context than it began in
# --------------------------------------------------------------------------


@pytest.mark.parametrize("nested", [False, True], ids=["conversation", "span-in-conversation"])
def test_an_abandoned_async_generator_leaves_no_conversation_behind(llm, nested):  # noqa: F811
    """`async for ... break` — a client that hung up mid-stream. The event loop
    closes the generator on a task of its own; the consumer's context must
    read as it did before the block, not as the dead block."""
    host, port = llm
    seen: dict[str, str | None] = {}

    async def turn():
        with wardex.conversation("chat", id="G"):
            if nested:
                with wardex.span("stream"):
                    yield 1
                    yield 2
            else:
                yield 1
                yield 2

    async def consume() -> None:
        async for _ in turn():
            break
        gc.collect()
        await asyncio.sleep(0.05)  # the loop runs the generator's `aclose()` here
        seen["ambient"] = _ambient_conversation()
        with wardex.span("next-request"):
            pass
        await asyncio.to_thread(_post, host, port, "after-hangup")

    spans = _shipped(lambda: asyncio.run(consume()))
    assert seen["ambient"] is None
    after = _named(spans, "next-request")
    assert _conv(after) is None
    assert _conv(_call(spans, "after-hangup")) is None
    closed = {s.context.span_id for s in spans if s.name in ("chat", "stream")}
    assert after.parent_span_id not in closed
    assert _conv(_named(spans, "chat")) == "G"
    assert counters.snapshot().get("tracing.block_scope_retired", 0) >= 1


def test_a_generator_closed_on_another_thread_leaves_no_conversation_behind():
    """The same boundary without the event loop, deterministically: the block
    is entered on this thread and ended on another one."""
    seen: dict[str, Any] = {}

    def turn():
        with wardex.conversation("chat", id="G"):
            yield 1
            yield 2

    def run() -> None:
        gen = turn()
        next(gen)
        assert _ambient_conversation() == "G"  # inside the block, as it should be
        closer = threading.Thread(target=gen.close)
        closer.start()
        closer.join()
        seen["ambient"] = _ambient_conversation()
        with wardex.span("after"):
            pass

    spans = _shipped(run)
    assert seen["ambient"] is None
    after = _named(spans, "after")
    assert _conv(after) is None
    assert after.parent_span_id != _named(spans, "chat").context.span_id


@pytest.mark.parametrize("abandoned", [False, True], ids=["ended-normally", "abandoned"])
def test_a_task_started_in_a_block_keeps_what_it_was_started_with(abandoned):
    """A task started inside a block is in the conversation and under the
    block's span, and keeps both after the block ends, however it ends: the
    consumer that abandoned a generator takes its own scope back, and the task
    the generator started is not part of that."""

    async def main() -> None:
        go = asyncio.Event()
        tasks: list[asyncio.Task[None]] = []

        async def background() -> None:
            await go.wait()
            with wardex.span("background"):
                pass

        async def turn():
            with wardex.conversation("chat", id="K"):
                tasks.append(asyncio.create_task(background()))
                yield 1
                yield 2

        async for _ in turn():
            if abandoned:
                break
        gc.collect()
        await asyncio.sleep(0.05)
        with wardex.span("consumer-after"):
            pass
        go.set()
        await tasks[0]

    spans = _shipped(lambda: asyncio.run(main()))
    background = _named(spans, "background")
    assert _conv(background) == "K"
    assert background.parent_span_id == _named(spans, "chat").context.span_id
    assert _conv(_named(spans, "consumer-after")) is None


def test_a_callback_bound_in_a_block_keeps_it_on_every_thread():
    """`bind_context` carries a block to deferred work, and keeps carrying it
    after the block ends elsewhere — on the thread that installed the block as
    on any other. The callback runs in a copy of the installer's Context, and a
    copy is not the installer: it does not take the installer's scope back."""
    seen: dict[str, str | None] = {}

    def deferred(where: str) -> None:
        seen[where] = _ambient_conversation()
        with wardex.span(f"bound-{where}"):
            pass

    def turn(bound: list[Callable[[str], None]]):
        with wardex.conversation("chat", id="G"):
            bound.append(wardex.bind_context(deferred))
            yield 1

    def run() -> None:
        bound: list[Callable[[str], None]] = []
        gen = turn(bound)
        next(gen)
        closer = threading.Thread(target=gen.close)
        closer.start()
        closer.join()
        bound[0]("installing-thread")
        other = threading.Thread(target=bound[0], args=("other-thread",))
        other.start()
        other.join()
        seen["installer"] = _ambient_conversation()

    spans = _shipped(run)
    assert seen == {"installing-thread": "G", "other-thread": "G", "installer": None}
    chat = _named(spans, "chat").context.span_id
    for where in ("installing-thread", "other-thread"):
        bound_span = _named(spans, f"bound-{where}")
        assert (_conv(bound_span), bound_span.parent_span_id) == ("G", chat)


def test_work_started_right_after_an_abandon_does_not_carry_the_dead_block():
    """On 3.12+ an asyncio consumer gets its scope back when the loop closes the
    generator, not at its next wardex read: a thread or a task it starts before
    reading anything carries none of the dead block. Before 3.12 a Task lends
    no Context, and the window stays open until the consumer's next read, as
    the `conversation()` docstring says: both halves are pinned here."""
    dead = None if sys.version_info >= (3, 12) else "DEAD"
    seen: dict[str, str | None] = {}

    async def main() -> None:
        async def turn():
            with wardex.conversation("chat", id="DEAD"):
                yield 1
                yield 2

        async def child() -> str | None:
            with wardex.span("started-after-hangup"):
                pass
            return _ambient_conversation()

        async for _ in turn():
            break
        gc.collect()
        await asyncio.sleep(0.05)
        seen["thread"] = await asyncio.to_thread(_ambient_conversation)
        seen["task"] = await asyncio.create_task(child())

    spans = _shipped(lambda: asyncio.run(main()))
    assert seen == {"thread": dead, "task": dead}
    assert _conv(_named(spans, "started-after-hangup")) == dead


def test_a_block_closed_from_another_thread_leaves_the_running_owner_alone(monkeypatch):
    """A sync generator whose block was installed by an asyncio task and is
    closed from another thread while that task is still running. The closing
    thread must not enter the owner's Context: if the owner's loop resumes the
    owner meanwhile, the loop cannot run the step and the task is lost. Taking
    the block back is left to the owner's next read.

    The wrapper below is where entering from afar used to happen. If it ever
    runs on the closing thread, it holds the owner's Context there until the
    owner's loop has tried to resume the owner, which makes the failure certain
    rather than a race."""
    errors: list[dict[str, Any]] = []
    seen: dict[str, Any] = {}
    ready = threading.Event()
    attempted = threading.Event()
    state: dict[str, Any] = {}

    def turn():
        with wardex.conversation("chat", id="R"):
            yield 1
            yield 2

    async def owner() -> None:
        state["go"] = asyncio.Event()
        gen = turn()
        next(gen)
        state["gen"] = gen
        ready.set()
        await state["go"].wait()
        seen["owner"] = _ambient_conversation()

    settle_in = _hub._settle_in

    def held_open(*args: Any) -> None:
        loop = state["loop"]

        def resume_owner() -> None:
            state["go"].set()
            loop.call_soon(loop.call_soon, attempted.set)  # after the owner's step

        loop.call_soon_threadsafe(resume_owner)
        attempted.wait(5)
        settle_in(*args)

    monkeypatch.setattr(_hub, "_settle_in", held_open)

    def drive() -> None:
        loop = asyncio.new_event_loop()
        state["loop"] = loop
        loop.set_exception_handler(lambda _loop, ctx: errors.append(ctx))
        try:
            loop.run_until_complete(asyncio.wait_for(owner(), 5))
        finally:
            loop.close()

    def run() -> None:
        # A daemon, so that a lost owner task fails this test instead of hanging the run.
        thread = threading.Thread(target=drive, daemon=True)
        thread.start()
        assert ready.wait(5)
        state["gen"].close()
        state["loop"].call_soon_threadsafe(state["go"].set)
        thread.join(10)
        seen["finished"] = not thread.is_alive()

    _shipped(run)
    assert errors == []
    assert seen == {"owner": None, "finished": True}


def test_a_retired_scope_does_not_keep_its_installer_alive():
    """What a retired scope remembers of the Context that installed it holds
    no task: an installer that abandons a generator and ends is freed, though a
    task it started inside the block lives on with the block's scope."""
    seen: dict[str, Any] = {}

    async def main() -> None:
        stop = asyncio.Event()
        children: list[asyncio.Task[None]] = []
        installers: list[weakref.ref[Any]] = []

        async def child() -> None:
            await stop.wait()

        async def installer() -> None:
            async def turn():
                with wardex.conversation("chat", id="M"):
                    children.append(asyncio.create_task(child()))
                    yield 1
                    yield 2

            async for _ in turn():
                break

        for _ in range(20):
            task = asyncio.create_task(installer())
            installers.append(weakref.ref(task))
            await task
            del task
        for _ in range(20):
            await asyncio.sleep(0)
        gc.collect()
        seen["alive"] = sum(1 for ref in installers if ref() is not None)
        stop.set()
        await asyncio.gather(*children)

    _shipped(lambda: asyncio.run(main()))
    assert seen["alive"] == 0


@pytest.mark.parametrize(
    ("enter", "isolates"),
    [
        pytest.param(lambda: wardex.continue_trace({}), True, id="continue_trace"),
        pytest.param(wardex.isolation_scope, True, id="isolation_scope"),
        pytest.param(wardex.new_scope, False, id="new_scope"),
    ],
)
def test_a_scope_entered_in_an_abandoned_generator_is_taken_back_too(enter, isolates):
    """The scope entries end the same way a conversation block does: the
    consumer that abandoned the generator reads its own scope again — its own
    isolation scope too, where the entry forked one, so a tag the request set
    stays the request's — and nothing raises out of the loop's cleanup task."""
    raised: list[object] = []

    async def main() -> None:
        asyncio.get_running_loop().set_exception_handler(lambda _loop, ctx: raised.append(ctx))

        async def turn():
            with wardex.conversation("chat", id="X"), enter():
                wardex.set_tag("request", "abandoned")
                yield 1
                yield 2

        async for _ in turn():
            break
        gc.collect()
        await asyncio.sleep(0.05)
        assert _ambient_conversation() is None
        assert ("request" in _hub.get_isolation_scope().tags) is not isolates
        with wardex.span("next-request"):
            pass

    spans = _shipped(lambda: asyncio.run(main()))
    assert _conv(_named(spans, "next-request")) is None
    assert _named(spans, "next-request").parent_span_id is None
    assert raised == []


def test_a_consumer_that_abandons_turn_after_turn_holds_no_chain_of_them():
    """A busy consumer — a WebSocket session — abandons each turn's stream and
    starts the next before the loop has closed the last. Each retired scope
    points past the ones its consumer already retired, so the consumer holds a
    link or two however many turns it has had, and the turns' scopes go."""
    turns = 300
    seen: dict[str, Any] = {}

    async def main() -> None:
        forks: list[weakref.ref[Any]] = []

        async def model_stream():
            for i in range(3):
                await asyncio.sleep(0)
                yield i

        async def pipeline(session: str):
            with wardex.conversation("chat", id=session):
                forks.append(weakref.ref(_hub._current_scope.get()))
                async for chunk in model_stream():
                    yield chunk

        for t in range(turns):
            async for _ in pipeline(f"S{t}"):
                break
        for _ in range(50):
            await asyncio.sleep(0)
        gc.collect()
        links, scope = 0, _hub._current_scope.get()
        while scope is not None and scope._retired_to is not None:
            links, scope = links + 1, scope._retired_to
        seen["links"] = links
        seen["alive"] = sum(1 for ref in forks if ref() is not None)
        seen["ambient"] = _ambient_conversation()

    _shipped(lambda: asyncio.run(main()))
    assert seen["links"] <= 3
    assert seen["alive"] <= 2
    assert seen["ambient"] is None


# --------------------------------------------------------------------------
# an openai-agents trace across a block boundary
# --------------------------------------------------------------------------

_AGENTS = {"adapters": AdaptersConfig(enabled=(AdapterName.OPENAI_AGENTS,))}


@pytest.mark.usefixtures("agents_env")
def test_a_trace_started_in_a_conversation_and_finished_outside_does_not_bring_it_back(llm):  # noqa: F811
    host, port = llm
    seen: dict[str, str | None] = {}

    def run() -> None:
        with wardex.conversation("chat", id="Y"):
            t = agents_trace("wf")
            t.start(mark_as_current=True)
        t.finish(reset_current=True)
        seen["ambient"] = _ambient_conversation()
        with wardex.span("after"):
            pass
        _post(host, port, "after-trace")

    spans = _shipped(run, **_AGENTS)
    assert seen["ambient"] is None
    assert _conv(_named(spans, "after")) is None
    assert _conv(_call(spans, "after-trace")) is None


@pytest.mark.usefixtures("agents_env")
def test_a_trace_finished_inside_a_conversation_leaves_the_block_its_own(llm):  # noqa: F811
    """The mirror image: the trace is pinned first and finished inside a block.
    The rest of the block keeps its id and its span as the parent, and the
    finished trace's own group id does not come back after the block."""
    host, port = llm

    def run() -> None:
        t = agents_trace("wf", group_id="GRP")
        t.start(mark_as_current=True)
        with wardex.conversation("chat", id="Z"):
            t.finish(reset_current=True)
            with wardex.span("rest-of-block"):
                pass
            _post(host, port, "in-block")
        with wardex.span("after"):
            pass
        _post(host, port, "after-block")

    spans = _shipped(run, **_AGENTS)
    rest = _named(spans, "rest-of-block")
    assert _conv(rest) == "Z"
    assert rest.parent_span_id == _named(spans, "chat").context.span_id
    assert _conv(_call(spans, "in-block")) == "Z"
    assert _conv(_named(spans, "after")) is None
    assert _conv(_call(spans, "after-block")) is None
