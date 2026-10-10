"""Every function shape a decorator can meet: its span covers the whole execution,
or the decorator refuses the shape where it is applied and says what to do.

The two failures this pins shut. A generator or async generator was wrapped like
a plain function, so its span closed when the call merely BUILT the generator —
about 0.03 ms, before any of the body ran — and every span the body opened
became the root of a trace of its own, with nothing saying so. And anything
that was not a plain function (a `staticmethod`, a `functools.partial`, a
callable object, the span name passed positionally) died at import with an
`AttributeError` about wardex's own internals.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import threading
import time

import pytest

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._decorators import agent, step, tool, workflow
from wardex_sdk._enums import StatusCode
from wardex_sdk._tracing import span
from wardex_sdk._types import Envelope
from wardex_sdk.transport._base import Transport

DECORATORS = {"workflow": workflow, "agent": agent, "tool": tool, "step": step}


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)


@pytest.fixture
def recording():
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), t))
    yield t
    _hub.reset_for_test()


def _spans(t: _Recording):
    _hub.get_client().flush()
    return [sp for env in t.envelopes for sp in env.spans]


def _by_name(t: _Recording):
    return {sp.name: sp for sp in _spans(t)}


def _parent(sp):
    return None if sp.parent_span_id is None else sp.parent_span_id.value


def _ambient_span_id() -> str | None:
    header = wardex_sdk.get_traceparent()
    return None if header is None else header.split("-")[2]


# ==========================================================================
# generators: the span stays open from the first step to the last
# ==========================================================================


def test_a_decorated_generator_covers_its_whole_iteration(recording):
    @workflow(name="stream")
    def stream():
        for i in range(3):
            time.sleep(0.01)
            with span(f"inside-{i}"):
                pass
            yield i

    items = []
    for item in stream():
        with span(f"consumer-{item}"):
            pass
        items.append(item)

    assert items == [0, 1, 2]
    spans = _by_name(recording)
    root = spans["stream"]
    assert root.end_time_ns - root.start_time_ns >= 30_000_000
    for i in range(3):
        inside = spans[f"inside-{i}"]
        assert _parent(inside) == root.context.span_id.value
        assert inside.context.trace_id.value == root.context.trace_id.value
        # The generator's span is current only while its own body runs: the
        # consumer's work between two items is not inside it.
        assert _parent(spans[f"consumer-{i}"]) != root.context.span_id.value


def test_a_decorated_async_generator_covers_its_whole_iteration(recording):
    @agent(name="astream")
    async def astream():
        for i in range(3):
            await asyncio.sleep(0.01)
            with span(f"inside-{i}"):
                pass
            yield i

    async def consume():
        out = []
        async for item in astream():
            with span(f"consumer-{item}"):
                pass
            out.append(item)
        return out

    assert asyncio.run(consume()) == [0, 1, 2]
    spans = _by_name(recording)
    root = spans["astream"]
    assert root.end_time_ns - root.start_time_ns >= 30_000_000
    for i in range(3):
        assert _parent(spans[f"inside-{i}"]) == root.context.span_id.value
        assert _parent(spans[f"consumer-{i}"]) != root.context.span_id.value


def test_the_generator_shape_survives_decoration(recording):
    """Frameworks branch on these predicates (a streaming response, a tool
    runner); a wrapper that answered False turned a stream into a call."""

    @tool
    def gen():
        yield 1

    @tool
    async def agen():
        yield 1

    assert inspect.isgeneratorfunction(gen)
    assert inspect.isasyncgenfunction(agen)


def test_a_span_held_open_across_a_yield_parents_the_next_steps_work(recording):
    @workflow(name="outer")
    def stream():
        with span("held"):
            yield 1
            with span("later"):
                pass
            yield 2

    assert list(stream()) == [1, 2]
    spans = _by_name(recording)
    assert _parent(spans["held"]) == spans["outer"].context.span_id.value
    assert _parent(spans["later"]) == spans["held"].context.span_id.value


def test_send_throw_and_the_return_value_reach_the_body_as_with_yield_from(recording):
    seen = []

    @step(name="echo")
    def echo():
        try:
            while True:
                seen.append((yield len(seen)))
        except KeyError:
            yield "handled"
        return "done"

    def drive():
        g = echo()
        assert next(g) == 0
        assert g.send("a") == 1
        assert g.throw(KeyError("k")) == "handled"
        with pytest.raises(StopIteration) as stop:
            next(g)
        return stop.value.value

    assert drive() == "done"
    assert seen == ["a"]
    assert _by_name(recording)["echo"].status is not StatusCode.ERROR


def test_closing_a_generator_early_closes_the_body_and_ends_the_span(recording):
    closed = []

    @tool(name="early")
    def early():
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    g = early()
    assert next(g) == 1
    g.close()
    assert closed == [True]
    spans = _by_name(recording)
    assert spans["early"].status is not StatusCode.ERROR


def test_an_exception_from_a_generator_body_marks_its_span_failed(recording):
    @tool(name="boom")
    def boom():
        yield 1
        raise ValueError("bad item")

    with pytest.raises(ValueError, match="bad item"):
        list(boom())
    sp = _by_name(recording)["boom"]
    assert sp.status is StatusCode.ERROR
    assert sp.error_type == "ValueError"


def test_an_async_generator_closed_early_closes_its_body(recording):
    closed = []

    @tool(name="aearly")
    async def aearly():
        try:
            yield 1
            yield 2
        finally:
            closed.append(True)

    async def main():
        g = aearly()
        assert await g.__anext__() == 1
        await g.aclose()

    asyncio.run(main())
    assert closed == [True]
    assert "aearly" in _by_name(recording)


def test_a_generator_resumed_on_another_thread_stays_in_its_trace(recording):
    @workflow(name="handoff")
    def handoff():
        with span("first"):
            pass
        yield 1
        with span("second"):
            pass
        yield 2

    g = handoff()
    assert next(g) == 1
    rest = []
    worker = threading.Thread(target=lambda: rest.extend(g))
    worker.start()
    worker.join(timeout=10)
    assert rest == [2]
    spans = _by_name(recording)
    root = spans["handoff"].context.span_id.value
    assert _parent(spans["first"]) == root
    assert _parent(spans["second"]) == root


# ==========================================================================
# the shape table: every cell covers the call, or is refused with guidance
# ==========================================================================


def _sync(tag):
    return _ambient_span_id()


async def _async(tag):
    await asyncio.sleep(0)
    return _ambient_span_id()


class _Holder:
    def method(self, tag):
        return _ambient_span_id()


class _CallableObject:
    def __call__(self, tag):
        return _ambient_span_id()


_WRAPPED = {
    "function": (lambda: _sync, "call"),
    "lambda": (lambda: lambda tag: _ambient_span_id(), "call"),
    "bound method": (lambda: _Holder().method, "call"),
    "partial": (lambda: functools.partial(_sync), "call"),
    "partial of an async function": (lambda: functools.partial(_async), "await"),
    "lru_cache": (lambda: functools.lru_cache(maxsize=None)(_sync), "call"),
    "async function": (lambda: _async, "await"),
}


@pytest.mark.parametrize("form", ["bare", "called"])
@pytest.mark.parametrize("api", sorted(DECORATORS))
@pytest.mark.parametrize("shape", sorted(_WRAPPED))
def test_every_supported_shape_runs_inside_its_span(recording, shape, api, form):
    make, how = _WRAPPED[shape]
    deco = DECORATORS[api]
    target = make()
    decorated = deco(target) if form == "bare" else deco(name="named")(target)
    seen = decorated("x") if how == "call" else asyncio.run(decorated("x"))
    (sp,) = _spans(recording)
    assert seen == sp.context.span_id.value.hex()
    assert sp.name == ("named" if form == "called" else _expected_bare_name(target))


def _expected_bare_name(target):
    return getattr(target, "__name__", None) or target.func.__name__


@pytest.mark.parametrize("api", sorted(DECORATORS))
def test_a_builtin_is_wrapped_with_no_call_site(recording, api):
    decorated = DECORATORS[api](len)
    assert decorated([1, 2, 3]) == 3
    (sp,) = _spans(recording)
    assert sp.name == "len"
    assert sp.call_site is None


@pytest.mark.parametrize("api", sorted(DECORATORS))
def test_above_staticmethod_and_classmethod_the_descriptor_is_kept(recording, api):
    deco = DECORATORS[api]

    class Tools:
        @deco
        @staticmethod
        def search(q):
            return ("static", q, _ambient_span_id())

        @deco
        @classmethod
        def build(cls, q):
            return (cls.__name__, q, _ambient_span_id())

    static_result = Tools.search("a")
    class_result = Tools().build("b")
    spans = _by_name(recording)
    assert static_result == ("static", "a", spans["search"].context.span_id.value.hex())
    assert class_result == ("Tools", "b", spans["build"].context.span_id.value.hex())


def test_a_partial_and_a_cache_wrapper_name_the_functions_own_source(recording):
    for decorated in (
        tool(functools.partial(_sync)),
        tool(functools.lru_cache(maxsize=None)(_sync)),
    ):
        decorated("x")
    for sp in _spans(recording):
        assert sp.call_site is not None
        assert sp.call_site.function == "_sync"
        assert sp.call_site.line == _sync.__code__.co_firstlineno


_REFUSED = {
    "a class": (lambda: _Holder, "cannot decorate the class _Holder"),
    "a callable object": (lambda: _CallableObject(), "or decorate the class's `__call__`"),
    "a cache over an async function": (
        lambda: functools.lru_cache(maxsize=None)(_async),
        "directly to the function, below the _lru_cache_wrapper decorator",
    ),
    "the name passed positionally": (lambda: "search", "(name='search')"),
    "a value that cannot be called": (lambda: 42, "got a int object, which cannot be called"),
}


@pytest.mark.parametrize("form", ["bare", "called"])
@pytest.mark.parametrize("api", sorted(DECORATORS))
@pytest.mark.parametrize("shape", sorted(_REFUSED))
def test_an_unsupported_shape_is_refused_where_it_is_decorated(recording, shape, api, form):
    make, how_to_fix = _REFUSED[shape]
    deco = DECORATORS[api]
    with pytest.raises(TypeError) as refused:
        deco(make()) if form == "bare" else deco(name="named")(make())
    message = str(refused.value)
    assert message.startswith(f"wardex.{api}(")
    assert how_to_fix in message
    assert _spans(recording) == []


def test_with_a_decorator_name_positionally_the_block_form_is_suggested(recording):
    with pytest.raises(TypeError, match=r"with wardex\.span\('x'\)"):
        with workflow("x"):
            pass


# ==========================================================================
# the six published context managers: `with` only, never a decorator
# ==========================================================================

_CONTEXT_MANAGERS = {
    "span": lambda: wardex_sdk.span("x"),
    "conversation": lambda: wardex_sdk.conversation("x"),
    "isolation_scope": wardex_sdk.isolation_scope,
    "new_scope": wardex_sdk.new_scope,
    "continue_trace": lambda: wardex_sdk.continue_trace(
        {"traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"}
    ),
    "continue_from_otel": wardex_sdk.continue_from_otel,
}


@pytest.mark.parametrize("name", sorted(_CONTEXT_MANAGERS))
def test_a_published_context_manager_refuses_to_decorate(recording, name):
    make = _CONTEXT_MANAGERS[name]
    with pytest.raises(TypeError) as refused:

        @make()
        async def handler():
            return 1

    message = str(refused.value)
    assert message.startswith(f"wardex.{name}() is a context manager, not a decorator")
    assert "with wardex." in message
    with make():
        pass


def test_continue_trace_still_joins_the_callers_trace_in_a_with_block(recording):
    header = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    with wardex_sdk.continue_trace({"traceparent": header}):
        with span("joined"):
            pass
    sp = _by_name(recording)["joined"]
    assert sp.context.trace_id.value.hex() == "0af7651916cd43dd8448eb211c80319c"
    assert sp.parent_span_id.value.hex() == "b7ad6b7169203331"


# ==========================================================================
# a span wardex failed to open still runs the generator's body
# ==========================================================================


def test_a_generator_whose_span_failed_to_open_still_runs(recording, monkeypatch):
    import wardex_sdk._tracing as tracing

    def broken(*args, **kwargs):
        raise RuntimeError("wardex bug")

    monkeypatch.setattr(tracing, "latch_ambient", broken)

    @workflow(name="degraded")
    def stream():
        yield 1
        yield 2

    assert list(stream()) == [1, 2]
    assert "degraded" not in _by_name(recording)
