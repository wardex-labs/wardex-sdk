"""The four decorators — `@workflow`, `@agent`, `@tool`, `@step` — and the shapes they wrap.

A decorator's span has to cover the decorated function's whole execution, and
what that is depends on the function's shape: the call for a plain function,
the `await` for an `async def`, every step from the first `next()` to the last
for a generator. Each shape gets its own wrapper below. Wrapping only the call
was right for the first and wrong for the other two: a generator's call merely
builds the generator, so its span closed before the body ran and every span the
body opened became the root of a trace of its own. A generator's wrapper is a
generator function itself, which frameworks branch on; the price is the one
every generator wrapper pays: its arguments reach the body at the first
`next()`, so a call with the wrong arguments raises there, not at the call.

Anything that is not a function of one of those shapes is refused HERE, when
the decorator is applied, with a `TypeError` that says what was refused, why,
and what to write instead. That is configuration time — an import, in practice
— which is where a wrong guess about how to attach wardex should fail. Before,
it failed in two worse ways: an `AttributeError` about `__code__` from inside
wardex, or a span that silently covered nothing.
"""

from __future__ import annotations

import functools
import inspect
import types
from collections.abc import Callable
from typing import Any

from . import _hub
from ._assembly import degraded_run, guard, report_once
from ._enums import OperationName, SpanKind
from ._source_paths import _call_site_file
from ._tracing import _NULL_PARENTAGE, Span, _debug_enabled, _span, _WithOnly
from ._types import AgentAttributes, CallSite, ToolAttributes
from .context._contextvar import fork_scope

_FUNCTION = "function"
_ASYNC = "async function"
_GENERATOR = "generator function"
_ASYNC_GENERATOR = "async generator function"


def _shape(obj: object) -> str:
    """What calling `obj` hands back, as the stdlib reads it off the code flags.

    `inspect` looks through bound methods and `functools.partial` on its own.
    """
    if inspect.isasyncgenfunction(obj):
        return _ASYNC_GENERATOR
    if inspect.isgeneratorfunction(obj):
        return _GENERATOR
    if inspect.iscoroutinefunction(obj):
        return _ASYNC
    return _FUNCTION


def _has_code(obj: object) -> bool:
    return isinstance(getattr(obj, "__code__", None), types.CodeType)


def _unwrapped(obj: object) -> Any:
    """The function a wrapper OBJECT says it wraps (`__wrapped__`), or None."""
    try:
        base = inspect.unwrap(obj)
    except Exception:  # a `__wrapped__` cycle, or a host `__getattr__` raising
        return None
    return base if base is not obj and _has_code(base) else None


def _resolve(api: str, fn: object) -> tuple[str, Any, bool]:
    """`(shape, code, is_wrapper_object)` for a callable the decorator can wrap;
    a `TypeError` otherwise.

    `code` is the function whose source the span's call site names: `fn` itself
    for a plain function (a `functools.wraps` wrapper included — its own code
    is what runs), what a bound method, a partial or a caching wrapper object
    stands for, and None for a builtin, which has no source.
    """
    kind = type(fn).__name__
    if isinstance(fn, str):
        raise TypeError(
            f"wardex.{api}({fn!r}): the span name is a keyword argument, so {fn!r} "
            f"was taken for the function to wrap. Write @wardex.{api}(name={fn!r}); "
            f"to trace a block of code rather than a function, use "
            f"`with wardex.span({fn!r}):`."
        )
    if isinstance(fn, type):
        raise TypeError(
            f"wardex.{api}() cannot decorate the class {fn.__name__}: it would "
            "replace the class with a function and trace only the construction of "
            "an instance. Decorate the method that does the work instead."
        )
    if not callable(fn):
        raise TypeError(
            f"wardex.{api}() decorates a function, and got a {kind} object, which "
            "cannot be called. Apply it to a function, method, generator or async "
            f"function: `@wardex.{api}` or `@wardex.{api}(name=...)`."
        )
    inner: Any = fn
    while True:
        if inspect.ismethod(inner):
            inner = inner.__func__
        elif isinstance(inner, functools.partial):
            inner = inner.func
        else:
            break
    if _has_code(inner):
        return _shape(fn), inner, False
    if inspect.isbuiltin(inner):
        return _FUNCTION, None, False
    base = _unwrapped(inner)
    if base is None:
        raise TypeError(
            f"wardex.{api}() cannot decorate a {kind} object. Apply @wardex.{api} to "
            "the plain function first — closest to its `def`, below any decorator "
            "that turns it into an object — or decorate the class's `__call__` "
            "method: the decorators wrap functions, and putting a function in place "
            "of an object hides the object's own type and attributes from whatever "
            "uses it (a framework's tool registry, say)."
        )
    if _shape(base) != _shape(fn):
        raise TypeError(
            f"wardex.{api}() cannot decorate this {kind} object. Apply @wardex.{api} "
            f"directly to the function, below the {kind} decorator: the object wraps "
            f"the {_shape(base)} {base.__name__!r} but is not itself one, so a span "
            "around a call to it would end before the body runs."
        )
    return _shape(fn), base, inner is fn


def _keep_api(wrapper: Any, obj: object) -> None:
    """Keep a wrapper object's own public API reachable on the function replacing it.

    `functools.wraps` copies a function's attributes, and a caching wrapper's
    are methods of its type — so `@wardex.tool` over `@functools.lru_cache`
    used to hide `cache_info()` and `cache_clear()`. Each public attribute the
    function does not already have is set on it, bound to the object.
    """
    with guard("tracing.decorator_api", debug=_debug_enabled()):
        for attr in dir(obj):
            if not attr.startswith("_") and not hasattr(wrapper, attr):
                setattr(wrapper, attr, getattr(obj, attr))


def _name_of(fn: object, code: Any) -> str:
    for candidate in (getattr(fn, "__name__", None), getattr(code, "__name__", None)):
        if isinstance(candidate, str) and candidate:
            return candidate
    return type(fn).__name__


def _call_site(code: Any) -> CallSite | None:
    """Where the decorated function is defined. Placed once, at decoration time.

    The call path pays nothing for it. A builtin has no source, and a failure
    to place one costs the call site, never the decoration: the span is whole
    without it.
    """
    if code is None:
        return None
    site = None
    with guard("tracing.call_site", debug=_debug_enabled()):
        source = code.__code__
        module = getattr(code, "__module__", None)
        site = CallSite(
            file=_call_site_file(source.co_filename, module),
            line=source.co_firstlineno,
            function=code.__name__,
            module=module,
        )
    return site


class _Steps:
    """What a decorated generator's body runs under: there for one step, gone between.

    A generator has no context of its own: each `next()` runs its body in the
    CONSUMER's context. Installed once for the generator's whole life, the span
    would stay the active parent in the consumer between items — so the
    consumer's own work would be parented under a span it is not inside, and
    taking the installation down from a step resumed on another thread or task
    would raise. So the span's scope goes in for exactly one step and comes out
    after it, and whatever the body left installed (a `with wardex.span()` still
    open across a `yield`) is what its next step resumes under. Only wardex's own
    scope variable moves; the host's context variables are left as they are.

    A span wardex failed to open has no scope to install. Its steps run with the
    degraded-run flag up instead, which is what `_begin` does for a whole block.
    """

    __slots__ = ("_degraded", "_outer", "_scope")

    def __init__(self, scope: Any, *, degraded: bool) -> None:
        self._scope = scope
        self._degraded = degraded
        self._outer: Any = None

    def __enter__(self) -> None:
        if self._degraded:
            flag = degraded_run()
            flag.__enter__()
            self._outer = flag
        elif self._scope is not None:
            self._outer = _hub._current_scope.get()
            _hub._current_scope.set(self._scope)

    def __exit__(self, *exc: object) -> None:
        if self._degraded:
            flag, self._outer = self._outer, None
            flag.__exit__(None, None, None)
        elif self._scope is not None:
            self._scope = _hub._current_scope.get()
            _hub._current_scope.set(self._outer)
            self._outer = None


def _steps_for(span: Span) -> _Steps:
    if span._draft._parentage is _NULL_PARENTAGE:
        return _Steps(None, degraded=True)
    scope = None
    with guard("tracing.generator_scope", debug=_debug_enabled()):
        scope = fork_scope(span.context)
    if scope is None:
        report_once(
            "wardex: internal error installing a decorated generator's span as the "
            "active parent; work inside it will be attached one level too high "
            "(re-run with debug=True for the traceback)",
            key="wardex.span.generator_scope",
        )
    return _Steps(scope, degraded=False)


def _decorate(
    fn: Callable[..., Any] | None,
    *,
    api: str,
    name: str | None,
    op: OperationName | None,
    kind: SpanKind = SpanKind.INTERNAL,
    agent: AgentAttributes | None = None,
    tool: ToolAttributes | None = None,
    names_workflow: bool = False,
) -> Callable[..., Any]:
    """The one body behind the four decorators.

    `fn` is non-None exactly when the decorator was applied BARE (`@wardex.tool`
    over the function itself); with keywords, `fn` is None and the returned
    `deco` is what wraps. `name=None` resolves to the function's name at
    decoration time, so the two forms name spans identically.
    """

    def deco(fn: Any) -> Any:
        if isinstance(fn, (staticmethod, classmethod)):
            # Above `@staticmethod`/`@classmethod`: wrap the function inside and
            # put the same descriptor back around it, so the class still binds it.
            rewrap = staticmethod if isinstance(fn, staticmethod) else classmethod
            return rewrap(deco(fn.__func__))
        shape, code, is_wrapper_object = _resolve(api, fn)
        wrapper = shaped(fn, shape, code)
        if is_wrapper_object:
            _keep_api(wrapper, fn)
        return wrapper

    def shaped(fn: Any, shape: str, code: Any) -> Any:
        span_name = name if name is not None else _name_of(fn, code)
        workflow_name = span_name if names_workflow else None
        cs = _call_site(code)

        def opened(*, stepped: bool = False) -> _WithOnly:
            return _WithOnly("span", _span(span_name, op, kind, agent, tool, stepped=stepped))

        if shape == _ASYNC_GENERATOR:

            @functools.wraps(fn)
            async def agen_wrapper(*args: Any, **kwargs: Any) -> Any:
                with opened(stepped=True) as s:
                    s.call_site, s.workflow_name = cs, workflow_name
                    steps = _steps_for(s)
                    with steps:
                        agen = fn(*args, **kwargs)
                    send, value = agen.asend, None
                    while True:
                        with steps:
                            try:
                                item = await send(value)
                            except StopAsyncIteration:
                                return
                        try:
                            value = yield item
                            send = agen.asend
                        except GeneratorExit:
                            with steps:
                                await agen.aclose()
                            raise
                        except BaseException as exc:
                            # Forwarded into the body, exactly as `yield from` would.
                            send, value = agen.athrow, exc

            return agen_wrapper

        if shape == _GENERATOR:

            @functools.wraps(fn)
            def gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                with opened(stepped=True) as s:
                    s.call_site, s.workflow_name = cs, workflow_name
                    steps = _steps_for(s)
                    with steps:
                        gen = fn(*args, **kwargs)
                    send, value = gen.send, None
                    while True:
                        with steps:
                            try:
                                item = send(value)
                            except StopIteration as stop:
                                return stop.value
                        try:
                            value = yield item
                            send = gen.send
                        except GeneratorExit:
                            with steps:
                                gen.close()
                            raise
                        except BaseException as exc:
                            # Forwarded into the body, exactly as `yield from` would.
                            send, value = gen.throw, exc

            return gen_wrapper

        if shape == _ASYNC:

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                with opened() as s:
                    s.call_site, s.workflow_name = cs, workflow_name
                    return await fn(*args, **kwargs)

            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with opened() as s:
                s.call_site, s.workflow_name = cs, workflow_name
                return fn(*args, **kwargs)

        return wrapper

    if fn is not None:
        return deco(fn)
    return deco


def workflow(
    fn: Callable[..., Any] | None = None, *, name: str | None = None
) -> Callable[..., Any]:
    return _decorate(
        fn, api="workflow", name=name, op=OperationName.INVOKE_WORKFLOW, names_workflow=True
    )


def agent(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    attributes: AgentAttributes | None = None,
) -> Callable[..., Any]:
    return _decorate(fn, api="agent", name=name, op=OperationName.INVOKE_AGENT, agent=attributes)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    attributes: ToolAttributes | None = None,
) -> Callable[..., Any]:
    return _decorate(fn, api="tool", name=name, op=OperationName.EXECUTE_TOOL, tool=attributes)


def step(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Callable[..., Any]:
    """One step of a larger run — a graph node, a pipeline stage, a phase.

    Maps to `OperationName.EXECUTE_STEP`, so its spans appear on
    operation-keyed dashboards like every other decorator's (the old `task()`
    mapped no operation, and its spans vanished from any view keyed on
    `gen_ai.operation.name`). "step" also stays clear of the "task" asyncio,
    Celery and LangGraph each already mean something else by.
    """
    return _decorate(fn, api="step", name=name, op=OperationName.EXECUTE_STEP)
