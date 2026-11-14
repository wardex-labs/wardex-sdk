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

import dataclasses
import functools
import inspect
import json
import types
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar, overload

from . import _hub, _values
from ._assembly import degraded_run, guard, report_once
from ._enums import OperationName, SpanKind, ToolExecutionType
from ._source_paths import _call_site_file
from ._tracing import _NULL_PARENTAGE, Span, _debug_enabled, _span, _WithOnly
from ._types import AgentAttributes, CallSite, ToolAttributes
from .context._contextvar import fork_scope
from .context._current_span import _CURRENT_SPAN

P = ParamSpec("P")
R = TypeVar("R")

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
    """The function's qualified name, `Researcher.run` rather than `run`.

    Qualified so two classes' `run` methods are two names. Everything up to a
    function's last `<locals>.` is dropped: the enclosing function of a closure
    or a factory is where it was defined, not what it is, so
    `make_app.<locals>.Researcher.run` reads as `Researcher.run`.
    """
    for candidate in (
        getattr(fn, "__qualname__", None),
        getattr(code, "__qualname__", None),
        getattr(fn, "__name__", None),
        getattr(code, "__name__", None),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate.rsplit("<locals>.", 1)[-1]
    return type(fn).__name__


def _identity_of(fn: object, code: Any) -> str | None:
    """The function's own name, `run` rather than `Researcher.run`.

    What the identity keys (`gen_ai.tool.name`, ...) carry: the name a model
    calls the tool by and an adapter reports for it, so the same tool is one
    identity whichever way it was traced. The span name stays qualified.
    """
    for candidate in (getattr(fn, "__name__", None), getattr(code, "__name__", None)):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


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
            function=_name_of(code, code),
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
    open across a `yield`) is what its next step resumes under, as the parent of
    new work and as `current_span()`. Only wardex's own variables move; the
    host's context variables are left as they are.

    A span wardex failed to open has no scope to install. Its steps run with the
    degraded-run flag up instead, which is what `_begin` does for a whole block.
    """

    __slots__ = ("_current", "_degraded", "_inner", "_outer", "_scope", "_span")

    def __init__(self, scope: Any, span: Span, *, degraded: bool) -> None:
        self._scope = scope
        self._span = span
        self._degraded = degraded
        self._outer: Any = None
        self._current: Any = None
        self._inner: Span | None = None  # a body's span still open after its last step

    def __enter__(self) -> None:
        self._span._outer = _CURRENT_SPAN.get()
        inner = self._inner
        # A closed inner span is fine here: `current_span()` steps past it.
        self._current = _CURRENT_SPAN.set(self._span if inner is None else inner)
        if self._degraded:
            flag = degraded_run()
            flag.__enter__()
            self._outer = flag
        elif self._scope is not None:
            self._outer = _hub._current_scope.get()
            _hub._current_scope.set(self._scope)

    def __exit__(self, *exc: object) -> None:
        current, self._current = self._current, None
        left = _CURRENT_SPAN.get()
        self._inner = None if left is self._span else left
        _CURRENT_SPAN.reset(current)  # one step runs in one Context
        if self._degraded:
            flag, self._outer = self._outer, None
            flag.__exit__(None, None, None)
        elif self._scope is not None:
            self._scope = _hub._current_scope.get()
            _hub._current_scope.set(self._outer)
            self._outer = None


def _steps_for(span: Span) -> _Steps:
    if span._draft._parentage is _NULL_PARENTAGE:
        return _Steps(None, span, degraded=True)
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
    return _Steps(scope, span, degraded=False)


def _decorate(
    fn: Callable[..., Any] | None,
    *,
    api: str,
    name: str | None,
    op: OperationName,
    agent: AgentAttributes | None = None,
    tool: ToolAttributes | None = None,
    capture_io: bool = False,
) -> Any:
    """The one body behind the four decorators.

    `fn` is non-None exactly when the decorator was applied BARE (`@wardex.tool`
    over the function itself); with keywords, `fn` is None and the returned
    `deco` is what wraps. `name=None` resolves at decoration time to the
    attributes' name, else the function's name, so the two forms name spans
    identically.

    The span is shaped like the one an adapter opens for the same operation:
    named `"{operation} {subject}"` (`invoke_agent researcher`), and carrying
    the identity semconv key: `gen_ai.agent.name` from an `AgentAttributes`,
    `gen_ai.tool.name` from a `ToolAttributes`, `gen_ai.workflow.name`, or
    `wardex.step.name`. Without a given name, the span name is the qualified
    one (`Researcher.run`, so two classes' methods differ), and so is the
    identity key, except a tool's: `gen_ai.tool.name` is the function's own
    (`search`), the name a model calls it by and an adapter reports. An
    agent block built here states no `wardex.agent.type`: running a function
    shows neither that it is the primary agent nor whose sub-agent it is.
    """
    if name is not None:
        _values.text(f"wardex.{api}() name", name)
    _values.block(f"wardex.{api}() attributes", AgentAttributes, agent)
    _values.block(f"wardex.{api}() attributes", ToolAttributes, tool)
    given = agent if agent is not None else tool
    if given is not None and name is not None and given.name != name:
        raise ValueError(
            f"wardex.{api}(name={name!r}) disagrees with attributes.name={given.name!r}: "
            "the span would name one and export the other. Give one of them, or the "
            "same name in both."
        )

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
        chosen = name or (given.name if given is not None else None)
        span_name = f"{op.value} {chosen or _name_of(fn, code)}"
        own = _identity_of(fn, code) if api == "tool" else None
        identity = chosen or own or _name_of(fn, code)
        agent_block = agent
        tool_block = tool
        if api == "agent" and agent_block is None:
            agent_block = AgentAttributes(name=identity, agent_type=None)
        if api == "tool":
            # The decorator ran the body in this process, which is the one fact
            # about the tool's execution it observed.
            if tool_block is None:
                tool_block = ToolAttributes(
                    name=identity, execution_type=ToolExecutionType.IN_PROCESS
                )
            elif tool_block.execution_type is ToolExecutionType.UNKNOWN:
                tool_block = dataclasses.replace(
                    tool_block, execution_type=ToolExecutionType.IN_PROCESS
                )
        workflow_name = identity if api == "workflow" else None
        step_name = identity if api == "step" else None
        cs = _call_site(code)
        signature = _signature(fn) if capture_io else None

        def opened(*, stepped: bool = False) -> _WithOnly:
            return _WithOnly(
                "span",
                _span(span_name, op, SpanKind.INTERNAL, agent_block, tool_block, stepped=stepped),
            )

        def begin(s: Span, args: Any, kwargs: Any) -> None:
            draft = s._draft
            draft.set_call_site(cs)
            draft.set_workflow_name(workflow_name)
            if step_name is not None:
                draft.set_extra("wardex.step.name", step_name)
            if capture_io:
                with guard("tracing.decorator_capture_input", debug=_debug_enabled()):
                    limit = _capture_limit()
                    if limit is not None:
                        draft._input_data = _arguments_json(signature, args, kwargs, limit)

        def end(s: Span, result: Any) -> Any:
            if capture_io:
                with guard("tracing.decorator_capture_output", debug=_debug_enabled()):
                    limit = _capture_limit()
                    if limit is not None:
                        s._draft._output_data = _result_bytes(result, limit)
            return result

        if shape == _ASYNC_GENERATOR:

            @functools.wraps(fn)
            async def agen_wrapper(*args: Any, **kwargs: Any) -> Any:
                with opened(stepped=True) as s:
                    begin(s, args, kwargs)
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
                    begin(s, args, kwargs)
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
                    begin(s, args, kwargs)
                    return end(s, await fn(*args, **kwargs))

            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with opened() as s:
                begin(s, args, kwargs)
                return end(s, fn(*args, **kwargs))

        return wrapper

    if fn is not None:
        return deco(fn)
    return deco


# -- capture_io: what the decorated function was called with and returned ----

#: The longest `repr()` kept for an argument JSON cannot hold. A repr is the
#: object's own code, and a host opting in to argument capture is not opting in
#: to one object's megabyte repr per call.
_REPR_MAX_CHARS = 1024


def _signature(fn: Any) -> inspect.Signature | None:
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):  # a builtin with no signature: arguments by position
        return None


def _repr_of(value: object) -> str:
    """`repr(value)`, cut; never raises, because a host's `__repr__` may."""
    try:
        text = repr(value)
    except Exception:
        return f"<unprintable {type(value).__name__}>"
    if len(text) > _REPR_MAX_CHARS:
        return text[:_REPR_MAX_CHARS] + "...<truncated>"
    return text


_ENCODER = json.JSONEncoder(ensure_ascii=False, default=_repr_of)

#: What JSON encoding raises for a value it cannot write: a key that is not a
#: str or number, a reference cycle, nesting past the recursion limit.
_UNENCODABLE = (TypeError, ValueError, RecursionError)


#: The last client's limits and the cap they resolve to: resolving reads the
#: core's table, which is not worth repeating on every captured call.
_resolved_cap: tuple[object, int] | None = None


def _capture_limit() -> int | None:
    """How many characters of a captured value are worth encoding, or None to skip.

    The export cap, `LimitsConfig.max_otlp_attribute_bytes`: past it the
    exporter cuts the value anyway, so encoding more would cost the host's
    thread and the span buffer for bytes that never ship. None before
    `init()`, where the span records nothing.
    """
    global _resolved_cap
    client = _hub.get_client()
    if client is None:
        return None
    limits = client.config.limits
    cached = _resolved_cap
    if cached is None or cached[0] is not limits:
        cached = _resolved_cap = (limits, limits.resolved()["max_otlp_attribute_bytes"])
    return cached[1]


def _bounded_json(value: object, limit: int) -> bytes:
    """`value`'s JSON text, encoded only until it is longer than `limit` characters.

    A value that does not fit is cut one character past `limit`, so the
    exporter still sees it over the cap, cuts it there and marks the span
    `otlp_attribute_truncated` as it does any payload it shortens. The
    incremental encoder is what stops early: a ten-million-item list costs
    what its first `limit` characters cost.
    """
    pieces: list[str] = []
    size = 0
    for piece in _ENCODER.iterencode(value):
        pieces.append(piece)
        size += len(piece)
        if size > limit:
            break
    return "".join(pieces)[: limit + 1].encode("utf-8", "backslashreplace")


def _encodable(value: object, limit: int) -> object:
    """`value` if JSON can write it, else its `repr()`."""
    try:
        _bounded_json(value, limit)
    except _UNENCODABLE:
        return _repr_of(value)
    return value


def _arguments_json(
    signature: inspect.Signature | None, args: Any, kwargs: Any, limit: int
) -> bytes:
    """The call's arguments by parameter name, as JSON; `repr()` for what JSON cannot hold.

    A method's `self` or `cls` is left out: it is the object, not an input.
    """
    arguments: dict[str, Any] | None = None
    if signature is not None:
        try:
            arguments = dict(signature.bind_partial(*args, **kwargs).arguments)
        except TypeError:  # the call itself is wrong, and the body will say so
            arguments = None
        else:
            first = next(iter(signature.parameters), None)
            if first in ("self", "cls"):
                arguments.pop(first, None)
    if arguments is None:
        arguments = {"args": list(args), **kwargs}
    try:
        return _bounded_json(arguments, limit)
    except _UNENCODABLE:
        return _bounded_json({key: _encodable(v, limit) for key, v in arguments.items()}, limit)


def _result_bytes(result: object, limit: int) -> bytes:
    if isinstance(result, str):
        return _values.raw("capture_io", result[: limit + 1])
    if isinstance(result, (bytes, bytearray)):
        return bytes(result[: limit + 1])
    if isinstance(result, memoryview):
        return bytes(result)[: limit + 1]
    try:
        return _bounded_json(result, limit)
    except _UNENCODABLE:
        return _bounded_json(_repr_of(result), limit)


# -- the four decorators ------------------------------------------------------
#
# Each has two overloads, so a type checker sees the wrapped function's own
# signature through both `@wardex.agent` and `@wardex.agent(name=...)`.


@overload
def workflow(fn: Callable[P, R]) -> Callable[P, R]: ...
@overload
def workflow(
    fn: None = None, *, name: str | None = None, capture_io: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def workflow(
    fn: Callable[..., Any] | None = None, *, name: str | None = None, capture_io: bool = False
) -> Any:
    """A whole run: `invoke_workflow {name}`, with `gen_ai.workflow.name`.
    `capture_io=True` records the arguments and return value, as on `@agent`."""
    return _decorate(
        fn, api="workflow", name=name, op=OperationName.INVOKE_WORKFLOW, capture_io=capture_io
    )


@overload
def agent(fn: Callable[P, R]) -> Callable[P, R]: ...
@overload
def agent(
    fn: None = None,
    *,
    name: str | None = None,
    attributes: AgentAttributes | None = None,
    capture_io: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def agent(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    attributes: AgentAttributes | None = None,
    capture_io: bool = False,
) -> Any:
    """An agent's invocation: `invoke_agent {name}`, with `gen_ai.agent.name`.

    `attributes` adds the rest of the agent's identity; its `name` must match
    `name=` when both are given. `capture_io=True` records the call's arguments
    and return value as the span's input and output (JSON, `repr()` for what
    JSON cannot hold), masked and capped at export like any payload.
    """
    return _decorate(
        fn,
        api="agent",
        name=name,
        op=OperationName.INVOKE_AGENT,
        agent=attributes,
        capture_io=capture_io,
    )


@overload
def tool(fn: Callable[P, R]) -> Callable[P, R]: ...
@overload
def tool(
    fn: None = None,
    *,
    name: str | None = None,
    attributes: ToolAttributes | None = None,
    capture_io: bool = False,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    attributes: ToolAttributes | None = None,
    capture_io: bool = False,
) -> Any:
    """A tool call: `execute_tool {name}`, with `gen_ai.tool.name`.

    The execution type is `in_process` (the decorator ran the body) unless
    `attributes` names another. `capture_io=True` records the arguments and
    result as `gen_ai.tool.call.arguments` and `gen_ai.tool.call.result`.
    """
    return _decorate(
        fn,
        api="tool",
        name=name,
        op=OperationName.EXECUTE_TOOL,
        tool=attributes,
        capture_io=capture_io,
    )


@overload
def step(fn: Callable[P, R]) -> Callable[P, R]: ...
@overload
def step(
    fn: None = None, *, name: str | None = None, capture_io: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
def step(
    fn: Callable[..., Any] | None = None, *, name: str | None = None, capture_io: bool = False
) -> Any:
    """One step of a larger run — a graph node, a pipeline stage, a phase.

    `execute_step {name}`, with `wardex.step.name`, as an adapter names a graph
    node. "step" also stays clear of the "task" asyncio, Celery and LangGraph
    each already mean something else by. `capture_io=True` as on `@agent`.
    """
    return _decorate(
        fn, api="step", name=name, op=OperationName.EXECUTE_STEP, capture_io=capture_io
    )
