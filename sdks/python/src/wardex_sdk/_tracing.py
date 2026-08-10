from __future__ import annotations

import functools
import inspect
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from . import _hub
from ._assembly import (
    EMPTY_AMBIENT,
    Evidence,
    ParentSource,
    SpanDraft,
    degraded_run,
    guard,
    latch_ambient,
    parent_is_closed_unit,
    report_once,
    resolve_observed,
    resolve_parentage,
)
from ._enums import CaptureSource, OperationName, SpanKind, StatusCode
from ._types import (
    AgentAttributes,
    CallSite,
    ConversationContext,
    GenAIAttributes,
    InternalSpan,
    ToolAttributes,
)
from .context._contextvar import fork_active_span

if TYPE_CHECKING:
    # Type-only: these three name the span machinery, which stays out of this
    # module's runtime namespace — the `noqa: ANN201` escape these properties
    # used to wear existed to dodge exactly this import, and a TYPE_CHECKING
    # import states the same fact without leaving the properties untyped.
    from ._types import CorrelationInfo, SpanContext, SpanId


class Span:
    """The object `wardex.span()` yields — a published surface over a `SpanDraft`.

    Every attribute below is a view onto the draft, so the manual path and the
    observation paths build the same object through the same validation. That is
    §6.4's point about the two `execute_tool` shapes: this path used to ship
    spans with no `capture_sources`, no `capture_integrity` and no
    `correlation`, purely because it was a second constructor.

    The draft is in MANUAL mode, which changes exactly two things and no more:
    the name is the host's (`wardex.span("anything")` is a published API), and
    `set_attribute` takes any key (so is it). `operation` stays a LABEL — the
    decorators make the typed block optional, so wardex has nothing to check a
    `SpanIntent` against, and requiring one would be a public API change.
    """

    # NO `__slots__`. This object is yielded into a `with` block the host owns,
    # and before the draft backed it, it was a plain object — so stashing
    # `s.my_tag = 1` on it worked. Under `__slots__` that raises `AttributeError`
    # in the middle of the host's own code, outside every `guard()`: wardex
    # breaking the host, which is precisely what I6 forbids.

    def __init__(self, draft: SpanDraft) -> None:
        self._draft = draft
        self.end_time_ns = 0

    @property
    def context(self) -> SpanContext:
        return self._draft.context

    @property
    def start_time_ns(self) -> int:
        return self._draft._start_ns

    @start_time_ns.setter
    def start_time_ns(self, value: int) -> None:
        self._draft._start_ns = value

    # Views onto the draft of what this object exposed as plain attributes
    # before the draft backed it. They stay WRITABLE: every one of them was a
    # settable attribute on a published object, and turning a working assignment
    # into an `AttributeError` inside a user's `with` block is a break wardex
    # gains nothing from — the draft still decides what a legal span is, in
    # `_finish()`, which is where the check belongs.
    #
    # `correlation` and `parent_span_id` are the two exceptions and they are
    # read-only on purpose: parentage is decided by `_assembly/_parentage.py`
    # (I1) and a span whose parent the host overwrote after the fact would
    # contradict the trace it was already forked into.

    @property
    def name(self) -> str:
        return self._draft.name

    @name.setter
    def name(self, value: str) -> None:
        self._draft.rename(value)

    @property
    def kind(self) -> SpanKind:
        return self._draft._kind or SpanKind.INTERNAL

    @kind.setter
    def kind(self, value: SpanKind) -> None:
        self._draft._kind = value

    @property
    def status(self) -> StatusCode:
        return self._draft._status

    @status.setter
    def status(self, value: StatusCode) -> None:
        self._draft.set_status(value, self._draft._status_message)

    @property
    def status_message(self) -> str:
        return self._draft._status_message

    @status_message.setter
    def status_message(self, value: str) -> None:
        self._draft._status_message = value

    @property
    def conversation(self) -> ConversationContext | None:
        return self._draft._conversation

    @conversation.setter
    def conversation(self, value: ConversationContext | None) -> None:
        self._draft.set_conversation(value)

    @property
    def correlation(self) -> CorrelationInfo | None:
        return self._draft._correlation

    @property
    def parent_span_id(self) -> SpanId | None:
        return self._draft._parentage.parent_span_id

    def set_status(self, code: StatusCode, message: str = "") -> None:
        self._draft.set_status(code, message)

    def set_error(self, error_type: str, message: str = "") -> None:
        """Name the failure. Pairs with `set_status(StatusCode.ERROR)`.

        Without this the published surface could express the state that makes a
        span illegal (`set_status(ERROR)`) and had no way to express the state
        that makes it legal — so the only span a host ever marks as failed was
        the only span the host could not keep. `_finish()` still supplies OTel's
        `_OTHER` when a manual span is ERROR with no type, so forgetting this
        call degrades the span rather than deleting it.
        """
        self._draft.set_error(error_type, message)

    def set_gen_ai(self, attrs: GenAIAttributes) -> None:
        self._draft.set_gen_ai(attrs)

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        self._draft.set_extra(key, value)

    @property
    def gen_ai(self) -> GenAIAttributes | None:
        return self._draft._gen_ai

    @gen_ai.setter
    def gen_ai(self, value: GenAIAttributes | None) -> None:
        self._draft._gen_ai = value

    @property
    def agent(self) -> AgentAttributes | None:
        return self._draft._agent

    @agent.setter
    def agent(self, value: AgentAttributes | None) -> None:
        self._draft._agent = value

    @property
    def tool(self) -> ToolAttributes | None:
        return self._draft._tool

    @tool.setter
    def tool(self, value: ToolAttributes | None) -> None:
        self._draft._tool = value

    @property
    def operation(self) -> OperationName | str | None:
        return self._draft._operation_label

    @operation.setter
    def operation(self, value: OperationName | str | None) -> None:
        self._draft.set_operation_label(value)

    @property
    def workflow_name(self) -> str | None:
        return self._draft._workflow_name

    @workflow_name.setter
    def workflow_name(self, value: str | None) -> None:
        self._draft.set_workflow_name(value)

    @property
    def call_site(self) -> CallSite | None:
        return self._draft._call_site

    @call_site.setter
    def call_site(self, value: CallSite | None) -> None:
        self._draft.set_call_site(value)

    @property
    def input_data(self) -> bytes:
        return self._draft._input_data

    @input_data.setter
    def input_data(self, value: bytes) -> None:
        self._draft._input_data = value

    @property
    def output_data(self) -> bytes:
        return self._draft._output_data

    @output_data.setter
    def output_data(self, value: bytes) -> None:
        self._draft._output_data = value

    def _finish(self) -> InternalSpan:
        # Private on purpose: the context manager owns completion. A public
        # `finish()` on a CM-yielded object was a second way to end the span,
        # and the one the `with` block then ran again on the way out.
        return self._draft.finish(self.end_time_ns or time.time_ns())


#: The parentage a span that will never be emitted hangs off. Minted ONCE at
#: import, through the sanctioned factory. Every degraded manual span shares it,
#: which costs nothing: none of them reaches a sink, so the trace it names has
#: no members. Each still gets its OWN draft — a shared one would accumulate
#: every host's `set_attribute` for the life of the process.
_NULL_PARENTAGE = resolve_parentage(EMPTY_AMBIENT, Evidence(ParentSource.UNRESOLVED))


@contextmanager
def _begin(
    name: str,
    kind: SpanKind,
    conversation: ConversationContext | None,
) -> Iterator[Span]:
    """Never raises an Exception of wardex's own making; the body ALWAYS runs.

    `wardex.span()` is the SDK's published context manager, so its `with` block
    is the host's own code — and every step below was once outside a guard:
    latching the ambient scope, resolving the edge, building the draft, forking
    the carrier, and handing the finished span to the client. A defect in any of
    them raised out of `with wardex.span(...)` and took the block with it.
    """
    draft = None
    ok = False
    with guard("tracing.manual_open", debug=_debug_enabled()):
        # A manual span is issued on the caller's own task, so the latch is here
        # and the evidence is the default (`AMBIENT`): a parent means the
        # ContextVar held one, a remote parent is re-labelled `header` by the
        # core, and no parent at all means this span deliberately roots a new
        # trace. Manual spans and adapter spans agree on all three because they
        # ask the same function — and they also build the same object through
        # the same constructor (I5).
        #
        # `resolve_observed`, not `resolve_parentage`, and the reason is that
        # this site OBSERVES a parent it did not open and cannot vet. Two things
        # it cannot vet, both of which used to arrive here as a confident 1.0
        # edge: a parent whose unit has already CLOSED — an activation fork the
        # adapter could not take down, so a `wardex.span()` after the run ended
        # became a child of an already-shipped span (design §10.3) — and a run
        # wardex itself failed to open, where the missing parent is wardex's
        # doing rather than the host's. The byte seams have asked this question
        # since §10.3; a hand-written span is issued on the same carriers and
        # inherits the same hazard, so it asks it too rather than having a
        # second answer.
        ambient = latch_ambient()
        parentage = resolve_observed(
            ambient, parent_closed=parent_is_closed_unit(ambient.span_context)
        )
        draft = SpanDraft.manual(
            parentage,
            name=name,
            kind=kind,
            start_ns=time.time_ns(),
            source=CaptureSource.MANUAL,
        )
        if conversation is not None:
            draft.set_conversation(conversation)
        ok = True
    if not ok:
        # A builder the host can still drive, over a draft nothing will emit —
        # and never the shared NULL_DRAFT, because `Span` is a view onto
        # a draft's own fields and a host that writes through it must not be
        # writing into every other degraded span in the process. There is no
        # `finally` below this yield: the span is already lost, and running the
        # emit path on a draft with no trace would be inventing one.
        report_once(
            f"[wardex] wardex.span({name!r}): internal error opening the span; "
            "this span and anything it would have parented will not be recorded "
            "(re-run with debug=True for the traceback)",
            key="wardex.span.manual_open",
        )
        # The host's block runs with nothing ambient, so under
        # `capture_mode=AGENT` every request inside it would be dropped at the
        # byte seam — a bug in wardex's own span turning into silence for the
        # work the span was opened to watch. The flag says the missing parent is
        # wardex's doing, and the gate and the seam's edge both read it.
        with degraded_run():
            yield Span(
                SpanDraft.manual(
                    _NULL_PARENTAGE, name=name, kind=kind, start_ns=0, source=CaptureSource.MANUAL
                )
            )
        return

    builder = Span(draft)
    fork = None
    forked = False
    with guard("tracing.manual_fork", debug=_debug_enabled()):
        fork = fork_active_span(draft.context)
        fork.__enter__()
        forked = True
    if not forked:
        # Half-entered at worst, so it must not be exited. This span still
        # ships; what is lost is everything opened INSIDE the block, which finds
        # whatever was standing before it instead.
        fork = None
        report_once(
            "[wardex] wardex.span(): internal error installing the span as the "
            "active parent; work inside this block will be attached one level "
            "too high (re-run with debug=True for the traceback)",
            key="wardex.span.manual_fork",
        )
    try:
        yield builder
    finally:
        if fork is not None:
            with guard("tracing.manual_fork_exit", debug=_debug_enabled()):
                fork.__exit__(None, None, None)
        finished = None
        # `_finish()` validates, and a vocabulary breach must not reach the host
        # (I6) — a `with wardex.span(...)` block would otherwise raise on the
        # way out of code that has nothing to do with wardex.
        with guard("tracing.manual_span", debug=_debug_enabled()):
            finished = builder._finish()
        client = None
        with guard("tracing.manual_client", debug=_debug_enabled()):
            client = _hub.get_client()
        if client is not None and finished is not None:
            with guard("tracing.manual_emit", debug=_debug_enabled()):
                client.capture_span(finished)


def _debug_enabled() -> bool:
    """Whether to log tracebacks. TOTAL, because every `guard()` below calls it.

    `debug=` is evaluated when the guard is CONSTRUCTED, which is one expression
    outside the block it is about to protect — so a client whose config read
    raises would break the host from inside the one call meant to prevent that.
    """
    debug = False
    with guard("tracing.debug_flag", debug=False):
        config = getattr(_hub.get_client(), "config", None)
        debug = bool(getattr(config, "debug", False))
    return debug


class _WithOnly:
    """`with` support over the generator CM, and NOTHING else — no decorator.

    `@contextmanager` returns a `ContextDecorator`, so `span("x")` used to be
    callable — and `@span("x")` on an `async def` compiled, ran, and silently
    closed the span before any awaited work started, because the decorator
    protocol wraps the CALL, which for a coroutine function merely builds the
    coroutine. This shape keeps the `with` protocol byte-for-byte (plain
    delegation over the generator CM) and turns the decorator misuse into a
    `TypeError` at decoration time, naming the decorators that do it right.
    """

    __slots__ = ("_api", "_cm")

    def __init__(self, api: str, cm: Any) -> None:
        self._api = api
        self._cm = cm

    def __enter__(self) -> Span:
        return self._cm.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
        return self._cm.__exit__(exc_type, exc, tb)

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            f"{self._api}() is a context manager; decorate with "
            "@workflow/@agent/@step/@tool instead"
        )


@contextmanager
def _conversation(name: str, *, id: str | None, op: OperationName | None) -> Iterator[Span]:
    conversation = ConversationContext(conversation_id=id if id is not None else str(uuid.uuid4()))
    # Reading the scope and stamping the conversation onto it are wardex's own
    # work, and they run BEFORE the host's block — so a failure here would take
    # the block with it. A conversation that could not be installed costs the
    # id on the spans inside; it does not cost the trace.
    scope = None
    installed = False
    with guard("tracing.conversation_scope", debug=_debug_enabled()):
        scope = _hub.get_current_scope()
        prev_conv = scope.conversation
        scope.conversation = conversation
        installed = True
    if not installed:
        scope = None
        report_once(
            "[wardex] wardex.conversation(): internal error reading the active "
            "scope; spans in this block will not carry a conversation id "
            "(re-run with debug=True for the traceback)",
            key="wardex.conversation.scope",
        )
    try:
        with _begin(name, SpanKind.INTERNAL, conversation) as builder:
            builder.operation = op
            yield builder
    finally:
        if scope is not None:
            with guard("tracing.conversation_scope_restore", debug=_debug_enabled()):
                scope.conversation = prev_conv


def conversation(
    name: str, *, id: str | None = None, op: OperationName | None = None
) -> Iterator[Span]:
    """Open a conversation: every span inside carries `gen_ai.conversation.id`.

    This is NOT a trace root, which is why it is not called "trace": the span
    it opens joins the ambient trace as a child like any other, and no new
    `trace_id` is minted here. What it opens is a CONVERSATION — the
    `gen_ai.conversation.id` every span captured inside the block is stamped
    with, which is how a backend groups the turns of one chat.

    `id=None` mints a fresh uuid4. An explicit `id` is used verbatim: a
    multi-turn chat app passes its own session id so that every turn joins ONE
    conversation instead of each turn becoming its own.
    """
    return _WithOnly("conversation", _conversation(name, id=id, op=op))


@contextmanager
def _span(
    name: str,
    op: OperationName | None,
    kind: SpanKind,
    agent: AgentAttributes | None,
    tool: ToolAttributes | None,
) -> Iterator[Span]:
    with _begin(name, kind, None) as builder:
        builder.agent = agent
        builder.tool = tool
        builder.operation = op
        yield builder


def span(
    name: str,
    op: OperationName | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    agent: AgentAttributes | None = None,
    tool: ToolAttributes | None = None,
) -> Iterator[Span]:
    """Open a hand-named span over the host's block and yield it as a `Span`."""
    return _WithOnly("span", _span(name, op, kind, agent, tool))


def _call_site(fn: Callable[..., Any]) -> CallSite:
    code = fn.__code__
    return CallSite(
        file=code.co_filename,
        line=code.co_firstlineno,
        function=fn.__name__,
        module=getattr(fn, "__module__", None),
    )


def _decorate(
    fn: Callable[..., Any] | None,
    *,
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
    `deco` is what wraps. `name=None` resolves to `fn.__name__` at decoration
    time, so the two forms name spans identically.
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        span_name = name if name is not None else fn.__name__
        workflow_name = span_name if names_workflow else None
        cs = _call_site(fn)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                with span(span_name, op=op, kind=kind, agent=agent, tool=tool) as s:
                    s.call_site = cs
                    s.workflow_name = workflow_name
                    return await fn(*args, **kwargs)

            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name, op=op, kind=kind, agent=agent, tool=tool) as s:
                s.call_site = cs
                s.workflow_name = workflow_name
                return fn(*args, **kwargs)

        return wrapper

    if fn is not None:
        return deco(fn)
    return deco


def workflow(
    fn: Callable[..., Any] | None = None, *, name: str | None = None
) -> Callable[..., Any]:
    return _decorate(fn, name=name, op=OperationName.INVOKE_WORKFLOW, names_workflow=True)


def agent(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    attributes: AgentAttributes | None = None,
) -> Callable[..., Any]:
    return _decorate(fn, name=name, op=OperationName.INVOKE_AGENT, agent=attributes)


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    attributes: ToolAttributes | None = None,
) -> Callable[..., Any]:
    return _decorate(fn, name=name, op=OperationName.EXECUTE_TOOL, tool=attributes)


def step(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Callable[..., Any]:
    """One step of a larger run — a graph node, a pipeline stage, a phase.

    Maps to `OperationName.EXECUTE_STEP`, so its spans appear on
    operation-keyed dashboards like every other decorator's (the old `task()`
    mapped no operation, and its spans vanished from any view keyed on
    `gen_ai.operation.name`). "step" also stays clear of the "task" asyncio,
    Celery and LangGraph each already mean something else by.
    """
    return _decorate(fn, name=name, op=OperationName.EXECUTE_STEP)
