from __future__ import annotations

import functools
import inspect
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from . import _hub
from ._enums import CaptureSource, OperationName, SpanKind, StatusCode
from ._types import (
    AgentAttributes,
    CallSite,
    ConversationContext,
    GenAIAttributes,
    InternalSpan,
    ToolAttributes,
)
from .assembly import SpanDraft, guard, latch_ambient, resolve_parentage
from .context._contextvar import fork_active_span


class SpanBuilder:
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
    def context(self):  # noqa: ANN201 — SpanContext; kept untyped to avoid the import
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
    # `finish()`, which is where the check belongs.
    #
    # `correlation` and `parent_span_id` are the two exceptions and they are
    # read-only on purpose: parentage is decided by `assembly/_parentage.py`
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
    def correlation(self):  # noqa: ANN201 — CorrelationInfo | None
        return self._draft._correlation

    @property
    def parent_span_id(self):  # noqa: ANN201 — SpanId | None
        return self._draft._parentage.parent_span_id

    def set_status(self, code: StatusCode, message: str = "") -> None:
        self._draft.set_status(code, message)

    def set_error(self, error_type: str, message: str = "") -> None:
        """Name the failure. Pairs with `set_status(StatusCode.ERROR)`.

        Without this the published surface could express the state that makes a
        span illegal (`set_status(ERROR)`) and had no way to express the state
        that makes it legal — so the only span a host ever marks as failed was
        the only span the host could not keep. `finish()` still supplies OTel's
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

    def finish(self) -> InternalSpan:
        return self._draft.finish(self.end_time_ns or time.time_ns())


@contextmanager
def _begin(
    name: str,
    kind: SpanKind,
    conversation: ConversationContext | None,
) -> Iterator[SpanBuilder]:
    # A manual span is issued on the caller's own task, so the latch is here and
    # the evidence is the default (`AMBIENT`): a parent means the ContextVar
    # held one, a remote parent is re-labelled `header` by the core, and no
    # parent at all means this span deliberately roots a new trace. Manual spans
    # and adapter spans agree on all three because they ask the same function —
    # and they also build the same object through the same constructor (I5).
    parentage = resolve_parentage(latch_ambient())
    draft = SpanDraft.manual(
        parentage,
        name=name,
        kind=kind,
        start_ns=time.time_ns(),
        source=CaptureSource.MANUAL,
    )
    if conversation is not None:
        draft.set_conversation(conversation)

    builder = SpanBuilder(draft)
    try:
        with fork_active_span(draft.context):
            yield builder
    finally:
        finished = None
        # `finish()` validates, and a vocabulary breach must not reach the host
        # (I6) — a `with wardex.span(...)` block would otherwise raise on the
        # way out of code that has nothing to do with wardex.
        with guard("tracing.manual_span", debug=_debug_enabled()):
            finished = builder.finish()
        client = _hub.get_client()
        if client is not None and finished is not None:
            client.capture_span(finished)


def _debug_enabled() -> bool:
    config = getattr(_hub.get_client(), "config", None)
    return bool(getattr(config, "debug", False))


@contextmanager
def trace(
    name: str,
    op: OperationName | None = None,
    tags: tuple[tuple[str, str], ...] = (),
) -> Iterator[SpanBuilder]:
    """Opens a top-level trace session and yields a SpanBuilder.

    ``op`` is applied to the builder immediately and serialized as ``gen_ai.operation.name``.
    ``tags`` is accepted for public API stability and nothing reads it: no scope tag
    reaches the span, so passing tags here changes nothing about what is exported.
    """
    conversation = ConversationContext(conversation_id=str(uuid.uuid4()))
    scope = _hub.get_current_scope()
    prev_conv = scope.conversation
    scope.conversation = conversation
    try:
        with _begin(name, SpanKind.INTERNAL, conversation) as builder:
            builder.operation = op
            yield builder
    finally:
        scope.conversation = prev_conv


@contextmanager
def span(
    name: str,
    op: OperationName | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    agent: AgentAttributes | None = None,
    tool: ToolAttributes | None = None,
) -> Iterator[SpanBuilder]:
    with _begin(name, kind, None) as builder:
        builder.agent = agent
        builder.tool = tool
        builder.operation = op
        yield builder


def _call_site(fn: Callable[..., Any]) -> CallSite:
    code = fn.__code__
    return CallSite(
        file=code.co_filename,
        line=code.co_firstlineno,
        function=fn.__name__,
        module=getattr(fn, "__module__", None),
    )


def _decorate(
    name: str,
    *,
    op: OperationName | None,
    kind: SpanKind = SpanKind.INTERNAL,
    agent: AgentAttributes | None = None,
    tool: ToolAttributes | None = None,
    workflow_name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        cs = _call_site(fn)

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                with span(name, op=op, kind=kind, agent=agent, tool=tool) as s:
                    s.call_site = cs
                    s.workflow_name = workflow_name
                    return await fn(*args, **kwargs)

            return awrapper

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(name, op=op, kind=kind, agent=agent, tool=tool) as s:
                s.call_site = cs
                s.workflow_name = workflow_name
                return fn(*args, **kwargs)

        return wrapper

    return deco


def workflow(*, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _decorate(name, op=OperationName.INVOKE_WORKFLOW, workflow_name=name)


def agent(
    *, name: str, agent: AgentAttributes | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _decorate(name, op=OperationName.INVOKE_AGENT, agent=agent)


def tool(
    *, name: str, tool: ToolAttributes | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _decorate(name, op=OperationName.EXECUTE_TOOL, tool=tool)


def task(*, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    return _decorate(name, op=None)
