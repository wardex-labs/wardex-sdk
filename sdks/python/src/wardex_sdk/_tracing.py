from __future__ import annotations

import functools
import inspect
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from . import _hub
from ._enums import OperationName, SpanKind, StatusCode
from ._types import (
    AgentAttributes,
    CallSite,
    ConversationContext,
    CorrelationInfo,
    GenAIAttributes,
    InternalSpan,
    SpanContext,
    ToolAttributes,
)
from .assembly import Parentage, latch_ambient, resolve_parentage
from .context._contextvar import fork_active_span


class SpanBuilder:
    def __init__(
        self,
        parentage: Parentage,
        context: SpanContext,
        name: str,
        kind: SpanKind,
        conversation: ConversationContext | None,
    ) -> None:
        self.context = context
        self.parent_span_id = parentage.parent_span_id
        self.correlation: CorrelationInfo | None = parentage.correlation
        self.name = name
        self.kind = kind
        self.conversation = conversation
        self.start_time_ns = time.time_ns()
        self.end_time_ns = 0
        self.status = StatusCode.UNSET
        self.status_message = ""
        self.gen_ai: GenAIAttributes | None = None
        self.agent: AgentAttributes | None = None
        self.tool: ToolAttributes | None = None
        self.input_data = b""
        self.output_data = b""
        # fields filled in by decorator sugar (Task 12)
        self.operation: OperationName | str | None = None
        self.workflow_name: str | None = None
        self.call_site: CallSite | None = None
        self._extra: list[tuple[str, str | int | float | bool]] = []

    def set_status(self, code: StatusCode, message: str = "") -> None:
        self.status = code
        self.status_message = message

    def set_gen_ai(self, attrs: GenAIAttributes) -> None:
        self.gen_ai = attrs

    def set_attribute(self, key: str, value: str | int | float | bool) -> None:
        self._extra.append((key, value))

    def finish(self) -> InternalSpan:
        extra_list: list[tuple[str, str | int | float | bool]] = []
        if self.operation is not None:
            op_val = (
                self.operation.value
                if isinstance(self.operation, OperationName)
                else self.operation
            )
            extra_list.append(("gen_ai.operation.name", op_val))
        extra_list.extend(self._extra)
        extra = tuple(extra_list)
        return InternalSpan(
            context=self.context,
            parent_span_id=self.parent_span_id,
            name=self.name,
            kind=self.kind,
            start_time_ns=self.start_time_ns,
            end_time_ns=self.end_time_ns or time.time_ns(),
            status=self.status,
            status_message=self.status_message,
            gen_ai=self.gen_ai,
            agent=self.agent,
            tool=self.tool,
            conversation=self.conversation,
            workflow_name=self.workflow_name,
            call_site=self.call_site,
            input_data=self.input_data,
            output_data=self.output_data,
            correlation=self.correlation,
            extra=extra,
        )


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
    # and adapter spans now agree on all three because they ask the same
    # function.
    parentage = resolve_parentage(latch_ambient())
    ctx = parentage.child_context()
    conv = conversation if conversation is not None else parentage.conversation

    builder = SpanBuilder(parentage, ctx, name, kind, conv)
    try:
        with fork_active_span(ctx):
            yield builder
    finally:
        finished = builder.finish()
        client = _hub.get_client()
        if client is not None:
            client.capture_span(finished)


@contextmanager
def trace(
    name: str,
    op: OperationName | None = None,
    tags: tuple[tuple[str, str], ...] = (),
) -> Iterator[SpanBuilder]:
    """Opens a top-level trace session and yields a SpanBuilder.

    ``op`` is applied to the builder immediately and serialized as ``gen_ai.operation.name``.
    ``tags`` is included in the signature for public API stability, but Scope -> Span tag
    application is not implemented in Phase 1 and will be wired up in a later stage.
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
