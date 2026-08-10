"""Span-scoped context forking — design §5.6.

Starting a span forks the *current* scope: the fork carries the new
active_span_context, and the ContextVar points at the fork for the span's
lifetime. asyncio copies the context on task creation, so concurrent tasks
each see their own fork — the shared-scope overwrite race (the old
known-limitation in _tracing.py) is structurally impossible.

`activate_span` is the general form and `fork_active_span` is now an alias of
it. The generalization is what a logical unit needs: a unit carries a
conversation identity and a tracestate alongside its span context, and a
carrier that installs only the context leaves the other two behind on every
task the unit spans — so a sub-agent's spans would silently lose the
conversation id the session issued.

`install_span` is the UNBALANCED form of the same fork, for a carrier that is
never exited. Both build the fork through `_fork`, so the override semantics
below are stated once.
"""

from __future__ import annotations

import contextvars
import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from .. import _hub
from .._scope import Scope
from .._types import ConversationContext, SpanContext


def _fork(
    prev: Scope,
    ctx: SpanContext,
    conversation: ConversationContext | None,
    tracestate: str | None,
) -> Scope:
    """Clone `prev` with `ctx` active. The one place the fork rule is written.

    `conversation` and `tracestate` are OVERRIDES, not assignments: `None`
    leaves whatever the cloned scope already carried. That is what keeps
    `activate_span` a true generalization of `fork_active_span` — the old
    two-line body inherited both fields from the parent scope, and a version
    that wrote `None` through would clear a conversation id merely because the
    caller did not restate it.

    Factored out rather than duplicated into `install_span` because a second
    copy of these five lines is a second place for that rule to drift, which is
    the drift the whole `_assembly/` extraction exists to end.
    """
    forked = prev.clone()
    forked.active_span_context = ctx
    if conversation is not None:
        forked.conversation = conversation
    if tracestate is not None:
        forked.tracestate = tracestate
    return forked


def install_span(
    ctx: SpanContext,
    *,
    conversation: ConversationContext | None = None,
    tracestate: str | None = None,
) -> Scope:
    """Fork the scope with NO finaliser. Returns the scope that was current.

    For a carrier that is never exited — a pin (design §5.6), whose whole point
    is that the driver task keeps the unit ambient for the rest of its life.
    `activate_span` is the balanced form and is wrong for that job: a `finally`
    is a promise that the block WILL be exited, so the only way a pin's intent
    can be violated is by that finaliser running, and a suspended generator
    runs its `finally` the moment nothing references it. A caller that reads
    `.installed` off a temporary `PinToken` and drops it would silently undo
    the fork on the statement that installed it.

    This form cannot be undone by dropping a reference. `restore_scope` is the
    deliberate, same-task way back, and there is no other.
    """
    prev = _hub.get_current_scope()
    _hub._current_scope.set(_fork(prev, ctx, conversation, tracestate))
    return prev


def restore_scope(prev: Scope) -> None:
    """Put `prev` back as the current scope. MUST run on the installing task.

    A plain `set()` rather than a Token reset, because `install_span` has no
    Token to reset — and because a `set()` from another task would land on THAT
    task's scope, silently. The caller checks task identity; see
    `_assembly/_units.py::_Carrier.remove`.
    """
    _hub._current_scope.set(prev)


@contextmanager
def activate_span(
    ctx: SpanContext,
    *,
    conversation: ConversationContext | None = None,
    tracestate: str | None = None,
) -> Iterator[None]:
    """Make `ctx` the ambient parent for the duration of this block.

    Enter and exit MUST happen on the same task/thread: the ContextVar Token
    returned by `set()` may only be reset in the Context it was created in, and
    resetting it elsewhere raises `ValueError` — into the host, on a path the
    host did not ask for. Every carrier in `_assembly/_units.py` is built on this
    rule rather than around it.

    `conversation` and `tracestate` are OVERRIDES, not assignments: `None`
    leaves whatever the cloned scope already carried — see `_fork`, which both
    forms share.
    """
    prev = _hub.get_current_scope()
    token = _hub._current_scope.set(_fork(prev, ctx, conversation, tracestate))
    try:
        yield
    finally:
        _hub._current_scope.reset(token)


#: The original name, kept because it is what `_tracing.py` and the tests call.
#: A plain alias rather than a wrapper: a delegating `fork_active_span` would be
#: a second entry point to grow a second opinion in, which is the drift the
#: whole `_assembly/` extraction exists to end.
fork_active_span = activate_span


def run_in_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``fn`` so it runs in a copy of the calling flow's context.

    Threads do not inherit contextvars automatically (asyncio tasks do).
    Capture happens at wrap time — wrap inside the span you want carried::

        thread = threading.Thread(target=wardex.run_in_context(work))

    NOT the mechanism `Unit.bind()` uses, and the difference is load-bearing:
    this replays ONE captured `contextvars.Context`, and entering the same
    Context twice concurrently raises ``RuntimeError: cannot enter context ...
    is already entered``. For a framework callback invoked from two tasks that
    error surfaces into USER CODE. `Unit.bind` enters a fresh fork per
    invocation instead.
    """
    ctx = contextvars.copy_context()

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return ctx.run(fn, *args, **kwargs)

    return wrapper
