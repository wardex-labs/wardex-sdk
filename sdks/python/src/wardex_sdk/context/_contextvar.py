"""Span-scoped context forking (Phase 4a) — design §5.6.

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
"""

from __future__ import annotations

import contextvars
import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from .. import _hub
from .._types import ConversationContext, SpanContext


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
    host did not ask for. Every carrier in `assembly/_units.py` is built on this
    rule rather than around it.

    `conversation` and `tracestate` are OVERRIDES, not assignments: `None`
    leaves whatever the cloned scope already carried. That is what keeps this a
    true generalization of `fork_active_span` — the old two-line body inherited
    both fields from the parent scope, and a version that wrote `None` through
    would clear a conversation id merely because the caller did not restate it.
    """
    forked = _hub.get_current_scope().clone()
    forked.active_span_context = ctx
    if conversation is not None:
        forked.conversation = conversation
    if tracestate is not None:
        forked.tracestate = tracestate
    token = _hub._current_scope.set(forked)
    try:
        yield
    finally:
        _hub._current_scope.reset(token)


#: The original name, kept because it is what `_tracing.py` and the tests call.
#: A plain alias rather than a wrapper: a delegating `fork_active_span` would be
#: a second entry point to grow a second opinion in, which is the drift the
#: whole `assembly/` extraction exists to end.
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
