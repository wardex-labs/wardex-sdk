"""Span-scoped context forking (Phase 4a).

Starting a span forks the *current* scope: the fork carries the new
active_span_context, and the ContextVar points at the fork for the span's
lifetime. asyncio copies the context on task creation, so concurrent tasks
each see their own fork — the shared-scope overwrite race (the old
known-limitation in _tracing.py) is structurally impossible.
"""

from __future__ import annotations

import contextvars
import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from .. import _hub
from .._types import SpanContext


@contextmanager
def fork_active_span(ctx: SpanContext) -> Iterator[None]:
    forked = _hub.get_current_scope().clone()
    forked.active_span_context = ctx
    token = _hub._current_scope.set(forked)
    try:
        yield
    finally:
        _hub._current_scope.reset(token)


def run_in_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``fn`` so it runs in a copy of the calling flow's context.

    Threads do not inherit contextvars automatically (asyncio tasks do).
    Capture happens at wrap time — wrap inside the span you want carried::

        thread = threading.Thread(target=wardex.run_in_context(work))
    """
    ctx = contextvars.copy_context()

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return ctx.run(fn, *args, **kwargs)

    return wrapper
