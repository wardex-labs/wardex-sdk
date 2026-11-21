"""What `wardex.current_span()` answers: the innermost span a `wardex.span()` block or a decorator
opened in this context.

Each span keeps the one it replaced (`Span._outer`), so a span that closed in another Context than
it opened in (a generator's body, finished elsewhere) is skipped rather than reported as current.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._tracing import Span

_CURRENT_SPAN: ContextVar[Span | None] = ContextVar("wardex_current_span", default=None)


def current_span() -> Span | None:
    """The innermost wardex span open in this context, or None outside every one.

    The span a `with wardex.span(...)` block or a `@workflow`/`@agent`/`@tool`/
    `@step` function is running under, so the code inside can call
    `set_input`, `set_output`, `set_attribute` or `add_event` on it. Spans an
    adapter opens for a framework are not returned: they are not the host's to
    write. Never raises. Before `init()` a block's span is returned as well; like
    the block, it records nothing.
    """
    return _open_from(_CURRENT_SPAN.get())


def _open_from(span: Span | None) -> Span | None:
    """`span`, or the first open span it was opened under."""
    while span is not None and not span._open:
        span = span._outer
    return span


def _make_current(span: Span) -> Any:
    """Make `span` what `current_span()` answers here; the token to undo it with."""
    span._outer = current_span()
    return _CURRENT_SPAN.set(span)


def _unmake_current(span: Span, token: Any) -> None:
    span._open = False
    # Closed in another Context, it stays in the one it opened in: linking it on
    # to closed spans would keep every span a long-lived task ever opened.
    span._outer = _open_from(span._outer)
    if token is None:
        return
    try:
        _CURRENT_SPAN.reset(token)
    except ValueError:
        # Ended in another Context than it began in (a generator's body
        # finished elsewhere): that Context keeps the span, which `_open`
        # now makes `current_span()` step past.
        pass
