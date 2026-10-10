"""The published context managers, kept honest about being used as decorators.

`@contextmanager` returns a `ContextDecorator`, so every published context
manager built on it is also callable as a decorator — and as one it wraps the
CALL. For a plain function that is the whole execution. For an `async def`, a
generator or an async generator, the call only builds the coroutine or the
generator and returns, so the block closes before any of the body runs and the
body runs outside it: no error, no warning, and a trace that is quietly wrong.

Two shapes, because the published managers differ in what their decorator form
has ever meant. `WithOnly` refuses every decorator use (`span()` and
`conversation()` always did; the four decorators are the way to trace a
function). `WithOrSyncDecorator` keeps the decorator form wherever it was right
— a plain function — and refuses it exactly where it never covered the body.
Both keep the `with` protocol byte-for-byte (plain delegation), and both refuse
at decoration time with what was refused, why, and what to write instead.
"""

from __future__ import annotations

import inspect
from typing import Any


class WithOnly:
    """A context manager that refuses to be used as a decorator."""

    __slots__ = ("_api", "_cm", "_instead")

    def __init__(self, api: str, cm: Any, instead: str) -> None:
        self._api = api
        self._cm = cm
        self._instead = instead

    def __enter__(self) -> Any:
        return self._cm.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool | None:
        return self._cm.__exit__(exc_type, exc, tb)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            f"wardex.{self._api}() is a context manager, not a decorator: as a "
            "decorator it would wrap only the call, which for an async function "
            "or a generator returns before the body runs, so the body would run "
            f"outside it. {self._instead}"
        )


class WithOrSyncDecorator(WithOnly):
    """A context manager whose decorator form is kept for plain functions only.

    Over a plain function the decorator form covers the whole call and always
    did, so it is delegated to the generator CM unchanged (`ContextDecorator`
    builds a fresh manager per call). Over an async function, a generator or an
    async generator it never did, and that is refused.
    """

    __slots__ = ()

    def __call__(self, fn: Any) -> Any:
        shape = _late_shape(fn)
        if shape is None:
            return self._cm(fn)
        raise TypeError(
            f"wardex.{self._api}() cannot decorate "
            f"{getattr(fn, '__name__', type(fn).__name__)!r} ({shape}): as a decorator "
            "it wraps only the call, and that call returns before the body runs, so the "
            f"body would run outside it. {self._instead}"
        )


def _late_shape(fn: Any) -> str | None:
    """The shape whose body runs after its call returns, or None for a plain call."""
    if inspect.isasyncgenfunction(fn):
        return "async generator function"
    if inspect.isgeneratorfunction(fn):
        return "generator function"
    if inspect.iscoroutinefunction(fn):
        return "async function"
    return None
