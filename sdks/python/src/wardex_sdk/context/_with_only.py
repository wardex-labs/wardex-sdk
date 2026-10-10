"""`with` support over a generator context manager, and nothing else.

`@contextmanager` returns a `ContextDecorator`, so every published context
manager built on it is also callable as a decorator — and as one it wraps the
CALL. For a plain function that happens to work. For an `async def`, a
generator or an async generator, the call only builds the coroutine or the
generator and returns, so the block closes before any of the body runs and the
body runs outside it: no error, no warning, and a trace that is quietly wrong.

`WithOnly` keeps the `with` protocol byte-for-byte (plain delegation) and turns
the decorator form into a `TypeError` at decoration time that says what was
refused, why, and what to write instead. Every published context manager wears
it, so the rule is the same whichever one a host reaches for.
"""

from __future__ import annotations

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

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            f"wardex.{self._api}() is a context manager, not a decorator: as a "
            "decorator it would wrap only the call, which for an async function "
            "or a generator returns before the body runs, so the body would run "
            f"outside it. {self._instead}"
        )
