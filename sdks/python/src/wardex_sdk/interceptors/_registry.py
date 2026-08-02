"""Manages interceptor install state — idempotent install/uninstall."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..assembly import guard
from ._base import InterceptorInterface

if TYPE_CHECKING:
    from .._client import Client


class InterceptorRegistry:
    def __init__(self) -> None:
        self._installed: dict[str, InterceptorInterface] = {}

    def install(self, interceptor: InterceptorInterface, client: Client | None) -> None:
        name = interceptor.name()
        if name in self._installed:
            return
        interceptor.install(client)
        self._installed[name] = interceptor

    def uninstall_all(self) -> None:
        """Uninstall every interceptor. Total: one failure cannot stop the rest.

        `PatchSet.restore_all()` already isolates the individual monkeypatches
        one level down, and this loop used to undo that: an interceptor whose
        `uninstall()` raised for any other reason — a tracker flushing a
        pending WS session, a shared-timing teardown — abandoned every
        interceptor queued behind it, left `_installed` populated (so the next
        `install()` silently no-ops by name, forever), and propagated out of
        `_lifecycle._teardown` before `client.close()`, losing every buffered
        span. That runs from `atexit`, where the exception goes nowhere anyone
        reads.

        Pop-then-uninstall, mirroring `restore_all()`: an entry leaves the
        table before its teardown is attempted, so a failure is not retried by
        a later call and cannot be double-counted, and an interrupted loop
        leaves exactly the interceptors it has not reached yet.
        """
        while self._installed:
            name = next(iter(self._installed))
            interceptor = self._installed.pop(name)
            with guard(f"interceptors.{name}.uninstall"):
                interceptor.uninstall()

    def is_installed(self, name: str) -> bool:
        return name in self._installed


_registry = InterceptorRegistry()


def get_registry() -> InterceptorRegistry:
    return _registry
