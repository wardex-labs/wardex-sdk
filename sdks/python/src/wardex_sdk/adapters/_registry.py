"""Manages adapter install state — idempotent install/uninstall."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..assembly import Limitation, guard
from ._base import AdapterInterface

if TYPE_CHECKING:
    from .._client import Client


class AdapterRegistry:
    def __init__(self) -> None:
        self._installed: dict[str, AdapterInterface] = {}

    def install(self, adapter: AdapterInterface, client: Client | None) -> None:
        name = adapter.name()
        if name in self._installed:
            return
        adapter.install(client)
        self._installed[name] = adapter

    def uninstall_all(self) -> None:
        """Uninstall every adapter. Total: one failure cannot stop the rest.

        Same rule as `InterceptorRegistry.uninstall_all`, and the same reason:
        this loop runs inside `_lifecycle._teardown`, immediately before
        `client.close()`. An adapter whose `uninstall()` raised took the close
        with it and every span still in the buffer, left `_installed`
        populated so the next `init()` silently no-ops for that adapter by
        name, and did it from `atexit`, where nothing reports the exception.
        Pop-then-uninstall so a failure is neither retried nor double-counted.
        """
        while self._installed:
            name = next(iter(self._installed))
            adapter = self._installed.pop(name)
            with guard(f"adapters.{name}.uninstall"):
                adapter.uninstall()

    def close_units_all(self, *, marker: Limitation) -> None:
        """Ask every INSTALLED adapter to close its open spans, and keep it installed.

        Iterates without popping, which is the difference from `uninstall_all`:
        this runs from the signal handler, where the process may or may not be
        about to die, and an adapter that stopped being installed because a
        shutdown signal arrived would stop capturing for a program that then
        carries on. Guarded per adapter for `uninstall_all`'s reason — one
        failure here must not cost the flush that follows it.
        """
        for name, adapter in list(self._installed.items()):
            with guard(f"adapters.{name}.close_units"):
                adapter.close_units(marker=marker)

    def is_installed(self, name: str) -> bool:
        return name in self._installed


_registry = AdapterRegistry()


def get_registry() -> AdapterRegistry:
    return _registry
