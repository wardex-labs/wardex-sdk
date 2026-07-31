"""Manages adapter install state — idempotent install/uninstall."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._limits import CaptureLimits
from ..assembly import Limitation, UnitRegistry, guard
from ._base import AdapterInterface
from ._context import AdapterContext
from ._sink import _ClientSink

if TYPE_CHECKING:
    from .._client import Client


class AdapterRegistry:
    def __init__(self) -> None:
        self._installed: dict[str, AdapterInterface] = {}

    def install(self, adapter: AdapterInterface, client: Client | None) -> None:
        name = adapter.name()
        if name in self._installed:
            return
        adapter.install(client, self._context_for(name, client))
        self._installed[name] = adapter

    @staticmethod
    def _context_for(name: str, client: Client | None) -> AdapterContext:
        """Build the surface this adapter will eventually be written against.

        Passed now and ignored by every adapter, so that the signature change
        lands apart from the behaviour change — an adapter migrating onto it is
        then a change to that adapter alone, not to the interface plus the
        registry plus everyone else at once.

        TRANSITIONAL: the registry it holds is its own, while each assembler
        still constructs one of its own too. Nothing opens a unit in this one
        yet, so the two cannot interact; the step that moves an adapter onto
        `ctx` is the step that collapses them.
        """
        config = getattr(client, "config", None)
        limits = config.limits if config is not None else CaptureLimits()
        return AdapterContext(
            name,
            units=UnitRegistry(sink=_ClientSink(client)),
            limits=limits.resolved(),
            debug=bool(getattr(config, "debug", False)),
        )

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
