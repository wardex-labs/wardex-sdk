"""Manages adapter install state — idempotent install/uninstall."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        for adapter in self._installed.values():
            adapter.uninstall()
        self._installed.clear()

    def is_installed(self, name: str) -> bool:
        return name in self._installed


_registry = AdapterRegistry()


def get_registry() -> AdapterRegistry:
    return _registry
