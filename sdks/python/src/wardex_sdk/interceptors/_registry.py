"""Manages interceptor install state — idempotent install/uninstall."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        for interceptor in self._installed.values():
            interceptor.uninstall()
        self._installed.clear()

    def is_installed(self, name: str) -> bool:
        return name in self._installed


_registry = InterceptorRegistry()


def get_registry() -> InterceptorRegistry:
    return _registry
