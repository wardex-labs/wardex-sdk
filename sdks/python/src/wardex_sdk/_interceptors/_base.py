"""Interceptor interface — spec §5.2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._client import Client


class InterceptorInterface(ABC):
    """Transport-layer I/O interceptor. Installed/uninstalled via monkey-patching.

    When Adapters are absent: generates its own CLIENT Span + transport attributes + raw I/O.
    """

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def install(self, client: Client | None) -> None: ...

    @abstractmethod
    def uninstall(self) -> None: ...
