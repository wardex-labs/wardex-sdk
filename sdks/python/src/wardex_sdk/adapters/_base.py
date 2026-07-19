"""Adapter interface — L1 framework adapters (mirror of InterceptorInterface)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._client import Client


class AdapterInterface(ABC):
    """Framework-level adapter. Installed/uninstalled via monkey-patching.

    Adapters observe framework surfaces (message streams, hooks) and emit
    semantic spans; they must never alter the host application's behavior.
    """

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def install(self, client: Client | None) -> None: ...

    @abstractmethod
    def uninstall(self) -> None: ...
