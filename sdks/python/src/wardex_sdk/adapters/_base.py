"""Adapter interface — L1 framework adapters (mirror of InterceptorInterface)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..assembly import Limitation

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

    def close_units(self, *, marker: Limitation) -> None:
        """Close whatever spans are still open, but stay installed.

        Concrete and not abstract, deliberately. This exists for the shutdown
        signal path, which most adapters have nothing to answer for — an
        adapter that holds no open span across calls is already correct doing
        nothing. Making it abstract would break every out-of-tree adapter to
        force them all to write the same empty body.
        """
        return None
