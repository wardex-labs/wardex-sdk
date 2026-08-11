"""Adapter interface — L1 framework adapters (mirror of InterceptorInterface)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from .._assembly import Limitation

if TYPE_CHECKING:
    from .._client import Client
    from ._context import AdapterContext


class AdapterInterface(ABC):
    """Framework-level adapter. Installed/uninstalled via monkey-patching.

    Adapters observe framework surfaces (message streams, hooks) and emit
    semantic spans; they must never alter the host application's behavior.
    """

    #: Exceptions this framework uses as CONTROL FLOW rather than as failure.
    #: A CLASSVAR, uniformly referenced: `_run` reads it through the context on
    #: EVERY causal path rather than at each call site, which is the mistake the
    #: design names — a per-path reference leaves the paths nobody remembered,
    #: and it is why Sentry still marks a `GraphBubbleUp` that escaped
    #: `Pregel.invoke` as an error.
    #:
    #: Declared here with an empty default so every adapter has one and the read
    #: needs no `getattr` fallback. An adapter whose framework's error classes
    #: can only be imported inside `install()` assigns
    #: `type(self).CONTROL_FLOW = (...)` there — the read happens later, when an
    #: exception is being classified, so install-time population suffices.
    CONTROL_FLOW: tuple[type[BaseException], ...] = ()

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def install(self, client: Client | None, ctx: AdapterContext | None = None) -> None:
        """Patch the framework's surface.

        `ctx` is the contract an adapter is moving onto — see
        `_adapters/_context.py`. It is optional while adapters migrate one at a
        time; an adapter that ignores it keeps today's behaviour exactly.
        """
        ...

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
