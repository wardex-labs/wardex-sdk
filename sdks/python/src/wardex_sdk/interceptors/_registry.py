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
        """Install one interceptor. A failure here costs its seam and nothing else.

        The same two defects `AdapterRegistry.install` closed, in the registry
        next door, and deliberately the same shape: this asymmetry written twice
        with two different remedies is how the two sides drift apart again.

        GUARDED, which it was not. `uninstall_all` has always been total and
        `install` was not, so an interceptor raising out of `install()` took
        `wardex.init()` down with it — a host that added observability got a
        crash at startup from the one component whose whole promise is never to
        alter the application. Here that is likelier than on the adapter side
        and less often anybody's mistake: these seams patch attributes of the
        stdlib and of third-party internals (`ssl.SSLSocket.recv`,
        `anyio._backends._asyncio.AsyncIOBackend.open_process`), so a version
        bump in a package the user never chose is an ordinary way for this to
        raise.

        FILED BEFORE CALLED, which is the other half. An `install()` that raises
        halfway has already patched part of a surface, and an interceptor the
        registry never recorded is one nothing can reach — those wrappers stayed
        in front of the host's sockets for the life of the process, with no
        object left able to remove them. Recording first is what lets the
        rollback below run at all.

        The undo is best-effort and says so rather than pretending: each
        interceptor owns its own `PatchSet`, so restoring it is that
        interceptor's `uninstall()`'s job, and one that sets its installed flag
        last will decline. The entry is dropped either way — a name left in the
        table is one every later `install()` skips by name, turning one failed
        install into a seam that is never attempted again.
        """
        name = interceptor.name()
        if name in self._installed:
            return
        self._installed[name] = interceptor

        ok = False
        with guard(f"interceptors.{name}.install"):
            interceptor.install(client)
            ok = True
        if ok:
            return

        with guard(f"interceptors.{name}.install_rollback"):
            interceptor.uninstall()
        self._installed.pop(name, None)

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
