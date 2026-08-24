"""Manages interceptor install state — idempotent install/uninstall."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._assembly import guard
from ._base import InterceptorInterface

if TYPE_CHECKING:
    from .._client import Client


class InterceptorRegistry:
    def __init__(self) -> None:
        self._installed: dict[str, InterceptorInterface] = {}

    def install(self, interceptor: InterceptorInterface, client: Client | None) -> None:
        """Install one interceptor. A failure here costs its seam and nothing else.

        The same two defects `AdapterRegistry.install` closed, in the registry
        next door, and the same shape wherever the two sides have the same
        problem: this asymmetry written twice with two different remedies is how
        they drift apart again. Where they now differ is the rollback, and the
        last paragraph but two says why — an adapter's undo is its patches, a
        seam's is its patches plus a refcount and a WebSocket flush.

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

        The undo really undoes, which is the third thing and the one the first
        two are worth nothing without. Each interceptor owns its own `PatchSet`,
        so restoring is its `uninstall()`'s job — and all three of them used to
        open with `if not self._installed: return` against a flag their
        `install()` sets LAST, so the rollback called an undo that declined,
        every time, for every interceptor in the product. They are now total:
        each undoes whatever it got as far as, and is safe to call on a seam
        that installed nothing (`_seam.ByteSeamInterceptor.uninstall`).

        Routing the patches through a registry-owned `PatchSet` — the shape
        `AdapterRegistry` uses, where `ctx.patches.restore_all()` runs
        unconditionally — was the other candidate and it undoes strictly less
        here. A seam's install is not only patches: it takes a refcounted
        reference on the shared connection-timing probe and it holds live
        WebSocket sessions whose spans are emitted at teardown. A registry that
        restored the patches and nothing else would leave that refcount raised
        forever, which is a `socket.connect` patch nothing can ever remove, and
        `uninstall()` would still have to be total for the rest — so it buys a
        second mechanism and keeps the bug.

        POP FIRST, then roll back. The pop used to sit after the rollback, where
        a `BaseException` out of `uninstall()` — `guard` re-raises those by
        design, because a `KeyboardInterrupt` is the host's control flow — flew
        past it and left the name in the table forever, and a name left in the
        table is one every later `install()` skips by name. Nothing reads the
        table during an `uninstall()`, so the earlier pop costs nothing; this is
        the order `uninstall_all` already documents for the same reason.
        """
        name = interceptor.name()
        if name in self._installed:
            return
        self._installed[name] = interceptor
        # `debug` is read here so a failed seam is one line on stderr under
        # `init(debug=True)` rather than a counter with no reader — which
        # `_diag.report_once` argues is the same as saying nothing. This is the
        # failure the guards above exist for; it is the last one that should be
        # undiagnosable.
        debug = bool(getattr(getattr(client, "config", None), "debug", False))

        ok = False
        with guard(f"interceptors.{name}.install", debug=debug):
            interceptor.install(client)
            ok = True
        if ok:
            return

        self._installed.pop(name, None)
        with guard(f"interceptors.{name}.install_rollback", debug=debug):
            interceptor.uninstall()

    def uninstall_all(self) -> None:
        """Uninstall every interceptor. Total: one failure cannot stop the rest.

        `PatchSet.restore_all()` already isolates the individual monkeypatches
        one level down, and this loop used to undo that: an interceptor whose
        `uninstall()` raised for any other reason — a tracker flushing a
        pending WS session, a shared-timing teardown — abandoned every
        interceptor queued behind it, left `_installed` populated (so the next
        `install()` silently no-ops by name, forever), and propagated out of
        `Runtime._teardown` before `client.close()`, losing every buffered
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

    def _at_fork_reinit(self) -> None:
        """Fork-child reset, delegated to every installed interceptor.

        The registry stays populated and every patch stays installed
        (I-fork-4): the child is still instrumented, it just must not trust
        per-connection state built for the parent's sockets. A seam that
        declares no `_at_fork_reinit` is stating it holds no per-process
        mutable state to reset — not even a PatchSet, whose lock row Q makes
        every holder replace (MCP stdio declares one for exactly that lock;
        its per-stream state rides closures the fork either carries validly
        or never touches — `_FORK_EXEMPT`). The coverage guard in
        `tests/test_fork_reinit_coverage.py` is what keeps that statement
        honest for every FUTURE holder.

        Per-seam `guard()` so one failing reset cannot abandon the rest — the
        same totality rule as `uninstall_all`, on the same kind of path.
        """
        for name, interceptor in list(self._installed.items()):
            reinit = getattr(interceptor, "_at_fork_reinit", None)
            if reinit is None:
                continue
            with guard(f"interceptors.{name}.fork_reinit_failed"):
                reinit()

    def is_installed(self, name: str) -> bool:
        return name in self._installed


def get_registry() -> InterceptorRegistry:
    """The process registry, which `Runtime` owns and builds on first use.

    Not a module singleton of its own any more: a registry nobody owns is a
    registry `reset_for_test()` cannot drain, which is how an interceptor
    installed by one test stayed in front of the host's sockets for every test
    behind it.
    """
    from .._runtime import runtime

    return runtime().interceptors
