"""Manages adapter install state — idempotent install/uninstall."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .._limits import CaptureLimits
from ..assembly import Limitation, UnitRegistry, counters, guard
from ._base import AdapterInterface
from ._context import AdapterContext
from ._sink import _ClientSink

if TYPE_CHECKING:
    from .._client import Client


class AdapterRegistry:
    def __init__(self) -> None:
        self._installed: dict[str, AdapterInterface] = {}
        self._contexts: dict[str, AdapterContext] = {}

    def install(self, adapter: AdapterInterface, client: Client | None) -> None:
        """Install one adapter. A failure here costs its spans and nothing else.

        GUARDED, which it was not. `uninstall_all` has always been total, and
        the asymmetry was the bug: an adapter raising out of `install()` took
        `wardex.init()` down with it, so a host that added observability got a
        crash at startup from the one component whose whole promise is never to
        alter the application.

        FILED BEFORE CALLED, which is the other half. An `install()` that raises
        halfway has already patched part of a framework's surface, and an
        adapter the registry never recorded is one `uninstall_all` never
        reaches — so those wrappers stayed in the host's classes for the life of
        the process, with nothing able to remove them. Recording first means the
        teardown path can always find it.

        The undo is best-effort today and says so rather than pretending: the
        adapter's patches live in a `PatchSet` it builds itself, so restoring
        them is its `uninstall()`'s job, and an adapter that sets its installed
        flag last will decline. Routing every patch through `ctx.patches` is
        what makes the undo unconditional, and that is a later step.
        """
        name = adapter.name()
        if name in self._installed:
            return
        ctx = self._context_for(name, client)
        self._installed[name] = adapter
        self._contexts[name] = ctx

        ok = False
        with guard(f"adapters.{name}.install"):
            adapter.install(client, ctx)
            ok = True
        if ok:
            return

        with guard(f"adapters.{name}.install_rollback"):
            adapter.uninstall()
        ctx.patches.restore_all()
        self._installed.pop(name, None)
        self._contexts.pop(name, None)

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
        resolved = limits.resolved()
        debug = bool(getattr(config, "debug", False))
        return AdapterContext(
            name,
            # The bounds are passed HERE and not left to the registry's defaults.
            # Once an adapter shares this registry, this is the only place a
            # user's `max_units` can reach it — a registry built with defaults
            # would ignore the setting in silence, which is the shape of bug
            # that looks like nothing at all until a workload crosses a cap the
            # user thought they had raised.
            units=UnitRegistry(
                sink=_ClientSink(client),
                max_units=resolved["max_units"],
                max_entries_per_unit=resolved["max_entries_per_unit"],
                debug=debug,
            ),
            limits=resolved,
            debug=debug,
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

        LIFO, mirroring `PatchSet`'s own restore order and for the same reason.
        Two adapters that patched the same attribute leave the later one's
        wrapper in place; undoing the EARLIER one first finds a value that is
        not its own, correctly declines to touch it, and reports a supersession
        — after which the later undo restores the earlier adapter's wrapper as
        though it were the host's original. Newest-first, each undo sees exactly
        what it installed.
        """
        self._drain(lambda name, adapter: adapter.uninstall(), "uninstall")

    def close_units_all(self, *, marker: Limitation) -> None:
        """Ask every INSTALLED adapter to close its open spans, and keep it installed.

        Keeps each adapter installed, which is the difference from
        `uninstall_all`: this runs from the signal handler, where the process
        may or may not be about to die, and an adapter that stopped being
        installed because a shutdown signal arrived would stop capturing for a
        program that then carries on.
        """
        self._sweep(
            list(self._installed.items()),
            lambda name, adapter: adapter.close_units(marker=marker),
            "close_units",
        )

    def _drain(self, act, where: str) -> None:  # noqa: ANN001
        """Pop every adapter newest-first and act on it. Nothing survives a failure."""
        ordered = []
        while self._installed:
            name = next(reversed(self._installed))
            ordered.append((name, self._installed.pop(name)))
            self._contexts.pop(name, None)
        self._sweep(ordered, act, where)

    @staticmethod
    def _sweep(items, act, where: str) -> None:  # noqa: ANN001
        """Act on every item, and let no exception cut the loop short.

        `guard()` deliberately re-raises `BaseException` — a `KeyboardInterrupt`
        or a `CancelledError` is the host's control flow and swallowing it would
        be the SDK deciding when the program stops. But this loop runs at
        teardown, so re-raising it IN PLACE abandons every adapter behind the
        one that was interrupted: their patches stay in the host's classes and
        their open spans are never emitted, from `atexit`, where nothing reports
        why. So it is captured, the sweep finishes, and the FIRST one raised is
        re-raised afterwards — control flow preserved, and nothing left behind
        to preserve it.
        """
        deferred: BaseException | None = None
        for name, adapter in items:
            try:
                with guard(f"adapters.{name}.{where}"):
                    act(name, adapter)
            except BaseException as exc:  # noqa: BLE001 — deferred, never swallowed
                counters.bump(f"adapters.{name}.{where}_interrupted")
                if deferred is None:
                    deferred = exc
        if deferred is not None:
            raise deferred

    def is_installed(self, name: str) -> bool:
        return name in self._installed


_registry = AdapterRegistry()


def get_registry() -> AdapterRegistry:
    return _registry
