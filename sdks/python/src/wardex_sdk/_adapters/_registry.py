"""Manages adapter install state — idempotent install/uninstall."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from .._assembly import Limitation, UnitRegistry, counters, guard
from .._config import AdaptersConfig
from .._limits import LimitsConfig, LimitsConsumer, limits_kwargs
from ._base import AdapterInterface
from ._context import AdapterContext
from ._sink import _ClientSink

if TYPE_CHECKING:
    from .._client import Client


def _declared_control_flow(adapter: AdapterInterface) -> tuple[type[BaseException], ...]:
    """`CONTROL_FLOW` as of NOW — the CLASS attribute, read per exception.

    A function rather than a lambda at the call site so the indirection has a
    name and one definition: the whole point is that no caller can hand
    `AdapterContext` a tuple it copied before `install()` ran.
    """
    return type(adapter).CONTROL_FLOW


def context_for(
    name: str,
    client: Client | None,
    adapter: AdapterInterface | None = None,
) -> AdapterContext:
    """Build the surface an adapter is written against. THE one construction.

    A module function rather than a method because an adapter installed outside
    the registry — every test that drives one directly — must be able to reach
    the same construction. Two constructions would be two registries, and
    `owner` scoping answers questions about ONE table: with two, `sole_live` and
    `close_all` cannot tell one adapter's units from another's inside either.
    """
    config = getattr(client, "config", None)
    limits = config.limits if config is not None else LimitsConfig()
    resolved = limits.resolved()
    debug = bool(getattr(config, "debug", False))
    # The adapter's own options group, picked off `AdaptersConfig` by the
    # registration row. Imported lazily because the package's `__init__`
    # imports THIS module at its top, before the registration table exists;
    # by the time a context is built the package is whole. The isinstance
    # guard keeps every test double that carries a fake config honest: a
    # config whose `adapters` is not the real group has no options to give.
    options = None
    adapters_config = getattr(config, "adapters", None)
    if isinstance(adapters_config, AdaptersConfig):
        from . import _options_for

        options = _options_for(name, adapters_config)
    return AdapterContext(
        name,
        # The bounds are passed HERE and not left to the registry's defaults.
        # Once an adapter shares this registry, this is the only place a
        # user's `max_units` can reach it — a registry built with defaults
        # would ignore the setting in silence, which is the shape of bug
        # that looks like nothing at all until a workload crosses a cap the
        # user thought they had raised.
        #
        # Through the PROJECTION, not by hand. Spelling the keywords here is
        # how `max_body_bytes` came to be the one bound this call forgot: the
        # list of what a registry takes lived at the call site, so it could be
        # short and still look complete. `_LIMIT_DELIVERY` holds that list once
        # and `tests/test_limits_wiring.py` holds it against the signature.
        units=UnitRegistry(
            sink=_ClientSink(client),
            debug=debug,
            **limits_kwargs(LimitsConsumer.UNIT_REGISTRY, resolved),
        ),
        limits=resolved,
        debug=debug,
        options=options,
        # A READER over one classvar, not the adapter and not a tuple. `install`
        # builds this context BEFORE it calls `adapter.install()`, and an adapter
        # can only import its framework's error classes in there — so a tuple
        # taken here would be `()` for the life of the process, and the whole
        # mechanism would ship green and dead.
        control_flow=None if adapter is None else partial(_declared_control_flow, adapter),
    )


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
        ctx = context_for(name, client, adapter)
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

    def uninstall_all(self) -> None:
        """Uninstall every adapter. Total: one failure cannot stop the rest.

        Same rule as `InterceptorRegistry.uninstall_all`, and the same reason:
        this loop runs inside `Runtime._teardown`, immediately before
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

    def _at_fork_reinit(self) -> None:
        """Fork-child reset, delegated to every installed adapter.

        `InterceptorRegistry._at_fork_reinit`'s twin, same rules: the
        registry stays populated, every patch stays installed (I-fork-4),
        only per-process mutable state resets, and an adapter that declares
        no `_at_fork_reinit` is stating it holds none — a statement the
        coverage guard in `tests/test_fork_reinit_coverage.py` verifies for
        every holder, present and future. Per-adapter `guard()` so one
        failing reset cannot abandon the rest.
        """
        for name, adapter in list(self._installed.items()):
            reinit = getattr(adapter, "_at_fork_reinit", None)
            if reinit is None:
                continue
            with guard(f"adapters.{name}.fork_reinit_failed"):
                reinit()
        # Every installed adapter's context owns a PatchSet whose lock the
        # child's teardown WILL take (`AdapterRegistry.uninstall` ->
        # `ctx.patches.restore_all()`), whether or not the adapter itself
        # declared a reset — LangGraph patches exclusively through its
        # context and declares none. Row Q therefore lives here, on the
        # OWNER of the contexts, not on each adapter's goodwill.
        for name, ctx in list(self._contexts.items()):
            with guard(f"adapters.{name}.fork_reinit_failed"):
                ctx._at_fork_reinit()

    def is_installed(self, name: str) -> bool:
        return name in self._installed


def get_registry() -> AdapterRegistry:
    """The process registry, which `Runtime` owns and builds on first use.

    Not a module singleton of its own any more, for `InterceptorRegistry.
    get_registry`'s reason: a registry nobody owns cannot be drained by the one
    reset, so an adapter left installed by one test silently no-ops the next
    `install()` of the same name.
    """
    from .._runtime import runtime

    return runtime().adapters
