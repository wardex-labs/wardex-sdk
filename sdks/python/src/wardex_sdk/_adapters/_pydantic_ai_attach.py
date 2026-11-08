"""How the pydantic-ai adapter reaches pydantic-ai's spans, keeps reaching them, and lets go.

The adapter module holds the normalizer (`_Bridge`); this module holds everything about WHERE the
spans come from, in three moments:

* Install. `Agent._instrument_default` is read and one route taken (`attach_target` in
  `_pydantic_ai_otel.py` decides it): own the instrumentation (Mode W: the attribute is patched
  to wardex's settings), attach a processor to the host's provider through its processor tuple
  (patched and restored by identity) or through its public `add_span_processor` (no remove, so
  it runs last), or find nothing to attach to (`unreachable`, declined by the adapter).
* Follow. For the install's lifetime `Agent.instrument_all` is wrapped (through the PatchSet,
  restored by identity). The host's call runs FIRST and is never altered: what it set is what
  pydantic uses, and stays so after `wardex.close()`. Then the value it set is read with the same
  route logic, and wardex's processor is attached to the provider behind it, once per provider:
  a `logfire.instrument_pydantic_ai()` made after `wardex.init()` (it calls
  `Agent.instrument_all(settings)`) no longer blinds the adapter. `instrument_all(True)` with the
  API's default provider keeps wardex's own settings, as install would. Turning instrumentation
  off, or onto a setup with no processor seam, is said once; there is nothing to attach to.
* Hand back. The PatchSet restores every patch it can by identity (a superseded one is the
  host's and is left alone); a processor that cannot be taken back (public route, or a tuple the
  host rebuilt) stays registered and inert, said once. A replacement the wrapper could NOT see is
  said once here: the host assigned `Agent._instrument_default` directly, or called an
  `instrument_all` reference it took before `init()`.

pydantic-ai and OpenTelemetry are imported inside functions only, like the adapter itself:
building the adapter object imports nothing of the host's.
"""

from __future__ import annotations

import functools
import inspect
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._assembly import report_once

if TYPE_CHECKING:
    from ._context import AdapterContext
    from ._pydantic_ai_otel import Bridge

#: Routes a processor goes in by; the others attach nothing.
_ATTACHING = ("tuple", "public")


def _version_of(current: object) -> object:
    """The instrumentation schema version the settings `current` emit at."""
    if current is True:
        from pydantic_ai.models.instrumented import InstrumentationSettings

        # Reading the signature builds nothing, so the global meter provider is untouched.
        return inspect.signature(InstrumentationSettings.__init__).parameters["version"].default
    return getattr(current, "version", 0)


def _route(current: object) -> tuple[str, Any]:
    """`(route, target)` for a value of `Agent._instrument_default`.

    The one global READ wardex makes is here, and only for `True`, which IS "use the global
    provider"; it is read each time, so a follow sees the provider as it is then.
    """
    import opentelemetry.trace as _otel_trace
    from pydantic_ai.models.instrumented import InstrumentationSettings

    from . import _pydantic_ai_otel as _otel

    if isinstance(current, InstrumentationSettings) or current is True:
        known = _otel_trace.get_tracer_provider() if current is True else None
        return _otel.attach_target(current, known)
    if current:
        return "unreachable", None
    return "not_enabled", None


class Attachment:
    """The adapter's hold on pydantic-ai's instrumentation, from install to hand-back.

    State is tuples rebound whole, never mutated in place: the records cross a fork with the
    patches they describe (the child's teardown needs them), and only the lock is reset.
    """

    def __init__(
        self,
        bridge: Bridge,
        is_installed: Callable[[], bool],
        on_schema: Callable[[AdapterContext, object], None],
    ) -> None:
        self._bridge = bridge
        self._is_installed = is_installed
        self._on_schema = on_schema
        self._ctx: AdapterContext | None = None
        self._agent: Any = None
        #: Serializes a follow against another follow and against `hand_back()`: check-then-attach
        #: must be one step, or two host threads could attach to one provider twice, or one could
        #: attach after the restore. Reentrant because `hand_back()` runs on close paths that a
        #: finalizer may reach on a thread already inside a follow.
        self._lock = threading.RLock()
        #: The one processor every attached provider receives; built on the first attach.
        self._processor: object | None = None
        #: `(route, target)` for every target the processor went to, in order.
        self._targets: tuple[tuple[str, Any], ...] = ()
        #: An install-time public target, added by `commit()` as install's last step.
        self._deferred: Any = None
        #: Mode W: wardex's own settings object.
        self._ours: object = None
        #: The value of `Agent._instrument_default` wardex last read or set. Hand-back compares
        #: it with the value in place to find a replacement the wrapper never saw.
        self._known: object = None

    def _at_fork_reinit(self) -> None:
        """Fork-child reset: a fresh lock, and nothing else (the records stay, I-fork-4)."""
        self._lock = threading.RLock()

    # -- install -------------------------------------------------------------

    def install(self, ctx: AdapterContext) -> tuple[str, object]:
        """Take the route the current value calls for: `(route, schema version)`.

        `unreachable` attached nothing (the adapter declines); `refused` is wardex's own Mode W
        patch refused by the PatchSet (counted there). Everything done here is undone by
        `hand_back()`; the one step with no undo, a public `add_span_processor`, waits for
        `commit()`.
        """
        from pydantic_ai import Agent
        from pydantic_ai.models.instrumented import InstrumentationSettings

        from . import _pydantic_ai_otel as _otel

        self._ctx, self._agent = ctx, Agent
        # An OWN class attribute: `instrument_all` is the one-liner that assigns it.
        current = vars(Agent).get("_instrument_default", False)
        self._known = current
        route, target = _route(current)
        if route == "unreachable":
            return route, 0
        if route == "not_enabled":
            ours = InstrumentationSettings(
                tracer_provider=_otel.Provider(self._bridge, self._is_installed),
                meter_provider=_otel.meter_provider(),
            )
            # The PatchSet form of `Agent.instrument_all(ours)`: restored by identity, and a
            # later host `instrument_all(x)` supersedes it and is left alone at hand-back.
            if not ctx.patches.patch(Agent, "_instrument_default", ours):
                return "refused", 0
            self._ours = self._known = ours
            return route, ours.version
        if route == "tuple" and self._attach_tuple(ctx, target):
            return route, _version_of(current)
        self._deferred = target  # a tuple wardex may not replace: the same object's public method
        return "public", _version_of(current)

    def commit(self) -> None:
        """Install's last steps: wrap `Agent.instrument_all`, then the public add (no undo)."""
        ctx, agent = self._ctx, self._agent
        if ctx is None or agent is None:
            return
        original = agent.instrument_all  # the plain function behind the staticmethod
        follow = self._follow

        @functools.wraps(original)
        def instrument_all(instrument: Any = True) -> Any:
            result = original(instrument)  # the host's call, first and unaltered
            follow()
            return result

        ctx.patches.patch(agent, "instrument_all", staticmethod(instrument_all))
        deferred, self._deferred = self._deferred, None
        if deferred is not None:
            self._attach_public(deferred)

    # -- follow --------------------------------------------------------------

    def _follow(self) -> None:
        """After a host `instrument_all(x)`: attach to x's provider, or say why not.

        Runs under the adapter's guard: a wardex fault here is counted, never raised into the
        host's `instrument_all` call, which has already done its work. `_known` moves to the new
        value only once it is followed or reported, so a follow that faults midway is said at
        hand-back as a replacement wardex could not follow. The install check is repeated under
        the lock, which `hand_back()` also holds: a follow racing `wardex.close()` either finishes
        before the restore or sees the uninstall and attaches nothing.
        """
        ctx, agent = self._ctx, self._agent
        if ctx is None or agent is None or not self._is_installed():
            return
        with ctx.guard("follow"), self._lock:
            if not self._is_installed():
                return
            current = vars(agent).get("_instrument_default", False)
            if current is self._known:
                return
            if current is self._ours:
                self._known = current
                return
            if not current:
                report_once(
                    "pydantic_ai adapter: pydantic-ai's instrumentation was turned off after "
                    "wardex.init(); the adapter records nothing of pydantic-ai's runs from then "
                    "on; wire-level model calls still are",
                    key="adapters.pydantic_ai.instrumentation_off",
                )
                ctx.count("instrumentation_off")
                self._known = current
                return
            route, target = _route(current)
            if route == "not_enabled" and self._ours is not None:
                # `True` with the API's default provider: pydantic's spans would go nowhere, as at
                # install, so wardex's own settings stay in place. The newer PatchSet record
                # restores the host's value at hand-back; the older one is superseded.
                if ctx.patches.patch(agent, "_instrument_default", self._ours):
                    self._known = self._ours
                    return
            if route not in _ATTACHING:
                report_once(
                    "pydantic_ai adapter: pydantic-ai's instrumentation was replaced after "
                    "wardex.init() with a setup that accepts no span processor wardex can attach "
                    "to; the adapter records nothing of pydantic-ai's runs from then on; "
                    "wire-level model calls still are",
                    key="adapters.pydantic_ai.instrumentation_not_followed",
                )
                ctx.count("instrumentation_not_followed")
                self._known = current
                return
            self._on_schema(ctx, _version_of(current))
            if self._attached(target):
                self._known = current
                return
            if not (route == "tuple" and self._attach_tuple(ctx, target)):
                self._attach_public(target)
            self._known = current
            report_once(
                "pydantic_ai adapter: pydantic-ai's instrumentation was replaced after "
                "wardex.init(); wardex attached to the new setup and keeps recording",
                key="adapters.pydantic_ai.instrumentation_followed",
            )
            ctx.count("instrumentation_followed")

    # -- attaching ------------------------------------------------------------

    def _processor_for(self) -> object:
        if self._processor is None:
            from . import _pydantic_ai_otel as _otel

            self._processor = _otel.Processor(self._bridge, self._is_installed)
        return self._processor

    def _attached(self, target: Any) -> bool:
        """Whether the processor already went to `target`: one provider is attached once."""
        if any(seen is target for _, seen in self._targets):
            return True
        processor = self._processor
        return processor is not None and processor in getattr(target, "_span_processors", ())

    def _attach_tuple(self, ctx: AdapterContext, target: Any) -> bool:
        processor = self._processor_for()
        processors = target._span_processors
        if not ctx.patches.patch(target, "_span_processors", (*processors, processor)):
            return False
        self._targets += (("tuple", target),)
        return True

    def _attach_public(self, target: Any) -> None:
        target.add_span_processor(self._processor_for())
        self._targets += (("public", target),)

    # -- hand back -------------------------------------------------------------

    def hand_back(self) -> None:
        """Restore what can be restored by identity; say once what stays behind.

        Holds the follow lock, so a host `instrument_all` racing `wardex.close()` cannot attach
        after the restore.
        """
        ctx = self._ctx
        if ctx is None:
            return
        with self._lock:
            self._hand_back(ctx)

    def _hand_back(self, ctx: AdapterContext) -> None:
        agent = self._agent
        if agent is not None and vars(agent).get("_instrument_default", False) is not self._known:
            report_once(
                "pydantic_ai adapter: pydantic-ai's instrumentation was replaced after "
                "wardex.init() without `Agent.instrument_all` (by assigning "
                "`Agent._instrument_default`, or through an `instrument_all` taken before "
                "init()), so wardex could not follow it and may have recorded none of "
                "pydantic-ai's runs from then on; call `Agent.instrument_all` instead",
                key="adapters.pydantic_ai.instrumentation_replaced",
            )
            ctx.count("instrumentation_replaced")
        ctx.patches.restore_all()
        processor = self._processor
        targets, self._targets = self._targets, ()
        self._ours = self._deferred = None
        # A public target has no remove; a tuple the host rebuilt meanwhile keeps wardex's in it.
        inert = [
            target
            for route, target in targets
            if route == "public" or processor in getattr(target, "_span_processors", ())
        ]
        if inert:
            report_once(
                "pydantic_ai adapter: wardex could not take its span processor back from the "
                "OpenTelemetry provider it attached to (the provider has no remove, or the host "
                "added a processor after wardex.init()); it stays registered but inert for the "
                "life of this process",
                key="adapters.pydantic_ai.processor_left_inert",
            )
        for _ in inert:
            ctx.count("processor_left_inert")


__all__ = ["Attachment"]
