"""The three verbs a BRIDGE adapter needs from its context, kept beside `AdapterContext`.

A bridge receives a framework's own spans (OpenTelemetry, for every bridge so far) and turns them
into units. Two things it needs cannot be built from the verbs every adapter has:

* **Following the framework's current span.** The framework may re-attach its span on a task
  wardex's carrier never reached — measured on pydantic-ai's streaming run, where the request is
  sent from the consumer task while wardex's ambient there is the enclosing run.
  `follow_current_span` lets the request latch parent that request to the unit this context holds
  for the span (`_assembly/_bridge_latch.py` bounds it to live descendants of the ambient unit).
* **Holding framework usage.** A bridge CHAT unit carries the framework's usage demoted, and the
  core promotes it at close only when no wire request was issued under the unit
  (`_assembly/_usage_hold.py`).

A mixin, not methods on `AdapterContext` itself, because that module is at its line budget; and so
the bridge verbs read as one set. `__slots__ = ()` because `AdapterContext` declares slots and a
base without them would give every context a `__dict__`.

The core never names OpenTelemetry: the adapter hands `follow_current_span` the framework's own
"current span" reader, so an adapter that could not import it never reaches here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .._assembly import Unit, set_span_resolver
from .._assembly import hold_framework_usage as _hold

if TYPE_CHECKING:
    from ._context import RunHandle


class BridgeVerbs:
    """Mixed into `AdapterContext`. Reads `name`, `peek` and `count` from it."""

    __slots__ = ()

    def follow_current_span(self, current: Callable[[], object]) -> None:
        """Let wire requests issued while a framework span is current be parented to its unit.

        `current` is the framework's "current span" reader (for OpenTelemetry,
        `opentelemetry.trace.get_current_span`), called on the issuing task. The span is looked up
        in this context's slots under the key `"handle"` — the key the uninstall sweep reads too,
        so a unit the resolver can find is a unit the teardown closes.
        """

        def resolve() -> Unit | None:
            entry = self.peek(current())
            handle = entry.get("handle") if entry else None
            if handle is None or handle.degraded:
                return None
            return handle._unit

        set_span_resolver(self, resolve)

    def unfollow_current_span(self) -> None:
        """Undo `follow_current_span`. Idempotent: safe before install and after uninstall."""
        set_span_resolver(self, None)

    def hold_framework_usage(self, handle: RunHandle, *, prefix: str) -> bool:
        """Declare `handle`'s own CHAT unit a framework-reported usage holder.

        `prefix` must be in this adapter's own extras namespace (`wardex.<name>.`), which binds a
        bridge to promoting only the numbers it demoted itself. Refused (False, counted) for a
        degraded handle or a foreign prefix.
        """
        if handle.degraded or not prefix.startswith(f"wardex.{self.name}."):
            self.count("usage_hold_refused")
            return False
        return _hold(handle.draft, prefix=prefix)


__all__ = ["BridgeVerbs"]
