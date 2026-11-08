"""The verbs a BRIDGE adapter needs from its context, kept beside `AdapterContext`.

A bridge receives a framework's own spans (OpenTelemetry, for every bridge so far) and turns them
into units. Two things it needs cannot be built from the verbs every adapter has:

* **Following the framework's current span.** The framework may re-attach its span on a task
  wardex's carrier never reached — measured on pydantic-ai's streaming run, where the request is
  sent from the consumer task while wardex's ambient there is the enclosing run.
  `follow_current_span` lets the request latch parent that request to the unit this context holds
  for the span (`_assembly/_bridge_latch.py` bounds it to live descendants of the ambient unit).
* **Following tool calls by name.** A byte seam may see a framework's tool call on a task the
  tool never ran on: pydantic-ai's MCP client writes the server's stdin from a task it spawned
  when the toolset connected. `follow_tool_calls` lets the seam parent the wire `tools/call` to
  the one live tool unit this context holds for that tool name and those arguments
  (`latch_tool_call`).
* **Holding framework usage.** A bridge CHAT unit carries the framework's usage demoted, and the
  core promotes it at close only when no model-call transaction was sealed under the unit
  (`_assembly/_usage_hold.py`).

A mixin, not methods on `AdapterContext` itself, because that module is at its line budget; and so
the bridge verbs read as one set. `__slots__ = ()` because `AdapterContext` declares slots and a
base without them would give every context a `__dict__`.

The core never names OpenTelemetry: the adapter hands `follow_current_span` the framework's own
"current span" reader, so an adapter that could not import it never reaches here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._assembly import Unit, set_span_resolver, set_tool_call_resolver
from .._assembly import hold_framework_usage as _hold

if TYPE_CHECKING:
    from ._context import RunHandle


class BridgeVerbs:
    """Mixed into `AdapterContext`. Reads `name`, `peek`, `slots_snapshot` and `count` from it."""

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

    def follow_tool_calls(
        self, call_of: Callable[[dict[str, Any]], tuple[str, str | None] | None]
    ) -> None:
        """Let a wire tool call be parented to the one live tool unit held for its name and args.

        `call_of` reads a slot and answers `(tool name, canonical arguments or None)` for the call
        its unit was opened for, or None for a slot that is not a tool's. The slots are walked the
        way the uninstall sweep walks them (`slots_snapshot`). A unit whose recorded arguments
        differ from the wire's is no candidate. Among the rest, the unit under `"handle"` is
        answered only when exactly ONE live, undegraded candidate is left: first among those whose
        arguments equal the wire's (proved), else among those that recorded none (by name only).
        Two concurrent calls of one tool with the same arguments cannot be told apart, so the
        answer is None and the ambiguity is counted.
        """

        def resolve(name: str, arguments: str | None) -> tuple[Unit, bool] | None:
            proved: list[Unit] = []
            named: list[Unit] = []
            for entry in self.slots_snapshot():
                handle = entry.get("handle")
                if handle is None or handle.degraded or not handle._unit.is_live:
                    continue
                call = call_of(entry)
                if call is None or call[0] != name:
                    continue
                if call[1] is None:
                    named.append(handle._unit)
                elif arguments is not None and call[1] == arguments:
                    proved.append(handle._unit)
            for found, by_arguments in ((proved, True), (named, False)):
                if len(found) > 1:
                    self.count("tool_call_ambiguous")
                    return None
                if found:
                    return found[0], by_arguments
            return None

        set_tool_call_resolver(self, resolve)

    def unfollow_tool_calls(self) -> None:
        """Undo `follow_tool_calls`. Idempotent: safe before install and after uninstall."""
        set_tool_call_resolver(self, None)

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
