"""The one function the byte seams call when they latch a request's parent.

`latch_parent(parent, conversation)` takes what the issuing task's scope said and answers what the
request is parented to. With no bridge adapter installed it is the identity — two module-global
truthiness tests and a tuple, no import, no lock — and that is the case in almost every process.

With one installed it does two things a bridge needs and nothing in the scope can give it.

**A resolver may deepen the parent.** A framework that emits OpenTelemetry spans keeps its own
current span in OpenTelemetry's context, and that context can be re-attached on a task wardex's
carrier never reached. Measured on pydantic-ai's streaming run: the framework's `chat` span is
opened on a model-request task and stays open while the CONSUMER task sends the HTTP request; at
the send, OpenTelemetry's current span on the consumer is that `chat` span, while wardex's ambient
unit there is the enclosing agent run. Without help the wire CHAT lands under the run, beside the
bridge's chat unit instead of inside it. A bridge registers a resolver — "the live unit I hold for
the framework span current on THIS task" — and the request is parented to that unit instead.

The resolver is bounded so that no framework span can move a request out of the tree it was
issued in: the scope's parent must BE the ambient unit (not a host span or a remote header
standing on top of it), and the resolved unit must be live, in the same registry, and a strict
descendant of the ambient unit by the registry's own ancestor walk (`UnitRegistry._is_descendant`)
— a framework id never becomes a parent. Where wardex had no ambient unit, nothing is invented.

The edge keeps the seam's ordinary strategy, `CONTEXTVAR` at 1.0: that is literally how the parent
was found, a context variable read in-task at issue (OpenTelemetry's rather than wardex's). No
`LinkReason` fits — all of them are causality between distinct spans, and this is containment —
so the audit trail is the counter `assembly._bridge_latch.parent_resolved`.

**The witness.** Whatever parent comes out is told to `_usage_hold.witness_request`, so a bridge
CHAT unit knows at its close whether a wire request was issued under it. It sits HERE, after the
resolver, because the wire span lands under exactly this parent: a unit whose usage stayed demoted
with no wire span under it, or the reverse, is the one inconsistency the pair exists to remove.

The core never names OpenTelemetry: the bridge's resolver reads it, and an adapter that could not
import it could not have installed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ._diag import counters, guard
from ._units import Unit, _ambient_unit
from ._usage_hold import witness_request

if TYPE_CHECKING:
    from .._types import ConversationContext, SpanContext

#: `(owner, resolver)` pairs. A TUPLE, replaced whole on every change: readers on the request path
#: never lock, and they see either the old set or the new one. Writers are adapter install and
#: uninstall, which the adapter registry already serializes. Survives a fork on purpose — the
#: adapters stay installed in the child, and their per-span slots (what a resolver reads) are reset
#: by the adapter registry's own fork path.
_RESOLVERS: tuple[tuple[object, Callable[[], Unit | None]], ...] = ()


def set_span_resolver(owner: object, resolver: Callable[[], Unit | None] | None) -> None:
    """Register (or, with None, remove) `owner`'s resolver. Idempotent; one resolver per owner.

    A resolver is a zero-argument callable answering "the live unit I hold whose framework span is
    current on THIS task", or None. It is called on the issuing task, inside the request path, so
    it must be cheap and must not issue requests itself.
    """
    global _RESOLVERS
    kept = tuple(pair for pair in _RESOLVERS if pair[0] is not owner)
    _RESOLVERS = (*kept, (owner, resolver)) if resolver is not None else kept


def latch_parent(
    parent: SpanContext | None,
    conversation: ConversationContext | None,
    *,
    witnessed: bool = True,
) -> tuple[SpanContext | None, ConversationContext | None]:
    """The parent and conversation a request issued from this scope is latched with.

    Call it exactly where the scope is read for a request, on the task that issues it.
    `witnessed=False` is for a caller that does not yet know whether the parent is the request's
    issuer (the h2 writer, which flushes other tasks' frames): it witnesses per stream itself.
    """
    if _RESOLVERS:
        parent, conversation = _resolved(parent, conversation)
    if witnessed:
        witness_request(parent)
    return parent, conversation


def _resolved(
    parent: SpanContext | None, conversation: ConversationContext | None
) -> tuple[SpanContext | None, ConversationContext | None]:
    """A registered resolver's unit, when it is a live descendant of the ambient unit; else the
    inputs. A resolver that raises costs the deepening, never the request."""
    with guard("assembly._bridge_latch.resolve"):
        entry = _ambient_unit.get()
        if entry is None or parent is None:
            return parent, conversation
        ambient = entry.unit
        if not ambient.is_live or parent != ambient.context:
            return parent, conversation
        for _owner, resolve in _RESOLVERS:
            unit = resolve()
            if (
                unit is None
                or unit is ambient
                or not unit.is_live
                or unit._registry is not ambient._registry
                or not ambient._registry._is_descendant(unit, ambient.context)
            ):
                continue
            counters.bump("assembly._bridge_latch.parent_resolved")
            return unit.context, unit.conversation or conversation
    return parent, conversation


__all__ = ["latch_parent", "set_span_resolver"]
