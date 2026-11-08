"""The one function the byte seams call when they latch a request's parent.

`latch_parent(parent, conversation)` takes what the issuing task's scope said and answers what the
request is parented to. With no bridge adapter installed it is the identity — one module-global
truthiness test and a tuple, no import, no lock — and that is the case in almost every process.

With one installed it does the one thing a bridge needs that nothing in the scope can give it.

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

**Not the witness.** The parent latched here travels on the transaction, and the seam's seal hands
it to `_usage_hold.witness_request` only when the endpoint table reads the path as a model call. A
request issued here is not yet evidence that any wire span will carry its usage.

**A tool-call resolver may deepen a byte seam's ambient.** The MCP stdio seam latches a
`tools/call` on the task that WRITES the server's stdin, and with pydantic-ai's MCP client that is
a task the client spawned when the toolset connected, not the tool's task. The framework's own
`execute_tool` span is open on the tool's task, which the writer never sees. A bridge registers a
second kind of resolver, "the one live tool unit I hold for a tool of this name and these
arguments", and `latch_tool_call` parents the wire `tools/call` to it.

The resolver answers with the unit AND whether the arguments proved the match: the wire request's
arguments, canonicalized (`canonical_arguments`), equal the arguments the framework recorded on the
unit at START. A unit whose recorded arguments differ is no candidate at all, so a tool of the same
name called by an unrelated client, or a concurrent call of the same tool with other arguments, is
never nested under it. A unit with no recorded arguments (the framework was told not to record
content) can match by name only. Bounds, mirroring `_resolved`: the unit is live and of
`EXECUTE_TOOL` intent, and then

* a match by name only must lie strictly below the writer's ambient unit, in its registry. When the
  ambient names no live unit, or the unit is elsewhere, the name is too weak to move the call: it
  keeps the seam's own placement.
* a match proved by the arguments may lie anywhere in the ambient unit's registry, or anywhere at
  all when the ambient names no unit. The writer task predates the run, or was spawned by another
  run sharing the same toolset, so its ambient says nothing about which run the call belongs to.

A candidate refused by those bounds is counted under `assembly._bridge_latch.tool_call_unproved`.
An unrelated client sending a tool of the same name with exactly the same arguments while that
unit is live is indistinguishable from the framework's own call and is nested under it. The audit
trail is the counter `assembly._bridge_latch.tool_call_resolved`.

The core never names OpenTelemetry: the bridge's resolver reads it, and an adapter that could not
import it could not have installed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from ._diag import counters, guard
from ._parentage import Ambient
from ._units import Unit, _ambient_unit
from ._vocab import SpanIntent

if TYPE_CHECKING:
    from .._types import ConversationContext, SpanContext

#: `(owner, resolver)` pairs. A TUPLE, replaced whole on every change: readers on the request path
#: never lock, and they see either the old set or the new one. Writers are adapter install and
#: uninstall, which the adapter registry already serializes. Survives a fork on purpose — the
#: adapters stay installed in the child, and their per-span slots (what a resolver reads) are reset
#: by the adapter registry's own fork path.
_RESOLVERS: tuple[tuple[object, Callable[[], Unit | None]], ...] = ()

#: A tool-call resolver: `(name, canonical arguments or None)` -> `(unit, proved by arguments)`.
ToolCallResolver = Callable[[str, "str | None"], "tuple[Unit, bool] | None"]

#: `(owner, resolver)` pairs answering a tool call with a unit. Same copy-on-write discipline and
#: the same fork reasoning as `_RESOLVERS`.
_TOOL_RESOLVERS: tuple[tuple[object, ToolCallResolver], ...] = ()


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
    parent: SpanContext | None, conversation: ConversationContext | None
) -> tuple[SpanContext | None, ConversationContext | None]:
    """The parent and conversation a request issued from this scope is latched with.

    Call it exactly where the scope is read for a request, on the task that issues it.
    """
    if _RESOLVERS:
        parent, conversation = _resolved(parent, conversation)
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


def canonical_arguments(value: object) -> str | None:
    """One spelling for a tool call's arguments, so the framework's and the wire's compare equal.

    `value` is the decoded arguments object, or a JSON text of it (how a framework records them on
    its span). Answers None, counted, when it is neither, which leaves a match by name only.
    """
    text = None
    with guard("assembly._bridge_latch.canonical_arguments"):
        decoded = json.loads(value) if isinstance(value, (str, bytes, bytearray)) else value
        text = json.dumps(decoded, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text


def set_tool_call_resolver(owner: object, resolver: ToolCallResolver | None) -> None:
    """Register (or, with None, remove) `owner`'s tool-call resolver. Idempotent; one per owner.

    A resolver answers "the one live tool unit I hold for a call of `name` with these canonical
    arguments", with whether the arguments proved it, or None when it holds none or more than one.
    It is called on the task that writes the request, so it must be cheap and must not issue
    requests itself.
    """
    global _TOOL_RESOLVERS
    kept = tuple(pair for pair in _TOOL_RESOLVERS if pair[0] is not owner)
    _TOOL_RESOLVERS = (*kept, (owner, resolver)) if resolver is not None else kept


def latch_tool_call(ambient: Ambient, name: str, arguments: object = None) -> tuple[Ambient, bool]:
    """The ambient a wire tool call named `name` is latched with, and whether it was deepened.

    `arguments` is the request's decoded arguments object (an MCP `tools/call` without one sends
    `{}`). Call it on the task that writes the request, right after `latch_ambient()`. With no
    tool-call resolver registered it is the identity.
    """
    if not _TOOL_RESOLVERS:
        return ambient, False
    unit = None
    with guard("assembly._bridge_latch.resolve_tool_call"):
        unit = _tool_unit(ambient, name, canonical_arguments(arguments))
    if unit is None:
        return ambient, False
    counters.bump("assembly._bridge_latch.tool_call_resolved")
    deepened = Ambient(
        span_context=unit.context,
        conversation=unit.conversation or ambient.conversation,
        tracestate=ambient.tracestate,
    )
    return deepened, True


def _tool_unit(ambient: Ambient, name: str, arguments: str | None) -> Unit | None:
    """The first registered resolver's answer that passes the bounds, else None."""
    entry = _ambient_unit.get()
    anchor = (
        entry.unit
        if entry is not None
        and entry.unit.is_live
        and ambient.span_context is not None
        and ambient.span_context == entry.unit.context
        else None
    )
    for _owner, resolve in _TOOL_RESOLVERS:
        answer = resolve(name, arguments)
        if answer is None:
            continue
        unit, proved = answer
        if (
            unit is anchor
            or not unit.is_live
            or getattr(unit.draft, "_intent", None) is not SpanIntent.EXECUTE_TOOL
        ):
            continue
        if anchor is None:
            inside = proved
        elif unit._registry is not anchor._registry:
            inside = False
        else:
            inside = proved or anchor._registry._is_descendant(unit, anchor.context)
        if not inside:
            counters.bump("assembly._bridge_latch.tool_call_unproved")
            continue
        return unit
    return None


__all__ = [
    "canonical_arguments",
    "latch_parent",
    "latch_tool_call",
    "set_span_resolver",
    "set_tool_call_resolver",
]
