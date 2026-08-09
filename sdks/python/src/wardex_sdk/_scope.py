from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ._types import ConversationContext, SpanContext


@dataclass(frozen=True, slots=True)
class UserInfo:
    id: str | None = None
    email: str | None = None
    username: str | None = None
    ip_address: str | None = None


@dataclass
class Scope:
    tags: dict[str, str] = field(default_factory=dict)
    user: UserInfo | None = None
    contexts: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_span_context: SpanContext | None = None
    conversation: ConversationContext | None = None
    tracestate: str | None = None

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value

    def set_user(self, user: UserInfo) -> None:
        self.user = user

    def set_context(self, key: str, value: dict[str, Any]) -> None:
        self.contexts[key] = value

    def clone(self) -> Scope:
        return Scope(
            tags=dict(self.tags),
            user=self.user,
            contexts=copy.deepcopy(self.contexts),
            active_span_context=self.active_span_context,
            conversation=self.conversation,
            tracestate=self.tracestate,
        )


def merge_scopes(global_: Scope, isolation: Scope, current: Scope) -> Scope:
    """Overwrites layers in Global → Isolation → Current order."""
    merged = global_.clone()
    for layer in (isolation, current):
        merged.tags.update(layer.tags)
        merged.contexts.update(copy.deepcopy(layer.contexts))
        if layer.user is not None:
            merged.user = layer.user
        if layer.active_span_context is not None:
            merged.active_span_context = layer.active_span_context
        if layer.conversation is not None:
            merged.conversation = layer.conversation
        if layer.tracestate is not None:
            merged.tracestate = layer.tracestate
    return merged


def merged_trace_fields(
    global_: Scope, isolation: Scope, current: Scope
) -> tuple[SpanContext | None, str | None]:
    """The two propagation fields `merge_scopes` would produce, and nothing else.

    Same layers, same last-non-None-wins precedence, deliberately adjacent to
    `merge_scopes` so that a change to the rule cannot be made in one of them
    and missed in the other. That adjacency is the whole reason this is not a
    hand-rolled walk somewhere in `context/`.

    Separate from `merge_scopes` because the W3C header readers need exactly
    these two immutable scalars, and the full merge cannot hand them over
    cheaply OR safely. It `copy.deepcopy`s `contexts` once per layer, over
    dicts the host application filled with objects of its own choosing: that is
    unbounded work on the outbound path of every request through a patched HTTP
    client, and it is host code. `deepcopy` raises on a value it cannot copy —
    a lock, a socket, a file — and runs whatever `__deepcopy__`/`__reduce__` the
    host defined, and it iterates the process-global scope's dicts, which
    another thread's `set_tag`/`set_context` can be mutating. `get_traceparent`
    is documented as the by-hand escape hatch for gRPC, Kafka and Celery send
    paths; a host that once put a lock in `set_context()` must not discover it
    there. Reading two scalars can do none of that.
    """
    active = global_.active_span_context
    tracestate = global_.tracestate
    for layer in (isolation, current):
        if layer.active_span_context is not None:
            active = layer.active_span_context
        if layer.tracestate is not None:
            tracestate = layer.tracestate
    return active, tracestate
