from __future__ import annotations

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

    def set_user(self, user: UserInfo | None) -> None:
        """Attach `user` to this scope; `None` clears it again."""
        self.user = user

    def set_context(self, key: str, value: dict[str, Any]) -> None:
        self.contexts[key] = value

    def clone(self) -> Scope:
        # Per-context dict copies, never `deepcopy`: `set_context()` takes
        # arbitrary host objects — a lock, a socket, an open file — and a fork
        # (`isolation_scope()`, `new_scope()`) must not run host
        # `__deepcopy__`/`__reduce__` or raise on a value that cannot be
        # copied. Copying each context dict is what the public surface needs:
        # `set_context()` replaces whole entries, so mutations on the fork
        # stay on the fork; the values themselves stay shared with the host.
        return Scope(
            tags=dict(self.tags),
            user=self.user,
            contexts={key: dict(value) for key, value in self.contexts.items()},
            active_span_context=self.active_span_context,
            conversation=self.conversation,
            tracestate=self.tracestate,
        )


def merge_scopes(global_: Scope, isolation: Scope, current: Scope) -> Scope:
    """Overwrites layers in Global → Isolation → Current order."""
    merged = global_.clone()
    for layer in (isolation, current):
        merged.tags.update(layer.tags)
        merged.contexts.update((key, dict(value)) for key, value in layer.contexts.items())
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
    cheaply. It copies every context dict once per layer — shallow copies, so
    no host `__deepcopy__` runs and nothing raises on a lock (see
    `Scope.clone`), but still per-key work over dicts the host application
    filled, on the outbound path of every request through a patched HTTP
    client, iterating the process-global scope's dicts, which another thread's
    `set_tag`/`set_context` can be mutating. `get_traceparent` is documented
    as the by-hand escape hatch for gRPC, Kafka and Celery send paths. Reading
    two scalars does none of that.
    """
    active = global_.active_span_context
    tracestate = global_.tracestate
    for layer in (isolation, current):
        if layer.active_span_context is not None:
            active = layer.active_span_context
        if layer.tracestate is not None:
            tracestate = layer.tracestate
    return active, tracestate


def merged_tags_and_user(
    global_: Scope, isolation: Scope, current: Scope
) -> tuple[dict[str, str], UserInfo | None]:
    """The tags and user `merge_scopes` would produce, and nothing else.

    Same layers, same precedence — tags dict-merged Global → Isolation →
    Current with later layers overriding keys, user last-non-None — and
    deliberately adjacent to `merge_scopes` so the rule cannot be changed in
    one of them and missed in the other, exactly like `merged_trace_fields`
    above.

    Separate from `merge_scopes` for `merged_trace_fields`' reason: the full
    merge copies every context dict, dicts the host filled with objects of
    its own choosing, and this reader runs on the capture path of every span.
    Tags are `str -> str` and `UserInfo` is frozen, so copying the tag dict is
    bounded and nothing here touches the contexts at all.
    """
    tags = dict(global_.tags)
    user = global_.user
    for layer in (isolation, current):
        tags.update(layer.tags)
        if layer.user is not None:
            user = layer.user
    return tags, user
