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
    return merged
