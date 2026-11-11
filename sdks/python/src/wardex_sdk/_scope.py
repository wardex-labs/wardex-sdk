from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any
from uuid import UUID

from ._types import ConversationContext, SpanContext
from ._values import AttributeValue, attribute_value, wire_text

#: The tag key `wardex.set_account()` writes: the customer's own customer, the
#: unit a dashboard compares behaviour across.
ACCOUNT_KEY = "wardex.account.id"
#: The prefix `set_context(name, ...)` stamps each scalar under, as
#: `wardex.context.<name>.<key>`.
CONTEXT_PREFIX = "wardex.context."
#: The most keys one `set_context` payload stamps onto a span; the rest stay on
#: the scope only. A payload is host data of any size, and every key becomes an
#: attribute on every span under the scope.
MAX_CONTEXT_KEYS = 32
#: The values a context key may hold and still be stamped: what an OTLP
#: attribute can carry without the SDK choosing a rendering for it.
_STAMPABLE = (str, int, float, bool)


@dataclass(frozen=True, slots=True)
class UserInfo:
    id: str | None = None
    email: str | None = None
    username: str | None = None
    ip_address: str | None = None

    def __post_init__(self) -> None:
        # Raised here, at the developer's own call: a dict or an object in one of
        # these fields would otherwise surface only at the span stamp, far away.
        # An int id (a database primary key) is a faithful id and is kept as its
        # decimal string, so `user.id` has one type on the wire whichever form
        # the host passed. An empty string is "no value", like None, so a field
        # forwarded from an anonymous user never fails the host's request.
        for spec in fields(self):
            value = getattr(self, spec.name)
            if spec.name == "id" and type(value) is int:
                object.__setattr__(self, "id", str(value))
            elif value == "":
                object.__setattr__(self, spec.name, None)
            elif isinstance(value, str):
                object.__setattr__(self, spec.name, wire_text(value))
            elif value is not None:
                allowed = "a str, an int or None" if spec.name == "id" else "a str or None"
                raise TypeError(
                    f"UserInfo.{spec.name} must be {allowed}, got {type(value).__name__}"
                )


#: `set_user()`'s "no positional argument" default, told apart from `None` (clear the user).
NO_USER: Any = object()


def user_from_call(user: object, keywords: dict[str, object]) -> UserInfo | None:
    """What `wardex.set_user(...)` was asked to attach, or a TypeError saying what it accepts.

    Accepted: a `UserInfo`, a user id (`str` or `int`), `None` (clear the user),
    or keyword fields alone (`id=`, `email=`, `username=`, `ip_address=`).
    `user` is `NO_USER` when the caller passed no positional argument. Keyword
    fields that are all None or empty (an anonymous user) clear the user, as
    an empty id does: a value forwarded from the host's data must not decide
    whether the host's request fails.
    """
    given = {name: value for name, value in keywords.items() if value is not None}
    if user is NO_USER:
        found = UserInfo(**given)  # type: ignore[arg-type]  # checked by UserInfo itself
        return None if found == UserInfo() else found
    if given:
        raise TypeError(f"{USER_FORMS}; pass either the positional form or keywords, not both")
    return checked_user(user)


#: What `set_user` accepts, for the TypeError a wrong call raises.
USER_FORMS = (
    "set_user() takes a UserInfo, a user id (str, int or uuid.UUID), None to clear the user, "
    "or keyword "
    "fields: set_user(id=..., email=..., username=..., ip_address=...)"
)


def checked_user(user: object) -> UserInfo | None:
    """`user` as a `UserInfo` (or None), from one positional value; TypeError for anything else."""
    if user is None or isinstance(user, UserInfo):
        return None if user == UserInfo() else user
    if isinstance(user, str) or type(user) is int:
        return None if user == "" else UserInfo(id=user)  # type: ignore[arg-type]
    if isinstance(user, UUID):
        return UserInfo(id=str(user))
    raise TypeError(f"{USER_FORMS}; got {type(user).__name__}")


def context_entries(name: object, value: object) -> tuple[dict[str, Any], list[str], int]:
    """`(stamped tags, skipped keys, keys over the cap)` for `set_context(name, value)`.

    A TypeError for a name that is not a non-empty str or a value that is not a
    mapping: the developer's own call, so it fails there. Only str/int/float/bool
    values are stamped; `None` is "no value" and is neither stamped nor reported.
    """
    if not isinstance(name, str) or not name:
        raise TypeError(f"set_context() takes a non-empty str name, got {name!r}")
    if not isinstance(value, Mapping):
        raise TypeError(
            f"set_context({name!r}, ...) takes a mapping of keys to values, "
            f"got {type(value).__name__}"
        )
    stamped: dict[str, Any] = {}
    skipped: list[str] = []
    over = 0
    for key, item in value.items():
        if item is None:
            continue
        if not isinstance(key, str) or not isinstance(item, _STAMPABLE):
            skipped.append(str(key))
        elif len(stamped) < MAX_CONTEXT_KEYS:
            stamped[f"{CONTEXT_PREFIX}{name}.{key}"] = (
                wire_text(item) if isinstance(item, str) else item
            )
        else:
            over += 1
    return stamped, skipped, over


@dataclass
class Scope:
    tags: dict[str, AttributeValue] = field(default_factory=dict)
    user: UserInfo | None = None
    contexts: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_span_context: SpanContext | None = None
    conversation: ConversationContext | None = None
    tracestate: str | None = None
    #: Set when the block that installed this scope ended in another Context
    #: than it began in (`_hub.retire`): the scope to read in its place, and
    #: the marker of the Context that installed it, the one that reads it so.
    #: Neither is a field a fork inherits.
    _retired_to: Scope | None = field(default=None, init=False, repr=False, compare=False)
    _retired_by: object = field(default=None, init=False, repr=False, compare=False)
    #: For a copy a write made (`_hub.write_scope`): the scope the block
    #: installed, which `_hub.take_down` and `retire` know it by. None on the
    #: scope a block installed itself.
    _origin: Scope | None = field(default=None, init=False, repr=False, compare=False)
    #: Created on first use in a Context that no wardex block, bound thread or
    #: span ever reached: a plain thread's empty context (`_hub.unseen_parent`).
    #: Kept by a write's copy, lost by any other fork.
    _implicit: bool = field(default=False, init=False, repr=False, compare=False)
    #: Installed by `continue_trace` for headers whose traceparent could not be
    #: parsed: a root started under it had a caller wardex could not read.
    _parent_unreadable: bool = field(default=False, init=False, repr=False, compare=False)

    def set_tag(self, key: str, value: AttributeValue | list | tuple | dict | None) -> None:
        """Set `key` under `Span.set_attribute`'s rule: None removes it, and a key or
        value the wire cannot hold raises here, at the caller's line."""
        stored = attribute_value("set_tag", key, value)
        if stored is None:
            self.tags.pop(key, None)
        else:
            self.tags[key] = stored

    def set_user(self, user: UserInfo | str | int | None) -> None:
        """Attach `user` (a `UserInfo` or a user id) to this scope; `None` clears it again.

        Anything else raises `TypeError` here, at the caller's line.
        """
        self.user = checked_user(user)

    def set_context(self, key: str, value: dict[str, Any]) -> tuple[list[str], int]:
        """Keep `value` under `key` and stamp its scalars as `wardex.context.<key>.*` tags.

        Replaces the stamped keys of an earlier payload under the same name.
        Returns what `context_entries` left out: `(skipped keys, keys over the cap)`.
        """
        stamped, skipped, over = context_entries(key, value)
        prefix = f"{CONTEXT_PREFIX}{key}."
        for stale in [tag for tag in self.tags if tag.startswith(prefix)]:
            del self.tags[stale]
        self.tags.update(stamped)
        self.contexts[key] = value
        return skipped, over

    def data_fork(self) -> Scope:
        """A copy of the tags, user and contexts alone: what an isolation scope holds.

        The trace position (span context, conversation, tracestate) stays with
        the current scope, so a fork of the data cannot carry a stale parent
        into the header readers that merge both layers.
        """
        return Scope(
            tags=dict(self.tags),
            user=self.user,
            contexts={key: dict(value) for key, value in self.contexts.items()},
        )

    def clone(self) -> Scope:
        # Per-context dict copies, never `deepcopy`: `set_context()` takes
        # arbitrary host objects — a lock, a socket, an open file — and a fork
        # (`isolation_scope()`, `new_scope()`) must not run host
        # `__deepcopy__`/`__reduce__` or raise on a value that cannot be
        # copied. Copying each context dict is what the public surface needs:
        # `set_context()` replaces whole entries, so mutations on the fork
        # stay on the fork; the values themselves stay shared with the host.
        fork = Scope(
            tags=dict(self.tags),
            user=self.user,
            contexts={key: dict(value) for key, value in self.contexts.items()},
            active_span_context=self.active_span_context,
            conversation=self.conversation,
            tracestate=self.tracestate,
        )
        fork._parent_unreadable = self._parent_unreadable
        return fork


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
