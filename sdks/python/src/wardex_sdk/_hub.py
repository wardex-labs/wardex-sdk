from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from ._client import Client
from ._runtime import runtime
from ._scope import Scope, UserInfo, merge_scopes, merged_tags_and_user, merged_trace_fields
from ._types import SpanContext

_global_scope: Scope = Scope()
_current_scope: ContextVar[Scope | None] = ContextVar("wardex_current_scope", default=None)
_isolation_scope: ContextVar[Scope | None] = ContextVar("wardex_isolation_scope", default=None)


def reset_for_test() -> None:
    """For test isolation — resets global state.

    The scopes are this module's own; everything else the SDK installed belongs
    to the `Runtime`, and `Runtime.reset()` is what undoes it — the client (and
    with it the background worker thread a dropped reference would leak), the
    interceptor and adapter registries, the chained signal handlers and the
    shared connection-timing probe. Resetting through the owner rather than
    naming the pieces here is the point: this helper used to reset the client
    alone, so every other state a test installed leaked into the next one.
    """
    global _global_scope
    runtime().reset()
    _global_scope = Scope()
    _current_scope.set(None)
    _isolation_scope.set(None)


def set_client(client: Client | None) -> None:
    runtime().set_client(client)


def get_client() -> Client | None:
    return runtime().client


def get_global_scope() -> Scope:
    return _global_scope


def get_isolation_scope() -> Scope:
    scope = _isolation_scope.get()
    if scope is None:
        scope = Scope()
        _isolation_scope.set(scope)
    return scope


def get_current_scope() -> Scope:
    scope = _current_scope.get()
    if scope is None:
        scope = Scope()
        _current_scope.set(scope)
    return scope


def get_merged_scope() -> Scope:
    return merge_scopes(get_global_scope(), get_isolation_scope(), get_current_scope())


def get_merged_trace_fields() -> tuple[SpanContext | None, str | None]:
    """The merged span context and tracestate, without materializing the merge."""
    return merged_trace_fields(get_global_scope(), get_isolation_scope(), get_current_scope())


def get_merged_tags_and_user() -> tuple[dict[str, str], UserInfo | None]:
    """The merged tags and user, without materializing the merge."""
    return merged_tags_and_user(get_global_scope(), get_isolation_scope(), get_current_scope())


@contextmanager
def new_scope() -> Iterator[Scope]:
    forked = get_current_scope().clone()
    token = _current_scope.set(forked)
    try:
        yield forked
    finally:
        _current_scope.reset(token)


@contextmanager
def isolation_scope() -> Iterator[Scope]:
    """Fork the current isolation scope for the block (Sentry 2.x semantics).

    A CLONE, not a blank scope: the ambient context — tags, user, contexts —
    is inherited, and mutations made inside the block stay on the fork and are
    discarded with it. The current scope is replaced with a fresh one for the
    block, exactly as before.
    """
    new_iso = get_isolation_scope().clone()
    iso_token = _isolation_scope.set(new_iso)
    cur_token = _current_scope.set(Scope())
    try:
        yield new_iso
    finally:
        _current_scope.reset(cur_token)
        _isolation_scope.reset(iso_token)
