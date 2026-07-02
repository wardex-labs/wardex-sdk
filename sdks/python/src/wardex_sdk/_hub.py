from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from ._client import Client
from ._scope import Scope, merge_scopes

_global_scope: Scope = Scope()
_current_scope: ContextVar[Scope | None] = ContextVar("wardex_current_scope", default=None)
_isolation_scope: ContextVar[Scope | None] = ContextVar("wardex_isolation_scope", default=None)
_client: Client | None = None


def reset_for_test() -> None:
    """For test isolation — resets global state."""
    global _global_scope, _client
    _global_scope = Scope()
    _current_scope.set(None)
    _isolation_scope.set(None)
    _client = None


def set_client(client: Client | None) -> None:
    global _client
    _client = client


def get_client() -> Client | None:
    return _client


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
    new_iso = Scope()
    iso_token = _isolation_scope.set(new_iso)
    cur_token = _current_scope.set(Scope())
    try:
        yield new_iso
    finally:
        _current_scope.reset(cur_token)
        _isolation_scope.reset(iso_token)
