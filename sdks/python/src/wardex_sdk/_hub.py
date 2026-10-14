from __future__ import annotations

import asyncio
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token

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


def installed_adapters() -> frozenset[str]:
    """The framework adapters the last `init()` installed and that are still installed, by name.

    For an observer that must not import `_adapters/` itself — the byte and stdio seams sit beside
    it, not above it — and still has to know which framework already records an operation, so the
    seam does not record it a second time. Never builds the adapter registry: one never built has
    installed nothing. Read from the registry, not from the adapter statuses: those are the last
    `init()`'s account and outlive `close()`, which has uninstalled the adapters they name.
    """
    return runtime().installed_adapters()


def get_global_scope() -> Scope:
    return _global_scope


def get_isolation_scope() -> Scope:
    scope = _isolation_scope.get()
    if scope is None:
        scope = Scope()
        _isolation_scope.set(scope)
    return scope if scope._retired_to is None else _settled(_isolation_scope, scope)


def get_current_scope() -> Scope:
    scope = _current_scope.get()
    if scope is None:
        scope = Scope()
        _current_scope.set(scope)
    return scope if scope._retired_to is None else _settled(_current_scope, scope)


class _Here:
    """The running Context, as one object: what a block remembers as its installer.

    A Context has no identity Python lets code read, and a copy of one (a task
    or a `bind_context` callback started inside a block) holds the same values.
    What tells them apart is a Token: only the Context that made it can reset
    it. So the marker holds the Token of its own installation, `here()` tests
    it on every call and installs a fresh marker where the test fails — in a
    copy, which then has a marker of its own.
    """

    __slots__ = ("task", "token")

    def __init__(self) -> None:
        loop = asyncio._get_running_loop()
        task = asyncio.current_task(loop) if loop is not None else None
        #: The asyncio Task running this Context, weakly: `_settle_from_afar`
        #: needs its Context, and a marker must not keep a finished Task alive.
        self.task = None if task is None else weakref.ref(task)
        self.token: Token[_Here | None] | None = None


_here: ContextVar[_Here | None] = ContextVar("wardex_here", default=None)


def here() -> _Here:
    """The marker of the Context running now; see `_Here`."""
    found = _here.get()
    if found is not None and found.token is not None:
        try:
            _here.reset(found.token)
        except (ValueError, RuntimeError):  # a copy of the Context that made it
            found = None
    if found is None:
        found = _Here()
    found.token = _here.set(found)
    return found


def retire(fork: Scope, prev: Scope, owner: _Here) -> None:
    """Make `fork` read as `prev` in `owner`, the Context that installed it.

    For a block that ends in another Context than it began in, where its Token
    cannot reset: a `with wardex.conversation()` held across a `yield` in an
    async generator the consumer abandoned is closed by the event loop on a task
    of its own. The consumer, which installed the block, still holds its scope
    and takes `prev` back (`_settle_from_afar` at once where it can, otherwise
    `_settled` on its next read), as a block that ended at home would have given
    it. A task or callback STARTED inside the block runs in a copy of that
    Context and goes on reading the block's scope, exactly as after a block that
    ended at home: what it was started with does not depend on how the block
    ended.

    `prev` is followed past every scope the same Context already retired, so a
    consumer that abandons stream after stream holds a link or two, never a
    chain through every turn it ever had.
    """
    while prev._retired_to is not None and prev._retired_by is owner:
        prev = prev._retired_to
    if prev is not fork:
        fork._retired_by = owner
        fork._retired_to = prev


def _settled(var: ContextVar[Scope | None], scope: Scope) -> Scope:
    """`scope` as the running Context reads it: what `retire` left, in the installer only."""
    mine = None
    while scope._retired_to is not None:
        if mine is None:
            mine = here()
        if scope._retired_by is not mine:
            break
        scope = scope._retired_to
        var.set(scope)
    return scope


def _settle_from_afar(
    owner: _Here, var: ContextVar[Scope | None], gone: Scope, prev: Scope
) -> None:
    """Hand the installer `prev` NOW, where its Task can lend its Context (3.12+).

    Lazy settlement waits for the installer's next read, and work it starts
    before that read (a thread, a task, a `copy_context()`) copies the retired
    scope. Only from the installer's OWN event loop, which is the usual case —
    an abandoned async generator finalized by its loop — because there the
    installer is necessarily suspended and its Context free to enter. From any
    other thread the installer may be resumed while its Context is entered
    here, and its loop then fails to run the step and loses the task; the
    installer's next read settles it instead.
    """
    task = owner.task() if owner.task is not None else None
    context_of = getattr(task, "get_context", None)
    if context_of is None or task.done() or task.get_loop() is not asyncio._get_running_loop():
        return
    try:
        context_of().run(_settle_in, owner, var, gone, prev)
    except RuntimeError:  # the Context is entered: the Task is running
        return


def _settle_in(owner: _Here, var: ContextVar[Scope | None], gone: Scope, prev: Scope) -> None:
    if var.get() is gone and here() is owner:
        var.set(prev)


def take_down(
    var: ContextVar[Scope | None],
    token: Token[Scope | None],
    installed: Scope,
    prev: Scope,
    owner: _Here,
) -> bool:
    """Undo `var.set(installed)` by its Token or, where the Token refuses because
    the block ended in another Context, by `retire`. True in that second case.

    The fallback sets `prev` here as well when `installed` is current here, and
    never otherwise: in a Context where it is NOT current, a plain `set()` would
    write one request's scope over whatever is.
    """
    try:
        var.reset(token)
    except ValueError:  # the Token was made in another Context
        if var.get() is installed:
            var.set(prev)
        retire(installed, prev, owner)
        _settle_from_afar(owner, var, installed, prev)
        return True
    return False


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
    prev, owner = get_current_scope(), here()
    forked = prev.clone()
    token = _current_scope.set(forked)
    try:
        yield forked
    finally:
        take_down(_current_scope, token, forked, prev, owner)


@contextmanager
def isolation_scope(*, keep_conversation: bool = False) -> Iterator[Scope]:
    """Fork the current isolation scope for the block (Sentry 2.x semantics).

    A CLONE, not a blank scope: the ambient context — tags, user, contexts —
    is inherited, and mutations made inside the block stay on the fork and are
    discarded with it. The current scope is replaced with a fresh one for the
    block, which is also how a host steps out of a `wardex.conversation(...)`.

    `keep_conversation` is for trace continuation (`continue_trace`): a request
    continued inside a conversation is still in it, so the fresh scope keeps the
    conversation while the trace (parent, tracestate) starts over.
    """
    owner = here()
    prev_iso, prev_cur = get_isolation_scope(), get_current_scope()
    new_iso = prev_iso.clone()
    fresh = Scope(conversation=prev_cur.conversation if keep_conversation else None)
    iso_token = _isolation_scope.set(new_iso)
    cur_token = _current_scope.set(fresh)
    try:
        yield new_iso
    finally:
        take_down(_current_scope, cur_token, fresh, prev_cur, owner)
        take_down(_isolation_scope, iso_token, new_iso, prev_iso, owner)
