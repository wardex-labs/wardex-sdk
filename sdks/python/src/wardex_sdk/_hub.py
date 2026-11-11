from __future__ import annotations

import asyncio
import logging
import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, TypeVar

from ._client import Client
from ._runtime import runtime
from ._scope import Scope, UserInfo, merge_scopes, merged_tags_and_user, merged_trace_fields
from ._suppress import suppress_capture
from ._types import SpanContext

_R = TypeVar("_R")

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
        scope._implicit = True
        _isolation_scope.set(scope)
        return scope
    return scope if _block_of(scope)._retired_to is None else _settled(_isolation_scope, scope)


def _block_of(scope: Scope) -> Scope:
    """The scope a block installed, for `scope` or for a copy a write made of it."""
    return scope if scope._origin is None else scope._origin


#: Keys `say_once` has already written. A dict, for `report_once`'s reason:
#: `setdefault` decides "was I first" in one C call. The values are inert.
_said: dict[str, object] = {}


def say_once(key: str, message: str) -> None:
    """Warn once per process on the `wardex_sdk` logger; never raise.

    For the layers below the diagnostics module, which cannot import it: the
    line is written here, with capture suppressed as that module does for a
    log handler that ships records over HTTP.
    """
    mark = object()
    if _said.setdefault(key, mark) is not mark:
        return
    try:
        with suppress_capture():
            logging.getLogger("wardex_sdk").warning(message)
    except Exception:  # noqa: BLE001 - a host log handler may raise; the host's call goes on
        return


def get_current_scope() -> Scope:
    scope = _current_scope.get()
    if scope is None:
        scope = Scope()
        scope._implicit = True
        _current_scope.set(scope)
        return scope
    return scope if _block_of(scope)._retired_to is None else _settled(_current_scope, scope)


#: `unseen_parent`'s reasons, as the `Ambient` that latched one carries them.
UNBOUND_THREAD = "unbound_thread"
TRACEPARENT_UNREADABLE = "traceparent_unreadable"


def unseen_parent(scope: Scope) -> str | None:
    """Why a trace root about to start under `scope` may have a parent wardex could not see.

    `scope` is the running Context's current scope with no active span, read
    where the work is issued. Two causes are observable, and each makes the
    root an unresolved edge instead of a confident one (no parent is guessed):
    a `continue_trace` whose traceparent could not be parsed, and a thread
    Python started with an empty context (a plain `Thread`, an executor worker)
    working while a span is open elsewhere in the process. A context that a
    wardex block, a bound thread or a span reached is not the second: there a
    root is what the code asked for.
    """
    if scope._parent_unreadable:
        return TRACEPARENT_UNREADABLE
    iso = _isolation_scope.get()
    if not scope._implicit or (iso is not None and not iso._implicit):
        return None
    thread = threading.current_thread()
    if thread is threading.main_thread() or thread.name.startswith("wardex-"):
        return None
    return UNBOUND_THREAD if runtime()._open_spans else None


def explain_unseen_parent(reason: str) -> None:
    """Say once why a root was recorded as unresolved; the traceparent case was said at entry."""
    if reason == UNBOUND_THREAD:
        say_once(
            UNBOUND_THREAD,
            "A span or model call started on a thread that carries no wardex context "
            "while a span was open elsewhere, so wardex cannot tell which trace it belongs "
            "to; it is recorded without a parent and marked parent_unresolved. If it belongs "
            "to the open span, start the thread's function through wardex.bind_context(fn) "
            "(also for executor.submit and loop.run_in_executor). If it is work of its own, "
            "open it inside wardex.isolation_scope(), wardex.continue_trace(headers) or the "
            "ASGI/WSGI middleware.",
        )


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
    while (block := _block_of(prev))._retired_to is not None and block._retired_by is owner:
        prev = block._retired_to
    if _block_of(prev) is not fork:
        fork._retired_by = owner
        fork._retired_to = prev


def _settled(var: ContextVar[Scope | None], scope: Scope) -> Scope:
    """`scope` as the running Context reads it: what `retire` left, in the installer only."""
    mine = None
    while (block := _block_of(scope))._retired_to is not None:
        if mine is None:
            mine = here()
        if block._retired_by is not mine:
            break
        scope = block._retired_to
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
    now = var.get()
    if now is not None and _block_of(now) is gone and here() is owner:
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
        now = var.get()
        if now is not None and _block_of(now) is installed:
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


def write_scope(var: ContextVar[Scope | None], apply: Callable[..., _R], *args: Any) -> _R:
    """Apply one write to the running Context's scope in `var` by installing a changed copy.

    A scope bound in a ContextVar is never changed in place by a write. Work
    started from this Context (an asyncio task, a `bind_context` thread, a
    `copy_context()`) holds the same object, so an in-place write would reach
    it: a dispatcher that sets one request's user and then starts the next
    request would retag every request it had already started. The copy is
    changed and installed here alone, so work started earlier keeps what it was
    started with and nothing written elsewhere arrives here. `apply` runs on
    the copy before it is installed: a write that raises changes nothing.
    """
    scope = get_current_scope() if var is _current_scope else get_isolation_scope()
    fork = scope.clone()
    fork._implicit = scope._implicit
    fork._origin = _block_of(scope)
    result = apply(fork, *args)
    var.set(fork)
    return result


def _block_field(name: str, trace: bool) -> property:
    def read(self: BlockScope) -> Any:
        return getattr(self._scope(trace), name)

    def write(self: BlockScope, value: Any) -> None:
        self._write(trace, Scope.set_user if name == "user" else _assign(name), value)

    return property(read, write)


def _assign(name: str) -> Callable[[Scope, Any], None]:
    return lambda scope, value: setattr(scope, name, value)


class BlockScope(Scope):
    """The `Scope` that `wardex.isolation_scope()` and `wardex.new_scope()` yield.

    Not an object of its own: the block's scope as the running Context sees
    it. Every write installs a changed copy (`write_scope`), so an object
    handed out at entry would miss the block's later writes and take writes
    that nothing reads. While the block is open this reads and writes the
    ContextVars, through the same copy-on-write as the module-level setters;
    once the block ends it holds a detached copy of what the block ended with.
    Tags, user and contexts live in `data_var`; the trace fields in `trace_var`.
    """

    tags = _block_field("tags", trace=False)
    user = _block_field("user", trace=False)
    contexts = _block_field("contexts", trace=False)
    active_span_context = _block_field("active_span_context", trace=True)
    conversation = _block_field("conversation", trace=True)
    tracestate = _block_field("tracestate", trace=True)

    def __init__(
        self, data_var: ContextVar[Scope | None], trace_var: ContextVar[Scope | None]
    ) -> None:
        self._vars = (data_var, trace_var)
        self._ended: tuple[Scope, Scope] | None = None

    def _scope(self, trace: bool) -> Scope:
        if self._ended is not None:
            return self._ended[trace]
        return get_current_scope() if self._vars[trace] is _current_scope else get_isolation_scope()

    def _write(self, trace: bool, apply: Callable[..., _R], *args: Any) -> _R:
        if self._ended is not None:
            return apply(self._ended[trace], *args)
        return write_scope(self._vars[trace], apply, *args)

    def set_tag(self, key: str, value: str) -> None:
        self._write(False, Scope.set_tag, key, value)

    def set_user(self, user: UserInfo | str | int | None) -> None:
        self._write(False, Scope.set_user, user)

    def set_context(self, key: str, value: dict[str, Any]) -> tuple[list[str], int]:
        return self._write(False, Scope.set_context, key, value)

    def end(self) -> None:
        self._ended = (self._scope(False).clone(), self._scope(True).clone())


@contextmanager
def public_isolation_scope() -> Iterator[Scope]:
    """`isolation_scope()` for the public API, yielding a `BlockScope` over the isolation scope."""
    with isolation_scope():
        yield from _handed(BlockScope(_isolation_scope, _isolation_scope))


@contextmanager
def public_new_scope() -> Iterator[Scope]:
    """`new_scope()` for the public API: identity in its isolation fork, trace in its current."""
    with new_scope():
        yield from _handed(BlockScope(_isolation_scope, _current_scope))


def _handed(handle: BlockScope) -> Iterator[Scope]:
    try:
        yield handle
    finally:
        handle.end()


@contextmanager
def new_scope() -> Iterator[Scope]:
    """Fork both the current scope and the isolation scope for the block.

    Yields the CURRENT fork, where a direct `scope.set_tag()` or a trace field
    lands. Module-level setters inside the block write to the isolation fork,
    so they too end with the block; a nested block forks this one, and work
    started inside reads it and writes copies of its own.
    """
    prev_cur, prev_iso, owner = get_current_scope(), get_isolation_scope(), here()
    forked = prev_cur.clone()
    data = prev_iso.data_fork()
    iso_token = _isolation_scope.set(data)
    cur_token = _current_scope.set(forked)
    try:
        yield forked
    finally:
        take_down(_current_scope, cur_token, forked, prev_cur, owner)
        take_down(_isolation_scope, iso_token, data, prev_iso, owner)


@contextmanager
def isolation_scope(*, restart_trace: bool = False) -> Iterator[Scope]:
    """Fork the current isolation scope for the block (Sentry 2.x semantics).

    A copy, not a blank scope: the ambient tags, user and contexts are
    inherited, and writes made inside the block stay on the fork and are
    discarded with it. The trace position is NOT touched: a span opened inside
    keeps the parent and conversation it would have had outside, because
    isolating identity must not cut the tree.

    `restart_trace` is for trace continuation (`continue_trace`) alone: the
    current scope is replaced for the block by a fresh one that keeps the
    conversation while the trace (parent, tracestate) starts over.
    """
    owner = here()
    prev_iso, prev_cur = get_isolation_scope(), get_current_scope()
    new_iso = prev_iso.data_fork()
    iso_token = _isolation_scope.set(new_iso)
    fresh = Scope(conversation=prev_cur.conversation) if restart_trace else None
    cur_token = None if fresh is None else _current_scope.set(fresh)
    try:
        yield new_iso
    finally:
        if fresh is not None and cur_token is not None:
            take_down(_current_scope, cur_token, fresh, prev_cur, owner)
        take_down(_isolation_scope, iso_token, new_iso, prev_iso, owner)
