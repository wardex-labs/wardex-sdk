"""One answer to "this connection is over", shared by every byte seam.

Four separate defects in `_interceptors/` were the same missing moment. State is
created the first time a connection is SEEN — a protocol tracker in
`_seam._conns`, a per-stream latch inside `_Http2Tracker`, a connect/handshake
pair in `_conn_timing` — and nothing anywhere destroyed it, because the seams
observe bytes and bytes stop arriving without saying so. What each site did
instead was guess:

  * `_seam._conns` is keyed by `id(obj)` and evicted only under a FIFO cap. A
    dead connection's entry therefore survives its socket, and CPython hands
    the same address to the next object of that size — so a brand new socket
    could be served the previous one's tracker, its sniff-latch verdict and its
    half-parsed request. That is not a leak, it is wrong data: an HTTPS
    connection landing on the id of a retired Redis one inherits
    `gate == "ignore"` and is never captured at all.
  * `_Http2Tracker._latch` pops per transaction, so a stream that ends without
    a response (RST_STREAM, GOAWAY, a server that simply stops) leaves an entry
    behind for as long as the tracker lives.
  * `ConnTimingStore` is keyed by fileno and popped only when a span is
    emitted, so every non-HTTP socket in the process squats a slot until the
    FIFO cap pushes it out — and what the cap pushes out first is the OLDEST
    entry, which is the live TLS connection that is still streaming a response.
    It then reports `connect_timing_unavailable` for a connection wardex
    measured perfectly well.

This module supplies the moment. `on_close(obj, hook)` registers a callback
that fires exactly once, and it fires from three places rather than one because
no one of them is enough:

  CLOSE — `socket.socket.close`/`_real_close` are patched, so the hook runs
  while the object is still intact and the memory is released promptly. This is
  the timely half, and the only half that can act on a POOLED connection, whose
  object the host holds for as long as the pool does.

  FINALIZE — a `weakref.finalize` backstop for everything that is dropped
  without `close()`: `ssl.SSLObject` has no close at all, an abandoned socket is
  closed by its own deallocator, and an exception path can lose either. This is
  the half that makes the id-reuse bug IMPOSSIBLE rather than unlikely: CPython
  invokes weakref callbacks during deallocation, before the address can be
  handed to anything else, so there is no instant at which a live object shares
  an id with a registered-but-unfired hook.

  CONNECTION LOST — `asyncio.sslproto.SSLProtocol.connection_lost` is patched,
  which is the only end-of-connection signal a POOLED `ssl.SSLObject` has. The
  async TLS seam's carrier has no `close()` for the first half to patch, and
  asyncio pins it to the protocol for the transport's whole life, so the second
  half does not run either until the pool itself is dropped: per-connection
  state for a keep-alive connection to a model provider was held for as long as
  the pool held the connection. That is where the original leak was found.
  The protocol object, however, IS told — this is the call that ends its
  transport — and it is holding the `SSLObject`, so the moment exists after all
  and only needed to be read off a private attribute one layer up.

That third patch is a best-effort improvement over GC timing, never a guarantee,
and it is the reason nothing sized by a connection may depend on any single
signal. It is absent under an event loop that implements TLS itself rather than
through `asyncio.sslproto` (uvloop is the one every user has), and under a
protocol subclass that overrides `connection_lost` without calling up. Both
degrade to the finalizer, which is late but correct — and behind both stands the
one bound that needs no signal at all: the explicit cap beside
`_Http2Tracker._latch`.

Nothing here may hold a strong reference to the observed object — a registry
that pins sockets would keep the host's file descriptors open, which is a
worse bug than the ones it fixes. `weakref.finalize` keeps its CALLBACK and
the callback's ARGUMENTS alive, never the referent, so the arguments are this
registry and an integer, and the hooks the seams register close over an id
rather than over the socket.

`detach()` is deliberately NOT a close event. `ssl.SSLContext.wrap_socket`
builds the `SSLSocket` from the plain socket's fileno and then calls
`sock.detach()` on the original: the fd lives on under a new owner, and firing
close hooks there would evict the connect timing of every TLS connection in the
process at the exact moment it was wrapped. A detach is a handoff, not an end.
"""

from __future__ import annotations

import socket
import weakref
from collections.abc import Callable
from typing import Any

from .._assembly import PatchSet, guard

__all__ = [
    "CloseProbe",
    "CloseRegistry",
    "close_registry",
    "install_shared_close_hook",
    "on_close",
    "uninstall_shared_close_hook",
]

#: Reusable and reentrant (it holds no per-entry state), so the per-callback
#: guard costs one attribute lookup rather than an allocation.
_FIRE = guard("interceptors.close_hook.fire")
#: Registering a finalizer on an object that ACCEPTS weak references should not
#: be able to fail; if it does, that is a wardex failure and belongs in a
#: counter. An object that simply refuses one is not — see `_supports_weakref`.
_REGISTER = guard("interceptors.close_hook.register")
#: Importing a stdlib module should not be able to fail either — see
#: `_ssl_protocol_class`, which is the one place this is entered.
_IMPORT_SSLPROTO = guard("interceptors.close_hook.sslproto_import")
#: The `SSLProtocol.connection_lost` wrapper, whose failure mode is worse than
#: an error: the read runs BEFORE the original, so an exception escaping it
#: stops asyncio ever scheduling the app protocol's `connection_lost` and the
#: host's `wait_closed()` never returns. A hung coroutine, not a traceback. The
#: attributes are private and belong to a class a third party may subclass with
#: a `__getattr__` or a property of its own, so the read is somebody else's code
#: and is treated as such.
_CONNECTION_LOST = guard("interceptors.close_hook.connection_lost")


class _Entry:
    """Every hook registered for ONE object, plus the finalizer that backstops them."""

    __slots__ = ("finalizer", "hooks")

    def __init__(self) -> None:
        #: (hook, also fire it from the finalizer)
        self.hooks: list[tuple[Callable[[], None], bool]] = []
        self.finalizer: weakref.finalize | None = None


class CloseRegistry:
    """id(obj) -> the hooks to run when that object's connection ends.

    Keyed by id and safe to key by id, which is the whole trick: an entry is
    removed either by the close hook firing or by the finalizer firing, and the
    finalizer runs before the address can be recycled. The one shape that
    escapes is an object that supports no weak reference AND is never closed;
    its entry stays, and a later object at the same address would append to it.
    Nothing wardex observes is that shape — `socket.socket` and `ssl.SSLObject`
    both support weak references — and the alternative (holding the object) is
    the bug this exists to avoid.

    NO LOCK, deliberately. `_fire` can run from a weakref callback, which lands
    wherever a reference count reaches zero — an attribute store, a `del`, a
    container eviction, a frame exit, or a cyclic collection at an allocation —
    on whatever thread happened to drop that reference, and so including a
    thread already inside `on_close`. A plain `Lock` there is a self-deadlock in
    the host's own code, and an `RLock` would only
    hide it while ordering nothing. What is left is what the dict itself
    guarantees: `pop`, `get` and insert are each single operations, and two
    threads cannot register the same object because they cannot both have just
    created it. `_seam._conns` is unlocked for the same reason.

    The price is that no method here may hold the table open across a step that
    can re-enter. `clear()` is the one that walks it, so it walks a snapshot:
    see the note there.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[int, _Entry] = {}

    def on_close(self, obj: Any, hook: Callable[[], None], *, on_finalize: bool = True) -> None:
        """Run `hook()` once, when `obj` is closed or collected.

        `on_finalize=False` restricts the hook to the EXPLICIT close. It exists
        for hooks whose action is keyed by something the object no longer owns
        once it is dead: a file descriptor is reissued by the kernel the moment
        it is released, so a hook that pops a fileno-keyed record must run while
        the fd still belongs to this object, and never afterwards from a
        finalizer that may be racing the next `connect()`.

        Registering the SAME hook twice registers it once. Both callers repeat:
        `ssl.SSLSocket.do_handshake` runs in a retry loop on a non-blocking
        socket and again on renegotiation, and the seam re-creates a connection
        state — and re-registers its retirement hook — every time the FIFO cap
        evicted a still-live entry. Appending there is an unbounded list on a
        long-lived socket, which is the shape this module exists to remove.
        """
        cid = id(obj)
        entry = self._entries.get(cid)
        if entry is None:
            entry = _Entry()
            entry.finalizer = _finalizer(self, obj, cid)
            self._entries[cid] = entry
        key = _hook_identity(hook)
        for existing, _ in entry.hooks:
            if _hook_identity(existing) == key:
                return
        entry.hooks.append((hook, on_finalize))

    def fire(self, obj: Any) -> None:
        """The object was closed. Runs every hook, at most once, in registration order."""
        self._fire(id(obj), finalizing=False)

    def forget(self, obj: Any) -> None:
        """Drop `obj`'s hooks WITHOUT running them."""
        entry = self._entries.pop(id(obj), None)
        if entry is not None and entry.finalizer is not None:
            entry.finalizer.detach()

    def clear(self) -> None:
        """Drop every registration without running anything (uninstall path).

        A SNAPSHOT, not `self._entries.values()`. This is the one method that
        holds the table open across something that can re-enter: `finalize.detach`
        is ordinary Python and allocates, and any socket in the process closing
        on another thread reaches `_fire`, which pops. Either mutates the dict
        mid-walk and `clear()` raises `RuntimeError: dictionary changed size
        during iteration` — out of `uninstall_shared_close_hook`, which the
        registries above it call after they have already restored their patches
        but before they clear their `_installed` flags. The exception is counted
        and swallowed, the flag stays True on a module singleton, and the next
        `init()` finds an already-installed probe: connection timing is silently
        never instrumented again for the life of the process.
        """
        for entry in list(self._entries.values()):
            if entry.finalizer is not None:
                entry.finalizer.detach()
        self._entries.clear()

    def tracked(self) -> int:
        """How many objects currently have hooks. Diagnostics and tests only."""
        return len(self._entries)

    def _finalized(self, cid: int) -> None:
        self._fire(cid, finalizing=True)

    def _fire(self, cid: int, *, finalizing: bool) -> None:
        entry = self._entries.pop(cid, None)
        if entry is None:
            return
        if entry.finalizer is not None and not finalizing:
            # Detached, not left to fire later into an empty table: a live
            # finalizer is a strong reference to this registry and to the hook
            # tuple, and there are as many of them as there have been sockets.
            entry.finalizer.detach()
        for hook, on_finalize in entry.hooks:
            if finalizing and not on_finalize:
                continue
            # One guard per hook: a seam whose eviction raises must not cost the
            # other seam its own. This runs inside `socket.close()` and inside
            # garbage collection, so it is also the last boundary before an
            # instrumentation failure would reach the host (I6).
            with _FIRE:
                hook()


def _hook_identity(hook: Callable[[], None]) -> Any:
    """What makes two registrations the same registration.

    Every hook the seams register is a `functools.partial` closing over an id or
    a fileno, and `partial` defines no `__eq__` — two built from the same
    function and the same argument are distinct objects that mean one thing. So
    the pair is compared, not the wrapper. Anything else (a plain function, a
    bound method, a lambda) answers for itself, which is the identity a caller
    would expect.
    """
    func = getattr(hook, "func", None)
    if func is None:
        return hook
    return (func, getattr(hook, "args", ()))


def _supports_weakref(obj: Any) -> bool:
    """Can `obj` carry a weak reference at all?

    ASKED, not attempted. `weakref.finalize` answers by raising `TypeError`, and
    catching that would file "the caller handed this seam an object that is not
    a socket" as a swallowed wardex failure — a `guard()` counter, which is the
    signal that means A SPAN WAS DELETED. It is the same distinction
    `_mcp_stdio` draws around an absent anyio: an absence is an answer, and
    recording it as a failure is how a counter stops being evidence.

    `__weakrefoffset__` is CPython's own answer to the question and is what
    `weakref` itself consults. An implementation that does not expose it reads
    as "no", which costs the finalize backstop and keeps the explicit close
    hook — degraded, never wrong.
    """
    return getattr(type(obj), "__weakrefoffset__", 0) != 0


def _finalizer(registry: CloseRegistry, obj: Any, cid: int) -> weakref.finalize | None:
    """A weak finalizer for `obj`, or None if it cannot carry one.

    `atexit=False` because the default is True: `weakref.finalize` would
    otherwise run every outstanding hook during interpreter shutdown, on a
    half-torn-down module graph, printing "Exception ignored in" at a host that
    has already stopped caring. Interpreter exit is not a connection close, and
    the SDK's own teardown (`_lifecycle`) is where shutdown is handled.
    """
    if not _supports_weakref(obj):
        return None
    fin: weakref.finalize | None = None
    with _REGISTER:
        registered = weakref.finalize(obj, registry._finalized, cid)
        registered.atexit = False
        fin = registered
    return fin


def _ssl_protocol_class() -> Any:
    """`asyncio.sslproto.SSLProtocol`, or None where there is no such thing.

    The import sits here rather than at module scope only to keep the failure
    LOCAL — inside the guard below, at the one moment the class is wanted. It
    buys nothing at import time and the docstring should not pretend otherwise:
    `import wardex_sdk` already leaves `asyncio.sslproto` and `ssl` in
    `sys.modules` by way of `asyncio.base_events`, and `install()` asks for the
    class unconditionally anyway — the plaintext seam and the timing probe both
    reach it through `_acquire_close_hook`.

    Guarded, and counted rather than swallowed, because unlike anyio this is not
    an optional dependency: `asyncio.sslproto` is stdlib, so a failure to import
    it is a fact about this runtime that a maintainer wants to see. Absent, the
    async TLS path keeps the finalizer and the per-connection caps and loses
    only promptness — degraded, never wrong.
    """
    sslproto: Any = None
    with _IMPORT_SSLPROTO:
        from asyncio import sslproto as _sslproto

        sslproto = _sslproto
    return getattr(sslproto, "SSLProtocol", None)


def _sslobj_of(protocol: Any) -> Any:
    """The `ssl.SSLObject` an `SSLProtocol` is driving, or None.

    Two spellings for one attribute, and both are private. Since the 3.11
    rewrite the protocol wraps the BIO itself and holds `_sslobj`; before it,
    the object lived one layer down in the `_SSLPipe` the protocol drove, which
    exposes it as `ssl_object`. Neither is guaranteed by anything, so both are
    read with a default and a miss costs the timely half only — the finalizer
    and the per-connection caps are still underneath.

    None is also the ordinary answer for a connection that died during the
    handshake: there is no `SSLObject` yet, and nothing registered a hook
    against one.
    """
    sslobj = getattr(protocol, "_sslobj", None)
    if sslobj is not None:
        return sslobj
    return getattr(getattr(protocol, "_sslpipe", None), "ssl_object", None)


class CloseProbe:
    """Turns the end of a connection into a registry event. Idempotent, fail-silent.

    THREE patches. Two of them are on `socket.socket`, because `close()` is not
    reliably the end of anything:

        def close(self):
            self._closed = True
            if self._io_refs <= 0:
                self._real_close()

    `http.client` depends on exactly that. When a response `will_close` — a
    `Connection: close` header, HTTP/1.0, or a body with no length framing —
    `getresponse()` calls `sock.close()` the moment the HEADERS are parsed
    ("this effectively passes the connection to the response"), and the body
    then arrives through the `makefile()` object that still holds the fd, which
    is to say through the seams' own patched `recv_into`. Firing at that
    `close()` retired the connection mid-response: the remaining body landed on
    a freshly built state, a response with no request ahead of it latches
    `gate = "ignore"`, and the span was never assembled.

    So the fire happens where the fd actually goes. `_real_close` is the single
    funnel that both the immediate close and the deferred one (`SocketIO.close`
    -> `_decref_socketios`) reach, and that `ssl.SSLSocket._real_close` enters
    through `super()`. The `close` patch stays as the timely path for the
    ordinary case, and declines while `_io_refs` is outstanding. Both firing is
    harmless: the registry pops an entry before running it, so the second is a
    no-op.

    `ssl.SSLSocket` inherits `close` and overrides `_real_close` with a
    `super()` call, so the TLS seam's objects arrive through these same two
    patches. `ssl.SSLObject` is not a socket and has neither, which is what the
    THIRD patch is for.

    `asyncio.sslproto.SSLProtocol.connection_lost` is that third, and it is a
    patch on somebody else's private class — asked for with `getattr` and
    skipped when absent, exactly as `_real_close` is. It earns the exception
    because the alternative is not "late", it is "never": an `SSLObject` in a
    connection pool is closed by nobody and collected by nobody, so this is the
    only moment at which its per-connection state can be released while the
    process still cares. The protocol is told its transport ended and is
    holding the object the seam keyed everything by; nothing else in the
    interpreter knows both facts at once.

    The `SSLObject` is read BEFORE the original runs, because 3.10 clears the
    `_SSLPipe` that holds it on the way out. Two attribute paths for the same
    thing: `SSLProtocol._sslobj` since the 3.11 rewrite, and the pipe's
    `ssl_object` before it.
    """

    __slots__ = ("_installed", "_patches", "_registry")

    def __init__(self, registry: CloseRegistry) -> None:
        self._registry = registry
        self._patches = PatchSet("interceptors.close_hook")
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        self._patches.patch(socket.socket, "close", self._mk_close(socket.socket.close))
        # Private, so it is asked for rather than assumed. Without it the probe
        # is still correct, only late: a `will_close` connection is then retired
        # by the finalizer instead of when its fd is released.
        real_close = getattr(socket.socket, "_real_close", None)
        if real_close is not None:
            self._patches.patch(socket.socket, "_real_close", self._mk_real_close(real_close))
        proto = _ssl_protocol_class()
        lost = getattr(proto, "connection_lost", None)
        if lost is not None:
            self._patches.patch(proto, "connection_lost", self._mk_connection_lost(lost))
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._patches.restore_all()
        self._installed = False

    def _mk_close(self, orig: Any):  # noqa: ANN202
        registry = self._registry

        def wrapper(this: Any, *a: Any, **k: Any) -> Any:
            # An outstanding `makefile()` reference means this call ends
            # nothing — the body of a `will_close` response is still to come
            # through it. `_real_close` fires when the last one goes.
            if not getattr(this, "_io_refs", 0):
                # BEFORE the real close, so a hook may still read what the
                # object is about to lose (its fileno, its peer). `fire` guards
                # each hook, so nothing here can stop the host's close.
                registry.fire(this)
            return orig(this, *a, **k)

        return wrapper

    def _mk_real_close(self, orig: Any):  # noqa: ANN202
        registry = self._registry

        def wrapper(this: Any, *a: Any, **k: Any) -> Any:
            # Still ahead of the fd going back to the kernel, which is what the
            # fileno-keyed hooks (`_conn_timing`) require of a close event.
            registry.fire(this)
            return orig(this, *a, **k)

        return wrapper

    def _mk_connection_lost(self, orig: Any):  # noqa: ANN202
        registry = self._registry

        def wrapper(this: Any, *a: Any, **k: Any) -> Any:
            # Ahead of the original for two reasons: the same one `_mk_close`
            # gives, and because 3.10's `connection_lost` drops the `_SSLPipe`
            # that owns the object we are looking for.
            #
            # Guarded because being ahead of the original is what makes a raise
            # here expensive: asyncio would never reach the `call_soon` that
            # tells the app protocol its transport ended, so the host waits on a
            # future nothing will complete. See `_CONNECTION_LOST`.
            with _CONNECTION_LOST:
                sslobj = _sslobj_of(this)
                if sslobj is not None:
                    registry.fire(sslobj)
            return orig(this, *a, **k)

        return wrapper


# --- Module singleton: the SSL seam, the plaintext seam and the timing probe
# all need the same patch on socket.socket.close, and must not each install one ---
_registry = CloseRegistry()
_probe = CloseProbe(_registry)
_refcount = 0


def close_registry() -> CloseRegistry:
    return _registry


def on_close(obj: Any, hook: Callable[[], None], *, on_finalize: bool = True) -> None:
    """Register `hook` on the shared registry. See `CloseRegistry.on_close`."""
    _registry.on_close(obj, hook, on_finalize=on_finalize)


def install_shared_close_hook() -> None:
    global _refcount
    if _refcount == 0:
        _probe.install()
    _refcount += 1


def uninstall_shared_close_hook() -> None:
    global _refcount
    if _refcount == 0:
        return
    _refcount -= 1
    if _refcount == 0:
        _probe.uninstall()
        # The registrations go with the patch. Every hook still in the table
        # belongs to a seam that has just been uninstalled, so running them
        # later would touch state that no longer exists — and keeping them would
        # keep that seam alive for as long as its sockets live.
        _registry.clear()
