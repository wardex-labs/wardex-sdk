"""One answer to "this connection is over", shared by every byte seam.

Four separate defects in `interceptors/` were the same missing moment. State is
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
that fires exactly once, and it fires from two places rather than one because
neither alone is enough:

  CLOSE — `socket.socket.close()` is patched, so the hook runs while the object
  is still intact and the memory is released promptly. This is the timely half,
  and the only half that can act on a connection whose object the host keeps
  alive in a pool.

  FINALIZE — a `weakref.finalize` backstop for everything that is dropped
  without `close()`: `ssl.SSLObject` has no close at all, an abandoned socket is
  closed by its own deallocator, and an exception path can lose either. This is
  the half that makes the id-reuse bug IMPOSSIBLE rather than unlikely: CPython
  invokes weakref callbacks during deallocation, before the address can be
  handed to anything else, so there is no instant at which a live object shares
  an id with a registered-but-unfired hook.

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

from ..assembly import PatchSet, guard

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
    at an arbitrary allocation on whatever thread happened to drop the last
    reference — including a thread already inside `on_close`. A plain `Lock`
    there is a self-deadlock in the host's own code, and an `RLock` would only
    hide it while ordering nothing. What is left is what the dict itself
    guarantees: `pop`, `get` and insert are each single operations, and two
    threads cannot register the same object because they cannot both have just
    created it. `_seam._conns` is unlocked for the same reason.
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
        """
        cid = id(obj)
        entry = self._entries.get(cid)
        if entry is None:
            entry = _Entry()
            entry.finalizer = _finalizer(self, obj, cid)
            self._entries[cid] = entry
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
        """Drop every registration without running anything (uninstall path)."""
        for entry in self._entries.values():
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


class CloseProbe:
    """Turns `socket.socket.close()` into a registry event. Idempotent, fail-silent.

    One patch, on `socket.socket`. `ssl.SSLSocket` inherits `close` (it overrides
    `_real_close`, not `close`), so the TLS seam's objects arrive here too and a
    second patch would double-fire. `ssl.SSLObject` is not a socket and has no
    close of its own; it reaches the registry through the finalizer only.
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
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._patches.restore_all()
        self._installed = False

    def _mk_close(self, orig: Any):  # noqa: ANN202
        registry = self._registry

        def wrapper(this: Any, *a: Any, **k: Any) -> Any:
            # BEFORE the real close, so a hook may still read the object it is
            # about to lose (its fileno, its peer). `fire` guards each hook
            # itself, so nothing here can stop the host's close from happening.
            registry.fire(this)
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
