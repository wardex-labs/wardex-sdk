"""Connection-layer timing measurement — TCP connect + TLS handshake.

sync: measures socket.connect / SSLSocket.do_handshake directly, keyed by fileno.
async: connect time is derived by subtraction — create_connection total time minus
       accumulated SSLObject.do_handshake time — stamped onto the SSLObject via
       a ContextVar→wrap_bio bridge.
fail-silent: measurement failures are silently ignored and never interfere with
       the original behavior.
"""

from __future__ import annotations

import asyncio
import socket
import ssl
import time
from contextvars import ContextVar
from functools import partial
from typing import Any

from .. import _wardex_native
from .._assembly import PatchSet
from ._close_hook import install_shared_close_hook, on_close, uninstall_shared_close_hook

# ContextVar that carries the timing record the async probe stamps onto the SSLObject
_establishing: ContextVar[_TimingRecord | None] = ContextVar("_wardex_establishing", default=None)

# List of event loop classes whose create_connection gets patched.
# macOS/Linux: _UnixSelectorEventLoop fully reimplements
# BaseEventLoop.create_connection (without calling super()), so the concrete
# class must be included as well.
_CONNECT_TARGETS: list[type] = [asyncio.base_events.BaseEventLoop]
try:
    import asyncio.unix_events  # macOS / Linux concrete loop

    _CONNECT_TARGETS.append(asyncio.unix_events._UnixSelectorEventLoop)
except (ImportError, AttributeError):
    pass
try:
    import asyncio.proactor_events  # Windows concrete loop

    _CONNECT_TARGETS.append(asyncio.proactor_events.BaseProactorEventLoop)
except (ImportError, AttributeError):
    pass
try:
    import uvloop

    _CONNECT_TARGETS.append(uvloop.Loop)
except ImportError:
    pass


# Timing record stamped onto the SSLObject by the async probe (create_connection path)
class _TimingRecord:
    __slots__ = ("connect_ms", "handshake_ms", "total_ms", "hs_start")

    def __init__(self) -> None:
        self.connect_ms = 0.0
        self.handshake_ms = 0.0
        self.total_ms = 0.0
        self.hs_start = 0.0


class ConnTimingStore:
    """Sync-path-only handoff buffer: fileno → (connect_ms, handshake_ms).

    A slot is released when the transaction that needed it is emitted (`pop`)
    or when its socket closes (`discard`, wired by the close hook in
    `ConnTimingProbe`). The FIFO cap is the BACKSTOP for whatever neither of
    those reaches, and it is a poor one on its own: it evicts the OLDEST entry,
    which on a busy process is the live TLS connection still streaming a
    response, while the socket that connected once and died an hour ago keeps
    its slot. That is where the spurious `connect_timing_unavailable` came from,
    and close-driven release is what keeps the cap from being reached at all.
    """

    def __init__(self, cap: int | None = None) -> None:
        self._by_fileno: dict[int, list[float]] = {}
        # None means "use the core default" — resolved here (rather than hardcoded)
        # so this can never silently drift from crates/wardex-limits.
        self._cap = cap if cap is not None else _wardex_native.limits_defaults()["max_connections"]

    def _slot(self, fileno: int) -> list[float]:
        slot = self._by_fileno.get(fileno)
        if slot is None:
            if len(self._by_fileno) >= self._cap:
                # FIFO eviction: remove the first entry in dict insertion order
                self._by_fileno.pop(next(iter(self._by_fileno)))
            slot = [0.0, 0.0]  # [connect_ms, handshake_ms]
            self._by_fileno[fileno] = slot
        return slot

    def set_connect(self, fileno: int, connect_ms: float) -> None:
        self._slot(fileno)[0] = connect_ms

    def set_handshake(self, fileno: int, handshake_ms: float) -> None:
        self._slot(fileno)[1] = handshake_ms

    def pop(self, fileno: int) -> tuple[float, float] | None:
        slot = self._by_fileno.pop(fileno, None)
        if slot is None:
            return None
        return (slot[0], slot[1])

    def discard(self, fileno: int) -> None:
        """Release a slot nobody will consume — the socket that owned it is gone.

        Separate from `pop` because the two are different events with the same
        mechanics: `pop` is "a span consumed this measurement", `discard` is
        "there will never be a span". Reading the difference off a discarded
        return value would make the close hook look like a consumer.
        """
        self._by_fileno.pop(fileno, None)

    def clear(self) -> None:
        self._by_fileno.clear()

    def set_cap(self, cap: int) -> None:
        """Re-apply the configured bound. The store OUTLIVES the seams that share it.

        `shared_timing_store` honours `cap` only on the branch that BUILDS the
        singleton, and nothing on the public path ever tears that singleton
        down: `uninstall_shared_timing` at refcount zero calls `clear()`, which
        empties `_by_fileno` and leaves the module global in place, and
        `reset_shared_timing()` — the one function that nulls it — is test-only,
        with `Runtime.reset()` as its sole caller. `wardex.close()` runs the
        teardown path, which touches neither. So `init(max_connections=A)`,
        `close()`, `init(max_connections=B)` used to keep A for the life of the
        process, and what a host saw was the spurious
        `connect_timing_unavailable` this class's docstring is about — a marker
        that points at no knob.

        No trim. The only way to reach a CHANGED cap is a re-init, and the
        refcount-zero uninstall on the way in already emptied the table; the
        two seams inside ONE init share one config, so they call this with the
        same number. Even if that stopped holding, `_slot`'s own `len(...) >=
        self._cap` check converges on the next connection, while evicting here
        would drop a LIVE connection's slot earlier than the FIFO would.
        """
        self._cap = cap


class ConnTimingProbe:
    """Connection-layer monkeypatch — idempotent, fail-silent."""

    def __init__(self, store: ConnTimingStore) -> None:
        self._store = store
        self._patches = PatchSet("interceptors.conn_timing")
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        # `socket.connect` is patched globally, so every non-TLS socket in the
        # process leaves an entry here too — a Redis client, a Postgres pool, a
        # health check. Nothing pops those (only an emitted span does), so they
        # used to sit in a FIFO-capped table until they pushed a LIVE TLS entry
        # out of it and that connection reported `connect_timing_unavailable`
        # for a measurement wardex had taken correctly.
        #
        # The close hook is the answer the cap was standing in for: a slot is
        # bound to its socket at `connect` and again at TLS-wrap time (the sync
        # handshake, where the SSLSocket that will ASK for the measurement first
        # exists — the plain socket the connect was measured on has already been
        # detached by then and is not the object anyone closes), and released
        # when that socket closes. The cap stays as the backstop.
        self._patch(socket.socket, "connect", self._mk_connect)
        self._patch(ssl.SSLSocket, "do_handshake", self._mk_sync_handshake)
        for cls in _CONNECT_TARGETS:
            self._patch(cls, "create_connection", self._mk_create_connection)
        self._patch(ssl.SSLContext, "wrap_bio", self._mk_wrap_bio)
        self._patch(ssl.SSLObject, "do_handshake", self._mk_async_handshake)
        # Taken LAST, immediately before the flag that authorizes the release.
        # `uninstall()` is gated on `_installed`, so a reference acquired ahead
        # of a patch that then raised would be a refcount this probe can never
        # give back — wardex's wrapper left on `socket.close` for the life of
        # the process, silently, after `wardex.close()`.
        install_shared_close_hook()
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._patches.restore_all()
        uninstall_shared_close_hook()
        self._installed = False

    def _patch(self, cls: type, attr: str, make_wrapper: Any) -> None:
        self._patches.patch(cls, attr, make_wrapper(getattr(cls, attr)))

    def _mk_connect(self, orig: Any):  # noqa: ANN202
        store = self._store

        def wrapper(this: Any, *a: Any, **k: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return orig(this, *a, **k)
            finally:
                try:
                    ms = (time.perf_counter() - t0) * 1000.0
                    fileno = this.fileno()
                    store.set_connect(fileno, ms)
                    _release_at_close(store, this, fileno)
                except Exception:
                    pass

        return wrapper

    def _mk_sync_handshake(self, orig: Any):  # noqa: ANN202
        store = self._store

        def wrapper(this: Any, *a: Any, **k: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return orig(this, *a, **k)
            finally:
                try:
                    ms = (time.perf_counter() - t0) * 1000.0
                    fileno = this.fileno()
                    store.set_handshake(fileno, ms)
                    # The connect was measured on the plain socket, which
                    # `wrap_socket` has already detached; THIS object is the one
                    # the host will close, so the slot is bound to it as well.
                    _release_at_close(store, this, fileno)
                except Exception:
                    pass

        return wrapper

    def _mk_create_connection(self, orig: Any):  # noqa: ANN202
        def wrapper(this: Any, *a: Any, **k: Any):  # async method
            async def run() -> Any:
                rec = _TimingRecord()
                token = _establishing.set(rec)
                t0 = time.perf_counter()
                try:
                    return await orig(this, *a, **k)
                finally:
                    # fail-silent: each statement handles its own exception independently
                    # to avoid masking the original exception
                    try:
                        rec.total_ms = (time.perf_counter() - t0) * 1000.0
                    except Exception:
                        pass
                    try:
                        _establishing.reset(token)
                    except Exception:
                        pass

            return run()

        return wrapper

    def _mk_wrap_bio(self, orig: Any):  # noqa: ANN202
        def wrapper(this: Any, *a: Any, **k: Any) -> Any:
            obj = orig(this, *a, **k)
            try:
                rec = _establishing.get()
                # Even when there's no ContextVar (e.g. stacks like anyio that separate
                # TCP from TLS), stamp an empty record onto the SSLObject so
                # do_handshake instrumentation still activates.
                if rec is None:
                    rec = _TimingRecord()
                obj._wardex_timing = rec
            except Exception:
                pass
            return obj

        return wrapper

    def _mk_async_handshake(self, orig: Any):  # noqa: ANN202
        def wrapper(this: Any, *a: Any, **k: Any) -> Any:
            rec = getattr(this, "_wardex_timing", None)
            # fail-silent: a setup failure in the pre-call step never interferes with the
            # original do_handshake
            try:
                if rec is not None and rec.hs_start == 0.0:
                    rec.hs_start = time.perf_counter()
            except Exception:
                pass
            try:
                return orig(this, *a, **k)
            finally:
                if rec is not None:
                    try:
                        rec.handshake_ms = (time.perf_counter() - rec.hs_start) * 1000.0
                    except Exception:
                        pass

        return wrapper


def _release_at_close(store: ConnTimingStore, obj: Any, fileno: int) -> None:
    """Give this connection's slot back when its socket is closed.

    `on_finalize=False`, and this is the reason that flag exists. The key is a
    FILE DESCRIPTOR, which the kernel reissues the instant it is released — and
    a finalizer runs AFTER the object's deallocator has already closed the fd,
    so a hook that popped by fileno from there could be racing another thread's
    `connect()` and would delete that connection's measurement instead of its
    own. An explicit `close()` has no such window: the hook runs before the real
    close, while the fd is still this socket's.

    A slot nobody releases is not lost data, only a slot; the FIFO cap still
    collects it eventually.
    """
    on_close(obj, partial(store.discard, fileno), on_finalize=False)


# --- Module singleton: shared by the SSL and plaintext seams without double-patching
# socket.connect ---
_shared_store: ConnTimingStore | None = None
_shared_probe: ConnTimingProbe | None = None
_shared_refcount = 0


def shared_timing_store(cap: int | None = None) -> ConnTimingStore:
    """The process-wide store, built on first use and RE-BOUND on every later one.

    `cap=None` means "whatever it already is" for a caller that only wants the
    object (`_socket.py`, `_ssl.py`), and the core default when there is
    nothing yet. A caller that HAS a number is an `install()` carrying a
    host's config, and its number wins — this store survives `close()`, so
    honouring the cap only at construction meant honouring only the first one.
    """
    global _shared_store, _shared_probe
    if _shared_store is None:
        _shared_store = ConnTimingStore(cap)  # cap=None → ConnTimingStore resolves the core default
        _shared_probe = ConnTimingProbe(_shared_store)
    elif cap is not None:
        _shared_store.set_cap(cap)
    return _shared_store


def install_shared_timing(cap: int | None = None) -> None:
    global _shared_refcount
    shared_timing_store(cap)  # ensure the singleton exists
    if _shared_refcount == 0:
        assert _shared_probe is not None
        _shared_probe.install()
    _shared_refcount += 1


def uninstall_shared_timing() -> None:
    global _shared_refcount
    if _shared_refcount == 0:
        return
    _shared_refcount -= 1
    if _shared_refcount == 0 and _shared_probe is not None:
        _shared_probe.uninstall()
        _shared_store.clear()  # type: ignore[union-attr]


def _at_fork_reinit() -> None:
    """Fork-child reset: empty the fileno-keyed store; keep patch and refcount.

    The store's keys are the PARENT's file descriptors: in the child the
    kernel reissues them the moment anything closes, so an inherited slot
    would donate the parent's connect/handshake milliseconds to whatever new
    socket lands on the same fileno. The probe's patch and its refcount stay
    — installation crossed the fork in the memory image and is still exactly
    as installed (I-fork-4); `reset_shared_timing` is NOT reusable here, it
    is the uninstall path and would rip `socket.connect` out from under the
    still-installed seams.

    The store itself is deliberately unlocked (single dict operations, same
    argument as `CloseRegistry`) — but the probe's `PatchSet` is not: its
    lock is held across whole `patch()`/`restore_all()` walks, and the
    child's teardown runs `uninstall_shared_timing()` -> `restore_all()`, so
    an inherited-held lock would hang the child there. Delegate the
    replacement (P/Q/R row Q). Reached by `Runtime.after_in_child` through
    `sys.modules`, so a process that never imported this module resets
    nothing that provably does not exist.
    """
    if _shared_store is not None:
        _shared_store.clear()
    if _shared_probe is not None:
        _shared_probe._patches._at_fork_reinit()


def reset_shared_timing() -> None:
    """Drop the probe, the store and the refcount outright. TEST-ONLY.

    The refcount exists so that two byte seams can share one `socket.connect`
    patch; it is NOT a way to force the probe out, because a seam that never
    took a reference must never be able to release one (a rolled-back install
    that decremented the count restored `socket.connect` out from under the
    seam still using it). This is the other operation — the owner, `Runtime.
    reset()`, saying that no seam is left to hold a reference — so it undoes the
    patch regardless of the count instead of counting down to it.
    """
    global _shared_store, _shared_probe, _shared_refcount
    if _shared_probe is not None:
        _shared_probe.uninstall()
    _shared_store = None
    _shared_probe = None
    _shared_refcount = 0
