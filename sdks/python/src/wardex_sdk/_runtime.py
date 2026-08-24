"""The one owner of everything wardex installs into the host process.

Every piece of interpreter-global state the SDK holds — the signal table,
atexit, the current client, the interceptor and adapter registries, and the
shared connection-timing probe — belongs to the single `Runtime` below
(design §4.3). Nothing else in the SDK keeps a mutable module global that
outlives a call.

WHY ONE OWNER. Those five things used to be five module globals spread over
four modules, and the teardown that undoes them was written twice — once in
`wardex.close()`, once here — with the order-sensitive comments duplicated on
both sides. Two copies of an ordering is an ordering that holds only until
someone edits one of them. Worse, no copy could see the other's state:
`reset_for_test()` reset one global out of five, so a test that installed an
interceptor charged every test behind it for the leak, and removing a feature
meant auditing four modules to find the pieces of it.

There is one install order and one uninstall order now, and both live on the
Runtime:

    install    client → atexit → signal handlers → interceptors → adapters
               → propagation
    uninstall  interceptors → adapters → propagation → client.close()

The uninstall order is not a preference. Interceptor uninstall flushes pending
WebSocket sessions through `client.capture_span`, which `client.close()`
refuses once `_closed` is set — an uninstall running after the close would
build every one of those spans and drop each on the floor. Adapters are
uninstalled here rather than left to the next `init()` for a different reason:
`AdapterRegistry.install` is idempotent BY NAME, so a stale registration makes
the next `init()` silently no-op for that adapter, forever.

Signal policy: our handler only flushes — it never swallows a signal and never
terminates the process itself. Whatever the app had installed (a handler,
default action, or "ignore") happens exactly as before, after our flush.

THE LOCK IS REENTRANT, and not out of habit. `install()` holds it and then
calls `_teardown()` and `_install_signal_handlers()`, which take it again on
the same thread; a plain `Lock` would deadlock the SDK's own `init()`. A
signal handler is the second re-entrant path — it can land on the thread that
already holds the lock, between any two bytecodes. A weakref finalizer is the
third: the byte seams' close hook registers one per connection, and CPython runs
such a callback out of the referent's DEALLOCATION — so SDK code can now
re-enter wherever a reference count reaches zero (and at any allocation, via a
cyclic collection), on ANY thread rather than only the main one. Neither of
those two reaches this lock today. The self-recursion does, and it is what makes
the reentrancy a property of the lock rather than a standing claim about who
calls it.

THE CLIENT IS READ WITHOUT THE LOCK, deliberately. `get_client()` sits on the
capture hot path and on the teardown path, and a reader that can block behind
a shutdown holding the lock is a reader that can stall the host application.
An attribute read is atomic; writers serialize among themselves, which is what
the lock is for.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

from ._assembly import Limitation, counters, diag_info, guard
from ._assembly._diag import diag_reset_for_new_process
from ._client import Client, _UnnamedTimeout

if TYPE_CHECKING:
    from ._adapters._registry import AdapterRegistry
    from ._config import WardexConfig
    from ._interceptors._registry import InterceptorRegistry

_SIGNALS = (signal.SIGINT, signal.SIGTERM)

#: Short bound — never delay shutdown (design §4.3).
#:
#: An `_UnnamedTimeout` and not a bare `2.0`, because the number is WARDEX'S and
#: the difference is load-bearing downstream. `Client.flush` cannot see who
#: called it; it reads whose number this is off the value. As a bare float this
#: reached the transport as a budget "the caller passed", and a host that hit
#: Ctrl-C against a slow backend was told on stderr that its own 2.0s budget had
#: cut an export short and that it should "pass a larger timeout" -- an
#: accusation about a number no host can pass, advising a knob that does not
#: exist. Worse, that line is one per key per process, so it burned the key and
#: silenced the report for a caller who later really did cut one short.
#:
#: It is still a real 2s bound; wearing this type changes nothing about how long
#: the handler waits, only about whom a cut-off export is attributed to.
#:
#: And a stalled backend at SIGTERM stays SILENT on this channel, deliberately.
#: The report exists to hand someone a number they can change, and here there is
#: none: `batching=BatchingConfig(flush_on_signals=False)` plus a handler of the
#: host's own is the only
#: lever, which is a documentation matter and not a line printed while the
#: process is being torn down. The fact is not hidden either -- the transport
#: still logs the failed POST under `debug`, at the layer that observed it.
_SIGNAL_FLUSH_TIMEOUT = _UnnamedTimeout(2.0, "<wardex's own signal-flush budget>")


def _handler(signum: int, frame: object) -> None:
    """What the signal table holds. A module function, never a bound method.

    The uninstall path restores a disposition only while `signal.getsignal()`
    still returns THIS object ("only put it back if it is still ours"), and a
    bound method is a fresh object on every attribute access — that identity
    check would read False forever, and wardex would never take its own handler
    back out of the host's signal table.
    """
    runtime().handle_signal(signum, frame)


def _after_in_child() -> None:
    """What `os.register_at_fork(after_in_child=...)` holds — CPython runs it
    in every forked child, after the fork, before `fork()` returns there.

    A module function holding a STRONG route to the runtime, and deliberately
    not OTel's `WeakMethod` shape: `_RUNTIME` is an immortal singleton this
    module owns, so there is nothing for a weak reference to protect against,
    and a weakly-held hook that silently went dead would be the duplicate-
    export bug coming back with no line of code to find it by.
    """
    _RUNTIME.after_in_child()


class Runtime:
    """Every process-global thing wardex owns, and the order it owns it in.

    One instance per interpreter (`runtime()`). `reset()` empties it IN PLACE
    rather than replacing it, so a registry reference taken before a reset is
    still the live one afterwards — a replaced registry would silently orphan
    every interceptor installed against the old one.
    """

    __slots__ = (
        "_adapters",
        "_atexit_registered",
        "_client",
        "_close_units",
        "_fork_hooks_registered",
        "_fork_reinit_us",
        "_interceptors",
        "_lock",
        "_prev_handlers",
        "_signals_installed",
    )

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._client: Client | None = None
        self._interceptors: InterceptorRegistry | None = None
        self._adapters: AdapterRegistry | None = None
        self._atexit_registered = False
        self._signals_installed = False
        self._fork_hooks_registered = False
        #: How long the last `after_in_child` took in THIS process, in µs.
        #: Diagnostics only (the fork tests assert an upper bound on it);
        #: 0 means "this process is not a fork child".
        self._fork_reinit_us = 0
        #: signum → the disposition wardex chained over
        self._prev_handlers: dict[int, Any] = {}
        #: `AdapterRegistry.close_units_all`, bound at signal-install time.
        #: Bound rather than looked up inside `handle_signal`, because a signal
        #: can land in the middle of an import and hand the handler a
        #: half-initialized module.
        self._close_units: Any = None

    # -- the client ---------------------------------------------------------

    @property
    def client(self) -> Client | None:
        """The process client. Read WITHOUT the lock — see the module docstring."""
        return self._client

    def set_client(self, client: Client | None) -> None:
        with self._lock:
            self._client = client

    # -- the registries -----------------------------------------------------
    #
    # Built on first use rather than at import, and reached through the runtime
    # rather than through a module global of their own. Both halves matter: the
    # lazy build keeps `_interceptors/` and `_adapters/` — which reach the native
    # extension at import time — off the paths that must work without it, and
    # single ownership is what lets `reset()` drain all of them.

    @property
    def interceptors(self) -> InterceptorRegistry:
        with self._lock:
            if self._interceptors is None:
                from ._interceptors import _registry

                self._interceptors = _registry.InterceptorRegistry()
            return self._interceptors

    @property
    def adapters(self) -> AdapterRegistry:
        with self._lock:
            if self._adapters is None:
                from ._adapters import _registry

                self._adapters = _registry.AdapterRegistry()
            return self._adapters

    # -- install ------------------------------------------------------------

    def install(self, client: Client, config: WardexConfig) -> None:
        """Make `client` the process client and install what `config` asks for.

        Tears the previous client down first, in full: `init()` may be called
        twice, and the second call must not leave the first one's interceptors
        bound to a closed client.
        """
        with self._lock:
            previous = self._client
            if previous is not None and previous is not client:
                self._teardown(previous)
            self._client = client
            if not self._atexit_registered:
                atexit.register(self._at_exit)
                self._atexit_registered = True
            self._register_fork_hooks()
            if config.batching.flush_on_signals:
                self._install_signal_handlers(debug=config.debug)
            else:
                self._uninstall_signal_handlers()

            from ._interceptors import install_configured_interceptors

            install_configured_interceptors(client, config)

            from ._adapters import install_configured_adapters

            install_configured_adapters(client, config)

            from .context._inject import install_propagation, uninstall_propagation

            # A re-init after a plain `close()` reaches here with the previous
            # run's propagation patches still in place and no previous CLIENT to
            # have carried them out, so the drop is unconditional.
            uninstall_propagation()
            if config.propagation.enabled:
                install_propagation()

    # -- uninstall ----------------------------------------------------------

    def teardown(self, *, timeout: float | None = None) -> None:
        """Uninstall everything and close the process client. Idempotent.

        The one implementation `wardex.close()`, `atexit` and re-init all reach.
        """
        with self._lock:
            self._teardown(self._client, timeout=timeout)

    def _teardown(self, client: Client | None, *, timeout: float | None = None) -> None:
        """THE uninstall order, run against `client`. Caller holds the lock.

        Takes the client as an argument rather than reading the slot, because
        re-init tears down the client being REPLACED while the slot still holds
        it and the new one is not installed yet.

        A registry that was never built is skipped rather than built to be
        drained: an empty registry uninstalls nothing, and constructing one here
        would import `_interceptors/` on a teardown path that may be running
        precisely because the native extension is absent.

        `timeout=None` means "the client's own shutdown default" — which
        `Client.close` resolves from its config's `batching.shutdown_timeout`
        — and it is expressed by NOT passing one. Restating a number here
        would put the same budget in two places, and `Client.close`
        distinguishes a budget wardex picked from one a host named by the
        value's TYPE — a default re-stated at this call site is
        indistinguishable from a host's, and gets the host blamed for an
        export it never bounded.
        """
        if self._interceptors is not None:
            self._interceptors.uninstall_all()
        if self._adapters is not None:
            self._adapters.uninstall_all()
        from .context._inject import uninstall_propagation

        uninstall_propagation()
        if client is None:
            return
        if timeout is None:
            client.close()
        else:
            client.close(timeout)

    def _at_exit(self) -> None:
        client = self._client
        if client is not None:
            self.teardown()

    # -- fork ---------------------------------------------------------------

    def _register_fork_hooks(self) -> None:
        """Register the child-side fork hook. Once per process, ever.

        `after_in_child` ONLY. No `before`, no `after_in_parent` — and that is
        a closed decision, not an omission (I-fork-2): fork changes nothing in
        the parent, so a `before` hook has nothing to protect, and one that
        took SDK locks could deadlock the host's own `os.fork()` against the
        existing buffer→spawn lock ordering (`capture_span` inside
        `_buffer_lock` reaches `ensure_alive`'s blocking `_spawn_lock`; a
        before hook acquiring spawn→buffer is that pair reversed). uWSGI's
        C-level fork also runs ONLY the child hook, so anything hung off the
        other two would silently not exist in the deployment this work
        targets. The host's `os.fork()` never waits on wardex.

        The registration is IRREVERSIBLE — `os.register_at_fork` has no
        unregister — which is why `reset()` keeps `_fork_hooks_registered`:
        clearing it would stack one more registration per init/reset cycle for
        the life of the pytest process. The hook itself tolerates any state
        (`after_in_child` skips empty slots), so staying registered is safe.

        Caller holds the lock (`install()`).
        """
        if self._fork_hooks_registered or not hasattr(os, "register_at_fork"):
            return
        os.register_at_fork(after_in_child=_after_in_child)
        self._fork_hooks_registered = True

    def after_in_child(self) -> None:
        """Child-side fork reset — the ONE piece of SDK code that runs at fork.

        The child inherits every table, buffer, lock and thread *record* of
        the parent, and none of the parent's threads. Everything process-global
        is therefore either replaced or emptied here, under one rule per kind:

        * locks are REPLACED, never acquired (I-fork-5) — any inherited lock
          can be held by a thread that does not exist in this process;
        * buffered/pending data is DISCARDED, never emitted (I-fork-3) — the
          parent owns it and the parent exports it; a child that shipped its
          copy is the N+1 duplication this hook removes;
        * patches and install records are KEPT (I-fork-4) — the monkeypatches
          crossed the fork in the memory image and are still in effect, and
          re-installing over them would double-wrap the host.

        STEP 0 RUNS BEFORE ANY `guard()`: `guard.__exit__` bumps `counters`,
        so entering one while `Counters._lock` is still the parent's — copied
        mid-`bump()`, say — would hang the child inside the very hook that
        exists to remove inherited-lock hangs. Step 0 is reassignments and
        `dict.clear()` only; none of it can raise.

        Steps read the SLOTS directly, never the `interceptors`/`adapters`
        properties: those take `self._lock` and BUILD a missing registry
        (importing `_interceptors/` on the way), and `_teardown`'s rule holds
        here too — a registry that was never built is skipped, not built to be
        reset. A `None` client is skipped the same way, which also covers
        fork-before-init and fork-after-close.

        Failures are isolated per step (each `with guard(...)`) and counted
        under `*.fork_reinit_failed`; CPython would only write the exception
        as unraisable and carry on anyway, so the guard buys isolation and a
        counter, not survival. Worst case the lazy-respawn PID check
        (`BatchWorker.is_alive`) still self-heals the worker on the next
        capture — that check stays, as the backstop for platforms where this
        hook never ran at all.

        `process.pid` needs nothing here: the drain stamps it live, so it is
        correct in every process without any fork-time work.
        """
        started = time.perf_counter()
        # step 0 — replace the locks the hook itself would otherwise step on.
        self._lock = threading.RLock()
        diag_reset_for_new_process()
        with guard("_runtime.fork_reinit_failed"):
            # step 1 — slots, directly.
            client = self._client
            # step 2 — client: locks, buffer, worker.
            if client is not None:
                with guard("client.fork_reinit_failed"):
                    client._at_fork_reinit()
            # step 3 — interceptors: per-connection tracking through the
            # registry (patches stay installed), then the shared module
            # singletons — the timing store's parent filenos, the close
            # registry's parent object ids, and the propagation module's own
            # lock (`uninstall_propagation()` is unconditionally on the
            # child's teardown path, so that lock inherited held would hang
            # atexit). All three module holders are reached through
            # `sys.modules`: never imported means nothing to reset.
            with guard("interceptors.fork_reinit_failed"):
                if self._interceptors is not None:
                    self._interceptors._at_fork_reinit()
                _fork_reinit_module("_interceptors._conn_timing")
                _fork_reinit_module("_interceptors._close_hook")
                _fork_reinit_module("context._inject")
            # step 4 — adapters: sessions/units dropped without emitting, the
            # inherited bridge receiver torn down the child-safe way. Then the
            # host's transport gets its say: wardex cannot reset a third-party
            # transport's internals (a pooled requests.Session shares TCP
            # sockets with the parent — that library's own fork problem), so
            # the extension point is a duck-typed `at_fork_child()` the
            # transport may implement; ours are stateless and don't.
            with guard("adapters.fork_reinit_failed"):
                if self._adapters is not None:
                    self._adapters._at_fork_reinit()
            if client is not None:
                with guard("transport.fork_reinit_failed"):
                    at_fork_child = getattr(client._transport, "at_fork_child", None)
                    if at_fork_child is not None:
                        at_fork_child()
            # step 5 — a multiprocessing fork child leaves through os._exit,
            # where atexit never runs; its tail rides util.Finalize instead.
            # Registered here (per child) because the registration answers a
            # per-process question — a grandchild registers its own.
            if client is not None:
                with guard("client.tail_flush.fork_reinit_failed"):
                    client._register_mp_tail_flush()
            # step 6 — after the resets, so this survives them.
            counters.bump("_runtime.fork_child_reinit")
        self._fork_reinit_us = int((time.perf_counter() - started) * 1e6)

    def reset(self) -> None:
        """Undo everything and forget it. TEST-ONLY.

        Every state this object owns, not one of them. The predecessor reset the
        client slot alone, so an interceptor, an adapter, a chained signal
        handler or a raised timing refcount left behind by one test was charged
        to whichever test ran next — the failure landing in a file that had done
        nothing wrong.
        """
        with self._lock:
            self._teardown(self._client)
            self._uninstall_signal_handlers()
            self._close_units = None
            self._client = None
            _reset_shared_timing()

    # -- signals ------------------------------------------------------------

    def handle_signal(self, signum: int, frame: object) -> None:
        client, prev, close_units = self._signal_state(signum)
        if client is not None:
            if prev is signal.SIG_DFL and close_units is not None:
                # Close live units BEFORE the flush, or the flush has nothing of
                # them to send. Gated on SIG_DFL because that is exactly the
                # disposition where the process ends in this handler, via
                # `os.kill`, and atexit does NOT run — measured for SIGTERM. Any
                # other disposition either exits through the interpreter (atexit
                # runs, and the adapter uninstall closes the units there) or
                # keeps the program running, and closing every live unit under a
                # program that carries on would truncate sessions still being
                # driven.
                #
                # No `try` around it, and that is not an oversight:
                # `close_units_all` guards every adapter individually with the
                # one sanctioned swallow (I6), so a failing adapter is already
                # counted and already cannot reach the flush below. Adding a
                # second net here would add an unsanctioned one and hide nothing
                # that is not already caught.
                close_units(marker=Limitation.UNIT_INTERRUPTED)
            try:
                # The shutdown arm of flush, not the public one: the process
                # dies in this handler (SIG_DFL is re-raised below), so
                # pending deferred parses are drained in FALLBACK — parsed
                # inside half the budget, shipped as PARSE_SKIPPED_AT_SHUTDOWN
                # after — and the export keeps a floor of the other half. A
                # plain flush() would KEEP the leftover, and a kept job here
                # is a job the process takes down with it.
                client._shutdown_flush(_SIGNAL_FLUSH_TIMEOUT)
            except Exception:
                pass  # a failed flush must never block the chain to the app's handler
        if callable(prev):
            prev(signum, frame)
        elif prev == signal.SIG_DFL:
            # Re-raise so the default action (and the exit code) stay unchanged.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        # SIG_IGN: respect the app's decision to ignore the signal.

    def _signal_state(self, signum: int) -> tuple[Client | None, Any, Any]:
        """What the handler needs, read under the lock when the lock is free.

        NON-BLOCKING, and that is the whole design of this method. A signal
        handler runs on the main thread; if `init()` or `close()` is running on
        another one, blocking here would make Ctrl-C wait for a shutdown to
        finish — the SDK stalling the host application at the exact moment the
        host asked it to stop. The reads below are single attribute and dict
        lookups either way, so the uncontended case gets a consistent snapshot
        and the contended one still answers.
        """
        acquired = self._lock.acquire(blocking=False)
        try:
            return self._client, self._prev_handlers.get(signum), self._close_units
        finally:
            if acquired:
                self._lock.release()

    def _install_signal_handlers(self, *, debug: bool) -> None:
        with self._lock:
            if self._signals_installed:
                return
            self._close_units = self.adapters.close_units_all
            if threading.current_thread() is not threading.main_thread():
                if debug:
                    diag_info("signal handlers skipped (init() not on main thread)")
                return
            try:
                for signum in _SIGNALS:
                    self._prev_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, _handler)
            except (ValueError, OSError) as exc:  # exotic embedding — never crash init
                # Roll back any handler we already installed this attempt.
                for signum, prev in self._prev_handlers.items():
                    if signal.getsignal(signum) is _handler:
                        signal.signal(signum, prev)
                self._prev_handlers.clear()
                if debug:
                    diag_info(f"signal handlers skipped ({exc})")
                return
            self._signals_installed = True

    def _uninstall_signal_handlers(self) -> None:
        with self._lock:
            if not self._signals_installed:
                self._prev_handlers.clear()
                return
            for signum, prev in self._prev_handlers.items():
                if signal.getsignal(signum) is _handler:  # only restore if still ours
                    signal.signal(signum, prev)
            self._prev_handlers.clear()
            self._signals_installed = False


def _reset_shared_timing() -> None:
    """Drop the refcounted connection-timing probe the byte seams share.

    Reached through `sys.modules` rather than by importing it, and the guard is
    exact rather than defensive: a module nobody has imported cannot be holding
    a probe, and importing `_interceptors/` from a reset path would reach the
    native extension for a state that provably does not exist.
    """
    module = sys.modules.get(f"{__package__}._interceptors._conn_timing")
    if module is not None:
        module.reset_shared_timing()


def _fork_reinit_module(name: str) -> None:
    """Run one module-global holder's `_at_fork_reinit`, iff it was imported.

    `_reset_shared_timing`'s rule, applied to the fork path: a module nobody
    imported holds pristine, unheld locks and empty tables — there is nothing
    to reset — and importing it from inside a fork hook would run arbitrary
    import-time code (some of it reaching the native extension) at the one
    moment the process should be doing reassignments and nothing else.
    """
    module = sys.modules.get(f"{__package__}.{name}")
    if module is not None:
        module._at_fork_reinit()


_RUNTIME = Runtime()


def runtime() -> Runtime:
    """The process runtime. See `Runtime` for why it is never replaced."""
    return _RUNTIME


def current_client() -> Client | None:
    return _RUNTIME.client
