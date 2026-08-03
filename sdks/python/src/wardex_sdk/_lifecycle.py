"""Process-global registrations — atexit, signal chaining, re-init teardown.

Everything that touches interpreter-global state (the signal table, atexit,
the "current client" slot) lives here, isolated from the rest of the SDK
(design §4.3). All exit paths converge on Client.close(), which is idempotent.

Signal policy: our handler only flushes — it never swallows a signal and never
terminates the process itself. Whatever the app had installed (a handler,
default action, or "ignore") happens exactly as before, after our flush.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import threading

from ._client import Client, _UnnamedTimeout
from ._config import WardexConfig
from .assembly import Limitation

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
#: none: `flush_on_signals=False` plus a handler of the host's own is the only
#: lever, which is a documentation matter and not a line printed while the
#: process is being torn down. The fact is not hidden either -- the transport
#: still logs the failed POST under `debug`, at the layer that observed it.
_SIGNAL_FLUSH_TIMEOUT = _UnnamedTimeout(2.0, "<wardex's own signal-flush budget>")

_current_client: Client | None = None
_atexit_registered = False
_signals_installed = False
_prev_handlers: dict[int, object] = {}  # signum → handler we chained over
#: `AdapterRegistry.close_units_all`, resolved at install time. Bound here
#: rather than imported in `_handler` for two reasons: this module reaches
#: `adapters` only inside a function (layering), and a signal can land in the
#: middle of an import and hand the handler a half-initialized module.
_close_units: object | None = None


def current_client() -> Client | None:
    return _current_client


def _teardown(client: Client) -> None:
    """Uninstall interceptors and adapters, then close.

    Uninstall must run first: it flushes pending interceptor state (e.g. WS
    sessions) via capture_span, which close() rejects once _closed is set —
    same ordering wardex.close() uses. Adapters are uninstalled here too, so
    a re-init() (or atexit) doesn't leave a previous adapter bound to the
    closed client — AdapterRegistry.install() is idempotent by name, so a
    stale registration would otherwise make the next init() silently no-op.
    """
    from .adapters._registry import get_registry as get_adapter_registry
    from .interceptors._registry import get_registry as get_interceptor_registry

    get_interceptor_registry().uninstall_all()
    get_adapter_registry().uninstall_all()
    client.close()


def install(client: Client, config: WardexConfig) -> None:
    """Make `client` the process-wide client; tear down the previous one."""
    global _current_client, _atexit_registered
    previous = _current_client
    if previous is not None:
        _teardown(previous)  # uninstall interceptors, flush remainder, join worker
    _current_client = client
    if not _atexit_registered:
        atexit.register(_atexit_handler)
        _atexit_registered = True
    if config.flush_on_signals:
        _install_signal_handlers(debug=config.debug)
    else:
        _uninstall_signal_handlers()


def _atexit_handler() -> None:
    client = _current_client
    if client is not None:
        _teardown(client)


def _handler(signum: int, frame: object) -> None:
    client = _current_client
    if client is not None:
        if _prev_handlers.get(signum) is signal.SIG_DFL and _close_units is not None:
            # Close live units BEFORE the flush, or the flush has nothing of
            # them to send. Gated on SIG_DFL because that is exactly the
            # disposition where the process ends in `_handler` below, via
            # `os.kill`, and atexit does NOT run — measured for SIGTERM. Any
            # other disposition either exits through the interpreter (atexit
            # runs, and the adapter uninstall closes the units there) or keeps
            # the program running, and closing every live unit under a program
            # that carries on would truncate sessions still being driven.
            #
            # No `try` around it, and that is not an oversight: `close_units_all`
            # guards every adapter individually with the one sanctioned swallow
            # (I6), so a failing adapter is already counted and already cannot
            # reach the flush below. Adding a second net here would add an
            # unsanctioned one and hide nothing that is not already caught.
            _close_units(marker=Limitation.UNIT_INTERRUPTED)
        try:
            client.flush(timeout=_SIGNAL_FLUSH_TIMEOUT)
        except Exception:
            pass  # a failed flush must never block the chain to the app's handler
    prev = _prev_handlers.get(signum)
    if callable(prev):
        prev(signum, frame)
    elif prev == signal.SIG_DFL:
        # Re-raise so the default action (and the exit code) stay unchanged.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    # SIG_IGN: respect the app's decision to ignore the signal.


def _install_signal_handlers(*, debug: bool) -> None:
    global _signals_installed, _close_units
    if _signals_installed:
        return
    from .adapters._registry import get_registry as get_adapter_registry

    _close_units = get_adapter_registry().close_units_all
    if threading.current_thread() is not threading.main_thread():
        if debug:
            print(
                "[wardex] signal handlers skipped (init() not on main thread)",
                file=sys.stderr,
            )
        return
    try:
        for signum in _SIGNALS:
            _prev_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handler)
    except (ValueError, OSError) as exc:  # exotic embedding/platform — never crash init
        # Roll back any handler we already installed this attempt.
        for signum, prev in _prev_handlers.items():
            if signal.getsignal(signum) is _handler:
                signal.signal(signum, prev)
        _prev_handlers.clear()
        if debug:
            print(f"[wardex] signal handlers skipped ({exc})", file=sys.stderr)
        return
    _signals_installed = True


def _uninstall_signal_handlers() -> None:
    global _signals_installed
    if not _signals_installed:
        _prev_handlers.clear()
        return
    for signum, prev in _prev_handlers.items():
        if signal.getsignal(signum) is _handler:  # only restore if still ours
            signal.signal(signum, prev)
    _prev_handlers.clear()
    _signals_installed = False
