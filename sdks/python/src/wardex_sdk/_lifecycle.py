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

from ._client import Client
from ._config import WardexConfig

_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_SIGNAL_FLUSH_TIMEOUT = 2.0  # short bound — never delay shutdown (design §4.3)

_current_client: Client | None = None
_atexit_registered = False
_signals_installed = False
_prev_handlers: dict[int, object] = {}  # signum → handler we chained over


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
    global _signals_installed
    if _signals_installed:
        return
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
