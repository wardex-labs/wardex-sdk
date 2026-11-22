"""Hermes Agent: the registration half. What Hermes holds, and what outlives one `init()`.

Owns: `attach(ctx)` (what the public `wardex_sdk.hermes_agent.register` calls), the one callback
per `CONTRACT` row that Hermes holds (built once, at import), `_ROUTE` (the adapter's one seam:
the installed `HermesAgentAdapter`, or None), `_Once` (Hermes's `next_call`, kept apart from
wardex's own work), the process-wide attachment facts `restate` reads, and the `wardex-hermes`
export thread.

Kept apart from `_hermes_agent.py` because its lifetime is Hermes's plugin load, not one `init()`.
Callbacks are registered once per plugin context and never removed by wardex (Hermes removes them
on unload), while the adapter they route to comes and goes with each `init()`. A callback that
finds `_ROUTE` empty does nothing: a hook returns None, a middleware frame calls `next_call()`
and returns its result. So a missing key, a `close()` or a second `init()` leaves Hermes's turns
untouched, and an embedder that calls `wardex_sdk.init()` after Hermes loaded the plugin records
without Hermes reloading anything.

Not on the path of a turn: a reader tracing one turn reads `_hermes_agent_events.py`,
`_hermes_agent_turns.py` and `_hermes_agent.py`.
"""

from __future__ import annotations

import os
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from .._assembly import DropReason, counters, guard, report_once
from .._assembly import record as ledger_record
from ._hermes_agent_events import CONTRACT, HOST_PROCESS_ENV, ContractRow, Surface

if TYPE_CHECKING:
    from ._hermes_agent import HermesAgentAdapter

__all__ = ["HOST_ISOLATION_LINE", "NOT_ATTACHED_DETAIL", "attach", "attached", "refused"]

#: One flush from the export thread waits at most this long for the transport.
EXPORT_TIMEOUT_S = 5.0
#: While a turn is live, the export thread wakes this often to run the idle net. Read at call
#: time, so a test can shorten it.
IDLE_SWEEP_S = 30.0

HOST_ISOLATION_LINE = (
    "wardex: Hermes runs this plugin in its plugin host process (plugins.isolation: host), where "
    "Hermes's model calls and tools are not visible, so wardex records no Hermes turn here. Run "
    "the wardex plugin in process to record turns"
)
NOT_ATTACHED_DETAIL = (
    "Hermes Agent is installed, but no Hermes plugin context has attached wardex in this "
    "process, so no Hermes turn is recorded. Run `hermes plugins enable wardex`. If Hermes runs "
    "plugins in its plugin host process (plugins.isolation: host), wardex cannot record there; "
    "run the plugin in process"
)

#: The route every registered callback reads: the installed adapter, or None. `install` sets it
#: through its context's `PatchSet` and `uninstall` restores it by identity, which is what the
#: conformance suite's seam check measures. Read without a lock: one attribute load.
_ROUTE: HermesAgentAdapter | None = None

#: Frames being delivered on this task, as (kind, frame id). A second delivery of the same frame
#: nested inside the first (wardex registered twice on one chain) only calls `next_call()`.
_DELIVERING: ContextVar[frozenset[tuple[str, str]]] = ContextVar(
    "wardex_hermes_delivering", default=frozenset()
)

#: (weak reference to a plugin context, registrations it accepted). One entry per context, so a
#: repeated `register(ctx)` registers nothing. A list compared by identity, because Hermes's
#: context type may define equality and not hashing.
_ATTACHED: list[tuple[Callable[[], object], int]] = []
#: Contract rows a Hermes context refused, in refusal order. Process-wide: a registration is.
_REFUSED: list[str] = []
#: Guards `_ATTACHED` and `_REFUSED`. Nothing is counted or reported while it is held.
_ATTACH_LOCK = threading.Lock()

#: Emptied by the first attach, which registers the fork reset with it: importing the module
#: registers nothing, and the reset is needed only once something here holds state.
_UNARMED: dict[int, Callable[..., None]] = (
    {0: os.register_at_fork} if hasattr(os, "register_at_fork") else {}
)


# -- what Hermes gets back ---------------------------------------------------------------------


class _Once:
    """Hermes's `next_call`, run at most once, with no argument, and its outcome kept.

    What a middleware callback hands back to Hermes is always `settle()`: Hermes's own result,
    or Hermes's own exception object, whatever wardex's work around the call did. A call wardex
    never made (its own work failed before reaching it) is made by `settle()`, so the frame runs
    exactly once either way.
    """

    __slots__ = ("_next", "called", "raised", "result")

    def __init__(self, next_call: Callable[[], Any]) -> None:
        self._next = next_call
        self.called = False
        self.raised: BaseException | None = None
        self.result: Any = None

    def __call__(self) -> Any:
        if self.called:
            return self.settle()
        self.called = True
        try:
            self.result = self._next()
        except BaseException as exc:
            self.raised = exc
            raise
        return self.result

    def settle(self) -> Any:
        if not self.called:
            return self()
        if self.raised is not None:
            raise self.raised
        return self.result


def _hook_callback(row: ContractRow) -> Callable[..., None]:
    def callback(**payload: Any) -> None:
        adapter = _ROUTE
        if adapter is not None:
            try:
                adapter.on_hook(row, payload)
            except Exception:  # noqa: BLE001 - a hook must not raise into Hermes
                counters.bump("adapters.hermes_agent.hook_escaped")
                ledger_record(DropReason.INTERNAL_SDK_ERROR, where="adapters.hermes_agent.hook")
        # Always None: Hermes injects a `pre_llm_call` return as context, blocks a tool on a
        # `pre_tool_call` one, and shows an `on_session_finalize` one to the user.
        return None

    callback.__name__ = callback.__qualname__ = f"wardex_{row.name}"
    return callback


def _frame_callback(row: ContractRow) -> Callable[..., Any]:
    frame_key = row.record.FRAME_KEY

    def callback(next_call: Callable[[], Any], **payload: Any) -> Any:
        adapter = _ROUTE
        if adapter is None:
            return next_call()
        frame_id = payload.get(frame_key)
        active = _DELIVERING.get()
        fid = (row.name, frame_id)
        if isinstance(frame_id, str) and fid in active:
            return next_call()
        token = _DELIVERING.set(active | {fid})
        once = _Once(next_call)
        try:
            adapter.on_frame(row, once, payload)
        except Exception as exc:
            if exc is once.raised:
                raise
            counters.bump("adapters.hermes_agent.frame_escaped")
            ledger_record(DropReason.INTERNAL_SDK_ERROR, where="adapters.hermes_agent.frame")
        finally:
            _DELIVERING.reset(token)
        return once.settle()

    callback.__name__ = callback.__qualname__ = f"wardex_{row.name}"
    return callback


#: The callback Hermes holds for each row, by row name. Built once, so every registration of a
#: row, from every plugin context, hands Hermes the same function.
_CALLBACKS: Mapping[str, Callable[..., Any]] = {
    row.name: _hook_callback(row) if row.surface is Surface.HOOK else _frame_callback(row)
    for row in CONTRACT
}


# -- the export thread -------------------------------------------------------------------------


class _Exporter:
    """One daemon thread, `wardex-hermes`, started by the first turn that opens.

    A turn's close only sets an Event, so no Hermes thread waits on export. The thread flushes
    when kicked and runs the idle net every IDLE_SWEEP_S while a turn is live; with no turn live
    it waits on the Event alone. It ends once the adapter is uninstalled, and the next turn
    after the next `init()` starts a new one. `drain` is the one flush on a Hermes thread: the
    `on_session_finalize` caller at shutdown, bounded by its timeout.

    Why per turn and not at exit: Hermes leaves its gateway with `os._exit`, so no plugin's
    exit handler runs there, and the CLI's one-shot turns run in that gateway.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._flush: Callable[[float | None], None] | None = None
        self._flush_due = False

    def bind(self, flush: Callable[[float | None], None]) -> None:
        """The public `wardex_sdk.flush`, handed in by the public entry (an adapter module may
        not import the package itself)."""
        self._flush = flush

    def start(self) -> None:
        """Start the thread if none is running, and wake it."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="wardex-hermes", daemon=True)
                self._thread.start()
        self._wake.set()

    def kick(self) -> None:
        """Ask for one flush soon, off the calling thread."""
        self._flush_due = True
        self.start()

    def wake(self) -> None:
        """Let a waiting thread look at the route again (the adapter was uninstalled)."""
        self._wake.set()

    def drain(self, timeout: float) -> None:
        """Flush on the calling thread, bounded by `timeout`. Shutdown only."""
        # Not named `flush`: the limitation census keys helpers by bare name, and the WebSocket
        # tracker's `flush(marker)` would make this argument read as a marker slot.
        send = self._flush
        if send is not None:
            with guard("adapters.hermes_agent.drain"):
                send(timeout)

    def _run(self) -> None:
        me = threading.current_thread()
        while True:
            adapter = _ROUTE
            self._wake.wait(
                IDLE_SWEEP_S if adapter is not None and adapter.has_live_turns() else None
            )
            self._wake.clear()
            with guard("adapters.hermes_agent.export_thread"):
                self._once()
            with self._lock:
                if _ROUTE is None and self._thread is me:
                    self._thread = None
                    return

    def _once(self) -> None:
        send = self._flush
        if self._flush_due and send is not None:
            self._flush_due = False
            send(EXPORT_TIMEOUT_S)
        adapter = _ROUTE
        if adapter is not None:
            adapter.reap_idle(time.monotonic())

    def _at_fork_reinit(self) -> None:
        # The child has no thread, and the parent's lock may have been held at the fork.
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread = None
        self._flush_due = False


_EXPORT = _Exporter()


# -- registration ------------------------------------------------------------------------------


def attach(
    ctx: object,
    *,
    init: Callable[[], None],
    initialized: Callable[[], bool],
    flush: Callable[[float | None], None],
) -> None:
    """Register wardex's callbacks on one Hermes plugin context. Never raises into Hermes.

    1. In Hermes's plugin host process: say HOST_ISOLATION_LINE once and return. Nothing is
       initialized and nothing is registered, because no model call is visible there.
    2. A context already attached: return, so a repeated `register(ctx)` registers once. A
       second profile home is a different context and gets its own registrations.
    3. wardex not initialized: `init()` from the environment. A missing destination leaves
       wardex without adapters, and the callbacks below stay inert until a later `init()`. An
       embedder's earlier `init()` is used as it is.
    4. Hand the export thread `flush`.
    5. One registration per `CONTRACT` row. A refusal (a raise, or Hermes's None after its load
       timed out) is kept for `diagnostics()` and said once.
    6. `ctx.on_unload`: forget the context when Hermes unloads the plugin.
    """
    if os.environ.get(HOST_PROCESS_ENV) == "1":
        report_once(HOST_ISOLATION_LINE, key="adapters.hermes_agent.host_isolation")
        return
    if _UNARMED:
        arm = _UNARMED.pop(0, None)  # one caller gets it, whatever the threads do
        if arm is not None:
            arm(after_in_child=_at_fork_reinit)
    ref = _weak(ctx)
    with _ATTACH_LOCK:
        _ATTACHED[:] = [entry for entry in _ATTACHED if entry[0]() is not None]
        if any(seen() is ctx for seen, _ in _ATTACHED):
            return
        _ATTACHED.append((ref, 0))
    with guard("adapters.hermes_agent.attach_init"):
        try:
            if not initialized():
                init()
        except ValueError as exc:
            report_once(
                f"wardex: init() refused the environment's configuration, so Hermes turns are "
                f"not recorded: {exc}",
                key="adapters.hermes_agent.init_refused",
            )
    _EXPORT.bind(flush)
    accepted = [row.name for row in CONTRACT if _register(ctx, row)]
    refusals = [row.name for row in CONTRACT if row.name not in accepted]
    with _ATTACH_LOCK:
        _ATTACHED[:] = [(r, len(accepted)) if r is ref else (r, n) for r, n in _ATTACHED]
        _REFUSED.extend(name for name in refusals if name not in _REFUSED)
    if refusals:
        ledger_record(
            DropReason.INSTRUMENTATION_UNAVAILABLE,
            n=len(refusals),
            where="adapters.hermes_agent.registration",
        )
        report_once(
            "wardex: Hermes refused the registration of " + ", ".join(refusals) + ", so those "
            "events are not recorded",
            key="adapters.hermes_agent.registration_refused",
        )
    on_unload = getattr(ctx, "on_unload", None)
    if callable(on_unload):
        with guard("adapters.hermes_agent.on_unload"):
            on_unload(_unload_for(ref))


def _weak(ctx: object) -> Callable[[], object]:
    try:
        return weakref.ref(ctx)
    except TypeError:
        # A context type with __slots__ and no __weakref__: held for the life of the process.
        counters.bump("adapters.hermes_agent.context_held_strongly")
        return lambda: ctx


def _register(ctx: object, row: ContractRow) -> bool:
    """One registration. True when Hermes accepted it."""
    accepted = False
    with guard(f"adapters.hermes_agent.register.{row.name}"):
        callback = _CALLBACKS[row.name]
        if row.surface is Surface.HOOK:
            lease = ctx.register_hook(row.name, callback)  # type: ignore[attr-defined]
        else:
            lease = ctx.register_middleware(row.name, callback)  # type: ignore[attr-defined]
        accepted = lease is not None
    return accepted


def _unload_for(ref: Callable[[], object]) -> Callable[[], None]:
    def wardex_unload() -> None:
        with _ATTACH_LOCK:
            _ATTACHED[:] = [entry for entry in _ATTACHED if entry[0] is not ref]

    return wardex_unload


def attached() -> int:
    """Plugin contexts alive in this process that accepted at least one registration."""
    with _ATTACH_LOCK:
        return sum(1 for ref, accepted in _ATTACHED if accepted and ref() is not None)


def refused() -> tuple[str, ...]:
    """Contract rows a Hermes context refused to register, in refusal order."""
    with _ATTACH_LOCK:
        return tuple(_REFUSED)


def _at_fork_reinit() -> None:
    """Fork reset: the lock and the export thread. The attachment records stay, because the
    child's Hermes still holds the registrations they describe, and `_ROUTE` stays, because the
    registry keeps the installed adapter across the fork and resets its turns itself."""
    global _ATTACH_LOCK
    _ATTACH_LOCK = threading.Lock()
    _EXPORT._at_fork_reinit()
