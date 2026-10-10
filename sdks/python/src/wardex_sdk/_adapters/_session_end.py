"""Where a Claude Agent SDK session ends: at its reader, at its transport's close, or late.

A session with the OTel bridge has to end AFTER the CLI has exited, because the CLI flushes its
last OTel export when its stdin closes, and the SDK closes stdin inside `transport.close()`. A merge
that runs before that close misses the batch, which then waits in a receiver slot that no session
claims. `ClaudeSDKClient.disconnect()` cancels the SDK's reader BEFORE it closes the transport, so
a bridge session whose reader stops is handed over to the close, which drains and merges it after
the SDK's own close has run. A session without the bridge has nothing to wait for: it ends at its
reader, or before the SDK's close, exactly where it always has.

A handover can fail in two ways, and each has a deadline. A host that drops a client without
`disconnect()` stops its reader at loop teardown, and no close follows. A host that breaks out of
`query()` just before the loop ends starts a close that the loop then abandons. A session waits here
until its deadline: `_HANDOFF_S` from its reader stopping, `_CLOSE_BUDGET_S` from its close
starting, and the drain's own budget plus `_HANDOFF_S` once the SDK's close has returned. If nothing
has ended it by then, the receiver's serve loop (`reap`, roughly twice a second) ends it with
whatever arrived and counts the late close. Every end goes through `_on_close`, which closes a
session once and ignores later calls, so the reaper and a slow close racing each other cannot
finalize a session twice. A close that is cancelled (`GeneratorExit`, when its coroutine is dropped)
skips the drain, because nothing may be awaited then, and still ends the session.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Awaitable
from typing import Any

from .._assembly import counters
from ._session_outcome import reader_stopped

#: From a reader stopping to its transport's close starting. The SDK makes that call right after
#: the cancel, in the same shielded `Query.close()`, so a close that is not under way by then is
#: not coming.
_HANDOFF_S = 1.0

#: From a bridge session's close starting to its end. Longer than `SubprocessCLITransport.close()`'s
#: own bounded worst case (lock, graceful exit, SIGTERM and SIGKILL waits of 5 s each).
_CLOSE_BUDGET_S = 30.0


class SessionEnd:
    """The handover table between a session's reader, its transport's close and the reaper."""

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter
        #: Transport key -> monotonic deadline, for bridge sessions whose end is pending.
        self._due: dict[int, float] = {}
        self._lock = threading.RLock()

    def _bridged(self, key: int) -> bool:
        adapter = self._adapter
        assembler = adapter._assembler
        if adapter._bridge is None or assembler is None:
            return False
        return assembler.bridge_route(key) is not None

    def _end(self, key: int) -> bool:
        """End the session, whoever gets here first. True when it was still pending here."""
        with self._lock:
            pending = self._due.pop(key, None) is not None
        with self._adapter._guard("adapters.anthropic.transport_close"):
            self._adapter._on_close(key, None)
        return pending

    def reader_ended(self, key: int, exc: BaseException) -> None:
        """The tee's reader stopped (`reader_stopped`: no failure) or failed (an error root)."""
        stopped = reader_stopped(exc)
        if stopped and self._bridged(key):
            with self._lock:
                self._due.setdefault(key, time.monotonic() + _HANDOFF_S)
            return
        self._adapter._on_close(key, None if stopped else repr(exc))

    async def transport_closed(self, key: int, closing: Awaitable[Any]) -> Any:
        """The transport's close. For a bridge session, `closing` (the SDK's own close) goes first,
        then the drain, then the merge, in nested `finally` blocks."""
        adapter = self._adapter
        if not self._bridged(key):
            self._end(key)
            return await closing
        with self._lock:
            self._due[key] = time.monotonic() + _CLOSE_BUDGET_S
        abandoned = False
        try:
            return await closing
        except GeneratorExit:
            abandoned = True
            raise
        finally:
            with self._lock:
                if key in self._due:  # the SDK's close is over: only the drain is left to wait
                    self._due[key] = time.monotonic() + adapter._drain_seconds + _HANDOFF_S
            try:
                if not abandoned:
                    with adapter._guard("adapters.anthropic.otel_bridge_drain"):
                        await adapter._drain_bridge(key)
            finally:
                self._end(key)

    def reap(self) -> None:
        """On the bridge receiver's serve loop: end every session whose deadline passed."""
        now = time.monotonic()
        with self._lock:
            late = [key for key, deadline in self._due.items() if deadline <= now]
        for key in late:
            if self._end(key):
                counters.bump("adapters.anthropic.otel_bridge.session_ended_late")

    def clear(self) -> None:
        """Uninstall: `close_all_sessions` ends whatever was still pending here."""
        with self._lock:
            self._due.clear()

    def _at_fork_reinit(self) -> None:
        """Fork child: the lock is REPLACED, never acquired; the parent's sessions are its own."""
        self._lock = threading.RLock()
        self._due = {}
