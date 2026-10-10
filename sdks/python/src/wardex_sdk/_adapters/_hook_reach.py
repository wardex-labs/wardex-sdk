"""Whether wardex's observation hooks reached a Claude Agent SDK session.

The adapter attaches its hooks by wrapping `claude_agent_sdk.query` and
`ClaudeSDKClient.__init__`. A host that ran `from claude_agent_sdk import
query` before `wardex.init()` holds the function as it was, so the sessions it
starts carry none of them — while the transport tee, patched on the CLASS,
still records each one from the CLI's output. Nothing fails, so until this
module nothing said so: sub-agents got no span of their own, hook inputs and
the prompt fallback were gone, and the run looked complete.

The evidence is on the wire. A session opens with an `initialize` control
request that lists every hook event registered for it, written BEFORE the
first user message — and the session's root opens on that message, not on the
handshake. So a handshake missing wardex's events waits here, per transport,
until the root exists; then the root carries `INSTRUMENTATION_DEGRADED`, the
session is counted, and the cause and its fix are reported once per process.

A false negative is accepted rather than guessed around: a host that
registers all six of wardex's events itself sends a handshake identical to one
that carries wardex's. The reverse cannot happen — an option set the adapter
prepared always carries every one of them.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable
from typing import Any

from .._assembly import Limitation, counters, report_once

#: Handshakes waiting for their root. One waits only from the `initialize`
#: write to the transport's next write (the user message that opens the
#: session) or its close, which the SDK issues back to back; the bound is for a
#: host that connects and then does neither.
_MAX_WAITING = 64

_NOTICE = (
    "anthropic_agent_sdk adapter: a Claude Agent SDK session started without wardex's hooks, so "
    "it is recorded from the CLI's output alone: sub-agents get no span of their own and the "
    "root carries instrumentation_degraded. The usual cause is `from claude_agent_sdk import "
    "query` running before wardex.init(); call init() first, or call claude_agent_sdk.query(...) "
    "through the module"
)


def handshake_lacks(data: str, events: frozenset[str]) -> bool | None:
    """For an outbound `initialize` control request, whether it lacks any of
    `events`; None for every other line. Nearly every line is another one,
    hence the substring test in front of the parse."""
    if '"initialize"' not in data:
        return None
    try:
        msg = json.loads(data)
    except ValueError:
        # A write that names `initialize` and is not JSON: no SDK writes one, so it is counted.
        counters.bump("adapters.anthropic.handshake_unreadable")
        return None
    if not isinstance(msg, dict) or msg.get("type") != "control_request":
        return None
    request = msg.get("request")
    if not isinstance(request, dict) or request.get("subtype") != "initialize":
        return None
    hooks = request.get("hooks")
    return not events <= set(hooks if isinstance(hooks, dict) else ())


class HookReach:
    """The handshakes that lacked wardex's hooks, held until their root opens."""

    def __init__(self, events: Iterable[str]) -> None:
        self._events = frozenset(events)
        self._waiting: dict[int, None] = {}
        self._lock = threading.RLock()

    def observe(self, key: int, data: str, assembler: Any) -> None:
        """Called after the assembler has seen this write: file a hook-less
        handshake, or mark the root one was waiting for once it is open."""
        lacks = handshake_lacks(data, self._events)
        with self._lock:
            if lacks is not None:
                self._waiting.pop(key, None)
                if lacks:
                    if len(self._waiting) >= _MAX_WAITING:
                        del self._waiting[next(iter(self._waiting))]
                        counters.bump("adapters.anthropic.hooks_absent_unmarked")
                    self._waiting[key] = None
                return
            if key not in self._waiting:
                return
            unit = assembler.unit_for(key)
            if unit is None:
                return
            del self._waiting[key]
        unit.note(Limitation.INSTRUMENTATION_DEGRADED)
        counters.bump("adapters.anthropic.hooks_absent")
        report_once(_NOTICE, key="adapters.anthropic_agent_sdk.hooks_absent")

    def forget(self, key: int) -> None:
        """The transport closed: a handshake still waiting never got a root."""
        with self._lock:
            self._waiting.pop(key, None)

    def clear(self) -> None:
        """Uninstall: no root this adapter would mark can open any more."""
        with self._lock:
            self._waiting.clear()

    def _at_fork_reinit(self) -> None:
        """Fork child: the lock is REPLACED, never acquired; the parent's handshakes are its own."""
        self._lock = threading.RLock()
        self._waiting = {}
