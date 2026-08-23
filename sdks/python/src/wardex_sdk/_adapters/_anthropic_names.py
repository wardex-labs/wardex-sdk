"""Tool-name reconciliation for the Anthropic Agent SDK — design §5.4.

One in-process tool call is seen by two observers under two DIFFERENT strings,
and unless both land on the same key `Unit.claim()` cannot arbitrate between
them — it becomes a duplicate generator instead of an arbiter. This module is
that single key space.

  * The in-process HANDLER sees the bare name the `@tool` decorator gave it
    (`greet`). The Python SDK does no namespacing at all: `mcp__` appears zero
    times in `claude_agent_sdk` 0.2.122.
  * The HOOK payload carries the name the CLI built:
    ``mcp__{sanitize(server)}__{sanitize(tool)}``, where `server` is the KEY of
    the user's `mcp_servers` dict — NOT `create_sdk_mcp_server(name=...)` — and
    `sanitize` maps every character outside ``[A-Za-z0-9_-]`` to ``_``.

Three consequences shape everything below.

1. **The server token is not knowable at wrap time.** The dict it comes from
   does not exist yet when `create_sdk_mcp_server` runs. So a `ServerHandle` is
   created with no token and filled in later, by IDENTITY of the server
   instance, once `ClaudeAgentOptions` exists.
2. **`CLAUDE_AGENT_SDK_MCP_NO_PREFIX` removes the prefix entirely** for
   ``type: "sdk"`` servers — precisely the ones wardex wraps. A key that
   assumes the prefix misses every tool in that configuration, so the reverse
   parse falls back to a bare-name index built at wrap time. That index is
   consulted ONLY in that configuration: while the CLI prefixes, a bare name
   can only be a builtin, and asking the index about it would let a wrapped
   tool capture a builtin's key.
3. **The sanitizer is not injective.** `"a.b"` and `"a b"` collapse onto one
   token, and `"a  b"` — like a dict key that simply contains `__` already —
   puts `__` INSIDE the token, so the separator stops being a separator. A
   reverse parse that splits on the FIRST `__` reads `mcp__a__b__greet` as
   server `a`, tool `b__greet`, while the handler builds server `a__b`, tool
   `greet`: two keys, `claim()` never meets itself, and the call ships TWICE
   with nothing marking it. So the prefixed parse is anchored on the servers
   this adapter actually wrapped — a token is accepted only if some handle
   claims it AND exports the tool name left over — and the positional split
   survives only as the fallback for servers wardex never saw, which have one
   observer and cannot be double-emitted.

Where a collapse is genuinely ambiguous this module refuses to guess:
`key_for_hook` returns None and the hook observation does not enter the
arbitration at all. Attributing it to one candidate would fail two ways at once
— a double emit for one server and a LOST handler span for the other, since the
hook would win a key the handler never claimed.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Mapping
from typing import Any

from .._assembly import UnitKey, counters
from .._limits import LimitsConfig

_PREFIX = "mcp__"
_UNSAFE = re.compile(r"[^a-zA-Z0-9_-]")
_NO_PREFIX_ENV = "CLAUDE_AGENT_SDK_MCP_NO_PREFIX"

#: What the CLI's own truthiness check rejects. Deliberately conservative: this
#: value decides only whether an HONESTY MARKER is attached, so reading a exotic
#: spelling as "enabled" costs a marker nobody needed, while reading it as
#: "disabled" would hide a real ambiguity.
_FALSY = frozenset({"", "0", "false", "no", "off"})


def sanitize(token: str) -> str:
    """The CLI's own sanitizer: every char outside ``[A-Za-z0-9_-]`` becomes ``_``.

    Idempotent, which is what lets `mcp_tool_key` be applied both to a raw dict
    key and to a token already recovered from a prefixed hook name without the
    two disagreeing by one character.
    """
    return _UNSAFE.sub("_", token)


def mcp_tool_key(server_token: str, tool_name: str) -> UnitKey:
    """THE key space. Both observers must land here or `claim()` cannot arbitrate."""
    return UnitKey("mcp.tool", f"{sanitize(server_token)}/{sanitize(tool_name)}")


def builtin_tool_key(tool_name: str) -> UnitKey:
    """A tool wardex did not wrap — a Claude Code builtin, or an external server.

    Its own namespace on purpose: nothing in this process can ever be a second
    observer of it, so it must not collide with an in-process tool that happens
    to share a name.
    """
    return UnitKey("tool.name", tool_name)


def prefix_disabled(env: Mapping[str, str] | None = None) -> bool:
    """Is the CLI shipping SDK tools under their BARE names right now?

    Read at call time rather than at install: the variable is consumed by the
    CLI subprocess, which is spawned later and from whatever environment the
    host has by then.
    """
    source = env if env is not None else os.environ
    return source.get(_NO_PREFIX_ENV, "").strip().lower() not in _FALSY


class ServerHandle:
    """One in-process MCP server wardex wrapped, and its token once it is known.

    Held by the handler wrapper itself (the closure captures it), so the wrapper
    reads the token at CALL time — by which point `_prepare_options` has run and
    resolved it. Reading it at wrap time would freeze in the placeholder.
    """

    __slots__ = ("instance", "name", "token", "tools")

    def __init__(self, name: str) -> None:
        #: `create_sdk_mcp_server(name=...)`. NOT the token — the fallback for it.
        self.name = name
        #: The sanitized `mcp_servers` dict key, or None until options are built.
        self.token: str | None = None
        #: The server object `create_sdk_mcp_server` returned, for identity matching.
        self.instance: Any = None
        self.tools: set[str] = set()

    @property
    def token_resolved(self) -> bool:
        return self.token is not None

    @property
    def effective_token(self) -> str:
        """The token, or the server's own name while the real one is unknown.

        The fallback is a GUESS and the caller marks it as one: if the dict key
        differs from the server name, the handler and the hook build different
        keys, `claim()` never meets itself and the call is emitted twice.
        """
        return self.token if self.token is not None else sanitize(self.name)

    def key_for(self, tool_name: str) -> UnitKey:
        return mcp_tool_key(self.effective_token, tool_name)


class McpToolCatalog:
    """Every in-process server this adapter wrapped, and the name space for them.

    Bounded like every other table in the SDK (I10): a host that builds a fresh
    server per query would otherwise accumulate handles for the process
    lifetime. Overflow drops the OLDEST handle and counts it — what is lost is a
    bare-name lookup, never a span: `key_for_hook` degrades to the builtin key
    space, which at worst lets one call be observed twice.
    """

    __slots__ = ("_handles", "_lock", "_max")

    def __init__(self) -> None:
        # From the CORE, never a Python literal — same rule as every other
        # bound. The core default is ALL this constructor resolves: the host's
        # value arrives through `apply_bound`, which is the single handle. A
        # second one here would let the delivery table point at `apply_bound`
        # while a caller quietly configured the table some other way, which is
        # the same "the guard says delivered, nobody delivers" illusion the
        # delivery census exists to remove.
        resolved = LimitsConfig().resolved()
        self._max = resolved["max_entries_per_unit"]
        # Reentrant, and for the reason `_diag._REPORT_LOCK` was made reentrant
        # rather than argued safe: the alternative is a claim about which
        # callers can arrive here, and such a claim holds only until the next
        # caller is added. Today no finalizer-borne path reaches this table —
        # the seams' close hooks end at `Client.capture_span` — but a weakref
        # callback lands wherever a reference count reaches zero (and at any
        # allocation, via a cyclic collection), and every block this lock guards
        # offers both: `ServerHandle(name)`, a list append and a `sanitize()`
        # result allocate, and `self._handles.pop(0)` drops the last reference
        # to a handle. So the one thing standing between this and a permanent
        # self-deadlock in the host's own `create_sdk_mcp_server()` call would
        # be a call-graph fact nobody re-checks. The lock is taken per
        # registration and per hook lookup — both far below span rate — so
        # making the property local to the lock costs nothing worth counting.
        #
        # Reentrancy alone is not the whole guarantee, because a reentrant lock
        # turns a hang into concurrent mutation. Every guarded block here must
        # therefore tolerate a nested `handle_for`, whose FIFO cap can pop from
        # `_handles` and append to it. The three that walk the list do so over a
        # snapshot for exactly that reason: walking the live list by index would
        # silently skip a handle after a pop, and a skipped handle is a server
        # whose token is never resolved — which the module docstring explains is
        # how one call gets emitted twice.
        self._lock = threading.RLock()
        self._handles: list[ServerHandle] = []

    def apply_bound(self, *, max_entries: int) -> None:
        """Re-bind this table's ceiling to the host's configured value.

        Exists because the catalog is built in the adapter's `__init__`, before
        the adapter has a client, so the bound cannot be a constructor
        argument. The core documents this table as one of the things
        `max_entries_per_unit` sizes, and until this method existed that
        documentation was a promise nothing kept.

        No trim, deliberately. The only writer of `_handles` is `handle_for`,
        whose only production caller is the `create_sdk_mcp_server` wrapper the
        adapter installs — and `install()` is single-shot, reachable a second
        time only through `uninstall()`, whose last statement clears this
        table. So this method is only ever called on an EMPTY one. Even if that
        stopped holding, `handle_for`'s own `while len(self._handles) >=
        self._max` loop converges on the next registration; trimming here would
        only move the same eviction earlier, under a counter whose published
        meaning (the README's "Resource limits" section) is overflow, not
        re-binding.
        """
        with self._lock:
            self._max = max_entries

    def handle_for(self, name: str, existing: ServerHandle | None = None) -> ServerHandle:
        """The handle for this registration — the previous one when there is one.

        `existing` is the handle recorded on an already-wrapped handler. Reusing
        it is what makes the "fresh options, reused `@tool` definitions" pattern
        work: the second `create_sdk_mcp_server` call builds a NEW server object,
        and if that produced a new handle the wrapper — which still closes over
        the first — would keep a token nobody ever resolves again.
        """
        with self._lock:
            if existing is not None:
                return existing
            handle = ServerHandle(name)
            while len(self._handles) >= self._max:
                self._handles.pop(0)
                counters.bump("adapters.anthropic.server_table_full")
            self._handles.append(handle)
            return handle

    def resolve_tokens(self, mcp_servers: Any) -> None:
        """Recover each server's CLI token from the options, by instance identity.

        The only moment this is knowable. Called from `_prepare_options`, which
        runs on every `query()` / `ClaudeSDKClient(...)`, so a handle registered
        under a different key later is re-resolved rather than left stale.
        """
        if not isinstance(mcp_servers, Mapping):
            return
        with self._lock:
            if not self._handles:
                return
            for key, config in mcp_servers.items():
                instance = config.get("instance") if isinstance(config, Mapping) else None
                if instance is None:
                    continue
                for handle in list(self._handles):
                    if handle.instance is instance:
                        handle.token = sanitize(str(key))

    def key_for_hook(self, raw: str) -> UnitKey | None:
        """Canonicalize a CLI-reported `tool_name` into the shared key space.

        None means "this observation cannot be attributed to a server, so do not
        claim and do not emit" — see the module docstring for why guessing here
        breaks in two directions at once.
        """
        if raw.startswith(_PREFIX):
            body = raw[len(_PREFIX) :]
            anchored = self._anchored(body)
            if len(anchored) == 1:
                return next(iter(anchored))
            if anchored:
                # Two wrapped servers both explain this string. Standing down
                # costs nothing the hook was going to contribute: for a tool
                # wardex wrapped, a resolved key only ever loses to the handler
                # in `outranked()` and opens nothing, so None reaches the same
                # place by a shorter route. Picking one, by contrast, would hand
                # the other server's handler a rival observer on its own key.
                counters.bump("adapters.anthropic.hook_tool_name_ambiguous")
                return None
            # No wrapped server explains it, so the tool is external — one
            # observer, no arbitration, and any stable key will do. The CLI's own
            # positional split is that key.
            server, sep, tool = body.partition("__")
            if sep:
                return mcp_tool_key(server, tool)
        if not prefix_disabled():
            # Gated for the same reason `ambiguous_bare` is. While the CLI
            # namespaces EVERY MCP tool — the in-process ones included — a bare
            # name can only be a builtin, so the bare-name index has nothing to
            # say about it. Consulting it anyway lets a wrapped tool named `Read`
            # capture the builtin `Read`'s key: with one such server the builtin
            # is misrouted, and once that server's handler claims the key at the
            # handler rank every later builtin `Read` is outranked and discarded;
            # with two, the observation is dropped outright, and `ambiguous_bare`
            # correctly reports False so no span carries a marker for the loss.
            return builtin_tool_key(raw)
        tokens = self._tokens_exporting(raw)
        if len(tokens) == 1:
            return mcp_tool_key(next(iter(tokens)), raw)
        if tokens:
            counters.bump("adapters.anthropic.hook_tool_name_ambiguous")
            return None
        return builtin_tool_key(raw)

    def ambiguous_bare(self, tool_name: str) -> bool:
        """Would the hook's view of this tool be ambiguous, right now?

        Gated on the environment variable and not merely on the name: while the
        CLI prefixes, two servers exporting `search` are told apart perfectly and
        claiming otherwise would be a marker the span cannot back.
        """
        if not prefix_disabled():
            return False
        return len(self._tokens_exporting(tool_name)) > 1

    def clear(self) -> None:
        with self._lock:
            self._handles.clear()

    def _anchored(self, body: str) -> set[UnitKey]:
        """Which wrapped servers explain `body` as ``<their token>__<their tool>``.

        The tool name is compared through `sanitize` because the two sides arrive
        differently seasoned: `body` has already been through the CLI's sanitizer,
        while `ServerHandle.tools` holds the raw `@tool` names. Skipping that
        would silently un-anchor every tool whose name contains a space or a dot
        and drop it back onto the positional split.

        A set, never a single token, for the same reason `_tokens_exporting`
        returns one: with two candidates there is no right answer, and quietly
        keeping one would be the guess this module refuses to make.
        """
        with self._lock:
            out: set[UnitKey] = set()
            for handle in list(self._handles):
                token = handle.effective_token
                head = token + "__"
                if not body.startswith(head):
                    continue
                tail = body[len(head) :]
                if any(tail == sanitize(name) for name in handle.tools):
                    out.add(mcp_tool_key(token, tail))
            return out

    def _tokens_exporting(self, tool_name: str) -> set[str]:
        """Which server tokens export this BARE tool name.

        A set, never a single token: two servers may export the same name, and a
        mapping that kept one would silently pick a winner — the exact shape
        `key_for_hook` returns None for.
        """
        with self._lock:
            return {h.effective_token for h in list(self._handles) if tool_name in h.tools}
