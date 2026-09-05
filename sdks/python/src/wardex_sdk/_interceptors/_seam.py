"""Byte seam interceptor base — shared logic for feeding trackers and assembling spans.

Common skeleton that streams app-layer plaintext bytes into a protocol tracker and
assembles a CLIENT span from the _Txn the tracker returns. SSL (wss/https) and
plaintext (ws/http) seams inherit this and override only seam-specific behavior
(tracker selection, timing resolution, scheme, etc). Works standalone, without adapters.
"""

from __future__ import annotations

import contextvars
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from .._assembly import (
    Ambient,
    Limitation,
    PatchSet,
    Prefilter,
    SpanDraft,
    TransportLabel,
    capture_mode_of,
    counters,
    diag_warning,
    guard,
    in_degraded_run,
    resolve_observed,
    should_capture,
)
from .._enums import (
    CaptureMode,
    CaptureSource,
    Direction,
    Protocol,
    StatusCode,
)
from .._limits import LimitsConfig, LimitsConsumer, limits_kwargs
from .._protocol import classify_path, parse_llm_semantics
from .._semantics import (
    USAGE_DROPPED_KEY,
    build_gen_ai,
    build_grpc_fields,
    embeddings_attrs,
    has_core_semantics,
    identifies_llm_call,
    provider_extras,
    ws_close_name,
)
from .._suppress import is_suppressed
from .._types import (
    HttpMeta,
    TransportAttributes,
    TransportTiming,
)
from ._base import InterceptorInterface
from ._close_hook import install_shared_close_hook, on_close, uninstall_shared_close_hook
from ._conn_timing import install_shared_timing, uninstall_shared_timing
from ._trackers import _Txn, _WebSocketTracker

if TYPE_CHECKING:
    from .._client import Client

# Headers are parsed but intentionally not recorded on the span (secret protection).
# Bodies are recorded, and their PII is masked later — in the Rust core, at encode
# time (`codec.encode_otlp_traces` takes the mode and the disabled categories), not
# here.
#
# --- Where "do we capture this?" is asked, and why it is asked twice ---
#
# The question has two halves, split by WHEN each can be answered rather than by
# what it means. Getting that split wrong is expensive in one direction and
# WRONG in the other, so it is written down here.
#
# INVARIANT — settled for the life of the connection, so `_capture_possible`
# answers them in the patch wrapper, before the send/recv buffer is copied and
# before a single byte reaches a tracker:
#
#   * no client on this seam. Both emit paths already returned on a `None`
#     client, so nothing behind one was ever going to become a span — but the
#     bodies were still accumulated in a tracker that is kept for the life of
#     the connection, which is memory held for an output that cannot exist.
#   * the self-exclusion flag. The exporter's own POST, which must never be
#     re-captured whatever else is true.
#   * a connection the sniff-latch already classified "ignore". `_gate` is
#     latched once, from the first bytes, and never re-decided — so a
#     TLS-backed Redis, Postgres or Kafka connection is settled for its whole
#     life, and every call on one was still materializing its buffer
#     (`bytes(data)`, a real copy whenever the caller passes a memoryview or a
#     bytearray, which is the shape the asyncio and httpx paths use) to
#     re-derive a verdict that was reached on the first write.
#
# CONTEXT-DEPENDENT — an answer about ONE transaction, and only ever correct
# for the instant it was asked, so they live in the module `_should_capture` on the
# response side: the shared policy's ambient-span clause and the LLM-semantic
# claim `_parse_semantics` earns. An agent unit can ACTIVATE after a connection
# was opened — a pooled keep-alive socket outlives any single run — so a
# per-connection early-out on "no agent unit is ambient right now" would
# silently drop every later request on that socket. That is data loss, and the
# parse it would save is the one the gate needs.
#
# And a THIRD moment since the parse moved off the caller's thread: the gate
# now usually runs on the finalize WORKER, over a `_PendingTxn` sealed at the
# response instant. What keeps its answer honest there is the seal — the
# prefilter verdict, the mode, and `copy_context()` all travel with the job —
# so the gate reads the response instant's facts wherever and whenever it
# actually executes.
#
# NOT here, deliberately: "the transport is a NoOpTransport". `init()` resolves
# a missing `transport=` argument to exactly that, so it is the out-of-the-box
# configuration rather than a statement that capture is off, and reading it as
# one would turn `wardex.init(intercept=True)` into a silent no-op.


def _accepted_prefix(data: Any, n: int) -> Any:
    """The first `n` bytes of a send buffer, without materializing the rest.

    `send`/`write` may report a SHORT write, and the caller then keeps the tail
    and calls again with the whole remainder — asyncio's plaintext writer does
    exactly that on 3.10/3.11 (`_write_ready` calls `send(self._buffer)` on one
    bytearray and then `del self._buffer[:n]`). `bytes(data)[:n]` copies that
    entire remainder before throwing away everything the kernel refused, so a
    multi-megabyte body costs a copy per call: quadratic in body size, and on
    the branch the gate deliberately leaves OPEN — a local plaintext model
    server is HTTP, so it latches "http" and pays this on every partial write.

    Returns something `bytes()` accepts rather than `bytes`, so that an exact
    `bytes` argument written in full stays the same object and costs nothing at
    all; the caller materializes once, immediately.

    The three arms are an isinstance chain rather than a `try`, because a
    handler here would be a silent swallow on a path that has a correct answer
    without one: anything that is not one of the three buffer types the socket
    API actually takes falls through to what this line used to be.
    """
    if isinstance(data, bytes):
        return data[:n]
    if isinstance(data, (bytearray, memoryview)):
        # `.cast("B")` so `n` is read as bytes for an itemsize > 1 view too,
        # which is what the caller's `n` counts.
        return memoryview(data).cast("B")[:n]
    return bytes(data)[:n]


def _http_error(txn: Any) -> bool:
    """Did the peer answer with a 4xx/5xx? Absent status reads as success."""
    return not 200 <= getattr(txn, "status", 200) < 400


def _is_llm_traffic(txn: Any, sem: Any) -> bool:
    """Is this an LLM call, for the capture policy and for the gen_ai block alike?

    One predicate for both questions on purpose: a transaction the gate admits
    as agent traffic and then leaves with `gen_ai=None` is worse than either
    answer alone — captured volume with no identity on it.

    The request-side half is admitted only for an HTTP ERROR, and that is the
    narrow reading rather than the tidy one. Both provider gates in the Rust
    parser are SUBSTRING matches on host and path, so an internal service at
    `anthropic-proxy.corp/v1/messages` whose body happens to carry a `model`
    field parses as an Anthropic chat call. On a 2xx that shape is genuinely
    ambiguous — it may be an LLM endpoint wardex cannot read, or not an LLM
    endpoint at all — and it is already dropped today, so admitting it would be
    a behaviour change nobody asked for on traffic nobody identified. A 4xx/5xx
    from a host and path that parse as a provider is not ambiguous in the same
    way: the request was addressed to a chat endpoint with a model on it, and
    the reply is the provider refusing. That is the call the fix is about.
    """
    return has_core_semantics(sem) or (_http_error(txn) and identifies_llm_call(sem))


class _ConnectionState:
    """Capture state for a single connection — holds the protocol tracker."""

    def __init__(self, tracker: Any, server_address: str, server_port: int) -> None:
        self.tracker = tracker
        self.server_address = server_address
        self.server_port = server_port
        self.timing_consumed = False
        self.gate: str | None = None  # None=undetermined, "http", "h2c", "h2", "ignore"
        # This state was built for a socket that OUTLIVED an os.fork(): the
        # tracker starts mid-stream, so the first span assembled from it
        # carries `TRACKING_RESET_AT_FORK` (once — the stamp clears it). Set
        # by `_state` from the seam's fork latch, never by the trackers.
        self.reset_at_fork = False
        # Guards the once-per-connection debug log below. Deliberately a
        # separate field from `gate`: `gate` is owned by the plaintext seam's
        # protocol sniff-latch (_socket.py), which never re-evaluates once
        # set — reusing it here would let a disable-log event permanently
        # gate off all further bytes on that seam, including a still-healthy
        # direction, as an unintended side effect of two unrelated concerns
        # sharing one field.
        self.disabled_logged = False

    def latched_off(self) -> bool:
        """Has the sniff-latch already ruled this connection out for good?

        The early gate acts on this answer, so what matters is that it is
        STABLE: each seam's `_gate` writes `gate` once, from the first bytes it
        sees, and never revisits it (see the two `_gate` docstrings, and
        `test_latch_stays_ignore_once_closed`). "ignore" is therefore a fact
        about the connection rather than about the call that observed it, which
        is what makes it safe to skip the buffer copy on every later call.

        Asked as "is it ignore" rather than "is it one of the live protocols"
        so that `None` — undetermined, the state of a connection whose first
        bytes have not arrived — reads as "keep going", never as "drop".
        """
        return self.gate == "ignore"


class ByteSeamInterceptor(InterceptorInterface):
    """Shared byte seam base — tracker feed + span assembly.
    Seam-specific behavior is a subclass hook."""

    def __init__(self) -> None:
        self._client: Client | None = None
        self._conns: dict[int, _ConnectionState] = {}
        # ids of connections that were being tracked when an os.fork() reset
        # this seam in the child — consumed one id at a time by `_state`, so
        # the first span on an INHERITED socket can say its tracking restarted
        # mid-stream (`TRACKING_RESET_AT_FORK`). Bounded by max_connections at
        # the only site that fills it (`_at_fork_reinit`).
        self._reset_at_fork_ids: set[int] = set()
        self._patches = PatchSet(f"interceptors.{self.name()}")
        self._installed = False
        # Does THIS seam hold a reference on the shared timing probe? A second
        # field rather than a reading of `_installed`, because `uninstall()` is
        # now total: it runs after an `install()` that raised, where the two
        # answers differ. Releasing a reference this seam never took decrements
        # the shared refcount to zero underneath the OTHER seam and rips
        # `socket.connect` back out from under a live installation.
        self._timing_held = False
        # The same bookkeeping for the shared close probe, and a SECOND flag for
        # the same reason: the two probes are refcounted independently, so a
        # single "did I acquire anything" answer would release one of them on
        # behalf of a seam that only ever took the other.
        self._close_hook_held = False
        # Defaults match the core's, so behavior is unchanged until _load_limits
        # resolves an actual config at install() time.
        self._limits: dict[str, int] = LimitsConfig().resolved()
        # The WS tracker's projected keywords, built ONCE per limits load rather
        # than per upgrade: a tracker is constructed on every WebSocket
        # handshake, and this is the one projection on a path whose rate is set
        # by the host's traffic. Initialized HERE as well as in `_load_limits`
        # because a seam driven without `install()` is a real shape in this
        # suite, and an attribute missing there would raise inside the host's
        # own socket call rather than anywhere a test looks.
        self._ws_kwargs: dict[str, int] = limits_kwargs(LimitsConsumer.WS_TRACKER, self._limits)
        self._native_limits: Any = None

    def _load_limits(self, client: Client | None) -> None:
        """Cache resolved limits at install time; config is frozen after init."""
        config = getattr(client, "config", None)
        lim = config.limits if config is not None else LimitsConfig()
        self._limits = lim.resolved()
        self._ws_kwargs = limits_kwargs(LimitsConsumer.WS_TRACKER, self._limits)
        self._native_limits = lim.to_native()

    def _fresh_patchset(self) -> PatchSet:
        """This seam's PatchSet, wired to the client's debug setting.

        Built at install() rather than in `__init__` because `config.debug` is
        not known until a client arrives, and a restore that fails invisibly
        under `debug=True` is exactly what `_diag` exists to prevent. The empty
        set `__init__` makes is what keeps `uninstall()` safe before any
        `install()`.
        """
        config = getattr(self._client, "config", None)
        return PatchSet(f"interceptors.{self.name()}", debug=bool(getattr(config, "debug", False)))

    # --- Uninstall (shared: both seams undo exactly the same three things) ---

    def uninstall(self) -> None:
        """Undo whatever this seam installed, however far `install()` got.

        TOTAL, which it was not. It used to open with `if not self._installed:
        return` while every `install()` sets that flag as its LAST statement, so
        the two never overlapped where it mattered: an `install()` that raised
        halfway had already patched part of `ssl.SSLSocket` or `socket.socket`,
        and the undo the registry then called declined to run. Those wrappers
        stayed in front of the host's sockets for the life of the process — the
        flag turned the rollback into a no-op for every interceptor this SDK
        ships, which is a guard that reads as a fix and is not one.

        The flag cannot answer "was anything patched?", because it is only ever
        set once everything was. The things that CAN answer it are the pieces
        themselves, and each is asked separately: `PatchSet.restore_all()` is
        idempotent and empty until the first `patch()` lands, `_timing_held` and
        `_close_hook_held` record the two acquisitions that are refcounted
        elsewhere and so must not be released twice or unearned, and `_conns` is
        empty until a byte flows.
        Every step is a no-op on a seam that never installed, which is what
        makes this safe to call unconditionally, twice, or on a fresh object.

        The WebSocket flush stays: a live WS session holds a span that only
        exists once the session ends, and dropping it at uninstall would be the
        SDK losing data at teardown — the reason `uninstall_all` runs before
        `client.close()` at all. It is `_retire`'s job now, because teardown and
        a socket closing are the same question asked at two altitudes.
        """
        self._patches.restore_all()
        self._release_probes()
        for st in list(self._conns.values()):
            self._retire(st, Limitation.WS_NO_CLOSE)
        self._conns.clear()
        self._installed = False

    def _at_fork_reinit(self) -> None:
        """Fork-child reset: drop the inherited connection table, remember it.

        NOT `uninstall()` in a smaller coat, and the differences are the
        design: the patches stay installed (I-fork-4 — they crossed the fork
        and still work), nothing is retired or emitted (`_retire` would EMIT
        the inherited WS sessions, which the parent owns and will emit
        itself — I-fork-3), and the probes keep their refcounts (the shared
        singletons reset their own state through the runtime, not through the
        seams). What goes is the per-connection state: an inherited entry
        would hand a recycled `id()` a dead connection's tracker and latched
        gate — the close-hook bug's cross-process edition — and an inherited
        LIVE socket's tracker holds a parse position that is a lie in a child
        that missed bytes.

        The ids are LATCHED before the clear, capped at `max_connections`
        newest-first (dict order is insertion order; anything beyond the cap
        was oldest and is dropped — the same drop-oldest posture as the table
        itself). `_state` consumes the latch: the first span assembled on a
        connection whose id survives here says `TRACKING_RESET_AT_FORK`
        instead of silently reporting mid-stream parses as clean ones — every
        OTHER path that discards a live `_ConnectionState` leaves a marker
        (`CONNECTION_EVICTED`, `WS_NO_CLOSE`), and the fork path may not be
        the one silent exception.

        Locks: this seam's table has none to replace (`_conns` is unlocked by
        the same argument `CloseRegistry` documents); the PatchSet's lock is
        the one inherited lock this seam owns, and it is replaced through the
        set's own reset.
        """
        self._patches._at_fork_reinit()
        ids = list(self._conns)
        cap = self._limits["max_connections"]
        if len(ids) > cap:
            ids = ids[-cap:]
        self._reset_at_fork_ids = set(ids)
        self._conns.clear()

    def _acquire_probes(self) -> None:
        """Take this seam's references on the two shared, refcounted probes.

        One call so that a seam cannot take the timing probe and forget the
        close hook: without the second, this seam's per-connection state is
        evicted only when the socket is COLLECTED, which a pooled connection
        may never be while the process runs.
        """
        self._acquire_timing()
        self._acquire_close_hook()

    def _release_probes(self) -> None:
        self._release_timing()
        self._release_close_hook()

    def _acquire_timing(self) -> None:
        """Take this seam's reference on the shared connection-timing probe.

        The flag is set AFTER the call, so a raising `install_shared_timing`
        leaves nothing for `_release_timing` to give back.
        """
        install_shared_timing(**limits_kwargs(LimitsConsumer.CONN_TIMING, self._limits))
        self._timing_held = True

    def _release_timing(self) -> None:
        """Give back the reference this seam took, at most once, and only if taken."""
        if not self._timing_held:
            return
        self._timing_held = False
        uninstall_shared_timing()

    def _acquire_close_hook(self) -> None:
        """Take this seam's reference on the shared `socket.close` probe."""
        install_shared_close_hook()
        self._close_hook_held = True

    def _release_close_hook(self) -> None:
        if not self._close_hook_held:
            return
        self._close_hook_held = False
        uninstall_shared_close_hook()

    # --- Subclass hooks ---

    @abstractmethod
    def _select_tracker(self, obj: Any) -> Any: ...

    @abstractmethod
    def _resolve_timing(
        self, obj: Any, st: _ConnectionState
    ) -> tuple[float, float, bool, tuple[Limitation, ...]]: ...

    def _guard(self, where: str) -> guard:
        """The one authorized swallow, wired to this seam's debug setting.

        Span assembly runs `SpanDraft.finish()`, which raises `VocabularyError`
        on a vocabulary breach, and I6 forbids that reaching the host. It is
        counted and — under `config.debug` — logged with a traceback, so the
        span this deletes is at least findable.
        """
        config = getattr(self._client, "config", None)
        return guard(where, debug=bool(getattr(config, "debug", False)))

    def _url_scheme(self, is_ws: bool) -> str:
        return "wss" if is_ws else "https"

    def _gate(self, st: _ConnectionState, data: bytes, phase: str) -> bool:
        return True

    def _capture_possible(self, obj: Any) -> bool:
        """Can this seam capture ANYTHING on this connection, right now?

        The early half of the split described at the top of this module. It
        runs in front of every `send` and every `recv` of every socket in the
        process, so it is three reads and no allocation — it has to cost less
        than the buffer copy it exists to skip.

        A conservative answer in one direction only: True means "nothing
        invariant rules this out", not "this will be captured". Everything that
        depends on the transaction — the policy's ambient-span clause, the
        LLM-semantic claim — is still ahead, in `_should_capture`.

        Asked again at the top of `_on_request_bytes`/`_on_response_bytes`
        rather than trusted from the wrapper that already asked: those two are
        the seam's real entry points, reached directly by every subclass and by
        the tests, and a guard that lives only in the patch wrappers is not
        there at all for half its callers. The repeat is a dict lookup and a
        ContextVar read against work measured in kilobytes.
        """
        if self._client is None:
            return False
        if is_suppressed():
            return False
        st = self._conns.get(id(obj))
        return st is None or not st.latched_off()

    def _transport_prefilter(self, st: _ConnectionState) -> Prefilter:
        """This seam's opinion about the connection itself, before the policy.

        `DEFER` is the base answer, and it is the honest one for a TLS seam: it
        knows nothing about the peer that `assembly.should_capture` does not
        already know better. A seam that DOES know something — the plaintext
        socket seam, which must never read a link-local metadata endpoint and
        must always honour `intercept_hosts` — overrides this, and only this.
        Overriding `_should_capture` itself is what produced the two bugs
        design §4.4 names.
        """
        return Prefilter.DEFER

    def _capture_source(self) -> CaptureSource:
        return CaptureSource.SSL

    # --- Connection state ---

    def _state(self, obj: Any) -> _ConnectionState:
        cid = id(obj)
        st = self._conns.get(cid)
        if st is None:
            addr, port = _peer(obj)
            st = _ConnectionState(self._select_tracker(obj), addr, port)
            if cid in self._reset_at_fork_ids:
                # This object was tracked when the fork reset the table, so it
                # is (to the limit of id() identity) an INHERITED connection:
                # the new tracker starts mid-stream. Consume the id — the
                # marker is a fact about the reset, and the reset happened
                # once.
                self._reset_at_fork_ids.discard(cid)
                st.reset_at_fork = True
            if len(self._conns) > self._limits["max_connections"]:
                old_cid = next(iter(self._conns))
                self._retire(self._conns.pop(old_cid), Limitation.CONNECTION_EVICTED)
            self._conns[cid] = st
            # The hook closes over the ID, never over `obj`: an eviction
            # mechanism that referenced the socket would keep the host's file
            # descriptor open for as long as this seam is installed, which is a
            # worse bug than the one it fixes (see `_close_hook`).
            on_close(obj, partial(self._connection_closed, cid))
        return st

    def _connection_closed(self, cid: int) -> None:
        """The connection behind `cid` is over: retire its state, keep its data.

        The FIFO cap in `_state` was the only thing that ever removed an entry,
        and it removes the wrong one — the oldest, which on a long-lived process
        is the connection still streaming, while the entries of connections that
        died an hour ago stay. Worse, the entry outliving its socket is what let
        a recycled `id()` serve a new connection the dead one's tracker and its
        latched "ignore" verdict.

        Idempotent by construction: the registry fires a hook at most once, and
        an already-evicted id pops nothing.
        """
        st = self._conns.pop(cid, None)
        if st is not None:
            self._retire(st, Limitation.WS_NO_CLOSE)

    def _retire(self, st: _ConnectionState, marker: Limitation) -> None:
        """The single place a connection's state stops existing.

        Three callers — the close hook, the FIFO cap and `uninstall()` — and
        they differ only in the marker they can honestly claim. The tracker
        decides what survives its own end: a WebSocket session is a span that
        exists ONLY at close, so it is emitted here rather than dropped, and
        every other tracker answers with an empty list after releasing whatever
        it was holding.

        Runs inside `socket.close()` and inside garbage collection, so it must
        stay short and must not raise; `_emit_ws` already carries the guard.
        """
        for txn in st.tracker.on_connection_close(marker):
            self._emit_ws(st, txn)

    # --- Tracker delegation + span assembly ---

    def _on_request_bytes(self, obj: Any, data: bytes) -> None:
        if not self._capture_possible(obj):
            return
        st = self._state(obj)
        if not self._gate(st, data, "request"):
            return
        addr, port = _peer(obj)
        st.server_address = addr
        st.server_port = port
        for txn in st.tracker.on_request_bytes(data):
            if txn.version == "websocket":
                self._emit_ws(st, txn)
            else:
                self._emit_span(obj, st, txn)

    def _on_response_bytes(self, obj: Any, data: bytes) -> None:
        if not self._capture_possible(obj):
            return
        st = self._state(obj)
        if not self._gate(st, data, "response"):
            return
        txns = st.tracker.on_response_bytes(data)
        try:
            # No span exists to carry a disable reason (the whole point of
            # the latch is that no message was ever parsed), so debug mode
            # logs it instead. Guarded by st.disabled_logged (not st.gate,
            # which the plaintext seam's sniff-latch owns) so a disabled
            # connection logs once, not once per subsequent read.
            if self._client is not None and self._client.config.debug:
                reason = getattr(st.tracker, "disabled_reason", lambda: None)()
                if reason is not None and not st.disabled_logged:
                    st.disabled_logged = True
                    diag_warning(f"parser disabled for {st.server_address}: {reason}")
        except Exception:  # noqa: BLE001 — debug-only logging must never break capture
            pass
        for txn in txns:
            if getattr(txn, "ws_upgrade", False):
                ws = _WebSocketTracker(
                    path=txn.ws_upgrade_path or "/",
                    deflate=txn.ws_deflate,
                    parent=txn.parent,
                    parent_closed=txn.parent_closed,
                    start_ns=txn.start_ns,
                    limits=self._native_limits,
                    **self._ws_kwargs,
                )
                st.tracker = ws
                if txn.ws_leftover:
                    for t2 in ws.on_response_bytes(txn.ws_leftover):
                        self._emit_ws(st, t2)
            elif txn.version == "websocket":
                self._emit_ws(st, txn)
            else:
                self._emit_span(obj, st, txn)

    def _prefilter_of(self, st: _ConnectionState) -> Prefilter:
        """This seam's transport prefilter, evaluated NOW and fail-open.

        `ALLOW` on an exception, deliberately (review M2-4): the old
        `_should_capture` swallowed a raising prefilter into `return True` —
        capture — and any narrower default here (`DEFER`, say) would turn
        today's captured traffic into a policy drop under AGENT with no
        parent, which is a new silent loss wearing an error handler. The
        guard makes the swallow counted instead of invisible.
        """
        pre = Prefilter.ALLOW
        with self._guard("interceptors.seam.prefilter"):
            pre = self._transport_prefilter(st)
        return pre

    def _seal(
        self, obj: Any, st: _ConnectionState, txn: _Txn, client: Client
    ) -> _PendingTxn | None:
        """Snapshot, on the CALLER's thread, everything a finalization may
        need that only this thread can read — and nothing else.

        This is the whole caller-side cost of a completed transaction:
        `_resolve_timing` (destructive and order-dependent, so it must run
        here whatever the gate later says — see its comment), the prefilter,
        a handful of attribute reads, and `copy_context()` (HAMT sharing,
        O(1)). The parse, the gate and the draft belong to the worker.

        `None` means the prefilter DENIED the connection, or the path is one
        the endpoint table EXCLUDES (`classify_path`) — either way cheaper
        than today, because the parse this skips was unconditionally paid
        before.

        The fork latch is consumed here rather than at assembly: `st` must
        not travel with the job (I-F1), so the first transaction SEALED on a
        fork-crossing connection carries the marker in its sealed
        timing markers. (Before the deferred split the stamp happened below
        the gate — first CAPTURED transaction; the seal is the earliest
        moment the fact can leave the connection state, and a gate-refused
        first transaction consumes the latch too; an EXCLUDED one does not —
        it returns above the latch block — so the marker lands on the
        connection's next sealed transaction.)
        """
        url_host = getattr(obj, "server_hostname", None) or st.server_address
        ct = txn.content_type or ""
        # gRPC: skip LLM semantic extraction (protobuf isn't LLM JSON).
        is_grpc = ct.startswith("application/grpc") and not ct.startswith("application/grpc-web")
        connect_ms, handshake_ms, reused, timing_markers = self._resolve_timing(obj, st)
        if classify_path(txn.path) == "excluded":
            # A telemetry upload (the OpenAI Agents SDK POSTs its whole run
            # record to /v1/traces/ingest). Not wardex's to copy: skipped in
            # every mode and above the allowlist, before any parse is queued
            # or body retained — counted, not spanned. Does not consume the
            # fork latch: the marker belongs on the first transaction that
            # becomes a span.
            counters.bump("interceptors.seam.path_excluded")
            return None
        if st.reset_at_fork:
            st.reset_at_fork = False
            timing_markers = (*timing_markers, Limitation.TRACKING_RESET_AT_FORK)
            counters.bump("interceptors.seam.tracking_reset_at_fork")
        pre = self._prefilter_of(st)
        if pre is Prefilter.DENY:
            return None
        config = getattr(client, "config", None)
        return _PendingTxn(
            txn=txn,
            url_host=url_host,
            url_scheme=self._url_scheme(False),
            capture_source=self._capture_source(),
            connection_id=str(id(obj)),
            server_address=st.server_address,
            server_port=st.server_port,
            is_grpc=is_grpc,
            prefilter=pre,
            mode=capture_mode_of(client),
            connect_ms=connect_ms,
            handshake_ms=handshake_ms,
            reused=reused,
            timing_markers=tuple(timing_markers),
            limits=self._native_limits,
            debug=bool(getattr(config, "debug", False)),
            ctx=contextvars.copy_context(),
            size=len(txn.request_body) + len(txn.response_body),
        )

    def _stamp_fork_reset(self, st: _ConnectionState, draft: Any) -> None:
        """Put `TRACKING_RESET_AT_FORK` on the FIRST span of a fork-crossing
        connection, and only that one.

        One helper for both build paths (HTTP and WS), because the fact is the
        connection's, not the transaction kind's. The flag clears on the first
        stamp: later spans on the same connection are parsed by a tracker that
        saw their whole exchange, and a marker repeated forever would read as
        a per-span defect rather than the one-time event it is.
        """
        if st.reset_at_fork:
            st.reset_at_fork = False
            draft.add_limitation(Limitation.TRACKING_RESET_AT_FORK)
            counters.bump("interceptors.seam.tracking_reset_at_fork")

    def _emit_span(self, obj: Any, st: _ConnectionState, txn: _Txn) -> None:
        """Seal on this thread; finalize on the worker — with two measured
        exceptions that assemble inline.

        gRPC (§3.6): there is nothing to defer — `parse_llm_semantics` was
        never called on protobuf, and `build_grpc_fields` is µs-grade framing
        (offset jumps, no decompression). Queueing it would let a 64 MiB
        protobuf body evict REAL parse jobs from the backlog, and a fallback
        that skipped the framing would downgrade the label/status of a span
        whose expensive step never existed. WebSocket rides `_emit_ws` below
        for the same class of reason.

        `capture_deferred` is a total function, but it stays inside the emit
        guard anyway: the blast radius of this path must remain exactly what
        it was before the split.
        """
        client = self._client
        if client is None:
            return
        span = None
        with self._guard("interceptors.seam.emit_span"):
            pending = self._seal(obj, st, txn, client)
            if pending is not None and pending.is_grpc:
                span = _assemble(pending, parse=True, extra=())
                pending = None
            if pending is not None:
                client.capture_deferred(pending)
        if span is not None:
            client.capture_span(span)

    def _emit_ws(self, st: _ConnectionState, txn: _Txn) -> None:
        client = self._client
        if client is None:
            return
        span = None
        with self._guard("interceptors.seam.emit_ws"):
            span = self._build_ws_span(st, txn)
        if span is not None:
            client.capture_span(span)

    def _build_ws_span(self, st: _ConnectionState, txn: _Txn) -> Any:
        # `sem=None`: a WS session carries no parsed LLM semantics (by
        # construction on this path), so it is captured only under ALL, an
        # allowlisted host, or a live local span. Inline — WS never defers
        # (§3.6) — but the DECISION is the same module `_gate` the deferred
        # path uses, composed with the same fail-open prefilter.
        if not _should_capture(
            self._prefilter_of(st), txn, None, mode=capture_mode_of(self._client)
        ):
            return None
        p = resolve_observed(_latched(txn), parent_closed=txn.parent_closed)

        code = txn.ws_close_code
        # Status based on close code: 1000/1001/none = OK, otherwise = ERROR
        if code is None or code in (1000, 1001):
            status_code = StatusCode.OK
            error_type = None
        else:
            status_code = StatusCode.ERROR
            error_type = ws_close_name(code)

        draft = SpanDraft.transport(
            p,
            label=TransportLabel.WEBSOCKET,
            subject=txn.path,
            source=self._capture_source(),
            start_ns=txn.start_ns,
        )
        self._stamp_fork_reset(st, draft)
        draft.set_extra("network.protocol.version", "websocket")
        draft.set_extra("ws.messages.sent", txn.ws_messages_sent)
        draft.set_extra("ws.messages.received", txn.ws_messages_received)
        draft.set_extra("ws.bytes.sent", txn.ws_bytes_sent)
        draft.set_extra("ws.bytes.received", txn.ws_bytes_received)
        if code is not None:
            draft.set_extra("ws.close_code", code)

        timing = TransportTiming(
            tcp_connect_ms=0.0,
            tls_handshake_ms=0.0,
            ttfb_ms=0.0,
            ttft_ms=0.0,
            transfer_ms=max(0.0, (txn.end_ns - txn.start_ns) / 1e6),
        )
        draft.set_transport(
            TransportAttributes(
                connection_id=str(id(st)),
                protocol=Protocol.HTTP,
                direction=Direction.OUTBOUND,
                timing=timing,
                request_size=txn.ws_bytes_sent,
                response_size=txn.ws_bytes_received,
                http=HttpMeta(
                    method="GET",
                    url=(
                        f"{self._url_scheme(True)}://{st.server_address}:{st.server_port}{txn.path}"
                    ),
                    status_code=101,
                ),
                connection_reused=False,
            )
        )
        draft.set_server(st.server_address, st.server_port)
        draft.set_status(status_code)
        draft.set_error(error_type)
        draft.set_io(input_data=txn.request_body, output_data=txn.response_body)
        draft.integrity.truncated(Limitation.WS_PAYLOAD_TRUNCATED in txn.ws_markers)
        for marker in txn.ws_markers:
            draft.add_limitation(marker)
        return draft.finish(txn.end_ns)


@dataclass(frozen=True, slots=True)
class _PendingTxn:
    """One sealed transaction — everything the finalization may read, and
    NOTHING else (invariants I-F1/I-F2).

    No socket, no `_ConnectionState`, no tracker, no seam, no client. That is
    not tidiness, it is the mechanism: `_assemble`/`_parse_semantics` below
    are MODULE functions over this dataclass, so "the worker read seam state
    that changed under it" is not a bug class that can be written — a
    re-install that swaps `seam._native_limits` cannot touch a sealed job,
    because the job holds its own `limits` and the seam is not in scope.
    (An earlier draft kept them as methods and declared the snapshot in
    prose; the prose was false within one review pass — `p.limits` was a
    dead field. Structure over promise.)

    `ctx` carries the IMMUTABLE-fact ContextVars (`in_degraded_run` and
    friends) — `run`/`fallback` execute inside it. Mutable Scope state
    travels separately, as the snapshot `Client.capture_deferred` takes
    (design §3.7). `size` is the queue's byte-accounting unit: the raw
    bodies this job keeps resident while it waits.
    """

    txn: _Txn
    url_host: str
    url_scheme: str
    capture_source: CaptureSource
    connection_id: str
    server_address: str
    server_port: int
    is_grpc: bool
    prefilter: Prefilter
    mode: CaptureMode
    connect_ms: float
    handshake_ms: float
    reused: bool
    timing_markers: tuple[Limitation, ...]
    limits: Any
    debug: bool
    ctx: contextvars.Context
    size: int

    def run(self) -> Any:
        return _assemble(self, parse=True, extra=())

    def fallback(self, marker: Limitation) -> Any:
        return _assemble(self, parse=False, extra=(marker,))


def _parse_semantics(p: _PendingTxn) -> Any:
    """LLM semantics for a sealed transaction, or None. A module function
    with no try: the ONE swallow for a raising parser is the guard in
    `_assemble`, which counts and (under debug) logs — the bare
    `except: return None` this replaces was an uncounted I6 violation.

    The limits are THE JOB'S snapshot, not the seam's live attribute: that
    is wiring, not prose — a re-install between seal and finalize cannot
    change what bounds this parse.
    """
    return parse_llm_semantics(
        p.url_host,
        p.txn.path,
        p.txn.request_body,
        p.txn.response_body,
        p.limits,
    )


def _should_capture(
    prefilter: Prefilter, txn: Any, sem: Any, *, mode: CaptureMode, unparsed: bool = False
) -> bool:
    """The seam's capture decision: the transport prefilter composed with the
    one shared policy. The old method of the same name, made a function of
    its inputs (so the worker can ask it over a sealed `_PendingTxn`).

    `unparsed` widens `degraded` (§3.9): "wardex is the reason a gate input
    is missing" now covers the SEMANTIC CLAIM as well as the parent — a
    fallback that skipped the parse cannot honestly answer `agent_semantic`,
    exactly as `degraded_run` cannot honestly answer `parent`. The policy's
    signature does not change; the widening is this caller's input.

    Failing OPEN around the composition stays this function's job rather
    than the policy's: `has_core_semantics` runs parser output through
    host-supplied objects and can raise, `should_capture` cannot — so the
    swallow sits where the risk is (design §5.1: losing data is worse than
    noise).
    """
    try:
        if prefilter is Prefilter.DENY:
            return False
        if prefilter is Prefilter.ALLOW:
            return True
        return should_capture(
            mode,
            parent=getattr(txn, "parent", None),
            agent_semantic=sem is not None and _is_llm_traffic(txn, sem),
            degraded=unparsed or in_degraded_run() or getattr(txn, "parent_evicted", False),
            parent_closed=getattr(txn, "parent_closed", False),
        )
    except Exception:
        return True  # losing data is worse than noise (design §5.1)


def _assemble(p: _PendingTxn, *, parse: bool, extra: tuple[Limitation, ...]) -> Any:
    """One sealed transaction, finished into a span — on whatever thread.

    The old `_build_span` from the gate down, rewritten over `_PendingTxn`
    fields so the answer is the same wherever it runs (I-F10). Two callers:
    the finalize worker (`run`/`fallback`, inside the sealed `ctx`) and the
    gRPC inline branch of `_emit_span` (§3.6 — same thread that sealed, so
    its ambient context IS the sealed one).

    `parse=False` is the fallback shape: the parse is SKIPPED — backlog
    eviction, shutdown budget, spawn failure — and `extra` carries the
    marker that says which. A parse that RAISES is the third shape: counted
    under the parse guard, marked `INSTRUMENTATION_DEGRADED`, and the span
    still ships (the defect this design fixes alongside the stall: the old
    swallow left AGENT-mode spans silently gone with a zero counter).

    §3.9's restraint: when the gate admits the span ONLY because of
    wardex's own degradation (`unparsed` flipped the answer), the raw
    bodies are withheld. The user's mode excluded this traffic; overload
    must not become the reason its payloads leave the process. Transport
    metadata, timing, status and markers stay — what happened is still
    said; what was SAID in the bodies is not.
    """
    txn = p.txn
    sem: Any = None
    parse_failed = False
    if parse and not p.is_grpc:
        # `parsed` distinguishes "the parser raised" (swallowed and counted
        # by the guard) from the parser's own honest None ("not an LLM
        # body") — `sem` cannot carry both facts.
        parsed = False
        with guard("interceptors.seam.parse", debug=p.debug):
            sem = _parse_semantics(p)
            parsed = True
        if not parsed:
            parse_failed = True
    unparsed = ((not parse) or parse_failed) and not p.is_grpc
    if not _should_capture(p.prefilter, txn, sem, mode=p.mode, unparsed=unparsed):
        return None
    withhold_bodies = unparsed and not _should_capture(
        p.prefilter, txn, sem, mode=p.mode, unparsed=False
    )

    edge = resolve_observed(
        _latched(txn),
        parent_closed=txn.parent_closed,
        parent_evicted=txn.parent_evicted,
    )
    url = f"{p.url_scheme}://{p.url_host}:{p.server_port}{txn.path}"
    transfer = max(0.0, (txn.end_ns - txn.start_ns) / 1e6 - txn.ttfb_ms)

    # TRANSPORT mode, not an intent (design §6.1 correction in `_vocab.py`):
    # the seam knows a request happened and, on the branches below, what the
    # body meant — but `HTTP POST /v1/messages` reports the observation, and
    # §6.2's twelve intents hold no member for uninterpreted traffic.
    draft = SpanDraft.transport(
        edge,
        label=TransportLabel.HTTP,
        subject=f"{txn.method} {txn.path}",
        source=p.capture_source,
        start_ns=txn.start_ns,
    )
    # Sealed markers: the timing story (`_resolve_timing`) plus, on a
    # fork-crossing connection's first sealed transaction, the fork latch.
    for marker in p.timing_markers:
        draft.add_limitation(marker)
    # Markers the protocol parser attached to the transaction (a body that
    # hit its cap, say). They arrive as MEMBERS: the string-to-member
    # crossing happens once, at the PyO3 boundary in `_protocol/_http1.py`,
    # which is where the Rust `&'static str` actually enters Python.
    #
    # A plain attribute access, not `getattr(txn, "limitations", ())`. The
    # default could never fire — `_Txn.limitations` is a declared field —
    # but it is the exact shape that fails silently if the field is ever
    # renamed or a non-`_Txn` reaches here: every parser marker would
    # vanish with no error and no counter. An AttributeError is the correct
    # outcome for that, and it is what the callers' guards are for.
    for marker in txn.limitations:
        draft.add_limitation(marker)
    # The queue's verdict about THIS finalization: PARSE_BACKLOG_FULL,
    # PARSE_SKIPPED_AT_SHUTDOWN, or INSTRUMENTATION_DEGRADED (spawn failure).
    for marker in extra:
        draft.add_limitation(marker)
    if parse_failed:
        # The parser RAISED (already counted by the guard above): the span
        # ships saying wardex broke, instead of vanishing with a green
        # counter — SEMANTIC_PARSE_FAILED would be a lie here (the parser
        # never returned an answer) and silence was the old defect.
        draft.add_limitation(Limitation.INSTRUMENTATION_DEGRADED)

    output_data = txn.response_body
    status_code = StatusCode.OK if 200 <= txn.status < 400 else StatusCode.ERROR
    # `finish()` refuses `status=ERROR` with no `error.type` and a refused
    # span is a DELETED span, so this may not be left `None`: without it
    # every 4xx/5xx on every byte seam — the rate limit, the auth failure,
    # the provider outage, i.e. the highest-value spans this SDK captures —
    # disappears into a counter. The value is the response status rendered
    # as a string, which is what OTel's HTTP-client semconv prescribes when
    # the instrumentation has no richer classification. That is exactly
    # wardex's position here: the seam observed a 500, it did not observe
    # why. Low cardinality, and a fact rather than a guess.
    error_type: str | None = str(txn.status) if status_code is StatusCode.ERROR else None
    draft.set_extra("network.protocol.version", txn.version)

    ct = txn.content_type or ""
    if ct.startswith("application/grpc-web"):
        # grpc-web uses different framing and is unsupported — leave it as plain h2 but mark it.
        draft.add_limitation(Limitation.GRPC_WEB_UNSUPPORTED)

    if p.is_grpc:
        # gRPC fields; `sem` is None on this path (the parse above skips
        # gRPC) and the framing walk ALWAYS runs — inline and fallback alike
        # (there IS no fallback for gRPC: §3.6, nothing was deferred) — so
        # the GRPC label, status and error type are never downgraded.
        _name, status_code, error_type, grpc_extra, grpc_markers = build_grpc_fields(txn, (), ())
        for key, value in grpc_extra:
            draft.set_extra(key, value)
        for marker in grpc_markers:
            draft.add_limitation(marker)
        # The helper falls back to plain h2 when the framing parse fails and
        # says so with FRAME_PARSE_FAILED; the label follows the same
        # decision, so the span's name and its marker cannot disagree.
        if Limitation.FRAME_PARSE_FAILED not in grpc_markers:
            draft.relabel(TransportLabel.GRPC, txn.path)
    else:
        # --- LLM semantics (parsed above the gate) ---
        if sem is not None:
            if sem.decoded_response is not None:
                output_data = bytes(sem.decoded_response)
            streamed = bool(getattr(sem, "reassembled_from_stream", False))
            # The same predicate the capture gate used. Asking a different
            # question here is what produced a span the policy admitted as
            # agent traffic and then shipped with `gen_ai=None`.
            identified = _is_llm_traffic(txn, sem)
            if identified:
                draft.set_gen_ai(build_gen_ai(sem))
                # The open half rides only on an IDENTIFIED span: the
                # `openai.*` scalars, the provider-usage mirror, and —
                # when the key-count bound dropped leaves — the marker,
                # the count and the diagnostics bump as ONE fact. An
                # unidentified span gets none of the family, so a marker
                # explaining keys that are not there cannot exist.
                for key, value in provider_extras(sem):
                    draft.set_extra(key, value)
                dropped = getattr(sem, "usage_dropped_count", 0)
                if dropped:
                    draft.add_limitation(Limitation.EXTRA_KEYS_DROPPED)
                    draft.set_extra(USAGE_DROPPED_KEY, dropped)
                    counters.bump("interceptors.seam.usage_leaves_dropped")
                embeddings = embeddings_attrs(sem)
                if embeddings is not None:
                    draft.set_embeddings(embeddings)
            if streamed and sem.stream_terminated is False:
                # Outside the identity gate on purpose: pure diagnostics
                # (no marker, no wire artifact) building the volume
                # evidence a STREAM_INCOMPLETE-class marker would need.
                counters.bump("interceptors.seam.stream_unterminated")
            if streamed:
                # Keyed on `streamed` rather than on the response having
                # yielded semantics: a reassembled stream is reassembled
                # whatever came back, and hanging these off the identity
                # test makes a known provider's usage-less stream stop
                # saying it was reassembled at all.
                if identified:
                    draft.add_limitation(Limitation.REASSEMBLED_FROM_STREAM)
                    if sem.output_tokens is None:
                        draft.add_limitation(Limitation.STREAM_USAGE_UNAVAILABLE)
                    if txn.version == "2":
                        draft.add_limitation(Limitation.TTFT_UNAVAILABLE_H2)
                else:
                    draft.add_limitation(Limitation.SSE_UNKNOWN_PROVIDER)
            elif not has_core_semantics(sem) and status_code is StatusCode.OK:
                # Narrowed to a SUCCESSFUL response the parser could not
                # read. A 4xx/5xx body is an error envelope; the parse did
                # not fail, so claiming it did sent every rate limit and
                # every auth failure out under a marker that says "wardex
                # could not understand this" when the truth — already on the
                # span as `status=ERROR` and `error.type` — is that the
                # provider refused.
                #
                # No need for semantic_parse_failed if tool_calls extraction succeeded.
                if sem.output_messages is None:
                    draft.add_limitation(Limitation.SEMANTIC_PARSE_FAILED)
            # Structured extraction of output.messages
            om = sem.output_messages
            if om is not None:
                draft.set_extra("gen_ai.output.messages", om)
                if sem.tool_args_unparsed:
                    draft.add_limitation(Limitation.TOOL_ARGS_UNPARSED)
                if sem.output_messages_has_unmapped:
                    draft.add_limitation(Limitation.OUTPUT_MESSAGES_UNMAPPED_PART)
            # Structured extraction of input.messages / system_instructions
            im = sem.input_messages
            if im is not None:
                draft.set_extra("gen_ai.input.messages", im)
                if sem.input_messages_has_unmapped:
                    draft.add_limitation(Limitation.INPUT_MESSAGES_UNMAPPED_PART)
            si = sem.system_instructions
            if si is not None:
                draft.set_extra("gen_ai.system_instructions", si)

    timing = TransportTiming(
        tcp_connect_ms=p.connect_ms,
        tls_handshake_ms=p.handshake_ms,
        ttfb_ms=txn.ttfb_ms,
        ttft_ms=txn.ttft_ms,
        transfer_ms=transfer,
    )
    # response_size is the wire (compressed) size, while output_data is the
    # decompressed body, so lengths may differ for gzip responses (intended behavior).
    draft.set_transport(
        TransportAttributes(
            connection_id=p.connection_id,
            protocol=Protocol.HTTP,
            direction=Direction.OUTBOUND,
            timing=timing,
            request_size=len(txn.request_body),
            response_size=len(txn.response_body),
            http=HttpMeta(method=txn.method, url=url, status_code=txn.status),
            connection_reused=p.reused,
        )
    )
    draft.set_server(p.server_address, p.server_port)
    draft.set_status(status_code)
    draft.set_error(error_type)
    if not withhold_bodies:
        # "attempted and succeeded", not "non-empty": the seam read both
        # bodies off the tracker, and a zero-length body is a captured
        # zero-length body. Withheld only under §3.9's restraint above.
        draft.set_io(input_data=txn.request_body, output_data=output_data)
    draft.integrity.truncated(txn.truncated)
    return draft.finish(txn.end_ns)


def _latched(txn: _Txn) -> Ambient:
    """The scope as it was when this transaction's request was ISSUED.

    Both emit paths run on the RESPONSE side, where the ambient context has
    already moved on — so neither may call `latch_ambient()` itself. The tracker
    did the latching at request time (`_trackers.py`, `self._parent`), and this
    wraps what it captured in the shape `resolve_parentage` consumes.

    `parent_closed` travels BESIDE this rather than inside the `Ambient`, for
    the same reason: it is a fact about the latch INSTANT, and an `Ambient` is
    the shape `resolve_parentage` consumes, not a place to keep one seam's
    bookkeeping. `_build_span` and `_build_ws_span` pass it explicitly.

    `conversation` and `tracestate` are None because the tracker latches neither
    today; that is exactly the pre-existing behaviour (the seam never set
    `InternalSpan.conversation`), and widening the latch to a full `Ambient`
    belongs with the seam decomposition (design §3.3) — it is a change to what
    the tracker captures at request time, not to how this function shapes what
    it already captured.
    """
    return Ambient(span_context=txn.parent, conversation=None, tracestate=None)


def _peer(obj: Any) -> tuple[str, int]:
    try:
        peer = obj.getpeername()
        return str(peer[0]), int(peer[1])
    except Exception:
        host = getattr(obj, "server_hostname", None) or "unknown"
        return str(host), 443
