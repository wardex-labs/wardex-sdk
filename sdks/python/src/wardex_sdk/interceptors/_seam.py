"""Byte seam interceptor base — shared logic for feeding trackers and assembling spans.

Common skeleton that streams app-layer plaintext bytes into a protocol tracker and
assembles a CLIENT span from the _Txn the tracker returns. SSL (wss/https) and
plaintext (ws/http) seams inherit this and override only seam-specific behavior
(tracker selection, timing resolution, scheme, etc). Works standalone, without adapters.
"""

from __future__ import annotations

import sys
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from .._enums import (
    CaptureSource,
    Direction,
    Protocol,
    StatusCode,
)
from .._limits import CaptureLimits
from .._types import (
    HttpMeta,
    TransportAttributes,
    TransportTiming,
)
from ..assembly import (
    Ambient,
    Limitation,
    PatchSet,
    Prefilter,
    SpanDraft,
    TransportLabel,
    capture_mode_of,
    guard,
    in_degraded_run,
    resolve_observed,
    should_capture,
)
from ..protocol import parse_llm_semantics
from ..semantics import (
    build_gen_ai,
    build_grpc_fields,
    has_core_semantics,
    identifies_llm_call,
    ws_close_name,
)
from ._base import InterceptorInterface
from ._conn_timing import install_shared_timing, uninstall_shared_timing
from ._trackers import _Txn, _WebSocketTracker

if TYPE_CHECKING:
    from .._client import Client

# Headers are parsed but intentionally not recorded on the span (secret protection).
# Bodies are recorded, and their PII is masked later — in the Rust core, at encode
# time (`codec.encode_otlp_traces` takes the mode and the disabled categories), not
# here.


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
        # Guards the once-per-connection debug log below. Deliberately a
        # separate field from `gate`: `gate` is owned by the plaintext seam's
        # protocol sniff-latch (_socket.py), which never re-evaluates once
        # set — reusing it here would let a disable-log event permanently
        # gate off all further bytes on that seam, including a still-healthy
        # direction, as an unintended side effect of two unrelated concerns
        # sharing one field.
        self.disabled_logged = False


class ByteSeamInterceptor(InterceptorInterface):
    """Shared byte seam base — tracker feed + span assembly.
    Seam-specific behavior is a subclass hook."""

    def __init__(self) -> None:
        self._client: Client | None = None
        self._conns: dict[int, _ConnectionState] = {}
        self._patches = PatchSet(f"interceptors.{self.name()}")
        self._installed = False
        # Does THIS seam hold a reference on the shared timing probe? A second
        # field rather than a reading of `_installed`, because `uninstall()` is
        # now total: it runs after an `install()` that raised, where the two
        # answers differ. Releasing a reference this seam never took decrements
        # the shared refcount to zero underneath the OTHER seam and rips
        # `socket.connect` back out from under a live installation.
        self._timing_held = False
        # Defaults match the core's, so behavior is unchanged until _load_limits
        # resolves an actual config at install() time.
        self._limits: dict[str, int] = CaptureLimits().resolved()
        self._native_limits: Any = None

    def _load_limits(self, client: Client | None) -> None:
        """Cache resolved limits at install time; config is frozen after init."""
        config = getattr(client, "config", None)
        lim = config.limits if config is not None else CaptureLimits()
        self._limits = lim.resolved()
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
        idempotent and empty until the first `patch()` lands, `_timing_held`
        records the one acquisition that is refcounted elsewhere and so must not
        be released twice or unearned, and `_conns` is empty until a byte flows.
        Every step is a no-op on a seam that never installed, which is what
        makes this safe to call unconditionally, twice, or on a fresh object.

        The WebSocket flush stays: a live WS session holds a span that only
        exists once the session ends, and dropping it at uninstall would be the
        SDK losing data at teardown — the reason `uninstall_all` runs before
        `client.close()` at all.
        """
        self._patches.restore_all()
        self._release_timing()
        for st in list(self._conns.values()):
            if isinstance(st.tracker, _WebSocketTracker):
                for txn in st.tracker.flush(Limitation.WS_NO_CLOSE):
                    self._emit_ws(st, txn)
        self._conns.clear()
        self._installed = False

    def _acquire_timing(self) -> None:
        """Take this seam's reference on the shared connection-timing probe.

        The flag is set AFTER the call, so a raising `install_shared_timing`
        leaves nothing for `_release_timing` to give back.
        """
        install_shared_timing(self._limits["max_connections"])
        self._timing_held = True

    def _release_timing(self) -> None:
        """Give back the reference this seam took, at most once, and only if taken."""
        if not self._timing_held:
            return
        self._timing_held = False
        uninstall_shared_timing()

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

    def _should_capture(self, st: _ConnectionState, txn: Any, sem: Any) -> bool:
        """Compose this seam's transport prefilter with the one shared policy.

        The policy itself lives in `assembly._policy` and is asked here, at the
        single point where both halves are known. Failing open around it is
        this method's job rather than the policy's: `has_core_semantics` runs
        parser output through host-supplied objects and can raise, while
        `should_capture` cannot — so the swallow stays where the risk is.
        """
        try:
            pre = self._transport_prefilter(st)
            if pre is Prefilter.DENY:
                return False
            if pre is Prefilter.ALLOW:
                return True
            return should_capture(
                capture_mode_of(self._client),
                parent=getattr(txn, "parent", None),
                agent_semantic=sem is not None and _is_llm_traffic(txn, sem),
                # "a span wardex FAILED to open is what should have been ambient
                # here". Without it the gate reads an absent parent as "not
                # agent work" and drops every request inside a run wardex broke
                # at the top of — the one case where the absent parent is
                # wardex's own doing rather than evidence about the traffic.
                degraded=in_degraded_run(),
            )
        except Exception:
            return True  # losing data is worse than noise (design §5.1)

    def _capture_source(self) -> CaptureSource:
        return CaptureSource.SSL

    # --- Connection state ---

    def _state(self, obj: Any) -> _ConnectionState:
        cid = id(obj)
        st = self._conns.get(cid)
        if st is None:
            addr, port = _peer(obj)
            st = _ConnectionState(self._select_tracker(obj), addr, port)
            if len(self._conns) > self._limits["max_connections"]:
                old_cid = next(iter(self._conns))
                old_st = self._conns.pop(old_cid)
                if isinstance(old_st.tracker, _WebSocketTracker):
                    for txn in old_st.tracker.flush(Limitation.CONNECTION_EVICTED):
                        self._emit_ws(old_st, txn)
            self._conns[cid] = st
        return st

    # --- Tracker delegation + span assembly ---

    def _on_request_bytes(self, obj: Any, data: bytes) -> None:
        from ._exclusion import is_suppressed

        if is_suppressed():
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
        from ._exclusion import is_suppressed

        if is_suppressed():
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
                    print(
                        f"[wardex] parser disabled for {st.server_address}: {reason}",
                        file=sys.stderr,
                    )
        except Exception:  # noqa: BLE001 — debug-only logging must never break capture
            pass
        for txn in txns:
            if getattr(txn, "ws_upgrade", False):
                ws = _WebSocketTracker(
                    path=txn.ws_upgrade_path or "/",
                    deflate=txn.ws_deflate,
                    parent=txn.parent,
                    start_ns=txn.start_ns,
                    limits=self._native_limits,
                    sample_cap=self._limits["ws_sample_bytes"],
                )
                st.tracker = ws
                if txn.ws_leftover:
                    for t2 in ws.on_response_bytes(txn.ws_leftover):
                        self._emit_ws(st, t2)
            elif txn.version == "websocket":
                self._emit_ws(st, txn)
            else:
                self._emit_span(obj, st, txn)

    def _parse_semantics(self, url_host: str, txn: _Txn) -> Any:
        """LLM semantics for this transaction, or None. Pure — no seam state.

        Split out of `_emit_span` because the gate needs its answer: whether a
        transaction is agent traffic is one of the policy's three inputs, so
        the parse has to happen before the gate can run. It is safe there
        precisely because it is pure — it reads bodies the tracker already
        buffered and touches nothing on the connection.
        """
        try:
            # The resolved limits must travel with the call: the semantic
            # parser bounds decompression by max_decoded_bytes, and omitting
            # them here would silently run on the core default.
            return parse_llm_semantics(
                url_host,
                txn.path,
                txn.request_body,
                txn.response_body,
                self._native_limits,
            )
        except Exception:
            return None

    def _emit_span(self, obj: Any, st: _ConnectionState, txn: _Txn) -> None:
        client = self._client
        if client is None:
            return
        span = None
        with self._guard("interceptors.seam.emit_span"):
            span = self._build_span(obj, st, txn)
        if span is not None:
            client.capture_span(span)

    def _build_span(self, obj: Any, st: _ConnectionState, txn: _Txn) -> Any:
        url_host = getattr(obj, "server_hostname", None) or st.server_address

        ct = txn.content_type or ""
        # gRPC: skip LLM semantic extraction (protobuf isn't LLM JSON).
        is_grpc = ct.startswith("application/grpc") and not ct.startswith("application/grpc-web")
        sem: Any = None if is_grpc else self._parse_semantics(url_host, txn)

        # ABOVE the gate on purpose, and it must stay there. `_resolve_timing`
        # is DESTRUCTIVE — it sets `st.timing_consumed` and pops this
        # connection's record out of the shared store (`_ssl.py`, `_socket.py`)
        # — and the fact it encodes is "was this the FIRST transaction on this
        # connection", which is what `connection_reused` means (span.proto:
        # "whether the connection was reused via pooling"). Move it below the
        # gate and "first" silently becomes "first CAPTURED", so on a keep-alive
        # connection whose first request was dropped, the next request — one
        # that reused an established socket and paid no connect or handshake —
        # reports the previous request's `tcp_connect_ms` with
        # `connection_reused=False`. Both are false, and a present span with a
        # false measurement is worse than a connect cost nobody claims.
        # Running it for a dropped transaction also keeps that connection's slot
        # from sitting in the FIFO-capped store until eviction.
        connect_ms, handshake_ms, reused, timing_markers = self._resolve_timing(obj, st)

        if not self._should_capture(st, txn, sem):
            return None

        p = resolve_observed(_latched(txn))
        url = f"{self._url_scheme(False)}://{url_host}:{st.server_port}{txn.path}"
        transfer = max(0.0, (txn.end_ns - txn.start_ns) / 1e6 - txn.ttfb_ms)

        # TRANSPORT mode, not an intent (design §6.1 correction in `_vocab.py`):
        # the seam knows a request happened and, on the branches below, what the
        # body meant — but `HTTP POST /v1/messages` reports the observation, and
        # §6.2's twelve intents hold no member for uninterpreted traffic.
        draft = SpanDraft.transport(
            p,
            label=TransportLabel.HTTP,
            subject=f"{txn.method} {txn.path}",
            source=self._capture_source(),
            start_ns=txn.start_ns,
        )
        for marker in timing_markers:
            draft.add_limitation(marker)
        # Markers the protocol parser attached to the transaction (a body that
        # hit its cap, say). They arrive as MEMBERS: the string-to-member
        # crossing happens once, at the PyO3 boundary in `protocol/_http1.py`,
        # which is where the Rust `&'static str` actually enters Python.
        #
        # It used to happen here as well, and that was survivable only while the
        # first conversion did not exist. Two `from_wire` calls in series is not
        # idempotent — the second is handed a member, finds no string key, and
        # returns None — so the marker would be dropped by the very code written
        # to preserve it. One boundary, and it is the earliest one.
        # A plain attribute access, not `getattr(txn, "limitations", ())`. The
        # default could never fire — `_Txn.limitations` is a declared field —
        # but it is the exact shape that fails silently if the field is ever
        # renamed or a non-`_Txn` reaches here: every parser marker would
        # vanish with no error and no counter. An AttributeError is the correct
        # outcome for that, and it is what the surrounding `guard()` is for.
        for marker in txn.limitations:
            draft.add_limitation(marker)

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

        if ct.startswith("application/grpc-web"):
            # grpc-web uses different framing and is unsupported — leave it as plain h2 but mark it.
            draft.add_limitation(Limitation.GRPC_WEB_UNSUPPORTED)

        if is_grpc:
            # gRPC fields; `sem` is already None (see the parse above).
            _name, status_code, error_type, grpc_extra, grpc_markers = build_grpc_fields(
                txn, (), ()
            )
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
            tcp_connect_ms=connect_ms,
            tls_handshake_ms=handshake_ms,
            ttfb_ms=txn.ttfb_ms,
            ttft_ms=txn.ttft_ms,
            transfer_ms=transfer,
        )
        # response_size is the wire (compressed) size, while output_data is the
        # decompressed body, so lengths may differ for gzip responses (intended behavior).
        draft.set_transport(
            TransportAttributes(
                connection_id=str(id(obj)),
                protocol=Protocol.HTTP,
                direction=Direction.OUTBOUND,
                timing=timing,
                request_size=len(txn.request_body),
                response_size=len(txn.response_body),
                http=HttpMeta(method=txn.method, url=url, status_code=txn.status),
                connection_reused=reused,
            )
        )
        draft.set_server(st.server_address, st.server_port)
        draft.set_status(status_code)
        draft.set_error(error_type)
        # "attempted and succeeded", not "non-empty": the seam read both bodies
        # off the tracker, and a zero-length body is a captured zero-length body.
        draft.set_io(input_data=txn.request_body, output_data=output_data)
        draft.integrity.truncated(txn.truncated)
        return draft.finish(txn.end_ns)

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
        # `agent_semantic=False`: a WS session carries no parsed LLM semantics
        # (`sem` is None on this path by construction), so it is captured only
        # under ALL, an allowlisted host, or a live local span.
        if not self._should_capture(st, txn, None):
            return None
        p = resolve_observed(_latched(txn))

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


def _latched(txn: _Txn) -> Ambient:
    """The scope as it was when this transaction's request was ISSUED.

    Both emit paths run on the RESPONSE side, where the ambient context has
    already moved on — so neither may call `latch_ambient()` itself. The tracker
    did the latching at request time (`_trackers.py`, `self._parent`), and this
    wraps what it captured in the shape `resolve_parentage` consumes.

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
