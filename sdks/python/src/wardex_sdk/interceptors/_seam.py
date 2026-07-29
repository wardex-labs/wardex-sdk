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
    OperationName,
    Protocol,
    ProviderName,
    StatusCode,
)
from .._limits import CaptureLimits
from .._types import (
    GenAIAttributes,
    HttpMeta,
    TransportAttributes,
    TransportTiming,
)
from ..assembly import (
    Ambient,
    Limitation,
    Prefilter,
    SpanDraft,
    TransportLabel,
    capture_mode_of,
    guard,
    resolve_parentage,
    should_capture,
)
from ..assembly._vocab import transport_name
from ..protocol import grpc_status_name, parse_grpc_frames, parse_llm_semantics
from ._base import InterceptorInterface
from ._trackers import _Txn, _WebSocketTracker

if TYPE_CHECKING:
    from .._client import Client

# Headers are parsed but intentionally not recorded on the span (secret protection);
# PII masking is Phase 3.


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
        self._orig: dict[str, Any] = {}
        self._installed = False
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
        this method's job rather than the policy's: `_has_core_semantics` runs
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
                agent_semantic=sem is not None and _has_core_semantics(sem),
            )
        except Exception:
            return True  # losing data is worse than noise (design §5.1)

    def _capture_source(self) -> CaptureSource:
        return CaptureSource.SSL

    # --- Monkeypatch helpers ---

    def _patch(self, cls: type, meth: str, wrapper: Any) -> None:
        key = f"{cls.__name__}.{meth}"
        self._orig[key] = getattr(cls, meth)
        setattr(cls, meth, wrapper)

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

        p = resolve_parentage(_latched(txn))
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
            _name, status_code, error_type, grpc_extra, grpc_markers = _build_grpc_fields(
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
                if _has_core_semantics(sem):
                    draft.set_gen_ai(_build_gen_ai(sem))
                    if streamed:
                        draft.add_limitation(Limitation.REASSEMBLED_FROM_STREAM)
                        if sem.output_tokens is None:
                            draft.add_limitation(Limitation.STREAM_USAGE_UNAVAILABLE)
                        if txn.version == "2":
                            draft.add_limitation(Limitation.TTFT_UNAVAILABLE_H2)
                elif streamed:
                    draft.add_limitation(Limitation.SSE_UNKNOWN_PROVIDER)
                else:
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
        p = resolve_parentage(_latched(txn))

        code = txn.ws_close_code
        # Status based on close code: 1000/1001/none = OK, otherwise = ERROR
        if code is None or code in (1000, 1001):
            status_code = StatusCode.OK
            error_type = None
        else:
            status_code = StatusCode.ERROR
            error_type = _ws_close_name(code)

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
    belongs with the seam decomposition (design §3.3), not with this step.
    """
    return Ambient(span_context=txn.parent, conversation=None, tracestate=None)


_OPERATION_MAP = {"chat": OperationName.CHAT, "embeddings": OperationName.EMBEDDINGS}
_PROVIDER_MAP = {"openai": ProviderName.OPENAI, "anthropic": ProviderName.ANTHROPIC}


def _has_core_semantics(sem: Any) -> bool:
    """True if at least one core semantic (model, tokens) is present."""
    return (
        sem.input_tokens is not None
        or sem.output_tokens is not None
        or sem.response_model is not None
    )


def _ws_close_name(code: int) -> str:
    return {
        1002: "protocol_error",
        1003: "unsupported_data",
        1007: "invalid_payload",
        1008: "policy_violation",
        1009: "message_too_big",
        1010: "mandatory_extension",
        1011: "internal_error",
    }.get(code, f"close_{code}")


def _build_grpc_fields(
    txn: _Txn,
    extra: tuple[tuple[str, str | int | float | bool], ...],
    limitations: tuple[Limitation, ...],
) -> tuple[
    str,
    StatusCode,
    str | None,
    tuple[tuple[str, str | int | float | bool], ...],
    tuple[Limitation, ...],
]:
    """Assemble gRPC span fields → (name, status_code, error_type, extra, limitations).

    Pure, and deliberately still returning a tuple rather than mutating a draft:
    it is the one branch of the seam with enough protocol logic to be worth
    testing without a socket, and it is where the census scanner's R6 rule finds
    the gRPC markers.

    On a framing-parse failure the span falls back to plain h2 (HTTP) fields plus
    `Limitation.FRAME_PARSE_FAILED`. The caller keys its own label off exactly
    that marker, so the name and the marker cannot disagree about whether this
    was gRPC.

    Markers are `Limitation` members as of step 3a, and two of these values
    changed name on the way in (§6.5.1): `grpc_parse_failed` became
    `FRAME_PARSE_FAILED` because a WebSocket framing failure is the same fact,
    and `grpc_compressed` became `PAYLOAD_COMPRESSED` because
    `TransportAttributes.protocol` already carries which protocol it was and
    encoding that into the marker duplicates a field.
    """
    try:
        req = parse_grpc_frames(txn.request_body)
        resp = parse_grpc_frames(txn.response_body)
    except Exception:
        http_status = StatusCode.OK if 200 <= txn.status < 400 else StatusCode.ERROR
        return (
            transport_name(TransportLabel.HTTP, f"{txn.method} {txn.path}"),
            http_status,
            # Plain-h2 fallback, so the plain-h2 error type: the HTTP status as
            # a string. Returning `None` here alongside an ERROR status is what
            # `finish()` deletes the span for.
            str(txn.status) if http_status is StatusCode.ERROR else None,
            extra,
            limitations + (Limitation.FRAME_PARSE_FAILED,),
        )

    name = transport_name(TransportLabel.GRPC, txn.path)
    code = txn.grpc_status
    status_code = StatusCode.ERROR if code not in (0, None) else StatusCode.OK
    error_type = grpc_status_name(code) if status_code is StatusCode.ERROR else None

    # "/pkg.Svc/Method" → service="pkg.Svc", method="Method"
    service, method = "", ""
    trimmed = txn.path.lstrip("/")
    if "/" in trimmed:
        service, method = trimmed.rsplit("/", 1)
    else:
        method = trimmed

    extra = extra + (
        ("rpc.system", "grpc"),
        ("rpc.service", service),
        ("rpc.method", method),
        ("rpc.grpc.request.message_count", len(req.messages)),
        ("rpc.grpc.response.message_count", len(resp.messages)),
    )
    if code is not None:
        extra = extra + (("rpc.grpc.status_code", code),)
    # If grpc-message is present, include it on the span — useful for diagnosing errors
    # (e.g. "NOT_FOUND: collection x missing")
    if txn.grpc_message:
        extra = extra + (("rpc.grpc.status_message", txn.grpc_message),)

    if code is None:
        limitations = limitations + (Limitation.GRPC_STATUS_UNAVAILABLE,)
    if any(m.compressed for m in req.messages) or any(m.compressed for m in resp.messages):
        limitations = limitations + (Limitation.PAYLOAD_COMPRESSED,)
    if req.truncated or resp.truncated:
        limitations = limitations + (Limitation.GRPC_MESSAGE_TRUNCATED,)

    return name, status_code, error_type, extra, limitations


def _build_gen_ai(sem: Any) -> GenAIAttributes:
    """Convert LlmSemantics → GenAIAttributes."""
    stops = tuple(sem.stop_sequences) if sem.stop_sequences else None
    finishes = tuple(sem.finish_reasons) if sem.finish_reasons else None
    return GenAIAttributes(
        operation=_OPERATION_MAP.get(sem.operation, sem.operation),
        provider=_PROVIDER_MAP.get(sem.provider, sem.provider),
        request_model=sem.request_model,
        response_model=sem.response_model,
        response_id=sem.response_id,
        input_tokens=sem.input_tokens,
        output_tokens=sem.output_tokens,
        cache_read_input_tokens=sem.cache_read_input_tokens,
        cache_creation_input_tokens=sem.cache_creation_input_tokens,
        reasoning_output_tokens=sem.reasoning_output_tokens,
        temperature=sem.temperature,
        max_tokens=sem.max_tokens,
        top_p=sem.top_p,
        top_k=sem.top_k,
        seed=sem.seed,
        frequency_penalty=sem.frequency_penalty,
        presence_penalty=sem.presence_penalty,
        choice_count=sem.choice_count,
        stop_sequences=stops,
        stream=sem.stream,
        finish_reasons=finishes,
        output_type=sem.output_type,
    )


def _peer(obj: Any) -> tuple[str, int]:
    try:
        peer = obj.getpeername()
        return str(peer[0]), int(peer[1])
    except Exception:
        host = getattr(obj, "server_hostname", None) or "unknown"
        return str(host), 443
