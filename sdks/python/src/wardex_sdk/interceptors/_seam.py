"""Byte seam interceptor base — shared logic for feeding trackers and assembling spans.

Common skeleton that streams app-layer plaintext bytes into a protocol tracker and
assembles a CLIENT span from the _Txn the tracker returns. SSL (wss/https) and
plaintext (ws/http) seams inherit this and override only seam-specific behavior
(tracker selection, timing resolution, scheme, etc). Works standalone, without adapters.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from .._enums import (
    CaptureMode,
    CaptureSource,
    Direction,
    OperationName,
    Protocol,
    ProviderName,
    SpanKind,
    StatusCode,
)
from .._limits import CaptureLimits
from .._types import (
    CaptureIntegrity,
    CorrelationInfo,
    GenAIAttributes,
    HttpMeta,
    InternalSpan,
    SpanContext,
    SpanId,
    TraceId,
    TransportAttributes,
    TransportTiming,
)
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
        self.gate: str | None = None  # None=undetermined, "http", "h2c", "ignore"


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
    ) -> tuple[float, float, bool, tuple[str, ...]]: ...

    def _url_scheme(self, is_ws: bool) -> str:
        return "wss" if is_ws else "https"

    def _gate(self, st: _ConnectionState, data: bytes, phase: str) -> bool:
        return True

    def _should_capture(self, st: _ConnectionState, txn: Any, sem: Any) -> bool:
        """Capture-policy gate (Phase 4c, design §5.1).

        AGENT (default): LLM-semantic traffic always; anything else only when
        the tracker latched a *local* wardex span as parent. Remote-only
        context (a joined trace with no local span) does not open the gate —
        service meshes attach traceparent to every request, and that must not
        resurrect the firehose. Fails open: losing data is worse than noise.
        """
        try:
            client = self._client
            if client is None or client.config.capture_mode is CaptureMode.ALL:
                return True
            if sem is not None and _has_core_semantics(sem):
                return True
            parent = getattr(txn, "parent", None)
            return parent is not None and not getattr(parent, "is_remote", False)
        except Exception:
            return True

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
                    for txn in old_st.tracker.flush("ws_evicted"):
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
        for txn in st.tracker.on_response_bytes(data):
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

    def _emit_span(self, obj: Any, st: _ConnectionState, txn: _Txn) -> None:
        if self._client is None:
            return
        active = txn.parent
        if active is not None:
            trace_id = active.trace_id
            parent_span_id: SpanId | None = active.span_id
            correlation: CorrelationInfo | None = CorrelationInfo(
                strategy="contextvar",
                active_span_id_at_capture=active.span_id,
                confidence=1.0,
            )
        else:
            trace_id = TraceId.generate()
            parent_span_id = None
            correlation = None

        ctx = SpanContext(trace_id=trace_id, span_id=SpanId.generate())
        url_host = getattr(obj, "server_hostname", None) or st.server_address
        url = f"{self._url_scheme(False)}://{url_host}:{st.server_port}{txn.path}"
        transfer = max(0.0, (txn.end_ns - txn.start_ns) / 1e6 - txn.ttfb_ms)

        connect_ms, handshake_ms, reused, limitations = self._resolve_timing(obj, st)

        # Default values for common span fields (HTTP path). If gRPC, _build_grpc_fields
        # overrides them.
        sem: Any = None
        gen_ai = None
        output_data = txn.response_body
        name = f"HTTP {txn.method} {txn.path}"
        status_code = StatusCode.OK if 200 <= txn.status < 400 else StatusCode.ERROR
        error_type: str | None = None
        extra: tuple[tuple[str, str | int | float | bool], ...] = (
            ("network.protocol.version", txn.version),
        )

        ct = txn.content_type or ""
        if ct.startswith("application/grpc-web"):
            # grpc-web uses different framing and is unsupported — leave it as plain h2 but mark it.
            limitations = limitations + ("grpc_web_unsupported",)

        is_grpc = ct.startswith("application/grpc") and not ct.startswith("application/grpc-web")

        if is_grpc:
            # gRPC: skip LLM semantic extraction (protobuf isn't LLM JSON), assemble gRPC fields.
            name, status_code, error_type, extra, limitations = _build_grpc_fields(
                txn, extra, limitations
            )
        else:
            # --- LLM semantic extraction (body parser) ---
            try:
                sem = parse_llm_semantics(url_host, txn.path, txn.request_body, txn.response_body)
            except Exception:
                sem = None
            if sem is not None:
                if sem.decoded_response is not None:
                    output_data = bytes(sem.decoded_response)
                streamed = bool(getattr(sem, "reassembled_from_stream", False))
                if _has_core_semantics(sem):
                    gen_ai = _build_gen_ai(sem)
                    if streamed:
                        limitations = limitations + ("reassembled_from_stream",)
                        if sem.output_tokens is None:
                            limitations = limitations + ("stream_usage_unavailable",)
                        if txn.version == "2":
                            limitations = limitations + ("ttft_unavailable_h2",)
                elif streamed:
                    limitations = limitations + ("sse_unknown_provider",)
                else:
                    # No need for semantic_parse_failed if tool_calls extraction succeeded.
                    if sem.output_messages is None:
                        limitations = limitations + ("semantic_parse_failed",)
                # Structured extraction of output.messages
                om = sem.output_messages
                if om is not None:
                    extra = extra + (("gen_ai.output.messages", om),)
                    if sem.tool_args_unparsed:
                        limitations = limitations + ("tool_args_unparsed",)
                    if sem.output_messages_has_unmapped:
                        limitations = limitations + ("output_messages_unmapped_part",)
                # Structured extraction of input.messages / system_instructions
                im = sem.input_messages
                if im is not None:
                    extra = extra + (("gen_ai.input.messages", im),)
                    if sem.input_messages_has_unmapped:
                        limitations = limitations + ("input_messages_unmapped_part",)
                si = sem.system_instructions
                if si is not None:
                    extra = extra + (("gen_ai.system_instructions", si),)

        timing = TransportTiming(
            tcp_connect_ms=connect_ms,
            tls_handshake_ms=handshake_ms,
            ttfb_ms=txn.ttfb_ms,
            ttft_ms=txn.ttft_ms,
            transfer_ms=transfer,
        )
        # response_size is the wire (compressed) size, while output_data is the
        # decompressed body, so lengths may differ for gzip responses (intended behavior).
        transport = TransportAttributes(
            connection_id=str(id(obj)),
            protocol=Protocol.HTTP,
            direction=Direction.OUTBOUND,
            timing=timing,
            request_size=len(txn.request_body),
            response_size=len(txn.response_body),
            http=HttpMeta(method=txn.method, url=url, status_code=txn.status),
            connection_reused=reused,
        )
        integrity = CaptureIntegrity(
            request_headers_captured=False,
            request_body_captured=True,
            response_headers_captured=False,
            response_body_captured=True,
            truncated=txn.truncated,
            limitations=limitations,
        )
        if not self._should_capture(st, txn, sem):
            return
        span = InternalSpan(
            context=ctx,
            parent_span_id=parent_span_id,
            name=name,
            kind=SpanKind.CLIENT,
            start_time_ns=txn.start_ns,
            end_time_ns=txn.end_ns,
            status=status_code,
            error_type=error_type,
            gen_ai=gen_ai,
            transport=transport,
            server_address=st.server_address,
            server_port=st.server_port,
            input_data=txn.request_body,
            output_data=output_data,
            capture_sources=(self._capture_source(),),
            capture_integrity=integrity,
            correlation=correlation,
            extra=extra,
        )
        self._client.capture_span(span)

    def _emit_ws(self, st: _ConnectionState, txn: _Txn) -> None:
        if self._client is None:
            return
        active = txn.parent
        if active is not None:
            trace_id = active.trace_id
            parent_span_id: SpanId | None = active.span_id
            correlation: CorrelationInfo | None = CorrelationInfo(
                strategy="contextvar",
                active_span_id_at_capture=active.span_id,
                confidence=1.0,
            )
        else:
            trace_id = TraceId.generate()
            parent_span_id = None
            correlation = None
        ctx = SpanContext(trace_id=trace_id, span_id=SpanId.generate())

        code = txn.ws_close_code
        # Status based on close code: 1000/1001/none = OK, otherwise = ERROR
        if code is None or code in (1000, 1001):
            status_code = StatusCode.OK
            error_type = None
        else:
            status_code = StatusCode.ERROR
            error_type = _ws_close_name(code)

        extra: tuple[tuple[str, str | int | float | bool], ...] = (
            ("network.protocol.version", "websocket"),
            ("ws.messages.sent", txn.ws_messages_sent),
            ("ws.messages.received", txn.ws_messages_received),
            ("ws.bytes.sent", txn.ws_bytes_sent),
            ("ws.bytes.received", txn.ws_bytes_received),
        )
        if code is not None:
            extra = extra + (("ws.close_code", code),)

        if not self._should_capture(st, txn, None):
            return

        timing = TransportTiming(
            tcp_connect_ms=0.0,
            tls_handshake_ms=0.0,
            ttfb_ms=0.0,
            ttft_ms=0.0,
            transfer_ms=max(0.0, (txn.end_ns - txn.start_ns) / 1e6),
        )
        transport = TransportAttributes(
            connection_id=str(id(st)),
            protocol=Protocol.HTTP,
            direction=Direction.OUTBOUND,
            timing=timing,
            request_size=txn.ws_bytes_sent,
            response_size=txn.ws_bytes_received,
            http=HttpMeta(
                method="GET",
                url=f"{self._url_scheme(True)}://{st.server_address}:{st.server_port}{txn.path}",
                status_code=101,
            ),
            connection_reused=False,
        )
        integrity = CaptureIntegrity(
            request_headers_captured=False,
            request_body_captured=True,
            response_headers_captured=False,
            response_body_captured=True,
            truncated="ws_payload_truncated" in txn.ws_markers,
            limitations=txn.ws_markers,
        )
        span = InternalSpan(
            context=ctx,
            parent_span_id=parent_span_id,
            name=f"WS {txn.path}",
            kind=SpanKind.CLIENT,
            start_time_ns=txn.start_ns,
            end_time_ns=txn.end_ns,
            status=status_code,
            error_type=error_type,
            transport=transport,
            server_address=st.server_address,
            server_port=st.server_port,
            input_data=txn.request_body,
            output_data=txn.response_body,
            capture_sources=(self._capture_source(),),
            capture_integrity=integrity,
            correlation=correlation,
            extra=extra,
        )
        self._client.capture_span(span)


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
    limitations: tuple[str, ...],
) -> tuple[
    str,
    StatusCode,
    str | None,
    tuple[tuple[str, str | int | float | bool], ...],
    tuple[str, ...],
]:
    """Assemble gRPC span fields → (name, status_code, error_type, extra, limitations).

    On parse failure, falls back to plain h2 (HTTP) fields + grpc_parse_failed marker.
    """
    try:
        req = parse_grpc_frames(txn.request_body)
        resp = parse_grpc_frames(txn.response_body)
    except Exception:
        http_status = StatusCode.OK if 200 <= txn.status < 400 else StatusCode.ERROR
        return (
            f"HTTP {txn.method} {txn.path}",
            http_status,
            None,
            extra,
            limitations + ("grpc_parse_failed",),
        )

    name = f"gRPC {txn.path}"
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
        limitations = limitations + ("grpc_status_unavailable",)
    if any(m.compressed for m in req.messages) or any(m.compressed for m in resp.messages):
        limitations = limitations + ("grpc_compressed",)
    if req.truncated or resp.truncated:
        limitations = limitations + ("grpc_message_truncated",)

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
