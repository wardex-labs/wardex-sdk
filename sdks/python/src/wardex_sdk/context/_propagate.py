"""Trace continuation and header emission.

continue_trace/get_traceparent are the universal escape hatch: the baton is
just a string, so manual propagation works over any channel that carries
strings — gRPC metadata, WS handshake headers, Celery/Kafka message headers.
Automatic injection (middleware, client patches) is sugar on top of these.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

from .. import _hub
from .._types import SpanContext
from ._w3c import format_traceparent, parse_traceparent, sanitize_tracestate


@contextmanager
def continue_trace(headers: Mapping[str, str]) -> Iterator[None]:
    """Join the distributed trace described by W3C headers.

    Always enters an isolation scope (per-request state isolation). A valid
    traceparent installs a remote parent; a missing/malformed one starts a
    fresh trace (W3C restart rule) — never raises.
    """
    parsed = None
    tracestate: str | None = None
    try:

        def _norm_str(x: object) -> str:
            return x.decode("latin-1") if isinstance(x, (bytes, bytearray)) else str(x)

        norm = {_norm_str(k).lower(): _norm_str(v) for k, v in dict(headers).items()}
        raw = norm.get("traceparent")
        if raw:
            parsed = parse_traceparent(raw)
        tracestate = sanitize_tracestate(norm.get("tracestate"))
    except Exception:
        parsed = None
        tracestate = None
    with _hub.isolation_scope():
        if parsed is not None:
            tid, sid, flags = parsed
            scope = _hub.get_current_scope()
            scope.active_span_context = SpanContext(
                trace_id=tid, span_id=sid, trace_flags=flags, is_remote=True
            )
            scope.tracestate = tracestate
        yield


def _emit_headers() -> dict[str, str]:
    """The W3C headers the ambient context describes. The pair is built here.

    ONE resolution for both public readers, and it is `_hub`'s merged one. The
    two used to disagree: `get_traceparent` read the span context off the
    CURRENT scope while `get_trace_headers` read the tracestate off the MERGED
    one. Nothing writes either field to the global scope today, so the two
    agreed by accident rather than by construction — the first host to seed a
    tracestate on an isolation scope would have got a `tracestate` header on
    requests whose `traceparent` came from somewhere else entirely, and the
    first to seed a span context there would have got a header pair from
    `get_trace_headers` that `get_traceparent` reported as absent.

    Merged, and not current, because merged is the wider of the two: every
    header that is emitted today is still emitted, and the layers
    `get_trace_headers` already honoured are now honoured by both. Narrowing to
    the current scope would have silently stopped forwarding a tracestate some
    host is relying on, and losing propagation data is the failure direction
    this SDK does not take.

    Two fields and not a whole merged Scope: `merged_trace_fields` applies the
    same precedence without `merge_scopes`' per-layer `deepcopy` of `contexts`.
    That copy is why this is not simply `_hub.get_merged_scope()` — it costs
    unboundedly much on the request path, it can RAISE on a host context value
    that does not copy, and neither reader here looks at tags, user, contexts
    or conversation.

    `tracestate` rides only where a `traceparent` goes: the spec gives no
    reading for vendor state without the context it annotates, and a receiver
    that gets one alone either drops it or attributes it to a trace of its own.
    """
    active, tracestate = _hub.get_merged_trace_fields()
    if active is None:
        return {}
    headers = {"traceparent": format_traceparent(active)}
    if tracestate:
        headers["tracestate"] = tracestate
    return headers


def get_traceparent() -> str | None:
    return _emit_headers().get("traceparent")


def get_trace_headers() -> dict[str, str]:
    return _emit_headers()


@contextmanager
def continue_from_otel() -> Iterator[None]:
    """Adopt the current OpenTelemetry span (if any) as a remote parent.

    Entry semantics match continue_trace (isolation + remote parent).
    No opentelemetry installed, or no active/valid span -> plain isolation.
    """
    ctx: SpanContext | None = None
    try:
        from opentelemetry import trace as _otel  # noqa: PLC0415

        sc = _otel.get_current_span().get_span_context()
        if sc.is_valid:
            from .._types import SpanId, TraceId  # noqa: PLC0415

            ctx = SpanContext(
                trace_id=TraceId(sc.trace_id.to_bytes(16, "big")),
                span_id=SpanId(sc.span_id.to_bytes(8, "big")),
                trace_flags=int(sc.trace_flags),
                is_remote=True,
            )
    except Exception:
        ctx = None
    with _hub.isolation_scope():
        if ctx is not None:
            _hub.get_current_scope().active_span_context = ctx
        yield
