"""Trace continuation and header emission (Phase 4b).

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
from ._w3c import format_traceparent, parse_traceparent


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
        norm = {str(k).lower(): str(v) for k, v in dict(headers).items()}
        raw = norm.get("traceparent")
        if raw:
            parsed = parse_traceparent(raw)
        tracestate = norm.get("tracestate") or None
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


def get_traceparent() -> str | None:
    active = _hub.get_current_scope().active_span_context
    if active is None:
        return None
    return format_traceparent(active)


def get_trace_headers() -> dict[str, str]:
    tp = get_traceparent()
    if tp is None:
        return {}
    headers = {"traceparent": tp}
    ts = _hub.get_merged_scope().tracestate
    if ts:
        headers["tracestate"] = ts
    return headers
