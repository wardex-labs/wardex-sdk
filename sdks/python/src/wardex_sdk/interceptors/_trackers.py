"""Protocol-neutral transaction + protocol-specific trackers.

When the SSL interceptor (_ssl.py) streams plaintext bytes into a tracker, the tracker
parses and correlates them according to the protocol (HTTP/1 or HTTP/2) and returns a
list of _Txn. Span assembly is performed by _ssl.py based solely on _Txn (protocol-agnostic).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .. import _hub, _wardex_native
from .._types import SpanContext
from ..protocol import WsParser
from ..protocol._http1 import Http1RequestParser, Http1ResponseParser
from ..protocol._http2 import Http2Parser


def _ttft_from_marks(marks: list[tuple[int, int]], header_len: int, start_ns: int) -> float:
    """marks=[(cumulative_wire_bytes, ns)]. TTFT (ms) is computed from the ns of the
    first mark that crosses the header_len boundary. Returns 0.0 if no mark crosses the
    boundary, or if the request start time is unknown (mid-connection capture)."""
    for cum, ns in marks:
        if cum > header_len:
            return max(0.0, (ns - start_ns) / 1e6) if start_ns else 0.0
    return 0.0


def _header_get(headers: object, name: str) -> str | None:
    """Looks up the value for name (case-insensitive) in ParsedMessage.headers
    (a sequence of tuples)."""
    target = name.lower()
    for h in headers:  # type: ignore[union-attr]
        if h[0].lower() == target:
            return h[1]
    return None


def _is_ws_upgrade_request(headers: object) -> bool:
    up = _header_get(headers, "upgrade")
    conn = _header_get(headers, "connection")
    return (
        up is not None
        and "websocket" in up.lower()
        and conn is not None
        and "upgrade" in conn.lower()
    )


@dataclass
class _Txn:
    """A single protocol-neutral transaction (request + response)."""

    method: str
    path: str
    status: int
    request_body: bytes
    response_body: bytes
    parent: SpanContext | None
    start_ns: int
    end_ns: int
    ttfb_ms: float
    truncated: bool = False
    version: str = "1.1"
    ttft_ms: float = 0.0
    content_type: str | None = None
    grpc_status: int | None = None
    grpc_message: str | None = None
    # WS upgrade signal (set on 101 detection — used by _ssl.py as the SWAP trigger)
    ws_upgrade: bool = False
    ws_upgrade_path: str | None = None
    ws_deflate: bool = False
    ws_leftover: bytes = b""
    # WS session span data (set on close/flush — filled in by _WebSocketTracker,
    # version=="websocket")
    ws_close_code: int | None = None
    ws_messages_sent: int = 0
    ws_messages_received: int = 0
    ws_bytes_sent: int = 0
    ws_bytes_received: int = 0
    ws_markers: tuple[str, ...] = ()


class _Http1Tracker:
    """HTTP/1.1 — per-direction parser + single-slot latch (unchanged from Slice 1 behavior)."""

    def __init__(self, limits: object | None = None) -> None:
        self._req = Http1RequestParser(limits)
        self._resp = Http1ResponseParser(limits)
        self._method: str | None = None
        self._path: str | None = None
        self._req_body: bytes = b""
        self._req_start_ns: int = 0
        self._resp_first_ns: int = 0
        self._parent: SpanContext | None = None
        self._resp_cum: int = 0
        self._resp_marks: list[tuple[int, int]] = []
        self._expect_ws: bool = False
        self._resp_raw: bytes = b""

    def on_request_bytes(self, data: bytes) -> list[_Txn]:
        if self._req_start_ns == 0:
            self._req_start_ns = time.time_ns()
            self._parent = _hub.get_current_scope().active_span_context
        for msg in self._req.feed(data):
            self._method = msg.method
            self._path = msg.url
            self._req_body = msg.body
            if _is_ws_upgrade_request(msg.headers):
                self._expect_ws = True
        return []

    def on_response_bytes(self, data: bytes) -> list[_Txn]:
        now = time.time_ns()
        if self._resp_first_ns == 0:
            self._resp_first_ns = now
        self._resp_cum += len(data)
        self._resp_marks.append((self._resp_cum, now))
        if self._expect_ws:
            self._resp_raw += data
        out: list[_Txn] = []
        for msg in self._resp.feed(data):
            # --- WS upgrade branch ---
            if self._expect_ws and msg.status_code == 101 and _is_ws_upgrade_request(msg.headers):
                ext = _header_get(msg.headers, "sec-websocket-extensions") or ""
                leftover = self._resp_raw[msg.header_len :]
                out.append(
                    _Txn(
                        method=self._method or "GET",
                        path=self._path or "/",
                        status=101,
                        request_body=b"",
                        response_body=b"",
                        parent=self._parent,
                        start_ns=self._req_start_ns or now,
                        end_ns=now,
                        ttfb_ms=0.0,
                        version="1.1",
                        ws_upgrade=True,
                        ws_upgrade_path=self._path or "/",
                        ws_deflate="permessage-deflate" in ext.lower(),
                        ws_leftover=leftover,
                    )
                )
                # After the upgrade, this tracker retires (_ssl.py SWAPs it for a
                # _WebSocketTracker). The remaining state (_resp_marks, etc.) is never
                # used again, so no reset is needed.
                self._expect_ws = False
                self._resp_raw = b""
                continue
            # Interim 1xx responses (100 Continue/103 Early Hints, etc.) are not final responses.
            # Skip them without emitting a span or resetting request state (wait for the
            # real final response that follows).
            # (101 upgrade is already handled in the branch above.)
            if msg.status_code is not None and 100 <= msg.status_code < 200:
                continue
            # --- Regular HTTP response (existing behavior) ---
            now = time.time_ns()
            ttfb = (
                max(0.0, (self._resp_first_ns - self._req_start_ns) / 1e6)
                if self._req_start_ns and self._resp_first_ns
                else 0.0
            )
            ttft = _ttft_from_marks(self._resp_marks, msg.header_len, self._req_start_ns)
            out.append(
                _Txn(
                    method=self._method or "?",
                    path=self._path or "/",
                    status=msg.status_code or 0,
                    request_body=self._req_body,
                    response_body=msg.body,
                    parent=self._parent,
                    start_ns=self._req_start_ns or now,
                    end_ns=now,
                    ttfb_ms=ttfb,
                    version="1.1",
                    ttft_ms=ttft,
                )
            )
            self._method = None
            self._path = None
            self._req_body = b""
            self._req_start_ns = 0
            self._resp_first_ns = 0
            self._parent = None
            self._resp_cum = 0
            self._resp_marks = []
            self._expect_ws = False
            self._resp_raw = b""
        return out

    def disabled_reason(self) -> str | None:
        return self._resp.disabled_reason() or self._req.disabled_reason()


class _Http2Tracker:
    """HTTP/2 — native parser + per-stream_id latch (multiplexing correlation)."""

    def __init__(self, limits: object | None = None) -> None:
        self._conn = Http2Parser(limits)
        # stream_id -> (active span at request time, request start ns)
        # TODO: evict stale entries for streams that closed without a response (Phase 3 close hook)
        self._latch: dict[int, tuple[SpanContext | None, int]] = {}

    def on_request_bytes(self, data: bytes) -> list[_Txn]:
        opened, txns = self._conn.feed(True, data)
        now = time.time_ns()
        for sid in opened:
            self._latch[sid] = (_hub.get_current_scope().active_span_context, now)
        # Always call _mk to pop the _latch entry (prevents leaks); status==0
        # (degenerate transaction) is excluded from the result
        out: list[_Txn] = []
        for t in txns:
            txn = self._mk(t)
            if t.status:
                out.append(txn)
        return out

    def on_response_bytes(self, data: bytes) -> list[_Txn]:
        # server push (PUSH_PROMISE) unsupported — response-side stream_id ignored
        _opened, txns = self._conn.feed(False, data)
        # Always call _mk to pop the _latch entry (prevents leaks); status==0
        # (degenerate transaction) is excluded from the result
        out: list[_Txn] = []
        for t in txns:
            txn = self._mk(t)
            if t.status:
                out.append(txn)
        return out

    def _mk(self, t: Any) -> _Txn:
        now = time.time_ns()
        parent, start = self._latch.pop(t.stream_id, (None, now))
        return _Txn(
            method=t.method or "?",
            path=t.path or "/",
            status=t.status,
            request_body=t.request_body,
            response_body=t.response_body,
            parent=parent,
            start_ns=start,
            end_ns=now,
            ttfb_ms=0.0,  # per-h2-stream first-byte not tracked (limitation, Phase 4)
            truncated=t.truncated,
            version="2",
            ttft_ms=0.0,  # per-h2-stream first-body-byte not tracked (limitation, Phase 4)
            content_type=getattr(t, "content_type", None),
            grpc_status=getattr(t, "grpc_status", None),
            grpc_message=getattr(t, "grpc_message", None),
        )


class _WebSocketTracker:
    """One WS connection — per-direction frame parser + aggregation + 64KB content sample.
    Emits 1 span on close/flush."""

    def __init__(
        self,
        path: str,
        deflate: bool,
        parent: SpanContext | None,
        start_ns: int,
        limits: object | None = None,
        sample_cap: int | None = None,
    ) -> None:
        self._sent = WsParser(limits)  # client -> server
        self._recv = WsParser(limits)  # server -> client
        self._path = path
        self._deflate = deflate
        self._parent = parent
        self._start_ns = start_ns
        # None means "use the core default" — resolved here (rather than hardcoded)
        # so this can never silently drift from crates/wardex-limits.
        self._sample_cap = (
            sample_cap
            if sample_cap is not None
            else _wardex_native.limits_defaults()["ws_sample_bytes"]
        )
        self._sent_msgs = 0
        self._recv_msgs = 0
        self._sent_bytes = 0
        self._recv_bytes = 0
        self._sample_in = bytearray()
        self._sample_out = bytearray()
        self._in_trunc = False
        self._out_trunc = False
        self._close_code: int | None = None
        self._closed = False
        self._emitted = False

    def on_request_bytes(self, data: bytes) -> list[_Txn]:
        r = self._sent.feed(data)
        for f in r.frames:
            self._sent_bytes += f.payload_len
            if f.close_code is not None:
                self._close_code = f.close_code
                self._closed = True
            if f.opcode == "close":
                self._closed = True
        self._sent_msgs += len(r.messages)
        self._in_trunc = self._append_sample(self._sample_in, r.messages) or self._in_trunc
        return self._maybe_emit()

    def on_response_bytes(self, data: bytes) -> list[_Txn]:
        r = self._recv.feed(data)
        for f in r.frames:
            self._recv_bytes += f.payload_len
            if f.close_code is not None:
                self._close_code = f.close_code
                self._closed = True
            if f.opcode == "close":
                self._closed = True
        self._recv_msgs += len(r.messages)
        self._out_trunc = self._append_sample(self._sample_out, r.messages) or self._out_trunc
        return self._maybe_emit()

    def _append_sample(self, buf: bytearray, messages: list[bytes]) -> bool:
        """Accumulate messages into buf (up to a total of self._sample_cap).
        Returns True if truncated."""
        truncated = False
        for m in messages:
            room = self._sample_cap - len(buf)
            if room <= 0:
                truncated = True
                break
            if len(m) > room:
                buf += m[:room]
                truncated = True
            else:
                buf += m
        return truncated

    def _maybe_emit(self) -> list[_Txn]:
        if self._closed and not self._emitted:
            return [self._build_txn(())]
        return []

    def flush(self, marker: str) -> list[_Txn]:
        if self._emitted:
            return []
        return [self._build_txn((marker,))]

    def _build_txn(self, extra_markers: tuple[str, ...]) -> _Txn:
        self._emitted = True
        markers = list(extra_markers)
        if self._in_trunc or self._out_trunc:
            markers.append("ws_payload_truncated")
        if self._deflate:
            markers.append("ws_compressed")
        if self._sent.is_disabled() or self._recv.is_disabled():
            markers.append("ws_parse_failed")
        now = time.time_ns()
        return _Txn(
            method="GET",
            path=self._path,
            status=101,
            request_body=bytes(self._sample_in),
            response_body=bytes(self._sample_out),
            parent=self._parent,
            start_ns=self._start_ns,
            end_ns=now,
            ttfb_ms=0.0,
            version="websocket",
            ws_close_code=self._close_code,
            ws_messages_sent=self._sent_msgs,
            ws_messages_received=self._recv_msgs,
            ws_bytes_sent=self._sent_bytes,
            ws_bytes_received=self._recv_bytes,
            ws_markers=tuple(markers),
        )
