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
from ..assembly import Limitation, parent_is_closed_unit
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


def _max_streams(limits: object | None) -> int:
    """The per-connection stream bound, for tables sized by a stream.

    Read off the resolved core limits rather than named again here: the Rust
    parser already bounds its own `streams` map with this number, and the
    correlation latch beside it holds at most one entry per stream that map
    opened. Two names for one quantity is how they drift.
    """
    cap = getattr(limits, "max_streams", None)
    if not isinstance(cap, int) or cap <= 0:
        cap = _wardex_native.limits_defaults()["max_streams"]
    return max(1, int(cap))


def _merge_markers(*groups: tuple[Limitation, ...]) -> tuple[Limitation, ...]:
    """Concatenate limitation markers, keeping first-seen order and dropping
    duplicates. A request and a response that both hit the body cap describe
    one limitation of the transaction, not two."""
    out: list[Limitation] = []
    for group in groups:
        for m in group:
            if m not in out:
                out.append(m)
    return tuple(out)


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
    #: Was `parent` latched off a unit that had ALREADY closed? Latched HERE,
    #: beside the parent and on the task that ISSUED the request, because the
    #: answer is a property of that instant: a request issued while the run was
    #: live is a child of the run's span whether or not the run finishes before
    #: the response arrives, and re-asking on the response side would orphan it.
    #: See `assembly._units.parent_is_closed_unit`.
    parent_closed: bool = False
    #: Was the latched parent DISCARDED by the tracker's own bound before this
    #: transaction arrived to claim it? Only `_Http2Tracker` can answer yes.
    #: Distinct from `parent is None`, which is the ordinary "nothing was
    #: ambient" and an honest trace root; this one says a parent was latched and
    #: wardex threw it away, which is a defect the span has to carry rather than
    #: a fact about the traffic. See `assembly._parentage.resolve_observed`.
    parent_evicted: bool = False
    truncated: bool = False
    # Capture-limitation markers the protocol parser attached to this
    # transaction, merged into the span's CaptureIntegrity.limitations by the
    # seam. Members, not strings: the parser's `&'static str` was resolved once
    # at the PyO3 boundary (`protocol/_http1.py`).
    limitations: tuple[Limitation, ...] = ()
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
    ws_markers: tuple[Limitation, ...] = ()


class _Http1Tracker:
    """HTTP/1.1 — per-direction parser + single-slot latch."""

    def __init__(self, limits: object | None = None) -> None:
        self._req = Http1RequestParser(limits)
        self._resp = Http1ResponseParser(limits)
        self._method: str | None = None
        self._path: str | None = None
        self._req_body: bytes = b""
        self._req_truncated: bool = False
        self._req_limitations: tuple[Limitation, ...] = ()
        self._req_start_ns: int = 0
        self._resp_first_ns: int = 0
        self._parent: SpanContext | None = None
        self._parent_closed: bool = False
        self._resp_cum: int = 0
        self._resp_marks: list[tuple[int, int]] = []
        self._expect_ws: bool = False
        self._resp_raw: bytes = b""

    def on_request_bytes(self, data: bytes) -> list[_Txn]:
        if self._req_start_ns == 0:
            self._req_start_ns = time.time_ns()
            self._parent = _hub.get_current_scope().active_span_context
            # Asked on THIS line, where the request is being issued, so that a
            # context a finished unit left standing is refused before it can
            # become a parent — or open the `capture_mode=AGENT` gate — for
            # traffic that has nothing to do with that run.
            self._parent_closed = parent_is_closed_unit(self._parent)
        for msg in self._req.feed(data):
            self._method = msg.method
            self._path = msg.url
            self._req_body = msg.body
            self._req_truncated = msg.truncated
            self._req_limitations = msg.limitations
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
                        parent_closed=self._parent_closed,
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
                    parent_closed=self._parent_closed,
                    start_ns=self._req_start_ns or now,
                    end_ns=now,
                    ttfb_ms=ttfb,
                    truncated=self._req_truncated or msg.truncated,
                    limitations=_merge_markers(self._req_limitations, msg.limitations),
                    version="1.1",
                    ttft_ms=ttft,
                )
            )
            self._method = None
            self._path = None
            self._req_body = b""
            self._req_truncated = False
            self._req_limitations = ()
            self._req_start_ns = 0
            self._resp_first_ns = 0
            self._parent = None
            self._parent_closed = False
            self._resp_cum = 0
            self._resp_marks = []
            self._expect_ws = False
            self._resp_raw = b""
        return out

    def disabled_reason(self) -> str | None:
        return self._resp.disabled_reason() or self._req.disabled_reason()

    def on_connection_close(self, marker: Limitation) -> list[_Txn]:
        """The connection ended. Nothing here survives it.

        A request whose response never arrived is not a transaction: there is no
        status, no end, and no ttfb, and a span assembled from the half of it
        that exists would assert things the seam never observed. So the accrued
        buffers are released — promptly, rather than whenever the last reference
        to this tracker happens to go — and the caller gets nothing to emit.
        """
        self._req_body = b""
        self._resp_raw = b""
        self._resp_marks = []
        return []


class _Http2Tracker:
    """HTTP/2 — native parser + per-stream_id latch (multiplexing correlation)."""

    def __init__(self, limits: object | None = None) -> None:
        self._conn = Http2Parser(limits)
        # stream_id -> (active span at request time, whether that span's unit
        # had already closed then, request start ns)
        #
        # `_mk` pops on every transaction, so the entries that accumulate are
        # the streams that end WITHOUT one: RST_STREAM, a GOAWAY that strands
        # everything above `last_stream_id`, a server that stops mid-response.
        # There is no per-stream close signal to act on — the parser reports
        # transactions, not stream lifecycles.
        #
        # So there are TWO bounds, because the first one is not reachable
        # everywhere. `on_connection_close` is the honest one and empties this
        # table outright — but it is driven by the close hook, and the async TLS
        # seam's carrier is an `ssl.SSLObject`: no `close()` to patch, and
        # pinned by asyncio's `SSLProtocol` for the life of a pooled connection,
        # so it is neither closed nor collected. That is exactly the h2
        # keep-alive to a model provider this leak was found on.
        #
        # The close hook now reaches that carrier too, through the protocol's
        # `connection_lost` — but only where asyncio's own TLS implementation is
        # the one running (not uvloop's) and only when the pool actually drops
        # the connection, which for a keep-alive to a model provider may be
        # never. A cap that needs no signal at all is what makes the bound
        # unconditional, and that is the FIFO cap below.
        self._latch: dict[int, tuple[SpanContext | None, bool, int]] = {}
        self._latch_cap = _max_streams(limits)
        #: The highest stream id the cap has evicted, and the whole memory of
        #: eviction this tracker keeps. One integer rather than a set of dropped
        #: ids, because a set is the same unbounded table again under a
        #: different name — and it is exact for the policy above: entries are
        #: inserted in increasing id order and dropped lowest-first, so the ids
        #: evicted are precisely the ones latched at or below this mark.
        #:
        #: What it buys is in `_mk`. An evicted entry that no transaction ever
        #: claims cost nothing and is worth saying nothing about; one that a
        #: LATE response then claims would otherwise ship as a clean trace root
        #: at confidence 1.0, which is a span asserting the host issued this
        #: request outside any agent work when wardex simply lost the parent.
        #:
        #: The one imprecision, stated rather than hidden: a transaction for a
        #: stream this tracker never saw opened (capture that began
        #: mid-connection) and whose id falls below the mark is reported as
        #: evicted. Reaching that needs a connection that has already stranded
        #: more than `_latch_cap` streams, and the claim it makes there —
        #: wardex does not know this span's parent and is not calling it a root
        #: — is still the true one.
        self._latch_evicted_below = 0

    def on_request_bytes(self, data: bytes) -> list[_Txn]:
        opened, txns = self._conn.feed(True, data)
        now = time.time_ns()
        parent = _hub.get_current_scope().active_span_context
        # Asked ONCE per feed rather than once per stream: every stream this
        # write opened was issued from this carrier at this instant.
        parent_closed = parent_is_closed_unit(parent) if opened else False
        for sid in opened:
            self._latch[sid] = (parent, parent_closed, now)
        # Drop-oldest, which for h2 is drop-lowest-stream-id: ids only ever
        # increase, so the entry evicted is the one likeliest to be stranded
        # already. Losing it costs that stream its parentage, never a span —
        # `_mk` falls back to (None, False, now) and says so on the span.
        while len(self._latch) > self._latch_cap:
            evicted = next(iter(self._latch))
            self._latch.pop(evicted)
            self._latch_evicted_below = max(self._latch_evicted_below, evicted)
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

    def on_connection_close(self, marker: Limitation) -> list[_Txn]:
        """Release the per-stream latch — the eviction the entries were waiting for.

        A latch entry is one `SpanContext` plus two scalars, so this is small
        money per connection and, without the cap in `__init__`, unbounded money
        over a process: an h2 client that resets a stream per cancelled request
        accumulates one entry per cancellation for the life of the connection,
        and a keep-alive h2 connection to a model provider lives as long as the
        process does. This is the release that costs nothing, where it is
        reachable; the cap bounds the paths where it is not.

        No transactions come back. A stream that never produced a response
        produced no status either, and the seam has nothing to say about it that
        would not be invented.
        """
        self._latch.clear()
        return []

    def _mk(self, t: Any) -> _Txn:
        now = time.time_ns()
        entry = self._latch.pop(t.stream_id, None)
        if entry is None:
            parent, parent_closed, start = None, False, now
            # The DECISION the cap owes the span. An absent latch entry has two
            # causes that look identical here and mean opposite things: nothing
            # was ambient when the request went out (an honest trace root, and
            # under `capture_mode=AGENT` the gate has usually dropped it long
            # before this line), or a parent WAS latched and the cap discarded
            # it. Shipping the second as the first is the one degradation a
            # consumer cannot detect downstream — same edge, same confidence,
            # no marker — so the bound reports itself, exactly as the unit
            # registry's does when it closes a root at `max_units`.
            #
            # A separate `Limitation` member was the alternative and is refused:
            # the vocabulary is closed on the wire, and what a user would read
            # off a new one — "wardex dropped what belongs on this span" —
            # `PARENT_UNRESOLVED` plus `INSTRUMENTATION_DEGRADED` already say,
            # from the site that owns the edge. `resolve_observed` attaches
            # them; this only reports the fact.
            #
            # It reports the EVICTION and not "a parent was lost", because the
            # two are not separable from here: what the entry held went with it.
            # That is also why the claim is never an over-reach on a stream that
            # had no parent to lose — every entry carries the REQUEST START
            # INSTANT as well, so `start` below is a fabrication on this path
            # regardless, the span's duration is near-zero and its start is the
            # response instant. Something that belongs on this span is missing
            # in every case, which is the whole content of the marker; a second
            # marker for the clock half would split one fact across two words.
            parent_evicted = t.stream_id <= self._latch_evicted_below
        else:
            parent, parent_closed, start = entry
            parent_evicted = False
        return _Txn(
            method=t.method or "?",
            path=t.path or "/",
            status=t.status,
            request_body=t.request_body,
            response_body=t.response_body,
            parent=parent,
            parent_closed=parent_closed,
            parent_evicted=parent_evicted,
            start_ns=start,
            end_ns=now,
            ttfb_ms=0.0,  # per-h2-stream first-byte not tracked (limitation)
            truncated=t.truncated,
            version="2",
            ttft_ms=0.0,  # per-h2-stream first-body-byte not tracked (limitation)
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
        parent_closed: bool = False,
        limits: object | None = None,
        sample_cap: int | None = None,
    ) -> None:
        self._sent = WsParser(limits)  # client -> server
        self._recv = WsParser(limits)  # server -> client
        self._path = path
        self._deflate = deflate
        self._parent = parent
        # Inherited from the UPGRADE transaction rather than re-latched: a WS
        # session's parent is the scope that issued the handshake, and so is the
        # question of whether that scope's unit had already died.
        self._parent_closed = parent_closed
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

    def flush(self, marker: Limitation) -> list[_Txn]:
        if self._emitted:
            return []
        return [self._build_txn((marker,))]

    #: The connection-close verb every tracker answers to. For a WS session it
    #: IS `flush`, and an ALIAS rather than a delegating wrapper: this is the one
    #: tracker with something to save at close — its span exists only once the
    #: session ends — so the two names must never be able to drift apart.
    #: Before the close hook, that span waited for `uninstall()` and was lost
    #: whenever the process never reached one.
    on_connection_close = flush

    def _build_txn(self, extra_markers: tuple[Limitation, ...]) -> _Txn:
        self._emitted = True
        markers = list(extra_markers)
        if self._in_trunc or self._out_trunc:
            markers.append(Limitation.WS_PAYLOAD_TRUNCATED)
        if self._deflate:
            # Census merge (§6.5.1): `ws_compressed` folded into
            # PAYLOAD_COMPRESSED. What is lost is which protocol it was, and
            # `TransportAttributes.protocol` already carries that.
            markers.append(Limitation.PAYLOAD_COMPRESSED)
        if self._sent.is_disabled() or self._recv.is_disabled():
            # Census merge: `ws_parse_failed` and `grpc_parse_failed` are one
            # fact — the framing layer failed, so the transport fields on this
            # span are partial or synthesized.
            markers.append(Limitation.FRAME_PARSE_FAILED)
        now = time.time_ns()
        return _Txn(
            method="GET",
            path=self._path,
            status=101,
            request_body=bytes(self._sample_in),
            response_body=bytes(self._sample_out),
            parent=self._parent,
            parent_closed=self._parent_closed,
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
