"""Protocol-specific trackers.

When the SSL interceptor (_ssl.py) streams plaintext bytes into a tracker, the tracker
parses and correlates them according to the protocol (HTTP/1 or HTTP/2) and returns a
list of _Txn (`_txn.py`). Span assembly is performed based solely on _Txn (protocol-agnostic).
"""

from __future__ import annotations

import re
import time
from typing import Any

from .. import _hub, _wardex_native
from .._assembly import Limitation, counters, parent_is_closed_unit, should_capture
from .._enums import CaptureMode
from .._protocol import WsParser
from .._protocol._http1 import Http1ResponseParser, declares_event_stream
from .._protocol._http2 import Http2Parser
from .._types import ConversationContext, ParsedMessage, SpanContext
from ._h2_issuer import IssuerLink
from ._http1_requests import RequestSide
from ._issue_scope import UNKNOWN_ISSUER, ScopeSnapshot, issued_scope
from ._txn import _name_path, _Txn


def _ttft_from_marks(marks: list[tuple[int, int]], header_len: int, start_ns: int) -> float | None:
    """marks=[(cumulative_wire_bytes, ns)]. TTFT (ms) is computed from the ns of the
    first mark that crosses the header_len boundary. None — not measured, which
    ships as an unset field rather than as a 0 ms reading — if no mark crosses the
    boundary (no body byte arrived), or if the request start time is unknown
    (mid-connection capture)."""
    for cum, ns in marks:
        if cum > header_len:
            return max(0.0, (ns - start_ns) / 1e6) if start_ns else None
    return None


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


def _admits(parent: SpanContext | None, parent_closed: bool) -> bool:
    """Would this latched parent by itself admit a transaction past the capture gate? Asked of the
    policy (`should_capture` under AGENT, the one mode in which a parent decides anything) rather
    than restated here, so the latch and the gate cannot disagree about what a parent is worth."""
    return should_capture(
        CaptureMode.AGENT, parent=parent, agent_semantic=False, parent_closed=parent_closed
    )


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


#: The `type` of the one client message the Responses WebSocket transport sends per call, searched
#: for anywhere in the first client message rather than matched as a prefix: key order
#: (`sort_keys`), a BOM or leading whitespace must not decide the connection's fate. The search is
#: bounded by the message, which the frame parser already caps. Consulted only when nothing hides
#: the payload.
_RESPONSES_CREATE = re.compile(rb'"type"\s*:\s*"response\.create"')
_StreamLatch = tuple[SpanContext | None, bool, ConversationContext | None, ScopeSnapshot, bool, int]


class _Http1Tracker:
    """HTTP/1.1 — a response parser, and a request side (`_http1_requests`) that says which
    request each final reply answers, including when the seam did not see every request byte."""

    def __init__(self, limits: object | None = None) -> None:
        self._requests = RequestSide(limits)
        self._resp = Http1ResponseParser(limits)
        self._resp_first_ns: int = 0
        self._resp_cum: int = 0
        self._resp_marks: list[tuple[int, int]] = []
        self._expect_ws: bool = False
        self._resp_raw: bytes = b""

    def _upgrade_requested(self) -> bool:
        request = self._requests.request
        return request is not None and _is_ws_upgrade_request(request.headers)

    def on_request_bytes(self, data: bytes) -> list[_Txn]:
        self._requests.feed(data)
        self._expect_ws = self._upgrade_requested()
        return []

    def on_response_bytes(self, data: bytes) -> list[_Txn]:
        if self._requests.decide():
            self._expect_ws = self._upgrade_requested()
        if self._resp.idle():
            # These bytes open a response, so its request is settled: HEAD and CONNECT frame their
            # answers, and only the request side knows which one this reply answers.
            self._resp.expect_response_to(self._requests.method)
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
                request, issue = self._requests.request, self._requests.issue
                path = (request.url if request is not None else None) or "/"
                out.append(
                    _Txn(
                        method=(request.method if request is not None else None) or "GET",
                        path=_name_path(path),
                        target=path,
                        status=101,
                        request_body=b"",
                        response_body=b"",
                        parent=issue.parent,
                        parent_closed=issue.parent_closed,
                        conversation=issue.conversation,
                        scope=issue.scope,
                        start_ns=issue.start_ns or now,
                        end_ns=now,
                        ttfb_ms=None,
                        version="1.1",
                        ws_upgrade=True,
                        ws_upgrade_path=path,
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
                # Re-base the byte marks on the final response's first byte, or
                # its TTFT would land on the arrival of its own header block.
                self._resp_marks = [(c - msg.header_len, ns) for c, ns in self._resp_marks]
                self._resp_cum -= msg.header_len
                continue
            if self._requests.method == "CONNECT" and 200 <= (msg.status_code or 0) < 300:
                # A tunnel opened; the calls inside are the TLS seam's. Its opening ships no span: a
                # span URL cannot yet carry an authority-form target (`host:443`).
                self._requests.take()
                self._reset()
                continue
            # --- Regular HTTP response (existing behavior) ---
            txn = self._response_txn(msg, time.time_ns())
            if txn is not None:
                out.append(txn)
            self._reset()
        return out

    def on_response_eof(self) -> list[_Txn]:
        """The peer closed its side (a read asked for bytes, got none): that ENDS a body with no
        framing, so it ships whole, now; one whose framing promised more ships marked."""
        msg = self._resp.flush(peer_closed=True)
        if msg is None:
            return []
        txn = self._response_txn(msg, time.time_ns())
        self._reset()
        return [txn] if txn is not None else []

    def _reset(self) -> None:
        """Forget the finished response. The request side already moved on: `take` readied it for
        the next request when it paired this one."""
        self._resp_first_ns = 0
        self._resp_cum = 0
        self._resp_marks = []
        self._expect_ws = False
        self._resp_raw = b""

    def _response_txn(self, msg: ParsedMessage, now: int) -> _Txn | None:
        """The transaction a final response completes at `now`, paired with the request it answers
        (`RequestSide.take`): None when not one byte of it was seen, `? /` when only part was (both
        counted). An `incomplete` response (the body is partial) is marked `FRAME_PARSE_FAILED` and
        `response_cut`: its 2xx is not a success."""
        taken = self._requests.take()
        if taken is None:
            return None
        request, issue = taken
        ttfb = (
            max(0.0, (self._resp_first_ns - issue.start_ns) / 1e6)
            if issue.start_ns and self._resp_first_ns
            else None
        )
        ttft = _ttft_from_marks(self._resp_marks, msg.header_len, issue.start_ns)
        path = (request.url if request is not None else None) or "/"
        cut = (Limitation.FRAME_PARSE_FAILED,) if msg.incomplete else ()
        return _Txn(
            method=(request.method if request is not None else None) or "?",
            path=_name_path(path),
            target=path,
            status=msg.status_code or 0,
            request_body=request.body if request is not None else b"",
            response_body=msg.body,
            parent=issue.parent,
            parent_closed=issue.parent_closed,
            conversation=issue.conversation,
            scope=issue.scope,
            start_ns=issue.start_ns or now,
            end_ns=now,
            ttfb_ms=ttfb,
            truncated=(request is not None and request.truncated) or msg.truncated,
            request_counted=request is not None and not request.truncated,
            response_counted=not msg.truncated,
            response_cut=msg.incomplete,
            limitations=_merge_markers(
                request.limitations if request is not None else (), msg.limitations, cut
            ),
            version="1.1",
            ttft_ms=ttft,
            event_stream=declares_event_stream(_header_get(msg.headers, "content-type")),
            content_encoding=_header_get(msg.headers, "content-encoding"),
        )

    def disabled_reason(self) -> str | None:
        return self._resp.disabled_reason() or self._requests.disabled_reason()

    def on_connection_close(self, marker: Limitation, *, still_open: bool = False) -> list[_Txn]:
        """Closed: a response in flight whose headers arrived ships (`_response_txn`), as what
        arrived and marked — its end (the peer's EOF, `on_response_eof`) was not seen. No headers,
        no transaction: nothing was observed. `marker` is the seam's reason, not this span's.
        `still_open` (FIFO cap, `uninstall()`): nothing ended, nothing ships. Buffers go anyway."""
        out: list[_Txn] = []
        if not still_open:
            msg = self._resp.flush(peer_closed=False)
            if msg is not None and (txn := self._response_txn(msg, time.time_ns())) is not None:
                out.append(txn)
        self._requests.release()
        self._resp_raw = b""
        self._resp_marks = []
        return out


class _Http2Tracker:
    """HTTP/2 — native parser + per-stream_id latch (multiplexing correlation)."""

    def __init__(self, limits: object | None = None) -> None:
        self._conn = Http2Parser(limits)
        # stream_id -> (parent span, whether its unit had already closed, the conversation and the
        # scope identity — the issuer's, where `_h2_issuer` proved it — whether it was proved,
        # request start ns)
        #
        # `_mk` pops on every transaction, so the entries that accumulate are the streams that end
        # WITHOUT one: RST_STREAM, a GOAWAY that strands everything above `last_stream_id`, a server
        # that stops mid-response. There is no per-stream close signal to act on — the parser
        # reports transactions, not stream lifecycles.
        #
        # So there are TWO bounds, because the first one is not reachable everywhere.
        # `on_connection_close` is the honest one and empties this table outright — but it is driven
        # by the close hook, and the async TLS seam's carrier is an `ssl.SSLObject`: no `close()` to
        # patch, and pinned by asyncio's `SSLProtocol` for the life of a pooled connection, so it is
        # neither closed nor collected. That is exactly the h2 keep-alive to a model provider this
        # leak was found on.
        #
        # The close hook now reaches that carrier too, through the protocol's `connection_lost` —
        # but only where asyncio's own TLS implementation is the one running (not uvloop's) and only
        # when the pool actually drops the connection, which for a keep-alive to a model provider
        # may be never. A cap that needs no signal at all is what makes the bound unconditional, and
        # that is the FIFO cap below.
        self._latch: dict[int, _StreamLatch] = {}
        self._latch_cap = _max_streams(limits)
        #: The highest stream id the cap has evicted, and the whole memory of eviction this tracker
        #: keeps. One integer rather than a set of dropped ids, because a set is the same unbounded
        #: table again under a different name — and it is exact for the policy above: entries are
        #: dropped lowest id first (not insertion order: a stream whose request body ends late is
        #: latched after higher ids), so the ids evicted are precisely the ones latched at or below
        #: this mark.
        #:
        #: What it buys is in `_mk`. An evicted entry that no transaction ever claims cost nothing
        #: and is worth saying nothing about; one that a LATE response then claims would otherwise
        #: ship as a clean trace root at confidence 1.0, which is a span asserting the host issued
        #: this request outside any agent work when wardex simply lost the parent.
        #:
        #: Reset by `on_connection_close` along with the latch itself: stream ids restart at 1 on a
        #: new connection, so a mark carried across one would name a different set of streams than
        #: the ones it was taken on.
        self._latch_evicted_below = 0
        #: The LOWEST stream id this tracker ever latched, and the floor that keeps the mark above
        #: from over-claiming. Capture can attach mid-connection: a response for a stream opened
        #: before the seam was watching has no latch entry either, and its id is strictly below
        #: anything this tracker put in the table. Without the floor such a stream reads as evicted
        #: once the cap has run — a span blaming wardex for a parent wardex was never in a position
        #: to hold. (It cannot open the capture gate: only an id `_evicted_admitting` recorded at
        #: the eviction can.) Zero means "nothing latched yet", which fails the test for every real
        #: stream id.
        self._latch_first = 0
        #: The evicted ids whose entry held a parent that would have admitted its transaction
        #: (`_admits`): the half of the mark above that `_mk` may hand the capture gate. Bounded by
        #: the same cap and dropped lowest id first, so it is not the unbounded table the mark
        #: avoids; an id it forgets reads as a parentless eviction, the gate staying shut.
        self._evicted_admitting: set[int] = set()
        #: Who opened each stream, read in the opening call itself (`_h2_issuer`).
        self._issuers = IssuerLink(self._latch_cap)

    def on_request_bytes(self, data: bytes) -> list[_Txn]:
        self._issuers.see(data)
        opened, txns = self._conn.feed(True, data)
        now = time.time_ns()
        # The WRITER's scope, which on a shared connection may be any task's: any
        # of them flushes the others' queued frames. So it is the parent only of
        # a stream with no proven issuer, and never anyone's conversation or
        # identity (`IssuerLink.latch`).
        parent = _hub.get_current_scope().active_span_context
        parent_closed = parent_is_closed_unit(parent) if opened else False
        for sid in opened:
            self._latch[sid] = (*self._issuers.latch(sid, parent, parent_closed), now)
        if opened:
            low = min(opened)
            self._latch_first = low if self._latch_first == 0 else min(self._latch_first, low)
        # Drop-lowest-stream-id: ids only ever increase, so the entry evicted
        # is the one likeliest to be stranded already. Lowest by id, not by
        # insertion — a stream announced late (its request body ended after
        # higher ids') is still the oldest request. Losing it costs that stream
        # its parentage, never a span — `_mk` falls back to a parentless entry
        # and says so on the span. `min` is a scan of at most `max_streams + 1`
        # keys, run only on the writes that overflow the cap.
        while len(self._latch) > self._latch_cap:
            evicted = min(self._latch)
            if _admits(*self._latch.pop(evicted)[:2]):
                self._evicted_admitting.add(evicted)
                if len(self._evicted_admitting) > self._latch_cap:
                    self._evicted_admitting.discard(min(self._evicted_admitting))
            self._latch_evicted_below = max(self._latch_evicted_below, evicted)
        return self._ship(txns)

    def on_response_bytes(self, data: bytes) -> list[_Txn]:
        # server push (PUSH_PROMISE) unsupported — response-side stream_id ignored
        _opened, txns = self._conn.feed(False, data)
        return self._ship(txns)

    def _ship(self, txns: list[Any]) -> list[_Txn]:
        """`_mk` every native transaction — it pops the latch entry, so skipping one would leak it —
        and keep the ones that describe an exchange.

        Two kinds are left out. status==0 is a degenerate transaction. A transaction with no method
        that the stream table did NOT evict is a response whose request this parser never observed:
        the host opened the stream before capture attached. Shipping it meant a `? /` span with no
        marker, indistinguishable from a request with no method and no path, and with a start
        instant invented at the response. It is counted instead. `? /` ships only with
        `H2_REQUEST_EVICTED`, which names the bound that took the request.
        """
        out: list[_Txn] = []
        for t in txns:
            txn = self._mk(t)
            if not t.status:
                continue
            if not t.method and not t.request_evicted:
                counters.bump("protocol.http2.request_unobserved")
                continue
            out.append(txn)
        return out

    def disabled_reason(self) -> str | None:
        """The native parser's latch reason. One connection, both directions,
        one parser — so unlike HTTP/1 there is a single reason to ask for."""
        return self._conn.disabled_reason()

    def on_connection_close(self, marker: Limitation, *, still_open: bool = False) -> list[_Txn]:
        """Release the per-stream latch — the eviction the entries were waiting for.

        A latch entry is two references and two scalars, so this is small
        money per connection and, without the cap in `__init__`, unbounded money
        over a process: an h2 client that resets a stream per cancelled request
        accumulates one entry per cancellation for the life of the connection,
        and a keep-alive h2 connection to a model provider lives as long as the
        process does. This is the release that costs nothing, where it is
        reachable; the cap bounds the paths where it is not.

        No transactions come back. A stream that never produced a response
        produced no status either, and the seam has nothing to say about it that
        would not be invented.

        The eviction bookkeeping goes with the entries, and it has to: h2 stream
        ids restart at 1 on the next connection, so a mark or a floor taken on
        the last one names a different set of streams here. No caller reuses a
        tracker across a close today — every `_retire` in `_seam.py` discards
        the `_ConnectionState` and the tracker inside it — but nothing declares
        that, and the cost of a stale mark is every unlatched stream on the new
        connection reporting a parent wardex never lost.
        """
        self._latch.clear()
        self._issuers.clear()
        self._evicted_admitting.clear()
        self._latch_evicted_below = 0
        self._latch_first = 0
        return []

    def _mk(self, t: Any) -> _Txn:
        now = time.time_ns()
        entry = self._latch.pop(t.stream_id, None)
        if entry is None:
            parent, parent_closed, conversation, proven, start = None, False, None, False, now
            scope = UNKNOWN_ISSUER  # whoever issued it, nothing here says who
            # The DECISION the cap owes the span. An absent latch entry has two causes that look
            # identical here and mean opposite things: nothing was ambient when the request went out
            # (an honest trace root, and under `capture_mode=AGENT` the gate has usually dropped it
            # long before this line), or a parent WAS latched and the cap discarded it. Shipping the
            # second as the first is the one degradation a consumer cannot detect downstream — same
            # edge, same confidence, no marker — so the bound reports itself, exactly as the unit
            # registry's does when it closes a root at `max_units`.
            #
            # A separate `Limitation` member was the alternative and is refused: the vocabulary is
            # closed on the wire, and what a user would read off a new one — "wardex dropped what
            # belongs on this span" — `PARENT_UNRESOLVED` plus `INSTRUMENTATION_DEGRADED` already
            # say, from the site that owns the edge. `resolve_observed` attaches them; this only
            # reports the fact.
            #
            # The MARKER reports the EVICTION and not "a parent was lost", and is no over-reach on a
            # stream that had no parent to lose: every entry carries the REQUEST START INSTANT as
            # well, so `start` below is a fabrication on this path regardless, the span's duration
            # is near-zero and its start is the response instant. Something that belongs on this
            # span is missing in every case, which is the whole content of the marker; a second
            # marker for the clock half would split one fact across two words.
            #
            # The capture GATE is told less (`parent_lost`): only that the entry held a parent that
            # would by itself have admitted the span, which the eviction recorded before the entry
            # went. An eviction that lost no parent, or one the gate refuses, lost nothing the gate
            # acts on; calling it degraded exported request and response bodies under a mode that
            # had filtered the span out.
            #
            # The marker is bounded at BOTH ends, and the floor is not decoration: the mark alone
            # would also claim a stream opened before capture attached, whose id is below everything
            # this tracker latched, and blame wardex for a parent it never held.
            parent_evicted = self._latch_first <= t.stream_id <= self._latch_evicted_below
            parent_lost = t.stream_id in self._evicted_admitting
            self._evicted_admitting.discard(t.stream_id)
        else:
            parent, parent_closed, conversation, scope, proven, start = entry
            parent_evicted = parent_lost = False
        # The OTHER half of the same bound: the native stream table evicted this stream's request
        # before its response completed. The response is a real observation — a status, an end — so
        # the span ships, but its `? /` is a display fallback for a request wardex lost, and it may
        # only ever appear with the marker that says so. Counted here, before the seam's status and
        # capture-mode filters, so the loss is visible even when no span survives them.
        #
        # A plain attribute read: a default here is the shape that would make every marker vanish
        # silently if the native field were ever renamed.
        request_evicted = bool(t.request_evicted)
        if request_evicted:
            counters.bump("protocol.http2.stream_evicted")
        return _Txn(
            method=t.method or "?",
            path=_name_path(t.path or "/"),
            target=t.path or "/",
            status=t.status,
            request_body=t.request_body,
            response_body=t.response_body,
            parent=parent,
            parent_closed=parent_closed,
            parent_evicted=parent_evicted,
            parent_lost=parent_lost,
            conversation=conversation,
            scope=scope,
            issuer_proven=proven,
            start_ns=start,
            end_ns=now,
            ttfb_ms=None,  # per-h2-stream first-byte not tracked: not measured
            truncated=t.truncated or request_evicted,
            # A request whose END_STREAM had not arrived when the response ended holds only what was
            # sent so far (an upload refused part-way): like HTTP/1's unfinished request, no size.
            request_counted=t.request_ended and not (t.request_truncated or request_evicted),
            response_counted=not t.response_truncated,
            stream_id=t.stream_id,
            limitations=(Limitation.H2_REQUEST_EVICTED,) if request_evicted else (),
            version="2",
            ttft_ms=None,  # per-h2-stream first-body-byte not tracked: not measured
            content_type=getattr(t, "content_type", None),
            event_stream=declares_event_stream(getattr(t, "content_type", None)),
            content_encoding=t.content_encoding,
            grpc_status=getattr(t, "grpc_status", None),
            grpc_message=getattr(t, "grpc_message", None),
        )


class _WebSocketTracker:
    """One WS connection — per-direction frame parser + aggregation + 64KB content sample.
    Emits 1 span once both Close frames have crossed, or on flush."""

    def __init__(
        self,
        path: str,
        deflate: bool,
        parent: SpanContext | None,
        start_ns: int,
        parent_closed: bool = False,
        limits: object | None = None,
        sample_cap: int | None = None,
        llm_upgrade: str | None = None,
        conversation: ConversationContext | None = None,
        scope: ScopeSnapshot | None = None,
    ) -> None:
        # "known_provider" | "unknown_host" | None: the endpoint table's answer about the upgrade
        # path (`classify_ws_upgrade`), decided by the seam at the swap site. The tracker only
        # confirms it — once: `_decide_llm` nulls this, so "already decided" and "nothing to
        # decide" are the same state and there is no second flag to keep in step with it.
        self._llm_upgrade = llm_upgrade
        self._llm_call = False
        self._llm_unconfirmed = False
        self._sent = WsParser(limits)  # client -> server
        self._recv = WsParser(limits)  # server -> client
        self._path = _name_path(path)
        self._target = path
        self._deflate = deflate
        self._parent = parent
        # Inherited from the UPGRADE transaction rather than re-latched: a WS
        # session's parent is the scope that issued the handshake, and so is the
        # question of whether that scope's unit had already died.
        self._parent_closed = parent_closed
        # The handshake's conversation, kept only while every message the client sends is issued in
        # it too: the session is ONE span, so a socket reused across conversations names none of
        # them rather than whichever one happened to open it.
        self._conversation = conversation
        # The handshake's identity, kept by the same rule as the conversation: only while every
        # message the client sends is issued under it too. A socket one tenant opened and another
        # writes to would otherwise ship the second tenant's payload under the first one's name.
        self._scope = scope
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
        #: The directions ("sent", "received") a Close frame crossed in. The session ends when both
        #: have: the peer may still send data after the first Close (RFC 6455 5.5.1), and its own
        #: Close is part of the session too, so a span built at the first one undercounted both.
        self._close_from: set[str] = set()
        self._emitted = False

    def on_request_bytes(self, data: bytes) -> list[_Txn]:
        r = self._sent.feed(data)
        self._sent_bytes += self._count(r.frames, "sent")
        self._sent_msgs += len(r.messages)
        # A parser that died can no longer tell a message from a frame, so every write counts then.
        if r.messages or self._sent.is_disabled():
            if self._conversation is not None:
                if _hub.get_current_scope().conversation != self._conversation:
                    self._conversation = None
            if self._scope is not None and self._scope != UNKNOWN_ISSUER:
                if issued_scope() != self._scope:
                    self._scope = UNKNOWN_ISSUER  # issued under more than one: names no one
        if self._llm_upgrade is not None:
            if r.messages:
                self._decide_llm(r.messages[0])
            elif self._sent.is_disabled():
                # The parser died on the first client frame (oversize, desync) and will never yield
                # a message. Bytes crossed all the same, so decide now on nothing readable: a
                # decision that waited for a message would be starved by the parse failure and the
                # connection would vanish without a counter.
                self._decide_llm(None)
        self._in_trunc = self._append_sample(self._sample_in, r.messages) or self._in_trunc
        return self._maybe_emit()

    def _decide_llm(self, first: bytes | None) -> None:
        # Decided once, on the first client message — the moment "a call crossed" becomes true — or,
        # when the client-direction parser disables before yielding one, on the bytes that killed it
        # (`first` is None then). The path alone is a suffix match; it is corroborated by the
        # provider's own host or, when nothing hides the payload (no permessage-deflate, a readable
        # message), by the Responses envelope itself. The tracker only records the answer on the
        # `_Txn` it emits at close; the seam reads it there and counts — so the two counters move
        # when the connection closes, not at this first message, and a span the capture gate then
        # refuses is still counted.
        if self._llm_upgrade == "known_provider" or (
            first is not None and not self._deflate and _RESPONSES_CREATE.search(first) is not None
        ):
            self._llm_call = True
        else:
            self._llm_unconfirmed = True
        self._llm_upgrade = None

    def on_response_bytes(self, data: bytes) -> list[_Txn]:
        r = self._recv.feed(data)
        self._recv_bytes += self._count(r.frames, "received")
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

    def _count(self, frames: list[Any], side: str) -> int:
        """The payload bytes in `frames`, noting a Close frame from `side`. The session's close
        code is the first Close frame's: the one that began the closing handshake."""
        for f in frames:
            if f.opcode == "close":
                if not self._close_from:
                    self._close_code = f.close_code
                self._close_from.add(side)
        return sum(f.payload_len for f in frames)

    def _maybe_emit(self) -> list[_Txn]:
        if len(self._close_from) == 2 and not self._emitted:
            return [self._build_txn(())]
        return []

    def flush(self, marker: Limitation, *, still_open: bool = False) -> list[_Txn]:
        """The socket closed or the seam let go before the closing handshake completed.
        `WS_NO_CLOSE` says no Close frame crossed at all, so a session that saw one does not take
        it; the seam decides from the ending whether the counts are whole."""
        if self._emitted:
            return []
        drop = marker is Limitation.WS_NO_CLOSE and bool(self._close_from)
        return [self._build_txn(() if drop else (marker,))]

    #: The connection-close verb every tracker answers to. For a WS session it IS `flush`, and an
    #: ALIAS rather than a delegating wrapper: this is the one tracker with something to save at
    #: close — its span exists only once the session ends — so the two names must never be able to
    #: drift apart. Before the close hook, that span waited for `uninstall()` and was lost whenever
    #: the process never reached one.
    on_connection_close = flush

    def _build_txn(self, extra_markers: tuple[Limitation, ...]) -> _Txn:
        self._emitted = True
        markers = list(extra_markers)
        if self._in_trunc or self._out_trunc:
            markers.append(Limitation.WS_PAYLOAD_TRUNCATED)
        if self._deflate:
            # Census merge (§6.5.1): `ws_compressed` folded into PAYLOAD_COMPRESSED. What is lost is
            # which protocol it was, and `TransportAttributes.protocol` already carries that.
            markers.append(Limitation.PAYLOAD_COMPRESSED)
        if self._sent.is_disabled() or self._recv.is_disabled():
            # Census merge: `ws_parse_failed` and `grpc_parse_failed` are one fact — the framing
            # layer failed, so the transport fields on this span are partial or synthesized.
            markers.append(Limitation.FRAME_PARSE_FAILED)
        if self._llm_call:
            markers.append(Limitation.WS_LLM_SEMANTICS_UNREAD)
        now = time.time_ns()
        return _Txn(
            method="GET",
            path=self._path,
            target=self._target,
            status=101,
            request_body=bytes(self._sample_in),
            response_body=bytes(self._sample_out),
            parent=self._parent,
            parent_closed=self._parent_closed,
            conversation=self._conversation,
            scope=self._scope,
            start_ns=self._start_ns,
            end_ns=now,
            ttfb_ms=None,
            version="websocket",
            ws_close_code=self._close_code,
            ws_messages_sent=self._sent_msgs,
            ws_messages_received=self._recv_msgs,
            ws_bytes_sent=self._sent_bytes,
            ws_bytes_received=self._recv_bytes,
            request_counted=not self._sent.is_disabled(),
            response_counted=not self._recv.is_disabled(),
            ws_markers=tuple(markers),
            ws_llm_call=self._llm_call,
            ws_llm_unconfirmed=self._llm_unconfirmed,
        )
