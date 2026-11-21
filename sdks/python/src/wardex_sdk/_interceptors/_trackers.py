"""Protocol-specific trackers.

When the SSL interceptor (_ssl.py) streams plaintext bytes into a tracker, the tracker
parses and correlates them according to the protocol (HTTP/1 or HTTP/2) and returns a
list of _Txn (`_txn.py`). Span assembly is performed based solely on _Txn (protocol-agnostic).
The WebSocket tracker an HTTP/1 upgrade swaps in is `_ws_tracker.py`.
"""

from __future__ import annotations

import time
from typing import Any

from .. import _hub, _wardex_native
from .._assembly import (
    Limitation,
    counters,
    latch_parent,
    parent_is_closed_unit,
    should_capture,
)
from .._enums import CaptureMode
from .._protocol import classify_path
from .._protocol._http1 import Http1ResponseParser, declares_event_stream
from .._protocol._http2 import Http2Parser
from .._types import ConversationContext, ParsedMessage, SpanContext
from ._h2_issuer import IssuerLink
from ._http1_requests import RequestSide, header_get, is_ws_upgrade_request
from ._issue_scope import UNKNOWN_ISSUER, ScopeSnapshot
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


#: A stream's latch entry: (parent, parent closed, conversation, scope, unseen parent, issuer
#: proven, request start ns).
_StreamLatch = tuple[
    SpanContext | None, bool, ConversationContext | None, ScopeSnapshot, str | None, bool, int
]


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
        return request is not None and is_ws_upgrade_request(request.headers)

    def keep_bodies(self, keep_all: bool) -> None:
        """The seam's answer, for the request bytes it feeds next, to "could the capture policy use
        this exchange whatever its response says?". False stores a request body only for an LLM
        endpoint path, or until its reply decides for a JSON POST elsewhere; and the reply's only
        when it is an event stream (`_reply_policy`)."""
        self._requests.keep_bodies(keep_all)

    def on_request_bytes(self, data: bytes) -> list[_Txn]:
        self._requests.feed(data)
        self._expect_ws = self._upgrade_requested()
        return []

    def _reply_policy(self) -> str:
        """The body policy for the reply about to open, from what its request's body was stored for:
        a model call's makes the reply one too (first claim on the in-flight budget); a body kept
        under "keep" keeps the reply's; a provisional or skipped one stores the reply's only if it
        is an event stream. No complete request waiting (one still arriving, or none seen) keeps
        it: nothing says the policy can do without it."""
        request = self._requests.request
        if request is None or request.body_use == "kept":
            return "keep"
        return "model_call" if request.body_use == "model_call" else "model_calls_only"

    def pass_unseen(self, n: int) -> bool:
        """Take `n` response bytes without seeing them, when they can only be a body this tracker
        is not storing and they do not end it (`Http1Parser.pass_unseen_body`). They leave no byte
        mark, so such a response has no TTFT (`_response_txn`)."""
        return self._resp.pass_unseen_body(n)

    def on_response_bytes(self, data: bytes) -> list[_Txn]:
        if self._requests.decide():
            self._expect_ws = self._upgrade_requested()
        if self._resp.idle():
            # These bytes open a response, so its request is settled: HEAD and CONNECT frame their
            # answers, and only the request side knows which one this reply answers.
            self._resp.expect_response_to(self._requests.method)
            self._resp.set_body_policy(self._reply_policy())
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
            if self._expect_ws and msg.status_code == 101 and is_ws_upgrade_request(msg.headers):
                ext = header_get(msg.headers, "sec-websocket-extensions") or ""
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
                        unseen_parent=issue.unseen_parent,
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
        # Reads inside a body left unstored may have gone by unseen (`pass_unseen`): not measured.
        response_kept = msg.body_use != "skipped"
        ttft = (
            _ttft_from_marks(self._resp_marks, msg.header_len, issue.start_ns)
            if response_kept
            else None
        )
        path = (request.url if request is not None else None) or "/"
        cut = (Limitation.FRAME_PARSE_FAILED,) if msg.incomplete else ()
        request_kept = request is None or request.body_use != "skipped"
        return _Txn(
            method=(request.method if request is not None else None) or "?",
            path=_name_path(path),
            target=path,
            status=msg.status_code or 0,
            request_body=request.body if request is not None else b"",
            response_body=msg.body,
            parent=issue.parent,
            parent_closed=issue.parent_closed,
            unseen_parent=issue.unseen_parent,
            conversation=issue.conversation,
            scope=issue.scope,
            start_ns=issue.start_ns or now,
            end_ns=now,
            ttfb_ms=ttfb,
            truncated=(request is not None and request.truncated) or msg.truncated,
            request_counted=request is not None and not request.truncated and request_kept,
            response_counted=not msg.truncated and response_kept,
            request_body_kept=request_kept,
            response_body_kept=response_kept,
            response_cut=msg.incomplete,
            limitations=_merge_markers(
                request.limitations if request is not None else (), msg.limitations, cut
            ),
            version="1.1",
            ttft_ms=ttft,
            event_stream=declares_event_stream(header_get(msg.headers, "content-type")),
            content_encoding=header_get(msg.headers, "content-encoding"),
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
        # `_mk` pops on every transaction, and `_forget_reset` on every RST_STREAM the parser
        # reports, so the entries that accumulate are the streams that end with neither: a GOAWAY
        # that strands everything above `last_stream_id`, a server that stops mid-response. There is
        # no close signal to act on for those — the parser reports transactions and resets, not
        # stream lifecycles.
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
        #: avoids; an id it forgets reads as a parentless eviction, the gate staying shut, and is
        #: counted (`protocol.http2.evicted_parent_forgotten`).
        self._evicted_admitting: set[int] = set()
        #: Who opened each stream, read in the opening call itself (`_h2_issuer`).
        self._issuers = IssuerLink(self._latch_cap)
        #: The request path of each latched stream that is a model call (`classify_path`), taken
        #: from the `feed` that opened it, which read it as the request ended: the native table
        #: evicts with the same cap inside that same `feed`, so no question asked afterwards could
        #: still answer it. Keys are latch keys.
        self._model_paths: dict[int, str] = {}
        #: Streams whose latch entry the cap evicted while it held a parent, with that parent and
        #: the model-call path (None for any other request), until their answer arrives (`_mk`).
        #: Bounded by the cap, lowest id dropped first; what is dropped or still here at close is
        #: counted as unanswered. A stream a RST_STREAM ended leaves uncounted (`_forget_reset`).
        self._evicted_calls: dict[int, tuple[SpanContext, str | None]] = {}

    def on_request_bytes(self, data: bytes) -> list[_Txn]:
        self._issuers.see(data)
        opened, txns, reset = self._conn.feed(True, data)
        now = time.time_ns()
        # The WRITER's scope, which on a shared connection may be any task's: any of them flushes
        # the others' queued frames. So it is the parent only of a stream with no proven issuer, and
        # never anyone's conversation or identity (`IssuerLink.latch`).
        scope = _hub.get_current_scope()
        parent = scope.active_span_context
        if opened:  # latched only for the streams this write opens
            parent, _ = latch_parent(parent, None)
        parent_closed = parent_is_closed_unit(parent) if opened else False
        unseen = _hub.unseen_parent(scope) if opened and parent is None else None
        for sid, path in opened:
            self._latch[sid] = (*self._issuers.latch(sid, parent, parent_closed, unseen), now)
            if path is not None and classify_path(path) == "llm_call":
                self._model_paths[sid] = path
        if opened:
            low = min(sid for sid, _ in opened)
            self._latch_first = low if self._latch_first == 0 else min(self._latch_first, low)
        self._forget_reset(reset)  # before the cap runs: a cancelled stream is not one it evicts
        # Drop-lowest-stream-id: ids only ever increase, so the entry evicted is the one likeliest
        # to be stranded already. Lowest by id, not by insertion — a stream announced late (its
        # request body ended after higher ids') is still the oldest request. Losing it costs that
        # stream its parentage, never a span — `_mk` falls back to a parentless entry and says so on
        # the span. `min` is a scan of at most `max_streams + 1` keys, run only on the writes that
        # overflow the cap.
        while len(self._latch) > self._latch_cap:
            evicted = min(self._latch)
            dropped = self._latch.pop(evicted)
            self._remember_evicted_call(evicted, dropped[0])
            if _admits(*dropped[:2]):
                self._evicted_admitting.add(evicted)
                if len(self._evicted_admitting) > self._latch_cap:
                    # A late answer to the forgotten stream is refused under AGENT like traffic
                    # outside agent work; this count is its only trace.
                    self._evicted_admitting.discard(min(self._evicted_admitting))
                    counters.bump("protocol.http2.evicted_parent_forgotten")
            self._latch_evicted_below = max(self._latch_evicted_below, evicted)
        return self._ship(txns)

    def _remember_evicted_call(self, stream_id: int, parent: SpanContext | None) -> None:
        """Remember a stream whose latch entry the cap evicted while it held a parent, so that its
        answer never arriving is counted rather than silent.

        For a model call it also keeps what the seal's model-call witness (`witness_request`) needs:
        its answer reaches `_seal` with no parent, so the seal would witness nothing, and a bridge
        chat unit above it would promote its framework usage while the wire span ships its own, one
        call counted twice. The witness is handed over with the answer (`_Txn.evicted_parent`), not
        given here: a call whose answer never arrives must leave the unit's number promoted, or the
        usage is lost with no wire span to carry it.

        Any other request keeps only the fact: its answer, if it comes, says the eviction on its own
        span (`_mk`), and one that never comes would otherwise leave no trace that wardex latched a
        stream under agent work and lost it. A stream that held no parent lost nothing."""
        path = self._model_paths.pop(stream_id, None)
        if parent is None:
            return
        self._evicted_calls[stream_id] = (parent, path)
        if len(self._evicted_calls) > self._latch_cap:
            self._count_unanswered(self._evicted_calls.pop(min(self._evicted_calls))[1])

    def _forget_reset(self, reset: list[int]) -> None:
        """Let go of every stream a RST_STREAM ended, from either peer, without counting it.

        No transaction will ever complete for such a stream, so its latch entry, model-call path
        and evicted-call record would otherwise wait for an answer that cannot come, and the cap or
        the close would then count it as `evicted_unanswered` / `evicted_call_unanswered`. A request
        the host itself cancelled, or one the server refused, is not a loss wardex caused: nothing
        that belongs on a span went missing."""
        for sid in reset:
            self._latch.pop(sid, None)
            self._model_paths.pop(sid, None)
            self._evicted_calls.pop(sid, None)

    @staticmethod
    def _count_unanswered(path: str | None) -> None:
        if path is not None:  # its usage stays on the unit, promoted; no wire span has it
            counters.bump("protocol.http2.evicted_call_unanswered")
        else:
            counters.bump("protocol.http2.evicted_unanswered")

    def on_response_bytes(self, data: bytes) -> list[_Txn]:
        # server push (PUSH_PROMISE) unsupported — response-side stream_id ignored
        _opened, txns, reset = self._conn.feed(False, data)
        self._forget_reset(reset)
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

        A latch entry is two references and two scalars, so this is small money per connection and,
        without the cap in `__init__`, unbounded money over a process: an h2 server that stops
        answering mid-response strands one entry per stream for the life of the connection (a reset
        stream no longer does: `_forget_reset`), and a keep-alive h2 connection to a model provider
        lives as long as the process does. This is the release that costs nothing, where it is
        reachable; the cap bounds the paths where it is not.

        At a real close (not `still_open`), a stream whose response status arrived and whose end
        never did ships as what arrived, marked `FRAME_PARSE_FAILED` and `response_cut` (`_mk`), as
        HTTP/1's response in flight does: a client that stops reading a stream at its last event
        never waits for the empty DATA frame carrying END_STREAM, and the usage that stream carried
        belongs on the wire record. A stream with no status ships nothing; the seam has nothing to
        say about it that would not be invented.

        The eviction bookkeeping goes with the entries, and it has to: h2 stream ids restart at 1 on
        the next connection, so a mark or a floor taken on the last one names a different set of
        streams here. No caller reuses a tracker across a close today — every `_retire` in
        `_seam.py` discards the `_ConnectionState` and the tracker inside it — but nothing declares
        that, and the cost of a stale mark is every unlatched stream on the new connection reporting
        a parent wardex never lost.
        """
        out = [] if still_open else self._ship(self._conn.finish())
        for _parent, path in self._evicted_calls.values():
            self._count_unanswered(path)
        self._evicted_calls.clear()
        self._model_paths.clear()
        self._latch.clear()
        self._issuers.clear()
        self._evicted_admitting.clear()
        self._latch_evicted_below = 0
        self._latch_first = 0
        return out

    def _mk(self, t: Any) -> _Txn:
        now = time.time_ns()
        entry = self._latch.pop(t.stream_id, None)
        model_path = self._model_paths.pop(t.stream_id, None)
        witness = None
        if entry is None:
            parent, parent_closed, conversation, proven, start = None, False, None, False, now
            unseen = None
            call = self._evicted_calls.pop(t.stream_id, None)
            if call is not None and call[1] is not None:
                witness, model_path = call  # the seal witnesses the call it evicted the parent of
                counters.bump("protocol.http2.evicted_witnessed")
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
            parent, parent_closed, conversation, scope, unseen, proven, start = entry
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
        cut = not t.response_ended  # handed back at close (`on_connection_close`): a partial body
        target = t.path or model_path or "/"  # a request the native table evicted: path kept here
        return _Txn(
            method=t.method or "?",
            path=_name_path(target),
            target=target,
            status=t.status,
            request_body=t.request_body,
            response_body=t.response_body,
            parent=parent,
            evicted_parent=witness,
            parent_closed=parent_closed,
            parent_evicted=parent_evicted,
            parent_lost=parent_lost,
            unseen_parent=unseen,
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
            response_cut=cut,
            stream_id=t.stream_id,
            # A body past its cap carries the marker HTTP/1 gives the same cut (`append_capped`), so
            # one query finds every capped body whatever version carried it. HTTP/2 reports the cut
            # per direction instead of as a marker, and this is where those two flags become one.
            limitations=_merge_markers(
                (Limitation.H2_REQUEST_EVICTED,) if request_evicted else (),
                (Limitation.BODY_CAP_EXCEEDED,) if t.truncated else (),
                (Limitation.FRAME_PARSE_FAILED,) if cut else (),
            ),
            version="2",
            ttft_ms=None,  # per-h2-stream first-body-byte not tracked: not measured
            content_type=getattr(t, "content_type", None),
            event_stream=declares_event_stream(getattr(t, "content_type", None)),
            content_encoding=t.content_encoding,
            grpc_status=getattr(t, "grpc_status", None),
            grpc_message=getattr(t, "grpc_message", None),
        )
