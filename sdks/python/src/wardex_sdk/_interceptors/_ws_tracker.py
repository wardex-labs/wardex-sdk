"""The WebSocket tracker an HTTP/1 upgrade swaps in for `_trackers._Http1Tracker`.

Per-direction frame parsers, aggregation and a bounded content sample; one `_Txn` per session
(`_txn.py`), built once both Close frames have crossed or on flush.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .. import _hub, _wardex_native
from .._assembly import Limitation
from .._protocol import WsParser
from .._types import ConversationContext, SpanContext
from ._issue_scope import UNKNOWN_ISSUER, ScopeSnapshot, issued_scope, same_issuer
from ._txn import _name_path, _Txn

#: The `type` of the one client message the Responses WebSocket transport sends per call, searched
#: for anywhere in the first client message rather than matched as a prefix: key order
#: (`sort_keys`), a BOM or leading whitespace must not decide the connection's fate. The search is
#: bounded by the message, which the frame parser already caps. Consulted only when nothing hides
#: the payload.
_RESPONSES_CREATE = re.compile(rb'"type"\s*:\s*"response\.create"')


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
        # message the client sends is issued under it too (`same_issuer`: every tag and user
        # field the handshake carried, unchanged; fields added later do not count). A socket one
        # tenant opened and another writes to would otherwise ship the second tenant's payload
        # under the first one's name.
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
                if not same_issuer(self._scope, issued_scope()):
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
