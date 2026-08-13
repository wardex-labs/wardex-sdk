"""Loopback OTLP receiver for the Anthropic Agent SDK OTel bridge.

The Claude Code CLI carries its own OpenTelemetry telemetry; with
``AnthropicAgentSdkConfig(otel_bridge=True)`` the adapter points the CLI's
exporter at this receiver (``_prepare_options`` injects the env) and the
assembler merges what arrives into the session tree at finalize. This module
is the LISTENING half only: it accepts, bounds, scrubs and files spans by
session — it interprets nothing (``_otel_merge`` does) and emits nothing
(the assembler does).

Shape and rules:

* stdlib only, ``http.server.ThreadingHTTPServer`` bound to ``127.0.0.1``
  port 0 — never an external interface. Hand-rolling a socket loop would
  re-implement HTTP/1.1 parsing (keep-alive, header folding) badly; the
  thread-per-request cost is bounded in practice by the loopback bind, the
  secret token (a request without it is refused before its body is read,
  with ``Connection: close``), and the per-socket timeout.
* only ``POST /v1/traces`` ever reads a body. Other paths answer 404, other
  methods 405, both without touching the body.
* the token header (``x-wardex-bridge``, ``secrets.token_urlsafe(32)`` minted
  per receiver) is compared with ``hmac.compare_digest``; absent or wrong →
  403 and no slot is created. Loopback is not authorization: any process on
  the machine can POST here, and only the CLI wardex spawned was handed the
  token.
* the body is bounded TWICE, both from the core limits table
  (``max_otel_bridge_body_bytes``): the declared Content-Length, and what a
  gzip body decompresses to — ``zlib.decompressobj(max_length=...)`` is the
  decompression-bomb bound the crate-level gunzip does not offer, which is
  why the stdlib does this instead of a new PyO3 binding. Over either → 413,
  counted, nothing parsed.
* per-session retention is bounded by ``max_otel_bridge_spans_per_session``:
  over → newest dropped + counted on the slot, so the merge can say what it
  never saw.
* identity PII is scrubbed AT THIS BOUNDARY, before a span is stored: the CLI
  stamps ``user.email`` / ``user.id`` / ``user.account_uuid`` /
  ``organization.id`` on every span, and ``user_prompt`` may carry raw prompt
  text under the CLI's own log flags. None of it ever sits in wardex memory.
* an undecodable POST answers 200 — destabilizing the CLI's exporter to
  report our own parse failure would trade the host's telemetry for a
  diagnostic — and is counted; the assembler turns it into the schema-drift
  marker when exactly one bridge session is live to own it.
* lifetime = the ADAPTER, not the session: one receiver serves N concurrent
  sessions, routed by trace id (the reservation minted at injection) with the
  span's own ``session.id`` attribute as fallback. ``close()`` rides the
  adapter's ``uninstall()``.

KNOWN GAP (out of scope by design): a user who exports ``OTEL_*`` MID-SESSION
is not caught — the never-hijack check in ``_prepare_options`` reads the env
once, at spawn time, which is the only moment the decision can still change
what the subprocess inherits.
"""

from __future__ import annotations

import hmac
import http.server
import secrets
import threading
import time
import zlib
from dataclasses import dataclass, field
from typing import Any

from .._assembly import counters, report_once
from .._native import native

#: The token header the injected `OTEL_EXPORTER_OTLP_HEADERS` carries.
_TOKEN_HEADER = "x-wardex-bridge"

#: Per-connection socket timeout. A stalled peer holds one daemon thread, not
#: the receiver: this is what turns "a local process opened a socket and went
#: quiet" into a bounded cost.
_SOCKET_TIMEOUT_S = 10.0


@dataclass
class _BridgeSlot:
    """One session's decoded, scrubbed, not-yet-merged CLI spans."""

    spans: list[dict] = field(default_factory=list)
    #: Resource attributes from the last request that fed this slot (scrubbed).
    resource: dict = field(default_factory=dict)
    #: Spans dropped by the per-session cap — the merge reports them.
    dropped: int = 0
    #: An undecodable POST was attributed to this slot (sole-live inference).
    schema_failed: bool = False
    #: ``time.monotonic()`` of the last arrival, for the drain's quiescence
    #: check. None until anything arrives.
    last_arrival: float | None = None
    #: Reverse keys, so ``take()`` can drop every index entry for this slot.
    trace_key: str | None = None
    session_key: str | None = None


def _scrub(attrs: Any) -> dict:
    """The identity denylist, applied before anything is stored (R12).

    ``user.*`` and ``organization.*`` as PREFIXES plus ``user_prompt`` by
    name: the named keys are what the spike measured on every span, and the
    prefixes are what keeps a new key in either family from arriving as a
    surprise. This is the denylist half; the merge's named allowlist is the
    structural guarantee on top of it.
    """
    if not isinstance(attrs, dict):
        return {}
    return {
        k: v
        for k, v in attrs.items()
        if not (k == "user_prompt" or k.startswith("user.") or k.startswith("organization."))
    }


class _OtelBridgeReceiver:
    """The in-process OTLP/HTTP receiver plus the per-session slot store."""

    def __init__(
        self,
        *,
        max_body_bytes: int,
        max_spans_per_session: int,
        max_sessions: int,
    ) -> None:
        self._max_body_bytes = max_body_bytes
        self._max_spans_per_session = max_spans_per_session
        self._max_sessions = max_sessions
        self.token = secrets.token_urlsafe(32)
        #: One lock guards both tables and every slot's contents.
        self._lock = threading.Lock()
        self._by_trace: dict[str, _BridgeSlot] = {}
        self._by_session_id: dict[str, _BridgeSlot] = {}

        receiver = self

        class _Server(http.server.ThreadingHTTPServer):
            daemon_threads = True
            # The default (True) would leak the port to every interface the
            # hostname resolves to on some platforms; the bind address below
            # is the whole security story and must stay literal.
            allow_reuse_address = False

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = _SOCKET_TIMEOUT_S

            def log_message(self, *args: Any) -> None:  # noqa: D102 — silence stderr
                pass

            def do_POST(self) -> None:  # noqa: N802 — http.server's spelling
                try:
                    receiver._handle_post(self)
                except Exception:  # noqa: BLE001 — a broken peer, not the host
                    counters.bump("adapters.anthropic.otel_bridge.handler_error")

            def do_GET(self) -> None:  # noqa: N802
                receiver._refuse(self, 405)

            def do_PUT(self) -> None:  # noqa: N802
                receiver._refuse(self, 405)

            def do_DELETE(self) -> None:  # noqa: N802
                receiver._refuse(self, 405)

        self._server = _Server(("127.0.0.1", 0), _Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="wardex-otel-bridge",
            daemon=True,
        )
        self._thread.start()

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # ------------------------------------------------------------------
    # HTTP handling
    # ------------------------------------------------------------------

    def _refuse(self, handler: Any, status: int) -> None:
        """Answer without reading the body — so the connection must close.

        HTTP/1.1 keep-alive would leave the unread body to be parsed as the
        next request line; ``Connection: close`` is what makes not reading it
        safe.
        """
        handler.send_response(status)
        handler.send_header("Content-Length", "0")
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.close_connection = True

    def _handle_post(self, handler: Any) -> None:
        if handler.path.split("?", 1)[0] != "/v1/traces":
            self._refuse(handler, 404)
            return
        supplied = handler.headers.get(_TOKEN_HEADER) or ""
        if not hmac.compare_digest(supplied.encode(), self.token.encode()):
            # The body is never read: an unauthorized peer does not get to
            # choose how many bytes this process buffers.
            counters.bump("adapters.anthropic.otel_bridge.token_rejected")
            self._refuse(handler, 403)
            return
        length = self._declared_length(handler)
        if length < 0 or length > self._max_body_bytes:
            counters.bump("adapters.anthropic.otel_bridge.body_rejected")
            self._refuse(handler, 413)
            return
        body = handler.rfile.read(length)
        if (handler.headers.get("Content-Encoding") or "").lower() == "gzip":
            body = self._gunzip_bounded(body)
            if body is None:
                counters.bump("adapters.anthropic.otel_bridge.body_rejected")
                self._refuse(handler, 413)
                return
        decoded: dict | None = None
        try:
            decoded = native.codec.decode_otlp_traces(body)
        except Exception:  # noqa: BLE001 — foreign bytes; fail-open is the contract
            counters.bump("adapters.anthropic.otel_bridge.decode_error")
        if decoded is None:
            # 200 on purpose: an error status would put the CLI's exporter
            # into retry against a receiver that will never accept the bytes.
            counters.bump("adapters.anthropic.otel_bridge.undecodable")
            report_once(
                "anthropic_agent_sdk otel bridge: a telemetry POST did not decode "
                "as OTLP; the session tree is unchanged (fail-open)",
                key="adapters.anthropic_agent_sdk.otel_bridge.undecodable",
            )
            self._note_schema_failure()
        else:
            self._ingest(decoded)
        handler.send_response(200)
        handler.send_header("Content-Type", "application/x-protobuf")
        handler.send_header("Content-Length", "0")
        handler.end_headers()

    def _declared_length(self, handler: Any) -> int:
        """Content-Length as an int; -1 for absent or unparseable.

        Absent folds into the same 413 as oversize: a chunked or unbounded
        body would have to be read to be measured, and reading before
        measuring is the exact order the cap forbids.
        """
        raw = handler.headers.get("Content-Length")
        if raw is None:
            return -1
        try:
            return int(raw)
        except ValueError:
            counters.bump("adapters.anthropic.otel_bridge.body_rejected")
            return -1

    def _gunzip_bounded(self, body: bytes) -> bytes | None:
        """Decompress with the bomb bound; None means over-cap or invalid."""
        try:
            d = zlib.decompressobj(wbits=31)
            out = d.decompress(body, self._max_body_bytes + 1)
        except zlib.error:
            counters.bump("adapters.anthropic.otel_bridge.gzip_error")
            return None
        if len(out) > self._max_body_bytes or d.unconsumed_tail:
            return None
        return out

    # ------------------------------------------------------------------
    # Slot store
    # ------------------------------------------------------------------

    def _note_schema_failure(self) -> None:
        """Attribute an undecodable POST to the SOLE live slot, if there is one.

        A counted inference: the body never decoded, so nothing routes it.
        With exactly one live bridge session the attribution is forced; with
        several, picking one would be a guess, and the counter alone speaks.
        """
        with self._lock:
            slots = set(map(id, self._by_trace.values())) | set(
                map(id, self._by_session_id.values())
            )
            if len(slots) != 1:
                return
            slot = next(iter(self._by_trace.values()), None) or next(
                iter(self._by_session_id.values())
            )
            slot.schema_failed = True
            slot.last_arrival = time.monotonic()

    def reserve(self, trace_id_hex: str) -> None:
        """File an empty slot under the trace id minted at env injection."""
        with self._lock:
            while len(self._by_trace) >= self._max_sessions:
                # FIFO over reservations: a query() that never spawned keeps
                # its slot only until the table needs the room (I10).
                oldest = next(iter(self._by_trace))
                dead = self._by_trace.pop(oldest)
                if dead.session_key:
                    self._by_session_id.pop(dead.session_key, None)
                counters.bump("adapters.anthropic.otel_bridge.reservation_evicted")
            slot = _BridgeSlot()
            slot.trace_key = trace_id_hex
            self._by_trace[trace_id_hex] = slot

    def _ingest(self, decoded: dict) -> None:
        """Scrub, route and file every span of one decoded request."""
        for rs in decoded.get("resource_spans", ()):
            resource_attrs = _scrub((rs.get("resource") or {}).get("attributes"))
            for ss in rs.get("scope_spans", ()):
                for span in ss.get("spans", ()):
                    span["attributes"] = _scrub(span.get("attributes"))
                    self._file(span, resource_attrs)

    def _file(self, span: dict, resource_attrs: dict) -> None:
        now = time.monotonic()
        with self._lock:
            slot = self._by_trace.get(span.get("trace_id") or "")
            session_id = span["attributes"].get("session.id")
            if slot is None and isinstance(session_id, str) and session_id:
                slot = self._by_session_id.get(session_id)
                if slot is None:
                    # session.id fallback routing: the CLI stamps it on every
                    # trace span (measured), so a session whose TRACEPARENT
                    # injection could not be read back still converges here.
                    if len(self._by_session_id) >= self._max_sessions:
                        counters.bump("adapters.anthropic.otel_bridge.span_unroutable")
                        return
                    slot = _BridgeSlot()
                    slot.session_key = session_id
                    self._by_session_id[session_id] = slot
            if slot is None:
                counters.bump("adapters.anthropic.otel_bridge.span_unroutable")
                return
            if isinstance(session_id, str) and session_id and slot.session_key is None:
                # Index a trace-routed slot by its session id too, so the two
                # lookups converge on one slot for the rest of the session.
                slot.session_key = session_id
                self._by_session_id[session_id] = slot
            slot.last_arrival = now
            if len(slot.spans) >= self._max_spans_per_session:
                slot.dropped += 1
                counters.bump("adapters.anthropic.otel_bridge.span_dropped")
                return
            slot.resource = resource_attrs
            slot.spans.append(span)

    def _find(self, trace_id_hex: str | None, session_id: str | None) -> _BridgeSlot | None:
        """Caller holds ``self._lock``."""
        if trace_id_hex:
            slot = self._by_trace.get(trace_id_hex)
            if slot is not None:
                return slot
        if session_id:
            return self._by_session_id.get(session_id)
        return None

    def take(self, trace_id_hex: str | None, session_id: str | None) -> _BridgeSlot | None:
        """Pop the session's slot from BOTH indexes; None means no data ever."""
        with self._lock:
            slot = self._find(trace_id_hex, session_id)
            if slot is None:
                return None
            if slot.trace_key:
                self._by_trace.pop(slot.trace_key, None)
            if slot.session_key:
                self._by_session_id.pop(slot.session_key, None)
            return slot

    def has_data(self, trace_id_hex: str | None, session_id: str | None) -> bool:
        """Whether at least one span arrived — the drain's precondition."""
        with self._lock:
            slot = self._find(trace_id_hex, session_id)
            return slot is not None and bool(slot.spans)

    def last_arrival(self, trace_id_hex: str | None, session_id: str | None) -> float | None:
        with self._lock:
            slot = self._find(trace_id_hex, session_id)
            return slot.last_arrival if slot is not None else None

    def close(self) -> None:
        """Stop serving and release the socket. Rides the adapter's uninstall."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)
