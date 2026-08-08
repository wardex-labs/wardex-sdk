"""SSL/TLS interceptor — monkeypatches ssl.SSLSocket (sync) + ssl.SSLObject (async).

Captures the plaintext application-layer bytes right before/after TLS, feeds them
into a protocol tracker, and assembles a CLIENT span from the _Txn the tracker
returns. Works standalone, without any adapter.
The shared skeleton (tracker feeding, span assembly, state) lives in the
ByteSeamInterceptor base; only the SSL-specific parts (patching the SSL classes,
ALPN-based tracker selection, connection timing) remain here.
"""

from __future__ import annotations

import ssl
from typing import TYPE_CHECKING, Any

from ..assembly import Limitation
from ._conn_timing import shared_timing_store
from ._seam import ByteSeamInterceptor, _accepted_prefix, _ConnectionState
from ._socket import _H2_PREFACE, _HTTP_METHODS
from ._trackers import _Http1Tracker, _Http2Tracker

if TYPE_CHECKING:
    from .._client import Client

__all__ = ["SSLInterceptor"]


class SSLInterceptor(ByteSeamInterceptor):
    """Global monkeypatch interceptor for the ssl module."""

    def name(self) -> str:
        return "ssl"

    # --- Install / uninstall ---

    def install(self, client: Client | None) -> None:
        if self._installed:
            return
        self._client = client
        self._load_limits(client)
        self._patches = self._fresh_patchset()
        # The original is read here and closed over by the wrapper, rather than
        # looked up per call out of a dict the uninstall clears. That dict was
        # how a wrapper still referenced by another library raised `KeyError`
        # into the host after `uninstall()`.
        sock, obj = ssl.SSLSocket, ssl.SSLObject
        self._patches.patch(sock, "send", self._mk_send("send", sock.send))
        self._patches.patch(sock, "recv", self._mk_recv(sock.recv))
        self._patches.patch(sock, "recv_into", self._mk_recv_into(sock.recv_into))
        self._patches.patch(obj, "write", self._mk_send("write", obj.write))
        self._patches.patch(obj, "read", self._mk_read(obj.read))
        self._acquire_timing()
        self._installed = True

    # `uninstall` is the base's: both seams undid the same three things, and the
    # copy here is what let one of them keep a stale `if not self._installed`.

    # --- Subclass hook implementations (SSL-specific) ---

    def _select_tracker(self, obj: Any) -> Any:
        try:
            if obj.selected_alpn_protocol() == "h2":
                return _Http2Tracker(self._native_limits)
        except Exception:
            pass
        return _Http1Tracker(self._native_limits)

    def _gate(self, st: _ConnectionState, data: bytes, phase: str) -> bool:
        """Sniff-latch: classify the connection once, from the first request bytes.

        Without this, every ssl.SSLSocket in the process — a TLS-backed Redis,
        Mongo, or Kafka client included — streams into the HTTP parser and
        accumulates there for the life of the connection (the plaintext seam
        has had this protection since it shipped; this ports it to TLS).

        ALPN is trusted first when present: the handshake already negotiated
        the protocol, so there is nothing to sniff. This matters because
        `send`/`write` may be called again after the h2 connection preface has
        already gone out, so requiring the preface to reappear in every call
        would misclassify a healthy h2 connection. `_select_tracker` already
        picked an _Http2Tracker from ALPN, so checking the tracker type here
        reuses that decision instead of re-deriving it.

        Otherwise the first request bytes decide, mirroring the plaintext seam:
        a response arriving before any request means the peer spoke first (a
        server-first protocol such as Postgres/MySQL over TLS), which cannot
        be classified, so it is ignored.
        """
        if st.gate is None:
            if isinstance(st.tracker, _Http2Tracker):
                st.gate = "h2"
            elif phase != "request":
                st.gate = "ignore"
            elif data.startswith(_H2_PREFACE):
                st.gate = "h2"
            elif data.startswith(_HTTP_METHODS):
                st.gate = "http"
            else:
                st.gate = "ignore"
        return st.gate in ("http", "h2")

    # --- send family (request) ---

    def _mk_send(self, meth: str, real: Any):  # noqa: ANN202
        def wrapper(this: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            ret = real(this, data, *args, **kwargs)
            try:
                # The gate comes FIRST, above the materialization. Materializing
                # is free for an exact `bytes` argument and a full copy of the
                # send buffer for anything else — a memoryview or bytearray,
                # which is what the asyncio and httpx paths hand to `write`.
                # It ran on every send of every SSLSocket in the process,
                # including the ones the sniff-latch had already ruled out, so
                # a TLS-backed Redis or Postgres client paid it for the life of
                # the connection for a verdict settled on its first write.
                if self._capture_possible(this):
                    sent = data
                    if meth in ("send", "write") and isinstance(ret, int):
                        sent = _accepted_prefix(data, ret)
                    self._on_request_bytes(this, bytes(sent))
            except Exception:
                pass
            return ret

        return wrapper

    # --- recv family (response) ---

    def _mk_recv(self, real: Any):  # noqa: ANN202
        def wrapper(this: Any, *args: Any, **kwargs: Any) -> Any:
            ret = real(this, *args, **kwargs)
            try:
                # Ahead of `bytes(ret)`, for the reason `_mk_send` gives.
                if self._capture_possible(this) and isinstance(ret, (bytes, bytearray)) and ret:
                    self._on_response_bytes(this, bytes(ret))
            except Exception:
                pass
            return ret

        return wrapper

    def _mk_recv_into(self, real: Any):  # noqa: ANN202
        def wrapper(this: Any, buffer: Any, *args: Any, **kwargs: Any) -> int:
            n = real(this, buffer, *args, **kwargs)
            try:
                if n and self._capture_possible(this):
                    self._on_response_bytes(this, bytes(buffer[:n]))
            except Exception:
                pass
            return n

        return wrapper

    def _mk_read(self, real: Any):  # noqa: ANN202
        def wrapper(this: Any, *args: Any, **kwargs: Any) -> Any:
            ret = real(this, *args, **kwargs)
            try:
                if self._capture_possible(this):
                    buffer = None
                    if len(args) >= 2:
                        buffer = args[1]
                    elif "buffer" in kwargs:
                        buffer = kwargs["buffer"]
                    if buffer is not None and isinstance(ret, int):
                        if ret:
                            self._on_response_bytes(this, bytes(buffer[:ret]))
                    elif isinstance(ret, (bytes, bytearray)) and ret:
                        self._on_response_bytes(this, bytes(ret))
            except Exception:
                pass
            return ret

        return wrapper

    # --- Connection timing resolution (SSL-specific) ---

    def _resolve_timing(
        self, obj: Any, st: _ConnectionState
    ) -> tuple[float, float, bool, tuple[Limitation, ...]]:
        """(tcp_connect_ms, tls_handshake_ms, connection_reused, limitations)."""
        if st.timing_consumed:
            return (0.0, 0.0, True, ())
        st.timing_consumed = True
        # sync: SSLSocket — look up the store by fileno
        if not isinstance(obj, ssl.SSLObject):
            try:
                popped = shared_timing_store().pop(obj.fileno())
            except Exception:
                popped = None
            if popped is not None:
                return (popped[0], popped[1], False, ())
            return (0.0, 0.0, False, (Limitation.CONNECT_TIMING_UNAVAILABLE,))
        # async: SSLObject — derive total−handshake from the stamped record
        # The anyio path handles TCP connect and TLS in separate layers, so when
        # total_ms is 0 the connect time cannot be derived. To avoid misreporting
        # that 0 as a "fast connection", attach CONNECT_TIMING_UNAVAILABLE.
        #
        # That marker used to be a distinct string, `async_connect_unavailable`.
        # The census (design §6.5.1) merged it: what it lost is the provenance —
        # a sync fileno miss versus this anyio layer split — and both assert the
        # same fact, `tcp_connect_ms` is unknown rather than zero, with the same
        # user action in either case (none).
        rec = getattr(obj, "_wardex_timing", None)
        if rec is not None:
            if rec.total_ms > 0.0:
                # raw-asyncio path: total was measured successfully → connect can be derived
                connect = max(0.0, rec.total_ms - rec.handshake_ms)
                return (connect, rec.handshake_ms, False, ())
            else:
                # anyio/httpx path: total could not be measured → connect=0 + marker
                return (0.0, rec.handshake_ms, False, (Limitation.CONNECT_TIMING_UNAVAILABLE,))
        return (0.0, 0.0, False, (Limitation.CONNECT_TIMING_UNAVAILABLE,))
