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

from ._conn_timing import install_shared_timing, shared_timing_store, uninstall_shared_timing
from ._seam import (
    ByteSeamInterceptor,
    _build_grpc_fields,  # backward-compat re-export (keeps the existing test import path)
    _ConnectionState,
)
from ._trackers import _Http1Tracker, _Http2Tracker, _WebSocketTracker

if TYPE_CHECKING:
    from .._client import Client

__all__ = ["SSLInterceptor", "_build_grpc_fields"]


class SSLInterceptor(ByteSeamInterceptor):
    """Global monkeypatch interceptor for the ssl module."""

    def name(self) -> str:
        return "ssl"

    # --- Install / uninstall ---

    def install(self, client: Client | None) -> None:
        if self._installed:
            return
        self._client = client
        self._patch(ssl.SSLSocket, "send", self._mk_send("send", "SSLSocket"))
        self._patch(ssl.SSLSocket, "recv", self._mk_recv("recv", "SSLSocket"))
        self._patch(ssl.SSLSocket, "recv_into", self._mk_recv_into())
        self._patch(ssl.SSLObject, "write", self._mk_send("write", "SSLObject"))
        self._patch(ssl.SSLObject, "read", self._mk_read())
        install_shared_timing()
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for key, fn in self._orig.items():
            cls_name, meth = key.split(".", 1)
            cls = ssl.SSLSocket if cls_name == "SSLSocket" else ssl.SSLObject
            setattr(cls, meth, fn)
        self._orig.clear()
        uninstall_shared_timing()
        for st in list(self._conns.values()):
            if isinstance(st.tracker, _WebSocketTracker):
                for txn in st.tracker.flush("ws_no_close"):
                    self._emit_ws(st, txn)
        self._conns.clear()
        self._installed = False

    # --- Subclass hook implementations (SSL-specific) ---

    def _select_tracker(self, obj: Any) -> Any:
        try:
            if obj.selected_alpn_protocol() == "h2":
                return _Http2Tracker()
        except Exception:
            pass
        return _Http1Tracker()

    # --- send family (request) ---

    def _mk_send(self, meth: str, cls_name: str):  # noqa: ANN202
        key = f"{cls_name}.{meth}"

        def wrapper(this: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            real = self._orig[key]
            ret = real(this, data, *args, **kwargs)
            try:
                sent = data
                if meth in ("send", "write") and isinstance(ret, int):
                    sent = bytes(data)[:ret]
                self._on_request_bytes(this, bytes(sent))
            except Exception:
                pass
            return ret

        return wrapper

    # --- recv family (response) ---

    def _mk_recv(self, meth: str, cls_name: str):  # noqa: ANN202
        key = f"{cls_name}.{meth}"

        def wrapper(this: Any, *args: Any, **kwargs: Any) -> Any:
            real = self._orig[key]
            ret = real(this, *args, **kwargs)
            try:
                if isinstance(ret, (bytes, bytearray)) and ret:
                    self._on_response_bytes(this, bytes(ret))
            except Exception:
                pass
            return ret

        return wrapper

    def _mk_recv_into(self):  # noqa: ANN202
        def wrapper(this: Any, buffer: Any, *args: Any, **kwargs: Any) -> int:
            real = self._orig["SSLSocket.recv_into"]
            n = real(this, buffer, *args, **kwargs)
            try:
                if n:
                    self._on_response_bytes(this, bytes(buffer[:n]))
            except Exception:
                pass
            return n

        return wrapper

    def _mk_read(self):  # noqa: ANN202
        def wrapper(this: Any, *args: Any, **kwargs: Any) -> Any:
            real = self._orig["SSLObject.read"]
            ret = real(this, *args, **kwargs)
            try:
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
    ) -> tuple[float, float, bool, tuple[str, ...]]:
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
            return (0.0, 0.0, False, ("connect_timing_unavailable",))
        # async: SSLObject — derive total−handshake from the stamped record
        # The anyio path handles TCP connect and TLS in separate layers, so when
        # total_ms is 0 the connect time cannot be derived. To avoid misreporting
        # that 0 as a "fast connection", attach the async_connect_unavailable marker.
        rec = getattr(obj, "_wardex_timing", None)
        if rec is not None:
            if rec.total_ms > 0.0:
                # raw-asyncio path: total was measured successfully → connect can be derived
                connect = max(0.0, rec.total_ms - rec.handshake_ms)
                return (connect, rec.handshake_ms, False, ())
            else:
                # anyio/httpx path: total could not be measured → connect=0 + marker
                return (0.0, rec.handshake_ms, False, ("async_connect_unavailable",))
        return (0.0, 0.0, False, ("connect_timing_unavailable",))
