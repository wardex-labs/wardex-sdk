"""Plaintext raw-socket interceptor — monkeypatches socket.socket.

Tracks only HTTP via a method sniff-latch, hard-excludes link-local addresses,
and emits LLM-only (+ allowlist). Reuses the existing _Http1Tracker/_WebSocketTracker.
h2c and uvloop async are not supported.
SSLSocket is a subclass of socket.socket but implements its own send/recv, so
this patch does not double-capture TLS application data (regression-safe).
"""

from __future__ import annotations

import ipaddress
import socket
from typing import TYPE_CHECKING, Any

from .._enums import CaptureSource
from ._conn_timing import install_shared_timing, shared_timing_store, uninstall_shared_timing
from ._seam import ByteSeamInterceptor, _ConnectionState, _has_core_semantics
from ._trackers import _Http1Tracker, _Http2Tracker

if TYPE_CHECKING:
    from .._client import Client

_HTTP_METHODS = (b"GET ", b"POST ", b"PUT ", b"DELETE ", b"HEAD ", b"PATCH ", b"OPTIONS ")

# HTTP/2 connection preface (prior-knowledge h2c). TLS h2 sends the same bytes.
_H2_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"


def _is_link_local(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr).is_link_local
    except ValueError:
        return False


class RawSocketInterceptor(ByteSeamInterceptor):
    """Plaintext socket.socket patch interceptor — captures only non-TLS HTTP traffic."""

    def __init__(self, intercept_hosts: list[str] | None = None) -> None:
        super().__init__()
        self._allow: set[str] = set(intercept_hosts or ())

    def name(self) -> str:
        return "socket"

    def install(self, client: Client | None) -> None:
        if self._installed:
            return
        self._client = client
        # base _patch uses the key f"{cls.__name__}.{meth}".
        # socket.socket.__name__ == "socket" → keys become "socket.send", etc.
        self._patch(socket.socket, "send", self._mk_send("send"))
        self._patch(socket.socket, "sendall", self._mk_sendall())
        self._patch(socket.socket, "recv", self._mk_recv("recv"))
        self._patch(socket.socket, "recv_into", self._mk_recv_into())
        install_shared_timing()
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        for key, fn in self._orig.items():
            _, meth = key.split(".", 1)
            setattr(socket.socket, meth, fn)
        self._orig.clear()
        uninstall_shared_timing()
        from ._trackers import _WebSocketTracker

        for st in list(self._conns.values()):
            if isinstance(st.tracker, _WebSocketTracker):
                for txn in st.tracker.flush("ws_no_close"):
                    self._emit_ws(st, txn)
        self._conns.clear()
        self._installed = False

    # --- Subclass hooks ---

    def _capture_source(self) -> CaptureSource:
        return CaptureSource.SOCKET

    def _select_tracker(self, obj: Any) -> Any:
        # Initial value is h1. When an h2c preface is detected, _gate SWAPs it to _Http2Tracker.
        return _Http1Tracker()

    def _url_scheme(self, is_ws: bool) -> str:
        return "ws" if is_ws else "http"

    def _resolve_timing(
        self, obj: Any, st: _ConnectionState
    ) -> tuple[float, float, bool, tuple[str, ...]]:
        if st.timing_consumed:
            return (0.0, 0.0, True, ())
        st.timing_consumed = True
        try:
            popped = shared_timing_store().pop(obj.fileno())
        except Exception:
            popped = None
        if popped is not None:
            return (popped[0], 0.0, False, ())  # plaintext: no TLS handshake
        return (0.0, 0.0, False, ("connect_timing_unavailable",))

    def _gate(self, st: _ConnectionState, data: bytes, phase: str) -> bool:
        """Sniff-latch: determine the protocol from the first request bytes; never re-decided.

        HTTP/1 method → "http", h2c connection preface → "h2c" (tracker SWAP),
        anything else (TLS records, redis, etc.) → "ignore". h2c is byte-identical
        to TLS h2, so once the preface is detected and the tracker is swapped to
        _Http2Tracker, the rest of the pipeline works unchanged.
        """
        if st.gate is None:
            if phase != "request":
                st.gate = "ignore"  # response arrives before any request → cannot decide
            elif _is_link_local(st.server_address):
                st.gate = "ignore"  # link-local (metadata endpoint) — skip parsing
            elif data.startswith(_HTTP_METHODS):
                st.gate = "http"
            elif data.startswith(_H2_PREFACE):
                st.gate = "h2c"  # plaintext HTTP/2 (prior-knowledge)
                st.tracker = _Http2Tracker()  # swap the h1 tracker for h2
            else:
                st.gate = "ignore"  # non-HTTP protocols such as Redis, Memcached, etc.
        return st.gate in ("http", "h2c")

    def _should_capture(self, st: _ConnectionState, txn: Any, sem: Any) -> bool:
        # Deliberate decision: an explicit `intercept_hosts` allowlist match bypasses
        # the 4c `capture_mode` policy gate (which normally requires an active local
        # span for non-LLM-semantic traffic — see _seam.py._emit_span). The user
        # naming a plaintext host here is a stronger, more specific opt-in than the
        # global capture_mode default; composing both gates would silently drop
        # traffic the user explicitly asked to capture. This only applies to hosts
        # the user listed by hand — it does not widen capture_mode=AGENT for anyone
        # else.
        if _is_link_local(st.server_address):
            return False
        if sem is not None and _has_core_semantics(sem):
            return True
        return self._in_allow(st)  # the allowlist is populated in Task 4

    def _in_allow(self, st: _ConnectionState) -> bool:
        if not self._allow:
            return False
        return (
            st.server_address in self._allow
            or f"{st.server_address}:{st.server_port}" in self._allow
        )

    # --- socket.socket wrappers (base_patch key = "socket.<meth>") ---

    def _mk_send(self, meth: str):  # noqa: ANN202
        key = f"socket.{meth}"

        def wrapper(this: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            real = self._orig[key]
            ret = real(this, data, *args, **kwargs)
            try:
                sent = bytes(data)[:ret] if isinstance(ret, int) else data
                self._on_request_bytes(this, bytes(sent))
            except Exception:
                pass
            return ret

        return wrapper

    def _mk_sendall(self):  # noqa: ANN202
        def wrapper(this: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            real = self._orig["socket.sendall"]
            ret = real(this, data, *args, **kwargs)
            try:
                self._on_request_bytes(this, bytes(data))
            except Exception:
                pass
            return ret

        return wrapper

    def _mk_recv(self, meth: str):  # noqa: ANN202
        key = f"socket.{meth}"

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
            real = self._orig["socket.recv_into"]
            n = real(this, buffer, *args, **kwargs)
            try:
                if n:
                    self._on_response_bytes(this, bytes(buffer[:n]))
            except Exception:
                pass
            return n

        return wrapper
