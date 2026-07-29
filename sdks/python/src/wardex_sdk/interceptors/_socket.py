"""Plaintext raw-socket interceptor — monkeypatches socket.socket.

Tracks only HTTP via a method sniff-latch. What it captures is not this seam's
decision: it contributes a `Prefilter` about the CONNECTION — link-local
addresses (cloud metadata) are hard-excluded whatever the mode says, an
explicit `intercept_hosts` match bypasses the mode — and everything else defers
to the one shared policy in `assembly._policy`, the same rule the TLS seam
answers to. Under the `agent` default that is LLM-semantic traffic plus
anything issued inside a live local wardex span; under `all` it is everything.
Reuses the existing _Http1Tracker/_WebSocketTracker.
h2c and uvloop async are not supported.
SSLSocket is a subclass of socket.socket but implements its own send/recv, so
this patch does not double-capture TLS application data (regression-safe).
"""

from __future__ import annotations

import ipaddress
import socket
from typing import TYPE_CHECKING, Any

from .._enums import CaptureSource
from ..assembly import Prefilter
from ._conn_timing import install_shared_timing, shared_timing_store, uninstall_shared_timing
from ._seam import ByteSeamInterceptor, _ConnectionState
from ._trackers import _Http1Tracker, _Http2Tracker

if TYPE_CHECKING:
    from .._client import Client

_HTTP_METHODS = (
    b"GET ",
    b"POST ",
    b"PUT ",
    b"DELETE ",
    b"HEAD ",
    b"PATCH ",
    b"OPTIONS ",
    b"CONNECT ",  # proxied connections open with this
    b"TRACE ",
)

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
        self._load_limits(client)
        # base _patch uses the key f"{cls.__name__}.{meth}".
        # socket.socket.__name__ == "socket" → keys become "socket.send", etc.
        self._patch(socket.socket, "send", self._mk_send("send"))
        self._patch(socket.socket, "sendall", self._mk_sendall())
        self._patch(socket.socket, "recv", self._mk_recv("recv"))
        self._patch(socket.socket, "recv_into", self._mk_recv_into())
        install_shared_timing(self._limits["max_connections"])
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
        return _Http1Tracker(self._native_limits)

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
                st.tracker = _Http2Tracker(self._native_limits)  # swap the h1 tracker for h2
            else:
                st.gate = "ignore"  # non-HTTP protocols such as Redis, Memcached, etc.
        return st.gate in ("http", "h2c")

    def _transport_prefilter(self, st: _ConnectionState) -> Prefilter:
        """What this seam knows about the PEER, and nothing beyond it.

        Two opinions, both about the connection rather than the traffic on it:

        DENY — link-local (cloud metadata endpoints). Never captured, whatever
        the mode says, because the bodies carry instance credentials.

        ALLOW — an explicit `intercept_hosts` match. The user naming a
        plaintext host by hand is a stronger, more specific opt-in than the
        global `capture_mode` default, so it stays a bypass rather than
        composing; composing it would silently drop traffic the user asked for
        by name. This only ever applies to hosts listed by hand.

        Everything else DEFERs, and that is the change. This method used to be
        `_should_capture`, overriding the base outright, which meant the
        plaintext seam re-implemented the LLM-semantics clause (fine) and
        silently dropped BOTH of the other two: `capture_mode=ALL` did nothing
        here, and a plaintext request issued inside a live wardex span was
        dropped while the identical request over TLS was captured. Deferring
        gives the shared policy back both clauses without giving up either
        opinion above.
        """
        if _is_link_local(st.server_address):
            return Prefilter.DENY
        if self._in_allow(st):
            return Prefilter.ALLOW
        return Prefilter.DEFER

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
