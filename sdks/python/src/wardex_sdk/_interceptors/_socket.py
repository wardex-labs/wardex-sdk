"""Plaintext raw-socket interceptor — monkeypatches socket.socket.

Tracks only HTTP via a method sniff-latch. What it captures is not this seam's
decision: it contributes a `Prefilter` about the CONNECTION — link-local
addresses (cloud metadata) are hard-excluded whatever the mode says, an
explicit `intercept_hosts` match bypasses the mode — and everything else defers
to the one shared policy in `assembly._policy`, the same rule the TLS seam
answers to. Under the `agent` default that is LLM-semantic traffic plus
anything issued inside a live local wardex span; under `all` it is everything.
Reuses the existing _Http1Tracker/_WebSocketTracker, and _Http2Tracker for h2c.

What it sees is what passes through a `socket.socket` method in Python: `send`,
`sendall`, `sendmsg` and `sendto` on the way out, `recv` and `recv_into` on the
way in. That covers synchronous clients and asyncio's selector event loop, whose
plaintext writer uses `send` for the first attempt of a `write()` and, from
Python 3.12, `sendmsg` for everything after it (and for every `writelines()`).
Bytes written below those methods are not seen: `os.sendfile` (what
`loop.sendfile`/`sock_sendfile` and `socket.sendfile` use where the OS has it),
`os.write` on the descriptor, uvloop (libuv writes and reads the descriptor
itself) and the Windows proactor loop (overlapped `WSASend`/`WSARecv`). A
response that arrives for a request the tracker did not see whole is counted,
never paired with another request — see `_Http1Tracker`.
SSLSocket is a subclass of socket.socket but implements its own send/recv, and
refuses `sendmsg` (and `sendto` once its TLS layer exists), so this patch does
not double-capture TLS application data (regression-safe).
"""

from __future__ import annotations

import ipaddress
import socket
from typing import TYPE_CHECKING, Any

from .._assembly import Limitation, Prefilter
from .._enums import CaptureSource
from .._protocol import REQUEST_METHODS as _HTTP_METHODS
from ._conn_timing import shared_timing_store
from ._seam import ByteSeamInterceptor, _accepted_prefix, _ConnectionState
from ._trackers import _Http1Tracker, _Http2Tracker

if TYPE_CHECKING:
    from .._client import Client

# HTTP/2 connection preface (prior-knowledge h2c). TLS h2 sends the same bytes.
_H2_PREFACE = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n"


def _accepted_prefix_vectored(buffers: list[Any], n: int) -> bytes:
    """The first `n` bytes of a `sendmsg` call: its buffers joined in order.

    `sendmsg` returns how many bytes the kernel took ACROSS all the buffers, and
    like `send` it may take fewer than were offered — asyncio passes its whole
    write queue and keeps the remainder for the next call. Only the accepted
    bytes may reach a tracker: the rest is offered again, and feeding it twice
    would put the same body bytes into the request twice.

    Each buffer is sliced, not copied whole, for the reason `_accepted_prefix`
    gives; `sendmsg` already refused anything that is not a contiguous buffer,
    so the `cast("B")` cannot meet a shape it does not take.
    """
    parts: list[Any] = []
    left = n
    for buf in buffers:
        if left <= 0:
            break
        view = memoryview(buf).cast("B")
        parts.append(view[:left])
        left -= len(view)
    return b"".join(parts)


def _is_link_local(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr).is_link_local
    except ValueError:
        return False


class RawSocketInterceptor(ByteSeamInterceptor):
    """Plaintext socket.socket patch interceptor — captures only non-TLS HTTP traffic."""

    def __init__(self, intercept_hosts: list[str] | None = None) -> None:
        super().__init__()
        # Case-folded on the way in, and matched case-folded below. What the
        # allowlist is compared against is usually a peer IP, where case cannot
        # differ — but `peer_address()` falls back to the connection's
        # `server_hostname` when `getpeername()` fails, and that string is
        # whatever the caller passed to connect. A user who wrote `MyBox.local`
        # in the config and a connection wardex names `mybox.local` are the same
        # host; hostnames are case-insensitive and an exact-string set said
        # otherwise. Folding once at construction keeps the per-connection check
        # the single set lookup it has to be.
        self._allow: set[str] = {h.lower() for h in intercept_hosts or ()}

    def name(self) -> str:
        return "socket"

    def install(self, client: Client | None) -> None:
        if self._installed:
            return
        self._client = client
        self._load_limits(client)
        self._patches = self._fresh_patchset()
        # Each wrapper closes over the original it replaces, rather than looking
        # it up per call in a dict the uninstall clears — that dict is how a
        # wrapper another library still holds raised `KeyError` into the host
        # after `uninstall()`.
        sock = socket.socket
        self._patches.patch(sock, "send", self._mk_send(sock.send))
        self._patches.patch(sock, "sendall", self._mk_sendall(sock.sendall))
        # Python 3.12+'s asyncio plaintext writer sends everything after a
        # `write()`'s first attempt through `sendmsg`; without this patch the
        # tail of every request larger than one `send` went by unseen.
        # Absent on Windows, where `socket.socket` has no `sendmsg`.
        if hasattr(sock, "sendmsg"):
            self._patches.patch(sock, "sendmsg", self._mk_sendmsg(sock.sendmsg))
        self._patches.patch(sock, "sendto", self._mk_sendto(sock.sendto))
        self._patches.patch(sock, "recv", self._mk_recv(sock.recv))
        self._patches.patch(sock, "recv_into", self._mk_recv_into(sock.recv_into))
        self._acquire_probes()
        self._installed = True

    # `uninstall` is the base's — see `ByteSeamInterceptor.uninstall`.

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
    ) -> tuple[float | None, float | None, bool | None, tuple[Limitation, ...]]:
        # Plaintext: there is no TLS handshake to time, so that interval is
        # always None — unset on the wire, never a 0 ms handshake.
        if st.timing_consumed:
            return (0.0, None, True, ())
        st.timing_consumed = True
        try:
            popped = shared_timing_store().pop(obj.fileno())
        except Exception:
            popped = None
        if popped is not None:
            # A record proves the seam saw this connection open, so this is its
            # first transaction. Its connect half is None when the connect was
            # not timed: a non-blocking connect returns before the handshake.
            if popped[0] is None:
                return (None, None, False, (Limitation.CONNECT_TIMING_UNAVAILABLE,))
            return (popped[0], None, False, ())
        # No connect record: the seam did not see this connection open — it
        # may have been opened, and used, before `init` — so whether this is
        # its first transaction is as unknown as how long the connect took.
        return (None, None, None, (Limitation.CONNECT_TIMING_UNAVAILABLE,))

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
        address = st.server_address.lower()
        return address in self._allow or f"{address}:{st.server_port}" in self._allow

    # --- socket.socket wrappers ---

    def _mk_send(self, real: Any):  # noqa: ANN202
        def wrapper(this: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            ret = real(this, data, *args, **kwargs)
            try:
                # Above the copies — see `_ssl._mk_send`. This patch sits on
                # `socket.socket`, so the buffers it would materialize belong to
                # every plaintext client in the process, HTTP or not.
                if self._capture_possible(this):
                    sent = _accepted_prefix(data, ret) if isinstance(ret, int) else data
                    self._on_request_bytes(this, bytes(sent))
            except Exception:
                pass
            return ret

        return wrapper

    def _mk_sendall(self, real: Any):  # noqa: ANN202
        def wrapper(this: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            ret = real(this, data, *args, **kwargs)
            try:
                if self._capture_possible(this):
                    self._on_request_bytes(this, bytes(data))
            except Exception:
                pass
            return ret

        return wrapper

    def _mk_sendmsg(self, real: Any):  # noqa: ANN202
        # One guard per wrapper, built here where the client's debug setting is
        # known: entering it is the whole per-call cost (see `assembly.guard`).
        feed = self._guard("interceptors.socket.sendmsg")

        def wrapper(this: Any, buffers: Any, *args: Any, **kwargs: Any) -> Any:
            # Ancillary data, flags and an address are not HTTP bytes: they pass
            # through untouched and are never fed.
            capture = False
            with feed:
                capture = hasattr(buffers, "__iter__") and self._capture_possible(this)
            if not capture:
                return real(this, buffers, *args, **kwargs)
            # Read twice — by the call and by the feed — and asyncio hands over a
            # one-shot `itertools.islice`. A list of the same objects is what
            # `sendmsg` builds from it anyway, so the call is unchanged, and an
            # iterator that raises raises here as it would have inside `sendmsg`.
            views = list(buffers)
            ret = real(this, views, *args, **kwargs)
            with feed:
                if isinstance(ret, int) and ret > 0:
                    self._on_request_bytes(this, _accepted_prefix_vectored(views, ret))
            return ret

        return wrapper

    def _mk_sendto(self, real: Any):  # noqa: ANN202
        feed = self._guard("interceptors.socket.sendto")

        def wrapper(this: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            ret = real(this, data, *args, **kwargs)
            with feed:
                if isinstance(ret, int) and self._capture_possible(this):
                    self._on_request_bytes(this, bytes(_accepted_prefix(data, ret)))
            return ret

        return wrapper

    def _mk_recv(self, real: Any):  # noqa: ANN202
        def wrapper(this: Any, *args: Any, **kwargs: Any) -> Any:
            ret = real(this, *args, **kwargs)
            try:
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
