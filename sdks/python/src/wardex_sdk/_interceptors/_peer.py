"""The peer address a byte seam reports, and what it reports when it has none.

`server.address`/`server.port` and the URL on a seam span come from here.
These answer with a READ address, and each is the TCP peer:

  * A connected INET socket's own `getpeername()` — every sync client, TLS or
    not, since an `ssl.SSLSocket` is a socket.
  * What a memory-BIO `ssl.SSLObject` rides on. The object has no
    `getpeername()`, but at the moment it is set up something that holds both
    it and the socket is in hand, and the close probe
    (`_close_hook.CloseProbe`) reads the peer there and stamps it onto the
    object as `TRANSPORT_PEER` — the way `_conn_timing` stamps
    `_wardex_timing` — for `peer_address` to prefer:
      - asyncio TLS (aiohttp, `loop.create_connection(ssl=...)`,
        `asyncio.open_connection`): `SSLProtocol.connection_made` is handed
        the raw socket transport, whose `peername` extra is the socket's
        `getpeername()`, while the protocol already owns the object;
      - anyio TLS (httpx's `AsyncClient`, and so the async OpenAI and
        Anthropic clients): `TLSStream.wrap` returns a stream whose public
        typed attributes name the object and the socket's remote address.
    Under TLS inside TLS (an HTTPS proxy) that is the proxy, exactly as a sync
    socket's `getpeername()` would say.

Everything else is a placeholder: the TLS server name when there is one, else
`unknown`, and port `UNRESOLVED_PORT` (0). The traffic is still captured — a
local model server over a unix socket is an LLM call to the user. The cases:

  * `AF_UNIX` (httpx's `uds=`, docker-py, local model servers): `getpeername()`
    answers with a path.
  * A socket that is not connected: `getpeername()` raises.
  * uvloop TLS. uvloop's `SSLProtocol` is its own Cython class, not
    `asyncio.sslproto.SSLProtocol`, so the patch that reads the transport
    never runs. (anyio TLS over uvloop still goes through `TLSStream.wrap`,
    and is read.)
  * An async TLS connection set up before wardex was installed — one a pool
    opened ahead of `init()`. The moment the peer is read has passed, and
    nothing later is taken as a stand-in for it.
  * anyio without its `abc`/`streams.tls` modules importable, or a stream
    that does not publish both attributes: nothing names the peer.

The seam's contract with that placeholder, stated once here so each call site
in `_seam.py` can stay one line:

  * PORT 0 IS THE MARK. No connected INET socket has peer port 0, so the port
    alone says "not read", and a seam that sees it attaches
    `Limitation.PEER_UNRESOLVED` and bumps `interceptors.seam.peer_unresolved`.
    Port 443 — what this fallback used to invent — shipped
    `http://unknown:443/...` with no marker, indistinguishable from a real TLS
    connection to a host reading its own spans.
  * THE URL RENDERS `:0`. Omitting the port is not neutral: a URL with no port
    reads as the scheme's default, which is the same invention again. `:0`
    agrees with `server.port`.
  * COUNTED PER UNIT THAT COULD BECOME A SPAN, never at the fallback. HTTP
    counts once per sealed transaction, in `_seal`, above the excluded-path
    return and the capture gate. A WebSocket session is one span built at
    close and never sealed, so it counts once per session, in
    `_build_ws_span`, above the same gate. Either way a unit that never
    becomes a span (refused by the mode, a telemetry upload) still counts. The
    fallback itself runs when the seam first builds a connection's state, but
    every unix socket in the process reaches that point on its first byte —
    asyncio's self-pipe is a `socketpair()` — so a count there would tally
    event loops rather than HTTP traffic wardex could not address.
  * ASKED ONCE. A connected socket's peer cannot change while the connection
    lives, whether `getpeername()` answered or not, so `_on_request_bytes`
    never calls it again (it used to on every send: a syscall per write, and
    on an unstamped `SSLObject` a raised-and-caught `AttributeError` per
    write). What can still arrive is a `server_hostname`, so while the host is
    the bare `UNRESOLVED_HOST` the seam re-reads that attribute alone
    (`placeholder_host`). A better host name leaves the port at 0, so the mark
    stays.
"""

from __future__ import annotations

from typing import Any

#: The host reported when neither the socket nor a TLS server name names the
#: peer — the one placeholder `_on_request_bytes` asks about again.
UNRESOLVED_HOST = "unknown"

#: The port reported whenever no INET address was read. See the module
#: docstring: it is the mark the seam keys `PEER_UNRESOLVED` on.
UNRESOLVED_PORT = 0

#: The attribute an `ssl.SSLObject` carries its transport's INET peer under,
#: named after `_conn_timing`'s `_wardex_timing`. Only `stamp_peer` writes it,
#: and only with a tuple that already passed `_is_inet`.
TRANSPORT_PEER = "_wardex_peer"


def _is_inet(peer: Any) -> bool:
    """Is `peer` an INET/INET6 `(host, port, ...)` tuple with an integer port?

    The SHAPE is checked rather than indexed, because indexing a unix path is
    not reliably an error: `"/9.sock"[1]` is `"9"`, and an old `int(peer[1])`
    read that path as host `/` on port 9 — a lie that looked resolved.
    """
    return isinstance(peer, tuple) and len(peer) >= 2 and isinstance(peer[1], int)


def peer_address(obj: Any) -> tuple[str, int]:
    """The connection's peer as `(host, port)`; port `UNRESOLVED_PORT` if unread.

    An address counts as read when it is an INET or INET6 tuple (`_is_inet`)
    from either source the module docstring names: the `TRANSPORT_PEER` stamp,
    asked first because an `SSLObject` that carries one has no `getpeername()`
    to ask, then `getpeername()` itself. A stamp of any other shape is not a
    read address and is passed over, so a test double whose attributes all
    answer still reaches its `getpeername()`.

    `getpeername()` raising or not existing is caught, and it is an answer
    rather than a failure: the caller learns it through the port.
    """
    try:
        peer = getattr(obj, TRANSPORT_PEER, None)
        if not _is_inet(peer):
            peer = obj.getpeername()
    except Exception:
        peer = None
    if _is_inet(peer):
        return str(peer[0]), peer[1]
    return placeholder_host(obj), UNRESOLVED_PORT


def stamp_peer(sslobj: Any, peer: Any) -> None:
    """Carry an INET `peer` to the seam on `sslobj`; anything else, drop.

    Called by the close probe once asyncio or anyio has named both halves.
    `sslobj` is None when nothing named an object, and a `peer` that is None
    (no socket under the transport) or not an INET tuple (a unix-socket
    transport's is a path) is an answer too: the object stays unstamped, so
    its spans keep port 0 and the marker. Anything that RAISES while the
    caller READ those halves is the caller's to count.
    """
    if sslobj is not None and _is_inet(peer):
        setattr(sslobj, TRANSPORT_PEER, peer)


def placeholder_host(obj: Any) -> str:
    """The host reported when no address was read: the TLS server name, else
    `UNRESOLVED_HOST`. An attribute read, never a syscall — the one part of the
    placeholder that can change after the connection's first byte."""
    return str(getattr(obj, "server_hostname", None) or UNRESOLVED_HOST)
