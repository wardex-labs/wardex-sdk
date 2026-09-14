"""The peer address a byte seam reports, and what it reports when it has none.

`server.address`/`server.port` and the URL on a seam span come from here. The
honest answer is not always an address: a non-INET socket (`AF_UNIX`, the family
httpx's `uds=`, docker-py and local model servers ride) answers `getpeername()`
with a path, and a socket that is not connected, or a memory-BIO `SSLObject`,
does not answer at all. The traffic is still captured — a local model server
over a unix socket is an LLM call to the user — but its address is a
placeholder: the TLS server name when there is one, else `unknown`, and port
`UNRESOLVED_PORT` (0).

THE WIDEST CASE IS NOT THE UNIX SOCKET. Asyncio and anyio TLS — httpx's
`AsyncClient`, so the async OpenAI and Anthropic clients, and aiohttp — encrypt
through a memory-BIO `SSLObject`, which has a `server_hostname` and no
`getpeername()`. Every call on that path reports port 0 and carries the marker.
It used to report 443: right for most public APIs, a guess all the same, and
wrong for any TLS server on another port. Nothing the seam holds names that
port — the object has no link to its transport, and the Host header is not
kept past the parser — so it is reported as unread rather than guessed.

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
    on an `SSLObject` a raised-and-caught `AttributeError` per write). What can
    still arrive is a `server_hostname`, so while the host is the bare
    `UNRESOLVED_HOST` the seam re-reads that attribute alone
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


def peer_address(obj: Any) -> tuple[str, int]:
    """The connection's peer as `(host, port)`; port `UNRESOLVED_PORT` if unread.

    An address counts as read only when `getpeername()` answers with an INET or
    INET6 tuple whose second item is an integer port. The SHAPE is checked
    rather than indexed, because indexing a unix path is not reliably an
    error: `"/9.sock"[1]` is `"9"`, and the old `int(peer[1])` read that path
    as host `/` on port 9 — a lie that looked resolved.

    `getpeername()` raising or not existing is caught, and it is an answer
    rather than a failure: the caller learns it through the port.
    """
    try:
        peer = obj.getpeername()
    except Exception:
        peer = None
    if isinstance(peer, tuple) and len(peer) >= 2 and isinstance(peer[1], int):
        return str(peer[0]), peer[1]
    return placeholder_host(obj), UNRESOLVED_PORT


def placeholder_host(obj: Any) -> str:
    """The host reported when no address was read: the TLS server name, else
    `UNRESOLVED_HOST`. An attribute read, never a syscall — the one part of the
    placeholder that can change after the connection's first byte."""
    return str(getattr(obj, "server_hostname", None) or UNRESOLVED_HOST)
