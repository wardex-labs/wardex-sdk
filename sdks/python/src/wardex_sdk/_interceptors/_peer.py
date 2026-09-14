"""The peer address a byte seam reports, and what it reports when it has none.

`server.address`/`server.port` and the URL on a seam span come from here. The
honest answer is not always an address: a non-INET socket (`AF_UNIX`, the family
httpx's `uds=`, docker-py and local model servers ride) answers `getpeername()`
with a path, and a socket that is not connected, or a memory-BIO `SSLObject`,
does not answer at all. The traffic is still captured — a local model server
over a unix socket is an LLM call to the user — but its address is a
placeholder: the TLS server name when there is one, else `unknown`, and port
`UNRESOLVED_PORT` (0).

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
  * COUNTED PER SEALED TRANSACTION, not per connection. Both counted sites sit
    above the capture gate, so a span the mode later refuses still counts. The
    fallback itself runs when the seam first builds a connection's state, but
    every unix socket in the process reaches that point on its first byte —
    asyncio's self-pipe is a `socketpair()` — so a count there would tally
    event loops rather than HTTP traffic wardex could not address.
  * ASKED ONCE. A peer address that was READ cannot change while the
    connection lives, so `_on_request_bytes` does not re-ask per send (it used
    to: a syscall per write, and on an `SSLObject` a raised-and-caught
    `AttributeError` per write). It re-asks only while the host is the bare
    `UNRESOLVED_HOST`, where a later `server_hostname` can still name the host;
    a better host name leaves the port at 0, so the mark stays.
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
    return str(getattr(obj, "server_hostname", None) or UNRESOLVED_HOST), UNRESOLVED_PORT
