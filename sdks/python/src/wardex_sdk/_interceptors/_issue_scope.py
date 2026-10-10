"""The scope identity a wire span is stamped with: the one its work was ISSUED under.

`Client._stamp_scope` folds the scope's tags and user (`set_tag`, `set_user`) into every exported
span. Read when the span is captured, that is the identity of whichever context captures it, and
the interceptors capture far from where the work was issued: a response read by the task that
holds an HTTP/2 connection's read lock, a socket closed or collected on another request's thread
(a WebSocket session's span exists only then), an MCP response read by a reader task that the
first request to open the session started. With one process serving many tenants, each of those
stamped one tenant's user and tags on another tenant's span.

So every producer snapshots the identity where it latches the parent, on the same line, and the
snapshot travels with the transaction to the client (`Client.capture_span(scope=)`, and
`_PendingTxn.scope` for a deferred parse). The snapshot is only as right as that latch: MCP stdio
latches on the task that writes stdin, which with the official `mcp` SDK is a writer task the
session's opener started, so there the identity, like the parent, is still the opener's.

Where the issuer is not known — an HTTP/2 stream whose opener was never proven (`_h2_issuer`), a
latch entry the stream cap dropped, a reply to a request the seam never saw — the span carries
`UNKNOWN_ISSUER`, which stamps nothing. A guess would be some other tenant's identity on this
call; a conversation id is never guessed for the same reason.
"""

from __future__ import annotations

from .. import _hub
from .._assembly import guard
from .._scope import UserInfo

#: `(tags, user)`: the merged tags and user of the scope a span's work was issued in, in the shape
#: `Client._admit(scope=)` stamps.
ScopeSnapshot = tuple[dict[str, str], UserInfo | None]
#: The snapshot of an issuer wardex did not observe: no tags and no user are stamped.
UNKNOWN_ISSUER: ScopeSnapshot = ({}, None)
#: The read runs inside the host's own send path, where a raise would fail the host's request.
_READ = guard("interceptors.issue_scope")


def issued_scope() -> ScopeSnapshot:
    """The calling context's merged tags and user, copied now: call it where the work is issued.

    Copied, not referenced: `Scope` is mutable, and a later `set_user` in the same context must not
    reach a request already sent. A read that raises (the host mutating a tag dict on another
    thread this instant) is counted by the guard and stamps nothing, as the deferred path's own
    snapshot in `Client.capture_deferred` does.
    """
    snapshot = UNKNOWN_ISSUER
    with _READ:
        snapshot = _hub.get_merged_tags_and_user()
    return snapshot


def same_issuer(first: ScopeSnapshot, later: ScopeSnapshot) -> bool:
    """Was `later` issued under the identity `first` names, for work that ships as ONE span (a
    WebSocket session)? The same user id, or neither has one, and the same value for every tag
    `first` carried. A tag added afterwards, or the same user gaining an email, is that identity
    annotating its own work, not another one; a different user id, a tag the first snapshot had
    now changed or gone (another tenant's request, or a thread outside every scope) is."""
    tags, user = first
    later_tags, later_user = later
    if getattr(user, "id", None) != getattr(later_user, "id", None):
        return False
    return all(later_tags.get(key) == value for key, value in tags.items())
