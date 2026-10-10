"""Which task issued each HTTP/2 stream, read where the stream is opened.

An HTTP/2 connection is shared, and its bytes are not written by the task that
issued them. httpcore (httpx's transport, so the OpenAI and Anthropic clients
built with `http2=True`) queues a request's frames on the connection's
`h2.connection.H2Connection` from the issuing task, and then whichever task next
holds the connection's write lock flushes EVERY queued frame, its own and the
other tasks'. Under `asyncio.gather` over one connection, the END_STREAM that
opens a stream in the h2 tracker was routinely written by a task that issued a
different stream. A scope read at the write latched one conversation's id, and
its parent, onto another conversation's LLM call: seven calls of eight swapped,
and the per-conversation token sums came out as zero and double.

The one place the issuer is still the caller is the state machine's own
`send_headers`, where the stream is opened: httpcore, urllib3's HTTP/2 and a
hand-driven `h2` all open a stream from the call that issues the request. So the
issuer is read THERE, into a table that belongs to that state machine, and the
tracker of the connection those frames are written to reads it. Linking the two
is the part that has to be proven rather than assumed, and it is proven once per
connection, by object identity:

* `data_to_send` returns the bytes the client is about to write. Until its state
  machine is linked, the latest such chunk is OFFERED: kept on the state
  machine's table and indexed by its object id on the calling thread.
* An h2 tracker that is not linked yet looks up every chunk it is handed in that
  index, and links only if the chunk IS the offered object (`is`, not `==`): the
  client wrote this state machine's output to this connection.

The index holds every pending offer on the thread, not just the last one,
because the write is not always the next thing the thread does. anyio's
plaintext `SocketStream.send` yields to the event loop before it writes, so with
several HTTP/2 connections opening under one `asyncio.gather` every client's
preface is offered before any of them is written. With one slot, each offer
replaced the last, the first write found another connection's chunk, and the
connections it did not link named no conversation for their first calls. An
entry is only ever a pointer: the proof is still that the written chunk is the
very object its state machine handed out.

The issuer is recorded whether or not the state machine is linked yet, and
BEFORE the original `send_headers` queues the frame, so no thread can flush the
frame first. A stream opened before the link (capture attached mid-connection,
or a first chunk that was not written as handed out) is therefore still proven
by the link a later chunk makes, as long as its request half ends after that.

A stream whose issuer was not read here is UNPROVEN, and names no conversation:
the state machine was never linked (a client on another HTTP/2 library, every
chunk copied before it was written) or the stream was opened before capture,
and the writer's scope is a guess on a shared connection. A conversation is
never guessed, and a request body's own id does not stand in for the unknown
one either (`_semantics.apply_request_conversation`): the host's, had it been
read, would have won. Nor is the scope identity guessed: no tag and no user
of the writer's scope is stamped on the stream's span (`_issue_scope`). Its
parent is latched as it was before this module existed.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import weakref
from typing import Any

from .. import _hub
from .._assembly import PatchSet, guard, parent_is_closed_unit
from .._types import ConversationContext, SpanContext
from ._issue_scope import UNKNOWN_ISSUER, ScopeSnapshot, issued_scope

#: What the issuing task's scope said when it opened a stream: its active span,
#: whether that span's unit had already closed, its conversation, and the tags
#: and user its span is stamped with (`_issue_scope`).
Issued = tuple[SpanContext | None, bool, ConversationContext | None, ScopeSnapshot]

#: The sizes of chunk offered for linking. An offer holds its bytes until the
#: state machine is linked or offers again, so a connection no tracker ever
#: proves (one wardex does not capture) keeps its latest small chunk for as long
#: as it lives. Linking needs one chunk only, and a small one comes early: the
#: connection preface a client writes first, the HEADERS httpcore writes apart
#: from the body, any later control frame. The floor is a frame header, which
#: also keeps out the empty and one-byte `bytes` objects CPython shares between
#: callers, for which `is` would prove nothing.
_OFFER_MIN, _OFFER_MAX = 9, 1 << 12
#: Streams recorded before a tracker proves the link (and so before it says its
#: own `max_streams`): the few opened between capture attaching mid-connection
#: and the first chunk written after it. Small, because a connection that wardex
#: never captures records every stream it opens and no tracker ever takes one.
_UNLINKED_CAP = 64
#: How many pending offers one thread indexes: one per connection whose chunk
#: has been handed out and not yet written. Oldest first out. An entry is an int
#: and a weak reference; losing one costs that connection the link on that
#: chunk, and the next chunk it offers tries again.
_INDEX_CAP = 1024
#: Per thread, `id(chunk) -> weak reference to the StreamIssuers that offered
#: it`. Per thread so the index never needs a lock: the offer and the write of
#: one chunk happen on one thread in every client this links.
_index = threading.local()
#: State machine -> its issuers. Weak at the key: the table dies with the host's
#: connection, and wardex keeps no connection alive.
_issuers_of: weakref.WeakKeyDictionary[Any, StreamIssuers] = weakref.WeakKeyDictionary()

#: Importing `h2` once it has been FOUND, as `_close_hook` imports anyio: its
#: absence is an answer, and present but unimportable is worth a count.
_IMPORT_H2 = guard("interceptors.h2_issuer.import")
#: The two wrappers run inside the host's HTTP/2 client. A raise here would fail
#: a request over a span attribute.
_RECORD = guard("interceptors.h2_issuer.record")
_OFFER = guard("interceptors.h2_issuer.offer")


class StreamIssuers:
    """One h2 state machine's issuers, by stream id, until each stream opens in
    a tracker; and the chunk it offers until a tracker proves the link.

    Bounded like the tracker's latch, lowest stream id first: an entry whose
    stream never ends its request half (reset before END_STREAM, or written
    before the link was proven) would otherwise stay for the life of the
    connection. Losing one costs that stream its proof, so it names no
    conversation, never the wrong one.
    """

    __slots__ = ("__weakref__", "_by_stream", "_cap", "_link", "_offered")

    def __init__(self, cap: int) -> None:
        self._by_stream: dict[int, Issued] = {}
        self._cap = cap
        #: The latest chunk `data_to_send` returned while unlinked.
        self._offered: bytes | None = None
        #: The tracker's end of the link, once a chunk proved it. Weak: a tracker
        #: that died (its connection retired, or reset by a fork) reads as
        #: unlinked, so the next chunk this state machine offers links its successor.
        self._link: weakref.ref[IssuerLink] | None = None

    @property
    def linked(self) -> bool:
        return self._link is not None and self._link() is not None

    def record(self, stream_id: int) -> bool:
        """The CALLER's scope, as the issuer of `stream_id`. Only the first call
        for a stream counts: a later `send_headers` on it sends trailers. True if
        this call recorded it."""
        if stream_id in self._by_stream:
            return False
        scope = _hub.get_current_scope()  # ONE read: parent and conversation are one fact
        parent = scope.active_span_context
        closed, identity = parent_is_closed_unit(parent), issued_scope()
        self._by_stream[stream_id] = (parent, closed, scope.conversation, identity)
        while len(self._by_stream) > self._cap:
            self._by_stream.pop(min(self._by_stream))
        return True

    def forget(self, stream_id: int) -> None:
        self._by_stream.pop(stream_id, None)

    def take(self, stream_id: int) -> Issued | None:
        return self._by_stream.pop(stream_id, None)

    def clear(self) -> None:
        self._by_stream.clear()

    def offer(self, chunk: bytes) -> None:
        self._offered = chunk
        index: dict[int, weakref.ref[StreamIssuers]] | None = getattr(_index, "by_id", None)
        if index is None:
            index = _index.by_id = {}
        index.pop(id(chunk), None)  # re-inserted at the young end
        index[id(chunk)] = weakref.ref(self)
        while len(index) > _INDEX_CAP:
            del index[next(iter(index))]


def claim(data: bytes, link: IssuerLink, cap: int) -> StreamIssuers | None:
    """The issuers of the state machine whose offered chunk `data` is, linked to
    `link`; or None, when `data` is no pending offer on this thread."""
    index = getattr(_index, "by_id", None)
    if not index:
        return None
    ref = index.pop(id(data), None)
    issuers = ref() if ref is not None else None
    # The id is only where to look. The proof is that the offer IS this object:
    # an entry left by an older chunk, whose id a new object now reuses, fails here.
    if issuers is None or issuers._offered is not data:
        return None
    issuers._offered = None
    issuers._link = weakref.ref(link)
    issuers._cap = cap  # the tracker's own `max_streams` from here on
    return issuers


class IssuerLink:
    """An h2 tracker's end of the link: the issuers of the state machine its
    connection's bytes are proven to come from, once a chunk proves it."""

    __slots__ = ("__weakref__", "_cap", "_issuers")

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._issuers: StreamIssuers | None = None

    def see(self, data: bytes) -> None:
        """Every request chunk the tracker is handed, until one proves the link."""
        if self._issuers is None:
            self._issuers = claim(data, self, self._cap)

    def latch(
        self, stream_id: int, parent: SpanContext | None, parent_closed: bool
    ) -> tuple[SpanContext | None, bool, ConversationContext | None, ScopeSnapshot, bool]:
        """`(parent, parent_closed, conversation, scope, proven)` for a stream that
        just opened. Its issuer's, where one was proven. Otherwise the WRITER's
        parent, which on a shared connection may be any task's, and so never
        anyone's conversation or identity: none is named, no tag or user is
        stamped (`UNKNOWN_ISSUER`), and `proven` says the issuer is unknown."""
        issued = self._issuers.take(stream_id) if self._issuers is not None else None
        if issued is not None:
            return (*issued, True)
        return (parent, parent_closed, None, UNKNOWN_ISSUER, False)

    def clear(self) -> None:
        if self._issuers is not None:
            self._issuers.clear()


def _issuers(conn: Any) -> StreamIssuers:
    found = _issuers_of.get(conn)
    if found is None:
        found = _issuers_of.setdefault(conn, StreamIssuers(_UNLINKED_CAP))
    return found


def _at_fork_reinit() -> None:
    """Fork-child reset: forget the parent's state machines and offers.

    Each table pairs a parent connection with the scopes of the parent's tasks,
    and each offer points at one. The child's trackers start unlinked (the
    seam's own reset drops them), so a state machine the child does use offers
    again and is proven again, from the child's own writes. The thread-local
    index is reset for the forking thread, the only thread the child has.
    Reached by `_close_hook._at_fork_reinit`, whose probe owns the patches.
    """
    _issuers_of.clear()
    _index.by_id = {}


def _mk_send_headers(orig: Any):  # noqa: ANN202
    def send_headers(this: Any, stream_id: Any, *args: Any, **kwargs: Any) -> Any:
        recorded: StreamIssuers | None = None
        # BEFORE the original: once it returns, the frame is queued and another
        # thread holding the write lock may flush it before a later record lands.
        with _RECORD:
            if this.config.client_side:
                issuers = _issuers(this)
                if issuers.record(stream_id):
                    recorded = issuers
        try:
            return orig(this, stream_id, *args, **kwargs)
        except BaseException:
            # A refused stream was never opened, and its id may be retried.
            if recorded is not None:
                with _RECORD:
                    recorded.forget(stream_id)
            raise

    return send_headers


def _mk_data_to_send(orig: Any):  # noqa: ANN202
    def data_to_send(this: Any, *args: Any, **kwargs: Any) -> Any:
        out = orig(this, *args, **kwargs)
        with _OFFER:
            if (
                isinstance(out, bytes)
                and _OFFER_MIN <= len(out) <= _OFFER_MAX
                and this.config.client_side
            ):
                issuers = _issuers(this)
                if not issuers.linked:
                    issuers.offer(out)
        return out

    return data_to_send


def _h2_connection_class() -> Any | None:
    """`h2.connection.H2Connection`, or None where `h2` is not installed."""
    if "h2" not in sys.modules and importlib.util.find_spec("h2") is None:
        return None
    found: Any | None = None
    with _IMPORT_H2:
        from h2.connection import H2Connection

        found = H2Connection
    return found


def patch_h2(patches: PatchSet) -> None:
    """Put the two reads on `H2Connection`, into the caller's `PatchSet`."""
    cls = _h2_connection_class()
    if cls is None:
        return
    patches.patch(cls, "send_headers", _mk_send_headers(cls.send_headers))
    patches.patch(cls, "data_to_send", _mk_data_to_send(cls.data_to_send))
