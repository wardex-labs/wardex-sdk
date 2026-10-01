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
issuer is read THERE and handed to the tracker of the connection those frames
are written to. Linking
the two is the part that has to be proven rather than assumed, and it is proven
once per connection, by object identity:

* `data_to_send` returns the bytes the client is about to write. Until its state
  machine is linked, those bytes are OFFERED on the calling thread.
* The next h2 tracker write on that thread takes the offer, and links only if
  the bytes it was handed ARE that object (`is`, not `==`): the client wrote
  this state machine's output to this connection. httpcore, urllib3 and a
  hand-driven `h2` all write the chunk in the call right after they take it, so
  an offer that does not match is a write that was not that chunk, and it is
  dropped.

A stream whose issuer was not read here names NO conversation: the state
machine was never linked (a client on another HTTP/2 library, a chunk copied
before it was written, capture attached after the stream was opened), and the
writer's scope is a guess on a shared connection. A conversation is never
guessed. Its parent is latched as it was before this module existed.
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

#: What the issuing task's scope said when it opened a stream: its active span,
#: whether that span's unit had already closed, and its conversation.
Issued = tuple[SpanContext | None, bool, ConversationContext | None]

#: The largest chunk offered for linking. An offer holds its bytes until the
#: thread's next h2 tracker write, and linking needs one small chunk only: the
#: connection preface a client writes first, or any later control frame. The
#: floor is a frame header, which also keeps out the empty and one-byte `bytes`
#: objects CPython shares between callers, for which `is` would prove nothing.
_OFFER_MIN, _OFFER_MAX = 9, 1 << 16
_offer = threading.local()
#: State machine -> weak reference to its linked connection's `StreamIssuers`.
#: Weak at both ends: the link keeps neither the host's connection nor the
#: tracker alive, and a dead tracker (its connection retired, or reset by a
#: fork) reads as "not linked", so the next offer links its successor.
_linked: weakref.WeakKeyDictionary[Any, weakref.ref[StreamIssuers]] = weakref.WeakKeyDictionary()

#: Importing `h2` once it has been FOUND, as `_close_hook` imports anyio: its
#: absence is an answer, and present but unimportable is worth a count.
_IMPORT_H2 = guard("interceptors.h2_issuer.import")
#: The two wrappers run inside the host's HTTP/2 client, after the original has
#: returned. A raise here would fail a request over a span attribute.
_RECORD = guard("interceptors.h2_issuer.record")
_OFFER = guard("interceptors.h2_issuer.offer")


class StreamIssuers:
    """One h2 connection's proven issuers, by stream id, until each stream opens.

    Bounded like the tracker's latch beside it, lowest stream id first: an
    entry whose stream never ends its request half (reset before END_STREAM)
    would otherwise stay for the life of the connection. Losing one costs that
    stream its proof, so it names no conversation, never the wrong one.
    """

    __slots__ = ("__weakref__", "_by_stream", "_cap", "linked")

    def __init__(self, cap: int) -> None:
        self._by_stream: dict[int, Issued] = {}
        self._cap = cap
        self.linked = False

    def claim(self, data: bytes) -> None:
        """Link to the state machine whose offered chunk `data` is, if it is one."""
        if self.linked:
            return
        offer = getattr(_offer, "pending", None)
        if offer is None:
            return
        _offer.pending = None
        conn = offer[0]()
        if conn is not None and offer[1] is data:
            _linked[conn] = weakref.ref(self)
            self.linked = True

    def record(self, stream_id: int) -> None:
        """The CALLER's scope, as the issuer of `stream_id`. Only the first call
        for a stream counts: a later `send_headers` on it sends trailers."""
        if stream_id in self._by_stream:
            return
        scope = _hub.get_current_scope()  # ONE read: parent and conversation are one fact
        parent = scope.active_span_context
        self._by_stream[stream_id] = (parent, parent_is_closed_unit(parent), scope.conversation)
        while len(self._by_stream) > self._cap:
            self._by_stream.pop(min(self._by_stream))

    def take(self, stream_id: int) -> Issued | None:
        return self._by_stream.pop(stream_id, None)

    def clear(self) -> None:
        self._by_stream.clear()


def _linked_issuers(conn: Any) -> StreamIssuers | None:
    ref = _linked.get(conn)
    return ref() if ref is not None else None


def _mk_send_headers(orig: Any):  # noqa: ANN202
    def send_headers(this: Any, stream_id: Any, *args: Any, **kwargs: Any) -> Any:
        ret = orig(this, stream_id, *args, **kwargs)
        # After the original, and only if it returned: a refused stream was never opened.
        with _RECORD:
            issuers = _linked_issuers(this)
            if issuers is not None:
                issuers.record(stream_id)
        return ret

    return send_headers


def _mk_data_to_send(orig: Any):  # noqa: ANN202
    def data_to_send(this: Any, *args: Any, **kwargs: Any) -> Any:
        out = orig(this, *args, **kwargs)
        with _OFFER:
            if (
                isinstance(out, bytes)
                and _OFFER_MIN <= len(out) <= _OFFER_MAX
                and this.config.client_side
                and _linked_issuers(this) is None
            ):
                _offer.pending = (weakref.ref(this), out)
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
