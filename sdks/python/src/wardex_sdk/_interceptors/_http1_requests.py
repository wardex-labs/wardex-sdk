"""Which request an HTTP/1 reply answers, when the seam did not see every request byte in order.

`_Http1Tracker` pairs each final reply on a connection with the request before it. That is the
whole story while the seam sees every request byte before the reply to it. Two things break it,
and this module is the request half of the tracker that copes with both:

* A reply arrives before its request was seen whole. A server may answer an upload early (a 413,
  a 401) while the client is still writing it, or the body went out where no `socket.socket`
  method carries it (`os.sendfile`, `os.write` on the descriptor). The reply ships without a
  request half, counted (`protocol.http1.request_unfinished`), and the parser is left inside that
  request: it is ORPHANED.
* After an orphan, the bytes the seam sees next are either its rest arriving late (the early
  answer) or the next request, the rest having gone by unseen. Neither the bytes nor their timing
  say which: a body may contain anything, a request line included, and a late rest and the next
  request can reach the seam in one `sendmsg` when asyncio flushes a queued buffer. So when the
  first chunk after the reply opens like a request, both readings are parsed side by side until
  the stream decides: a reading its parser rejects is dropped, and a reply goes to the reading
  that holds more of a request a client would send (`_Reading.claim`), the continuing one on a
  tie, since in it every byte was seen.

A reply with no request byte at all since the last reply pairs with nothing, and is counted
(`protocol.http1.request_unobserved`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from .. import _hub
from .._assembly import counters, parent_is_closed_unit
from .._protocol import REQUEST_METHODS
from .._protocol._http1 import Http1RequestParser
from .._types import ConversationContext, ParsedMessage, SpanContext


@dataclass(frozen=True)
class Issue:
    """Where a request was issued, read where its first byte was seen."""

    start_ns: int
    parent: SpanContext | None
    #: Asked on the same line, so that a context a finished unit left standing is refused before
    #: it can become a parent, or open the `capture_mode=AGENT` gate, for traffic that has nothing
    #: to do with that run (`assembly.parent_is_closed_unit`).
    parent_closed: bool
    conversation: ConversationContext | None


#: What a reply pairs with when no request byte was latched: its start is the reply's own.
NOT_SEEN = Issue(0, None, False, None)


def _opens_like_a_request(data: bytes) -> bool:
    """Could `data` begin a request: a method and its space, or the start a short write cut."""
    return data.startswith(REQUEST_METHODS) or any(m.startswith(data) for m in REQUEST_METHODS)


def _sent_by_a_client(method: str | None) -> bool:
    return method is not None and f"{method} ".encode() in REQUEST_METHODS


def _issued_here() -> Issue:
    scope = _hub.get_current_scope()  # ONE read: parent and conversation are one fact
    parent = scope.active_span_context
    return Issue(time.time_ns(), parent, parent_is_closed_unit(parent), scope.conversation)


class _Reading:
    """One parse of a connection's request bytes: the request not yet answered, and its issue."""

    __slots__ = ("issue", "orphan", "parser", "request")

    def __init__(self, limits: object | None) -> None:
        self.parser = Http1RequestParser(limits)
        self.issue: Issue | None = None
        self.request: ParsedMessage | None = None  # complete, not yet answered
        self.orphan = False  # the parser is inside a request whose reply already shipped

    def feed(self, data: bytes) -> None:
        if self.issue is None and not self.orphan:
            self.issue = _issued_here()
        for msg in self.parser.feed(data):
            if self.orphan:  # its late rest completed it, and its reply already shipped
                self.orphan = False
            else:
                self.request = msg
        if self.orphan and self.parser.disabled_reason() is not None:
            self.orphan = False  # a parser that stopped is inside nothing
        # The orphan ended inside this chunk and the next request began after it, in the same
        # chunk: that request's first byte was seen here, so it was issued here.
        if self.issue is None and not self.orphan:
            if self.request is not None or not self.parser.idle():
                self.issue = _issued_here()

    def claim(self) -> int:
        """How much this reading holds of the request a reply would answer: the LAST one it began.
        Ranked: complete, with a method a client would send (3); in flight with such a method (2);
        of another method (1); nothing, a header block still arriving, or the orphan itself, whose
        reply already shipped (0).

        The method is what tells a real request from the tail of one this reading took as an
        orphan's rest: with one byte of `POST /v1` unseen, `OST /v1` parses as a request line.
        """
        if self.orphan:
            return 0
        method = self.parser.method_in_flight()
        if method is not None:
            return 2 if _sent_by_a_client(method) else 1
        if self.request is None or not self.parser.idle():
            return 0
        return 3 if _sent_by_a_client(self.request.method) else 1


class RequestSide:
    """The request half of one HTTP/1 connection: what the next final reply pairs with."""

    __slots__ = ("_fresh", "_just_orphaned", "_limits", "_reading")

    def __init__(self, limits: object | None = None) -> None:
        self._limits = limits
        self._reading = _Reading(limits)
        #: The other reading of what followed an orphan: its rest went by unseen, and these bytes
        #: open the next request. Only while the stream has not decided between the two.
        self._fresh: _Reading | None = None
        #: No request byte yet since a reply orphaned its request. The next request can begin only
        #: at the first chunk after that reply: anything seen before it was the orphan's rest.
        self._just_orphaned = False

    @property
    def request(self) -> ParsedMessage | None:
        """The complete request the next final reply answers, if one is waiting."""
        return self._reading.request

    @property
    def issue(self) -> Issue:
        """Where the request the next final reply answers was issued."""
        return self._reading.issue or NOT_SEEN

    def disabled_reason(self) -> str | None:
        return self._reading.parser.disabled_reason()

    def feed(self, data: bytes) -> None:
        if self._just_orphaned:
            self._just_orphaned = False
            if _opens_like_a_request(data):
                self._fresh = _Reading(self._limits)
        fresh = self._fresh
        if fresh is not None:
            fresh.feed(data)
        self._reading.feed(data)
        if fresh is None:
            return
        if fresh.parser.disabled_reason() is not None:
            self._fresh = None  # not a request after all: those bytes were the orphan's rest
        elif self._reading.parser.disabled_reason() is not None:
            self._reading, self._fresh = fresh, None  # the rest went unseen: they were a request

    def decide(self) -> bool:
        """A reply is arriving, so a request was sent: settle on the reading that holds more of
        one (`_Reading.claim`), the continuing one on a tie, since in it every byte was seen. Says
        whether that changed which request waits.
        """
        fresh, self._fresh = self._fresh, None
        if fresh is None or fresh.claim() <= self._reading.claim():
            return False
        self._reading = fresh
        return True

    def take(self) -> tuple[ParsedMessage | None, Issue] | None:
        """At a final reply: the request it answers and where that request was issued, and the
        side is ready for the next one. None when not one byte of it was seen (counted): there is
        nothing to pair the reply with, and nothing about it to say.

        A request seen only in part pairs as `None` with its issue, counted, and is orphaned: the
        parser stays inside it, so that its rest, if it comes, is not read as the next request.
        A stopped parser pairs every reply as `None`, as it always has; the seam counts the stop.
        """
        self.decide()
        r = self._reading
        request, issue = r.request, r.issue
        r.request = r.issue = None
        if request is None and r.parser.disabled_reason() is None:
            if issue is None:
                counters.bump("protocol.http1.request_unobserved")
                return None
            counters.bump("protocol.http1.request_unfinished")
            r.orphan = self._just_orphaned = True
        return request, issue or NOT_SEEN

    def release(self) -> None:
        """The connection ended: drop what it was holding."""
        self._reading.request = None
        self._reading.orphan = self._just_orphaned = False
        self._fresh = None
