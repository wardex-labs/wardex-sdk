"""The transaction record every tracker returns, and how a span names and addresses it.

A tracker (`_trackers.py`) turns plaintext bytes into `_Txn`s; the seam (`_seam.py`) assembles a
span from a `_Txn` alone, whatever protocol produced it. `_name_path` and `_url_target` are the two
readings of the request target the span ships: its name and its URL.
"""

from __future__ import annotations

from dataclasses import dataclass

from .._assembly import Limitation
from .._types import ConversationContext, SpanContext
from ._issue_scope import ScopeSnapshot


@dataclass
class _Txn:
    """A single protocol-neutral transaction (request + response)."""

    method: str
    #: The request target as a span NAME may show it (`_name_path`).
    path: str
    status: int
    request_body: bytes
    response_body: bytes
    parent: SpanContext | None
    start_ns: int
    end_ns: int
    #: None when this tracker cannot time the first response byte (an HTTP/2
    #: stream, a capture that joined mid-connection): not measured, not zero.
    ttfb_ms: float | None
    #: Was `parent` latched off a unit that had ALREADY closed? Latched HERE, beside the parent and
    #: on the task that ISSUED the request, because the answer is a property of that instant: a
    #: request issued while the run was live is a child of the run's span whether or not the run
    #: finishes before the response arrives, and re-asking on the response side would orphan it. See
    #: `assembly._units.parent_is_closed_unit`.
    parent_closed: bool = False
    #: Was the latched parent DISCARDED by the tracker's own bound before this transaction arrived
    #: to claim it? Only `_Http2Tracker` can answer yes. Distinct from `parent is None`, which is
    #: the ordinary "nothing was ambient" and an honest trace root; this one says a parent was
    #: latched and wardex threw it away, which is a defect the span has to carry rather than a fact
    #: about the traffic. See `assembly._parentage.resolve_observed`.
    parent_evicted: bool = False
    #: The conversation the request was ISSUED in, latched beside `parent` for the same reason.
    conversation: ConversationContext | None = None
    #: The tags and user the span is stamped with, snapshotted beside `parent` for the same reason
    #: (`_issue_scope`). None only for a record built by hand: the client then reads its own scope.
    scope: ScopeSnapshot | None = None
    #: False where no h2 issuer was proven (`_h2_issuer`): `conversation` is unknown, not none.
    issuer_proven: bool = True
    truncated: bool = False
    #: Was every byte of this half counted? False when its body went past the capture cap (a prefix
    #: was kept), the half was lost or never finished parsing, or a WebSocket direction's frame
    #: parser stopped. The seam ships a size only for a half that was.
    request_counted: bool = True
    response_counted: bool = True
    #: `ParsedMessage.incomplete`: status and headers were observed, how the exchange ended was not.
    response_cut: bool = False
    #: The HTTP/2 stream this transaction rode; None on HTTP/1.
    stream_id: int | None = None
    # Capture-limitation markers the protocol parser attached to this transaction, merged into the
    # span's CaptureIntegrity.limitations by the seam. Members, not strings: the parser's
    # `&'static str` was resolved once at the PyO3 boundary (`_protocol/_http1.py`).
    limitations: tuple[Limitation, ...] = ()
    version: str = "1.1"
    #: See `ttfb_ms`; also None when no body byte arrived.
    ttft_ms: float | None = None
    content_type: str | None = None
    event_stream: bool = False  # see `declares_event_stream`
    content_encoding: str | None = None  # the response's; see `sniff_decoded_body`
    grpc_status: int | None = None
    grpc_message: str | None = None
    # WS upgrade signal (set on 101 detection — used by _ssl.py as the SWAP trigger)
    ws_upgrade: bool = False
    ws_upgrade_path: str | None = None
    ws_deflate: bool = False
    ws_leftover: bytes = b""
    # WS session span data (set on close/flush — filled in by _WebSocketTracker,
    # version=="websocket")
    ws_close_code: int | None = None
    ws_messages_sent: int = 0
    ws_messages_received: int = 0
    ws_bytes_sent: int = 0
    ws_bytes_received: int = 0
    ws_markers: tuple[Limitation, ...] = ()
    #: Confirmed LLM calls crossed this WebSocket connection and wardex read
    #: none. A capture claim for the gate (`_should_capture`), so the marked
    #: span ships under the default mode instead of being gated out with
    #: sem=None. The seam counts it (`interceptors.seam.ws_llm_semantics_unread`)
    #: when it builds the connection's span.
    ws_llm_call: bool = False
    #: The request target as sent, query included (`_url_target` reads it).
    #: `None` when the transaction never had one beyond `path`.
    target: str | None = None
    #: The upgrade path was a WebSocket-capable LLM row on a host that is not the provider's, and
    #: nothing corroborated an LLM call: no claim, but a recognised path must not vanish uncounted.
    #: The seam counts it (`interceptors.seam.ws_llm_endpoint_unconfirmed`) when it builds the
    #: connection's span. Exclusive with `ws_llm_call`.
    ws_llm_unconfirmed: bool = False


def _name_path(target: str) -> str:
    """The part of a request target a span NAME may carry.

    A name groups spans; it is not a place for data. The query and fragment
    are the call's arguments and go to the URL, where the masker judges each
    one; the userinfo of an absolute-form target (`http://user:pw@host/p`) is
    a credential and goes nowhere near a name. Byte positions, not a URL
    parser: a malformed target is shortened at worst.
    """
    path = target.split("?", 1)[0].split("#", 1)[0]
    scheme, sep, rest = path.partition("://")
    if not sep:
        return path
    authority, slash, tail = rest.partition("/")
    return f"{scheme}://{authority.rpartition('@')[2]}{slash}{tail}"


def _url_target(txn: _Txn, withhold: bool = False) -> str:
    """The request target a span's URL carries: the whole target as sent.

    The query rides in the URL like a body rides in the payload, and under the same policy: when the
    seam withholds a transaction's bodies (capture admitted only by wardex's own degradation), the
    query is withheld with them. Credentials in it are the native masker's to replace, and a URL's
    userinfo is replaced even when masking is off.
    """
    if withhold or txn.target is None:
        return txn.path
    return txn.target
