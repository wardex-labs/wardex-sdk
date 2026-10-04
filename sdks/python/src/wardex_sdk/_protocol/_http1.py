"""Thin wrapper that maps the native Http1Parser to ParsedMessage."""

from __future__ import annotations

import zlib

from .. import _wardex_native
from .._assembly import Limitation, counters, guard
from .._enums import Protocol
from .._types import ParsedMessage
from ._base import ProtocolParserInterface

#: The methods an HTTP/1 request line opens with, each with its space. The seams sniff a
#: connection's first bytes against it; `_Http1Tracker` tells a new request from the late
#: rest of an unfinished one with it.
REQUEST_METHODS = (
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


def _resolve_markers(raw: object) -> tuple[Limitation, ...]:
    """Rust marker strings -> `Limitation` members. THE boundary, and the only one.

    The native parsers hand back `&'static str`, so this is where the closed
    vocabulary is actually entered. A string with no member cannot be carried
    forward — `CaptureIntegrity.limitations` holds members now — so the choice is
    between dropping it in silence and saying so. It says so: the drop is
    counted and, under `init(debug=True)`, logged.

    It should be unreachable. `tests/test_limitation_census.py` reads every
    marker literal in `crates/` on every test run and fails if one has no member
    here, which makes drift a CI failure rather than a runtime surprise. This
    branch is what keeps that guarantee honest if someone ever bypasses it.
    """
    out: list[Limitation] = []
    for value in raw.limitations:  # type: ignore[attr-defined]
        member = Limitation.from_wire(value)
        if member is None:
            counters.bump("protocol.limitation_unresolved")
            continue
        out.append(member)
    return tuple(out)


def declares_event_stream(content_type: str | None) -> bool:
    """Did this Content-Type value declare a Server-Sent Events stream?

    The media type alone, case-insensitive, parameters dropped — the reading
    `crates/wardex-protocol/src/http1.rs::bare_media_type` gives the header
    when it picks a body cap. The header is the stream's declaration by
    protocol, so the body is not consulted: a stream whose bytes no parser can
    read (an encoding it does not decode, a cap that cut it mid-character) is
    still the stream the server said it was.

    Both HTTP trackers ask it. Their `_Txn.event_stream` is a field of its own
    rather than `content_type`, which only the HTTP/2 tracker fills and which
    also picks the gRPC branches. The HTTP/2 parser's value is the response's
    Content-Type, or the request's when the response named none;
    `text/event-stream` is a response format no client sends as a request
    body, so in practice it reads the response's declaration there too.
    """
    if not content_type:
        return False
    return content_type.split(";", 1)[0].strip().lower() == "text/event-stream"


#: The one coding layer the SDK undoes. Both inflaters (the semantic parser's and
#: `inflate_body` below) recognise gzip and zlib by the bytes' own header.
_INFLATED_CODINGS = frozenset({"gzip", "x-gzip", "deflate"})


def sniff_decoded_body(
    content_encoding: str | None, wire: bytes, body: bytes, whole: bool = True
) -> bool | None:
    """The event-stream sniff over a response body, or None when the SDK never read that body.

    `wire` is the body as sent, still in its content coding, and `body` what the SDK made of it,
    inflated when it could. The sniff answers only for bytes that are the body's content: the
    response declared no `Content-Encoding` (`identity` is none), or declared one gzip or deflate
    layer and the SDK inflated it (`body` is not `wire`). Any other declared coding (`br`, `zstd`,
    more than one layer), or a gzip or deflate nothing inflated (raw deflate, a corrupt stream),
    leaves bytes the SDK did not read, whatever they look like: a brotli stream can begin with `[`,
    and a few can even be valid text. The sniff itself answers None for a body that is not text
    (`crates/wardex-protocol/src/sse.rs::sniff`).

    `whole=False`: `body` is only the part of the content the SDK read, a prefix cut by a capture
    cap as sent or by the inflate cap. Event lines in that part were seen, so it answers True; an
    answer of "not a stream" is about the rest too, which nobody read, so it answers None instead.
    A cap can end the part inside a character, so the sniff reads it up to the last whole one.
    """
    codings = [c.strip().lower() for c in (content_encoding or "").split(",")]
    codings = [c for c in codings if c and c != "identity"]
    if codings and not (len(codings) == 1 and codings[0] in _INFLATED_CODINGS and body != wire):
        return None
    if whole:
        return _wardex_native.protocol.sniff_event_stream(body)  # type: ignore[no-any-return]
    return True if _wardex_native.protocol.sniff_event_stream(_whole_chars(body)) else None


def _whole_chars(part: bytes) -> bytes:
    """`part` without the one UTF-8 character a cut may have split at its end; else unchanged."""
    try:
        part.decode("utf-8")
    except UnicodeDecodeError as err:
        if err.reason == "unexpected end of data":  # only a sequence the end cut short
            return part[: err.start]
    return part


def _to_parsed(raw: object) -> ParsedMessage:
    # raw: _wardex_native.protocol.RawHttpMessage
    return ParsedMessage(
        protocol=Protocol.HTTP,
        method=raw.method,  # type: ignore[attr-defined]
        url=raw.path,  # type: ignore[attr-defined]
        status_code=raw.status,  # type: ignore[attr-defined]
        headers=tuple(tuple(h) for h in raw.headers),  # type: ignore[attr-defined]
        body=raw.body,  # type: ignore[attr-defined]
        header_len=raw.header_len,  # type: ignore[attr-defined]
        truncated=raw.truncated,  # type: ignore[attr-defined]
        limitations=_resolve_markers(raw),
        incomplete=raw.incomplete,  # type: ignore[attr-defined]
    )


class _Http1Parser(ProtocolParserInterface):
    def __init__(self, is_request: bool, limits: object | None = None) -> None:
        self._native = _wardex_native.protocol.Http1Parser(is_request, limits)

    def protocol_name(self) -> str:
        return "http/1.1"

    def feed(self, data: bytes) -> list[ParsedMessage]:
        return [_to_parsed(m) for m in self._native.feed(data)]

    def expect_response_to(self, method: str) -> None:
        """The request the next final response answers used `method`: a response to HEAD has no
        body whatever its headers declare, and a 2xx to CONNECT opens a tunnel nothing in which is
        HTTP (RFC 9112 §6.3). Only the side that saw the request knows; one slot, for the next one.
        """
        self._native.expect_response_to(method)

    def flush(self, peer_closed: bool = False) -> ParsedMessage | None:
        """The response in flight when the stream ended, or None.

        `peer_closed`: the peer's EOF was observed, which is what ends a body
        with no framing — that one comes back whole. Anything else in flight
        comes back `incomplete`: a Content-Length or chunked framing that
        promised more, or an unframed body whose end nobody saw. None for a
        header block that never completed, and on the request parser.
        """
        raw = self._native.finish(peer_closed)
        return _to_parsed(raw) if raw is not None else None

    def disabled_reason(self) -> str | None:
        return self._native.disabled_reason()  # type: ignore[no-any-return]

    def idle(self) -> bool:
        """Between messages: the last one completed and no byte of the next has arrived."""
        return self._native.is_idle()  # type: ignore[no-any-return]

    def method_in_flight(self) -> str | None:
        """The method of the request whose header block parsed and whose body is arriving."""
        return self._native.method_in_flight()  # type: ignore[no-any-return]


class Http1RequestParser(_Http1Parser):
    def __init__(self, limits: object | None = None) -> None:
        super().__init__(True, limits)


class Http1ResponseParser(_Http1Parser):
    def __init__(self, limits: object | None = None) -> None:
        super().__init__(False, limits)


def inflate_body(body: bytes, limits: object | None) -> tuple[bytes, bool]:
    """A gzip- or zlib-compressed body as the bytes it carries, and whether it is just a prefix.

    A captured body is what a debugger reads and what masking scans, and
    neither can see through compression: a gzipped OAuth token response
    shipped its `access_token` as base64 anyone could gunzip. Recognised by
    its header, as the semantic parser recognises it.

    Bounded by the smaller of `max_decoded_bytes` and `max_opaque_body_bytes`: the compressed bytes
    were admitted under some cap, and inflating must not turn a 13 KB download into a
    multi-megabyte span. A body that inflates past the bound keeps its inflated prefix and says so
    (the body's whole content was not read), and the caller marks the transaction truncated —
    never the compressed bytes, whose secrets anyone could recover. Accepted only when the stream
    completed, or when a stream cut short (by the capture cap) inflated to text: a plain body that
    merely starts like a zlib header decodes to noise, and is returned as it was.
    """
    cap = min(
        getattr(limits, "max_decoded_bytes", 0) or 0,
        getattr(limits, "max_opaque_body_bytes", 0) or 0,
    )
    is_gzip = body[:2] == b"\x1f\x8b"
    # A zlib header: deflate method, and a check value divisible by 31 —
    # which a text body that merely starts with `x` almost never is.
    is_zlib = len(body) >= 2 and body[0] & 0x0F == 8 and (body[0] << 8 | body[1]) % 31 == 0
    if not cap or not (is_gzip or is_zlib):
        return body, False
    result = None
    with guard("interceptors.inflate"):
        result = _inflate_once(body, cap)
    if result is None or not result[0]:
        return body, False
    out, complete = result
    # Text, give or take the one character a cut stream may end inside.
    text_len = len(out.decode("utf-8", "ignore").encode())
    if not complete and len(out) <= cap and text_len < len(out) - 3:
        return body, False
    if len(out) > cap:
        return out[:cap], True
    return out, False


def _inflate_once(body: bytes, cap: int) -> tuple[bytes, bool]:
    """Up to `cap + 1` inflated bytes, and whether the stream completed."""
    inflater = zlib.decompressobj(wbits=47)  # 47: a gzip or a zlib header
    out = inflater.decompress(body, cap + 1)
    return out, inflater.eof
