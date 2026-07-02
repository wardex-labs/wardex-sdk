"""Thin wrapper that maps the native Http1Parser to ParsedMessage."""

from __future__ import annotations

from .. import _wardex_native
from .._enums import Protocol
from .._types import ParsedMessage
from ._base import ProtocolParserInterface


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
    )


class _Http1Parser(ProtocolParserInterface):
    def __init__(self, is_request: bool) -> None:
        self._native = _wardex_native.protocol.Http1Parser(is_request)

    def protocol_name(self) -> str:
        return "http/1.1"

    def feed(self, data: bytes) -> list[ParsedMessage]:
        return [_to_parsed(m) for m in self._native.feed(data)]

    def flush(self) -> ParsedMessage | None:
        raw = self._native.flush_truncated()
        return _to_parsed(raw) if raw is not None else None


class Http1RequestParser(_Http1Parser):
    def __init__(self) -> None:
        super().__init__(True)


class Http1ResponseParser(_Http1Parser):
    def __init__(self) -> None:
        super().__init__(False)
