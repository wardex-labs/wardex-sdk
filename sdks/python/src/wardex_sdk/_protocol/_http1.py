"""Thin wrapper that maps the native Http1Parser to ParsedMessage."""

from __future__ import annotations

from .. import _wardex_native
from .._assembly import Limitation, counters
from .._enums import Protocol
from .._types import ParsedMessage
from ._base import ProtocolParserInterface


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
    )


class _Http1Parser(ProtocolParserInterface):
    def __init__(self, is_request: bool, limits: object | None = None) -> None:
        self._native = _wardex_native.protocol.Http1Parser(is_request, limits)

    def protocol_name(self) -> str:
        return "http/1.1"

    def feed(self, data: bytes) -> list[ParsedMessage]:
        return [_to_parsed(m) for m in self._native.feed(data)]

    def flush(self) -> ParsedMessage | None:
        raw = self._native.flush_truncated()
        return _to_parsed(raw) if raw is not None else None

    def disabled_reason(self) -> str | None:
        return self._native.disabled_reason()  # type: ignore[no-any-return]


class Http1RequestParser(_Http1Parser):
    def __init__(self, limits: object | None = None) -> None:
        super().__init__(True, limits)


class Http1ResponseParser(_Http1Parser):
    def __init__(self, limits: object | None = None) -> None:
        super().__init__(False, limits)
