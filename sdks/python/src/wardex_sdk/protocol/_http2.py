"""Thin wrapper for the native Http2Parser."""

from __future__ import annotations

from typing import Any

from .. import _wardex_native


class Http2Parser:
    """Incremental parser for a single h2 connection. Passes through to the native Http2Parser."""

    def __init__(self) -> None:
        self._native = _wardex_native.protocol.Http2Parser()

    def feed(self, from_client: bool, data: bytes) -> tuple[list[int], list[Any]]:
        # returns: (opened_request_streams, [native Http2Transaction ...])
        return self._native.feed(from_client, data)
