"""Thin wrapper for the native Http2Parser."""

from __future__ import annotations

from typing import Any

from .. import _wardex_native


class Http2Parser:
    """Incremental parser for a single h2 connection. Passes through to the native Http2Parser."""

    def __init__(self, limits: object | None = None) -> None:
        self._native = _wardex_native.protocol.Http2Parser(limits)

    def feed(
        self, from_client: bool, data: bytes
    ) -> tuple[list[tuple[int, str | None]], list[Any], list[int]]:
        """Feed `data`, read in the direction `from_client` names.

        Returns `(opened, transactions, reset)`. `opened` is one `(stream_id, path)` pair per stream
        whose request ended in this call, in arrival order; `path` is the request `:path` read as
        the request ended, kept across the stream table evicting the request's header block before
        its end, and None when the request carried none or that memory dropped it under the same
        bound. It is read there because the stream table evicts within the same call, so no later
        question could answer it. `transactions` are the native transactions whose response ended
        in this call. `reset` is the stream ids a RST_STREAM ended in this call, in arrival order,
        whichever peer sent it: no transaction will ever complete for them.
        """
        return self._native.feed(from_client, data)  # type: ignore[no-any-return]

    def disabled_reason(self) -> str | None:
        """Why the connection latched off, if it did — a debug string, never a
        span marker: no transaction exists to carry it."""
        return self._native.disabled_reason()  # type: ignore[no-any-return]

    def finish(self) -> list[Any]:
        """The connection closed: every stream whose response status arrived and whose response
        never ended, as native transactions with `response_ended` False."""
        return self._native.finish()  # type: ignore[no-any-return]
