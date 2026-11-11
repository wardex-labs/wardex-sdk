"""WSGI middleware — trace continuation for Flask / Django-WSGI apps.

The request's block stays open until the server has iterated and closed the
body, so work a streaming response does while it is iterated (an LLM reply
streamed to the client) is in the request's trace and isolation scope too.
"""

from __future__ import annotations

import contextvars
import sys
from collections.abc import Iterator
from typing import Any

from ._propagate import continue_trace


class WardexWsgiMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    def __call__(self, environ: Any, start_response: Any) -> Any:
        try:
            headers = {
                k[5:].replace("_", "-").lower(): v
                for k, v in environ.items()
                if isinstance(k, str) and k.startswith("HTTP_")
            }
        except Exception:
            headers = {}
        # The server iterates the body after this returns, on a thread whose
        # context may serve the next request; one private Context holds the
        # block from the call to the body's close().
        ctx = contextvars.copy_context()
        block = continue_trace(headers)
        ctx.run(block.__enter__)
        try:
            body = ctx.run(self.app, environ, start_response)
        except BaseException:
            ctx.run(block.__exit__, *sys.exc_info())
            raise
        if _runs_no_code(body, environ):
            ctx.run(block.__exit__, None, None, None)
            return body
        return _RequestBody(body, ctx, block)


def _runs_no_code(body: Any, environ: Any) -> bool:
    """A list, a tuple or the server's file wrapper runs no app code while it
    is served, and wrapping it would hide its length or its sendfile path."""
    if isinstance(body, (list, tuple)):
        return True
    wrapper = environ.get("wsgi.file_wrapper") if isinstance(environ, dict) else None
    return isinstance(wrapper, type) and isinstance(body, wrapper)


class _RequestBody:
    """The app's body, iterated and closed inside the request's Context."""

    def __init__(self, body: Any, ctx: contextvars.Context, block: Any) -> None:
        self._body = body
        self._ctx = ctx
        self._block = block
        self._it: Iterator[Any] | None = None
        self._open = True

    def __iter__(self) -> _RequestBody:
        return self

    def __next__(self) -> Any:
        if self._it is None:
            self._it = self._ctx.run(iter, self._body)
        return self._ctx.run(next, self._it)

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        close = getattr(self._body, "close", None)
        try:
            if close is not None:
                self._ctx.run(close)
        finally:
            self._ctx.run(self._block.__exit__, None, None, None)
