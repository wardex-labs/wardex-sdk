"""Pure ASGI 3 middleware — one line of user code to join distributed traces.

Framework-agnostic: works for FastAPI/Starlette/Django ASGI alike. Handles
http and websocket scopes; lifespan passes through untouched. wardex-internal
failures are swallowed (fail-open) — app exceptions always propagate.
"""

from __future__ import annotations

from typing import Any

from ._propagate import continue_trace


class WardexMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        try:
            headers = {
                k.decode("latin-1"): v.decode("latin-1") for k, v in scope.get("headers") or []
            }
        except Exception:
            headers = {}
        with continue_trace(headers):
            await self.app(scope, receive, send)
