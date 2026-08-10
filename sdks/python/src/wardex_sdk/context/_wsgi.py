"""WSGI middleware — trace continuation for Flask / Django-WSGI apps.

Known v1 limitation (documented): the context covers the app callable only.
Streaming responses (work done while iterating the returned iterable) run
outside the joined context; framework handlers (Flask views, Django views)
do their work inside the callable, which is the case this targets.
"""

from __future__ import annotations

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
        with continue_trace(headers):
            return self.app(environ, start_response)
