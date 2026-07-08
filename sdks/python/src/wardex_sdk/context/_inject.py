"""Opt-in W3C header injection into HTTP client libraries (Phase 4b).

The byte seam stays observe-only forever; this module is the single place
wardex mutates user traffic, and only when propagate_trace=True. Everything
is fail-silent: a failed patch or header computation must never break the
user's HTTP call. Patched libraries: httpx (sync+async), requests, aiohttp —
each is a soft dependency (try-import).
"""

from __future__ import annotations

import fnmatch
from typing import Any

from .. import _hub
from ._propagate import get_trace_headers

_orig: dict[str, Any] = {}


def _build_inject_headers(host: str) -> dict[str, str]:
    try:
        from ..interceptors import _exclusion  # noqa: PLC0415

        if _exclusion.is_suppressed():
            return {}
        client = _hub.get_client()
        if client is None:
            return {}
        cfg = client.config
        if not cfg.propagate_trace:
            return {}
        targets = cfg.propagate_targets
        if targets is not None and not any(fnmatch.fnmatch(host, p) for p in targets):
            return {}
        return get_trace_headers()
    except Exception:
        return {}


def _install_httpx() -> None:
    try:
        import httpx  # noqa: PLC0415
    except ImportError:
        return
    if "httpx.Client.send" in _orig:
        return

    orig_send = httpx.Client.send
    orig_async_send = httpx.AsyncClient.send

    def _apply(request: Any) -> None:
        try:
            if "traceparent" in request.headers:
                return
            for k, v in _build_inject_headers(request.url.host or "").items():
                request.headers[k] = v
        except Exception:
            pass

    def send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        _apply(request)
        return orig_send(self, request, *args, **kwargs)

    async def async_send(self: Any, request: Any, *args: Any, **kwargs: Any) -> Any:
        _apply(request)
        return await orig_async_send(self, request, *args, **kwargs)

    _orig["httpx.Client.send"] = orig_send
    _orig["httpx.AsyncClient.send"] = orig_async_send
    httpx.Client.send = send
    httpx.AsyncClient.send = async_send


def _uninstall_httpx() -> None:
    try:
        import httpx  # noqa: PLC0415
    except ImportError:
        return
    if "httpx.Client.send" in _orig:
        httpx.Client.send = _orig.pop("httpx.Client.send")
        httpx.AsyncClient.send = _orig.pop("httpx.AsyncClient.send")


def install_propagation() -> None:
    _install_httpx()


def uninstall_propagation() -> None:
    _uninstall_httpx()
