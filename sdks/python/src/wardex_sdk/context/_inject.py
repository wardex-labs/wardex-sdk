"""Opt-in W3C header injection into HTTP client libraries (Phase 4b).

The byte seam stays observe-only forever; this module is the single place
wardex mutates user traffic, and only when propagate_trace=True. Everything
is fail-silent: a failed patch or header computation must never break the
user's HTTP call. Patched libraries: httpx (sync+async), requests, aiohttp —
each is a soft dependency (try-import).

The patches go through `assembly.PatchSet` like every other patch site in the
SDK. This one is module-level rather than per-instance state, because the
targets are process-wide classes and `install_propagation()` is a module
function, so the set lives beside them — but the guarantee it buys is the same
one, and it matters more here than almost anywhere: `httpx.Client.send`,
`requests.Session.send` and `aiohttp.ClientSession._request` are the exact
attributes OpenTelemetry's HTTP instrumentors patch. The unconditional
`setattr` this replaced destroyed whatever they had installed after wardex,
on every re-`init()` and every `close()`, and left the host running an
interception nobody could see. Now a superseded attribute is left alone and
counted (`Limitation.PATCH_SUPERSEDED`).
"""

from __future__ import annotations

import fnmatch
from typing import Any

from .. import _hub
from ..assembly import PatchSet
from ._propagate import get_trace_headers

_patches: PatchSet | None = None
"""The live set, or None while nothing is patched. Built at install time."""

_installed: set[str] = set()
"""Which libraries are patched — the per-library idempotence check.

Separate from `_patches` because a PatchSet does not (and should not) answer
"is this attribute already yours"; asking it would make the install path
depend on the restore bookkeeping. Cleared together with the set, so the two
can never disagree about what is installed.
"""


def _patchset() -> PatchSet:
    """The module-level set, created on first patch of an install cycle.

    Built here rather than at import so that `config.debug` — which is not
    known until a client exists — reaches the restore path. A restore that
    fails invisibly under `debug=True` is what `_diag` exists to prevent.
    """
    global _patches
    if _patches is None:
        config = getattr(_hub.get_client(), "config", None)
        _patches = PatchSet("context.inject", debug=bool(getattr(config, "debug", False)))
    return _patches


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
    if "httpx" in _installed:
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

    patches = _patchset()
    patches.patch(httpx.Client, "send", send)
    patches.patch(httpx.AsyncClient, "send", async_send)
    _installed.add("httpx")


def _install_requests() -> None:
    try:
        import requests  # noqa: PLC0415
    except ImportError:
        return
    if "requests" in _installed:
        return
    from urllib.parse import urlparse  # noqa: PLC0415

    orig_send = requests.Session.send

    def send(self: Any, request: Any, **kwargs: Any) -> Any:
        try:
            if "traceparent" not in request.headers:
                host = urlparse(request.url).hostname or ""
                for k, v in _build_inject_headers(host).items():
                    request.headers[k] = v
        except Exception:
            pass
        return orig_send(self, request, **kwargs)

    _patchset().patch(requests.Session, "send", send)
    _installed.add("requests")


def _install_aiohttp() -> None:
    try:
        import aiohttp  # noqa: PLC0415
        from multidict import CIMultiDict  # noqa: PLC0415
        from yarl import URL  # noqa: PLC0415
    except ImportError:
        return
    if "aiohttp" in _installed:
        return

    orig_request = aiohttp.ClientSession._request

    async def _request(self: Any, method: Any, str_or_url: Any, **kwargs: Any) -> Any:
        try:
            host = URL(str_or_url).host or ""
            # CIMultiDict accepts mappings and iterables of pairs while
            # preserving duplicate keys (both are valid aiohttp LooseHeaders).
            merged = CIMultiDict(kwargs.get("headers") or {})
            if "traceparent" not in merged:  # CIMultiDict lookup is case-insensitive
                inject = _build_inject_headers(host)
                if inject:
                    merged.extend(inject)
                    kwargs["headers"] = merged
        except Exception:
            pass
        return await orig_request(self, method, str_or_url, **kwargs)

    _patchset().patch(aiohttp.ClientSession, "_request", _request)
    _installed.add("aiohttp")


def install_propagation() -> None:
    """Patch every importable client library. Idempotent, per library."""
    _install_httpx()
    _install_requests()
    _install_aiohttp()


def uninstall_propagation() -> None:
    """Undo every patch this module installed. Safe to call when none are.

    Restores newest-first, skips any attribute another library has taken over
    since, and guards each restore individually — `PatchSet.restore_all()`.
    The set is dropped rather than reused so the next `install_propagation()`
    picks up the current client's `debug` setting.
    """
    global _patches
    if _patches is not None:
        _patches.restore_all()
        _patches = None
    _installed.clear()
