"""Opt-in W3C header injection into HTTP client libraries.

The byte seam stays observe-only forever; this module is the single place
wardex mutates user traffic, and only when propagation.enabled is True. Everything
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
from collections.abc import Callable
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


def _matches_target(host: str, patterns: tuple[str, ...]) -> bool:
    """Glob-match an outbound host against the configured target patterns.

    `fnmatchcase` against both sides lowercased, rather than plain `fnmatch`.
    Hostnames are case-insensitive, so `API.MyCorp.com` has to match
    `*.mycorp.com` — and `fnmatch` gets that wrong in two directions at once:
    it defers to `os.path.normcase`, which folds case on Windows and does
    nothing on Linux or macOS. An allowlist that admits a host on one operating
    system and refuses the same host on another is worse than either answer
    consistently applied, because it turns a propagation gap into something
    only one developer's machine can reproduce.

    Only the host is folded. The pattern is folded too rather than documented
    as "write it lowercase", since a user who typed `*.MyCorp.com` in a config
    file meant the same set of hosts.
    """
    lowered = host.lower()
    return any(fnmatch.fnmatchcase(lowered, pattern.lower()) for pattern in patterns)


def _build_inject_headers(host: str) -> dict[str, str]:
    try:
        from .. import _suppress  # noqa: PLC0415

        if _suppress.is_suppressed():
            return {}
        client = _hub.get_client()
        if client is None:
            return {}
        cfg = client.config
        if not cfg.propagation.enabled:
            return {}
        targets = cfg.propagation.targets
        if targets is not None and not _matches_target(host, targets):
            return {}
        return get_trace_headers()
    except Exception:
        return {}


def _headers_to_add(host: str, has_header: Callable[[str], bool]) -> dict[str, str]:
    """The headers wardex may add to one outbound request — the whole rule.

    ONE rule for all three patched libraries: wardex only ever ADDS a header
    the caller has not already set. It never replaces one, and never appends a
    second copy of one. This is the same principle the byte seam is built on,
    applied to the one place wardex is allowed to write: whatever the host
    application put on the wire is what goes on the wire.

    `has_header` is the library's own case-insensitive membership test, and it
    has to see EVERY layer that will actually be sent. aiohttp merges a
    session's default headers with the per-request ones long after the patch
    runs, so a predicate that looks only at the per-request mapping answers
    "absent" for a `traceparent` the session already carries — and wardex then
    writes a per-request header that outranks the host's session default,
    silently rewriting the trace context of every call on that session.

    `traceparent` gates the whole injection. A caller who set one owns the
    trace context for this request, and pairing their traceparent with our
    tracestate would attribute vendor state to a trace that never carried it.

    `tracestate` set WITHOUT a traceparent is likewise left alone, and this is
    where the three libraries used to disagree: httpx and requests assigned
    into a case-insensitive mapping and replaced the caller's value, while
    aiohttp `extend`ed a CIMultiDict and put both on the wire — one request,
    two `tracestate` headers, which is not a shape the W3C spec defines a
    reading for. Dropping ours keeps the rule above intact in all three.
    """
    if has_header("traceparent"):
        return {}
    headers = _build_inject_headers(host)
    if headers and has_header("tracestate"):
        headers.pop("tracestate", None)
    return headers


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
            # `httpx.Headers` is case-insensitive, and by the time `send` runs
            # the Client has already merged its own default headers into the
            # Request — so this one mapping is every layer that will be sent.
            headers = request.headers
            add = _headers_to_add(request.url.host or "", headers.__contains__)
            for k, v in add.items():
                headers[k] = v
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
            # A PreparedRequest: `Session.prepare_request` has already folded
            # the session's headers into this CaseInsensitiveDict, so like
            # httpx there is a single layer to consult here.
            headers = request.headers
            host = urlparse(request.url).hostname or ""
            for k, v in _headers_to_add(host, headers.__contains__).items():
                headers[k] = v
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
            # aiohttp is the one library of the three that has NOT merged its
            # session defaults yet: `_prepare_headers` does that after this
            # call, and a per-request header wins there. So the session's own
            # headers are the second layer this predicate has to see — without
            # them, a host that set `traceparent` once on the ClientSession had
            # it overwritten on every single request. Both lookups are
            # case-insensitive (CIMultiDict).
            defaults = getattr(self, "headers", None)

            def _has(name: str) -> bool:
                return name in merged or (defaults is not None and name in defaults)

            inject = _headers_to_add(host, _has)
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
