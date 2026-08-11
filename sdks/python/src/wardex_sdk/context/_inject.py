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
import functools
import importlib
import threading
from collections.abc import Callable
from typing import Any

from .. import _hub
from .._assembly import PatchSet
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

_install_lock = threading.RLock()
"""Serializes install/uninstall, so the idempotence above is real.

`"httpx" in _installed` and `_installed.add("httpx")` sit either side of three
attribute swaps, and two threads that arrive between them both read False and
both patch. The second one captures the FIRST one's wrapper as its `orig_send`
and the PatchSet records a restore target that is already wardex's own code —
so the stack is two deep, every outbound request runs the injection twice, and
`uninstall_propagation()` peels off one layer and leaves the host permanently
patched by a wardex that believes it has left.

The process-wide `init()` path is serialized a level up by the runtime lock and
would not reach that on its own. This lock is here because the guarantee
belongs to the module that owns the state: `install_propagation()` is reachable
directly, the runtime's lock is not this module's to rely on, and "safe as long
as the only caller keeps holding a different lock" is not a property anyone can
see from here.

REENTRANT, like every other lock this SDK puts on its teardown path, and for
the reason `_runtime` states there: a signal handler lands on the thread that
already holds the lock, between any two bytecodes. wardex chains to the
application's previous SIGINT/SIGTERM handler, so a host handler that calls
`close()` — an ordinary shutdown handler — re-enters this module on the thread
sitting inside `install_propagation()`, and a plain `Lock` turns that into a
permanent hang at Ctrl-C. A finalizer running `close()` from a `__del__` is the
same shape. Reentry is still not FREE with an RLock, only survivable, which is
what `_install_generation` is for.

Ordering, since it is not a leaf: `PatchSet.patch()` takes the patch set's own
RLock beneath this one, and a refused patch bumps the diagnostics counters'
RLock beneath that. The order is `Runtime._lock -> _install_lock ->
PatchSet._lock / Counters._lock` and never the reverse. What keeps it acyclic
is that `_patchset()` reaches the client through `Runtime.client`, the property
that is deliberately read WITHOUT the runtime lock; giving that property a lock
would close the cycle.
"""

_install_generation = 0
"""Bumped by every uninstall, so an install can tell one ran underneath it.

An RLock stops the deadlock and nothing else: a signal-driven
`uninstall_propagation()` that lands between two `_install_*` calls acquires
the lock the outer install holds, restores what is patched so far, and returns
into an install that then goes on patching for a wardex that has already torn
itself down. The generation makes that visible — the install reads it once,
re-reads it after each library, and unwinds instead of finishing. Recoverable
either way (the next `init()`/`close()` pair unwinds it), but "patched after
teardown" is a state the host cannot see and should not have to.
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


@functools.lru_cache(maxsize=16)
def _folded_patterns(patterns: tuple[str, ...]) -> tuple[str, ...]:
    """The configured allowlist, case-folded once rather than per request.

    The fold used to live in `PropagationConfig.__post_init__`, which put it
    where a user could see it — and made the config LIE about itself: the
    capitals they typed read back lowercased, the one value mutation the
    round-trips-as-written rule forbids. So the config now keeps the patterns
    exactly as written, and the matcher — the only consumer that needs both
    sides of the comparison folded — folds them here. Cached on the tuple, so
    one configured allowlist is folded once however many outbound calls it
    admits; the config is immutable and a process holds a handful of configs
    over its life, which is what the small bound is sized to.
    """
    return tuple(pattern.lower() for pattern in patterns)


def _matches_target(host: str, patterns: tuple[str, ...]) -> bool:
    """Glob-match an outbound host against the configured target patterns.

    `fnmatchcase` against lowercased inputs, rather than plain `fnmatch`.
    Hostnames are case-insensitive, so `API.MyCorp.com` has to match
    `*.mycorp.com` — and `fnmatch` gets that wrong in two directions at once:
    it defers to `os.path.normcase`, which folds case on Windows and does
    nothing on Linux or macOS. An allowlist that admits a host on one operating
    system and refuses the same host on another is worse than either answer
    consistently applied, because it turns a propagation gap into something
    only one developer's machine can reproduce.

    BOTH sides are folded by the injector: the host here, on every call, and
    the patterns in `_folded_patterns`, once per configured allowlist — the
    config itself round-trips as written and never folds anything.
    """
    lowered = host.lower()
    return any(fnmatch.fnmatchcase(lowered, pattern) for pattern in _folded_patterns(patterns))


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

    `traceparent` gates the whole injection, and `tracestate` does not. The two
    are not symmetric, and the asymmetry is the decision here rather than an
    oversight, because the headers are not symmetric: `traceparent` NAMES the
    trace and `tracestate` only annotates whatever trace is named. A caller who
    set a traceparent owns the context of this request, so we add nothing —
    pairing their traceparent with our tracestate would attribute our vendor
    state to a trace that never carried it.

    A caller who set only a `tracestate` has not established a context at all:
    the spec gives a receiver no way to act on vendor state with no traceparent
    beside it, so the header they wrote is one every conformant hop ignores.
    Declining our traceparent as well would break the trace link on that
    request — losing propagation, the failure direction this SDK does not take
    — to protect a header that was already inert. So we add the traceparent and
    leave their bytes exactly as written.

    That does put wardex's traceparent on the wire next to a tracestate wardex
    did not write, which is the mirror of the mis-attribution above and is
    accepted knowingly: their tracestate was unattributed before we touched the
    request, the alternatives are to destroy host bytes (replace it) or to drop
    the trace (suppress everything), and of the three this is the only one that
    neither loses the host's data nor loses the trace.

    Dropping OUR tracestate in that case is also where the three libraries used
    to disagree: httpx and requests assigned into a case-insensitive mapping
    and replaced the caller's value, while aiohttp `extend`ed a CIMultiDict and
    put both on the wire — one request, two `tracestate` headers, which is not
    a shape the W3C spec defines a reading for.
    """
    if has_header("traceparent"):
        return {}
    headers = _build_inject_headers(host)
    if headers and has_header("tracestate"):
        headers.pop("tracestate", None)
    return headers


def _optional(name: str) -> Any | None:
    """Import a soft dependency, or None if the host does not have it.

    Called from OUTSIDE `_install_lock` — see `install_propagation`.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _install_httpx(httpx: Any | None) -> None:
    if httpx is None or "httpx" in _installed:
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


def _install_requests(requests: Any | None) -> None:
    if requests is None or "requests" in _installed:
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


def _install_aiohttp(aiohttp: Any | None, multidict: Any | None, yarl: Any | None) -> None:
    if aiohttp is None or multidict is None or yarl is None or "aiohttp" in _installed:
        return
    CIMultiDict = getattr(multidict, "CIMultiDict", None)
    URL = getattr(yarl, "URL", None)
    if CIMultiDict is None or URL is None:
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
    """Patch every importable client library. Idempotent, per library.

    Idempotent under concurrency too, not just under repetition — see
    `_install_lock`.

    The soft dependencies are resolved BEFORE the lock is taken. An import runs
    arbitrary third-party module-level code under CPython's import machinery,
    and it touches nothing this lock protects, so holding the lock across it
    only buys a way for `close()` — and the `atexit` hook behind it — to wait
    out a cold aiohttp import, or to hang for good behind a thread wedged in a
    slow import hook. Interpreter shutdown is not a good place to discover that.
    """
    libraries = (
        (_install_httpx, (_optional("httpx"),)),
        (_install_requests, (_optional("requests"),)),
        (_install_aiohttp, (_optional("aiohttp"), _optional("multidict"), _optional("yarl"))),
    )
    with _install_lock:
        generation = _install_generation
        for install, modules in libraries:
            install(*modules)
            if _install_generation != generation:
                # An uninstall re-entered underneath us — a signal handler, or a
                # finalizer, calling `close()` on this very thread. It has
                # already restored everything it could see; what it could not
                # see is the library we patched after it returned. Unwind that
                # rather than leave the host patched by a wardex that has run
                # its teardown, and do not carry on with the rest: the teardown
                # was the later decision.
                _restore_locked()
                return


def _restore_locked() -> None:
    """Undo the patches and drop the bookkeeping. Caller holds `_install_lock`."""
    global _patches
    if _patches is not None:
        _patches.restore_all()
        _patches = None
    _installed.clear()


def uninstall_propagation() -> None:
    """Undo every patch this module installed. Safe to call when none are.

    Restores newest-first, skips any attribute another library has taken over
    since, and guards each restore individually — `PatchSet.restore_all()`.
    The set is dropped rather than reused so the next `install_propagation()`
    picks up the current client's `debug` setting.

    Under the same lock as the install, so a `close()` racing an `init()` on
    another thread cannot restore the attributes between the install's swap and
    its bookkeeping — which would leave `_installed` claiming a library that is
    no longer patched, and the next `install_propagation()` skipping it.
    """
    global _install_generation
    with _install_lock:
        _install_generation += 1
        _restore_locked()
