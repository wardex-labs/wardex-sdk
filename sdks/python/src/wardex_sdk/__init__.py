"""Wardex SDK — observability for AI agents."""

from __future__ import annotations

import sys as _sys
import warnings as _warnings
from collections.abc import Iterator as _Iterator
from collections.abc import Sequence as _Sequence
from contextlib import contextmanager as _contextmanager
from typing import Any as _Any
from urllib.parse import urlsplit as _urlsplit

from . import _hub, _runtime
from ._client import _FOLLOW_TRANSPORT_TIMEOUT
from ._client import Client as _Client
from ._config import (
    BackendConfig,
    BatchingConfig,
    PIIConfig,
    PropagationConfig,
    WardexConfigWarning,
)
from ._config import _resolve_config as _resolve_config_from_env
from ._enums import (
    AdapterName,
    AgentType,
    CaptureMode,
    InterceptorName,
    OperationName,
    OutputType,
    PIICategory,
    PIIMode,
    ProviderName,
    SnapshotType,
    SpanKind,
    StatusCode,
    ToolExecutionType,
    ToolType,
)
from ._limits import LimitsConfig

# NOT public: imported for use by `init()` and `close()` below, and the
# underscore aliases are what make the non-export literal — `wardex_sdk.
# NATIVE_OK` no longer resolves at all, instead of resolving while being
# deliberately absent from `__all__`. Said here rather than left to be guessed,
# because the pair looks like a supported feature probe and is not one: the
# degraded mode it describes is ANNOUNCED, once, on stderr by `init()`, and a
# host does not have to ask. Their home is `wardex_sdk._native`, private, and
# they are free to change shape there.
from ._native import NATIVE_OK as _NATIVE_OK
from ._native import unavailable_reason as _unavailable_reason
from ._scope import Scope, UserInfo
from ._snapshot_api import capture_state_snapshot
from ._tracing import agent, span, task, tool, trace, workflow
from ._types import (
    AgentAttributes,
    CallSite,
    ConversationContext,
    GenAIAttributes,
    InputRef,
    ToolAttributes,
    ToolDefinition,
    ToolDefinitionSet,
)

# Private on purpose: `before_send=`'s annotation needs the protocol, but the
# callback type's public spelling is a later slice of this API batch (it is
# renamed with the hook). The signature-walk test records the debt in
# `_KNOWN_UNEXPORTED`.
from ._types import BeforeSendCallback as _BeforeSendCallback
from ._version import __version__
from .context._asgi import WardexMiddleware
from .context._contextvar import run_in_context
from .context._propagate import (
    continue_from_otel,
    continue_trace,
    get_trace_headers,
    get_traceparent,
)
from .context._wsgi import WardexWSGIMiddleware
from .transport._base import Transport
from .transport._console import ConsoleTransport
from .transport._noop import NoOpTransport
from .transport._otlp_http import OtlpHttpTransport

__all__ = [
    "__version__",
    "init",
    "trace",
    "span",
    "workflow",
    "agent",
    "task",
    "tool",
    "capture_state_snapshot",
    "set_tag",
    "set_user",
    "isolation_scope",
    "new_scope",
    "flush",
    "close",
    "run_in_context",
    "continue_trace",
    "continue_from_otel",
    "get_traceparent",
    "get_trace_headers",
    "WardexMiddleware",
    "WardexWSGIMiddleware",
    # Config groups — every one of them is passed to `init()` by name
    "BackendConfig",
    "BatchingConfig",
    "LimitsConfig",
    "PIIConfig",
    "PropagationConfig",
    "WardexConfigWarning",
    # Enums — importable directly from user code
    "AdapterName",
    "AgentType",
    "CaptureMode",
    "InterceptorName",
    "OperationName",
    "OutputType",
    "PIICategory",
    "PIIMode",
    "ProviderName",
    "SnapshotType",
    "SpanKind",
    "StatusCode",
    "ToolExecutionType",
    "ToolType",
    # Types
    "UserInfo",
    "Scope",
    "AgentAttributes",
    "CallSite",
    "ConversationContext",
    "GenAIAttributes",
    "InputRef",
    "ToolAttributes",
    "ToolDefinition",
    "ToolDefinitionSet",
    # Transport
    "Transport",
    "NoOpTransport",
    "ConsoleTransport",
    "OtlpHttpTransport",
]


def _traces_endpoint(endpoint: str) -> str:
    """The URL the default OTLP/HTTP transport actually POSTs to.

    THE ENDPOINT RULE (documented on `BackendConfig.endpoint`): a configured
    URL whose path is empty or `/` is a collector base address, so the OTLP
    traces path `/v1/traces` is appended; a URL with an explicit path is used
    verbatim. Applied here, at transport construction, and NEVER written back
    into `config.backend.endpoint` — the config reads back as written.
    """
    if _urlsplit(endpoint).path in ("", "/"):
        return endpoint.rstrip("/") + "/v1/traces"
    return endpoint


def init(
    *,
    transport: Transport | None = None,
    backend: BackendConfig | None = None,
    pii: PIIConfig | None = None,
    batching: BatchingConfig | None = None,
    limits: LimitsConfig | None = None,
    propagation: PropagationConfig | None = None,
    adapters: tuple[AdapterName, ...] | None = None,
    interceptors: tuple[InterceptorName, ...] | None = None,
    intercept: bool = True,
    intercept_hosts: _Sequence[str] | None = None,
    capture_mode: CaptureMode = CaptureMode.AGENT,
    release: str | None = None,
    environment: str | None = None,
    before_send: _BeforeSendCallback | None = None,
    debug: bool = False,
) -> None:
    """Initialize wardex: resolve the config, build the client, install it.

    The keyword parameters are `WardexConfig`'s fields plus `transport=` — a
    drift test holds the two signatures together — and `None` for a config
    group means that group's defaults.

    RESOLUTION ORDER, per field: an explicit argument wins; an unset one falls
    back to its environment variable; only then does the default apply. The
    frozen env contract:

        backend.api_key      WARDEX_API_KEY
        backend.endpoint     WARDEX_ENDPOINT, else
                             OTEL_EXPORTER_OTLP_TRACES_ENDPOINT, else
                             OTEL_EXPORTER_OTLP_ENDPOINT
        release              WARDEX_RELEASE
        environment          WARDEX_ENVIRONMENT
        debug                WARDEX_DEBUG=true (case-insensitive)

    so `wardex.init()` with only `WARDEX_ENDPOINT` set is a working first run.
    What the environment resolved is written into the config the client
    carries — `client.config` answers with the resolved values, not the bare
    arguments. `WARDEX_DEBUG` can only turn `debug` ON: `debug=False` is this
    signature's default and therefore cannot veto the variable.

    TRANSPORT RESOLUTION, in precedence order: an explicit `transport=`
    carries its own address and wins outright; otherwise a configured
    `backend.endpoint` builds the default OTLP/HTTP exporter against it (a
    bare collector address gets `/v1/traces` appended; an explicit path is
    used verbatim — see `BackendConfig.endpoint`); with neither, wardex
    installs `NoOpTransport`, captures into nothing, and says so once on
    stderr. When a setting loses a precedence fight — the endpoint under an
    explicit `transport=`, PII exemptions under `PIIMode.OFF`, an
    `interceptors=` selection under `intercept=False` — a
    `WardexConfigWarning` is emitted, because a config value that loses in
    silence is indistinguishable from one that was honoured.

    `intercept=True` is the default: `init()` is the consent and
    zero-instrumentation capture is the product. Mutation of outbound traffic
    (`propagation`) stays opt-in; PII masking stays on.
    """
    config = _resolve_config_from_env(
        backend=backend,
        pii=pii,
        batching=batching,
        limits=limits,
        propagation=propagation,
        adapters=adapters,
        interceptors=interceptors,
        intercept=intercept,
        intercept_hosts=intercept_hosts,
        capture_mode=capture_mode,
        release=release,
        environment=environment,
        before_send=before_send,
        debug=debug,
    )
    # The config is built FIRST so that a caller's bad argument still raises
    # the same TypeError/ValueError it raises with a working wheel. Degraded
    # mode must not turn a programming error into a shrug.
    if not _NATIVE_OK:
        # Without the core there is nothing to capture WITH, and the failure is
        # one nobody can fix from Python. Returning here is what makes degraded
        # mode complete rather than half-applied: no client is built, so no
        # interceptor, adapter, propagation patch, atexit hook or signal handler
        # is ever installed, and every one of those modules -- each of which
        # still reaches the extension -- stays unreachable by construction.
        #
        # The stderr line is not optional. Silently doing nothing is the other
        # failure mode and it is the worse one: it looks exactly like a backend
        # that is up and receiving no traffic, so nobody goes looking.
        print(
            f"[wardex] native extension unavailable, wardex is disabled: "
            f"nothing will be captured or exported ({_unavailable_reason()})",
            file=_sys.stderr,
        )
        return
    # The three known conflict cases, announced UNCONDITIONALLY as warnings —
    # they used to hide behind `debug`, which printed them in exactly the
    # configuration nobody runs. Each is a setting another setting disables:
    # legal, but never to be confused with a setting that was honoured.
    if transport is not None and config.backend.endpoint:
        _warnings.warn(
            "backend endpoint ignored: transport= carries its own address",
            WardexConfigWarning,
            stacklevel=2,
        )
    if config.pii.mode is PIIMode.OFF and config.pii.disabled_categories:
        _warnings.warn(
            "pii disabled_categories has no effect when pii mode is OFF",
            WardexConfigWarning,
            stacklevel=2,
        )
    if config.interceptors is not None and not config.intercept:
        # Detected here rather than at install time: `_interceptors/` keeps the
        # matching behavior (a selection under intercept=False installs
        # nothing) and this is its one announcement.
        _warnings.warn(
            "interceptors=... has no effect without intercept=True",
            WardexConfigWarning,
            stacklevel=2,
        )
    if config.debug:
        # One line, at install time, with the RESOLVED config — the env
        # fallbacks and canonicalized collections included. `api_key` is
        # `repr=False`, so no credential can ride along.
        print(f"[wardex] resolved config: {config!r}", file=_sys.stderr)
    if transport is not None:
        resolved_transport = transport
    elif config.backend.endpoint:
        resolved_transport = OtlpHttpTransport(
            _traces_endpoint(config.backend.endpoint), debug=config.debug
        )
    else:
        resolved_transport = NoOpTransport()
        # Unconditional, like the degraded-mode line and for the same reason: a
        # wardex that captures into nothing looks exactly like a backend that
        # is up and receiving no traffic, so nobody goes looking.
        print(
            "[wardex] no transport or backend.endpoint configured: capturing, exporting nothing",
            file=_sys.stderr,
        )
    resolved_transport.set_pii_policy(
        config.pii.mode.value,
        tuple(sorted(c.value for c in config.pii.disabled_categories)),
    )
    # The transport encodes, so the encoder's ceilings are its business too --
    # `max_otlp_attribute_bytes` and `max_otlp_request_bytes` are configured
    # here and enforced there, and a transport that never received them would
    # advertise both knobs and honour neither.
    resolved_transport.set_limits(config.limits.to_native())
    client = _Client(config, resolved_transport)
    # One call, because there is one install order and the `Runtime` owns it:
    # the client slot, atexit, the signal handlers, the interceptor and adapter
    # registries and the propagation patches, in the order `close()` undoes them
    # in. Spelling the sequence out here is what let this function and the
    # teardown paths drift apart about what "installed" means.
    _runtime.runtime().install(client, config)


def set_tag(key: str, value: str) -> None:
    _hub.get_global_scope().set_tag(key, value)


def set_user(user: UserInfo) -> None:
    _hub.get_global_scope().set_user(user)


@_contextmanager
def isolation_scope() -> _Iterator[_Any]:
    with _hub.isolation_scope() as s:
        yield s


@_contextmanager
def new_scope() -> _Iterator[_Any]:
    with _hub.new_scope() as s:
        yield s


def flush(timeout: float | None = None) -> None:
    """Send everything buffered and wait for it.

    `timeout=None` — the default — means FOLLOW THE TRANSPORT'S CONFIGURED
    TIMEOUT: a bare `flush()` is "send what you have, I will wait", so it does
    not cap the POST below what the transport was configured for (an
    `OtlpHttpTransport(timeout=10.0)` gets its 10 seconds). Pass a number for a
    real wall-clock bound: `flush(2.0)` returns within about two seconds
    whatever the transport was configured for. `close()` is the other operation
    and follows `batching.shutdown_timeout` instead; see below.

    `None` is mapped to an internal sentinel here, at the public boundary, and
    every layer below asks that budget what it IS rather than comparing it to
    a known object: any budget wardex picked for itself is an
    `_UnnamedTimeout`, and the one that means "follow the transport" says so
    in a field. So the distinction this signature draws between "no argument"
    and an explicit number survives every layer it passes through -- and
    survives being copied, deepcopied or pickled on the way, which an identity
    check could not have.
    """
    client = _hub.get_client()
    if client is not None:
        client.flush(_FOLLOW_TRANSPORT_TIMEOUT if timeout is None else timeout)


def close(timeout: float | None = None) -> None:
    """Uninstall everything, drain what is buffered, and close the transport.

    `timeout` bounds each shutdown step. `None` — the default — means FOLLOW
    `batching.shutdown_timeout` (5 seconds unless configured): the number a
    bare `close()` spends has one home, on the config, shared with the atexit
    hook and re-init's teardown of the previous client. Unlike `flush()` this
    default does NOT follow the transport, deliberately: close() runs when the
    process is going away, and an unbounded one ate the whole termination
    grace period on the way out. Pass a number (`close(30.0)`) for a
    caller-owned budget when keeping the tail matters more than exiting
    promptly.

    `None` is expressed downward by NOT passing a budget: the runtime hands
    the client its own default, a sentinel that resolves to the client's
    `batching.shutdown_timeout`, and `Client.close` asks that budget's TYPE
    rather than comparing it to any one object — so it can tell "wardex picked
    this number" from "the host asked for this number". Only the second is a
    number anyone chose, and only the second can be blamed for an export it
    cuts short. Nothing on this path compares budgets by identity, which is
    what lets a default be copied and still mean what it says.
    """
    if not _NATIVE_OK:
        # `init()` returned before installing anything, so there is nothing to
        # uninstall and no client to drain. The return still comes first rather
        # than being left to the runtime to discover: `_adapters/` reaches the
        # extension at IMPORT time, so a `close()` in a host's shutdown path --
        # an atexit hook, a `finally`, a test teardown -- would raise ImportError
        # out of a teardown that cannot handle it, and the process would die on
        # the way out instead of on the way in.
        return
    # The uninstall ORDER is the runtime's, not this function's. It used to be
    # written out here and again in the re-init/atexit path, with the same
    # order-sensitive reasoning copied onto both -- and the copies could not see
    # each other's state, so `close()` dropped the propagation patches and the
    # atexit path did not.
    _runtime.runtime().teardown(timeout=timeout)


# Root-namespace hygiene: the stdlib names above are imported under
# underscore aliases rather than deleted after use, because `typing.
# get_type_hints` resolves this module's string annotations against its
# globals — a deleted `_Any` would make `init`'s signature unresolvable to
# every consumer that asks. `annotations` is different: it is the
# future-feature object the first import statement binds, nothing resolves
# through it, and it is a leak like any other. `tests/test_public_surface.py`
# holds the resulting invariant: the public namespace is `__all__`, the
# dunders, underscore-private names, and the three public subpackages
# (`transport`, `context`, `testing`) — nothing else.
del annotations
