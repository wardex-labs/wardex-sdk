"""Wardex SDK — observability for AI agents."""

from __future__ import annotations

import sys as _sys
from collections.abc import Iterator as _Iterator
from contextlib import contextmanager as _contextmanager
from typing import Any as _Any

from . import _hub, _runtime
from ._client import _FOLLOW_TRANSPORT_TIMEOUT, _SHUTDOWN_TIMEOUT
from ._client import Client as _Client
from ._config import (
    BackendConfig,
    BatchingPolicy,
    PIIPolicy,
    PropagationPolicy,
    RetentionPolicy,
)
from ._config import WardexConfig as _WardexConfig
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
    RetentionClass,
    SnapshotType,
    SpanKind,
    StatusCode,
    ToolExecutionType,
    ToolType,
)
from ._limits import CaptureLimits

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
    "BatchingPolicy",
    "CaptureLimits",
    "PIIPolicy",
    "PropagationPolicy",
    "RetentionPolicy",
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
    "RetentionClass",
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


def init(
    *,
    transport: Transport | None = None,
    intercept: bool = False,
    intercept_hosts: list[str] | None = None,
    **config_kwargs: _Any,
) -> None:
    config = _WardexConfig(
        intercept=intercept,
        intercept_hosts=tuple(intercept_hosts) if intercept_hosts else None,
        **config_kwargs,
    )
    # The config is built FIRST so that a caller's bad keyword still raises the
    # same TypeError it raises with a working wheel. Degraded mode must not turn
    # a programming error into a shrug.
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
    # Transport resolution, in precedence order: an explicit `transport=`
    # carries its own address and wins outright; otherwise a configured
    # `backend.endpoint` builds the default OTLP/HTTP exporter; otherwise
    # NoOpTransport. When both are given the endpoint loses, and under `debug`
    # it says so — a config value that loses a precedence fight in silence is
    # indistinguishable from one that was honoured.
    if transport is not None:
        resolved_transport = transport
        if config.backend.endpoint and config.debug:
            print(
                "[wardex] backend endpoint ignored: transport= carries its own address",
                file=_sys.stderr,
            )
    elif config.backend.endpoint:
        resolved_transport = OtlpHttpTransport(config.backend.endpoint, debug=config.debug)
    else:
        resolved_transport = NoOpTransport()
    resolved_transport.set_pii_policy(
        config.pii.mode.value,
        tuple(sorted(c.value for c in config.pii.disabled_categories)),
    )
    # The transport encodes, so the encoder's ceilings are its business too --
    # `max_otlp_attribute_bytes` and `max_otlp_request_bytes` are configured
    # here and enforced there, and a transport that never received them would
    # advertise both knobs and honour neither.
    resolved_transport.set_limits(config.limits.to_native())
    if config.pii.mode.value == "off" and config.pii.disabled_categories and config.debug:
        print(
            "[wardex] pii disabled_categories has no effect when pii mode is OFF",
            file=_sys.stderr,
        )
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


def flush(timeout: float = _FOLLOW_TRANSPORT_TIMEOUT) -> None:
    """Send everything buffered and wait for it.

    With no argument the budget is the transport's own configured timeout -- a
    bare `flush()` is "send what you have, I will wait", so it does not cap the
    POST below what the transport was configured for (an
    `OtlpHttpTransport(timeout=10.0)` gets its 10 seconds). Pass a number for a
    real wall-clock bound: `flush(2.0)` returns within about two seconds
    whatever the transport was configured for. `close()` is the other operation
    and keeps its own tight default; see below.

    The sentinel default is forwarded as it stands, and every layer below asks
    it what it IS rather than comparing it to a known object: any budget wardex
    picked for itself is an `_UnnamedTimeout`, and the one that means "follow
    the transport" says so in a field. So the distinction this signature draws
    between "no argument" and an explicit number that happens to equal the
    default survives every layer it passes through -- and survives being
    copied, deepcopied or pickled on the way, which an identity check could not
    have.
    """
    client = _hub.get_client()
    if client is not None:
        client.flush(timeout)


def close(timeout: float = _SHUTDOWN_TIMEOUT) -> None:
    """Uninstall everything, drain what is buffered, and close the transport.

    `timeout` bounds each shutdown step and defaults to 5 seconds. Unlike
    `flush()` this default does NOT follow the transport, deliberately: close()
    runs when the process is going away, and an unbounded one ate the whole
    termination grace period on the way out. Pass a larger budget when
    keeping the tail matters more than exiting promptly.

    The default is a sentinel carrying that same 5.0, and `Client.close` asks
    its TYPE rather than comparing it to this one object, so it can tell "wardex
    picked 5 seconds" from "the host asked for 5 seconds". Only the second is a
    number anyone chose, and only the second can be blamed for an export it cuts
    short. Asking the type is what closed the door an identity check left open:
    the signal handler's own 2s budget is not this object either, and used to be
    blamed on the host. Nothing on this path compares budgets by identity any
    more, which is what lets the default be copied and still mean what it says.
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
