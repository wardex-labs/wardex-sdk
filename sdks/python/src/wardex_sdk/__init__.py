"""Wardex SDK — observability for AI agents."""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from . import _hub, _wardex_native  # noqa: F401  (verifies native module loads)
from ._client import Client
from ._config import WardexConfig
from ._enums import (
    AdapterName,
    CaptureMode,
    Direction,
    InterceptorName,
    Modality,
    OperationName,
    OutputType,
    PIICategory,
    PIIMode,
    Protocol,
    ProviderName,
    RetentionClass,
    SessionStatus,
    SpanKind,
    StatusCode,
    ToolType,
)
from ._limits import CaptureLimits
from ._scope import UserInfo
from ._tracing import agent, span, task, tool, trace, workflow
from ._types import (
    GenAIAttributes,
    InputRef,
    InternalStateSnapshot,
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
    "CaptureLimits",
    # Enums — importable directly from user code
    "AdapterName",
    "CaptureMode",
    "Direction",
    "InterceptorName",
    "Modality",
    "OperationName",
    "OutputType",
    "PIICategory",
    "PIIMode",
    "Protocol",
    "ProviderName",
    "RetentionClass",
    "SessionStatus",
    "SpanKind",
    "StatusCode",
    "ToolType",
    # Types
    "UserInfo",
    "GenAIAttributes",
    "InputRef",
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
    **config_kwargs: Any,
) -> None:
    config = WardexConfig(
        intercept=intercept,
        intercept_hosts=tuple(intercept_hosts) if intercept_hosts else None,
        **config_kwargs,
    )
    resolved_transport = transport or NoOpTransport()
    resolved_transport.set_pii_policy(
        config.pii_mode.value,
        tuple(sorted(c.value for c in config.pii_disabled_categories)),
    )
    if config.pii_mode.value == "off" and config.pii_disabled_categories and config.debug:
        print(
            "[wardex] pii_disabled_categories has no effect when pii_mode=OFF",
            file=sys.stderr,
        )
    client = Client(config, resolved_transport)
    from . import _lifecycle

    _lifecycle.install(client, config)
    _hub.set_client(client)
    if config.intercept:
        from .interceptors._registry import get_registry
        from .interceptors._ssl import SSLInterceptor

        get_registry().install(SSLInterceptor(), client)

        from .interceptors._mcp_stdio import McpStdioInterceptor

        get_registry().install(McpStdioInterceptor(), client)

        from .interceptors._socket import RawSocketInterceptor

        get_registry().install(RawSocketInterceptor(list(config.intercept_hosts or ())), client)

    from .adapters import install_configured_adapters

    install_configured_adapters(client, config)

    from .context._inject import install_propagation, uninstall_propagation

    uninstall_propagation()  # re-init: drop patches from a previous init
    if config.propagate_trace:
        install_propagation()


def set_tag(key: str, value: str) -> None:
    _hub.get_global_scope().set_tag(key, value)


def set_user(user: UserInfo) -> None:
    _hub.get_global_scope().set_user(user)


@contextmanager
def isolation_scope() -> Iterator[Any]:
    with _hub.isolation_scope() as s:
        yield s


@contextmanager
def new_scope() -> Iterator[Any]:
    with _hub.new_scope() as s:
        yield s


def capture_state_snapshot(
    *,
    snapshot_type: str = "turn_start",
    turn_index: int = 0,
    conversation_state: bytes = b"",
    input_refs: Iterable[InputRef | tuple[str, str]] = (),
    attributes: Mapping[str, str | int | float | bool] | None = None,
    tool_definitions: ToolDefinitionSet | None = None,
) -> None:
    client = _hub.get_client()
    if client is None:
        return
    scope = _hub.get_current_scope()
    active = scope.active_span_context
    if active is None:
        return
    norm_refs = tuple(
        r if isinstance(r, InputRef) else InputRef(key=r[0], content_hash=r[1]) for r in input_refs
    )
    norm_attrs = tuple(attributes.items()) if attributes else ()
    snapshot = InternalStateSnapshot(
        trace_id=active.trace_id,
        span_id=active.span_id,
        timestamp_ns=time.time_ns(),
        snapshot_type=snapshot_type,
        turn_index=turn_index,
        conversation_state=conversation_state,
        tool_definitions=tool_definitions,
        attributes=norm_attrs,
        input_refs=norm_refs,
    )
    client.capture_snapshot(snapshot)


def flush(timeout: float = 5.0) -> None:
    client = _hub.get_client()
    if client is not None:
        client.flush(timeout)


def close(timeout: float = 5.0) -> None:
    from .interceptors._registry import get_registry

    # Interceptor uninstall flushes (client.capture_span) any pending WS sessions.
    # This must run before client.close() sets _closed=True, or the WS-close span
    # would be blocked and lost.
    get_registry().uninstall_all()

    from .adapters._registry import get_registry as _adapter_registry

    _adapter_registry().uninstall_all()

    from .context._inject import uninstall_propagation

    uninstall_propagation()

    client = _hub.get_client()
    if client is not None:
        client.close(timeout)
