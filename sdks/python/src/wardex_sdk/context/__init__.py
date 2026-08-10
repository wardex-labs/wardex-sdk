"""User-facing context helpers: carrying a trace across tasks and processes.

Only the helpers a host application calls live on this package's surface. The
scope plumbing (`activate_span`, `install_span`, ...) is internal vocabulary —
`wardex_sdk.context._contextvar` is its home, and internal consumers import it
from there directly.
"""

from ._contextvar import run_in_context
from ._propagate import (
    continue_from_otel,
    continue_trace,
    get_trace_headers,
    get_traceparent,
)

__all__ = [
    "continue_from_otel",
    "continue_trace",
    "get_trace_headers",
    "get_traceparent",
    "run_in_context",
]
