"""Self-exclusion — prevents the exporter's outbound POST from being re-captured by the byte seam.

A ContextVar guard. The transport sets it while POSTing → the seam (ByteSeamInterceptor) skips it.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_exporting: ContextVar[bool] = ContextVar("wardex_exporting", default=False)


def is_suppressed() -> bool:
    return _exporting.get()


@contextmanager
def suppress_capture() -> Iterator[None]:
    token = _exporting.set(True)
    try:
        yield
    finally:
        _exporting.reset(token)
