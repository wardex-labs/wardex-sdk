"""Self-exclusion — prevents the exporter's outbound POST from being re-captured by the byte seam.

A ContextVar guard. The transport sets it while POSTing → the seam (ByteSeamInterceptor) skips it.

It lives at the package root, beside `_hub` and `_scope`, because both ends of
the guard sit BELOW the observers: `transport/` sets it and `context/` reads it,
and while it lived under `interceptors/` those two were the only modules in the
SDK importing a layer above their own. The guard itself never had any
interceptor-specific content — it is a flag and a context manager — so the
inversion bought nothing and cost the one-way arrow in design §3.1.
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
