"""Self-exclusion — prevents the exporter's outbound POST from being re-captured by the byte seam.

A ContextVar guard. While it is set, the seam (ByteSeamInterceptor) captures nothing and no trace
header is injected. The CLIENT sets it around everything on the export path that can reach host
code -- its whole drain (`before_send_envelope`, `Transport.export`, `Transport.flush`) and the
transport's `close()` -- so a `Transport` the host wrote is excluded without knowing this module
exists. That is why it is not exported: a transport needs nothing from it, and a public switch that
stops capture would also be a way to hide host traffic. The built-in transports still enter it
around their own POST, so they stay excluded when called outside a drain.

It lives at the package root, beside `_hub` and `_scope`, because both ends of
the guard sit BELOW the observers: `transport/` sets it and `context/` reads it,
and while it lived under `_interceptors/` those two were the only modules in the
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
