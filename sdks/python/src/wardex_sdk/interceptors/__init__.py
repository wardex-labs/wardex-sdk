"""Interceptors: transport-layer I/O interceptors — selected via Config.interceptors."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from .._enums import InterceptorName
from ._base import InterceptorInterface
from ._registry import InterceptorRegistry, get_registry
from ._ssl import SSLInterceptor

if TYPE_CHECKING:
    from .._client import Client
    from .._config import WardexConfig

__all__ = [
    "InterceptorInterface",
    "InterceptorRegistry",
    "SSLInterceptor",
    "get_registry",
    "install_configured_interceptors",
]


def _ssl_interceptor(config: WardexConfig) -> InterceptorInterface:
    # Reached through the MODULE and not through this package's own re-export
    # above, so the class is resolved when the seam is built rather than when
    # this package was first imported. `init()` used to import inside its own
    # body and therefore had that property for free; taking the name bound at
    # line 12 instead would freeze `_ssl.SSLInterceptor` as it stood at import
    # time, and the substitution the isolation suite performs — swapping in a
    # seam whose `install()` raises, to prove one broken seam does not cost the
    # other two — would silently install the real one and measure nothing.
    from . import _ssl

    return _ssl.SSLInterceptor()


# The two builders below import INSIDE the call, and that is not a style choice:
# `_mcp_stdio` needs anyio, so a module-level import here would make the whole
# package unimportable on a host that has none — and `close()` imports this
# package unconditionally, from teardown paths that cannot handle an ImportError.
def _mcp_stdio_interceptor(config: WardexConfig) -> InterceptorInterface:
    from ._mcp_stdio import McpStdioInterceptor

    return McpStdioInterceptor()


def _socket_interceptor(config: WardexConfig) -> InterceptorInterface:
    from ._socket import RawSocketInterceptor

    return RawSocketInterceptor(list(config.intercept_hosts or ()))


#: name -> builder, and the ITERATION ORDER is the install order. Both facts
#: live in this one table so that adding an interceptor is one module plus one
#: row, and so that `config.interceptors` cannot reorder the seams: SSL patches
#: `ssl.SSLSocket`, the socket seam patches `socket.socket` underneath it, and
#: which one wraps first is not a caller's decision to make.
_INTERCEPTORS: dict[InterceptorName, Callable[[WardexConfig], InterceptorInterface]] = {
    InterceptorName.SSL: _ssl_interceptor,
    InterceptorName.MCP_STDIO: _mcp_stdio_interceptor,
    InterceptorName.SOCKET: _socket_interceptor,
}


def install_configured_interceptors(client: Client | None, config: WardexConfig) -> None:
    """Install the interceptors the config asks for. `intercept` is the switch.

    `config.interceptors` chooses WHICH; `None` means every one there is, which
    is what `intercept=True` did for as long as the field was declared and never
    read. Selecting a subset used to require editing `init()`.

    The wanted set is built by walking the TABLE and keeping what was asked for,
    never by walking the request. That is what makes the loop total: every name
    it reaches has a builder by construction, so no lookup here can fail and no
    silent skip is needed to cover one. A value that is not an `InterceptorName`
    at all is refused earlier, by `WardexConfig`, where the mistake was made.

    No guard around the build/install pair, deliberately. `InterceptorRegistry.
    install` is total — it isolates the failure, rolls the half-install back and
    keeps going — so a second one here would catch nothing and would put the
    recovery in two places, which is how the two stop agreeing. What the
    registry cannot cover is the IMPORT inside each builder, so each interceptor
    module is responsible for staying importable without its optional
    third-party seam (see `_mcp_stdio` on anyio).
    """
    if not config.intercept:
        if config.interceptors is not None and config.debug:
            # The same shape as `init()`'s pii_disabled_categories line, and for
            # the same reason: a refinement of a switch that is off is not an
            # error, but silence about it is how a user concludes the selection
            # was honoured.
            print(
                "[wardex] interceptors=... has no effect without intercept=True",
                file=sys.stderr,
            )
        return
    wanted = (
        tuple(_INTERCEPTORS)
        if config.interceptors is None
        else tuple(name for name in _INTERCEPTORS if name in config.interceptors)
    )
    for name in wanted:
        get_registry().install(_INTERCEPTORS[name](config), client)
