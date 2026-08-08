"""Interceptors: transport-layer I/O interceptors — selected via Config.interceptors."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from .._enums import InterceptorName
from ._base import InterceptorInterface
from ._registry import InterceptorRegistry, get_registry

if TYPE_CHECKING:
    from .._client import Client
    from .._config import WardexConfig

__all__ = [
    "InterceptorInterface",
    "InterceptorRegistry",
    "get_registry",
    "install_configured_interceptors",
]


# Every builder below imports INSIDE the call, and none of it is a style choice.
#
# `_ssl` was the one exception and it was the wrong one twice over. Imported at
# module level, `ssl.SSLInterceptor` would be resolved when this PACKAGE was
# first imported rather than when the seam is built — so the substitution the
# isolation suite performs (swapping in a seam whose `install()` raises, to
# prove one broken seam does not cost the other two) would silently install the
# real one and measure nothing. And importing it here dragged the whole TLS seam
# — with the native extension underneath it — into every import of this package,
# including the ones `Runtime` makes from teardown paths that exist precisely to
# work when that extension does not.
#
# `_mcp_stdio` needs anyio, so a module-level import would make the package
# unimportable on a host that has none.
def _ssl_interceptor(config: WardexConfig) -> InterceptorInterface:
    from . import _ssl

    return _ssl.SSLInterceptor()


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
