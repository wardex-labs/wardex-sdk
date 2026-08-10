"""L1 framework adapters — auto-detected at init(), opt-out via Config.adapters."""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING, NamedTuple

from .._enums import AdapterName
from ._registry import get_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from .._client import Client
    from .._config import WardexConfig
    from ._base import AdapterInterface


class _Registration(NamedTuple):
    """What this SDK knows about one adapter: how to FIND its framework, and
    how to BUILD it.

    ONE row, and that is the whole point. These two facts used to live in a
    detection table and an `if`-chain, and a registration split across two
    places can be written half-way in either direction: a name in the detection
    table with no branch auto-detects the framework and then installs nothing,
    silently, while a branch with no table entry is unreachable unless the user
    names the adapter explicitly. Neither half-write fails anything — they are
    the shape of "the adapter shipped and captured nothing".
    """

    detect: str
    """The distribution module probed to decide whether the framework is here."""

    build: Callable[[], AdapterInterface]
    """Constructs the adapter, importing its module on the way.

    A thunk rather than a dotted path, so that the import stays a LITERAL one.
    The layering rules read this package's imports statically and can follow
    `importlib.import_module("wardex_sdk._adapters._x")`; a name computed from a
    table row is reported as unauditable, and an adapter registry is the last
    place that should be the first hole in those rules.
    """


def _anthropic_agent_sdk() -> AdapterInterface:
    from ._anthropic_agent_sdk import AnthropicAgentSdkAdapter

    return AnthropicAgentSdkAdapter()


def _langgraph() -> AdapterInterface:
    from ._langgraph import LangGraphAdapter

    return LangGraphAdapter()


#: Every adapter this SDK ships, in install order. Adding one is this row plus
#: its module — nothing else in this file, and no branch anywhere.
_ADAPTERS: dict[AdapterName, _Registration] = {
    AdapterName.ANTHROPIC_AGENT_SDK: _Registration("claude_agent_sdk", _anthropic_agent_sdk),
    AdapterName.LANGGRAPH: _Registration("langgraph", _langgraph),
}

#: AdapterName -> distribution package to probe for auto-detection. DERIVED from
#: `_ADAPTERS` and never maintained beside it: a second literal table is exactly
#: the drift the single row above exists to make impossible.
_DETECT_PACKAGES: dict[AdapterName, str] = {name: row.detect for name, row in _ADAPTERS.items()}


def _detect_package(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:  # noqa: BLE001 — detection must never raise
        return False


def _make_adapter(name: AdapterName) -> AdapterInterface | None:
    """The adapter for `name`, or None when this SDK ships none for it.

    None is an ANSWER rather than a failure: `AdapterName` names frameworks
    wardex intends to support before their adapter exists, so a user who asks
    for one of those by name gets the same nothing a user whose framework is
    absent gets, and `init()` carries on.
    """
    row = _ADAPTERS.get(name)
    return None if row is None else row.build()


def install_configured_adapters(client: Client | None, config: WardexConfig) -> None:
    if config.adapters is not None:
        wanted = list(config.adapters)
    else:
        wanted = [name for name, pkg in _DETECT_PACKAGES.items() if _detect_package(pkg)]
    for name in wanted:
        try:
            adapter = _make_adapter(name)
            if adapter is not None:
                get_registry().install(adapter, client)
        except Exception as exc:  # noqa: BLE001 — a broken adapter must not break init()
            print(f"[wardex] adapter {name.value} failed to load ({exc})", file=sys.stderr)
            continue
