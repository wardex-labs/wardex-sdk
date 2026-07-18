"""L1 framework adapters — auto-detected at init(), opt-out via Config.adapters."""

from __future__ import annotations

import importlib.util
import sys
from typing import TYPE_CHECKING

from .._enums import AdapterName
from ._registry import get_registry

if TYPE_CHECKING:
    from .._client import Client
    from .._config import WardexConfig

# AdapterName -> distribution package to probe for auto-detection
_DETECT_PACKAGES: dict[AdapterName, str] = {
    AdapterName.ANTHROPIC_AGENT_SDK: "claude_agent_sdk",
}


def _detect_package(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:  # noqa: BLE001 — detection must never raise
        return False


def _make_adapter(name: AdapterName):
    if name is AdapterName.ANTHROPIC_AGENT_SDK:
        from ._anthropic_agent_sdk import AnthropicAgentSdkAdapter

        return AnthropicAgentSdkAdapter()
    return None


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
