"""L1 framework adapters — auto-detected at init(), selected via `AdaptersConfig.enabled`."""

from __future__ import annotations

import importlib.util
from operator import attrgetter
from typing import TYPE_CHECKING, NamedTuple

from .._assembly import counters, diag_info, diag_warning, report_once
from .._config import AdaptersConfig, _non_default_adapter_options
from .._enums import AdapterName
from ._probe import probe
from ._registry import get_registry

if TYPE_CHECKING:
    from collections.abc import Callable

    from .._client import Client
    from .._config import WardexConfig
    from ._base import AdapterInterface


class _Registration(NamedTuple):
    """What this SDK knows about one adapter: how to FIND its framework, how to
    BUILD it, and how to PICK its options.

    ONE row, and that is the whole point. These facts used to live in a
    detection table and an `if`-chain, and a registration split across two
    places can be written half-way in either direction: a name in the detection
    table with no branch auto-detects the framework and then installs nothing,
    silently, while a branch with no table entry is unreachable unless the user
    names the adapter explicitly. Neither half-write fails anything — they are
    the shape of "the adapter shipped and captured nothing".
    """

    detect: str
    """The framework's import name, probed to decide whether it is here."""

    distribution: str
    """The framework's distribution name, as `importlib.metadata` knows it.

    What the pre-build probe checks `detect` against: a module that resolves
    somewhere other than this distribution's package is a project-local
    shadow, declined before the adapter is built — and so before its
    `install()` could import, and thereby EXECUTE, the local package.
    """

    build: Callable[[], AdapterInterface]
    """Constructs the adapter, importing its module on the way.

    A thunk rather than a dotted path, so that the import stays a LITERAL one.
    The layering rules read this package's imports statically and can follow
    `importlib.import_module("wardex_sdk._adapters._x")`; a name computed from a
    table row is reported as unauditable, and an adapter registry is the last
    place that should be the first hole in those rules.
    """

    options: Callable[[AdaptersConfig], object] | None = None
    """Picks this adapter's own options group off `AdaptersConfig`, keyed by
    the canonical name — the per-adapter field name equals the `AdapterName`
    value equals `adapter.name()`, so the accessor is an `attrgetter` on that
    one identifier. `None` for an adapter that has no config class yet (a
    per-adapter class is created with its first real option, never ahead of
    it), and `AdapterContext.options` is `None` for it.
    """


def _anthropic_agent_sdk() -> AdapterInterface:
    from ._anthropic_agent_sdk import AnthropicAgentSdkAdapter

    return AnthropicAgentSdkAdapter()


def _langgraph() -> AdapterInterface:
    from ._langgraph import LangGraphAdapter

    return LangGraphAdapter()


def _openai_agents() -> AdapterInterface:
    from ._openai_agents import OpenAIAgentsAdapter

    return OpenAIAgentsAdapter()


#: Every adapter this SDK ships, in install order. Adding one is this row plus
#: its module — nothing else in this file, and no branch anywhere.
_ADAPTERS: dict[AdapterName, _Registration] = {
    AdapterName.ANTHROPIC_AGENT_SDK: _Registration(
        "claude_agent_sdk",
        "claude-agent-sdk",
        _anthropic_agent_sdk,
        attrgetter("anthropic_agent_sdk"),
    ),
    AdapterName.LANGGRAPH: _Registration("langgraph", "langgraph", _langgraph),
    AdapterName.OPENAI_AGENTS: _Registration("agents", "openai-agents", _openai_agents),
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


def _framework_present(name: str, *, explicit: bool, debug: bool) -> bool:
    """Whether the framework of the adapter called `name` is here to be
    adapted. Run by `AdapterRegistry.install` — ONCE, in front of
    `adapter.install()`, the step that imports the framework — for every way
    in, and without importing anything of the host's. An adapter with no
    registration row (every test double) has no framework to probe and is
    present by definition.

    Shadowed is said once with both paths and counted under
    `adapters.<name>.shadowed`, whichever path the adapter came in by.

    Absent splits on who asked. Under auto-detection it is silent (one
    debug line): a config shared across services legitimately names
    frameworks some of them lack. Under `enabled=` the user NAMED the
    framework, and a host that ships it without dist-info (a PyInstaller
    bundle built without `copy_metadata`, a vendored checkout on
    `PYTHONPATH`) is still a host with the framework — so when the import
    system finds the module the install goes ahead on the adapter's own
    import, the way it did before the probe existed. The shadow check needs
    a distribution to compare against and cannot run there, which is said
    once as a warning and counted under `distribution_absent_explicit`.
    """
    row = next((r for member, r in _ADAPTERS.items() if member.value == name), None)
    if row is None:
        return True
    verdict = probe(row.distribution, row.detect, where=f"adapters.{name}")
    if verdict.outcome == "present":
        return True
    if verdict.outcome == "shadowed":
        report_once(
            f"{name} adapter: the module '{row.detect}' resolved to {verdict.found}, "
            f"which is not the installed {row.distribution} distribution ({verdict.expected}); "
            "the adapter declined. Rename the local package or fix sys.path",
            key=f"adapters.{name}.shadowed",
        )
        counters.bump(f"adapters.{name}.shadowed")
        return False
    if explicit and _detect_package(row.detect):
        report_once(
            f"{name} adapter: the module '{row.detect}' was found without package "
            f"metadata for {row.distribution}, so the shadow check was skipped; "
            "installing because adapters.enabled names it",
            key=f"adapters.{name}.distribution_absent_explicit",
        )
        counters.bump(f"adapters.{name}.distribution_absent_explicit")
        return True
    counters.bump(f"adapters.{name}.distribution_absent")
    if debug:
        diag_info(f"{name} adapter: distribution {row.distribution} not installed")
    return False


def _make_adapter(name: AdapterName) -> AdapterInterface | None:
    """The adapter for `name`. A member exists iff its adapter ships.

    That doctrine (recorded on `AdapterName`, held by a registry test that
    keeps the enum and this table equal) makes a miss here unreachable except
    through drift between the two — kept as a `None` rather than a `KeyError`
    because a drift that shipped anyway must not turn `init()` into a crash.
    """
    row = _ADAPTERS.get(name)
    return None if row is None else row.build()


def _options_for(name: str, adapters: AdaptersConfig) -> object | None:
    """The options group `adapters` carries for the adapter called `name`.

    Selected via the REGISTRATION ROW, so how an adapter's options are found
    lives on the same one row as how its framework is found and how it is
    built. `None` when the row has no options accessor (no config class yet)
    and for a name this SDK ships no adapter for — every test double.
    """
    for member, row in _ADAPTERS.items():
        if member.value == name:
            return None if row.options is None else row.options(adapters)
    return None


def install_configured_adapters(client: Client | None, config: WardexConfig) -> None:
    """Install what `config.adapters.enabled` selects.

    `None` auto-detects installed frameworks, `()` installs none, a tuple
    installs exactly what it names — the same three answers the flat
    `config.adapters` tuple gave before the group existed.
    """
    if config.adapters.enabled is not None:
        wanted = list(config.adapters.enabled)
    else:
        wanted = [name for name, pkg in _DETECT_PACKAGES.items() if _detect_package(pkg)]
        # Configured-but-not-installed has TWO announcement channels, split by
        # who caused the absence. An adapter EXCLUDED BY `enabled=` while its
        # options are set is a contradiction the user wrote into one config
        # object, so `init()` announces it unconditionally with a
        # `WardexConfigWarning` (the mirror of interceptors under
        # intercept=False). An adapter merely NOT DETECTED — this branch — is
        # environment-dependent and legitimate in a config shared across
        # services, so it is one stderr line under debug only.
        if config.debug:
            for name in _non_default_adapter_options(config.adapters):
                if name not in wanted:
                    diag_info(
                        f"{name.value} options set but the adapter is not installed (not detected)"
                    )
    explicit = config.adapters.enabled is not None
    for name in wanted:
        try:
            adapter = _make_adapter(name)
            if adapter is not None:
                get_registry().install(adapter, client, explicit=explicit)
        except Exception as exc:  # noqa: BLE001 — a broken adapter must not break init()
            diag_warning(f"adapter {name.value} failed to load ({exc})")
            continue
