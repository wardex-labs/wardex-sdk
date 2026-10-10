"""L1 framework adapters — auto-detected at init(), selected via `AdaptersConfig.enabled`."""

from __future__ import annotations

import importlib.util
import shutil
from operator import attrgetter
from typing import TYPE_CHECKING, NamedTuple

from .._assembly import counters, diag_info, guard, report_once
from .._config import AdaptersConfig, _non_default_adapter_options
from .._diagnostics import AdapterStatus
from .._enums import AdapterName, AdapterState
from ._measured import MEASURED, Measured, installed_version, is_measured, spelled
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

    detect: str | None
    """The framework's import name, probed to decide whether it is here.
    `None` for a framework the host RUNS rather than imports — see
    `executable`."""

    distribution: str | None
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

    executable: str | None = None
    """For a framework that is a CLI the host spawns rather than a package it
    imports: the executable's name, looked up on `PATH` to auto-detect it.

    There is no module to probe and nothing to shadow, so such a row has
    `detect` and `distribution` of `None`. What is detected is only that the
    CLI is installed; whether a given spawn IS that CLI is the adapter's own
    decision, made per process. Auto-detection keyed on the CLI keeps the
    adapter's process-wide patch off every host that could never run it; a
    host that runs the CLI from a path off `PATH` names the adapter in
    `enabled=`.
    """


def _codex_exec() -> AdapterInterface:
    from ._codex_exec import CodexExecAdapter

    return CodexExecAdapter()


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
    AdapterName.CODEX_EXEC: _Registration(
        None,
        None,
        _codex_exec,
        attrgetter("codex_exec"),
        executable="codex",
    ),
    AdapterName.LANGGRAPH: _Registration("langgraph", "langgraph", _langgraph),
    AdapterName.OPENAI_AGENTS: _Registration("agents", "openai-agents", _openai_agents),
}

#: AdapterName -> distribution package to probe for auto-detection. DERIVED from
#: `_ADAPTERS` and never maintained beside it: a second literal table is exactly
#: the drift the single row above exists to make impossible.
_DETECT_PACKAGES: dict[AdapterName, str | None] = {
    name: row.detect for name, row in _ADAPTERS.items()
}


def _detect_package(module_name: str | None) -> bool:
    if module_name is None:
        return False
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:  # noqa: BLE001 — detection must never raise
        return False


def _detect_executable(name: str | None) -> bool:
    if name is None:
        return False
    found = False
    with guard("adapters.detect_executable"):  # detection must never raise
        found = shutil.which(name) is not None
    return found


def _detected(row: _Registration) -> bool:
    """Whether auto-detection finds `row`'s framework: its module, or its CLI."""
    if row.detect is not None:
        return _detect_package(row.detect)
    return _detect_executable(row.executable)


def _framework_absence(name: str, *, explicit: bool, debug: bool) -> AdapterState | None:
    """Why the framework of the adapter called `name` is NOT here to be
    adapted — `SHADOWED` or `ABSENT` — or None when it is. Run by
    `AdapterRegistry.install` — ONCE, in front of
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
    if row is None or row.detect is None or row.distribution is None:
        # No registration row (a test double), or a CLI framework: there is no
        # module to import here and so nothing to shadow — the adapter decides
        # per spawn whether a process is its CLI.
        return None
    verdict = probe(row.distribution, row.detect, where=f"adapters.{name}")
    if verdict.outcome == "present":
        return None
    if verdict.outcome == "shadowed":
        report_once(
            f"{name} adapter: the module '{row.detect}' resolved to {verdict.found}, "
            f"which is not the installed {row.distribution} distribution ({verdict.expected}); "
            "the adapter declined. Rename the local package or fix sys.path",
            key=f"adapters.{name}.shadowed",
        )
        counters.bump(f"adapters.{name}.shadowed")
        return AdapterState.SHADOWED
    if explicit and _detect_package(row.detect):
        report_once(
            f"{name} adapter: the module '{row.detect}' was found without package "
            f"metadata for {row.distribution}, so the shadow check was skipped; "
            "installing because adapters.enabled names it",
            key=f"adapters.{name}.distribution_absent_explicit",
        )
        counters.bump(f"adapters.{name}.distribution_absent_explicit")
        return None
    counters.bump(f"adapters.{name}.distribution_absent")
    if debug:
        diag_info(f"{name} adapter: distribution {row.distribution} not installed")
    return AdapterState.ABSENT


#: Each distribution an adapter's table entry names, with its installed version
#: (None when it is not installed).
_Found = tuple[tuple[Measured, str | None], ...]


def _framework_versions(name: str) -> _Found:
    """The installed version of every distribution the adapter called `name`
    was measured against, read from package metadata — nothing is imported.
    Empty for an adapter with no table entry (a test double, or a CLI)."""
    member = next((m for m in MEASURED if m.value == name), None)
    entries = MEASURED[member] if member is not None else ()
    return tuple((entry, installed_version(entry.distribution)) for entry in entries)


def _status(name: str, state: AdapterState, found: _Found, detail: str = "") -> AdapterStatus:
    """`name`'s status in `state`. Measured-ness is judged only for an adapter
    whose framework is here and was tried — installed, unsupported or failed —
    and only from versions that could be read."""
    if not found:
        return AdapterStatus(name, state, detail=detail)
    primary, version = found[0]
    tried = state in (AdapterState.INSTALLED, AdapterState.UNSUPPORTED, AdapterState.FAILED)
    readable = [(entry, v) for entry, v in found if v is not None]
    measured = all(is_measured(entry, v) for entry, v in readable) if tried and readable else None
    ranges = tuple(f"{minor}.x" for minor in primary.minors)
    return AdapterStatus(name, state, primary.distribution, version, measured, ranges, detail)


def _versions_said(found: _Found) -> str:
    """ "langgraph 1.2.10, langgraph-prebuilt 1.1.0": what was found, unknowns included."""
    return ", ".join(f"{entry.distribution} {v or '(version unknown)'}" for entry, v in found)


def _say(name: str, key: str, line: str, *, count: bool = True) -> str:
    """One `[wardex]` line per adapter per kind, counted under `key`, and kept
    as the status's `detail` so it is readable after stderr has scrolled."""
    report_once(line, key=f"adapters.{name}.{key}")
    if count:
        counters.bump(f"adapters.{name}.{key}")
    return line


def _installed_status(name: str) -> AdapterStatus:
    """An adapter that installed. Says so once when a framework version it
    depends on is outside what this release was tested against: it stays
    installed — most of what it records is still right — but the reader of
    its tree has to know the tree may be wrong somewhere."""
    found = _framework_versions(name)
    off = [(entry, v) for entry, v in found if v is not None and not is_measured(entry, v)]
    if not off:
        return _status(name, AdapterState.INSTALLED, found)
    what = ", ".join(f"{entry.distribution} {v} (tested: {spelled(entry)})" for entry, v in off)
    pins = ", ".join(
        f"{entry.distribution} to a tested release ({spelled(entry)})" for entry, _ in off
    )
    line = _say(
        name,
        "unmeasured_version",
        f"{name} adapter: this wardex release was not tested against {what}. The "
        "adapter is installed and recording, but parts of what it records may be wrong. "
        f"Pin {pins} for a tested setup, or upgrade wardex-sdk",
    )
    return _status(name, AdapterState.INSTALLED, found, line)


def _not_installed(
    name: str, state: AdapterState, key: str, line: str, found: _Found, *, count: bool = True
) -> AdapterStatus:
    """THE one place an adapter whose framework is here is filed as not
    installed — `UNSUPPORTED` or `FAILED` — so whatever else has to know that
    a whole framework's spans are missing is told from here and nowhere else."""
    return _status(name, state, found, _say(name, key, line, count=count))


def _unsupported_status(name: str) -> AdapterStatus:
    """An adapter whose `install()` returned without installing: its framework
    is here, and its surface is not the one the adapter was written for."""
    found = _framework_versions(name)
    status = _status(name, AdapterState.UNSUPPORTED, found)
    seen = _versions_said(found) or "its framework"
    if status.measured is False:
        off = [entry for entry, v in found if v is not None and not is_measured(entry, v)]
        pins = ", ".join(
            f"{entry.distribution} to a tested release ({spelled(entry)})" for entry in off
        )
        then = f"Pin {pins}, or upgrade wardex-sdk"
    else:
        # A version the release WAS tested on, or one that could not be read:
        # the surface should have been there, so the installation itself is
        # suspect — not something a version pin fixes.
        then = "Reinstall the framework; if this persists, report it with the debug=True output"
    return _not_installed(
        name,
        AdapterState.UNSUPPORTED,
        "unsupported",
        f"{name} adapter: found {seen}, but not the surface this wardex release was "
        f"written for, so the adapter did not install and its spans will be absent. {then}",
        found,
    )


def _failed_status(name: str) -> AdapterStatus:
    """An adapter that raised while it was being built or installed.

    The registry attempts the rollback, but an adapter whose patches live
    outside `ctx.patches` undoes them only through its own `uninstall()`, so
    the line says what is known — the adapter did not finish installing — and
    does not promise that nothing of it is left behind.
    """
    found = _framework_versions(name)
    seen = f" ({_versions_said(found)})" if found else ""
    return _not_installed(
        name,
        AdapterState.FAILED,
        "install_failed",
        f"{name} adapter: an error was raised while installing it{seen}, so it did not "
        "finish installing and its spans may be missing. Re-run with debug=True to see "
        "the traceback",
        found,
        # Not counted again: the guard the error passed through already counted
        # it, under `adapters.<name>.install` or `adapters.<name>.load`.
        count=False,
    )


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
        wanted = [name for name, row in _ADAPTERS.items() if _detected(row)]
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
    registry = get_registry()
    for name in wanted:
        # A broken adapter must not break init(). Guarded rather than caught, so
        # the failure is counted under `adapters.<name>.load` and its traceback
        # prints under debug, and said once as that adapter's failure line.
        loaded = False
        with guard(f"adapters.{name.value}.load", debug=bool(config.debug)):
            adapter = _make_adapter(name)
            if adapter is not None:
                registry.install(adapter, client, explicit=explicit)
            loaded = True
        if not loaded:
            registry.record(_failed_status(name.value))
    # Every adapter this release ships gets a status, so `wardex.diagnostics()`
    # answers for the ones that were never tried as well: left out by `enabled=`,
    # or not detected.
    for member in _ADAPTERS:
        if member not in wanted:
            state = AdapterState.DISABLED if explicit else AdapterState.ABSENT
            registry.record(_status(member.value, state, _framework_versions(member.value)))
