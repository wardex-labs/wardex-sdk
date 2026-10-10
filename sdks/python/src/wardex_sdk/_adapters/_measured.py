"""Which framework versions this release was tested against, and the verdict for one.

An adapter reaches into a framework this SDK does not control, and a framework
release can change what flows through the surface the adapter patches without
changing the surface itself: the names and signatures the adapter probes stay,
so it installs, and the tree it records is quietly wrong. The probe cannot see
that. Only running the adapter's suite against the release can, so the table
below records the releases that suite was run against, and every other version
is said to be unmeasured when an adapter installs on it. The adapter still
installs: most of what it records is usually still right, and a framework that
ships a patch every few days would otherwise have its tree switched off most of
the time.

THE UNIT IS THE MINOR VERSION, final releases only. A pre-release, dev, post or
local version was never the subject of a measurement, so it is unmeasured even
when its minor is listed. A patch inside a measured minor that is known to
break the suite is listed under `known_bad`.

No dependency on `packaging`: the SDK has none at runtime, and a regular
expression over the release segment is all the comparison needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .._assembly import guard
from .._enums import AdapterName
from ._probe import installed_distribution

__all__ = ["MEASURED", "Measured", "installed_version", "is_measured", "spelled"]


@dataclass(frozen=True, slots=True)
class Measured:
    """One distribution an adapter depends on, and the minors it was measured on."""

    distribution: str
    minors: tuple[str, ...]
    """Measured minor versions, as "major.minor"."""

    known_bad: tuple[str, ...] = ()
    """Exact versions inside a measured minor that fail the adapter's suite."""


#: Every adapter this release ships, and what its suite was last run against
#: (2026-10-10). An empty tuple means the adapter's framework has no version
#: read at `init()`: Codex is a CLI, and reading its version would mean
#: spawning it. A framework release outside this table is said to be unmeasured
#: once, when the adapter installs on it.
MEASURED: dict[AdapterName, tuple[Measured, ...]] = {
    AdapterName.ANTHROPIC_AGENT_SDK: (Measured("claude-agent-sdk", ("0.2",)),),
    AdapterName.CODEX_EXEC: (),
    # The tool seam lives in a separately versioned distribution that langgraph
    # depends on, so a release of either one can move what the adapter patches.
    AdapterName.LANGGRAPH: (
        Measured("langgraph", ("1.2",)),
        Measured("langgraph-prebuilt", ("1.1",)),
    ),
    # 0.23 is not measured: its runs record wrong parents and error types.
    AdapterName.OPENAI_AGENTS: (Measured("openai-agents", ("0.22",)),),
}

_FINAL = re.compile(r"(\d+)\.(\d+)(?:\.\d+)*")


def is_measured(entry: Measured, version: str) -> bool:
    """Whether `version` of `entry.distribution` is one the adapter was measured on."""
    found = _FINAL.fullmatch(version)
    if found is None or version in entry.known_bad:
        return False
    return f"{int(found[1])}.{int(found[2])}" in entry.minors


def spelled(entry: Measured) -> str:
    """The measured range as a person reads it: "0.21.x, 0.22.x"."""
    return ", ".join(f"{minor}.x" for minor in entry.minors)


def installed_version(distribution: str) -> str | None:
    """The installed version of `distribution`, or None when it is not installed.

    Read from package metadata, so nothing is imported. A metadata read that
    raises is counted, not raised: a version wardex could not read is reported
    as unknown rather than taking `init()` down.
    """
    dist = installed_distribution(distribution)
    if dist is None:
        return None
    version = None
    with guard("adapters.version_read"):
        version = dist.version
    return version or None  # an empty Version field is no version
