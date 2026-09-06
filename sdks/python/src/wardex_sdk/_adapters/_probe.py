"""Where a framework's package would come from, decided WITHOUT importing it.

Shared by every adapter and run by the registry BEFORE an adapter is built.
The registry's auto-detection is a bare `find_spec`, and an adapter's
`install()` imports its framework: a project-local regular package called
`langgraph/` or `agents/` (an app folder named after the framework is the
common shape) would otherwise have its body EXECUTED by that import on the
host's own `wardex.init()` — the host-behaviour change an adapter exists
to never make — before the adapter declined in silence. The three answers:

- ABSENT: no distribution of that name is installed. Nothing is imported;
  absence is an answer, not a failure, and it is not reported.
- SHADOWED: the distribution is installed but `import <module>` would answer
  a package that is not the distribution's. Declined, loudly, naming both
  paths; the local package is never imported.
- PRESENT: the module the import system would answer IS the distribution's,
  directly or through an editable install of it.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import NamedTuple
from urllib.parse import unquote, urlsplit

from .._assembly import guard

__all__ = ["Probe", "editable_root", "installed_distribution", "probe", "shadow_path"]


class Probe(NamedTuple):
    outcome: str
    """`"present"`, `"absent"` or `"shadowed"`."""

    found: str | None = None
    """The resolved module path, when shadowed."""

    expected: str | None = None
    """The installed distribution's path, when shadowed."""


def probe(distribution: str, module: str, *, where: str) -> Probe:
    """The verdict for one framework: `distribution` is its distribution
    name, `module` its import name, `where` the counter namespace a corrupt
    editable record is counted under (`adapters.<name>`)."""
    dist = installed_distribution(distribution)
    if dist is None:
        return Probe("absent")
    shadow = shadow_path(module, dist, where=where)
    if shadow is not None:
        return Probe("shadowed", shadow[0], shadow[1])
    return Probe("present")


def installed_distribution(distribution: str) -> importlib.metadata.Distribution | None:
    """The distribution, found WITHOUT importing anything: a host with a local
    package of the framework's name and no distribution answers `None` here
    and nothing of theirs is imported.

    The absence is an ANSWER, spelled as the stdlib spells it. The
    exception-free `packages_distributions()` was tried first and answers
    `None` for this very wheel on the 3.10 floor (it reads `top_level.txt`
    only there), which would have declined the adapter on every 3.10 host.
    """
    try:
        return importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def shadow_path(
    module: str, dist: importlib.metadata.Distribution, *, where: str
) -> tuple[str, str] | None:
    """`(resolved module path, distribution path)` when the package that
    `import <module>` WOULD answer is not the installed distribution's, else None.

    Read from the import system's spec, so the answer costs no import: a
    project-local package is named here, with both paths, BEFORE anything of
    it could run — and a local package missing a submodule the adapter needs
    is reported the same way rather than declined in silence by a failed
    import. A module the import system cannot locate answers None and leaves
    the import step to decline; a namespace package has no origin and does
    the same.

    An EDITABLE install (`pip install -e`, `uv --editable`, a workspace
    member) resolves the module to the project tree, never to
    `site-packages`, so the path comparison alone would decline the adapter
    on the very machine the framework is developed on. The distribution's
    own record of where it came from (PEP 610's `direct_url.json`) settles
    it: a module under that editable root IS the installed distribution. A
    corrupt record is counted under `<where>.direct_url_read`, not raised.
    """
    spec = importlib.util.find_spec(module)
    origin = spec.origin if spec is not None else None
    if not origin:
        return None
    found = Path(origin).parent.resolve()
    expected = Path(str(dist.locate_file(module))).resolve()
    if found == expected:
        return None
    editable = None
    with guard(f"{where}.direct_url_read"):
        editable = editable_root(dist)
    if editable is not None and (found == editable or editable in found.parents):
        return None
    return str(found), str(expected)


def editable_root(dist: importlib.metadata.Distribution) -> Path | None:
    """The project directory an EDITABLE install of `dist` points at, or None
    for a regular install (no `direct_url.json`, or one that records an
    archive, an index, or a non-editable local directory whose package was
    COPIED into site-packages and so must still match `locate_file`)."""
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict) or not (data.get("dir_info") or {}).get("editable"):
        return None
    url = urlsplit(str(data.get("url", "")))
    if url.scheme != "file":
        return None
    return Path(unquote(url.path)).resolve()
