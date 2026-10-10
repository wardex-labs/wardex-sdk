"""What wardex can say about itself: the types `wardex.diagnostics()` returns.

Kept apart from `_types.py` on purpose. The adapters file their own status, and
the import rules forbid an adapter module from naming `_types` at all — that is
where the span machinery lives, and an adapter that can name a span id can mint
a parent edge. Nothing here is span machinery: these are plain records of what
was installed and what was found, so they sit at the vocabulary layer, where
both the adapters and the public surface can reach them.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._enums import AdapterState

__all__ = ["AdapterStatus", "Diagnostics"]


@dataclass(frozen=True, slots=True)
class AdapterStatus:
    """One adapter's state in this process, and the framework version it found.

    `measured` is whether this wardex release was tested against the installed
    `version`: True or False when it was checked, None when there was nothing to
    check (the adapter is not installed, or its framework is a CLI whose
    version is not read at `init()`). `measured_versions` is what was tested,
    as "0.22.x". `detail` is the line wardex said about this adapter, if it
    said one, so the reason is readable after stderr has scrolled away.
    """

    name: str
    state: AdapterState
    framework: str | None = None
    version: str | None = None
    measured: bool | None = None
    measured_versions: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """What `wardex.diagnostics()` returns: wardex's own account of this process.

    `initialized` is False before `init()` and after `close()`, and then the
    other fields are empty because there is nothing installed to describe.
    `adapters` has one entry per adapter this release ships.
    """

    initialized: bool
    adapters: tuple[AdapterStatus, ...] = ()
