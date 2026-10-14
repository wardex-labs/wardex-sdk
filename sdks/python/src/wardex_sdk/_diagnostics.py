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

__all__ = ["AdapterStatus", "Diagnostics", "Loss"]


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
class Loss:
    """One kind of thing wardex did not record, counted since the process started.

    `reason` is a closed vocabulary (`queue_overflow`, `internal_sdk_error`,
    ...) and `unit` says what was counted (`span`, `request`, `adapter`, ...).
    `nature` is `"loss"` when the thing is gone, `"partial"` when it was
    recorded incomplete, and `"none"` when nothing was lost because the drop
    was asked for (a `before_send_envelope` hook that returned None).
    `meaning` and `next_step` are the sentences wardex uses for this reason
    everywhere it reports one. `where` breaks the reason down by the place in
    wardex that counted it (`internal_sdk_error` by where it raised, an adapter
    that did not install as `adapters.<name>`), across all of the reason's
    units.

    The counts only grow: sending them to a receiver does not reset them, so
    two readings can be subtracted to get what was lost between them. They are
    exact for this process whether or not they were sent. `undeliverable` is
    how many of `count` never will be: the report carrying them was dropped
    because more reports awaited the receiver's confirmation than wardex holds
    (counted itself as `ledger_overflow`).
    """

    reason: str
    unit: str
    count: int
    nature: str
    meaning: str = ""
    next_step: str = ""
    where: tuple[tuple[str, int], ...] = ()
    undeliverable: int = 0


@dataclass(frozen=True, slots=True)
class Diagnostics:
    """What `wardex.diagnostics()` returns: wardex's own account of this process.

    `initialized` is True between `init()` and `close()`. `adapters` is what
    became of each adapter this release ships at the most recent `init()`, and
    it is kept after `close()`, so a check made once the run is over still
    reads it; before the first `init()` it is empty. `losses` counts every
    kind of loss since the process started, one entry per reason and unit
    that has happened at least once; a forked child starts its own from zero.
    """

    initialized: bool
    adapters: tuple[AdapterStatus, ...] = ()
    losses: tuple[Loss, ...] = ()
