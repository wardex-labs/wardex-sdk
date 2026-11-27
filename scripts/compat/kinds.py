"""Every kind of point the compatibility run knows, and which kinds may turn the run red.

A kind is one question asked of the same device: "does wardex hold on this framework
release" is `framework`. The plan has one rule per kind that turns its sources into
points. Nothing after the plan branches on a kind's name: the executor runs the recipe a
point carries, and the report reads the point's own fields and the two declarations here.

Standard library only: the plan and the report import this.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The kinds whose failure inside a claim, or whose planned point with no result, turns the
#: run red. Every other kind is watching: it is recorded, it opens and rewrites issues, and
#: it never turns the run red.
#:
#: Promotion is adding one line here, in a commit of its own. A change that introduces a
#: kind never adds it here, and copying a neighbouring `Kind(...)` entry cannot make a kind
#: red-capable, because red-capability is not a field of the entry.
RED_CAPABLE: frozenset[str] = frozenset(
    {
        "framework",
    }
)


@dataclass(frozen=True)
class Kind:
    #: Words for issues and pages.
    label: str
    #: The runs this kind is part of.
    cadences: frozenset[str]
    #: A result older than this many days is said to be that old, never shown as current.
    fresh_days: int
    #: Fewer points than this for one subject, on a day the kind is due, is a plan problem.
    min_points: int
    #: Said in an issue under a failing point of this kind: how the item closes.
    closes: str


KINDS: dict[str, Kind] = {
    "framework": Kind(
        label="framework release",
        cadences=frozenset({"daily"}),
        fresh_days=3,
        min_points=2,
        closes="when it passes again, or when the tested range no longer claims it",
    ),
    # The newest wardex release on PyPI, installed from PyPI and run with the tests and the
    # table of its own tag: what a user who types `pip install` has today, as opposed to
    # what the default branch would give them.
    "published": Kind(
        label="published wardex release",
        cadences=frozenset({"daily"}),
        fresh_days=3,
        min_points=1,
        closes="when it passes, or when a newer wardex release takes its place",
    ),
}

#: The cron lines of `compat.yml` and the cadence each one starts. A test holds the
#: workflow's `schedule:` to this table.
CRONS: dict[str, str] = {
    "23 19 * * *": "daily",
}


def is_red_capable(kind: str) -> bool:
    """The one question the verdict asks of a kind."""
    return kind in RED_CAPABLE


def due(cadence: str) -> tuple[str, ...]:
    """The kinds a run of `cadence` plans, in declaration order."""
    return tuple(name for name, kind in KINDS.items() if cadence in kind.cadences)
