"""The shapes the compatibility run keeps between days, and the one reader of each.

Two things outlive a run: the record lines on the records branch, which are never
rewritten, and `state.json`, which every run rewrites. Schema 1 wrote both with no notion
of a kind of point. Schema 2 keeps every schema-1 field under its name and meaning and
adds fields, so an older line or file is read by giving each new field the only value it
could have had then. Nothing on the records branch is edited to make that true.

Standard library only: the plan, the executor and the report import this.
"""

from __future__ import annotations

import json

SCHEMA = 2

#: Where the suite lives inside whichever tree a point tests.
TESTS_DIR = "sdks/python/tests"

#: What a point said before kinds existed: every point was a framework release tested with
#: the wheel built from the run's own commit on the Linux runner, and could turn the run red.
_THEN = {
    "kind": "framework",
    "watching": False,
    "runner": "linux-x64",
    "sdk": {"origin": "build"},
}


def read_record(line: dict) -> dict:
    """A record line of any schema, with every schema-2 field present."""
    if line.get("schema", 1) >= SCHEMA:
        return line
    run = {"cadence": "daily", "scope": "unknown", **line.get("run", {})}
    adapter = line.get("adapter", "")
    return {**_THEN, "subject": adapter, "adapters": [adapter], "plan": None, **line, "run": run}


def identity_of(point: dict) -> dict:
    """What `state.json` keeps about a point besides what it last said: enough to show it
    on an issue or a page on a day no run planned it."""
    adapter = point.get("adapter", "")
    return {
        "kind": point.get("kind", _THEN["kind"]),
        "subject": point.get("subject", adapter),
        "adapters": point.get("adapters", [adapter]),
        "adapter": adapter,
        "role": point["role"],
        "targets": point["targets"],
        "python": point.get("python", ""),
        "runner": point.get("runner", _THEN["runner"]),
        "sdk": point.get("sdk", _THEN["sdk"]),
        "in_claim": point["in_claim"],
        "watching": point.get("watching", _THEN["watching"]),
        "fresh_days": point.get("fresh_days", 3),
    }


def upgrade_state(state: dict) -> dict:
    """`state.json` of any schema as schema 2, in memory.

    A schema-1 entry has no identity: `point` is None until a run plans that key again,
    and an entry no plan makes any more is dropped by the run that covers its kind."""
    if state.get("schema", 1) >= SCHEMA:
        return state
    points = {key: {"point": None, **entry} for key, entry in (state.get("points") or {}).items()}
    return {
        "schema": SCHEMA,
        "points": points,
        "displaced": state.get("displaced") or {},
        "kinds": {},
        "last_scheduled": state.get("last_scheduled"),
    }


#: A release that goes out without a passing compatibility gate carries this file in its
#: own tree. The gate reads it from the tree being released; the plan reads it from the
#: release's tag, so the page of that release says so for as long as the page exists.
OVERRIDE = "scripts/compat/release-override.json"


def prose(text: object, limit: int = 200) -> str:
    """One line of someone's own words, in any language, with nothing that is markup.

    What is left holds no backtick, so it can be printed inside a code span, where no
    other character means anything either: a link, a mention and emphasis stay text."""
    kept = "".join(ch if ch.isprintable() and ch not in "`|<>[]" else " " for ch in str(text))
    return " ".join(kept.split())[:limit]


def quoted(text: object, limit: int = 200) -> str:
    """`text` as a code span that cannot be broken out of."""
    return f"`{prose(text, limit)}`"


def parse_override(text: str, version: str) -> dict | None:
    """The override `text` holds for `version`, or None.

    One left behind by an earlier release names that release and excuses nothing, and one
    that gives no reason is not one."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("version") != version:
        return None
    reason = data.get("reason")
    if not isinstance(reason, str):
        return None
    # The reason is printed on a public page. It has to say something: a few letters at
    # the least.
    kept = prose(reason)
    if sum(ch.isalpha() for ch in kept) < 3:
        return None
    return {"version": version, "reason": kept}
