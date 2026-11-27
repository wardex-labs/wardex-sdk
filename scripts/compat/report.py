"""Turn one run's point results into records, issues and the run's colour.

The only job with write access runs this, and it installs nothing: it reads
the JSON the point jobs left, checks its shape, and writes plain data. A point
job executes framework releases published the same day, so nothing it
produced is trusted beyond being a string of an expected form.

The run is red in exactly two cases: a release inside the tested range does
not pass, or a planned point left no result. Everything else that needs a
person (a newer release that fails, a newer release that passes and could be
added to the range, an index outage that lasts) reaches them as an issue.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compat import kinds, model, render  # noqa: E402
from compat.plan import claims, displaced_line  # noqa: E402
from compat.verdict import (  # noqa: E402
    FAILED,
    GREEN,
    NOT_OURS,
    OUTCOME_WORDS,
    RED,
    is_watching,
    judge,
)

SCHEMA = model.SCHEMA
LABEL = "compat"
#: The exit code of a run judged red. Its own number, so the workflow can tell a verdict
#: from this script crashing, which leaves no verdict and must be reported as such.
RED_EXIT = 10
#: A release inside the tested range that has failed on this many days running is no
#: longer waiting for a fix: its issue starts printing the command that stops claiming it.
#: Nothing narrows a claim on its own.
LAPSE_NARROW_DAYS = 14
_UNSAFE = re.compile(r"[^A-Za-z0-9 _./:\[\]()=,<>+*#-]")

ROLE_WORDS = {
    "floor": "oldest tested release",
    "minor_newest": "newest patch of a tested line",
    "claim_top": "newest tested release",
    "claim_top_oldpy": "newest tested release, oldest Python",
    "beyond": "newer than the tested range",
    "displaced": "failed as the newest tested release, since replaced",
}
CAUSE_WORDS = {
    "new_version": "the newer release",
    "upstream": "a dependency release (the same wardex passes on the packages of the last pass)",
    "ours": "a wardex change (it fails on the packages of the last pass too)",
    "no_baseline": "unknown: this point has no earlier pass to compare with",
    "not_determined": "not determined: a second run could not settle it",
}


def clean(value: object, limit: int = 200) -> str:
    """`value` as text that is safe inside a code span of a public issue."""
    return _UNSAFE.sub("?", str(value))[:limit]


def code(value: object, limit: int = 200) -> str:
    return f"`{clean(value, limit)}`"


# --- reading and judging ------------------------------------------------------


def _shape_problem(data: dict) -> str:
    """Why a result is not of the form the rest of this module relies on, or ""."""

    def texts(value: object) -> bool:
        return isinstance(value, list) and all(isinstance(item, str) for item in value)

    tests = data.get("tests")
    if tests is not None:
        if not isinstance(tests, dict):
            return "malformed tests"
        for field in ("failed_ids", "skipped_ids", "files", "excluded_files", "plugins"):
            if not texts(tests.get(field, [])):
                return f"malformed tests.{field}"
        for field in ("passed", "failed", "errors", "skipped"):
            if not isinstance(tests.get(field, 0), int):
                return f"malformed tests.{field}"
    for field in ("attach", "install", "counterfactual", "rerun", "sdk", "resolved", "uploaded"):
        if data.get(field) is not None and not isinstance(data[field], dict):
            return f"malformed {field}"
    dropped = (data.get("install") or {}).get("dropped_optional", [])
    if not isinstance(dropped, list) or not all(texts(group) for group in dropped):
        return "malformed install.dropped_optional"
    changed = (data.get("counterfactual") or {}).get("changed", {})
    if not isinstance(changed, dict) or not all(
        isinstance(pair, list) and len(pair) == 2 for pair in changed.values()
    ):
        return "malformed counterfactual.changed"
    if data.get("cause") not in (None, *CAUSE_WORDS):
        return "unknown cause"
    return ""


def load_results(plan: dict, results_dir: Path) -> dict[str, dict]:
    """One result per planned point. A missing or malformed one is `no_result`.

    Each artifact is read from its own directory and must name the point the
    directory is for, so one point cannot speak for another.
    """
    results = {}
    for point in plan["points"]:
        key = point["key"]
        path = results_dir / f"compat-point-{key}" / "result.json"
        problem = ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or data.get("schema") != SCHEMA:
                problem = "unknown schema"
            elif (data.get("key"), data.get("adapter")) != (key, point["adapter"]):
                problem = "the result names a different point"
            elif data.get("targets") != point["targets"]:
                problem = "the result names different target versions"
            elif data.get("outcome") not in OUTCOME_WORDS:
                problem = "unknown outcome"
            else:
                problem = _shape_problem(data)
        except (OSError, ValueError) as error:
            data, problem = {}, type(error).__name__
        if problem:
            data = {"outcome": "no_result", "note": problem}
        # Who the point is comes from the plan, never from what the point job wrote. What
        # the point measured (the interpreter that ran, the wheel it imported) stays its own.
        identity = model.identity_of(point)
        data.update({k: point[k] for k in ("key", "adapter", "role", "targets", "in_claim")})
        data.update({k: v for k, v in identity.items() if k not in ("python", "sdk")})
        data["planned_python"] = point.get("python", "")
        data["planned_sdk"] = identity["sdk"]
        data["planned"] = point
        data.setdefault("cause", None)
        results[key] = data
    return results


def apply_skip_rule(results: dict[str, dict], state: dict) -> None:
    """A pass that skips a test its last pass ran is not a pass.

    A skipped test looks green. Compared only while the selected files are
    byte-identical, so a commit that adds a skipped test is not held against
    the point, and only a newly skipped test counts: fewer skips is more tested.
    """
    for key, result in results.items():
        last = state.get("points", {}).get(key, {}).get("last_green") or {}
        tests = result.get("tests") or {}
        if (
            result["outcome"] in GREEN
            and last.get("selection_sha256") == tests.get("selection_sha256")
            and last.get("skipped_ids") is not None
            and set(tests.get("skipped_ids", [])) - set(last["skipped_ids"])
        ):
            result["outcome"] = "skips_changed"


# --- records and state --------------------------------------------------------


def records(plan: dict, results: dict[str, dict], run: dict) -> list[dict]:
    """One line per point: everything a later reader needs, in data they can trust.

    Every field a point wrote is rebuilt here from cleaned text, numbers and
    known words. The line goes to a public branch, and stage two reads it.
    """
    lines = []
    for key, result in sorted(results.items()):
        tests = result.get("tests") or {}
        replay = result.get("counterfactual") or {}
        lines.append(
            {
                "schema": SCHEMA,
                "run": run,
                "cutoff": plan["cutoff"],
                "kind": result.get("kind", "framework"),
                "subject": result.get("subject", result["adapter"]),
                "adapters": result.get("adapters", [result["adapter"]]),
                "watching": is_watching(result),
                "sdk": _sdk(result),
                "runner": result.get("runner", "linux-x64"),
                # The point as it was planned, so the same point can be built again from this
                # line on a later day. From the plan job, which runs no third-party code.
                "plan": _recipe(result.get("planned")),
                "key": key,
                "adapter": result["adapter"],
                "role": result["role"],
                "targets": result["targets"],
                "uploaded": {
                    clean(name, 80): clean(stamp, 30)
                    for name, stamp in (result.get("uploaded") or {}).items()
                },
                "in_claim": result["in_claim"],
                "isolated": result.get("isolated") is True,
                "python": clean(result.get("python", "")),
                "os": clean(result.get("os", "")),
                "arch": clean(result.get("arch", "")),
                "runner_image": clean(result.get("runner_image", "")),
                "uv_version": clean(result.get("uv_version", "")),
                "sdk_wheel_sha256": clean((result.get("sdk") or {}).get("wheel_sha256", "")),
                "evidence": clean(result.get("evidence", "")),
                "replay_of": clean(result.get("replay_of") or ""),
                "resolved": {
                    clean(name, 80): clean(version, 40)
                    for name, version in (result.get("resolved") or {}).items()
                },
                "lock_sha256": clean(result.get("lock_sha256", "")),
                "dropped_optional": [
                    [clean(requirement, 80) for requirement in group]
                    for group in (result.get("install") or {}).get("dropped_optional", [])
                    if isinstance(group, list)
                ],
                "tests": {
                    "files": [clean(name) for name in tests.get("files", [])],
                    "excluded_files": [clean(name) for name in tests.get("excluded_files", [])],
                    "selection_sha256": clean(tests.get("selection_sha256", "")),
                    "passed": tests.get("passed"),
                    "failed": tests.get("failed"),
                    "errors": tests.get("errors"),
                    "skipped": tests.get("skipped"),
                    "timed_out": tests.get("timed_out") is True,
                    "failed_ids": [clean(node, 300) for node in tests.get("failed_ids", [])],
                    "skipped_ids": [clean(node, 300) for node in tests.get("skipped_ids", [])],
                    "plugins": [
                        clean(name) for name in tests.get("plugins", []) if isinstance(name, str)
                    ],
                },
                "attach": {
                    k: clean(v) if isinstance(v, str) else v
                    for k, v in (result.get("attach") or {}).items()
                    if k in ("state", "measured", "in_table", "lines_said", "verdict")
                    and (v is None or isinstance(v, (str, bool, int)))
                },
                "install_outcome": clean((result.get("install") or {}).get("outcome", "")),
                "rerun_passed": (result.get("rerun") or {}).get("passed") is True
                if result.get("rerun")
                else None,
                "note": clean(result.get("harness_error") or result.get("note") or "", 300),
                "point_cutoff": clean(result.get("cutoff", ""), 30),
                "outcome": result["outcome"],
                "cause": result["cause"],
                "replay_on_last_pass": {
                    "ran": replay.get("ran") is True,
                    "passed": replay.get("passed") is True,
                    "changed": {
                        clean(name, 80): [clean(a, 40), clean(b, 40)]
                        for name, (a, b) in (
                            (n, pair)
                            for n, pair in (replay.get("changed") or {}).items()
                            if isinstance(pair, list) and len(pair) == 2
                        )
                    },
                }
                if replay
                else None,
                "duration_s": result["duration_s"]
                if isinstance(result.get("duration_s"), (int, float))
                else None,
            }
        )
    return lines


def _sdk(result: dict) -> dict:
    """Which wardex a point ran: where it came from, as planned, and the version it imported.

    A point that never got as far as importing it still says which release it was for."""
    planned = result.get("planned_sdk") or {"origin": "build"}
    measured = (result.get("sdk") or {}).get("version", "")
    said = {name: clean(value, 80) for name, value in planned.items()}
    return {**said, "version": clean(measured or planned.get("version", ""), 40)}


def _recipe(point: dict | None) -> dict | None:
    """The part of a planned point a record line does not already hold."""
    if not point:
        return None
    kept = ("python", "floats", "cutoff", "assume_measured", "install", "probe", "check")
    return {name: point[name] for name in (*kept, "budget_s", "fresh_days") if name in point}


def next_displaced(state: dict, plan: dict, results: dict[str, dict]) -> dict:
    """Releases inside the tested range that failed and that no rule picks any more.

    The newest release of a line is a point only while it is the newest. One
    that fails there and is then replaced by a newer patch would drop out of
    every later run while the README still calls it tested. It is remembered
    here and the next plan makes it a point of its own. It leaves the list when
    that point passes, or when the plan stops making the point: the table lists
    the release as bad, the range no longer holds it, or the index withdrew it.
    """
    kept: dict[str, list[dict]] = {}
    for adapter, items in (state.get("displaced") or {}).items():
        info = plan["adapters"].get(adapter)
        if info is None or info["problems"]:
            # A run that did not plan this adapter in full has no say about its list.
            kept[adapter] = list(items)
            continue
        for item in items:
            result = results.get(item["key"]) or _picked_again(item, results)
            # No point at all means the plan no longer claims the release. A point that ran
            # it settles it either way: a pass ends it, and a failure at a rule's own point
            # is on that point's record from here on.
            if result is None or result["outcome"] in GREEN:
                continue
            if result["role"] != "displaced" and result["outcome"] in FAILED:
                continue
            kept.setdefault(adapter, []).append(item)
    for key, result in sorted(results.items()):
        before = state.get("points", {}).get(key, {}).get("last_checked") or {}
        entries = plan["adapters"].get(result["adapter"], {}).get("entries") or []
        if (
            not _own(result)
            or not result["in_claim"]
            or result["role"] == "displaced"
            or before.get("outcome") not in FAILED
            or before.get("targets") == result["targets"]
            or not entries
        ):
            continue
        rows = {entry["distribution"]: entry for entry in entries}
        # A range narrowed past the release, or a release now listed as bad, is settled.
        # Every pinned distribution is asked of its own row.
        if not all(
            dist in rows and isinstance(v, str) and claims(rows[dist], v)
            for dist, v in before["targets"].items()
        ):
            continue
        python = before.get("python") or result.get("planned_python") or ""
        item = {
            "key": f"{result['adapter']}--displaced--{displaced_line(before['targets'], python)}",
            "targets": before["targets"],
            "python": python,
            "failed_on": before["date"],
        }
        mine = kept.setdefault(result["adapter"], [])
        if all(other["key"] != item["key"] for other in mine):
            mine.append(item)
    return {adapter: items for adapter, items in sorted(kept.items()) if items}


def release_id(adapter: str, targets: dict, python: object) -> str:
    """One release on one interpreter, as the key its failure is counted under."""
    return f"{adapter}|{python or ''}|" + json.dumps(targets, sort_keys=True)


def _counted(result: dict) -> bool:
    """Whether a result counts toward how long its release has been failing on this build.

    A red-capable point of the wheel this run built, inside the range, with today's
    packages. Not the floor, which installs the release with the packages of its own
    time: failing there and failing with today's are two findings, each with its own
    first day. Not a published wardex either: that is another build's count."""
    return _own(result) and result["in_claim"] and result["role"] != "floor"


def next_failing(state: dict, results: dict[str, dict], run: dict) -> dict[str, str]:
    """For each release and interpreter failing on this build, the first day of the failure
    it is in now.

    Kept by release and not by point, because a release changes hands: a newer patch
    replaces it and its own point takes over, that patch is withdrawn and the rule's point
    runs it again, the range grows and another rule's point picks it up. Whoever runs it,
    it has been failing since the same day. A failure by any point that runs it keeps or
    starts the count; a pass with no failure beside it ends it; a day nobody could check
    it changes nothing."""
    failing = dict(state.get("failing") or {})
    said: dict[str, set[str]] = {}
    for result in results.values():
        if _counted(result):
            release = release_id(result["adapter"], result["targets"], result.get("planned_python"))
            said.setdefault(release, set()).add(result["outcome"])
    for release, outcomes in said.items():
        if outcomes & FAILED:
            failing.setdefault(release, run["date"])
        elif outcomes & GREEN:
            failing.pop(release, None)
    return failing


def _still_asked(failing: dict, plan: dict | None, results: dict, displaced: dict) -> dict:
    """`failing` without the releases nobody asks about any more: no point runs them and
    no list remembers them, in an adapter this run planned in full. A release that left the
    range, or was listed as bad, takes its count with it."""
    if not plan:
        return failing
    asked = {
        release_id(r["adapter"], r["targets"], r.get("planned_python")) for r in results.values()
    }
    asked |= {
        release_id(adapter, item["targets"], item.get("python"))
        for adapter, items in displaced.items()
        for item in items
        if isinstance(item.get("targets"), dict)
    }
    whole = {name for name, info in plan["adapters"].items() if not info["problems"]}
    return {
        release: day
        for release, day in failing.items()
        if release in asked or release.split("|", 1)[0] not in whole
    }


def _picked_again(item: dict, results: dict[str, dict]) -> dict | None:
    """The rule's point that runs a remembered release today, when one of the rules picks it
    again (the release that replaced it was withdrawn) and the plan makes no second point."""
    for result in results.values():
        same = (
            result["targets"] == item["targets"] and result.get("planned_python") == item["python"]
        )
        # Not the floor: it installs the dependencies of the release's own time, and what
        # failed was the release with today's.
        picked = result["in_claim"] and result["role"] not in ("displaced", "floor")
        if same and picked and _own(result):
            return result
    return None


def _own(result: dict) -> bool:
    """Whether a result speaks for the wheel this run built, as a red-capable point.

    The list of replaced failures is about that build alone. The same release failing on
    a published wardex is that release's finding, on its own page and under its own
    heading, and must neither start nor settle anything here."""
    return not is_watching(result) and _planned_sdk(result).get("origin", "build") == "build"


_ABOUT = ("label", "evidence", "replay_of", "claim", "said", "newest")


def _prose(text: object) -> str:
    """A sentence for a page: printable, bounded, and with nothing that is table or markup
    syntax. Code marks are dropped and the words kept, where `clean` would print `?`."""
    kept = "".join(ch for ch in str(text) if " " <= ch <= "~" and ch not in "`|<>[]")
    return kept[:240]


def _versions(state: dict, plan: dict | None, run: dict) -> dict:
    """What the pages say about each wardex build besides its points: which build it is,
    and each adapter's range and evidence as that build states them.

    Kept per build because a page is rendered on days its build was not planned. A run of
    one adapter updates that adapter and leaves the others as they were."""
    versions = {name: dict(about) for name, about in (state.get("versions") or {}).items()}
    if not plan:
        return versions

    def merged(old: dict, adapters: dict) -> dict:
        kept = dict(old.get("adapters") or {})
        for name, info in adapters.items():
            kept[name] = {field: info.get(field) for field in _ABOUT}
        return kept

    main = versions.get("main") or {}
    versions["main"] = {
        "version": run["sdk_version"],
        "commit": run["sdk_commit"],
        "date": run["date"],
        "adapters": merged(main, plan["adapters"]),
        "claims": [
            {name: _prose(claim.get(name) or "") for name in ("says", "proof", "why")}
            for claim in plan.get("claims", main.get("claims") or [])
        ],
    }
    release = plan.get("published")
    if release:
        old = versions.get(release["version"]) or {}
        versions[release["version"]] = {
            "ref": release["ref"],
            "commit": release["commit"],
            "date": run["date"],
            "adapters": merged(old, release["adapters"]),
        }
    return versions


def next_state(
    state: dict, results: dict[str, dict], run: dict, results_dir: Path, plan: dict | None = None
) -> dict:
    """The small file the next run reads: each point's last record, last check and last pass,
    and the failed releases a newer one has replaced."""
    points = dict(state.get("points", {}))
    locks: dict[str, Path] = {}
    failing = next_failing(state, results, run)
    for key, result in results.items():
        entry = dict(points.get(key, {}))
        tests = result.get("tests") or {}
        entry["point"] = model.identity_of(
            {
                **result,
                "python": result.get("planned_python", ""),
                "sdk": result.get("planned_sdk", {"origin": "build"}),
            }
        )
        entry["last"] = {
            "date": run["date"],
            "outcome": result["outcome"],
            "cause": result["cause"],
            "targets": result["targets"],
            "url": run["url"],
            "failed_ids": [clean(node, 300) for node in tests.get("failed_ids", [])[:200]],
            "excluded_files": [clean(name) for name in tests.get("excluded_files", [])[:200]],
        }
        if (entry.get("last_checked") or {}).get("targets") not in (None, result["targets"]):
            # What this point last said was about a release it no longer tests. Whether that
            # release is still owed a run was settled by `next_displaced`, once.
            del entry["last_checked"]
        if result["outcome"] in GREEN | FAILED:
            # The last day this release was actually run. A later day on which it could not be
            # installed, or on which the run itself broke, says nothing new about the release
            # and must not erase what this said.
            entry["last_checked"] = {
                "date": run["date"],
                "outcome": result["outcome"],
                "cause": result["cause"],
                "targets": result["targets"],
                "python": result.get("planned_python"),
                "failed_ids": [clean(node, 300) for node in tests.get("failed_ids", [])[:200]],
            }
        # The first day of the failure this release is in now. A day it could not be checked
        # neither starts nor ends one; a pass, or another release under this key, ends it.
        since = entry.get("failing_since") or {}
        if _counted(result):
            # Counted by release, so the count survives the release changing hands.
            release = release_id(result["adapter"], result["targets"], result.get("planned_python"))
            entry.pop("failing_since", None)
            if release in failing:
                entry["failing_since"] = {"date": failing[release], "targets": result["targets"]}
        else:
            if result["outcome"] in GREEN or since.get("targets") not in (None, result["targets"]):
                entry.pop("failing_since", None)
            if result["outcome"] in FAILED:
                entry.setdefault(
                    "failing_since", {"date": run["date"], "targets": result["targets"]}
                )
        # Days, not runs: a second run on the same day is the same outage seen twice.
        if result["outcome"] in NOT_OURS:
            if entry.get("not_ours_last") != run["date"]:
                entry["not_ours_days"] = entry.get("not_ours_days", 0) + 1
            entry["not_ours_last"] = run["date"]
        else:
            entry["not_ours_days"] = 0
            entry.pop("not_ours_last", None)
        lock = results_dir / f"compat-point-{key}" / "lock.txt"
        if result["outcome"] in GREEN and lock.is_file():
            entry["last_green"] = {
                "date": run["date"],
                "targets": result["targets"],
                "sdk_commit": run["sdk_commit"],
                "selection_sha256": tests.get("selection_sha256"),
                "skipped_ids": tests.get("skipped_ids", []),
                "files": tests.get("files", []),
                "lock": f"last-green/{key}.txt",
            }
            locks[key] = lock
        points[key] = entry
    scope = (plan or {}).get("scope") or {}
    forgotten: list[str] = []
    for key in [key for key in points if key not in results]:
        # A point this run was in a position to plan and did not: the question is no longer
        # asked (the release left the range, a newer line replaced it). Anything outside
        # what the run covers is left exactly as the run that does cover it wrote it.
        identity = points[key].get("point") or {}
        kind = identity.get("kind", "framework")
        subject = identity.get("subject") or key.split("--")[0]
        # A kind nobody declares any more has no run that could cover it, so the first
        # whole run forgets its points; otherwise they would stay on an issue for good.
        retired = kind not in kinds.KINDS and not scope.get("partial", True)
        # A point about another wardex release than the one this run planned is that
        # release's history, kept for its page.
        build = (identity.get("sdk") or {}).get("version")
        builds = scope.get("builds", {}).get(kind)
        if not retired and build is not None and builds is not None and build not in builds:
            continue
        if retired or subject in scope.get("covers", {}).get(kind, ()):
            del points[key]
            forgotten.append(key)
    seen = dict(state.get("kinds") or {})
    if not scope.get("partial", True):
        for kind in scope.get("covers", {}):
            mine = [r for r in results.values() if r.get("kind", "framework") == kind]
            seen[kind] = {"last_complete": run["date"], "points": len(mine)}
    out = {
        "schema": SCHEMA,
        "points": points,
        "kinds": seen,
        "last_scheduled": state.get("last_scheduled"),
        "release": state.get("release"),
        "versions": _versions(state, plan, run),
    }
    if (plan or {}).get("published"):
        out["release"] = plan["published"]["version"]
    out["displaced"] = (
        next_displaced(state, plan, results) if plan else state.get("displaced") or {}
    )
    out["failing"] = _still_asked(failing, plan, results, out["displaced"])

    def keys(remembered: dict) -> set[str]:
        return {item["key"] for items in remembered.values() for item in items}

    for key in keys(state.get("displaced") or {}) - keys(out["displaced"]):
        # Settled. If the release is ever remembered again, what its point said before
        # that must not outrank the newer failure.
        if key in points:
            points[key] = {k: v for k, v in points[key].items() if k != "last_checked"}
    if run["event"] == "schedule":
        out["last_scheduled"] = run["started_at"]
    out["_locks"] = locks
    # A forgotten point's lock goes with it. A key that is planned again later starts with
    # no last pass at all, instead of with a lock the state knows nothing about.
    out["_forget"] = forgotten
    return out


# --- issues -------------------------------------------------------------------


def _version(result: dict) -> str:
    return ", ".join(f"{dist} {version}" for dist, version in result["targets"].items())


def _planned_sdk(result: dict) -> dict:
    """Where a row's wardex came from, whether the row is of this run or remembered."""
    planned = result.get("planned_sdk")
    return planned if planned is not None else result.get("sdk") or {"origin": "build"}


def _wardex(result: dict) -> str:
    sdk = _planned_sdk(result)
    if sdk.get("origin", "build") == "build":
        return "this commit"
    return f"{code(sdk.get('version', ''), 40)} from PyPI"


def _table(rows: list[dict]) -> list[str]:
    lines = [
        "| Point | wardex | Version | Python | Result | Cause |",
        "|---|---|---|---|---|---|",
    ]
    for result in rows:
        lines.append(
            f"| {ROLE_WORDS.get(result['role'], clean(result['role']))} | {_wardex(result)} "
            f"| {code(_version(result))} "
            f"| {code(result.get('python') or '-')} | {OUTCOME_WORDS[result['outcome']]} "
            f"| {CAUSE_WORDS.get(result['cause'], '')} |"
        )
    return lines


def _left_out(rows: list[dict]) -> list[str]:
    """Test files a point did not run, said plainly: fewer tests is not the same pass."""
    lines = []
    for result in rows:
        excluded = (result.get("tests") or {}).get("excluded_files") or []
        if excluded:
            names = ", ".join(code(name) for name in excluded)
            lines.append(
                f"- On {code(_version(result))}, not run: {names}. What these files need could "
                "not be installed beside this release."
            )
    return lines


def _fingerprint(items: list) -> str:
    if not items:
        return "clear"
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()[:16]


def _remembered_rows(state: dict | None, subject: str, results: dict[str, dict]) -> dict:
    """What the state holds about `subject` that this run did not run, as result rows.

    A point of another cadence, or one this run did not cover, said what it said on the
    day it last ran. It stays on the issue, dated, until the run that covers it says
    otherwise: a daily run must not close what a weekly run found."""
    rows = {}
    for key, entry in ((state or {}).get("points") or {}).items():
        identity, last = entry.get("point"), entry.get("last") or {}
        if key in results or not identity or identity.get("subject") != subject:
            continue
        if identity.get("kind", "framework") not in kinds.KINDS:
            continue  # a kind that was retired asks nothing of anyone
        build = (identity.get("sdk") or {}).get("version")
        if build is not None and build != (state or {}).get("release"):
            # A wardex release a newer one replaced. Its points are history, shown on its
            # own page; nobody is asked to do anything about them.
            continue
        said, checked = last, entry.get("last_checked") or {}
        if last.get("outcome") not in GREEN | FAILED and checked.get("targets") == identity.get(
            "targets"
        ):
            # Its own last run could not check it (the index was down, it would not
            # install). What it said the last time it was checked is what is known.
            said = checked
        rows[key] = {
            **identity,
            "key": key,
            "planned_python": identity.get("python", ""),
            "outcome": said.get("outcome", "no_result"),
            "cause": said.get("cause"),
            "as_of": said.get("date", ""),
            "tests": {
                "failed_ids": said.get("failed_ids", []),
                "excluded_files": said.get("excluded_files", []),
            },
        }
    return rows


def _item_id(adapter: str, item: dict) -> str:
    return release_id(adapter, item["targets"], item.get("python"))


def _lapse(result: dict, known: dict, run: dict) -> str:
    """What a release that has failed for two weeks adds to its item: the way out that is
    not a fix. The README goes on calling the release tested until a person runs it."""
    since = known.get(result["key"], {}).get("failing_since") or {}
    if since.get("targets") != result["targets"]:
        return ""
    return _lapse_since(since.get("date"), result["targets"], run)


def _lapse_since(began: object, targets: dict, run: dict) -> str:
    """`_lapse` for a release and the first day of its failure, wherever that day is kept:
    on its point, or on the item that remembers it while it has no point of its own."""
    since, result = {"date": began}, {"targets": targets}
    try:
        days = (date.fromisoformat(run["date"]) - date.fromisoformat(since["date"])).days
    except (KeyError, TypeError, ValueError):
        return ""
    if days < LAPSE_NARROW_DAYS:
        return ""
    # One command for each release the point pins. Where it pins a framework and a
    # companion, which of the two broke is for a person to say; naming only the first
    # would send them to list the wrong release.
    commands = [
        code(f"python3 scripts/compat/claim.py bad {dist} {version}")
        for dist, version in result["targets"].items()
    ]
    which = (
        commands[0]
        if len(commands) == 1
        else "for the release that broke, " + " or ".join(commands)
    )
    return (
        f" It has failed since {code(since['date'])}, {days} days. To stop calling it tested "
        f"instead: {which}."
    )


def _kind_label(result: dict) -> str:
    kind = kinds.KINDS.get(result.get("kind", "framework"))
    label = kind.label if kind else clean(result.get("kind", ""), 40)
    sdk = _planned_sdk(result)
    if sdk.get("origin", "build") != "build":
        label += f" {code(sdk.get('version', ''), 40)}"
    return label


def _kind_closes(result: dict) -> str:
    kind = kinds.KINDS.get(result.get("kind", "framework"))
    return kind.closes if kind and kind.closes else "when it passes"


def adapter_issue(
    name: str, plan: dict, results: dict[str, dict], run: dict, state: dict | None = None
) -> dict:
    """What the issue for one adapter should say after this run.

    `state` holds what each point said the last time it was actually passed or
    failed. A release inside the range that failed then, and could not be
    checked today (the index was down, it would not install), is still failing
    as far as anyone knows: it stays on the list, worded as such.
    """
    info = plan["adapters"][name]
    known = (state or {}).get("points", {})
    remembered = {item["key"]: item for item in (state or {}).get("displaced", {}).get(name, [])}
    counted = (state or {}).get("failing") or {}
    # Which rows are remembered is decided here, from the state. A field of the row cannot
    # say so: a point job writes its own row, and it is not believed about that.
    old = _remembered_rows(state, name, results)
    results = {**old, **results}
    mine = [r for r in results.values() if r.get("subject", r["adapter"]) == name]
    watched = []
    todo, marks, failing, unchecked = [], [], [], []
    spoken: set[str] = set()
    for result in mine:
        version, outcome = code(_version(result)), result["outcome"]
        tests = result.get("tests") or {}
        nodes = tests.get("failed_ids", [])
        if outcome == "fail" and nodes:
            what = f"fails {len(nodes)} tests"
        elif outcome == "fail":
            what = "does not finish its tests (the run timed out, or failed outside any test)"
        else:
            what = OUTCOME_WORDS[outcome]
        as_of = old[result["key"]]["as_of"] if result["key"] in old else ""
        when = f" (last run {code(as_of)})" if as_of else ""
        if is_watching(result):
            # Reported, and never an argument for red. Said apart, so nobody reads a
            # watching failure as a broken promise or a promise as merely watched.
            if outcome in GREEN or (outcome in NOT_OURS and not result["in_claim"]):
                continue
            watched.append(
                f"{_kind_label(result)}: {version} {what}{when}. Closes {_kind_closes(result)}."
            )
            marks.append([result["key"], "watching", outcome, nodes])
            if nodes:
                failing.append((result, nodes))
            continue
        if result["key"] in old:
            # Not run by this run: what it said stands, dated, and nothing new is claimed.
            # No "passes, widen the range" from a pass nobody saw today.
            if not result["in_claim"]:
                continue
            if outcome not in RED:
                unchecked.append(f"{version} (not run by this run)")
                continue
            todo.append(
                f"{version} is inside the tested range and did not pass when it was last "
                f"run, on {code(as_of)}: {what}. This run did not run it."
            )
            spoken.add(result["key"])
            marks.append([result["key"], "remembered", outcome, nodes])
            if nodes:
                failing.append((result, nodes))
            continue
        if result["in_claim"] and outcome in RED:
            cause = CAUSE_WORDS.get(result["cause"], "not determined")
            closes = (
                "This closes when it passes again, or when the tested range no longer claims it."
            )
            if result["role"] == "displaced":
                closes = (
                    "A newer release has taken its place, so no other point tests it. This "
                    "closes when it passes again, or when the table of tested versions lists "
                    "it under `known_bad`."
                )
            todo.append(
                f"{version} is inside the tested range and does not pass: {what}. "
                f"Cause: {cause}. {closes}{_lapse(result, known, run)}"
            )
            if _lapse(result, known, run):
                # Said once, as a comment, on the day the command first appears.
                marks.append([result["key"], "lapsed"])
        elif result["in_claim"] and outcome not in GREEN:
            # Not checked today. What it said when it last was checked still stands.
            before = known.get(result["key"], {}).get("last_checked") or {}
            item = remembered.get(result["key"]) or next(
                (i for i in remembered.values() if _picked_again(i, {"": result})), None
            )
            pending = False
            if item and before.get("targets") != result["targets"]:
                # A replaced release whose own point has not run it yet: what is known about
                # it is the failure it was remembered for.
                before = {"outcome": "fail", "targets": item["targets"], "date": item["failed_on"]}
                pending = True
            unchecked.append(f"{version} ({OUTCOME_WORDS[outcome]})")
            if before.get("outcome") not in RED or before.get("targets") != result["targets"]:
                continue
            # A release waiting for its own point has its count on the item that remembers it.
            lapse = (
                _lapse_since(counted.get(_item_id(name, item)), item["targets"], run)
                if pending
                else _lapse(result, known, run)
            )
            todo.append(
                f"{version} is inside the tested range. It did not pass when last checked, on "
                f"{code(before['date'])}, and could not be checked today: "
                f"{OUTCOME_WORDS[outcome]}.{lapse}"
            )
            outcome, nodes = before["outcome"], before.get("failed_ids", [])
            if pending:
                # The same mark as on the day it was remembered: that its own point could
                # not run it yet is no news about the release.
                marks.append([result["key"], "displaced", None, []])
            else:
                marks.append([result["key"], outcome, before.get("cause"), nodes])
            if lapse:
                # The same mark as on a day it ran and failed: an outage in the middle of
                # a long failure is not news about the lapse.
                marks.append([result["key"], "lapsed"])
            continue
        elif not result["in_claim"] and outcome in ("fail", "attach_failed", "skips_changed"):
            todo.append(
                f"{version} is newer than the tested range and {what}. Users on it are told "
                "once that it is untested. This closes when the adapter supports it and the "
                "range includes it."
            )
        elif not result["in_claim"] and outcome in GREEN and info["evidence"] == "live":
            todo.append(
                f"{version} is newer than the tested range and passes. The range can be widened "
                "to include it: the table of tested versions, the test pin and the README."
            )
        else:
            continue
        marks.append([result["key"], outcome, result["cause"], nodes])
        if nodes:
            failing.append((result, nodes))

    today = {key: result for key, result in results.items() if key not in old}
    for item in remembered.values():
        # Spoken for already: by its own point or the point of a rule that picked it again,
        # run today, or by the row above that remembers its own point failing.
        if item["key"] in today or item["key"] in spoken or _picked_again(item, today):
            continue
        # Remembered by this run, or by an earlier one whose point for it has not been able
        # to check it since: the failure it was remembered for is what is known.
        version = code(", ".join(f"{dist} {v}" for dist, v in item["targets"].items()))
        lapse = _lapse_since(counted.get(_item_id(name, item)), item["targets"], run)
        todo.append(
            f"{version} is inside the tested range and did not pass when it was last run, on "
            f"{code(item['failed_on'])}. A newer release has taken its place, so it is run on "
            "its own from the next run on. This closes when it passes, or when the table of "
            f"tested versions lists it under `known_bad`.{lapse}"
        )
        marks.append([item["key"], "displaced", None, []])
        if lapse:
            # The way out was already on the issue while the rule's own point ran the
            # release. It does not leave for the day the release changes hands.
            marks.append([item["key"], "lapsed"])

    said = info.get("said") or {}
    claim = "; ".join(
        f"{code(dist)} " + clean(said.get(dist) or ", ".join(f"{line}.x" for line in lines), 80)
        for dist, lines in info["claim"].items()
    )
    fingerprint = _fingerprint(sorted(marks, key=json.dumps))
    body = [
        f"<!-- compat-key: {name} -->",
        f"<!-- compat-fingerprint: {fingerprint} -->",
        f"The daily compatibility run tests the wardex adapter for {code(info['label'])} against "
        "exact framework releases, each in an environment that holds only this framework. "
        "Every run rewrites this issue, and it closes itself when nothing below needs doing.",
        "",
        f"- Tested range: {claim}",
        f"- Newest release on PyPI: {code(info['newest'])}",
        f"- Last run: {code(run['date'])}, wardex {code(run['sdk_version'])} at "
        f"{code(run['sdk_commit'][:10])} ([run]({run['url']}))",
    ]
    if info["evidence"] != "live":
        body.append(
            f"- What a pass proves here: the framework's Python package ran, against recorded "
            f"output of {code(info['replay_of'])}. It does not show that a newer build of that "
            "program behaves the same."
        )
    if info["prerelease_ahead"]:
        body.append(f"- A pre-release exists and is not run: {code(info['prerelease_ahead'])}")
    body += [f"- Note: {clean(note)}" for note in info.get("notes", [])]
    body += ["", "### What needs doing", ""]
    body += [f"- {line}" for line in todo] or ["Nothing."]
    if watched:
        body += ["", "### Watching (never turns the run red)", ""]
        body += [f"- {line}" for line in watched]
    body += ["", "### Points", "", *_table(mine)]
    if _left_out(mine):
        body += ["", *_left_out(mine)]
    for result, nodes in failing:
        body += ["", f"### Failing tests on {code(_version(result))}", ""]
        body += [f"- {code(node, 300)}" for node in nodes[:40]]
        if len(nodes) > 40:
            body.append(f"- and {len(nodes) - 40} more, in the run's log")
    # Said when the issue closes. The run tests the default branch, so a published wardex
    # release older than the commit that fixed this does not contain the fix; and a release
    # that could not be checked today is named, not counted as passing.
    closing = (
        "Nothing here needs doing any more: every release inside the tested range that could be "
        f"checked today passes on wardex {code(run['sdk_version'])} at "
        f"{code(run['sdk_commit'][:10])}."
    )
    if unchecked:
        closing += " Not checked today: " + "; ".join(unchecked) + "."
    closing += (
        " If a change to wardex is what fixed it, a wardex release published before that "
        "commit does not contain the fix. Closing."
    )
    return {
        "key": name,
        "title": f"Compatibility: {info['label']}",
        "body": "\n".join(body) + "\n",
        "fingerprint": fingerprint,
        "open": bool(todo or watched),
        "changes": [*todo, *watched],
        "closing": closing,
    }


def health_issue(plan: dict, results: dict[str, dict], state: dict, run: dict) -> dict:
    """What the one issue about the run itself should say."""
    todo, marks = [], []
    for problem in plan.get("problems", []):
        todo.append(f"The plan could not be made in full: {code(problem)}")
        marks.append(["plan", problem])
    for name, info in sorted(plan["adapters"].items()):
        for problem in info["problems"]:
            todo.append(f"{code(info['label'])}: {code(problem)}")
            marks.append([name, problem])
    for problem in plan.get("unplanned", []):
        todo.append(f"Not planned, and so not run (this never turns the run red): {code(problem)}")
        marks.append(["unplanned", problem])
    for key, result in sorted(results.items()):
        outcome = result["outcome"]
        streak = state.get("points", {}).get(key, {}).get("not_ours_days", 0)
        if outcome in ("no_result", "harness_error"):
            detail = result.get("harness_error") or result.get("note") or ""
            said = f" It said: {code(detail, 300)}" if detail else ""
            todo.append(f"{code(key)} left no usable result.{said}")
        elif outcome == "test_deps_unresolvable":
            todo.append(
                f"{code(key)}: {code(_version(result))} installs on its own, but not beside what "
                "the tests need. A test dependency has not caught up with it, so nothing was "
                "tested on this release."
            )
        elif outcome in NOT_OURS and streak >= 2:
            todo.append(
                f"{code(key)}: {code(_version(result))} {OUTCOME_WORDS[outcome]}, "
                f"on {streak} days running."
            )
        else:
            continue
        marks.append([key, outcome])
    body = [
        "<!-- compat-key: health -->",
        f"<!-- compat-fingerprint: {_fingerprint(sorted(marks))} -->",
        "Problems with the daily compatibility run itself, as opposed to with a framework "
        "release. Every run rewrites this issue, and it closes itself when the list is empty.",
        "",
        f"- Last run: {code(run['date'])} ([run]({run['url']}))",
        "",
        "### What needs doing",
        "",
    ]
    body += [f"- {line}" for line in todo] or ["Nothing."]
    flaky = sorted(key for key, result in results.items() if result["outcome"] == "flaky")
    if flaky:
        body += ["", "### Points where a test failed once and passed on a second run", ""]
        body += [f"- {code(key)}" for key in flaky]
    return {
        "key": "health",
        "title": "Compatibility: daily run health",
        "body": "\n".join(body) + "\n",
        "fingerprint": _fingerprint(sorted(marks)),
        "open": bool(todo),
        "changes": todo,
    }


def died_issue(url: str = "") -> dict:
    """The issue for a run that ended without a verdict; `url` empty means none did.

    Opened by the run that died, which knows only its own address, and closed
    by the next run that reaches a verdict. Its own key, so the watchdog, which
    answers a different question, never closes it.
    """
    todo = [f"A run ended before it could report: {url}"] if url else []
    body = [
        "<!-- compat-key: run-died -->",
        f"<!-- compat-fingerprint: {_fingerprint(todo)} -->",
        "A daily compatibility run that ends without a verdict leaves no result to read, "
        "so it says so here. The next run that reaches a verdict closes this.",
        "",
        "### What needs doing",
        "",
    ]
    body += [f"- {line}" for line in todo] or ["Nothing."]
    return {
        "key": "run-died",
        "title": "Compatibility: a daily run ended without a verdict",
        "body": "\n".join(body) + "\n",
        "fingerprint": _fingerprint(todo),
        "open": bool(todo),
        "changes": todo,
        "closing": "A later run reached a verdict. Closing.",
    }


def wanted_issues(plan: dict, results: dict[str, dict], state: dict, run: dict) -> list[dict]:
    """Every issue this run has a say about.

    A run of one adapter knows nothing about the others, so it has no say
    about the two issues that speak for the whole run and must not close them.
    """
    wanted = [adapter_issue(name, plan, results, run, state) for name in sorted(plan["adapters"])]
    if not plan.get("partial"):
        wanted += [health_issue(plan, results, state, run), died_issue()]
    return wanted


class GitHub:
    """The few issue calls the report makes, through the `gh` already on the runner."""

    def __init__(self, repo: str, *, write: bool) -> None:
        self.repo, self.write, self.log = repo, write, []

    def _api(self, method: str, path: str, payload: dict | None = None):
        cmd = ["gh", "api", "--method", method, f"repos/{self.repo}/{path}"]
        if method == "GET":
            cmd += ["--paginate", "--slurp"]
        done = subprocess.run(
            cmd + (["--input", "-"] if payload is not None else []),
            input=json.dumps(payload) if payload is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(done.stdout) if done.stdout.strip() else None

    def issues(self) -> list[dict]:
        """Every issue carrying the label, open or closed.

        The list endpoint, not search: search lags behind an issue opened a
        minute ago, and a lagging lookup is how a duplicate gets opened.
        """
        pages = self._api("GET", f"issues?labels={LABEL}&state=all&per_page=100") or []
        return [issue for page in pages for issue in page if "pull_request" not in issue]

    def ensure_label(self) -> None:
        if not self.write:
            return
        try:
            self._api("POST", "labels", {"name": LABEL, "color": "5319e7"})
        except subprocess.CalledProcessError:
            pass  # it already exists

    def do(self, what: str, method: str, path: str, payload: dict) -> None:
        self.log.append(what)
        if self.write:
            self._api(method, path, payload)


def sync_issue(gh: GitHub, wanted: dict, existing: list[dict], assignee: str = "") -> None:
    """Make the issue with this key say what `wanted` says, without duplicating it.

    Found by a marker in the body, never by title, so a renamed issue is still
    the same issue. An open one is preferred to a closed one, so that closing a
    duplicate by hand never leaves a live failure with no open issue. A person
    who closes the only one while it still has something to say is respected
    until what it has to say changes.
    """
    marker = f"<!-- compat-key: {wanted['key']} -->"
    same = sorted(
        (issue for issue in existing if marker in (issue.get("body") or "")),
        key=lambda issue: (issue["state"] == "open", issue["number"]),
    )
    mine = same[-1] if same else None
    for extra in same[:-1]:
        if extra["state"] == "open":
            gh.do(
                f"close duplicate #{extra['number']}",
                "PATCH",
                f"issues/{extra['number']}",
                {"state": "closed", "state_reason": "not_planned"},
            )
    was = None
    if mine:
        found = re.search(r"<!-- compat-fingerprint: (\w+) -->", mine["body"] or "")
        was = found.group(1) if found else None
    changed = was != wanted["fingerprint"]
    if mine is None:
        if wanted["open"]:
            payload = {"title": wanted["title"], "body": wanted["body"], "labels": [LABEL]}
            if assignee:
                payload["assignees"] = [assignee]
            gh.do(f"open {wanted['title']}", "POST", "issues", payload)
        return
    path = f"issues/{mine['number']}"
    if mine["state"] == "closed":
        if not wanted["open"]:
            if changed:
                # Closed by a person while it failed, and now it passes: record that quietly,
                # so the same failure coming back is seen as a change and reopens it.
                gh.do(f"rewrite #{mine['number']}", "PATCH", path, {"body": wanted["body"]})
            return
        if not changed:
            return
    gh.do(f"rewrite #{mine['number']}", "PATCH", path, {"body": wanted["body"]})
    if wanted["open"]:
        if mine["state"] == "closed":
            gh.do(f"reopen #{mine['number']}", "PATCH", path, {"state": "open"})
        if changed:
            text = "What needs doing changed:\n\n" + "\n".join(f"- {c}" for c in wanted["changes"])
            gh.do(f"comment on #{mine['number']}", "POST", f"{path}/comments", {"body": text})
    else:
        text = wanted.get("closing") or "Nothing here needs doing any more. Closing."
        gh.do(f"comment on #{mine['number']}", "POST", f"{path}/comments", {"body": text})
        gh.do(f"close #{mine['number']}", "PATCH", path, {"state": "closed"})


# --- the run -------------------------------------------------------------------


def summary(plan: dict, results: dict[str, dict], reasons: list[str], actions: list[str]) -> str:
    lines = ["## Compatibility run", ""]
    lines.append("**Red:** " + "; ".join(clean(r) for r in reasons) if reasons else "**Green.**")
    for name, info in sorted(plan["adapters"].items()):
        mine = [r for r in results.values() if r["adapter"] == name]
        lines += ["", f"### {clean(info['label'])}", "", *_table(mine)]
        if _left_out(mine):
            lines += ["", *_left_out(mine)]
        for result in mine:
            nodes = (result.get("tests") or {}).get("failed_ids", [])
            if nodes:
                lines += ["", f"Failing on {code(_version(result))}:"]
                lines += [f"- {code(node, 300)}" for node in nodes[:40]]
    lines += ["", "### Issue actions", ""]
    lines += [f"- {clean(action)}" for action in actions] or ["- none"]
    return "\n".join(lines) + "\n"


def run_info(plan: dict) -> dict:
    env = os.environ.get
    started = plan["cutoff"]
    return {
        "id": env("GITHUB_RUN_ID", ""),
        "attempt": env("GITHUB_RUN_ATTEMPT", ""),
        "url": f"{env('GITHUB_SERVER_URL', 'https://github.com')}/"
        f"{env('GITHUB_REPOSITORY', '')}/actions/runs/{env('GITHUB_RUN_ID', '')}",
        "event": env("GITHUB_EVENT_NAME", ""),
        "cadence": plan.get("cadence", "daily"),
        "scope": "partial" if plan.get("partial") else "all",
        "ref": env("GITHUB_REF", ""),
        "started_at": started,
        "date": started[:10],
        "sdk_source": "main" if env("GITHUB_REF") == "refs/heads/main" else "branch",
        "sdk_commit": env("GITHUB_SHA", ""),
        "sdk_version": env("COMPAT_SDK_VERSION", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--results", required=True, help="one directory per point artifact")
    parser.add_argument("--state-dir", required=True, help="a checkout of the records branch")
    parser.add_argument("--write", action="store_true", help="write records and issues")
    parser.add_argument("--preview", help="a directory to render the public pages into")
    args = parser.parse_args(argv)

    state_dir, results_dir = Path(args.state_dir), Path(args.results)
    try:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("no plan: the run cannot be judged", file=sys.stderr)
        return RED_EXIT
    state_file = state_dir / "state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
    state = model.upgrade_state(state)
    run = run_info(plan)

    results = load_results(plan, results_dir)
    apply_skip_rule(results, state)
    reasons = judge(plan, results)
    new_state = next_state(state, results, run, results_dir, plan)
    locks, forgotten = new_state.pop("_locks"), new_state.pop("_forget")

    gh = GitHub(os.environ.get("GITHUB_REPOSITORY", ""), write=args.write)
    wanted = wanted_issues(plan, results, new_state, run)
    existing = gh.issues() if os.environ.get("GITHUB_REPOSITORY") else []
    gh.ensure_label()
    for issue in wanted:
        sync_issue(gh, issue, existing, os.environ.get("COMPAT_ASSIGNEE", ""))

    if args.write:
        day = state_dir / "records" / run["date"][:4] / run["date"][5:7]
        day.mkdir(parents=True, exist_ok=True)
        with (day / f"{run['date'][8:10]}.jsonl").open("a", encoding="utf-8") as handle:
            for line in records(plan, results, run):
                handle.write(json.dumps(line, sort_keys=True) + "\n")
        (state_dir / "last-green").mkdir(exist_ok=True)
        for key in forgotten:
            for suffix in (".txt", ".files.json"):
                (state_dir / "last-green" / f"{key}{suffix}").unlink(missing_ok=True)
        for key, lock in locks.items():
            (state_dir / "last-green" / f"{key}.txt").write_bytes(lock.read_bytes())
            files = new_state["points"][key]["last_green"]["files"]
            (state_dir / "last-green" / f"{key}.files.json").write_text(
                json.dumps(files), encoding="utf-8"
            )
        state_file.write_text(json.dumps(new_state, indent=1, sort_keys=True), encoding="utf-8")
        render.write_site(state_dir, new_state, run["date"])
    if args.preview:
        # What the pages would say after this run, for a run that writes nothing.
        render.write_site(Path(args.preview), new_state, run["date"])

    text = summary(
        plan, results, reasons, gh.log if args.write else [f"(dry run) {a}" for a in gh.log]
    )
    print(text)
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(text)
    return RED_EXIT if reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
