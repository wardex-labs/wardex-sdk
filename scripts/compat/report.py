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
from collections.abc import Iterable
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compat import gate, kinds, model, render  # noqa: E402
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
    "prerelease": "pre-release ahead of the newest release",
    "devbranch": "head of the default branch",
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
                # What the installer said when it refused: for frameworks that cannot be
                # installed together, this is the finding.
                "install_detail": _refusal(result, 400),
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


#: Points whose release nobody is owed a run of when the point leaves the plan: the floor
#: runs a release with another time's packages, and a replaced release's own point leaves
#: the plan only when the plan no longer claims the release.
_NOT_OWED = ("floor", "displaced")


def next_displaced(
    state: dict, plan: dict, results: dict[str, dict], *, late: bool = False
) -> dict:
    """Releases inside the tested range that failed and that no rule picks any more.

    The newest release of a line is a point only while it is the newest. One
    that fails there and is then replaced by a newer patch would drop out of
    every later run while the README still calls it tested. It is remembered
    here and the next plan makes it a point of its own. It leaves the list when
    that point passes, or when the plan stops making the point: the table lists
    the release as bad, the range no longer holds it, or the index withdrew it.

    The plan of a run that is behind (`late`) is no evidence of what is planned now: a
    release it made no point of stays remembered, and a point it has no result for is not
    a point that is no longer planned.
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
            if result is None and late:
                kept.setdefault(adapter, []).append(item)
                continue
            # No point at all means the plan no longer claims the release. A point that ran
            # it settles it either way: a pass ends it, and a failure at a rule's own point
            # is on that point's record from here on.
            if result is None or result["outcome"] in GREEN:
                continue
            if result["role"] != "displaced" and result["outcome"] in FAILED:
                continue
            kept.setdefault(adapter, []).append(item)
    for key, entry in sorted((state.get("points") or {}).items()):
        before = entry.get("last_checked") or {}
        if before.get("outcome") not in FAILED or not isinstance(before.get("targets"), dict):
            continue
        result = results.get(key)
        if result is not None:
            # Its point is still planned, on another release now.
            moved = _own(result) and result["in_claim"] and result["role"] != "displaced"
            if not moved or before["targets"] == result["targets"]:
                continue
            adapter = result["adapter"]
            python = before.get("python") or result.get("planned_python") or ""
        elif late:
            continue
        else:
            # Its point is no longer planned at all: the rules moved (the floor now sits on
            # its line, the line it was the newest patch of became the top one). The
            # release is as untested as one a newer patch replaced.
            identity = entry.get("point") or {}
            mine = (
                not identity.get("watching")
                and (identity.get("sdk") or {}).get("origin", "build") == "build"
            )
            if not mine or not identity.get("in_claim") or identity.get("role") in _NOT_OWED:
                continue
            adapter = identity.get("adapter", "")
            python = before.get("python") or identity.get("python") or ""
        info = plan["adapters"].get(adapter) or {}
        entries = info.get("entries") or []
        if not entries or info.get("problems"):
            continue
        rows = {row["distribution"]: row for row in entries}
        # A range narrowed past the release, or a release now listed as bad, is settled.
        # Every pinned distribution is asked of its own row.
        if not all(
            dist in rows and isinstance(v, str) and claims(rows[dist], v)
            for dist, v in before["targets"].items()
        ):
            continue
        # So is one the index no longer serves, for any distribution the point pinned: no
        # point can be made for it.
        served = info.get("served")
        if isinstance(served, dict) and not all(
            v in served.get(dist, ()) for dist, v in before["targets"].items()
        ):
            continue
        item = {
            "key": f"{adapter}--displaced--{displaced_line(before['targets'], python)}",
            "targets": before["targets"],
            "python": python,
            "failed_on": before["date"],
        }
        # A rule's point that checked the same release on the same interpreter today
        # speaks for it: its pass settles it, its failure is on that point's record.
        spoke = _picked_again(item, results)
        if spoke is not None and spoke["outcome"] in GREEN | FAILED:
            continue
        mine_ = kept.setdefault(adapter, [])
        if all(other["key"] != item["key"] for other in mine_):
            mine_.append(item)
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
    # Asked by a point that counts, or by the list. A point above the range, the floor or
    # another wardex build running the same release asks a different question, and must
    # not keep this one's count alive after the release left the range.
    asked = {
        release_id(r["adapter"], r["targets"], r.get("planned_python"))
        for r in results.values()
        if _counted(r)
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
    # Which build the rows are of. Only a whole run that asked about the tested ranges
    # ran them all at its own commit: a run of one adapter, or one of a cadence that plans
    # no framework point, must not put its commit on rows another run found.
    whole = not plan.get("partial") and bool(plan["adapters"])
    stamp = run if whole or not main else {f"sdk_{k}": main.get(k) for k in ("version", "commit")}
    versions["main"] = {
        "version": stamp["sdk_version"],
        "commit": stamp["sdk_commit"],
        "date": run["date"] if whole or not main else main.get("date"),
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
            "override": release.get("override"),
            "date": run["date"],
            "adapters": merged(old, release["adapters"]),
        }
    return versions


#: How long the list of runs remembers a run. The release gate reads the last two days.
RUN_LIST_DAYS = 7
_RUN_KEPT = "id attempt started_at date scope cadence ref sdk_commit url".split()
_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")


def run_list(before: object, run: dict, *, now: str | None = None) -> list[dict]:
    """The runs that wrote records, this one last: what the release gate takes as the
    list of runs there were. A run that planned no point writes no line, and is here.

    One entry per run and attempt. `red` and `points` are there once the run has a
    verdict; a state built before that (a preview) lists the run without them, and the
    gate reads such an entry as one it cannot use. `now` is the day this is written
    (UTC today when not given), which is not always the run's: a run is dated by its plan."""
    entry = {field: run.get(field) for field in _RUN_KEPT}
    entry.update({field: run[field] for field in ("red", "points") if field in run})
    today = run["date"]
    now = now or datetime.now(timezone.utc).date().isoformat()
    oldest = (date.fromisoformat(today) - timedelta(days=RUN_LIST_DAYS)).isoformat()
    # A state from before the list existed has none, and that is a list of nothing. Anything
    # else that is not a list is damage, and is kept as that.
    held = [] if before is None else before if isinstance(before, list) else [before]
    # The newest day there is: the day this is written, this run's, and those the list
    # knows. A run is dated by its plan, and failed jobs run again weeks later still carry
    # the old plan's day: a mark dated by such a run would be old the day it was made, and
    # the next ordinary run would drop it.
    days = [old["date"] for old in held if gate.entry_ok(old) and _DAY.fullmatch(old["date"])]
    latest = max([today, now, *days])
    kept = []
    for old in held:
        # Dated by its plan, and so possibly after this run: failed jobs of yesterday's run,
        # run again today, write with yesterday's plan after today's run has written.
        if gate.entry_ok(old) and _DAY.fullmatch(old["date"]):
            mine = (old["id"], old["attempt"]) == (entry["id"], entry["attempt"])
            if old["date"] >= oldest and not mine:
                kept.append(old)
            continue
        # Not an entry this function writes. It may have been the run that said a commit
        # is not fit to release, so it is not dropped by the next run that comes along: it
        # stays, as something the gate cannot read, for as long as an entry would.
        seen = old.get("unreadable_since") if isinstance(old, dict) else None
        if not (isinstance(seen, str) and _DAY.fullmatch(seen) and seen <= latest):
            seen = latest
        if seen >= oldest:
            kept.append({"unreadable_since": seen})
    return [*kept, entry]


#: What `runs` is set to when the state holds no list and the records show there was one.
LIST_GONE = "gone"


def behind(state: dict, run: dict, tip: str | None, ran: Iterable[str] = ()) -> str:
    """Why this run is not the one to say how things stand now, or "".

    The state is what the newest run of the newest code knows. Two runs are not that. A
    run planned before a run that has already reported: its failed jobs were run again
    after a later run wrote. And a run started again at a commit that is no longer the tip
    of its branch: GitHub re-runs a run with the code of its own commit, which does not
    know what was declared since. Either, acting as the newest word, would undo what the
    newer run wrote, forget points its plan has not heard of and close their issues. Such
    a run adds what only it knows and takes nothing away: it forgets nothing, writes
    nothing over what a later run found, and has no say about how the run itself stands.

    `tip` is the commit the run's branch points at now, or empty when that is not known.
    `ran` are the points this run has a result for. The list of runs forgets a run after a
    week and a run can be started again for longer than that, so a point that itself holds
    a newer word than this run's says the same thing the list would have said."""
    mine = gate.run_of({"run": run})
    for entry in state.get("runs") or [] if isinstance(state.get("runs"), list) else []:
        if not gate.entry_ok(entry) or entry["ref"] != run["ref"]:
            continue
        same = (entry.get("cadence") or kinds.ISSUE_WRITER) == run["cadence"]
        if same and mine is not None and gate.run_of({"run": entry}) > mine:
            return "a run planned after this one has already reported"
    if off_tip(run, tip):
        return (
            "this is a run started again at a commit that is no longer the tip of its branch"
            if tip
            else "this is a run started again, and what its branch points at now is not known"
        )
    points = state.get("points") or {}
    for key in ran:
        last = (points.get(key) or {}).get("last") or {}
        # Only a point that names the run that wrote it: one written before runs were kept
        # on points is told by its day, which a run planned later the same day shares.
        if _named(last) and spoken_for(last, run):
            return "a point this run ran already holds the word of a newer build or a later run"
    return ""


def off_tip(run: dict, tip: str | None) -> bool:
    """Whether this is a run started again at a commit its branch no longer points at.

    A first attempt starts at the tip, whatever is pushed while it runs. A later attempt
    runs the code of the run's own commit, and builds that commit's wheel."""
    return str(run.get("attempt") or "1") != "1" and tip != run["sdk_commit"]


def _named(last: dict) -> bool:
    """Whether what a point last said names the run that said it."""
    by = last.get("run")
    return isinstance(by, list) and len(by) == 2 and all(isinstance(part, str) for part in by)


def spoken_for(last: dict, run: dict) -> bool:
    """Whether what a point last said is the word of a newer build, or a later word of
    the same build, which a run that is behind must leave standing.

    Not by the day: two runs planned on one day are told apart. And not by the plan
    alone: every job of an old run, run again, makes a plan today, and still builds its
    old commit. Two things are known of the run that wrote a point, its id and its
    commit, and of them:

    - Another commit is another build, and which is the newer cannot be read from the
      commits. It can be read from the runs: a run keeps its id and its commit through
      every attempt, ids rise with each new run, a branch moves forward and a run starts
      at its tip. The run created later built the newer commit, whenever either was
      planned.
    - The same commit is the same build, and the word planned later is the later one.

    A point written before the run was kept on it is told by its day, and on the same day
    is left standing."""
    by = last.get("run")
    if _named(last):
        if by[1] != run["id"] and last.get("commit") != run["sdk_commit"]:
            # Ids are numbers written as text: the longer is the greater.
            return (len(by[1]), by[1]) > (len(run["id"]), run["id"])
        return (by[0], by[1]) > (run["started_at"], run["id"])
    return str(last.get("date") or "") >= run["date"]


def list_was_kept(state_dir: Path, days: list[str]) -> bool:
    """Whether a record line of the last week was written by a report that kept the list.

    Such a line carries how many points its run wrote lines for, a stamp that came with the
    list. A state with no list beside such lines did not start without one: the list was
    removed, and the next run must not be what quietly starts it again from nothing."""
    for day in days:
        try:
            start = date.fromisoformat(day)
        except ValueError:
            continue
        for back in range(RUN_LIST_DAYS + 1):
            at = start - timedelta(days=back)
            path = state_dir / "records" / f"{at:%Y}" / f"{at:%m}" / f"{at:%d}.jsonl"
            if not path.is_file():
                continue
            for raw in path.read_bytes().splitlines():
                try:
                    stamp = json.loads(raw.decode("utf-8"))["run"]["points"]
                except Exception:  # noqa: BLE001, S112 - a line that is not one says nothing
                    continue
                if isinstance(stamp, int) and not isinstance(stamp, bool):
                    return True
    return False


def next_state(
    state: dict,
    results: dict[str, dict],
    run: dict,
    results_dir: Path,
    plan: dict | None = None,
    *,
    late: bool = False,
) -> dict:
    """The small file the next run reads: each point's last record, last check and last pass,
    and the failed releases a newer one has replaced.

    `late` is a run that is behind (`behind`). It adds what only it knows: a point no
    later run has spoken for takes its result, and a release it saw failing that no point
    runs any more is remembered. It takes nothing away: nothing is forgotten, and a point
    a newer build or a later run has spoken for keeps what it said (`spoken_for`)."""
    points = dict(state.get("points", {}))
    locks: dict[str, Path] = {}
    found = results
    if late:
        results = {
            key: result
            for key, result in found.items()
            if not spoken_for((points.get(key) or {}).get("last") or {}, run)
        }
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
            # Which run wrote this, as the list of runs orders them: a run that is behind
            # leaves standing what a run planned after it said.
            "run": [run["started_at"], run["id"]],
            # Which commit of this repository ran it: a run of one adapter, and the weekly
            # run, leave rows of another commit than the one the page is headed with.
            "commit": run["sdk_commit"],
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
    gone: list[dict] = []
    for key in [] if late else [key for key in points if key not in results]:
        # A point this run was in a position to plan and did not: the question is no longer
        # asked (the release left the range, a newer line replaced it). Anything outside
        # what the run covers is left exactly as the run that does cover it wrote it.
        identity = points[key].get("point") or {}
        kind = identity.get("kind", "framework")
        subject = identity.get("subject") or key.split("--")[0]
        # A kind nobody declares any more has no run that could cover it, so the first
        # whole run forgets its points; otherwise they would stay on an issue for good.
        # Only the run that writes issues does this, so that the issue a retired check kept
        # open hears of it from the same run that forgets it.
        whole = not scope.get("partial", True)
        writer = scope.get("cadence", kinds.ISSUE_WRITER) == kinds.ISSUE_WRITER
        retired = kind not in kinds.KINDS and whole and writer
        # A subject its kind's rule no longer knows (a set of frameworks taken out of the
        # declarations) is planned by no run again. The run that writes issues forgets it,
        # whichever run used to plan it, and leaves a trace so that its issue is closed.
        known = scope.get("declared", {}).get(kind)
        undeclared = whole and writer and known is not None and subject not in known
        # A point about another wardex release than the one this run planned is that
        # release's history, kept for its page.
        build = (identity.get("sdk") or {}).get("version")
        builds = scope.get("builds", {}).get(kind)
        if not retired and build is not None and (builds is None or build not in builds):
            # Also on a day the newest release could not be read: with no release planned,
            # every remembered one is another release than the one this run planned.
            continue
        if retired or undeclared or subject in scope.get("covers", {}).get(kind, ()):
            last = points[key].get("last") or {}
            if retired or undeclared:
                # The question itself went away: no later run knows of this point, so this
                # run leaves a trace for the issue it may have kept open. The issue is
                # closed by the same run, and says a check that was not passing is gone,
                # not fixed.
                gone.append(
                    {
                        "subject": subject,
                        "key": key,
                        "kind": kind,
                        "date": last.get("date"),
                        "undeclared": not retired,
                        "failing": last.get("outcome") not in GREEN,
                    }
                )
            del points[key]
            forgotten.append(key)
    seen = dict(state.get("kinds") or {})
    if not scope.get("partial", True) and not late:
        for kind in scope.get("covers", {}):
            mine = [r for r in results.values() if r.get("kind", "framework") == kind]
            seen[kind] = {"last_complete": run["date"], "points": len(mine)}
    out = {
        "schema": SCHEMA,
        "points": points,
        "kinds": seen,
        "last_scheduled": state.get("last_scheduled"),
        "release": state.get("release"),
        # Which build the pages are of is the later run's to say.
        "versions": state["versions"]
        if late and state.get("versions")
        else _versions(state, plan, run),
        "runs": run_list(state.get("runs"), run),
        # Of this run only: the next run has nothing left to retire.
        "retired": gone,
    }
    if (plan or {}).get("published") and not late:
        out["release"] = plan["published"]["version"]
    out["displaced"] = (
        next_displaced(state, plan, results, late=late) if plan else state.get("displaced") or {}
    )
    if late:
        # A release it saw failing, under a point a later run has moved to another
        # release: no point runs it any more, and this run is the only one that knows.
        for key in sorted(set(found) - set(results)):
            result, now = found[key], (points[key].get("last") or {}).get("targets")
            if not (_counted(result) and result["outcome"] in FAILED):
                continue
            if now == result["targets"] or result["role"] == "displaced":
                continue
            python = result.get("planned_python") or ""
            item = {
                "key": f"{result['adapter']}--displaced--"
                f"{displaced_line(result['targets'], python)}",
                "targets": result["targets"],
                "python": python,
                "failed_on": run["date"],
            }
            kept = out["displaced"].setdefault(result["adapter"], [])
            if all(other["key"] != item["key"] for other in kept):
                kept.append(item)
            failing.setdefault(
                release_id(result["adapter"], result["targets"], python), run["date"]
            )
        out["displaced"] = {
            name: items for name, items in sorted(out["displaced"].items()) if items
        }
    out["failing"] = failing if late else _still_asked(failing, plan, results, out["displaced"])

    def keys(remembered: dict) -> set[str]:
        return {item["key"] for items in remembered.values() for item in items}

    for key in keys(state.get("displaced") or {}) - keys(out["displaced"]):
        # Settled. If the release is ever remembered again, what its point said before
        # that must not outrank the newer failure.
        if key in points:
            points[key] = {k: v for k, v in points[key].items() if k != "last_checked"}
    writer = run.get("cadence", kinds.ISSUE_WRITER) == kinds.ISSUE_WRITER
    if run["event"] == "schedule" and writer and not late:
        out["last_scheduled"] = run["started_at"]
    # What a run that is behind found for points a later run has spoken for: on its lines,
    # and on no issue as today's word.
    out["_older"] = sorted(set(found) - set(results))
    out["_locks"] = locks
    # A forgotten point's lock goes with it. A key that is planned again later starts with
    # no last pass at all, instead of with a lock the state knows nothing about.
    out["_forget"] = forgotten
    return out


# --- issues -------------------------------------------------------------------


def _version(result: dict) -> str:
    # A scoped npm package is named with an at-sign, and no at-sign is ever written to an
    # issue: the scope and the name say which package it is.
    return ", ".join(
        f"{dist.removeprefix('@')} {version}" for dist, version in result["targets"].items()
    )


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
            f"| {_point_words(result)} | {_wardex(result)} "
            f"| {code(_version(result))} "
            f"| {code(result.get('python') or '-')} | {_outcome_words(result)} "
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


#: What an issue says of a subject that is not an adapter with a tested range.
_NO_RANGE = {
    "evidence": "live",
    "replay_of": None,
    "claim": {},
    "said": {},
    "newest": None,
    "prerelease_ahead": None,
    "notes": [],
    "problems": [],
}


def subject_info(name: str, plan: dict) -> dict:
    """What is known about a subject: the adapter's facts when it is one the plan worked
    out, else its name alone. A set of frameworks run together is a subject, and so is a
    program with no tested range; neither has a row in the table of tested versions."""
    info = plan["adapters"].get(name)
    if info:
        return info
    label = ((plan.get("subjects") or {}).get(name) or {}).get("label")
    return {**_NO_RANGE, "label": label if isinstance(label, str) and label else name}


def _outcome_words(result: dict) -> str:
    """What a row's outcome is called. Several frameworks that will not install are not a
    matter of the interpreter: they cannot be installed together."""
    if result["outcome"] == "not_installable" and len(result.get("adapters") or []) > 1:
        return "cannot be installed together"
    return OUTCOME_WORDS[result["outcome"]]


def _refusal(result: dict, limit: int) -> str:
    """What the installer said when it refused, on one line."""
    detail = (result.get("install") or {}).get("detail", "")
    return clean(" ".join(str(detail).split()), limit)


def _point_words(result: dict) -> str:
    """What a row is, in the table of an issue or a summary: its role, and its kind when
    that is not the plain question of a framework release."""
    role = ROLE_WORDS.get(result["role"], clean(result["role"]))
    built = _planned_sdk(result).get("origin", "build") == "build"
    if result.get("kind", "framework") == "framework" or not built:
        # A release from PyPI is told apart by the column that says which wardex it is.
        return role
    return f"{_kind_label(result)}: {role}"


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
    info = subject_info(name, plan)
    known = (state or {}).get("points", {})
    remembered = {item["key"]: item for item in (state or {}).get("displaced", {}).get(name, [])}
    counted = (state or {}).get("failing") or {}
    # Which rows are remembered is decided here, from the state. A field of the row cannot
    # say so: a point job writes its own row, and it is not believed about that.
    old = _remembered_rows(state, name, results)
    results = {**old, **results}
    mine = [r for r in results.values() if r.get("subject", r["adapter"]) == name]
    watched = []
    # Lines of watching points that left no result. They keep an open issue open and open
    # none: what a point could not say is no news about its subject.
    held: list[str] = []
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
            what = _outcome_words(result)
            said = _refusal(result, 300)
            if outcome == "not_installable" and said:
                # In a code span: what an installer prints can hold anything.
                what += f" (the installer said: {code(said, 300)})"
        as_of = old[result["key"]]["as_of"] if result["key"] in old else ""
        when = f" (last run {code(as_of)})" if as_of else ""
        if is_watching(result):
            # Reported, and never an argument for red. Said apart, so nobody reads a
            # watching failure as a broken promise or a promise as merely watched.
            if outcome in GREEN:
                continue
            # Not checked today: it left no result (the run's own failure, on the health
            # issue as that), or it could not be installed. Several frameworks that will
            # not install together are not that: it is what a point of that kind finds.
            found = outcome == "not_installable" and len(result.get("adapters") or []) > 1
            unseen = outcome in ("no_result", "harness_error")
            if unseen or (outcome in NOT_OURS and not found):
                before = known.get(result["key"], {}).get("last_checked") or {}
                if before.get("outcome") in FAILED and before.get("targets") == result["targets"]:
                    # What the point said when this release was last checked still stands,
                    # and is the same thing to say as on the day it said it: a day nothing
                    # could be checked must not close an issue.
                    watched.append(
                        f"{_kind_label(result)}: {version} did not pass when it was last "
                        f"checked, on {code(before.get('date', ''))}, and could not be checked "
                        f"since ({what}). Closes {_kind_closes(result)}."
                    )
                    marks.append(
                        [result["key"], "watching", before["outcome"], before.get("failed_ids", [])]
                    )
                    continue
                if unseen or not result["in_claim"]:
                    # It says nothing of the subject, in either direction: a wheel that did
                    # not build must not open an issue for everything that is watched, and
                    # a release above the range that will not install asks nothing of
                    # anyone. Neither is a reason to close what is open.
                    held.append(
                        f"{_kind_label(result)}: {version} could not be checked{when} "
                        f"({what}), so nothing is known of it. This neither opens nor closes "
                        "the issue."
                    )
                    continue
            watched.append(
                f"{_kind_label(result)}: {version} {what}{when}. Closes {_kind_closes(result)}."
            )
            # As the state keeps them, so that the day after says the same thing.
            marks.append(
                [result["key"], "watching", outcome, [clean(node, 300) for node in nodes[:200]]]
            )
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
                    "No rule picks it any more, so no other point tests it. This "
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
        elif not result["in_claim"] and outcome not in GREEN:
            # Above the range and not checked today. What it said when this release was
            # last checked still stands: a day the index was down is no day it passed.
            before = known.get(result["key"], {}).get("last_checked") or {}
            if before.get("outcome") not in FAILED or before.get("targets") != result["targets"]:
                # Nothing is known of it, and nothing is asked: it opens no issue, and is
                # no reason to close one.
                held.append(
                    f"{version} is newer than the tested range and could not be checked today "
                    f"({OUTCOME_WORDS[outcome]}), so nothing is known of it. This neither opens "
                    "nor closes the issue."
                )
                continue
            todo.append(
                f"{version} is newer than the tested range. It did not pass when it was last "
                f"checked, on {code(before.get('date', ''))}, and could not be checked today: "
                f"{OUTCOME_WORDS[outcome]}. This closes when the adapter supports it and the "
                "range includes it."
            )
            marks.append(
                [
                    result["key"],
                    before["outcome"],
                    before.get("cause"),
                    before.get("failed_ids", []),
                ]
            )
            continue
        elif not result["in_claim"] and outcome in GREEN and info["evidence"] == "live":
            todo.append(
                f"{version} is newer than the tested range and passes. The range can be widened "
                "to include it: the table of tested versions, the test pin and the README."
            )
        else:
            continue
        # The names as the state keeps them, so that a day the release cannot be checked
        # says the same thing as the day it failed, whatever the names hold.
        marks.append(
            [result["key"], outcome, result["cause"], [clean(n, 300) for n in nodes[:200]]]
        )
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
            f"{code(item['failed_on'])}. No rule picks it any more (a newer release took its "
            "place, or the points moved), so it is run on its own from the next run on. This "
            "closes when it passes, or when the table of tested versions lists it under "
            f"`known_bad`.{lapse}"
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
    if info["claim"]:
        about = [
            f"The daily compatibility run tests the wardex adapter for {code(info['label'])} "
            "against exact framework releases, each in an environment that holds only this "
            "framework. Every run rewrites this issue, and it closes itself when nothing below "
            "needs doing.",
            "",
            f"- Tested range: {claim}",
            f"- Newest release on PyPI: {code(info['newest'])}",
        ]
    else:
        about = [
            f"What the compatibility run finds for {code(info['label'])}, which has no row of "
            "its own in the table of tested versions. Every daily run rewrites this issue, and "
            "it closes itself when nothing below needs doing.",
            "",
        ]
    body = [
        f"<!-- compat-key: {name} -->",
        f"<!-- compat-fingerprint: {fingerprint} -->",
        *about,
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
        body.append(
            f"- A pre-release ahead of the newest release exists: {code(info['prerelease_ahead'])}"
        )
    body += [f"- Note: {clean(note)}" for note in info.get("notes", [])]
    body += ["", "### What needs doing", ""]
    body += [f"- {line}" for line in todo] or ["Nothing."]
    if watched:
        body += ["", "### Watching (never turns the run red)", ""]
        body += [f"- {line}" for line in watched]
    if held:
        body += ["", "### Not checked today", ""]
        body += [f"- {line}" for line in held]
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
    traces = [
        item
        for item in (state or {}).get("retired") or []
        if isinstance(item, dict) and item.get("subject") == name
    ]
    if not mine and traces:
        # Nothing was checked, so nothing passed: the question itself went away.
        closing = f"Nothing here is asked any more: no run plans a check of {code(name)} again."
    left = []
    for item in traces:
        if not item.get("failing", True):
            continue
        # Not fixed: no longer asked. Said, so a closed issue is not read as a pass.
        why = (
            "nothing declares it any more"
            if item.get("undeclared")
            else f"its kind ({code(item.get('kind', ''))}) was retired"
        )
        left.append(
            f"The check {code(item.get('key', ''))} was not passing when it was last run "
            f"({code(item.get('date', ''))}) and is no longer run: {why}. That is not a pass."
        )
    closing += "".join(f" {sentence}" for sentence in left)
    if left and (todo or watched or held):
        # The issue stays open for something else, and the trace is of this run alone:
        # said here and in a comment of its own, or it would never be said.
        body += ["", "### No longer run", "", *[f"- {sentence}" for sentence in left]]
    if closing.startswith("Nothing here is asked"):
        closing += " Closing."
    else:
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
        # Nothing to do, and something that could not be looked at: the issue stays as it
        # is, open or not.
        "hold": bool(held) and not (todo or watched),
        "changes": [*todo, *watched],
        "closing": closing,
        "left": left,
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
                f"{code(key)}: {code(_version(result))} {_outcome_words(result)}, "
                f"on {streak} days running."
            )
        else:
            continue
        marks.append([key, outcome])
    mine = list(marks)
    cadence = plan.get("cadence", kinds.ISSUE_WRITER)
    # What a run of another cadence found wrong with itself, as it left it in the state:
    # that run writes no issue, so this is where it is said.
    for other, kept in sorted((state.get("health") or {}).items()):
        if other == cadence or not isinstance(kept, dict):
            continue
        for mark in kept.get("marks") or []:
            if not (isinstance(mark, list) and len(mark) == 2):
                continue
            todo.append(
                f"From the {clean(other, 20)} run of {code(kept.get('date', ''))}: "
                f"{code(mark[0])}: {code(mark[1])}"
            )
            marks.append([f"{other}:{mark[0]}", str(mark[1])])
    # And whether the other cadences run at all: a schedule that stopped says nothing.
    for name, kind in kinds.KINDS.items():
        if cadence in kind.cadences:
            continue
        last = ((state.get("kinds") or {}).get(name) or {}).get("last_complete")
        try:
            age = (date.fromisoformat(run["date"]) - date.fromisoformat(last)).days
        except (TypeError, ValueError):
            continue  # never completed yet: there is no day to count from
        if age > kind.fresh_days:
            todo.append(
                f"No whole run has run the {kind.label} points since {code(last)} "
                f"({age} days). Their schedule may have stopped."
            )
            marks.append([name, "stale"])
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
        # This run's own problems, for the state of a run that writes no issue.
        "marks": [[str(a), str(b)] for a, b in mine],
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


def wanted_issues(
    plan: dict,
    results: dict[str, dict],
    state: dict,
    run: dict,
    *,
    late: bool = False,
    quiet: frozenset[str] | set[str] = frozenset(),
) -> list[dict]:
    """Every issue this run has a say about.

    A run of one adapter knows nothing about the others, so it has no say
    about the two issues that speak for the whole run and must not close them.

    A run that is behind (`late`) has a say about a subject only when it is the newest
    word on all of it: it ran it, and no later run has spoken for any point of it that
    this run ran (`quiet` holds the subjects for which one has).
    """
    if plan.get("cadence", kinds.ISSUE_WRITER) != kinds.ISSUE_WRITER:
        # One run writes issues. What this one found is in the state, and is on the issue,
        # dated, the next time that run reports.
        return []
    subjects = set(plan["adapters"])
    if not plan.get("partial"):
        # Also what is not an adapter of the plan: a subject with a result today, or with
        # a point the state still remembers (a kind another cadence runs, a set of frameworks).
        subjects |= {r.get("subject", r["adapter"]) for r in results.values()}
        for entry in (state.get("points") or {}).values():
            identity = entry.get("point") or {}
            if identity.get("kind", "framework") in kinds.KINDS and identity.get("subject"):
                subjects.add(identity["subject"])
        # And a subject this run has just forgotten a point of, because its kind was
        # retired or nothing declares it any more. No later run will know of it, so this
        # one has the say about what it left open.
        for item in state.get("retired") or []:
            if isinstance(item, dict) and item.get("subject"):
                subjects.add(str(item["subject"]))
    if late:
        # Rewritten from this run's results, an issue a later run wrote would lose what
        # that run found and this one did not run, and close.
        subjects = {r.get("subject", r["adapter"]) for r in results.values()} - set(quiet)
    wanted = [adapter_issue(name, plan, results, run, state) for name in sorted(subjects)]
    if not plan.get("partial") and not late:
        # How the run itself stands is the newest run's to say: a run that is behind would
        # put an old plan's problems back, and close what a later run that died opened.
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
    if wanted.get("hold"):
        # An open issue says today's lines and stays open, with no comment: nothing
        # changed that anyone can act on. One that is closed, or not there, is left alone.
        if mine and mine["state"] == "open":
            body = wanted["body"]
            if was:
                # What it last had to say is still what it has to say.
                body = body.replace(
                    f"<!-- compat-fingerprint: {wanted['fingerprint']} -->",
                    f"<!-- compat-fingerprint: {was} -->",
                )
            path = f"issues/{mine['number']}"
            gh.do(f"rewrite #{mine['number']}", "PATCH", path, {"body": body})
            if wanted.get("left"):
                # A check that left while it was not passing: of this run alone, so now.
                text = "No longer run:\n\n" + "\n".join(f"- {c}" for c in wanted["left"])
                gh.do(f"comment on #{mine['number']}", "POST", f"{path}/comments", {"body": text})
        return
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
        if wanted.get("left"):
            # A check that left while it was not passing. The body says so today only.
            text = "No longer run:\n\n" + "\n".join(f"- {c}" for c in wanted["left"])
            gh.do(f"comment on #{mine['number']}", "POST", f"{path}/comments", {"body": text})
    else:
        text = wanted.get("closing") or "Nothing here needs doing any more. Closing."
        gh.do(f"comment on #{mine['number']}", "POST", f"{path}/comments", {"body": text})
        gh.do(f"close #{mine['number']}", "PATCH", path, {"state": "closed"})


# --- the run -------------------------------------------------------------------


def summary(plan: dict, results: dict[str, dict], reasons: list[str], actions: list[str]) -> str:
    lines = ["## Compatibility run", ""]
    lines.append("**Red:** " + "; ".join(clean(r) for r in reasons) if reasons else "**Green.**")
    subjects = set(plan["adapters"]) | {r.get("subject", r["adapter"]) for r in results.values()}
    for name in sorted(subjects):
        info = subject_info(name, plan)
        mine = [r for r in results.values() if r.get("subject", r["adapter"]) == name]
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


def gate_rehearsal(plan: dict, results: dict[str, dict], run: dict) -> str:
    """What the release gate would say of this commit, from this run's own lines.

    Printed by every run, so the gate's verdict is seen daily and not for the first time
    on the day of a release. It decides nothing here."""
    said = ["", "### If this commit were released now", ""]
    proving = gate.PROOF in kinds.due(run.get("cadence") or kinds.ISSUE_WRITER)
    if run.get("scope") != "all" or run.get("ref") != gate.MAIN_REF or not proving:
        # A run of one adapter, a run on a branch, or a run of a cadence that does not ask
        # about the tested ranges proves nothing to the gate whatever it found, and saying
        # "no run" from inside a run would be a strange thing to read.
        return "\n".join(
            [
                *said,
                "This run cannot say: the release gate reads only whole runs on the default "
                "branch that ask about the tested ranges, and this is not one.",
                "",
            ]
        )
    try:
        decision = gate.decide(
            records(plan, results, run),
            commit=run["sdk_commit"],
            today=run["date"],
            claims=gate.claimed(gate.REPO),
            runs=run_list([], run),
        )
    except Exception as broken:  # noqa: BLE001 - a rehearsal must never cost the run its verdict
        return "\n".join(
            [*said, f"The release gate could not be rehearsed: {type(broken).__name__}"]
        )
    if decision["pass"]:
        said.append(
            "The release gate would pass: every point that speaks for a tested range passed."
        )
    else:
        said.append("The release gate would stop:")
        # The gate's own sentences, made of fixed words and code spans it cleaned itself.
        said += ["", *[f"- {reason}" for reason in decision["reasons"]]]
    return "\n".join(said) + "\n"


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
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("runs") is None and list_was_kept(state_dir, [run["date"], today]):
        state = {**state, "runs": LIST_GONE}

    results = load_results(plan, results_dir)
    apply_skip_rule(results, state)
    reasons = judge(plan, results)
    # On every record line of the run, so a reader of the lines alone (the release gate)
    # knows the run's colour, also when the reason is one no point holds.
    run["red"] = bool(reasons)
    # And how many lines the run writes, so a reader can tell that one is missing.
    run["points"] = len(results)
    tip = os.environ.get("COMPAT_TIP")
    late = behind(state, run, tip, results)
    new_state = next_state(state, results, run, results_dir, plan, late=bool(late))
    locks, forgotten = new_state.pop("_locks"), new_state.pop("_forget")
    # What the run found that a later run has already spoken for.
    older = set(new_state.pop("_older"))
    # What each cadence other than the one that writes issues last found wrong with itself.
    health = {k: v for k, v in (state.get("health") or {}).items() if isinstance(v, dict)}
    cadence = plan.get("cadence", kinds.ISSUE_WRITER)
    if not plan.get("partial") and not late:
        health.pop(cadence, None)
        if cadence != kinds.ISSUE_WRITER:
            own = health_issue(plan, results, new_state, run)["marks"]
            health[cadence] = {"date": run["date"], "marks": own}
    new_state["health"] = health

    # The pages are of the day they are written, which for a run that is behind is not the
    # day of its plan.
    as_of = max(run["date"], today) if late else run["date"]
    gh = GitHub(os.environ.get("GITHUB_REPOSITORY", ""), write=args.write)
    quiet = {results[key].get("subject", results[key]["adapter"]) for key in older}
    wanted = wanted_issues(plan, results, new_state, run, late=bool(late), quiet=quiet)
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
        render.write_site(state_dir, new_state, as_of)
    if args.preview:
        # What the pages would say after this run, for a run that writes nothing.
        render.write_site(Path(args.preview), new_state, as_of)

    text = summary(
        plan, results, reasons, gh.log if args.write else [f"(dry run) {a}" for a in gh.log]
    )
    if late:
        text += (
            f"\n### This run is behind\n\nIt adds what only it knows and takes nothing away: "
            f"{late}. It forgets nothing, writes nothing over what a later run found "
            f"({len(older)} of its {len(results)} results are on its lines only), and has no "
            "say about the run's own health. Start a new run to say how things stand now "
            "(`gh workflow run compat.yml`).\n"
        )
    text += gate_rehearsal(plan, results, run)
    print(text)
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(text)
    return RED_EXIT if reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
