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
from pathlib import Path

SCHEMA = 1
LABEL = "compat"
#: The exit code of a run judged red. Its own number, so the workflow can tell a verdict
#: from this script crashing, which leaves no verdict and must be reported as such.
RED_EXIT = 10
#: A release inside the tested range in one of these states turns the run red.
RED = frozenset({"fail", "attach_failed", "no_result", "harness_error", "skips_changed"})
#: Nothing wardex or its tests did: reported only once it has lasted two days running.
NOT_OURS = frozenset({"install_network", "not_installable", "framework_import_error"})
GREEN = frozenset({"pass", "flaky"})
_UNSAFE = re.compile(r"[^A-Za-z0-9 _./:\[\]()=,<>+*#-]")

ROLE_WORDS = {
    "floor": "oldest tested release",
    "minor_newest": "newest patch of a tested line",
    "claim_top": "newest tested release",
    "claim_top_oldpy": "newest tested release, oldest Python",
    "beyond": "newer than the tested range",
}
OUTCOME_WORDS = {
    "pass": "pass",
    "flaky": "pass (a test failed once and passed when its file was run again)",
    "fail": "tests fail",
    "attach_failed": "the adapter did not attach as the SDK says it should",
    "skips_changed": "the set of skipped tests changed",
    "no_result": "no result",
    "harness_error": "the run itself failed",
    "install_network": "could not be installed (index or network)",
    "not_installable": "cannot be installed on this Python",
    "test_deps_unresolvable": "the test dependencies cannot be installed beside it",
    "framework_import_error": "the framework does not import, with or without wardex",
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
        data.update({k: point[k] for k in ("key", "adapter", "role", "targets", "in_claim")})
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


def judge(plan: dict, results: dict[str, dict]) -> list[str]:
    """The reasons this run is red. Empty means green."""
    reasons = [f"plan: {problem}" for problem in plan.get("problems", [])]
    for name, info in plan["adapters"].items():
        reasons += [f"plan: {name}: {problem}" for problem in info["problems"]]
        if sum(1 for p in plan["points"] if p["adapter"] == name) < 2:
            reasons.append(f"plan: {name} has fewer than two points")
    if not plan["points"]:
        reasons.append("plan: no points")
    for key, result in results.items():
        if result["outcome"] in ("no_result", "harness_error"):
            reasons.append(f"{key}: {OUTCOME_WORDS[result['outcome']]}")
        elif result["in_claim"] and result["outcome"] in RED:
            reasons.append(f"{key}: {OUTCOME_WORDS[result['outcome']]}")
    return reasons


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


def next_state(state: dict, results: dict[str, dict], run: dict, results_dir: Path) -> dict:
    """The small file the next run reads: each point's last record, last check and last pass."""
    points = dict(state.get("points", {}))
    locks: dict[str, Path] = {}
    for key, result in results.items():
        entry = dict(points.get(key, {}))
        tests = result.get("tests") or {}
        entry["last"] = {
            "date": run["date"],
            "outcome": result["outcome"],
            "cause": result["cause"],
            "targets": result["targets"],
            "url": run["url"],
        }
        if result["outcome"] in GREEN | RED and result["outcome"] != "no_result":
            # The last day this point was actually passed or failed. A later day on which it
            # could not be installed says nothing new, and must not erase what this said.
            entry["last_checked"] = {
                "date": run["date"],
                "outcome": result["outcome"],
                "cause": result["cause"],
                "targets": result["targets"],
                "failed_ids": tests.get("failed_ids", []),
            }
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
    out = {"schema": SCHEMA, "points": points, "last_scheduled": state.get("last_scheduled")}
    if run["event"] == "schedule":
        out["last_scheduled"] = run["started_at"]
    out["_locks"] = locks
    return out


# --- issues -------------------------------------------------------------------


def _version(result: dict) -> str:
    return ", ".join(f"{dist} {version}" for dist, version in result["targets"].items())


def _table(rows: list[dict]) -> list[str]:
    lines = ["| Point | Version | Python | Result | Cause |", "|---|---|---|---|---|"]
    for result in rows:
        lines.append(
            f"| {ROLE_WORDS.get(result['role'], clean(result['role']))} | {code(_version(result))} "
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
    mine = [r for r in results.values() if r["adapter"] == name]
    todo, marks, failing, unchecked = [], [], [], []
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
        if result["in_claim"] and outcome in RED:
            cause = CAUSE_WORDS.get(result["cause"], "not determined")
            todo.append(
                f"{version} is inside the tested range and does not pass: {what}. "
                f"Cause: {cause}. This closes when it passes again, "
                "or when the tested range no longer claims it."
            )
        elif result["in_claim"] and outcome not in GREEN:
            # Not checked today. What it said when it last was checked still stands.
            before = known.get(result["key"], {}).get("last_checked") or {}
            unchecked.append(f"{version} ({OUTCOME_WORDS[outcome]})")
            if before.get("outcome") not in RED or before.get("targets") != result["targets"]:
                continue
            todo.append(
                f"{version} is inside the tested range. It did not pass when last checked, on "
                f"{code(before['date'])}, and could not be checked today: "
                f"{OUTCOME_WORDS[outcome]}."
            )
            outcome, nodes = before["outcome"], before.get("failed_ids", [])
            marks.append([result["key"], outcome, before.get("cause"), nodes])
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

    claim = "; ".join(
        f"{code(dist)} " + ", ".join(f"{line}.x" for line in lines)
        for dist, lines in info["claim"].items()
    )
    body = [
        f"<!-- compat-key: {name} -->",
        f"<!-- compat-fingerprint: {_fingerprint(sorted(marks))} -->",
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
        "fingerprint": _fingerprint(sorted(marks)),
        "open": bool(todo),
        "changes": todo,
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
    args = parser.parse_args(argv)

    state_dir, results_dir = Path(args.state_dir), Path(args.results)
    try:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        print("no plan: the run cannot be judged", file=sys.stderr)
        return RED_EXIT
    state_file = state_dir / "state.json"
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
    run = run_info(plan)

    results = load_results(plan, results_dir)
    apply_skip_rule(results, state)
    reasons = judge(plan, results)
    new_state = next_state(state, results, run, results_dir)
    locks = new_state.pop("_locks")

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
        for key, lock in locks.items():
            (state_dir / "last-green" / f"{key}.txt").write_bytes(lock.read_bytes())
            files = new_state["points"][key]["last_green"]["files"]
            (state_dir / "last-green" / f"{key}.files.json").write_text(
                json.dumps(files), encoding="utf-8"
            )
        state_file.write_text(json.dumps(new_state, indent=1, sort_keys=True), encoding="utf-8")

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
