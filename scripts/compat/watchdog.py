"""Say so when the daily compatibility run has stopped running, or when one run died.

Two questions, two issues, so that answering one never closes the other.

Is the schedule still firing? Asked of the Actions run list, not of the
records: a scheduled workflow that stops (its cron line's owner lost access,
the file no longer parses, GitHub switched schedules off after sixty quiet
days) starts no run at all, and a run that starts and dies is the other
question. Three or more calendar days since the last scheduled run means two
whole days passed without one, which is more than a late start explains
(scheduled runs here have started up to eleven hours late). Before the first
scheduled run ever, the clock starts when GitHub registered the workflow.

Did a run end without a verdict? The run itself says so, with `--run-died`,
and the next run that reaches a verdict closes it (`report.py`).

Always exits 0, and when it cannot find out, it changes nothing: an alarm is
never closed on the strength of an answer it did not get.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compat.report import GitHub, _fingerprint, code, died_issue, sync_issue  # noqa: E402

WORKFLOW = "compat.yml"
STALE_DAYS = 3


def _get(repo: str, path: str):
    done = subprocess.run(
        ["gh", "api", f"repos/{repo}/{path}"], capture_output=True, text=True, check=True
    )
    return json.loads(done.stdout)


def last_scheduled(repo: str) -> tuple[str, bool]:
    """The instant the schedule last fired, and whether it ever has.

    When it never has, the instant is when GitHub first registered the
    workflow: the moment from which a scheduled run could be expected. Not the
    date of the commit that added the file, which can be days older than the
    push that brought it to the default branch. Raises when GitHub does not
    answer, so the caller changes nothing.
    """
    runs = _get(repo, f"actions/workflows/{WORKFLOW}/runs?event=schedule&per_page=1")
    if runs.get("workflow_runs"):
        return runs["workflow_runs"][0]["created_at"], True
    return _get(repo, f"actions/workflows/{WORKFLOW}")["created_at"], False


def wanted(last: str, now: datetime, *, ever: bool = True) -> dict:
    """What the "is it still running" issue should say."""
    then = datetime.strptime(last[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    quiet = (now.date() - then.date()).days
    todo = []
    if quiet >= STALE_DAYS:
        since = (
            f"The schedule last started a run on {code(last[:10])}, {quiet} days ago."
            if ever
            else f"The workflow has been on the default branch since {code(last[:10])}, "
            f"{quiet} days, and the schedule has never started a run."
        )
        todo.append(
            f"{since} Check the Actions tab: a scheduled workflow is switched off after sixty "
            "days without repository activity, and stops when its workflow file no longer parses."
        )
    fingerprint = _fingerprint([last[:10]] if todo else [])
    body = [
        "<!-- compat-key: watchdog -->",
        f"<!-- compat-fingerprint: {fingerprint} -->",
        "Whether the daily compatibility run is still being started by its schedule. Written "
        "by a separate workflow, because a run that never starts cannot report that it did not.",
        "",
        "### What needs doing",
        "",
    ]
    body += [f"- {line}" for line in todo] or ["Nothing."]
    return {
        "key": "watchdog",
        "title": "Compatibility: the daily run is not running",
        "body": "\n".join(body) + "\n",
        "fingerprint": fingerprint,
        "open": bool(todo),
        "changes": todo,
        "closing": "The schedule is starting runs again. Closing.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--now", help="rehearse: take this UTC instant (ISO 8601) as now")
    parser.add_argument("--last", help="rehearse: take this as when the schedule last fired")
    parser.add_argument("--run-died", default="", help="URL of a run that ended without a verdict")
    args = parser.parse_args(argv)
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    now = (
        datetime.strptime(args.now[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        if args.now
        else datetime.now(timezone.utc)
    )
    try:
        gh = GitHub(repo, write=True)
        if args.run_died:
            issue = died_issue(args.run_died)
        else:
            last, ever = (args.last, True) if args.last else last_scheduled(repo)
            issue = wanted(last, now, ever=ever)
            print(f"the schedule last fired: {last} (ever: {ever}); alarm: {issue['open']}")
        gh.ensure_label()
        sync_issue(gh, issue, gh.issues(), os.environ.get("COMPAT_ASSIGNEE", ""))
        for action in gh.log:
            print(action)
    except Exception as broken:  # noqa: BLE001 - a watchdog that fails is a red mark on a commit
        print(f"::warning::the watchdog could not finish: {type(broken).__name__}: {broken}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
