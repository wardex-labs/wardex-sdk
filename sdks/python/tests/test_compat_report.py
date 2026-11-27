"""What the daily compatibility run concludes from its points, and what it tells whom.

`scripts/compat/report.py` reads the results the point jobs left and decides
three things: the run's colour, what each issue says, and what the next run
will compare against. A red run has to mean one thing, a release inside the
tested range that does not pass, or people stop reading it; an issue has to
exist once, or it becomes noise. Both are driven here on invented results,
with a stand-in for the issue tracker.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from compat import point, report, watchdog  # noqa: E402

_RUN = {
    "id": "1",
    "attempt": "1",
    "url": "https://example.invalid/runs/1",
    "event": "schedule",
    "ref": "refs/heads/main",
    "started_at": "2026-11-26T19:40:00Z",
    "date": "2026-11-26",
    "sdk_source": "main",
    "sdk_commit": "0123456789abcdef",
    "sdk_version": "0.7.0b1",
}


def _point(role: str, version: str, *, line: str = "") -> dict:
    return {
        "key": "--".join(filter(None, ("openai_agents", role, line))),
        "adapter": "openai_agents",
        "role": role,
        "targets": {"openai-agents": version},
        "in_claim": role != "beyond",
    }


def _plan(*points: dict, evidence: str = "live") -> dict:
    return {
        "schema": 1,
        "cutoff": "2026-11-26T19:40:00Z",
        "problems": [],
        "points": list(points),
        "adapters": {
            "openai_agents": {
                "label": "openai-agents",
                "evidence": evidence,
                "replay_of": None if evidence == "live" else "a recorded program 1.0",
                "claim": {"openai-agents": ["0.22", "0.23"]},
                "newest": "0.26.0",
                "prerelease_ahead": None,
                "problems": [],
            }
        },
    }


_FLOOR = _point("floor", "0.22.0")
_TOP = _point("claim_top", "0.23.1")
_ABOVE = _point("beyond", "0.26.0", line="0.26")


def _result(point: dict, outcome: str = "pass", **more) -> dict:
    tests = {
        "files": ["test_openai_agents_adapter.py"],
        "selection_sha256": "s1",
        "passed": 10,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "failed_ids": more.pop("failed_ids", []),
        "skipped_ids": more.pop("skipped_ids", []),
    }
    return {**point, "schema": 1, "outcome": outcome, "cause": None, "tests": tests, **more}


def _results(*results: dict) -> dict:
    return {result["key"]: result for result in results}


def _write(results_dir: Path, result: dict, lock: str | None = "a==1\n") -> None:
    folder = results_dir / f"compat-point-{result['key']}"
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps(result), encoding="utf-8")
    if lock is not None:
        (folder / "lock.txt").write_text(lock, encoding="utf-8")


# --- the run's colour ----------------------------------------------------------


def test_a_run_whose_tested_releases_all_pass_is_green():
    plan = _plan(_FLOOR, _TOP, _ABOVE)
    assert report.judge(plan, _results(_result(_FLOOR), _result(_TOP), _result(_ABOVE))) == []


def test_a_tested_release_that_fails_turns_the_run_red():
    plan = _plan(_FLOOR, _TOP)
    reasons = report.judge(plan, _results(_result(_FLOOR), _result(_TOP, "fail")))
    assert reasons == ["openai_agents--claim_top: tests fail"]


def test_a_release_above_the_range_that_fails_does_not_turn_the_run_red():
    plan = _plan(_FLOOR, _TOP, _ABOVE)
    results = _results(_result(_FLOOR), _result(_TOP), _result(_ABOVE, "fail"))
    assert report.judge(plan, results) == []


@pytest.mark.parametrize("outcome", sorted(report.NOT_OURS | {"test_deps_unresolvable"}))
def test_an_index_outage_or_an_uninstallable_release_is_not_a_red_run(outcome):
    plan = _plan(_FLOOR, _TOP)
    assert report.judge(plan, _results(_result(_FLOOR), _result(_TOP, outcome))) == []


def test_a_planned_point_with_no_result_turns_the_run_red_wherever_it_sits(tmp_path):
    plan = _plan(_FLOOR, _TOP, _ABOVE)
    _write(tmp_path, _result(_FLOOR))
    _write(tmp_path, _result(_TOP))
    results = report.load_results(plan, tmp_path)
    assert results[_ABOVE["key"]]["outcome"] == "no_result"
    assert report.judge(plan, results) == ["openai_agents--beyond--0.26: no result"]


def test_a_result_that_names_another_point_or_another_version_is_not_believed(tmp_path):
    plan = _plan(_FLOOR, _TOP)
    _write(tmp_path, _result(_FLOOR))
    forged = {**_result(_FLOOR), "key": _TOP["key"], "targets": {"openai-agents": "0.22.0"}}
    _write(tmp_path, forged)
    assert report.load_results(plan, tmp_path)[_TOP["key"]]["outcome"] == "no_result"

    other = tmp_path / "other"
    _write(other, _result(_FLOOR))
    (other / f"compat-point-{_TOP['key']}").mkdir()
    wrong = other / f"compat-point-{_TOP['key']}" / "result.json"
    wrong.write_text(json.dumps(_result(_FLOOR)), encoding="utf-8")
    assert report.load_results(plan, other)[_TOP["key"]]["outcome"] == "no_result"


def test_an_outcome_the_report_does_not_know_is_no_result(tmp_path):
    plan = _plan(_FLOOR, _TOP)
    _write(tmp_path, _result(_FLOOR))
    _write(tmp_path, _result(_TOP, "splendid"))
    assert report.load_results(plan, tmp_path)[_TOP["key"]]["outcome"] == "no_result"


def test_a_plan_that_could_not_cover_an_adapter_is_red_not_quietly_smaller():
    assert report.judge(_plan(_FLOOR), _results(_result(_FLOOR))) == [
        "plan: openai_agents has fewer than two points"
    ]
    assert "plan: no points" in report.judge(_plan(), {})


def test_a_pass_that_skips_different_tests_than_the_last_pass_is_not_a_pass():
    results = _results(_result(_TOP, skipped_ids=["a.py::test_x"]))
    last = {"selection_sha256": "s1", "skipped_ids": []}
    report.apply_skip_rule(results, {"points": {_TOP["key"]: {"last_green": last}}})
    assert results[_TOP["key"]]["outcome"] == "skips_changed"


def test_new_skips_are_not_held_against_a_point_whose_test_files_changed():
    results = _results(_result(_TOP, skipped_ids=["a.py::test_x"]))
    last = {"selection_sha256": "an earlier selection", "skipped_ids": []}
    report.apply_skip_rule(results, {"points": {_TOP["key"]: {"last_green": last}}})
    assert results[_TOP["key"]]["outcome"] == "pass"


# --- what the issue says -------------------------------------------------------


def test_nothing_to_do_means_no_issue():
    plan = _plan(_FLOOR, _TOP)
    issue = report.adapter_issue("openai_agents", plan, _results(_result(_FLOOR)), _RUN)
    assert (issue["open"], issue["fingerprint"]) == (False, "clear")


def test_a_failing_release_above_the_range_opens_the_issue_and_names_its_tests():
    plan = _plan(_FLOOR, _TOP, _ABOVE)
    failed = _result(_ABOVE, "fail", cause="new_version", failed_ids=["t.py::test_tree"])
    issue = report.adapter_issue(
        "openai_agents", plan, _results(_result(_FLOOR), _result(_TOP), failed), _RUN
    )
    assert issue["open"]
    assert issue["title"] == "Compatibility: openai-agents"
    assert (
        "`openai-agents 0.26.0` is newer than the tested range and fails 1 tests" in issue["body"]
    )
    assert "- `t.py::test_tree`" in issue["body"]
    assert "<!-- compat-key: openai_agents -->" in issue["body"]


def test_a_passing_release_above_the_range_is_offered_for_widening_only_on_live_evidence():
    results = _results(_result(_FLOOR), _result(_TOP), _result(_ABOVE))
    live = report.adapter_issue("openai_agents", _plan(_FLOOR, _TOP, _ABOVE), results, _RUN)
    assert live["open"] and "The range can be widened" in live["body"]

    replay = _plan(_FLOOR, _TOP, _ABOVE, evidence="wrapper-live")
    recorded = report.adapter_issue("openai_agents", replay, results, _RUN)
    assert not recorded["open"], "a pass against recorded output is no ground to widen a claim"
    assert "against recorded output of `a recorded program 1.0`" in recorded["body"]


def test_the_issue_carries_nothing_a_point_could_inject():
    """A point job runs code published today; whatever it wrote reaches a public issue
    only as characters that cannot mention anyone, link anywhere or leave a code span."""
    hostile = "t.py::test_x @everyone [click](https://evil.example) `escape` <img src=x>"
    failed = _result(_ABOVE, "fail", cause="new_version", failed_ids=[hostile])
    failed["python"] = "3.14 @someone"
    plan = _plan(_FLOOR, _TOP, _ABOVE)
    body = report.adapter_issue("openai_agents", plan, _results(failed), _RUN)["body"]
    assert "@" not in body
    assert body.count("`") % 2 == 0
    outside_code_spans = re.sub(r"`[^`\n]*`", "", body)
    for fragment in ("evil", "everyone", "img", "click"):
        assert fragment not in outside_code_spans


def test_the_fingerprint_changes_with_what_fails_and_not_with_the_date():
    plan = _plan(_FLOOR, _TOP, _ABOVE)
    failed = _result(_ABOVE, "fail", cause="new_version", failed_ids=["t.py::test_a"])
    first = report.adapter_issue("openai_agents", plan, _results(failed), _RUN)
    later = report.adapter_issue(
        "openai_agents", plan, _results(failed), {**_RUN, "date": "2026-11-27"}
    )
    assert first["fingerprint"] == later["fingerprint"]
    worse = _result(_ABOVE, "fail", cause="new_version", failed_ids=["t.py::test_a", "t.py::b"])
    changed = report.adapter_issue("openai_agents", plan, _results(worse), _RUN)
    assert changed["fingerprint"] != first["fingerprint"]


# --- one issue, not many -------------------------------------------------------


class _Tracker(report.GitHub):
    """Records what the report would do to the issue tracker and does none of it."""

    def __init__(self) -> None:
        super().__init__("example/repo", write=False)


def _wanted(open_: bool = True, fingerprint: str = "f1") -> dict:
    return {
        "key": "openai_agents",
        "title": "Compatibility: openai-agents",
        "body": f"<!-- compat-key: openai_agents -->\n<!-- compat-fingerprint: {fingerprint} -->\n",
        "fingerprint": fingerprint,
        "open": open_,
        "changes": ["something needs doing"],
    }


def _existing(state: str = "open", fingerprint: str = "f1", title: str = "x") -> dict:
    body = f"<!-- compat-key: openai_agents -->\n<!-- compat-fingerprint: {fingerprint} -->\n"
    return {"number": 7, "state": state, "title": title, "body": body}


def test_the_first_failure_opens_one_issue_assigned_to_whoever_is_configured():
    tracker = _Tracker()
    report.sync_issue(tracker, _wanted(), [], "someone")
    assert tracker.log == ["open Compatibility: openai-agents"]


def test_an_open_issue_is_rewritten_not_duplicated_and_stays_quiet_while_nothing_changes():
    tracker = _Tracker()
    report.sync_issue(tracker, _wanted(), [_existing(title="renamed by a person")])
    assert tracker.log == ["rewrite #7"]


def test_a_change_in_what_fails_is_said_once_as_a_comment():
    tracker = _Tracker()
    report.sync_issue(tracker, _wanted(fingerprint="f2"), [_existing(fingerprint="f1")])
    assert tracker.log == ["rewrite #7", "comment on #7"]


def test_an_issue_with_nothing_left_to_do_is_closed_with_a_word():
    tracker = _Tracker()
    report.sync_issue(tracker, _wanted(open_=False, fingerprint="clear"), [_existing()])
    assert tracker.log == ["rewrite #7", "comment on #7", "close #7"]


def test_an_issue_a_person_closed_stays_closed_until_what_fails_changes():
    tracker = _Tracker()
    report.sync_issue(tracker, _wanted(), [_existing(state="closed")])
    assert tracker.log == []

    report.sync_issue(tracker, _wanted(fingerprint="f2"), [_existing(state="closed")])
    assert tracker.log == ["rewrite #7", "reopen #7", "comment on #7"]


def test_an_issue_the_run_closed_is_reopened_when_the_failure_returns():
    tracker = _Tracker()
    report.sync_issue(tracker, _wanted(), [_existing(state="closed", fingerprint="clear")])
    assert tracker.log == ["rewrite #7", "reopen #7", "comment on #7"]


def test_a_closed_issue_with_nothing_to_do_is_left_alone():
    tracker = _Tracker()
    wanted = _wanted(open_=False, fingerprint="clear")
    report.sync_issue(tracker, wanted, [_existing(state="closed", fingerprint="clear")])
    assert tracker.log == []


def test_another_keys_issue_is_never_touched():
    tracker = _Tracker()
    other = {"number": 3, "state": "open", "title": "x", "body": "<!-- compat-key: health -->"}
    report.sync_issue(tracker, _wanted(), [other])
    assert tracker.log == ["open Compatibility: openai-agents"]


# --- the health issue ----------------------------------------------------------


def test_an_outage_is_reported_only_once_it_has_lasted_two_days():
    plan = _plan(_FLOOR, _TOP)
    results = _results(_result(_FLOOR), _result(_TOP, "install_network"))
    once = {"points": {_TOP["key"]: {"not_ours_days": 1}}}
    assert not report.health_issue(plan, results, once, _RUN)["open"]
    twice = {"points": {_TOP["key"]: {"not_ours_days": 2}}}
    issue = report.health_issue(plan, results, twice, _RUN)
    assert issue["open"] and "on 2 days running" in issue["body"]


def test_a_release_our_test_dependencies_exclude_is_reported_the_same_day():
    """The framework installs for its users; only our test environment cannot follow.
    That is the day a new release line lands, and nothing was tested on it."""
    plan = _plan(_FLOOR, _TOP, _ABOVE)
    results = _results(_result(_ABOVE, "test_deps_unresolvable"))
    issue = report.health_issue(plan, results, {}, _RUN)
    assert (
        issue["open"] and "installs on its own, but not beside what the tests need" in issue["body"]
    )


def test_a_point_that_left_no_result_is_a_health_problem():
    plan = _plan(_FLOOR, _TOP)
    results = _results(_result(_TOP, "harness_error", harness_error="AssertionError: wrong"))
    assert "left no usable result" in report.health_issue(plan, results, {}, _RUN)["body"]


# --- what the next run compares against ----------------------------------------


def test_only_a_pass_becomes_the_last_pass_and_only_a_schedule_counts_as_scheduled(tmp_path):
    _write(tmp_path, _result(_FLOOR))
    _write(tmp_path, _result(_TOP, "fail"))
    results = _results(_result(_FLOOR), _result(_TOP, "fail"))
    earlier = {"points": {_TOP["key"]: {"last_green": {"date": "2026-11-20"}}}}

    state = report.next_state(earlier, results, _RUN, tmp_path)
    assert state["points"][_FLOOR["key"]]["last_green"]["date"] == "2026-11-26"
    assert state["points"][_FLOOR["key"]]["last_green"]["lock"].endswith("--floor.txt")
    assert state["points"][_TOP["key"]]["last_green"] == {"date": "2026-11-20"}
    assert set(state["_locks"]) == {_FLOOR["key"]}
    assert state["last_scheduled"] == _RUN["started_at"]

    by_hand = report.next_state(state, results, {**_RUN, "event": "workflow_dispatch"}, tmp_path)
    assert by_hand["last_scheduled"] == _RUN["started_at"]
    never = report.next_state({}, results, {**_RUN, "event": "workflow_dispatch"}, tmp_path)
    assert never["last_scheduled"] is None


def test_the_outage_streak_counts_days_not_runs_and_resets_on_anything_else(tmp_path):
    """A report job run again, or a second run by hand the same day, is the same outage
    seen twice. Counted as two it would raise the alarm a day early."""
    results = _results(_result(_TOP, "install_network"))
    state = report.next_state({}, results, _RUN, tmp_path)
    state.pop("_locks")
    assert state["points"][_TOP["key"]]["not_ours_days"] == 1
    state = report.next_state(state, results, _RUN, tmp_path)
    state.pop("_locks")
    assert state["points"][_TOP["key"]]["not_ours_days"] == 1, "the same day again"
    state = report.next_state(state, results, {**_RUN, "date": "2026-11-27"}, tmp_path)
    state.pop("_locks")
    assert state["points"][_TOP["key"]]["not_ours_days"] == 2
    state = report.next_state(state, _results(_result(_TOP, "fail")), _RUN, tmp_path)
    assert state["points"][_TOP["key"]]["not_ours_days"] == 0


def test_a_record_line_holds_only_plain_data_about_its_point():
    line = report.records(_plan(_TOP), _results(_result(_TOP)), _RUN)[0]
    assert line["schema"] == 1
    assert line["run"]["sdk_commit"] == _RUN["sdk_commit"]
    assert (line["key"], line["outcome"], line["in_claim"]) == (_TOP["key"], "pass", True)
    json.dumps(line)


# --- the watchdog ----------------------------------------------------------------


def _at(day: str) -> datetime:
    return datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def test_two_whole_days_without_a_scheduled_run_is_an_alarm_and_a_late_start_is_not():
    last = "2026-11-20T19:40:00Z"
    assert not watchdog.wanted(last, _at("2026-11-21"))["open"]
    assert not watchdog.wanted(last, _at("2026-11-22"))["open"]
    stale = watchdog.wanted(last, _at("2026-11-23"))
    assert stale["open"] and "3 days ago" in stale["body"]
    assert stale["key"] == "watchdog"


def test_a_schedule_that_has_never_fired_is_an_alarm_once_the_workflow_is_three_days_old():
    """Before the first scheduled run there is no last run to be late. The clock starts
    when the workflow reached the default branch, or a schedule that never worked
    would never be noticed."""
    added = "2026-11-26T10:00:00Z"
    assert not watchdog.wanted(added, _at("2026-11-28"), ever=False)["open"]
    never = watchdog.wanted(added, _at("2026-11-29"), ever=False)
    assert never["open"] and "has never started a run" in never["body"]


def test_a_run_that_died_has_its_own_issue_which_only_a_verdict_closes():
    """The watchdog answers "is the schedule firing". If it shared an issue with "a run
    died", the next push would close a death notice nobody had read."""
    died = report.died_issue("https://example.invalid/runs/9")
    assert died["open"] and died["key"] == "run-died"
    assert "ended before it could report" in died["body"]
    assert watchdog.wanted("2026-11-26T19:40:00Z", _at("2026-11-26"))["key"] != died["key"]

    cleared = report.died_issue()
    assert not cleared["open"]
    tracker = _Tracker()
    existing = {"number": 9, "state": "open", "title": "x", "body": died["body"]}
    report.sync_issue(tracker, cleared, [existing])
    assert tracker.log == ["rewrite #9", "comment on #9", "close #9"]


def test_a_run_of_one_adapter_has_no_say_about_the_issues_that_speak_for_the_whole_run():
    plan = _plan(_FLOOR, _TOP)
    results = _results(_result(_FLOOR), _result(_TOP))
    assert [i["key"] for i in report.wanted_issues(plan, results, {}, _RUN)] == [
        "openai_agents",
        "health",
        "run-died",
    ]
    partial = {**plan, "partial": True}
    assert [i["key"] for i in report.wanted_issues(partial, results, {}, _RUN)] == ["openai_agents"]


# --- the lock a point leaves, and the one it replays -----------------------------

_LOCK = """\
/tmp/dist/wardex_sdk-0.7.0b1-cp310-abi3-manylinux_2_17_x86_64.whl \\
    --hash=sha256:aaa
openai-agents==0.23.1 \\
    --hash=sha256:bbb \\
    --hash=sha256:ccc
Typing_Extensions==4.15.0 ; python_full_version < '3.13' \\
    --hash=sha256:ddd
httpx[http2]==0.28.1 \\
    --hash=sha256:eee
"""


def test_a_lock_is_read_as_one_entry_per_requirement_with_its_hashes():
    entries = point.lock_entries(_LOCK)
    assert len(entries) == 4
    assert entries[1].count("--hash=") == 2


def test_the_pinned_versions_of_a_lock_are_read_by_normalized_name():
    assert point.pinned(_LOCK) == {
        "openai-agents": "0.23.1",
        "typing-extensions": "4.15.0",
        "httpx": "0.28.1",
    }


# --- the rules a point applies, each one broken in turn ---------------------------


@pytest.mark.parametrize(
    ("exit_code", "passed", "failed", "errors", "ok"),
    [
        (0, 10, 0, 0, True),
        (0, 0, 0, 0, False),
        (1, 10, 1, 0, False),
        (0, 10, 0, 1, False),
        (0, 10, 1, 0, False),
        (5, 0, 0, 0, False),
        (2, 3, 0, 0, False),
        (124, 10, 0, 0, False),
    ],
    ids=[
        "clean",
        "nothing-ran",
        "a-failure",
        "a-teardown-error",
        "junit-says-failed",
        "none-collected",
        "collection-error",
        "timed-out",
    ],
)
def test_a_pass_is_a_clean_exit_that_ran_something(exit_code, passed, failed, errors, ok):
    counts = {"passed": passed, "failed": failed, "errors": errors, "skipped": 0}
    assert point.is_pass(exit_code, counts) is ok


@pytest.mark.parametrize(
    ("state", "measured", "in_table", "lines_said", "verdict"),
    [
        ("installed", True, True, 0, "ok"),
        ("installed", False, False, 1, "ok"),
        ("unsupported", None, True, 1, "not_installed"),
        ("waiting_for_import", None, True, 0, "not_installed"),
        ("installed", False, True, 1, "flag_mismatch"),
        ("installed", True, False, 0, "flag_mismatch"),
        ("installed", True, True, 1, "flag_mismatch"),
        ("installed", False, False, 0, "flag_mismatch"),
        ("installed", None, True, 0, "flag_mismatch"),
    ],
    ids=[
        "tested-and-silent",
        "untested-and-said",
        "declined",
        "never-installed",
        "sdk-flags-a-tested-release",
        "sdk-vouches-for-an-untested-one",
        "a-line-on-a-tested-release",
        "silent-on-an-untested-one",
        "version-unknown",
    ],
)
def test_the_attach_verdict_holds_the_adapter_to_the_table(
    state, measured, in_table, lines_said, verdict
):
    seen = {"state": state, "measured": measured, "in_table": in_table, "lines_said": lines_said}
    assert point.attach_verdict(seen) == verdict


def test_a_replay_installs_every_locked_package_except_the_wardex_wheel():
    kept = point.without_wardex(point.lock_entries(_LOCK))
    assert len(kept) == 3
    assert all("wardex_sdk-" not in entry for entry in kept)
    assert point.pinned("\n".join(kept)) == point.pinned(_LOCK)


def test_the_cause_that_needs_no_second_run():
    cause = point.cause_without_replay
    assert cause(in_claim=False, frozen=False, has_baseline=True) == "new_version"
    assert cause(in_claim=True, frozen=True, has_baseline=False) == "no_baseline"
    assert cause(in_claim=True, frozen=True, has_baseline=True) == "ours"
    # A floor younger than its dependency window still moves, like any other point.
    assert cause(in_claim=True, frozen=False, has_baseline=True) is None


# --- what a later reader of the report found, kept from coming back ---------------


def _failed_yesterday(point: dict, nodes: list[str]) -> dict:
    checked = {
        "date": "2026-11-25",
        "outcome": "fail",
        "cause": "upstream",
        "targets": point["targets"],
        "failed_ids": nodes,
    }
    return {"points": {point["key"]: {"last_checked": checked}}}


@pytest.mark.parametrize("today", sorted(report.NOT_OURS | {"test_deps_unresolvable"}))
def test_a_release_that_failed_and_cannot_be_checked_today_is_still_on_the_list(today):
    """Yesterday a tested release failed; today it could not even be installed. The run
    is not red, and the issue must neither close nor forget what it knew."""
    plan = _plan(_FLOOR, _TOP)
    nodes = ["t.py::test_tree"]
    yesterday = report.adapter_issue(
        "openai_agents",
        plan,
        _results(_result(_FLOOR), _result(_TOP, "fail", cause="upstream", failed_ids=nodes)),
        _RUN,
    )
    results = _results(_result(_FLOOR), _result(_TOP, today))
    issue = report.adapter_issue(
        "openai_agents", plan, results, _RUN, _failed_yesterday(_TOP, nodes)
    )
    assert issue["open"]
    assert "did not pass when last checked, on `2026-11-25`" in issue["body"]
    assert issue["fingerprint"] == yesterday["fingerprint"], "no news, so no comment"

    tracker = _Tracker()
    report.sync_issue(tracker, issue, [_existing(fingerprint=yesterday["fingerprint"])])
    assert tracker.log == ["rewrite #7"]


def test_what_a_point_last_said_is_carried_only_for_the_same_release_inside_the_range():
    plan = _plan(_FLOOR, _TOP, _ABOVE)
    nodes = ["t.py::test_tree"]
    # A release above the range that cannot be installed today needs nothing doing.
    above = _results(_result(_ABOVE, "install_network"))
    assert not report.adapter_issue(
        "openai_agents", plan, above, _RUN, _failed_yesterday(_ABOVE, nodes)
    )["open"]
    # A different release now sits at this point: yesterday's failure was another version's.
    moved = _failed_yesterday(_TOP, nodes)
    moved["points"][_TOP["key"]]["last_checked"]["targets"] = {"openai-agents": "0.23.0"}
    top = _results(_result(_TOP, "install_network"))
    assert not report.adapter_issue("openai_agents", plan, top, _RUN, moved)["open"]
    # A release that passed when last checked is not turned into a failure by an outage.
    assert not report.adapter_issue("openai_agents", plan, top, _RUN, {})["open"]


def test_the_closing_word_names_what_could_not_be_checked_instead_of_counting_it_as_passing():
    plan = _plan(_FLOOR, _TOP)
    results = _results(_result(_FLOOR), _result(_TOP, "not_installable"))
    issue = report.adapter_issue("openai_agents", plan, results, _RUN)
    assert not issue["open"]
    assert "Not checked today: `openai-agents 0.23.1` (cannot be installed" in issue["closing"]
    clean = report.adapter_issue(
        "openai_agents", plan, _results(_result(_FLOOR), _result(_TOP)), _RUN
    )
    assert "Not checked today" not in clean["closing"]


def test_an_outage_does_not_erase_what_a_point_said_when_it_was_last_checked(tmp_path):
    failed = _results(_result(_TOP, "fail", cause="upstream", failed_ids=["t.py::test_a"]))
    state = report.next_state({}, failed, _RUN, tmp_path)
    state.pop("_locks")
    assert state["points"][_TOP["key"]]["last_checked"]["failed_ids"] == ["t.py::test_a"]

    outage = _results(_result(_TOP, "install_network"))
    later = report.next_state(state, outage, {**_RUN, "date": "2026-11-27"}, tmp_path)
    assert later["points"][_TOP["key"]]["last_checked"]["date"] == "2026-11-26"
    assert later["points"][_TOP["key"]]["last"]["outcome"] == "install_network"


def test_the_last_pass_remembers_which_files_it_ran(tmp_path):
    _write(tmp_path, _result(_TOP))
    state = report.next_state({}, _results(_result(_TOP)), _RUN, tmp_path)
    assert state["points"][_TOP["key"]]["last_green"]["files"] == ["test_openai_agents_adapter.py"]


def test_a_test_no_longer_skipped_is_more_tested_not_less():
    results = _results(_result(_TOP, skipped_ids=[]))
    last = {"selection_sha256": "s1", "skipped_ids": ["a.py::test_x"]}
    report.apply_skip_rule(results, {"points": {_TOP["key"]: {"last_green": last}}})
    assert results[_TOP["key"]]["outcome"] == "pass"


@pytest.mark.parametrize(
    "poison",
    [
        {"tests": "a string where a table should be"},
        {"tests": {"failed_ids": "not-a-list"}},
        {"tests": {"failed_ids": [1, 2]}},
        {"tests": {"passed": "many"}},
        {"tests": {"plugins": 5}},
        {"cause": "<img src=x>"},
        {"attach": ["not", "a", "table"]},
        {"resolved": "x"},
        {"install": {"dropped_optional": 5}},
        {"install": {"dropped_optional": [["ok"], "not-a-group"]}},
        {"counterfactual": {"ran": True, "changed": ["a", "b"]}},
        {"counterfactual": {"ran": True, "changed": {"a": "not-a-pair"}}},
    ],
)
def test_a_result_of_the_wrong_shape_is_no_result_and_cannot_crash_the_report(tmp_path, poison):
    plan = _plan(_FLOOR, _TOP)
    _write(tmp_path, _result(_FLOOR))
    _write(tmp_path, {**_result(_TOP), **poison})
    results = report.load_results(plan, tmp_path)
    assert results[_TOP["key"]]["outcome"] == "no_result"
    report.records(plan, results, _RUN)
    report.adapter_issue("openai_agents", plan, results, _RUN)
    report.health_issue(plan, results, {}, _RUN)


def test_a_record_line_carries_nothing_raw_from_a_point():
    """The records branch is public and read by later tooling: every string a point wrote
    arrives cleaned and bounded, and a field of the wrong type arrives as nothing."""
    hostile = "@everyone `tick` " + "x" * 5000
    result = _result(_TOP, "fail", cause="upstream", failed_ids=[hostile])
    result.update(
        python=hostile,
        uploaded={hostile: hostile},
        resolved={hostile: hostile},
        install={"dropped_optional": [[hostile]]},
        counterfactual={"ran": True, "passed": True, "changed": {hostile: [hostile, hostile]}},
        duration_s="soon",
    )
    line = json.dumps(report.records(_plan(_TOP), _results(result), _RUN))
    assert "@" not in line and "`" not in line
    assert "x" * 301 not in line
    assert '"duration_s": null' in line


def test_the_health_issue_puts_a_points_own_words_inside_a_code_span():
    """`harness_error` is text a point wrote. Outside a code span a link, an image or
    an issue reference in it would render in a public issue."""
    said = "boom [click](https://evil.example) <img src=x> #1 owner/repo#2"
    results = _results(_result(_TOP, "harness_error", harness_error=said))
    body = report.health_issue(_plan(_FLOOR, _TOP), results, {}, _RUN)["body"]
    outside_code_spans = re.sub(r"`[^`\n]*`", "", body)
    for fragment in ("evil", "click", "img", "#1", "repo#2"):
        assert fragment not in outside_code_spans


def test_a_duplicate_issue_is_closed_and_the_newest_one_is_kept():
    older = {**_existing(), "number": 5}
    long_closed = {**_existing(state="closed"), "number": 3}
    tracker = _Tracker()
    report.sync_issue(tracker, _wanted(), [long_closed, older, _existing()])
    assert tracker.log == ["close duplicate #5", "rewrite #7"]


def test_closing_one_duplicate_by_hand_does_not_leave_a_live_failure_with_no_open_issue():
    """A person closes the newer of two. The older open one is the issue from then on;
    taking the closed one and closing the other as its duplicate would leave none open."""
    older_open = {**_existing(), "number": 5}
    newer_closed = _existing(state="closed")
    tracker = _Tracker()
    report.sync_issue(tracker, _wanted(), [older_open, newer_closed])
    assert tracker.log == ["rewrite #5"]


def test_a_failure_that_returns_reopens_an_issue_a_person_closed_while_it_failed():
    """Closed by hand while failing, then it passed, then the same failure came back.
    The pass is written into the closed issue quietly, so the return is seen as a change."""
    tracker = _Tracker()
    passing = _wanted(open_=False, fingerprint="clear")
    report.sync_issue(tracker, passing, [_existing(state="closed", fingerprint="f1")])
    assert tracker.log == ["rewrite #7"]

    tracker = _Tracker()
    report.sync_issue(tracker, _wanted(), [_existing(state="closed", fingerprint="clear")])
    assert tracker.log == ["rewrite #7", "reopen #7", "comment on #7"]


def test_a_failure_with_no_failing_test_is_not_called_zero_failures():
    timed_out = _result(_ABOVE, "fail", cause="new_version")
    body = report.adapter_issue(
        "openai_agents", _plan(_FLOOR, _TOP, _ABOVE), _results(timed_out), _RUN
    )["body"]
    assert "fails 0 tests" not in body
    assert "does not finish its tests" in body


def test_a_replay_cannot_explain_a_failure_in_a_file_it_does_not_run():
    """The replay runs the files of the last pass. A test file added since is not among
    them, the replay would pass, and a wardex change would be blamed on a dependency."""
    assert point.replay_can_tell(["a.py::test_x"], ["a.py", "b.py"])
    assert not point.replay_can_tell(["new.py::test_x"], ["a.py", "b.py"])
    assert not point.replay_can_tell(["a.py::test_x", "new.py::test_y"], ["a.py"])
    assert not point.replay_can_tell([], ["a.py"]), "nothing named, nothing to explain"


def test_a_record_keeps_what_the_probe_counted_and_what_a_point_said_of_itself():
    result = _result(_TOP, "fail", cause="ours")
    result.update(
        attach={
            "state": "installed",
            "measured": True,
            "in_table": True,
            "lines_said": 0,
            "verdict": "ok",
            "detail": "left out",
        },
        rerun={"passed": False},
        install={"outcome": "ok", "dropped_optional": []},
        cutoff="2026-09-02T12:00:00Z",
    )
    line = report.records(_plan(_TOP), _results(result), _RUN)[0]
    assert line["attach"] == {
        "state": "installed",
        "measured": True,
        "in_table": True,
        "lines_said": 0,
        "verdict": "ok",
    }
    assert (line["rerun_passed"], line["install_outcome"]) == (False, "ok")
    assert line["point_cutoff"] == "2026-09-02T12:00:00Z"


def test_a_stale_schedule_alarm_names_the_day_it_went_quiet():
    """Two different quiet spells are two different alarms; the second must not be
    taken for the first and stay closed."""
    first = watchdog.wanted("2026-11-01T00:00:00Z", _at("2026-11-10"))
    second = watchdog.wanted("2026-11-20T00:00:00Z", _at("2026-11-26"))
    assert first["open"] and second["open"]
    assert first["fingerprint"] != second["fingerprint"]
    assert watchdog.wanted("2026-11-20T00:00:00Z", _at("2026-11-21"))["fingerprint"] == "clear"


# --- a failed release that a newer one replaced --------------------------------

_ENTRY = {"distribution": "openai-agents", "minors": ["0.22", "0.23"], "known_bad": [], "since": ""}


def _plan_with_table(*points: dict, entry: dict = _ENTRY) -> dict:
    plan = _plan(*points)
    plan["adapters"]["openai_agents"]["entries"] = [entry]
    return plan


def _displaced_point(version: str) -> dict:
    return {**_point("displaced", version, line=f"{version}-py3.14"), "python": "3.14"}


def _top_failed_yesterday(version: str = "0.23.1") -> dict:
    state = _failed_yesterday(_point("claim_top", version), ["t.py::test_tree"])
    state["points"][_TOP["key"]]["last_checked"]["python"] = "3.14"
    return state


def test_a_failed_release_that_a_newer_one_replaced_is_remembered_and_kept_on_the_issue(tmp_path):
    """Yesterday the newest tested release failed. Today a newer patch is the newest and
    passes. The old one is still a release the README calls tested, and no point runs it
    any more: without this it would leave the issue as if it had been fixed."""
    newer = _point("claim_top", "0.23.2")
    plan = _plan_with_table(_FLOOR, newer)
    results = _results(_result(_FLOOR), _result(newer))
    assert report.judge(plan, results) == []

    state = report.next_state(_top_failed_yesterday(), results, _RUN, tmp_path, plan)
    assert state["displaced"] == {
        "openai_agents": [
            {
                "key": "openai_agents--displaced--0.23.1-py3.14",
                "targets": {"openai-agents": "0.23.1"},
                "python": "3.14",
                "failed_on": "2026-11-25",
            }
        ]
    }
    issue = report.adapter_issue("openai_agents", plan, results, _RUN, state)
    assert issue["open"]
    (line,) = issue["changes"]
    assert "`openai-agents 0.23.1`" in line and "did not pass when it was last run" in line
    assert "`known_bad`" in line
    assert issue["fingerprint"] != "clear", "an open issue that reads as having nothing to say"

    again = report.next_state(state, results, _RUN, tmp_path, plan)
    assert again["displaced"] == {}, "the plan made no point of it: it is no longer claimed"


def test_a_release_the_range_no_longer_claims_is_not_remembered(tmp_path):
    """Narrowing the range, or listing the release as bad, is how such a failure is
    settled. The run that first sees the narrower table must let the issue close."""
    newer = _point("claim_top", "0.23.2")
    results = _results(_result(_FLOOR), _result(newer))
    for entry in (
        {**_ENTRY, "known_bad": ["0.23.1"]},
        {**_ENTRY, "since": "0.23.2"},
        {**_ENTRY, "minors": ["0.22"]},
    ):
        plan = _plan_with_table(_FLOOR, newer, entry=entry)
        state = report.next_state(_top_failed_yesterday(), results, _RUN, tmp_path, plan)
        assert state["displaced"] == {}, entry
        assert not report.adapter_issue("openai_agents", plan, results, _RUN, state)["open"]


@pytest.mark.parametrize("before", ["no_result", "harness_error", "install_network", "pass"])
def test_only_a_release_that_was_run_and_failed_is_remembered(tmp_path, before):
    """A release that could not be looked at is not known to be bad."""
    state = _top_failed_yesterday()
    state["points"][_TOP["key"]]["last_checked"]["outcome"] = before
    newer = _point("claim_top", "0.23.2")
    plan = _plan_with_table(_FLOOR, newer)
    results = _results(_result(_FLOOR), _result(newer))
    assert report.next_state(state, results, _RUN, tmp_path, plan)["displaced"] == {}


def test_a_displaced_release_that_still_fails_is_red_and_says_how_it_closes(tmp_path):
    newer, old = _point("claim_top", "0.23.2"), _displaced_point("0.23.1")
    plan = _plan_with_table(_FLOOR, newer, old)
    failing = _result(old, "fail", cause="no_baseline", failed_ids=["t.py::test_tree"])
    results = _results(_result(_FLOOR), _result(newer), failing)
    assert report.judge(plan, results) == [f"{old['key']}: tests fail"]

    remembered = {"displaced": {"openai_agents": [{"key": old["key"], **_remembered_item()}]}}
    state = report.next_state(remembered, results, _RUN, tmp_path, plan)
    assert [item["key"] for item in state["displaced"]["openai_agents"]] == [old["key"]]
    issue = report.adapter_issue("openai_agents", plan, results, _RUN, state)
    (line,) = issue["changes"]
    assert "A newer release has taken its place, so no other point tests it." in line
    assert "lists it under `known_bad`" in line
    assert "failed as the newest tested release, since replaced" in issue["body"]


def _remembered_item() -> dict:
    return {"targets": {"openai-agents": "0.23.1"}, "python": "3.14", "failed_on": "2026-11-25"}


@pytest.mark.parametrize(
    ("today", "kept"),
    [
        ("pass", False),
        ("flaky", False),
        ("fail", True),
        ("install_network", True),
        ("no_result", True),
    ],
)
def test_a_displaced_release_leaves_the_list_only_when_it_passes(tmp_path, today, kept):
    newer, old = _point("claim_top", "0.23.2"), _displaced_point("0.23.1")
    plan = _plan_with_table(_FLOOR, newer, old)
    results = _results(_result(_FLOOR), _result(newer), _result(old, today))
    remembered = {"displaced": {"openai_agents": [{"key": old["key"], **_remembered_item()}]}}
    state = report.next_state(remembered, results, _RUN, tmp_path, plan)
    assert bool(state["displaced"]) is kept


def test_a_run_that_did_not_plan_an_adapter_in_full_leaves_its_list_alone(tmp_path):
    """A run of one adapter, or a day the index could not list an adapter's releases, has
    nothing to say about another adapter's remembered failures."""
    remembered = {
        "displaced": {
            "langgraph": [{"key": "langgraph--displaced--1.2.3-1.1.0-py3.14"}],
            "openai_agents": [{"key": "openai_agents--displaced--0.23.1-py3.14"}],
        }
    }
    plan = _plan_with_table(_FLOOR, _TOP)
    plan["adapters"]["openai_agents"]["problems"] = ["no installable release"]
    results = _results(_result(_FLOOR), _result(_TOP))
    state = report.next_state(remembered, results, _RUN, tmp_path, plan)
    assert state["displaced"] == remembered["displaced"]


def test_the_issue_says_where_a_tested_range_starts():
    plan = _plan(_FLOOR, _TOP)
    plan["adapters"]["openai_agents"]["said"] = {"openai-agents": "0.22.x from 0.22.3, 0.23.x"}
    issue = report.adapter_issue("openai_agents", plan, _results(_result(_FLOOR)), _RUN)
    assert "- Tested range: `openai-agents` 0.22.x from 0.22.3, 0.23.x" in issue["body"]


def test_a_replaced_release_is_remembered_once_even_while_its_successor_cannot_be_checked(tmp_path):
    """The newer release cannot be installed for days. The older one is remembered on the
    first of them, runs, passes and is done; it must not be remembered again each day
    because the point that replaced it has still said nothing."""
    newer, old = _point("claim_top", "0.23.2"), _displaced_point("0.23.1")
    plan = _plan_with_table(_FLOOR, newer)
    stuck = _results(_result(_FLOOR), _result(newer, "install_network"))
    first = report.next_state(_top_failed_yesterday(), stuck, _RUN, tmp_path, plan)
    assert [item["key"] for item in first["displaced"]["openai_agents"]] == [old["key"]]
    assert "last_checked" not in first["points"][newer["key"]]

    plan = _plan_with_table(_FLOOR, newer, old)
    passed = _results(_result(_FLOOR), _result(newer, "install_network"), _result(old))
    second = report.next_state(first, passed, _RUN, tmp_path, plan)
    assert second["displaced"] == {}
    third = report.next_state(second, stuck, _RUN, tmp_path, _plan_with_table(_FLOOR, newer))
    assert third["displaced"] == {}


@pytest.mark.parametrize("today", sorted(report.NOT_OURS | {"test_deps_unresolvable"}))
def test_a_displaced_release_that_cannot_be_checked_stays_on_the_issue(tmp_path, today):
    """Its own point has never run it, so that point has said nothing yet. What is known is
    the failure it was remembered for, and a day it cannot be installed must not close the
    issue over it. Nor may each such day post a comment."""
    newer, old = _point("claim_top", "0.23.2"), _displaced_point("0.23.1")
    plan = _plan_with_table(_FLOOR, newer, old)
    results = _results(_result(_FLOOR), _result(newer), _result(old, today))
    remembered = {"displaced": {"openai_agents": [{"key": old["key"], **_remembered_item()}]}}
    assert report.judge(plan, results) == []

    state = report.next_state(remembered, results, _RUN, tmp_path, plan)
    issue = report.adapter_issue("openai_agents", plan, results, _RUN, state)
    assert issue["open"] and state["displaced"]
    (line,) = issue["changes"]
    assert "did not pass when last checked, on `2026-11-25`, and could not be checked today" in line

    tomorrow = report.next_state(state, results, _RUN, tmp_path, plan)
    again = report.adapter_issue("openai_agents", plan, results, _RUN, tomorrow)
    assert again["open"] and again["fingerprint"] == issue["fingerprint"]


def test_a_day_the_run_itself_broke_does_not_erase_a_failure(tmp_path):
    """The release failed. The next day the harness broke on it, which says nothing about
    the release. The day after, a newer patch replaced it: it is still owed a run."""
    plan = _plan_with_table(_FLOOR, _TOP)
    broke = _results(_result(_FLOOR), _result(_TOP, "harness_error"))
    state = report.next_state(_top_failed_yesterday(), broke, _RUN, tmp_path, plan)
    assert state["points"][_TOP["key"]]["last_checked"]["outcome"] == "fail"

    newer = _point("claim_top", "0.23.2")
    plan = _plan_with_table(_FLOOR, newer)
    after = report.next_state(
        state, _results(_result(_FLOOR), _result(newer)), _RUN, tmp_path, plan
    )
    assert [item["targets"] for item in after["displaced"]["openai_agents"]] == [
        {"openai-agents": "0.23.1"}
    ]


@pytest.mark.parametrize(
    ("today", "kept", "open_"),
    [("pass", False, False), ("fail", False, True), ("install_network", True, True)],
)
def test_a_remembered_release_a_rule_picks_again_is_settled_only_by_a_run(
    tmp_path, today, kept, open_
):
    """The patch that replaced it was withdrawn, so the newest-release rule picks the old
    one again and the plan makes no second point. A pass ends it; a failure is on that
    point's own record from then on; a day it cannot be installed settles nothing."""
    again = {**_point("claim_top", "0.23.1"), "planned_python": "3.14"}
    plan = _plan_with_table(_FLOOR, again)
    results = _results(
        _result(_FLOOR),
        _result(again, today, failed_ids=["t.py::test_tree"] if today == "fail" else []),
    )
    key = "openai_agents--displaced--0.23.1-py3.14"
    remembered = {"displaced": {"openai_agents": [{"key": key, **_remembered_item()}]}}
    state = report.next_state(remembered, results, _RUN, tmp_path, plan)
    assert bool(state["displaced"]) is kept
    assert report.adapter_issue("openai_agents", plan, results, _RUN, state)["open"] is open_


def test_a_replaced_failure_survives_the_whole_path_from_results_to_the_next_plan(
    tmp_path, monkeypatch
):
    """Three runs through `main`, with files on disk as the workflow has them: the release
    fails on both interpreters, a newer patch replaces it, and the plan of the day after
    runs it again on each interpreter it failed on."""
    from compat import plan as planner

    table = [
        {"distribution": "openai-agents", "minors": ("0.22", "0.23"), "known_bad": (), "since": ""}
    ]
    history = {
        "openai-agents": [
            {
                "version": v,
                "files": [
                    {"uploaded": f"{d}T12:00:00Z", "yanked": False, "requires_python": ">=3.10"}
                ],
            }
            for v, d in (("0.22.0", "2026-08-19"), ("0.23.1", "2026-10-02"))
        ]
    }
    monkeypatch.setattr(planner, "read_measured", lambda: {"openai_agents": table})
    monkeypatch.setattr(planner, "fetch_releases", lambda dist: history[dist])
    monkeypatch.setattr(report.GitHub, "_api", lambda self, method, path, payload=None: [])
    said: list[str] = []
    monkeypatch.setattr(report.GitHub, "do", lambda self, what, *rest: said.append(what))
    for name in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "GITHUB_REPOSITORY"):
        monkeypatch.delenv(name, raising=False)
    records = tmp_path / "records"
    records.mkdir()

    def run_day(day: int, failing: set[str]) -> tuple[int, dict, list[dict]]:
        work = tmp_path / f"day{day}"
        work.mkdir()
        stamp = f"2026-11-{day:02d}T19:40:00Z"
        argv = [
            "--out",
            str(work / "plan.json"),
            "--cutoff",
            stamp,
            "--state",
            str(records / "state.json"),
        ]
        assert planner.main(argv) == 0
        points = json.loads((work / "plan.json").read_text())["points"]
        for planned in points:
            failed = planned["role"] in failing
            result = {
                "schema": 1,
                "key": planned["key"],
                "adapter": planned["adapter"],
                "targets": planned["targets"],
                "outcome": "fail" if failed else "pass",
                "cause": None,
                "tests": {
                    "failed_ids": ["t.py::test_tree"] if failed else [],
                    "skipped_ids": [],
                    "files": ["t.py"],
                    "passed": 1,
                    "failed": int(failed),
                    "errors": 0,
                    "skipped": 0,
                },
            }
            _write(work / "results", result)
        argv = [
            "--plan",
            str(work / "plan.json"),
            "--results",
            str(work / "results"),
            "--state-dir",
            str(records),
            "--write",
        ]
        code = report.main(argv)
        return code, json.loads((records / "state.json").read_text()), points

    code, state, _ = run_day(20, {"claim_top", "claim_top_oldpy"})
    assert code == report.RED_EXIT and state["displaced"] == {}
    assert state["points"]["openai_agents--claim_top"]["last_checked"]["python"] == "3.14"

    history["openai-agents"].append(
        {
            "version": "0.23.2",
            "files": [
                {"uploaded": "2026-11-21T08:00:00Z", "yanked": False, "requires_python": ">=3.10"}
            ],
        }
    )
    said.clear()
    code, state, points = run_day(21, set())
    assert code == 0 and not [p for p in points if p["role"] == "displaced"]
    assert said == ["open Compatibility: openai-agents"], "green, and the issue still says so"
    assert sorted(item["key"] for item in state["displaced"]["openai_agents"]) == [
        "openai_agents--displaced--0.23.1-py3.10",
        "openai_agents--displaced--0.23.1-py3.14",
    ]

    code, state, points = run_day(22, {"displaced"})
    assert {p["key"]: p["python"] for p in points if p["role"] == "displaced"} == {
        "openai_agents--displaced--0.23.1-py3.10": "3.10",
        "openai_agents--displaced--0.23.1-py3.14": "3.14",
    }
    assert code == report.RED_EXIT and len(state["displaced"]["openai_agents"]) == 2

    table[0]["known_bad"] = ("0.23.1",)
    code, state, points = run_day(23, set())
    assert code == 0 and state["displaced"] == {}
    assert not [p for p in points if p["role"] == "displaced"]
