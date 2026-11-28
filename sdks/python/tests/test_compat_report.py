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
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from compat import gate, kinds, model, point, render, report, verdict, watchdog  # noqa: E402
from compat import plan as planner  # noqa: E402

_FIXTURES = Path(__file__).parent / "fixtures" / "compat"


@pytest.fixture(autouse=True)
def _two_kinds_that_are_not_red_capable(monkeypatch):
    """Kinds for these tests to watch: one of every run, one of a weekly run."""
    daily = kinds.Kind("anything", frozenset({"daily"}), 3, 1, "when it passes")
    weekly = kinds.Kind("weekly thing", frozenset({"weekly"}), 9, 1, "when it passes")
    monkeypatch.setitem(kinds.KINDS, "anything", daily)
    monkeypatch.setitem(kinds.KINDS, "weekly_thing", weekly)


@pytest.fixture(autouse=True)
def _no_published_release(monkeypatch):
    """No test here reaches PyPI or a tag unless it says so."""
    nothing = {"problems": [], "points": [], "adapters": {}}
    monkeypatch.setattr(planner, "fetch_published", lambda cutoff, only=None: nothing)


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
    return {
        **point,
        "schema": report.SCHEMA,
        "outcome": outcome,
        "cause": None,
        "tests": tests,
        **more,
    }


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
    assert line["schema"] == report.SCHEMA == 2
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
    assert "No rule picks it any more, so no other point tests it." in line
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
    monkeypatch.setattr(planner, "read_measured", lambda *paths: {"openai_agents": table})
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
                "schema": report.SCHEMA,
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


def test_a_pass_on_another_interpreter_or_at_the_floor_settles_nothing(tmp_path):
    """It failed on Python 3.10 with today's packages. The same release passing on 3.14,
    or at the floor with the packages of its own time, says nothing about that."""
    key = "openai_agents--displaced--0.23.1-py3.10"
    item = {"key": key, **_remembered_item(), "python": "3.10"}
    newest = {**_point("claim_top", "0.23.1"), "planned_python": "3.14"}
    oldest = {**_point("claim_top_oldpy", "0.23.1"), "planned_python": "3.10"}
    floor = {**_point("floor", "0.23.1"), "planned_python": "3.10"}
    plan = _plan_with_table(floor, newest, oldest)
    results = _results(_result(floor), _result(newest), _result(oldest, "install_network"))
    before = {"displaced": {"openai_agents": [item]}}
    state = report.next_state(before, results, _RUN, tmp_path, plan)
    assert [kept["key"] for kept in state["displaced"]["openai_agents"]] == [key]
    assert report.adapter_issue("openai_agents", plan, results, _RUN, state)["open"]


def test_a_pair_is_settled_by_listing_either_of_its_releases_as_bad(tmp_path):
    """LangGraph is tested as two distributions pinned together. The issue says listing
    the release as bad closes it; that has to hold for the companion's release too."""
    pair = {"langgraph": "1.2.3", "langgraph-prebuilt": "1.1.0"}
    newer = {"langgraph": "1.2.3", "langgraph-prebuilt": "1.1.1"}
    top = {
        "key": "langgraph--claim_top",
        "adapter": "langgraph",
        "role": "claim_top",
        "targets": newer,
        "in_claim": True,
    }
    checked = {
        "date": "2026-11-25",
        "outcome": "fail",
        "cause": None,
        "targets": pair,
        "python": "3.14",
        "failed_ids": ["t.py::test_tree"],
    }
    before = {"points": {top["key"]: {"last_checked": checked}}}
    rows = [
        {"distribution": "langgraph", "minors": ["1.2"], "known_bad": [], "since": ""},
        {"distribution": "langgraph-prebuilt", "minors": ["1.1"], "known_bad": [], "since": ""},
    ]

    def remembered(entries: list[dict]) -> dict:
        plan = _plan(top)
        plan["adapters"]["langgraph"] = {**plan["adapters"]["openai_agents"], "entries": entries}
        return report.next_state(before, _results(_result(top)), _RUN, tmp_path, plan)["displaced"]

    assert [item["targets"] for item in remembered(rows)["langgraph"]] == [pair]
    assert remembered([rows[0], {**rows[1], "known_bad": ["1.1.0"]}]) == {}
    assert remembered([{**rows[0], "known_bad": ["1.2.3"]}, rows[1]]) == {}


def test_what_a_displaced_point_said_is_forgotten_when_its_release_is_settled(tmp_path):
    """Its point passed and the release left the list. Remembered again after a later
    failure, that old pass must not outrank the failure it is remembered for."""
    newer, old = _point("claim_top", "0.23.2"), _displaced_point("0.23.1")
    plan = _plan_with_table(_FLOOR, newer, old)
    item = {"key": old["key"], **_remembered_item()}
    passed = _results(_result(_FLOOR), _result(newer), _result(old))
    before = {"displaced": {"openai_agents": [item]}}
    state = report.next_state(before, passed, _RUN, tmp_path, plan)
    assert state["displaced"] == {} and "last_checked" not in state["points"][old["key"]]

    # The same when the plan simply stops making the point (the release was listed as bad):
    # nothing ran it today, and what it said before goes with it.
    said = {"date": "2026-11-27", "outcome": "fail", "targets": old["targets"], "python": "3.14"}
    held = {
        "displaced": {"openai_agents": [item]},
        "points": {old["key"]: {"last_checked": said}},
    }
    without = _plan_with_table(_FLOOR, newer)
    results = _results(_result(_FLOOR), _result(newer))
    dropped = report.next_state(held, results, _RUN, tmp_path, without)
    assert dropped["displaced"] == {} and "last_checked" not in dropped["points"][old["key"]]
    assert held["points"][old["key"]]["last_checked"] == said, "the state read is not edited"
    untouched = report.next_state({**held, "displaced": {}}, results, _RUN, tmp_path, without)
    assert untouched["points"][old["key"]]["last_checked"] == said, "only on leaving the list"

    state["displaced"] = {"openai_agents": [{**item, "failed_on": "2026-11-30"}]}
    stuck = _results(_result(_FLOOR), _result(newer), _result(old, "install_network"))
    later = report.next_state(state, stuck, _RUN, tmp_path, plan)
    issue = report.adapter_issue("openai_agents", plan, stuck, _RUN, later)
    assert issue["open"] and "on `2026-11-30`" in issue["changes"][0]


# --- kinds: what was written before them stays readable ---------------------------


def test_a_record_line_written_before_kinds_reads_as_what_it_was():
    """Lines on the records branch are never rewritten. A line from before kinds existed
    is read with the only values its new fields could have had then, and every field it
    did have comes back untouched."""
    text = (_FIXTURES / "records-schema-1.jsonl").read_text(encoding="utf-8")
    for raw in text.splitlines():
        old = json.loads(raw)
        assert old["schema"] == 1
        new = model.read_record(old)
        assert {name: new[name] for name in old if name != "run"} == {
            name: value for name, value in old.items() if name != "run"
        }
        assert {name: new["run"][name] for name in old["run"]} == old["run"]
        assert (new["kind"], new["watching"], new["sdk"]) == (
            "framework",
            False,
            {"origin": "build"},
        )
        assert (new["subject"], new["adapters"]) == (old["adapter"], [old["adapter"]])
        assert new["run"]["cadence"] == "daily" and new["plan"] is None
        assert json.loads(raw) == old, "reading does not edit the line"
    line = report.records(_plan(_TOP), _results(_result(_TOP)), _RUN)[0]
    assert model.read_record(line) is line


def test_the_state_written_before_kinds_is_carried_over_whole(tmp_path):
    """The first run after kinds land reads the state the last run before them wrote.
    Every point keeps what it said, its last pass and its lock, and gains who it is."""
    old = json.loads((_FIXTURES / "state-schema-1.json").read_text(encoding="utf-8"))
    kept = json.loads(json.dumps(old))
    state = model.upgrade_state(old)
    assert old == kept, "upgrading does not edit what was read"
    assert state["schema"] == 2 and set(state["points"]) == set(old["points"])
    for key, entry in state["points"].items():
        assert entry["point"] is None
        assert {name: entry[name] for name in old["points"][key]} == old["points"][key]
    assert model.upgrade_state(state) is state

    floor = {**_point("floor", "0.22.0"), "python": "3.10"}
    top = {**_point("claim_top", "0.23.1"), "python": "3.14"}
    plan = _plan(floor, top)
    plan["scope"] = {
        "cadence": "daily",
        "covers": {"framework": ["openai_agents"]},
        "partial": False,
    }
    _write(tmp_path, _result(floor))
    _write(tmp_path, _result(top))
    after = report.next_state(state, report.load_results(plan, tmp_path), _RUN, tmp_path, plan)
    mine = after["points"]["openai_agents--floor"]
    assert mine["last_green"]["date"] == _RUN["date"]
    assert mine["point"]["kind"] == "framework" and mine["point"]["python"] == "3.10"
    # Planned today, kept. Covered and no longer planned, forgotten. Another adapter's,
    # which this plan did not cover, exactly as it was.
    assert "openai_agents--beyond--0.26" not in after["points"]
    assert after["points"]["langgraph--floor"] == state["points"]["langgraph--floor"]
    assert after["kinds"] == {"framework": {"last_complete": _RUN["date"], "points": 2}}


def test_a_run_of_one_adapter_forgets_nothing_and_completes_no_kind(tmp_path):
    state = model.upgrade_state(json.loads((_FIXTURES / "state-schema-1.json").read_text()))
    plan = _plan(_TOP)
    plan["scope"] = {
        "cadence": "daily",
        "covers": {"framework": ["openai_agents"]},
        "partial": True,
    }
    after = report.next_state(state, _results(_result(_TOP)), _RUN, tmp_path, plan)
    assert "openai_agents--floor" not in after["points"], "its own adapter is covered"
    assert "pydantic_ai--floor" in after["points"] and after["kinds"] == {}
    untouched = report.next_state(state, _results(_result(_TOP)), _RUN, tmp_path)
    assert set(untouched["points"]) == set(state["points"]), "no plan, no say"


# --- kinds: watching is never an argument for red ---------------------------------


@pytest.mark.parametrize("outcome", sorted(verdict.OUTCOME_WORDS))
@pytest.mark.parametrize("in_claim", [True, False])
def test_a_watching_point_cannot_turn_the_run_red_whatever_it_says(outcome, in_claim):
    watching = {**_TOP, "key": "watched--top", "kind": "anything", "watching": True}
    watching["in_claim"] = in_claim
    plan = _plan(_FLOOR, _TOP, watching)
    results = _results(_result(_FLOOR), _result(_TOP), _result(watching, outcome))
    assert report.judge(plan, results) == []
    counted = {**watching, "watching": False}
    reasons = report.judge(
        plan, _results(_result(_FLOOR), _result(_TOP), _result(counted, outcome))
    )
    red = outcome in ("no_result", "harness_error") or (in_claim and outcome in verdict.RED)
    assert bool(reasons) is red


def test_only_the_plan_decides_whether_a_point_is_watching(tmp_path):
    """A point job runs third-party code. What it writes about itself is not believed:
    who it is and whether it may turn the run red come from the plan."""
    lying = {**_result(_TOP, "fail"), "watching": True, "kind": "bench", "subject": "nobody"}
    _write(tmp_path, lying)
    plan = _plan({**_TOP, "kind": "framework", "subject": "openai_agents", "watching": False})
    (loaded,) = report.load_results(plan, tmp_path).values()
    assert (loaded["kind"], loaded["subject"], loaded["watching"]) == (
        "framework",
        "openai_agents",
        False,
    )
    assert report.judge(plan, {loaded["key"]: loaded}) != []


def test_a_watching_failure_is_said_apart_and_keeps_the_issue_open_on_a_green_run():
    watched = {**_point("claim_top", "0.23.1"), "key": "watched--top", "watching": True}
    plan = _plan(_FLOOR, _TOP, watched)
    nodes = ["t.py::test_tree"]
    results = _results(_result(_FLOOR), _result(_TOP), _result(watched, "fail", failed_ids=nodes))
    assert report.judge(plan, results) == []
    issue = report.adapter_issue("openai_agents", plan, results, _RUN)
    assert issue["open"] and issue["fingerprint"] != "clear"
    assert "### Watching (never turns the run red)" in issue["body"]
    needs, watching = issue["body"].split("### Watching (never turns the run red)")
    assert "Nothing." in needs.split("### What needs doing")[1]
    assert "framework release: `openai-agents 0.23.1` fails 1 tests" in watching
    assert "`t.py::test_tree`" in issue["body"]

    passing = _results(_result(_FLOOR), _result(_TOP), _result(watched))
    assert not report.adapter_issue("openai_agents", plan, passing, _RUN)["open"]


def test_a_run_does_not_close_what_a_run_it_did_not_cover_found():
    """Issues are read from everything the state knows about a subject. A point of
    another cadence said what it said on the day it ran, and stays said, dated, until
    the run that covers it says otherwise."""
    weekly = {**_point("claim_top", "0.23.1"), "key": "weekly--top", "watching": True}
    weekly["kind"], weekly["subject"] = "anything", "openai_agents"
    state = {
        "points": {
            "weekly--top": {
                "point": model.identity_of(weekly),
                "last": {
                    "date": "2026-11-22",
                    "outcome": "fail",
                    "cause": None,
                    "failed_ids": ["t.py::test_weekly"],
                    "excluded_files": [],
                },
            }
        }
    }
    plan = _plan(_FLOOR, _TOP)
    results = _results(_result(_FLOOR), _result(_TOP))
    issue = report.adapter_issue("openai_agents", plan, results, _RUN, state)
    assert issue["open"] and "(last run `2026-11-22`)" in issue["body"]
    assert "`t.py::test_weekly`" in issue["body"]
    again = report.adapter_issue("openai_agents", plan, results, _RUN, state)
    assert again["fingerprint"] == issue["fingerprint"], "no news, no comment"


_PLANNED_TOP = {
    **_TOP,
    "python": "3.14",
    "kind": "framework",
    "subject": "openai_agents",
    "adapters": ["openai_agents"],
    "watching": False,
    "sdk": {"origin": "build"},
    "runner": "linux-x64",
    "install": {"requires": ["h2"]},
    "check": {"files": ["test_x.py"]},
    "probe": {"imports": ["agents"]},
    "cutoff": "2026-11-26T19:40:00Z",
    "assume_measured": False,
    "floats": [],
    "budget_s": 2700,
    "fresh_days": 3,
}


def test_a_record_line_keeps_what_the_plan_said_and_what_the_point_measured_apart(tmp_path):
    """Who the point is comes from the plan. Which interpreter ran and which wheel it
    imported are what the point saw, and a record without them cannot be reproduced."""
    measured = {
        "python": "3.14.2",
        "sdk": {"wheel": "w.whl", "wheel_sha256": "ab" * 32, "version": "0.7.0b1"},
        "runner": "a-runner-of-its-own",
    }
    _write(tmp_path, _result(_TOP, **measured))
    plan = _plan(_PLANNED_TOP)
    plan["cadence"] = "daily"
    results = report.load_results(plan, tmp_path)
    (line,) = report.records(plan, results, {**_RUN, "cadence": "daily", "scope": "all"})
    assert (line["schema"], line["kind"], line["subject"], line["adapters"], line["watching"]) == (
        2,
        "framework",
        "openai_agents",
        ["openai_agents"],
        False,
    )
    assert line["python"] == "3.14.2" and line["sdk_wheel_sha256"] == "ab" * 32
    assert line["sdk"] == {"origin": "build", "version": "0.7.0b1"}
    assert line["runner"] == "linux-x64", "the runner is the plan's to say"
    assert line["plan"] == {
        "python": "3.14",
        "floats": [],
        "cutoff": "2026-11-26T19:40:00Z",
        "assume_measured": False,
        "install": {"requires": ["h2"]},
        "probe": {"imports": ["agents"]},
        "check": {"files": ["test_x.py"]},
        "budget_s": 2700,
        "fresh_days": 3,
    }
    assert line["run"]["cadence"] == "daily"
    json.dumps(line)

    issue = report.adapter_issue("openai_agents", plan, results, _RUN)
    assert "| `3.14.2` |" in issue["body"], "the table shows the interpreter that ran"
    state = report.next_state({}, results, _RUN, tmp_path, plan)
    kept = state["points"][_TOP["key"]]
    assert kept["point"]["python"] == "3.14" and kept["point"]["sdk"] == {"origin": "build"}
    assert kept["last_checked"]["python"] == "3.14", "the interpreter line the plan asked for"


def test_a_watching_point_does_not_make_up_the_number_of_points_an_adapter_needs():
    """Two points is the least that tests a range. A watching point beside one framework
    point is not the second: it cannot turn the run red, so it cannot stand for one that can."""
    watching = {**_TOP, "key": "watched--top", "kind": "anything", "watching": True}
    plan = _plan(_TOP, watching)
    results = _results(_result(_TOP), _result(watching))
    assert report.judge(plan, results) == ["plan: openai_agents has fewer than two points"]
    assert report.judge(_plan(_TOP, _FLOOR, watching), _results(_result(_TOP))) == []


def _kept(point: dict, outcome: str, nodes: list[str], day: str = "2026-11-22") -> dict:
    """A state entry as a run that covered `point` left it."""
    last = {
        "date": day,
        "outcome": outcome,
        "cause": None,
        "failed_ids": nodes,
        "excluded_files": [],
    }
    return {"point": model.identity_of(point), "last": last}


def test_what_a_run_did_not_run_is_never_worded_as_checked_today():
    """An adapter the plan could not cover keeps its points in the state. Its issue is
    still written, and a failure from another day must read as that day's."""
    plan = _plan()
    plan["adapters"]["openai_agents"]["problems"] = ["the index did not answer"]
    above_passed = {**_ABOVE, "kind": "framework", "subject": "openai_agents"}
    state = {
        "points": {
            _TOP["key"]: _kept(_TOP, "fail", ["t.py::test_tree"]),
            _FLOOR["key"]: _kept(_FLOOR, "pass", []),
            _ABOVE["key"]: _kept(above_passed, "pass", []),
        }
    }
    issue = report.adapter_issue("openai_agents", plan, {}, _RUN, state)
    (line,) = [line for line in issue["body"].splitlines() if line.startswith("- `openai-agents 0")]
    assert line == (
        "- `openai-agents 0.23.1` is inside the tested range and did not pass when it was last "
        "run, on `2026-11-22`: fails 1 tests. This run did not run it."
    )
    assert "can be widened" not in issue["body"], "nobody saw that pass today"
    assert "`t.py::test_tree`" in issue["body"] and issue["open"]
    assert (
        report.adapter_issue("openai_agents", plan, {}, _RUN, state)["fingerprint"]
        == (issue["fingerprint"])
    )
    a_day_later = {**_RUN, "date": "2026-11-27"}
    assert (
        report.adapter_issue("openai_agents", plan, {}, a_day_later, state)["fingerprint"]
        == (issue["fingerprint"])
    ), "the same old news on a new day is not a comment"
    assert f"<!-- compat-fingerprint: {issue['fingerprint']} -->" in issue["body"]

    state["points"][_TOP["key"]]["last"]["outcome"] = "pass"
    assert not report.adapter_issue("openai_agents", plan, {}, _RUN, state)["open"]


def test_a_failure_outlives_a_day_it_could_not_be_checked_and_a_day_it_was_not_planned(tmp_path):
    """Three days through the state: the floor fails, then the index is down, then the
    plan cannot cover the adapter at all. What is known on the third day is still the
    failure of the first, and the issue must not close on it."""
    covered = _covering({"framework": ["openai_agents"]})
    day = lambda n: {**_RUN, "date": f"2026-11-2{n}"}  # noqa: E731
    nodes = ["t.py::test_tree"]
    state = report.next_state(
        {},
        _results(_result(_FLOOR, "fail", failed_ids=nodes), _result(_TOP)),
        day(4),
        tmp_path,
        covered,
    )
    state.pop("_locks"), state.pop("_forget")
    state = report.next_state(
        state,
        _results(_result(_FLOOR, "install_network"), _result(_TOP)),
        day(5),
        tmp_path,
        covered,
    )
    state.pop("_locks"), state.pop("_forget")
    assert state["points"][_FLOOR["key"]]["last"]["outcome"] == "install_network"

    uncovered = _plan()
    uncovered["adapters"]["openai_agents"]["problems"] = ["the index did not answer"]
    uncovered["scope"] = {"cadence": "daily", "covers": {"framework": []}, "partial": False}
    state = report.next_state(state, {}, day(6), tmp_path, uncovered)
    issue = report.adapter_issue("openai_agents", uncovered, {}, day(6), state)
    assert issue["open"] and issue["changes"] == [
        "`openai-agents 0.22.0` is inside the tested range and did not pass when it was last "
        "run, on `2026-11-24`: fails 1 tests. This run did not run it."
    ]
    assert "`openai-agents 0.23.1` (not run by this run)" in issue["closing"]


def test_a_point_cannot_say_of_itself_that_it_was_not_run_today(tmp_path):
    """Which rows are remembered is read from the state. A result that calls itself old
    is today's result, worded as today's, and a release above the range stays listed."""
    forged = {"as_of": "2020-01-01", "fresh_days": 9999}
    watched = {**_TOP, "key": "watched--top", "watching": True}
    _write(tmp_path, _result(_FLOOR))
    _write(tmp_path, _result(_TOP, "fail", failed_ids=["t.py::test_a"], **forged))
    _write(tmp_path, _result(_ABOVE, "fail", failed_ids=["t.py::test_b"], **forged))
    _write(tmp_path, _result(watched, "fail", failed_ids=["t.py::test_c"], **forged))
    with_watched = _plan(_FLOOR, _TOP, _ABOVE, watched)
    body = report.adapter_issue(
        "openai_agents", with_watched, report.load_results(with_watched, tmp_path), _RUN
    )["body"]
    assert "### Watching" in body and "(last run" not in body and "2020-01-01" not in body
    plan = _plan(_FLOOR, _TOP, _ABOVE)
    results = report.load_results(plan, tmp_path)
    honest = _results(
        _result(_FLOOR),
        _result(_TOP, "fail", failed_ids=["t.py::test_a"]),
        _result(_ABOVE, "fail", failed_ids=["t.py::test_b"]),
    )
    said = report.adapter_issue("openai_agents", plan, results, _RUN)
    wanted = report.adapter_issue("openai_agents", plan, honest, _RUN)
    assert said["changes"] == wanted["changes"] and said["fingerprint"] == wanted["fingerprint"]
    assert "2020-01-01" not in said["body"] and len(said["changes"]) == 2
    state = report.next_state({}, results, _RUN, tmp_path, plan)
    assert state["points"][_TOP["key"]]["point"]["fresh_days"] == 3


def test_a_remembered_release_above_the_range_is_never_called_tested():
    plan = _plan()
    plan["adapters"]["openai_agents"]["problems"] = ["the index did not answer"]
    above = {**_ABOVE, "kind": "framework", "subject": "openai_agents"}
    state = {"points": {_ABOVE["key"]: _kept(above, "fail", ["t.py::test_tree"])}}
    issue = report.adapter_issue("openai_agents", plan, {}, _RUN, state)
    assert not issue["open"] and "inside the tested range" not in issue["body"]


def test_a_replaced_failure_outlives_a_day_it_could_not_be_checked_and_a_day_nothing_ran(tmp_path):
    """Four days: the newest tested release fails, a newer patch replaces it, its own point
    cannot install it, and then the plan cannot cover the adapter. Its point never ran
    it, so the only thing known about it is the failure it was remembered for, and the
    issue says that once and stays open."""
    old = _point("claim_top", "0.23.1")
    new = {**_point("claim_top", "0.23.2"), "python": "3.14"}
    covering = {"cadence": "daily", "covers": {"framework": ["openai_agents"]}, "partial": False}
    state: dict = {}

    def day(n: int, plan: dict, *results: dict) -> dict:
        nonlocal state
        run = {**_RUN, "date": f"2026-11-2{n}"}
        loaded = _results(*results)
        for result in loaded.values():
            result.setdefault("planned_python", "3.14")
        state = report.next_state(state, loaded, run, tmp_path, plan)
        state.pop("_locks"), state.pop("_forget")
        return report.adapter_issue("openai_agents", plan, loaded, run, state)

    first = _plan_with_table(_FLOOR, {**old, "python": "3.14"})
    first["scope"] = covering
    assert day(1, first, _result(_FLOOR), _result(old, "fail", failed_ids=["t.py::a"]))["open"]
    second = _plan_with_table(_FLOOR, new)
    second["scope"] = covering
    assert day(2, second, _result(_FLOOR), _result(new))["open"]
    (item,) = state["displaced"]["openai_agents"]
    own = {**_point("displaced", "0.23.1"), "key": item["key"], "python": "3.14"}
    third = _plan_with_table(_FLOOR, new, own)
    third["scope"] = covering
    outage = day(3, third, _result(_FLOOR), _result(new), _result(own, "install_network"))
    assert outage["open"] and "could not be checked today" in outage["changes"][0]

    uncovered = _plan_with_table()
    uncovered["adapters"]["openai_agents"]["problems"] = ["the index did not answer"]
    uncovered["scope"] = {**covering, "covers": {"framework": []}}
    silent = day(4, uncovered)
    assert state["displaced"]["openai_agents"] == [item]
    assert silent["open"] and silent["changes"] == [
        "`openai-agents 0.23.1` is inside the tested range and did not pass when it was last "
        "run, on `2026-11-21`. No rule picks it any more (a newer release took its place, or "
        "the points moved), so it is run on its own from the next run on. This closes when "
        "it passes, or when the table of tested versions lists it under `known_bad`."
    ]


def test_a_remembered_failure_is_said_once_when_its_own_point_and_the_list_both_hold_it():
    """The displaced point failed on its own day, and the list still names the release. On
    a day nothing ran, one line speaks for it, with the cause of the day it was checked."""
    own = {**_point("displaced", "0.23.1"), "key": "openai_agents--displaced--0.23.1-py3.14"}
    entry = _kept(own, "install_network", [], "2026-11-25")
    entry["last"]["cause"] = None
    entry["last_checked"] = {
        "date": "2026-11-24",
        "outcome": "fail",
        "cause": "upstream",
        "targets": dict(own["targets"]),
        "failed_ids": ["t.py::test_tree"],
    }
    item = {**_remembered_item(), "key": own["key"]}
    state = {"points": {own["key"]: entry}, "displaced": {"openai_agents": [item]}}
    plan = _plan_with_table()
    plan["adapters"]["openai_agents"]["problems"] = ["the index did not answer"]
    issue = report.adapter_issue("openai_agents", plan, {}, _RUN, state)
    assert len(issue["changes"]) == 1 and "on `2026-11-24`: fails 1 tests" in issue["changes"][0]
    assert "| tests fail | a dependency release (" in issue["body"]


@pytest.mark.parametrize("seed", range(600))
def test_a_known_failure_keeps_its_issue_open_through_any_week(tmp_path, seed):
    """A week of invented days: releases pass, fail, cannot be installed or leave no
    result; a newer patch replaces the newest; on some days the plan cannot cover the
    adapter at all. Whatever the order, a release inside the range whose last real check
    failed is named on an open issue, and a day everything ran and passed with nothing
    owed has nothing to do."""
    rng = random.Random(seed)
    covering = {"cadence": "daily", "covers": {"framework": ["openai_agents"]}, "partial": False}
    state: dict = {}
    last_check: dict[str, str] = {}
    top = "0.23.1"
    for n in range(1, 8):
        run = {**_RUN, "date": f"2026-11-2{n}"}
        covered = rng.random() > 0.25
        if covered and top == "0.23.1" and rng.random() < 0.3:
            top = "0.23.2"
        points = []
        if covered:
            points = [{**_FLOOR, "python": "3.10"}, {**_point("claim_top", top), "python": "3.14"}]
            for item in (state.get("displaced") or {}).get("openai_agents", []):
                version = item["targets"]["openai-agents"]
                if version != top:
                    own = {**_point("displaced", version), "key": item["key"], "python": "3.14"}
                    points.append(own)
        plan = _plan_with_table(*points)
        plan["scope"] = covering if covered else {**covering, "covers": {"framework": []}}
        if not covered:
            plan["adapters"]["openai_agents"]["problems"] = ["the index did not answer"]
        results = {}
        for planned in points:
            outcome = rng.choice(["pass", "pass", "fail", "install_network", "no_result"])
            nodes = ["t.py::test_tree"] if outcome == "fail" else []
            result = _result(planned, outcome, failed_ids=nodes)
            result["planned_python"] = planned["python"]
            results[planned["key"]] = result
            if outcome in ("pass", "fail"):
                last_check[planned["targets"]["openai-agents"]] = outcome
        state = report.next_state(state, results, run, tmp_path, plan)
        state.pop("_locks"), state.pop("_forget")
        issue = report.adapter_issue("openai_agents", plan, results, run, state)
        owed = sorted(version for version, outcome in last_check.items() if outcome == "fail")
        said = "\n".join(issue["changes"])
        for version in owed:
            assert f"`openai-agents {version}`" in said, (n, version, issue["changes"])
        assert issue["open"] or not owed
        if covered and not owed and all(r["outcome"] == "pass" for r in results.values()):
            assert not issue["open"], (n, issue["changes"])
        assert len(issue["changes"]) == len(set(issue["changes"]))
        lines = [change.split("` is ")[0] for change in issue["changes"]]
        assert len(lines) == len(set(lines)), f"day {n}: one release, one line: {issue['changes']}"


def test_what_a_point_said_of_another_release_is_not_said_of_this_one():
    plan = _plan()
    plan["adapters"]["openai_agents"]["problems"] = ["the index did not answer"]
    entry = _kept(_TOP, "install_network", [])
    entry["last_checked"] = {
        "date": "2026-11-20",
        "outcome": "fail",
        "cause": None,
        "targets": {"openai-agents": "0.23.0"},
        "failed_ids": ["t.py::test_tree"],
    }
    issue = report.adapter_issue("openai_agents", plan, {}, _RUN, {"points": {_TOP["key"]: entry}})
    assert not issue["open"] and "2026-11-20" not in issue["body"]
    entry["last_checked"]["targets"] = dict(_TOP["targets"])
    again = report.adapter_issue("openai_agents", plan, {}, _RUN, {"points": {_TOP["key"]: entry}})
    assert again["open"] and "on `2026-11-20`: fails 1 tests" in again["body"]


def test_a_failure_that_stops_being_run_is_news_once():
    """Today's failure and the same failure remembered are two different things to read:
    the second says nobody is running it. The issue says so in a comment, one time."""
    plan = _plan(_FLOOR, _TOP)
    nodes = ["t.py::test_tree"]
    today = report.adapter_issue(
        "openai_agents",
        plan,
        _results(_result(_FLOOR), _result(_TOP, "fail", failed_ids=nodes)),
        _RUN,
    )
    unplanned = _plan()
    state = {"points": {_TOP["key"]: _kept(_TOP, "fail", nodes, "2026-11-26")}}
    remembered = report.adapter_issue("openai_agents", unplanned, {}, _RUN, state)
    assert today["open"] and remembered["open"]
    assert today["fingerprint"] != remembered["fingerprint"]


def test_the_fingerprint_does_not_depend_on_the_order_results_arrive_in():
    plan = _plan(_FLOOR, _TOP, _ABOVE)
    rows = [
        _result(_FLOOR, "fail", failed_ids=["t.py::test_a"]),
        _result(_TOP, "fail", failed_ids=["t.py::test_b"]),
        _result(_ABOVE, "fail", failed_ids=["t.py::test_c"]),
    ]
    forward = report.adapter_issue("openai_agents", plan, _results(*rows), _RUN)
    backward = report.adapter_issue("openai_agents", plan, _results(*reversed(rows)), _RUN)
    assert forward["fingerprint"] == backward["fingerprint"]
    assert f"<!-- compat-fingerprint: {backward['fingerprint']} -->" in backward["body"]


def test_the_version_a_point_says_it_imported_is_cleaned_like_everything_else_it_says(tmp_path):
    measured = {"sdk": {"wheel": "w.whl", "wheel_sha256": "ab", "version": "0.7`|<b>\n" + "9" * 90}}
    _write(tmp_path, _result(_TOP, **measured))
    plan = _plan(_PLANNED_TOP)
    (line,) = report.records(plan, report.load_results(plan, tmp_path), _RUN)
    assert len(line["sdk"]["version"]) == 40 and not re.search(r"[`|\n]", line["sdk"]["version"])


@pytest.mark.parametrize("in_claim, said", [(True, True), (False, False)])
def test_a_watching_point_that_could_not_be_installed_is_said_only_inside_the_range(in_claim, said):
    watched = {**_TOP, "key": "watched--top", "watching": True, "in_claim": in_claim}
    plan = _plan(_FLOOR, _TOP, watched)
    results = _results(_result(_FLOOR), _result(_TOP), _result(watched, "not_installable"))
    issue = report.adapter_issue("openai_agents", plan, results, _RUN)
    assert issue["open"] is said
    assert ("### Watching (never turns the run red)" in issue["body"]) is said


def test_a_watching_failure_is_news_once():
    """It reaches the comment that says what changed, like anything else on the issue."""
    watched = {**_TOP, "key": "watched--top", "watching": True}
    plan = _plan(_FLOOR, _TOP, watched)
    failing = _result(watched, "fail", failed_ids=["t.py::test_tree"])
    issue = report.adapter_issue(
        "openai_agents", plan, _results(_result(_FLOOR), _result(_TOP), failing), _RUN
    )
    (change,) = issue["changes"]
    assert change.startswith("framework release: `openai-agents 0.23.1` fails 1 tests.")
    assert change.endswith(f"Closes {kinds.KINDS['framework'].closes}.")
    undeclared = {**failing, "kind": "nobody declared"}
    said = report.adapter_issue(
        "openai_agents", plan, _results(_result(_FLOOR), _result(_TOP), undeclared), _RUN
    )["changes"][0]
    assert said.startswith("nobody declared: ") and said.endswith("Closes when it passes.")


def _covering(covers: dict, *, partial: bool = False) -> dict:
    plan = _plan(_FLOOR, _TOP)
    plan["scope"] = {"cadence": "daily", "covers": covers, "partial": partial}
    return plan


def test_a_run_forgets_only_points_of_a_kind_and_subject_it_covers(tmp_path):
    """The subject is read from what the state kept about the point, not from its key:
    a point of another kind may be about an adapter its key does not start with."""
    other_kind = {**_TOP, "key": "openai_agents--weekly", "kind": "weekly_thing"}
    other_subject = {**_TOP, "key": "openai_agents--about_langgraph", "subject": "langgraph"}
    renamed = {**_TOP, "key": "combo--one", "subject": "openai_agents"}
    gone = {**_TOP, "key": "openai_agents--minor_newest--0.21"}
    state = {"points": {p["key"]: _kept(p, "pass", []) for p in (other_kind, other_subject)}}
    state["points"].update({p["key"]: _kept(p, "pass", []) for p in (renamed, gone)})
    state["points"]["openai_agents--before_kinds"] = {"point": None, "last": {}}
    plan = _covering({"framework": ["openai_agents"]})
    after = report.next_state(state, _results(_result(_FLOOR), _result(_TOP)), _RUN, tmp_path, plan)
    assert sorted(after["_forget"]) == [
        "combo--one",
        "openai_agents--before_kinds",
        "openai_agents--minor_newest--0.21",
    ]
    assert {"openai_agents--weekly", "openai_agents--about_langgraph"} <= set(after["points"])
    assert not set(after["_forget"]) & set(after["points"])


def test_a_whole_run_counts_the_points_of_each_kind_it_covers(tmp_path):
    other = {**_TOP, "key": "watched--top", "kind": "weekly_thing", "watching": True}
    results = _results(_result(_FLOOR), _result(_TOP), _result(other))
    for result in results.values():
        result.setdefault("kind", "framework")
    plan = _covering({"framework": ["openai_agents"], "weekly_thing": ["openai_agents"]})
    before = {"kinds": {"never_today": {"last_complete": "2026-11-01", "points": 9}}}
    after = report.next_state(before, results, _RUN, tmp_path, plan)
    assert after["kinds"] == {
        "framework": {"last_complete": "2026-11-26", "points": 2},
        "weekly_thing": {"last_complete": "2026-11-26", "points": 1},
        "never_today": {"last_complete": "2026-11-01", "points": 9},
    }
    partial = _covering({"framework": ["openai_agents"]}, partial=True)
    assert report.next_state(before, results, _RUN, tmp_path, partial)["kinds"] == before["kinds"]


def test_the_state_holds_a_points_words_cleaned_and_in_bounded_number(tmp_path):
    """`state.json` is public and issues are built from it on days the point did not run."""
    nodes = [f"t.py::test_{n}" for n in range(300)]
    nodes[0] = "t.py::test_x`\n- [click](https://example.invalid)" + "y" * 500
    result = _result(_TOP, "fail", failed_ids=nodes)
    result["tests"]["excluded_files"] = ["a`b.py"] * 300
    kept = report.next_state({}, _results(result), _RUN, tmp_path)["points"][_TOP["key"]]
    last, checked = kept["last"], kept["last_checked"]
    assert len(last["failed_ids"]) == 200 and len(last["excluded_files"]) == 200
    assert len(last["failed_ids"][0]) <= 300
    assert checked["failed_ids"] == last["failed_ids"], "the same words, wherever they are kept"
    assert not re.search(r"[`\n]", last["failed_ids"][0] + last["excluded_files"][0])


def test_a_forgotten_point_takes_its_last_pass_with_it(tmp_path, monkeypatch):
    """Through `main`, on files: the lock of a key no plan makes any more is removed with
    its state entry, and nothing of a point the run did not cover is touched."""
    monkeypatch.setattr(report.GitHub, "_api", lambda self, method, path, payload=None: [])
    monkeypatch.setattr(report.GitHub, "do", lambda self, what, *rest: None)
    for name in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "GITHUB_REPOSITORY"):
        monkeypatch.delenv(name, raising=False)
    records, results_dir = tmp_path / "records", tmp_path / "results"
    (records / "last-green").mkdir(parents=True)
    gone = {**_TOP, "key": "openai_agents--minor_newest--0.21"}
    elsewhere = {**_TOP, "key": "langgraph--claim_top", "adapter": "langgraph"}
    state = {
        "schema": 2,
        "points": {p["key"]: _kept(p, "pass", []) for p in (gone, elsewhere)},
        "displaced": {"langgraph": [{"key": "langgraph--displaced--1", "targets": {}}]},
        "kinds": {},
        "last_scheduled": "2026-11-25T19:40:00Z",
    }
    (records / "state.json").write_text(json.dumps(state), encoding="utf-8")
    for key in (gone["key"], elsewhere["key"]):
        (records / "last-green" / f"{key}.txt").write_text("a==1\n", encoding="utf-8")
        (records / "last-green" / f"{key}.files.json").write_text("[]", encoding="utf-8")
    plan = _covering({"framework": ["openai_agents"]}, partial=True)
    plan["partial"] = True
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    _write(results_dir, _result(_FLOOR))
    _write(results_dir, _result(_TOP))

    argv = ["--plan", str(tmp_path / "plan.json"), "--results", str(results_dir)]
    argv += ["--state-dir", str(records)]
    assert report.main(argv) == 0
    assert (records / "last-green" / f"{gone['key']}.txt").is_file(), "a dry run writes nothing"

    assert report.main([*argv, "--write"]) == 0
    left = sorted(path.name for path in (records / "last-green").iterdir())
    assert left == [
        "langgraph--claim_top.files.json",
        "langgraph--claim_top.txt",
        "openai_agents--claim_top.files.json",
        "openai_agents--claim_top.txt",
        "openai_agents--floor.files.json",
        "openai_agents--floor.txt",
    ]
    written = json.loads((records / "state.json").read_text())
    assert gone["key"] not in written["points"] and elsewhere["key"] in written["points"]
    assert "_forget" not in written and "_locks" not in written
    assert written["displaced"]["langgraph"] and written["last_scheduled"] == "2026-11-25T19:40:00Z"


def test_a_state_written_before_kinds_keeps_everything_it_remembered():
    old = {
        "schema": 1,
        "points": {"a--floor": {"last": {"outcome": "pass"}}},
        "displaced": {"openai_agents": [{"key": "openai_agents--displaced--1"}]},
        "last_scheduled": "2026-11-25T19:40:00Z",
    }
    new = model.upgrade_state(old)
    assert new["displaced"] == old["displaced"] and new["last_scheduled"] == old["last_scheduled"]
    assert new["points"] == {"a--floor": {"point": None, "last": {"outcome": "pass"}}}
    assert (new["schema"], new["kinds"]) == (2, {})
    assert model.upgrade_state(new) is new


def test_every_declared_kind_has_words_and_a_way_to_close(monkeypatch):
    monkeypatch.undo()  # the kinds the repository declares, without the two of these tests
    for name, kind in kinds.KINDS.items():
        assert kind.label and kind.cadences and kind.fresh_days > 0, name
        assert name in kinds.RED_CAPABLE or kind.closes, name
    assert kinds.due("daily") == ("framework", "published")
    assert kinds.RED_CAPABLE == {"framework"}, "a new kind starts by being watched"


# --- a kind that is retired, and a run that holds only watching kinds ------------


def test_the_points_of_a_retired_kind_ask_nothing_and_are_forgotten_by_a_whole_run(tmp_path):
    """A kind taken out of the declarations has no run that could ever cover it. Left
    alone, a failure it once found would keep an issue open for good."""
    gone = {**_TOP, "key": "retired--top", "kind": "retired_kind", "watching": True}
    state = {"points": {gone["key"]: _kept(gone, "fail", ["t.py::test_tree"])}}
    plan = _plan(_FLOOR, _TOP)
    results = _results(_result(_FLOOR), _result(_TOP))
    assert not report.adapter_issue("openai_agents", plan, results, _RUN, state)["open"]

    partial = _covering({"framework": ["openai_agents"]}, partial=True)
    assert gone["key"] in report.next_state(state, results, _RUN, tmp_path, partial)["points"]
    whole = _covering({"framework": ["openai_agents"]})
    after = report.next_state(state, results, _RUN, tmp_path, whole)
    assert gone["key"] not in after["points"] and after["_forget"] == [gone["key"]]

    watched = {**gone, "key": "weekly--top", "kind": "weekly_thing"}
    state = {"points": {watched["key"]: _kept(watched, "fail", ["t.py::test_tree"])}}
    kept = report.next_state(state, results, _RUN, tmp_path, whole)
    assert watched["key"] in kept["points"], "a declared kind's points wait for their own run"


def test_a_run_that_holds_only_watching_kinds_is_never_red():
    """Red needs a red-capable question. A cadence that asks none cannot be red for
    having no points, and whatever its watching points say is not a reason either."""
    watched = {**_TOP, "key": "weekly--top", "kind": "weekly_thing", "watching": True}
    weekly = {**_plan(watched), "cadence": "weekly", "adapters": {}}
    assert report.judge(weekly, _results(_result(watched, "fail"))) == []
    assert report.judge({**weekly, "points": []}, {}) == []
    daily = {**_plan(), "adapters": {}}
    assert report.judge(daily, {}) == ["plan: no points"]
    assert report.judge({**daily, "cadence": "daily"}, {}) == ["plan: no points"]


def test_a_record_line_says_when_its_point_was_only_watched():
    watched = {**_TOP, "key": "watched--top", "kind": "anything", "watching": True}
    plan = _plan(_TOP, watched)
    lines = {
        line["key"]: line
        for line in report.records(plan, _results(_result(_TOP), _result(watched)), _RUN)
    }
    assert lines["watched--top"]["watching"] is True and lines[_TOP["key"]]["watching"] is False
    assert lines["watched--top"]["kind"] == "anything"


# --- the published release -------------------------------------------------------

_SDK = {"origin": "pypi", "version": "0.7.0b1", "ref": "python-v0.7.0b1", "commit": "77b59d9"}


def _published(point: dict, version: str = "0.7.0b1") -> dict:
    """`point` as the plan makes it for the release `version` on PyPI."""
    return {
        **point,
        "key": f"published--{version}--{point['key']}",
        "kind": "published",
        "subject": point["adapter"],
        "watching": True,
        "python": "3.14",
        "sdk": {**_SDK, "version": version, "ref": f"python-v{version}"},
    }


def _loaded(*results: dict) -> dict:
    """Results as `load_results` hands them on: what the plan said kept beside what ran."""
    for result in results:
        result["planned_sdk"] = result.get("sdk", {"origin": "build"})
        result["planned_python"] = result.get("python", "")
    return _results(*results)


def _release_plan(*points: dict, version: str = "0.7.0b1") -> dict:
    plan = _plan(_FLOOR, _TOP, *points)
    plan["published"] = {
        "version": version,
        "ref": f"python-v{version}",
        "commit": "77b59d9",
        "adapters": {"openai_agents": dict(plan["adapters"]["openai_agents"])},
    }
    plan["scope"] = {
        "cadence": "daily",
        "covers": {"framework": ["openai_agents"], "published": ["openai_agents"]},
        "partial": False,
        "builds": {"published": [version]},
    }
    return plan


def test_a_published_points_record_says_which_release_it_ran_even_when_it_ran_nothing(tmp_path):
    ran, never = _published(_TOP), _published(_ABOVE)
    measured = {"wheel": "", "wheel_sha256": "", "version": "0.7.0b1"}
    _write(tmp_path, _result(ran, sdk=measured))
    plan = _release_plan(ran, never)
    results = report.load_results(plan, tmp_path)
    lines = {line["key"]: line for line in report.records(plan, results, _RUN)}
    for key in (ran["key"], never["key"]):
        assert lines[key]["sdk"] == _SDK and lines[key]["kind"] == "published"
        assert lines[key]["watching"] is True
    assert lines[never["key"]]["tests"]["passed"] is None
    assert lines[_TOP["key"]]["sdk"] == {"origin": "build", "version": ""}


def test_a_failing_published_release_is_said_by_version_and_never_turns_the_run_red():
    top, above = _published(_TOP), _published(_ABOVE)
    plan = _release_plan(top, above)
    nodes = ["t.py::test_tree"]
    results = _loaded(
        _result(_FLOOR),
        _result(_TOP),
        _result(top, "fail", failed_ids=nodes),
        _result(above, "fail", failed_ids=nodes),
    )
    assert report.judge(plan, results) == []
    issue = report.adapter_issue("openai_agents", plan, results, _RUN)
    watching = issue["body"].split("### Watching (never turns the run red)")[1]
    assert (
        "- published wardex release `0.7.0b1`: `openai-agents 0.23.1` fails 1 tests. Closes "
        "when it passes, or when a newer wardex release takes its place." in watching
    )
    assert "- published wardex release `0.7.0b1`: `openai-agents 0.26.0` fails 1 tests." in watching
    assert (
        "| newest tested release | `0.7.0b1` from PyPI | `openai-agents 0.23.1` |" in issue["body"]
    )
    assert "| newest tested release | this commit | `openai-agents 0.23.1` |" in issue["body"]
    assert "Nothing." in issue["body"].split("### What needs doing")[1].split("###")[0]


def test_the_points_of_a_release_a_newer_one_replaced_are_kept_and_ask_nothing(tmp_path):
    """They are that release's history, shown on its page. The run that plans the newer
    release forgets nothing of the older one, and no issue asks anyone to fix it."""
    old_top = _published(_TOP, "0.7.0b1")
    new_top = _published(_TOP, "0.7.0b2")
    first = _release_plan(old_top)
    state = report.next_state(
        {},
        _loaded(_result(_FLOOR), _result(_TOP), _result(old_top, "fail")),
        _RUN,
        tmp_path,
        first,
    )
    assert state["release"] == "0.7.0b1" and sorted(state["versions"]) == ["0.7.0b1", "main"]
    assert report.adapter_issue("openai_agents", first, {}, _RUN, state)["open"]

    later = _release_plan(new_top, version="0.7.0b2")
    a_day_on = {**_RUN, "date": "2026-11-27"}
    results = _loaded(_result(_FLOOR), _result(_TOP), _result(new_top))
    state = report.next_state(state, results, a_day_on, tmp_path, later)
    assert old_top["key"] in state["points"] and old_top["key"] not in state["_forget"]
    assert state["release"] == "0.7.0b2"
    assert sorted(state["versions"]) == ["0.7.0b1", "0.7.0b2", "main"]
    assert state["versions"]["0.7.0b1"]["date"] == "2026-11-26", "as its last run left it"
    issue = report.adapter_issue("openai_agents", later, results, a_day_on, state)
    assert not issue["open"] and "`0.7.0b1` from PyPI" not in issue["body"]
    assert "`0.7.0b2` from PyPI" in issue["body"]
    assert [p["key"] for p in render.rows(state, "0.7.0b1")["openai_agents"]] == [old_top["key"]]


def test_a_point_of_the_planned_release_that_is_no_longer_planned_is_forgotten(tmp_path):
    top, above = _published(_TOP), _published(_ABOVE)
    both = _release_plan(top, above)
    results = _loaded(_result(_FLOOR), _result(_TOP), _result(top), _result(above))
    state = report.next_state({}, results, _RUN, tmp_path, both)
    only_top = _release_plan(top)
    del results[above["key"]]
    after = report.next_state(state, results, _RUN, tmp_path, only_top)
    assert after["_forget"] == [above["key"]] and above["key"] not in after["points"]


def test_what_the_pages_say_about_a_build_is_kept_per_build_and_merged_by_adapter(tmp_path):
    plan = _release_plan(_published(_TOP))
    state = report.next_state({}, {}, _RUN, tmp_path, plan)
    main = state["versions"]["main"]
    assert (main["version"], main["commit"], main["date"]) == (
        "0.7.0b1",
        "0123456789abcdef",
        "2026-11-26",
    )
    about = main["adapters"]["openai_agents"]
    assert about == {
        "label": "openai-agents",
        "evidence": "live",
        "replay_of": None,
        "claim": {"openai-agents": ["0.22", "0.23"]},
        "said": None,
        "newest": "0.26.0",
    }
    assert state["versions"]["0.7.0b1"]["ref"] == "python-v0.7.0b1"

    state["versions"]["main"]["adapters"]["langgraph"] = {"label": "langgraph"}
    partial = _plan(_TOP)
    after = report.next_state(state, {}, {**_RUN, "date": "2026-11-27"}, tmp_path, partial)
    assert sorted(after["versions"]["main"]["adapters"]) == ["langgraph", "openai_agents"]
    assert after["release"] == "0.7.0b1", "a run that planned no release keeps the last one"
    assert report.next_state(state, {}, _RUN, tmp_path)["versions"] == state["versions"]


def test_what_a_watching_kind_could_not_plan_is_on_the_health_issue_and_is_not_red():
    plan = _plan(_FLOOR, _TOP)
    plan["unplanned"] = ["published: the newest wardex release could not be read (OSError)"]
    results = _results(_result(_FLOOR), _result(_TOP))
    assert report.judge(plan, results) == []
    issue = report.health_issue(plan, results, {}, _RUN)
    assert issue["open"] and (
        "- Not planned, and so not run (this never turns the run red): `published: the newest "
        "wardex release could not be read (OSError)`" in issue["body"]
    )
    plan["unplanned"] = []
    assert not report.health_issue(plan, results, {}, _RUN)["open"]


def test_a_run_that_writes_renders_the_site_and_one_that_does_not_can_preview_it(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(report.GitHub, "_api", lambda self, method, path, payload=None: [])
    monkeypatch.setattr(report.GitHub, "do", lambda self, what, *rest: None)
    for name in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "GITHUB_REPOSITORY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COMPAT_SDK_VERSION", "0.7.0b1")
    records, results_dir, preview = tmp_path / "records", tmp_path / "results", tmp_path / "preview"
    records.mkdir()
    top = {
        **_published(_TOP),
        **{k: v for k, v in _PLANNED_TOP.items() if k not in _published(_TOP)},
    }
    plan = _release_plan(top)
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    for result in (_result(_FLOOR), _result(_TOP), _result(top)):
        _write(results_dir, result)
    argv = ["--plan", str(tmp_path / "plan.json"), "--results", str(results_dir)]
    argv += ["--state-dir", str(records)]

    assert report.main([*argv, "--preview", str(preview)]) == 0
    assert not (records / "state.json").exists() and not (records / "site").exists()
    wanted = ["0.7.0b1/README.md", "0.7.0b1/openai_agents.svg", "README.md", "main/README.md"]
    made = sorted(
        str(p.relative_to(preview / "site")) for p in (preview / "site").rglob("*") if p.is_file()
    )
    assert made == wanted

    assert report.main([*argv, "--write"]) == 0
    for name in wanted:
        assert (records / "site" / name).read_bytes() == (preview / "site" / name).read_bytes()
    cell = (records / "site/0.7.0b1/openai_agents.svg").read_text(encoding="utf-8")
    assert "0.23.1 passes" in cell and "26 Nov" in cell
    assert "| newest tested release | `openai-agents 0.23.1` |" in (
        records / "site/main/README.md"
    ).read_text(encoding="utf-8")


# --- a proof that stays broken ---------------------------------------------------


def test_a_release_that_fails_for_two_weeks_gets_the_command_that_stops_claiming_it(tmp_path):
    """Nothing narrows a claim on its own. After fourteen days of failing the issue stops
    waiting for a fix in silence: it prints the one command, and says so once."""
    plan = _plan(_FLOOR, _TOP)
    nodes = ["t.py::test_tree"]
    state: dict = {}
    seen = {}
    for day in (10, 15, 23, 24, 25):
        run = {**_RUN, "date": f"2026-11-{day}"}
        outcome = "install_network" if day == 15 else "fail"
        results = _results(_result(_FLOOR), _result(_TOP, outcome, failed_ids=nodes))
        state = report.next_state(state, results, run, tmp_path, plan)
        state.pop("_locks"), state.pop("_forget")
        seen[day] = report.adapter_issue("openai_agents", plan, results, run, state)
    assert state["points"][_TOP["key"]]["failing_since"] == {
        "date": "2026-11-10",
        "targets": {"openai-agents": "0.23.1"},
    }, "a day it could not be checked does not restart the count"
    assert "claim.py" not in seen[23]["body"]
    assert (
        "It has failed since `2026-11-10`, 14 days. To stop calling it tested instead: "
        "`python3 scripts/compat/claim.py bad openai-agents 0.23.1`." in seen[24]["changes"][0]
    )
    assert seen[23]["fingerprint"] != seen[24]["fingerprint"], "said once, as a comment"
    assert seen[24]["fingerprint"] == seen[25]["fingerprint"] and "15 days" in seen[25]["body"]

    # An outage in the middle of a long failure is not news about the lapse: the command
    # stays, and the fingerprint does not move there or back.
    run = {**_RUN, "date": "2026-11-26"}
    outage = _results(_result(_FLOOR), _result(_TOP, "install_network"))
    state = report.next_state(state, outage, run, tmp_path, plan)
    state.pop("_locks"), state.pop("_forget")
    down = report.adapter_issue("openai_agents", plan, outage, run, state)
    assert down["fingerprint"] == seen[25]["fingerprint"]
    assert "could not be checked today" in down["changes"][0]
    assert "claim.py bad openai-agents 0.23.1" in down["changes"][0]

    passing = _results(_result(_FLOOR), _result(_TOP))
    state = report.next_state(state, passing, {**_RUN, "date": "2026-11-26"}, tmp_path, plan)
    assert "failing_since" not in state["points"][_TOP["key"]]


def test_a_new_release_under_the_same_point_starts_its_own_count(tmp_path):
    plan = _plan(_FLOOR, _TOP)
    old = report.next_state(
        {}, _results(_result(_TOP, "fail")), {**_RUN, "date": "2026-11-01"}, tmp_path, plan
    )
    newer = {**_TOP, "targets": {"openai-agents": "0.23.2"}}
    state = report.next_state(
        old, _results(_result(newer, "fail")), {**_RUN, "date": "2026-11-20"}, tmp_path, plan
    )
    assert state["points"][_TOP["key"]]["failing_since"]["date"] == "2026-11-20"
    issue = report.adapter_issue(
        "openai_agents", _plan(_FLOOR, newer), _results(_result(newer, "fail")), _RUN, state
    )
    assert "claim.py" not in issue["body"]


def test_what_could_not_be_planned_changes_the_health_issue_once_and_a_claim_is_printed_whole(
    tmp_path,
):
    plan = _plan(_FLOOR, _TOP)
    results = _results(_result(_FLOOR), _result(_TOP))
    quiet = report.health_issue(plan, results, {}, _RUN)["fingerprint"]
    plan["unplanned"] = ["published: the newest wardex release could not be read (OSError)"]
    assert report.health_issue(plan, results, {}, _RUN)["fingerprint"] != quiet

    plan["claims"] = [
        {
            "says": "Verified against `codex-cli` 0.160.0; a | b <i> [x](y)",
            "proof": None,
            "why": "w",
        }
    ]
    kept = report.next_state({}, results, _RUN, tmp_path, plan)["versions"]["main"]["claims"]
    assert kept == [
        {"says": "Verified against codex-cli 0.160.0; a  b i x(y)", "proof": "", "why": "w"}
    ]
    page = render.page({"versions": {"main": {"claims": kept}}}, render.MAIN, "2026-11-26")
    assert "| Verified against codex-cli 0.160.0; a  b i x(y) | nothing yet: w |" in page


def test_a_red_capable_question_is_not_answered_by_watching_points():
    """However many watching points a plan holds, a run that was due to ask whether the
    built wheel holds and planned no point for it is red."""
    watched = {**_TOP, "key": "watched--top", "kind": "anything", "watching": True}
    plan = {**_plan(watched), "adapters": {}}
    assert report.judge(plan, _results(_result(watched))) == ["plan: no points"]


def test_the_way_out_names_every_release_a_point_pins(tmp_path):
    """A framework and its companion are pinned together. Which of the two broke is for a
    person to say; a command for the first alone would list the wrong release."""
    pair = {**_TOP, "targets": {"langgraph": "1.2.24", "langgraph-prebuilt": "1.1.3"}}
    plan = _plan(_FLOOR, pair)
    state = report.next_state(
        {}, _results(_result(pair, "fail")), {**_RUN, "date": "2026-11-01"}, tmp_path, plan
    )
    state.pop("_locks"), state.pop("_forget")
    run = {**_RUN, "date": "2026-11-20"}
    results = _results(_result(_FLOOR), _result(pair, "fail"))
    issue = report.adapter_issue("openai_agents", plan, results, run, state)
    assert (
        "To stop calling it tested instead: for the release that broke, "
        "`python3 scripts/compat/claim.py bad langgraph 1.2.24` or "
        "`python3 scripts/compat/claim.py bad langgraph-prebuilt 1.1.3`." in issue["changes"][0]
    )


def test_how_long_a_release_has_failed_goes_with_the_release_when_a_newer_one_replaces_it(
    tmp_path,
):
    """It failed as the newest tested release for ten days and then as its own point. The
    fourteen days are counted from the first failure, not from the day its key changed."""
    old = {**_point("claim_top", "0.23.1"), "python": "3.14"}
    new = {**_point("claim_top", "0.23.2"), "python": "3.14"}
    covering = {"cadence": "daily", "covers": {"framework": ["openai_agents"]}, "partial": False}
    state: dict = {}
    issue = {}
    for day in range(1, 16):
        run = {**_RUN, "date": f"2026-11-{day:02d}"}
        points = [_FLOOR, old if day <= 10 else new]
        for item in (state.get("displaced") or {}).get("openai_agents", []):
            points.append({**_point("displaced", "0.23.1"), "key": item["key"], "python": "3.14"})
        plan = _plan_with_table(*points)
        plan["scope"] = covering
        results = {}
        for planned in points:
            failing = planned["targets"] == old["targets"]
            result = _result(planned, "fail" if failing else "pass")
            result["planned_python"] = planned.get("python", "3.14")
            results[planned["key"]] = result
        state = report.next_state(state, results, run, tmp_path, plan)
        state.pop("_locks"), state.pop("_forget")
        issue = report.adapter_issue("openai_agents", plan, results, run, state)
    (item,) = state["displaced"]["openai_agents"]
    release = report.release_id("openai_agents", item["targets"], "3.14")
    assert state["failing"] == {release: "2026-11-01"}
    assert state["points"][item["key"]]["failing_since"]["date"] == "2026-11-01"
    assert "It has failed since `2026-11-01`, 14 days." in "\n".join(issue["changes"])
    assert "claim.py bad openai-agents 0.23.1" in "\n".join(issue["changes"])


def test_what_the_pages_know_of_a_release_survives_runs_of_one_adapter(tmp_path):
    plan = _release_plan(_published(_TOP))
    state = report.next_state({}, {}, _RUN, tmp_path, plan)
    state["versions"]["0.7.0b1"]["adapters"]["langgraph"] = {"label": "langgraph"}
    after = report.next_state(state, {}, {**_RUN, "date": "2026-11-27"}, tmp_path, plan)
    assert sorted(after["versions"]["0.7.0b1"]["adapters"]) == ["langgraph", "openai_agents"]


def test_a_release_failing_on_one_wardex_build_starts_no_count_on_another(tmp_path):
    """The published release has failed on this framework release for weeks. The day the
    default branch first fails on it, on the same interpreter, is day one for the default
    branch; and the other way round."""
    top = {**_TOP, "python": "3.14"}
    held = {**_published(top), "python": "3.14"}
    plan = _release_plan(held)

    def day(state: dict, date: str, ours: str, theirs: str) -> dict:
        results = _loaded(_result(_FLOOR), _result(top, ours), _result(held, theirs))
        after = report.next_state(state, results, {**_RUN, "date": date}, tmp_path, plan)
        after.pop("_locks"), after.pop("_forget")
        return after

    state = day({}, "2026-11-01", "pass", "fail")
    state = day(state, "2026-11-20", "fail", "fail")
    assert state["points"][top["key"]]["point"]["python"] == "3.14"
    assert state["points"][held["key"]]["point"]["python"] == "3.14"
    assert state["points"][top["key"]]["failing_since"]["date"] == "2026-11-20"
    assert state["points"][held["key"]]["failing_since"]["date"] == "2026-11-01"
    run = {**_RUN, "date": "2026-11-20"}
    results = _loaded(_result(_FLOOR), _result(top, "fail"), _result(held, "fail"))
    issue = report.adapter_issue("openai_agents", plan, results, run, state)
    assert "claim.py" not in issue["body"]

    state = day({}, "2026-11-01", "fail", "pass")
    state = day(state, "2026-11-20", "fail", "fail")
    assert state["points"][held["key"]]["failing_since"]["date"] == "2026-11-20"
    assert state["points"][top["key"]]["failing_since"]["date"] == "2026-11-01"


def test_a_replaced_release_whose_own_point_cannot_run_yet_is_no_news(tmp_path):
    """It was remembered yesterday. Today its own point is planned for the first time and
    the index is down: the same thing is known as yesterday, so nobody is told again."""
    new = {**_point("claim_top", "0.23.2"), "python": "3.14"}
    plan = _plan_with_table(_FLOOR, new)
    results = _results(_result(_FLOOR), _result(new))
    for result in results.values():
        result["planned_python"] = "3.14"
    state = report.next_state(_top_failed_yesterday(), results, _RUN, tmp_path, plan)
    state.pop("_locks"), state.pop("_forget")
    remembered = report.adapter_issue("openai_agents", plan, results, _RUN, state)
    (item,) = state["displaced"]["openai_agents"]
    own = {**_point("displaced", "0.23.1"), "key": item["key"], "python": "3.14"}
    plan = _plan_with_table(_FLOOR, new, own)
    run = {**_RUN, "date": "2026-11-27"}
    results = _results(_result(_FLOOR), _result(new), _result(own, "install_network"))
    for result in results.values():
        result["planned_python"] = "3.14"
    state = report.next_state(state, results, run, tmp_path, plan)
    waiting = report.adapter_issue("openai_agents", plan, results, run, state)
    assert waiting["open"] and waiting["fingerprint"] == remembered["fingerprint"]


def test_a_failure_of_the_published_release_is_never_a_replaced_failure_of_this_build(tmp_path):
    """The published wardex fails on the newest tested release while the default branch
    passes it. When a newer patch arrives, nothing of the default branch is owed a run:
    the list of replaced failures is about the wheel this run built."""
    held = {**_published(_TOP), "python": "3.14"}
    covering = {
        "cadence": "daily",
        "covers": {"framework": ["openai_agents"], "published": ["openai_agents"]},
        "partial": False,
        "builds": {"published": ["0.7.0b1"]},
    }
    first = _plan_with_table(_FLOOR, {**_TOP, "python": "3.14"}, held)
    first["scope"] = covering
    results = _loaded(_result(_FLOOR), _result(_TOP), _result(held, "fail"))
    state = report.next_state({}, results, {**_RUN, "date": "2026-11-25"}, tmp_path, first)
    state.pop("_locks"), state.pop("_forget")

    new = {**_point("claim_top", "0.23.2"), "python": "3.14"}
    held_new = {**_published(new), "key": held["key"]}
    second = _plan_with_table(_FLOOR, new, held_new)
    second["scope"] = covering
    results = _loaded(_result(_FLOOR), _result(new), _result(held_new))
    state = report.next_state(state, results, _RUN, tmp_path, second)
    assert state["displaced"] == {}
    issue = report.adapter_issue("openai_agents", second, results, _RUN, state)
    assert not issue["open"]


def test_a_pass_of_the_published_release_settles_nothing_of_this_build(tmp_path):
    item = {**_remembered_item(), "key": "openai_agents--displaced--0.23.1-py3.14"}
    state = {"displaced": {"openai_agents": [item]}}
    own = {**_point("displaced", "0.23.1"), "key": item["key"], "python": "3.14"}
    held = {**_published(_TOP), "python": "3.14"}
    plan = _plan_with_table(_FLOOR, own, held)
    results = _loaded(_result(_FLOOR), _result(own, "install_network"), _result(held))
    for result in results.values():
        result["planned_python"] = "3.14"
    after = report.next_state(state, results, _RUN, tmp_path, plan)
    assert after["displaced"]["openai_agents"] == [item], "still owed a run of its own"


def test_the_way_out_stays_on_the_issue_while_a_release_changes_hands(tmp_path):
    """It has failed for more than two weeks and the command is on its issue. The day a
    newer patch replaces it, and the day its own point cannot install it, the command is
    still there, with the count from the first failure."""
    old = {**_point("claim_top", "0.23.1"), "python": "3.14"}
    new = {**_point("claim_top", "0.23.2"), "python": "3.14"}
    state: dict = {}
    said = {}
    for day in range(1, 23):
        run = {**_RUN, "date": f"2026-11-{day:02d}"}
        points = [_FLOOR, old if day < 20 else new]
        for item in (state.get("displaced") or {}).get("openai_agents", []):
            points.append({**_point("displaced", "0.23.1"), "key": item["key"], "python": "3.14"})
        plan = _plan_with_table(*points)
        results = {}
        for planned in points:
            outcome = "pass"
            if planned["targets"] == old["targets"]:
                outcome = "install_network" if day == 21 else "fail"
            result = _result(planned, outcome)
            result["planned_python"] = planned.get("python", "3.14")
            results[planned["key"]] = result
        state = report.next_state(state, results, run, tmp_path, plan)
        state.pop("_locks"), state.pop("_forget")
        said[day] = report.adapter_issue("openai_agents", plan, results, run, state)
    assert "claim.py" not in said[14]["body"]
    for day in range(15, 23):
        (line,) = [c for c in said[day]["changes"] if "0.23.1" in c]
        days = day - 1
        assert f"It has failed since `2026-11-01`, {days} days." in line, day
        assert "claim.py bad openai-agents 0.23.1" in line, day
    assert "No rule picks it any more" in said[20]["changes"][0], "the day it was replaced"
    assert "could not be checked today" in said[21]["changes"][0], "its own point's first day"
    assert said[20]["fingerprint"] == said[21]["fingerprint"]


@pytest.mark.parametrize(
    "other",
    [
        {"watching": True, "sdk": {"origin": "pypi", "version": "0.7.0b1"}},
        {"watching": False, "sdk": {"origin": "pypi", "version": "0.7.0b1"}},
        {"watching": True, "sdk": {"origin": "build"}},
    ],
)
def test_only_a_red_capable_point_of_this_build_makes_or_speaks_for_a_replaced_failure(
    tmp_path, other
):
    """Another wardex build, or a kind that is only watched, running the same release on
    the same interpreter: its failure starts nothing here, and its pass does not stand in
    for the run the replaced release is owed."""
    new = {**_point("claim_top", "0.23.2"), "python": "3.14"}
    beside = {**_TOP, "key": "beside--top", "python": "3.14", "kind": "anything", **other}
    plan = _plan_with_table(_FLOOR, new, beside)

    failed = _loaded(_result(_FLOOR), _result(new), _result(beside, "fail"))
    before = {"points": {"beside--top": _failed_yesterday(_TOP, [])["points"][_TOP["key"]]}}
    before["points"]["beside--top"]["last_checked"]["targets"] = {"openai-agents": "0.23.0"}
    after = report.next_state(before, failed, _RUN, tmp_path, plan)
    assert after["displaced"] == {}, "its failure on a release it left is not this build's"

    item = {**_remembered_item(), "key": "openai_agents--displaced--0.23.1-py3.14"}
    state = {"displaced": {"openai_agents": [item]}}
    passed = _loaded(_result(_FLOOR), _result(new), _result(beside))
    issue = report.adapter_issue("openai_agents", plan, passed, _RUN, state)
    assert issue["open"] and "No rule picks it any more" in issue["changes"][0]


def _one_day(state: dict, date: str, tmp_path, *outcomes: tuple) -> dict:
    """One covered day of `(point, outcome)` pairs through the state, as the report does."""
    points = [planned for planned, _ in outcomes]
    plan = _plan_with_table(*points)
    plan["scope"] = {
        "cadence": "daily",
        "covers": {"framework": ["openai_agents"]},
        "partial": False,
    }
    results = {}
    for planned, outcome in outcomes:
        result = _result(planned, outcome)
        result["planned_python"] = planned["python"]
        results[planned["key"]] = result
    after = report.next_state(state, results, {**_RUN, "date": date}, tmp_path, plan)
    after.pop("_locks"), after.pop("_forget")
    return after


@pytest.mark.parametrize("dead_day", ["fail", "install_network", "no_result"])
@pytest.mark.parametrize("taker", ["claim_top", "minor_newest"])
def test_a_count_follows_its_release_to_whichever_point_of_this_build_runs_it_next(
    tmp_path, taker, dead_day
):
    """The patch that replaced a failing release is withdrawn, or the range grows, and a
    rule's own point runs the old release again. It has been failing since the first day,
    whoever holds it now, also when the day it changed hands left no verdict."""
    floor = {**_FLOOR, "python": "3.10"}
    first = {**_point("minor_newest", "0.22.3"), "python": "3.14"}
    state = _one_day({}, "2026-11-01", tmp_path, (floor, "pass"), (first, "fail"))
    again = {**_point(taker, "0.22.3"), "key": f"openai_agents--{taker}--x", "python": "3.14"}
    state = _one_day(state, "2026-11-20", tmp_path, (floor, "pass"), (again, dead_day))
    assert first["key"] not in state["points"], "the first point is gone, the count is not"
    state = _one_day(state, "2026-11-21", tmp_path, (floor, "pass"), (again, "fail"))
    assert state["points"][again["key"]]["failing_since"]["date"] == "2026-11-01"

    state = _one_day(state, "2026-11-22", tmp_path, (floor, "pass"), (again, "pass"))
    assert state["failing"] == {} and "failing_since" not in state["points"][again["key"]]
    state = _one_day(state, "2026-11-23", tmp_path, (floor, "pass"), (again, "fail"))
    assert state["points"][again["key"]]["failing_since"]["date"] == "2026-11-23"


def test_a_count_is_by_release_and_interpreter_and_ends_when_nobody_asks_any_more(tmp_path):
    top = {**_point("claim_top", "0.23.1"), "python": "3.14"}
    oldpy = {**_point("claim_top_oldpy", "0.23.1"), "python": "3.10"}
    state = _one_day({}, "2026-10-20", tmp_path, (top, "pass"), (oldpy, "fail"))
    state = _one_day(state, "2026-11-03", tmp_path, (top, "fail"), (oldpy, "fail"))
    assert state["points"][top["key"]]["failing_since"]["date"] == "2026-11-03"
    assert state["points"][oldpy["key"]]["failing_since"]["date"] == "2026-10-20"

    newer = {**_point("claim_top", "0.23.2"), "python": "3.14"}
    state = _one_day(state, "2026-11-05", tmp_path, (newer, "pass"))
    on = {
        py: report.release_id("openai_agents", {"openai-agents": "0.23.1"}, py)
        for py in ("3.14", "3.10")
    }
    assert state["failing"] == {on["3.14"]: "2026-11-03", on["3.10"]: "2026-10-20"}, (
        "replaced on 3.14, and its point on 3.10 no longer planned: both are still owed a run"
    )
    assert sorted(item["python"] for item in state["displaced"]["openai_agents"]) == [
        "3.10",
        "3.14",
    ]

    listed = {**_ENTRY, "known_bad": ["0.23.1"]}
    plan = _plan_with_table(newer, entry=listed)
    plan["scope"] = {
        "cadence": "daily",
        "covers": {"framework": ["openai_agents"]},
        "partial": False,
    }
    result = _result(newer)
    result["planned_python"] = "3.14"
    settled = report.next_state(
        state, {newer["key"]: result}, {**_RUN, "date": "2026-11-06"}, tmp_path, plan
    )
    assert settled["failing"] == {} and settled["displaced"] == {}, "listed as bad: nobody asks"

    uncovered = _plan_with_table()
    uncovered["adapters"]["openai_agents"]["problems"] = ["the index did not answer"]
    kept = report.next_state(state, {}, {**_RUN, "date": "2026-11-06"}, tmp_path, uncovered)
    assert kept["failing"] == state["failing"], "a run that did not cover it forgets nothing"


def test_the_floors_count_and_the_count_with_todays_packages_are_kept_apart(tmp_path):
    """A range of one release: the floor and the oldest-interpreter point run the same
    release on the same interpreter, one with the packages of its own time and one with
    today's. Each failure has its own first day."""
    floor = {**_point("floor", "0.23.1"), "python": "3.10"}
    oldpy = {**_point("claim_top_oldpy", "0.23.1"), "python": "3.10"}
    plan = _plan(floor, oldpy)

    def day(state: dict, date: str, at_floor: str, with_todays: str) -> dict:
        results = _results(_result(floor, at_floor), _result(oldpy, with_todays))
        for result in results.values():
            result["planned_python"] = "3.10"
        after = report.next_state(state, results, {**_RUN, "date": date}, tmp_path, plan)
        after.pop("_locks"), after.pop("_forget")
        return after

    state = day(day({}, "2026-11-01", "fail", "pass"), "2026-11-20", "fail", "fail")
    assert state["points"][oldpy["key"]]["failing_since"]["date"] == "2026-11-20"
    state = day(day({}, "2026-11-01", "pass", "fail"), "2026-11-20", "fail", "fail")
    assert state["points"][floor["key"]]["failing_since"]["date"] == "2026-11-20"


def test_a_pass_beside_a_failure_of_the_same_release_ends_no_count(tmp_path):
    """Two points on one release and interpreter on one day, one passing and one failing:
    it is still failing, whichever the report happens to read first."""
    one = {**_point("claim_top", "0.23.1"), "python": "3.14"}
    two = {**_point("minor_newest", "0.23.1"), "python": "3.14"}
    state = _one_day({}, "2026-11-03", tmp_path, (one, "fail"), (two, "fail"))
    for order in ((one, "pass"), (two, "fail")), ((two, "fail"), (one, "pass")):
        after = _one_day(state, "2026-11-04", tmp_path, *order)
        assert list(after["failing"].values()) == ["2026-11-03"]
    after = _one_day(state, "2026-11-04", tmp_path, (one, "pass"), (two, "install_network"))
    assert after["failing"] == {}, "a pass and nothing against it ends the count"


def test_only_a_release_inside_the_range_is_counted_and_an_uncovered_day_keeps_every_count(
    tmp_path,
):
    top = {**_point("claim_top", "0.23.1"), "python": "3.14"}
    above = {**_ABOVE, "python": "3.14"}
    state = _one_day({}, "2026-11-01", tmp_path, (top, "fail"), (above, "fail"))
    counted = report.release_id("openai_agents", top["targets"], "3.14")
    assert state["failing"] == {counted: "2026-11-01"}, "above the range nothing is claimed"
    assert "failing_since" not in state["points"][above["key"]] or (
        state["points"][above["key"]]["failing_since"]["date"] == "2026-11-01"
    )

    uncovered = _plan_with_table()
    uncovered["adapters"]["openai_agents"]["problems"] = ["the index did not answer"]
    kept = report.next_state(state, {}, {**_RUN, "date": "2026-11-02"}, tmp_path, uncovered)
    assert kept["failing"] == {counted: "2026-11-01"} and kept["displaced"] == {}


def test_a_failing_release_whose_point_leaves_the_plan_is_still_owed_a_run(tmp_path):
    """The rules moved: the floor now sits on the line this release was the newest patch
    of, so no point runs it with today's packages any more. It failed when it was last
    checked and the range still claims it, so it is remembered like a release a newer
    patch replaced. Not when the index withdrew it, and not when a point checked it today."""
    floor = {**_FLOOR, "python": "3.10"}
    patch = {**_point("minor_newest", "0.22.3"), "python": "3.14"}
    top = {**_TOP, "python": "3.14"}
    state = _one_day({}, "2026-11-01", tmp_path, (floor, "pass"), (patch, "fail"), (top, "pass"))

    def without_it(served: list[str], *more: tuple) -> dict:
        plan = _plan_with_table(floor, top, *[planned for planned, _ in more])
        plan["adapters"]["openai_agents"]["served"] = {"openai-agents": served}
        plan["scope"] = {
            "cadence": "daily",
            "covers": {"framework": ["openai_agents"]},
            "partial": False,
        }
        results = {}
        for planned, outcome in ((floor, "pass"), (top, "pass"), *more):
            result = _result(planned, outcome)
            result["planned_python"] = planned["python"]
            results[planned["key"]] = result
        after = report.next_state(state, results, {**_RUN, "date": "2026-11-02"}, tmp_path, plan)
        return {
            "state": after,
            "issue": report.adapter_issue(
                "openai_agents", plan, results, {**_RUN, "date": "2026-11-02"}, after
            ),
        }

    owed = without_it(["0.22.0", "0.22.3", "0.23.1"])
    (item,) = owed["state"]["displaced"]["openai_agents"]
    assert (item["targets"], item["python"], item["failed_on"]) == (
        {"openai-agents": "0.22.3"},
        "3.14",
        "2026-11-01",
    )
    assert patch["key"] not in owed["state"]["points"]
    assert owed["issue"]["open"] and "No rule picks it any more" in owed["issue"]["changes"][0]
    assert list(owed["state"]["failing"].values()) == ["2026-11-01"]

    withdrawn = without_it(["0.22.0", "0.23.1"])
    assert withdrawn["state"]["displaced"] == {} and not withdrawn["issue"]["open"]
    assert without_it([])["state"]["displaced"] == {}, "nothing of the distribution is served"

    again = {**_point("claim_top_oldpy", "0.22.3"), "key": "openai_agents--other", "python": "3.14"}
    for outcome, remembered in (("pass", False), ("fail", False), ("install_network", True)):
        seen = without_it(["0.22.0", "0.22.3", "0.23.1"], (again, outcome))
        assert bool(seen["state"]["displaced"]) is remembered, outcome
        assert seen["issue"]["open"] is (outcome != "pass"), outcome


def test_a_count_is_not_kept_alive_by_a_point_that_does_not_count(tmp_path):
    """A release left the range and passes above it; later the range takes it back and it
    fails. That is day one, not the day it first failed before it left."""
    inside = {**_point("claim_top", "0.24.0"), "python": "3.14"}
    state = _one_day({}, "2026-11-01", tmp_path, (inside, "fail"))
    above = {**_point("beyond", "0.24.0", line="0.24"), "python": "3.14"}
    top = {**_TOP, "python": "3.14"}
    state = _one_day(state, "2026-11-11", tmp_path, (top, "pass"), (above, "pass"))
    assert state["failing"] == {}
    state = _one_day(state, "2026-11-21", tmp_path, (inside, "fail"))
    assert list(state["failing"].values()) == ["2026-11-21"]


def test_a_pair_whose_companion_release_was_withdrawn_is_owed_nothing(tmp_path):
    """The framework is pinned with a companion. The companion's release is withdrawn, so
    the pair that failed cannot be installed any more: no point can be made for it, and
    remembering it would put a release nobody can run on the issue for a day."""
    pair = {
        **_TOP,
        "python": "3.14",
        "targets": {"langgraph": "1.2.1", "langgraph-prebuilt": "1.1.2"},
    }
    state = _one_day({}, "2026-11-01", tmp_path, (pair, "fail"))
    moved = {**pair, "targets": {"langgraph": "1.2.1", "langgraph-prebuilt": "1.1.1"}}
    rows = [
        {"distribution": "langgraph", "minors": ["1.2"], "known_bad": [], "since": ""},
        {"distribution": "langgraph-prebuilt", "minors": ["1.1"], "known_bad": [], "since": ""},
    ]

    def day(served: dict) -> dict:
        plan = _plan(moved)
        plan["adapters"]["openai_agents"].update(entries=rows, served=served)
        plan["scope"] = {
            "cadence": "daily",
            "covers": {"framework": ["openai_agents"]},
            "partial": False,
        }
        result = _result(moved, "fail")
        result["planned_python"] = "3.14"
        return report.next_state(
            state, {moved["key"]: result}, {**_RUN, "date": "2026-11-02"}, tmp_path, plan
        )

    gone = day({"langgraph": ["1.2.1"], "langgraph-prebuilt": ["1.1.1"]})
    assert gone["displaced"] == {} and len(gone["failing"]) == 1, "only the pair that runs today"
    still = day({"langgraph": ["1.2.1"], "langgraph-prebuilt": ["1.1.1", "1.1.2"]})
    assert [item["targets"] for item in still["displaced"]["openai_agents"]] == [pair["targets"]]


@pytest.mark.parametrize(
    "left",
    [
        {**_point("floor", "0.22.3"), "key": "left--floor"},
        {**_point("displaced", "0.22.3"), "key": "left--displaced"},
        {**_point("minor_newest", "0.22.3"), "key": "left--watched", "watching": True},
        {**_point("minor_newest", "0.22.3"), "key": "left--pypi", "sdk": {"origin": "pypi"}},
        {**_point("beyond", "0.22.3"), "key": "left--beyond"},
    ],
)
def test_not_every_failing_point_that_leaves_the_plan_is_owed_a_run(tmp_path, left):
    """The floor ran the release with another time's packages; a replaced release's own
    point leaves the plan only when the plan stopped claiming it; a watched point, another
    wardex build and a release above the range were never this build's promise."""
    left = {**left, "python": "3.14"}
    state = _one_day({}, "2026-11-01", tmp_path, (left, "fail"))
    state["points"][left["key"]]["point"].update(
        watching=left.get("watching", False), sdk=left.get("sdk", {"origin": "build"})
    )
    top = {**_TOP, "python": "3.14"}
    after = _one_day(state, "2026-11-02", tmp_path, (top, "pass"))
    assert after["displaced"] == {} and after["failing"] == {}


def test_a_schedule_that_never_fired_is_dated_from_the_day_its_file_reached_the_default_branch(
    monkeypatch,
):
    """GitHub registers a workflow when it first sees the file on any branch. A rehearsal
    on a branch days before the merge must not count as days the schedule stayed silent."""
    answers = {
        "runs": {"workflow_runs": []},
        "registered": "2026-11-26T03:17:24.000+09:00",
        "commits": [
            {"commit": {"committer": {"date": "2026-11-27T20:45:01Z"}}},
            {"commit": {"committer": {"date": "2026-11-27T17:46:37Z"}}},
        ],
    }

    def get(repo: str, path: str):
        if path.startswith("actions/workflows/compat.yml/runs"):
            return answers["runs"]
        if path.startswith("commits?path=.github/workflows/compat.yml"):
            assert path.endswith("&per_page=100"), "a short page would end at a newer commit"
            return answers["commits"]
        assert path == "actions/workflows/compat.yml"
        return {"created_at": answers["registered"]}

    monkeypatch.setattr(watchdog, "_get", get)
    since, ever = watchdog.last_scheduled("o/r")
    assert (since, ever) == ("2026-11-27T17:46:37Z", False)
    assert not watchdog.wanted(since, _at("2026-11-28"), ever=False)["open"]
    assert watchdog.wanted(since, _at("2026-11-30"), ever=False)["open"]

    answers["commits"] = []
    assert watchdog.last_scheduled("o/r") == ("2026-11-25T18:17:24Z", False), "said in UTC"
    answers["registered"] = "2026-11-29T00:00:00Z"
    answers["commits"] = [{"commit": {"committer": {"date": "2026-11-27T17:46:37Z"}}}]
    assert watchdog.last_scheduled("o/r")[0] == "2026-11-29T00:00:00Z", "the later of the two"
    answers["runs"] = {"workflow_runs": [{"created_at": "2026-11-28T04:00:00Z"}]}
    assert watchdog.last_scheduled("o/r") == ("2026-11-28T04:00:00Z", True)


# --- the release gate, rehearsed by every run ------------------------------------


def _rehearsed(*results: dict, ref: str = "refs/heads/main", scope: str = "all") -> str:
    run = {**_RUN, "ref": ref, "scope": scope, "cadence": "daily", "red": False}
    run["points"] = len(results)
    plan = _plan(*[{k: v for k, v in r.items() if k in _FLOOR} for r in results])
    return report.gate_rehearsal(plan, _results(*results), run)


def test_every_run_says_what_the_release_gate_would_say_of_its_commit(monkeypatch):
    monkeypatch.setattr(report.gate, "claimed", lambda root: {"openai_agents": 2})
    assert "The release gate would pass" in _rehearsed(_result(_FLOOR), _result(_TOP))
    stopped = _rehearsed(_result(_FLOOR, "fail"), _result(_TOP))
    assert "The release gate would stop:" in stopped
    assert (
        "- `openai_agents--floor`: `openai-agents 0.22.0` is inside the tested range and: tests "
        "fail\n" in stopped
    )
    for elsewhere in (
        _rehearsed(_result(_FLOOR), _result(_TOP), ref="refs/heads/some-branch"),
        _rehearsed(_result(_FLOOR), _result(_TOP), scope="partial"),
    ):
        assert "This run cannot say" in elsewhere and "would" not in elsewhere


def test_a_rehearsal_that_breaks_costs_the_run_nothing(monkeypatch):
    def broken(root):
        raise ValueError("no table")

    monkeypatch.setattr(report.gate, "claimed", broken)
    assert "could not be rehearsed: ValueError" in _rehearsed(_result(_FLOOR), _result(_TOP))


def test_the_page_of_a_release_that_overrode_its_gate_says_so(tmp_path):
    plan = _release_plan(_published(_TOP))
    plan["published"]["override"] = {"version": "0.7.0b1", "reason": "the floor fails upstream"}
    state = report.next_state({}, {}, _RUN, tmp_path, plan)
    page = render.page(state, "0.7.0b1", "2026-11-26")
    assert (
        "**This release carried an override of the compatibility gate**, which lets a release "
        "go out without a recent pass for what it says it is tested on. The reason given in "
        "its own tree: `the floor fails upstream`\n" in page
    )
    assert "override of the compatibility gate" not in render.page(state, render.MAIN, "2026-11-26")
    # In the words it was given, in any language, and never as markup: the state file is
    # read back from a branch, so the page cleans the reason again and quotes it.
    for reason, shown in (
        ("상위 프레임워크 버그로 측정 불가", "`상위 프레임워크 버그로 측정 불가`"),
        ("a` | <b> [x](https://e.invalid) **PASS**", "`a b x (https://e.invalid) **PASS**`"),
    ):
        plan["published"]["override"] = {"version": "0.7.0b1", "reason": reason}
        state = report.next_state(state, {}, _RUN, tmp_path, plan)
        assert f"its own tree: {shown}\n" in render.page(state, "0.7.0b1", "2026-11-26")
    # A state file read back from the branch may hold anything where the override was.
    for wrong in (
        "yes",
        ["r"],
        7,
        {},
        {"reason": ""},
        {"reason": "  "},
        {"reason": "``"},
        {"reason": 3},
    ):
        state["versions"]["0.7.0b1"]["override"] = wrong
        assert "override of the" not in render.page(state, "0.7.0b1", "2026-11-26")
    plan["published"]["override"] = None
    state = report.next_state(state, {}, _RUN, tmp_path, plan)
    assert "override of the compatibility gate" not in render.page(state, "0.7.0b1", "2026-11-26")


def test_a_run_writes_its_colour_on_every_line_it_records(tmp_path, monkeypatch, capsys):
    """The release gate reads lines and nothing else. A run that was red because its plan
    could not cover a tested release has no line that says so, unless every line does."""
    monkeypatch.setattr(report.GitHub, "_api", lambda self, method, path, payload=None: [])
    monkeypatch.setattr(report.GitHub, "do", lambda self, what, *rest: None)
    for name in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "GITHUB_REPOSITORY"):
        monkeypatch.delenv(name, raising=False)

    def lines_of(plan: dict, where: str) -> tuple[int, list[dict]]:
        records, results_dir = tmp_path / where / "records", tmp_path / where / "results"
        records.mkdir(parents=True)
        (tmp_path / where / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        _write(results_dir, _result(_FLOOR))
        _write(results_dir, _result(_TOP))
        argv = ["--plan", str(tmp_path / where / "plan.json"), "--results", str(results_dir)]
        code = report.main([*argv, "--state-dir", str(records), "--write"])
        (day,) = (records / "records").rglob("*.jsonl")
        return code, [json.loads(line) for line in day.read_text(encoding="utf-8").splitlines()]

    code, green = lines_of(_plan(_FLOOR, _TOP), "green")
    assert code == 0 and [line["run"]["red"] for line in green] == [False, False]
    assert [line["run"]["points"] for line in green] == [2, 2], "and how many lines it wrote"
    assert "### If this commit were released now" in capsys.readouterr().out, "in every summary"
    problem = _plan(_FLOOR, _TOP)
    problem["adapters"]["openai_agents"]["problems"] = ["0.22.0 admits no supported interpreter"]
    code, red = lines_of(problem, "red")
    assert code == report.RED_EXIT and [line["run"]["red"] for line in red] == [True, True]
    assert all(line["outcome"] == "pass" for line in red), "no line of its own says why"


def test_the_gate_reads_what_the_report_wrote_and_a_run_with_no_point_is_still_a_run(
    tmp_path, monkeypatch
):
    """The report and the gate, end to end, on one records folder. A whole run on the
    default branch passes the gate. A later run at the same commit that could plan no
    point at all writes no line; the list of runs is what makes it the newest run."""
    monkeypatch.setattr(report.GitHub, "_api", lambda self, method, path, payload=None: [])
    monkeypatch.setattr(report.GitHub, "do", lambda self, what, *rest: None)
    for name in ("GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "GITHUB_REPOSITORY"):
        monkeypatch.delenv(name, raising=False)
    commit = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_SHA", commit)
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    records = tmp_path / "records"
    records.mkdir()

    def run(plan: dict, run_id: str, *results: dict) -> int:
        monkeypatch.setenv("GITHUB_RUN_ID", run_id)
        work = tmp_path / f"{run_id}-{len(list(tmp_path.iterdir()))}"
        (work / "results").mkdir(parents=True, exist_ok=True)
        (work / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
        for result in results:
            _write(work / "results", result)
        argv = ["--plan", str(work / "plan.json"), "--results", str(work / "results")]
        return report.main([*argv, "--state-dir", str(records), "--write"])

    def decided() -> dict:
        return gate.decide(
            gate.read_lines(records, "2026-11-26"),
            commit=commit,
            today="2026-11-26",
            claims={"openai_agents": 2},
            runs=gate.read_runs(records),
        )

    # A watching point that fails is one of the run's lines, and stops nothing.
    watched = _published(_TOP)
    morning = {**_release_plan(watched), "partial": False}
    results = (_result(_FLOOR), _result(_TOP), _result(watched, "fail"))
    assert run(morning, "1", *results) == 0
    listed = gate.read_runs(records)
    assert [(e["id"], e["attempt"], e["red"], e["points"]) for e in listed] == [
        ("1", "1", False, 3)
    ]
    assert listed[0]["scope"] == "all" and listed[0]["ref"] == "refs/heads/main"
    assert decided()["pass"] is True

    # Later the same day the index lists nothing: no point, a red run, not one line.
    evening = {**_plan(), "partial": False, "cutoff": "2026-11-26T21:00:00Z"}
    evening["adapters"] = {"openai_agents": {**morning["adapters"]["openai_agents"]}}
    evening["adapters"]["openai_agents"]["problems"] = ["no release could be listed"]
    assert run(evening, "2") == report.RED_EXIT
    assert [(e["id"], e["red"], e["points"]) for e in gate.read_runs(records)] == [
        ("1", False, 3),
        ("2", True, 0),
    ]
    assert len(gate.read_lines(records, "2026-11-26")) == 3, "the second run wrote no line"
    stopped = decided()
    assert stopped["pass"] is False and "ended red" in stopped["reasons"][0]

    # The same run and attempt again is one entry.
    assert run(evening, "2") == report.RED_EXIT
    assert len(gate.read_runs(records)) == 2

    # Every job of the first run run again the next day: a second attempt with a start of
    # its own. Both attempts stay listed, the first one's lines are still lines of a run
    # the list knows, and the newest thing the commit said is green again.
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "2")
    again = {**morning, "cutoff": "2026-11-27T08:00:00Z"}
    assert run(again, "1", *results) == 0
    assert [(e["id"], e["attempt"], e["date"]) for e in gate.read_runs(records)] == [
        ("1", "1", "2026-11-26"),
        ("2", "1", "2026-11-26"),
        ("1", "2", "2026-11-27"),
    ]
    tomorrow = gate.decide(
        gate.read_lines(records, "2026-11-27"),
        commit=commit,
        today="2026-11-27",
        claims={"openai_agents": 2},
        runs=gate.read_runs(records),
    )
    assert tomorrow["pass"] is True, tomorrow["reasons"]

    # The failed jobs of yesterday evening's run, run again now: written with yesterday's
    # plan, after today's entry. Nothing already listed is damage for being dated later.
    assert run(evening, "2") == report.RED_EXIT
    assert [(e["id"], e["attempt"], e["date"]) for e in gate.read_runs(records)] == [
        ("1", "1", "2026-11-26"),
        ("2", "1", "2026-11-26"),
        ("1", "2", "2026-11-27"),
        ("2", "2", "2026-11-26"),
    ]
    still = gate.decide(
        gate.read_lines(records, "2026-11-27"),
        commit=commit,
        today="2026-11-27",
        claims={"openai_agents": 2},
        runs=gate.read_runs(records),
    )
    assert still["pass"] is True, still["reasons"]


def _entry(day: str, run_id: str = "1", **more) -> dict:
    listed = {**_RUN, "id": run_id, "date": day, "scope": "all", "red": False, "points": 2}
    kept = (*report._RUN_KEPT, "red", "points")
    return {k: v for k, v in {**listed, **more}.items() if k in kept}


_NOW = {**_RUN, "id": "9", "date": "2026-12-04", "scope": "all", "red": False, "points": 2}


def test_the_list_of_runs_keeps_a_week_and_is_not_capped():
    before = [_entry("2026-11-26"), _entry("2026-11-27", "2"), _entry("2026-12-04", "3")]
    assert [e["id"] for e in report.run_list(before, _NOW)] == ["2", "3", "9"], "seven days"
    many = [_entry("2026-12-03", f"r{n}") for n in range(300)]
    assert len(report.run_list(many, _NOW)) == 301, "as many as there were, however many"
    assert [e["id"] for e in report.run_list(None, _NOW)] == ["9"], "no list yet is no damage"
    # Failed jobs of an older run, run again after a newer run has written: the newer
    # run's entry is dated after this one, and is as good an entry as it was.
    newer = _entry("2026-12-05", "n")
    assert report.run_list([newer], _NOW) == [newer, report.run_list([], _NOW)[0]]


@pytest.mark.parametrize(
    "damaged",
    [
        "not a list",
        [3],
        [{"id": "7"}],
        [_entry("2026-12-03", red="no")],
        [_entry("")],
        [_entry("soon")],
        [{"unreadable_since": "9999-12-31"}],
        [{"unreadable_since": 5}],
    ],
    ids=["text", "number", "half", "red", "undated", "words", "mark-future", "mark-type"],
)
def test_a_damaged_entry_is_not_dropped_by_the_next_run_and_ages_out_like_any_other(damaged):
    """It may have been the run that said a commit is not fit to release. The run that
    comes along next must not be what makes the gate forget it."""
    kept = report.run_list(damaged, _NOW)
    assert kept == [{"unreadable_since": "2026-12-04"}, kept[-1]] and kept[-1]["id"] == "9"
    assert not gate.entry_ok(kept[0]) and gate.entry_ok(kept[-1])
    week = report.run_list(kept, {**_NOW, "id": "10", "date": "2026-12-11"})
    assert week[0] == {"unreadable_since": "2026-12-04"}, "still there a week later"
    gone = report.run_list(week, {**_NOW, "id": "11", "date": "2026-12-12"})
    assert [e.get("id") for e in gone] == ["10", "11"]
