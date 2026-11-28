"""The release gate: a commit goes out only with a recent pass for what it says it is tested on.

The gate reads record lines and decides. Nothing here publishes anything or
needs a tag: the decision is a function of the lines, the commit, the day and
the claims of the tree, and every test below hands it those four.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))

from compat import gate, manifest  # noqa: E402

_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_TODAY = "2026-12-03"
_CLAIMS = {"openai_agents": 2}


def _run(day: str = _TODAY, *, commit: str = _COMMIT, run_id: str = "7", **more) -> dict:
    return {
        "id": run_id,
        "url": f"https://example.invalid/runs/{run_id}",
        "ref": "refs/heads/main",
        "scope": "all",
        "cadence": "daily",
        "date": day,
        "started_at": f"{day}T19:40:00Z",
        "sdk_commit": commit,
        "red": False,
        **more,
    }


def _line(key: str, outcome: str = "pass", *, run: dict | None = None, **more) -> dict:
    return {
        "schema": 2,
        "run": run or _run(),
        "key": f"openai_agents--{key}",
        "kind": "framework",
        "subject": "openai_agents",
        "adapter": "openai_agents",
        "watching": False,
        "in_claim": key != "beyond",
        "targets": {"openai-agents": "0.23.1"},
        "outcome": outcome,
        **more,
    }


def _decide(*lines: dict, claims: dict | None = None, today: str = _TODAY) -> dict:
    return gate.decide(list(lines), commit=_COMMIT, today=today, claims=claims or _CLAIMS)


def test_a_commit_whose_every_claim_passed_yesterday_may_be_released():
    run = _run("2026-12-02")
    decision = _decide(_line("floor", run=run), _line("claim_top", run=run))
    assert decision == {"pass": True, "reasons": [], "run": "https://example.invalid/runs/7"}


def test_a_commit_no_run_looked_at_is_stopped_and_told_how_to_get_one():
    other = _run(commit="f" * 40)
    decision = _decide(_line("floor", run=other), _line("claim_top", run=other))
    assert decision["pass"] is False and decision["run"] is None
    assert decision["reasons"] == [
        "no whole compatibility run on the default branch at 0123456789 in the last 2 days"
    ]
    said = gate.words(decision, None, _COMMIT)
    assert "**STOP.**" in said and "gh workflow run compat.yml --ref main" in said


@pytest.mark.parametrize(
    "change",
    [
        {"date": "2026-11-30"},
        {"ref": "refs/heads/some-branch"},
        {"scope": "partial"},
        {"scope": "unknown"},
    ],
)
def test_an_old_run_a_branch_run_and_a_run_of_one_adapter_prove_nothing(change):
    run = _run(**change)
    assert _decide(_line("floor", run=run), _line("claim_top", run=run))["pass"] is False


@pytest.mark.parametrize("outcome", ["fail", "attach_failed", "skips_changed", "install_network"])
def test_a_release_inside_the_range_that_did_not_pass_stops_the_release(outcome):
    """An outage too: a range nobody could check that day is not a range that passed."""
    decision = _decide(_line("floor"), _line("claim_top"), _line("minor_newest", outcome))
    assert decision["pass"] is False and len(decision["reasons"]) == 1
    assert decision["reasons"][0].startswith(
        "`openai_agents--minor_newest`: `openai-agents 0.23.1` is inside the tested range and: "
    )


@pytest.mark.parametrize("outcome", ["no_result", "harness_error"])
def test_a_point_that_left_no_result_stops_the_release_wherever_it_sits(outcome):
    decision = _decide(_line("floor"), _line("claim_top"), _line("beyond", outcome))
    assert decision["pass"] is False and "openai_agents--beyond" in decision["reasons"][0]


def test_a_failure_above_the_range_and_a_watching_failure_stop_nothing():
    watching = _line("claim_top", "fail", kind="published", watching=True)
    watching["key"] = "published--0.7.0b1--openai_agents--claim_top"
    lost = {**watching, "key": "published--lost", "outcome": "no_result"}
    decision = _decide(_line("floor"), _line("claim_top"), _line("beyond", "fail"), watching, lost)
    assert decision["pass"] is True


def test_a_claim_with_too_few_passing_points_is_not_proven_by_the_rest():
    """A run that planned one adapter in full and lost another's points is green for the
    first only. The tree being released still says the second is tested."""
    decision = _decide(
        _line("floor"), _line("claim_top"), claims={"openai_agents": 2, "langgraph": 2}
    )
    assert decision["reasons"] == [
        "langgraph: 0 passing points inside its tested range, 2 are needed"
    ]
    one = _decide(_line("floor"), _line("beyond"))
    assert one["reasons"] == [
        "openai_agents: 1 passing points inside its tested range, 2 are needed"
    ]
    counted = _decide(
        _line("floor"), _line("claim_top", kind="published", watching=True), _line("beyond")
    )
    assert counted["pass"] is False, "a watching pass does not stand in for a framework point"


def test_the_newest_whole_run_on_the_commit_is_the_one_that_decides():
    morning, evening = _run(run_id="1"), _run(run_id="2")
    evening["started_at"] = f"{_TODAY}T21:00:00Z"
    red_then_green = _decide(
        _line("floor", "fail", run=morning),
        _line("claim_top", run=morning),
        _line("floor", run=evening),
        _line("claim_top", run=evening),
    )
    assert red_then_green["pass"] is True and red_then_green["run"].endswith("/2")
    green_then_red = _decide(
        _line("floor", run=morning),
        _line("claim_top", run=morning),
        _line("floor", "fail", run=evening),
        _line("claim_top", run=evening),
    )
    assert green_then_red["pass"] is False


def _records(tmp_path: Path, day: str, *lines: dict) -> Path:
    year, month, number = day.split("-")
    folder = tmp_path / "records" / year / month
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{number}.jsonl").open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")
    return tmp_path


def test_the_gate_reads_the_last_days_of_the_records_and_skips_what_is_not_a_line(tmp_path):
    _records(tmp_path, "2026-12-02", _line("floor"), _line("claim_top"))
    _records(tmp_path, "2026-11-30", _line("too_old"))
    (tmp_path / "records/2026/12/03.jsonl").write_text('not json\n[1]\n{"schema": 1}\n', "utf-8")
    with (tmp_path / "records/2026/12/03.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_line("after_the_bad_lines")) + "\n")
    _records(tmp_path, "2026-12-01", _line("two_days_ago"))
    lines = gate.read_lines(tmp_path, _TODAY)
    assert sorted(line["key"] for line in lines) == [
        "openai_agents--after_the_bad_lines",
        "openai_agents--claim_top",
        "openai_agents--floor",
        "openai_agents--two_days_ago",
    ]
    assert gate.read_lines(tmp_path / "nowhere", _TODAY) == []


def test_a_line_written_before_kinds_cannot_let_a_release_through(tmp_path):
    """Such a line does not say whether its run was whole, so it proves no claim."""
    old = {k: v for k, v in _line("floor").items() if k not in ("schema", "kind", "watching")}
    old["run"] = {k: v for k, v in old["run"].items() if k not in ("scope", "cadence")}
    _records(tmp_path, _TODAY, {**old, "schema": 1}, {**old, "schema": 1, "key": "x--claim_top"})
    decision = gate.decide(
        gate.read_lines(tmp_path, _TODAY), commit=_COMMIT, today=_TODAY, claims=_CLAIMS
    )
    assert decision["pass"] is False and decision["run"] is None


def test_the_claims_of_a_tree_are_the_adapters_its_table_gives_a_range():
    claims = gate.claimed(_REPO)
    assert set(claims) == set(manifest.ADAPTERS) and set(claims.values()) == {2}
    assert not set(claims) & set(manifest.REPLAY_ONLY)


def _tree(tmp_path: Path, version: str, override: object) -> Path:
    (tmp_path / "sdks/python").mkdir(parents=True)
    (tmp_path / "sdks/python/pyproject.toml").write_text(
        f'[project]\nname = "wardex-sdk"\nversion = "{version}"\n', encoding="utf-8"
    )
    if override is not None:
        (tmp_path / "scripts/compat").mkdir(parents=True)
        (tmp_path / gate.OVERRIDE).write_text(
            override if isinstance(override, str) else json.dumps(override), encoding="utf-8"
        )
    return tmp_path


def test_an_override_is_for_one_version_and_has_to_say_why(tmp_path):
    good = {"version": "0.7.0b2", "reason": "langgraph 1.2.24 fails upstream; fix in 0.7.0b3"}
    tree = _tree(tmp_path / "a", "0.7.0b2", good)
    assert gate.tree_version(tree) == "0.7.0b2"
    assert gate.read_override(tree, "0.7.0b2") == good
    assert gate.read_override(tree, "0.7.0b3") is None, "left behind by an earlier release"
    assert gate.read_override(_tree(tmp_path / "b", "0.7.0b2", None), "0.7.0b2") is None
    assert gate.read_override(_tree(tmp_path / "c", "0.7.0b2", "{not json"), "0.7.0b2") is None
    silent = {"version": "0.7.0b2", "reason": "  "}
    assert gate.read_override(_tree(tmp_path / "d", "0.7.0b2", silent), "0.7.0b2") is None
    assert gate.read_override(_tree(tmp_path / "e", "0.7.0b2", ["0.7.0b2"]), "0.7.0b2") is None
    for n, nothing in enumerate([None, False, 0, ["because"], {"why": "x"}, "``", "[]<>|"]):
        said = {"version": "0.7.0b2", "reason": nothing}
        tree = _tree(tmp_path / f"n{n}", "0.7.0b2", said)
        assert gate.read_override(tree, "0.7.0b2") is None, nothing
    marked = {"version": "0.7.0b2", "reason": "see [the issue](https://e.invalid) `now`\n## x"}
    kept = gate.read_override(_tree(tmp_path / "m", "0.7.0b2", marked), "0.7.0b2")
    assert kept == {"version": "0.7.0b2", "reason": "see the issue (https://e.invalid) now ## x"}
    for n, wordless in enumerate(["...", "---", "!!!", "123", "*", "? ?", "ab"]):
        tree = _tree(tmp_path / f"w{n}", "0.7.0b2", {"version": "0.7.0b2", "reason": wordless})
        assert gate.read_override(tree, "0.7.0b2") is None, wordless
    korean = {"version": "0.7.0b2", "reason": "하한 버전이 업스트림에서 깨져 먼저 내보냄"}
    said = gate.read_override(_tree(tmp_path / "k", "0.7.0b2", korean), "0.7.0b2")
    assert said == korean, "a reason is words in any language"
    long = {"version": "0.7.0b2", "reason": "because " * 100}
    assert (
        len(gate.read_override(_tree(tmp_path / "l", "0.7.0b2", long), "0.7.0b2")["reason"]) == 200
    )


def _main(monkeypatch, tmp_path, records: Path, *, override: object = None) -> tuple[int, str]:
    tree = _tree(tmp_path / "tree", "0.7.0b2", override)
    monkeypatch.setattr(gate, "REPO", tree)
    monkeypatch.setattr(gate, "claimed", lambda root: dict(_CLAIMS))
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    code = gate.main(["--commit", _COMMIT, "--records", str(records), "--today", _TODAY])
    return code, summary.read_text(encoding="utf-8")


def test_the_gate_passes_stops_and_is_overridden_out_loud(tmp_path, monkeypatch, capsys):
    green = _records(tmp_path / "green", _TODAY, _line("floor"), _line("claim_top"))
    code, said = _main(monkeypatch, tmp_path / "1", green)
    assert code == 0 and said.startswith("## Release gate: compatibility\n\n**PASS.**")
    assert "https://example.invalid/runs/7" in said and said in capsys.readouterr().out

    red = _records(tmp_path / "red", _TODAY, _line("floor", "fail"), _line("claim_top"))
    code, said = _main(monkeypatch, tmp_path / "2", red)
    assert code == gate.STOP_EXIT == 20 and "**STOP.**" in said and "OVERRIDDEN" not in said
    assert "- `openai_agents--floor`: `openai-agents 0.23.1` is inside the tested range " in said

    reason = "the floor fails on a dependency release; tracked upstream"
    code, said = _main(
        monkeypatch, tmp_path / "3", red, override={"version": "0.7.0b2", "reason": reason}
    )
    assert code == 0 and "**STOP.**" in said, "the reasons are still printed"
    assert (
        f"**OVERRIDDEN** for `0.7.0b2` by `{gate.OVERRIDE}` in this tree. Reason given: "
        f"`{reason}`. The release goes on" in said
    )

    code, said = _main(
        monkeypatch, tmp_path / "4", red, override={"version": "0.7.0b1", "reason": reason}
    )
    assert code == gate.STOP_EXIT and "OVERRIDDEN" not in said

    code, said = _main(
        monkeypatch, tmp_path / "5", green, override={"version": "0.7.0b2", "reason": reason}
    )
    assert code == 0 and "OVERRIDDEN" not in said and "**PASS.**" in said
    # The plan reads the file from the release's tag whatever the gate said, so a file
    # nobody needed would still put the sentence on the release's page.
    assert f"`{gate.OVERRIDE}` in this tree names this version and was not needed." in said


def test_a_run_two_days_old_still_proves_and_one_three_days_old_does_not():
    for day, proven in (("2026-12-03", True), ("2026-12-01", True), ("2026-11-30", False)):
        run = _run(day)
        assert _decide(_line("floor", run=run), _line("claim_top", run=run))["pass"] is proven, day


def test_the_last_attempt_of_a_run_is_what_the_run_says():
    """A run whose failed jobs were run again writes its lines a second time, under the
    same run and the same instant. What it said the first time is no longer what it says,
    whichever order the lines are read in."""
    first, second = _run(attempt="1"), _run(attempt="2")
    red = [_line("floor", "harness_error", run=first), _line("claim_top", run=first)]
    green = [_line("floor", run=second), _line("claim_top", run=second)]
    assert _decide(*red, *green)["pass"] is True
    assert _decide(*green, *red)["pass"] is True
    went_red = [_line("floor", "fail", run=second), _line("claim_top", run=second)]
    passed_first = [_line("floor", run=first), _line("claim_top", run=first)]
    assert _decide(*passed_first, *went_red)["pass"] is False
    assert _decide(_line("floor", run=_run(attempt=None)), _line("claim_top"))["pass"] is True


def test_a_point_recorded_twice_is_one_point():
    """Two points is the least that proves a range. One point written twice is still one,
    and a point that says pass on one line and fail on another did not pass."""
    once = _line("floor")
    assert _decide(once, once)["reasons"] == [
        "openai_agents: 1 passing points inside its tested range, 2 are needed"
    ]
    across = [_line("floor", run=_run(attempt="1")), _line("floor", run=_run(attempt="2"))]
    assert _decide(*across)["pass"] is False, "nor is the same point of two attempts two"
    mixed = _decide(_line("floor"), _line("claim_top"), _line("claim_top", "fail"))
    assert mixed["pass"] is False and "openai_agents--claim_top" in mixed["reasons"][0]


def test_only_a_framework_point_proves_a_tested_range():
    other = _line("claim_top", kind="something_else")
    other["key"] = "something_else--top"
    decision = _decide(_line("floor"), other)
    assert decision["reasons"] == [
        "openai_agents: 1 passing points inside its tested range, 2 are needed"
    ]


def test_a_commit_is_matched_whole_and_a_tree_with_no_claims_proves_nothing():
    lines = [_line("floor"), _line("claim_top")]
    short = gate.decide(lines, commit=_COMMIT[:12], today=_TODAY, claims=_CLAIMS)
    assert short["pass"] is False and "the whole commit id is needed" in short["reasons"][0]
    whole = gate.decide(lines, commit="f" * 40, today=_TODAY, claims=_CLAIMS)
    assert whole["pass"] is False and "whole commit id" not in whole["reasons"][0]
    empty = gate.decide(lines, commit=_COMMIT, today=_TODAY, claims={})
    assert empty == {
        "pass": False,
        "run": None,
        "reasons": ["this tree gives no adapter a tested range"],
    }


def test_an_adapter_with_no_row_in_the_table_is_not_a_claim(monkeypatch):
    table = {"openai_agents": [], "langgraph": [{"distribution": "langgraph"}], "not_run": [{}]}
    monkeypatch.setattr(gate.plan, "read_measured", lambda *paths: table)
    assert gate.claimed(_REPO) == {"langgraph": 2}


def test_a_missing_or_unreadable_attempt_is_the_first(tmp_path):
    """Lines written before attempts were recorded, beside a later attempt of the same
    run: the later attempt is still what the run says."""
    old, garbled, second = _run(attempt=None), _run(attempt="x"), _run(attempt="2")
    for first in (old, garbled):
        red = [_line("floor", "fail", run=first), _line("claim_top", run=first)]
        green = [_line("floor", run=second), _line("claim_top", run=second)]
        assert _decide(*red, *green)["pass"] is True
        assert _decide(*green, *red)["pass"] is True


def test_a_point_that_says_both_did_not_pass_whichever_line_comes_last():
    lines = [_line("floor"), _line("claim_top", "fail"), _line("claim_top")]
    assert _decide(*lines)["pass"] is False
    assert _decide(lines[0], lines[2], lines[1])["pass"] is False


def test_a_run_that_ended_red_proves_nothing_even_when_every_line_passes():
    """A run is also red for what its plan could not make a point of: a tested release
    that admits no supported interpreter, an adapter the index would not list. No line
    holds that. The lines carry the run's colour so the gate does not have to guess."""
    red = _run(red=True)
    decision = _decide(_line("floor", run=red), _line("claim_top", run=red))
    assert (
        decision["pass"] is False
        and "ended red for a reason no point holds" in (decision["reasons"][0])
    )
    with_reason = _decide(_line("floor", "fail", run=red), _line("claim_top", run=red))
    assert "floor" in with_reason["reasons"][0], "a point that says why is reason enough"
    assert not any("no point holds" in reason for reason in with_reason["reasons"])
    green = _run(red=False)
    assert _decide(_line("floor", run=green), _line("claim_top", run=green))["pass"] is True
    retried = [
        _line("floor", run=_run(red=True, attempt="1")),
        _line("claim_top", run=_run(red=True, attempt="1")),
    ]
    retried += [
        _line("floor", run=_run(red=False, attempt="2")),
        _line("claim_top", run=_run(red=False, attempt="2")),
    ]
    assert _decide(*retried)["pass"] is True, "the last attempt's colour"


def test_an_override_file_that_does_not_apply_is_said_not_ignored(tmp_path, monkeypatch, capsys):
    red = _records(tmp_path / "red", _TODAY, _line("floor", "fail"), _line("claim_top"))
    code, said = _main(
        monkeypatch, tmp_path / "1", red, override={"version": "0.7.0b2", "reason": "..."}
    )
    assert code == gate.STOP_EXIT and "OVERRIDDEN" not in said
    assert f"`{gate.OVERRIDE}` is in this tree and does not apply." in said
    assert '"version": "0.7.0b2"' in said
    code, said = _main(monkeypatch, tmp_path / "2", red)
    assert code == gate.STOP_EXIT and "does not apply" not in said, "no file, nothing to say"
    good = {"version": "0.7.0b2", "reason": "the floor fails upstream; tracked there"}
    applies = _tree(tmp_path / "3", "0.7.0b2", good)
    assert gate.rejected_override(applies, "0.7.0b2") == "", "a file that applies is not rejected"
    assert gate.rejected_override(applies, "0.7.0b3") != "", "left behind by an earlier release"


# --- what a third reader found ---------------------------------------------------


def test_lines_that_do_not_say_how_their_run_ended_prove_nothing():
    """Lines written before a run recorded its colour. Passing lines alone are not proof
    that the run was green, so the gate does not take them for it."""
    silent = {k: v for k, v in _run().items() if k != "red"}
    decision = _decide(_line("floor", run=silent), _line("claim_top", run=silent))
    assert decision["reasons"] == [
        "the run's lines do not say how it ended; run it again at this commit"
    ]
    half = _decide(_line("floor"), _line("claim_top", run=silent))
    assert half["pass"] is False


def test_nothing_a_record_line_holds_is_markup_in_what_the_gate_prints():
    nasty = "k](https://e.invalid)|x`y`<b>**z**"
    line = _line("floor", "fail", targets={"d|e [1](https://e.invalid)": "`1`"})
    line["key"] = nasty
    line["run"] = _run(url="https://e.invalid/x)[y](https://f.invalid")
    decision = _decide(line, _line("claim_top"))
    said = gate.words(decision, None, _COMMIT)
    first = decision["reasons"][0]
    assert first.startswith(
        "`k](https://e.invalid)|x?y?<b>**z**`: `d|e [1](https://e.invalid) ?1?`"
    )
    assert all(reason.count("`") % 2 == 0 for reason in decision["reasons"])
    assert decision["run"] == "" and "f.invalid" not in said, (
        "an address that is not one is dropped"
    )


@pytest.mark.parametrize(
    "damage",
    [
        {"targets": ["openai-agents"]},
        {"outcome": ["pass"]},
        {"outcome": "passed"},
        {"subject": ["x"]},
    ],
)
def test_a_damaged_line_of_the_deciding_run_stops_the_release_and_is_not_a_crash(damage):
    decision = _decide(_line("floor"), _line("claim_top"), _line("minor_newest", **damage))
    if "targets" in damage or "subject" in damage:
        assert decision["pass"] is True, "nothing the verdict reads was damaged"
    else:
        assert decision["reasons"] == [
            "`openai_agents--minor_newest`: a result this gate cannot read"
        ]
    beyond = _decide(_line("floor"), _line("claim_top"), _line("beyond", outcome="?"))
    assert beyond["pass"] is False, "a result nobody can read is not a failure above the range"


def test_a_records_file_with_bytes_that_are_not_text_keeps_its_good_lines(tmp_path):
    _records(tmp_path, _TODAY, _line("floor"), _line("claim_top"))
    day = tmp_path / "records/2026/12/03.jsonl"
    day.write_bytes(day.read_bytes() + b"\xff\xfe not text\n" + b'{"run": {}, "targets": 3}\n')
    lines = gate.read_lines(tmp_path, _TODAY)
    assert [line.get("key") for line in lines][:2] == [
        "openai_agents--floor",
        "openai_agents--claim_top",
    ]
    decision = gate.decide(lines, commit=_COMMIT, today=_TODAY, claims=_CLAIMS)
    assert decision["pass"] is True


def test_an_override_that_cannot_be_read_stops_the_release_and_says_the_file_is_there(
    tmp_path, monkeypatch
):
    red = _records(tmp_path / "red", _TODAY, _line("floor", "fail"), _line("claim_top"))
    tree = _tree(tmp_path / "bytes" / "tree", "0.7.0b2", "{}")
    (tree / gate.OVERRIDE).write_bytes(b"\xff\xfe")
    assert gate.read_override(tree, "0.7.0b2") is None
    assert "does not apply" in gate.rejected_override(tree, "0.7.0b2")
    folder = _tree(tmp_path / "folder" / "tree", "0.7.0b2", None)
    (folder / gate.OVERRIDE).mkdir(parents=True)
    assert gate.read_override(folder, "0.7.0b2") is None
    assert "does not apply" in gate.rejected_override(folder, "0.7.0b2")
    link = _tree(tmp_path / "link" / "tree", "0.7.0b2", None)
    (link / gate.OVERRIDE).parent.mkdir(parents=True)
    (link / gate.OVERRIDE).symlink_to(tmp_path / "nowhere")
    assert "does not apply" in gate.rejected_override(link, "0.7.0b2")
    monkeypatch.setattr(gate, "REPO", tree)
    monkeypatch.setattr(gate, "claimed", lambda root=None: dict(_CLAIMS))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    code = gate.main(["--commit", _COMMIT, "--records", str(red), "--today", _TODAY])
    assert code == gate.STOP_EXIT


def test_a_reason_is_printed_as_written_in_any_language_and_never_as_markup(tmp_path, monkeypatch):
    red = _records(tmp_path / "red", _TODAY, _line("floor", "fail"), _line("claim_top"))
    korean = "상위 프레임워크 버그로 측정 불가"
    code, said = _main(
        monkeypatch, tmp_path / "1", red, override={"version": "0.7.0b2", "reason": korean}
    )
    assert code == 0 and f"Reason given: `{korean}`." in said
    loud = "ok** **PASS** see https://e.invalid/x _x_ ~~y~~ @someone"
    code, said = _main(
        monkeypatch, tmp_path / "2", red, override={"version": "0.7.0b2", "reason": loud}
    )
    assert code == 0 and f"Reason given: `{loud}`." in said, "inside a code span nothing is markup"
