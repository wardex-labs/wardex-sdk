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


def _stamped(lines: tuple) -> list:
    """Each line with the number of points its attempt holds, as the report stamps it,
    unless the line already says a number or does not say which run it is of."""
    held: dict[tuple, set] = {}
    for line in lines:
        run = line.get("run") if isinstance(line, dict) else None
        if gate.run_of(line) is not None:
            held.setdefault((gate.run_of(line), str(run.get("attempt"))), set()).add(
                json.dumps(line.get("key"), sort_keys=True, default=str)
            )
    out = []
    for line in lines:
        if gate.run_of(line) is None or "points" in line["run"] or "red" not in line["run"]:
            out.append(line)
            continue
        count = len(held[(gate.run_of(line), str(line["run"].get("attempt")))])
        out.append({**line, "run": {**line["run"], "points": count}})
    return out


_LISTED = ("id", "attempt", "started_at", "date", "scope", "ref", "sdk_commit", "url")


def _listed(lines: list) -> list[dict]:
    """The list of runs the report would have kept for `lines`: one entry per run and
    attempt, with the colour and the count its lines carry."""
    entries: dict[tuple, dict] = {}
    for line in lines:
        if gate.run_of(line) is None:
            continue
        run = line["run"]
        entry = {field: run.get(field) for field in _LISTED}
        entry.update(red=run.get("red") is True, points=run.get("points") or 0)
        if gate._attempt(line) is None:
            # The list is the report's own and is not what a test of a damaged line damages.
            entry["attempt"] = None
        if not gate._count(entry["points"], least=0):
            entry["points"] = 0
        entries.setdefault((gate.run_of(line), str(run.get("attempt"))), entry)
    return list(entries.values())


def _judge(lines: list, runs: list | None = None, **more) -> dict:
    """`gate.decide` on lines exactly as given, with the list of runs they imply."""
    runs = _listed(lines) if runs is None else runs
    return gate.decide(
        lines, commit=_COMMIT, today=more.pop("today", _TODAY), claims=_CLAIMS, runs=runs, **more
    )


def _decide(*lines: dict, claims: dict | None = None, today: str = _TODAY) -> dict:
    stamped = _stamped(lines)
    return gate.decide(
        stamped, commit=_COMMIT, today=today, claims=claims or _CLAIMS, runs=_listed(stamped)
    )


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
        for line in _stamped(lines):
            handle.write(json.dumps(line) + "\n")
    state = tmp_path / "state.json"
    before = json.loads(state.read_text("utf-8"))["runs"] if state.is_file() else []
    state.write_text(json.dumps({"schema": 2, "runs": before + _listed(_stamped(lines))}), "utf-8")
    return tmp_path


def test_the_gate_reads_the_last_days_of_the_records_and_no_older(tmp_path):
    _records(tmp_path, "2026-12-02", _line("floor"), _line("claim_top"))
    _records(tmp_path, "2026-11-30", _line("too_old"))
    _records(tmp_path, "2026-12-01", _line("two_days_ago"))
    (tmp_path / "records/2026/12/02.jsonl").open("a").write("\n   \n")
    lines = gate.read_lines(tmp_path, _TODAY)
    assert sorted(line["key"] for line in lines) == [
        "openai_agents--claim_top",
        "openai_agents--floor",
        "openai_agents--two_days_ago",
    ], "an empty line is not a line"
    assert gate.read_lines(tmp_path / "nowhere", _TODAY) == []


@pytest.mark.parametrize(
    "garbage",
    [
        b"not json",
        b"[1]",
        b'{"schema": 1}',
        b"\xff\xfe not text",
        b"[" * 200_000,
        b'{"run": {}, "schema": "two"}',
        b'{"run": "not a run", "key": "x"}',
        b'{"run": {"id": "7", "started_at": "x", "scope": null, "ref": "r", "sdk_commit": "c",'
        b' "date": "d"}}',
    ],
    ids=["text", "list", "no-run", "bytes", "nested", "schema", "run-str", "scope-null"],
)
def test_a_line_nobody_can_read_stops_the_release_and_is_named(tmp_path, garbage):
    """It is never skipped. The line that cannot be read may be the one that said a point
    failed, and a truncated line says nothing about what it held."""
    _records(tmp_path, _TODAY, _line("floor"), _line("claim_top"))
    day = tmp_path / "records/2026/12/03.jsonl"
    runs = gate.read_runs(tmp_path)
    assert _judge(gate.read_lines(tmp_path, _TODAY), runs)["pass"] is True
    day.write_bytes(day.read_bytes() + garbage + b"\n")
    lines = gate.read_lines(tmp_path, _TODAY)
    assert lines[-1] == {"unreadable": "records/2026/12/03.jsonl:3"}
    decision = _judge(lines, runs)
    assert decision == {
        "pass": False,
        "run": None,
        "reasons": [
            "1 line(s) in the records of the last 2 days cannot be read (the first is "
            "`records/2026/12/03.jsonl:3`); whichever run they belong to, it cannot be taken "
            "as whole"
        ],
    }


def test_a_line_that_left_its_run_stops_the_release():
    """A green run with one point that could not be installed is stopped by that point.
    Damage that moves the point's line out of the run must not be what lets it pass. The
    list of runs is what the report wrote; the lines are what is left of them."""
    blocked = _line("minor_newest", "install_network")
    whole = _stamped((_line("floor"), _line("claim_top"), blocked))
    runs = _listed(whole)
    assert _judge(whole, runs)["pass"] is False
    for field, value in [
        ("scope", "partial"),
        ("ref", "refs/heads/x"),
        ("sdk_commit", "f" * 40),
        ("date", "2020-01-01"),
    ]:
        moved = {**whole[2], "run": {**whole[2]["run"], field: value}}
        decision = _judge([*whole[:2], moved], runs)
        assert decision["pass"] is False, field
        assert (
            "does not agree with the list of runs about which run it is" in decision["reasons"][0]
        )
    for field in ("scope", "id", "started_at"):
        lost = {**whole[2], "run": {**whole[2]["run"], field: None}}
        decision = _judge([*whole[:2], lost], runs)
        assert decision["pass"] is False and "cannot be read" in decision["reasons"][0]
    # Gone altogether, or under the name of a run the list knows: three planned, two here.
    missing = ["the run planned 3 points and 2 have a line; a line is missing or was added"]
    assert _judge(whole[:2], runs)["reasons"] == missing
    other = _stamped((_line("floor", run=_run(run_id="1", commit="f" * 40)),))
    renamed = {**whole[2], "run": other[0]["run"]}
    assert _judge([*whole[:2], *other, renamed], [*runs, *_listed(other)])["reasons"] == missing
    # Under the name of a run nobody listed, at this commit: the lines were changed.
    unknown = {**whole[2], "run": {**whole[2]["run"], "id": "8"}}
    assert _judge([*whole[:2], unknown], runs)["reasons"] == [
        "the records hold lines of a run the list of runs does not know (`8`)"
    ]
    green = _stamped((_line("floor"), _line("claim_top"), _line("minor_newest")))
    added = [*green, {**green[0], "key": "x--added"}]
    assert _judge(added, _listed(green))["reasons"] == [
        "the run planned 3 points and 4 have a line; a line is missing or was added"
    ]
    uneven = [green[0], green[1], {**green[2], "run": {**green[2]["run"], "points": 2}}]
    assert _judge(uneven, _listed(green))["reasons"] == [
        "the run's lines do not agree with the list of runs about how it ended"
    ]
    silent = [{**line, "run": {**line["run"], "red": None}} for line in green]
    assert _judge(silent, _listed(green))["pass"] is False


def test_the_newest_run_is_the_one_that_started_last_whatever_its_id():
    early, late = _run(run_id="9"), _run(run_id="10")
    late["started_at"] = f"{_TODAY}T21:00:00Z"
    green = [_line("floor", run=early), _line("claim_top", run=early)]
    red = [_line("floor", "fail", run=late), _line("claim_top", run=late)]
    assert _decide(*green, *red)["pass"] is False
    assert _decide(*red, *green)["pass"] is False
    early["started_at"], late["started_at"] = late["started_at"], early["started_at"]
    assert _decide(*green, *red)["pass"] is True


def test_lines_of_a_run_the_list_does_not_hold_matter_only_at_this_commit():
    """Records written before the list existed, or by a run of another commit or of a
    branch: not this release's business. At this commit they mean the list was changed."""
    green = _stamped((_line("floor"), _line("claim_top")))
    old = _stamped((_line("floor", "fail", run=_run(run_id="3", commit="f" * 40)),))
    branch = _stamped((_line("floor", "fail", run=_run(run_id="4", ref="refs/heads/x")),))
    assert _judge([*green, *old, *branch], _listed(green))["pass"] is True
    here = _stamped((_line("floor", "fail", run=_run(run_id="5")),))
    assert "does not know (`5`)" in _judge([*green, *here], _listed(green))["reasons"][0]
    # And lines that put themselves at this commit, of a run the list puts elsewhere.
    for field, value in [("sdk_commit", "f" * 40), ("ref", "refs/heads/x"), ("scope", "partial")]:
        moved = [{**entry, field: value} for entry in _listed(here)]
        decision = _judge([*green, *here], [*_listed(green), *moved])
        assert decision["reasons"] == [
            "the records hold lines that do not agree with the list of runs about which run "
            "they are of (`5`)"
        ], field


def test_a_run_that_wrote_no_line_is_still_the_newest_run():
    """The newest run at the commit planned no point at all: the index would not list a
    release, so it ended red and wrote nothing. An older green run at the same commit is
    not what the commit last said."""
    morning = _run(run_id="1")
    green = _stamped((_line("floor", run=morning), _line("claim_top", run=morning)))
    assert _judge(green)["pass"] is True
    evening = {
        **{field: _run(run_id="2").get(field) for field in _LISTED},
        "started_at": f"{_TODAY}T21:00:00Z",
        "red": True,
        "points": 0,
    }
    decision = _judge(green, [*_listed(green), evening])
    assert decision == {
        "pass": False,
        "run": "https://example.invalid/runs/2",
        "reasons": [
            "the run ended red for a reason no point holds (its plan could not cover "
            "something the tree says is tested); see the run"
        ],
    }
    # And a run of one line whose line is gone is a run with a line missing, not no run.
    lone = _stamped((_line("floor", "no_result", run={**_run(run_id="3"), "red": True}),))
    lone[0]["run"]["started_at"] = f"{_TODAY}T22:00:00Z"
    listed = [*_listed(green), *_listed(lone)]
    assert _judge([*green, *lone], listed)["pass"] is False
    assert _judge(green, listed)["pass"] is False, "its only line deleted"
    assert _judge(green, _listed(green))["pass"] is True


@pytest.mark.parametrize(
    "runs",
    [
        None,
        [3],
        [{"id": "7"}],
        "not a list",
    ],
    ids=["none", "number", "half", "text"],
)
def test_records_without_a_readable_list_of_runs_prove_nothing(tmp_path, runs):
    lines = _stamped((_line("floor"), _line("claim_top")))
    decision = gate.decide(lines, commit=_COMMIT, today=_TODAY, claims=_CLAIMS, runs=runs)
    assert decision["pass"] is False and "list of" in decision["reasons"][0]
    for field, value in [
        ("red", "no"),
        ("red", None),
        ("points", "2"),
        ("points", -1),
        ("attempt", "x"),
    ]:
        entry = {**_listed(lines)[0], field: value}
        assert _judge(lines, [entry])["reasons"] == [
            "the list of runs in `state.json` of the records cannot be read"
        ], field
    _records(tmp_path / "a", _TODAY, *lines)
    assert gate.read_runs(tmp_path / "a") == _listed(lines)
    assert gate.read_runs(tmp_path / "nowhere") is None
    for text in ("{not json", "[]", '{"runs": 3}', "{}"):
        (tmp_path / "a/state.json").write_text(text, encoding="utf-8")
        assert gate.read_runs(tmp_path / "a") is None


def test_a_line_written_before_kinds_cannot_let_a_release_through(tmp_path):
    """Such a line does not say whether its run was whole, so it proves no claim."""
    old = {k: v for k, v in _line("floor").items() if k not in ("schema", "kind", "watching")}
    old["run"] = {k: v for k, v in old["run"].items() if k not in ("scope", "cadence")}
    _records(tmp_path, _TODAY, {**old, "schema": 1}, {**old, "schema": 1, "key": "x--claim_top"})
    decision = _judge(gate.read_lines(tmp_path, _TODAY), gate.read_runs(tmp_path))
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


def test_only_a_framework_point_proves_a_tested_range(monkeypatch):
    """Also once another kind may turn the run red: what the README's ranges claim is
    proven by framework points and by nothing else."""
    monkeypatch.setattr(gate.kinds, "RED_CAPABLE", frozenset({"framework", "something_else"}))
    other = _line("claim_top", kind="something_else")
    other["key"] = "something_else--top"
    decision = _decide(_line("floor"), other)
    assert decision["reasons"] == [
        "openai_agents: 1 passing points inside its tested range, 2 are needed"
    ]


def test_a_commit_is_matched_whole_and_a_tree_with_no_claims_proves_nothing():
    lines = [_line("floor"), _line("claim_top")]
    lines = _stamped(lines)
    runs = _listed(lines)
    assert gate.decide(lines, commit=_COMMIT, today=_TODAY, claims=_CLAIMS, runs=runs)["pass"]
    short = gate.decide(lines, commit=_COMMIT[:12], today=_TODAY, claims=_CLAIMS, runs=runs)
    assert short["pass"] is False and "the whole commit id is needed" in short["reasons"][0]
    whole = gate.decide(lines, commit="f" * 40, today=_TODAY, claims=_CLAIMS, runs=runs)
    assert whole["pass"] is False and "whole commit id" not in whole["reasons"][0]
    empty = gate.decide(lines, commit=_COMMIT, today=_TODAY, claims={}, runs=runs)
    assert empty == {
        "pass": False,
        "run": None,
        "reasons": ["this tree gives no adapter a tested range"],
    }


def test_an_adapter_with_no_row_in_the_table_is_not_a_claim(monkeypatch):
    table = {"openai_agents": [], "langgraph": [{"distribution": "langgraph"}], "not_run": [{}]}
    monkeypatch.setattr(gate.plan, "read_measured", lambda *paths: table)
    assert gate.claimed(_REPO) == {"langgraph": 2}


def test_a_line_that_does_not_say_its_attempt_is_of_the_first():
    """Lines written before attempts were recorded, beside a later attempt of the same
    run: the later attempt is still what the run says."""
    old, second = _run(attempt=None), _run(attempt=2)
    red = [_line("floor", "fail", run=old), _line("claim_top", run=old)]
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
    """The list says the run ended green with two points. Lines that do not say so
    themselves are not the lines that run wrote, and passing lines alone are not proof."""
    said = _stamped((_line("floor"), _line("claim_top")))
    runs = _listed(said)
    assert _judge(said, runs)["pass"] is True
    for missing in ("red", "points"):
        quiet = [
            {**line, "run": {k: v for k, v in line["run"].items() if k != missing}} for line in said
        ]
        for lines in (quiet, [said[0], quiet[1]]):
            assert _judge(lines, runs)["reasons"] == [
                "the run's lines do not agree with the list of runs about how it ended"
            ], missing


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


_DAMAGE = [
    ("targets", ["openai-agents"]),
    ("targets", {"openai-agents": 7}),
    ("outcome", ["pass"]),
    ("outcome", "passed"),
    ("outcome", None),
    ("subject", ["x"]),
    ("kind", 3),
    ("key", {"a": 1}),
    ("watching", "false"),
    ("watching", 0),
    ("watching", True),
    ("in_claim", "no"),
    ("in_claim", None),
    ("run.red", 0),
    ("run.red", "false"),
    ("run.red", []),
    ("run.attempt", "x"),
    ("run.attempt", float("inf")),
    ("run.attempt", True),
    ("run.attempt", 0),
    ("run.attempt", "0"),
    ("run.attempt", "9" * 5000),
    ("run.points", "3"),
    ("run.points", True),
    ("run.points", 0),
    ("run.points", 2.0),
    ("run.attempt", [1]),
]


@pytest.mark.parametrize(("field", "value"), _DAMAGE, ids=[f"{f}={v!r}" for f, v in _DAMAGE])
def test_a_line_of_the_wrong_shape_stops_the_release_and_is_never_guessed_at(field, value):
    """Every field the verdict reads is held to the type the report writes. Read loosely,
    `"watching": "false"` hides a failure and `"red": 0` is a green run."""
    bad = _line("minor_newest", "fail" if field in ("watching", "in_claim") else "pass")
    if field.startswith("run."):
        bad["run"] = {**bad["run"], field[4:]: value}
    else:
        bad[field] = value
    decision = _decide(_line("floor"), _line("claim_top"), bad)
    assert decision["pass"] is False and len(decision["reasons"]) == 1
    assert decision["reasons"][0].endswith(f": a line of this run cannot be read ({field})")
    assert decision["reasons"][0].count("`") == 2
    assert "**STOP.**" in gate.words(decision, None, _COMMIT)


def test_a_line_of_another_run_can_be_any_shape_as_long_as_it_says_which_run():
    """Only the run that decides is held to the shape of a result: what a line of a branch
    run or of another commit says about its point is not this release's business."""
    branch = _run(run_id="3", ref="refs/heads/x")
    elsewhere = _line("floor", "fail", run=branch, watching="?", targets=3)
    older = _line("floor", outcome=[], run=_run(run_id="4", commit="f" * 40))
    assert _decide(_line("floor"), _line("claim_top"), elsewhere, older)["pass"] is True


def test_a_gate_that_cannot_decide_stops_and_says_so(tmp_path, monkeypatch, capsys):
    green = _records(tmp_path / "green", _TODAY, _line("floor"), _line("claim_top"))

    def broken(root=None):
        raise RecursionError("the table of tested versions could not be read")

    monkeypatch.setattr(gate, "REPO", _tree(tmp_path / "tree", "0.7.0b2", None))
    monkeypatch.setattr(gate, "claimed", broken)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    code = gate.main(["--commit", _COMMIT, "--records", str(green), "--today", _TODAY])
    assert code == gate.STOP_EXIT
    assert (
        "- the gate could not read what it decides from (RecursionError)" in capsys.readouterr().out
    )


def test_an_override_nested_past_reason_is_no_override_and_no_crash(tmp_path):
    tree = _tree(tmp_path / "tree", "0.7.0b2", "[" * 200_000)
    assert gate.read_override(tree, "0.7.0b2") is None
    assert "does not apply" in gate.rejected_override(tree, "0.7.0b2")


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
    # Characters that print as nothing, or turn the line around, are not kept either.
    hidden = "the floor\u202e fails\u200b upstream"
    code, said = _main(
        monkeypatch, tmp_path / "0", red, override={"version": "0.7.0b2", "reason": hidden}
    )
    assert code == 0 and "Reason given: `the floor fails upstream`." in said
    loud = "ok** **PASS** see https://e.invalid/x _x_ ~~y~~ @someone"
    code, said = _main(
        monkeypatch, tmp_path / "2", red, override={"version": "0.7.0b2", "reason": loud}
    )
    assert code == 0 and f"Reason given: `{loud}`." in said, "inside a code span nothing is markup"


# --- what a fifth reader found ---------------------------------------------------


def test_every_line_under_one_key_is_judged_and_a_key_that_disagrees_proves_nothing():
    """The report writes one line per point per attempt. Two under one key is damage, and
    it is never settled by looking at the first of them."""
    above = _line("claim_top", "fail", in_claim=False)
    for second in (_line("claim_top", "no_result"), _line("claim_top", "fail")):
        for lines in ([above, second], [second, above]):
            decision = _decide(_line("floor"), _line("minor_newest"), *lines)
            assert (
                decision["pass"] is False and "openai_agents--claim_top" in decision["reasons"][0]
            )
    mixed = [_line("claim_top"), _line("claim_top", in_claim=False)]
    assert _decide(_line("floor"), *mixed)["reasons"] == [
        "openai_agents: 1 passing points inside its tested range, 2 are needed"
    ]
    elsewhere = [_line("claim_top"), _line("claim_top", subject="langgraph")]
    assert _decide(_line("floor"), *elsewhere)["pass"] is False
    same = _line("claim_top", "fail")
    assert len(_decide(_line("floor"), _line("minor_newest"), same, same)["reasons"]) == 1


def test_a_damaged_line_stops_the_release_wherever_in_the_run_it_sits():
    bad = _line("minor_newest", "fail", watching="false")
    good = [_line("floor"), _line("claim_top")]
    for lines in ([bad, *good], [good[0], bad, good[1]], [*good, bad]):
        assert _decide(*lines)["pass"] is False
    # In an attempt a later one replaced, too: what the run wrote is not all readable.
    first, second = _run(attempt="1"), _run(attempt="2")
    earlier = _line("minor_newest", "fail", run=first, watching="false")
    later = [_line("floor", run=second), _line("claim_top", run=second)]
    decision = _decide(earlier, *later)
    assert decision["reasons"] == [
        "`openai_agents--minor_newest`: a line of this run cannot be read (watching)"
    ]
    assert gate.damage({"run": 3}) == "run"


def test_a_run_outside_actions_has_no_attempt_number_and_is_the_first():
    local = _run(attempt="")
    assert _decide(_line("floor", run=local), _line("claim_top", run=local))["pass"] is True


def test_a_run_whose_date_is_not_a_date_cannot_be_read():
    undated = _run(date=5)
    decision = _decide(_line("floor", run=undated), _line("claim_top", run=undated))
    assert decision["run"] is None and "2 line(s)" in decision["reasons"][0]


def test_a_tree_whose_version_cannot_be_read_has_no_release_to_excuse(
    tmp_path, monkeypatch, capsys
):
    red = _records(tmp_path / "red", _TODAY, _line("floor", "fail"), _line("claim_top"))
    nameless = {"version": "", "reason": "the floor fails upstream"}
    tree = _tree(tmp_path / "tree", "0.7.0b2", nameless)
    pyproject = tree / "sdks/python/pyproject.toml"
    for text in (None, b"\xff\xfe", b'[project]\nversion = "a b`c"\n', b"[project]\n"):
        if text is None:
            pyproject.unlink()
        else:
            pyproject.write_bytes(text)
        assert gate.tree_version(tree) == ""
        assert gate.read_override(tree, gate.tree_version(tree)) is None
        monkeypatch.setattr(gate, "REPO", tree)
        monkeypatch.setattr(gate, "claimed", lambda root=None: dict(_CLAIMS))
        # A summary that cannot be written changes no verdict either.
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path))
        code = gate.main(["--commit", _COMMIT, "--records", str(red), "--today", _TODAY])
        said = capsys.readouterr().out
        assert code == gate.STOP_EXIT and "OVERRIDDEN" not in said
        assert "the version of this tree could not be read" in said
    pyproject.write_text("[project]\nversion = '0.7.0b2'\n", encoding="utf-8")
    assert gate.tree_version(tree) == "0.7.0b2", "either kind of quote is a version line"
