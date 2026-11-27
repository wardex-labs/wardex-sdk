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
        "openai_agents--minor_newest: openai-agents 0.23.1 is inside the tested range and: "
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
    lines = gate.read_lines(tmp_path, _TODAY)
    assert [line["key"] for line in lines] == ["openai_agents--floor", "openai_agents--claim_top"]
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
    assert "- openai_agents--floor: openai-agents 0.23.1 is inside the tested range and: " in said

    reason = "the floor fails on a dependency release; tracked upstream"
    code, said = _main(
        monkeypatch, tmp_path / "3", red, override={"version": "0.7.0b2", "reason": reason}
    )
    assert code == 0 and "**STOP.**" in said, "the reasons are still printed"
    assert (
        f"**OVERRIDDEN** for 0.7.0b2 by `{gate.OVERRIDE}` in this tree. Reason given: {reason}."
        in said
    )

    code, said = _main(
        monkeypatch, tmp_path / "4", red, override={"version": "0.7.0b1", "reason": reason}
    )
    assert code == gate.STOP_EXIT and "OVERRIDDEN" not in said

    code, said = _main(
        monkeypatch, tmp_path / "5", green, override={"version": "0.7.0b2", "reason": reason}
    )
    assert code == 0 and "OVERRIDDEN" not in said, "an override nobody needed is not mentioned"
