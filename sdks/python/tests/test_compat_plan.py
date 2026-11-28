"""Which framework versions the daily compatibility run picks, and that nothing drops out of it.

The run lives in `scripts/compat/` and its plan is computed, not written down:
the table of tested versions plus what PyPI lists at one instant decide the
points. These tests drive that computation on invented release histories, and
hold the manifest to the table and to the test files, because an adapter or a
file missing from the run is a claim nobody is checking any more.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from wardex_sdk import AdapterName
from wardex_sdk._adapters import _measured
from wardex_sdk._adapters._measured import MEASURED, is_measured

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))

sys.path.insert(0, str(_REPO / "scripts" / "compat" / "pytest_plugins"))

import compat_widen  # noqa: E402
from compat import kinds, manifest, model, plan  # noqa: E402
from compat import point as point_runner  # noqa: E402

_TESTS = Path(__file__).parent
_CUTOFF = "2026-11-26T00:00:00Z"
_NOTHING_PUBLISHED = {"problems": [], "points": [], "adapters": {}}


@pytest.fixture(autouse=True)
def _no_published_release(monkeypatch):
    """No test here reaches PyPI or a tag unless it says so."""
    monkeypatch.setattr(plan, "fetch_published", lambda cutoff, only=None: _NOTHING_PUBLISHED)


def _release(version: str, day: str, *, python: str = ">=3.10", yanked: bool = False) -> dict:
    stamp = f"{day}T12:00:00Z"
    return {
        "version": version,
        "files": [{"uploaded": stamp, "yanked": yanked, "requires_python": python}],
    }


def _points(entries: list[dict], releases: dict, adapter: str = "openai_agents") -> dict:
    planned = plan.plan_adapter(adapter, entries, releases, _CUTOFF)
    assert planned["info"]["problems"] == []
    return {point["key"]: point for point in planned["points"]}


_OPENAI = [{"distribution": "openai-agents", "minors": ("0.22", "0.23"), "known_bad": ()}]
_OPENAI_HISTORY = {
    "openai-agents": [
        _release("0.21.9", "2026-08-01"),
        _release("0.22.0", "2026-08-19"),
        _release("0.22.3", "2026-09-17"),
        _release("0.23.1", "2026-10-02"),
        _release("0.24.0", "2026-10-15"),
        _release("0.25.0", "2026-11-18"),
        _release("0.26.0", "2026-11-24"),
        _release("0.26.1", "2026-11-25"),
        _release("0.27.0rc1", "2026-11-25"),
    ]
}


def test_the_points_are_the_floor_each_tested_line_the_top_twice_and_the_lines_above():
    points = _points(_OPENAI, _OPENAI_HISTORY)
    assert {key: point["targets"]["openai-agents"] for key, point in points.items()} == {
        "openai_agents--floor": "0.22.0",
        "openai_agents--minor_newest--0.22": "0.22.3",
        "openai_agents--claim_top": "0.23.1",
        "openai_agents--claim_top_oldpy": "0.23.1",
        "openai_agents--beyond--0.24": "0.24.0",
        "openai_agents--beyond--0.25": "0.25.0",
        "openai_agents--beyond--0.26": "0.26.1",
    }
    assert points["openai_agents--floor"]["python"] == "3.10"
    assert points["openai_agents--claim_top"]["python"] == "3.14"
    assert points["openai_agents--claim_top_oldpy"]["python"] == "3.10"


def test_only_a_point_above_the_range_is_outside_the_claim_and_assumed_measured():
    points = _points(_OPENAI, _OPENAI_HISTORY)
    for key, point in points.items():
        above = "--beyond--" in key
        assert point["in_claim"] is not above, key
        assert point["assume_measured"] is above, key


def test_a_floor_resolves_with_the_dependencies_of_its_own_time_and_nothing_else_does():
    points = _points(_OPENAI, _OPENAI_HISTORY)
    assert points["openai_agents--floor"]["cutoff"] == "2026-09-02T12:00:00Z"
    assert {p["cutoff"] for k, p in points.items() if k != "openai_agents--floor"} == {_CUTOFF}


def test_a_release_no_longer_served_or_known_bad_is_not_the_floor():
    history = {
        "openai-agents": [
            {"version": "0.22.0", "files": []},
            _release("0.22.1", "2026-08-20", yanked=True),
            _release("0.22.2", "2026-08-21"),
            _release("0.22.3", "2026-09-17"),
            _release("0.23.1", "2026-10-02"),
        ]
    }
    entries = [{**_OPENAI[0], "known_bad": ("0.22.2",)}]
    assert _points(entries, history)["openai_agents--floor"]["targets"] == {
        "openai-agents": "0.22.3"
    }


def test_a_release_uploaded_after_the_cutoff_is_not_planned():
    history = {
        "openai-agents": [*_OPENAI_HISTORY["openai-agents"], _release("0.28.0", "2026-11-27")]
    }
    versions = {p["targets"]["openai-agents"] for p in _points(_OPENAI, history).values()}
    assert "0.28.0" not in versions


def test_a_prerelease_is_recorded_and_never_run():
    planned = plan.plan_adapter("openai_agents", _OPENAI, _OPENAI_HISTORY, _CUTOFF)
    assert planned["info"]["prerelease_ahead"] == "0.27.0rc1"
    assert all("rc" not in p["targets"]["openai-agents"] for p in planned["points"])


def test_at_most_three_lines_above_the_range_are_run_and_they_are_the_newest():
    history = {
        "openai-agents": [
            _release("0.23.1", "2026-10-02"),
            *(_release(f"0.{minor}.0", "2026-11-01") for minor in range(24, 30)),
        ]
    }
    entries = [{"distribution": "openai-agents", "minors": ("0.23",), "known_bad": ()}]
    above = sorted(key for key in _points(entries, history) if "--beyond--" in key)
    assert above == [f"openai_agents--beyond--0.{minor}" for minor in (27, 28, 29)]


def test_a_whole_major_claim_has_no_per_line_points_and_looks_above_by_major():
    entries = [{"distribution": "pydantic-ai-slim", "minors": ("2",), "known_bad": ()}]
    history = {
        "pydantic-ai-slim": [
            _release("2.0.0", "2026-06-23"),
            _release("2.54.0", "2026-10-03"),
            _release("2.74.0", "2026-11-25", python=">=3.11"),
            _release("3.0.0", "2026-11-25", python=">=3.11"),
            _release("3.0.1", "2026-11-25", python=">=3.11"),
        ]
    }
    points = _points(entries, history, "pydantic_ai")
    assert {key: p["targets"]["pydantic-ai-slim"] for key, p in points.items()} == {
        "pydantic_ai--floor": "2.0.0",
        "pydantic_ai--claim_top": "2.74.0",
        "pydantic_ai--claim_top_oldpy": "2.54.0",
        "pydantic_ai--beyond--3": "3.0.1",
    }


def test_a_range_tested_from_a_release_on_has_its_floor_there():
    """The table can say a range is tested from one release on. Nothing below it is a
    point, because nothing below it is claimed."""
    entries = [
        {"distribution": "pydantic-ai-slim", "minors": ("2",), "known_bad": (), "since": "2.54"}
    ]
    history = {
        "pydantic-ai-slim": [
            _release("2.0.0", "2026-06-23"),
            _release("2.53.0", "2026-10-02"),
            _release("2.54.0", "2026-10-03"),
            _release("2.74.0", "2026-11-25", python=">=3.11"),
        ]
    }
    points = _points(entries, history, "pydantic_ai")
    assert {key: p["targets"]["pydantic-ai-slim"] for key, p in points.items()} == {
        "pydantic_ai--floor": "2.54.0",
        "pydantic_ai--claim_top": "2.74.0",
        "pydantic_ai--claim_top_oldpy": "2.54.0",
    }
    assert plan.claims(entries[0], "2.54.0")
    assert not plan.claims(entries[0], "2.53.0")


@pytest.mark.parametrize("since", ["2.54", "2.54.0"])
@pytest.mark.parametrize(
    ("version", "claimed"),
    [("2.54", True), ("2.54.0", True), ("2.53.9", False), ("2.9.0", False), ("2.100.0", True)],
)
def test_where_a_range_starts_is_compared_as_numbers_like_the_sdk_does(since, version, claimed):
    """Compared as text, 2.9 would sort above 2.54 and 2.100 below it."""
    row = {"minors": ("2",), "known_bad": (), "since": since}
    assert plan.claims(row, version) is claimed
    assert is_measured(_measured.Measured("x", ("2",), since=since), version) is claimed


def test_the_plan_and_the_sdk_call_the_same_releases_tested():
    """Two implementations of one rule, because the plan cannot import the package. Held
    together over every spelling of a final release the index could serve."""
    rows = [
        (("2",), "", ()),
        (("2",), "2.54", ("2.60.0",)),
        (("0.22", "0.23"), "", ("0.23.0",)),
        (("0.22", "0.23"), "0.22.3", ()),
        (("1.2",), "1.2.0", ()),
    ]
    versions = [
        ".".join(str(n) for n in parts)
        for parts in (
            [(a,) for a in (0, 1, 2, 3)]
            + [(a, b) for a in (0, 1, 2, 3) for b in (0, 9, 22, 23, 53, 54, 60, 100)]
            + [(a, b, c) for a in (0, 1, 2) for b in (0, 2, 22, 23, 54, 60) for c in (0, 3, 10)]
            + [(2, 54, 0, 1), (0, 22, 3, 0)]
        )
    ] + ["02.54", "2.054.0", "0.022.3", "00.23.1"]
    for minors, since, bad in rows:
        entry = _measured.Measured("x", minors, known_bad=bad, since=since)
        row = {"minors": minors, "known_bad": bad, "since": since}
        for version in versions:
            assert plan.claims(row, version) is is_measured(entry, version), (row, version)


# --- a failed release that a newer one replaced --------------------------------

_WITH_023_0 = {
    "openai-agents": [
        _release("0.22.0", "2026-08-19"),
        _release("0.23.0", "2026-09-30"),
        _release("0.23.1", "2026-10-02"),
    ]
}


def _remembered(version: str, python: str = "3.14") -> dict:
    targets = {"openai-agents": version}
    return {
        "key": f"openai_agents--displaced--{plan.displaced_line(targets, python)}",
        "targets": targets,
        "python": python,
        "failed_on": "2026-11-25",
    }


def test_a_failed_release_that_a_newer_one_replaced_keeps_a_point_of_its_own():
    """The newest release of a line is a point only while it is the newest. One that
    failed there is still inside the range the README calls tested, so it stays a point."""
    item = _remembered("0.23.0")
    planned = plan.plan_adapter("openai_agents", _OPENAI, _WITH_023_0, _CUTOFF, [item])
    (point,) = [p for p in planned["points"] if p["role"] == "displaced"]
    assert point["key"] == item["key"] == "openai_agents--displaced--0.23.0-py3.14"
    assert point["targets"] == {"openai-agents": "0.23.0"}
    assert (point["python"], point["in_claim"], point["assume_measured"]) == ("3.14", True, False)
    assert point["cutoff"] == _CUTOFF

    # On the interpreter it failed on, which is not always the newest.
    old = _remembered("0.23.0", python="3.10")
    planned = plan.plan_adapter("openai_agents", _OPENAI, _WITH_023_0, _CUTOFF, [item, old])
    assert {p["key"]: p["python"] for p in planned["points"] if p["role"] == "displaced"} == {
        "openai_agents--displaced--0.23.0-py3.14": "3.14",
        "openai_agents--displaced--0.23.0-py3.10": "3.10",
    }

    whole = plan.build_plan(
        {"openai_agents": _OPENAI}, _WITH_023_0, _CUTOFF, {"displaced": {"openai_agents": [item]}}
    )
    assert item["key"] in {p["key"] for p in whole["points"]}


@pytest.mark.parametrize(
    ("entries", "item"),
    [
        ([{**_OPENAI[0], "known_bad": ("0.23.0",)}], _remembered("0.23.0")),
        ([{**_OPENAI[0], "since": "0.23.1"}], _remembered("0.23.0")),
        (_OPENAI, _remembered("0.21.9")),
        (_OPENAI, _remembered("0.23.7")),
        (_OPENAI, _remembered("0.23.0", python="2.7")),
        (_OPENAI, {"key": "x", "targets": {}, "python": "3.14"}),
    ],
    ids=[
        "listed-as-bad",
        "range-starts-later",
        "outside-the-range",
        "not-on-the-index",
        "no-such-interpreter",
        "names-no-release",
    ],
)
def test_a_remembered_release_that_is_no_longer_claimed_is_not_planned(entries, item):
    planned = plan.plan_adapter("openai_agents", entries, _WITH_023_0, _CUTOFF, [item])
    assert [p for p in planned["points"] if p["role"] == "displaced"] == []
    assert planned["info"]["problems"] == []


def test_a_remembered_pair_whose_companion_was_withdrawn_is_not_planned_and_does_not_crash():
    """LangGraph is remembered as a pair. With the companion's release gone from the index
    the pair cannot be installed again, and a plan that died on it would die every day."""
    entries = [
        {"distribution": "langgraph", "minors": ("1.2",), "known_bad": ()},
        {"distribution": "langgraph-prebuilt", "minors": ("1.1",), "known_bad": ()},
    ]
    history = {
        "langgraph": [_release("1.2.0", "2026-05-12"), _release("1.2.24", "2026-11-20")],
        "langgraph-prebuilt": [_release("1.1.0", "2026-05-12"), _release("1.1.3", "2026-11-19")],
    }
    gone = {"langgraph": "1.2.0", "langgraph-prebuilt": "1.1.2"}
    here = {"langgraph": "1.2.0", "langgraph-prebuilt": "1.1.3"}
    items = [{"key": "k", "targets": targets, "python": "3.14"} for targets in (gone, here)]
    planned = plan.plan_adapter("langgraph", entries, history, _CUTOFF, items)
    assert [p["targets"] for p in planned["points"] if p["role"] == "displaced"] == [here]
    assert planned["info"]["problems"] == []

    # Listing the companion's release as bad settles the pair, as the issue says it does.
    listed = [entries[0], {**entries[1], "known_bad": ("1.1.3",)}]
    history["langgraph-prebuilt"].append(_release("1.1.4", "2026-11-22"))
    planned = plan.plan_adapter("langgraph", listed, history, _CUTOFF, items)
    assert [p for p in planned["points"] if p["role"] == "displaced"] == []

    # A pair remembered before the table stopped naming its companion: skipped, not a crash.
    stray = [{"key": "k", "targets": {"langgraph": "1.2.0", "gone-from-the-table": "1.0"}}]
    stray[0]["python"] = "3.14"
    planned = plan.plan_adapter("langgraph", entries, history, _CUTOFF, stray)
    assert [p for p in planned["points"] if p["role"] == "displaced"] == []
    assert planned["info"]["problems"] == []


def test_the_floor_does_not_stand_in_for_a_release_that_failed_with_todays_packages():
    """The floor installs the dependencies of the release's own time. A release that
    failed with today's is still owed a run with today's, even when the floor is the same
    release on the same interpreter."""
    entries = [
        {"distribution": "pydantic-ai-slim", "minors": ("2",), "known_bad": (), "since": "2.54"}
    ]
    history = {
        "pydantic-ai-slim": [
            _release("2.54.0", "2026-10-03"),
            _release("2.55.0", "2026-10-09"),
        ]
    }
    targets = {"pydantic-ai-slim": "2.54.0"}
    item = {"key": "k", "targets": targets, "python": "3.10"}
    planned = plan.plan_adapter("pydantic_ai", entries, history, _CUTOFF, [item])
    roles = {p["role"]: p for p in planned["points"] if p["targets"] == targets}
    assert set(roles) == {"floor", "displaced"}
    assert roles["floor"]["cutoff"] != _CUTOFF and roles["displaced"]["cutoff"] == _CUTOFF


def test_the_plan_is_handed_what_the_last_report_remembered(tmp_path, monkeypatch):
    """The whole path, not the function: the workflow fetches `state.json` from the records
    branch and gives it to `plan.py`, and `main` gives it to the plan."""
    monkeypatch.setattr(plan, "read_measured", lambda: {"openai_agents": _OPENAI})
    monkeypatch.setattr(plan, "fetch_releases", lambda dist: _WITH_023_0[dist])
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"displaced": {"openai_agents": [_remembered("0.23.0")]}}))
    out = tmp_path / "plan.json"
    assert plan.main(["--out", str(out), "--cutoff", _CUTOFF, "--state", str(state)]) == 0
    keys = {point["key"] for point in json.loads(out.read_text())["points"]}
    assert "openai_agents--displaced--0.23.0-py3.14" in keys
    missing = tmp_path / "none.json"
    assert plan.main(["--out", str(out), "--cutoff", _CUTOFF, "--state", str(missing)]) == 0

    step = _jobs("compat.yml")["plan"]
    assert "git show FETCH_HEAD:state.json > plan/state.json" in step
    assert "plan.py --out plan/plan.json --state plan/state.json" in step


def test_a_remembered_release_a_rule_picks_again_gets_no_second_point():
    planned = plan.plan_adapter(
        "openai_agents", _OPENAI, _WITH_023_0, _CUTOFF, [_remembered("0.23.1")]
    )
    assert [p["key"] for p in planned["points"] if p["targets"] == {"openai-agents": "0.23.1"}] == [
        "openai_agents--claim_top",
        "openai_agents--claim_top_oldpy",
    ]


def test_a_point_falls_back_to_the_nearest_interpreter_its_release_admits():
    entries = [{"distribution": "pydantic-ai-slim", "minors": ("2",), "known_bad": ()}]
    history = {
        "pydantic-ai-slim": [
            _release("2.60.0", "2026-10-20", python=">=3.11,<3.14"),
            _release("2.61.0", "2026-10-21", python=">=3.11,<3.14"),
        ]
    }
    planned = plan.plan_adapter("pydantic_ai", entries, history, _CUTOFF)
    points = {p["key"]: p for p in planned["points"]}
    assert points["pydantic_ai--floor"]["python"] == "3.11"
    assert points["pydantic_ai--claim_top"]["python"] == "3.13"
    assert "pydantic_ai--claim_top_oldpy" not in points
    # A point that cannot exist is worth saying and is not a reason for a red run.
    assert planned["info"]["notes"] == ["no tested release of pydantic-ai-slim admits Python 3.10"]
    assert planned["info"]["problems"] == []


def test_a_companion_distribution_is_pinned_with_the_floor_and_with_the_top():
    entries = [
        {"distribution": "langgraph", "minors": ("1.2",), "known_bad": ()},
        {"distribution": "langgraph-prebuilt", "minors": ("1.1",), "known_bad": ()},
    ]
    history = {
        "langgraph": [_release("1.2.0", "2026-05-12"), _release("1.2.24", "2026-11-20")],
        "langgraph-prebuilt": [_release("1.1.0", "2026-05-12"), _release("1.1.3", "2026-11-19")],
    }
    points = _points(entries, history, "langgraph")
    assert points["langgraph--floor"]["targets"] == {
        "langgraph": "1.2.0",
        "langgraph-prebuilt": "1.1.0",
    }
    assert points["langgraph--claim_top"]["targets"] == {
        "langgraph": "1.2.24",
        "langgraph-prebuilt": "1.1.3",
    }
    assert points["langgraph--claim_top_oldpy"]["floats"] == ["langgraph-prebuilt"]


def test_a_range_with_nothing_installable_in_it_is_a_problem_not_an_empty_plan():
    history = {"openai-agents": [_release("0.30.0", "2026-11-01")]}
    planned = plan.plan_adapter("openai_agents", _OPENAI, history, _CUTOFF)
    assert planned["points"] == []
    assert "no installable release" in planned["info"]["problems"][0]


def test_a_point_that_cannot_exist_is_a_note_and_never_a_problem():
    """`problems` turn the run red. A release no supported interpreter admits is the
    framework's doing, said in the issue, and red would be a false alarm every day."""
    history = {
        "openai-agents": [
            _release("0.22.0", "2026-08-19"),
            _release("0.23.1", "2026-10-02"),
            _release("0.24.0", "2026-10-15", python=">=3.15"),
        ]
    }
    planned = plan.plan_adapter("openai_agents", _OPENAI, history, _CUTOFF)
    assert planned["info"]["problems"] == []
    assert planned["info"]["notes"] == [
        "openai-agents 0.24.0 admits no supported interpreter; not run"
    ]
    assert all("--beyond--" not in point["key"] for point in planned["points"])


def test_a_tested_release_that_no_interpreter_can_run_is_a_problem():
    """Inside the range the README calls it tested. If it cannot be run at all, a green
    run would be vouching for a release nothing looked at."""
    history = {
        "openai-agents": [
            _release("0.22.0", "2026-08-19"),
            _release("0.23.1", "2026-10-02", python=">=3.15"),
        ]
    }
    planned = plan.plan_adapter("openai_agents", _OPENAI, history, _CUTOFF)
    assert planned["info"]["problems"] == [
        "openai-agents 0.23.1 admits no supported interpreter; not run"
    ]


def test_a_plan_of_one_adapter_says_it_is_partial(tmp_path, monkeypatch):
    """The report must not read one adapter's plan as the whole run."""
    monkeypatch.setattr(plan, "read_measured", lambda: {"openai_agents": _OPENAI})
    monkeypatch.setattr(plan, "fetch_releases", lambda dist: _OPENAI_HISTORY[dist])
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    out = tmp_path / "plan.json"
    assert plan.main(["--out", str(out), "--cutoff", _CUTOFF, "--only", "openai_agents"]) == 0
    assert json.loads(out.read_text())["partial"] is True
    assert plan.main(["--out", str(out), "--cutoff", _CUTOFF]) == 0
    whole = json.loads(out.read_text())
    assert whole["partial"] is False
    assert len(whole["points"]) == 7


def test_the_same_inputs_give_the_same_plan():
    measured = {"openai_agents": _OPENAI}
    first = plan.build_plan(measured, _OPENAI_HISTORY, _CUTOFF)
    assert first == plan.build_plan(measured, _OPENAI_HISTORY, _CUTOFF)
    assert first["problems"] == []


@pytest.mark.parametrize(
    ("spec", "python", "admitted"),
    [
        ("", "3.10", True),
        (">=3.10", "3.10", True),
        (">=3.11", "3.10", False),
        (">=3.11", "3.14", True),
        ("<3.14,>=3.11", "3.14", False),
        ("<3.14,>=3.11", "3.13", True),
        (">=3.10.0", "3.10", True),
        ("<4.0.0,>=3.11.0", "3.11", True),
        ("!=3.12.*", "3.12", False),
        ("!=3.12.*", "3.13", True),
        ("~=3.10", "3.14", True),
        (">3.10", "3.10", False),
        ("garbage", "3.10", True),
    ],
)
def test_requires_python_is_read_as_the_installer_reads_it(spec, python, admitted):
    assert plan.admits(spec, python) is admitted


# --- the manifest against the table and the files -----------------------------


def test_the_plan_reads_the_table_the_sdk_ships():
    """The plan parses the table from source; it must be the table `init()` consults."""
    shipped = {
        name.value: [
            {
                "distribution": entry.distribution,
                "minors": entry.minors,
                "known_bad": entry.known_bad,
                "since": entry.since,
            }
            for entry in entries
        ]
        for name, entries in MEASURED.items()
    }
    assert plan.read_measured() == shipped


def test_the_plan_spells_a_tested_range_as_the_sdk_does():
    """The issue and the line `init()` prints name the same range, in the same words."""
    made = [
        _measured.Measured("x", ("0.22", "0.23")),
        _measured.Measured("x", ("0.22", "0.23"), since="0.22.3"),
        _measured.Measured("x", ("2",), since="2.54"),
        *(entry for entries in MEASURED.values() for entry in entries),
    ]
    for entry in made:
        row = {"minors": entry.minors, "since": entry.since}
        assert plan.spelled(row) == _measured.spelled(entry), entry


def test_every_adapter_in_the_table_is_run_or_named_as_replay_only():
    for name, entries in MEASURED.items():
        if entries:
            spec = manifest.ADAPTERS[name.value]
            assert list(spec["targets"]) == [entry.distribution for entry in entries], name
        else:
            assert name.value in manifest.REPLAY_ONLY, name
    assert set(manifest.ADAPTERS) | set(manifest.REPLAY_ONLY) == {name.value for name in MEASURED}


#: Test files that import a framework and are not run at that framework's points, each
#: with the reason. They run where every framework is installed together, in the pull
#: request job, and nowhere else. Shorter is better.
_SEVERAL = (
    "exercises several frameworks in one process, so no single-framework environment holds it"
)
_NOT_AT_A_POINT = {
    "test_adapter_install_on_import.py": _SEVERAL,
    "test_adapter_registry.py": _SEVERAL,
    "test_strict_mode.py": _SEVERAL,
    "test_uninstall_isolation.py": _SEVERAL,
}


def _reaches(path: Path, seen: set[Path]) -> set[str]:
    """Every module `path` imports, at any depth of indentation, through the suite's own
    helper modules. An import inside a test function needs the framework just the same."""
    if path in seen:
        return set()
    seen.add(path)
    reached = set()
    for module in re.findall(r"^[ \t]*(?:import|from) (\w+)", path.read_text("utf-8"), re.M):
        helper = _TESTS / f"{module}.py"
        reached |= _reaches(helper, seen) if helper.is_file() else {module}
    return reached


def test_every_test_file_that_imports_a_framework_runs_at_that_frameworks_points():
    """A file left out of the selection is a file whose assertions no point checks.
    Followed through helper modules: a file that reaches a framework only through
    `_some_support.py` needs that framework installed just the same."""
    unlisted = set(_NOT_AT_A_POINT)
    for name, spec in manifest.ADAPTERS.items():
        selected = {
            path.name
            for pattern in (*spec["tests"], *(t for o in spec["optional"] for t in o["tests"]))
            for path in _TESTS.glob(pattern)
        }
        frameworks = {module.split(".")[0] for module in spec["imports"]}
        importing = {
            path.name for path in _TESTS.glob("test_*.py") if frameworks & _reaches(path, set())
        }
        assert importing - selected - set(_NOT_AT_A_POINT) == set(), name
        unlisted -= importing
    assert unlisted == set(), "listed as not run at a point, but imports no framework"


def test_every_optional_test_file_is_also_in_the_adapters_selection():
    for name, spec in manifest.ADAPTERS.items():
        selected = {path.name for pattern in spec["tests"] for path in _TESTS.glob(pattern)}
        for optional in spec["optional"]:
            assert set(optional["tests"]) <= selected, name


def test_no_requirement_in_the_manifest_carries_an_upper_bound():
    """A lower bound says what a test file needs; anything else decides what users get."""
    for spec in manifest.ADAPTERS.values():
        optional = (r for o in spec["optional"] for r in o["requires"])
        for requirement in (*spec["requires"], *optional):
            assert not re.search(r"<|==|~=|!=", requirement), requirement
    assert all(re.fullmatch(r"[a-z-]+==[\d.]+", tool) for tool in manifest.TOOLS)


# --- the session plugin a point above the range runs with ----------------------


def test_the_switch_measures_an_installed_release_outside_the_table(monkeypatch):
    """A point above the range asks whether the suite would pass with the
    release added to the table; the switch adds it and hands the table back."""
    real = _measured.installed_version
    monkeypatch.setattr(
        _measured,
        "installed_version",
        lambda dist: "0.99.1" if dist == "openai-agents" else real(dist),
    )
    (entry,) = MEASURED[AdapterName.OPENAI_AGENTS]
    assert not is_measured(entry, "0.99.1")

    before = compat_widen.widen()
    try:
        (widened,) = MEASURED[AdapterName.OPENAI_AGENTS]
        assert is_measured(widened, "0.99.1")
        assert is_measured(widened, "0.22.0"), "the tested lines stay tested"
        assert not is_measured(widened, "0.98.0")
    finally:
        MEASURED.update(before)
    assert MEASURED[AdapterName.OPENAI_AGENTS] == (entry,)


def test_the_table_is_as_it_was_after_a_widened_session(monkeypatch):
    """The plugin's session fixture is this block: widened inside, restored on the way
    out, also when the session ends in an error."""
    real = _measured.installed_version
    monkeypatch.setattr(
        _measured,
        "installed_version",
        lambda dist: "0.99.1" if dist == "openai-agents" else real(dist),
    )
    shipped = dict(MEASURED)
    with pytest.raises(RuntimeError), compat_widen.widened():
        (inside,) = MEASURED[AdapterName.OPENAI_AGENTS]
        assert is_measured(inside, "0.99.1")
        raise RuntimeError("the session broke")
    assert dict(MEASURED) == shipped
    text = (_REPO / "scripts/compat/pytest_plugins/compat_widen.py").read_text(encoding="utf-8")
    assert "    with widened():\n        yield\n" in text


def test_the_widening_plugin_reads_only_names_a_published_release_already_has():
    """A point that tests a published wardex release runs that release's own tests, so the
    plugin cannot live in the tests tree and cannot lean on anything newer than the three
    names the table module has carried since it first shipped."""
    text = (_REPO / "scripts/compat/pytest_plugins/compat_widen.py").read_text(encoding="utf-8")
    used = set(re.findall(r"_measured\.(\w+)", text))
    assert used == {"MEASURED", "installed_version", "is_measured"}
    assert "compat_widen" not in (_TESTS / "conftest.py").read_text(encoding="utf-8")


# --- kinds: one device, many questions -------------------------------------------


def test_promotion_is_one_line_and_only_declared_kinds_can_be_promoted():
    """Whether a kind may turn the run red is not a field of its declaration: it is one
    name on one line of a set, so copying a neighbouring entry promotes nothing."""
    text = (_REPO / "scripts/compat/kinds.py").read_text(encoding="utf-8")
    (body,) = re.findall(
        r"RED_CAPABLE: frozenset\[str\] = frozenset\(\n    \{\n(.*?)    \}\n\)", text, re.S
    )
    lines = [line for line in body.splitlines() if line.strip()]
    assert [line.strip() for line in lines] == [f'"{name}",' for name in sorted(kinds.RED_CAPABLE)]
    assert kinds.RED_CAPABLE <= set(kinds.KINDS)
    assert "red" not in {field.lower() for field in kinds.Kind.__dataclass_fields__}


def _history_for(name: str) -> tuple[list[dict], dict]:
    """A table row per target of `name` and two invented releases of each."""
    entries = [
        {"distribution": dist, "minors": ("1.0",), "known_bad": (), "since": ""}
        for dist in manifest.ADAPTERS[name]["targets"]
    ]
    history = {
        entry["distribution"]: [_release("1.0.0", "2026-08-01"), _release("1.0.1", "2026-11-01")]
        for entry in entries
    }
    return entries, history


@pytest.mark.parametrize("name", sorted(manifest.ADAPTERS))
def test_every_planned_point_says_its_kind_and_carries_its_whole_recipe(name):
    """The executor reads the point and nothing else, so whatever the manifest says about
    an adapter has to be in every one of its points, whole."""
    entries, history = _history_for(name)
    spec = manifest.ADAPTERS[name]
    points = _points(entries, history, name)
    assert len(points) >= 2
    for key, planned in points.items():
        assert (planned["kind"], planned["subject"], planned["adapters"]) == (
            "framework",
            name,
            [name],
        ), key
        assert planned["watching"] is False and planned["sdk"] == {"origin": "build"}, key
        assert planned["runner"] == "linux-x64"
        assert planned["install"] == {
            "extras": dict(spec["extras"]),
            "requires": list(spec["requires"]),
            "optional": [
                {"requires": list(extra["requires"]), "tests": list(extra["tests"])}
                for extra in spec["optional"]
            ],
            "tools": list(manifest.TOOLS),
        }
        assert planned["probe"] == {"imports": list(spec["imports"]), "adapter": name}
        assert planned["check"] == {
            "how": "suite",
            "files": list(spec["tests"]),
            "plugins": list(manifest.PLUGINS),
            "test_timeout_s": manifest.TEST_TIMEOUT_S,
            "suite_timeout_s": manifest.SUITE_TIMEOUT_S,
            "evidence": spec["evidence"],
            "replay_of": spec["replay_of"],
        }
        assert planned["budget_s"] == manifest.POINT_BUDGET_S
        assert planned["fresh_days"] == kinds.KINDS["framework"].fresh_days


def test_a_kind_that_is_not_red_capable_is_planned_watching(monkeypatch):
    """The stamp the whole verdict leans on. A new kind is watching because its name is
    not on the list, without anyone remembering to say so."""
    entries, history = _history_for("openai_agents")
    (base,) = [p for p in _points(entries, history).values() if p["role"] == "claim_top"]
    spec = manifest.ADAPTERS["openai_agents"]
    watching = kinds.Kind("a new question", frozenset({"daily"}), 5, 1, "when it passes")
    monkeypatch.setitem(kinds.KINDS, "new_question", watching)
    made = plan.compile_point(base, kind="new_question", subject="openai_agents", spec=spec)
    assert made["watching"] is True and made["kind"] == "new_question"
    assert made["fresh_days"] == 5, "how long its proof lasts is the kind's to say"
    monkeypatch.setattr(kinds, "RED_CAPABLE", kinds.RED_CAPABLE | {"new_question"})
    again = plan.compile_point(base, kind="new_question", subject="openai_agents", spec=spec)
    assert again["watching"] is False


def _runner(
    tmp_path, planned: dict, monkeypatch, tests_tree: str | None = None
) -> tuple[point_runner.Point, list]:
    """A point runner over one planned point, with every command it would run captured."""
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "wardex_sdk-0.0.0-py3-none-any.whl").write_bytes(b"not a wheel")
    made = {"cutoff": _CUTOFF, "points": [planned]}
    (tmp_path / "plan.json").write_text(json.dumps(made), encoding="utf-8")
    args = point_runner.argparse.Namespace(
        plan=str(tmp_path / "plan.json"),
        key=planned["key"],
        wheel_dir=str(tmp_path / "dist"),
        work=str(tmp_path / "work"),
        out=str(tmp_path / "out"),
        last_green_lock=None,
        tests_tree=tests_tree,
    )
    ran: list = []

    def run(cmd, *, cwd=None, env=None, timeout=900):
        call = {"cmd": [str(part) for part in cmd], "env": env, "timeout": timeout, "cwd": cwd}
        ran.append(call)
        return 0, "", ""

    monkeypatch.setattr(point_runner, "run", run)
    made_point = point_runner.Point(args)
    made_point.deadline = point_runner.time.monotonic() + planned["budget_s"]
    return made_point, ran


def _planned(role: str, adapter: str = "pydantic_ai") -> dict:
    entries, history = _history_for(adapter)
    history[entries[0]["distribution"]].append(_release("1.1.0", "2026-11-20"))
    points = plan.plan_adapter(adapter, entries, history, _CUTOFF)["points"]
    return next(p for p in points if p["role"] == role)


def test_the_executor_runs_the_recipe_the_point_carries(tmp_path, monkeypatch):
    """Not the manifest's, and not a constant of its own: change the point and the commands
    change with it."""
    planned = _planned("claim_top")
    planned["check"].update(test_timeout_s=7, suite_timeout_s=91, plugins=["one.plugin", "two"])
    planned["install"]["tools"] = ["pytest==1.2.3"]
    planned["probe"]["imports"] = ["a_module", "another"]
    runner, ran = _runner(tmp_path, planned, monkeypatch)

    runner.pytest(tmp_path / "venv", ["test_a.py", "test_b.py"], "first")
    (call,) = ran
    cmd = call["cmd"]
    assert cmd[3:5] == ["sdks/python/tests/test_a.py", "sdks/python/tests/test_b.py"]
    assert "--timeout=7" in cmd and call["timeout"] == 91
    assert [cmd[i + 1] for i, part in enumerate(cmd) if part == "-p"] == [
        "no:cacheprovider",
        "one.plugin",
        "two",
    ]
    assert "PYTHONPATH" not in call["env"] and call["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"

    wanted = runner._requirements((0, 1))
    assert "pydantic-ai-slim[openai]==1.0.1" in wanted.splitlines() and "pytest==1.2.3" in wanted
    for requirement in ("opentelemetry-sdk", "fastmcp-slim[client]>=4.1", "boto3"):
        assert requirement in wanted.splitlines()
    assert "boto3" not in runner._requirements((0,)).splitlines()

    ran.clear()
    runner.control_import(tmp_path / "venv")
    assert ran[0]["cmd"][-1] == "import a_module, another"

    runner.dropped_tests = ["test_pydantic_ai_mcp.py"]
    assert "test_pydantic_ai_mcp.py" not in runner.files()
    assert "test_pydantic_ai_adapter.py" in runner.files()
    assert runner.excluded == ["test_pydantic_ai_mcp.py"]


def test_the_widening_plugin_is_loaded_only_for_a_point_that_asks_for_it(tmp_path, monkeypatch):
    """And from the run's own plugin directory, which is the whole of PYTHONPATH there: a
    source tree on the path would shadow the installed wheel."""
    planned = _planned("beyond", "openai_agents")
    assert planned["assume_measured"] is True
    runner, ran = _runner(tmp_path, planned, monkeypatch)
    runner.pytest(tmp_path / "venv", ["test_a.py"], "first")
    cmd, env = ran[0]["cmd"], ran[0]["env"]
    assert cmd[cmd.index("compat_widen") - 1] == "-p"
    assert env["PYTHONPATH"] == str(_REPO / "scripts/compat/pytest_plugins")
    assert (_REPO / "scripts/compat/pytest_plugins/compat_widen.py").is_file()


def test_a_point_runs_out_of_time_by_its_own_budget(tmp_path, monkeypatch):
    planned = _planned("claim_top")
    planned["budget_s"] = 200
    runner, ran = _runner(tmp_path, planned, monkeypatch)
    runner.deadline = 0.0
    steps = lambda self, venv: self.pytest(venv, ["test_a.py"], "first")  # noqa: E731
    monkeypatch.setattr(point_runner.Point, "_steps", steps)
    runner.execute()
    assert 60 <= ran[0]["timeout"] <= 140, "what the point has left, less a minute to write up"


def test_the_executor_cannot_ask_what_kind_of_point_it_runs(tmp_path, monkeypatch):
    """It follows the recipe in the point. A new kind is new values in the same fields; if
    the executor read the kind it would grow a branch per kind. Held by running it: a point
    with no kind at all, and one of a kind nobody declared, run the same commands."""

    def commands(change) -> list:
        planned = _planned("claim_top", "langgraph")
        change(planned)
        where = tmp_path / str(len(list(tmp_path.iterdir())))
        where.mkdir()
        runner, ran = _runner(where, planned, monkeypatch)
        runner.pytest(where / "venv", runner.files(), "first")
        runner.control_import(where / "venv")
        lines = runner._requirements((0,)).replace(str(where), "<work>")
        for call in ran:
            call["cmd"] = [part.replace(str(where), "<work>") for part in call["cmd"]]
            call["cwd"] = str(call["cwd"]).replace(str(where), "<work>")
            call.pop("timeout")
        return [lines, ran]

    def strip(planned: dict) -> None:
        for name in ("kind", "subject", "adapters", "watching", "runner", "sdk", "fresh_days"):
            del planned[name]

    def disguise(planned: dict) -> None:
        planned.update(kind="nobody_declared", subject="someone:else", watching=True)

    assert commands(lambda planned: None) == commands(strip) == commands(disguise)
    for name in ("point.py", "attach_probe.py"):
        text = (_REPO / "scripts/compat" / name).read_text(encoding="utf-8")
        imported = re.findall(r"^\s*(?:from compat import|import compat\.)\s*(.+)$", text, re.M)
        assert not any(re.search(r"\b(kinds|manifest)\b", line) for line in imported), name


def test_a_plan_says_what_it_speaks_for(tmp_path, monkeypatch):
    """The report forgets a point only inside what the plan covered. An adapter the index
    would not list is not covered, and a run of one adapter covers that adapter only."""
    whole = plan.build_plan({"openai_agents": _OPENAI}, _OPENAI_HISTORY, _CUTOFF)
    assert whole["scope"] == {
        "cadence": "daily",
        "covers": {"framework": ["openai_agents"]},
        "partial": False,
        "builds": {},
    }
    assert whole["schema"] == model.SCHEMA == 2
    assert whole["published"] is None and whole["unplanned"] == []

    unlisted = {"openai-agents": [_release("0.30.0", "2026-11-01")]}
    short = plan.build_plan({"openai_agents": _OPENAI}, unlisted, _CUTOFF)
    assert short["scope"]["covers"] == {"framework": []}

    monkeypatch.setattr(plan, "read_measured", lambda: {"openai_agents": _OPENAI})
    monkeypatch.setattr(plan, "fetch_releases", lambda dist: _OPENAI_HISTORY[dist])
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    out = tmp_path / "plan.json"
    assert plan.main(["--out", str(out), "--cutoff", _CUTOFF, "--only", "openai_agents"]) == 0
    assert json.loads(out.read_text())["scope"]["partial"] is True


def test_a_run_of_a_cadence_the_kind_is_not_part_of_plans_none_of_its_points():
    """Such a run has nothing to say about an adapter, and so nothing to close."""
    weekly = plan.build_plan({"openai_agents": _OPENAI}, _OPENAI_HISTORY, _CUTOFF, cadence="weekly")
    assert (weekly["points"], weekly["adapters"]) == ([], {})
    assert weekly["scope"]["covers"] == {} and weekly["cadence"] == "weekly"


def test_the_schedule_in_the_workflow_is_the_table_of_cadences():
    text = (_REPO / ".github/workflows/compat.yml").read_text(encoding="utf-8")
    assert set(re.findall(r'- cron: "([^"]+)"', text)) == set(kinds.CRONS)


# --- a fork that inherits the workflows -----------------------------------------

_UPSTREAM = "github.repository == 'wardex-labs/wardex-sdk'"


def _jobs(workflow: str) -> dict[str, str]:
    """`job id -> its text`, read as text: the suite has no YAML parser on its floor."""
    text = (_REPO / ".github/workflows" / workflow).read_text(encoding="utf-8")
    parts = re.split(r"^  ([a-z_]+):\n", text.split("\njobs:\n")[1], flags=re.M)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


def test_the_upstream_the_workflows_name_is_the_repository_the_package_names():
    urls = (_REPO / "sdks/python/pyproject.toml").read_text(encoding="utf-8")
    assert 'Repository = "https://github.com/wardex-labs/wardex-sdk"' in urls


def test_a_fork_starts_no_scheduled_run_and_writes_no_issue_or_record():
    """A fork inherits the workflow files and their schedules. It must not grow issues
    and a records branch of its own, and a run a contributor starts there by hand still
    has to work, reporting to its own summary."""
    jobs = _jobs("compat.yml")
    for name in ("plan", "wheel"):
        assert f"    if: github.event_name != 'schedule' || {_UPSTREAM}\n" in jobs[name], name
    assert "needs: [plan, wheel]" in jobs["point"], "skipped with the plan"
    assert "if: always() && needs.plan.result != 'skipped'" in jobs["report"]
    (write,) = re.findall(r"WRITE: \$\{\{ (.*) \}\}", jobs["report"])
    assert write == f"github.ref == 'refs/heads/main' && {_UPSTREAM} && '1' || ''"
    (commit,) = re.findall(r"if: (.*steps\.report\.outputs\.judged == 'true')", jobs["report"])
    assert _UPSTREAM in commit
    (died,) = re.findall(r"if: >-\n((?:      .*\n)+)", jobs["died"])
    assert " ".join(died.split()) == (
        f"always() && github.ref == 'refs/heads/main' && {_UPSTREAM} "
        "&& needs.plan.result != 'skipped' && needs.report.outputs.judged != 'true'"
    )

    (watch,) = re.findall(r"if: >-\n((?:      .*\n)+)", _jobs("compat-watchdog.yml")["watch"])
    assert " ".join(watch.split()) == (
        f"{_UPSTREAM} && (github.event_name != 'pull_request' "
        "|| github.event.pull_request.head.repo.full_name == github.repository)"
    )


def test_only_the_report_and_the_two_alarm_jobs_may_write():
    """A point runs framework code published the same day. Nothing but the jobs that
    install nothing may hold a token that can write."""
    writers = {
        name
        for workflow in ("compat.yml", "compat-watchdog.yml")
        for name, text in _jobs(workflow).items()
        if re.search(r"^      \w+: write$", text, flags=re.M)
    }
    assert writers == {"report", "died", "watch"}


# --- the release workflow's list of what it publishes ---------------------------


def test_the_release_publishes_exactly_the_artifacts_its_build_matrix_uploads():
    """`publish` names each wheel artifact instead of taking `wheels-*`, so that only the
    `build` job can have produced what reaches PyPI. A platform added to the matrix and
    not to that list would silently not be published; one left in the list would fail."""
    text = (_REPO / ".github/workflows/release-python.yml").read_text(encoding="utf-8")
    build, publish = text.split("\n  smoke:")[0], text.split("\n  publish:")[1]
    built = {
        f"wheels-{runner}-{target}"
        for runner, target in re.findall(r"runner: ([\w.-]+),\s+target: (\w+),", build)
    }
    assert len(built) == 5
    assert set(re.findall(r"name: (wheels-[\w.-]+)", publish)) == built
    assert "pattern:" not in publish


# --- the published release -------------------------------------------------------

_TAG = {
    "ref": "python-v0.7.0b1",
    "commit": "77b59d9" + "0" * 33,
    "table": {
        "openai_agents": _OPENAI,
        "codex_exec": [],
        "an_adapter_this_checkout_dropped": [
            {"distribution": "gone", "minors": ("1",), "known_bad": ()}
        ],
    },
    "tests": ["conftest.py", "test_openai_agents_adapter.py", "test_unrelated.py"],
}


def _file(day: str, *, yanked: bool = False) -> dict:
    return {"uploaded": f"{day}T12:00:00Z", "yanked": yanked, "requires_python": ">=3.10"}


@pytest.mark.parametrize(
    "served, newest",
    [
        ({"0.6.0b2": "2026-10-01", "0.7.0b9": "2026-11-20", "0.7.0b10": "2026-11-21"}, "0.7.0b10"),
        ({"0.9.0": "2026-10-01", "0.10.0": "2026-11-20"}, "0.10.0"),
        (
            {"0.7.0": "2026-10-01", "0.7.0.post1": "2026-11-20", "0.7.1.dev3": "2026-11-21"},
            "0.7.0.post1",
        ),
        ({"0.7.0rc1": "2026-11-01", "0.7.0": "2026-11-02", "0.7.1b1": "2026-11-03"}, "0.7.1b1"),
        ({"0.7.0": "2026-11-02", "0.7.0rc2": "2026-11-03", "0.7.0b9": "2026-11-04"}, "0.7.0"),
        ({"0.7.0b1": "2026-11-20", "0.8.0": "2026-11-27"}, "0.7.0b1"),
        (
            {"0.7.0b1": "2026-11-20", "0.8.0.dev1": "2026-11-21", "1.0+local": "2026-11-21"},
            "0.7.0b1",
        ),
        ({}, None),
    ],
)
def test_the_newest_wardex_release_counts_pre_releases_and_ranks_a_final_above_them(served, newest):
    """What `pip install --pre` gives is what a user has, and so far every release is one.
    A release uploaded after the run's instant does not exist for the run."""
    releases = [{"version": v, "files": [_file(day)]} for v, day in served.items()]
    assert plan.newest_wardex(releases, _CUTOFF) == newest


def test_a_yanked_wardex_release_is_not_the_newest():
    releases = [
        {"version": "0.7.0b1", "files": [_file("2026-11-20")]},
        {"version": "0.7.0b2", "files": [_file("2026-11-21", yanked=True)]},
    ]
    assert plan.newest_wardex(releases, _CUTOFF) == "0.7.0b1"
    one_wheel_pulled = [_file("2026-11-21", yanked=True), _file("2026-11-21")]
    releases.append({"version": "0.7.0b3", "files": one_wheel_pulled})
    assert plan.newest_wardex(releases, _CUTOFF) == "0.7.0b3", "one file still served is enough"


def test_the_published_release_is_run_on_the_top_of_its_own_range_and_the_newest_above_it():
    """Two points an adapter: the newest release the release's own table claims, and the
    newest one above that range. Picked by the rule that plans a framework point; what
    differs is in the point, which is why the executor needs no branch for it."""
    made = plan.plan_published("0.7.0b1", _TAG, _OPENAI_HISTORY, _CUTOFF)
    assert sorted(made["adapters"]) == ["openai_agents"], "only what both sides know"
    top, above = made["points"]
    assert (top["key"], above["key"]) == (
        "published--0.7.0b1--openai_agents--claim_top",
        "published--0.7.0b1--openai_agents--beyond--0.26",
    )
    assert (top["targets"], above["targets"]) == (
        {"openai-agents": "0.23.1"},
        {"openai-agents": "0.26.1"},
    )
    assert (top["in_claim"], above["in_claim"]) == (True, False)
    assert (top["assume_measured"], above["assume_measured"]) == (False, True)
    framework = {p["key"]: p for p in _points(_OPENAI, _OPENAI_HISTORY).values()}
    for point, twin in ((top, "openai_agents--claim_top"), (above, "openai_agents--beyond--0.26")):
        assert (point["kind"], point["subject"], point["watching"]) == (
            "published",
            "openai_agents",
            True,
        )
        assert point["sdk"] == {
            "origin": "pypi",
            "version": "0.7.0b1",
            "ref": "python-v0.7.0b1",
            "commit": _TAG["commit"],
        }
        assert point["fresh_days"] == kinds.KINDS["published"].fresh_days
        same = ("install", "probe", "check", "python", "cutoff", "role", "adapter", "budget_s")
        assert {k: point[k] for k in same} == {k: framework[twin][k] for k in same}


def test_a_release_whose_tag_holds_none_of_an_adapters_tests_is_said_and_not_run():
    tag = {**_TAG, "tests": ["conftest.py", "test_unrelated.py"]}
    made = plan.plan_published("0.7.0b1", tag, _OPENAI_HISTORY, _CUTOFF)
    assert made["points"] == []
    assert made["adapters"]["openai_agents"]["problems"] == [
        "the tag python-v0.7.0b1 holds none of its test files"
    ]


def test_published_points_join_the_plan_and_their_problems_never_reach_the_verdict():
    published = {"problems": [], **plan.plan_published("0.7.0b1", _TAG, _OPENAI_HISTORY, _CUTOFF)}
    made = plan.build_plan(
        {"openai_agents": _OPENAI}, _OPENAI_HISTORY, _CUTOFF, published=published
    )
    keys = [p["key"] for p in made["points"]]
    assert keys[-2:] == [p["key"] for p in published["points"]] and len(keys) == len(set(keys))
    assert made["published"] == {
        "version": "0.7.0b1",
        "ref": "python-v0.7.0b1",
        "commit": _TAG["commit"],
        "override": None,
        "adapters": published["adapters"],
    }
    assert made["scope"]["covers"]["published"] == ["openai_agents"]
    assert made["scope"]["builds"] == {"published": ["0.7.0b1"]}
    assert made["problems"] == [] and made["unplanned"] == []

    broken = {**_TAG, "tests": []}
    cannot = {"problems": [], **plan.plan_published("0.7.0b1", broken, _OPENAI_HISTORY, _CUTOFF)}
    made = plan.build_plan({"openai_agents": _OPENAI}, _OPENAI_HISTORY, _CUTOFF, published=cannot)
    assert made["problems"] == [], "a watching kind that cannot be planned is not a red run"
    assert made["unplanned"] == [
        "published: openai_agents: the tag python-v0.7.0b1 holds none of its test files"
    ]
    assert made["scope"]["covers"]["published"] == [], "and it forgets nothing"

    unread = {"problems": ["the newest wardex release could not be read (OSError)"]}
    made = plan.build_plan({"openai_agents": _OPENAI}, _OPENAI_HISTORY, _CUTOFF, published=unread)
    assert made["published"] is None and "published" not in made["scope"]["covers"]
    assert made["unplanned"] == ["published: the newest wardex release could not be read (OSError)"]

    weekly = plan.build_plan(
        {"openai_agents": _OPENAI}, _OPENAI_HISTORY, _CUTOFF, cadence="weekly", published=published
    )
    assert weekly["points"] == [] and weekly["published"] is None


def test_a_day_the_release_cannot_be_read_is_said_and_never_raised(monkeypatch):
    monkeypatch.undo()

    def down(distribution):
        raise OSError("no route")

    monkeypatch.setattr(plan, "fetch_releases", down)
    assert plan.fetch_published(_CUTOFF) == {
        "problems": ["the newest wardex release could not be read (OSError)"],
        "points": [],
        "adapters": {},
    }
    monkeypatch.setattr(plan, "fetch_releases", lambda distribution: [])
    assert plan.fetch_published(_CUTOFF)["problems"] == ["PyPI lists no wardex release"]


def _a_tag_in_git(
    monkeypatch, *, held: bool, shallow: bool, override: str | None = None
) -> list[tuple]:
    """`plan._git` answered from memory: a tag that is here, or one that must be fetched."""
    calls: list[tuple] = []
    fetched = {"done": held}
    measured = (_REPO / "sdks/python/src/wardex_sdk/_adapters/_measured.py").read_text("utf-8")
    enums = (_REPO / "sdks/python/src/wardex_sdk/_enums.py").read_text("utf-8")

    def git(*args: str) -> str:
        calls.append(args)
        if args[0] == "rev-parse" and "--is-shallow-repository" in args:
            return "true\n" if shallow else "false\n"
        if args[0] == "rev-parse":
            if not fetched["done"]:
                raise plan.subprocess.CalledProcessError(1, "git")
            return "c0ffee" + "0" * 34 + "\n"
        if args[0] == "fetch":
            fetched["done"] = True
            return ""
        if args[0] == "show" and args[1].endswith("release-override.json"):
            if override is None:
                raise plan.subprocess.CalledProcessError(128, "git")
            if override == "not text":
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
            return override
        if args[0] == "show":
            return measured if args[1].endswith("_measured.py") else enums
        if args[0] == "ls-tree":
            return "test_b.py\nconftest.py\ntest_a.py\n"
        raise AssertionError(args)

    monkeypatch.setattr(plan, "_git", git)
    return calls


def test_a_release_is_read_from_its_own_tag(monkeypatch):
    """Its table and its test files, at the tag's commit: the README a user of that release
    reads is that tag's, whatever this checkout says today."""
    calls = _a_tag_in_git(monkeypatch, held=True, shallow=False)
    tag = plan.read_tag("0.7.0b1")
    commit = "c0ffee" + "0" * 34
    assert (tag["ref"], tag["commit"]) == ("python-v0.7.0b1", commit)
    assert tag["table"] == plan.read_measured() and tag["tests"] == [
        "conftest.py",
        "test_a.py",
        "test_b.py",
    ]
    assert not any(call[0] == "fetch" for call in calls)
    shown = [call[-1] for call in calls if call[0] in ("show", "ls-tree")]
    assert len(shown) == 4 and tag["override"] is None
    assert all(path.startswith(f"{commit}:") for path in shown), "the commit, not a moving name"


@pytest.mark.parametrize("shallow", [True, False])
def test_a_tag_the_checkout_does_not_hold_is_fetched_without_cutting_a_full_clone(
    monkeypatch, shallow
):
    calls = _a_tag_in_git(monkeypatch, held=False, shallow=shallow)
    assert plan.read_tag("0.7.0b1")["commit"].startswith("c0ffee")
    (fetch,) = [call for call in calls if call[0] == "fetch"]
    assert fetch[-2:] == ("origin", "refs/tags/python-v0.7.0b1:refs/tags/python-v0.7.0b1")
    assert ("--depth" in fetch) is shallow


def test_the_matrix_tells_a_point_job_whose_tests_to_check_out(tmp_path, monkeypatch):
    published = {"problems": [], **plan.plan_published("0.7.0b1", _TAG, _OPENAI_HISTORY, _CUTOFF)}
    monkeypatch.setattr(plan, "fetch_published", lambda cutoff, only=None: published)
    monkeypatch.setattr(plan, "read_measured", lambda: {"openai_agents": _OPENAI})
    monkeypatch.setattr(plan, "fetch_releases", lambda dist: _OPENAI_HISTORY[dist])
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert plan.main(["--out", str(tmp_path / "plan.json"), "--cutoff", _CUTOFF]) == 0
    matrix = json.loads(output.read_text().removeprefix("matrix="))["include"]
    trees = {row["key"]: row["tree"] for row in matrix}
    assert trees["openai_agents--claim_top"] == ""
    assert trees["published--0.7.0b1--openai_agents--claim_top"] == _TAG["commit"]
    assert all(set(row) == {"key", "python", "tree"} for row in matrix)


def _published_point(role: str = "claim_top") -> dict:
    made = plan.plan_published("0.7.0b1", _TAG, _OPENAI_HISTORY, _CUTOFF)
    return next(p for p in made["points"] if p["role"] == role)


def test_a_published_point_installs_the_release_from_pypi_and_runs_its_tags_tests(
    tmp_path, monkeypatch
):
    tree = tmp_path / "tag"
    (tree / "sdks/python/tests").mkdir(parents=True)
    for name in ("test_openai_agents_adapter.py", "test_langgraph_adapter.py"):
        (tree / "sdks/python/tests" / name).write_text("", encoding="utf-8")
    runner, ran = _runner(tmp_path, _published_point(), monkeypatch, tests_tree=str(tree))
    (tmp_path / "dist" / "wardex_sdk-0.0.0-py3-none-any.whl").unlink()

    wanted = runner._requirements(()).splitlines()
    assert wanted[0] == "wardex-sdk==0.7.0b1" and "openai-agents==0.23.1" in wanted
    assert not any(line.endswith(".whl") for line in wanted)
    assert runner.result["sdk"] == {"wheel": "", "wheel_sha256": ""}

    assert runner.files() == ["test_openai_agents_adapter.py"], "only what the tag holds"
    runner.pytest(tmp_path / "venv", runner.files(), "first")
    runner.control_import(tmp_path / "venv")
    assert ran[0]["cwd"] == tree.resolve() and ran[0]["cmd"][3] == (
        "sdks/python/tests/test_openai_agents_adapter.py"
    )


def test_a_published_point_is_never_run_with_this_checkouts_tests(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="its own tests"):
        _runner(tmp_path, _published_point(), monkeypatch)


def test_a_published_point_that_imports_another_wardex_than_planned_is_a_broken_harness(
    tmp_path, monkeypatch
):
    tree = tmp_path / "tag"
    tree.mkdir()
    planned = _published_point()
    runner, _ = _runner(tmp_path, planned, monkeypatch, tests_tree=str(tree))
    venv = tmp_path / "venv"

    def seen(version: str):
        about = {
            "python": "3.14.2",
            "sdk": version,
            "file": str(venv / "lib/wardex_sdk/__init__.py"),
            "dists": {"openai-agents": "0.23.1"},
        }
        return lambda cmd, **kwargs: (0, json.dumps(about), "")

    monkeypatch.setattr(point_runner, "run", seen("0.7.0b1"))
    assert runner._verify(venv) is True and runner.result["sdk"]["version"] == "0.7.0b1"
    monkeypatch.setattr(point_runner, "run", seen("0.7.0b2"))
    with pytest.raises(AssertionError, match="wardex 0.7.0b2 is installed, not 0.7.0b1"):
        runner._verify(venv)


def test_a_run_of_one_adapter_plans_the_published_points_of_that_adapter_only(monkeypatch):
    monkeypatch.undo()
    table = {
        "openai_agents": _OPENAI,
        "langgraph": [{"distribution": "langgraph", "minors": ("1.2",), "known_bad": ()}],
    }
    tag = {
        **_TAG,
        "table": table,
        "tests": ["test_openai_agents_adapter.py", "test_langgraph_adapter.py"],
    }
    history = {**_OPENAI_HISTORY, "langgraph": [_release("1.2.0", "2026-08-01")]}
    history["wardex-sdk"] = [{"version": "0.7.0b1", "files": [_file("2026-11-20")]}]
    asked: list[str] = []

    def releases(distribution: str) -> list[dict]:
        asked.append(distribution)
        return history[distribution]

    monkeypatch.setattr(plan, "fetch_releases", releases)
    monkeypatch.setattr(plan, "read_tag", lambda version: {**tag, "table": dict(table)})
    one = plan.fetch_published(_CUTOFF, "openai_agents")
    assert sorted(one["adapters"]) == ["openai_agents"] and "langgraph" not in asked
    assert sorted(plan.fetch_published(_CUTOFF)["adapters"]) == ["langgraph", "openai_agents"]


def test_a_published_point_is_probed_and_fingerprinted_in_its_own_tree(tmp_path, monkeypatch):
    tree = tmp_path / "tag"
    (tree / "sdks/python/tests").mkdir(parents=True)
    (tree / "sdks/python/tests/test_openai_agents_adapter.py").write_text("# the tag's", "utf-8")
    runner, _ = _runner(tmp_path, _published_point(), monkeypatch, tests_tree=str(tree))
    venv = tmp_path / "venv"
    seen: list = []

    def run(cmd, *, cwd=None, env=None, timeout=900):
        seen.append({"cmd": [str(part) for part in cmd], "cwd": cwd})
        about = {"python": "3.14.2", "sdk": "0.7.0b1", "dists": {"openai-agents": "0.23.1"}}
        about["file"] = str(venv / "lib/wardex_sdk/__init__.py")
        return 0, json.dumps(about), ""

    monkeypatch.setattr(point_runner, "run", run)
    assert runner._verify(venv) is True
    assert seen[0]["cwd"] == tree.resolve(), "probed where the tests will run"

    monkeypatch.setattr(point_runner.Point, "install", lambda self, venv: True)
    monkeypatch.setattr(point_runner.Point, "control_import", lambda self, venv: True)
    monkeypatch.setattr(point_runner.Point, "attach", lambda self, venv: True)
    passed = {"ok": True, "exit_code": 0, "seconds": 1.0, "failed_ids": [], "skipped_ids": []}
    monkeypatch.setattr(point_runner.Point, "pytest", lambda self, venv, files, tag: dict(passed))
    runner.deadline = point_runner.time.monotonic() + 600
    runner._steps(venv)
    here = _REPO / "sdks/python/tests/test_openai_agents_adapter.py"
    theirs = point_runner.sha256(tree / "sdks/python/tests/test_openai_agents_adapter.py")
    assert point_runner.sha256(here) != theirs
    wanted = point_runner.hashlib.sha256(theirs.encode()).hexdigest()
    assert runner.result["tests"]["selection_sha256"] == wanted


def test_a_published_point_that_fails_is_replayed_with_the_release_and_never_blamed_on_wardex(
    tmp_path, monkeypatch
):
    """The wheel from PyPI and the tests of its tag are what they were at the last pass.
    If the packages of that pass fail too, no wardex change explains it."""
    tree = tmp_path / "tag"
    tree.mkdir()
    runner, ran = _runner(tmp_path, _published_point(), monkeypatch, tests_tree=str(tree))
    lock = tmp_path / "last-green.txt"
    lock.write_text("openai-agents==0.23.1 \\\n    --hash=sha256:" + "a" * 64 + "\n", "utf-8")
    (runner.out / "lock.txt").write_text(lock.read_text("utf-8"), "utf-8")
    runner.last_green = lock
    verdicts = iter([False, True])

    def replay(self, venv, files, tag):
        return {"ok": next(verdicts), "timed_out": False}

    monkeypatch.setattr(point_runner.Point, "pytest", replay)
    runner.counterfactual(["test_a.py"], ["test_a.py::test_x"])
    install = next(call["cmd"] for call in ran if "--no-deps" in call["cmd"])
    assert install[-1] == "wardex-sdk==0.7.0b1"
    assert runner.result["cause"] == "not_determined"
    runner.counterfactual(["test_a.py"], ["test_a.py::test_x"])
    assert runner.result["cause"] == "upstream"


def test_the_plan_says_which_releases_of_each_pinned_distribution_the_index_serves():
    """The report asks this before it remembers a failed release: one the index withdrew,
    of the framework or of a companion it is pinned with, cannot be run again."""
    history = {
        "langgraph": [_release("1.2.0", "2026-08-01"), _release("1.2.1", "2026-09-01")],
        "langgraph-prebuilt": [
            _release("1.1.0", "2026-08-01"),
            _release("1.1.1", "2026-09-01", yanked=True),
            _release("1.2.0", "2026-10-01"),
        ],
    }
    rows = [
        {"distribution": "langgraph", "minors": ("1.2",), "known_bad": (), "since": ""},
        {"distribution": "langgraph-prebuilt", "minors": ("1.1",), "known_bad": (), "since": ""},
    ]
    info = plan.plan_adapter("langgraph", rows, history, _CUTOFF)["info"]
    assert info["served"] == {"langgraph": ["1.2.0", "1.2.1"], "langgraph-prebuilt": ["1.1.0"]}


def test_a_release_that_overrode_its_gate_carries_that_into_the_plan(monkeypatch):
    """The override is in the release's own tree, so it is read from the tag like the
    table is, and travels with the plan to the page of that release."""
    said = json.dumps({"version": "0.7.0b1", "reason": "the floor fails upstream"})
    _a_tag_in_git(monkeypatch, held=True, shallow=False, override=said)
    tag = plan.read_tag("0.7.0b1")
    assert tag["override"] == {"version": "0.7.0b1", "reason": "the floor fails upstream"}
    made = plan.plan_published(
        "0.7.0b1", {**_TAG, "override": tag["override"]}, _OPENAI_HISTORY, _CUTOFF
    )
    whole = plan.build_plan(
        {"openai_agents": _OPENAI}, _OPENAI_HISTORY, _CUTOFF, published={"problems": [], **made}
    )
    assert whole["published"]["override"] == tag["override"]

    stale = json.dumps({"version": "0.6.0", "reason": "an earlier release"})
    _a_tag_in_git(monkeypatch, held=True, shallow=False, override=stale)
    assert plan.read_tag("0.7.0b1")["override"] is None
    # A file that is not text is no override, and does not cost the release its points.
    _a_tag_in_git(monkeypatch, held=True, shallow=False, override="not text")
    unreadable = plan.read_tag("0.7.0b1")
    assert unreadable["override"] is None and unreadable["tests"]
