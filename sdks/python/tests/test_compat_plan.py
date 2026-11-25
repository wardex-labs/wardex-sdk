"""Which framework versions the daily compatibility run picks, and that nothing drops out of it.

The run lives in `scripts/compat/` and its plan is computed, not written down:
the table of tested versions plus what PyPI lists at one instant decide the
points. These tests drive that computation on invented release histories, and
hold the manifest to the table and to the test files, because an adapter or a
file missing from the run is a claim nobody is checking any more.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from conftest import (
    COMPAT_ASSUME_MEASURED,
    assume_installed_versions_measured,
    compat_switch_is_on,
)
from wardex_sdk import AdapterName
from wardex_sdk._adapters import _measured
from wardex_sdk._adapters._measured import MEASURED, is_measured

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))

from compat import manifest, plan  # noqa: E402

_TESTS = Path(__file__).parent
_CUTOFF = "2026-11-26T00:00:00Z"


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
            }
            for entry in entries
        ]
        for name, entries in MEASURED.items()
    }
    assert plan.read_measured() == shipped


def test_every_adapter_in_the_table_is_run_or_named_as_replay_only():
    for name, entries in MEASURED.items():
        if entries:
            spec = manifest.ADAPTERS[name.value]
            assert list(spec["targets"]) == [entry.distribution for entry in entries], name
        else:
            assert name.value in manifest.REPLAY_ONLY, name
    assert set(manifest.ADAPTERS) | set(manifest.REPLAY_ONLY) == {name.value for name in MEASURED}


#: Test files that import a framework at module level and are not run at that
#: framework's points, each with the reason. Empty is the goal.
_NOT_AT_A_POINT = {
    "test_adapter_envelopes.py": "imports two frameworks at once, so no single-framework "
    "environment can collect it",
}


def _reaches(path: Path, seen: set[Path]) -> set[str]:
    """Every module `path` imports at module level, through the suite's own helper modules."""
    if path in seen:
        return set()
    seen.add(path)
    reached = set()
    for module in re.findall(r"^(?:import|from) (\w+)", path.read_text("utf-8"), re.M):
        helper = _TESTS / f"{module}.py"
        reached |= _reaches(helper, seen) if helper.is_file() else {module}
    return reached


def test_every_test_file_that_imports_a_framework_runs_at_that_frameworks_points():
    """A file left out of the selection is a file whose assertions no point checks.
    Followed through helper modules: a file that reaches a framework only through
    `_some_support.py` needs that framework installed just the same."""
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


def test_the_switch_is_on_only_when_set_to_one():
    """An ordinary run, where the variable is absent, must never widen the table."""
    assert compat_switch_is_on({COMPAT_ASSUME_MEASURED: "1"})
    assert not compat_switch_is_on({})
    assert not compat_switch_is_on({COMPAT_ASSUME_MEASURED: "0"})
    assert not compat_switch_is_on({COMPAT_ASSUME_MEASURED: ""})


def test_the_switch_is_not_a_variable_init_would_warn_about():
    """`init()` warns once about any `WARDEX_*` variable it does not read. A switch of
    that prefix puts that warning into every test that asserts wardex said nothing."""
    assert not COMPAT_ASSUME_MEASURED.startswith("WARDEX_")


# --- the test-only switch a point above the range runs with -------------------


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

    before = assume_installed_versions_measured()
    try:
        (widened,) = MEASURED[AdapterName.OPENAI_AGENTS]
        assert is_measured(widened, "0.99.1")
        assert is_measured(widened, "0.22.0"), "the tested lines stay tested"
        assert not is_measured(widened, "0.98.0")
    finally:
        MEASURED.update(before)
    assert MEASURED[AdapterName.OPENAI_AGENTS] == (entry,)


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
