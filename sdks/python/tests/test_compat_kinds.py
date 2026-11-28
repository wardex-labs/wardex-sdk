"""The other things a compatibility run looks at, as kinds of the same device.

Frameworks installed together, another instrumentation library beside an adapter, a
pre-release, the programs themselves, a Hermes release tag and a framework's development
branch each enter the plan as points of a kind of their own. Every one of them is
watching: recorded and reported, and never what turns a run red. Nothing here runs a
framework or reaches a network; the facts a rule reads from outside are handed to it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))

from compat import kinds, manifest, plan, render, report, verdict  # noqa: E402
from compat import point as point_runner  # noqa: E402

_TESTS = Path(__file__).parent
_CUTOFF = "2026-11-26T00:00:00Z"
_HEAD = "0123456789abcdef0123456789abcdef01234567"
_NEW = ("combination", "neighbour", "prerelease", "cli", "hermes", "devbranch")


def _release(version: str, day: str, *, python: str = ">=3.10") -> dict:
    stamp = f"{day}T12:00:00Z"
    return {
        "version": version,
        "files": [{"uploaded": stamp, "yanked": False, "requires_python": python}],
    }


def _world() -> tuple[dict, dict]:
    """A table row per target of every adapter and two invented releases of each."""
    measured, releases = {}, {}
    for name, spec in manifest.ADAPTERS.items():
        measured[name] = [
            {"distribution": dist, "minors": ("1.0",), "known_bad": (), "since": ""}
            for dist in spec["targets"]
        ]
        for dist in spec["targets"]:
            releases[dist] = [_release("1.0.0", "2026-08-01"), _release("1.0.1", "2026-11-01")]
    return measured, releases


def _facts(**change: object) -> dict:
    facts = {
        "npm": {
            "@openai/codex": [
                {"version": "0.160.0", "uploaded": "2026-10-01T00:00:00Z"},
                {"version": "0.172.0", "uploaded": "2026-11-24T00:00:00Z"},
                {"version": "0.173.0-alpha.1", "uploaded": "2026-11-25T00:00:00Z"},
                {"version": "0.174.0", "uploaded": "2026-11-27T00:00:00Z"},
            ]
        },
        "github_releases": {
            "NousResearch/hermes-agent": [
                {"tag": "v0.23.5", "published": "2026-11-25T03:00:00Z", "prerelease": False},
                {
                    "tag": "canary.1-v0.23.5",
                    "published": "2026-11-25T09:00:00Z",
                    "prerelease": True,
                },
                {"tag": "v0.23.3", "published": "2026-11-20T00:00:00Z", "prerelease": False},
                {"tag": "v2026.9.24", "published": "2026-09-24T00:00:00Z", "prerelease": False},
                {"tag": "v0.24.0", "published": "2026-11-27T00:00:00Z", "prerelease": False},
                {"tag": "v0.23.6", "published": "2026-11-25T12:00:00Z", "prerelease": True},
                {"tag": "v1.2.3.4", "published": "2026-11-25T13:00:00Z", "prerelease": False},
                {"tag": "v9.9", "published": "2026-11-25T14:00:00Z", "prerelease": False},
                {
                    "tag": "v\u0661.\u0660.\u0660",
                    "published": "2026-11-25T15:00:00Z",
                    "prerelease": False,
                },
                {
                    "tag": "v\uff11.\uff12.\uff13",
                    "published": "2026-11-25T16:00:00Z",
                    "prerelease": False,
                },
                {"tag": "v\u0669.0.0", "published": "2026-11-25T17:00:00Z", "prerelease": False},
                {"tag": "v9.\u0669.0", "published": "2026-11-25T18:00:00Z", "prerelease": False},
                {"tag": "v9.0.\u0669", "published": "2026-11-25T19:00:00Z", "prerelease": False},
            ]
        },
        "heads": {declared["repository"]: _HEAD for declared in manifest.DEV_BRANCHES.values()},
    }
    facts.update(change)
    return facts


def _plan(cadence: str = "daily", facts: dict | None = None, **more: object) -> dict:
    measured, releases = _world()
    releases = {**releases, **more.pop("releases", {})}
    return plan.build_plan(
        measured,
        releases,
        _CUTOFF,
        {},
        cadence=cadence,
        facts=_facts() if facts is None else facts,
        **more,
    )


def _of(made: dict, kind: str) -> dict[str, dict]:
    return {point["key"]: point for point in made["points"] if point["kind"] == kind}


# --- what every new kind has in common --------------------------------------------


def test_every_kind_but_the_first_is_watching_and_has_words():
    assert kinds.RED_CAPABLE == {"framework"}
    for name in _NEW:
        kind = kinds.KINDS[name]
        assert kind.label and kind.closes and kind.cadences and kind.fresh_days > 0, name
        assert not kinds.is_red_capable(name), name
    assert kinds.due("weekly") == ("devbranch",)
    assert set(kinds.due("daily")) == set(kinds.KINDS) - {"devbranch"}
    assert kinds.ISSUE_WRITER in set(kinds.CRONS.values())


def test_a_daily_plan_holds_points_of_every_daily_kind_and_none_can_turn_it_red():
    made = _plan()
    held = {point["kind"] for point in made["points"]}
    assert held == set(kinds.due("daily")) - {"published", "prerelease"}
    assert made["problems"] == [] and made["unplanned"] == []
    for point in made["points"]:
        assert point["watching"] is (point["kind"] != "framework"), point["key"]
    # Every one of them failing, and every result of theirs missing: still green.
    results = {
        point["key"]: {**point, "outcome": "no_result" if n % 2 else "fail", "cause": None}
        for n, point in enumerate(made["points"])
        if point["kind"] != "framework"
    }
    results.update(
        {
            p["key"]: {**p, "outcome": "pass", "cause": None}
            for p in made["points"]
            if p["kind"] == "framework"
        }
    )
    assert verdict.judge(made, results) == []


def test_each_rule_says_which_subjects_it_planned_in_full():
    made = _plan()
    covers = made["scope"]["covers"]
    assert covers["combination"] == sorted(manifest.COMBINATIONS)
    assert covers["neighbour"] == sorted({d["adapter"] for d in manifest.NEIGHBOURS.values()})
    assert covers["cli"] == sorted(manifest.CLI)
    assert covers["hermes"] == ["hermes_agent"]
    assert covers["prerelease"] == sorted(manifest.ADAPTERS)
    assert set(made["subjects"]) >= set(manifest.COMBINATIONS) | {"codex_exec", "hermes_agent"}


def test_a_plan_of_one_adapter_or_one_made_without_outside_facts_holds_tested_ranges_only():
    measured, releases = _world()
    for made in (
        plan.build_plan(measured, releases, _CUTOFF, {}, partial=True, facts=_facts()),
        plan.build_plan(measured, releases, _CUTOFF, {}),
    ):
        assert {point["kind"] for point in made["points"]} == {"framework"}
        assert set(made["scope"]["covers"]) == {"framework"}


def test_every_driver_a_point_names_is_a_script_pytest_does_not_collect():
    made = {**_of(_plan(), "neighbour"), **_of(_plan(), "cli"), **_of(_plan(), "hermes")}
    assert len(made) == 8
    for key, point in made.items():
        check = point["check"]
        assert check["how"] == "driver" and check["files"] == [], key
        script = _TESTS / check["driver"]
        assert script.is_file() and not script.name.startswith("test_"), key
        assert "{out}/driver.json" in check["args"], "where the executor reads the verdict"
        assert check["suite_timeout_s"] == manifest.DRIVER_TIMEOUT_S


# --- frameworks together ----------------------------------------------------------


def test_a_combination_is_every_member_at_its_newest_tested_release_with_every_members_tests():
    points = _of(_plan(), "combination")
    assert sorted(points) == sorted(f"combination--{name}" for name in manifest.COMBINATIONS)
    pair = points["combination--openai_agents__pydantic_ai"]
    assert pair["subject"] == "openai_agents__pydantic_ai"
    assert pair["adapters"] == ["openai_agents", "pydantic_ai"]
    assert pair["targets"] == {"openai-agents": "1.0.1", "pydantic-ai-slim": "1.0.1"}
    assert pair["in_claim"] is True and pair["role"] == "claim_top"
    both = [manifest.ADAPTERS[name] for name in pair["adapters"]]
    assert set(pair["check"]["files"]) == {t for spec in both for t in spec["tests"]}
    assert set(pair["install"]["requires"]) == {r for spec in both for r in spec["requires"]}
    assert pair["install"]["extras"] == {"pydantic-ai-slim": "[openai]"}
    assert [probe["adapter"] for probe in pair["probe"]["adapters"]] == pair["adapters"]
    everything = points["combination--all_frameworks"]
    assert set(everything["targets"]) == {
        d for s in manifest.ADAPTERS.values() for d in s["targets"]
    }
    assert everything["check"]["evidence"] == "wrapper-live", "one member replays a program"


def test_a_combination_with_a_member_that_has_no_newest_tested_point_is_said_and_not_planned():
    measured, releases = _world()
    releases["pydantic-ai-slim"] = [_release("9.0.0", "2026-11-01")]
    made = plan.build_plan(measured, releases, _CUTOFF, {}, facts=_facts())
    assert "combination--openai_agents__pydantic_ai" not in _of(made, "combination")
    assert "combination--openai_agents__langgraph" in _of(made, "combination")
    said = [line for line in made["unplanned"] if line.startswith("combination: ")]
    assert said == [
        "combination: openai_agents__pydantic_ai: no newest tested point for pydantic_ai",
        "combination: all_frameworks: no newest tested point for pydantic_ai",
    ]
    assert made["scope"]["covers"]["combination"] == ["openai_agents__langgraph"]


def test_frameworks_whose_newest_tested_releases_need_different_interpreters_are_not_combined():
    old_only = [_release("1.0.0", "2026-08-01"), _release("1.0.1", "2026-11-01", python="<3.14")]
    made = _plan(releases={"pydantic-ai-slim": old_only})
    assert "combination--openai_agents__pydantic_ai" not in _of(made, "combination")
    assert (
        "combination: openai_agents__pydantic_ai: its members do not share an interpreter"
        in made["unplanned"]
    )


# --- beside another instrumentation library ---------------------------------------


def test_a_neighbour_point_installs_the_library_and_is_checked_by_the_driver():
    points = _of(_plan(), "neighbour")
    assert sorted(points) == sorted(f"neighbour--{name}" for name in manifest.NEIGHBOURS)
    for name, declared in manifest.NEIGHBOURS.items():
        point = points[f"neighbour--{name}"]
        spec = manifest.ADAPTERS[declared["adapter"]]
        assert point["subject"] == declared["adapter"] and point["adapter"] == declared["adapter"]
        assert set(declared["requires"]) <= set(point["install"]["requires"])
        assert set(spec["requires"]) <= set(point["install"]["requires"])
        assert point["install"]["optional"] == []
        assert point["check"]["driver"] == manifest.NEIGHBOUR_DRIVER
        assert point["check"]["args"][0] == name
        assert point["probe"]["adapter"] == declared["adapter"], "it still has to attach"


# --- a pre-release ----------------------------------------------------------------


def test_a_prerelease_ahead_of_the_newest_final_is_run_and_one_behind_it_is_not():
    ahead = {"openai-agents": [*_world()[1]["openai-agents"], _release("1.1.0rc1", "2026-11-20")]}
    made = _plan(releases=ahead)
    (point,) = _of(made, "prerelease").values()
    assert point["key"] == "prerelease--openai_agents" and point["role"] == "prerelease"
    assert point["targets"] == {"openai-agents": "1.1.0rc1"}
    assert point["in_claim"] is False and point["assume_measured"] is True
    assert point["check"]["how"] == "suite" and point["check"]["files"]
    behind = {"openai-agents": [*_world()[1]["openai-agents"], _release("1.0.1rc1", "2026-10-20")]}
    assert _of(_plan(releases=behind), "prerelease") == {}
    late = {"openai-agents": [*_world()[1]["openai-agents"], _release("1.1.0rc1", "2026-11-27")]}
    assert _of(_plan(releases=late), "prerelease") == {}, "published after the run's instant"
    unsupported = _release("1.1.0rc1", "2026-11-20", python=">=3.99")
    nobody = _plan(releases={"openai-agents": [*_world()[1]["openai-agents"], unsupported]})
    assert _of(nobody, "prerelease") == {}
    assert (
        "prerelease: openai_agents: 1.1.0rc1 admits no supported interpreter" in nobody["unplanned"]
    )
    assert "openai_agents" not in nobody["scope"]["covers"]["prerelease"]


# --- the programs themselves -------------------------------------------------------


def test_the_program_inside_a_wheel_is_run_at_the_newest_tested_release():
    point = _of(_plan(), "cli")["cli--anthropic_agent_sdk"]
    assert point["targets"] == {"claude-agent-sdk": "1.0.1"} and point["in_claim"] is True
    assert point["check"]["args"][0] == "claude" and point["check"]["evidence"] == "cli-live"
    assert point["check"]["replay_of"] is None, "nothing is replayed here"
    assert "npm" not in point["install"]


def test_an_npm_program_is_run_at_the_release_the_readme_names_and_at_the_newest():
    points = _of(_plan(), "cli")
    named, newest = points["cli--codex_exec--claim_top"], points["cli--codex_exec--beyond"]
    assert named["targets"] == {"@openai/codex": "0.160.0"} and named["in_claim"] is True
    assert newest["targets"] == {"@openai/codex": "0.172.0"} and newest["in_claim"] is False
    for point in (named, newest):
        assert point["install"]["npm"] == point["targets"]
        assert point["probe"] == {"imports": [], "adapter": None}, "there is no package to probe"
        assert point["subject"] == "codex_exec" and point["check"]["args"][0] == "codex"
    # A pre-release of the program, and one published after the run's instant, are not it.
    only = _facts(
        npm={"@openai/codex": [{"version": "0.160.0", "uploaded": "2026-10-01T00:00:00Z"}]}
    )
    assert sorted(_of(_plan(facts=only), "cli")) == [
        "cli--anthropic_agent_sdk",
        "cli--codex_exec--claim_top",
    ]


def test_a_registry_that_cannot_be_read_or_a_named_release_that_is_gone_is_said():
    silent = _plan(facts=_facts(npm={}))
    assert "cli--anthropic_agent_sdk" in _of(silent, "cli") and len(_of(silent, "cli")) == 1
    assert silent["unplanned"] == [
        "cli: codex_exec: the npm registry could not be read for @openai/codex"
    ]
    assert silent["scope"]["covers"]["cli"] == ["anthropic_agent_sdk"]
    gone = _facts(
        npm={"@openai/codex": [{"version": "0.172.0", "uploaded": "2026-11-24T00:00:00Z"}]}
    )
    made = _plan(facts=gone)
    assert sorted(_of(made, "cli")) == ["cli--anthropic_agent_sdk", "cli--codex_exec--beyond"]
    assert made["unplanned"] == [
        "cli: codex_exec: @openai/codex 0.160.0, the release named as tested, is not listed"
    ]
    assert made["problems"] == [], "never a reason for red"
    assert made["scope"]["covers"]["cli"] == ["anthropic_agent_sdk"], "its rows are kept"


# --- a Hermes release tag ----------------------------------------------------------


def test_hermes_is_run_at_the_named_tag_and_at_the_newest_release_by_when_it_was_published():
    points = _of(_plan(), "hermes")
    named, newest = points["hermes--claim_top"], points["hermes--beyond"]
    assert named["targets"] == {"hermes-agent": "v0.23.3"} and named["in_claim"] is True
    # Not the canary, not the tag of an older numbering, not a tag that is no release
    # number of three parts, not the one published after the cutoff.
    assert newest["targets"] == {"hermes-agent": "v0.23.5"} and newest["in_claim"] is False
    for point in (named, newest):
        assert point["install"]["how"] == "self", "the driver builds Hermes's own environment"
        assert point["subject"] == "hermes_agent" and point["adapter"] == "hermes_agent"
        assert point["check"]["driver"] == manifest.HERMES_DRIVER
        assert point["check"]["args"][:4] == ["--tag", "{target}", "--wheel", "{wheel}"]
    unread = _plan(facts=_facts(github_releases={}))
    assert _of(unread, "hermes") == {} and unread["scope"]["covers"]["hermes"] == []
    assert unread["unplanned"] == [
        "hermes: the releases of NousResearch/hermes-agent could not be read"
    ]


# --- a development branch, weekly --------------------------------------------------


def test_a_weekly_plan_holds_the_development_branches_and_nothing_of_the_daily_kinds():
    made = _plan("weekly")
    assert {point["kind"] for point in made["points"]} == {"devbranch"}
    assert made["adapters"] == {} and made["problems"] == [], "it speaks for no tested range"
    assert made["scope"]["covers"] == {"devbranch": sorted(manifest.DEV_BRANCHES)}
    assert verdict.judge(made, {}) == [], "a run with only watching kinds owes no point"
    graph = _of(made, "devbranch")["devbranch--langgraph"]
    assert graph["role"] == "devbranch" and graph["in_claim"] is False
    assert graph["targets"] == {
        "langgraph": f"head-{_HEAD[:10]}",
        "langgraph-prebuilt": f"head-{_HEAD[:10]}",
    }
    base = "git+https://github.com/langchain-ai/langgraph@" + _HEAD
    assert graph["install"]["sources"] == {
        "langgraph": f"{base}#subdirectory=libs/langgraph",
        "langgraph-prebuilt": f"{base}#subdirectory=libs/prebuilt",
    }
    assert graph["assume_measured"] is True and graph["check"]["how"] == "suite"
    agents = _of(made, "devbranch")["devbranch--openai_agents"]
    assert agents["install"]["sources"] == {
        "openai-agents": "git+https://github.com/openai/openai-agents-python@" + _HEAD
    }


@pytest.mark.parametrize("head", [None, "", "main", _HEAD[:12], _HEAD + "0", 7])
def test_a_head_that_is_not_a_commit_id_plans_nothing_and_is_said(head):
    heads = {declared["repository"]: head for declared in manifest.DEV_BRANCHES.values()}
    made = _plan("weekly", facts=_facts(heads=heads))
    assert made["points"] == [] and made["scope"]["covers"]["devbranch"] == []
    assert len(made["unplanned"]) == len(manifest.DEV_BRANCHES)
    assert all("could not be read" in line for line in made["unplanned"])


# --- what the plan reads from outside ----------------------------------------------


def test_the_cadence_is_the_one_asked_for_else_the_cron_that_started_the_run():
    assert plan.cadence_of(None, None) == plan.cadence_of("", "") == "daily"
    for cron, cadence in kinds.CRONS.items():
        assert plan.cadence_of(None, cron) == cadence
    assert plan.cadence_of("weekly", "23 19 * * *") == "weekly", "asked for outranks the cron"
    assert plan.cadence_of(None, "0 0 * * *") == "daily", "a cron nobody declared"
    with pytest.raises(SystemExit):
        plan.cadence_of("hourly", None)


def test_the_fetchers_read_what_the_registries_say_and_a_failure_leaves_the_fact_out(monkeypatch):
    asked = []

    def get(url, headers=None):
        asked.append((url, headers or {}))
        if "registry.npmjs.org" in url:
            return {
                "time": {"created": "x", "modified": "y", "0.160.0": "2026-10-01T10:00:00.123Z"}
            }
        return [
            {
                "tag_name": "v0.23.3",
                "published_at": "2026-11-20T00:34:38Z",
                "prerelease": False,
                "draft": False,
            },
            {
                "tag_name": "v9",
                "published_at": "2026-11-21T00:00:00Z",
                "prerelease": False,
                "draft": True,
            },
        ]

    monkeypatch.setattr(plan, "_get", get)
    monkeypatch.setattr(plan, "fetch_head", lambda repository: _HEAD)
    monkeypatch.setenv("GH_TOKEN", "a-token")
    assert plan.fetch_npm("@openai/codex") == [
        {"version": "0.160.0", "uploaded": "2026-10-01T10:00:00Z"}
    ]
    assert asked[-1][0] == "https://registry.npmjs.org/@openai%2Fcodex"
    listed = plan.fetch_github_releases("o/r")
    assert listed == [
        {"tag": "v0.23.3", "published": "2026-11-20T00:34:38Z", "prerelease": False},
        {"tag": "v9", "published": "2026-11-21T00:00:00Z", "prerelease": True},
    ]
    assert asked[-1][1] == {"Authorization": "Bearer a-token"}
    monkeypatch.delenv("GH_TOKEN")
    plan.fetch_github_releases("o/r")
    assert asked[-1][1] == {}, "no token, no header"

    daily = plan.fetch_facts("daily")
    assert set(daily["npm"]) == {"@openai/codex"} and daily["heads"] == {}
    assert set(daily["github_releases"]) == {manifest.HERMES["repository"]}
    weekly = plan.fetch_facts("weekly")
    assert weekly["npm"] == {} and len(weekly["heads"]) == len(manifest.DEV_BRANCHES)

    def broken(*args, **more):
        raise OSError("the network is down")

    monkeypatch.setattr(plan, "_get", broken)
    monkeypatch.setattr(plan, "fetch_head", broken)
    assert plan.fetch_facts("daily") == {"npm": {}, "github_releases": {}, "heads": {}}
    assert plan.fetch_facts("weekly") == {"npm": {}, "github_releases": {}, "heads": {}}


# --- the executor follows the recipe ------------------------------------------------


def _runner(tmp_path, planned: dict, monkeypatch, answers=None) -> tuple[point_runner.Point, list]:
    """A point runner over one planned point, with every command it would run captured.
    `answers(cmd)` may return `(code, out, err)` for a command; the default is success."""
    (tmp_path / "dist").mkdir(parents=True)
    (tmp_path / "dist" / "wardex_sdk-0.0.0-py3-none-any.whl").write_bytes(b"not a wheel")
    (tmp_path / "plan.json").write_text(
        json.dumps({"cutoff": _CUTOFF, "points": [planned]}), "utf-8"
    )
    args = point_runner.argparse.Namespace(
        plan=str(tmp_path / "plan.json"),
        key=planned["key"],
        wheel_dir=str(tmp_path / "dist"),
        work=str(tmp_path / "work"),
        out=str(tmp_path / "out"),
        last_green_lock=None,
        tests_tree=None,
    )
    ran: list = []

    def run(cmd, *, cwd=None, env=None, timeout=900):
        cmd = [str(part) for part in cmd]
        ran.append({"cmd": cmd, "env": env, "timeout": timeout})
        return (answers(cmd) if answers else None) or (0, "", "")

    monkeypatch.setattr(point_runner, "run", run)
    made = point_runner.Point(args)
    made.deadline = point_runner.time.monotonic() + planned["budget_s"]
    return made, ran


def test_a_target_from_a_source_commit_is_required_by_its_address_and_an_npm_one_not_at_all(
    tmp_path, monkeypatch
):
    graph = _of(_plan("weekly"), "devbranch")["devbranch--langgraph"]
    runner, _ = _runner(tmp_path / "a", graph, monkeypatch)
    lines = runner._requirements(()).splitlines()
    assert f"langgraph @ {graph['install']['sources']['langgraph']}" in lines
    assert not any(line.startswith("langgraph==") for line in lines)
    codex = _of(_plan(), "cli")["cli--codex_exec--claim_top"]
    runner, _ = _runner(tmp_path / "b", codex, monkeypatch)
    assert not any("codex" in line for line in runner._requirements(()).splitlines())


def test_a_lock_with_source_lines_is_installed_in_two_steps_and_then_checked(tmp_path, monkeypatch):
    """Index files with their hashes required; the lines from a commit after them, without
    their dependencies; then the whole for consistency. A lock of index files alone is
    installed as before, in one step."""
    graph = _of(_plan("weekly"), "devbranch")["devbranch--langgraph"]
    runner, ran = _runner(tmp_path / "a", graph, monkeypatch)
    lock = runner.work / "lock.txt"
    lock.write_text(
        "anyio==4.0 \\\n    --hash=sha256:aa\n"
        "langgraph @ git+https://example.invalid/x@abc#subdirectory=libs/langgraph\n"
        "numpy==2.0 \\\n    --hash=sha256:bb\n",
        encoding="utf-8",
    )
    assert runner._sync(runner.work / "venv", lock) == (0, "")
    steps = [call["cmd"] for call in ran]
    assert [step[:3] for step in steps] == [
        ["uv", "pip", "sync"],
        ["uv", "pip", "install"],
        ["uv", "pip", "check"],
    ]
    assert "--require-hashes" in steps[0] and "--no-deps" in steps[1]
    index = (runner.work / "lock-index.txt").read_text(encoding="utf-8")
    assert "anyio==4.0" in index and "numpy==2.0" in index and "git+" not in index
    assert (
        (runner.work / "lock-sources.txt")
        .read_text(encoding="utf-8")
        .startswith("langgraph @ git+")
    )

    plain, ran = _runner(tmp_path / "b", graph, monkeypatch)
    lock = plain.work / "lock.txt"
    lock.write_text("anyio==4.0 \\\n    --hash=sha256:aa\n", encoding="utf-8")
    assert plain._sync(plain.work / "venv", lock) == (0, "")
    assert len(ran) == 1 and "--require-hashes" in ran[0]["cmd"]

    def refuse(cmd):
        return (2, "", "not consistent") if cmd[:3] == ["uv", "pip", "check"] else None

    broken, _ = _runner(tmp_path / "c", graph, monkeypatch, refuse)
    lock = broken.work / "lock.txt"
    lock.write_text("x @ git+https://example.invalid/x@abc\n", encoding="utf-8")
    assert broken._sync(broken.work / "venv", lock) == (2, "not consistent")


def test_an_npm_package_is_installed_into_the_work_folder_and_its_programs_put_on_the_path(
    tmp_path, monkeypatch
):
    codex = _of(_plan(), "cli")["cli--codex_exec--claim_top"]
    runner, ran = _runner(tmp_path / "a", codex, monkeypatch)
    manifest_file = runner.work / "npm/node_modules/@openai/codex/package.json"
    manifest_file.parent.mkdir(parents=True)
    manifest_file.write_text(json.dumps({"version": "0.160.0"}), encoding="utf-8")
    assert runner.install_npm() is True
    assert ran[0]["cmd"][:2] == ["npm", "install"] and ran[0]["cmd"][-1] == "@openai/codex@0.160.0"
    assert runner.path == [str(runner.work / "npm/node_modules/.bin")]
    assert runner.result["resolved"] == {"npm:@openai/codex": "0.160.0"}
    manifest_file.write_text(json.dumps({"version": "0.161.0"}), encoding="utf-8")
    with pytest.raises(AssertionError, match="installed 0.161.0"):
        runner.install_npm()

    for said, outcome in (
        ("npm error code E404", "not_installable"),
        ("ECONNRESET", "install_network"),
    ):
        failing, _ = _runner(
            tmp_path / outcome, codex, monkeypatch, lambda cmd, said=said: (1, "", said)
        )
        assert failing.install_npm() is False
        assert failing.result["install"]["outcome"] == outcome


def _driven(tmp_path, monkeypatch, point: dict, *, code: int, verdict: object = None, again=None):
    """Run the driver step of `point` with a driver that exits `code` and leaves `verdict`."""
    calls = []

    def answer(cmd):
        if point["check"]["driver"] not in " ".join(cmd):
            return None
        calls.append(cmd)
        said = again if again is not None and len(calls) > 1 else (code, verdict)
        out = Path(next(a for a in cmd if a.endswith("driver.json")))
        out.parent.mkdir(parents=True, exist_ok=True)
        if said[1] is not None:
            out.write_text(said[1] if isinstance(said[1], str) else json.dumps(said[1]), "utf-8")
        return said[0], "the driver's output", "the driver's errors"

    runner, ran = _runner(tmp_path, point, monkeypatch, answer)
    runner.work.mkdir(parents=True, exist_ok=True)
    return runner, ran, calls


def test_a_driver_that_passes_fails_or_could_not_prepare_gives_the_outcome_it_stands_for(
    tmp_path, monkeypatch
):
    point = _of(_plan(), "hermes")["hermes--claim_top"]
    ok = {"ok": True, "failed": [], "checks": {"held": 7}}
    runner, ran, calls = _driven(tmp_path / "pass", monkeypatch, point, code=0, verdict=ok)
    runner._driven(runner.work / "venv")
    assert runner.result["outcome"] == "pass" and runner.result["tests"]["passed"] == 7
    assert runner.result["tests"]["files"] == [manifest.HERMES_DRIVER]
    assert calls[0][0] == sys.executable, "the check installs for itself: no environment of ours"
    args = calls[0][2:]
    assert args[:2] == ["--tag", "v0.23.3"] and args[3].endswith(
        "wardex_sdk-0.0.0-py3-none-any.whl"
    )
    assert args[5] == f"{runner.work}/hermes" and args[7] == f"{runner.out}/driver.json"

    bad = {"ok": False, "failed": ["the smoke failed in cli mode", "x" * 999], "checks": {}}
    runner, _, calls = _driven(tmp_path / "fail", monkeypatch, point, code=1, verdict=bad)
    runner._driven(runner.work / "venv")
    assert runner.result["outcome"] == "fail" and runner.result["cause"] == "not_determined"
    assert runner.result["tests"]["failed_ids"] == ["the smoke failed in cli mode", "x" * 300]
    assert len(calls) == 2 and runner.result["rerun"] == {"passed": False}, "run once more first"

    for index, odd in enumerate(("abc", ["a name", 7], {"a": 1}, False, 0, "", {})):
        unnamed = {"ok": False, "failed": odd, "checks": {}}
        runner, _, _ = _driven(
            tmp_path / f"odd{index}", monkeypatch, point, code=1, verdict=unnamed
        )
        runner._driven(runner.work / "venv")
        assert runner.result["outcome"] == "fail"
        assert runner.result["tests"]["failed_ids"] == [
            "e2e_hermes_tag.py: its list of failures is not a list of names"
        ], odd

    runner, _, calls = _driven(
        tmp_path / "flaky", monkeypatch, point, code=1, verdict=bad, again=(0, ok)
    )
    runner._driven(runner.work / "venv")
    assert runner.result["outcome"] == "flaky"

    runner, _, calls = _driven(tmp_path / "gone", monkeypatch, point, code=3, verdict={"ok": False})
    runner._driven(runner.work / "venv")
    assert runner.result["outcome"] == "install_network", "nothing wardex or its tests did"
    assert len(calls) == 1

    newer = _of(_plan(), "hermes")["hermes--beyond"]
    runner, _, _ = _driven(tmp_path / "newer", monkeypatch, newer, code=1, verdict=bad)
    runner._driven(runner.work / "venv")
    assert runner.result["cause"] == "new_version", "a release nobody called tested"


@pytest.mark.parametrize(
    ("code", "verdict"),
    [
        (0, None),
        (0, "not json"),
        (0, {"ok": False}),
        (0, [1]),
        (0, {"ok": True, "failed": [1, None]}),
        (0, {"ok": True, "failed": "abc"}),
        (0, {"ok": True, "failed": False}),
        (0, {"ok": True, "failed": 0}),
        (0, {"ok": True, "failed": ""}),
        (0, {"ok": True, "failed": {}}),
        (1, None),
        (2, {"ok": True}),
        (137, None),
    ],
    ids=[
        "no-verdict",
        "not-json",
        "says-not-ok",
        "a-list",
        "failures-not-names",
        "failures-a-string",
        "failures-false",
        "failures-zero",
        "failures-empty-text",
        "failures-empty-object",
        "failed-silently",
        "odd-exit",
        "killed",
    ],
)
def test_a_driver_that_leaves_no_verdict_it_can_stand_behind_is_the_harnesss_own_failure(
    tmp_path, monkeypatch, code, verdict
):
    """Never a pass: a script that exits 0 without saying `ok` proved nothing."""
    point = _of(_plan(), "hermes")["hermes--claim_top"]
    runner, _, _ = _driven(tmp_path, monkeypatch, point, code=code, verdict=verdict)
    with pytest.raises(RuntimeError, match="the driver ended with exit"):
        runner._driven(runner.work / "venv")
    assert runner.result["outcome"] is None


def test_a_driver_that_runs_out_of_time_is_a_failure_with_no_second_run(tmp_path, monkeypatch):
    point = _of(_plan(), "hermes")["hermes--claim_top"]
    runner, _, calls = _driven(tmp_path, monkeypatch, point, code=124, verdict={"ok": True})
    runner._driven(runner.work / "venv")
    assert runner.result["outcome"] == "fail" and runner.result["tests"]["timed_out"] is True
    assert len(calls) == 1


def test_every_adapter_of_a_point_is_probed_and_one_that_does_not_attach_decides(
    tmp_path, monkeypatch
):
    pair = _of(_plan(), "combination")["combination--openai_agents__pydantic_ai"]
    seen = {"state": "installed", "measured": True, "in_table": True, "lines_said": 0}

    def answer(cmd):
        if "attach_probe.py" not in " ".join(cmd):
            return None
        wrong = {**seen, "state": "waiting_for_import"} if cmd[2] == "pydantic_ai" else seen
        return 0, json.dumps(wrong), ""

    runner, ran = _runner(tmp_path, pair, monkeypatch, answer)
    assert runner.attach(runner.work / "venv") is False
    probes = [call["cmd"][2] for call in ran if "attach_probe.py" in " ".join(call["cmd"])]
    assert probes == ["openai_agents", "pydantic_ai"]
    assert runner.result["attach"]["verdict"] == "not_installed"


def test_a_line_about_a_library_named_after_a_framework_is_not_a_line_about_the_framework():
    """wardex says a line when another instrumentation library is installed. The probe
    counted it as the adapter speaking on a tested release, because the library's name
    ends in the framework's."""
    from compat import attach_probe

    library = "openinference-instrumentation-openai-agents is installed. If it instruments ..."
    own = "openai-agents 0.26.0 is outside the range wardex was tested on (0.22.x, 0.23.x)"
    lines = [
        library,
        own,
        "pin openai-agents<0.24 to get back",
        "my-openai-agents and openai-agents2",
    ]
    assert attach_probe.about(lines, ["openai-agents"]) == [own, lines[2]]
    assert attach_probe.about([library], ["openai-agents"]) == []
    assert attach_probe.about(
        ["langgraph-prebuilt 1.1.3"], ["langgraph", "langgraph-prebuilt"]
    ) == ["langgraph-prebuilt 1.1.3"]
    assert attach_probe.about(["(openai-agents)", "openai-agents."], ["openai-agents"]) == [
        "(openai-agents)",
        "openai-agents.",
    ]


# --- the report: one issue per subject, written by one run --------------------------


def _result(point: dict, outcome: str = "pass", **more: object) -> dict:
    return {**point, "outcome": outcome, "cause": None, "planned": point, **more}


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
    "cadence": "daily",
}


def test_a_subject_that_is_not_an_adapter_has_an_issue_of_its_own():
    made = _plan()
    results = {p["key"]: _result(p) for p in made["points"]}
    failing = "combination--openai_agents__pydantic_ai"
    results[failing] = _result(results[failing], "fail", tests={"failed_ids": ["t.py::test_x"]})
    results["hermes--beyond"] = _result(results["hermes--beyond"], "fail", tests={"failed_ids": []})
    wanted = {issue["key"]: issue for issue in report.wanted_issues(made, results, {}, _RUN)}
    assert set(manifest.ADAPTERS) | set(manifest.COMBINATIONS) | {
        "codex_exec",
        "hermes_agent",
    } <= set(wanted)
    pair = wanted["openai_agents__pydantic_ai"]
    assert pair["title"] == "Compatibility: openai-agents + pydantic-ai" and pair["open"] is True
    assert "has no row of its own in the table of tested versions" in pair["body"]
    assert "- Tested range:" not in pair["body"]
    assert "### Watching (never turns the run red)" in pair["body"]
    assert (
        "frameworks together: `openai-agents 1.0.1, pydantic-ai-slim 1.0.1` fails 1 tests"
        in pair["body"]
    )
    hermes = wanted["hermes_agent"]
    assert hermes["title"] == "Compatibility: hermes-agent" and hermes["open"] is True
    assert "| Hermes release: newer than the tested range |" in hermes["body"]
    assert wanted["codex_exec"]["open"] is False, "nothing to do: the issue is not opened"
    assert wanted["openai_agents"]["open"] is False
    assert verdict.judge(made, results) == []


def test_a_run_of_another_cadence_writes_no_issue_and_its_findings_wait_in_the_state(tmp_path):
    weekly = _plan("weekly")
    results = {
        p["key"]: _result(p, "fail", tests={"failed_ids": ["t.py::test_y"]})
        for p in weekly["points"]
    }
    run = {**_RUN, "cadence": "weekly", "date": "2026-11-23"}
    assert report.wanted_issues(weekly, results, {}, run) == []
    state = report.next_state({}, results, run, tmp_path, weekly)
    assert sorted(state["points"]) == ["devbranch--langgraph", "devbranch--openai_agents"]
    assert state["last_scheduled"] is None, "the daily schedule is the one the watchdog watches"

    # Three days later the daily run speaks for it, dated, and keeps the issue open.
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    after = report.next_state(state, passing, _RUN, tmp_path, daily)
    assert "devbranch--langgraph" in after["points"], "a daily run forgets nothing of a weekly kind"
    wanted = {issue["key"]: issue for issue in report.wanted_issues(daily, passing, after, _RUN)}
    graph = wanted["langgraph"]
    assert graph["open"] is True
    assert (
        "development branch: `langgraph head-0123456789, langgraph-prebuilt head-0123456789` "
        "fails 1 tests (last run `2026-11-23`)" in graph["body"]
    )


def test_a_failing_check_retired_with_its_kind_is_named_when_its_issue_closes(tmp_path):
    """A kind that could turn the run red is taken out of the declarations while one of its
    points is failing inside a claim. The first whole run forgets the point, and the issue
    it kept open would close as if it had been fixed."""
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    old = {
        "kind": "retired_kind",
        "subject": "openai_agents",
        "adapter": "openai_agents",
        "role": "claim_top",
        "targets": {"openai-agents": "1.0.1"},
        "in_claim": True,
        "watching": False,
    }
    failed = {"date": "2026-11-25", "outcome": "fail", "targets": old["targets"]}
    watched = {**old, "watching": True}
    state = {
        "points": {
            "retired--openai_agents": {"point": old, "last": failed},
            "retired--watched": {"point": watched, "last": failed},
            "retired--passing": {"point": old, "last": {**failed, "outcome": "pass"}},
            "retired--above": {"point": {**old, "in_claim": False}, "last": failed},
            # A point of a kind that is still declared, which the plan no longer makes: the
            # question stopped being asked (the release left the range). Not a retirement.
            "openai_agents--minor_newest--0.9": {
                "point": {**old, "kind": "framework", "role": "minor_newest"},
                "last": failed,
            },
        }
    }
    after = report.next_state(state, passing, _RUN, tmp_path, daily)
    assert not [key for key in after["points"] if key.startswith("retired--")]
    assert "openai_agents--minor_newest--0.9" not in after["points"]
    # Every point that went away with the kind leaves a trace, and the ones that were not
    # passing are named: a watched one and one above the range were on the issue too.
    assert {item["key"]: item["failing"] for item in after["retired"]} == {
        "retired--openai_agents": True,
        "retired--watched": True,
        "retired--passing": False,
        "retired--above": True,
    }
    assert after["retired"][0] == {
        "subject": "openai_agents",
        "key": "retired--openai_agents",
        "kind": "retired_kind",
        "date": "2026-11-25",
        "undeclared": False,
        "failing": True,
    }
    issue = report.adapter_issue("openai_agents", daily, passing, _RUN, after)
    assert issue["open"] is False
    assert issue["closing"].startswith("Nothing here needs doing any more: every release inside")
    for key in ("retired--openai_agents", "retired--watched", "retired--above"):
        assert (
            f"The check `{key}` was not passing when it was last run (`2026-11-25`) and "
            "is no longer run: its kind (`retired_kind`) was retired. That is not a pass."
            in issue["closing"]
        ), key
    assert "retired--passing" not in issue["closing"]
    assert (
        "retired" not in report.adapter_issue("langgraph", daily, passing, _RUN, after)["closing"]
    )
    later = report.next_state(after, passing, _RUN, tmp_path, daily)
    assert later["retired"] == [], "said once, by the run that retired it"


def test_what_a_weekly_run_found_wrong_with_itself_is_on_the_daily_health_issue():
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    state = {
        "health": {
            "weekly": {
                "date": "2026-11-23",
                "marks": [["devbranch--langgraph", "no_result"], "x", ["a"]],
            },
            "daily": {"date": "2026-11-20", "marks": [["old", "stale"]]},
        },
        "kinds": {"devbranch": {"last_complete": "2026-11-23", "points": 2}},
    }
    health = report.health_issue(daily, passing, state, _RUN)
    assert health["changes"] == [
        "From the weekly run of `2026-11-23`: `devbranch--langgraph`: `no_result`"
    ]
    assert health["open"] is True and health["marks"] == []
    quiet = report.health_issue(daily, passing, {"kinds": state["kinds"]}, _RUN)
    assert quiet["open"] is False


@pytest.mark.parametrize(
    ("last", "stale"), [("2026-11-16", False), ("2026-11-15", True), (None, False), (7, False)]
)
def test_a_cadence_that_has_stopped_running_is_said_by_the_one_that_still_runs(last, stale):
    """Ten days is how old a weekly result may be. Never completed is no day to count from."""
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    # A kind of this run's own cadence is never called stale by the run that runs it.
    state = {
        "kinds": {
            "devbranch": {"last_complete": last},
            "combination": {"last_complete": "2026-01-01"},
        }
    }
    health = report.health_issue(daily, passing, state, _RUN)
    assert health["open"] is stale
    if stale:
        assert health["changes"] == [
            "No whole run has run the development branch points since `2026-11-15` (11 days). "
            "Their schedule may have stopped."
        ]


def test_frameworks_that_cannot_be_installed_together_are_said_with_what_the_installer_said():
    made = _plan()
    results = {p["key"]: _result(p) for p in made["points"]}
    key = "combination--openai_agents__pydantic_ai"
    refused = "No solution found when resolving dependencies: openai>=3 and openai<1"
    printed = "error: " + refused.replace(": ", ":\n  cause: ", 1) + "\n"
    refused = " ".join(printed.split())
    results[key] = _result(
        results[key], "not_installable", install={"outcome": "not_installable", "detail": printed}
    )
    issue = report.adapter_issue("openai_agents__pydantic_ai", made, results, _RUN, {})
    assert issue["open"] is True
    assert (
        "frameworks together: `openai-agents 1.0.1, pydantic-ai-slim 1.0.1` cannot be installed "
        f"together (the installer said: `{refused}`)" in issue["body"]
    )
    assert "| cannot be installed together |" in issue["body"], "in the table of points too"
    alone = report.adapter_issue("langgraph", made, results, _RUN, {})
    assert "cannot be installed together" not in alone["body"]
    (line,) = [row for row in report.records(made, results, _RUN) if row["key"] == key]
    assert line["outcome"] == "not_installable" and line["install_detail"] == refused
    assert verdict.judge(made, results) == []


@pytest.mark.parametrize("kind", _NEW)
def test_promoting_a_kind_is_one_line_and_only_then_can_its_points_turn_the_run_red(
    kind, monkeypatch
):
    ahead = {"openai-agents": [*_world()[1]["openai-agents"], _release("1.1.0rc1", "2026-11-20")]}
    cadence = "weekly" if kind == "devbranch" else "daily"

    def lost() -> tuple[dict, dict]:
        made = _plan(cadence, releases=ahead)
        mine = [p for p in made["points"] if p["kind"] == kind]
        assert mine, kind
        results = {p["key"]: _result(p) for p in made["points"]}
        results[mine[0]["key"]] = _result(mine[0], "no_result")
        return made, results

    assert verdict.judge(*lost()) == [], "watching: its lost result is reported, not red"
    monkeypatch.setattr(kinds, "RED_CAPABLE", frozenset({"framework", kind}))
    made, results = lost()
    assert all(not p["watching"] for p in made["points"] if p["kind"] == kind)
    assert len(verdict.judge(made, results)) == 1, "the same lost result, once the line is there"


# --- the public page ---------------------------------------------------------------


def test_the_page_says_which_sentences_a_watching_kind_proves_and_names_the_new_rows():
    from compat import render

    proofs = {claim["says"]: claim["proof"] for claim in manifest.CLAIMS}
    assert proofs["Verified against `codex-cli` 0.160.0."] == "cli"
    assert proofs["Hermes Agent (through a Hermes plugin; measured on v0.23.3)"] == "hermes"
    assert all(proof is None or proof in kinds.KINDS for proof in proofs.values())
    lines = render._claims(
        [
            {"says": "a", "proof": "framework", "why": ""},
            {"says": "b", "proof": "cli", "why": ""},
            {"says": "c", "proof": "", "why": "nothing runs it"},
        ]
    )
    assert "| a | framework release points, daily |" in lines
    assert "| b | live program points, daily (watching) |" in lines
    assert "| c | nothing yet: nothing runs it |" in lines
    for role in ("prerelease", "devbranch"):
        assert role in render._ROLE_WORDS and role in report.ROLE_WORDS


# --- what the first reader of this change found --------------------------------------


def test_the_plan_says_every_subject_each_kind_declares_whatever_it_could_plan():
    """A fact of the manifest, the same on every run: a daily plan says what the weekly
    kind declares too, since the daily run is the one that forgets and closes."""
    made = _plan(facts=_facts(npm={}, github_releases={}))
    declared = made["scope"]["declared"]
    assert declared == plan.declared() == _plan("weekly")["scope"]["declared"]
    assert declared == {
        "framework": sorted(manifest.ADAPTERS),
        "published": sorted(manifest.ADAPTERS),
        "combination": sorted(manifest.COMBINATIONS),
        "neighbour": sorted({d["adapter"] for d in manifest.NEIGHBOURS.values()}),
        "prerelease": sorted(manifest.ADAPTERS),
        "cli": sorted(manifest.CLI),
        "hermes": ["hermes_agent"],
        "devbranch": sorted(manifest.DEV_BRANCHES),
    }
    assert set(declared) == set(kinds.KINDS), "every kind says what it declares"
    assert made["scope"]["covers"]["cli"] == ["anthropic_agent_sdk"], "not what it covered"
    assert made["scope"]["covers"]["hermes"] == []


_GONE = {
    "kind": "combination",
    "subject": "a_set_no_longer_declared",
    "adapter": "openai_agents",
    "adapters": ["openai_agents", "langgraph"],
    "role": "claim_top",
    "targets": {"openai-agents": "1.0.1"},
    "in_claim": True,
    "watching": True,
}


@pytest.mark.parametrize("kind", ["combination", "devbranch"])
def test_a_subject_taken_out_of_the_declarations_is_forgotten_and_its_issue_closes(tmp_path, kind):
    """A set of frameworks is removed from the manifest while its point is failing. No run
    plans it again, so no run covers it, and its issue would stay open for good. The same
    for a branch only the weekly run looked at: the daily run is the one that closes."""
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    failed = {"date": "2026-11-25", "outcome": "fail", "targets": _GONE["targets"]}
    key = f"{kind}--a_set_no_longer_declared"
    state = {"points": {key: {"point": {**_GONE, "kind": kind}, "last": failed}}}
    before = {i["key"]: i for i in report.wanted_issues(daily, passing, state, _RUN)}
    assert before["a_set_no_longer_declared"]["open"] is True, "it is on an issue until then"
    # A run of one adapter and the weekly run leave it: neither closes issues.
    partial = {**daily, "partial": True, "scope": {**daily["scope"], "partial": True}}
    assert key in report.next_state(state, passing, _RUN, tmp_path, partial)["points"]
    weekly = _plan("weekly")
    week = {p["key"]: _result(p) for p in weekly["points"]}
    kept = report.next_state(state, week, {**_RUN, "cadence": "weekly"}, tmp_path, weekly)
    assert key in kept["points"] and kept["retired"] == []
    # The daily run forgets it and, in the same run, has a say about its issue: closed.
    after = report.next_state(kept, passing, _RUN, tmp_path, daily)
    assert key not in after["points"]
    wanted = {i["key"]: i for i in report.wanted_issues(daily, passing, after, _RUN)}
    issue = wanted["a_set_no_longer_declared"]
    assert issue["open"] is False
    assert issue["closing"] == (
        "Nothing here is asked any more: no run plans a check of `a_set_no_longer_declared` "
        f"again. The check `{key}` was not passing when it was last run (`2026-11-25`) and is "
        "no longer run: nothing declares it any more. That is not a pass. Closing."
    )
    # The run after that knows nothing of it, and has nothing to say.
    later = report.next_state(after, passing, {**_RUN, "id": "2"}, tmp_path, daily)
    assert later["retired"] == []
    assert "a_set_no_longer_declared" not in {
        i["key"] for i in report.wanted_issues(daily, passing, later, _RUN)
    }


def test_a_passing_subject_taken_out_of_the_declarations_closes_without_a_word_of_failure(
    tmp_path,
):
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    last = {"date": "2026-11-25", "outcome": "pass", "targets": _GONE["targets"]}
    state = {"points": {"combination--a_set_no_longer_declared": {"point": _GONE, "last": last}}}
    after = report.next_state(state, passing, _RUN, tmp_path, daily)
    wanted = {i["key"]: i for i in report.wanted_issues(daily, passing, after, _RUN)}
    closing = wanted["a_set_no_longer_declared"]["closing"]
    assert closing.startswith("Nothing here is asked any more") and "not passing" not in closing
    # A plan written before declarations were said forgets nothing it cannot be sure of.
    old = {**daily, "scope": {k: v for k, v in daily["scope"].items() if k != "declared"}}
    assert (
        "combination--a_set_no_longer_declared"
        in report.next_state(state, passing, _RUN, tmp_path, old)["points"]
    )


def test_a_subject_the_plan_could_not_plan_today_keeps_its_issue_from_the_state(tmp_path):
    """The registry cannot be read, so today's plan holds no point of the program. What it
    said yesterday is still what is known, and still on its issue."""
    yesterday = _plan()
    results = {p["key"]: _result(p) for p in yesterday["points"]}
    key = "cli--codex_exec--claim_top"
    results[key] = _result(results[key], "fail", tests={"failed_ids": ["one execute_tool span"]})
    state = report.next_state({}, results, {**_RUN, "date": "2026-11-25"}, tmp_path, yesterday)
    today = _plan(facts=_facts(npm={}))
    passing = {p["key"]: _result(p) for p in today["points"]}
    assert not [p for p in today["points"] if p["subject"] == "codex_exec"]
    after = report.next_state(state, passing, _RUN, tmp_path, today)
    assert key in after["points"], "declared and not covered today, so not forgotten"
    wanted = {i["key"]: i for i in report.wanted_issues(today, passing, after, _RUN)}
    assert wanted["codex_exec"]["open"] is True
    assert "(last run `2026-11-25`)" in wanted["codex_exec"]["body"]


def test_only_the_run_that_writes_issues_retires_a_kind(tmp_path):
    """So the issue a retired check kept open hears of it from the run that forgets it."""
    old = {**_GONE, "kind": "retired_kind", "subject": "openai_agents", "watching": False}
    failed = {"date": "2026-11-25", "outcome": "fail", "targets": old["targets"]}
    key = "retired--openai_agents"
    state = {"points": {key: {"point": old, "last": failed}}}
    weekly = _plan("weekly")
    week = {p["key"]: _result(p) for p in weekly["points"]}
    first = report.next_state(state, week, {**_RUN, "cadence": "weekly"}, tmp_path, weekly)
    assert key in first["points"] and not first.get("retired")
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    then = report.next_state(first, passing, _RUN, tmp_path, daily)
    assert key not in then["points"]
    assert [item["key"] for item in then["retired"]] == [key]


def test_a_run_that_did_not_ask_about_the_tested_ranges_does_not_put_its_commit_on_the_page(
    tmp_path,
):
    """The weekly run is at a newer commit than the daily run whose rows the page shows."""
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    state = report.next_state({}, passing, {**_RUN, "sdk_commit": "a" * 40}, tmp_path, daily)
    assert state["versions"]["main"]["commit"] == "a" * 40
    weekly = _plan("weekly")
    week = {p["key"]: _result(p) for p in weekly["points"]}
    newer = {**_RUN, "sdk_commit": "b" * 40, "sdk_version": "0.7.0b3", "date": "2026-11-30"}
    after = report.next_state(state, week, {**newer, "cadence": "weekly"}, tmp_path, weekly)
    main = after["versions"]["main"]
    assert (main["commit"], main["version"], main["date"]) == ("a" * 40, "0.7.0b1", "2026-11-26")
    partial = {**daily, "partial": True, "scope": {**daily["scope"], "partial": True}}
    one = report.next_state(state, passing, newer, tmp_path, partial)
    assert one["versions"]["main"]["commit"] == "a" * 40, "nor does a run of one adapter"
    whole = report.next_state(state, passing, newer, tmp_path, daily)
    assert whole["versions"]["main"]["commit"] == "b" * 40, "the run that ran every row does"
    first = report.next_state({}, week, {**newer, "cadence": "weekly"}, tmp_path, weekly)
    assert first["versions"]["main"]["commit"] == "b" * 40, "with no page yet, it says its own"


def test_what_an_installer_printed_cannot_be_markup_on_the_issue():
    made = _plan()
    results = {p["key"]: _result(p) for p in made["points"]}
    key = "combination--openai_agents__pydantic_ai"
    printed = "error: see [here](https://example.invalid) @someone `x` <!-- compat-key: y -->"
    results[key] = _result(
        results[key], "not_installable", install={"outcome": "not_installable", "detail": printed}
    )
    body = report.adapter_issue("openai_agents__pydantic_ai", made, results, _RUN, {})["body"]
    (said,) = [line for line in body.splitlines() if "the installer said" in line]
    inside = said.split("the installer said: ", 1)[1]
    assert inside.startswith("`error: see [here]") and "`). Closes when" in inside, said
    assert inside.count("`") == 2, "one span, and nothing the installer printed closes it"
    assert "@" not in inside, "nobody is mentioned"
    assert body.count("<!-- compat-key:") == 1


def test_the_summary_has_a_section_for_every_subject_with_a_result():
    made = _plan()
    results = {p["key"]: _result(p) for p in made["points"]}
    text = report.summary(made, results, [], [])
    for subject in {p["subject"] for p in made["points"]}:
        label = report.subject_info(subject, made)["label"]
        assert f"\n### {report.clean(label)}\n" in text, subject
    assert len({report.subject_info(p["subject"], made)["label"] for p in made["points"]}) == 9


def test_a_subject_that_lost_one_declared_check_and_keeps_others_closes_as_what_it_is(tmp_path):
    """The branch of an adapter is taken out of the weekly declarations. Its tested range is
    still run and passes, so the issue closes on that, and says the branch check went away."""
    assert "pydantic_ai" not in manifest.DEV_BRANCHES and "pydantic_ai" in manifest.ADAPTERS
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    branch = {**_GONE, "kind": "devbranch", "subject": "pydantic_ai", "adapter": "pydantic_ai"}
    failed = {"date": "2026-11-23", "outcome": "fail", "targets": {"pydantic-ai-slim": "head-1"}}
    state = {"points": {"devbranch--pydantic_ai": {"point": branch, "last": failed}}}
    after = report.next_state(state, passing, _RUN, tmp_path, daily)
    assert "devbranch--pydantic_ai" not in after["points"]
    issue = report.adapter_issue("pydantic_ai", daily, passing, _RUN, after)
    assert issue["open"] is False
    assert issue["closing"].startswith("Nothing here needs doing any more: every release inside")
    assert (
        "The check `devbranch--pydantic_ai` was not passing when it was last run (`2026-11-23`) "
        "and is no longer run: nothing declares it any more. That is not a pass."
        in issue["closing"]
    )
    assert issue["closing"].endswith("does not contain the fix. Closing.")


def test_a_scoped_npm_package_is_named_by_scope_and_name_with_no_at_sign_on_an_issue():
    made = _plan()
    results = {p["key"]: _result(p) for p in made["points"]}
    text = report.summary(made, results, [], [])
    assert "| `openai/codex 0.160.0` |" in text and "?openai" not in text
    key = "cli--codex_exec--claim_top"
    results[key] = _result(results[key], "fail", tests={"failed_ids": ["one execute_tool span"]})
    body = report.adapter_issue("codex_exec", made, results, _RUN, {})["body"]
    assert "`openai/codex 0.160.0`" in body and "@" not in body and "?openai" not in body


# --- what the second reader found ----------------------------------------------------


def test_a_verdict_that_names_no_failure_or_leaves_the_list_out_is_a_pass(tmp_path, monkeypatch):
    point = _of(_plan(), "hermes")["hermes--claim_top"]
    for index, said in enumerate(({"ok": True}, {"ok": True, "failed": None})):
        verdict = {**said, "checks": {"held": 3}}
        runner, _, _ = _driven(tmp_path / str(index), monkeypatch, point, code=0, verdict=verdict)
        runner._driven(runner.work / "venv")
        assert runner.result["outcome"] == "pass", said


def test_an_adapter_taken_out_of_the_manifest_is_forgotten_and_its_issue_closes(tmp_path):
    """Its points of every kind, the first two included; what an older wardex release
    found stays, as that release's history."""
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    was = {
        "kind": "framework",
        "subject": "an_adapter_that_left",
        "adapter": "an_adapter_that_left",
        "role": "claim_top",
        "targets": {"left": "1.0.1"},
        "in_claim": True,
        "watching": False,
    }
    failed = {"date": "2026-11-25", "outcome": "fail", "targets": was["targets"]}
    shipped = {**was, "kind": "published", "watching": True}
    state = {
        "release": "0.7.0b1",
        "points": {
            "an_adapter_that_left--claim_top": {"point": was, "last": failed},
            "published--0.7.0b1--an_adapter_that_left--claim_top": {
                "point": {**shipped, "sdk": {"origin": "pypi", "version": "0.7.0b1"}},
                "last": failed,
            },
            "published--0.6.0--an_adapter_that_left--claim_top": {
                "point": {**shipped, "sdk": {"origin": "pypi", "version": "0.6.0"}},
                "last": failed,
            },
        },
    }
    daily["scope"]["builds"] = {"published": ["0.7.0b1"]}
    before = {i["key"]: i for i in report.wanted_issues(daily, passing, state, _RUN)}
    assert before["an_adapter_that_left"]["open"] is True
    after = report.next_state(state, passing, _RUN, tmp_path, daily)
    assert [key for key in after["points"] if "that_left" in key] == [
        "published--0.6.0--an_adapter_that_left--claim_top"
    ]
    wanted = {i["key"]: i for i in report.wanted_issues(daily, passing, after, _RUN)}
    issue = wanted["an_adapter_that_left"]
    assert issue["open"] is False
    assert issue["closing"].startswith(
        "Nothing here is asked any more: no run plans a check of `an_adapter_that_left` again."
    )
    assert issue["closing"].count("nothing declares it any more. That is not a pass.") == 2
    # An adapter the index would not list today is still declared, and nothing is forgotten.
    assert "openai_agents" in daily["scope"]["declared"]["framework"]


def test_a_subject_whose_only_checks_went_with_their_kind_has_its_issue_closed(tmp_path):
    """A watching kind is retired while its point fails, and its subject has no point of
    any other kind. Nothing would ever touch its issue again."""
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    only = {
        "kind": "a_kind_that_was_retired",
        "subject": "a_program",
        "adapter": "a_program",
        "role": "claim_top",
        "targets": {"a-program": "v1.0.0"},
        "in_claim": True,
        "watching": True,
    }
    failed = {"date": "2026-11-25", "outcome": "fail", "targets": only["targets"]}
    state = {"points": {"gone--a_program": {"point": only, "last": failed}}}
    after = report.next_state(state, passing, _RUN, tmp_path, daily)
    assert after["points"].get("gone--a_program") is None
    wanted = {i["key"]: i for i in report.wanted_issues(daily, passing, after, _RUN)}
    issue = wanted["a_program"]
    assert issue["open"] is False
    assert issue["closing"] == (
        "Nothing here is asked any more: no run plans a check of `a_program` again. The check "
        "`gone--a_program` was not passing when it was last run (`2026-11-25`) and is no longer "
        "run: its kind (`a_kind_that_was_retired`) was retired. That is not a pass. Closing."
    )
    # With a trace of each sort and no row left, both are said.
    both = {
        "points": {
            "gone--a_program": {"point": only, "last": failed},
            "combination--a_program": {"point": {**only, "kind": "combination"}, "last": failed},
        }
    }
    after = report.next_state(both, passing, _RUN, tmp_path, daily)
    closing = report.adapter_issue("a_program", daily, passing, _RUN, after)["closing"]
    assert closing.startswith("Nothing here is asked any more")
    assert "was retired. That is not a pass." in closing
    assert "nothing declares it any more. That is not a pass." in closing
    # A run of one adapter has no say about it.
    partial = {**daily, "partial": True, "scope": {**daily["scope"], "partial": True}}
    assert "a_program" not in {
        i["key"] for i in report.wanted_issues(partial, passing, after, _RUN)
    }


def test_the_page_says_which_rows_another_run_left_at_another_commit(tmp_path):
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    state = report.next_state({}, passing, {**_RUN, "sdk_commit": "a" * 40}, tmp_path, daily)
    assert state["points"]["openai_agents--floor"]["last"]["commit"] == "a" * 40
    page = render.page(state, render.MAIN, "2026-11-26")
    assert "another commit" not in page
    weekly = _plan("weekly")
    week = {p["key"]: _result(p) for p in weekly["points"]}
    later = {**_RUN, "cadence": "weekly", "sdk_commit": "b" * 40, "date": "2026-11-30"}
    state = report.next_state(state, week, later, tmp_path, weekly)
    page = render.page(state, render.MAIN, "2026-11-30")
    assert "commit `aaaaaaaaaa`)." in page
    assert (
        "2 of the rows below were last run by another run, at another commit of `main`: "
        "`devbranch--langgraph` at `bbbbbbbbbb`, `devbranch--openai_agents` at `bbbbbbbbbb`."
    ) in page
    # Rows from before the commit was kept say nothing, and are not counted.
    for entry in state["points"].values():
        entry["last"].pop("commit", None)
    assert "another commit" not in render.page(state, render.MAIN, "2026-11-30")


def test_a_set_that_has_not_installed_for_days_is_said_in_its_own_words_on_the_health_issue():
    made = _plan()
    results = {p["key"]: _result(p) for p in made["points"]}
    key = "combination--openai_agents__pydantic_ai"
    results[key] = _result(results[key], "not_installable", install={"detail": "no solution"})
    state = {"points": {key: {"not_ours_days": 2}}}
    body = report.health_issue(made, results, state, _RUN)["body"]
    assert (
        f"`{key}`: `openai-agents 1.0.1, pydantic-ai-slim 1.0.1` cannot be installed together, "
        "on 2 days running." in body
    )
    assert "on this Python" not in body


# --- what the third reader found -----------------------------------------------------


def test_a_check_that_leaves_while_its_issue_stays_open_is_said_on_it_that_day(tmp_path):
    """Something else keeps the issue open, and the trace is of one run: said in the body
    that day and in a comment of its own, or the later closing would never say it."""
    daily = _plan()
    results = {p["key"]: _result(p) for p in daily["points"]}
    key = "pydantic_ai--claim_top"
    results[key] = _result(results[key], "fail", tests={"failed_ids": ["t.py::test_x"]})
    assert "pydantic_ai" not in manifest.DEV_BRANCHES
    left = {**_GONE, "kind": "devbranch", "subject": "pydantic_ai", "adapter": "pydantic_ai"}
    state = {
        "points": {
            "devbranch--pydantic_ai": {
                "point": left,
                "last": {"date": "2026-11-25", "outcome": "not_installable", "targets": {}},
            }
        }
    }
    after = report.next_state(state, results, _RUN, tmp_path, daily)
    assert [(item["key"], item["failing"]) for item in after["retired"]] == [
        ("devbranch--pydantic_ai", True)
    ], "a check that could not be installed was not passing either"
    issue = report.adapter_issue("pydantic_ai", daily, results, _RUN, after)
    sentence = (
        "The check `devbranch--pydantic_ai` was not passing when it was last "
        "run (`2026-11-25`) and is no longer run: nothing declares it any more. That is not a "
        "pass."
    )
    assert issue["open"] is True and issue["left"] == [sentence]
    assert f"### No longer run\n\n- {sentence}" in issue["body"]
    done: list[tuple] = []
    gh = report.GitHub("o/r", write=False)
    gh.do = lambda what, method, path, payload: done.append((what, payload))  # type: ignore[method-assign]
    existing = [{"number": 7, "state": "open", "body": issue["body"]}]
    report.sync_issue(gh, issue, existing)
    assert [what for what, _ in done] == ["rewrite #7", "comment on #7"]
    assert done[-1][1]["body"] == f"No longer run:\n\n- {sentence}"
    # The day after, the trace is gone and nothing is said twice.
    later = report.next_state(after, results, {**_RUN, "id": "2"}, tmp_path, daily)
    again = report.adapter_issue("pydantic_ai", daily, results, _RUN, later)
    assert again["left"] == [] and "No longer run" not in again["body"]
    # An issue that closes says it in its closing words, and has no second comment.
    passing = {p["key"]: _result(p) for p in daily["points"]}
    closed = report.adapter_issue("pydantic_ai", daily, passing, _RUN, after)
    assert closed["open"] is False and sentence in closed["closing"]
    assert "No longer run" not in closed["body"]


def test_what_an_older_release_found_stays_on_a_day_the_newest_cannot_be_read(tmp_path):
    """An adapter has left the manifest. With no release planned today the run cannot tell
    which release is the current one, and every remembered one is history to it."""
    daily = _plan()
    assert daily["scope"]["builds"] == {}, "no published release was planned"
    passing = {p["key"]: _result(p) for p in daily["points"]}
    was = {**_GONE, "kind": "published", "subject": "an_adapter_that_left", "adapter": "x"}
    last = {"date": "2026-11-25", "outcome": "pass", "targets": {"x": "1"}}
    key = "published--0.6.0--an_adapter_that_left--claim_top"
    state = {
        "release": "0.7.0b1",
        "points": {
            key: {"point": {**was, "sdk": {"origin": "pypi", "version": "0.6.0"}}, "last": last}
        },
    }
    after = report.next_state(state, passing, _RUN, tmp_path, daily)
    assert key in after["points"] and after["retired"] == []
    # A point of the build itself has no release to be the history of, and goes.
    own = {
        "points": {
            "an_adapter_that_left--floor": {"point": {**was, "kind": "framework"}, "last": last}
        }
    }
    assert (
        "an_adapter_that_left--floor"
        not in report.next_state(own, passing, _RUN, tmp_path, daily)["points"]
    )


def test_the_page_that_stands_in_for_an_unreleased_version_names_other_commits_too(tmp_path):
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    state = report.next_state({}, passing, {**_RUN, "sdk_commit": "a" * 40}, tmp_path, daily)
    weekly = _plan("weekly")
    week = {p["key"]: _result(p) for p in weekly["points"]}
    later = {**_RUN, "cadence": "weekly", "sdk_commit": "b" * 40, "date": "2026-11-30"}
    state = report.next_state(state, week, later, tmp_path, weekly)
    page = render.page(state, render.MAIN, "2026-11-30", standing_in_for="0.7.0b2")
    assert (
        "No row here is a result of the published wheel. 2 of the rows below were last run" in page
    )
    assert "`devbranch--langgraph` at `bbbbbbbbbb`" in page
    # More than a dozen are counted, and the first dozen named.
    for index in range(11):
        entry = json.loads(json.dumps(state["points"]["devbranch--langgraph"]))
        state["points"][f"devbranch--z{index:02}"] = entry
    page = render.page(state, render.MAIN, "2026-11-30")
    assert (
        "13 of the rows below" in page and "`devbranch--z09` at `bbbbbbbbbb`, and 1 more." in page
    )
    assert "devbranch--z10` at" not in page.split("\n")[2]
    del state["points"]["devbranch--z10"]
    page = render.page(state, render.MAIN, "2026-11-30")
    assert "12 of the rows below" in page and "`devbranch--z09` at `bbbbbbbbbb`." in page
    # The page of a published release is of that release, and says nothing of the sort.
    shipped = {**_GONE, "sdk": {"origin": "pypi", "version": "0.7.0b1"}}
    state["points"]["published--0.7.0b1--x"] = {"point": shipped, "last": {"commit": "c" * 40}}
    assert "another commit" not in render.page(state, "0.7.0b1", "2026-11-30")


def _synced(issue: dict, existing: list[dict]) -> list[tuple]:
    done: list[tuple] = []
    gh = report.GitHub("o/r", write=False)
    gh.do = lambda what, method, path, payload: done.append((what, payload))  # type: ignore[method-assign]
    report.sync_issue(gh, issue, existing)
    return done


def test_a_wheel_that_did_not_build_opens_no_issue_for_what_is_watched(tmp_path):
    """Every point left no result: the run's own failure, on the health issue. A watching
    point that left none says nothing about its subject, so nothing is opened for a set of
    frameworks, a program or a release tag."""
    made = _plan()
    nothing = {p["key"]: _result(p, "no_result") for p in made["points"]}
    wanted = {i["key"]: i for i in report.wanted_issues(made, nothing, {}, _RUN)}
    for subject in ("openai_agents__pydantic_ai", "all_frameworks", "codex_exec", "hermes_agent"):
        issue = wanted[subject]
        assert issue["open"] is False and issue["changes"] == [], subject
        assert _synced(issue, []) == [], "nothing is opened"
    assert wanted["openai_agents"]["open"] is True, "a tested range with no proof is said"
    assert "cli--codex_exec--claim_top` left no usable result" in wanted["health"]["body"]
    line = (
        "- live program: `openai/codex 0.160.0` could not be checked (no result), so nothing "
        "is known of it."
    )
    body = wanted["codex_exec"]["body"]
    assert f"### Not checked today\n\n{line}" in body and "### Watching" not in body
    broke = {
        **nothing,
        "cli--codex_exec--claim_top": _result(
            nothing["cli--codex_exec--claim_top"], "harness_error"
        ),
    }
    assert report.adapter_issue("codex_exec", made, broke, _RUN, {})["open"] is False


def test_a_day_a_failed_release_could_not_be_checked_keeps_its_line(tmp_path):
    """Yesterday a watching point failed. Today it left no result, on the same release.
    What it said then still stands, and is the same thing to say: no comment, no close."""
    made = _plan()
    key = "cli--codex_exec--claim_top"
    failed = {p["key"]: _result(p) for p in made["points"]}
    failed[key] = _result(failed[key], "fail", tests={"failed_ids": ["one execute_tool span"]})
    state = report.next_state({}, failed, {**_RUN, "date": "2026-11-25"}, tmp_path, made)
    yesterday = report.adapter_issue("codex_exec", made, failed, _RUN, state)
    assert yesterday["open"] is True
    opened = [{"number": 7, "state": "open", "body": yesterday["body"]}]
    nothing = {p["key"]: _result(p, "no_result") for p in made["points"]}
    after = report.next_state(state, nothing, _RUN, tmp_path, made)
    same = report.adapter_issue("codex_exec", made, nothing, _RUN, after)
    assert same["open"] is True
    assert (
        "live program: `openai/codex 0.160.0` did not pass when it was last checked, on "
        "`2026-11-25`, and could not be checked since (no result)." in same["body"]
    )
    assert same["fingerprint"] == yesterday["fingerprint"]
    said = [line for line in same["body"].splitlines() if line.startswith("- live program: ")]
    assert len([line for line in said if "`openai/codex 0.160.0`" in line]) == 1, said
    assert len(said) == 2, "and the other point, which passed and left no result, has its own"
    assert [what for what, _ in _synced(same, opened)] == ["rewrite #7"]
    # What it last said of another release is not about this one, however the state came
    # to hold it.
    stale = json.loads(json.dumps(after))
    for word in ("last_checked", "last_said"):
        stale["points"][key][word]["targets"] = {"@openai/codex": "0.150.0"}
    assert report.adapter_issue("codex_exec", made, nothing, _RUN, stale)["open"] is False
    # A state written before what a watched point found was kept still has the day it ran.
    older = json.loads(json.dumps(after))
    del older["points"][key]["last_said"]
    assert report.adapter_issue("codex_exec", made, nothing, _RUN, older) == same


def test_an_issue_that_closes_names_what_could_not_be_checked(tmp_path):
    """A watching point failed, moved to another release, and left no result. The failure
    was of a release it no longer runs, and nothing is known of the new one: the issue
    closes, as it would had the new release passed, and its closing words say that nothing
    was checked. They are not the words of a fix. A failure of the new release opens it."""
    made = _plan()
    key = "cli--codex_exec--claim_top"
    failed = {p["key"]: _result(p) for p in made["points"]}
    failed[key] = _result(failed[key], "fail", tests={"failed_ids": ["one execute_tool span"]})
    state = report.next_state({}, failed, {**_RUN, "date": "2026-11-25"}, tmp_path, made)
    yesterday = report.adapter_issue("codex_exec", made, failed, _RUN, state)
    opened = [{"number": 7, "state": "open", "body": yesterday["body"]}]
    nothing = {p["key"]: _result(p, "no_result") for p in made["points"]}
    moved = {**nothing, key: {**nothing[key], "targets": {"@openai/codex": "0.161.0"}}}
    after = report.next_state(state, moved, _RUN, tmp_path, made)
    assert "last_checked" not in after["points"][key] and "last_said" not in after["points"][key]
    other = report.adapter_issue("codex_exec", made, moved, _RUN, after)
    assert other["open"] is False
    assert other["closing"] == (
        "Nothing here is known to need doing, and nothing of it could be checked today: "
        "`openai/codex 0.161.0` (no result); `openai/codex 0.172.0` (no result). It opens "
        "again if what is checked next does not pass. Closing."
    )
    done = _synced(other, opened)
    assert [what for what, _ in done] == ["rewrite #7", "comment on #7", "close #7"]
    assert done[1][1]["body"] == other["closing"]
    # With the other point checked and passing, both are said.
    partly = {**moved, "cli--codex_exec--beyond": _result(moved["cli--codex_exec--beyond"])}
    closing = report.adapter_issue("codex_exec", made, partly, _RUN, after)["closing"]
    assert closing.startswith("Nothing here needs doing any more: every release inside")
    assert "Not checked today: `openai/codex 0.161.0` (no result)." in closing
    # The new release fails: the closed issue opens again, as for any failure.
    fails = {**partly, key: {**failed[key], "targets": {"@openai/codex": "0.161.0"}}}
    later = report.next_state(after, fails, {**_RUN, "id": "3"}, tmp_path, made)
    again = report.adapter_issue("codex_exec", made, fails, _RUN, later)
    closed = [{"number": 7, "state": "closed", "body": other["body"]}]
    assert [what for what, _ in _synced(again, closed)] == [
        "rewrite #7",
        "reopen #7",
        "comment on #7",
    ]
    # And an issue a person closed while it failed opens again when what fails changes, as
    # it always did: a day nothing could be checked in between does not hide that.
    quiet = _synced(other, [{"number": 7, "state": "closed", "body": yesterday["body"]}])
    assert [what for what, _ in quiet] == ["rewrite #7"]
    assert f"<!-- compat-fingerprint: {other['fingerprint']} -->" in quiet[0][1]["body"]


def test_what_the_weekly_run_could_not_check_opens_no_issue_either(tmp_path):
    """The weekly run's wheel did not build and the branch head had moved. Its rows are
    remembered with no result: the daily run, which writes the issues, opens none for
    them."""
    daily, weekly = _plan(), _plan("weekly")
    passing = {p["key"]: _result(p) for p in daily["points"]}
    week = {p["key"]: _result(p, "no_result") for p in weekly["points"]}
    state = report.next_state({}, week, {**_RUN, "cadence": "weekly"}, tmp_path, weekly)
    state = report.next_state(state, passing, {**_RUN, "id": "2"}, tmp_path, daily)
    issue = report.adapter_issue("langgraph", daily, passing, _RUN, state)
    assert issue["open"] is False
    assert (
        "- development branch: `langgraph head-0123456789, langgraph-prebuilt head-0123456789` "
        "could not be checked (last run `2026-11-26`) (no result), so nothing is known of it."
        in issue["body"]
    )
    assert _synced(issue, []) == []
    assert "Not checked today: `langgraph head-0123456789" in issue["closing"]
    # With something else to do, it is an issue like any other, and the line is on it.
    failing = {
        **passing,
        "langgraph--claim_top": _result(
            passing["langgraph--claim_top"], "fail", tests={"failed_ids": ["t.py::test_x"]}
        ),
    }
    busy = report.adapter_issue("langgraph", daily, failing, _RUN, state)
    assert busy["open"] is True
    assert "could not be checked (last run `2026-11-26`) (no result)" in busy["body"]
    assert "could not be checked" not in "".join(busy["changes"]), "and no change to act on"


@pytest.mark.parametrize("today", ["install_network", "not_installable", "no_result"])
@pytest.mark.parametrize("key", ["hermes--beyond", "hermes--claim_top"])
def test_a_watching_release_that_failed_keeps_its_line_on_a_day_it_cannot_be_checked(
    tmp_path, key, today
):
    """Above the range or inside it. Yesterday it failed; today the index was down, or it
    would not install, or the run left nothing. The issue neither closes nor comments: what
    is known is what was known."""
    made = _plan()
    nodes = ["smoke: cli mode {'a': 1} `x`", "y" * 320]
    failed = {p["key"]: _result(p) for p in made["points"]}
    failed[key] = _result(failed[key], "fail", tests={"failed_ids": nodes})
    state = report.next_state({}, failed, {**_RUN, "date": "2026-11-25"}, tmp_path, made)
    yesterday = report.adapter_issue("hermes_agent", made, failed, _RUN, state)
    opened = [{"number": 7, "state": "open", "body": yesterday["body"]}]
    results = {**{p["key"]: _result(p) for p in made["points"]}, key: _result(failed[key], today)}
    after = report.next_state(state, results, _RUN, tmp_path, made)
    issue = report.adapter_issue("hermes_agent", made, results, _RUN, after)
    assert issue["open"] is True
    assert (
        "did not pass when it was last checked, on `2026-11-25`, and could not be" in issue["body"]
    )
    assert issue["fingerprint"] == yesterday["fingerprint"], "the same thing to say"
    assert [what for what, _ in _synced(issue, opened)] == ["rewrite #7"]


def test_a_watching_failure_of_another_sort_is_remembered_as_what_it_was(tmp_path):
    made = _plan()
    key = "hermes--beyond"
    failed = {p["key"]: _result(p) for p in made["points"]}
    failed[key] = _result(failed[key], "attach_failed")
    state = report.next_state({}, failed, {**_RUN, "date": "2026-11-25"}, tmp_path, made)
    yesterday = report.adapter_issue("hermes_agent", made, failed, _RUN, state)
    nothing = {**failed, key: _result(failed[key], "no_result")}
    after = report.next_state(state, nothing, _RUN, tmp_path, made)
    issue = report.adapter_issue("hermes_agent", made, nothing, _RUN, after)
    assert issue["open"] is True and issue["fingerprint"] == yesterday["fingerprint"]


def test_a_watching_release_above_the_range_that_will_not_install_asks_nothing(tmp_path):
    """With no failure known of it, no issue is opened for it, however long it will not
    install; and an issue open for something else that was since fixed closes over it."""
    made = _plan()
    key = "hermes--beyond"
    results = {p["key"]: _result(p) for p in made["points"]}
    results[key] = _result(results[key], "not_installable")
    issue = report.adapter_issue("hermes_agent", made, results, _RUN, {})
    assert issue["open"] is False
    assert "### Not checked today" in issue["body"] and "### Watching" not in issue["body"]
    assert _synced(issue, []) == []
    opened = [{"number": 7, "state": "open", "body": "<!-- compat-key: hermes_agent -->"}]
    done = _synced(issue, opened)
    assert [what for what, _ in done] == ["rewrite #7", "comment on #7", "close #7"]
    assert "Not checked today: `hermes-agent v0.23.5` (cannot be installed" in done[1][1]["body"]
    # Inside the range it is said, as it was: the release the README names does not install.
    inside = {
        **results,
        key: _result(results[key]),
        "hermes--claim_top": _result(results["hermes--claim_top"], "not_installable"),
    }
    assert report.adapter_issue("hermes_agent", made, inside, _RUN, {})["open"] is True
    # And a set of frameworks that will not install together is what its point finds.
    sets = {**results, key: _result(results[key])}
    conflict = "combination--openai_agents__pydantic_ai"
    sets[conflict] = _result(sets[conflict], "not_installable", install={"detail": "x"})
    assert report.adapter_issue("openai_agents__pydantic_ai", made, sets, _RUN, {})["open"] is True
    # Also the day after its tests failed: that they no longer install together is news,
    # and not a day the set could not be checked.
    failed = {**results, key: _result(results[key])}
    failed[conflict] = _result(failed[conflict], "fail", tests={"failed_ids": ["t.py::test_x"]})
    state = report.next_state({}, failed, {**_RUN, "date": "2026-11-25"}, tmp_path, made)
    after = report.next_state(state, sets, _RUN, tmp_path, made)
    body = report.adapter_issue("openai_agents__pydantic_ai", made, sets, _RUN, after)["body"]
    assert "cannot be installed together (the installer said: `x`)" in body
    assert "did not pass when it was last checked" not in body
    # A day the index was down is such a day, for a set too: what it last said stands.
    down = {**sets, conflict: _result(sets[conflict], "install_network")}
    after = report.next_state(state, down, _RUN, tmp_path, made)
    issue = report.adapter_issue("openai_agents__pydantic_ai", made, down, _RUN, after)
    before = report.adapter_issue("openai_agents__pydantic_ai", made, failed, _RUN, state)
    assert issue["open"] is True and issue["fingerprint"] == before["fingerprint"]


def test_a_check_that_left_is_said_when_nothing_else_could_be_checked(tmp_path):
    """A check leaves while it was not passing, on a day the subject's other point left no
    result. The issue closes, and its closing words say both: what left was not passing,
    and nothing else was looked at."""
    made = _plan()
    results = {p["key"]: _result(p) for p in made["points"]}
    for key in ("hermes--beyond", "hermes--claim_top"):
        results[key] = _result(results[key], "no_result")
    left = {**_GONE, "kind": "devbranch", "subject": "hermes_agent", "adapter": "hermes_agent"}
    last = {"date": "2026-11-25", "outcome": "fail", "targets": {}}
    state = {"points": {"devbranch--hermes_agent": {"point": left, "last": last}}}
    after = report.next_state(state, results, _RUN, tmp_path, made)
    issue = report.adapter_issue("hermes_agent", made, results, _RUN, after)
    assert issue["open"] is False and len(issue["left"]) == 1
    assert issue["closing"].startswith("Nothing here is known to need doing, and nothing of it")
    assert (
        "The check `devbranch--hermes_agent` was not passing when it was last run"
        in issue["closing"]
    )
    assert issue["closing"].endswith("That is not a pass. Closing.")
    opened = [{"number": 7, "state": "open", "body": "<!-- compat-key: hermes_agent -->"}]
    assert [what for what, _ in _synced(issue, opened)] == [
        "rewrite #7",
        "comment on #7",
        "close #7",
    ]
    assert _synced(issue, []) == [], "and no issue is opened to say it"


def test_what_another_run_saw_pass_is_not_something_checked_today(tmp_path):
    """The index is down for every point of an adapter today. Last week its branch passed.
    That row is remembered, and is no check of today: the closing words are those of a day
    nothing could be checked, not of a day everything checked passed."""
    daily, weekly = _plan(), _plan("weekly")
    week = {p["key"]: _result(p) for p in weekly["points"]}
    state = report.next_state({}, week, {**_RUN, "cadence": "weekly"}, tmp_path, weekly)
    # Its tested range could not be installed; what is watched of it left no result.
    down = {}
    for p in daily["points"]:
        how = "pass"
        if p["subject"] == "langgraph":
            how = "install_network" if p["kind"] == "framework" else "no_result"
        down[p["key"]] = _result(p, how)
    state = report.next_state(state, down, {**_RUN, "id": "2"}, tmp_path, daily)
    issue = report.adapter_issue("langgraph", daily, down, _RUN, state)
    assert issue["open"] is False
    assert issue["closing"].startswith(
        "Nothing here is known to need doing, and nothing of it could be checked today: "
    )
    assert "passes" not in issue["closing"]


def test_a_check_that_left_is_said_on_an_issue_only_a_watching_failure_keeps_open(tmp_path):
    made = _plan()
    results = {p["key"]: _result(p) for p in made["points"]}
    results["hermes--beyond"] = _result(results["hermes--beyond"], "fail")
    left = {**_GONE, "kind": "devbranch", "subject": "hermes_agent", "adapter": "hermes_agent"}
    last = {"date": "2026-11-25", "outcome": "fail", "targets": {}}
    state = {"points": {"devbranch--hermes_agent": {"point": left, "last": last}}}
    after = report.next_state(state, results, _RUN, tmp_path, made)
    issue = report.adapter_issue("hermes_agent", made, results, _RUN, after)
    assert issue["open"] is True and len(issue["left"]) == 1
    assert (
        "### No longer run\n\n- The check `devbranch--hermes_agent` was not passing"
        in issue["body"]
    )


@pytest.mark.parametrize("today", ["harness_error", "no_result", "install_network"])
@pytest.mark.parametrize(
    ("subject", "key", "found"),
    [
        (
            "openai_agents__pydantic_ai",
            "combination--openai_agents__pydantic_ai",
            "not_installable",
        ),
        ("hermes_agent", "hermes--claim_top", "not_installable"),
        ("hermes_agent", "hermes--claim_top", "test_deps_unresolvable"),
    ],
)
def test_what_a_watching_point_found_stands_on_a_day_it_says_nothing(
    tmp_path, subject, key, found, today
):
    """Yesterday a watching point found that its release will not install as asked. Today
    the run broke, left nothing, or could not reach the index. Nothing contradicts what it
    found: the issue stays open, says it again, and posts no comment."""
    made = _plan()
    passing = {p["key"]: _result(p) for p in made["points"]}
    refused = {**passing, key: _result(passing[key], found, install={"detail": "no wheel"})}
    state = report.next_state({}, refused, {**_RUN, "date": "2026-11-25"}, tmp_path, made)
    yesterday = report.adapter_issue(subject, made, refused, _RUN, state)
    assert yesterday["open"] is True
    opened = [{"number": 7, "state": "open", "body": yesterday["body"]}]
    silent = {**passing, key: _result(passing[key], today)}
    after = report.next_state(state, silent, _RUN, tmp_path, made)
    issue = report.adapter_issue(subject, made, silent, _RUN, after)
    assert issue["open"] is True
    assert (
        "when it was last checked, on `2026-11-25`, and could not be checked since"
        in (issue["body"])
    )
    assert issue["fingerprint"] == yesterday["fingerprint"]
    assert [what for what, _ in _synced(issue, opened)] == ["rewrite #7"]
    # The day it is checked again and finds the same, nothing changed either.
    again = report.next_state(after, refused, {**_RUN, "id": "2"}, tmp_path, made)
    assert (
        report.adapter_issue(subject, made, refused, _RUN, again)["fingerprint"]
        == (yesterday["fingerprint"])
    )


def test_a_day_the_index_is_down_opens_no_issue_for_what_is_watched(tmp_path):
    """Everything passed yesterday. Today no point could reach the index. That is the run's
    trouble, said on the health issue when it lasts; it is no news about any subject."""
    made = _plan()
    passing = {p["key"]: _result(p) for p in made["points"]}
    state = report.next_state({}, passing, {**_RUN, "date": "2026-11-25"}, tmp_path, made)
    down = {key: _result(result, "install_network") for key, result in passing.items()}
    after = report.next_state(state, down, _RUN, tmp_path, made)
    subjects = {p.get("subject", p["adapter"]) for p in made["points"]}
    for subject in sorted(subjects):
        issue = report.adapter_issue(subject, made, down, _RUN, after)
        assert issue["open"] is False, subject
        assert _synced(issue, []) == [], subject
    # Inside the range a release that will not install as asked is still what it finds.
    refused = {
        **down,
        "hermes--claim_top": _result(passing["hermes--claim_top"], "not_installable"),
    }
    assert report.adapter_issue("hermes_agent", made, refused, _RUN, after)["open"] is True


def test_a_watching_failure_with_more_names_than_are_kept_is_the_same_news_the_next_day(
    tmp_path,
):
    """A watching point fails with more test names than the state keeps, and the next day
    it cannot be checked. What its issue marks is what the state can say again."""
    made = _plan()
    key = "hermes--beyond"
    names = [f"t.py::test_{'x' * 320}{n}" for n in range(250)]
    passing = {p["key"]: _result(p) for p in made["points"]}
    failed = {**passing, key: _result(passing[key], "fail", tests={"failed_ids": names})}
    state = report.next_state({}, failed, {**_RUN, "date": "2026-11-25"}, tmp_path, made)
    yesterday = report.adapter_issue("hermes_agent", made, failed, _RUN, state)
    down = {**passing, key: _result(passing[key], "install_network")}
    after = report.next_state(state, down, _RUN, tmp_path, made)
    assert (
        report.adapter_issue("hermes_agent", made, down, _RUN, after)["fingerprint"]
        == (yesterday["fingerprint"])
    )


def test_a_pass_the_weekly_run_saw_is_named_as_not_run_by_the_daily_run(tmp_path):
    """The daily run does not run the development branch. When it writes the words an issue
    closes with, the branch's pass is named as not run today, not counted as checked."""
    weekly = _plan("weekly")
    run = {**_RUN, "cadence": "weekly", "date": "2026-11-23"}
    passed = {p["key"]: _result(p) for p in weekly["points"]}
    state = report.next_state({}, passed, run, tmp_path, weekly)
    daily = _plan()
    passing = {p["key"]: _result(p) for p in daily["points"]}
    after = report.next_state(state, passing, _RUN, tmp_path, daily)
    closing = report.adapter_issue("langgraph", daily, passing, _RUN, after)["closing"]
    assert (
        "langgraph head-0123456789, langgraph-prebuilt head-0123456789` (not run by this run)"
        in (closing)
    )
