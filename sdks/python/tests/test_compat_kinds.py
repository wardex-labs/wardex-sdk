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

from compat import kinds, manifest, plan, report, verdict  # noqa: E402
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


# --- a Hermes release tag ----------------------------------------------------------


def test_hermes_is_run_at_the_named_tag_and_at_the_newest_release_by_when_it_was_published():
    points = _of(_plan(), "hermes")
    named, newest = points["hermes--claim_top"], points["hermes--beyond"]
    assert named["targets"] == {"hermes-agent": "v0.23.3"} and named["in_claim"] is True
    # Not the canary, not the tag of an older numbering, not the one published after the cutoff.
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

    bad = {"ok": False, "failed": ["the smoke failed in cli mode", 7, "x" * 999], "checks": {}}
    runner, _, calls = _driven(tmp_path / "fail", monkeypatch, point, code=1, verdict=bad)
    runner._driven(runner.work / "venv")
    assert runner.result["outcome"] == "fail" and runner.result["cause"] == "not_determined"
    assert runner.result["tests"]["failed_ids"] == ["the smoke failed in cli mode", "x" * 300]
    assert len(calls) == 2 and runner.result["rerun"] == {"passed": False}, "run once more first"

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
        (1, None),
        (2, {"ok": True}),
        (137, None),
    ],
    ids=[
        "no-verdict",
        "not-json",
        "says-not-ok",
        "a-list",
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
    assert after["retired"] == [
        {
            "subject": "openai_agents",
            "key": "retired--openai_agents",
            "kind": "retired_kind",
            "date": "2026-11-25",
        }
    ]
    issue = report.adapter_issue("openai_agents", daily, passing, _RUN, after)
    assert issue["open"] is False
    assert (
        "The check `retired--openai_agents` was failing when it was last run (`2026-11-25`) and "
        "is no longer run: its kind (`retired_kind`) was retired. That is not a pass."
        in issue["closing"]
    )
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
        f"together (the installer said: {refused})" in issue["body"]
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
