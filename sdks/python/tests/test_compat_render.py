"""The public pages and README cells are a function of the state file and the day.

Nothing here runs a framework. The state is built by hand in the shape the
report writes, and each test names one thing a reader of the page relies on.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))

from compat import render  # noqa: E402

_TODAY = "2026-11-28"
_RELEASE = "0.7.0b1"


def _identity(adapter: str, role: str, version: str, *, release: str | None = None, **more) -> dict:
    sdk = {"origin": "build"}
    if release:
        sdk = {"origin": "pypi", "version": release, "ref": f"python-v{release}"}
    return {
        "kind": "published" if release else "framework",
        "subject": adapter,
        "adapters": [adapter],
        "adapter": adapter,
        "role": role,
        "targets": {adapter.replace("_", "-"): version},
        "python": "3.14",
        "runner": "linux-x64",
        "sdk": sdk,
        "in_claim": role != "beyond",
        "watching": bool(release),
        "fresh_days": 3,
        **more,
    }


def _entry(identity: dict, outcome: str = "pass", day: str = _TODAY, nodes=(), passed="") -> dict:
    last = {"date": day, "outcome": outcome, "cause": None, "failed_ids": list(nodes)}
    entry = {"point": identity, "last": last}
    if outcome == "pass" or passed:
        entry["last_green"] = {"date": passed or day}
    return entry


def _state(**points: dict) -> dict:
    about = {"label": "openai-agents", "evidence": "live", "newest": "0.26.0"}
    about["claim"] = {"openai-agents": ["0.22", "0.23"]}
    return {
        "schema": 2,
        "points": dict(points),
        "versions": {
            "main": {"version": _RELEASE, "commit": "0123456789abcdef", "adapters": {}},
            _RELEASE: {"ref": f"python-v{_RELEASE}", "adapters": {"openai_agents": about}},
        },
    }


def _published(**outcomes: tuple) -> dict:
    """A state holding one published point per role given, as `role=(version, outcome, day)`."""
    points = {}
    for role, (version, outcome, day) in outcomes.items():
        identity = _identity("openai_agents", role, version, release=_RELEASE)
        points[f"published--openai_agents--{role}"] = _entry(identity, outcome, day)
    return _state(**points)


def _cell(state: dict) -> tuple[str, str]:
    return render.cell_words(render.rows(state, _RELEASE)["openai_agents"], _TODAY)


# --- what a row says today -------------------------------------------------------


@pytest.mark.parametrize(
    "outcome, day, status, words",
    [
        ("pass", "2026-11-28", "pass", "pass"),
        ("pass", "2026-11-25", "pass", "pass"),
        ("pass", "2026-11-24", "stale", "passed, not run since 24 Nov 2026"),
        ("fail", "2026-11-28", "fail", "tests fail"),
        ("fail", "2026-11-20", "fail", "tests fail, not run since 20 Nov 2026"),
        ("no_result", "2026-11-28", "stale", "no result"),
        ("not_installable", "2026-11-28", "stale", None),
    ],
)
def test_an_old_pass_is_shown_as_when_it_passed_and_never_as_passing(outcome, day, status, words):
    entry = _entry(_identity("openai_agents", "claim_top", "0.23.1"), outcome, day)
    said = render.status_of(entry, _TODAY)
    assert said["status"] == status and said["ran"] == day
    if words is not None:
        assert said["words"] == words


def test_a_release_above_the_range_that_cannot_be_installed_is_not_a_lapse():
    entry = _entry(_identity("openai_agents", "beyond", "0.26.0"), "not_installable")
    assert render.status_of(entry, _TODAY)["status"] == "none"


def test_a_row_with_no_date_at_all_is_not_passing():
    entry = _entry(_identity("openai_agents", "claim_top", "0.23.1"), "pass", "not a date")
    assert render.status_of(entry, _TODAY)["status"] == "stale"


# --- which points belong on which page -------------------------------------------


def test_a_page_holds_the_points_of_its_own_build_and_no_other():
    state = _state(
        **{
            "openai_agents--claim_top": _entry(_identity("openai_agents", "claim_top", "0.23.1")),
            "published--openai_agents--claim_top": _entry(
                _identity("openai_agents", "claim_top", "0.23.1", release=_RELEASE)
            ),
            "published-old": _entry(
                _identity("openai_agents", "claim_top", "0.22.9", release="0.6.0")
            ),
            "before-kinds": {"point": None, "last": {"outcome": "pass", "date": _TODAY}},
        }
    )
    keys = lambda version: [p["key"] for p in render.rows(state, version).get("openai_agents", [])]  # noqa: E731
    assert keys(render.MAIN) == ["openai_agents--claim_top"]
    assert keys(_RELEASE) == ["published--openai_agents--claim_top"]
    assert keys("0.6.0") == ["published-old"]


# --- the README cell -------------------------------------------------------------


def test_a_cell_says_what_passes_what_is_newer_and_when():
    state = _published(claim_top=("0.23.1", "pass", _TODAY), beyond=("0.26.0", "fail", _TODAY))
    assert _cell(state) == ("pass", "0.23.1 passes · newer 0.26.0 fails · 28 Nov")


def test_a_failure_above_the_range_never_colours_the_cell_and_one_inside_always_does():
    above = _published(claim_top=("0.23.1", "pass", _TODAY), beyond=("0.26.0", "fail", _TODAY))
    inside = _published(claim_top=("0.23.1", "fail", _TODAY), beyond=("0.26.0", "pass", _TODAY))
    assert _cell(above)[0] == "pass"
    assert _cell(inside) == ("fail", "0.23.1 fails · newer 0.26.0 passes · 28 Nov")


def test_a_cell_names_the_span_of_releases_it_ran():
    state = _published(floor=("0.22.0", "pass", _TODAY), claim_top=("0.23.1", "pass", _TODAY))
    assert _cell(state) == ("pass", "0.22.0 to 0.23.1 pass · 28 Nov")


def test_a_cell_whose_proof_lapsed_says_since_when_instead_of_passing():
    state = _published(claim_top=("0.23.1", "pass", "2026-11-20"))
    assert _cell(state) == ("stale", "0.23.1 not proven since 20 Nov 2026")
    never = _published(claim_top=("0.23.1", "no_result", _TODAY))
    assert _cell(never) == ("stale", "0.23.1 not proven")


def test_an_adapter_with_no_point_gets_a_cell_that_claims_nothing():
    assert render.cell_words([], _TODAY) == ("none", "no result")


def test_a_cell_is_well_formed_and_holds_one_line_of_escaped_text():
    svg = render.svg_cell("fail", 'a < b & "c" · d')
    root = ET.fromstring(svg)  # noqa: S314 - text this test just made
    space = "{http://www.w3.org/2000/svg}"
    assert root.tag == f"{space}svg" and root.get("height") == "20"
    assert root.find(f"{space}title").text == 'a < b & "c" · d'
    assert root.find(f"{space}text").text == 'a < b & "c" · d'
    assert root.get("aria-label") == 'a < b & "c" · d'
    assert root.find(f"{space}circle").get("fill") == "#cf222e"
    wide = ET.fromstring(render.svg_cell("pass", "x" * 60))  # noqa: S314
    assert int(wide.get("width")) > int(root.get("width")), "the pill is as wide as its words"


# --- the site --------------------------------------------------------------------


def test_the_same_state_on_the_same_day_renders_the_same_bytes(tmp_path):
    state = _published(claim_top=("0.23.1", "pass", _TODAY), beyond=("0.26.0", "fail", _TODAY))
    first = render.site(state, _TODAY)
    assert first == render.site(state, _TODAY)
    assert sorted(first) == [
        f"{_RELEASE}/README.md",
        f"{_RELEASE}/openai_agents.svg",
        "README.md",
        "main/README.md",
    ]
    stale = tmp_path / "site" / "0.1.0" / "gone.svg"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")
    written = render.write_site(tmp_path, state, _TODAY)
    assert written == sorted(first) and not stale.exists(), "the site is replaced whole"
    for name, text in first.items():
        assert (tmp_path / "site" / name).read_text(encoding="utf-8") == text


def test_a_release_page_says_which_wheel_and_which_tests_and_marks_what_is_watched():
    nodes = [f"t.py::test_{n}" for n in range(45)]
    state = _published(claim_top=("0.23.1", "pass", _TODAY))
    beyond = _identity("openai_agents", "beyond", "0.26.0", release=_RELEASE)
    state["points"]["published--openai_agents--beyond--0.26"] = _entry(beyond, "fail", nodes=nodes)
    page = render.page(state, _RELEASE, _TODAY)
    assert f"`wardex-sdk=={_RELEASE}` as PyPI serves it" in page
    assert f"its own tag `python-v{_RELEASE}`" in page
    assert "- Tested range, as this build states it: `openai-agents` 0.22.x, 0.23.x" in page
    assert (
        "| published wardex release (watching) | newest tested release | `openai-agents 0.23.1` "
        in page
    )
    assert "| tests fail | 28 Nov 2026 | - |" in page
    assert "- `t.py::test_39`" in page and "t.py::test_40" not in page
    assert "- and 5 more" in page


def test_nothing_a_point_or_a_plan_wrote_can_become_markup_or_a_path(tmp_path):
    nasty = "x`|<script>](http://e.invalid)\n## y"
    state = _published(claim_top=("0.23.1", "fail", _TODAY))
    entry = state["points"]["published--openai_agents--claim_top"]
    entry["last"]["failed_ids"] = [nasty]
    entry["point"]["role"] = nasty
    entry["point"]["python"] = nasty
    state["versions"][_RELEASE]["adapters"]["openai_agents"]["label"] = nasty
    state["versions"]["../../outside"] = {"adapters": {"openai_agents": {}}}
    state["versions"][_RELEASE]["adapters"]["../evil"] = {}
    files = render.site(state, _TODAY)
    assert all(".." not in name for name in files)
    page = files[f"{_RELEASE}/README.md"]
    for line in page.splitlines():
        assert "<script>" not in line and not line.startswith("## y")
        assert line.count("`") % 2 == 0, line
    assert all(line.count("|") == 8 for line in page.splitlines() if line.startswith("| "))
    render.write_site(tmp_path, state, _TODAY)
    assert not (tmp_path / "outside").exists() and not (tmp_path.parent / "outside").exists()


def test_a_point_that_could_not_be_checked_is_shown_as_what_it_last_said():
    """Today the index was down. The last time this release was checked it failed, and the
    page says that, as its issue does; had it passed within the last three days, it would
    still be passing."""
    identity = _identity("openai_agents", "claim_top", "0.23.1")
    targets = identity["targets"]
    failed = _entry(identity, "install_network", passed="2026-11-20")
    failed["last_green"]["targets"] = targets
    failed["last_checked"] = {"date": "2026-11-27", "outcome": "fail", "targets": targets}
    said = render.status_of(failed, _TODAY)
    assert said["status"] == "fail" and said["words"].startswith(
        "failed when last checked, on 27 Nov 2026; now: "
    )

    passed = _entry(identity, "install_network", passed="2026-11-27")
    passed["last_green"]["targets"] = targets
    passed["last_checked"] = {"date": "2026-11-27", "outcome": "pass", "targets": targets}
    said = render.status_of(passed, _TODAY)
    assert (said["status"], said["words"][:26]) == ("pass", "passed on 27 Nov 2026; now")
    passed["last_green"]["date"] = "2026-11-20"
    assert render.status_of(passed, _TODAY)["status"] == "stale"
    passed["last_green"]["date"] = "2026-11-25"
    assert render.status_of(passed, _TODAY)["status"] == "pass", "three days is still proof"
    passed["last_green"]["date"] = passed["last_checked"]["date"] = _TODAY
    assert render.status_of(passed, _TODAY)["status"] == "pass", "a pass made earlier today"

    other = _entry(identity, "install_network")
    other["last_checked"] = {
        "date": "2026-11-27",
        "outcome": "fail",
        "targets": {"openai-agents": "0.23.0"},
    }
    assert render.status_of(other, _TODAY)["status"] == "stale", "another release's failure"


def test_a_lapsed_cell_dates_the_lapse_by_what_is_not_proven_and_not_by_what_is():
    state = _published(floor=("0.22.0", "pass", _TODAY), claim_top=("0.23.1", "pass", "2026-11-18"))
    assert _cell(state) == ("stale", "0.22.0 to 0.23.1 not proven since 18 Nov 2026")
    two = _published(
        floor=("0.22.0", "pass", "2026-11-10"), claim_top=("0.23.1", "pass", "2026-11-18")
    )
    assert _cell(two) == ("stale", "0.22.0 to 0.23.1 not proven since 10 Nov 2026")
    never = _published(
        floor=("0.22.0", "no_result", _TODAY), claim_top=("0.23.1", "pass", "2026-11-18")
    )
    assert _cell(never) == ("stale", "0.22.0 to 0.23.1 not proven")


def test_a_cell_speaks_for_the_newest_release_above_the_range():
    state = _published(claim_top=("0.23.1", "pass", _TODAY))
    for line, outcome in (("0.24.0", "fail"), ("0.26.1", "pass"), ("0.25.0", "fail")):
        identity = _identity("openai_agents", "beyond", line, release=_RELEASE)
        state["points"][f"published--beyond--{line}"] = _entry(identity, outcome)
    assert _cell(state) == ("pass", "0.23.1 passes \u00b7 newer 0.26.1 passes \u00b7 28 Nov")


def test_the_site_is_never_written_through_a_link(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "keep.txt").write_text("not the site", encoding="utf-8")
    records = tmp_path / "records"
    records.mkdir()
    (records / "site").symlink_to(elsewhere, target_is_directory=True)
    render.write_site(records, _published(claim_top=("0.23.1", "pass", _TODAY)), _TODAY)
    assert sorted(p.name for p in elsewhere.iterdir()) == ["keep.txt"]
    assert not (records / "site").is_symlink() and (records / "site/README.md").is_file()


@pytest.mark.parametrize("checked", ["fail", "attach_failed", "skips_changed"])
def test_every_way_a_release_can_have_failed_is_carried_over_an_outage(checked):
    identity = _identity("openai_agents", "claim_top", "0.23.1")
    entry = _entry(identity, "not_installable")
    entry["last_checked"] = {
        "date": "2026-11-27",
        "outcome": checked,
        "targets": identity["targets"],
    }
    assert render.status_of(entry, _TODAY)["status"] == "fail"


@pytest.mark.parametrize("outcome", ["no_result", "harness_error"])
def test_a_point_that_left_no_result_is_never_green_whatever_it_said_yesterday(outcome):
    """The run is red for it and its issue says it does not pass. A cell that stayed green
    beside that would be the page contradicting the run."""
    identity = _identity("openai_agents", "claim_top", "0.23.1")
    entry = _entry(identity, outcome, passed="2026-11-27")
    entry["last_green"]["targets"] = identity["targets"]
    entry["last_checked"] = {
        "date": "2026-11-27",
        "outcome": "pass",
        "targets": identity["targets"],
    }
    assert render.status_of(entry, _TODAY)["status"] == "stale"
    failed = {**entry, "last_checked": {**entry["last_checked"], "outcome": "fail"}}
    assert render.status_of(failed, _TODAY)["status"] == "fail", "a known failure stays one"
    state = _state(**{"published--top": entry})
    entry["point"] = _identity("openai_agents", "claim_top", "0.23.1", release=_RELEASE)
    assert _cell(state) == ("stale", "0.23.1 not proven since 27 Nov 2026")


def test_the_index_lists_the_newest_release_first():
    state = _state()
    for version in ("0.9.0", "0.7.0", "0.10.0", "0.7.0b1", "0.7.0.post1", "0.7.0rc1", "0.7.0b10"):
        state["versions"][version] = {"adapters": {}}
    listed = [
        line.split("]")[0].removeprefix("- [wardex ")
        for line in render.index(state, _TODAY).splitlines()
        if line.startswith("- [wardex")
    ]
    assert listed == ["0.10.0", "0.9.0", "0.7.0.post1", "0.7.0", "0.7.0rc1", "0.7.0b10", "0.7.0b1"]
    more = {"0.7.0.post10": {}, "0.7.0.post2": {}, "v0.6": {}, "a b": {}, "../x": {}}
    more["0.7.0.dev1"] = more["0.7.0+local"] = None
    state["versions"].update(more)
    again = render.index(state, _TODAY)
    assert again.index("0.7.0.post10]") < again.index("0.7.0.post2]") < again.index("0.7.0.post1]")
    pages = {name.split("/")[0] for name in render.site(state, _TODAY) if "/" in name} - {"main"}
    linked = set(re.findall(r"\]\(([^/)]+)/README\.md\)", again)) - {"main"}
    assert linked == pages and "v0.6" not in again, "the index links only pages that are written"


def test_a_failure_waiting_for_its_own_point_is_on_the_default_branchs_page():
    """Between the day a newer release replaces a failing one and the day its own point
    first runs, no row holds it. The page must not look clean in that gap."""
    state = _state(
        **{"openai_agents--claim_top": _entry(_identity("openai_agents", "claim_top", "0.23.2"))}
    )
    item = {"key": "openai_agents--displaced--0.23.1-py3.14", "failed_on": "2026-11-27"}
    item["python"] = "3.14"
    item["targets"] = {"openai-agents": "0.23.1"}
    state["displaced"] = {"openai_agents": [item]}
    page = render.page(state, render.MAIN, _TODAY)
    assert (
        "- `openai-agents 0.23.1` did not pass when it was last run, on 27 Nov 2026. No rule "
        "picks it any more; it is run on its own from the next run on." in page
    )
    own = _identity("openai_agents", "displaced", "0.23.1")
    state["points"][item["key"]] = _entry(own, "fail")
    assert "No rule picks it" not in render.page(state, render.MAIN, _TODAY), "its row says it now"

    del state["points"][item["key"]]
    again = _identity("openai_agents", "claim_top", "0.23.1")
    state["points"]["openai_agents--claim_top"] = _entry(again, "install_network")
    assert "No rule picks it" not in render.page(state, render.MAIN, _TODAY), (
        "a rule picked it again"
    )

    state["displaced"]["openai_agents"] = [{**item, "python": "3.10"}]
    page = render.page(state, render.MAIN, _TODAY)
    assert "No rule picks it" in page, "the same release on another interpreter is another failure"
    floor = _identity("openai_agents", "floor", "0.23.1")
    state["points"] = {"openai_agents--floor": _entry(floor)}
    state["displaced"]["openai_agents"] = [item]
    assert "No rule picks it" in render.page(state, render.MAIN, _TODAY), "the floor is not it"
    state["points"] = {"openai_agents--claim_top": _entry(again, "install_network")}

    odd = ["not an item", {"targets": "nor this"}, {"targets": {"x": None}}]
    state["displaced"]["openai_agents"] = odd
    page = render.page(state, render.MAIN, _TODAY)
    assert "- `x None` did not pass when it was last run, on an unknown day." in page
    state["displaced"]["openai_agents"] = "not a list"
    assert "No rule picks it" not in render.page(state, render.MAIN, _TODAY)
