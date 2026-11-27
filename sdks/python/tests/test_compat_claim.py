"""The claim command: one source for what wardex says it is tested on.

The table of tested versions is the source. The test pin and every version
phrase and install line in the README are written from it, and the first test
here is the one that fails when a person edits one of them by hand.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts"))

from compat import claim, kinds, manifest, plan  # noqa: E402

_FILES = (claim.MEASURED, claim.ENUMS, claim.PYPROJECT, claim.README)


def test_the_table_the_test_pins_and_the_readme_say_the_same_thing():
    """Every tested range is stated once and copied by a command. A README sentence, an
    install line or a pin that says something else fails here, with the place named."""
    assert claim.drift(_REPO) == []


@pytest.fixture
def tree(tmp_path) -> Path:
    """The four files a claim lives in, copied where a command can rewrite them."""
    for name in _FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_REPO / name, tmp_path / name)
    return tmp_path


def _row(root: Path, dist: str) -> dict:
    return claim.table(root)[dist]


def _texts(root: Path) -> dict[str, str]:
    return {name: (root / name).read_text(encoding="utf-8") for name in _FILES}


def _a_row_with_two_lines(root: Path) -> dict:
    return next(row for row in claim.table(root).values() if len(row["minors"]) > 1)


def _line_after(row: dict) -> str:
    top = max(row["minors"], key=claim._vkey).split(".")
    return ".".join([*top[:-1], str(int(top[-1]) + 1)])


def test_widening_a_range_changes_the_table_the_pin_and_every_readme_phrase(tree, capsys):
    row = _a_row_with_two_lines(tree)
    dist, line = row["distribution"], _line_after(row)
    before = _texts(tree)
    assert claim.main(["widen", dist, line], root=tree) == 0
    after = _row(tree, dist)
    assert after["minors"] == (*row["minors"], line) and after["since"] == row["since"]
    said = plan.spelled(after)
    assert said.endswith(f"{line}.x") and f"`{dist}` {said}" in _texts(tree)[claim.README]
    assert f"`{dist}` {plan.spelled(row)})" not in _texts(tree)[claim.README]
    lower, upper = claim.bounds(after)
    assert f'"{dist}>={lower},<{upper}"' in _texts(tree)[claim.PYPROJECT]
    assert f'"{dist}>={lower},<{upper}"' in _texts(tree)[claim.README], "the install line too"
    assert claim.drift(tree) == []
    assert f"{dist}: tested on {said}" in capsys.readouterr().out

    assert claim.main(["narrow", dist, "--drop", line], root=tree) == 0
    assert _texts(tree) == before, "taking the line out again gives back the same bytes"


def test_a_comment_beside_a_row_survives_the_rewrite(tree):
    source = (tree / claim.MEASURED).read_text(encoding="utf-8")
    comments = [line for line in source.splitlines() if line.strip().startswith("#")]
    row = _a_row_with_two_lines(tree)
    claim.main(["widen", row["distribution"], _line_after(row)], root=tree)
    after = (tree / claim.MEASURED).read_text(encoding="utf-8")
    assert [line for line in after.splitlines() if line.strip().startswith("#")] == comments


def test_dropping_the_lowest_line_moves_the_pins_floor_and_forgets_a_start_inside_it(tree):
    row = _a_row_with_two_lines(tree)
    dist, lowest, second = row["distribution"], *sorted(row["minors"], key=claim._vkey)[:2]
    assert claim.main(["narrow", dist, "--since", f"{lowest}.3"], root=tree) == 0
    assert _row(tree, dist)["since"] == f"{lowest}.3"
    assert f'"{dist}>={lowest}.3,<' in _texts(tree)[claim.PYPROJECT]
    assert f"{lowest}.x from {lowest}.3" in _texts(tree)[claim.README]
    assert claim.drift(tree) == []

    assert claim.main(["narrow", dist, "--drop", lowest], root=tree) == 0
    after = _row(tree, dist)
    assert lowest not in after["minors"] and after["since"] == ""
    assert f'"{dist}>={second},<' in _texts(tree)[claim.PYPROJECT]
    assert claim.drift(tree) == []


def test_a_release_listed_as_bad_is_no_longer_claimed_and_moves_no_pin(tree):
    row = _a_row_with_two_lines(tree)
    dist = row["distribution"]
    version = f"{sorted(row['minors'], key=claim._vkey)[-1]}.0"
    assert plan.claims(row, version)
    pins = _texts(tree)[claim.PYPROJECT]
    assert claim.main(["bad", dist, version], root=tree) == 0
    after = _row(tree, dist)
    assert after["known_bad"] == (version,) and not plan.claims(after, version)
    assert _texts(tree)[claim.PYPROJECT] == pins and claim.drift(tree) == []


@pytest.mark.parametrize(
    "argv",
    [
        ["widen", "no-such-distribution", "1.0"],
        ["widen", "{dist}", "{lowest}"],
        ["narrow", "{dist}", "--drop", "9.9"],
        ["narrow", "{dist}", "--since", "9.9.9"],
        ["bad", "{dist}", "9.9.9"],
        ["widen", "{dist}", "{lowest}.x"],
        ["widen", "{dist}", "7"],
        ["widen", "{dist}", "{gap}"],
        ["bad", "{dist}", "latest"],
        ["widen", "pydantic-ai-slim", "1"],
    ],
)
def test_a_command_that_makes_no_sense_changes_nothing_and_says_why(tree, argv):
    row = _a_row_with_two_lines(tree)
    lowest = sorted(row["minors"], key=claim._vkey)[0]
    top = max(row["minors"], key=claim._vkey).split(".")
    gap = ".".join([*top[:-1], str(int(top[-1]) + 2)])
    argv = [part.format(dist=row["distribution"], lowest=lowest, gap=gap) for part in argv]
    before = _texts(tree)
    with pytest.raises(SystemExit) as stopped:
        claim.main(argv, root=tree)
    assert stopped.value.code not in (0, None) and isinstance(stopped.value.code, str)
    assert _texts(tree) == before


def test_the_only_line_of_a_range_cannot_be_dropped(tree):
    row = next(row for row in claim.table(tree).values() if len(row["minors"]) == 1)
    with pytest.raises(SystemExit):
        claim.main(["narrow", row["distribution"], "--drop", row["minors"][0]], root=tree)


def test_a_hand_edit_is_named_by_file_and_line_and_sync_writes_it_back(tree, capsys):
    row = _a_row_with_two_lines(tree)
    dist, said = row["distribution"], plan.spelled(row)
    lower, upper = claim.bounds(row)
    before = _texts(tree)
    readme = before[claim.README]
    wrong = readme.replace(f"`{dist}` {said}", f"`{dist}` 9.1.x, 9.2.x")
    wrong = wrong.replace(f"{dist}>={lower},<{upper}", f"{dist}>={lower},<99")
    assert wrong != readme
    (tree / claim.README).write_text(wrong, encoding="utf-8")
    pins = before[claim.PYPROJECT].replace(f'"{dist}>={lower},<{upper}"', f'"{dist}>={lower},<99"')
    (tree / claim.PYPROJECT).write_text(pins, encoding="utf-8")

    assert claim.main(["check"], root=tree) == 1
    out = capsys.readouterr().out
    line = wrong.count("\n", 0, wrong.index(f"`{dist}` 9.1.x")) + 1
    assert f"README.md:{line}: says `{dist}` 9.1.x, 9.2.x; the table says {said}" in out
    assert f"installs {dist}>={lower},<99" in out
    assert f"{claim.PYPROJECT}: {dist} is pinned >={lower},<99" in out
    assert "the framework table has no row saying" in out

    assert claim.main(["sync"], root=tree) == 0
    assert _texts(tree) == before and claim.main(["check"], root=tree) == 0


def test_a_distribution_with_no_test_pin_is_drift(tree):
    row = _a_row_with_two_lines(tree)
    dist = row["distribution"]
    text = (tree / claim.PYPROJECT).read_text(encoding="utf-8")
    lower, upper = claim.bounds(row)
    (tree / claim.PYPROJECT).write_text(
        text.replace(f'"{dist}>={lower},<{upper}"', '"something-else>=1,<2"'), encoding="utf-8"
    )
    assert f"{claim.PYPROJECT}: no test pin for {dist}" in claim.drift(tree)


# --- every sentence that says "tested" ------------------------------------------

_CLAIMING = re.compile(
    r"tested on|(?:is|are) tested|tested against|[Vv]erified against|[Mm]easured (?:on|against)"
)


def _readme_sentences() -> list[str]:
    """The README as sentences, each on one line, so a claim can be looked up whole."""
    text = re.sub(r"\s+", " ", (_REPO / claim.README).read_text(encoding="utf-8"))
    return [part.strip() for part in re.split(r"(?<=[.:|]) ", text) if part.strip()]


def test_every_listed_claim_is_a_sentence_the_readme_still_has():
    """An entry whose words the README no longer has is proving a sentence nobody reads."""
    readme = re.sub(r"\s+", " ", (_REPO / claim.README).read_text(encoding="utf-8"))
    for entry in manifest.CLAIMS:
        assert entry["says"] in readme, entry["says"]
        if entry["proof"] is None:
            assert entry["why"], f"an unproven sentence says what stands behind it: {entry['says']}"
        else:
            assert entry["proof"] in kinds.KINDS and kinds.KINDS[entry["proof"]].cadences


def test_every_sentence_of_the_readme_that_claims_a_test_is_listed_or_comes_from_the_table():
    """A new sentence saying wardex was tested on something has to say what proves it: it
    is either a version phrase the table of tested versions writes, or an entry in the list
    of claims, where having no scheduled proof is a public fact."""
    rows = claim.table(_REPO)
    from_the_table = [f"`{dist}` {plan.spelled(row)}" for dist, row in rows.items()]
    unlisted = [
        sentence
        for sentence in _readme_sentences()
        if _CLAIMING.search(sentence)
        and not any(phrase in sentence for phrase in from_the_table)
        and not any(
            entry["says"] in sentence or sentence in entry["says"] for entry in manifest.CLAIMS
        )
    ]
    assert unlisted == []


def test_a_line_that_would_leave_a_hole_or_reach_under_a_stated_start_is_refused_in_words(tree):
    """One pin cannot leave a release line out of its middle, so the install line would
    admit releases the table does not claim. And a range that starts inside its lowest
    line starts there because what lies below does not pass."""
    with pytest.raises(SystemExit) as hole:
        claim.main(["widen", "openai-agents", "0.25"], root=tree)
    assert "would leave 0.24.x between the tested lines" in str(hole.value)
    with pytest.raises(SystemExit) as under:
        claim.main(["widen", "pydantic-ai-slim", "1"], root=tree)
    assert "is tested from 2.54 on; 1.x lies below that" in str(under.value)
    assert claim._between(["0.22", "0.23"], "0.24") == ""
    assert claim._between(["0.22", "0.23"], "0.21") == ""
    assert claim._between(["0.22", "0.25"], "0.28") == "0.26.x, 0.27.x"
    assert claim._between(["0.22"], "0.20") == "0.21.x"
    assert claim._between(["0.9"], "1.0") == "", "a new prefix is the framework's to number"
    assert claim._between(["2"], "4") == "3.x"


def test_an_install_line_with_extras_is_held_to_the_table_like_any_other(tree):
    text = (tree / claim.README).read_text(encoding="utf-8")
    assert '"pydantic-ai-slim[openai]>=2.54,<3"' in text
    assert claim.main(["narrow", "pydantic-ai-slim", "--since", "2.60"], root=tree) == 0
    after = (tree / claim.README).read_text(encoding="utf-8")
    assert '"pydantic-ai-slim[openai]>=2.60,<3"' in after and claim.drift(tree) == []
    (tree / claim.README).write_text(
        after.replace("[openai]>=2.60,<3", "[openai]>=2.0,<3"), "utf-8"
    )
    assert any("installs pydantic-ai-slim>=2.0,<3" in line for line in claim.drift(tree))


def test_releases_listed_as_bad_are_kept_in_order_and_once(tree):
    row = _a_row_with_two_lines(tree)
    dist = row["distribution"]
    low, high = sorted(row["minors"], key=claim._vkey)
    for version in (f"{high}.1", f"{low}.9", f"{low}.10"):
        assert claim.main(["bad", dist, version], root=tree) == 0
    assert _row(tree, dist)["known_bad"] == (f"{low}.9", f"{low}.10", f"{high}.1")
    with pytest.raises(SystemExit):
        claim.main(["bad", dist, f"{low}.9"], root=tree)
