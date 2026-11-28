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
    assert "leaves 0.24.x between the tested lines" in str(hole.value)
    with pytest.raises(SystemExit) as under:
        claim.main(["widen", "pydantic-ai-slim", "1"], root=tree)
    assert "is tested from 2.54 on; 1.x lies below that" in str(under.value)
    assert claim.holes(["0.22", "0.23", "0.24"]) == ""
    assert claim.holes(["0.21", "0.22", "0.23"]) == ""
    assert claim.holes(["0.22", "0.25", "0.28"]) == "0.23.x, 0.24.x, 0.26.x, 0.27.x"
    assert claim.holes(["0.9", "1.0"]) == "", "a new prefix is the framework's to number"
    assert claim.holes(["0.3", "1.7"]) == "", "nothing is counted across two prefixes"
    assert claim.holes(["0.25", "0.22", "0.23"]) == "0.24.x", "in whatever order they come"
    assert claim.holes(["0.22", "1.5", "1.7"]) == "1.6.x", "and a hole is looked for inside one"
    assert claim.holes(["2", "4"]) == "3.x"


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


def test_taking_a_line_out_of_the_middle_is_refused_like_adding_one_past_a_gap(tree):
    assert claim.main(["widen", "openai-agents", "0.24"], root=tree) == 0
    before = _texts(tree)
    with pytest.raises(SystemExit) as hole:
        claim.main(["narrow", "openai-agents", "--drop", "0.23"], root=tree)
    assert "leaves 0.23.x between the tested lines" in str(hole.value) and _texts(tree) == before
    assert claim.main(["narrow", "openai-agents", "--drop", "0.24"], root=tree) == 0
    assert claim.main(["narrow", "openai-agents", "--drop", "0.22"], root=tree) == 0


def test_a_table_someone_left_a_hole_in_is_drift(tree):
    path = tree / claim.MEASURED
    text = path.read_text(encoding="utf-8")
    holed = text.replace('("0.22", "0.23")', '("0.22", "0.24")')
    assert holed != text
    path.write_text(holed, encoding="utf-8")
    assert any("skips 0.23.x between its tested lines" in line for line in claim.drift(tree))


@pytest.mark.parametrize(
    "written",
    [
        "`{dist}`\n9.1.x, 9.2.x",
        "`{dist}`  9.1.x,\n9.2.x",
        "`{dist}` 9.1.x\n   from 9.1.4, 9.2.x",
        "`{dist}`\t9.1.x,\t9.2.x",
        "> `{dist}`\n> 9.1.x,\n> 9.2.x",
        "`{dist}` 9.1.x,\r\n9.2.x",
        "'{dist}>=9.1,<9.3'",
        '"{dist} >= 9.1, < 9.3"',
    ],
)
def test_a_claim_that_wraps_or_is_spelled_with_other_quotes_or_spaces_is_still_held(tree, written):
    """The README wraps its sentences. A phrase or an install line that says another
    range is drift however it is broken across lines, quoted or spaced, and `sync` writes
    it back."""
    dist = _a_row_with_two_lines(tree)["distribution"]
    path = tree / claim.README
    path.write_text(
        path.read_text(encoding="utf-8") + "\n" + written.format(dist=dist) + "\n", "utf-8"
    )
    assert len(claim.drift(tree)) == 1
    assert claim.main(["sync"], root=tree) == 0 and claim.drift(tree) == []
    after = path.read_text(encoding="utf-8")
    assert "9.1" not in after and "9.2" not in after and "<9.3" not in after


def test_a_phrase_the_table_writes_is_not_drift_for_being_wrapped(tree):
    row = _a_row_with_two_lines(tree)
    dist, said = row["distribution"], plan.spelled(row)
    assert ", " in said
    path = tree / claim.README
    wrapped = f"\n`{dist}` " + said.replace(", ", ",\n", 1) + "\n"
    path.write_text(path.read_text("utf-8") + wrapped, "utf-8")
    assert claim.drift(tree) == []


@pytest.mark.parametrize(
    "written",
    [
        "`{dist}`\n\n9.x of the documentation",
        "### `{dist}`\n\n9.x and later",
        "`{dist}` is described in section 9.x",
        "`{dist}` > 9.x is not how a range is written",
        "`{dist}` {said},\n\n9.x is another paragraph",
    ],
)
def test_a_version_like_word_in_the_next_paragraph_is_not_a_claim(tree, written):
    """One line break inside a phrase is a wrapped sentence. A blank line ends the
    paragraph, and what follows says something else."""
    dist = _a_row_with_two_lines(tree)["distribution"]
    path = tree / claim.README
    said = plan.spelled(_a_row_with_two_lines(tree))
    added = written.replace("{dist}", dist).replace("{said}", said)
    path.write_text(path.read_text("utf-8") + "\n" + added + "\n", "utf-8")
    before = path.read_text("utf-8")
    assert claim.drift(tree) == []
    assert claim.main(["sync"], root=tree) == 0 and path.read_text("utf-8") == before


def test_a_file_keeps_the_line_endings_it_came_with(tree):
    for name in _FILES:
        path = tree / name
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert claim.drift(tree) == []
    row = _a_row_with_two_lines(tree)
    assert claim.main(["widen", row["distribution"], _line_after(row)], root=tree) == 0
    assert claim.drift(tree) == []
    for name in (claim.MEASURED, claim.PYPROJECT, claim.README):
        data = (tree / name).read_bytes()
        assert data.count(b"\r\n") == data.count(b"\n") > 0, name
    bumped = (tree / claim.PYPROJECT).read_bytes().replace(b'"0.7.0b1"', b'"9.9.9"', 1)
    (tree / claim.PYPROJECT).write_bytes(bumped)
    assert claim.main(["sync"], root=tree) == 0
    data = (tree / claim.README).read_bytes()
    assert b"/site/9.9.9/" in data and data.count(b"\r\n") == data.count(b"\n")

    assert "9.1" not in path.read_text(encoding="utf-8")


def test_a_range_wrapped_inside_a_quoted_paragraph_is_read_as_one_range(tree):
    row = _a_row_with_two_lines(tree)
    dist, said = row["distribution"], plan.spelled(row)
    path = tree / claim.README
    quoted = f"\n> `{dist}` " + said.replace(", ", ",\n> ", 1) + "\n"
    path.write_text(path.read_text("utf-8") + quoted, "utf-8")
    assert claim.drift(tree) == []
    wrong = quoted.replace(said.split(", ")[0], "9.1.x")
    path.write_text(path.read_text("utf-8").replace(quoted, wrong), "utf-8")
    (found,) = claim.drift(tree)
    assert f"says `{dist}` 9.1.x, " in found and "> " not in found


# --- the README's table of what was run ------------------------------------------


def _block(root: Path) -> str:
    text = (root / claim.README).read_text(encoding="utf-8")
    return text[text.index(claim.BLOCK_OPEN) : text.index(claim.BLOCK_CLOSE)]


def test_the_readmes_table_of_what_was_run_has_a_row_for_every_adapter_the_run_tests():
    """Written from the table of tested versions, the manifest and this tree's version;
    the first test in this file fails when it is edited by hand."""
    rows = [line for line in _block(_REPO).splitlines() if line.startswith("| ")][1:]
    assert [row.split(" | ")[0].removeprefix("| ") for row in rows] == [
        spec["title"] for spec in manifest.ADAPTERS.values()
    ]
    version = re.search(r'^version\s*=\s*"([^"]+)"', (_REPO / claim.PYPROJECT).read_text(), re.M)[1]
    for (name, spec), row in zip(manifest.ADAPTERS.items(), rows, strict=True):
        for entry in plan.read_measured()[name]:
            assert f"`{entry['distribution']}` {plan.spelled(entry)}" in row
        assert f"/compat-results/site/{version}/{name}.svg" in row
        assert f"/blob/compat-results/site/{version}/README.md#{spec['label']})" in row
        assert row.count("|") == 5, "one image, one link, nothing that splits the row"


def test_a_version_bump_moves_the_tables_images_to_the_new_release(tree, capsys):
    """A README shows its own release. The release script runs `sync` after the bump, and
    a bump without it fails the check."""
    path = tree / claim.PYPROJECT
    text = path.read_text(encoding="utf-8")
    bumped = re.sub(r'(?m)^version\s*=\s*".*"', 'version         = "9.9.9"', text)
    assert bumped != text
    path.write_text(bumped, encoding="utf-8")
    assert claim.main(["check"], root=tree) == 1
    assert "compat-table markers" in capsys.readouterr().out
    assert claim.main(["sync"], root=tree) == 0 and claim.drift(tree) == []
    block = _block(tree)
    assert "Today, on wardex 9.9.9" in block and block.count("/site/9.9.9/") == 2 * len(
        manifest.ADAPTERS
    )
    release = (_REPO / "scripts/release.py").read_text(encoding="utf-8")
    assert '"scripts/compat/claim.py", "sync"' in release and "str(README)" in release


def test_widening_a_range_rewrites_the_tables_row_too(tree):
    row = _a_row_with_two_lines(tree)
    line = _line_after(row)
    assert claim.main(["widen", row["distribution"], line], root=tree) == 0
    assert f"`{row['distribution']}` {plan.spelled(_row(tree, row['distribution']))}" in _block(
        tree
    )
    assert claim.drift(tree) == []


def test_a_readme_without_the_markers_is_left_without_a_table(tree):
    path = tree / claim.README
    text = path.read_text(encoding="utf-8")
    start, end = text.index(claim.BLOCK_OPEN), text.index(claim.BLOCK_CLOSE)
    path.write_text(text[:start] + text[end + len(claim.BLOCK_CLOSE) :], encoding="utf-8")
    assert claim.main(["sync"], root=tree) == 0
    assert claim.BLOCK_OPEN not in path.read_text(encoding="utf-8") and claim.drift(tree) == []
