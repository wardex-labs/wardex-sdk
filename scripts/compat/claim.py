"""Change what wardex says it is tested on, in every place that says it, with one command.

    python3 scripts/compat/claim.py check
    python3 scripts/compat/claim.py widen openai-agents 0.24
    python3 scripts/compat/claim.py narrow openai-agents --drop 0.22
    python3 scripts/compat/claim.py narrow pydantic-ai-slim --since 2.54
    python3 scripts/compat/claim.py bad openai-agents 0.23.0
    python3 scripts/compat/claim.py sync

The table of tested versions (`_adapters/_measured.py`) is the one source. The
test pin in `sdks/python/pyproject.toml` and every version phrase and install
line in `README.md` are written from it; `check` names each place that
disagrees and exits 1, and the test suite runs it. A person runs the other
commands and commits the result: nothing here opens a pull request, and no
run changes a claim on its own.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compat import plan  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MEASURED = "sdks/python/src/wardex_sdk/_adapters/_measured.py"
ENUMS = "sdks/python/src/wardex_sdk/_enums.py"
PYPROJECT = "sdks/python/pyproject.toml"
README = "README.md"

# The README wraps its sentences, so the white space inside a phrase may hold one line
# break. Not a blank line: that ends the paragraph, and what follows is another sentence.
_GAP = r"(?:[ \t]+|[ \t]*\n[ \t]*)"
_LINE = rf"\d+(?:\.\d+)*\.x(?:{_GAP}from{_GAP}\d+(?:\.\d+)*)?"
_RANGE = rf"{_LINE}(?:,{_GAP}{_LINE})*"
_BOUNDS = r"\s*>=\s*(?P<lower>\d+(?:\.\d+)*)\s*,\s*<\s*(?P<upper>\d+(?:\.\d+)*)\s*"


_NUMBER = re.compile(r"\d+(?:\.\d+)*")


def holes(lines: list[str]) -> str:
    """The release lines a table of `lines` skips, as words, or "".

    A tested range is written as one pin, and a pin cannot leave a line out of its middle:
    a table with a hole makes the install line admit releases nobody claims. Only inside
    one prefix (0.23 and 0.25 skip 0.24). Whether 0.9 is followed by 0.10 or by 1.0 is the
    framework's to decide, so lines of different prefixes are taken as adjacent."""
    keys = sorted(_vkey(line) for line in lines)
    skipped = []
    for low, high in zip(keys, keys[1:], strict=False):
        if low[:-1] == high[:-1]:
            prefix = ".".join(map(str, low[:-1]))
            skipped += [
                f"{prefix}.{n}.x" if prefix else f"{n}.x" for n in range(low[-1] + 1, high[-1])
            ]
    return ", ".join(skipped)


def _vkey(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def table(root: Path) -> dict[str, dict]:
    """Every row of the table of tested versions, by distribution."""
    measured = plan.read_measured(root / MEASURED, root / ENUMS)
    return {entry["distribution"]: entry for entries in measured.values() for entry in entries}


def bounds(entry: dict) -> tuple[str, str]:
    """The test pin a row asks for: from its lowest line, or the release its range
    starts at, to just under the line after its highest."""
    lines = sorted(entry["minors"], key=_vkey)
    top = _vkey(lines[-1])
    return entry["since"] or lines[0], ".".join(map(str, (*top[:-1], top[-1] + 1)))


def _tuple(items: tuple[str, ...]) -> str:
    """A tuple of strings as the formatter leaves it: `("0.2",)`, `("0.22", "0.23")`."""
    inside = ", ".join(f'"{item}"' for item in items)
    return f"({inside},)" if len(items) == 1 else f"({inside})"


def _call(entry: dict) -> str:
    """A row as the source spells it."""
    text = f'Measured("{entry["distribution"]}", {_tuple(entry["minors"])}'
    if entry["known_bad"]:
        text += f", known_bad={_tuple(entry['known_bad'])}"
    if entry["since"]:
        text += f', since="{entry["since"]}"'
    return text + ")"


# --- the three files, each rewritten from a row ---------------------------------


def write_table(source: str, entry: dict) -> str:
    """`source` with the row of `entry`'s distribution replaced, comments untouched."""
    lines = source.splitlines(keepends=True)
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Measured"):
            continue
        first = node.args[0] if node.args else None
        if not (isinstance(first, ast.Constant) and first.value == entry["distribution"]):
            continue
        start = sum(len(line) for line in lines[: node.lineno - 1]) + node.col_offset
        end = sum(len(line) for line in lines[: node.end_lineno - 1]) + node.end_col_offset
        return source[:start] + _call(entry) + source[end:]
    raise SystemExit(f"{entry['distribution']} has no row in the table of tested versions")


def _pin(dist: str) -> re.Pattern:
    name = rf"{re.escape(dist)}(?:\[[a-z0-9,_-]+\])?"
    return re.compile(rf"(?P<head>(?P<quote>[\"']){name}){_BOUNDS}(?P<tail>(?P=quote))")


def write_pins(text: str, entry: dict) -> str:
    """Every quoted `dist>=a,<b` requirement in `text` set to the row's bounds."""
    lower, upper = bounds(entry)
    return _pin(entry["distribution"]).sub(rf"\g<head>>={lower},<{upper}\g<tail>", text)


def _phrase(dist: str) -> re.Pattern:
    return re.compile(rf"(?P<head>`{re.escape(dist)}`{_GAP})(?P<range>{_RANGE})")


def write_readme(text: str, entry: dict) -> str:
    """Every "`dist` 0.22.x, 0.23.x" phrase and every install line for `dist`."""
    text = _phrase(entry["distribution"]).sub(rf"\g<head>{plan.spelled(entry)}", text)
    return write_pins(text, entry)


# --- check ----------------------------------------------------------------------


def drift(root: Path) -> list[str]:
    """Each place that says something else than the table, in words a person can act on."""
    rows = table(root)
    readme = (root / README).read_text(encoding="utf-8")
    pyproject = (root / PYPROJECT).read_text(encoding="utf-8")
    found = []
    for dist, entry in sorted(rows.items()):
        said, want = plan.spelled(entry), bounds(entry)
        pins = [(m["lower"], m["upper"]) for m in _pin(dist).finditer(pyproject)]
        if not pins:
            found.append(f"{PYPROJECT}: no test pin for {dist}")
        for pin in pins:
            if pin != want:
                found.append(
                    f"{PYPROJECT}: {dist} is pinned >={pin[0]},<{pin[1]}; the table says "
                    f"{said}, which is >={want[0]},<{want[1]}"
                )
        if holes(list(entry["minors"])):
            found.append(
                f"{MEASURED}: {dist} skips {holes(list(entry['minors']))} between its tested "
                "lines; a pin cannot say that, so the install line admits them"
            )
        for match in _phrase(dist).finditer(readme):
            wrote = " ".join(match["range"].split())
            if wrote != said:
                line = readme.count("\n", 0, match.start()) + 1
                found.append(f"{README}:{line}: says `{dist}` {wrote}; the table says {said}")
        for match in _pin(dist).finditer(readme):
            if (match["lower"], match["upper"]) != want:
                line = readme.count("\n", 0, match.start()) + 1
                found.append(
                    f"{README}:{line}: installs {dist}>={match['lower']},<{match['upper']}; the "
                    f"table says {said}, which is >={want[0]},<{want[1]}"
                )
    for _, entries in sorted(plan.read_measured(root / MEASURED, root / ENUMS).items()):
        for entry in entries[:1]:
            phrase = f"(tested on `{entry['distribution']}` {plan.spelled(entry)})"
            if phrase not in readme:
                found.append(f"{README}: the framework table has no row saying {phrase}")
    return found


# --- commands -------------------------------------------------------------------


def changed(entry: dict, args: argparse.Namespace) -> dict:
    """The row after the command, or exit with what is wrong with the command."""
    lines, bad, since = list(entry["minors"]), list(entry["known_bad"]), entry["since"]
    dist = entry["distribution"]
    for given in (getattr(args, "line", None), getattr(args, "drop", None)):
        width = {len(line.split(".")) for line in lines}
        if given is not None and (
            not _NUMBER.fullmatch(given) or len(given.split(".")) not in width
        ):
            raise SystemExit(f"{given!r} is not a release line as the table spells one: {lines[0]}")
    for given in (getattr(args, "version", None), getattr(args, "since", None)):
        if given is not None and not _NUMBER.fullmatch(given):
            raise SystemExit(f"{given!r} is not a release number")
    if args.command == "widen":
        if args.line in lines:
            raise SystemExit(f"{dist} {args.line}.x is already in the table")
        lowest = min(lines, key=_vkey)
        if since and _vkey(args.line) < _vkey(lowest):
            # The range starts inside its lowest line because the releases below that start
            # do not pass. A line under it would call them tested again without a word.
            raise SystemExit(
                f"{dist} is tested from {since} on; {args.line}.x lies below that. Decide where "
                f"the range starts first: claim.py narrow {dist} --since <release>"
            )
        lines.append(args.line)
    elif args.command == "bad":
        if not plan.claims(entry, args.version):
            raise SystemExit(
                f"{entry['distribution']} {args.version} is not inside the tested range"
            )
        bad.append(args.version)
    elif args.drop:
        if args.drop not in lines or len(lines) == 1:
            raise SystemExit(f"{args.drop} is not a line that can be dropped from {lines}")
        lines.remove(args.drop)
    else:
        since = args.since
    lines.sort(key=_vkey)
    if holes(lines):
        # The install line would admit releases the table does not claim.
        raise SystemExit(
            f"{dist} would be tested on {', '.join(f'{line}.x' for line in lines)}, which leaves "
            f"{holes(lines)} between the tested lines, and a pin cannot say that. Change the "
            "lines next to each other, or list single releases with `bad`."
        )
    if since and not (since == lines[0] or since.startswith(lines[0] + ".")):
        # The range starts inside its lowest line. A start left behind by a dropped line
        # would call a whole line tested and judge it untested.
        if args.command == "narrow" and args.since:
            raise SystemExit(f"--since must be a release of the lowest line, {lines[0]}.x")
        since = ""
    return {
        "distribution": entry["distribution"],
        "minors": tuple(lines),
        "known_bad": tuple(sorted(set(bad), key=_vkey)),
        "since": since,
    }


def apply(root: Path, entry: dict) -> list[str]:
    """Write `entry` to the table, then the pin and the README from it."""
    touched = []
    for name, write in ((MEASURED, write_table), (PYPROJECT, write_pins), (README, write_readme)):
        path = root / name
        before = path.read_text(encoding="utf-8")
        after = write(before, entry)
        if after != before:
            path.write_text(after, encoding="utf-8")
            touched.append(name)
    return touched


def main(argv: list[str] | None = None, root: Path = REPO) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="name every place that disagrees with the table")
    commands.add_parser("sync", help="rewrite the pin and the README from the table")
    widen = commands.add_parser("widen", help="add a release line to a tested range")
    widen.add_argument("distribution")
    widen.add_argument("line", help="a release line, as the table spells it: 0.24, or 3")
    narrow = commands.add_parser("narrow", help="take a release line or the bottom of one out")
    narrow.add_argument("distribution")
    how = narrow.add_mutually_exclusive_group(required=True)
    how.add_argument("--drop", help="a release line to remove")
    how.add_argument("--since", help="the first release of the lowest line that is tested")
    bad = commands.add_parser("bad", help="list one release inside the range as not tested")
    bad.add_argument("distribution")
    bad.add_argument("version")
    args = parser.parse_args(argv)

    if args.command == "check":
        found = drift(root)
        for line in found:
            print(line)
        if found:
            print(
                "\nThe table of tested versions is the source: python3 scripts/compat/claim.py sync"
            )
        return 1 if found else 0

    rows = table(root)
    if args.command == "sync":
        touched = sorted({name for entry in rows.values() for name in apply(root, entry)})
    else:
        if args.distribution not in rows:
            raise SystemExit(
                f"{args.distribution} is not in the table; it has {', '.join(sorted(rows))}"
            )
        entry = changed(rows[args.distribution], args)
        touched = apply(root, entry)
        print(f"{entry['distribution']}: tested on {plan.spelled(entry)}")
    for name in touched:
        print(f"wrote {name}")
    left = drift(root)
    for line in left:
        print(f"still to do by hand: {line}")
    if touched:
        print(
            "Run `ruff format` on what changed, add a CHANGELOG line, and commit the three "
            "files together."
        )
    return 1 if left else 0


if __name__ == "__main__":
    raise SystemExit(main())
