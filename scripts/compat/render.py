"""The public pages and README cells, written from `state.json` and nothing else.

    site/README.md                 which wardex versions have a page
    site/main/README.md            every point of the default branch's build, every kind
    site/<version>/README.md       the same for one published wardex release
    site/<version>/<adapter>.svg   the README's cell for that adapter on that release

Nothing here is typed by a person and nothing is fetched: the same state on the
same day renders the same bytes. A README names its own version's cells, so
the page of a release on PyPI shows that release and not the newest one.

Every string that reaches a page was cleaned when the report put it in the
state, is cleaned again here, and is written inside a code span or XML-escaped.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import date
from pathlib import Path
from xml.sax.saxutils import escape

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compat import kinds, model  # noqa: E402
from compat.verdict import FAILED, GREEN, NOT_OURS, OUTCOME_WORDS  # noqa: E402

# A pass older than this proves nothing about today: the row says when it last
# passed instead of that it passes. The same three days after which the
# watchdog calls the schedule stopped, so one late start lapses nothing.
LAPSE_DAYS = 3

MAIN = "main"
REPO_URL = "https://github.com/wardex-labs/wardex-sdk"
RECORDS_BRANCH = "compat-results"

_VERSION = re.compile(r"[0-9][0-9A-Za-z.+!-]{0,40}")
_NAME = re.compile(r"[a-z][a-z0-9_]{0,40}")
_MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()

# The dot is the only colour in a cell. The words say the same thing without it.
_DOT = {"pass": "#2da44e", "fail": "#cf222e", "stale": "#bf8700", "none": "#8c959f"}
_INK, _RULE = "#6e7781", "#8c959f"
_CHAR_PX = 6.62  # one 11px monospace glyph; `textLength` holds the text to it on any font


def clean(text: object, limit: int = 120) -> str:
    """Printable ASCII, no markup characters, bounded: safe in a code span."""
    kept = "".join(ch if " " <= ch <= "~" and ch not in "`|<>" else "?" for ch in str(text))
    return kept[:limit]


def code(text: object, limit: int = 120) -> str:
    return f"`{clean(text, limit)}`" if str(text) else "-"


def day_words(day: str) -> str:
    """`2026-11-28` as `28 Nov 2026`; anything else as nothing."""
    try:
        parsed = date.fromisoformat(day)
    except (TypeError, ValueError):
        return ""
    return f"{parsed.day} {_MONTHS[parsed.month - 1]} {parsed.year}"


def _age(day: str, today: str) -> int | None:
    try:
        return (date.fromisoformat(today) - date.fromisoformat(day)).days
    except (TypeError, ValueError):
        return None


def _recent(day: str, today: str, days: int) -> bool:
    age = _age(day, today)
    return age is not None and age <= days


_WARDEX = re.compile(r"(\d+(?:\.\d+)*)(?:(a|b|rc)(\d+))?(?:\.post(\d+))?")


def _release_order(version: str) -> tuple:
    """Newest last, for wardex's own versions: a final after its pre-releases, a post
    release after its final."""
    match = _WARDEX.fullmatch(version)
    rank = {"a": 0, "b": 1, "rc": 2, None: 3}[match[2]]
    numbers = tuple(int(part) for part in match[1].split("."))
    return (numbers, rank, int(match[3] or 0), int(match[4] or 0))


def _published_versions(state: dict) -> list[str]:
    """The releases that get a page, oldest first. One list for the index and for the
    files, so the index never links to a page nobody wrote."""
    held = state.get("versions") or {}
    safe = [v for v in held if v != MAIN and _VERSION.fullmatch(v) and _WARDEX.fullmatch(v)]
    return sorted(safe, key=_release_order)


def status_of(entry: dict, today: str) -> dict:
    """What one remembered point says today: a word, a status and its dates.

    `pass` only when the point passed on its last run and that run is recent.
    An old pass is `stale` (the proof lapsed), a failure is `fail` however old,
    and a point that could not be run says so with the date of its last pass.
    """
    last, identity = entry.get("last") or {}, entry.get("point") or {}
    outcome, ran = last.get("outcome", "no_result"), last.get("date", "")
    days = int(identity.get("fresh_days") or LAPSE_DAYS)
    targets = identity.get("targets")

    def about_this_release(kept: dict) -> dict:
        return kept if kept.get("targets") in (None, targets) else {}

    passed = about_this_release(entry.get("last_green") or {}).get("date", "")
    checked = about_this_release(entry.get("last_checked") or {})
    age = _age(ran, today)
    fresh = age is not None and age <= days
    cannot = OUTCOME_WORDS.get(outcome, "no result")
    if outcome in GREEN and fresh:
        status, words = "pass", OUTCOME_WORDS[outcome]
    elif outcome in GREEN:
        status, words = "stale", f"passed, not run since {day_words(ran)}"
    elif outcome in FAILED:
        status, words = "fail", OUTCOME_WORDS[outcome]
        if not fresh:
            words += f", not run since {day_words(ran)}"
    elif checked.get("outcome") in FAILED:
        # Its last run did not check it, whatever the reason. The last time it was checked
        # it failed, and that is what is known: the same thing its issue says.
        status = "fail"
        words = f"failed when last checked, on {day_words(checked.get('date', ''))}; now: {cannot}"
    elif outcome in NOT_OURS and checked.get("outcome") in GREEN and _recent(passed, today, days):
        # Only for a release that could not be looked at. A point that left no result is
        # a broken run, and the run is red for it: its cell must not be green beside that.
        status = "pass"
        words = f"passed on {day_words(passed)}; now: {cannot}"
    else:
        status = "none" if outcome in NOT_OURS and not identity.get("in_claim") else "stale"
        words = cannot
    return {"status": status, "words": words, "ran": ran, "passed": passed, "outcome": outcome}


def rows(state: dict, version: str) -> dict[str, list[dict]]:
    """The remembered points of one wardex build, by subject.

    `MAIN` is the default branch's own build; anything else is that published release."""
    grouped: dict[str, list[dict]] = {}
    for key, entry in sorted((state.get("points") or {}).items()):
        identity = entry.get("point")
        if not identity:
            continue
        sdk = identity.get("sdk") or {}
        built = sdk.get("origin", "build") == "build"
        if (version == MAIN) != built or (not built and sdk.get("version") != version):
            continue
        grouped.setdefault(identity.get("subject", ""), []).append({"key": key, **entry})
    return grouped


def _targets(identity: dict) -> str:
    return ", ".join(f"{dist} {version}" for dist, version in identity["targets"].items())


def _primary(identity: dict) -> str:
    return next(iter(identity["targets"].values()), "")


def _vkey(version: str) -> tuple:
    return tuple(int(part) if part.isdigit() else -1 for part in re.split(r"[.]", version))


def cell_words(points: list[dict], today: str) -> tuple[str, str]:
    """One adapter's cell: the status of its dot, and one line of words.

    Inside the tested range first, in the fewest words that stay exact: the
    releases run, and whether all of them pass. Then the newest release above
    the range, which is outside the claim and never colours the dot.
    """
    inside = [p for p in points if p["point"]["in_claim"]]
    above = [p for p in points if not p["point"]["in_claim"]]
    said = {p["key"]: status_of(p, today) for p in points}
    parts: list[str] = []
    status = "none"
    if inside:
        versions = sorted({_primary(p["point"]) for p in inside}, key=_vkey)
        span = versions[0] if len(versions) == 1 else f"{versions[0]} to {versions[-1]}"
        states = {said[p["key"]]["status"] for p in inside}
        bad = sorted(
            {_primary(p["point"]) for p in inside if said[p["key"]]["status"] == "fail"},
            key=_vkey,
        )
        if bad:
            status = "fail"
            parts.append(f"{', '.join(bad)} fails")
        elif states == {"pass"}:
            status = "pass"
            parts.append(f"{span} passes" if len(versions) == 1 else f"{span} pass")
        else:
            status = "stale"
            # Since the oldest of the last passes of what is not proven now. One of them
            # that never passed leaves no date to give.
            lapsed = [
                said[p["key"]]["passed"] for p in inside if said[p["key"]]["status"] != "pass"
            ]
            since = day_words(min(lapsed)) if all(lapsed) else ""
            parts.append(f"{span} not proven since {since}" if since else f"{span} not proven")
    if above:
        newest = max(above, key=lambda p: _vkey(_primary(p["point"])))
        word = {"pass": "passes", "fail": "fails"}.get(said[newest["key"]]["status"], "not checked")
        parts.append(f"newer {_primary(newest['point'])} {word}")
    ran = max((said[p["key"]]["ran"] for p in points), default="")
    if ran and status != "stale":
        # The day it was last run, without the year: the page behind the cell has the year.
        parts.append(day_words(ran).rsplit(" ", 1)[0])
    return status, " · ".join(clean(part, 60) for part in parts) or "no result"


def svg_cell(status: str, words: str) -> str:
    """A flat pill: a rule, one status dot, one line of monospace."""
    text = escape(words)
    length = round(len(words) * _CHAR_PX, 1)
    width = round(length + 30)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" role="img" '
        f'aria-label="{escape(words, {chr(34): "&quot;"})}">'
        f"<title>{text}</title>"
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="19" rx="9.5" fill="none" '
        f'stroke="{_RULE}" stroke-opacity="0.5"/>'
        f'<circle cx="10.5" cy="10" r="3.5" fill="{_DOT[status]}"/>'
        f'<text x="20" y="13.8" font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" '
        f'font-size="11" fill="{_INK}" textLength="{length}" lengthAdjust="spacingAndGlyphs">'
        f"{text}</text></svg>\n"
    )


_ROLE_WORDS = {
    "floor": "oldest tested release",
    "minor_newest": "newest patch of a tested line",
    "claim_top": "newest tested release",
    "claim_top_oldpy": "newest tested release, oldest Python",
    "beyond": "newer than the tested range",
    "displaced": "tested release a newer one replaced",
}


def _kind_label(identity: dict) -> str:
    kind = kinds.KINDS.get(identity.get("kind", "framework"))
    return kind.label if kind else clean(identity.get("kind", ""), 40)


def _section(name: str, info: dict, points: list[dict], today: str) -> list[str]:
    label = clean(info.get("label") or name, 60)
    said = info.get("said") or {}
    claim = "; ".join(
        f"{code(dist)} {clean(said.get(dist) or ', '.join(f'{line}.x' for line in lines), 80)}"
        for dist, lines in (info.get("claim") or {}).items()
    )
    lines = [f"## {label}", ""]
    if claim:
        lines.append(f"- Tested range, as this build states it: {claim}")
    if info.get("evidence") == "live":
        lines.append("- Evidence: the framework itself runs in the test.")
    elif info.get("evidence"):
        lines.append(
            "- Evidence: the framework's Python package runs against recorded output of "
            f"{code(info.get('replay_of') or 'a program')}, replayed. A pass does not show that "
            "a newer build of that program behaves the same."
        )
    if info.get("newest"):
        lines.append(f"- Newest release on PyPI when last planned: {code(info['newest'])}")
    lines += [
        "",
        "| Check | What | Version | Python | Result | Last run | Last pass |",
        "|---|---|---|---|---|---|---|",
    ]
    failing: list[tuple[str, list[str]]] = []
    for point in points:
        identity, now = point["point"], status_of(point, today)
        role = _ROLE_WORDS.get(identity["role"], clean(identity["role"], 40))
        watching = " (watching)" if identity.get("watching") else ""
        lines.append(
            f"| {_kind_label(identity)}{watching} | {role} | {code(_targets(identity))} "
            f"| {code(identity.get('python', ''))} | {now['words']} "
            f"| {day_words(now['ran']) or '-'} | {day_words(now['passed']) or '-'} |"
        )
        nodes = (point.get("last") or {}).get("failed_ids") or []
        if now["status"] == "fail" and nodes:
            failing.append((_targets(identity), nodes))
    for version, nodes in failing:
        lines += ["", f"Failing tests on {code(version)}:", ""]
        lines += [f"- {code(node, 300)}" for node in nodes[:40]]
        if len(nodes) > 40:
            lines.append(f"- and {len(nodes) - 40} more")
    return [*lines, ""]


def page(state: dict, version: str, today: str) -> str:
    """The detailed page of one wardex build."""
    about = (state.get("versions") or {}).get(version) or {}
    grouped = rows(state, version)
    if version == MAIN:
        title = "Compatibility of the default branch"
        what = (
            f"What the daily run found for wardex built from `main` "
            f"(version string {code(about.get('version', ''))}, commit "
            f"{code(str(about.get('commit', ''))[:10])})."
        )
    else:
        title = f"Compatibility of wardex {clean(version, 40)}"
        what = (
            f"What the daily run found for `wardex-sdk=={clean(version, 40)}` as PyPI serves it, "
            f"with the tests of its own tag {code(about.get('ref', ''))}."
        )
    lines = [f"# {title}", "", what, ""]
    override = about.get("override") if version != MAIN else None
    if isinstance(override, dict) and model.says_something(override.get("reason")):
        lines += [
            "**This release carried an override of the compatibility gate**, which lets a "
            "release go out without a recent pass for what it says it is tested on. The reason "
            f"given in its own tree: {model.quoted(override.get('reason', ''))}",
            "",
        ]
    lines += [
        f"Generated on {day_words(today)} from the run records on this branch; nothing here is "
        "written by hand. A row marked *watching* is run and reported and never turns the run "
        f"red. A pass older than {LAPSE_DAYS} days is shown as when it last passed, not as "
        "passing. "
        f"How to reproduce a row: [AGENTS.md]({REPO_URL}/blob/main/AGENTS.md).",
        "",
    ]
    adapters = about.get("adapters") or {}
    for name in sorted(grouped):
        lines += _section(name, adapters.get(name) or {}, grouped[name], today)
        if version == MAIN:
            lines += _replaced(grouped[name], (state.get("displaced") or {}).get(name) or [])
    if not grouped:
        lines += ["No point of this build has been run yet.", ""]
    lines += _claims(about.get("claims") or [])
    return "\n".join(lines)


def _replaced(points: list[dict], items: list[dict]) -> list[str]:
    """Releases that failed and that a newer one replaced, before their own point has run:
    still inside the range, known to fail, and in no row of the table above."""
    have = {point["key"] for point in points}
    # A rule that picks the release again speaks for it in its own row: the same release on
    # the same interpreter, and not at the floor, which installs another time's packages.
    run = [
        (point["point"].get("targets"), point["point"].get("python"))
        for point in points
        if point["point"].get("in_claim")
        and point["point"].get("role") not in ("displaced", "floor")
    ]
    lines = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("targets"), dict):
            continue
        if item.get("key") in have or (item["targets"], item.get("python")) in run:
            continue
        version = ", ".join(f"{dist} {v}" for dist, v in item["targets"].items())
        lines.append(
            f"- {code(version)} did not pass when it was last run, on "
            f"{day_words(item.get('failed_on', '')) or 'an unknown day'}. No rule picks it "
            "any more; it is run on its own from the next run on."
        )
    return [*lines, ""] if lines else []


def _claims(claims: list[dict]) -> list[str]:
    """What the README says was tested, sentence by sentence, and what proves each."""
    if not claims:
        return []
    lines = [
        "## What the README says, and what proves it",
        "",
        "Each sentence of the README that says wardex was tested, verified or measured "
        "against something, and the scheduled check behind it. Where there is none, that "
        "is said here instead of left to be assumed.",
        "",
        "| The README says | Proven on a schedule by |",
        "|---|---|",
    ]
    for claim in claims:
        kind = kinds.KINDS.get(claim.get("proof") or "")
        if kind:
            proof = f"{kind.label} points, {' and '.join(sorted(kind.cadences))}"
        else:
            proof = f"nothing yet: {clean(claim.get('why') or 'no scheduled run', 200)}"
        lines.append(f"| {clean(claim.get('says', ''), 200)} | {proof} |")
    return [*lines, ""]


def index(state: dict, today: str) -> str:
    versions = _published_versions(state)
    lines = [
        "# wardex compatibility",
        "",
        "Which framework releases each wardex build was run against, and what happened. "
        f"Generated on {day_words(today)} from the run records on this branch.",
        "",
        "- [the default branch](main/README.md)",
    ]
    lines += [f"- [wardex {clean(v, 40)}]({clean(v, 40)}/README.md)" for v in reversed(versions)]
    return "\n".join(lines) + "\n"


def site(state: dict, today: str) -> dict[str, str]:
    """Every file of the site, by its path under `site/`."""
    files = {"README.md": index(state, today), "main/README.md": page(state, MAIN, today)}
    for version in _published_versions(state):
        about = state["versions"][version] or {}
        files[f"{version}/README.md"] = page(state, version, today)
        grouped = rows(state, version)
        for name in sorted(set(about.get("adapters") or {}) | set(grouped)):
            if not _NAME.fullmatch(name):
                continue
            status, words = cell_words(grouped.get(name, []), today)
            files[f"{version}/{name}.svg"] = svg_cell(status, words)
    return files


def write_site(state_dir: Path, state: dict, today: str) -> list[str]:
    """Replace `site/` whole, so a page nothing renders any more does not linger."""
    root = state_dir / "site"
    if root.is_symlink():
        root.unlink()  # never write through a link someone put where the site goes
    shutil.rmtree(root, ignore_errors=True)
    written = []
    for name, text in sorted(site(state, today).items()):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(name)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state-dir", required=True, help="a checkout of the records branch")
    parser.add_argument("--today", required=True, help="the run's date, YYYY-MM-DD")
    args = parser.parse_args(argv)
    state = json.loads((Path(args.state_dir) / "state.json").read_text(encoding="utf-8"))
    for name in write_site(Path(args.state_dir), state, args.today):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
