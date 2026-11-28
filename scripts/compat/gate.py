"""Whether a commit may be released: each thing it says it is tested on has a recent pass.

    python3 scripts/compat/gate.py --commit <sha> --records <a checkout of compat-results>

Run by the release workflow before anything is published, in the tree being
released. It reads the record lines the compatibility run wrote and asks one
question of them: did a whole run on the default branch, at exactly this
commit, in the last two days, pass every point that speaks for a claim this
tree makes? A point of a watching kind is not asked. It writes nothing and
needs no permission beyond reading two branches.

Exit 0 lets the release go on. Exit 20 stops it and prints what is missing and
the command that produces it. A release that has to go out anyway carries
`scripts/compat/release-override.json` in its own tree, naming its version and
the reason; the gate then lets it through and says so, and the page of that
release says so for as long as the page exists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compat import kinds, manifest, model, plan  # noqa: E402
from compat.verdict import GREEN, OUTCOME_WORDS, is_watching  # noqa: E402

STOP_EXIT = 20
#: A pass older than this says nothing about the commit as it will be installed today:
#: frameworks release daily. Two days leaves room for a release cut the morning after.
MAX_AGE_DAYS = 2
MAIN_REF = "refs/heads/main"
OVERRIDE = model.OVERRIDE
REPO = Path(__file__).resolve().parents[2]
_VERSION_LINE = re.compile(r'^version\s*=\s*"([^"]+)"', re.M)


_RUN_URL = re.compile(r"https://[A-Za-z0-9./_-]{1,200}")


def code(text: object, limit: int = 200) -> str:
    """What a record line holds, as a code span: printable ASCII and never markup."""
    kept = "".join(ch if " " <= ch <= "~" and ch != "`" else "?" for ch in str(text))
    return f"`{kept[:limit]}`"


def _url(text: object) -> str:
    """A run's address as a record line gives it, or "" when it is anything but one."""
    return text if isinstance(text, str) and _RUN_URL.fullmatch(text) else ""


def read_lines(records: Path, today: str) -> list[dict]:
    """Every record line of the last `MAX_AGE_DAYS` days, as schema-2 lines.

    A line that cannot be read is skipped: one bad line must not hide the good ones."""
    lines = []
    for back in range(MAX_AGE_DAYS + 1):
        day = date.fromisoformat(today) - timedelta(days=back)
        path = records / "records" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.jsonl"
        if not path.is_file():
            continue
        for raw in path.read_bytes().splitlines():
            try:
                line = json.loads(raw.decode("utf-8"))
                if isinstance(line, dict) and isinstance(line.get("run"), dict):
                    lines.append(model.read_record(line))
            except (ValueError, TypeError, KeyError, AttributeError):
                continue
    return lines


def claimed(root: Path) -> dict[str, int]:
    """The adapters this tree says are tested, and how many proving points each needs."""
    table = plan.read_measured(
        root / plan.MEASURED_PY.relative_to(plan.REPO), root / plan.ENUMS_PY.relative_to(plan.REPO)
    )
    need = kinds.KINDS["framework"].min_points
    return {name: need for name, rows in table.items() if rows and name in manifest.ADAPTERS}


def _attempt(line: dict) -> int:
    try:
        return int(line["run"].get("attempt") or 1)
    except (TypeError, ValueError):
        return 1


def _outcome(line: dict) -> str:
    """The line's outcome, or "" when it holds anything but a word."""
    outcome = line.get("outcome")
    return outcome if isinstance(outcome, str) else ""


def decide(lines: list[dict], *, commit: str, today: str, claims: dict[str, int]) -> dict:
    """`{"pass", "reasons", "run"}` for releasing `commit` on `today`.

    The newest whole run on the default branch at exactly `commit` decides, and of that
    run its last attempt: a run whose failed jobs were run again writes its lines a second
    time, and what it said the first time is no longer what it says. A point recorded
    twice is one point, and passes only if every line of it does."""
    stop = {"pass": False, "run": None}
    if not claims:
        # A tree that gives no adapter a tested range has nothing a pass could prove,
        # which is more likely a table that could not be read than a release with no claims.
        return {**stop, "reasons": ["this tree gives no adapter a tested range"]}
    oldest = (date.fromisoformat(today) - timedelta(days=MAX_AGE_DAYS)).isoformat()
    runs: dict[tuple, list[dict]] = {}
    for line in lines:
        run = line["run"]
        whole = run.get("scope") == "all" and run.get("ref") == MAIN_REF
        if whole and run.get("sdk_commit") == commit and str(run.get("date", "")) >= oldest:
            runs.setdefault((str(run.get("started_at", "")), str(run.get("id", ""))), []).append(
                line
            )
    if not runs:
        short = "" if len(commit) >= 40 else " (the whole commit id is needed, not a prefix)"
        return {
            **stop,
            "reasons": [
                f"no whole compatibility run on the default branch at {commit[:10]} in the "
                f"last {MAX_AGE_DAYS} days{short}"
            ],
        }
    newest = runs[max(runs)]
    last = max(_attempt(line) for line in newest)
    points: dict[str, list[dict]] = {}
    for line in newest:
        if _attempt(line) == last and not is_watching(line):
            points.setdefault(str(line.get("key", "")), []).append(line)
    reasons = []
    proving: dict[str, int] = {}
    for key, said in sorted(points.items()):
        line = next((entry for entry in said if _outcome(entry) not in GREEN), said[0])
        outcome, key = _outcome(line), code(key, 120)
        held = line.get("targets")
        targets = ", ".join(f"{d} {v}" for d, v in (held if isinstance(held, dict) else {}).items())
        if outcome in ("no_result", "harness_error"):
            reasons.append(f"{key}: {OUTCOME_WORDS[outcome]}")
        elif outcome not in OUTCOME_WORDS:
            # Not a word any run writes: the line is damaged, wherever its point sits.
            reasons.append(f"{key}: a result this gate cannot read")
        elif line.get("in_claim") and outcome not in GREEN:
            reasons.append(
                f"{key}: {code(targets)} is inside the tested range and: {OUTCOME_WORDS[outcome]}"
            )
        elif line.get("in_claim") and line.get("kind") == "framework":
            subject = str(line.get("subject", ""))
            proving[subject] = proving.get(subject, 0) + 1
    for name, need in sorted(claims.items()):
        if proving.get(name, 0) < need:
            reasons.append(
                f"{name}: {proving.get(name, 0)} passing points inside its tested range, "
                f"{need} are needed"
            )
    ended = {line["run"].get("red") for line in newest if _attempt(line) == last}
    if True in ended and not reasons:
        # A run is also red for what its plan could not make a point of: a tested release
        # that admits no supported interpreter, an adapter the index would not list. No
        # line holds that, and a release needs the run to have been green.
        reasons.append(
            "the run ended red for a reason no point holds (its plan could not cover "
            "something the tree says is tested); see the run"
        )
    elif ended != {False} and not reasons:
        # Lines written before a run recorded how it ended. Passing lines alone are not
        # proof that it ended green, so they are not taken for it.
        reasons.append("the run's lines do not say how it ended; run it again at this commit")
    return {"pass": not reasons, "reasons": reasons, "run": _url(newest[0]["run"].get("url"))}


def read_override(root: Path, version: str) -> dict | None:
    """The override this tree carries for its own version, or None.

    One left behind by an earlier release names that release and excuses nothing."""
    try:
        return model.parse_override((root / OVERRIDE).read_text(encoding="utf-8"), version)
    except (OSError, ValueError):  # missing, a directory, or not text
        return None


def rejected_override(root: Path, version: str) -> str:
    """What to say when the tree carries an override file that does not apply, or "".

    A gate that fails closed in silence leaves the person who wrote the file guessing."""
    path = root / OVERRIDE
    # Anything at that path counts, a directory or a dangling link too: whoever put it
    # there meant something by it.
    if not (path.exists() or path.is_symlink()) or read_override(root, version) is not None:
        return ""
    return (
        f"`{OVERRIDE}` is in this tree and does not apply. It has to be a JSON object with "
        f'"version": "{version}" (the version of this tree) and a "reason" that says in '
        "words why the release goes out without the proof."
    )


def tree_version(root: Path) -> str:
    match = _VERSION_LINE.search((root / "sdks/python/pyproject.toml").read_text(encoding="utf-8"))
    return match[1] if match else ""


def words(decision: dict, override: dict | None, commit: str, rejected: str = "") -> str:
    """What the gate prints: the decision, why, and what to do about it."""
    lines = ["## Release gate: compatibility", ""]
    if decision["pass"]:
        lines.append(
            f"**PASS.** Every point that speaks for a tested range passed at {commit[:10]}: "
            f"{decision['run']}"
        )
        if override:
            lines += [
                "",
                f"`{OVERRIDE}` in this tree names this version and was not needed. Remove it "
                "before releasing: the plan reads it from the release's tag, and the page of "
                "this release would say it carried an override.",
            ]
    else:
        lines.append(
            f"**STOP.** {commit[:10]} has no recent pass for what it says it is tested on:"
        )
        lines += ["", *[f"- {reason}" for reason in decision["reasons"]], ""]
        if decision["run"]:
            lines.append(f"The run that says so: {decision['run']}")
        lines += [
            "To get the proof, run the compatibility workflow on the default branch at this "
            "commit and wait for it to end green:",
            "",
            "    gh workflow run compat.yml --ref main",
            "",
            "If a release inside a tested range no longer passes, fix it or stop claiming it "
            "(`python3 scripts/compat/claim.py`), then run again.",
        ]
        if override:
            lines += [
                "",
                f"**OVERRIDDEN** for {code(override['version'], 40)} by `{OVERRIDE}` in this "
                f"tree. Reason given: {model.quoted(override['reason'])}. The release goes on, "
                "and its page will say so.",
            ]
        elif rejected:
            lines += ["", rejected]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--commit", required=True, help="the commit being released")
    parser.add_argument("--records", required=True, help="a checkout of the records branch")
    parser.add_argument("--today", help="YYYY-MM-DD (UTC); default today")
    args = parser.parse_args(argv)
    today = args.today or datetime.now(timezone.utc).date().isoformat()
    decision = decide(
        read_lines(Path(args.records), today), commit=args.commit, today=today, claims=claimed(REPO)
    )
    override = read_override(REPO, tree_version(REPO))
    rejected = "" if decision["pass"] else rejected_override(REPO, tree_version(REPO))
    text = words(decision, override, args.commit, rejected)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)
    return 0 if decision["pass"] or override else STOP_EXIT  # an override only matters on a stop


if __name__ == "__main__":
    raise SystemExit(main())
