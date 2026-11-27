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


def clean(text: object, limit: int = 200) -> str:
    return "".join(ch if " " <= ch <= "~" else "?" for ch in str(text))[:limit]


def read_lines(records: Path, today: str) -> list[dict]:
    """Every record line of the last `MAX_AGE_DAYS` days, as schema-2 lines.

    A line that is not JSON is skipped: one bad line must not hide the good ones."""
    lines = []
    for back in range(MAX_AGE_DAYS + 1):
        day = date.fromisoformat(today) - timedelta(days=back)
        path = records / "records" / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}.jsonl"
        if not path.is_file():
            continue
        for text in path.read_text(encoding="utf-8").splitlines():
            try:
                line = json.loads(text)
            except ValueError:
                continue
            if isinstance(line, dict) and isinstance(line.get("run"), dict):
                lines.append(model.read_record(line))
    return lines


def claimed(root: Path) -> dict[str, int]:
    """The adapters this tree says are tested, and how many proving points each needs."""
    table = plan.read_measured(
        root / plan.MEASURED_PY.relative_to(plan.REPO), root / plan.ENUMS_PY.relative_to(plan.REPO)
    )
    need = kinds.KINDS["framework"].min_points
    return {name: need for name, rows in table.items() if rows and name in manifest.ADAPTERS}


def decide(lines: list[dict], *, commit: str, today: str, claims: dict[str, int]) -> dict:
    """`{"pass", "reasons", "run"}` for releasing `commit` on `today`."""
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
        return {
            "pass": False,
            "run": None,
            "reasons": [
                f"no whole compatibility run on the default branch at {commit[:10]} in the "
                f"last {MAX_AGE_DAYS} days"
            ],
        }
    newest = runs[max(runs)]
    reasons = []
    proving: dict[str, int] = {}
    for line in sorted(newest, key=lambda entry: str(entry.get("key", ""))):
        if is_watching(line):
            continue
        outcome, key = line.get("outcome"), clean(line.get("key", ""), 120)
        targets = ", ".join(f"{d} {v}" for d, v in (line.get("targets") or {}).items())
        if outcome in ("no_result", "harness_error"):
            reasons.append(f"{key}: {OUTCOME_WORDS[outcome]}")
        elif line.get("in_claim") and outcome not in GREEN:
            said = OUTCOME_WORDS.get(outcome, "an unknown result")
            reasons.append(f"{key}: {clean(targets)} is inside the tested range and: {said}")
        elif line.get("in_claim") and line.get("kind") == "framework":
            proving[line.get("subject", "")] = proving.get(line.get("subject", ""), 0) + 1
    for name, need in sorted(claims.items()):
        if proving.get(name, 0) < need:
            reasons.append(
                f"{name}: {proving.get(name, 0)} passing points inside its tested range, "
                f"{need} are needed"
            )
    return {"pass": not reasons, "reasons": reasons, "run": clean(newest[0]["run"].get("url", ""))}


def read_override(root: Path, version: str) -> dict | None:
    """The override this tree carries for its own version, or None.

    One left behind by an earlier release names that release and excuses nothing."""
    try:
        return model.parse_override((root / OVERRIDE).read_text(encoding="utf-8"), version)
    except OSError:
        return None


def tree_version(root: Path) -> str:
    match = _VERSION_LINE.search((root / "sdks/python/pyproject.toml").read_text(encoding="utf-8"))
    return match[1] if match else ""


def words(decision: dict, override: dict | None, commit: str) -> str:
    """What the gate prints: the decision, why, and what to do about it."""
    lines = ["## Release gate: compatibility", ""]
    if decision["pass"]:
        lines.append(
            f"**PASS.** Every point that speaks for a tested range passed at {commit[:10]}: "
            f"{decision['run']}"
        )
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
                f"**OVERRIDDEN** for {override['version']} by `{OVERRIDE}` in this tree. Reason "
                f"given: {override['reason']}. The release goes on, and its page will say so.",
            ]
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
    override = None if decision["pass"] else read_override(REPO, tree_version(REPO))
    text = words(decision, override, args.commit)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)
    return 0 if decision["pass"] or override else STOP_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
