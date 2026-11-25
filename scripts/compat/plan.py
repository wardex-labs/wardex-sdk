"""Compute today's points: which exact framework versions the daily run tests.

Inputs are the table of tested versions, the manifest, and what PyPI lists at
one cutoff instant. The same three give the same plan, so a run re-attempted
days later resolves what the first attempt resolved. Nothing here is written
into a workflow file: a new framework release changes the plan by existing.

A point is one framework version on one interpreter:

- ``floor``: the lowest release still served inside the tested range, on the
  oldest interpreter, with the dependencies of its own time.
- ``minor_newest``: the newest patch of each tested release line below the
  top one, for a framework whose tree differs from line to line.
- ``claim_top``: the newest release inside the tested range, on the newest
  interpreter; ``claim_top_oldpy`` is the same question on the oldest one.
- ``beyond``: the newest patch of each release line above the tested range.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compat import manifest  # noqa: E402

SCHEMA = 1
REPO = Path(__file__).resolve().parents[2]
MEASURED_PY = REPO / "sdks/python/src/wardex_sdk/_adapters/_measured.py"
ENUMS_PY = REPO / "sdks/python/src/wardex_sdk/_enums.py"

_FINAL = re.compile(r"\d+(?:\.\d+)*")
_CLAUSE = re.compile(r"\s*(>=|<=|==|!=|~=|>|<)\s*([0-9]+(?:\.[0-9]+)*)(\.\*)?\s*")


def vkey(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(text: str) -> datetime:
    return datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def read_measured(measured_py: Path = MEASURED_PY, enums_py: Path = ENUMS_PY) -> dict:
    """The table of tested versions, read from source without importing the package.

    The plan job runs before any wheel exists, and importing the package
    would load the native extension. Returns
    ``{adapter: [{"distribution", "minors", "known_bad"}, ...]}``.
    """
    names: dict[str, str] = {}
    for node in ast.walk(ast.parse(enums_py.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ClassDef) and node.name == "AdapterName":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant):
                    names[stmt.targets[0].id] = stmt.value.value
    table: dict[str, list[dict]] = {}
    for node in ast.walk(ast.parse(measured_py.read_text(encoding="utf-8"))):
        target = getattr(node, "target", None) or (getattr(node, "targets", None) or [None])[0]
        if not (isinstance(target, ast.Name) and target.id == "MEASURED"):
            continue
        for key, value in zip(node.value.keys, node.value.values, strict=True):
            entries = []
            for call in value.elts:
                args = [ast.literal_eval(arg) for arg in call.args]
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                entries.append(
                    {
                        "distribution": args[0],
                        "minors": tuple(args[1]) if len(args) > 1 else tuple(kwargs["minors"]),
                        "known_bad": tuple(kwargs.get("known_bad", ())),
                    }
                )
            table[names[key.attr]] = entries
    if not table:
        raise ValueError(f"no MEASURED table found in {measured_py}")
    return table


def fetch_releases(distribution: str) -> list[dict]:
    """Every file PyPI lists for `distribution`, as plain data.

    The JSON API rather than the simple index because it carries upload times
    and yank flags in one request. Each release keeps its files, because a
    release is installable at an instant only if a file had been uploaded by
    then. Tried three times: a plan that cannot be made leaves the whole run
    without a result, and one dropped connection is not worth that.
    """
    url = f"https://pypi.org/pypi/{distribution}/json"
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                payload = json.load(response)
            break
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(15 * (attempt + 1))
    releases = []
    for version, files in payload["releases"].items():
        releases.append(
            {
                "version": version,
                "files": [
                    {
                        "uploaded": f["upload_time_iso_8601"][:19] + "Z",
                        "yanked": bool(f.get("yanked")),
                        "requires_python": f.get("requires_python") or "",
                    }
                    for f in files
                ],
            }
        )
    return releases


def admits(requires_python: str, python: str) -> bool:
    """Whether a ``Requires-Python`` value lets `python` ("3.10") install the release.

    Only the comparison forms that appear in practice. A clause this cannot
    read is taken as admitting: the installer is the authority, and a point
    that then fails to install is recorded as exactly that.
    """
    have = vkey(python)
    for clause in filter(None, (part.strip() for part in requires_python.split(","))):
        match = _CLAUSE.fullmatch(clause)
        if match is None:
            continue
        operator, text, wildcard = match.groups()
        # An interpreter is named by two numbers and stands for every patch of it, so a
        # bound is compared on as many numbers as the interpreter has.
        want = vkey(text)[: len(have)]
        want += (0,) * (len(have) - len(want))
        mine = have[: len(vkey(text))] if wildcard else have
        if operator == ">=" and not have >= want:
            return False
        if operator == ">" and not have > want:
            return False
        if operator == "<=" and not have <= want:
            return False
        if operator == "<" and not have < want:
            return False
        if operator == "==" and mine != vkey(text)[: len(mine)]:
            return False
        if operator == "!=" and mine == vkey(text)[: len(mine)]:
            return False
        if operator == "~=" and not (have >= want and have[0] == want[0]):
            return False
    return True


def _finals(releases: list[dict], cutoff: str) -> dict[str, dict]:
    """Final releases installable at `cutoff`: ``{version: {"uploaded", "requires_python"}}``."""
    out = {}
    for release in releases:
        if not _FINAL.fullmatch(release["version"]):
            continue
        live = [f for f in release["files"] if not f["yanked"] and f["uploaded"] <= cutoff]
        if live:
            out[release["version"]] = {
                "uploaded": min(f["uploaded"] for f in live),
                "requires_python": live[0]["requires_python"],
            }
    return out


def _line(version: str, width: int) -> str:
    return ".".join(version.split(".")[:width])


def _claimed(finals: dict, entry: dict) -> list[str]:
    def inside(version: str) -> bool:
        parts = version.split(".")
        two = ".".join(parts[:2])
        return (two in entry["minors"] or parts[0] in entry["minors"]) and (
            version not in entry["known_bad"]
        )

    return sorted((v for v in finals if inside(v)), key=vkey)


def _python_for(finals: dict, version: str, prefer: str) -> str | None:
    """`prefer` when the release admits it, else the nearest interpreter that does."""
    spec = finals[version]["requires_python"]
    if admits(spec, prefer):
        return prefer
    order = manifest.SUPPORTED_PYTHONS
    ranked = order if prefer == manifest.PYTHONS[0] else tuple(reversed(order))
    return next((python for python in ranked if admits(spec, python)), None)


def _prerelease_ahead(releases: list[dict], finals: dict, cutoff: str) -> str | None:
    """A pre-release published beyond the newest final, if there is one. Recorded, not run."""
    newest = max(finals, key=vkey) if finals else None
    ahead = []
    for release in releases:
        version = release["version"]
        head = re.match(r"\d+(?:\.\d+)*", version)
        if head is None or _FINAL.fullmatch(version):
            continue
        if any(not f["yanked"] and f["uploaded"] <= cutoff for f in release["files"]) and (
            newest is None or vkey(head.group()) > vkey(newest)
        ):
            ahead.append(version)
    return ahead[-1] if ahead else None


def plan_adapter(name: str, entries: list[dict], releases: dict, cutoff: str) -> dict:
    """One adapter's points and the facts the report states about it."""
    spec = manifest.ADAPTERS[name]
    primary, companions = entries[0], entries[1:]
    finals = _finals(releases[primary["distribution"]], cutoff)
    claimed = _claimed(finals, primary)
    info = {
        "label": spec["label"],
        "evidence": spec["evidence"],
        "replay_of": spec["replay_of"],
        "claim": {entry["distribution"]: list(entry["minors"]) for entry in entries},
        "newest": max(finals, key=vkey) if finals else None,
        "prerelease_ahead": _prerelease_ahead(releases[primary["distribution"]], finals, cutoff),
        # A problem means the adapter could not be planned and turns the run red. A note is
        # a fact about the plan worth reading (a point that cannot exist) and does not.
        "problems": [],
        "notes": [],
    }
    if not claimed:
        info["problems"].append(
            f"no installable release of {primary['distribution']} lies inside its tested range"
        )
        return {"info": info, "points": []}

    width = max(len(minor.split(".")) for minor in primary["minors"])
    top, floor = claimed[-1], claimed[0]
    oldest_py, newest_py = manifest.PYTHONS

    companion_finals = {
        entry["distribution"]: _finals(releases[entry["distribution"]], cutoff)
        for entry in companions
    }

    def companion_pins(pick: int) -> dict[str, str]:
        pins = {}
        for entry in companions:
            versions = _claimed(companion_finals[entry["distribution"]], entry)
            if versions:
                pins[entry["distribution"]] = versions[pick]
        return pins

    def uploaded(targets: dict[str, str]) -> dict[str, str]:
        return {
            dist: (finals if dist == primary["distribution"] else companion_finals[dist])[version][
                "uploaded"
            ]
            for dist, version in targets.items()
        }

    points: list[dict] = []

    def add(role: str, version: str, prefer: str, *, pins: dict | None = None, line: str = ""):
        python = _python_for(finals, version, prefer)
        if python is None:
            # Above the range this is the framework's own business. Inside it, a release the
            # README calls tested cannot be run at all, and that must not pass in silence.
            said = f"{primary['distribution']} {version} admits no supported interpreter"
            info["notes" if role == "beyond" else "problems"].append(f"{said}; not run")
            return
        targets = {primary["distribution"]: version, **(pins or {})}
        stamps = uploaded(targets)
        in_claim = role != "beyond"
        point_cutoff = cutoff
        if role == "floor":
            window = timedelta(days=manifest.FLOOR_WINDOW_DAYS)
            point_cutoff = min(cutoff, iso(parse_iso(max(stamps.values())) + window))
        points.append(
            {
                "key": "--".join(filter(None, (name, role, line))),
                "adapter": name,
                "role": role,
                "targets": targets,
                "floats": [
                    e["distribution"] for e in companions if e["distribution"] not in targets
                ],
                "python": python,
                "in_claim": in_claim,
                "cutoff": point_cutoff,
                "assume_measured": not in_claim,
                "uploaded": stamps,
            }
        )

    add("floor", floor, oldest_py, pins=companion_pins(0))
    lines = sorted({_line(v, width) for v in claimed}, key=vkey)
    for line in lines[:-1] if width > 1 else ():
        newest_in_line = [v for v in claimed if _line(v, width) == line][-1]
        if newest_in_line != floor:
            add("minor_newest", newest_in_line, newest_py, line=line)
    add("claim_top", top, newest_py, pins=companion_pins(-1))
    old_enough = [v for v in claimed if admits(finals[v]["requires_python"], oldest_py)]
    if old_enough:
        add("claim_top_oldpy", old_enough[-1], oldest_py)
    else:
        info["notes"].append(
            f"no tested release of {primary['distribution']} admits Python {oldest_py}"
        )

    above = sorted((v for v in finals if vkey(v) > vkey(top)), key=vkey)
    above = [
        v for v in above if v not in primary["known_bad"] and _line(v, width) != _line(top, width)
    ]
    beyond_lines = sorted({_line(v, width) for v in above}, key=vkey)[-manifest.BEYOND_LIMIT :]
    for line in beyond_lines:
        add("beyond", [v for v in above if _line(v, width) == line][-1], newest_py, line=line)
    return {"info": info, "points": points}


def build_plan(measured: dict, releases: dict, cutoff: str) -> dict:
    adapters, points, problems = {}, [], []
    for name, entries in sorted(measured.items()):
        if not entries:
            if name not in manifest.REPLAY_ONLY:
                problems.append(f"{name} has no tested versions and no replay entry")
            continue
        if name not in manifest.ADAPTERS:
            problems.append(f"{name} is in the table of tested versions but not in the manifest")
            continue
        planned = plan_adapter(name, entries, releases, cutoff)
        adapters[name] = planned["info"]
        points.extend(planned["points"])
    return {
        "schema": SCHEMA,
        "cutoff": cutoff,
        "adapters": adapters,
        "replay_only": manifest.REPLAY_ONLY,
        "points": points,
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="where to write plan.json")
    parser.add_argument("--cutoff", help="UTC instant (ISO 8601); default now")
    parser.add_argument("--only", help="plan one adapter only")
    args = parser.parse_args(argv)

    cutoff = iso(parse_iso(args.cutoff)) if args.cutoff else iso(datetime.now(timezone.utc))
    measured = read_measured()
    if args.only:
        measured = {args.only: measured[args.only]}
    releases = {
        entry["distribution"]: fetch_releases(entry["distribution"])
        for entries in measured.values()
        for entry in entries
    }
    plan = build_plan(measured, releases, cutoff)
    # A plan of one adapter says nothing about the others; the report must not read it
    # as the whole run.
    plan["partial"] = bool(args.only)
    Path(args.out).write_text(json.dumps(plan, indent=1, sort_keys=True), encoding="utf-8")
    for point in plan["points"]:
        targets = ", ".join(f"{d}=={v}" for d, v in point["targets"].items())
        print(f"{point['key']:<44} py{point['python']:<5} {targets}")
    for problem in plan["problems"]:
        print(f"problem: {problem}", file=sys.stderr)
    matrix = {"include": [{"key": p["key"], "python": p["python"]} for p in plan["points"]]}
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write("matrix=" + json.dumps(matrix, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
